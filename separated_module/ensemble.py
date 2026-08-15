"""
ensemble.py
Auto-split from the original monolithic AbstractIntegratedModule file.
Part of the `edge_ai_framework` package.

NOTE ON IMPORTS: several classes in the original monolith reference each
other in both directions (e.g. MLP <-> Transformer, IntegratedPipeline <->
WeightedEnsemblePredictor). To keep each file importable on its own without
triggering circular-import errors, sibling modules are imported as whole
modules (`from . import other_module`) rather than `from .other_module
import ClassName`. Class names used from another module are therefore
qualified, e.g. `pipeline.IntegratedPipeline`. This does not change any
behavior -- it's purely how the reference is spelled.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta, timezone
import sqlite3
import json
import joblib
import ast
import re
import sys
import threading
import time
from collections import deque, defaultdict, Counter
import socket
import pickle
import hashlib
import ssl
import os
import glob
import asyncio
import queue
import uuid
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Tuple, Optional, Dict, List
from enum import IntEnum, Enum
import traceback
from concurrent.futures import TimeoutError as FutureTimeoutError
import concurrent.futures
import secrets
import ipaddress
from functools import wraps
import hmac
import aiohttp
import psutil
import io
import struct

from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
import stat

try:
    from AbstractOptimizedModules import (
        optimized_sigmoid,
        optimized_sigmoid_deriv,
        optimized_tanh_deriv,
        optimized_lstm_cell_forward,
        optimized_project_heads,
        optimized_ame_encoder,
        optimized_anisotropy,
        optimized_cosine_similarity,
        optimized_softmax_2d,
        optimized_dynamic_weighted_ensemble,
        optimized_qkv_weight_grad,
        optimized_qkv_input_grad,
        optimized_lstm_cell_backward,
    )
    _OPT_AVAILABLE = True
except ImportError:
    _OPT_AVAILABLE = False

try:
    import abstract_weights_core as wc
    _RUST_MODULE_AVAILABLE = True
except ImportError:
    _RUST_MODULE_AVAILABLE = False

from .common import *  # noqa: F401,F403
from . import activations
from . import distributed_inference
from . import explainability
from . import lstm
from . import mlp
from . import model_storage
from . import pipeline
from . import query_node
from . import transformer

class WeightedEnsemblePredictor:
    """
    Combines predictions from the transformer.Transformer (model2), the mlp.MLP (model3), and
    optionally the lstm.LSTMEngine into a single probability distribution.

    This is the central "decision fusion" layer of the pipeline: instead of
    trusting any one model, it looks at how confident/focused each model is
    on a given input and blends their outputs accordingly. Several blending
    strategies are implemented (equal average, confidence-weighted,
    attention-weighted, a small learned "meta" weighting, and a
    hand-tuned "dynamic" heuristic — see `predict_ensemble` for the
    dispatch logic).

    It also owns:
      - An `explainability.ExplainabilityModule` instance, used to turn raw probabilities
        into a human-readable explanation + confidence summary.
      - A `model_storage.ModelStorage`/`query_node.QueryNode` pair, used to persist and recall past
        attention states ("attention memory") so that similar future inputs
        can reuse a previous decision instead of recomputing it, and to
        broker peer-assistance requests when the pipeline is unsure.

    Attributes:
        pipeline: The parent `pipeline.IntegratedPipeline`, giving access to model2
            (transformer.Transformer), model3 (mlp.MLP), lstm_engine, and shared config
            (e.g. `confidence_threshold`).
        storage (model_storage.ModelStorage): Persistence layer for attention memory.
        inference: The `distributed_inference.AgentDistributedInference` (P2P) handler used to
            escalate ambiguous predictions to peer agents.
        query_node (query_node.QueryNode): Used to check whether a peer/node
            "agreement" has been established before doing extra
            explainability work.
        transformer_weight / mlp_weight (float): Manually calibrated
            ensemble weights (see `calibrate_weights`); used by the
            "calibration" ensemble method.
        calibration_history (list): Reserved for tracking calibration runs
            over time (populated elsewhere/left for future use).
        explainer (explainability.ExplainabilityModule): Produces per-prediction
            explanations and confidence scores.
        memory_name (str): Key/namespace used when reading and writing
            attention memory via `storage`.
        error_counts / pred_counts: Reserved counters for adaptive error
            tracking (not initialized here beyond None; set/consumed
            elsewhere).
        error_decay (float): Decay factor intended for exponentially
            discounting older errors when they are tracked.
        self_attn_weights: Cache of the most recently seen attention
            weights, used as a fallback when no matching memory entry is
            found (see `attention_memory_gate`).
        memory (dict): The loaded attention-memory dict (or `{}` if this is
            the first time this `memory_name` is used), keyed by memory
            tags (e.g. `'TA'` for "text attention").
    """

    def __init__(self, pipeline, distribution, memory_name):
        self.pipeline = pipeline
        self.storage = model_storage.ModelStorage(pipeline, memory_name, db_path='activity_log.db')
        self.inference = distribution
        self.query_node = query_node.QueryNode(pipeline, memory_name, self.storage)

        self.transformer_weight = 0.5  # Initial equal weight
        self.mlp_weight = 0.5 # initial equal mlp weight
        self.calibration_history = []
        self.explainer = explainability.ExplainabilityModule(pipeline, self)
        self.memory_name = memory_name
        self.db_path = 'activity_log.db'

        self.error_counts = None
        self.pred_counts = None
        self.error_decay = 0.85

        self.self_attn_weights = None

        # Load previously saved attention memory for this memory_name, if any
        # exists on disk/DB. This lets the ensemble reuse past attention
        # states for near-duplicate inputs (see attention_memory_gate) rather
        # than always recomputing from scratch.
        if not self.storage.memory_exists(self.memory_name, type='transformer.Transformer'):
            self.memory = {}
        else:
            self.memory = self.storage.memory_retrieval(self.memory_name, type_func='transformer.Transformer', verbose=True)

    def _get_lstm_probs(self, input_ids, X_mlp, label_bins=None, confidence_level=0.90):
        """
        Convert lstm.LSTMEngine prediction output into a probability
        distribution compatible with trans_probs and mlp_probs
        for use in _dynamic_weighted_ensemble.

        Returns:
            lstm_probs : np.ndarray (batch_size, n_classes) or None if engine not ready
            lstm_weight_hint : float — calibrated confidence scalar for weighting
        """
        engine = self.pipeline.lstm_engine

        # guard — engine must be calibrated before predict() is callable
        if engine is None or engine.residual_std is None:
            print('[-] LSTM engine not ready, skipping LSTM probs')
            return None, 0.0

        # X_mlp is (batch, features) — LSTM expects (T, input_size) per sample
        # treating each feature vector as a single timestep sequence
        if X_mlp.ndim == 1:
            X_mlp = X_mlp.reshape(1, -1)

        batch_size = X_mlp.shape[0]
        n_classes = self.pipeline.model2.output.shape[1]  # align with transformer output

        lstm_probs = np.zeros((batch_size, n_classes))
        overall_confidences = []

        for i in range(batch_size):
            # shape each sample into (T=1, input_size) — single timestep
            x_seq = X_mlp[i].reshape(1, -1)

            try:
                result = engine.predict(
                    x_seq,
                    label_bins=label_bins,
                )
            except AssertionError:
                # calibrate() not called yet — skip gracefully
                print(f'[-] LSTM engine not calibrated for sample {i}, skipping')
                return None, 0.0

            # result['prediction'] is (T,) — take last timestep as scalar score
            raw_score = float(result['prediction'][-1])
            overall_confidences.append(result['overall'])

            # convert scalar score to class probabilities
            # label_confidence gives {label: prob} if label_bins were passed
            if result['label_confidence'] is not None:
                # map label_bins order to class indices
                label_probs = list(result['label_confidence'].values())
                n_label = len(label_probs)

                row = np.zeros(n_classes)
                row[:min(n_label, n_classes)] = label_probs[:n_classes]

                # renormalize in case n_label != n_classes
                row_sum = row.sum()
                if row_sum > 0:
                    row /= row_sum
                else:
                    row[0] = 1.0  # fallback — assign all mass to class 0

            else:
                # no label_bins — use mc_confidence at last timestep as a
                # soft signal: spread probability mass using mc_mean as logit
                mc_mean_last = float(result['mc_mean'][-1])
                mc_conf_last = float(result['mc_confidence'][-1])

                # build a soft peaked distribution using the score as a logit
                logits = np.full(n_classes, -mc_conf_last)
                target_class = int(np.clip(round(raw_score), 0, n_classes - 1))
                logits[target_class] = mc_conf_last

                # softmax here 
                logits -= logits.max()
                row = np.exp(logits)
                row /= row.sum()

            lstm_probs[i] = row

        # weight hint — average overall confidence across batch
        # lower residual_std = engine is well calibrated = higher weight earned
        mean_overall = float(np.mean(overall_confidences))
        lstm_weight_hint = mean_overall / (1.0 + engine.residual_std)

        return lstm_probs, lstm_weight_hint


    def attention_memory_gate(self, probs, x):
        """
        Look up whether a near-duplicate input has already been processed,
        by comparing `x` against every stored 'TA' (text-attention) memory
        entry via cosine similarity (threshold >= 0.85).

        This is a cache-lookup, not a computation: it never runs the models,
        it only decides whether a previous result can be reused.

        Args:
            probs: Unused placeholder kept for call-site symmetry with other
                methods that take `probs` (this method itself doesn't need
                a probability vector to do the lookup).
            x: The current input representation to compare against stored
                memory keys via `pipeline.cosine_similarity`.

        Returns:
            A 4-tuple. On a cache hit: (texts, x2, x3, x4) — the cached
            payload for the matching memory entry (see `modular_attention_saving`
            for what x2/x3/x4 represent: sanitized mlp.MLP probs / transformer
            probs / attention weights). On a cache miss but with a
            recent self-attention state available: (None, None, None,
            attn_weights). On a full miss: (None, None, None, None).

        Note:
            If multiple memory entries match, the loop below silently keeps
            iterating and only the *last* match's values are returned — this
            is existing behavior, not new.
        """
        memory = self.memory
        cache_attn_memory = [key for key, (_, inp, _, _, _) in memory.items() if key.startswith('TA') and self.pipeline.cosine_similarity(x, inp) >= 0.85]

        if cache_attn_memory:
            print('[+] Found matching attention memory!')
            for memo in cache_attn_memory:
                texts, _, x2, x3, x4 = memory[memo]

            return texts, x2, x3, x4

        else:
            print('🔄 No Matching Attention Weights!')
            if self.self_attn_weights is not None:
                print('|| Using current attention weights because of no matches found.')
                attn_weights = self.self_attn_weights
                return None, None, None, attn_weights

            return None, None, None, None

    def _sanitize_for_storage(self, obj, _depth=0, _max_depth=10):
        """
        Recursively strip Ellipsis objects and '...' string artifacts
        from any structure before saving to database/memory.

        """
        if _depth > _max_depth:
            print(f'[⚠️] _sanitize_for_storage: max depth {_max_depth} reached, '
                f'truncating to avoid infinite recursion')
            return None

        # Case 1 — literal Ellipsis object
        if obj is Ellipsis:
            print('[⚠️] _sanitize_for_storage: found literal Ellipsis, replacing with None')
            return None

        # Case 2 — string containing "..." artifacts
        if isinstance(obj, str):
            if obj.strip() == '...':
                return None
            if '...' in obj:
                cleaned = obj.replace('...', '').strip()
                if cleaned != obj:
                    print(f'[⚠️] _sanitize_for_storage: stripped "..." from string: '
                        f'"{obj[:40]}..." → "{cleaned[:40]}"')
                return cleaned if cleaned else None
            return obj

        # Case 3 — numpy array — check for object dtype containing Ellipsis
        if isinstance(obj, np.ndarray):
            if obj.dtype == object:
                flat = obj.ravel()
                has_ellipsis = any(v is Ellipsis for v in flat)
                if has_ellipsis:
                    print(f'[⚠️] _sanitize_for_storage: array contains Ellipsis '
                        f'objects, replacing with 0.0')
                    cleaned = np.array([
                        0.0 if v is Ellipsis else v for v in flat
                    ]).reshape(obj.shape)
                    return cleaned
            # numeric arrays never contain Ellipsis
            return obj

        # Case 4 — dict — recurse into keys/values
        if isinstance(obj, dict):
            cleaned = {}
            for k, v in obj.items():
                # keys should never legitimately be Ellipsis or "..."
                if k is Ellipsis or (isinstance(k, str) and k.strip() == '...'):
                    print(f'[⚠️] _sanitize_for_storage: dropping key that is Ellipsis/"..."')
                    continue
                cleaned_v = self._sanitize_for_storage(v, _depth + 1, _max_depth)
                if cleaned_v is not None or v is None:
                    cleaned[k] = cleaned_v
            return cleaned

        # Case 5 — list/tuple — recurse into elements
        if isinstance(obj, (list, tuple)):
            cleaned = [
                self._sanitize_for_storage(item, _depth + 1, _max_depth)
                for item in obj
            ]
            # remove None entries that came FROM ellipsis stripping,
            # but preserve legitimately-None entries at the same position
            # by only dropping items that were Ellipsis/"..." originally
            result = [c for c, orig in zip(cleaned, obj)
                    if not (orig is Ellipsis or
                            (isinstance(orig, str) and orig.strip() == '...'))]
            return tuple(result) if isinstance(obj, tuple) else result

        # everything else (int, float, bool, None) — pass through unchanged
        return obj           

    def modular_attention_saving(self, text, X, X2, X3, X4):
        """
        Persist the current attention/prediction context under the 'TA' key
        so that `attention_memory_gate` can retrieve it for similar future
        inputs.

        All array-like payloads (X, X2, X3, X4) are passed through
        `_sanitize_for_storage` first to strip any Ellipsis/'...' artifacts
        that could otherwise corrupt the DB entry.

        Args:
            text: The raw text (or texts) associated with this memory entry.
            X: Primary array payload (e.g. input embedding) — sanitized then
                stored, but notably the sanitized copy of X (`clean_X`) is
                *not* the one written into memory (see note below).
            X2, X3, X4: Additional payloads (by convention elsewhere: mlp.MLP
                probs, transformer probs, attention weights) — sanitized and
                stored.

        Note:
            The tuple written to `self.memory['TA']` is
            `(clean_X, text, clean_X2, clean_X3, clean_X4)` — i.e. `text`
            takes the second slot, not `clean_X`'s neighbor as the naming
            might suggest. `attention_memory_gate` unpacks with the matching
            order `(_, inp, _, _, _)` / `(texts, _, x2, x3, x4)`, so this is
            consistent with the reader, just worth knowing if you extend it.
        """
        memory_name = self.memory_name

        clean_X = self._sanitize_for_storage(X)
        clean_X2 = self._sanitize_for_storage(X2)
        clean_X3 = self._sanitize_for_storage(X3)
        clean_X4 = self._sanitize_for_storage(X4)

        self.memory['TA'] = clean_X, text, clean_X2, clean_X3, clean_X4

        self.storage.save_model_dict(memory_name, self.memory, type='transformer.Transformer', model_type='attention')

        print('🚀 Memory Probability Added!')




    # NOTE: `explainability_prediction_batch` is defined twice in this class
    # (here, and again further below). Python keeps only the *second*
    # definition — this first one is dead code and is never actually called.
    # It's left in place/documented here for reference, but any bug fix
    # should go into the second definition (the one that actually runs).
    def explainability_prediction_batch(self, texts, mlp_probs, trans_probs, attn_weights, show_explanation=False):
        """
        [Shadowed duplicate — see the second `explainability_prediction_batch`
        definition below, which is the one Python actually binds/uses.]

        Per-text explainability pass that, unlike the second definition,
        indexes mlp_probs/trans_probs/attn_weights per-text (i.e. assumes a
        batch of probability rows, one per text) rather than passing the
        same probs to every text.
        """
        results = []
        for i, text in enumerate(texts):
            text_mlp_probs = mlp_probs[i] if i < len(mlp_probs) else mlp_probs
            text_trans_probs = trans_probs[i] if i < len(trans_probs) else trans_probs
            text_attn_weights = attn_weights[i] if i < len(attn_weights) else attn_weights

            result = self.predict_single(text, text_mlp_probs, text_trans_probs, text_attn_weights, show_explanation)
            results.append(result)
            explanation = result['explanation']
            print(f'[+] Explanation: {explanation}')
        
        return results
    

    def credibility_summarized_prediction(self, input_ids, mlp_probs, trans_probs, attn_weights, type=None):
        """
        Run the (shadowed/active) `explainability_prediction_batch` over the
        current batch of texts, print a summary (prediction distribution +
        average confidence), and persist the result into attention memory.

        Args:
            input_ids: Token ids for the current batch — only used when
                `type == 'transformer.Transformer'`, to re-query attention memory via
                `attention_memory_gate`.
            mlp_probs, trans_probs, attn_weights: Probability/attention
                tensors for the batch; may be overwritten below if a
                transformer.Transformer-memory lookup succeeds.
            type: When `'transformer.Transformer'`, attempts to pull texts/probs from
                attention memory first, falling back to `pipeline.texts`.
                Any other value skips the memory lookup and uses
                `pipeline.texts` directly.

        Side effects:
            Prints a batch summary (count, avg confidence, label
            distribution) and calls `modular_attention_saving` to persist
            the (possibly memory-recalled) context back into memory.
        """
        if type == 'transformer.Transformer':
            texts, mlp_probs, trans_probs, attn_weights = self.attention_memory_gate(input_ids)
            if not texts:
                texts = self.pipeline.texts
        else:
            texts = self.pipeline.texts

        results = self.explainability_prediction_batch(texts, mlp_probs, trans_probs, attn_weights)

        # Calculate summary
        predictions = [r['prediction'] for r in results]
        confidences = [r['confidence'] for r in results]
        
        distribution = Counter(predictions)
        
        print("\n📊 Batch Summary:")
        print(f"   Total: {len(results)} predictions")
        print(f"   Avg Confidence: {np.mean(confidences):.1%}")
        print(f"   Distribution: {dict(distribution)}")
        
        self.modular_attention_saving(input_ids, texts, mlp_probs, trans_probs, attn_weights )


    def explain_past_memory(self, probs, input_ids):
        """
        Try to resolve a prediction from attention memory; if no memory
        match exists, escalate to a peer agent for assistance instead of
        guessing blindly.

        Args:
            probs: Fallback probabilities to return if the peer-assistance
                path also fails.
            input_ids: The current input, used both as the memory-lookup key
                and forwarded to the peer request.

        Returns:
            Either an ensemble of the recalled memory (via
            `_dynamic_weighted_ensemble`) when a memory match is found, or
            whatever `inference._handle_peer_agent_request` returns (falling
            back to the original `probs` on error).

        Side effects:
            On failure to get a peer response, reports the failure via
            `inference.report_failure` so it can be tracked/surfaced
            upstream.
        """
        _, mlp_probs, trans_probs, attn_weights = self.attention_memory_gate(probs, input_ids)
        self.self_attn_weights = attn_weights

        self_attn_weights = self.self_attn_weights

        if trans_probs is not None:
            print('[+] Attention memory retrieved! ')
            method = 'memory_retrieval'

            self.credibility_summarized_prediction(input_ids, mlp_probs, trans_probs, attn_weights, type='pipeline')
            ensemble_probs = self._dynamic_weighted_ensemble(
                trans_probs, mlp_probs, attn_weights, input_ids
            )

            return ensemble_probs
        else:
            print("[-] Ambiguity present, Requesting peer assistance... ")
        
            try:
                probs = self.inference._handle_peer_agent_request(probs, self_attn_weights, input_ids, type='DevicePeer', agreement=False)
                return probs
            except Exception as e:
                print(f'|| Error initiating peer request: {e}, returning regular probs.')
                self.inference.report_failure(id(self), 'processing', reason=f'{e}')                        

                return probs


    def predict_single(self, text, mlp_probs, trans_probs, attn_weights, show_explanation=True, batch_size=2):
        """
        Produce a single explained prediction for one piece of text by
        delegating to `explainability.ExplainabilityModule._get_prediction_details`.

        If an LSTM result is already cached on the pipeline (from an earlier
        step in the current request), it's threaded through so the
        explanation can incorporate the LSTM's contribution too.

        Args:
            text: The input text this prediction is for (used for display /
                explanation context, not re-tokenized here).
            mlp_probs, trans_probs, attn_weights: Pre-computed probability /
                attention tensors for this single item.
            show_explanation: Passed through but not directly used in this
                method body (the explainer decides its own verbosity);
                kept for API symmetry with batch callers.
            batch_size: Caps how much of the attention data the explainer
                processes at once (see inline note) to avoid memory blowups.

        Returns:
            dict with keys: 'prediction' (final label), 'confidence'
            (final confidence score), 'explanation' (human-readable string),
            and 'details' (the full raw result dict from the explainer).
        """
        # small batch size is used to prevent memory overflow in explanation module when processing large attention weights, as it computes detailed explanations that can be memory intensive.
        cache = self.pipeline.cache
        if cache is not None and "lstm_result" in cache:
            lstm_result = self.pipeline.cache['lstm_result']
            result, explanation = self.explainer._get_prediction_details(text, mlp_probs, trans_probs, attn_weights, lstm_result=lstm_result, batch_size=batch_size)
        else:
            result, explanation = self.explainer._get_prediction_details(text, mlp_probs, trans_probs, attn_weights, batch_size=batch_size)            
        return {
            'prediction': result['final_label'],
            'confidence': result['final_confidence'],
            'explanation': explanation,
            'details': result
        }
    
    

    def explainability_prediction_batch(self, texts, mlp_probs, trans_probs, attn_weights, show_explanation=False):
        """
        [This is the definition that actually runs — Python overrides the
        earlier same-named method above with this one.]

        Explain every text in `texts` using the *same* mlp_probs/trans_probs/
        attn_weights for each call (unlike the shadowed definition above,
        this one does not index per-text — it broadcasts one shared set of
        probabilities across the whole batch of texts).

        Args:
            texts: Iterable of input texts to explain.
            mlp_probs, trans_probs, attn_weights: Shared probability/attention
                tensors applied to every text in this batch.
            show_explanation: Forwarded to `predict_single` (see its
                docstring — currently informational only at that layer).

        Returns:
            List of per-text result dicts, each shaped like the dict
            returned by `predict_single`.
        """
        results = []
        for text in texts:
            result = self.predict_single(text, mlp_probs, trans_probs, attn_weights, show_explanation)
            results.append(result)
            explanation = result['explanation']
            print(f'[+] Explanation: {explanation}')
        
        return results
    


    def predict_ensemble(self, input_ids, X_mlp, y_true, method='dynamic', embedded=False):
        """
        Main entry point for producing a fused prediction: runs the
        transformer.Transformer, mlp.MLP, and (if calibrated) LSTM engine, then combines
        their outputs using the strategy named by `method`.

        Args:
            input_ids: Token ids for the transformer.Transformer path (also passed to
                the LSTM helper as raw input for shaping purposes).
            X_mlp: Feature vector(s) for the mlp.MLP path (and, reshaped, for
                the LSTM path — see `_get_lstm_probs`).
            y_true: Ground-truth labels, forwarded to the mlp.MLP's forward
                pass (used e.g. for any loss-aware behavior inside model3).
            method: One of:
                - 'equal'       — plain average of transformer + mlp.MLP probs.
                - 'confidence'  — weight each model by its own max-prob
                                  confidence.
                - 'dynamic'     — heuristic blend based on model agreement,
                                  attention focus, and anisotropy (default;
                                  see `_dynamic_weighted_ensemble`).
                - 'attention'   — weight primarily by attention entropy
                                  (see `_attention_weighted_ensemble`).
                - 'meta'        — small hand-built "meta-features -> weight"
                                  heuristic passed through an AME encoder
                                  (see `_meta_ensemble`).
                - 'calibration' — grid-searches a fixed transformer/mlp.MLP
                                  split via `calibrate_weights`, then applies
                                  it through `_attention_weighted_ensemble`.
                Any other value raises `ValueError`.
            embedded: Passed through to the transformer.Transformer's forward pass to
                indicate whether `input_ids` are already embeddings.

        Returns:
            Tuple of `(ensemble_probs, details_dict)` where `details_dict`
            contains the raw 'transformer', 'mlp', 'ensemble' arrays and the
            `method` used. If the resulting ensemble is degenerate (all-NaN/
            inf), falls back to returning the mlp.MLP probabilities alone with
            `'method': None` in the details.

        Side effects:
            If a node "agreement" is established (via `query_node`) and the
            pipeline has explainability display enabled, this also triggers
            a full explainability pass (`credibility_summarized_prediction`),
            which prints a batch summary and updates attention memory.
        """
        label_bins=None
        if self.pipeline.cache and 'label_bins' in self.pipeline.cache:
            print('[=] label_bins cache found!')
            label_bins = self.pipeline.cache['label_bins']

        AME = self.pipeline.model2.AME_Encoder(input_ids)

        trans_probs, attn_weights = self.pipeline.model2.forward(input_ids, AME=AME, embedded=embedded)
        mlp_probs = self.pipeline.model3.forward(X_mlp, y=y_true)
        lstm_probs, lstm_weight_hint = self._get_lstm_probs(input_ids, X_mlp, label_bins=label_bins)

        established_agreement = self.query_node._establish_node_connection("PredictEnsemble")
        
        if method == 'equal':
            # Simple average
            ensemble_probs = (trans_probs + mlp_probs) / 2
            
        elif method == 'confidence':
            # Weight by confidence (max probability)
            trans_conf = np.max(trans_probs, axis=1, keepdims=True)
            mlp_conf = np.max(mlp_probs, axis=1, keepdims=True)
            
            # Normalize weights
            total_conf = trans_conf + mlp_conf + 1e-8
            trans_weight = trans_conf / total_conf
            mlp_weight = mlp_conf / total_conf
            
            ensemble_probs = trans_weight * trans_probs + mlp_weight * mlp_probs
            
        elif method == 'dynamic':
            # Dynamic weighting based on agreement and attention
            ensemble_probs = self._dynamic_weighted_ensemble(
                trans_probs, mlp_probs, attn_weights, input_ids,
                lstm_probs=lstm_probs,
                lstm_weight_hint=lstm_weight_hint
            )              
            
        elif method == 'attention':
            # Use attention to weight transformer vs mlp.MLP
            ensemble_probs = self._attention_weighted_ensemble(
                trans_probs, mlp_probs, attn_weights
            )
            
        elif method == 'meta':
            # Meta-learner that decides weights
            ensemble_probs = self._meta_ensemble(
                trans_probs, mlp_probs, attn_weights, X_mlp,
                lstm_probs=lstm_probs,
                lstm_weight_hint=lstm_weight_hint
            )
        
        elif method == 'calibration':
            calibrated_weight = self.calibrate_weights(input_ids, X_mlp, y_true, step=3)  
            ensemble_probs = self._attention_weighted_ensemble(
                trans_probs, mlp_probs, calibrated_weight
            )
                                    
        else:
            print(f"[=] Unknown method: {method}")
            raise ValueError("Invalid ensemble method!")            

        
        if established_agreement and self.pipeline.show_explainability_details:
            print('[✅] Agreement established, generating explainability features.')
            try:
                print('=== COMPLETE EXPLAINABILITY PREDICTION ==') 
                self.credibility_summarized_prediction(input_ids, mlp_probs, trans_probs, attn_weights, type='pipeline')
            except Exception as e:
                print(f'[-] Cant get explainability features! : {e}')
                traceback.print_exc()
        else:
            print('[-] No agreement established, skipping explainability features.')


        try:
            ensemble_probs = ensemble_probs / ensemble_probs.sum(axis=1, keepdims=True)
        except:
            ensemble_probs = ensemble_probs / ensemble_probs.sum()

        if ensemble_probs is None or np.isnan(ensemble_probs).any() or np.isinf(ensemble_probs).any():
            print('[-] Ensemble probs is invalid , using mlp.MLP probs as.')
            return mlp_probs, {
            'transformer': trans_probs,
            'mlp': mlp_probs,
            'ensemble': mlp_probs,
            'method': None              
            }   

        return ensemble_probs, {
            'transformer': trans_probs,
            'mlp': mlp_probs,
            'ensemble': ensemble_probs,
            'method': method
        }
        
    def anisotropy_measurement(self, x):
        """
        Estimate how "directionally uneven" (anisotropic) an array's local
        gradients are, used as a proxy for how peaky/reliable an attention
        map is (higher anisotropy ~ gradients vary a lot in magnitude across
        the array ~ more structured/confident signal).

        Computed as std(gradient norms) / mean(gradient norms), i.e. the
        coefficient of variation of the gradient-magnitude field. Falls back
        through several strategies if the primary computation fails (a
        smaller sub-block gradient, then a flat default), and ultimately
        falls back to `pipeline.confidence_threshold` if everything errors
        out — the goal is to always return *some* usable scalar rather than
        raise, since this feeds directly into ensemble weighting.

        Args:
            x: Array-like input (typically an attention map) to measure.

        Returns:
            float anisotropy score (higher = more anisotropic/structured).
            Uses a Cython/optimized implementation (`optimized_anisotropy`)
            when available for speed, with an identical pure-Python fallback
            below it.
        """
        eps = 1e-5
        if _OPT_AVAILABLE:
            x = np.asarray(x)            
            x = x.reshape(x.shape[0], -1)            
            return optimized_anisotropy(np.asarray(x, dtype=np.float64))

        try:
            try:
                grads = np.gradient(x)
                # Stack gradients into a single array of vectors and find the norm of each
                # automatically handles multi-dimensional arrays (e.g., 2D, 3D volumes)
                stacked_grads = np.stack(grads, axis=-1)
                norms = np.linalg.norm(stacked_grads, axis=-1)
                
                # Safely filter out potential NaNs or infs (common at array boundaries)
                valid_norms = norms[np.isfinite(norms)]
                
                if len(valid_norms) == 0:
                    return 0.0 # Return zero or appropriate default if no valid values exist
                    
                # Calculate statistics using the clean data
                std_val = np.std(valid_norms)
                mean_val = np.mean(valid_norms)
                
                anisotropy = std_val / (mean_val + eps)    
                if np.isnan(anisotropy) or np.isinf(anisotropy):
                    anisotropy = self.pipeline.confidence_threshold
            except:
                try:
                    gradient = np.gradient(x)
                except:
                    subnet = x[:min(10, x.shape[0]), :min(10, x.shape[1])]
                    gradient = np.gradient(subnet.flatten())

                val = [np.linalg.norm(v) for v in gradient]
                anisotropy = np.std(val) / np.mean(val) + eps 
        except Exception as e:
            print(f'[!] Cant calculate anisotropy due to: {e}')   
            anisotropy = self.pipeline.confidence_threshold

        return anisotropy
        


    def _dynamic_weighted_ensemble(self, trans_probs, mlp_probs, attn_weights,
                                    input_ids, lstm_probs=None, lstm_weight_hint=0.0):
        """
        The default ('dynamic') ensembling strategy. For every sample in the
        batch, derives a per-model trust weight from several signals and
        blends the (zero-padded, class-aligned) probability rows
        accordingly:

          - transformer.Transformer weight (`trans_cf`/`tw`) grows with attention focus
            (low entropy across the attention map -> `attn_growth` high) and
            with the anisotropy of that attention map, then gets boosted
            when the transformer and mlp.MLP predictions already agree.
          - mlp.MLP weight (`mw`) grows with how low-entropy (peaked) the mlp.MLP's
            own output distribution is, scaled by `mean_pred_counts` (an
            mlp.MLP-side calibration signal) and by model agreement.
          - LSTM weight (`lw`), when available, uses the externally supplied
            `lstm_weight_hint` (from `_get_lstm_probs`, itself a function of
            the engine's calibration quality) boosted when the LSTM's own
            top prediction matches either other model.

        All three weights are then renormalized to sum to 1 per sample and
        used as a convex combination of the (renormalized) probability rows.

        A fast compiled path (`optimized_dynamic_weighted_ensemble`) is used
        when available; the remainder of the method is an equivalent pure
        NumPy fallback that runs sample-by-sample.

        Args:
            trans_probs, mlp_probs: transformer.Transformer/mlp.MLP probability arrays,
                shape (batch, n_classes_x) — class counts may differ and are
                zero-padded up to the max before combining.
            attn_weights: Per-sample attention weights/maps used to derive
                `attn_focus` (via std) and anisotropy; if None, a flat 0.5
                is used for every sample.
            input_ids: Unused directly in this method body (kept for a
                consistent call signature with sibling ensemble methods).
            lstm_probs: Optional LSTM probability array; if None, the LSTM
                branch is skipped entirely (`has_lstm=False`).
            lstm_weight_hint: Scalar or per-sample array of LSTM confidence
                hints (see `_get_lstm_probs`).

        Returns:
            np.ndarray of shape (batch, n_classes) — the blended ensemble.
            On any exception, logs the error and safely falls back to a copy
            of `mlp_probs`.
        """
        # normalize all inputs to guaranteed 2D float64
        mean_pred_counts = np.mean(self.pipeline.model3.pred_counts)
        try:
            trans_probs = np.asarray(trans_probs, dtype=np.float64)
            mlp_probs   = np.asarray(mlp_probs,   dtype=np.float64)

            if trans_probs.ndim == 1: trans_probs = trans_probs[np.newaxis, :]
            if mlp_probs.ndim   == 1: mlp_probs   = mlp_probs[np.newaxis, :]

            B = trans_probs.shape[0]

            n_trans = trans_probs.shape[1]
            n_mlp   = mlp_probs.shape[1]

            has_lstm = lstm_probs is not None
            if has_lstm:
                lstm_probs = np.asarray(lstm_probs, dtype=np.float64)
                if lstm_probs.ndim == 1: lstm_probs = lstm_probs[np.newaxis, :]
                n_lstm = lstm_probs.shape[1]
            else:
                lstm_probs = np.zeros((B, 1), dtype=np.float64)  # dummy, not used
                n_lstm = 0

            n_classes = max(n_trans, n_mlp, n_lstm)
            print(f"🔄 Aligning classes: trans={n_trans} mlp={n_mlp} "
                f"lstm={n_lstm} → {n_classes}")

            if attn_weights is not None:
                attn_arr = np.asarray(attn_weights, dtype=np.float64)
                attn_flat = attn_arr.reshape(B, -1) if attn_arr.ndim > 1 \
                            else np.tile(attn_arr.ravel(), (B, 1))
            else:
                attn_flat = np.full((B, 1), 0.5, dtype=np.float64)

            # per-sample lstm_weight_hints
            if isinstance(lstm_weight_hint, (int, float)):
                lstm_weight_hints = np.full(B, float(lstm_weight_hint), dtype=np.float64)
            else:
                lstm_weight_hints = np.asarray(lstm_weight_hint, dtype=np.float64)
                if lstm_weight_hints.ndim == 0:
                    lstm_weight_hints = np.full(B, float(lstm_weight_hints))

                
            if _OPT_AVAILABLE:
                try:
                    print('[+] Using Optimized Dynamic weighted ensemble Method.')
                    return optimized_dynamic_weighted_ensemble(
                        np.ascontiguousarray(trans_probs),
                        np.ascontiguousarray(mlp_probs),
                        np.ascontiguousarray(attn_flat),
                        np.ascontiguousarray(lstm_probs),
                        np.ascontiguousarray(lstm_weight_hints),
                        float(self.pipeline.confidence_threshold),
                        has_lstm,
                        mean_pred_counts
                    )
                except Exception as e:
                    print(f'[=] Error in optimized dynamic weighted ensemble: {e}, using regular dynamic ensemble method.')
                    pass

            # pure Python fallback 
            ensemble = np.zeros((B, n_classes))
            for i in range(B):
                trans_row = np.zeros(n_classes)
                mlp_row   = np.zeros(n_classes)
                trans_row[:n_trans] = trans_probs[i]
                mlp_row[:n_mlp]     = mlp_probs[i]
                trans_row /= trans_row.sum() + 1e-8
                mlp_row   /= mlp_row.sum()   + 1e-8

                trans_pred = int(np.argmax(trans_probs[i]))
                mlp_pred   = int(np.argmax(mlp_probs[i]))
                agreement  = 1.0 if trans_pred == mlp_pred else 0.3

                attn         = attn_flat[i]
                attn_focus   = float(np.std(attn)) if attn.size > 1 else 0.5
                attn_growth  = 1.0 / (1.0 + np.exp(-attn_focus))
                anisotropy   = self.anisotropy_measurement(attn.reshape(1, -1))
                attn_limit   = (1.0 - attn_focus + attn_growth) * anisotropy
                trans_cf     = attn_growth + attn_limit * attn_focus

                mlp_entropy  = -np.sum(mlp_probs[i] * np.log(mlp_probs[i] + 1e-8))
                mlp_cf       = 1.0 / (1.0 + mlp_entropy)

                tw = trans_cf * (1.0 + agreement) / 2.0
                mw = mlp_cf  * mean_pred_counts * (1.0 + agreement) / 2.0

                if has_lstm:
                    lstm_row = np.zeros(n_classes)
                    lstm_row[:n_lstm] = lstm_probs[i]
                    lstm_row /= lstm_row.sum() + 1e-8
                    lstm_pred = int(np.argmax(lstm_probs[i]))
                    la = 1.0 if (lstm_pred == trans_pred or lstm_pred == mlp_pred) \
                        else self.pipeline.confidence_threshold
                    lw    = float(lstm_weight_hints[i]) * (1.0 + la) / 2.0
                    total = tw + mw + lw + 1e-8
                    ensemble[i] = (tw/total) * trans_row + \
                                (mw/total) * mlp_row   + \
                                (lw/total) * lstm_row
                else:
                    total = tw + mw + 1e-8
                    ensemble[i] = (tw/total) * trans_row + (mw/total) * mlp_row

            return ensemble
        
        except Exception as e:
            print(f'[!] Cant do ensemble prediction due ensemble prediction due to: {e}, returning mlp.MLP probabilities')
            return mlp_probs.copy()


    
    def _attention_weighted_ensemble(self, trans_probs, mlp_probs, attn_weights):
        """
        Ensembling strategy driven primarily by attention entropy: a
        low-entropy (sharply focused) attention distribution is taken as
        evidence the transformer.Transformer should be trusted more; the mlp.MLP receives
        whatever trust "budget" is left over (scaled by `mean_pred_counts`).

        Args:
            trans_probs, mlp_probs: Probability arrays, shape
                (batch, n_classes_x); zero-padded to a common class count
                before blending.
            attn_weights: Per-sample attention arrays. If `None` entirely,
                short-circuits to a plain 50/50 average of the two inputs.
                If present but a given sample's attention array is empty
                (`.size == 0`), falls back to a activations.sigmoid-based estimate
                combined with the batch-level anisotropy instead of entropy.

        Returns:
            np.ndarray of shape (batch, n_classes) with the blended
            probabilities (one row per sample, transformer-trust +
            mlp-trust convex combination).
        """
        mean_pred_counts = np.mean(self.pipeline.model3.pred_counts)
        if attn_weights is None:
            return (trans_probs + mlp_probs) / 2
        
        batch_size = trans_probs.shape[0]
        ensemble = np.zeros_like(trans_probs)

        n_trans_classes = trans_probs.shape[1]        
        n_mlp_classes = mlp_probs.shape[1]
        
        anisotropy = self.anisotropy_measurement(attn_weights)            
        n_classes = max(n_trans_classes, n_mlp_classes)       
        for i in range(batch_size):
            trans_row = np.zeros(n_classes)
            mlp_row = np.zeros(n_classes)         
            trans_row[:n_trans_classes] = trans_probs[i]
            mlp_row[:n_mlp_classes] = mlp_probs[i]
            
            trans_row = trans_row / (trans_row.sum() + 1e-8)
            mlp_row = mlp_row / (mlp_row.sum() + 1e-8)

            if i < len(attn_weights):
                attn = attn_weights[i]              
                # Attention entropy: lower entropy = more focused = trust transformer more
                if attn.size > 0:
                    attn_flat = attn.flatten()
                    attn_entropy = -np.sum(attn_flat * np.log(attn_flat + 1e-8)) / np.log(len(attn_flat))
                    trans_trust = 1.0 - attn_entropy  # 0 to 1
                else:
                    attn_focus = 1.0 / (1.0 + np.exp(-attn))
                    trans_trust = attn_focus * anisotropy
            else:
                attn_focus = 1.0 / (1.0 + np.exp(-attn))
                attn_limit = 1.0 - np.exp(-attn_focus)
                trans_trust = attn_limit * (1.0 - anisotropy)
            
            # mlp.MLP gets the rest
            mlp_trust = (1.0 - trans_trust) * mean_pred_counts

            try:
                ensemble[i] = trans_trust * trans_row + mlp_trust * mlp_row 
            except:
                ensemble = trans_trust * trans_row + mlp_trust * mlp_row 
        
        return ensemble
    
    def _meta_ensemble(self, trans_probs, mlp_probs, attn_weights, X_mlp, 
                         lstm_probs=None, lstm_weight_hint=0.0):
        """
        A lightweight "meta-learner-style" ensembling strategy: builds a
        small hand-crafted feature vector per sample (per-model confidence,
        per-model spread/std, cross-model agreement flag, attention std/max)
        and pushes it through the pipeline's `AME_Encoder` + activations.sigmoid to get
        a single scalar "agreement-boosted base weight" per sample, which is
        then used to split trust between the transformer and mlp.MLP (whichever
        of the two has higher raw confidence gets `base_weight`, the other
        gets `1 - base_weight`). Note `X_mlp` is accepted for interface
        symmetry but is not itself fed into the feature vector — the meta
        features are derived purely from the models' own outputs.

        Args:
            trans_probs, mlp_probs: Probability arrays to combine, shape
                (batch, n_classes_x); zero-padded to a common class count.
            attn_weights: Optional per-sample attention arrays contributing
                two extra meta-features (std, max); replaced with a neutral
                `threshold_feature` placeholder when unavailable.
            X_mlp: The mlp.MLP's raw feature input — accepted but unused in the
                current feature construction (see note above).
            lstm_probs, lstm_weight_hint: Optional LSTM branch, folded into
                the final blend the same way as in `_dynamic_weighted_ensemble`
                (weight derived from `lstm_weight_hint`, boosted on
                agreement with either other model).

        Returns:
            np.ndarray of shape (batch, n_classes) with the blended
            probabilities.
        """
        lstm_row = None
        
        if trans_probs.ndim == 1: trans_probs = trans_probs[np.newaxis, :]
        if mlp_probs.ndim   == 1: mlp_probs   = mlp_probs[np.newaxis, :]

        B = trans_probs.shape[0]

        batch_size = trans_probs.shape[0]
        threshold_feature = 0.1 + self.pipeline.confidence_threshold
    
        n_trans_classes = trans_probs.shape[1]        
        n_mlp_classes = mlp_probs.shape[1]

        has_lstm = lstm_probs is not None
        if has_lstm:
            lstm_probs = np.asarray(lstm_probs, dtype=np.float64)
            if lstm_probs.ndim == 1: lstm_probs = lstm_probs[np.newaxis, :]
            n_lstm = lstm_probs.shape[1]
        else:
            lstm_probs = np.zeros((B, 1), dtype=np.float64)  # dummy, not used
            n_lstm = 0
 
        n_classes = max(n_trans_classes, n_lstm, n_mlp_classes)
        print(f"🔄 Aligning classes: trans={n_trans_classes} mlp={n_mlp_classes} "
            f"lstm={n_lstm} → {n_classes}") 
            
        if isinstance(lstm_weight_hint, (int, float)):
            lstm_weight_hints = np.full(B, float(lstm_weight_hint), dtype=np.float64)
        else:
            lstm_weight_hints = np.asarray(lstm_weight_hint, dtype=np.float64)
            if lstm_weight_hints.ndim == 0:
                lstm_weight_hints = np.full(B, float(lstm_weight_hints))

        # Create meta features
        meta_features = []
        for i in range(batch_size):
            trans_row = np.zeros(n_classes)
            mlp_row = np.zeros(n_classes)
             
            trans_row[:n_trans_classes] = trans_probs[i]
            mlp_row[:n_mlp_classes] = mlp_probs[i]
            if has_lstm:
                lstm_row = np.zeros(n_classes)
                lstm_row[:n_lstm] = lstm_probs[i]
                lstm_row = lstm_row / (lstm_row.sum() + 1e-8)

            trans_row = trans_row / (trans_row.sum() + 1e-8)
            mlp_row = mlp_row / (mlp_row.sum() + 1e-8)

            if lstm_row is None:
                features = [
                    np.max(trans_row),           # transformer.Transformer confidence
                    np.max(mlp_row),              # mlp.MLP confidence
                    np.std(trans_row),             # transformer.Transformer spread
                    np.std(mlp_row),               # mlp.MLP spread
                    1.0 if np.argmax(trans_row) == np.argmax(mlp_row) else 0.0,  # Agreement
                ]
            else:
                features = [
                    np.max(trans_row),           # transformer.Transformer confidence
                    np.max(mlp_row),              # mlp.MLP confidence
                    np.max(lstm_row),
                    np.std(trans_row),             # transformer.Transformer spread
                    np.std(mlp_row),               # mlp.MLP spread
                    np.std(lstm_row),
                    1.0 if np.argmax(trans_row) == np.argmax(mlp_row) else 0.0,  # Agreement
                ]    

            # Add attention stats if available
            if attn_weights is not None and i < len(attn_weights):
                attn = attn_weights[i]
                if attn.size > 0:
                    features.append(np.std(attn))
                    features.append(np.max(attn))
                else:
                    features.extend([threshold_feature, threshold_feature])
            else:
                features.extend([threshold_feature, threshold_feature])
            
            meta_features.append(features)
        
        meta_features = np.array(meta_features) 
        featured_AME = self.pipeline.AME_Encoder(meta_features) 
        AME_sigmoid = 1.0 / (1.0 + np.exp(-featured_AME))

        ensemble = np.zeros_like(trans_probs)
        
        for i in range(batch_size):
            # Calculate weight based on meta features
            trans_conf = meta_features[i, 0]
            mlp_conf = meta_features[i, 1]
            agreement_idx = 6 if has_lstm else 4
            agreement = meta_features[i, agreement_idx]
            
            trans_pred = int(np.argmax(trans_probs[i]))
            mlp_pred   = int(np.argmax(mlp_probs[i])) 
                                  
            # Boost weight when models agree
            base_weight = threshold_feature + AME_sigmoid * agreement
            
            # Adjust based on relative confidence
            if trans_conf > mlp_conf:
                trans_weight = base_weight
                mlp_weight = 1.0 - base_weight
            else:
                trans_weight = 1.0 - base_weight
                mlp_weight = base_weight
                
            if has_lstm:
                if lstm_row is None:
                    lstm_row = np.zeros(n_classes)
                    lstm_row[:n_lstm] = lstm_probs[i]
                    lstm_row /= lstm_row.sum() + 1e-8

                lstm_pred = int(np.argmax(lstm_probs[i]))
                la = 1.0 if (lstm_pred == trans_pred or lstm_pred == mlp_pred) \
                    else self.pipeline.confidence_threshold
                lw    = float(lstm_weight_hints[i]) * (1.0 + la) / 2.0
                total = trans_weight + mlp_weight + lw + 1e-8
                ensemble[i] = (trans_weight/total) * trans_row + \
                            (mlp_weight/total) * mlp_row   + \
                            (lw/total) * lstm_row
            else:
                try:
                    ensemble[i] = trans_weight * trans_row + mlp_weight * mlp_row 
                except:
                    ensemble = trans_weight * trans_row + mlp_weight * mlp_row                
        
        return ensemble
    
    def calibrate_weights(self, input_ids, X_mlp, y_true, step=3):
        """
        Simple grid search over a fixed transformer/mlp.MLP split (11 evenly
        spaced weights from 0.0 to 1.0) to find the mixture that maximizes
        accuracy on the current batch, run for `step` repetitions per
        candidate weight to average out noise. Used by the 'calibration'
        ensemble method in `predict_ensemble`.

        Args:
            input_ids: Token ids used to compute the (cached) AME encoding
                and re-run the transformer forward pass at each step.
            X_mlp: mlp.MLP feature input, forwarded to `pipeline.mlp.forward`.
            y_true: One-hot (or similar) ground-truth labels used to compute
                accuracy for each candidate weight.
            step: Number of repeated forward passes per weight candidate
                (averages out any stochastic effects, e.g. dropout).

        Returns:
            float — `best_weight`, the transformer weight (0..1) that
            achieved the highest measured accuracy; the complementary mlp.MLP
            weight is `1 - best_weight`.

        Side effects:
            Mutates `self.transformer_weight` / `self.mlp_weight` in place,
            setting them to the winning split before returning.
        """
        print("\n🔧 Calibrating ensemble weights...")
        
        best_weight = self.pipeline.confidence_threshold + self.pipeline.confidence_threshold
        best_accuracy = 0

        AME = self.pipeline.model2.AME_Encoder(input_ids)

        # Try different weights
        for w in np.linspace(0, 1, 11):
            self.transformer_weight = w
            self.mlp_weight = 1 - w
            
            correct = 0
            total = 0
            for i in range(step):
                trans_probs, _ = self.pipeline.model2.forward(input_ids, AME=AME, embedded=True)
                mlp_probs = self.pipeline.mlp.forward(X_mlp)
                
                ensemble = w * trans_probs + (1-w) * mlp_probs
                preds = np.argmax(ensemble, axis=1)
                true = np.argmax(y_true, axis=1)
                
                correct += np.sum(preds == true)
                total += len(preds)
            
            accuracy = correct / total
            print(f" || Weight: {w:.1f}: Accuracy: {accuracy:.2%}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_weight = w

        self.transformer_weight = best_weight
        self.mlp_weight = 1.0 - best_weight

        print(f"\n✅ Optimal weights: transformer.Transformer: {best_weight:.2f}, mlp.MLP={1-best_weight:.2f}")
        print(f"[-] Validation accuracy: {best_accuracy:.2%}")
        
        return best_weight

# Cross-session automation module that allows exporting and importing of sessions, syncing with another device, and listing available sessions for better management and continuity of work across different environments.


