"""
prediction_manager.py
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
from . import deployment
from . import mlp
from . import peer_agent
from . import pipeline
from . import transformer

class PipelinePredictionManager:
    """
    Label-aware prediction front-end that sits on top of an
    `pipeline.IntegratedPipeline` and turns its raw model outputs into calibrated,
    rule-checked, human-facing predictions.

    Where `pipeline.IntegratedPipeline` owns the models (transformer.Transformer/mlp.MLP/LSTM/
    ensemble) and low-level encode/train/predict plumbing, this class owns
    the *labels* side of the problem: it loads the label CSV
    (`load_labels_from_csv`), keeps running per-class error/prediction
    counters used for confidence calibration (`error_counts`,
    `pred_counts`, `decay`), and provides the higher-level prediction
    entry points that callers (including `peer_agent.ConsecutivePeerAgent` and
    `deployment.CohesiveAgentDeployment`) actually use:

    - `regular_prediction_method` / `simple_pass_prediction`: single-pass
      prediction over a batch of titles, optionally showing per-class
      probabilities.
    - `advanced_prediction_method`: the "full" prediction path — computes
      anisotropy/AME-style diagnostics, decides whether an ensemble vs. a
      single model should be trusted for a given sample
      (`_compute_need_ensemble_method`), optionally routes long/sequential
      inputs to the transformer.Transformer (`_check_for_transformer_sequences`), and
      applies calibration/penalization (`calibration_penalized_check`)
      before returning a final label.
    - Train/validation split helpers (`dynamic_test_size`,
      `_prepare_train_val_split`) used when the manager needs to hold out
      data for calibration rather than relying on the caller to pass a
      pre-split set.
    - `display_hybrid_results`: pretty-prints a prediction payload (top-k
      labels + probabilities) for logging/debugging.

    Calibration model
    -------------------
    `error_counts[c]` / `pred_counts[c]` track, per label index `c`, how
    often a prediction for that class was wrong vs. made at all. `decay`
    (taken from `pipeline.error_decay`) exponentially fades old
    observations so the manager adapts to recent model behavior rather
    than being dominated by early history. These counters feed into
    `_compute_need_ensemble_method` and `calibration_penalized_check` to
    down-weight classes the model has recently been unreliable on.

    Attributes:
        pipeline (pipeline.IntegratedPipeline): Owning pipeline; supplies models,
            encoders, and configuration (e.g. `error_decay`,
            `use_transformer`, `confidence_threshold`).
        titles (List[str]): Raw label-CSV example texts, in file order.
        y_raw (List[int]): Numeric label ids parallel to `titles`.
        label_map (Dict[str, int]): String label -> numeric id mapping
            derived from the CSV (sorted for determinism).
        error_counts (np.ndarray): Running (decayed) wrong-prediction
            count per class, shape `(len(label_map),)`.
        pred_counts (np.ndarray): Running (decayed) total-prediction count
            per class, same shape as `error_counts`.
        decay (float): Exponential decay factor applied when updating the
            above counters (from `pipeline.error_decay`).

    Raises:
        Warning: If the label CSV could not be loaded (`label_map` ends up
            None) — the manager is not usable for label-aware prediction
            without a valid label map, so construction fails loudly rather
            than leaving the instance in a half-initialized state.
    """

    def __init__(self, pipeline, label_csv='labels.csv', target_title='title', label='label'):
        """
        Load the label CSV and initialize per-class calibration counters.

        Args:
            pipeline (pipeline.IntegratedPipeline): Owning pipeline; used for
                `error_decay` and (later) model access during prediction.
            label_csv (str): Filename or path of the labeled examples CSV,
                resolved via `load_labels_from_csv`'s search-path logic.
            target_title (str): Name of the CSV column holding the input
                text for each example.
            label (str): Name of the CSV column holding the string label
                for each example.

        Raises:
            Warning: If loading the CSV failed and no usable `label_map`
                could be built (see `load_labels_from_csv`).
        """
        self.pipeline = pipeline
        try:
            print("📖 Loading labels from text file...")
            self.titles, self.y_raw, self.label_map = self.load_labels_from_csv(label_csv, target_title, label)
        except Exception as e:
            print(f"Error loading labels: {e}")
            self.titles, self.y_raw, self.label_map = None, None, None

        if self.label_map is not None:   
            self.error_counts = np.zeros(len(self.label_map), dtype=np.float64)
            self.pred_counts  = np.zeros(len(self.label_map), dtype=np.float64)
            self.decay        = self.pipeline.error_decay   # how fast old errors fade — tunable
        else:
            raise Warning('[!] Label map is None, consider adding label map!')

        print(f"✅ Loaded {len(self.titles)} labeled examples")

    def load_labels_from_csv(self, filename, target_title, label):
        """
        Load CSV from multiple common locations — no need to place
        file next to the script.

        Search order:
        1. Absolute path (if filename is already absolute)
        2. Current working directory
        3. Script directory
        4. User home directory
        5. Common data folders (Downloads, Documents, Desktop)
        """

        # build candidate paths
        candidates = []

        # 1 — absolute path as-is
        if os.path.isabs(filename):
            candidates.append(filename)
        else:
            # 2 — current working directory
            candidates.append(os.path.join(os.getcwd(), filename))

            # 3 — script directory
            try:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                candidates.append(os.path.join(script_dir, filename))
            except NameError:
                pass

            # 4 — home directory
            candidates.append(os.path.join(os.path.expanduser('~'), filename))

            # 5 — common data folders
            home = os.path.expanduser('~')
            for folder in ['Downloads', 'Documents', 'Desktop', 'Data', 'data']:
                candidates.append(os.path.join(home, folder, filename))

            # 6 — sys.path entries (useful in notebooks)
            for p in sys.path:
                if p:
                    candidates.append(os.path.join(p, filename))

        # find first existing path
        filepath = None
        for candidate in candidates:
            if os.path.exists(candidate):
                filepath = candidate
                break

        if filepath is None:
            print(f"❌ Could not find '{filename}' in any of these locations:")
            for c in candidates[:6]:  # show first 6 only
                print(f"   {c}")
            print(f"\n💡 Tip: place your CSV in one of these folders or pass the full path:")
            print(f"   {os.getcwd()}\\{filename}")
            print(f"   {os.path.expanduser('~')}\\Downloads\\{filename}")
            return [], [], {}

        print(f"✅ Found CSV at: {filepath}")

        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            print(f"❌ Failed to read CSV: {e}")
            return [], [], {}

        # validate columns exist
        missing = [c for c in [target_title, label] if c not in df.columns]
        if missing:
            print(f"❌ Missing columns: {missing}")
            print(f"   Available columns: {list(df.columns)}")
            return [], [], {}

        print(f"✅ Loaded CSV with columns: {list(df.columns)}")

        # drop rows with missing values in target columns
        before = len(df)
        df = df.dropna(subset=[target_title, label])
        dropped = before - len(df)
        if dropped > 0:
            print(f"⚠️ Dropped {dropped} rows with missing values")

        # extract and clean
        titles       = df[target_title].astype(str).str.strip('"').tolist()
        string_labels = df[label].astype(str).tolist()

        print(f"📊 Found {len(titles)} examples")
        print(f"📊 Labels: {set(string_labels)}")

        # create numeric label map
        unique_labels = sorted(set(string_labels))
        label_map     = {lbl: i for i, lbl in enumerate(unique_labels)}
        y             = [label_map[lbl] for lbl in string_labels]

        return titles, y, label_map


    def regular_prediction_method(self, titles=None, label_map=None, rules=None, X=None, y=None, show_proba=False, top_k=3, batch_size=2, use_transformer=True):
        """
        "Full" prediction pass over a batch of titles: builds/derives
        features if not supplied, runs both the transformer.Transformer and mlp.MLP,
        blends their probabilities into a single calibrated distribution
        per example, and returns per-example results (and optionally a
        top-k probability breakdown).

        High-level flow:
        1. If `titles` + `rules` are given but `X`/`y` are not, generates
           features via `pipeline.data_preparation` +
           `pipeline.mlp_training_features` (auto feature engineering).
        2. If `X`/`y` *are* given, splits off a validation slice with
           `_prepare_train_val_split`, one-hot encodes `y` if needed, and
           z-score normalizes `X` using the training slice's mean/std
           (this method evaluates on that held-out `X_val`/`y_val`, not
           the training slice itself).
        3. Ensures the transformer.Transformer has been prepped for this batch
           (`pipeline.transformer_utilities`), builds token-id input
           (`pipeline.input_encoding` or `_features_to_sequence`), and
           pulls LSTM probabilities from the cached `label_bins` if
           available.
        4. If `use_transformer` and a vocabulary exists, encodes titles and
           runs `model2.forward` (transformer.Transformer) to get `trans_probs` +
           attention weights; otherwise falls back to mlp.MLP-only.
        5. Runs `model3.forward` (mlp.MLP) to get `mlp_probs`, clamps any
           out-of-range predicted indices to a valid class, then for each
           example blends `trans_probs`/`mlp_probs` — trusting the mlp.MLP's
           *class choice* but boosting its probability mass using an
           anisotropy-derived confidence rate before renormalizing — with
           `pipeline._calibrate_probs` as an internal fallback if the
           direct blend fails.

        Args:
            titles: Batch of input texts (or `(text, expected_label)`
                tuples) to predict on.
            label_map (Dict[str, int]): String label -> class-index
                mapping used to build `reverse_map` for turning predicted
                indices back into label strings.
            rules: Feature-engineering rules forwarded to
                `pipeline.data_preparation`/`mlp_training_features` when
                `X`/`y` need to be auto-generated.
            X, y: Optional pre-computed features/targets. When provided,
                a validation split is carved out and normalized before use.
            show_proba (bool): Whether callers want per-class probability
                detail alongside the top prediction (consumed further down
                in the method body / by the caller's display logic).
            top_k (int): Number of top classes to retain when reporting
                probability breakdowns.
            batch_size (int): Batch size passed through to
                `pipeline.transformer_utilities`.
            use_transformer (bool): Whether to run the transformer.Transformer at all;
                when False (or no vocabulary is available), prediction is
                mlp.MLP-only.

        Returns:
            The per-example prediction results collected while iterating
            `titles` (see method body for exact payload shape); on
            exception, errors are caught by the surrounding `try` and
            handled per the `except` block below.
        """
        try:
            dataset = None
            X_gen = None
            use_embedded = False
            attn_weights = None

            trans_probs = None
            mlp_probs = None
            target_probs = None

            print(f"\n[🚀] Regular Prediction Initiated...")
            self.pipeline.titles = titles
            self.pipeline.labels = label_map

            reverse_map = {v: k for k, v in label_map.items()}
            num_classes = self.pipeline._get_num_classes(label_map=label_map)

            if titles is not None and rules is not None:
                print(f"[🔍] Preparing data for {len(titles)} titles with {len(rules)} length of rules.")
                if X is None and y is None or X is None or y is None:
                    print('[🔄] Creating automatic X samples because X is not provided manually.')
                    dataset, X_gen = self.pipeline.data_preparation(titles, label_map)  
                    _, y, _, _ = self.pipeline.mlp_training_features(rules, dataset)                  
                else:
                    dataset, _ = self.pipeline.data_preparation(titles, label_map)
                    
            if X is not None or y is not None and isinstance(X, (np.ndarray, list)) and isinstance(y, (np.ndarray, list)) and len(X) > 0 and len(y) > 0:
                X_train, X_val, y_train, y_val = self._prepare_train_val_split(
                    X, y, min_val_per_class=5, min_frac=0.1, max_frac=0.3
                )
                
                onehot_validation = self.pipeline._validate_onehot(y_train)
                if onehot_validation:
                    if num_classes != len(label_map):
                        num_classes = len(label_map)
                    if num_classes > np.max(y_train):
                        y_train = np.eye(num_classes)[np.asarray(y_train)]
                        y_val = np.eye(num_classes)[np.asarray(y_val)]
                    else:
                        print('[⚠️] Warning: Y onehot encoding fails, Returning y samples as is, This may cause Exploding Gradient in mlp.MLP Training!')
                        y_train = y_train
                        y_val = y_val

                X_mean = X_train.mean(axis=0)
                X_std = X_train.std(axis=0) + 1e-8

                X_train = (X_train - X_mean) / X_std
                X = (X_val - X_mean) / X_std
                y = y_val.copy()

            if X_gen is not None:
                self.pipeline.transformer_utilities(X_provided=X, X_raw=X_gen, y_true=y, rules=rules, datasets=dataset, label_map=label_map, batch_size=batch_size)
            else:
                self.pipeline.transformer_utilities(X_provided=X, X_raw=X, y_true=y, rules=rules, datasets=dataset, label_map=label_map, batch_size=batch_size)

            if dataset is not None:
                input_ids, _ = self.pipeline.input_encoding(dataset)
            else:
                input_ids = self.pipeline._features_to_sequence(X)
                
            if X is None and X_gen is not None:
                X = X_gen

            if isinstance(input_ids, (list, np.ndarray)):
                use_embedded = True

            if self.pipeline.cache and 'label_bins' in self.pipeline.cache:
                print('[=] label_bins cache found!')
                label_bins = self.pipeline.cache['label_bins']
                if X_gen is None:
                    X_gen = X

                lstm_probs, _ = self.pipeline.ensemble._get_lstm_probs(input_ids, X_gen, label_bins=label_bins)      
            else:
                lstm_probs = None


            if use_transformer and hasattr(self.pipeline, 'vocab') and self.pipeline.vocab:
                print("[🔄] Using transformer.Transformer for probability calibration")

                if titles is not None and len(titles) > 0:
                    # Encode titles for transformer
                    input_ids_list = []
                    for title in titles:
                        # Handle both string and tuple inputs
                        if isinstance(title, tuple):
                            title = title[0]
                        # Encode to token IDs using pipeline's vocabulary
                    ids = self.pipeline.encode(title, self.pipeline.vocab)
                    input_ids_list.append(np.array(ids))
                
                    input_ids = np.array(input_ids_list)
                    AME = self.pipeline.model2.AME_Encoder(input_ids) 
                    try:
                        trans_probs, attn_weights = self.pipeline.model2.forward(input_ids, AME=AME, embedded=use_embedded)
                    except:
                        trans_probs, attn_weights = self.pipeline.model2.forward(input_ids, AME=AME, embedded=False)

                else:
                    AME = self.pipeline.model2.AME_Encoder(input_ids)
                    trans_probs, attn_weights = self.pipeline.model2.forward(input_ids, AME=AME, embedded=use_embedded)
            else:
                print("[⚡] Using mlp.MLP only for predictions")
                trans_probs = None
        
            if not hasattr(self.pipeline, 'tfidf') or self.pipeline.tfidf is None:
                self.pipeline.initialize_fitting(titles)
            
            # Prepare texts for mlp.MLP
            if titles is not None and len(titles) > 0:
                if isinstance(titles[0], tuple):
                    mlp_titles = [t[0] for t in titles]
                else:
                    mlp_titles = titles
                    
                X_tfidf = self.pipeline.tfidf.transform(mlp_titles).toarray() 
            else:
                X_tfidf = X

            # Forward pass through mlp.MLP
            mlp_probs = self.pipeline.model3.forward(X, y=y)
           
            # Validate all mlp.MLP predictions at once
            mlp_pred_indices = np.argmax(mlp_probs, axis=1)
            if num_classes <= 0:
                num_classes = self.pipeline._get_num_classes(mlp_probs=mlp_probs)

            valid_mask = mlp_pred_indices < num_classes
            if not np.all(valid_mask):
                invalid_count = np.sum(~valid_mask)
                # Replace invalid indices with argmax within valid range
                for i in range(len(mlp_pred_indices)):
                    valid_probs = mlp_probs[i][:num_classes] if num_classes > 0 else mlp_probs[i]
                    if len(valid_probs) > 0:
                        mlp_pred_indices[i] = int(np.argmax(valid_probs))
                    else:
                        mlp_pred_indices[i] = 0  # Default to first class   
                             
            results = []
            if titles is not None and len(titles) > 0:
                for i, title in enumerate(titles):
                    # Handle tuple inputs
                    if isinstance(title, tuple):
                        display_title = title[0]
                        expected_label = title[1] if len(title) > 1 else None
                    else:
                        display_title = title
                        expected_label = None
                    
                    # mlp.MLP prediction
                    mlp_class_idx = mlp_pred_indices[i]
                    mlp_class_idx = min(mlp_class_idx, num_classes - 1)  # Clamped to valid range
                    
                    mlp_confidence = mlp_probs[i][mlp_class_idx]
                    mlp_label = reverse_map.get(mlp_class_idx, f"unknown_{mlp_class_idx}")
                    anisotropy = self.pipeline.anisotropy_measurement(input_ids)
                    anisotropic_rate = 1.0 / (1.0 + np.exp(-anisotropy)) if anisotropy is not None else 1.0

                    # transformer.Transformer prediction (if available)
                    if trans_probs is not None:
                        if trans_probs.shape[0] > i:
                            trans_probs_i = trans_probs[i]
                        else:
                            trans_probs_i = trans_probs[-1]  # fallback to last if mismatch
                        
                        trans_class_idx = np.argmax(trans_probs_i)
                        trans_confidence = trans_probs_i[trans_class_idx]
                        trans_label = reverse_map.get(trans_class_idx, f"unknown_{trans_class_idx}")
                        
                        # Calibrated probabilities (blend of mlp.MLP and transformer.Transformer)
                        if use_transformer:
                            # Boost mlp.MLP's prediction in transformer probabilities
                            calibrated = trans_probs_i.copy()
                            try:
                                calibrated[mlp_class_idx] = max(calibrated[mlp_class_idx], anisotropic_rate)
                                calibrated /= calibrated.sum()
                            except Exception as e:
                                calibrated = self.pipeline._calibrate_probs(mlp_probs, mlp_pred_indices, attn_weights, input_ids)
            
                            final_probs = calibrated
                            final_class_idx = mlp_class_idx  # Trust mlp.MLP's class decision
                            try:
                                final_confidence = final_probs[final_class_idx]
                            except IndexError:
                                final_confidence = np.max(final_probs) if isinstance(final_probs, np.ndarray) else final_probs

                            if isinstance(final_confidence, np.ndarray):
                                final_confidence = np.max(final_confidence)
                                            
                        else:
                            final_probs = mlp_probs[i]
                            final_class_idx = mlp_class_idx
                            final_confidence = mlp_confidence
                    else:
                        final_probs = mlp_probs[i]
                        final_class_idx = mlp_class_idx
                        final_confidence = mlp_confidence
                        trans_label = reverse_map.get(mlp_class_idx, f"unknown_{mlp_class_idx}")
                        trans_confidence = mlp_confidence
                    
                    final_label = reverse_map.get(final_class_idx, f"unknown_{final_class_idx}")
                    
                    result = {
                        'title': display_title,
                        'expected': expected_label,
                        'predicted': final_label,
                        'confidence': final_confidence,
                        'index': final_class_idx,
                        'mlp_prediction': mlp_label,
                        'mlp_confidence': mlp_confidence,
                    }
                    
                    if trans_label is not None:
                        result['transformer_prediction'] = trans_label
                        result['transformer_confidence'] = trans_confidence

                    agreement = trans_label  == mlp_label
                    
                    
                    # Include top-k predictions if requested
                    if show_proba:
                        top_indices = np.argsort(final_probs)[-top_k:][::-1]
                        top_predictions = []
                        for idx in top_indices:
                            if idx in reverse_map:
                                top_predictions.append({
                                    'label': reverse_map[idx],
                                    'confidence': final_probs[idx]
                                })
                            else:
                                top_predictions.append({
                                    'label': f"unknown_{idx}",
                                    'confidence': final_probs[idx]
                                })
                        result['top_predictions'] = top_predictions
                        
                        mlp_top_indices = np.argsort(mlp_probs[i])[-top_k:][::-1]
                        mlp_top = []
                        for idx in mlp_top_indices:
                            if idx in reverse_map:
                                mlp_top.append({
                                    'label': reverse_map[idx],
                                    'confidence': mlp_probs[i][idx]
                                })
                        result['mlp_top_predictions'] = mlp_top
                        
                        if trans_probs is not None:
                            trans_top_indices = np.argsort(trans_probs[i])[-top_k:][::-1]
                            trans_top = []
                            for idx in trans_top_indices:
                                if idx in reverse_map:
                                    trans_top.append({
                                        'label': reverse_map[idx],
                                        'confidence': trans_probs[i][idx]
                                    })
                            result['transformer_top_predictions'] = trans_top
                    
                    results.append(result)
            else:
                n_samples = mlp_probs.shape[0]

                lstm_pred_indices = np.argmax(lstm_probs, axis=1) if lstm_probs is not None else None
                if mlp_probs is None:
                    logits = self.pipeline.mlp.forward(X) if X is not None else X_tfidf
                    mlp_probs = self.pipeline._softmax(logits)

                target_probs = self.calibration_penalized_check(mlp_probs, mlp_pred_indices)
                target_pred_indices = np.argmax(target_probs, axis=1)

                for i in range(n_samples):
                    outcome = self._compute_sample_prediction(
                        i, mlp_probs, target_probs, target_pred_indices,
                        trans_probs=trans_probs, lstm_probs=lstm_probs,
                        lstm_pred_indices=lstm_pred_indices,
                        attn_weights=attn_weights, input_ids=input_ids,
                        num_classes=num_classes, reverse_map=reverse_map
                    )

                    result = {
                        "title": f"Unknown",
                        "expected": f"Unknown",
                        **outcome
                    }
                    results.append(result)

            # Display results

            if titles is not None and len(titles) > 0:
                print("\n" + "="*70)
                print("🎯 HYBRID PREDICTION RESULTS (mlp.MLP + transformer.Transformer)")
                print("="*70)
                
                correct_count = 0
                for result in results:
                    print(f"\n📌 '{result['title']}'")
                    
                    if result.get('expected'):
                        status = "✓" if result['predicted'] == result['expected'] else "✗"
                        print(f"   Expected: {result['expected']} {status}")
                    
                    print(f"   🎯 FINAL PREDICTION: {result['predicted']} ({result['confidence']:.1%})")
                    print(f"   ⚡ mlp.MLP: {result['mlp_prediction']} ({result['mlp_confidence']:.1%})")
                    
                    if result.get('transformer_prediction'):
                        arrow = "⬆️" if result['transformer_confidence'] > result['mlp_confidence'] else "⬇️"
                        print(f"   🌀 transformer.Transformer: {result['transformer_prediction']} ({result['transformer_confidence']:.1%}) {arrow}")
                    
                    if show_proba and 'top_predictions' in result:
                        print("\n   🔍 Top possibilities (calibrated):")
                        for j, pred in enumerate(result['top_predictions'][:top_k], 1):
                            bar = '█' * int(pred['confidence'] * 20)
                            print(f"      {j}. {pred['label']:20s} {bar} {pred['confidence']:.1%}")
                    
                    if result.get('expected') and result['predicted'] == result['expected']:
                        correct_count += 1
                
                if results and results[0].get('expected'):
                    accuracy = correct_count / len(results)
                    print(f"\n📊 Accuracy: {correct_count}/{len(results)} = {accuracy:.1%}")

            verbose = False
            if float(results[0]['confidence']) < self.pipeline.confidence_threshold:
                verbose = True

            payload = {
                'X_samples': X,
                'input_ids': input_ids
            }    

            if titles is not None and len(titles) > 0:
                self.display_hybrid_results(payload, final_class_idx, results, top_k, verbose=verbose)

            # Use results directly - they already contain calibrated predictions
            chosen_label = results[0]['predicted'] if results else None
            confidence = results[0]['confidence'] if results else None

            if isinstance(chosen_label, int) or isinstance(chosen_label, np.integer):
                chosen_label = str(chosen_label)
                
            # Only recalibrate if models disagreed AND we have valid results
            if results and not results[0].get('models_agree', True):
                print("\n[⚠️] Disagreement detected between mlp.MLP and transformer.Transformer predictions. Using calibrated probabilities for final decision.")
                calibrated_probs = self.pipeline.hybrid_prediction(rules, input_ids, dataset, X=X, y=y, use_embedded=use_embedded)
               
                if calibrated_probs is not None and len(calibrated_probs) > 0:
                    final_idx = int(np.argmax(calibrated_probs[:num_classes]))
                    if final_idx > len(reverse_map):
                        final_idx = int(np.argmax(calibrated_probs[:len(reverse_map)-1]))
                        print(f"[⚠️] Clamping {final_idx} → {final_idx}") 

                    final_idx = int(min(final_idx, num_classes - 1))  # Ensure index is within valid range
                    chosen_label = reverse_map.get(final_idx, f"unknown_{final_idx}")
                    try:
                        confidence = float(calibrated_probs[0][final_idx])   
                    except:
                        confidence = float(calibrated_probs[0][len(reverse_map)-1]) if isinstance(calibrated_probs[0], (float, int)) else 0.0             
                            
            if isinstance(chosen_label, str) and chosen_label.startswith("unknown") or float(confidence) < self.pipeline.confidence_threshold:
                print(f"\n[⚠️] Final prediction is {chosen_label} with uncertain confidence: {confidence:.1%}. Consider collecting more data or adjusting the model.")
            else:
                print(f"\n[🎯] Final chosen label for input: {chosen_label} || Confidence: {confidence:.1%}")  

            if results and results[0]['confidence'] > self.pipeline.confidence_threshold:
                results[0]['predicted'] = chosen_label
                results[0]['confidence'] = confidence

        except Exception as e:
            print(f"[=] Error during prediction: {e}")
            traceback.print_exc()
            results = []

        return results



    def simple_pass_prediction(self, pipeline, titles=None, label_map=None, rules=None, X=None, X_raw=None, y=None, show_proba=True, top_k=3, batch_size=2):
        """
        Simpler, hybrid-model prediction pass used when the caller wants
        the pipeline's own `hybrid_prediction`/`predict_proba` combination
        rather than the manual transformer.Transformer+mlp.MLP blend done in
        `regular_prediction_method`.

        Flow mirrors the first half of `regular_prediction_method` (derive
        `X`/`y` from `titles`+`rules` if not supplied, carve out and
        normalize a validation split via `_prepare_train_val_split`,
        prep the transformer.Transformer via `transformer_utilities`), but then hands
        off to the pipeline's own hybrid path:
        - Encodes `datasets`/`X` into `input_datasets`
          (`transformer_input_encoding` or `_features_to_sequence`).
        - Gets calibrated probabilities via
          `pipeline.predict_proba(..., type='Hybrid', embedded=True)`.
        - Gets the actual hybrid prediction via
          `pipeline.hybrid_prediction(...)`, retrying with
          `use_embedded=False` if the embedded path raises.
        - Pulls in LSTM probabilities from the cached `label_bins` when
          available, for use alongside the hybrid result.

        Args:
            pipeline (pipeline.IntegratedPipeline): Explicit pipeline reference used
                for the hybrid calls in this method (note: `self.pipeline`
                is also set/used elsewhere in the method — both should
                refer to the same instance).
            titles: Batch of input texts to predict on.
            label_map (Dict[str, int]): String label -> class-index map.
            rules: Feature-engineering rules for auto-generating `X`/`y`
                when not supplied directly.
            X, X_raw, y: Optional pre-computed features / raw features /
                targets.
            show_proba (bool): Whether to include per-class probability
                detail in the returned result.
            top_k (int): Number of top classes to report when showing
                probabilities.
            batch_size (int): Batch size forwarded to
                `transformer_utilities`.

        Returns:
            The hybrid prediction result produced by
            `pipeline.hybrid_prediction` (augmented with LSTM probability
            info when a `label_bins` cache is available); see method body
            for the exact downstream shape.
        """
        self.pipeline.titles = titles
        self.pipeline.labels = label_map   

        num_classes = self.pipeline._get_num_classes(label_map=label_map)
        try:

            if titles is not None and rules is not None:
                print(f"[🔍] Preparing data for {len(titles)} titles with {len(rules)} length of rules.")
                if X is None and y is None or X is None or y is None:
                    print('[🔄] Creating automatic X samples because X Samples is not provided manually.')
                    datasets, X_gen = self.pipeline.data_preparation(titles, label_map)  
                    _, y, _, _ = self.pipeline.mlp_training_features(rules, datasets)                  
                else:
                    datasets, _ = self.pipeline.data_preparation(titles, label_map)

            if X is not None or y is not None and len(X) > 0 and len(y) > 0:
                X_train, X_val, y_train, y_val = self._prepare_train_val_split(
                    X, y, min_val_per_class=5, min_frac=0.1, max_frac=0.3
                )
                
                onehot_validation = self.pipeline._validate_onehot(y_train)
                if onehot_validation:
                    if num_classes != len(label_map):
                        num_classes = len(label_map)
                    if num_classes > np.max(y_train):
                        y_train = np.eye(num_classes)[np.asarray(y_train)]
                        y_val = np.eye(num_classes)[np.asarray(y_val)]
                    else:
                        print('[⚠️] Warning: Y onehot encoding fails, Returning y samples as is, This may cause Exploding Gradient in mlp.MLP Training!')
                        y_train = y_train
                        y_val = y_val

                X_mean = X_train.mean(axis=0)
                X_std = X_train.std(axis=0) + 1e-8

                X_train = (X_train - X_mean) / X_std
                X = (X_val - X_mean) / X_std
                y = y_val.copy()

            if X_gen is not None:
                self.pipeline.transformer_utilities(X_provided=X_train, X_raw=X_gen, y_true=y_train, rules=rules, datasets=dataset, label_map=label_map, batch_size=batch_size)
            else:
                self.pipeline.transformer_utilities(X_provided=X_train, X_raw=X, y_true=y_train, rules=rules, datasets=dataset, label_map=label_map, batch_size=batch_size)

            reverse_map = {v: k for k, v in label_map.items()}

            if datasets is not None:
                input_datasets = self.pipeline.transformer_input_encoding(datasets)
            else:
                input_datasets = self.pipeline._features_to_sequence(X)

            pred_probs = self.pipeline.predict_proba(input_datasets, X_raw, type='Hybrid', embedded=True)[0]
            try:
                pred_result = self.pipeline.hybrid_prediction(rules, input_datasets, datasets, X=X, y=y, use_embedded=True)
            except:
                pred_result = self.pipeline.hybrid_prediction(rules, input_datasets, datasets, X=X, y=y, use_embedded=False)

            if X is None and X_gen is not None:
                X = X_gen
            
            if self.pipeline.cache and 'label_bins' in self.pipeline.cache:
                print('[=] label_bins cache found!')
                label_bins = self.pipeline.cache['label_bins']
                lstm_probs, _ = self.pipeline.ensemble._get_lstm_probs(input_datasets, X, label_bins=label_bins)      
            else:
                lstm_probs = None

            print("\n[🔍] Prediction result structure:")
            print(f"[=] Type: {type(pred_result)}")
            print(f"[=] Length: {len(pred_result) if isinstance(pred_result, tuple) else 1}")

            if isinstance(pred_result, tuple):
                if len(pred_result) == 3:
                    pred_indices = pred_result[0]
                    hybrid_probs = pred_result[1]  # Use different variable name
                    attn_weights = pred_result[2]
                    print("✅ Extracted: indices, probs, attention")
                elif len(pred_result) == 2:
                    pred_indices = pred_result[0]
                    hybrid_probs = pred_result[1]  # Use different variable name
                    print("✅ Extracted: indices, probs")
                else:
                    print(f"⚠️ Unknown tuple format with {len(pred_result)} elements")
                    pred_indices = pred_result[0]
                    hybrid_probs = pred_result[1] if len(pred_result) > 1 else None
            else:
                pred_indices = pred_result
                hybrid_probs = None
                print("✅ Single value return")
        
            # Use hybrid_probs if available, otherwise use pred_probs
            final_probs = hybrid_probs if hybrid_probs is not None else pred_probs
        
            if isinstance(pred_indices, (list, tuple)) and len(pred_indices) > 0:
                if isinstance(pred_indices[0], (np.ndarray, list)):

                    pred_indices = np.array([p[0] if isinstance(p, (np.ndarray, list)) else p 
                                        for p in pred_indices])
                else:
                    pred_indices = np.array(pred_indices)
            elif isinstance(pred_indices, np.ndarray):
                if pred_indices.ndim > 1:
                    pred_indices = pred_indices.flatten()
            else:
                pred_indices = np.array([pred_indices])
        
            print(f"\n[📊] Processed predictions:")
            print(f"[=] pred_indices shape: {pred_indices.shape}")
            print(f"[=] pred_indices: {pred_indices}")
        
            if final_probs is not None:
                print(f"[=] final_probs shape: {final_probs.shape if hasattr(final_probs, 'shape') else 'unknown'}")
        
            if final_probs is not None and isinstance(final_probs, np.ndarray) and final_probs.ndim == 1:
                final_probs = final_probs.reshape(1, -1)

            if titles is not None and len(titles) > 0:
                n_samples = len(titles)
            else:
                n_samples = pred_probs.shape[0] if hasattr(pred_probs, 'shape') else 0

            if len(pred_indices) < n_samples:
                print(f"[⚠️] Padding predictions from {len(pred_indices)} to {n_samples}")
                last_idx = pred_indices[-1] if len(pred_indices) > 0 else 0
                pred_indices = np.pad(pred_indices, (0, n_samples - len(pred_indices)), 
                                mode='constant', constant_values=last_idx)

            results = []
            best_idx = -1
            best_confidence = -1
            
            # Determine rows and cols from final_probs
            if final_probs is not None and hasattr(final_probs, 'shape'):
                rows = final_probs.shape[0]
                cols = final_probs.shape[1] if len(final_probs.shape) > 1 else 1
            elif final_probs is not None:
                rows = len(final_probs)
                cols = len(final_probs[0]) if rows > 0 and hasattr(final_probs[0], '__len__') else 1
            else:
                rows, cols = 0, 0
            
            for i in range(n_samples):
                class_idx = int(pred_indices[i]) if i < len(pred_indices) else 0
                    
            if final_probs is not None and i < rows and class_idx < cols:
                if hasattr(final_probs, 'shape'):
                    confidence = final_probs[i, class_idx]
                else:
                    if isinstance(final_probs[i], (list, np.ndarray)):
                        if i < len(final_probs):
                            confidence = final_probs[i][class_idx]
                        else:
                            confidence = np.mean(final_probs[class_idx])
                    else:
                        if i < len(final_probs):
                            confidence = float(final_probs[i])  # Single value
                        else:
                            confidence = float(np.mean(final_probs))

                        
                if confidence > best_confidence:
                    best_idx = i
                    best_confidence = confidence


            if titles is not None and len(titles) > 0:   
                for i, title in enumerate(titles):
                    if i < len(pred_indices):
                        class_idx = int(pred_indices[i])
                    else:
                        class_idx = 0
                        
                    # Get confidence from final_probs
                    if final_probs is not None and i < rows and class_idx < cols:
                        if hasattr(final_probs, 'shape'):
                            confidence = final_probs[i, class_idx]
                        else:  # list
                            if isinstance(final_probs[i], (list, np.ndarray)):
                                confidence = final_probs[i][class_idx]
                            else:
                                confidence = float(final_probs[i])  # Single value
                    else:
                        # Fallback: use max probability instead of min
                        if final_probs is not None and i < len(final_probs):
                            if isinstance(final_probs[i], (list, np.ndarray)):
                                confidence = max(final_probs[i])
                            else:
                                confidence = float(final_probs[i])
                        else:
                            confidence = 0.0
                
                    label = reverse_map.get(class_idx, f"unknown_{class_idx}")

                    result = {
                    'title': title,
                    'predicted': label,
                    'confidence': confidence,
                    'index': class_idx,
                    'is_best': (i == best_idx)
                    }
                    
                    if show_proba and i < rows and cols > 1:
                        if hasattr(final_probs, 'shape'):
                            probs_row = final_probs[i]
                        else:
                            if isinstance(final_probs[i], (list, np.ndarray)):
                                probs_row = np.array(final_probs[i])
                            else:
                                probs_row = np.array([final_probs[i]])
                    
                        if len(probs_row) > 1:
                            top_indices = np.argsort(probs_row)[-top_k:][::-1]
                            top_predictions = []
                            for idx in top_indices:
                                if idx in reverse_map:
                                    top_predictions.append({
                                    'label': reverse_map[idx],
                                    'confidence': float(probs_row[idx])
                                    })
                            result['top_predictions'] = top_predictions
                
                    results.append(result)
            
                print("\n" + "="*70)
                print("[🎯] LABEL PREDICTIONS")
                print("="*70)
            
                for i, result in enumerate(results):
                    print(f"\n[📌] Label: {i+1}. '{result['title']}'")
                
                    best_marker = "[🏆] BEST" if result.get('is_best') else ""
                    print(f"   → {result['predicted']} ({result['confidence']}){best_marker}")
                
                    if show_proba and 'top_predictions' in result:
                        print(" [  Top possibilities:")
                        for j, pred in enumerate(result['top_predictions'][:top_k], 1):
                            bar = '█' * int(pred['confidence'] * 20)
                            print(f"      {j}. {pred['label']} {bar} {pred['confidence']} %")
                
                # Return the best result (not inside loop)
                best_idx = int(np.argmax(final_probs[:, pred_indices] if final_probs is not None and hasattr(final_probs, 'shape') else [r['confidence'] for r in results]))
                if best_idx >= 0:
                    best_result = results[best_idx]
                    if isinstance(best_result['predicted'], str) and best_result['predicted'].startswith("unknown") or best_result['confidence'] < self.pipeline.confidence_threshold:
                        print(f"\n[⚠️] Final prediction is {best_result['predicted']} with uncertain confidence. Consider collecting more data or adjusting the model.")
                    else:
                        print(f"\n✨ Most confident: '{best_result['title']}' → {best_result['predicted']} ({best_result['confidence']:.1%})")
                    return best_result['predicted'], best_result['confidence'], best_result['confidence']
                elif results:
                    # Fallback: return first result if no best found
                    predicted = results[0]['predicted']
                    predicted_confidence = results[0]['confidence']
                    if isinstance(predicted, str) and predicted.startswith("unknown") and predicted_confidence < self.pipeline.confidence_threshold:
                        print(f"\n[⚠️] Final prediction is {predicted} with uncertain confidence: {predicted_confidence:.1%}. Consider more consistent data for the model to learn from.")
                    else:
                        print(f"\n[🎯] Final chosen label for input: {predicted} || Confidence: {predicted_confidence:.1%}")  
                    
                    return predicted, predicted_confidence

            else:
                print("[=] No titles provided for prediction, skipping accuracy validation.")
                
                lstm_pred_indices = np.argmax(lstm_probs, axis=1) if lstm_probs is not None else None
                best_pred_indices = np.argmax(final_probs, axis=1)

                for i in range(n_samples):
                    outcome = self._compute_sample_prediction(
                        i, pred_probs, final_probs, best_pred_indices,
                        trans_probs=trans_probs, lstm_probs=lstm_probs,
                        lstm_pred_indices=lstm_pred_indices,
                        attn_weights=attn_weights, input_ids=input_datasets,
                        num_classes=num_classes, reverse_map=reverse_map
                    )

                predicted = outcome['predicted']
                predicted_confidence = outcome['confidence']

        except Exception as e:
            print(f"[=] Error during robust prediction: {e}")
            predicted = None
            predicted_confidence = None
        return predicted, predicted_confidence
        
    def calculate_entropy(self, probs):
        """
        Shannon entropy of a (batch of) categorical probability
        distribution(s): `-sum(p * log(p))` along the last axis.

        Used as a general-purpose uncertainty measure — higher entropy
        means the distribution is closer to uniform (more uncertain),
        lower entropy means it's concentrated on one class (more
        confident). A small epsilon (`1e-10`) is added inside the log to
        avoid `log(0)` for zero-probability classes.

        Args:
            probs (np.ndarray): Probability array; entropy is computed
                along the last axis, so shape `(..., num_classes)` returns
                shape `(...)`.

        Returns:
            np.ndarray | float: Entropy value(s), same shape as `probs`
            minus the last axis.
        """
        return -np.sum(probs * np.log(probs + 1e-10), axis=-1)

    def _compute_sample_prediction(
        self, i, mlp_probs, target_probs, target_pred_indices,
        trans_probs=None, lstm_probs=None, lstm_pred_indices=None,
        attn_weights=None, input_ids=None, num_classes=None,
        need_ensemble_method=False, reverse_map=None, eps=1e-8
    ):
        """
        Pure per-sample ensemble computation — no title/label dependency.
        Returns a dict with mlp/trans/lstm/final confidence and class index,
        usable both for titled prediction loops and raw batch prediction.

        This is the core fusion logic shared by the batch-prediction paths
        in this class: given per-model probability arrays for sample index
        `i`, it decides which model's output (or which blend of them) to
        trust as the final answer.

        Decision logic (informal):
        - The mlp.MLP's argmax class/confidence is always computed first and
          used as the fallback answer; an out-of-range mlp.MLP index is
          handled defensively by re-deriving the argmax within a
          restricted (`reverse_map`-sized) window, or returning an
          explicit `is_valid=False` error result if even that fails.
        - LSTM class/confidence are read from precomputed
          `lstm_pred_indices`/`lstm_probs` for sample `i` (not
          recomputed here) when both are supplied; otherwise LSTM falls
          back to mirroring the mlp.MLP's class with no confidence value.
        - If transformer.Transformer output (`trans_probs` + `attn_weights`) is
          available:
            - When `need_ensemble_method` is True, all available model
              confidences are converted into normalized weights (each
              roughly `own_confidence / (own + other + eps)`), the
              pipeline's `_calibrate_probs` is used to get a calibrated
              distribution, and the final probability vector is a
              confidence-weighted blend of the target/mlp.MLP and calibrated
              transformer.Transformer/LSTM distributions, re-calibrated once more
              before being returned. `models_agree` reflects whether mlp.MLP
              and transformer.Transformer picked the same class.
            - Otherwise (simple heuristic mode), the single most confident
              of mlp.MLP/transformer.Transformer/LSTM is used directly, with a special
              case: if the chosen confidence exceeds 0.95 (suspiciously
              overconfident), the method instead tries multiplying
              mlp.MLP and LSTM distributions together (when their shapes
              match) or falls back to the target/transformer.Transformer probabilities,
              as a way of tempering runaway single-model confidence.
        - If no transformer.Transformer output is available at all, the result is
          simply the mlp.MLP's own prediction.

        Args:
            i (int): Index of the sample within the batch to compute a
                result for.
            mlp_probs (np.ndarray): mlp.MLP output probabilities, shape
                `(batch, num_classes)`.
            target_probs (np.ndarray): "Base"/target probability array
                (e.g. from `predict_proba`) used as the target-model
                signal in the blend.
            target_pred_indices (np.ndarray): Precomputed argmax indices
                for `target_probs`.
            trans_probs (np.ndarray | None): transformer.Transformer output
                probabilities, if available.
            lstm_probs (np.ndarray | None): LSTM output probabilities, if
                available.
            lstm_pred_indices (np.ndarray | None): Precomputed argmax
                indices for `lstm_probs`.
            attn_weights: transformer.Transformer attention weights, passed through to
                `pipeline._calibrate_probs` for calibration.
            input_ids: Token ids for this batch, also passed through to
                `pipeline._calibrate_probs`.
            num_classes (int | None): Number of classes; inferred from
                `mlp_probs.shape[1]` if not given.
            need_ensemble_method (bool): Selects the calibrated-blend path
                (True) vs. the simple confidence-heuristic path (False).
            reverse_map (Dict[int, str] | None): Class index -> label
                string map used to populate the human-readable prediction
                fields.
            eps (float): Small constant to avoid division by zero when
                normalizing confidence weights.

        Returns:
            Dict: Always includes `is_valid`, `predicted`, `predicted_idx`,
            `confidence`, `mlp_class`, `mlp_prediction`, `mlp_confidence`,
            `trans_class`, `trans_prediction`, `trans_confidence`,
            `lstm_class`, `lstm_confidence`, `models_agree`, and
            `final_probs`. On an unrecoverable out-of-range mlp.MLP index, a
            reduced error dict with `is_valid=False` and an `error` message
            is returned instead.
        """
        reverse_map = reverse_map or {}
        num_classes = num_classes or mlp_probs.shape[1]
        models_agree = False

        # ── mlp.MLP layer ─────────────────────────────────────────────
        mlp_class_idx  = int(np.argmax(mlp_probs[i]))
        is_valid_index = 0 <= mlp_class_idx < num_classes

        if not is_valid_index:
            try:
                mlp_class_idx = int(np.argmax(mlp_probs[:len(reverse_map)-1]))
                mlp_confidence = float(mlp_probs[:len(reverse_map)-1][mlp_class_idx]) 
                mlp_label      = reverse_map.get(mlp_class_idx, f"unknown_{mlp_class_idx}")
                return {
                    'predicted'  : mlp_label,
                    'confidence' : mlp_confidence,
                    'mlp_class'  : mlp_class_idx,
                    'predicted_idx'      : mlp_class_idx,
                    'models_agree': models_agree,
                    'final_probs' : mlp_probs[:len(reverse_map)-1],
                    'is_valid'   : True,
                }
            except:
                return {
                    'predicted'  : None,
                    'confidence' : 0.0,
                    'mlp_class'  : 0,
                    'predicted_idx'      : 0,
                    'models_agree': False,
                    'final_probs' : mlp_probs,
                    'is_valid'   : False,
                    'error'      : f'class_index_out_of_range(idx={mlp_class_idx}, num_classes={num_classes})'
                }               

        mlp_confidence = float(mlp_probs[i][mlp_class_idx]) 
        mlp_label      = reverse_map.get(mlp_class_idx, f"unknown_{mlp_class_idx}")

        # ── LSTM ────────────────────────────────────────────
        # lstm_pred_indices passed in precomputed, not recomputed per-sample
        if lstm_probs is not None and lstm_pred_indices is not None:
            lstm_class_idx  = int(lstm_pred_indices[i]) 
            lstm_confidence = float(lstm_probs[i][lstm_class_idx])
        else:
            # explicit default so lstm_class_idx is always defined, even if LSTM is not used
            lstm_class_idx  = mlp_class_idx
            lstm_confidence = None

        # ── target/base probs ────────────────────────────────
        target_class_idx = int(target_pred_indices[i])
        target_confidence = float(target_probs[i][target_class_idx])

        # ── transformer.Transformer ───────────────────────────────────────
        if trans_probs is not None and attn_weights is not None:
            trans_probs_i = np.asarray(trans_probs[i])

            if trans_probs_i.ndim == 0 or trans_probs_i.size == 1:
                trans_class_idx  = target_class_idx
                trans_confidence = target_confidence
            else:
                trans_class_idx  = int(np.argmax(trans_probs_i))
                trans_confidence = float(trans_probs_i[trans_class_idx])

            trans_label = reverse_map.get(trans_class_idx, f"unknown_{trans_class_idx}")

            if need_ensemble_method:
                calibration = self.pipeline._calibrate_probs(
                    target_probs, target_pred_indices, attn_weights, input_ids
                )

                mlp_weight   = mlp_confidence   / (target_confidence + trans_confidence + eps)
                trans_weight = trans_confidence / (target_confidence + trans_confidence + eps)
                lstm_weight  = None
                if lstm_confidence is not None:
                    lstm_weight = lstm_confidence / (target_confidence + lstm_confidence + eps)

                cal_len = len(calibration[0]) if calibration.ndim > 1 else len(calibration)
                calibration_weighting = (
                    calibration[target_class_idx]
                    if target_class_idx < cal_len else 0.0
                )

                if lstm_confidence is not None and lstm_weight is not None:
                    final_probs = (mlp_weight   * target_probs[i][:cal_len] +
                                trans_weight * calibration[i][:cal_len] +
                                lstm_weight  * calibration[i][:cal_len])
                else:
                    final_probs = (mlp_weight   * target_probs[i][:cal_len] +
                                trans_weight * calibration[i][:cal_len])

                final_probs = self.pipeline._calibrate_probs(final_probs, mlp_class_idx, attn_weights, input_ids)
                final_class_idx = target_class_idx

                try:
                    final_confidence = final_probs[final_class_idx]
                except IndexError:
                    final_confidence = (np.max(final_probs)
                                        if isinstance(final_probs, np.ndarray)
                                        else np.mean(final_probs))

                if isinstance(final_confidence, np.ndarray):
                    final_confidence = float(np.max(final_confidence))

                models_agree = mlp_class_idx == trans_class_idx

            else:
                if lstm_confidence is None:
                    lstm_confidence = mlp_confidence
                models_agree = False

                if (mlp_confidence > trans_confidence and
                    mlp_confidence > lstm_confidence and
                    mlp_confidence <= 0.95):
                    final_probs      = mlp_probs[i]
                    final_class_idx  = mlp_class_idx
                    final_confidence = mlp_confidence

                elif trans_confidence > lstm_confidence:
                    final_probs      = trans_probs[i]
                    final_class_idx  = trans_class_idx
                    final_confidence = trans_confidence

                else:
                    final_probs      = lstm_probs[i] if lstm_probs is not None else mlp_probs[i]
                    final_class_idx  = lstm_class_idx if lstm_probs is not None else mlp_class_idx
                    final_confidence = lstm_confidence if lstm_probs is not None else mlp_confidence

                    if final_confidence > 0.95:
                        if lstm_probs is not None and mlp_probs.shape == lstm_probs.shape:
                            final_probs      = mlp_probs[i] * lstm_probs[i]
                            final_class_idx  = mlp_class_idx
                            final_confidence = mlp_confidence * lstm_confidence
                        else:
                            if lstm_probs is not None:
                                final_probs = (target_probs[i]
                                            if len(target_probs[i]) == num_classes
                                            else lstm_probs[i])
                            else:
                                final_probs = trans_probs[i]
                            final_class_idx  = target_class_idx
                            final_confidence = target_confidence * (lstm_confidence or 1.0)

                models_agree = mlp_class_idx == trans_class_idx

        else:
            # no transformer available — mlp.MLP/LSTM only
            trans_class_idx  = None
            trans_confidence = None
            trans_label      = None
            final_probs      = mlp_probs[i]
            final_class_idx  = mlp_class_idx
            final_confidence = mlp_confidence
            agreement        = False
            models_agree     = False

        return {
            'is_valid'          : True,
            'predicted'         : reverse_map.get(final_class_idx, f"unknown_{final_class_idx}"),
            'predicted_idx'             : final_class_idx,
            'confidence'        : float(final_confidence),
            'mlp_class'         : mlp_class_idx,
            'mlp_prediction'    : mlp_label,
            'mlp_confidence'    : mlp_confidence,
            'trans_class'       : trans_class_idx,
            'trans_prediction'  : trans_label,
            'trans_confidence'  : trans_confidence,
            'lstm_class'        : lstm_class_idx if lstm_probs is not None else None,
            'lstm_confidence'   : lstm_confidence,
            'models_agree'      : models_agree,
            'final_probs'       : final_probs,
        }

    def dynamic_test_size(self, y, min_val_per_class=5, min_frac=0.1, max_frac=0.3):
        """
        Pick test_size so the smallest class still gets a usable validation sample,
        without departing too far from typical train/val ratios.
        """
        classes, counts = np.unique(y, return_counts=True)
        n_total = len(y)
        smallest_class_count = counts.min()

        # fraction needed to guarantee min_val_per_class from the rarest class
        needed_frac = min_val_per_class / smallest_class_count

        # clamp to a sane range so it doesn't swing to extremes
        test_size = np.clip(needed_frac, min_frac, max_frac)
        return float(test_size)

    def _prepare_train_val_split(self, X, y, min_val_per_class=5,
                                min_frac=0.1, max_frac=0.3, random_state=42):
        """
        Safely aligns X/y sample counts and splits into train/val,
        never guessing via transpose, always checking stratify viability.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim > 1 and y.shape[1] == 1:
            y = y.ravel()   # flatten (N,1) labels to (N,) for stratify compatibility

        # no transpose guessing. If shapes disagree, aligned to the
        # SHORTER sample count explicitly, with a warning — this is the
        # honest "we don't know which samples correspond.
        if X.shape[0] != y.shape[0]:
            print(f'[⚠️] X/y sample count mismatch: X={X.shape[0]}, '
                f'y={y.shape[0]} — aligning to shorter length. '
                f'This likely indicates an upstream bug where X and y '
                f'became desynchronized; investigate the caller.')
            min_samples = min(X.shape[0], y.shape[0])
            X = X[:min_samples]
            y = y[:min_samples]

        if X.shape[0] < 4:
            print(f'[⚠️] Only {X.shape[0]} samples — too few for a train/val '
                f'split, returning all data as training set, no validation')
            return X, np.empty((0,) + X.shape[1:]), y, np.empty((0,) + y.shape[1:])

        dynamic_test_bed = self.dynamic_test_size(
            y, min_val_per_class=min_val_per_class,
            min_frac=min_frac, max_frac=max_frac
        )

        # check stratify viability BEFORE attempting it
        unique_classes, class_counts = np.unique(y, return_counts=True)
        can_stratify = np.all(class_counts >= 2) and len(unique_classes) > 1

        if not can_stratify:
            sparse_classes = unique_classes[class_counts < 2]
            print(f'[⚠️] Cannot stratify: class(es) {sparse_classes.tolist()} '
                f'have fewer than 2 samples — falling back to non-stratified split')

        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=dynamic_test_bed,
                stratify=y if can_stratify else None,
                random_state=random_state
            )
        except ValueError as e:
            raise Warning(f'[⚠️] train_test_split failed even without stratify: {e} — '
                 'The model cant learn with very Few Y Signals inside the sample,'
                f'Cannot proceed with Training and prediction and make sure to have Y samples with proper shapes!')

        return X_train, X_val, y_train, y_val

    def _compute_need_ensemble_method(self, anisotropy, AME, error_counts,
                                    anisotropy_threshold=0.3,
                                    ame_threshold=0.3,
                                    error_threshold=0.3,
                                    min_signals_required=1):
        """
        Ensemble fallback is a safety net — it should trigger if ANY
        single strong risk signal fires, not only when all signals
        happen to align simultaneously. Uses max() over error_counts
        so a single persistently-wrong class triggers assistance even
        when most other classes are healthy.
        """
        signals = []

        if anisotropy is not None and anisotropy > anisotropy_threshold:
            signals.append(f'anisotropy={anisotropy:.3f}')

        if AME is not None and AME > ame_threshold:
            signals.append(f'AME={AME:.3f}')

        if error_counts is not None and len(error_counts) > 0:
            # max: catches localized single-class failure
            # that a flat average would dilute away
            worst_class_idx = int(np.argmax(error_counts))
            max_error = float(error_counts[worst_class_idx])
            if max_error > error_threshold:
                signals.append(f'error_rate(class={worst_class_idx}, '
                            f'value={max_error:.2f})')

        need_ensemble = len(signals) >= min_signals_required

        if need_ensemble:
            print(f'[=] Ensemble method triggered by: {", ".join(signals)}')

        return need_ensemble


    def _check_for_transformer_sequences(self, sequence_inputs, min_seq_len=3, 
                                min_samples=10,  min_AME=0.5, min_anisotropy=0.5):
        """
        Decide whether it's actually worth running the (comparatively
        expensive, parameter-heavy) transformer.Transformer on `sequence_inputs`, or
        whether the data's shape/statistics indicate the mlp.MLP alone is the
        appropriate model.

        Computes a 0-5 "readiness score" by checking, independently:
        1. Enough samples (`n_samples >= min_samples`) to justify the
           transformer.Transformer's parameter count.
        2. Long enough sequences (`T >= min_seq_len`) for attention to have
           multiple positions to relate to each other.
        3. Sufficient anisotropy (`pipeline.anisotropy_measurement`) —
           low anisotropy suggests the data lacks directional/sequential
           structure attention would exploit.
        4. Bounded Abstract Modelling Error (`pipeline.AME_Encoder`) — high
           AME indicates sample complexity that may not translate into
           useful transformer signal.
        5. The mlp.MLP's own recent error rate (`pipeline.model3.error_counts`
           averaged) — if the mlp.MLP is already struggling, that's treated as
           a point in favor of trying the transformer.Transformer.

        Note: despite the name "score", each failed/borderline check adds
        to `score`, and the method only greenlights transformer.Transformer usage when
        `score == 5`, i.e. *every* check tipped in favor — this is a
        conservative, all-signals-required gate (contrast with
        `_compute_need_ensemble_method`, which triggers on any single
        strong signal).

        Args:
            sequence_inputs: Candidate sequence/feature batch to evaluate,
                coerced to an array; its shape determines `n_samples`/`T`.
            min_seq_len (int): Minimum sequence length threshold.
            min_samples (int): Minimum sample-count threshold.
            min_AME (float): Maximum acceptable Abstract Modelling Error
                before it counts against transformer.Transformer usage.
            min_anisotropy (float): Minimum anisotropy required.

        Returns:
            bool: True if all five checks favor using the transformer.Transformer,
            False otherwise.
        """
        score = 0
        sequence_inputs = np.asarray(sequence_inputs)
        n_samples = sequence_inputs.shape[0] if sequence_inputs.ndim > 0 else 0
        T = sequence_inputs.shape[1] if sequence_inputs.ndim > 1 else 1

        # ── Check 1: sample count ────────────────────────────────
        if n_samples < min_samples:
            score += 1
            print(f'[=] only {n_samples} samples (need >= {min_samples}) — '
                        f'too little data to justify transformer.Transformer parameter count')

        # ── Check 2: sequence length ─────────────────────────────
        if T < min_seq_len:
            score += 1
            print(f'[=] sequence length T={T} (need >= {min_seq_len}) — '
                        f'attention has too few positions to model '
                        f'meaningful relationships')

        # ── Check 3: anisotropy   ──
        # ── zero new computation cost                              ──
        anisotropy = self.pipeline.anisotropy_measurement(sequence_inputs)
        if anisotropy < min_anisotropy:
            score += 1
            print(f'[=] anisotropy={anisotropy:.4f} (need >= {min_anisotropy}) — '
                        f'data lacks directional/sequential structure; '
                        f'mlp.MLP alone is the appropriate model for this '
                        f'geometry.')  


        AME = self.pipeline.AME_Encoder(sequence_inputs)  
        if AME > min_AME:
            score += 1 
            print(f'[=] Result of Abstract Modelling Error is High, Complexity in samples is High.')

        error_counts = np.mean(self.pipeline.model3.error_counts)
        if error_counts > 0.5:
            score += 1
            print('[=] mlp.MLP Experienced lots of Errors, Considering to use transformer.Transformer Prediction.')

        propriate_single_transformer = score == 5
        print(f'[+] transformer.Transformer Prediction allowed: {propriate_single_transformer}')
        return propriate_single_transformer

    def advanced_prediction_method(self, titles=None, label_map=None, rules=None,
                                X=None, y=None,
                                show_proba=False, top_k=3, 
                                use_transformer=True, return_attention=False,
                                save_results=True, batch_size=2):
        """
        Primary, "best effort" prediction entry point for the pipeline —
        the most feature-complete of the three prediction methods on this
        class (`regular_prediction_method` and `simple_pass_prediction`
        being the simpler/faster alternatives). This is what
        `peer_agent.ConsecutivePeerAgent.predict_local` calls for the rich-input case,
        and is expected to be the main path used in production/autonomous
        operation.

        End-to-end flow:
        1. **Feature preparation** — if `X`/`y` are not supplied, derives
           them from `titles`/`rules` via `pipeline.data_preparation` +
           `pipeline.mlp_training_features`; ensures the transformer.Transformer's
           supporting state is up to date via `pipeline.transformer_utilities`,
           and builds `input_ids`/`sequence_ids` token encodings
           (falling back to `_features_to_sequence` when direct dataset
           encoding produces too many "weak" all-zero rows).
        2. **transformer.Transformer decision** — if `use_transformer` and a
           vocabulary is available, runs the transformer.Transformer
           (`model2.forward`) over the encoded input, computing
           `anisotropy` (`pipeline.anisotropy_measurement`) and Abstract
           Modelling Error (`AME`, `model2.AME_Encoder`) along the way;
           otherwise proceeds mlp.MLP-only and still computes best-effort
           anisotropy/AME estimates for use in ensemble-need scoring.
        3. **Cache short-circuit** — before doing full model inference,
           checks `pipeline.accurate_cache_lookup` for a sufficiently
           similar previously-seen sample (similarity >= 0.95); if found,
           the cached prediction/confidence is reused directly, skipping
           the (expensive) ensemble computation below.
        4. **mlp.MLP forward pass** — runs `model3.forward`, clamping any
           out-of-range predicted class indices into a valid range the
           same way `regular_prediction_method` does.
        5. **Per-sample fusion** — for samples not served by the cache,
           calls `_compute_need_ensemble_method` (using the anisotropy/AME/
           `error_counts` computed above) to decide, per batch, whether the
           calibrated-ensemble blend or the simple-heuristic blend should
           be used, then delegates the actual per-sample number-crunching
           to `_compute_sample_prediction` for each item.
        6. **Calibration & penalization** — applies
           `calibration_penalized_check` to temper overconfident results
           against this manager's running `error_counts`/`pred_counts`.
        7. **Second-opinion / autonomous override** — under certain
           conditions (e.g. results retrieved from history/cache
           disagreeing with the fresh prediction), computes a second
           candidate label (`sec_chosen_label`/`sec_confidence`); if that
           second prediction scores a higher accuracy in
           `display_hybrid_results` (`sec_correct > correct`), or if the
           pipeline is running `autonomous` and the second prediction's
           confidence clears `pipeline.confidence_threshold`, the second
           label overrides the primary one as the final answer.
        8. **Wrap-up** — logs a results breakdown via
           `display_hybrid_results`, updates model performance stats
           (`pipeline.evaluate_mlp_performance`), clears
           `pipeline.cache` (this call's transient cache state is
           considered spent once a final answer is produced), and returns.

        On any unhandled exception during the advanced flow, the method
        catches it, logs a traceback, and falls back to
        `regular_prediction_method` with the same inputs; if even that
        fails, it returns a safe `(None, None, 0.0)`-shaped result rather
        than raising, so callers (e.g. the P2P prediction path) always get
        a usable return value.

        Args:
            titles: Batch of input texts (or `(text, expected_label)`
                tuples) to predict on.
            label_map (Dict[str, int]): String label -> class index map;
                required (raises `ValueError` if None).
            rules: Feature-engineering rules used to auto-derive `X`/`y`
                when not supplied.
            X, y: Optional pre-computed features/targets.
            show_proba (bool): Whether to retain per-class probability
                detail for display/return.
            top_k (int): Number of top classes to report in probability
                breakdowns.
            use_transformer (bool): Whether to attempt transformer.Transformer
                inference at all (subject to vocabulary availability).
            return_attention (bool): Whether attention weights should be
                retained/exposed alongside the prediction (consumed by
                downstream display/return logic).
            save_results (bool): Whether this call's results should be
                persisted (e.g. into pipeline caches/history).
            batch_size (int): Batch size forwarded to
                `pipeline.transformer_utilities`.

        Returns:
            Tuple[Any, Any, float]: `(results, chosen_label, confidence)`
            — `results` is the detailed per-sample result list/dict
            structure built during the method (or from
            `regular_prediction_method`/None on total failure),
            `chosen_label` is the final predicted label (possibly the
            "second opinion" label if it won out), and `confidence` is the
            associated confidence score.

        Raises:
            ValueError: If `label_map` is None.
        """
        try:
            # ____ init temporary layer ____
            eps = 1e-5
            trans_probs = None
            attn_weights = None
            sequence_ids = None

            X_train = None
            X_val = None
            y_val = None
            y_train = None

            input_ids = None
            anisotropy = None
            final_class_idx = None

            AME = None
            use_embedded = False
            dataset = None
            target_probs = None

            X_gen = None
            sec_chosen_label = None

            sec_confidence = 0.0
            final_confidence = 0.0
            min_signal = 1e-3

            correct = 0
            sec_correct= 0
            performance_score = self.pipeline.performance_result

            if label_map is None:
                raise ValueError("[!] label_map must be provided for all prediction methods!")

            print("\n[🚀] Starting Advanced Hybrid Prediction Method")
            
            reverse_map = {v: k for k, v in label_map.items()}
            num_classes = self.pipeline._get_num_classes(label_map=label_map)

            self.pipeline.titles = titles
            self.pipeline.labels = label_map

            if X is not None and y is not None:
                print(f"[🔍] Processing X and y samples for prediction and Training...")
                if isinstance(X, (str, np.str_)):
                    clean_str = str(X).replace('[', '').replace(']', '')
                    X = np.fromstring(clean_str, sep=' ')
                if isinstance(X, np.ndarray) and np.issubdtype(X.dtype, np.character):
                    # catches arrays filled with string text
                    clean_str = ' '.join(X.astype(str).flatten()).replace('[', '').replace(']', '')
                    X = np.fromiter(
                            (x for x in clean_str.split() if x != "..."), dtype=float
                        )   

                if isinstance(y, (str, np.str_)):
                    clean_str = str(y).replace('[', '').replace(']', '')
                    y = np.fromstring(clean_str, sep=' ')
                if isinstance(y, np.ndarray) and np.issubdtype(y.dtype, np.character):
                    # catches arrays filled with string text
                    clean_str = ' '.join(y.astype(str).flatten()).replace('[', '').replace(']', '')
                    y = np.fromiter(
                            (x for x in clean_str.split() if x != "..."), dtype=float
                        )
                

                if len(X) < 5:
                    print(f"[⚠️] Warning: Only {len(X)} samples provided. Consider providing more samples for better prediction accuracy!")

                if 0 in X.shape:
                    print(f"[⚠️] Warning: X has zero samples in the total shapes, X shapes: {X.shape}. Creating an empty array with shape (1, n_features) for processing..")
                    X = np.empty((1, X.shape[1]))  # Created an empty array with shape (1, n_features)
                    X = X.reshape(1, -1)  # Reshape to (1, n_features) if empty but has features

                X_train, X_val, y_train, y_val = self._prepare_train_val_split(
                    X, y, min_val_per_class=5, min_frac=0.1, max_frac=0.3
                )
                
                onehot_validation = self.pipeline._validate_onehot(y_train)
                if onehot_validation:
                    if num_classes != len(label_map):
                        num_classes = len(label_map)
                    if num_classes > np.max(y_train):
                        y_train = np.eye(num_classes)[np.asarray(y_train)]
                        y_val = np.eye(num_classes)[np.asarray(y_val)]
                    else:
                        print('[⚠️] Warning: Y onehot encoding fails, Returning y samples as is, This may cause Exploding Gradient in mlp.MLP Training!')
                        y_train = y_train
                        y_val = y_val

                X_mean = X_train.mean(axis=0)
                X_std = X_train.std(axis=0) + 1e-8

                X_train = (X_train - X_mean) / X_std
                X = (X_val - X_mean) / X_std
                y = y_val.copy()

            if titles is not None and rules is not None:
                print(f"[🔍] Preparing data for {len(titles)} titles with {len(rules)} length of rules.")
                if X is None and y is None or X is None or y is None:
                    print('[🔄] Creating automatic X samples because X is not provided manually.')
                    dataset, X_gen = self.pipeline.data_preparation(titles, label_map)  
                    _, y, _, _ = self.pipeline.mlp_training_features(rules, dataset)                  
                else:
                    dataset, _ = self.pipeline.data_preparation(titles, label_map)
                    
            if y_train is None:
                y_train = y
            if X_train is None:
                X = X_gen if X_gen is not None else X

            if X_gen is not None:
                self.pipeline.transformer_utilities(X_provided=X_train, X_raw=X_gen, y_true=y_train, rules=rules, datasets=dataset, label_map=label_map, batch_size=batch_size, max_samples_for_focused_fit=len(X))
            else:
                self.pipeline.transformer_utilities(X_provided=X_train, X_raw=X, y_true=y_train, rules=rules, datasets=dataset, label_map=label_map, batch_size=batch_size, max_samples_for_focused_fit=len(X))

            if dataset is not None:
                input_ids, _ = self.pipeline.input_encoding(dataset)
            elif dataset is not None and X_train is not None:
                input_ids, _ = self.pipeline.input_encoding(dataset)
                row_sums = input_ids.sum(axis=1)
                weak_rows = np.where(row_sums < min_signal)[0]
                weak_ratio = len(weak_rows) / len(input_ids)
                if weak_ratio > 0.3:
                    print(f'[=] Zero rows abundant in Input indices: {weak_ratio:.1%}, using input indices from X samples...')
                    input_ids = self.pipeline._features_to_sequence(X_train)
            else:
                input_ids = self.pipeline._features_to_sequence(X_train)


            if len(input_ids) == 0:
                print('[⚠️] input_encoding produced no samples — skipping '
                    'transformer training/prediction for this call and transformer results will be Replaced by mlp.MLP or LSTM.')
                self.pipeline.use_transformer = False
    
            if use_transformer and hasattr(self.pipeline, 'vocab') and self.pipeline.vocab:
                print("\n[🔄] Running transformer.Transformer prediction method (transformer.Transformer)")
                if dataset is not None:
                
                    input_ids_list = []
                    for title in titles:
                        if isinstance(title, tuple):
                            title = title[0]
                        ids = self.pipeline.encode(title, self.pipeline.vocab)
                        input_ids_list.append(np.array(ids))
                    
                    input_ids = np.array(input_ids_list)
                    sequence_ids = self.pipeline.sequence_encoding(dataset, label_map=label_map)
                    if X is not None:
                        anisotropy = self.pipeline.anisotropy_measurement(X_train)
                    else:
                        anisotropy = self.pipeline.anisotropy_measurement(sequence_ids)

                    # Get transformer predictions with attention
                    print(f"[⚡] anisotropy rate detected on input: {anisotropy:.1%}.")                
                    use_embedded = True
                    
                    AME = self.pipeline.model2.AME_Encoder(sequence_ids)                
                    trans_probs, attn_weights = self.pipeline.model2.forward(sequence_ids, AME=AME, embedded=use_embedded)

                else:
                    if X is not None:
                        anisotropy = self.pipeline.anisotropy_measurement(X_train)
                    else:
                        anisotropy = self.pipeline.anisotropy_measurement(input_ids)

                    # Get transformer predictions with attention
                    print(f"[⚡] anisotropy rate detected on input: {anisotropy:.1%}.")                
                    use_embedded = True

                    AME = self.pipeline.model2.AME_Encoder(input_ids) 

                    trans_probs, attn_weights = self.pipeline.model2.forward(input_ids, AME=AME, embedded=use_embedded)
            
            else:
                print("\n[⚡] Running mlp.MLP-only predictions")
                print("[⚡] Note: transformer.Transformer not available, so transformer.Transformer results will be replaced with mlp.MLP results.")

                anisotropy = self.pipeline.anisotropy_measurement(X) if X is not None else self.pipeline.confidence_threshold

                try:
                    AME = self.pipeline.model2.AME_Encoder(X) if X is not None else self.pipeline.confidence_threshold
                except:
                    AME = self.pipeline.confidence_threshold


            if X is None or len(X) == 0 or isinstance(X, int) or (isinstance(X, np.ndarray) and X.size == 0):
                # Get mlp.MLP predictions
                titles = None
                if titles is not None and len(titles) > 0:
                    if isinstance(titles[0], tuple):
                        mlp_titles = [t[0] for t in titles]
                    else:
                        mlp_titles = titles
                    
                    if not hasattr(self.pipeline, 'tfidf') or self.pipeline.tfidf is None:
                        self.pipeline.initialize_fitting(mlp_titles)

                    if isinstance(mlp_titles, (list, tuple, np.ndarray)):   
                        titles = mlp_titles[0]    
                        if isinstance(mlp_titles, (list, tuple, np.ndarray)):     
                            titles = titles[0] 
                    
                    if not isinstance(mlp_titles, list) or isinstance(titles, str):
                        if titles is not None:
                            mlp_titles = titles
                        X = self.pipeline.tfidf.transform(mlp_titles).toarray()
                
                if X_gen is not None:
                    X = X_gen
                if X is not None and len(X) > 0:
                    X = np.asarray(X)
                
            # mlp.MLP forward pass  
            mlp_probs = self.pipeline.model3.forward(X, y=y_val)
          
             # Validate all mlp.MLP predictions at once
            mlp_pred_indices = np.argmax(mlp_probs, axis=1)
            if num_classes <= 0:
                num_classes = self.pipeline._get_num_classes(mlp_probs=mlp_probs)

            valid_mask = mlp_pred_indices < num_classes
            if not np.all(valid_mask):
                invalid_count = np.sum(~valid_mask)
                # Replace invalid indices with argmax within valid range
                for i in range(len(mlp_pred_indices)):
                    valid_probs = mlp_probs[i][:num_classes] if num_classes > 0 else mlp_probs[i]
                    if len(valid_probs) > 0 and i < len(mlp_pred_indices):
                        mlp_pred_indices[i] = int(np.argmax(valid_probs))
                    else:
                        mlp_pred_indices[i] = 0  # Default to first class  

            if sequence_ids is not None:
                print("\n[🔍] Using sequence encoding for transformer input due to low anisotropy.")
                input_ids = sequence_ids.copy()
            # verify samples for accurate answer from cache
            print('[🔍] Verifying Samples for possible predicted output in cache for accurate answer...')
            cached = self.pipeline.accurate_cache_lookup.lookup(
                x_mlp=X, 
                input_ids=input_ids)

            if cached is not None:
                if cached['similarity'] >= 0.95:                
                    print(f"[💎] Using verified cache "
                    f"(combined_sim={cached['similarity']:.1%}, "
                    f"hits={cached['hit_count']})")
                    result = {
                        'predicted': cached['prediction'],
                        'confidence': float(cached['confidence']),
                        'index': int(cached['index']),
                        'models_agree': True,
                        }    

                    print(f"\n[💎] Verified chosen label for samples: {cached['prediction']} || Confidence: {cached['confidence']:.1%}")  
                    
                    return result, cached['prediction'], cached['confidence']
                else:
                    print(f'[!] Similarity: {cached['similarity']} is low, Cannot pick label due to low certainty, Initiating advanced prediction...')
                    target_probs = mlp_probs.copy()
            else:
                print('[=] No verified output from cache available that matched samples, starting advanced prediction...')
                if self.pipeline.use_transformer:
                    if isinstance(input_ids, (list, np.ndarray)):
                        use_embedded = True

                    target_probs = self.pipeline.predict_proba(input_ids, X, type='Hybrid', embedded=use_embedded)
                else:
                    target_probs = mlp_probs.copy()
           
                target_probs = target_probs[:mlp_probs.shape[0], :mlp_probs.shape[1]] 
                target_probs = self.pipeline.model3.continuous_predictive_correction(self, target_probs, mlp_pred_indices, AME=AME, anisotropy=anisotropy)  
            
            target_pred_indices = np.argmax(target_probs, axis=1)    

            if self.pipeline.cache and 'label_bins' in self.pipeline.cache:
                print('[=] label_bins cache found!')
                label_bins = self.pipeline.cache['label_bins']
                lstm_probs, _ = self.pipeline.ensemble._get_lstm_probs(input_ids, X, label_bins=label_bins)      
            else:
                lstm_probs = None

            
            threshold = 0.5 + (self.pipeline.confidence_threshold + np.mean(self.error_counts)) / 2
            need_ensemble_method = self._compute_need_ensemble_method(anisotropy, AME, self.error_counts,
                                    anisotropy_threshold=threshold,
                                    ame_threshold=threshold,
                                    error_threshold=threshold,
                                    min_signals_required=1)
            
            results = []
            attention_data = [] if return_attention else None

            if titles is not None and len(titles) > 0:
                for i, title in enumerate(titles):
                    # Parse input
                    if isinstance(title, tuple):
                        display_title = title[0]
                        expected_label = title[1] if len(title) > 1 else None
                    else:
                        display_title = title
                        expected_label = None
                    
                    # mlp.MLP prediction     
                    if i < len(mlp_pred_indices):           
                        mlp_class_idx = int(mlp_pred_indices[i])
                    else:
                        mlp_class_idx = int(target_pred_indices[i] if i < len(target_pred_indices) else target_pred_indices[0])

                    is_valid_index = 0 <= mlp_class_idx < num_classes
                    if not is_valid_index:
                        print(f'[⚠️] Invalid mlp_class_idx={mlp_class_idx} for sample {i} '
                            f'(num_classes={num_classes}, title="{display_title}") '
                            f'— marking as low-confidence unknown, NOT defaulting to class 0')

                        results.append({
                            'title'          : display_title,
                            'expected'       : expected_label,
                            'predicted'      : None,            # explicit unknown, not a fake class
                            'confidence'     : 0.0,
                            'mlp_class'      : None,
                            'is_valid'       : False,
                            'error'          : f'class_index_out_of_range(idx={mlp_class_idx}, num_classes={num_classes})'
                        })      
                        continue              

                    if i < len(mlp_probs):
                        mlp_confidence = mlp_probs[i][mlp_class_idx]
                    else:
                        mlp_confidence = target_probs[i][mlp_class_idx] if i < len(target_probs) else target_probs[0][mlp_class_idx]

                    mlp_label = reverse_map.get(mlp_class_idx, f"unknown_{mlp_class_idx}")

                    if lstm_probs is not None:
                        lstm_pred_indices = np.argmax(lstm_probs, axis=1)
                        lstm_class_idx = lstm_pred_indices[i]              
                        lstm_confidence = lstm_probs[i][lstm_class_idx]
                    else:
                        lstm_confidence = None

                    if i < len(target_pred_indices):
                        target_class_idx = target_pred_indices[i]
                    else:
                        target_class_idx = target_pred_indices[0]
                    
                    if i < len(target_probs):
                        target_confidence = target_probs[i][target_class_idx]
                    else:
                        target_confidence = target_probs[0][target_class_idx]
                
                    # transformer.Transformer prediction and blending
                    if trans_probs is not None and attn_weights is not None:
                        if i < len(trans_probs):
                            trans_probs_i = trans_probs[i]

                        trans_class_idx = np.argmax(trans_probs_i)
                        if isinstance(trans_probs_i, float):
                            trans_confidence = target_confidence
                        else:
                            trans_confidence = trans_probs_i[trans_class_idx]

                        trans_label = reverse_map.get(trans_class_idx, f"unknown_{trans_class_idx}")

                        if need_ensemble_method:
                            print(f"[🔄] Ensemble method activated for sample {i} due to high anisotropy")
                            calibration = self.pipeline._calibrate_probs(target_probs, target_pred_indices, attn_weights, input_ids)
                            # Blend predictions (mlp.MLP decides class, transformer calibrates confidence)
                            mlp_weight = mlp_confidence / (target_confidence + trans_confidence + eps)
                            trans_weight = trans_confidence / (target_confidence + trans_confidence + eps)
                            if lstm_confidence is not None:
                                lstm_weight = lstm_confidence / (target_confidence + lstm_confidence + eps)
                                
                            calibration_weighting = calibration[target_class_idx] if target_class_idx < len(calibration) else 0.0
                                
                            # Weighted blend: calibration_weighting * calibrated + (1-weight) * mlp
                            if lstm_confidence is not None and lstm_weight is not None:
                                final_probs = mlp_weight * target_probs[i][:len(calibration)] + trans_weight * calibration[i][:len(calibration)] + lstm_weight * calibration[i][:len(calibration)]
                            else:
                                final_probs = mlp_weight * target_probs[i][:len(calibration)] + trans_weight * calibration[i][:len(calibration)]
                                
                            final_probs = self.pipeline._calibrate_probs(final_probs, mlp_class_idx, attn_weights, input_ids)
                            final_class_idx = target_class_idx
                            try:
                                final_confidence = final_probs[final_class_idx]
                            except IndexError:
                                final_confidence = np.max(final_probs) if isinstance(final_probs, np.ndarray) else np.mean(final_probs)

                            if isinstance(final_confidence, np.ndarray):
                                final_confidence = np.max(final_confidence)

                            # Calculate agreement
                            agreement = mlp_class_idx == trans_class_idx
                        else:
                            print(f"[🔄] Ensemble method not activated for sample {i} due to unmet conditions")
                            if lstm_confidence is None:
                                lstm_confidence = mlp_confidence
                            if mlp_confidence > trans_confidence and mlp_confidence > lstm_confidence and not mlp_confidence > 0.95:
                                final_probs = mlp_probs[i] if i < len(mlp_probs) else mlp_probs[0]
                                final_class_idx = mlp_class_idx
                                final_confidence = mlp_confidence
                                print(f"[🔄] mlp.MLP chosen for sample {i} due to highest confidence: {mlp_confidence:.1%}")
                            elif trans_confidence > lstm_confidence:
                                final_probs = trans_probs[i] if i < len(mlp_probs) else trans_probs[0]
                                final_class_idx = trans_class_idx
                                final_confidence = trans_confidence
                                print(f"[🔄] transformer.Transformer chosen for sample {i} due to highest confidence: {trans_confidence:.1%}")
                            else:
                                chosen_probs = mlp_probs[i] if i < len(mlp_probs) else mlp_probs[0]

                                final_probs = lstm_probs[i] if lstm_probs is not None else mlp_probs[0]
                                final_class_idx = lstm_class_idx if lstm_probs is not None else mlp_class_idx
                                final_confidence = lstm_confidence if lstm_probs is not None else mlp_confidence
                                if final_confidence > 0.95:
                                    if mlp_probs.shape == lstm_probs.shape:
                                        final_probs = chosen_probs * lstm_probs[i]
                                        final_class_idx = mlp_class_idx
                                        final_confidence = mlp_confidence * lstm_confidence
                                    else:
                                        if lstm_probs is not None:
                                            final_probs = target_probs[i] if len(target_probs) == num_classes and i < len(target_probs) else lstm_probs[i]
                                        else:
                                            final_probs = trans_probs[i] if i < len(mlp_probs) else np.mean(trans_probs)
                                            
                                        final_class_idx = target_class_idx
                                        final_confidence = target_confidence * lstm_confidence

                                else:
                                    print(f"[🔄] No model chosen for sample {i} due to low confidence: mlp.MLP={mlp_confidence:.1%}, transformer.Transformer={trans_confidence:.1%}, LSTM={lstm_confidence:.1%}")

                            agreement = mlp_class_idx == trans_class_idx

                    else:
                        if i < len(mlp_probs):
                            final_probs = mlp_probs[i]
                        else:
                            final_probs = mlp_probs[0]

                        final_class_idx = target_class_idx
                        final_confidence = target_confidence[0] if isinstance(target_confidence, np.ndarray) else target_confidence
                        if isinstance(final_confidence, np.ndarray) or isinstance(final_confidence, list):
                            final_confidence = np.max(final_confidence)

                        trans_label = None
                        trans_confidence = 0.0
                        agreement = False
                    
                    final_label = reverse_map.get(final_class_idx, f"unknown_{final_class_idx}")
                    # Build result
                    result = {
                        'title': display_title,
                        'expected': expected_label,
                        'predicted': final_label,
                        'confidence': float(final_confidence),
                        'predicted_idx': int(final_class_idx),
                        'mlp_prediction': mlp_label,
                        'mlp_confidence': float(mlp_confidence),
                        'models_agree': bool(agreement),
                        'sec_predicted': None,
                        'sec_confidence': 0.0,
                        'sec_index': None,
                    }
                    
                    if trans_label is not None:
                        result['transformer_prediction'] = trans_label
                        result['transformer_confidence'] = float(trans_confidence)
                    
                    # Add top-k predictions
                    if isinstance(final_probs, (int, float)):
                        if use_transformer:
                            final_probs = self._calibrate_probs(mlp_probs, mlp_class_idx, attn_weights, input_ids)
                        else:
                            final_probs = mlp_probs

                    final_probs = final_probs[:num_classes] if num_classes > 0 else final_probs
                    if show_proba:
                        top_indices = np.argsort(final_probs)[-top_k:][::-1]
                        result['top_predictions'] = [
                            {
                                'label': reverse_map.get(idx, f"unknown_{idx}"),
                                'confidence': float(final_probs[idx])
                            }
                            for idx in top_indices if idx in reverse_map
                        ]
                        
                        # mlp.MLP top predictions
                        mlp_probs_i = mlp_probs[i][:num_classes] if num_classes > 0 else mlp_probs[i]
                        mlp_top = np.argsort(mlp_probs_i)[-top_k:][::-1]
                        result['mlp_top'] = [
                            {
                                'label': reverse_map.get(idx, f"unknown_{idx}"),
                                'confidence': float(mlp_probs_i[idx])
                            }
                            for idx in mlp_top if idx in reverse_map
                        ]
                        
                        # transformer.Transformer top predictions
                        if trans_probs is not None:
                            if trans_probs.ndim > 1:
                                trans_probs = trans_probs[i][:num_classes] if num_classes > 0 else trans_probs[i]
                            else:
                                trans_probs = trans_probs.copy()
                            if trans_probs is not None:
                                trans_top = np.argsort(trans_probs)[-top_k:][::-1]
                                result['transformer_top'] = [
                                    {
                                        'label': reverse_map.get(idx, f"unknown_{idx}"),
                                        'confidence': float(trans_probs[idx])
                                    }
                                    for idx in trans_top if idx in reverse_map
                                ]
                    
                    results.append(result)
                    
                    # Collect attention data if requested
                    if return_attention and attn_weights is not None:
                        attention_data.append({
                            'title': display_title,
                            'attention': attn_weights[i].tolist() if i < len(attn_weights) else None
                        })

            else:
                print("[=] Initiating Continuous sample prediction without Titles.")

                transformer_takes = self._check_for_transformer_sequences(input_ids, min_seq_len=3, 
                                min_samples=10, min_AME=0.5, min_anisotropy=0.5)
                if not transformer_takes:
                    n_samples = mlp_probs.shape[0]
                    chosen_probs = mlp_probs.copy()
                else:
                    n_samples = trans_probs.shape[0]
                    chosen_probs = trans_probs.copy()

                lstm_pred_indices = np.argmax(lstm_probs, axis=1) if lstm_probs is not None else None
                for i in range(n_samples):
                    outcome = self._compute_sample_prediction(
                        i, chosen_probs, target_probs, target_pred_indices,
                        trans_probs=trans_probs, lstm_probs=lstm_probs,
                        lstm_pred_indices=lstm_pred_indices,
                        attn_weights=attn_weights, input_ids=input_ids,
                        num_classes=num_classes, reverse_map=reverse_map
                    )

                    result = {
                        "title": f"Unknown",
                        "expected": 'Unknown',
                        **outcome
                    }
                    results.append(result)

                if results is not None and isinstance(results[0], dict):
                    try:
                        final_probs = results[0]['final_probs'] if results else None
                        final_class_idx = results[0]['predicted_idx'] if results else None
                        agreement = results[0]['models_agree'] if results else None
                    except:
                        final_probs = results[0].get('final_probs', None)
                        final_class_idx = results[0].get('predicted_idx', None)
                        agreement = results[0].get('models_agree', None)
                else:
                    final_probs = mlp_probs
                    final_class_idx = target_pred_indices
                    agreement = False

            # Display results
            verbose = False
            if float(results[0]['confidence']) < self.pipeline.confidence_threshold:
                verbose = True

            chosen_label = results[0].get('predicted') if results else None
            confidence = results[0].get('confidence') if results else 0.0
            if chosen_label is None:
                for i in range(num_classes):
                    mlp_class_idx = int(mlp_pred_indices[i])
                    is_valid_index = 0 <= mlp_class_idx < num_classes

                    if mlp_class_idx > len(reverse_map) or not is_valid_index:
                        mlp_class_idx = int(np.argmax(mlp_probs[:len(reverse_map)-1]))
                        print(f"[⚠️] Clamping index {mlp_class_idx} → {mlp_class_idx}")  

                    if is_valid_index:
                        mlp_confidence = float(mlp_probs[i][mlp_class_idx]) 
                    else:
                        mlp_confidence = float(mlp_probs[:len(reverse_map)-1][mlp_class_idx]) 
                        
                    mlp_label      = reverse_map.get(mlp_class_idx, f"unknown_{mlp_class_idx}")
                    results.append({
                        'predicted'  : mlp_label,
                        'confidence' : mlp_confidence,
                        'mlp_class'  : mlp_class_idx,
                        'index'      : mlp_class_idx,
                        'models_agree': False,
                        'final_probs' : mlp_probs,
                        'is_valid'   : True,
                    })
                
                chosen_label = results[0].get('predicted') if results else None
                confidence = results[0].get('confidence') if results else 0.0

            if isinstance(chosen_label, int) or isinstance(chosen_label, np.integer):
                chosen_label = str(chosen_label)
                
            if isinstance(confidence, (np.ndarray, list)):
                confidence = np.mean(confidence)

            return_single_condition = (
                not self.pipeline.use_transformer or
                not need_ensemble_method 

            )
            if return_single_condition:
                print('[=] Displaying Results....')
                if not results[0].get('models_agree', True) and self.pipeline.use_transformer:
                    trans_confidence = results[0].get('trans_confidence', 1e-8)
                    confidence = (confidence + trans_confidence / 2) + 1e-5
                    print(f'[=] Calibrating confidence around: {confidence:.1%}')
                    
                payload = {
                    'X_samples': X,
                    'input_ids': input_ids
                }
                print('[=] Displaying Test Performance Results....')
                if titles is not None and len(titles) > 0:
                    correct, sec_correct = self.display_hybrid_results(payload, final_class_idx, results, top_k, verbose=True)
                    
                if isinstance(chosen_label, str) and chosen_label.startswith("unknown") or float(confidence) < self.pipeline.confidence_threshold:
                    if chosen_label is None or chosen_label.startswith('unknown'):
                        chosen_label = 'Unknown'
                        confidence = 1.0 - confidence  #Invert confidence for unknown class
                        print(f"\n[⚠️] Final prediction is {chosen_label} with uncertain confidence: {confidence:.1%}. Consider more consistent data for the model to learn from.")
                    else:
                        print(f"\n[🎯] Predicted label: {chosen_label} || With Certain Confidence: {confidence:.1%}")
                    
                return results, chosen_label, confidence

            else:
                if results[0].get('models_agree', True) and confidence > self.pipeline.confidence_threshold and not chosen_label.startswith("unknown"):
                    print(f"\n[🎯] Proper Confidence of Final chosen label for input: {chosen_label} || Confidence: {confidence:.1%}")
                    return results, chosen_label, confidence
                    
                # Only recalibrate if models disagreed
                elif results and not results[0].get('models_agree', True) or not self.pipeline.agreement and self.pipeline.use_transformer:

                    need_peer_condition = not results[0].get('models_agree', True) and self.pipeline.peer_assistance_threshold > 0.3 and self.pipeline.use_transformer
                    print("\n[⚠️] Disagreement detected between mlp.MLP and transformer.Transformer predictions. Usig calibrated probabilities for final decision.")
                    if confidence < self.pipeline.confidence_threshold and need_peer_condition:
                        print('|| Uncertain advanced prediction, requesting peer assistance if allowed...')
                        final_probs = self.pipeline._handle_distributed_connections(final_probs, attn_weights, input_ids, agreement) 

                        final_idx = final_probs[0].argmax()
                        original_idx = final_idx

                        if final_idx > len(reverse_map):
                            final_idx = int(np.argmax(final_probs[:len(reverse_map)-1]))
                            print(f"[⚠️] Clamping {final_idx} → {final_idx}")               
                        final_idx = int(final_idx)  

                        chosen_label = reverse_map.get(final_idx, f"unknown_{final_idx}")
                        if chosen_label.startswith('unknown'):
                            final_idx = int(np.argmax(final_probs[:len(reverse_map)-1]))

                        try:
                            confidence = float(final_probs[final_idx])   
                        except:
                            confidence = float(final_probs[0][len(reverse_map)-1]) if isinstance(final_probs[0], (float, int)) else 0.0      
               
                    elif self.pipeline.autonomous and need_peer_condition and attn_weights is not None:
                        if agreement is None:
                            agreement = False

                        print('[||] Iniating local peer output search in database for best output...')
                        final_probs = self.pipeline.distribution._handle_peer_agent_request(final_probs, attn_weights, input_ids, type='DevicePeer', agreement=agreement)

                        final_idx = final_probs[0].argmax()
                        original_idx = final_idx              

                        if final_idx > len(reverse_map):
                            final_idx = int(np.argmax(final_probs[:len(reverse_map)-1]))
                            print(f"[⚠️] Clamping {final_idx} → {final_idx}")               
                        final_idx = int(final_idx)  

                        chosen_label = reverse_map.get(final_idx, f"unknown_{final_idx}")
                        try:
                            confidence = float(final_probs[final_idx])   
                        except:
                            confidence = float(final_probs[0][len(reverse_map)-1]) if isinstance(final_probs[0], (float, int)) else 0.0      


                    elif not results[0].get('models_agree', True) and confidence > self.pipeline.confidence_threshold:
                        if final_confidence is not None and confidence < self.pipeline.confidence_threshold:
                            print("\n[⚠️] Low confidence detected, but both models don't agree. Using calibrated probabilities for final decision to ensure robustness.")
                            final_probs = self.pipeline.hybrid_prediction(rules, input_ids, dataset, X=X, y=y, use_embedded=use_embedded)
                        
                            final_idx = final_probs[0].argmax()
                            original_idx = final_idx
                            
                            if final_idx > len(reverse_map):
                                final_idx = int(np.argmax(final_probs[:len(reverse_map)-1]))
                                print(f"[⚠️] Clamping {final_idx} → {final_idx}")               
                            final_idx = int(final_idx)  
                        else:
                            print('[🎯] Stable confidence established, But both Models doesnt Agree, Re-evaluating...')   
                            final_probs = self.pipeline.hybrid_prediction(rules, input_ids, dataset, X=X, y=y, use_embedded=use_embedded)
                        
                            final_idx = final_probs[0].argmax()
                            original_idx = final_idx
                            if final_idx > len(reverse_map):
                                final_idx = int(np.argmax(final_probs[:len(reverse_map)-1]))
                                print(f"[⚠️] Clamping {final_idx} → {final_idx}")               
                            final_idx = int(final_idx)    

                        chosen_label = reverse_map.get(final_idx, f"unknown_{final_idx}")
                        try:
                            confidence = float(final_probs[0][final_idx])   
                        except:
                            confidence = float(final_probs[0][len(reverse_map)-1]) if isinstance(final_probs[0], (float, int)) else 0.0             
                    else:
                        if self.pipeline.use_transformer and need_ensemble_method:
                            print("\n[⚠️] Uncertain confidence and disagreement detected. Using ensemble method for final decision.")
                            input_forward = sequence_ids if sequence_ids is not None else input_ids
                            final_probs, details = self.pipeline.ensemble.predict_ensemble(input_forward, X, y, method='dynamic', embedded=use_embedded)
                        
                            final_idx = final_probs[0].argmax()
                            original_idx = final_idx 

                            if final_idx > len(reverse_map):
                                final_idx = int(np.argmax(final_probs[:len(reverse_map)-1]))
                                print(f"[⚠️] Clamping {final_idx} → {final_idx}")               
                            final_idx = int(final_idx)          

                            chosen_label = reverse_map.get(final_idx, f"unknown_{final_idx}")
                            try:
                                confidence = float(final_probs[0][final_idx])   
                            except:
                                confidence = float(final_probs[0][len(reverse_map)-1]) if isinstance(final_probs[0], (float, int)) else 0.0             
                        else:
                            if final_probs is None:
                                final_probs = mlp_probs

                            final_probs = self.calibration_penalized_check(final_probs, target_pred_indices[0])

                            final_idx = final_probs[0].argmax() if final_probs is not None else target_probs[0].argmax()
                            
                            original_idx = final_idx 

                            if final_idx > len(reverse_map):
                                final_idx = int(np.argmax(final_probs[:len(reverse_map)-1]))
                                print(f"[⚠️] Clamping {final_idx} → {final_idx}")               
                            final_idx = int(final_idx)          

                            chosen_label = reverse_map.get(final_idx, f"unknown_{final_idx}")
                            if final_probs is None:
                                final_probs = target_probs.copy()

                            try:
                                try:
                                    confidence = float(final_probs[0][final_idx])   
                                except:
                                    confidence = float(final_probs[0][len(reverse_map)-1]) if isinstance(final_probs[0], (float, int)) else 0.0             
                            except:
                                try:
                                    confidence = float(final_probs[final_idx]) 
                                except:
                                    confidence = self.pipeline.confidence_threshold

                elif confidence < self.pipeline.confidence_threshold and not self.pipeline.agreement and not results[0].get('models_agree', True):
                    if trans_probs is not None:
                        prob_entropy = self.calculate_entropy(final_probs)
                        normalized_entropy = prob_entropy / np.log(prob_entropy.shape[-1]) if prob_entropy.shape[-1] > 1 else 0
                        attn_quality = 1.0 / (1.0 + np.exp(-attn_weights.mean()) + eps) if attn_weights is not None else 0.5
                        anisotropy = self.pipeline.anisotropy_measurement(attn_weights.mean() if attn_weights is not None else 0.5)

                    else:
                        normalized_entropy = self.calculate_entropy(input_ids)  # Max entropy for uniform distribution
                        attn_quality = self.pipeline.confidence_threshold
                        anisotropy = self.anisotropy_measurement(input_ids) if hasattr(self.pipeline, 'anisotropy_measurement') else 0.5

                    mean_entropy = np.mean(normalized_entropy)

                    use_simple_prediction = (
                    anisotropy < 0.3 or
                    mean_entropy < 0.5 or  # High uncertainty
                    results[0].get('confidence', 0) < 0.4 or  # Low confidence
                    not results[0].get('models_agree', True) or  # Disagreement
                    attn_quality < 0.4
                    )

                    if use_simple_prediction:
                        print("\n[⚡] Condition is poorly unviable to handle agreement. Using robust prediction method for better reliability.")
                        predicted_label, confidence = self.simple_pass_prediction(self.pipeline, titles=titles, label_map=label_map, X_raw=X, y=y, show_proba=show_proba, top_k=top_k)
                        if predicted_label is not None:
                            print(f"\n[🎯] Robust prediction result: {predicted_label} with confidence {confidence:.1%}")
                            return _, predicted_label, confidence

                    else:
                        final_idx = final_probs[0].argmax()
                        original_idx = final_idx

                        if final_idx > len(reverse_map):
                            final_idx = int(np.argmax(final_probs[:len(reverse_map)-1]))
                            print(f"[⚠️] Clamping {final_idx} → {final_idx}")                      
                        final_idx = int(final_idx) 

                        chosen_label = reverse_map.get(final_idx, f"unknown_{final_idx}")
                        if final_probs is not None:
                            try:
                                confidence = float(final_probs[0][final_idx])   
                            except:
                                confidence = float(final_probs[0][len(reverse_map)-1]) if isinstance(final_probs[0], (float, int)) else 0.0  
            
                        else:
                            final_probs = target_probs.copy() 

                else:
                    print("\n[🎯] Using initial Regular final prediction as final decision.")
                    final_idx = final_probs[0].argmax()

                    if final_idx > len(reverse_map):
                        final_idx = int(np.argmax(final_probs[:len(reverse_map)-1]))
                        print(f"[⚠️] Clamping {final_idx} → {final_idx}")                    
                    final_idx = int(final_idx)

                    chosen_label = reverse_map.get(final_idx, f"unknown_{final_idx}")

                    try:
                        confidence = float(final_probs[0][final_idx])   
                    except:
                        confidence = float(final_probs[0][len(reverse_map)-1]) if isinstance(final_probs[0], (float, int)) else 0.0             
                
            if confidence > 0.8:
                confidence = (confidence + performance_score) / 2 
                print(f'[⚡] Confidence calibrated to be around: {confidence}')

            if isinstance(chosen_label, str) and chosen_label.startswith("unknown") or float(confidence) < self.pipeline.confidence_threshold:
                if chosen_label.startswith("unknown"):
                    chosen_label = 'Unknown'
                    confidence = 1.0 - confidence  #Invert confidence for unknown class

                print(f"\n[⚠️] Final prediction is {chosen_label} with uncertain confidence: {confidence:.1%}. Consider more consistent data for the model to learn from.")
            else:
                print(f"\n[🎯] Final chosen label for input: {chosen_label} || Confidence: {confidence:.1%}")  

            try:
                consecutive_probs = self.pipeline.distribution._handle_peer_agent_request(target_probs, attn_weights, input_ids, type='DevicePeer', agreement=agreement)
                sec_final_idx = consecutive_probs[0].argmax()

                if sec_final_idx > len(reverse_map):
                    sec_final_idx = int(np.argmax(consecutive_probs[:len(reverse_map)-1]))
                    print(f"[⚠️] Clamping {sec_final_idx} → {sec_final_idx}")                    
                sec_final_idx = int(sec_final_idx)

                sec_chosen_label = reverse_map.get(sec_final_idx, f"unknown_{sec_final_idx}")
                try:
                    sec_confidence = float(consecutive_probs[0][sec_final_idx])   
                except:
                    sec_confidence = float(consecutive_probs[0][len(reverse_map)-1]) if isinstance(consecutive_probs[0], (float, int)) else self.pipeline.confidence_threshold  

                if isinstance(confidence, (np.ndarray, list)):
                    confidence = np.mean(confidence)
                if isinstance(sec_confidence, (np.ndarray, list)):
                    sec_confidence = np.mean(sec_confidence) 
                    
                if sec_confidence > 0.8:
                    sec_confidence = (sec_confidence + performance_score) + eps / 2 

                if 'sec_predicted' in results and results['sec_predicted'] is None and sec_chosen_label:
                    print('========== Second Prediction Initiative ==========')
                    print(f'[⚡] My Second Prediction: {sec_chosen_label}') 
                    print(f'[⚡] Confidence: {sec_confidence:.1%}')  

                    results[0]['sec_predicted'] = sec_chosen_label
                    results[0]['sec_confidence'] = sec_confidence
                    results[0]['sec_index'] = sec_final_idx

                    if confidence > results[0]['confidence']:
                        results[0]['predicted'] = chosen_label
                        results[0]['confidence'] = confidence
                        results[0]['predicted_idx'] = final_idx
                else:
                    print('[!] No prediction in results cache are found!')

            except Exception as e:
                print(f'[!] Error initiating second prediction in Advanced prediction method: {e} ')

                results[0]['sec_predicted'] = chosen_label
                results[0]['sec_confidence'] = confidence
                results[0]['sec_index'] = final_idx


        except Exception as e:
            print(f"[!] Error in advanced prediction method: {e}, Initiating regular prediction method...")
            traceback.print_exc()            
            try:
                results = self.regular_prediction_method(titles=titles, label_map=label_map, rules=rules, X=X, y=y, show_proba=False, top_k=3, batch_size=2, use_transformer=self.pipeline.use_transformer)
                chosen_label = results[0]['predicted']
                confidence = results[0]['confidence']
            except Exception as error:
                print(f'[= ! =] Error in all prediction method: {error}')
                traceback.print_exc()
                results, chosen_label, confidence = None, None, 0.0

        print('[=] Displaying Results....')
        payload = {
            'X_samples': X,
            'input_ids': input_ids
        }
        if titles is not None and len(titles) > 0:
            correct, sec_correct = self.display_hybrid_results(payload, final_class_idx, results, top_k, verbose=True)

        if sec_chosen_label and sec_correct > correct:
            print(f'[⚡] Second Prediction: {sec_chosen_label} has higher accuracies, relying on: {sec_chosen_label} as final label.')
            chosen_label = sec_chosen_label # overrides previous chosen label if accuracy is higher
        elif self.pipeline.autonomous and sec_confidence > self.pipeline.confidence_threshold:
            print(f'[⚡] Autonomous Prediction used second predicted label: {sec_chosen_label}')
            chosen_label = sec_chosen_label
        else:
            print(f'[⚡] Final Prediction: {chosen_label} with confidence: {confidence:.1%}')
            chosen_label = chosen_label

        self.pipeline.evaluate_mlp_performance(X, y, self.label_map)
        # delete pipelines cache
        print('[🔍] Pipelines Cache Cleaned!')
        self.pipeline.cache.clear()

        return results, chosen_label, confidence


    def calibration_penalized_check(self, final_probs, predicted_index):
        """
        Reputation-based probability calibration: down-weight classes this
        manager has recently gotten wrong a lot, using the running
        `error_counts`/`pred_counts` counters (see class docstring).

        Steps:
        1. Decays both counters by `pipeline.error_decay` (older
           observations matter less over time).
        2. Increments `pred_counts[predicted_index]` for the class just
           predicted (also self-healing: if the counters' shape no longer
           matches `len(label_map)` — e.g. after a label map change — they
           are reset to zeros of the correct size first).
        3. For each class `c`, computes an error rate
           `error_counts[c] / pred_counts[c]` and converts it into a
           activations.sigmoid-shaped "reputation" multiplier
           `1 / (1 + error_rate)` — 0% error leaves probability unchanged,
           50% error roughly multiplies by 0.67, 100% error roughly
           multiplies by 0.5 (never drives probability all the way to
           zero, since a class can recover its reputation over time).
        4. Multiplies each class's entry in `final_probs` by its
           reputation factor, then renormalizes so probabilities still sum
           to 1.
        5. Re-adapts `pred_counts`/`error_counts` shape to match
           `final_probs` if they've drifted out of sync (e.g. a differently
           sized batch/label set).

        Any exception during calibration is caught and logged; in that
        case `final_probs` is returned unmodified (calibration is treated
        as a best-effort enhancement, not something that should ever break
        prediction).

        Args:
            final_probs (np.ndarray | None): Probability vector to
                calibrate in place. If None, returned as-is with a warning.
            predicted_index (int): Class index that was predicted for this
                sample, used to update `pred_counts`.

        Returns:
            The calibrated (renormalized) probability array, or the
            original `final_probs` if it was None or calibration failed.
        """
        # update class reputation.

        decay = self.pipeline.error_decay
        self.error_counts *= decay
        self.pred_counts  *= decay

        if final_probs is None:
            print('[!] Warning final probabilities is None! returning the None probabilities...')
            return final_probs

        try:

            self.pred_counts[predicted_index] += 1.0
            n_classes = len(self.label_map)

            self.pred_counts = self.pred_counts[0] if isinstance(self.pred_counts[0], np.ndarray) and self.pred_counts.ndim > 1 else self.pred_counts

            if len(self.pred_counts) != n_classes:
                self.pred_counts = np.zeros(n_classes, dtype=np.float64)
                self.pred_counts[predicted_index] += 1.0
                self.pred_counts = self.pred_counts[0] if isinstance(self.pred_counts[0], np.ndarray) and self.pred_counts.ndim > 1 else self.pred_counts

            for c in range(n_classes):
                if len(self.pred_counts) < c:
                    if isinstance(self.pred_counts[c], (int, float)) and self.pred_counts[c] > 0:
                        error_rate    = self.error_counts[c] / (self.pred_counts[c] + 1e-8)
                        # activations.sigmoid-shaped dampening — never goes negative
                        # error_rate=0.0 → multiplier=1.0 (no change)
                        # error_rate=0.5 → multiplier≈0.67
                        # error_rate=1.0 → multiplier≈0.5
                        reputation    = 1.0 / (1.0 + error_rate)
                        if isinstance(error_rate, (list, np.ndarray)):
                            if len(error_rate) != len(final_probs):
                                reputation = 1.0 / (1.0 + np.mean(error_rate))

                        if c < len(final_probs):
                            final_probs[c]  *= reputation  
         
            prob_sum = final_probs.sum()
            if prob_sum > 1e-8:
                final_probs /= prob_sum
                
            # re adapt shape of pred_counts and error_counts if they don't match prob shape
            if self.pred_counts.shape != final_probs.shape:
                self.pred_counts = np.zeros_like(final_probs)
                self.pred_counts *= decay
            if self.error_counts.shape != final_probs.shape:
                self.error_counts = np.zeros_like(final_probs)
                self.error_counts *= decay
                

        except Exception as e:
            print(f'[!] Cant check and calibrate probs based on penalty due to: {e}') 

        return final_probs 

        
    def display_hybrid_results(self, payload, predicted_index, results, top_k=3, verbose=False):
        """
        Pretty-print a batch of prediction results and, as a side effect,
        score them against any expected labels and feed correct/incorrect
        outcomes back into the pipeline's learning state.

        For each entry in `results` that carries an `'expected'` label:
        - Compares the primary `predicted` (and, if present, `sec_predicted`
          "second opinion") label against `expected`, printing a ✅/❌
          status line for each.
        - On a correct primary prediction, calls
          `pipeline.accurate_cache_lookup.add_verified(...)` to record the
          sample as a verified example (so future similar inputs can be
          served straight from cache — see the cache short-circuit in
          `advanced_prediction_method`), and increments the running
          `correct` counter.
        - On a correct second-opinion prediction, does the same for the
          second-opinion cache entry and increments `sec_correct`.
        - On any incorrect prediction, increments `self.error_counts` for
          the predicted class index, feeding the reputation-based
          calibration used by `calibration_penalized_check`.

        Also prints, for every result regardless of whether `expected` is
        known: the final chosen label + confidence (with a ✓/⚠️ agreement
        indicator), the mlp.MLP's own prediction/confidence, the transformer.Transformer's
        prediction/confidence (with an up/down arrow showing which model
        was more confident) when available, and a top-k breakdown when
        `'top_predictions'` is present in the result.

        Args:
            payload (Dict): `{'X_samples', 'input_ids'}` — the feature/
                token representations for this batch, forwarded to
                `add_verified` calls so the cache can match future similar
                inputs.
            predicted_index: Index of the primary prediction (accepted for
                the method's signature; per-result indices are read
                individually from each `results[i]['predicted_idx']`).
            results (List[Dict]): Per-sample result dicts as produced by
                `advanced_prediction_method` (title, predicted label,
                confidence, mlp.MLP/transformer.Transformer breakdown, optional
                `expected`/`sec_predicted`/`top_predictions`, etc).
            top_k (int): Number of top predictions to display per sample
                when `'top_predictions'` is present.
            verbose (bool): Reserved for controlling print verbosity by
                callers (accepted for API compatibility with call sites).

        Returns:
            Tuple[int, int]: `(correct, sec_correct)` — counts of samples
            whose primary / second-opinion prediction matched the expected
            label, respectively.
        """
        print("\n" + "="*80)
        print("[🎯] == PREDICTION RESULTS == ")
        print("="*80)

        correct = 0
        sec_correct = 0
        error = 0
        total_with_expected = 0
        X_samples, input_ids = payload['X_samples'], payload['input_ids']   


        for idx, result in enumerate(results):
            print(f"\n{idx+1}. 📌 '{result['title']}'")
            
            if result.get('expected'):
                total_with_expected += 1
                status = ": ✅" if result['predicted'] == result['expected'] else ": ❌"
                print(f"[=] First Expectation: {result['expected']} || Model Answer: {status}")

                if 'sec_predicted' in result:
                    sec_status = ": ✅" if result['sec_predicted'] == result['expected'] else ": ❌"                
                    print(f"[=] Second Expectation: {result['expected']} || Model Answer: {sec_status}")    
                    if 'sec_predicted' in result and 'sec_confidence' in result and result['sec_predicted'] == result['expected']:
                        self.pipeline.accurate_cache_lookup.add_verified(
                            X_samples, input_ids, 
                            result['sec_predicted'], result['sec_confidence'], result['sec_index'],
                            source='automatic_verified')

                        sec_correct += 1
                    else:
                        error += 1
                        if isinstance(self.error_counts[result.get('predicted_idx')], (int, float)):
                            self.error_counts[result.get('predicted_idx')] += 1.0

                if result['predicted'] == result['expected']:
                    if 'sec_index' in result and result['sec_index'] is None:
                        result['sec_index'] = None

                    if 'predicted_idx' in result and 'confidence' in result and 'predicted' in result:
                        self.pipeline.accurate_cache_lookup.add_verified(
                            X_samples, input_ids, 
                            result['predicted'], result['confidence'], result['predicted_idx'],
                            source='automatic_verified')      

                        correct += 1
                        
                else:
                    error += 1
                    if isinstance(self.error_counts[result.get('predicted_idx')], (int, float)):
                        self.error_counts[result.get('predicted_idx')] += 1.0
    
            # Agreement indicator
            agree_symbol = "✓" if result.get('models_agree', True) else "⚠️"
            print(f"[=] {agree_symbol} FINAL: {result['predicted']} ({result['confidence']:.1%})")

            # mlp.MLP vs transformer.Transformer
            print(f"      ├─ [⚡] mlp.MLP: {result['mlp_prediction']} ({result['mlp_confidence']:.1%})")
            if result.get('transformer_prediction'):
                arrow = "⬆️" if result['transformer_confidence'] > result['mlp_confidence'] else "⬇️"
                print(f"      └─ [🌀] transformer.Transformer: {result['transformer_prediction']} ({result['transformer_confidence']:.1%}) {arrow}")
            
            # Top predictions
            if 'top_predictions' in result:
                print(f"\n [🔍] Top {top_k} possibilities:")
                for j, pred in enumerate(result['top_predictions'][:top_k], 1):
                    bar = '█' * int(pred['confidence'] * 20)
                    print(f"         {j}. {pred['label']:20s} {bar} {pred['confidence']:.1%}")
        
        if total_with_expected > 0:
            accuracy = correct / total_with_expected
            print(f"\n📊 Accuracy: {correct}/{total_with_expected} = {accuracy:.1%}")

        return correct, sec_correct



