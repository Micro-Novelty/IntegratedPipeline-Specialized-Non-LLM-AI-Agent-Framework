"""
pipeline.py
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
from . import automation
from . import caching_security
from . import distributed_inference
from . import ensemble
from . import layers
from . import lstm
from . import mlp
from . import model_storage
from . import query_node
from . import transformer

class IntegratedPipeline:
    """
    Top-level orchestrator that wires together every model and subsystem in
    this framework into one usable prediction pipeline.

    `IntegratedPipeline` is effectively the "root object": it owns the
    transformer.Transformer (`model2`), the mlp.MLP (`model3`/`mlp`), the LSTM
    (`network_model`/`scrapper_model`/`lstm_engine`), the ensemble fuser
    (`ensemble`, a `ensemble.WeightedEnsemblePredictor`), the P2P/security layer
    (`distribution`, an `distributed_inference.AgentDistributedInference`), persistence
    (`storage`, a `model_storage.ModelStorage`), similarity-based answer caching
  (`accurate_cache_lookup`), cross-session import/export
    (`session_automation`), and batching (`batcher`). Most other classes in
    this file receive `self` (the pipeline) as a constructor argument and
    reach back into it for shared config and sibling components — so this
    class is the hub the rest of the framework is built around.

    Broadly, its methods fall into a few groups:
      - **Setup / encoding**: text -> token/feature encoding
        (`text_encoder`, `encode`, `input_encoding`, `sequence_encoding`,
        `transformer_input_encoding`), vocabulary and model initialization
        (`initialize_fitting`, `initialize_model_encoding`,
        `initialize_model_`, `automatic_parameterization`,
        `automatic_dense_layer`).
      - **Memory-backed shortcuts**: before running the (relatively
        expensive) models, several "gate" methods check whether a
        similar-enough input has already been seen and can reuse a cached
        result (`model_memory_gate`, `model_probability_gate`,
        `_gate_from_list`, `_batch_model_memory_gate`), backed by
        `modular_prediction_saving`/`modular_probability_saving` and
        cosine-similarity helpers (`cosine_similarity`,
        `cosine_robust_similarity`).
      - **Prediction**: the actual inference entry points, both single-item
        and batched, both synchronous and async (`predict_single`,
        `predict_async`, `prediction_batch`, `_batch_prediction_core`,
        `_process_batch_chunk`, `hybrid_prediction`,
        `_batch_hybrid_prediction`, `predict_proba`,
        `_batch_predict_proba`, `mlp_predict`), including peer-assisted
        calibration when local confidence is low (`_calibrate_probs`,
        `_handle_distributed_connections`).
      - **Training**: driving mlp.MLP/transformer.Transformer/LSTM training end-to-end
        (`train`, `transformer_utilities`, `_should_train_transformer`,
        `mlp_training_features`, `k_fold_cross_validate`,
        `evaluate_mlp_performance`, LSTM-specific sample prep
        `_set_lstm_samples`/`_extract_cache_samples_for_lstm`/
        `lstm_setup_inference`).
      - **Shape/array plumbing**: a large family of small numeric-safety
        helpers (`_safe_to_2d`, `_pad_ragged_to_array`,
        `_coerce_to_2d_float`, `_safe_to_2d_float`, `_normalize_to_2d`,
        `_convert_to_2d_float`, `_safe_convert`, `shape_adaptation`) that
        exist because inputs to this pipeline can arrive in many shapes
        (ragged lists, 1D/2D/3D arrays, scalars) and every model expects a
        consistent 2D float array.
      - **Memory validation/repair**: guarding against corrupted or
        malformed persisted state (`is_memory_corrupted`,
        `_validate_tuple_memory`, `_validate_list_memory`,
        `_validate_dict_memory`, `_validate_properties`,
        `_sanitize_for_storage`).
      - **Networking convenience**: thin pass-throughs to `distribution`
        for connecting to peers and checking network status
        (`connect_peers`, `get_network_status`).

    This class is implemented as a (lock-guarded) singleton — see the
    `_integrated_pipeline_lock` / `_singleton_initialized` guard in
    `__init__` — so repeated instantiation with the same process reuses the
    already-initialized pipeline instead of rebuilding all sub-components.

    Attributes (selected):
        model2 / model3: The transformer.Transformer and mlp.MLP instances, set once
            trained/initialized (start as `None`).
        network_model / scrapper_model / lstm_engine: LSTM network, its
            training/inference engine wrapper, and the calibrated engine
            actually used at inference time (starts `None` until
            calibrated).
        ensemble (ensemble.WeightedEnsemblePredictor): Fuses model2/model3/lstm
            outputs into a final prediction.
        distribution (distributed_inference.AgentDistributedInference): P2P/security layer for
            this pipeline instance.
        storage (model_storage.ModelStorage): Persistence for all memory namespaces.
        confidence_threshold (float): Global threshold used across the
            pipeline/ensemble to decide when a prediction is "confident
            enough" versus needing peer assistance or fallback handling.
        memory (dict): This pipeline's own persisted memory (distinct from
            the ensemble's attention memory), validated for corruption on
            load via `is_memory_corrupted`.
    """

    def __init__(self, memory_name='agent_memory', 
                  use_async=False, agent_port=None, 
                  ssl_cert_file=None, ssl_key_file=None, 
                  ssl_context=None, client_ssl_context=None,
                  secret_key=None, 
                  shared_auth_token=None, predict_manager=None,
                  bind_host=None, security_level=None):
        """
        Build (or, if already built for this process, reuse) the full
        pipeline: model placeholders, training hyperparameters, the P2P
        (`distribution`), ensemble, storage, caching, and batching
        subsystems, then loads and validates any previously persisted
        pipeline memory for `memory_name`.

        Args:
            memory_name: Namespace used across storage/ensemble/query-node
                for this pipeline's persisted state.
            use_async: If True, sets up the async message queue
                (`_setup_async_queue`) for the P2P layer.
            agent_port: TCP port for the P2P listener; falls back to the
                `AGENT_PORT` env var, then 5555.
            ssl_cert_file, ssl_key_file, ssl_context, client_ssl_context:
                Forwarded to `distributed_inference.AgentDistributedInference` for TLS setup.
            secret_key, shared_auth_token: Forwarded for P2P message
                signing/authentication.
            predict_manager: Optional external predictor forwarded to the
                P2P layer for answering peer predict requests.
            bind_host, security_level: Forwarded to `distributed_inference.AgentDistributedInference`
                to control listen interface and default access policy.

        Side effects:
            - Registers this instance as the process singleton
              (`_singleton_initialized`); a second call with the same
              memory_name inside the same process short-circuits and
              returns without re-initializing.
            - Loads persisted pipeline memory from `storage` and, if
              corruption is detected (`is_memory_corrupted`), attempts an
              automatic reset/fix depending on the failure reason.
            - Registers a synthetic `'local'` entry in
              `distribution.remote_agents` so the pipeline can treat "ask
              myself" the same way as "ask a peer" where convenient.
        """
        # Only initialized once and when allowed

        print('[= MEMORY =] Initializing IntegratedPipeline with memory name:', memory_name)         
        with _integrated_pipeline_lock:
            super().__init__()

            if hasattr(self, '_singleton_initialized'):
                print(f"[===] IntegratedPipeline already initialized, reusing...")
                return
            
            self._singleton_initialized = True
            self._init_params = {
                'memory_name': memory_name,
                'port': agent_port,
                'secret_key': secret_key,
                'ssl_cert_file': ssl_cert_file,
                'ssl_key_file': ssl_key_file,
                'shared_auth_token': shared_auth_token
        
            }  

        self.ssl_cert_file = ssl_cert_file
        self.ssl_key_file = ssl_key_file
        self.secret_key = secret_key
        self._use_async = use_async
        self.port = agent_port if agent_port else int(os.environ.get('AGENT_PORT', 5555))
        self.shared_auth_token = shared_auth_token
        self.manager = None

        self.memory_name = memory_name

        self.client_ssl_context = None
        self.ssl_context = None

        self.input_size = 1
        self.hidden = 32
        self.output_size = 1
        self.dropout_rate = 0.1
        self.max_size = 500
        self.error_decay = 0.85
        self.performance_result = 1.0
        self.max_ram_allowed = 150

        # === TRAINING SETUP ===
        self.mlp_training_epochs = 1000
        self.mlp_lr = 0.1
        self.transformer_lr = 0.1
        self.transformer_d_model = 32
        self.transformer_heads = 4
        self.transformer_training_epochs = 100
        self.lstm_training_epochs = 100
        self.lstm_lr = 5e-2
        self.lstm_hidden_dim = 64

        self.unsuitable_tolerance = False
        self.unsuitable_conditions = False
        self.unsuitable_peer_request  = False

        # Main component setup
        self.standard_scaler = StandardScaler()
        self.tfidf = TfidfVectorizer(max_features=70)
        # LSTM __init__ setup
        self.network_model = lstm.LSTMNetwork(self, input_size=self.input_size, hidden_size=self.hidden, output_size=self.output_size)
        self.scrapper_model = lstm.LSTMEngine(self, self.network_model, dropout=self.dropout_rate, n_samples=50)        
        self.lstm_engine = None
        self.lstm_n_samples = 0

        self.storage = model_storage.ModelStorage(self, memory_name, db_path='activity_log.db')
        self.distribution = distributed_inference.AgentDistributedInference(self, self.storage, memory_name, port=self.port, 
                                                         use_async=use_async, secret_key=self.secret_key, 
                                                         ssl_cert_file=ssl_cert_file, ssl_key_file=ssl_key_file, 
                                                         ssl_context=self.ssl_context, client_ssl_context=self.client_ssl_context,
                                                         shared_auth_token=self.shared_auth_token, predict_manager=self.manager,
                                                         bind_host=bind_host, security_level=security_level)        
        self.ensemble = ensemble.WeightedEnsemblePredictor(self, self.distribution, memory_name)        
        self.session_automation = automation.CrossSessionAutomation(self)
        self.batcher = automation.AutoBatcherAutomation(self)
        self.query_node = query_node.QueryNode(self, memory_name, self.storage)
        self.accurate_cache_lookup = caching_security.AccurateAnswerCache(self, similarity_threshold=0.85, max_size=self.max_size)

        self._agent_mode = os.environ.get('AGENT_MODE', 'single')
        self._agent_port = int(os.environ.get('AGENT_PORT', 5555))
        self._use_async = os.environ.get('USE_ASYNC_QUEUE', 'true').lower() == 'true'

        # Special token indices — reserve before any real words
        self._PAD_IDX = 0
        self._UNK_IDX = 1
        self._SPECIAL_TOKENS = {'[PAD]': self._PAD_IDX, '[UNK]': self._UNK_IDX}       

        print(f'[= PORT =] IntegratedPipeline initialized on port {self._agent_port}')
         
        # Queue for managing async operations
        self._async_tasks = set()
        self._loop = None  

        self.mlp = mlp.MLP()
        self.focused_mlp = mlp.MLP()

        self.X = None
        self.vocab_size = None
        self.model2 = None
        self.model3 = None
        self.texts = None
        self.intents = None
        self.role_bot = None
        self.batch_timer = None
        self.reverse_map = None
        self.rules = None
        self.titles = None
        self.labels = None

        self.use_transformer = True
        self.agreement = False
        self.external_peer_enabled = False
        self.autonomous = False 
        self.show_explainability_details = True    
        self.freeze_learning = False 
        self._cache_save_count = None
        self._prob_save_count = None

        self.temperature = 1.0
        self.max_seq_len = 16

        self.memory_name = memory_name

        self.pending_batch = []
        self.temporary_id = []

        self.final_conf_score = 0.0
        self.timeout = 120
        self.confidence_threshold = 0.45  
        self.peer_assistance_threshold = 0.0              
        self.agent_id = random.randint(0, 10000)

        self.vocab = {}
        self.cache = {}

        if not self.storage.memory_exists(memory_name, type='Pipeline'):
            self.memory = {}
        else:
            print(f'|| Found Matched Memory: {memory_name}!')
            self.memory = self.storage.memory_retrieval(memory_name, type_func='Pipeline', verbose=True)

            is_corrupted, reason, suggested_fix = self.is_memory_corrupted(self.memory)
            if is_corrupted:
                print(f'[⚠️] MEMORY CORRUPTION DETECTED!')
                print(f'    Reason: {reason}')
                print(f'    Suggestion: {suggested_fix}')
                
                # Auto-fix based on severity
                if 'deserialization' in reason.lower():
                    print(f'[!] Deserialization error - resetting memory')
                    self.storage.fix_corrupted_memory(memory_name)
                    self.memory = {}
                elif 'unexpected' in reason.lower():
                    print(f'[!] Unexpected shape in memory - resetting memory')
                    self.storage.fix_corrupted_memory(memory_name)
                    self.memory = {}
                else:
                    print(f'[!] Keeping memory but will validate on access')
                
            else:
                print(f'[✅] Memory validation passed: {reason}')              

        if use_async:
            self._setup_async_queue()
            
        self.distribution.remote_agents['local'] = {
            'sock': None,  # No socket needed for local
            'host': 'localhost',
            'port': self.port,
            'trust': 1.0,
            'last_seen': datetime.now(),
            'failures': 0
        }

    def _validate_properties(self, memory):
        """
        Normalize a raw "memory" value (persisted model state / cached
        feature vector) into a numeric array-like the pipeline can use,
        or None if it can't be salvaged.

        Handles a few messy real-world storage formats: a bare scalar is
        rejected (can't represent a class vector), a stringified array
        (e.g. `"[1.0 2.0 3.0]"`) is parsed back into floats via
        `np.fromstring`, and a numpy array of string/char dtype is
        similarly flattened and re-parsed. A plain Python list is checked
        against `_get_num_classes()` and rejected if its length doesn't
        match the current class count (stale memory from a different
        label set).

        Args:
            memory: Raw memory value as loaded from storage — may be a
                scalar, string, string-dtype ndarray, list, or already a
                usable numeric array.

        Returns:
            The cleaned-up memory value, or None if it was a bare scalar
            or a list whose length doesn't match the current number of
            classes.
        """
        if isinstance(memory, (float, int)):
            print('[!] Memory is single scalar! returning 0.0 similarity')
            return None

        if isinstance(memory, (str, np.str_)):
            clean_str = str(memory).replace('[', '').replace(']', '')
            memory = np.fromstring(clean_str, sep=' ')
        if isinstance(memory, np.ndarray) and np.issubdtype(memory.dtype, np.character):
            # catches arrays filled with string text
            clean_str = ' '.join(memory.astype(str).flatten()).replace('[', '').replace(']', '')
            memory = np.fromiter(
                    (v for v in clean_str.split() if memory != "..."), dtype=float
                )   

        if isinstance(memory, list) and self.model3 is not None:
            if len(memory) == self._get_num_classes():
                return memory
            else:
                print('[!] Memory length is not tied with current number of classes! skipping Memory.')
                return None

        return memory

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

    
    def is_memory_corrupted(self, memory, num_classes: int = None) -> tuple:
        """
        Robust memory corruption detection here.

        Returns:
            (is_corrupted: bool, reason: str, suggested_fix: str)
        """
        if num_classes is None:
            num_classes = self._get_num_classes() or 0

        # ___ Case 0: Checks for string based _______
        memory = self._validate_properties(memory)
        if memory is None:
            return None

        # ── Case 1: None ──────────────────────────────────────────────
        if memory is None:
            return (True, "Memory is None", "Initialize new memory dict")

        # ── Case 2: numpy array ───────────────────────────────────────
        if isinstance(memory, np.ndarray):
            if memory.ndim == 1 and memory.shape[0] == num_classes:
                return (False, "Valid probability array", None)
            if memory.ndim == 2 and memory.shape[1] == num_classes:
                return (False, "Valid probability matrix", None)
            if memory.shape[0] == 0:
                return (True, "Empty array", "Clear and reinitialize")
            return (True,
                    f"Unexpected array shape: {memory.shape} for {num_classes} classes",
                    "Clear memory and retrain model")

        # ── Case 3: tuple ─────────────────────────────────────────────
        # tuples are valid memory entries — (input, output) pairs
        # only corrupt if: single element, or contains non-serializable types
        if isinstance(memory, tuple):
            return self._validate_tuple_memory(memory, num_classes)

        # ── Case 4: list ──────────────────────────────────────────────
        if isinstance(memory, list):
            return self._validate_list_memory(memory, num_classes)


        # ── Case 5: dict ──────────────────────────────────────────────
        if isinstance(memory, dict):
            return self._validate_dict_memory(memory, num_classes)

        # ── Case 6: scalar numbers — sometimes stored as cached probs ──
        if isinstance(memory, (int, float, np.integer, np.floating)):
            if 0.0 <= float(memory) <= 1.0:
                return (False, "Valid scalar probability", None)
            return (True,
                    f"Scalar value {memory} out of [0,1] range",
                    "Clear and reinitialize")

        return (True,
                f"Unexpected memory type: {type(memory).__name__}",
                "Clear and reinitialize memory")


    def k_fold_cross_validate(self, X, y, input_dim, n_classes, k=5, seed=42,
                            epochs=None, lr=None, log_to_diagnostics=True):
        """
        Orchestrates init → train → predict across k folds. Each fold gets
        a FRESH model (no weight carryover between folds), and the model
        that existed before k-fold started is restored afterward — k-fold
        here is purely an evaluation procedure, not a permanent side effect.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        # preserve pre-existing model, restore it once CV is done
        original_model = getattr(self, 'model3', None)

        recorder = None
        if log_to_diagnostics:
            try:
                from abstract_diagnostics import get_recorder
                recorder = get_recorder(self.memory_name)
            except ImportError:
                pass   # diagnostics package not installed — silently skip, no crash

        aggregate_cm    = np.zeros((n_classes, n_classes), dtype=int)
        fold_accuracies = []

        for fold_idx, (X_tr, y_tr, X_val, y_val) in enumerate(
            self.model3.k_fold_split(X, y, k=k, seed=seed)
        ):
            # FRESH model every fold — this is the load-bearing line
            self.initialize_model_(X_tr, input_dim, n_classes)
            self.model3.train(X_tr, y_tr,
                            epochs=epochs or self.mlp_training_epochs,
                            lr=lr or self.mlp_lr)

            y_pred = self.model3.forward(X_val)
            cm = self.model3.confusion_matrix(y_val, y_pred, n_classes)
            aggregate_cm += cm

            fold_acc = np.trace(cm) / max(cm.sum(), 1)
            fold_accuracies.append(fold_acc)
            print(f'[=] Fold {fold_idx + 1}/{k}: accuracy={fold_acc:.2%}')

            if recorder:
                recorder.log_scalar('mlp/kfold_accuracy', fold_acc, step=fold_idx)

        self.model3 = original_model   # restore — CV was evaluation only

        mean_acc, std_acc = float(np.mean(fold_accuracies)), float(np.std(fold_accuracies))
        print(f'[=] K-Fold CV complete: {mean_acc:.2%} ± {std_acc:.2%}')

        if recorder:
            recorder.log_scalar('mlp/kfold_mean_accuracy', mean_acc)
            recorder.log_scalar('mlp/kfold_std_accuracy', std_acc)

        return {
            'fold_accuracies': fold_accuracies,
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'confusion_matrix': aggregate_cm,
        }

    def evaluate_mlp_performance(self, X, y, label_map, k=5, seed=42):
        """
        input_dim and n_classes are ALWAYS derived from the actual X/y
        given to this call — never accepted as separate parameters that
        could silently drift out of sync with the real data (the exact
        bug class this whole session was about).
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # derived input_dim directly from X, matching this call's
        # ACTUAL data.
        input_dim = X.shape[1]
        model_classes = self._get_num_classes()

        #derived n_classes the same authoritative way used
        # throughout the rest of the pipeline (_get_num_classes as
        # primary source, cross-checked against y itself)

        y_arr = np.asarray(y)

        onehot_validation = self._validate_onehot(y)
        if onehot_validation:
            if model_classes != len(label_map):
                model_classes = len(label_map)
            if model_classes > np.max(y):
                y = np.eye(model_classes)[np.asarray(y)]
            else:
                print('[⚠️] Warning: Proper Y onehot encoding fails, modifying number of classes to exactly match Y values to do one final one hot encoding...')
                try:
                    model_classes = np.max(y) + 1
                    y = np.eye(model_classes)[np.asarray(y)]
                except:
                    print('[!] Error: One hot encoding failed, please check your Y values and label map for consistency, passing raw Y and skipping mlp.MLP Training...')
                    unsuitable = True
                    y = y.copy()  # fallback to raw y if one-hot fails

        if y_arr.ndim > 1:
            n_classes_from_y = y_arr.shape[1]
        else:
            n_classes_from_y = int(y_arr.max()) + 1

        if model_classes is not None and model_classes != n_classes_from_y:
            print(f'[⚠️] evaluate_mlp_performance: n_classes mismatch — '
                f'model reports {model_classes}, y data implies '
                f'{n_classes_from_y}. Using y-derived value ({n_classes_from_y}) '
                f'since k-fold must match the ACTUAL labels being evaluated.')
        n_classes = n_classes_from_y

        print(f'[=] evaluate_mlp_performance: derived input_dim={input_dim}, '
            f'n_classes={n_classes} from provided X/y (shapes: '
            f'X={X.shape}, y={y_arr.shape})')

        result = self.k_fold_cross_validate(X, y, input_dim, n_classes, k=k, seed=seed)

        print('========= mlp.MLP Performance Evaluation Summary ============')
        print(f'[=>] mlp.MLP performance evaluation complete: mean accuracy={result["mean_accuracy"]:.2%}, '
              f'std accuracy={result["std_accuracy"]:.2%}')
        print(f'[=>] Confusion matrix:\n{result["confusion_matrix"]}')
        print(f'[=>] Fold accuracies: {result["fold_accuracies"]}')
        print(f'[=>] K-Fold CV complete: {result["mean_accuracy"]:.2%} ± {result["std_accuracy"]:.2%}')
        return result


    def _validate_tuple_memory(self, memory: tuple, num_classes: int) -> tuple:
        """
        Validate tuple memory entry.
        Valid: (input, output), (input, label), (features, probs)
        Invalid: single-element, or contains obviously corrupt values
        """
        # single element tuple — almost always a mistake since prob is never single
        if len(memory) < 2:
            return (True,
                    f"Single-element tuple — likely wrapping error",
                    "Unwrap or clear entry")

        # check each element for validity here
        for element_idx, element in enumerate(memory[:4]):  # check first 4 elements max first

            # None elements in a tuple are suspicious
            if element is None:
                return (True,
                        f"Tuple contains None at position {element_idx}",
                        "Clear this memory entry")

            # string elements — check for weird symbols that may appearr here
            if isinstance(element, str):
                # allow normal text but flag obviously corrupted strings
                if len(element) == 0:
                    return (True,
                            f"Tuple contains empty string at position {element_idx}",
                            "Clear this memory entry")
                # check for non-printable / control characters hiding
                non_printable = sum(1 for c in element if ord(c) < 32 and c not in '\n\t\r')
                if non_printable > 0:
                    return (True,
                            f"Tuple string at position {element_idx} contains "
                            f"{non_printable} non-printable characters",
                            "Clear this memory entry")
                # suspiciously long string — likely serialization artifact from pickle
                if len(element) > 2000:
                    return (True,
                            f"Tuple string at position {element_idx} is "
                            f"suspiciously long ({len(element)} chars)",
                            "Clear this memory entry")
                

            # numpy array elements — check shape sanity first
            elif isinstance(element, np.ndarray):
                if element.size == 0:
                    return (True,
                            f"Tuple contains empty array at position {element_idx}",
                            "Clear this memory entry")
                if not np.isfinite(element).all():
                    return (True,
                            f"Tuple array at position {element_idx} contains "
                            f"NaN or Inf values",
                            "Clear this memory entry")

            # numeric scalars — sanity check
            elif isinstance(element, (int, float, np.integer, np.floating)):
                if not np.isfinite(float(element)):
                    return (True,
                            f"Tuple contains non-finite scalar at position {element_idx}",
                            "Clear this memory entry")

            # nested tuple/list — valid.
            elif isinstance(element, (tuple, list)):
                if len(element) == 0:
                    return (True,
                            f"Tuple contains empty sequence at position {element_idx}",
                            "Clear this memory entry")

            # other types — flag as suspicious, can be deleted.
            else:
                type_name = type(element).__name__
                if type_name not in ('bool', 'bool_', 'datetime'):
                    return (True,
                            f"Tuple contains unexpected type {type_name} "
                            f"at position {element_idx}",
                            "Need to Clear this memory entry")

        return (False, f"Valid tuple memory entry (len={len(memory)})", None)


    def _validate_list_memory(self, memory: list, num_classes: int) -> tuple:
        """Validate list memory — probability lists, entry lists, feature lists."""
        if not memory:
            return (True, "Empty list", "Initialize new memory dict")

        # probability list — exact class count, all numeric
        if len(memory) == num_classes:
            sample = memory[:min(5, len(memory))]
            if all(isinstance(x, (int, float, np.integer, np.floating)) for x in sample):
                vals = [float(x) for x in sample]
                if all(0.0 <= v <= 1.0 for v in vals):
                    return (False, "Valid probability list", None)

        # list of valid memory entries — tuples or lists of length >= 2
        sample = memory[:min(5, len(memory))]
        if all(isinstance(item, (tuple, list)) and len(item) >= 2 for item in sample):
            # need to validate each tuple entry
            for item in sample:
                corrupted, reason, fix = self._validate_tuple_memory(
                    tuple(item) if isinstance(item, list) else item,
                    num_classes
                )
                if corrupted:
                    return (True,
                            f"List contains corrupted entry: {reason}",
                            fix)
            return (False, "Valid memory entries list", None)

        # numpy arrays in list
        if any(isinstance(item, np.ndarray) for item in sample):
            arrays = [item for item in memory if isinstance(item, np.ndarray)]
            if all(a.shape[0] == num_classes or
                (a.ndim > 1 and a.shape[1] == num_classes) for a in arrays):
                return (False, "Valid list of probability arrays", None)
            return (True,
                    f"List contains arrays with mismatched shapes",
                    "Clear and retrain")

        # hybrid feature list — wrong type for memory storage
        if len(memory) != num_classes:
            return (True,
                    f"Hybrid feature list (length {len(memory)}, "
                    f"expected {num_classes}) stored as memory",
                    "Clear memory and retrain")

        return (True,
                f"Suspicious list contents — type: {type(memory[0]).__name__}",
                "Inspect memory contents")


    def _validate_dict_memory(self, memory: dict, num_classes: int) -> tuple:
        """Validate dict memory — expected primary format."""
        if not memory:
            return (False, "Empty dict (no memory yet)", None)

        valid_keys = {'TW', 'MW', 'TP', 'MP', 'TA', 'local',
                    '_cached_probs', '_data'}

        for key, value in memory.items():
            # numeric keys — deserialization artifact
            if isinstance(key, (int, float)):
                return (True,
                        f"Dict has numeric key: {key}",
                        "Likely deserialization error, clear memory")

            # suspiciously long key
            if len(str(key)) > 100:
                return (True,
                        f"Dict has very long key ({len(str(key))} chars)",
                        "Possible corruption, clear memory")

            # validate array values
            if isinstance(value, np.ndarray):
                if num_classes > 0 and value.ndim > 0:
                    if value.shape[0] != num_classes and \
                    (value.ndim < 2 or value.shape[1] != num_classes):
                        return (True,
                                f"Array shape {value.shape} doesn't match "
                                f"{num_classes} classes in key '{key}'",
                                "Hybrid feature stored incorrectly, clear entry")
                if value.size == 0:
                    return (True,
                            f"Empty array in key '{key}'",
                            "Corrupted array, clear entry")
                if not np.isfinite(value).all():
                    return (True,
                            f"NaN/Inf values in array at key '{key}'",
                            "Corrupted values, clear entry")

            # validate tuple values in dict
            elif isinstance(value, tuple):
                corrupted, reason, fix = self._validate_tuple_memory(
                    value, num_classes
                )
                if corrupted:
                    return (True,
                            f"Corrupted tuple value at key '{key}': {reason}",
                            fix)

        return (False, "Valid dict structure", None)


    def initialize_fitting(self, text):
        """
        Fit the pipeline's TF-IDF vectorizer on a corpus of text and cache
        the resulting vocabulary size.

        Args:
            text: Iterable of raw text documents to fit `self.tfidf` on.

        Side effects:
            Sets `self.vocab_size` to the number of TF-IDF features
            learned, which downstream sizing logic
            (`automatic_parameterization`, `initialize_model_encoding`)
            uses as the mlp.MLP's input dimensionality.
        """
        self.tfidf.fit_transform(text).toarray()
        vocab_size = len(self.tfidf.get_feature_names_out())
        self.vocab_size = vocab_size
        

    def initialize_model_encoding(self, X, y_raw):
        """
        One-hot encode raw integer labels and, on first use, build+attach
        the mlp.MLP's layers sized for the current vocabulary/class count.

        Class-count resolution logic: if the mlp.MLP has no existing output
        dimension (`_get_num_classes()` is None), the class count is
        inferred fresh from how many unique labels appear in this batch
        (legitimate for first-time initialization). If the model already
        exists but this batch contains *more* distinct classes than the
        model currently supports, the model is treated as needing to grow
        to accommodate them. If the batch contains *fewer* classes than
        the model supports (a normal, expected case — not every batch
        will contain every class), the model's existing (larger) class
        count is kept so the one-hot width doesn't shrink and desync
        from the model's actual output layer.

        Out-of-range labels are logged and left as an all-zero one-hot row
        rather than raising or silently mapping to the wrong class.

        Args:
            X: Feature batch, forwarded to layer-sizing helpers
                (`automatic_dense_layer`) — used for shape info, not
                mutated here.
            y_raw (array-like[int]): Raw integer class labels for this
                batch.

        Returns:
            np.ndarray: One-hot encoded labels, shape
            `(len(y_raw), num_classes)`.

        Side effects:
            Sets `self.embedding_dim`; builds new layers.Dense layers sized via
            `automatic_parameterization`/`automatic_dense_layer` and adds
            them to `self.mlp` (both as regular layers and "feed" layers).
        """
        vocab_size = self.vocab_size

        # canonical source first, since model output dimension
        # (once it exists) is the true authority on num_classes
        num_classes = self._get_num_classes()

        unique_in_batch = len(np.unique(y_raw))

        if num_classes is None:
            # no model exists yet — this is the legitimate case where
            # inferring from y_raw is correct (e.g. first-time initialization)
            num_classes = unique_in_batch
            print(f'[=] No existing model — initializing with {num_classes} classes '
                f'from this batch')
        elif unique_in_batch > num_classes:
            # batch contains MORE classes than the model supports
            # this is a real problem — can't onehot-encode into a smaller space
            print(f'[⚠️] Batch contains {unique_in_batch} unique classes but model '
                f'only supports {num_classes} — expanding to {unique_in_batch}')
            num_classes = unique_in_batch
        elif unique_in_batch < num_classes:
            # batch just doesn't happen to contain all classes.
            # use the model's full num_classes so onehot stays the right width
            print(f'[=] Batch only contains {unique_in_batch}/{num_classes} classes '
                f'— using full model class count for onehot width')
            # num_classes stays as the model's true value, no change needed

        y_onehot = np.zeros((len(y_raw), num_classes))

        for idx, label in enumerate(y_raw):
            label_idx = int(label)
            if 0 <= label_idx < num_classes:
                y_onehot[idx, label_idx] = 1.0
            else:
                print(f'[⚠️] Label {label_idx} out of range for {num_classes} classes '
                    f'at sample {idx} — skipping onehot assignment')
                # leaves that row as all-zeros rather than crashing or
                # silently assigning to a wrong class

        automatic_change = self.automatic_parameterization(vocab_size, num_classes)
        self.embedding_dim = automatic_change
        layer1, layer2 = self.automatic_dense_layer(X, vocab_size, num_classes)

        model = self.mlp  

        model.add(layer1)
        model.add(layer2)

        model.feed_add(layer1)
        model.feed_add(layer2)

        return y_onehot



    def initialize_model_(self, X, input_dim, num_classes):
        """
                Build a brand-new `self.model3` (mlp.MLP) sized for `input_dim` ->
                `num_classes`, plus an extra wider "feed" pair of layers used
                elsewhere for feed-forward augmentation. Unlike
                `initialize_model_encoding`, this always creates a fresh model —
                it does not add layers onto an existing `self.mlp`.

                Args:
                    X: Sample batch, only used to size layers.Dense layers.
                    input_dim: Width of the input feature vector.
                    num_classes: Number of output classes.
        """
        automatic_change = self.automatic_parameterization(input_dim, num_classes)

        automatic_change = self.automatic_parameterization(input_dim, num_classes)

        layer1= layers.Dense(X, input_dim, automatic_change, activation="relu")
        layer2 = layers.Dense(X, automatic_change, num_classes, activation='relu')
        
        abundant_layer = int(automatic_change * 10)
        first_feed_layer = layers.Dense(X, input_dim, abundant_layer, activation="relu")
        sec_feed_layer = layers.Dense(X, abundant_layer, num_classes, activation="relu")

        self.model3 = mlp.MLP() 

        self.model3.add(layer1)
        self.model3.add(layer2)

        self.model3.feed_add(first_feed_layer)   
        self.model3.feed_add(sec_feed_layer)                

        
    def automatic_parameterization(self, input_size, num_classes):
        """
                Heuristic for picking a hidden-layer width: half the product of
                input size and class count. Used throughout the pipeline
                wherever a layers.Dense layer needs to be auto-sized.
        """
        parameters = input_size * num_classes / 2
        parameters = int(parameters)
        return parameters


    def automatic_dense_layer(self, X, input_dim, num_classes):
        """
                Build a (layers.Dense, layers.Dense) hidden->output layer pair auto-sized via
                `automatic_parameterization`, for `input_dim` -> `num_classes`.
                Does not attach the layers to any model — caller is responsible
                for calling `.add()`/`.feed_add()`.
        """
        vocab_size = self.vocab_size 

        automatic_change = self.automatic_parameterization(input_dim, num_classes)

        layer1= layers.Dense(X, input_dim, automatic_change, activation="relu")
        layer2 = layers.Dense(X, automatic_change, num_classes, activation="relu")

        return layer1, layer2



    def text_encoder(self, texts):
        """
        Build vocabulary from texts incrementally.
        Reserves 0=[PAD], 1=[UNK].
        """
        vocab = self.vocab

        if not vocab:
            vocab.update(self._SPECIAL_TOKENS)

        # track next free index from actual max value in vocab,
        # not len() which breaks if any entries were ever removed
        idx = max(vocab.values()) + 1 if vocab else len(self._SPECIAL_TOKENS)

        for item in texts:
            text = item[0] if isinstance(item, tuple) else item
            if not isinstance(text, str) or not text.strip():
                continue   # skip None/empty safely

            for word in text.lower().split():
                if word not in vocab:
                    vocab[word] = idx
                    idx += 1


    def encode(self, sentence, vocab, max_len=None):
        """
        Encode sentence to token ids.
        Unknown words → [UNK] (1). Shorter sequences → [PAD] (0) padded.
        """
        # guard against None/empty input
        if not isinstance(sentence, str) or not sentence.strip():
            pad_len = max_len or 6
            return [self._PAD_IDX] * pad_len

        # dynamic max_len from vocab's seen sentence lengths
        # if not explicitly provided, derive from pipeline config or vocab size
        if max_len is None:
            max_len = getattr(self, 'max_seq_len', 16)   # configurable, default 16 not 6

        tokens = sentence.lower().split()
        ids    = [vocab.get(w, self._UNK_IDX) for w in tokens]

        # informative truncation warning with actual token count
        if len(tokens) > max_len:
            print(f'[!] Truncated "{sentence[:40]}..." '
                f'({len(tokens)} tokens → {max_len})')

        # pad or truncate to exact max_len
        ids = ids[:max_len]
        ids.extend([self._PAD_IDX] * (max_len - len(ids)))

        return ids

    def input_encoding(self, datasets):
        """
                Turn a list of `(text, intent)` pairs into `(input_ids_list, y_true)`
                ready for the transformer.Transformer.

                Handles empty/None datasets, drops entries with empty/None text,
                and reconciles this batch's class count against any existing
                `self.model2` / `self.label_map`: if the batch introduces new
                classes the class count grows; if it's a subset of known classes,
                intents are remapped to the stored GLOBAL label indices (via
                `self.label_map`) rather than re-deriving batch-local indices, so
                one-hot columns stay consistent across calls. Also (re)creates
                `self.model2` sized for the resolved `num_classes` and populates
                `self.texts` / `self.intents` / `self.reverse_map` as a side
                effect.

                Returns:
                    (input_ids_list, y_true): token-id arrays per text and the
                    corresponding one-hot label matrix.
        """
        if datasets is None or len(datasets) == 0:
            print('[⚠️] input_encoding: datasets is empty — returning '
                'empty input_ids_list and empty y_true. Caller must '
                'handle this case explicitly rather than passing it '
                'downstream to the transformer.Transformer.')
            self.texts       = []
            self.intents     = []
            self.reverse_map = {}
            return [], np.zeros((0, self._get_num_classes() or 1))

        texts   = [d[0] for d in datasets]
        intents = [d[1] for d in datasets]

        intent_to_id = {intent: i for i, intent in enumerate(sorted(set(intents)))}
        batch_classes = len(intent_to_id)

        # guard against malformed entries silently producing
        # empty texts/intents even when datasets itself isn't empty
        valid_pairs = [(t, i) for t, i in zip(texts, intents)
                    if t is not None and str(t).strip()]
        if len(valid_pairs) < len(texts):
            dropped = len(texts) - len(valid_pairs)
            print(f'[⚠️] input_encoding: dropped {dropped} entries with '
                f'empty/None text')

        if not valid_pairs:
            print('[⚠️] input_encoding: no valid (non-empty) text entries '
                'remain after filtering — returning empty result')
            self.texts       = []
            self.intents     = []
            self.reverse_map = {}
            return [], np.zeros((0, self._get_num_classes() or 1))

        texts, intents = zip(*valid_pairs)
        texts   = list(texts)
        intents = list(intents)

        model_classes = self._get_num_classes()

        if model_classes is None:
            num_classes = batch_classes
            print(f'[=] No existing model — using batch class count: {num_classes}')

        elif batch_classes > model_classes:
            print(f'[⚠️] input_encoding: batch has {batch_classes} classes '
                f'but model only supports {model_classes} — '
                f'expanding num_classes to {batch_classes}')
            num_classes = batch_classes

        elif batch_classes < model_classes:
            # batch is a SUBSET of known classes 
            print(f'[=] input_encoding: batch has {batch_classes}/{model_classes} '
                f'classes — using full model class count')
            num_classes = model_classes

            # remap intent_to_id to GLOBAL indices from the stored label_map
            # so index 2 in this batch actually means class 2 globally,
            if hasattr(self, 'label_map') and self.label_map:
                remapped = {}
                for intent in intent_to_id:
                    if intent in self.label_map:
                        remapped[intent] = self.label_map[intent]
                    else:
                        # genuinely new intent not in global map — append at end
                        remapped[intent] = max(self.label_map.values()) + 1
                        print(f'[⚠️] Unknown intent "{intent}" not in label_map — '
                            f'assigned index {remapped[intent]}')
                intent_to_id = remapped
        else:
            num_classes = model_classes

        labels = [intent_to_id[i] for i in intents]

        # validate all labels are in range 
        max_label = max(labels) if labels else 0
        if max_label >= num_classes:
            print(f'[⚠️] Max label index {max_label} >= num_classes {num_classes} '
                f'— expanding num_classes to {max_label + 1}')
            num_classes = max_label + 1

        reverse_map = {i: intent for intent, i in intent_to_id.items()}

        self.texts      = texts
        self.intents    = intents
        self.reverse_map = reverse_map

        self.model2 = transformer.Transformer(
            vocab_size=len(self.vocab),
            d_model=self.transformer_d_model,
            n_heads=self.transformer_heads,
            num_classes=num_classes
        )

        # safe y_true construction with explicit bounds guard per row
        y_true = np.zeros((len(labels), num_classes))
        for i, l in enumerate(labels):
            if 0 <= l < num_classes:
                y_true[i, l] = 1.0
            else:
                print(f'[⚠️] Label {l} out of range for num_classes={num_classes} '
                    f'at sample {i} (intent="{intents[i]}") — row left as zeros')

        input_ids_list = [
            np.array(self.encode(text, self.vocab))
            for text in texts
        ]

        return input_ids_list, y_true


    def cosine_robust_similarity(self, a, b):
        """
                Cosine similarity that tolerates messy input: stringified
                arrays (e.g. `"[0.1 0.2 ...]"`), object/character-dtype numpy
                arrays, and mismatched shapes. Falls back through several
                best-effort strategies (subset slicing, flatten-and-truncate)
                before giving up and returning a low fixed similarity (0.1)
                rather than raising. Prefers the optimized native
                implementation (`optimized_cosine_similarity`) when available.
                Returns a value clipped to [-1.0, 1.0], or 0.0 for scalar input.
        """
        if isinstance(b, (float, int)) or isinstance(a, (float, int)):
            print('[!] Value is single scalar! returning 0.0 similarity')
            return 0.0

        if isinstance(a, (str, np.str_)):
            clean_str = str(a).replace('[', '').replace(']', '')
            a = np.fromstring(clean_str, sep=' ')         

        if isinstance(a, np.ndarray) and np.issubdtype(a.dtype, np.character):
            # catches arrays filled with string text
            clean_str = ' '.join(a.astype(str).flatten()).replace('[', '').replace(']', '')
            skip_values = {"...", "NaN", "null"}
            a = np.fromiter((v for v in clean_str.split() if v not in skip_values), dtype=float)

        if isinstance(b, (str, np.str_)):
            clean_str = str(b).replace('[', '').replace(']', '')
            b = np.fromstring(clean_str, sep=' ')
        if isinstance(b, np.ndarray) and np.issubdtype(b.dtype, np.character):
            # catches arrays filled with string text
            clean_str = ' '.join(b.astype(str).flatten()).replace('[', '').replace(']', '')
            skip_values = {"...", "NaN", "null"}
            b = np.fromiter((v for v in clean_str.split() if v not in skip_values), dtype=float)
         

        if isinstance(a[0], (np.ndarray, list)):
            norm_a = np.linalg.norm(a[0])
        else:
            norm_a = np.linalg.norm(a) 

        if isinstance(b[0], (np.ndarray, list)):
            norm_b = np.linalg.norm(b[0])
        else:
            norm_b = np.linalg.norm(b)

        if _OPT_AVAILABLE:
            if len(a.shape) > 1:
                a = np.asarray(a)
                a = a.reshape(-1)  
            if len(b.shape) > 1:
                b = np.asarray(b)
                b  = b.reshape(-1)
            return optimized_cosine_similarity(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))     
                  
        try:
            dot_product = np.dot(a, b)
        except:
            try: #detect inhomogenous shape
                dot_product = np.dot(a.flatten(), b[:a.flatten().shape[0]])
            except:
                try:
                    dot_product = np.dot(a[:b.shape[0]], b.flatten()[:a.shape[0]])
                except:
                    print('[-] No similarity due to inhomogenous shapes and failed attempts to find subsets, returning low similarity score.')
                    return 0.1
    
        dot_product = np.dot(a, b)
        cosine = dot_product / (norm_a * norm_b)
    
        return np.clip(cosine, -1.0, 1.0)

    # ===== async setup ======
    def _setup_async_queue(self):
        """
                Register the async-queue message handlers this pipeline needs
                (`custom_prediction`, `model_update`) on `self.distribution`,
                then start its queue processor and health checker. No-op if
                `self.distribution` has no async queue configured
                (`use_async=False`).
        """
        # Setup async queue handlers
        if not self.distribution or not self.distribution.use_async:
            return
            
        # Register custom handlers
        self.distribution.message_queue.register_handler(
            'custom_prediction', 
            self._handle_custom_prediction_async
        )

        self.distribution.message_queue.register_handler(
            'model_update',
            self._handle_model_update_async
        )
        
        # Start queue processor if not already running
        self.distribution.start_queue_processor()
        
        # Start health checker for async mode
        if self.distribution.use_async:
            self.distribution._start_health_checker()
            
        print("✅ Async message queue initialized")
    
    async def _handle_custom_prediction_async(self, message):
        """
        Async-queue handler (registered in `_setup_async_queue`) for a
        peer's `'custom_prediction'` message: runs `predict_single` on the
        requested text and returns a small success/error dict rather than
        raising, since this runs inside the message-queue dispatcher.
        """
        # Handle custom prediction requests asynchronously.
        try:
            text = message.payload.get('text', '')
            result = self.predict_single(text)
            return {
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'success': True
            }
        except Exception as e:
            return {
                'prediction': None,
                'confidence': 0.0,
                'success': False,
                'error': str(e)
            }
    
    async def _handle_model_update_async(self, message):
        """
        Async-queue handler (registered in `_setup_async_queue`) for a
        peer's `'model_update'` message: applies incoming `weights` via
        `self.update_weights` if present, returning a success/error dict.
        """
        # Handle model update requests asynchronously.
        try:
            weights = message.payload.get('weights')
            if weights:
                self.update_weights(weights)
                return {'success': True, 'message': 'Model updated'}
            return {'success': False, 'message': 'No weights provided'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    # ============ Async Prediction Methods ============
    
    def predict_async(self, text: Any, callback=None, timeout=30):
        """
                Fire off `predict_single` on a background thread via
                `distribution.message_queue` and invoke `callback(result)` when
                it completes, without blocking the caller. Requires the async
                queue to be set up (`_setup_async_queue`); intended for UI/
                server contexts where predictions shouldn't block the event
                loop or main thread.
        """
        # Async prediction with callback support.
        if not self.distribution or not self.distribution.use_async:
            # Fallback to sync prediction
            result = self.predict_single(text)
            if callback:
                callback(result)
            return result
        
        return self.distribution.request_prediction_async(
            agent_id='local',
            text=text,
            callback=callback
        )
    
    async def predict_async_await(self, text, timeout=30):
        """
        `await`-friendly counterpart to `predict_async`: runs
        `predict_single` in a thread-pool executor when the async queue
        isn't set up, otherwise awaits
        `distribution.request_prediction_async` with a hard timeout
        (`timeout + 5`s), raising `TimeoutError` if it's exceeded.
        """
        # Async prediction with await support.
        if not self.distribution or not self.distribution.use_async:
            # Fallback to sync prediction (run in thread)
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, self.predict_single, text
            )   
            
        try:
            result = await asyncio.wait_for(
                self.distribution.request_prediction_async('local', text, timeout=timeout),
                timeout=timeout + 5
            )
            return result
        except asyncio.TimeoutError:
            raise TimeoutError(f"[-] Prediction timed out after {timeout + 5}s")
        except Exception as e:
            logger.error(f"[-] Async prediction failed: {e}")
            traceback.print_exc()
            raise



    
    async def predict_batch_async(self, texts: List[str], callback=None):
        """
        Async counterpart to `prediction_batch`: falls back to the
        synchronous `prediction_batch` (optionally invoking `callback`)
        when the async queue isn't set up, otherwise delegates to
        `distribution.request_batch_prediction_async`.
        """
        # Batch async predictions.
        if not self.distribution or not self.distribution.use_async:
            # Fallback to sync batch prediction
            results = self.prediction_batch(texts)
            if callback:
                callback(results)
            return results

        return await self.distribution.request_batch_prediction_async(
            agent_id='local',
            texts=texts,
            callback=callback
        )


    # ============ Distributed Agent Methods ============
    
    def connect_peers(self, peer_addresses: List[tuple]):
        """
                Convenience pass-through: connect this pipeline's P2P layer to
                each `(host, port)` in `peer_addresses` via
                `self.distribution.connect_to_agent`.
        """
        # Connect to multiple peer agents.
        if not self.distribution:
            self.init_distributed()
        
        results = []
        for host, port in peer_addresses:
            try:
                sock = self.distribution.connect_to_agent(host, port)
                results.append({'host': host, 'port': port, 'success': sock is not None})
            except Exception as e:
                results.append({'host': host, 'port': port, 'success': False, 'error': str(e)})
        
        return results
    
    async def broadcast_to_peers(self, message_type: str, payload: dict):
        """
        Send `payload` under `message_type` to every connected peer via
        `self.distribution.broadcast`, run off the event loop thread
        (`asyncio.to_thread`) so it doesn't block. Raises if
        `self.distribution` hasn't been initialized.
        """
        # Broadcast message to all connected peers.
        if not self.distribution:
            raise RuntimeError("Distributed inference not initialized")
        
        return await asyncio.to_thread(
            self.distribution.broadcast,
            message_type, payload, timeout=10
        )
    
    def get_network_status(self):
        """
                Convenience pass-through to `self.distribution` for a quick
                summary of P2P health: number of connected peers, whether the
                server is listening, and this agent's own id/port.
        """
        # Get current network status.
        if not self.distribution:
            return {'status': 'not_initialized'}
        
        return {
            'status': 'active',
            'connected_agents': len(self.distribution.remote_agents),
            'queue_stats': self.distribution.get_queue_stats(),
            'mode': 'async' if self.distribution.use_async else 'sync'
        }
    
    # ============ Lifecycle Management ============

    async def shutdown_async(self):
        """
        Cancel any still-pending async tasks tracked in
        `self._async_tasks`, wait for them to finish, and stop the
        distribution (P2P) layer. Narrower than `shutdown` below — this
        only tears down the async-task/queue side, not the P2P server
        itself.
        """
        # Graceful shutdown of async components.
        print("🛑 Shutting down async components...")
        
        # Cancel all pending async tasks
        for task in self._async_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self._async_tasks:
            await asyncio.gather(*self._async_tasks, return_exceptions=True)
        
        # Shutdown distributed inference
        if self.distribution:
            self.distribution.stop()
        
        print("✅ Async shutdown complete")
    
    async def shutdown(self):
        """
        Full pipeline shutdown: stops the distribution layer's message
        processing and its listening server, signals `_shutdown_event` if
        present, and cancels/awaits any pending async tasks.
        """
        if self.distribution:
            self.distribution.stop()      # sync call
            self.distribution.stop_server()
            
            
            if hasattr(self, '_shutdown_event'):
                self._shutdown_event.set()

            # cancel async tasks
            if self._async_tasks:
                for task in self._async_tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*self._async_tasks, return_exceptions=True)


    def cosine_similarity(self, a, b):
        """
                Simpler, stricter cosine similarity than `cosine_robust_similarity`:
                still tolerates stringified/character-dtype arrays and coerces
                both inputs to float arrays, but handles shape mismatches with
                one layer of subset-slicing (rather than several fallback
                attempts) before giving up. Returns 0.0 on unrecoverable
                failure. Prefer `cosine_robust_similarity` for genuinely messy
                memory entries; this is the lighter-weight variant used on the
                hot prediction path.
        """
        eps = 1e-5
        if isinstance(a, (str, np.str_)):
            clean_str = str(a).replace('[', '').replace(']', '')
            a = np.fromstring(clean_str, sep=' ')         

        if isinstance(a, np.ndarray) and np.issubdtype(a.dtype, np.character):
            # catches arrays filled with string text
            clean_str = ' '.join(a.astype(str).flatten()).replace('[', '').replace(']', '')
            skip_values = {"...", "NaN", "null"}
            a = np.fromiter((v for v in clean_str.split() if v not in skip_values), dtype=float)

        if isinstance(b, (str, np.str_)):
            clean_str = str(b).replace('[', '').replace(']', '')
            b = np.fromstring(clean_str, sep=' ')
        if isinstance(b, np.ndarray) and np.issubdtype(b.dtype, np.character):
            # catches arrays filled with string text
            clean_str = ' '.join(b.astype(str).flatten()).replace('[', '').replace(']', '')
            skip_values = {"...", "NaN", "null"}
            b = np.fromiter((v for v in clean_str.split() if v not in skip_values), dtype=float)

        if isinstance(b, (float, int)) or isinstance(a, (float, int)):
            print('[!] Value is single scalar! returning 0.0 similarity')
            return 0.0

        b = b[0]

        if _OPT_AVAILABLE:
            if len(a.shape) > 1:
                a = np.asarray(a)
                a = a.reshape(-1)  
            if len(b.shape) > 1:
                b = np.asarray(b)
                b  = b.reshape(-1)

            return optimized_cosine_similarity(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))     

        try:
            # Handle variable b
            if isinstance(b, (str, np.str_)):
                clean_str = str(b).replace('[', '').replace(']', '')
                b = np.fromstring(clean_str, sep=' ')
            elif isinstance(b, np.ndarray) and np.issubdtype(b.dtype, np.character):
                clean_str = ' '.join(b.astype(str).flatten()).replace('[', '').replace(']', '')
                b = np.fromstring(clean_str, sep=' ')
            else:
                b = np.asarray(b, dtype=float)

            # handle variable a            
            if isinstance(a, (str, np.str_)):
                clean_str = str(a).replace('[', '').replace(']', '')
                a = np.fromstring(clean_str, sep=' ')
            elif isinstance(a, np.ndarray) and np.issubdtype(a.dtype, np.character):
                # catches arrays filled with string text
                clean_str = ' '.join(a.astype(str).flatten()).replace('[', '').replace(']', '')
                try:
                    a = np.fromstring(clean_str, sep=' ')
                except:
                    clean_string = clean_str.strip(",")  
                    a = np.fromiter(
                        (x for x in clean_string.split() if x != "..."), dtype=float
                    )
            else:
                # Ensure standard float array if it was integers or objects
                a = np.asarray(a, dtype=float)

            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
        
            if len(a.shape) > 1 and len(b.shape) > 1 and a.shape[1] != b.shape[0]:
                a = np.asarray(a)
                b = np.asarray(b)
                subset_a = a[0, :b.shape[0]]
                subset_b = a[:subset_a.shape[0], 0]

                try:
                    try:
                        dot_product = np.dot(subset_a, subset_b)
                    except:
                        dot_product = np.dot(subset_a[0, :min(subset_a.shape[1], subset_b.shape[0])], subset_b[:min(subset_a.shape[1], subset_b.shape[0])])
                except:
                    try:
                        dot_product = np.dot(a.flatten(), b[:a.flatten().shape[0]])
                    except:
                        try:
                            dot_product = np.dot(a[:b.shape[0]], b.flatten()[:a.shape[0]])
                        except:
                            print('[-] No similarity due to inhomogenous shapes and failed attempts to find subsets, returning low similarity score.')
                            return 0.1
        
            else:
                subset_a = a[:a.shape[0]]
                subset_b = b[:subset_a.shape[0]]

                try:
                    dot_product = np.dot(subset_a, subset_b) 
                except:
                    try:
                        subset_b_2 = subset_b[:subset_a.shape[1], :subset_a.shape[0]]  
                        dot_product = np.dot(subset_a, subset_b_2)

                    except:
                        print('[-] No similarity due to inhomogenous shapes and failed attempts to find subsets, returning low similarity score.')
    
                        return 0.0         

            cosine = dot_product / (norm_a * norm_b)
            cosine = np.clip(cosine, -1.0, 1.0)

        except Exception as e:
            print(f'[!] Cant calculate cosine similarity: {e}')
            cosine = 0.0

        return cosine  

    def anisotropy_measurement(self, x):
        """
                Measure how "directional" (as opposed to uniform/isotropic) the
                local structure of `x` is, via the coefficient of variation
                (std/mean) of gradient-vector norms. Used as a rough proxy for
                input complexity elsewhere in the pipeline (e.g. gating whether
                cache augmentation is worth it). Falls back to
                `self.confidence_threshold` on NaN/Inf or any conversion
                failure rather than propagating an error.
        """
        eps = 1e-5
    
        try:
            x = self._safe_convert(x)
        except:
            x = self._safe_to_2d_float(x)

        if _OPT_AVAILABLE:
            x = np.asarray(x)
            x = x.reshape(-1, 1)
            return optimized_anisotropy(np.asarray(x, dtype=np.float64))

        if isinstance(x, list):
            print(f'[=] Converting list to array with shape: {len(x)}')
            x = np.array(x)
            x = x.reshape(x.shape[0], -1)  # Flatten if necessary

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
            anisotropy = self.confidence_threshold

        if np.isnan(anisotropy) or np.isinf(anisotropy):
            anisotropy = self.confidence_threshold

        return anisotropy

    def modular_prediction_saving(self, X, X2, output):
        """
                Persist the most recent transformer.Transformer/mlp.MLP prediction pair into
                `self.memory` under the `'TW'` (transformer.Transformer) / `'MW'` (mlp.MLP) keys
                and flush it to `self.storage`. Inputs are passed through
                `_sanitize_for_storage` first. Resets `self.memory` to `{}` if it
                was ever mutated into a non-dict type.
        """
        memory_name = self.memory_name

        X = self._sanitize_for_storage(X)
        X2 = self._sanitize_for_storage(X2)
        output = self._sanitize_for_storage(output)

        if self.memory is None:
            self.memory = {}

        elif not isinstance(self.memory, dict):
            print(f'[!] Warning: memory was {type(self.memory)}, converting to dict')
            # Try to convert or reset
            if isinstance(self.memory, np.ndarray):
                # If memory was an array, changing type - log it
                print(f'[!] Converting memory from array to dict, old shape: {self.memory.shape}')
            self.memory = {}

        self.memory['TW'] = X, output # transformers Weight
        self.memory['MW'] = X2, output # mlp.MLP Weight
        try:
            print('🚀 Memory Prediction Added!')
            self.storage.save_model_dict(memory_name, self.memory, type='Pipeline', model_type='prediction')
        except Exception as e:
            print(f'[!] Cant save model memory: {e}')


    def modular_probability_saving(self, X, X2, prob):
        """
                Same as `modular_prediction_saving` but for probability outputs,
                stored under `'TP'` / `'MP'`.
        """
        memory_name = self.memory_name

        X = self._sanitize_for_storage(X)
        X2 = self._sanitize_for_storage(X2)
        prob = self._sanitize_for_storage(prob)

        if self.memory is None:
            self.memory = {}

        elif not isinstance(self.memory, dict):
            print(f'[!] Warning: memory was {type(self.memory)}, converting to dict')
            # Try to convert or reset
            if isinstance(self.memory, np.ndarray):
                # If memory was an array, changing type - log it
                print(f'[!] Converting memory from array to dict, old shape: {self.memory.shape}')
            self.memory = {}
    
        self.memory['TP'] = (X, prob)
        self.memory['MP'] = (X2, prob)
        
        # Save to database
        try:
            self.storage.save_model_dict(memory_name, self.memory, type='Pipeline', model_type='probs')
            print('🚀 Memory Probability Added!')
        except Exception as e:
            print(f'[!] Failed to save memory: {e}')
       
    def _cross_session_availability(self):
        """
                Interactive CLI menu (stdin `input()`) for cross-session memory
                housekeeping: export/import a named memory session, sync memory
                with another device over the network, or list existing sessions.
                Delegates the actual work to `self.session_automation`
                (`automation.CrossSessionAutomation`); this method is just the menu/prompt
                layer around it. Swallows all exceptions and prints a warning
                rather than propagating.
        """
        try:
            print('=== CROSS SESSION MEMORY AVAILABILITY AND TRANSFER ===')
            print('1. Export memory session')
            print('2. Import memory session')
            print('3. Sync with another device')
            print('4. List sessions')

            chosen = input('[=] Choose Options [1/2/3/4]: ')
            if chosen != '3':
                filename = input('[=] Insert filename name to export or import or list session (ex. name: memory_session): ')

            if chosen == '1':
                if filename:
                    print(f'[+] Exporting memory session with name {filename}...')
                    self.session_automation.export_session(filename)
                    print(f'🚀 {filename} successfully exported as json!')
                else:
                    print('[-] Invalid filename!')
                    pass

            elif chosen == '2':
                if isinstance(filename, str):
                    json_converted = filename + '.json'
                    print(f'[=] Importing memory session, filename: {json_converted}...')
                    self.session_automation.import_session(json_converted)
                    print(f'[+] Successfully imported {json_converted}! ')
                else:
                    print('[-] Invalid filename!')
                    pass

            elif chosen == '3':
                ip_number = input('[=] Insert ip number of your device for syncing: ')  
                if ip_number:
                    print('[=] Syncing with device to export memory session...')  
                    self.session_automation.sync_with_another_device(ip_number, port=5000) 
                else:
                    print('[-] Invalid filename!')
                    pass

            elif chosen == '4':
                print('[=] Listing sessions...')                
                print('[!] Note: you must put the common name of memory sessions you have, (ex: memory_sessions)')
                print('if your most memory sessions you have contains memory_ name in front, insert it in the input.')
                self.session_automation.list_sessions(filename)
            else:
                print('[-] Invalid options! ')
                pass

        except Exception as e:
            print(f'[-] Warning! Error detected during session availability: {e}')
            pass


    def model_memory_gate(self, x, x2):
        """
                Look up a previously-saved exact prediction (not probability —
                see `model_probability_gate` for that) for input `(x, x2)` in
                `self.memory`, tolerating several legacy memory shapes: a raw
                ndarray (reshaped/validated against the current class count), a
                list (delegated to `_gate_from_list`), or the normal dict form
                keyed by `'TW'`/`'MW'` prefixes. Any corrupted dict entries are
                deleted in place as they're encountered. Matching uses
                `cosine_robust_similarity`/`cosine_similarity` against stored
                inputs with a fixed 0.8 threshold.

                Returns the cached output array, or None if nothing matches
                closely enough (or the memory is unusable).
        """
        memory = self.memory
        is_corrupted, reason, _ = self.is_memory_corrupted(memory)      

        if isinstance(memory, np.ndarray):
            if self._get_num_classes() and memory.shape[-1] == self._get_num_classes():
                if is_corrupted:
                    print('[!] Memory corruption detected, Trying possible conversion to extract memory...')
                    print(f'[REASON]: {reason}')
                else:
                    print('[+] Memory is a direct array, converting...')
                               
                num_classes = self._get_num_classes()                 
                if memory.ndim == 1:                   
                    print(f'[+] Memory is flat array shape {memory.shape}, converting to 2D...')
                    converted_memory = memory.reshape(1, -1)
                    
                    if converted_memory.shape[-1] == num_classes:
                        print(f'[+] Memory converted to shape {converted_memory.shape}, returning it.')
                        return converted_memory
                    else:
                        print(f'[!] Shape mismatch! Got {converted_memory.shape[-1]} classes, expected {num_classes}')
                        return None
                
                # Case 2: Already 2D array (1, 8) or (N, 8)
                elif memory.ndim == 2:
                    if memory.shape[-1] == num_classes:
                        print(f'[+] Memory already in correct shape {memory.shape}, returning it.')
                        return memory
                    else:
                        print(f'[!] Memory shape mismatch! shape: {memory.shape}, expected classes: {num_classes}')
                        return None
                
                # Case 3: Higher dimensional array
                else:
                    print(f'[!] Memory has unexpected dimensions: {memory.ndim}D, shape: {memory.shape}, No matching memory!')
                    return None

            elif not self._get_num_classes():
                print('[!] Cant get number of classes from transformer and mlp.MLP model!')
                return None
            else:
                print(f'[!] Memory shape mismatch! shape: {memory.shape}, No matching memory! ')
                return None

        if isinstance(memory, list):
            print('[==] Memory is a list, checking contents...')
            if len(memory) > 0 and isinstance(memory[0], np.ndarray):
                print('[+] Got probability features inside list!')
                return memory[0].copy()
            else:
                if self._gate_from_list(memory, x, x2):
                    return self._gate_from_list(memory, x, x2)
                else:
                    print('[!] No matching item from memory!')
                    return None

        if isinstance(memory, dict):
            for key, value in list(memory.items()):
                is_val_corrupted, reason, _ = self.is_memory_corrupted(value)
                if is_val_corrupted and not key.startswith('supervised'):
                    print(f'[!] Removing corrupted entry {key}: {reason}')
                    del memory[key] 

            cache_trans_memory = [key for key, (inp) in memory.items() if key.startswith('TW') and (isinstance(inp, np.ndarray) or isinstance(inp, list)) and self.cosine_robust_similarity(x, inp) >= 0.8]
            cache_mlp_memory =  [key for key, (inp2) in memory.items() if key.startswith('MW') and (isinstance(inp2, np.ndarray) or isinstance(inp2, list)) and self.cosine_similarity(x2, inp2) >= 0.8]

            if cache_mlp_memory and cache_trans_memory:
                for memo in cache_trans_memory:
                    _, out = memory[memo]
                    
                for memo2  in cache_mlp_memory:
                    _, out = memory[memo2]

                if isinstance(out, str):
                    out = np.array([float(x) for x in out.strip('[]').split(',')])  # Convert string to numpy array

                output = out.copy()
                return output      
            else:
                if len(cache_mlp_memory) > 0:
                    print('[+] Found matching memory from mlp past memory!')                
                    for memo in cache_mlp_memory:
                        _, out = memory[memo] 
                        if isinstance(out, str):
                            out = np.array([float(x) for x in out.strip('[]').split(',')])  # Convert string to numpy array

                    output = out.copy() 
                    return output 

                elif len(cache_trans_memory) > 0:
                    print('[+] Found matching memory from transformer past memory!')                
                    for memo in cache_trans_memory:
                        _, out = memory[memo] 
                        if isinstance(out, str):
                            out = np.array([float(x) for x in out.strip('[]').split(',')])  # Convert string to numpy array

                    output = out.copy() 
                    return output

                else:
                    print('🔄 No Matching Memory!')
                    return None     
        else:
            print('[!] No matching memory types!')
            return None

    def _gate_from_list(self, memory: list, x, x2) -> Optional[np.ndarray]:
        """
                Search a list-shaped `memory` for an entry whose stored input is
                similar enough to `(x, x2)`, skipping any obviously-corrupted
                entries (scalar/None stored inputs, or values that fail array
                conversion) instead of raising.
        """
        print('[=] Extracting from list memory.. handling possible corruption...')

        for item in memory:
            if not isinstance(item, (tuple, list)) or len(item) < 2:
                continue

            stored_x   = item[0]
            stored_out = item[1]

            # guard against corrupted float entries
            if isinstance(stored_x, (int, float)):
                print(f'[!] Skipping corrupted entry — stored_x is scalar: {stored_x}')
                continue

            # guard against None
            if stored_x is None or stored_out is None:
                continue

            # ensure array before similarity
            try:
                stored_x_arr = self._to_numpy_array(stored_x)
                if stored_x_arr is None or stored_x_arr.ndim == 0:
                    continue
            except Exception as e:
                print(f'[!] Could not convert stored_x to array: {e}')
                continue

            try:
                print('[=] Checking similarity from stored X data...')
                if self.cosine_robust_similarity(x, stored_x_arr) >= 0.775:
                    return self._to_numpy_array(stored_out)
            except Exception as e:
                print(f'[!] Similarity check failed: {e}')
                continue

        print('[=] Cant get item from memory, possible dangerous data corruption!')
        return None


    def _to_numpy_array(self, value) -> Optional[np.ndarray]:
        """
        Convert various return types to numpy array.
        Handles: array, list, tuple, string, scalar.
        """
        if value is None:
            print('[!] Value is None!')
            return None
        
        # check here numpy array
        if isinstance(value, np.ndarray):
            return value.copy() if value.size > 0 else None
        
        # List or tuple of numbers
        if isinstance(value, (list, tuple)):
            try:
                arr = np.array(value, dtype=np.float32)
                if arr.ndim == 0:
                    arr = arr.reshape(1)
                return arr
            except:
                return None
        
        # String (from database JSON)
        if isinstance(value, str):
            try:
                # JSON first
                if value.startswith('['):
                    parsed = json.loads(value)
                    return np.array(parsed, dtype=np.float32)
                # space-separated
                parts = value.strip('[]').split()
                if parts:
                    return np.array([float(p) for p in parts], dtype=np.float32)
            except:
                pass
            return None
        
        # Scalar number
        if isinstance(value, (int, float)):
            return np.array([value], dtype=np.float32)
        
        print(f'[!] Cannot convert to array: {type(value)}')
        return None



    def model_probability_gate(self, x, x2):
        """
                Probability-cache counterpart to `model_memory_gate`: same
                legacy-shape handling (ndarray / list / dict), but reads the
                `'TP'`/`'MP'`-prefixed dict keys with a stricter 0.85 similarity
                threshold, and prefers a transformer.Transformer-sourced hit over an
                mlp.MLP-sourced one when both exist. Returns the cached probability
                array, or None.
        """
        output_trans = None
        output_mlp = None
        out = None

        memory = self.memory
        is_corrupted, reason, _ = self.is_memory_corrupted(memory)

        if isinstance(memory, np.ndarray):
            if self._get_num_classes() and not memory.shape[-1] == self._get_num_classes():
                if is_corrupted:
                    print('[!] Memory corruption detected, Trying possible conversion to extract memory...')
                    print(f'[MEMORY FAULT REASON]: {reason}')
                else:
                    print('[+] Memory is a direct probability, converting...')

                num_classes = self._get_num_classes()       

                if memory.ndim == 1:                  
                    print(f'[+] Memory is flat array shape {memory.shape}, converting to 2D...')
                    converted_memory = memory.reshape(1, -1)
                    
                    if converted_memory.shape[-1] == num_classes:
                        print(f'[+] Memory converted to shape {converted_memory.shape}, returning it.')
                        return converted_memory
                    else:
                        print(f'[!] Shape mismatch! Got {converted_memory.shape[-1]} classes, expected {num_classes}')
                        return None
                
                # Case 2: Already 2D array (1, 8) or (N, 8)
                elif memory.ndim == 2:
                    if memory.shape[-1] == num_classes:
                        print(f'[+] Memory already in correct shape {memory.shape}, returning it.')
                        return memory
                    else:
                        print(f'[!] Memory shape mismatch! shape: {memory.shape}, expected classes: {num_classes}')
                        return None
                
                # Case 3: Higher dimensional array
                else:
                    print(f'[!] Memory has unexpected dimensions: {memory.ndim}D, shape: {memory.shape}, No matching memory!')
                    return None

            elif not self._get_num_classes():
                print('[!] Cant get number of classes from transformer and mlp.MLP model!')
                return None
            else:
                print(f'[!] Memory shape mismatch! shape: {memory.shape}, No matching memory! ')
                return None

        if isinstance(memory, list):
            print('[==] Memory is a list, checking contents...')
            if len(memory) > 0 and isinstance(memory[0], np.ndarray):
                print('[+] Got probability features inside list!')
                return memory[0].copy()
            else:
                self._gate_from_list(memory, x, x2)

        if isinstance(memory, dict):
            for key, value in list(memory.items()):
                is_val_corrupted, reason, _ = self.is_memory_corrupted(value)
                if is_val_corrupted:
                    print(f'[!] Removing corrupted entry {key}: {reason}')
                    del memory[key] 

            cache_trans_memory = [key for key, (inp) in memory.items() if key.startswith('TP') and (isinstance(inp, np.ndarray) or isinstance(inp, list)) and self.cosine_robust_similarity(x, inp) >= 0.85]
            cache_mlp_memory =  [key for key, (inp2) in memory.items() if key.startswith('MP') and (isinstance(inp2, np.ndarray) or isinstance(inp2, list)) and self.cosine_similarity(x2, inp2) >= 0.85]

            if len(cache_mlp_memory) > 0 or len(cache_trans_memory) > 0:
                print('[+] Memory length found:')
                print(f'[=] mlp.MLP Memory length: {len(cache_mlp_memory)}')
                print(f'[=] transformer.Transformer Memory length: {len(cache_trans_memory)}')

                if cache_trans_memory is not None:
                    for memo in cache_trans_memory:
                        _, output_trans = memory[memo]
                if cache_mlp_memory is not None:
                    for memo2  in cache_mlp_memory:
                        _, output_mlp = memory[memo2]

                if output_trans is not None:
                    out = output_trans.copy()
                elif output_mlp is not None:
                    out = output_mlp.copy()
                else:
                    print('[!] No matched memory from given samples.')

                if out is not None and isinstance(out, str):
                    out = np.array([float(x) for x in out.strip('[]').split(',')])  # Convert string to numpy array

                output = out if out is not None else None
                return output  
                
            else:
                print('🔄 No Matching Probability!')
                return None
        else:
            print('[!] No matching memory types!')
            return None

    def prediction_batch(self, texts):
        """
                Batched prediction entry point over raw `texts`: fits/encodes
                each text for both the transformer.Transformer (`encode`) and mlp.MLP (tf-idf),
                stacks them into `(batch, seq_len)` / `(batch, features)`
                arrays, derives `y_true` (from `self.labels`/`self.titles` if
                set, otherwise a placeholder), and delegates to
                `_batch_prediction_core`.
        """
        self.initialize_fitting(texts)
        if not texts:
            return []
        
        # Prepare batch inputs

        input_ids_list = []
        X_raw_list = []

        for text in texts:
            # Prepare transformer input
            input_ids = np.array([self.encode(text, self.vocab)])
            input_ids_list.append(input_ids)
            
            # Prepare mlp.MLP input
            if not hasattr(self, 'tfidf') or self.tfidf is None:
                self.initialize_fitting([text])
            X_raw = self.tfidf.transform([text]).toarray()
            X_raw_list.append(X_raw)
        
        # Stack into batches
        batch_input_ids = np.vstack(input_ids_list)  # (batch_size, seq_len)
        batch_X_raw = np.vstack(X_raw_list) # (batch_size, features)

        if self.labels is None and self.titles is None:
            _, y_true = self.input_encoding(list(zip(texts, texts)))  # No-value labels for y_true
        else:
            dataset, _ = self.data_preparation(self.titles, self.labels)
            y_true = self.input_encoding(dataset)
        
        # Run batch prediction through your existing logic
        return self._batch_prediction_core(batch_input_ids, batch_X_raw, y_true)



    def _batch_model_memory_gate(self, batch_input_ids, batch_X_raw):
        """
                Per-item wrapper around `model_memory_gate` for an entire batch:
                returns a list the same length as the batch, with `None` in the
                positions that had no cached match (so the caller knows which
                rows still need a fresh model prediction).
        """
        batch_probs = [None] * len(batch_input_ids)
        
        for i in range(len(batch_input_ids)):
            probs = self.model_memory_gate(
                batch_input_ids[i:i+1], 
                batch_X_raw[i:i+1]
            )
            if probs is not None:
                arr = np.array(probs)
                if arr.ndim > 1:
                    arr = arr[0]
                elif arr.ndim == 0:
                    arr = arr.reshape(1)
                batch_probs[i] = arr.copy()
        
        return batch_probs

    def _coerce_batch_size(self, batch_size, default=32):
        """Single, clear path for batch_size type coercion."""
        try:
            arr = np.asarray(batch_size)
            if arr.ndim == 0:
                return int(arr)
            elif arr.size > 0:
                return int(arr.flat[0])
            else:
                return default
        except (TypeError, ValueError):
            return default



    def _batch_prediction_core(self, batch_input_ids: np.ndarray, batch_X: np.ndarray, 
                                batch_size: Any = None, show_progress: bool = True) -> np.ndarray:
        """
        Robust batch prediction with dynamic shape handling.
        """
        if len(batch_input_ids) == 0:
            return np.array([])
        
        # Auto-calculate optimal batch size if not provided
        if batch_size is None:
            batch_size = self._calculate_optimal_batch_size(batch_input_ids, batch_X)
        
        if isinstance(batch_size, (list, np.ndarray)):
            try:
                try:
                    batch_size = int(batch_size[0]) if len(batch_size) > 0 else 32
                except:
                    batch_size = int(batch_size[0][0])
            except:
                # robust single clear path when both batch size try blocks fails
                batch_size = self._coerce_batch_size(batch_size, default=32)

        else:
            if isinstance(batch_size, (tuple, list, np.ndarray)):
                batch_size = batch_size[0]
                batch_size = len(batch_size)
            else:
                batch_size = batch_size = self._coerce_batch_size(batch_size, default=32)
            
        n_samples = len(batch_input_ids)
        chunks = []
        if isinstance(batch_size, (tuple, list)):
            batch_size = batch_size[0] 

        print(f'[=] Total samples: {n_samples}, using batch size: {batch_size}, total batches: {((n_samples - 1) // batch_size) + 1}')
        
        for i in range(0, n_samples, batch_size):
            chunk = (
                batch_input_ids[i:i + batch_size],
                batch_X[i:i + batch_size]
            )
            chunks.append(chunk)
        
        # ✅ Determine number of classes dynamically from first successful chunk
        num_classes = self._get_num_classes()
        batch_probs = np.zeros((n_samples, num_classes)) 
        
        for chunk_idx, (chunk_ids, chunk_X) in enumerate(chunks):
            if show_progress:
                start_idx = chunk_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                print(f"\r📊 Processing batch {chunk_idx + 1}/{len(chunks)} (samples {start_idx}-{end_idx})...", end="")
            
            try:
                # Process chunk with memory gate
                chunk_probs = self._process_batch_chunk(chunk_ids, chunk_X)
                
                # ✅ Handle numpy array conversion
                if isinstance(chunk_probs, list):
                    chunk_probs = np.array(chunk_probs)
                
                # ✅ Determine number of classes from first successful chunk
                if num_classes is None:
                    num_classes = chunk_probs.shape[1] if chunk_probs.ndim > 1 else 1

                    # Initialize results array
                    print(f'\n[=] Detected {num_classes} classes from chunk {chunk_idx + 1}')
                
                # ✅ Handle dimension mismatches
                if chunk_probs.ndim == 1:
                    chunk_probs = chunk_probs.reshape(-1, 1)
                  

                # ✅ If chunk has different number of classes, pad or trim
                if chunk_probs.shape[1] != num_classes:
                    print(f'[=] Shape mismatch: chunk has {chunk_probs.shape[1]} classes, expected {num_classes}')
                    
                    if chunk_probs.shape[1] > num_classes:
                        # Trim extra classes
                        chunk_probs = chunk_probs[:, :num_classes]
                        print(f'[=] Trimmed to {chunk_probs.shape[1]} classes')
                    else:
                        # Pad missing classes
                        padded = np.zeros((chunk_probs.shape[0], num_classes))
                        padded[:, :chunk_probs.shape[1]] = chunk_probs
                        chunk_probs = padded
                        print(f'[=] Padded to {chunk_probs.shape[1]} classes')
                
                # Place in results
                start_idx = chunk_idx * batch_size
                end_idx = start_idx + len(chunk_probs)
                batch_probs[start_idx:end_idx] = chunk_probs
                
            except Exception as e:
                print(f"\n⚠️ Chunk {chunk_idx + 1} failed: {e}")
                traceback.print_exc()
                
                # Fill failed chunk with zeros
                start_idx = chunk_idx * batch_size
                end_idx = start_idx + len(chunk_ids)
                
                if batch_probs is None:
                    # If first chunk failed, initialize with default
                    num_classes = self._get_num_classes()
                    batch_probs = np.zeros((n_samples, num_classes))
                
                batch_probs[start_idx:end_idx] = 0
        
        if show_progress:
            print(f"\r✅ Batch complete: {n_samples} samples processed")
        
        return batch_probs if batch_probs is not None else np.array([])

    def _process_batch_chunk(self, chunk_ids: np.ndarray, chunk_X: np.ndarray) -> Any:
        """
        Process a single chunk - core batch logic with memory gate.
        """
        use_embedded = False
        chunk_probs = self._batch_model_memory_gate(chunk_ids, chunk_X)

        num_classes = self._get_num_classes()
        if num_classes is None:
            print('[⚠️] num_classes unavailable in _process_batch_chunk — '
                'cannot safely process this chunk')
            return np.zeros((len(chunk_ids), 1))   # minimal safe fallback shape

        needs_fresh = [i for i, p in enumerate(chunk_probs) if p is None]
        need_ensemble = (
            len(chunk_ids) > 100 and
            len(chunk_X) > 100 
        )
        if needs_fresh:
            fresh_ids = chunk_ids[needs_fresh]
            fresh_X   = chunk_X[needs_fresh]

            if need_ensemble:
                fresh_probs, _ = self.ensemble.predict_ensemble(
                    fresh_ids, fresh_X,
                    np.zeros((len(fresh_ids), num_classes)),
                    method='dynamic', embedded=False
                )
            else:
                fresh_probs = self.model3.forward(fresh_X)
                performance_result = self.model3.performance_calculation(fresh_X)
                self.performance_result = performance_result

                try:
                    fresh_trans_probs, _ = self.model2.forward(fresh_ids, embedded=False)
                except:
                    use_embedded = True
                    fresh_trans_probs, _ = self.model2.forward(fresh_ids, embedded=True)
                    
                fresh_mlp_confidence = fresh_probs.max(axis=1, keepdims=True)
                fresh_trans_confidence = fresh_trans_probs.max(axis=1, keepdims=True)

                if fresh_mlp_confidence >= fresh_trans_confidence:
                    fresh_probs = fresh_probs.copy()
                else:
                    fresh_probs = fresh_trans_probs.copy()

                if np.std(fresh_probs) < 0.3 or np.mean(fresh_probs) < 0.3:
                    fresh_probs = self.predict_proba(
                        fresh_ids, fresh_X,
                        type='Hybrid', embedded=use_embedded
                    )                    


            # validate fresh_probs shape before assignment
            fresh_probs = np.asarray(fresh_probs)
            if fresh_probs.ndim == 1:
                fresh_probs = fresh_probs[np.newaxis, :]

            cached_count = 0   # track global cache count

            for i, fresh_idx in enumerate(needs_fresh):
                row = fresh_probs[i]

                # guard — ensure row matches expected width before storing
                if row.shape[0] != num_classes:
                    aligned = np.zeros(num_classes)
                    min_len = min(row.shape[0], num_classes)
                    aligned[:min_len] = row[:min_len]
                    row = aligned

                chunk_probs[fresh_idx] = row

                # cap caching meaningfully
                if cached_count < 2:
                    self.modular_prediction_saving(
                        fresh_ids[i:i+1],
                        fresh_X[i:i+1],
                        row[np.newaxis, :]
                    )
                    cached_count += 1


        return np.array([
            p if p is not None else np.zeros(num_classes)
            for p in chunk_probs
        ])



    def _calculate_optimal_batch_size(self, batch_input_ids: np.ndarray, batch_X: Any=None) -> Any:
        """
        Calculate optimal batch size based on available memory.
        """
        try:
            # Estimate memory per sample
            sample_size = batch_input_ids[0].nbytes + batch_X[0].nbytes if hasattr(batch_X, '__len__') else 1024
            available_memory = psutil.virtual_memory().available
            max_samples = int(available_memory * 0.1 / sample_size)  # Use 10% of memory
            return min(64, max(8, max_samples))
        except:
            # Fallback to conservative batch size
            return 32

    def _get_num_classes(self, label_map: dict = None, mlp_probs: np.ndarray = None) -> Any:
        """
        Single source of truth for num_classes across the entire pipeline.

        Resolution priority:
        1. Actual model output shape (model2.output or mlp final layer)
            — this is authoritative since argmax indices are bounded by this
        2. label_map length — used only as cross-check / fallback
        3. mlp_probs.shape[1] — used only as last-resort fallback

        Logs a warning if sources disagree, since that disagreement
        is what causes invalid prediction indices downstream.
        """
        model_classes = None

        # primary source — actual model output dimension
        if hasattr(self, 'model2') and self.model2:
            model_classes = self.model2.output.shape[1]
        if hasattr(self, 'mlp') and self.mlp.layers:
            model_classes = self.mlp.layers[-1].b.shape[1]

        # cross-check against label_map if provided
        if label_map is not None:
            label_classes = len(label_map)
            if model_classes is not None and label_classes != model_classes:
                print(f'[⚠️] num_classes mismatch: model={model_classes} '
                    f'label_map={label_classes} — using model output as source of truth')
            if model_classes is None:
                model_classes = label_classes

        # cross-check against mlp_probs if provided
        elif mlp_probs is not None:
            probs_classes = mlp_probs.shape[1] if mlp_probs.ndim > 1 else len(mlp_probs[0])
            if model_classes is not None and probs_classes != model_classes:
                print(f'[⚠️] num_classes mismatch: model={model_classes} '
                    f'mlp_probs={probs_classes} — using model output as source of truth')
            if model_classes is None:
                model_classes = probs_classes

        # Dangerous Fallbacks (some of This may corrupt predictions)
        if model_classes is None:
            if self.manager:
                model_classes = len(self.manager.label_map)
            if self.vocab:
                raise Warning("[!] Model Classes is still None, using amount of vocab as number of clasess")
                model_classes = len(self.vocab)
            else:
                try:
                    model_classes = 1 # Fallback if everything else Fails.
                    raise Warning('[!] All Possible Methods of getting Number of classes fails! This may corrupt possible prediction downstream, Consider restart prediction and initialized models correctly with correct samples and label map!')
                except:
                    model_classes = 1
                    pass

        return model_classes
        
  
    def predict_async(self, text, callback=None):
        """
                Same as the earlier `predict_async` overload (queue a
                `predict_single` call on a background thread and invoke
                `callback` on completion) but without a timeout parameter — a
                second definition of this method name shadows the first;
                Python keeps only this later one.
        """
        try:
            id_req = self.batcher.add_request(text, callback)
            result = self.batcher.get_result(id_req, timeout=10)
            self.batcher.cleanup_stale()
            return id_req

        except Exception as e:
            print(f'[=] error in automatic batcher: {e}')
            return None
    

    def predict_single(self, text):
        """
                Thin synchronous wrapper that runs a single-text prediction
                through the full hybrid pipeline and returns its result
                immediately (as opposed to `predict_async`, which defers to a
                background thread).
        """
        result = [None]
        
        def callback(r):
            result[0] = r
        
        self.predict_async(text, callback)
        
        # Wait for result
        while result[0] is None:
            time.sleep(0.001)
        
        return result[0]
 

    def _batch_hybrid_prediction(self, batch_input_ids, batch_X_raw, y_true, embedded=True):
        """
                Batch-mode counterpart to the single-item probability-agreement
                logic in `predict_proba`: runs mlp.MLP and transformer.Transformer forward
                passes over the whole batch, compares their per-row argmax
                predictions, and for rows where they disagree, calibrates via
                `_calibrate_probs`; rows where they agree get a confidence boost
                on the agreed class instead. Also opportunistically saves a
                couple of fresh probability results into memory
                (`modular_probability_saving`, capped via `_prob_save_count`).
        """
        print('[+] Initiating hybrid prediction batching...')

        num_classes = self._get_num_classes()
        if num_classes is None:
            print('[⚠️] num_classes unavailable — cannot safely batch predict')
            return np.zeros((len(batch_input_ids), 1))

        zero_row    = np.zeros(num_classes)   # reusable fallback shape
        batch_probs = self._batch_model_memory_gate(batch_input_ids, batch_X_raw)
        fresh_probs = None                    # FIX 1 — explicit init, no UnboundLocalError

        needs_prediction = [i for i, p in enumerate(batch_probs) if p is None]
        need_ensemble = (
            len(batch_input_ids) > 100 and
            len(batch_X_raw) > 100 
        )

        if needs_prediction:
            fresh_input_ids = batch_input_ids[needs_prediction]
            fresh_X_raw     = batch_X_raw[needs_prediction]

            # slice y_true to match the fresh subset only
            if y_true is not None and hasattr(y_true, '__len__'):
                fresh_y_true = y_true[needs_prediction] \
                            if len(y_true) == len(batch_input_ids) \
                            else np.zeros((len(needs_prediction), num_classes))
            else:
                fresh_y_true = np.zeros((len(needs_prediction), num_classes))

            if need_ensemble:
                fresh_probs, _ = self.ensemble.predict_ensemble(
                    fresh_input_ids, fresh_X_raw,
                    np.zeros((len(fresh_input_ids), num_classes)),
                    method='dynamic', embedded=embedded
                )
            else:
                fresh_probs = self.model3.forward(fresh_X_raw, y=y_true)
                try:
                    fresh_trans_probs, _ = self.model2.forward(fresh_input_ids, embedded=False)
                except:
                    fresh_trans_probs, _ = self.model2.forward(fresh_input_ids, embedded=embedded)

                fresh_mlp_confidence = fresh_probs.max(axis=1, keepdims=True)
                fresh_trans_confidence = fresh_trans_probs.max(axis=1, keepdims=True)

                if fresh_mlp_confidence >= fresh_trans_confidence:
                    fresh_probs = fresh_probs.copy()
                else:
                    fresh_probs = fresh_trans_probs.copy()

                if np.std(fresh_probs) < 0.3 or np.mean(fresh_probs) < 0.3:
                    fresh_probs, _ = self.ensemble.predict_ensemble(
                        fresh_input_ids, fresh_X_raw,
                        np.zeros((len(fresh_input_ids), num_classes)),
                        method='dynamic', embedded=embedded
                    )   

            # validate fresh_probs shape before assignment
            fresh_probs = np.asarray(fresh_probs)
            if fresh_probs.ndim == 1:
                fresh_probs = fresh_probs[np.newaxis, :]

            for i, idx in enumerate(needs_prediction):
                row = fresh_probs[i]

                # align to expected num_classes if shape drifted
                if row.shape[0] != num_classes:
                    aligned = np.zeros(num_classes)
                    min_len = min(row.shape[0], num_classes)
                    aligned[:min_len] = row[:min_len]
                    row = aligned

                batch_probs[idx] = row

                # instance-level counter
                if getattr(self, '_cache_save_count', 0) < 2:
                    self.modular_prediction_saving(
                        fresh_input_ids[i:i+1],
                        fresh_X_raw[i:i+1],
                        row[np.newaxis, :]
                    )
                    self._cache_save_count = getattr(self, '_cache_save_count', 0) + 1

        # build valid_probs with guaranteed consistent shape
        valid_probs = []
        for i, p in enumerate(batch_probs):
            if p is None:
                # zero_row always defined, no UnboundLocalError
                valid_probs.append(zero_row.copy())
            elif isinstance(p, list):
                arr = np.array(p, dtype=np.float64)
                if arr.shape[0] != num_classes:
                    aligned = np.zeros(num_classes)
                    aligned[:min(arr.shape[0], num_classes)] = arr[:num_classes]
                    arr = aligned
                valid_probs.append(arr)
            elif isinstance(p, np.ndarray):
                if p.shape[0] != num_classes:
                    aligned = np.zeros(num_classes)
                    aligned[:min(p.shape[0], num_classes)] = p[:num_classes]
                    p = aligned
                valid_probs.append(p)
            else:
                print(f'[⚠️] Unexpected type in batch_probs[{i}]: {type(p)} — using zeros')
                valid_probs.append(zero_row.copy())

        # single explicit conversion with clear error, no nested bare except
        if not valid_probs:
            print('[⚠️] No valid probabilities collected — returning zeros')
            return np.zeros((len(batch_input_ids), num_classes))

        try:
            result = np.stack(valid_probs)   # stack guarantees shape (N, num_classes)
                                            # np.array() on ragged list gives object array
                                            # np.stack() fails fast if shapes disagree
            return result
        except ValueError as e:
            print(f'[⚠️] Stack failed (shape mismatch): {e} — '
                f'shapes: {[p.shape for p in valid_probs]}')
            # last resort return
            return valid_probs
        

    def _batch_predict_proba(self, batch_input_ids, batch_X, type='Hybrid'):
        """
                Batch counterpart to `predict_proba`'s single-item probability
                gate + fresh-prediction flow: checks `model_probability_gate`
                per row, runs transformer.Transformer/mlp.MLP predictions for the rows that miss,
                calibrates disagreements between the two models
                (`_calibrate_probs`) or boosts agreement confidence, and
                assembles a uniform `(batch, num_classes)` array — using the
                supplied `output_memory` (from `_batch_model_memory_gate`) or
                zeros to fill any row that ends up an unexpected type.
        """
        batch_size = len(batch_input_ids)
        
        output_memory = self._batch_model_memory_gate(batch_input_ids, batch_X)    
        num_classes = self._get_num_classes()
        if num_classes is None:
            print('[⚠️] num_classes unavailable in prediction batching function!')
            return np.zeros((batch_size, 1))

        zero_row    = np.zeros(num_classes)
        batch_probs = [None] * batch_size
        fresh_probs = None   # explicit init, no UnboundLocalError

        for i in range(batch_size):
            probs = self.model_probability_gate(
                batch_input_ids[i:i+1],
                batch_X[i:i+1]
            )
            if probs is not None:
                batch_probs[i] = probs[0]
                    

        needs_prediction = [i for i, p in enumerate(batch_probs) if p is None]

        try:
            if needs_prediction:
                fresh_input_ids = batch_input_ids[needs_prediction]
                fresh_X         = batch_X[needs_prediction]

                transformer_pred, fresh_probs, attn_weights = self.model2.predict(fresh_input_ids)
                mlp_pred = self.mlp.forward(fresh_X)

                # coerce indices to int 
                mlp_pred_indices   = np.argmax(mlp_pred,   axis=1).astype(int)
                trans_pred_indices = np.argmax(fresh_probs, axis=1).astype(int)

                for i, idx in enumerate(needs_prediction):

                    # validate indices in range before use
                    mlp_cls   = int(mlp_pred_indices[i])
                    trans_cls = int(trans_pred_indices[i])

                    mlp_cls   = mlp_cls   if 0 <= mlp_cls   < num_classes else 0
                    trans_cls = trans_cls if 0 <= trans_cls < num_classes else 0

                    if mlp_cls != trans_cls:
                        calibrated = self._calibrate_probs(
                            fresh_probs[i:i+1],
                            [mlp_cls],
                            attn_weights[i:i+1] if attn_weights is not None else None,
                            fresh_input_ids[i:i+1]
                        )
                        # validate calibrated output shape
                        row = np.asarray(calibrated[0])
                        if row.shape[0] != num_classes:
                            aligned = np.zeros(num_classes)
                            aligned[:min(row.shape[0], num_classes)] = row[:num_classes]
                            row = aligned
                        batch_probs[idx] = row

                    else:
                        # models agree — boost confidence on agreed class
                        probs_i        = fresh_probs[i].copy() if i < len(fresh_probs) else fresh_probs[0]
                        probs_i[trans_cls] = min(probs_i[trans_cls] * 1.2, 0.95)
                        row_sum        = probs_i.sum()
                        probs_i       /= row_sum if row_sum > 1e-8 else 1.0
                        batch_probs[idx] = probs_i

                    # instance-level save counter, not local idx_total
                    if getattr(self, '_prob_save_count', 0) < 2:
                        self.modular_probability_saving(
                            fresh_input_ids[i:i+1],
                            fresh_X[i:i+1],
                            np.array([batch_probs[idx]])
                        )
                        self._prob_save_count = getattr(self, '_prob_save_count', 0) + 1

            else:
                raise Warning('[!] Data is None before batching!')

            # safe final assembly with consistent shape
            valid_probs = []
            for i, p in enumerate(batch_probs):
                if p is None:
                    valid_probs.append(zero_row.copy())
                elif isinstance(p, list):
                    arr = np.array(p, dtype=np.float64)
                    if arr.shape[0] != num_classes:
                        aligned = np.zeros(num_classes)
                        aligned[:min(arr.shape[0], num_classes)] = arr[:num_classes]
                        arr = aligned
                    valid_probs.append(arr)
                elif isinstance(p, np.ndarray):
                    if p.shape[0] != num_classes:
                        aligned = np.zeros(num_classes)
                        aligned[:min(p.shape[0], num_classes)] = p[:num_classes]
                        p = aligned
                    valid_probs.append(p)
                else:
                    if output_memory is not None:
                        print('[=] Unexpected Sample type in batch probability, Using previous memory to fill gaps in Samples Ambiguity')
                        valid_probs.append(output_memory)
                    else:
                        print(f'[⚠️] Unexpected Sample type in batch_probs[{i}]: {type(p)} — using zeros to fill in valid probability')
                        valid_probs.append(zero_row.copy())
                    
            try:
                return np.stack(valid_probs)
            except ValueError as e:
                print(f'[⚠️] Stack failed: {e} — shapes: {[p.shape for p in valid_probs]}')
                return valid_probs
        except Exception as e:
            print(f'[⚠️] Error occured in probabilities batching: {e}')
            return None


    def hybrid_prediction(self, rules, input_ids, dataset, X=None, y=None, use_embedded=True):
        """
                Ensemble-based prediction for a single item: checks
                `model_memory_gate` first, and if there's no cached hit (or
                `self.agreement` is False), asks the ensemble
                (`self.ensemble.predict_ensemble`) for a fresh prediction. In
                non-autonomous mode this includes an interactive CLI prompt to
                pick which ensemble method (`dynamic`/`meta`/`attention`) to use,
                or to hand full control to autonomous mode. For a batch input
                (`input_ids` with more than one row), delegates entirely to
                `_batch_hybrid_prediction`. Saves any freshly-computed
                prediction back into memory via `modular_prediction_saving`.
        """
        if X is None or y is None:
            X, y, _, _ = self.feature_generation(rules, dataset) 
        
        if isinstance(input_ids, list):
            try:
                input_ids = np.asarray(input_ids)
            except Exception as e:
                input_ids = self._safe_to_2d_float(input_ids)
                
        if len(input_ids.shape) == 2 and input_ids.shape[0] > 1:
            # this is batch mode version
            return self._batch_hybrid_prediction(input_ids, X, y, embedded=use_embedded)

        probs = self.model_memory_gate(input_ids, X)

        if probs is None or not self.agreement:
            if not self.autonomous:
                print('= Prediction Method needed: ')
                print('[1]. dynamic')
                print('[2]. meta')
                print('[3]. attention')

                print('[-] Autonomous prediction give the model full control of its dynamic prediction, without any user input.')
                choose_method = input('[=] Autonomous prediction initiated? [Y/N] (press N to insert manual prediction method): ')

                if choose_method == 'Y':
                    self.autonomous = True
                    probs, details = self.ensemble.predict_ensemble(input_ids, X, y, method='dynamic', embedded=use_embedded) 

                else:
                    method = input('|| Choose one method (ex: dynamic): ')
                    if method:
                        probs, details = self.ensemble.predict_ensemble(input_ids, X, y, method=method, embedded=use_embedded)
                    else:
                        print('|| Invalid Method.. returning to dynamic prediction..')
                        probs, details = self.ensemble.predict_ensemble(input_ids, X, y, method='dynamic', embedded=use_embedded)    
            else:
                print('[+] Autonomous dynamic prediction: ')
                probs, details = self.ensemble.predict_ensemble(input_ids, X, y, method='dynamic', embedded=use_embedded) 

            self.modular_prediction_saving(input_ids, X, probs)
            print('🚀 Memory Added!')

        return probs

    def _handle_distributed_connections(self, probs, self_attn_weights, input_ids, agreement):
        """
                Interactive (or autonomous) router for asking a peer to help
                resolve an uncertain prediction. Offers a choice between an
                in-device peer (`self.distribution._handle_peer_agent_request`
                with `type='DevicePeer'`) and an external peer reached by IP
                (connect, then either request a prediction with an ensemble vote
                across all known remote agents, or hand the uncertainty request
                straight to `_handle_peer_agent_request` with
                `type='ExternalPeer'`). In autonomous mode the in-device path is
                taken automatically. Falls back to raising the
                `peer_assistance_threshold` and enabling external peers if local
                peer trust turns out to be low. Returns the (possibly
                peer-adjusted) `probs`.
        """
        print('=== AGENT DISTRIBUTIED INFERENCE HANDLING ===')
        print('1. Handle local In-device Peer')
        print('2. Handle external-device Peer')

        program = None
        if not self.autonomous:
            program = input('[=] Pick your choice [1/2] (choose N to skip): ')

        if program == '1' or self.autonomous:
            print('=== IN-DEVICE PEER REQUEST INITIATED ===')
            probs = self.distribution._handle_peer_agent_request(probs, self_attn_weights, input_ids, type='DevicePeer', agreement=agreement)
            if self.distribution.query_node.peer_trust < self.confidence_threshold:
                print('[-] Peer trust is low, broadcasting ping to check for better peers...')
                alive_agents = self.distribution.broadcast_ping()

                if alive_agents:
                    print(f'[+] Alive agents: {alive_agents} identified, enabling external peer connections for better assistance...')
                    self.external_peer_enabled = bool(alive_agents)
                    self.autonomous = False

        elif program == '2' or self.external_peer_enabled:
            print('=== EXTERNAL PEER REQUEST INITIATED ===')
            ip_number = input('[=] Insert IP Number to connect with peer: ')

            if ip_number:
                try:
                    distributed_a = self.distribution
                    
                    distributed_a.start_server()
                    if distributed_a.connect_to_agent(ip_number, 5555):
                        print(f'[=+=] Successfully connected to external peer at {ip_number}!')


                        print('=== EXTERNAL PEER REQUEST INITIATED ===')
                        print('[1]. Request prediction')
                        print('[2]. Handle Peer uncertainty')
                        sec_program = input('[=] Pick your choice [1/2]: ')    

                        if sec_program == '1':
                            for intent in self.intents:
                                result = distributed_a.request_prediction_method(self, intent)
                                print(f"[+] Remote prediction: {result}")
                    
                                # Ensemble vote across all agents
                                votes = []
                                list_probs = []

                            # Check network status
                            print('=== Checking network status with broadcast ping... ===')
                            alive_agents = distributed_a.broadcast_ping()  
                            if alive_agents:
                                print(f'[+] Alive agents: {alive_agents} identified, requesting ensemble votes...')                          
                                for agent_id in distributed_a.remote_agents:
                                    peer_probs, vote = distributed_a.request_ensemble_vote(agent_id, intent)
                                    if vote:
                                        votes.append(vote)
                                        list_probs.append(peer_probs)
                                        
                                        for vote in votes:
                                            print(f'[+] Prediction: {vote['prediction']}')
                                            print(f'[+] Confidence: {vote['confidence']}')
                                            print(f'[+] Trust: {vote['trust_score']}')
                                            if vote['confidence'] > self.confidence_threshold:
                                                for i in range(len(probs)):
                                                    probs = probs[i] * (1.0 + vote['trust_score'] * vote['confidence'])
                                                probs = probs.copy() / np.sum(probs) # Normalize after adjustment
                                else:
                                    print(f'[-] No alive agents found, Total: {alive_agents} Agent found. Using local prediction only.')
                                    probs = self.distribution._handle_peer_agent_request(probs, self_attn_weights, input_ids, type='ExternalPeer', agreement=agreement)

                            distributed_a.print_network_status()

                        elif sec_program == '2':
                            probs = self.distribution._handle_peer_agent_request(probs, self_attn_weights, input_ids, type='ExternalPeer', agreement=agreement)     
                    else:
                        print(f'[-] No Peer agents found, Using local prediction only.')
                        probs = self.distribution._handle_peer_agent_request(probs, self_attn_weights, input_ids, type='DevicePeer', agreement=agreement) 


                except Exception as e:
                    print(f'[-] Error establishing connections: {e}, returning previous probs.')
                    self.distribution.report_failure(id(self), 'processing', reason=f'{e}')

            else:
                print(f'[-] Invalid Choice... returning previous probs.')
                self.distribution.report_failure(id(self), 'processing', reason="InvalidChoice")                        

        elif program == 'N':
            print('|| Skipping Peer connections, returning previous probs')
        else:
            print('[-] Invalid choice! returning previous probs')
            
        return probs



    def mlp_predict(self, X):
        """
                Minimal mlp.MLP-only prediction path: fits/transforms `X` through
                tf-idf if it's raw text, then returns raw mlp.MLP logits (no
                transformer.Transformer, no ensembling, no memory gating).
        """
        if isinstance(X, str) or isinstance(X[0], str):
            self.initialize_fitting(X)            
            X_tfidf = self.tfidf.transform(X).toarray() 

        logits = self.mlp.prediction(X_tfidf)
        return logits

   
    def predict_proba(self, input_ids, X, type='Hybrid', embedded=False):
        """
                Core single/batch probability entry point. For a single item,
                runs the Hybrid path when `type='Hybrid'` and
                `self.use_transformer` is set: gets transformer.Transformer + mlp.MLP
                predictions, checks whether their argmax predictions agree with
                each other (and, if cached probabilities exist, with the cached
                memory's prediction too). On agreement, boosts confidence on the
                agreed class; on disagreement, either asks a peer
                (`_handle_distributed_connections`) or calibrates locally
                (`_calibrate_probs`), depending on autonomy and whether cached
                memory already gives high confidence. Delegates 2D multi-row
                input to `_batch_predict_proba`, and mlp.MLP-only input (no
                `input_ids`) straight to `self.model3`.

                Args:
                    input_ids: Encoded token ids for the transformer.Transformer, or None
                        for an mlp.MLP-only call.
                    X: mlp.MLP feature input.
                    type: `'Hybrid'` to use both models; anything else falls
                        through to mlp.MLP-only behavior for the `input_ids is None`
                        branch.
                    embedded: Forwarded to `self.model2.predict` — whether
                        `input_ids` are already embedded vectors.

                Returns:
                    Probability array, calibrated/boosted as described above.
        """
        eps = 1e-5
        probs_memory = self.model_probability_gate(input_ids, X)
        if isinstance(self.storage.id_history, list) or isinstance(self.storage.id_history, np.ndarray) and not self.agent_id in self.storage.id_history:
            self.storage.id_history.append(self.agent_id)
            id_history = self.storage.id_history
        else:
            self.temporary_id.append(self.agent_id)
            id_history = self.temporary_id

        if input_ids is None and X is not None:
            logits = self.model3.forward(X)
            return self.model3._softmax(logits)

        if input_ids is None and X is None:
            raise Warning('[!] X and input indices are None! cannot proceed with Batching probabilities!')

        if self.use_transformer:
            is_batch = len(input_ids.shape) == 2 and input_ids.shape[0] > 1
        else:
            is_batch = False

        AME = self.AME_Encoder(input_ids)
        AMR = 1.0 / (1.0 + np.exp(-AME))
        
        if is_batch:
            return self._batch_predict_proba(input_ids, X, type) 

        if type == 'Hybrid' and self.use_transformer:
            print('[=] Hybrid based classification method.')
            transformer_pred, probs, attn_weights = self.model2.predict(input_ids, embedded=embedded)
            mlp_pred = self.mlp.forward(X)

            if mlp_pred.ndim == 1:
                mlp_pred = mlp_pred.reshape(1, -1)
            # Ensure transformer_pred is 2D
            if transformer_pred.ndim == 1:
                transformer_pred = transformer_pred.reshape(1, -1)

            mlp_pred_indices = np.argmax(mlp_pred, axis=1)
            trans_pred_indices = np.argmax(transformer_pred, axis=1)
           
            if probs_memory is None:
                agreement = np.allclose(mlp_pred_indices, trans_pred_indices, rtol=eps)
          
            else:
                # memory agreement must match previously detected learned patterns, contextual transformer prediction isnot needed
                try:
                    probs_memory_ = np.argmax(probs_memory) 
                except:
                    probs_memory_= np.argmax(probs_memory, axis=1)

                agreement = np.allclose(mlp_pred_indices, probs_memory_, rtol=eps)

            need_peer_condition = not agreement and probs_memory is None
            self.agreement = agreement

            if not agreement:
                self.peer_assistance_threshold += 0.1                                     
                # if both pattern are still conflicting, use contextual relations for sorting regularization.
                if not self.autonomous and need_peer_condition:
                    print('|| Uncertain prediction, requesting peer assistance if allowed...')
                    probs = self._handle_distributed_connections(probs, attn_weights, input_ids, agreement)
                else:
                    need_calibration_condition = not agreement and self.final_conf_score > self.confidence_threshold
                    if need_calibration_condition:
                        print('[||] Uncertain prediction, but memory exist, skipping peer assistance and calibrating with attention because of high confidence...')                        
                        probs = self._calibrate_probs(probs, mlp_pred_indices, attn_weights, input_ids)       
                    else:
                        print('[-] Uncertain prediction, needing local peer assistance...')                        
                        probs = self._handle_distributed_connections(probs, attn_weights, input_ids, agreement)

            else:
                self.peer_assistance_threshold -= 0.2               
                print('[+] Both Models agree, Normalizing prediction with confidence boost...')
                for i, target in enumerate(mlp_pred_indices):
                    if i < len(probs):
                        probs[i, target] = min(probs[i, target] * 1.2, 0.95)
                        probs[i] /= probs[i].sum()
                    else:
                        probs /= probs.sum()

                    
            if not agreement and probs_memory is not None:
                self.storage.save_peer_needs_dict(self.memory_name, probs_memory, mlp_pred, id_history)   
            else:
                self.storage.save_peer_needs_dict(self.memory_name, probs, mlp_pred, id_history)

            self.modular_probability_saving(input_ids, X, probs)
            print('🚀 Memory Added!')
            return probs
        else:
            print(f'[=] mlp.MLP Based classification method. transformer.Transformer usage permission: {self.use_transformer}')
            logits = self.mlp.forward(X)

            if probs_memory is not None:
                mlp_pred_indices = np.argmax(logits, axis=1)
                probs_memo = np.array(probs_memory)
                try:
                    probs_memory_ = np.argmax(probs_memory) 
                except:
                    probs_memory_ = np.argmax(probs_memory, axis=1)

                agreement = np.allclose(mlp_pred_indices, probs_memory_, rtol=eps)
                
                if not agreement:
                    # if both pattern are still conflicting, used latest prediction
                    probs = logits.copy()       
                else:
                    for i, target in enumerate(mlp_pred_indices):
                        probs_memo[i, target] = min(probs_memo[i, target] * AMR, 0.95)
                        probs_memo[i] /= probs_memo[i].sum()

            return logits

    def _refit_sparse_data(self, X_features, texts, threshold=0.3):
        """Refit TF-IDF if zero-row ratio exceeds threshold."""
        X_features = np.asarray(X_features, dtype=np.float32)
        if X_features.ndim == 1:
            X_features = X_features.reshape(1, -1)        
            X_features = np.asarray(X_features)
                            
        zero_rows = np.where(X_features.sum(axis=1) == 0)[0]
        zero_ratio = len(zero_rows) / len(X_features)
        
        if zero_ratio > threshold:
            print(f'[!] {len(zero_rows)} zero rows ({zero_ratio:.0%}), refitting on current batch')
            if isinstance(texts, str):
                self.tfidf.fit([texts])
                X_features = self.tfidf.transform([texts]).toarray()
            elif isinstance(texts[0], str):
                self.tfidf.fit([texts[0]])
                X_features = self.tfidf.transform([texts[0]]).toarray()
            else:
                X_features = X_features
            
            # second pass — fill remaining zeros with checksum fingerprint
            zero_rows = np.where(X_features.sum(axis=1) == 0)[0]
            for i in zero_rows:
                text = texts[i] if isinstance(texts[i], str) else str(texts[i])
                checksum = int(hashlib.md5(text.encode()).hexdigest(), 16)
                rng = np.random.default_rng(checksum)
                X_features[i] = rng.uniform(0.01, 0.1, size=X_features.shape[1])
                print(f'[!] Row {i} still zero after refit, checksum fallback applied')
        
        return X_features

    def data_preparation(self, titles, labels):
        """
                Zip `titles` and `labels` into the `[(text, label), ...]`
                dataset format most of this pipeline's encoding methods expect,
                and store them on `self.titles` / `self.labels`.
        """
        datasets = []
        raw = []
        for title in titles:
            tupled_title = (str(title))
            datasets.append(tupled_title)
            raw.append(str(title))

        for label in labels:
            tupled_label = (str(label))
            datasets.append(tupled_label)
            raw.append(str(label))

        self.initialize_fitting(raw)
        X_raw = self.tfidf.transform(raw).toarray()

        X_raw = self._refit_sparse_data(X_raw, raw)
        return datasets, X_raw

    
    def _calibrate_probs(self, probs, target_preds, attn_weights, input_ids):
        """
                Nudge `probs` toward each row's mlp.MLP-predicted class
                (`target_preds`) when the transformer.Transformer and mlp.MLP disagreed,
                weighted by `(1 - AME_Encoder(attn_weights))` — i.e. the boost is
                stronger when the attention pattern for that row looks simple/
                confident (low AME) and gentler when it looks complex. Also
                tracks a running `self.temperature` from the per-row abstract
                scores, used by `_softmax` elsewhere. Renormalizes each row
                after adjustment; falls back to a uniform distribution for any
                row whose sum collapses to ~0.

                Args:
                    probs: `(batch, n_classes)` probabilities to adjust.
                    target_preds: Per-row class index to boost toward.
                    attn_weights: Per-row attention tensor(s) fed into
                        `AME_Encoder`; may be None.
                    input_ids: Unused directly here beyond shape bookkeeping —
                        kept for API symmetry with callers.

                Returns:
                    The calibrated, renormalized `probs` array.
        """
        calibrated = probs.copy()
        mlp_target_int = None

        if isinstance(input_ids, list):
            input_ids = np.array(input_ids)
            
        try:
            target_preds = np.asarray(target_preds).ravel().astype(int)
        except (TypeError, ValueError) as e:
            print(f'[!] target_preds could not be coerced to int array: '
                f'type={type(target_preds)} error={e}')
            return calibrated   # return uncalibrated 

        n_classes = probs.shape[1] if probs.ndim > 1 else probs.shape[0]
        batch_size = len(target_preds)
        eps = 1e-5

        temperature_accum = []

        attn_len = len(attn_weights) if attn_weights is not None else 0

        for i in range(batch_size):
            # consistent bound check, no off-by-one
            mlp_target = target_preds[i] if i < attn_len else target_preds[0]

            # anisotropy needs an array-like input
            if attn_weights is None:
                # no attention available 
                anisotropy = eps
            elif i < attn_len:
                anisotropy = self.anisotropy_measurement(attn_weights[i])
            else:
                anisotropy = self.anisotropy_measurement(attn_weights[0])

            if attn_weights is not None and i < attn_len:
                attn = attn_weights[i]
                score_quality = np.std(attn) if attn.size > 0 else self.confidence_threshold
                abstract_score = self.confidence_threshold + score_quality * anisotropy
            else:
                if attn_weights is not None and attn_len > 0:
                    #  use last valid index 
                    fallback_attn = attn_weights[min(i, attn_len - 1)]
                    score_quality = 1.0 / (1.0 + np.exp(-fallback_attn))
                else:
                    score_quality = self.confidence_threshold  # neutral default
                abstract_score = (1.0 - np.mean(score_quality)) + eps

            temp = (1.0 - abstract_score) + score_quality * anisotropy
            if isinstance(temp, np.ndarray):
                temp = float(np.clip(np.mean(temp), 1e-5, 5.0))
            temperature_accum.append(temp)

            try:
                mlp_target_int = int(mlp_target)
            except (TypeError, ValueError) as e:
                print(f'[!] mlp_target coercion failed at i={i}: '
                    f'type={type(mlp_target)} value={mlp_target} error={e}')
                continue

            # also guard against array type slipping through
            if isinstance(mlp_target, np.ndarray):
                if mlp_target.size == 1:
                    mlp_target_int = int(mlp_target.flat[0])
                    print(f'[!] mlp_target was ndarray, extracted scalar: {mlp_target_int}')
                else:
                    print(f'[!] mlp_target was multi-element array {mlp_target.shape} at i={i} — skipping')
                    continue

            # bounds guard before indexing 
            if 0 <= mlp_target < n_classes and i < calibrated.shape[0]:
                if mlp_target_int is None:
                    mlp_target_int = int(mlp_target.flat[0])

                if len(calibrated.shape) >= 2:
                    calibrated[i, mlp_target_int] = min(
                        calibrated[i, mlp_target_int] * (1.5 * (1.0 - abstract_score)), 0.95
                    )
                else:
                    if mlp_target_int < len(calibrated):
                        calibrated[mlp_target_int] = min(
                            calibrated[mlp_target_int] * (1.5 * (1.0 - abstract_score)), 0.95
                        )
                    else:
                        calibrated = min(calibrated * (1.5 * (1.0 - abstract_score)), 0.95)

            if i <= len(calibrated):
                try:
                    row_sum = calibrated[i].sum()
                except IndexError:
                    row_sum = calibrated[0].sum()
            else:
                row_sum = calibrated[0].sum()

            if row_sum > eps:
                if i <= len(calibrated):
                    try:
                        calibrated[i] /= row_sum
                    except:
                        calibrated /= row_sum
                else:
                    calibrated /= row_sum
            else:
                try:
                    calibrated[i] = np.full(n_classes, 1.0 / n_classes)
                except:
                    calibrated = np.full(n_classes, 1.0 / n_classes)

        self.temperature = float(np.mean(temperature_accum)) if temperature_accum else 1.0

        return calibrated
    
    def _softmax(self, x):
        """
                Temperature-scaled softmax using `self.temperature` (as set by
                `_calibrate_probs`). Handles both 1D and 2D input; falls back to
                the 2D broadcasting path if the initial shape branch raises.
        """
        temp = self.temperature

        try:
            if len(x.shape) > 1:
                x_dip = x / temp
                exp_x = np.exp(x_dip - np.max(x_dip, axis=1, keepdims=True))
                softmax = exp_x / np.sum(exp_x, axis=1, keepdims=True)     
            else:
                x_dip = x / temp
                exp_x = np.exp(x_dip - np.max(x_dip))
                softmax = exp_x / np.sum(exp_x)
        except:
            x_dip = x / temp
            exp_x = np.exp(x_dip - np.max(x_dip, axis=1, keepdims=True))


        softmax = exp_x / np.sum(exp_x, axis=1, keepdims=True)     
        return softmax


    def validate_writable_path(self, path):
        """
                Check whether `path` can actually be written to before
                attempting a real save: verifies the parent directory exists
                and is writable, that an existing file at `path` isn't
                read-only, and performs an actual scratch-file write/delete as
                a final confirmation. Returns `(is_writable, message)`.
        """
        try:
            path = os.path.expanduser(path)
        
            directory = os.path.dirname(path) or '.'
        
            if not os.path.exists(directory):
                return False, f"Directory does not exist: {directory}"
        
            if not os.access(directory, os.W_OK):
                return False, f"No write permission for directory: {directory}"
        
            if os.path.exists(path):
                if not os.access(path, os.W_OK):
                    return False, f"File exists but is not writable: {path}"
        
            test_file = os.path.join(directory, f".test_write_{os.getpid()}")
            try:
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
            except Exception as e:
                return False, f"Write test failed: {e}"
        
            return True, "Path is writable"
        
        except Exception as e:
            return False, f"Validation error: {e}"

    def safe_pickle_save_with_feedback(self, data, suggested_path):
        """
                Interactive fallback save path: prompts the user (stdin) for
                where to save `data` via `joblib.dump`, offering `suggested_path`
                as a hint, validating the chosen path with
                `validate_writable_path`, auto-naming the file if a bare
                directory is given, and reporting permission or other save
                errors without raising.
        """
        print("\n" + "="*50)
        if suggested_path:
            print(f"|| Suggested path: {suggested_path}")
        
        user_path = input("|| Enter path to save pickle file (or 'cancel' to skip): ").strip()
        
        if user_path.lower() == 'cancel':
            print("|| Save cancelled.")
            pass
        
        user_path = user_path.strip('"').strip("'")
        user_path = os.path.expanduser(user_path)
        
        if os.path.isdir(user_path):
            from datetime import datetime
            default_filename = f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            user_path = os.path.join(user_path, default_filename)
            print(f"|| Using filename: {default_filename}")
        
        valid, message = self.validate_writable_path(user_path)
        
        if valid:
            try:
                os.makedirs(os.path.dirname(user_path), exist_ok=True)
                
                with open(user_path, 'wb') as f:
                    joblib.dump(data, f)
                
                print(f"✓ Successfully saved to: {user_path}")
                pass
                
            except PermissionError as e:
                print(f"✗ Permission denied: {e}")
                print("|| Try a different location (like your Desktop or Documents folder)")
                pass
            except Exception as e:
                print(f"✗ Save failed: {e}")
                pass

        else:
            print(f"✗ Invalid path: {message}")
            print("Tips:")
            print("  - Use a path in your home directory: ~/Documents/myfile.pkl")
            print("  - Make sure the directory exists and is writable")
            print("  - Try saving to Desktop or Documents folder")
            pass


    def auto_generate_labels_from_texts(self, rules, texts):
        """
                Rule-based auto-labeling: for each text, apply `rules` (a list of
                `(regex_pattern, label)` pairs) in order and assign the first
                matching label, or `'other'` if nothing matches. Prints the
                resulting label distribution for a quick sanity check.
        """
        y_raw = []
        self.rules = rules

        for text in texts:
            text_lower = text.lower()
            matched = False
            for pattern, label in rules:
                if re.search(pattern, text_lower):
                    y_raw.append(label)
                    matched = True
                    break
            if not matched:
                y_raw.append('other')

        print("\n[📊] Auto-generated label distribution:")
        for label, count in sorted(Counter(y_raw).items()):
            print(f"   {label}: {count} ({count/len(texts)*100:.1f}%)")

        return y_raw


    def mlp_training_features(self, rules, dataset, label_map=None):
        """
                Normalize a `dataset` in any of three supported shapes —
                `[(features, label), ...]`, `[[feat1, feat2, ..., label], ...]`,
                or raw texts needing rule-based auto-labeling
                (`auto_generate_labels_from_texts`) — into mlp.MLP-ready
                `(X_mlp, y_indices, n_classes, input_dim)`.

                When `label_map` is given, class indices are taken from that
                GLOBAL map (extending it in place for any genuinely new label)
                instead of being re-derived from just this batch, so index
                `2` always means the same class across calls. Also reports any
                classes with fewer than 2 samples (too few to stratify/validate
                on downstream).

                Returns label-encoded `y_indices` (not one-hot) — callers that
                need one-hot convert it themselves after any train/val split,
                once `n_classes` is fixed.
        """
        print("\n[🔄] Preparing mlp.MLP data from dataset format")

        if isinstance(dataset[0], tuple) and len(dataset[0]) == 2:
            print('[=] Dataset Type 1: [(features, label), ...]')
            features_list, labels_list = [], []
            for features, label in dataset:
                features_list.append(features)
                labels_list.append(label)
            X_mlp = np.array(features_list)
            y_raw = np.array(labels_list)

        elif isinstance(dataset[0], (list, np.ndarray)) and len(dataset[0]) > 1:
            print('[=] Dataset Type 2: [feature1, feature2, ..., label]')
            texts  = [item[:-1] for item in dataset]
            labels = [item[-1]  for item in dataset]
            X_mlp = np.array(texts)
            y_raw = np.array(labels)

        else:
            print('[=] Dataset type 3: raw texts, auto-labeling via rules')
            X_mlp = dataset.copy()
            y_raw = self.auto_generate_labels_from_texts(rules, dataset)

        # Use the GLOBAL label map if given, instead of a batch-local one.
        # This keeps class indices consistent across every call, no matter
        # which classes happen to be present in this particular batch.
        if label_map is not None:
            label_to_idx = label_map
            unknown = set(y_raw) - set(label_to_idx.keys())
            if unknown:
                print(f'[⚠️] mlp_training_features: labels {sorted(unknown)} not in '
                    f'label_map — assigning new indices at the end')
                next_idx = max(label_to_idx.values()) + 1
                for lbl in sorted(unknown):
                    label_to_idx[lbl] = next_idx
                    next_idx += 1
        else:
            unique_labels = sorted(set(y_raw))
            label_to_idx  = {l: i for i, l in enumerate(unique_labels)}

        n_classes = len(label_to_idx)
        y_indices = np.array([label_to_idx[l] for l in y_raw])

        # Report scarcity HERE, at generation time — not three calls downstream.
        idx_counts = np.bincount(y_indices, minlength=n_classes)
        scarce = [ (i, c) for i, c in enumerate(idx_counts) if 0 < c < 2 ]
        if scarce:
            reverse = {i: l for l, i in label_to_idx.items()}
            print(f'[⚠️] mlp_training_features: classes with <2 samples: '
                f'{[(reverse[i], c) for i, c in scarce]} — these cannot be '
                f'stratified/validated on downstream. Consider collecting more '
                f'data for these classes, or merging them into a broader bucket.')

        if isinstance(X_mlp, np.ndarray) and X_mlp.ndim > 1:
            input_dim = X_mlp.shape[1]
        elif isinstance(X_mlp, np.ndarray):
            input_dim = 1
        else:
            input_dim = len(X_mlp[0]) if len(X_mlp) > 0 else 0

        print(f"\n✅ mlp.MLP data ready:")
        print(f"[=] X shape: {X_mlp.shape if isinstance(X_mlp, np.ndarray) else len(X_mlp)}")
        print(f"[=] input_dim: {input_dim}")
        print(f"[=] y shape: {y_indices.shape} (label-encoded, {n_classes} classes)")
        print(f"[=] Classes: {label_to_idx}")

        # Return label-encoded y, not one-hot. Callers that need one-hot
        # (e.g. for a loss function) convert AFTER splitting, with n_classes fixed.
        return X_mlp, y_indices, n_classes, input_dim


    def shape_adaptation(self, X, target_features):
        """
        Adapts X's FEATURE dimension (columns) to match target_features.
        Sample count (rows) is never touched — only column width changes.

        Args:
            X: (n_samples, n_features) array
            target_features: desired number of feature columns
        """
        try:
            if X.ndim == 1:
                X = X.reshape(1, -1)

            n_samples, n_features = X.shape

            if n_features == target_features:
                return X

            print(f'[⚠️] shape_adaptation: X has {n_features} features, '
                f'target is {target_features} — adapting columns only '
                f'(rows={n_samples} unchanged)')

            X_adapted = np.zeros((n_samples, target_features))
            min_features = min(n_features, target_features)
            X_adapted[:, :min_features] = X[:, :min_features]

            if n_features > target_features:
                print(f'[⚠️] shape_adaptation: TRUNCATED {n_features - target_features} '
                    f'feature columns ({n_features} → {target_features})')
            else:
                print(f'[=] shape_adaptation: PADDED {target_features - n_features} '
                    f'feature columns with zeros')
        except Exception as e:
            print(f'[-] Fallback to primitive shape adaptation due to {e}')
            print(f'[!] WARNING: This may pad shapes aggressively! ')
            inp = X.shape[1]
            tuple_ver = (inp, inp)
            if X.shape != tuple_ver:
                X = X[:inp, :inp]            

        return X_adapted



    def _safe_to_2d(self, x) -> np.ndarray:
        """
        Convert ANY input shape to a well-formed 2D float64 array.
        Handles: ragged lists, 1D arrays, 4D attention tensors,
                string arrays, scalar inputs, None.

        Strategy: PAD to max length.
        """
        if x is None:
            return None

        # string input — parse to numeric first
        if isinstance(x, (str, np.str_)):
            clean = x.replace('[', '').replace(']', '').strip()
            try:
                X = np.fromstring(clean, sep=' ', dtype=np.float64)
                return X.reshape(1, -1) if X.size > 0 else None
            except ValueError:
                return None

        # try direct conversion first — works for homogeneous inputs
        try:
            X = np.asarray(x, dtype=np.float64)
        except ValueError:
            # inhomogeneous shape — pad to max length
            X = self._pad_ragged_to_array(x)
            if X is None:
                return None

        # handle string dtype arrays
        if np.issubdtype(X.dtype, np.character):
            clean = ' '.join(X.astype(str).flatten()).replace('[', '').replace(']', '')
            try:
                vals = [float(v) for v in clean.split() if v != '...']
                X    = np.array(vals, dtype=np.float64)
            except ValueError:
                return np.asarray(x, dtype=object)

        # normalize to exactly 2D
        X = np.squeeze(X)

        if X.ndim == 0:
            return np.array([[float(X)]])

        if X.ndim == 1:
            return X.reshape(1, -1)

        if X.ndim == 2:
            # even-dimension crop (your existing logic, preserved)
            rows, cols = X.shape
            new_rows = rows - 1 if (rows % 2 != 0 and rows > 1) else rows
            new_cols = cols - 1 if (cols % 2 != 0 and cols > 1) else cols
            return X[:new_rows, :new_cols].astype(np.float64)

        if X.ndim > 2:
            # attention weights (B,H,T,T) or similar — flatten to (B, H*T*T)
            # preserves batch structure while giving AME a sensible 2D view
            return X.reshape(X.shape[0], -1).astype(np.float64)

        return np.asarray(x, dtype=object)


    def _pad_ragged_to_array(self, x) -> np.ndarray:
        """
        Convert a ragged list-of-lists to a 2D array by padding shorter
        rows with zeros to match the longest row.
        Preserves per-row geometric structure for gradient computation.
        """
        try:
            rows = [np.asarray(item, dtype=np.float64).ravel()
                    for item in x]
        except (TypeError, ValueError):
            return np.asarray(x, dtype=object)

        if not rows:
            return None

        max_len = max(len(r) for r in rows)
        if max_len == 0:
            return None

        padded = np.zeros((len(rows), max_len), dtype=np.float64)
        for i, row in enumerate(rows):
            padded[i, :len(row)] = row

        return padded




    def _coerce_to_2d_float(self, x) -> np.ndarray:
        """
        NumPy 1.24+ raises ValueError on ragged asarray() without dtype=object.
        This method handles that cross-platform consistently.
        """
        if x is None:
            return None

        if isinstance(x, np.ndarray):
            return self._normalize_to_2d(x.astype(np.float64, copy=False))

        # list/tuple — check for raggedness 
        if isinstance(x, (list, tuple)) and len(x) > 0:
            first = x[0]

            # check if items are sequences of potentially different lengths here
            if isinstance(first, (list, tuple, np.ndarray)):
                lengths = set()
                for item in x:
                    if isinstance(item, np.ndarray):
                        lengths.add(item.size)
                    elif isinstance(item, (list, tuple)):
                        lengths.add(len(item))
                    else:
                        lengths.add(1)

                if len(lengths) > 1:
                    # RAGGED — pad to uniform length.
                    max_len = max(lengths)
                    print(f'[=] AME_Encoder: ragged input detected '
                        f'(lengths: {lengths}) — padding to {max_len}')
                    padded = np.zeros((len(x), max_len), dtype=np.float64)
                    for i, item in enumerate(x):
                        arr = np.asarray(item, dtype=np.float64).ravel()
                        padded[i, :len(arr)] = arr
                    return padded

        # homogeneous — safe to call asarray directly
        try:
            X = np.asarray(x, dtype=np.float64)
            return self._normalize_to_2d(X)
        except ValueError:
            # last resort — try with dtype=object then convert
            try:
                X = np.asarray(x, dtype=object)
                flat = np.array([
                    float(v) for v in X.ravel()
                    if v is not None
                ], dtype=np.float64)
                return flat.reshape(1, -1)
            except Exception:
                print('[!] Cant Process and Convert X samples!')
                return np.asarray(x, dtype=object)

    def _safe_to_2d_float(self, x) -> np.ndarray:
        """
        Attempt direct numpy conversion first (fast path for normal inputs).
        Only activates ragged/string handling when direct conversion fails.
        This avoids over-inspection that returns None for valid inputs.
        """
        try:
            if x is None:
                raise Warning('[!] X samples is None!')
                return np.asarray(x, dtype=object)

            # FAST PATH — already a numeric numpy array, most common case
            if isinstance(x, np.ndarray):
                if np.issubdtype(x.dtype, np.floating) or \
                np.issubdtype(x.dtype, np.integer):
                    return self._normalize_to_2d(x.astype(np.float64, copy=False))
                # non-numeric numpy array — string/object dtype
                return self._safe_to_2d(x)

            # ATTEMPT direct conversion — works for all homogeneous inputs
            # including: flat lists, lists of same-length arrays, 2D lists, etc.
            try:
                X = np.asarray(x, dtype=np.float64)
                return self._normalize_to_2d(X)

            except ValueError:
                # NumPy 1.24+ ragged array — activate padding path
                print(f'[=] _to_2d_float: ragged input detected — padding to uniform shape')
                return self._coerce_to_2d_float(x)

            except TypeError:
                # string content or incompatible types — activate safe path
                print(f'[=] _to_2d_float: type conversion failed — trying safe path')
                return self._safe_to_2d(x)

        except Exception as e:
            print(f'[!] _to_2d_float: unexpected error: {e} — using Robust method')
            try:
                result = self._convert_to_2d_float(x)
            except Exception as e:
                print('[!] cant Convert and calculate samples! - Using dtype object to compensate for the failure')
                return np.asarray(x, dtype=object)


    def _normalize_to_2d(self, X: np.ndarray) -> np.ndarray:
        """Squeeze and reshape any array to exactly 2D."""
        X = np.squeeze(X)
        if X.ndim == 0:
            return np.array([[float(X)]])
        if X.ndim == 1:
            return X.reshape(1, -1)
        if X.ndim == 2:
            rows, cols = X.shape
            new_rows = rows - 1 if (rows % 2 != 0 and rows > 1) else rows
            new_cols = cols - 1 if (cols % 2 != 0 and cols > 1) else cols
            return X[:new_rows, :new_cols]
        if X.ndim > 2:
            return X.reshape(X.shape[0], -1)
        return X

    def _convert_to_2d_float(self, x) -> np.ndarray:
        """
        Adaptive gate — detects input characteristics and routes to
        the appropriate normalization path:
        
        _coerce_to_2d_float  → when input is ragged (variable-length sequences)
                            or when NumPy 1.24+ would raise ValueError
        _safe_to_2d          → when input has string/object dtype, nested
                            structures, or needs deep type conversion
        direct numpy         → when input is already a clean numeric array
        """
        if x is None:
            raise Warning('[!] X samples is None!')
            return None

        # FAST PATH — already a clean numpy float array, skip both functions
        if isinstance(x, np.ndarray):
            if np.issubdtype(x.dtype, np.floating) or np.issubdtype(x.dtype, np.integer):
                return self._normalize_to_2d(x.astype(np.float64, copy=False))
            # non-numeric numpy array → needs _safe_to_2d for string/object handling
            return self._safe_to_2d(x)

        # string input - always _safe_to_2d (handles parsing logic)
        if isinstance(x, (str, np.str_)):
            return self._safe_to_2d(x)

        # list/tuple — detect ragged vs homogeneous vs string content
        if isinstance(x, (list, tuple)) and len(x) > 0:
            first = x[0]

            # strings inside list → _safe_to_2d
            if isinstance(first, (str, np.str_)):
                return self._safe_to_2d(x)

            # nested sequences — check for raggedness
            if isinstance(first, (list, tuple, np.ndarray)):
                lengths = set()
                has_strings = False

                for item in x:
                    if isinstance(item, (str, np.str_)):
                        has_strings = True
                        break
                    elif isinstance(item, np.ndarray):
                        lengths.add(item.size)
                        if np.issubdtype(item.dtype, np.character):
                            has_strings = True
                            break
                    elif isinstance(item, (list, tuple)):
                        lengths.add(len(item))

                # string content anywhere → _safe_to_2d
                if has_strings:
                    return self._safe_to_2d(x)

                # ragged numeric - _coerce_to_2d_float (pads to uniform length)
                if len(lengths) > 1:
                    return self._coerce_to_2d_float(x)

                # homogeneous numeric - direct numpy, fastest path
                try:
                    X = np.asarray(x, dtype=np.float64)
                    return self._normalize_to_2d(X)
                except ValueError:
                    # for edge case — fall back to coerce
                    return self._coerce_to_2d_float(x)

            # flat list of scalars 
            try:
                X = np.asarray(x, dtype=np.float64)
                return self._normalize_to_2d(X)
            except (ValueError, TypeError):
                return self._safe_to_2d(x)

        # scalar input
        try:
            return np.array([[float(x)]])
        except (TypeError, ValueError):
            print('[!] Cant normalize and Convert X Samples! - Using dtype object to compensate failure!')
            return np.asarray(x, dtype=object)

    def _safe_convert(self, x):
        """
                Lighter-weight 2D-float coercion used on the AME/anisotropy
                path: parses stringified/character-dtype arrays, squeezes to
                2D, and applies the same even-dimension cropping as
                `_safe_to_2d` — but without that method's ragged-list padding or
                higher-dimensional flattening, so it's only suitable for inputs
                that are already numeric-ish and at most 2D.
        """
        X = np.asarray(x)

        if isinstance(X, (str, np.str_)):
            clean_str = str(X).replace('[', '').replace(']', '')
            X = np.fromstring(clean_str, sep=' ')
        if isinstance(X, np.ndarray) and np.issubdtype(X.dtype, np.character):
            # catches arrays filled with string text
            clean_str = ' '.join(X.astype(str).flatten()).replace('[', '').replace(']', '')
            X = np.fromiter(
                    (x for x in clean_str.split() if x != "..."), dtype=float
                )   

        X = np.squeeze(X)
        if X.ndim == 1:
            X = np.atleast_2d(X).T  

        # Handle cropping for a 2D array
        if X.ndim == 2:
            rows, cols = X.shape
            new_rows = rows - 1 if (rows % 2 != 0 and rows > 1) else rows
            new_cols = cols - 1 if (cols % 2 != 0 and cols > 1) else cols
            
            X = X[:new_rows, :new_cols]

        return X 

    def AME_Encoder(self, x):
        """
                "Abstract Modelling Energy" — a scalar complexity score for `x`
                combining its magnitude and its local gradient energy
                (`log1p(|x|) * log1p(|grad(x)|)`), used as a proxy for how hard
                an input is to model confidently. Prefers the optimized native
                implementation (`optimized_ame_encoder`) when available. Falls
                back to `(1 - confidence_threshold) + eps` on empty input, a
                near-zero result, or any computation error, so it never returns
                exactly 0 (callers use it as a divisor/activations.sigmoid input elsewhere).
        """
        eps = 1e-5

        try:
            try:
                X = self._safe_convert(x)
            except:
                X = self._safe_to_2d_float(x)
                
            if _OPT_AVAILABLE and np.asarray(X).ndim == 2:
                AME = optimized_ame_encoder(np.asarray(X, dtype=np.float64))  
                if isinstance(AME, (list, np.ndarray, tuple)):
                    AME = np.mean(AME)
                    
                return AME   

            if len(X) == 0:
                print('[!] X size is 0, AME Will be replaced by minimum confidence threshold')
                return self.pipeline.confidence_threshold

            if len(X.shape) > 1 and X.shape[1] == 1:
                gradient = np.gradient(X, axis=0)  # Calculate vertically instead of horizontally
            else:
                gradient = np.gradient(X) 
            
            if len(X.shape) > 1 and X.shape[1] == 1 and len(gradient.shape) > 1 and gradient.shape[1] == 1:
                grad_energy = np.mean(np.linalg.norm(gradient, axis=-1))       
                X_mag = np.mean(np.linalg.norm(X, axis=-1))
            else:
                grad_energy = np.mean(np.linalg.norm(gradient))       
                X_mag = np.mean(np.linalg.norm(X))

            AME = np.log1p(X_mag) * np.log1p(grad_energy) 

            if AME <= eps:
                AME = (1.0 - self.confidence_threshold) + eps
            
        except Exception as e:
            print(f'[!] Cant calculate AME from samples due to: {e}, using normalized value...')
            AME = (1.0 - self.confidence_threshold) + eps

        return AME

    def feature_generation(self, rules, dataset):
        """
                End-to-end mlp.MLP feature pipeline: normalize `dataset` via
                `mlp_training_features`, fit/transform through tf-idf, and
                adapt the resulting feature width to match `input_dim`
                (`shape_adaptation`).

                Returns:
                    `(X, y, input_dim, n_classes)` ready for `model3.train`.
        """
        X_raw, y, n_classes, input_dim = self.mlp_training_features(rules, dataset)
            
        self.initialize_fitting(X_raw)            
        X_tfidf = self.tfidf.transform(X_raw).toarray()
        X = X_tfidf.copy() 

        X = self.shape_adaptation(X, input_dim)  

        return X, y, input_dim, n_classes  

    def _set_lstm_samples(self, X, Y, min_samples_for_split=10, use_cache_augmentation=True):
        """
        Reshape X, Y for LSTM input. If sample count is too small for a
        meaningful train/val split, augment with verified entries from
        caching_security.AccurateAnswerCache before reshaping.
        """
        try:
            X = np.array(X)
            Y = np.array(Y)

            # augment from accurate_cache before reshaping.
            if use_cache_augmentation and len(X) < min_samples_for_split:
                if hasattr(self, 'accurate_cache') and self.accurate_cache_lookup.cache:
                    print(f'[=] Only {len(X)} samples — augmenting from accurate_cache '
                        f'(has {len(self.accurate_cache_lookup.cache)} verified entries)')

                    cached_X, cached_Y = self._extract_cache_samples_for_lstm(
                        target_count=min_samples_for_split - len(X)
                    )

                    if len(cached_X) > 0:
                        X = np.concatenate([X, cached_X], axis=0)
                        Y = np.concatenate([Y, cached_Y], axis=0)
                        print(f'[=] Augmented to {len(X)} samples using '
                            f'{len(cached_X)} verified cache entries')
                    else:
                        print('[=] No suitable cache entries found for augmentation')

            X = X[..., np.newaxis]
            Y = Y[..., np.newaxis]
        except Exception as e:
            print(f'[!] Error in seeting LSTM Samples: {e}, filling gaps with regular newaxis to populate data.')
            X = X = np.array(X)[..., np.newaxis]
            Y = np.array(Y)[..., np.newaxis]

        print('[=] Successfully set up LSTM Samples:')
        print(f'[=] X.shape: {X.shape}')
        print(f'[=] Y.shape: {Y.shape}')

        return X, Y

    def _sync_vocab_to_embedding(self):
        """Extend token_embedding rows if vocab has grown since model creation.
        Preserves all existing trained weights — only appends new rows.
        """
        if self.model2 is None:
            return
        current_emb_size = self.model2.token_embedding.shape[0]
        current_vocab_size = len(self.vocab)
        if current_vocab_size > current_emb_size:
            d_model = self.model2.d_model
            n_new = current_vocab_size - current_emb_size
            new_rows = np.random.randn(n_new, d_model) * 0.02
            self.model2.token_embedding = np.vstack([
                self.model2.token_embedding, new_rows
            ])
            print(f'[=] token embedding grown: {current_emb_size} → {current_vocab_size}')


    def _extract_cache_samples_for_lstm(self, target_count):
        """
        Pull verified-correct entries from caching_security.AccurateAnswerCache to use as
        additional LSTM training samples — only entries with multiple
        confirmed hits (hit_count >= 1) to avoid using unverified noise.
        """
        if not hasattr(self, 'accurate_cache') or not self.accurate_cache_lookup.cache:
            return np.array([]), np.array([])

        candidates = [
            entry for entry in self.accurate_cache_lookup.cache.values()
            if entry.get('hit_count', 0) >= 1   # only entries confirmed at least once
        ]

        # prioritize highest-confidence, most-confirmed entries first
        candidates.sort(key=lambda e: (e['hit_count'], e['confidence']), reverse=True)
        selected = candidates[:target_count]

        if not selected:
            return np.array([]), np.array([])

        cached_X = np.array([e['x_mlp'] for e in selected])
        cached_Y = np.array([
            e['prediction'] if isinstance(e['prediction'], (int, float))
            else e['confidence']   # fallback if prediction is a label string
            for e in selected
        ])

        return cached_X, cached_Y



    def _get_last_scalar(self, arr):
        """
        Extract the scalar value from the LAST timestep, robust to
        whether arr is 1D, 2D, or 3D — avoids hardcoding an index
        pattern that only works for one specific shape.
        """
        arr = np.asarray(arr)

        if arr.ndim == 0:
            return float(arr)
        if arr.ndim == 1:
            return float(arr[-1])          # (T,) → last timestep value
        if arr.ndim == 2:
            return float(arr[-1, 0])       # (T, 1) or (T, features) → last timestep, first feature
        if arr.ndim >= 3:
            return float(arr[-1, 0, 0])    # (T, features, 1) style → drill down consistently

        return float(np.asarray(arr).flat[-1])  # ultimate fallback, never crashes


    def _get_ensemble_confidence_for_true_class(self, X, Y, input_ids=None, eps=1e-3):
        """
        Computes, per sample, how confident the mlp.MLP + transformer.Transformer ensemble
        is in that sample's OWN true class — used as a genuine continuous
        LSTM training target instead of the raw class index.

        X        : hybrid_X after shape_adaptation — same format self.mlp
                    (self.model3) and self.model3.train() already consume.
        Y         : raw class indices, shape (n_samples,) or (n_samples, 1)
        input_ids : sequence_inputs — same format self.model2 (transformer.Transformer)
                    consumes with embedded=True. Optional; mlp.MLP-only
                    confidence is used if not provided or transformer.Transformer
                    is unavailable.

        Returns : (n_samples,) float64 array, each value in (eps, 1.0]
                    — never exactly 0, to avoid degenerate regression
                    targets downstream (consistent with the eps-floor
                    pattern used elsewhere in this codebase).
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        Y = np.asarray(Y).reshape(-1).astype(int)
        n_samples = X.shape[0]

        if len(Y) != n_samples:
            print(f'[⚠️] _get_ensemble_confidence_for_true_class: '
                f'X has {n_samples} samples but Y has {len(Y)} — '
                f'aligning to the shorter length')
            n_samples = min(n_samples, len(Y))
            X = X[:n_samples]
            Y = Y[:n_samples]

        confidences = np.full(n_samples, eps, dtype=np.float64)

        # ── mlp.MLP confidence, batched ──────────────────────────────
        mlp_probs = None
        if hasattr(self, 'mlp') and self.mlp is not None:
            try:
                if X.shape != self.model3.layers[0].W.shape:
                    X = np.reshape(X, (n_samples, self.model3.layers[0].W.shape[0]))
        
                mlp_probs = np.asarray(self.model3.forward(X, y=Y), dtype=np.float64)
                if mlp_probs.ndim == 1:
                    mlp_probs = mlp_probs.reshape(1, -1)
            except Exception as e:
                print(f'[⚠️] mlp.MLP forward failed in ensemble confidence: {e} '
                    f'— mlp.MLP contribution will be skipped')
                mlp_probs = None

        # ── transformer.Transformer confidence, batched ──────────────────────
        trans_probs = None
        if (getattr(self, 'use_transformer', False) and
                hasattr(self, 'model2') and self.model2 is not None and
                input_ids is not None):
            try:
                input_ids_arr = np.asarray(input_ids)
                if len(input_ids_arr) != n_samples:
                    print(f'[⚠️] input_ids length ({len(input_ids_arr)}) != '
                        f'n_samples ({n_samples}) for ensemble confidence '
                        f'— transformer.Transformer contribution will be skipped')
                else:
                    fwd_result = self.model2.forward(input_ids_arr, embedded=True)
                    # forward may return (probs, attn_weights) or just probs
                    trans_probs = fwd_result[0] if isinstance(fwd_result, tuple) else fwd_result
                    trans_probs = np.asarray(trans_probs, dtype=np.float64)
                    if trans_probs.ndim == 1:
                        trans_probs = trans_probs.reshape(1, -1)
            except Exception as e:
                print(f'[⚠️] transformer.Transformer forward failed in ensemble confidence: '
                    f'{e} — transformer.Transformer contribution will be skipped')
                trans_probs = None

        if mlp_probs is None and trans_probs is None:
            print('[⚠️] Neither mlp.MLP nor transformer.Transformer produced usable probs — '
                'falling back to normalized raw class index as target')
            n_classes = self._get_num_classes() or (int(Y.max()) + 1)
            return np.clip(Y.astype(np.float64) / max(n_classes - 1, 1), eps, 1.0)

        # ── combine per-sample, with per-model class-bound safety ─
        for i in range(n_samples):
            true_class = int(Y[i])
            sample_confs = []

            if mlp_probs is not None:
                n_cls = mlp_probs.shape[1]
                if 0 <= true_class < n_cls:
                    sample_confs.append(float(mlp_probs[i, true_class]))
                else:
                    print(f'[⚠️] Sample {i}: true_class={true_class} out of '
                        f'range for mlp.MLP output width {n_cls} — skipping Most '
                        f'mlp.MLP contribution for this sample')
                    sample_confs.append(float(mlp_probs[i, n_cls - 1]))  

            if trans_probs is not None:
                n_cls = trans_probs.shape[1]
                if 0 <= true_class < n_cls:
                    sample_confs.append(float(trans_probs[i, true_class]))
                else:
                    print(f'[⚠️] Sample {i}: true_class={true_class} out of '
                        f'range for transformer.Transformer output width {n_cls} — '
                        f'skipping Most transformer.Transformer contribution for this sample')
                    sample_confs.append(float(trans_probs[i, n_cls - 1]))

            if sample_confs:
                confidences[i] = max(np.mean(sample_confs), eps)
            # else: stays at eps floor — explicit, visible fallback, not silent 0

        return confidences


    def lstm_setup_inference(self, raw_X, raw_Y, input_ids=None):
        """
                Prepare and train/calibrate the LSTM branch (`self.lstm_engine`)
                on top of the mlp.MLP/transformer.Transformer outputs.

                Builds LSTM-ready samples (`_set_lstm_samples`, optionally
                augmented from `caching_security.AccurateAnswerCache` when the input looks
                complex enough via `AME_Encoder`/AMR and peer assistance isn't
                already elevated), computes a continuous ensemble-confidence
                training target per sample
                (`_get_ensemble_confidence_for_true_class`) rather than training
                on raw class indices, splits 80/20 into train/val, and fits a
                fresh `lstm.LSTMEngine` calibrated against that validation split.
                Also derives descriptive confidence bins (Low/Moderate/High/
                Extreme) from the validation targets via
                `derive_bins_from_data`, for later human-readable reporting.
        """
        print("\n" + "=" * 55)
        print("===== LSTM SETUP INFERENCE =====")
        print('[=] LSTM Setup is initiated for Longer short term memory.')

        scaler_y = self.standard_scaler 
        scrapper_engine = self.scrapper_model

        # build dataset for calibration
        AME = self.AME_Encoder(raw_X)  # geometric complexity scalar
        AMR = 1.0 / (1.0 + np.exp(-AME))  # abstract modelling rate

        augmentation = AMR > self.confidence_threshold and self.peer_assistance_threshold < 0.15 
        X, Y = self._set_lstm_samples(raw_X, raw_Y, use_cache_augmentation=augmentation)

        Y = np.asarray(Y)
        if self.use_transformer:
            ensemble_confidence = self._get_ensemble_confidence_for_true_class(X, Y, input_ids)
        else:
            ensemble_confidence = Y.copy()  # fallback to raw Y if no transformer

        n_train = int(0.8 * len(X)) # 80% of the data training is used for training
        X_val   = X[n_train:]
        Y_val   = ensemble_confidence[n_train:]

        X_train_full = X[:n_train]  # this should go to fit_stm
        Y_train_full = ensemble_confidence[:n_train]

        n_display = len(X_val)
        if n_display > 3:
            n_display = 3

        print('[= FIT =] Fitting Short Term Memory...')
        n_samples = scrapper_engine.lstm_optimal_samples(scrapper_engine, X_val[0])
        self.lstm_n_samples = n_samples

        label_bins = scrapper_engine.derive_bins_from_data(
            Y_val.ravel(),
            n_bins=4,
            labels=["Low", "Moderate", "High", "Extreme"],
            value_range=(0.0, 1.0)
        )
        # {"Low": (5.0, 42.3), "Moderate": (42.3, 68.1),
        #  "High": (68.1, 98.7), "Extreme": (98.7, 219.0)}

        # build and calibrate engine
        self.lstm_engine = lstm.LSTMEngine(self, self.network_model, 
                                       dropout=self.dropout_rate, n_samples=self.lstm_n_samples)
        
        engine = self.lstm_engine

        self.lstm_engine.fit_stm(X_train_full, Y_train_full, epochs=self.lstm_training_epochs, hidden=self.lstm_hidden_dim, lr=self.lstm_lr)
        engine.calibrate(X_val, Y_val)

        print("\n[= LSTM INSIGHT =] Per-sample confidence report:")
        print(f"[*] {'#':<4} {'Predicted':>10} {'Actual':>10} "
            f"[*] {'Confidence':>12} {'Gate Uncert':>13} "
            f"[*] {'90% Interval':>20}  Label Confidence")
        print("  " + "─" * 100)

        # n display to see the prediction result over specific batch
        for j in range(n_display):
            result = engine.predict(X_val[j], label_bins=label_bins)

            p      = result["prediction"][-1]
            actual = self._get_last_scalar(Y_val[j])
            conf   = result["overall"]
            gate_u = result["gate_uncertainty"][-1]
            lo     = result["interval_low"][-1]
            hi     = result["interval_high"][-1]
            lc     = result["label_confidence"]

            best_label = max(lc, key=lc.get)
            label_str  = "  ".join(f"{k}={v:.0%}" for k,v in lc.items())

            print(f"  {j:<4} {p:>+10.4f} {actual:>+10.4f} "
                f"{conf:>12.1%} {gate_u:>13.3f} "
                f"  [{lo:+.3f}, {hi:+.3f}]"
                f"  {label_str}")

        print('[=+=] ==== STATUS REPORT ====')
        print("\n[=] Confidence breakdown for sample 0:")
        r = engine.predict(X_val[0], label_bins=label_bins)

        print(f"    MC mean (last step)   : {r['mc_mean'][-1]:+.4f}")
        print(f"    MC std  (last step)   : {r['mc_std'][-1]:.4f}  "
            f"← tight = certain")
        print(f"    Gate uncertainty      : {r['gate_uncertainty'][-1]:.4f}  "
            f"← low = stable memory")
        print(f"    MC confidence         : {r['mc_confidence'][-1]:.1%}")
        print(f"    Overall confidence    : {r['overall']:.1%}")
        print(f"    90% interval          : "
            f"[{r['interval_low'][-1]:+.4f}, {r['interval_high'][-1]:+.4f}]")
        print(f"    Label confidence      : {r['label_confidence']}")

        self.cache['lstm_result'] = result
        self.cache['label_bins'] = label_bins


    # necessary functions to reduce wasteful training in similar scarce environment
    def training_necessary_condition(self, input_ids, x):
        """
                Decide whether there's enough usable signal in `input_ids`/`x` to
                even attempt training this round (as opposed to
                `_should_train_transformer`, which is transformer.Transformer-specific and
                factors in RAM/anisotropy thresholds too). Used as a cheap
                upfront gate before the heavier training path runs.
        """
        eps = 1e-5
        final_conf = self.final_conf_score
        confidence_threshold = self.confidence_threshold
        unsuitable_training = False

        probs = self.model_memory_gate(input_ids, x)
        cache = self.accurate_cache_lookup.lookup(
                x_mlp=x, 
                input_ids=input_ids)
        cached = cache is not None and cache['similarity'] >= 0.95

        anisotropy = self.anisotropy_measurement(input_ids)
        AME = self.AME_Encoder(input_ids)
        AMR = 1.0 / (1.0 + np.exp(-AME)) 

        ALR = 1.0 / (1.0 + np.exp(-anisotropy)) # anisotropic logistic rate
        AAC = (1.0 - ALR) * AMR + eps # anisotropic abstract coefficient

        self.confidence_threshold = anisotropy * AAC 
        if np.isnan(self.confidence_threshold) or np.isinf(self.confidence_threshold):
            self.confidence_threshold = 0.45

        print(f'[||] Confidence threshold set to: {self.confidence_threshold} || Final Confidence Score: {final_conf}')

        # training is wasteful if the model processes little abstraction divergence without necessary context on a similar environment
        # AMR is guaranteed to give sufficient ratio on how modelling error error could be sufficient enough to guarantee the model successful training 
        # (not too high that it shows unstability, not too low that it shows rigidity), high anisotropy correlates to a much complex non linearity that the model will have a hard time adjusting
        # Too high AAC means the model is likely to be in a regime where training could lead to overfitting or divergence due to insufficient modelling capacity relative to the complexity of the data, especially if the confidence score is also low, indicating that the model is not currently confident in its predictions and may not benefit from further training on this data.
        unsuitable_tolerance = probs is not None and cached or AAC > 0.75
        unsuitable_conditions = anisotropy > 0.85 or final_conf > confidence_threshold or self.freeze_learning
        unsuitable_peer_request = probs is not None and self.peer_assistance_threshold > self.confidence_threshold

        self.unsuitable_tolerance = unsuitable_tolerance
        self.unsuitable_conditions = unsuitable_conditions
        self.unsuitable_peer_request  = unsuitable_peer_request

        if self.unsuitable_tolerance or self.unsuitable_conditions or self.unsuitable_peer_request:
            print(f'[==] Unsuitable training condition detected! Tolerance: {unsuitable_tolerance} || Unsuitable Conditions: {unsuitable_conditions}')
            print(f'[==] Peer assistance condition: {unsuitable_peer_request} || Peer assistance threshold: {self.peer_assistance_threshold}')
            unsuitable_training = True

        print('== Training Condition evaluation == ')
        print(f'[==] Unsuitable training condition Evaluation || Unsuitable Tolerance: {unsuitable_tolerance} || Unsuitable Conditions: {unsuitable_conditions} || Unsuitable Peer Assistance: {unsuitable_peer_request}')
        print('[==] Final Decision on Training: ' + ('Unsuitable' if unsuitable_training else 'Suitable') + ' for training.')

        return unsuitable_training

    def sequence_encoding(self, datasets=None, label_map=None, max_len=32):
        """
                Batch counterpart to `encode`: turn a `[(text, label), ...]`
                dataset into a padded `(n_samples, max_len)` token-id array plus
                a one-hot `y_true`, reconciling class indices against
                `label_map` the same way `input_encoding`/`mlp_training_features`
                do when one is supplied, so all three encoders stay index
                -consistent for the same pipeline instance.
        """
        if not datasets:
            raise Warning('[!] Dataset is None or empty! Make sure you provide a dataset or create it automatically.')

        if not self.model2:
            intents = [d[1] for d in datasets]
            intent_to_id = {intent: i for i, intent in enumerate(sorted(set(intents)))}
            num_classes = self._get_num_classes(label_map=label_map)
            self.model2 = transformer.Transformer(
                vocab_size=len(self.vocab),
                d_model=self.transformer_d_model,
                n_heads=self.transformer_heads,
                num_classes=num_classes
            )

        self._sync_vocab_to_embedding()   # once, not per-item

        vocab_size = self.model2.token_embedding.shape[0]   # for bounds guard

        input_sequences = []
        for item in datasets:
            text = item[0] if isinstance(item, tuple) else item

            token_ids = self.encode(text, self.vocab, max_len=max_len)
            token_ids = np.asarray(token_ids)

            # bounds guard before indexing, with informative context
            out_of_bounds = (token_ids < 0) | (token_ids >= vocab_size)
            if out_of_bounds.any():
                n_bad = out_of_bounds.sum()
                print(f'[⚠️] sequence_encoding: {n_bad} token id(s) out of '
                    f'range [0, {vocab_size}) for text="{str(text)[:50]}" '
                    f'— clamping to valid range. This usually means vocab '
                    f'grew after model2 was constructed; consider rebuilding '
                    f'model2 or resizing token_embedding.')
                token_ids = np.clip(token_ids, 0, vocab_size - 1)

            token_embs = self.model2.token_embedding[token_ids]
            pos_embs   = self.model2.pos_embedding[:max_len]

            sequence_input = token_embs + pos_embs
            input_sequences.append(sequence_input)

        return np.stack(input_sequences)


    def transformer_pooled_features(self, sequence_inputs):
        """Mean/max/std pooling over the sequence dimension."""
        sequence_inputs = np.asarray(sequence_inputs)

        # guard degenerate single-timestep input, where std=0
        # everywhere is mathematically correct.
        if sequence_inputs.shape[1] == 1:
            print('[=] transformer_pooled_features: single-timestep input, '
                'std_pool will be all zeros (no variance across T=1)')
            print('[=] Reshaping sequence inputs to 2 dimension...')
            sequence_inputs = sequence_inputs.reshape(-1, 1)

        mean_pool = np.mean(sequence_inputs, axis=1)
        max_pool  = np.max(sequence_inputs, axis=1)
        std_pool  = np.std(sequence_inputs, axis=1)

        return np.concatenate([mean_pool, max_pool, std_pool], axis=-1)

    def _features_to_sequence(self, X_provided, d_model=None, min_seq_len=2):
        """
        Convert a flat (n_samples, n_features) array into a pseudo-sequence
        (n_samples, T, d_model). Hardened against locale issues, malformed
        strings, and degenerate padding — guarantees a finite output.
        """

        # ---- 1. String parsing: locale-independent, never raises on bad tokens ----
        def _safe_parse_string(s):
            clean_str = (str(s).replace('[', '').replace(']', '')
                                .replace('...', '').replace(',', ' ').strip())
            skip = {"nan", "null", "none", "inf", "-inf", "infinity", ""}
            vals = []

            for tok in clean_str.split():
                if tok.lower() in skip:
                    vals.append(0.0)
                    continue
                try:
                    v = float(tok)  # locale-independent, unlike np.fromstring
                    vals.append(v if np.isfinite(v) else 0.0)
                except ValueError:
                    vals.append(0.0)  # malformed token -> safe default, not a crash
            return np.array(vals, dtype=np.float64)

        if isinstance(X_provided, (str, np.str_)):
            X_provided = _safe_parse_string(X_provided)
    

        if isinstance(X_provided, np.ndarray) and np.issubdtype(X_provided.dtype, np.character):
            joined = ' '.join(X_provided.astype(str).flatten())
            X_provided = _safe_parse_string(joined)

        X_provided = np.asarray(X_provided, dtype=np.float64)
        if X_provided.ndim == 1:
            X_provided = X_provided.reshape(1, -1)

        # ---- 2. Sanitize the raw input itself (covers upstream NaN/Inf too) ----
        if not np.isfinite(X_provided).all():
            n_bad = (~np.isfinite(X_provided)).sum()
            print(f"[!] _features_to_sequence: sanitizing {n_bad} non-finite input values")
            X_provided = np.nan_to_num(X_provided, nan=0.0, posinf=0.0, neginf=0.0)

        n_samples, n_features = X_provided.shape
        n_features = max(n_features, 1)  # guard against degenerate empty input

        # ---- 3. Resolve d_model, preferring exact divisors (avoids padding) ----
        if d_model is None:
            if hasattr(self, 'model2') and getattr(self, 'model2', None) is not None:
                d_model = self.model2.d_model
            else:
                candidates = [d for d in range(1, n_features + 1) if n_features % d == 0]
                valid = [d for d in candidates if min_seq_len <= n_features // d <= 16]
                d_model = max(valid) if valid else max(1, n_features // 8)
        d_model = max(1, int(d_model))

        T = max(int(np.ceil(n_features / d_model)), min_seq_len)
        padded_width = T * d_model
        pad_amount = padded_width - n_features

        # ---- 4. Pad with wrapped real data, never constant zeros ----
        if pad_amount > 0:
            if n_features >= 1:
                X_padded = np.pad(X_provided, ((0, 0), (0, pad_amount)), mode='wrap')
            else:
                X_padded = np.zeros((n_samples, padded_width))
        else:
            X_padded = X_provided[:, :padded_width]

        sequence_inputs = X_padded.reshape(n_samples, T, d_model)

        # ---- 5. Break any exact-zero-variance timestep (root cause of LayerNorm NaN) ----
        # Add a deterministic, tiny, reproducible perturbation only where variance is 0.
        stds = sequence_inputs.std(axis=-1, keepdims=True)
        flat_mask = (stds < 1e-12)
        if flat_mask.any():
            rng = np.random.default_rng(seed=0)  # deterministic -> reproducible across runs
            noise = rng.normal(0.0, 1e-6, size=sequence_inputs.shape)
            sequence_inputs = np.where(flat_mask, sequence_inputs + noise, sequence_inputs)

        # ---- 6. Hard safety net: guarantee finite output no matter what happened above ----
        if not np.isfinite(sequence_inputs).all() or not np.isfinite(sequence_inputs.std(axis=-1)).all():
            n_bad = (~np.isfinite(sequence_inputs)).sum()
            print(f"[!] _features_to_sequence: FINAL sanitize caught {n_bad} non-finite values "
                f"(this should be rare — investigate if it triggers often)")
            sequence_inputs = np.nan_to_num(sequence_inputs, nan=0.0, posinf=0.0, neginf=0.0)

        print(f'[=] Converted X_provided {X_provided.shape} → '
            f'sequence_inputs {sequence_inputs.shape} '
            f'(d_model={d_model}, T={T}, padded {pad_amount} values)')

        return sequence_inputs


    def _sanitize_string_chars(self, x):
        """
                Strip characters from `x` (typically raw text) that would break
                downstream vocabulary/tokenization — e.g. control characters and
                anything that survived encoding as literal escape sequences —
                without altering normal punctuation or whitespace-separated
                words.
        """
        if isinstance(x, (str, np.str_)):
            clean_str = str(x).replace('[', '').replace(']', '').replace('...', '').strip()
            x = np.fromstring(clean_str, sep=' ')

        if isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.character):
            # catches arrays filled with string text
            clean_str = ' '.join(x.astype(str).flatten()).replace('[', '').replace(']', '')
            skip_values = {"...", "nan", "null"}
            x = np.fromiter((v for v in clean_str.split() if v not in skip_values), dtype=float)

        return x
        
    def _validate_onehot(self, y_true, context="train()"):
        """
        Robust one-hot validation for y_true, going beyond a simple .any() check.
        Catches: out-of-range values, negative values, non-binary floats,
        rows that don't sum to 1, and all-zero rows — each with a distinct,
        actionable error message instead of a generic failure.
        """
        y_true = np.asarray(y_true, dtype=np.float64)

        if y_true.ndim == 1:
            y_true = y_true[np.newaxis, :]

        issues = []

        # 1. Values outside [0, 1].
        out_of_range = (y_true < 0) | (y_true > 1)
        if out_of_range.any():
            bad_rows = np.where(out_of_range.any(axis=1))[0]
            issues.append(
                f"values outside [0,1] at rows {bad_rows.tolist()[:5]} "
                f"(sample values: {y_true[bad_rows[0]].tolist() if len(bad_rows) else []}) "
                f"— looks like raw label indices, not one-hot"
            )

        # 2. Non-binary values (e.g. soft labels or corrupted floats like 0.5, 0.999999)
        #    Only check this if values ARE in [0,1] — no point double-reporting.
        in_range = ~out_of_range
        non_binary = in_range & (~np.isclose(y_true, 0, atol=1e-6)) & (~np.isclose(y_true, 1, atol=1e-6))
        if non_binary.any():
            bad_rows = np.where(non_binary.any(axis=1))[0]
            issues.append(
                f"non-binary values at rows {bad_rows.tolist()[:5]} "
                f"(sample values: {y_true[bad_rows[0]].tolist() if len(bad_rows) else []}) "
                f"— expected exactly 0 or 1 per class slot"
            )

        # 3. Row sums must be exactly 1 (each sample has exactly one true class)
        row_sums = y_true.sum(axis=1)
        bad_sum_rows = np.where(~np.isclose(row_sums, 1.0, atol=1e-3))[0]
        if len(bad_sum_rows) > 0:
            issues.append(
                f"rows not summing to 1 at indices {bad_sum_rows.tolist()[:5]} "
                f"(sums: {row_sums[bad_sum_rows[:5]].tolist()}) "
                f"— either multi-label rows, all-zero rows, or raw indices"
            )

        if issues:
            print(f"[>] [{context}] y sample is not properly one-hot encoded, one-hot encoding y sample...")
            return True

        return False

    def _should_train_transformer(self, sequence_inputs, X_raw,
                                min_seq_len=3, min_anisotropy=0.35,
                                min_samples=10, ram_headroom_mb=80):
        """
        Decides whether transformer.Transformer training is worth its RAM/compute cost
        for THIS data, reusing existing AME/anisotropy signals rather than
        adding new expensive computation. Built specifically for
        memory-constrained edge deployment (e.g. Raspberry Pi Zero).

        Returns (should_train: bool, reason: str) so callers can log why should_train.
        
        """
        sequence_inputs = np.asarray(sequence_inputs)
        n_samples = sequence_inputs.shape[0] if sequence_inputs.ndim > 0 else 0
        T = sequence_inputs.shape[1] if sequence_inputs.ndim > 1 else 1

        # ── Check 1: sample count ────────────────────────────────
        if n_samples < min_samples:
            return False, (f'only {n_samples} samples (need >= {min_samples}) — '
                        f'too little data to justify transformer.Transformer parameter count')

        # ── Check 2: sequence length ─────────────────────────────
        if T < min_seq_len:
            return False, (f'sequence length T={T} (need >= {min_seq_len}) — '
                        f'attention has too few positions to model '
                        f'meaningful relationships')

        # ── Check 3: anisotropy   ──
        # ── zero new computation cost                              ──
        anisotropy = self.anisotropy_measurement(sequence_inputs)
        if anisotropy < min_anisotropy:
            return False, (f'anisotropy={anisotropy:.4f} (need >= {min_anisotropy}) — '
                        f'data lacks directional/sequential structure; '
                        f'mlp.MLP alone is the appropriate model for this '
                        f'geometry (same finding as make_moons earlier)')

        # ── Check 4: RAM headrooms, optional, graceful if unavailable ──
        if ram_headroom_mb is not None:
            try:
                available_mb = psutil.virtual_memory().available / (1024 * 1024)
                if available_mb < ram_headroom_mb:
                    return False, (f'only {available_mb:.0f}MB RAM available '
                                f'(need >= {ram_headroom_mb}MB headroom) — '
                                f'skipping transformer.Transformer to protect device stability')
            except ImportError:
                pass   # psutil not installed — skip memory check silently,
            

        return True, (f'n={n_samples}, T={T}, anisotropy={anisotropy:.4f} — '
                    f'sufficient structure and data to justify transformer.Transformer training')

    def transformer_utilities(self, X_provided= None, X_raw=None, y_true=None, rules=None, 
                              datasets=None, label_map=None, batch_size=2, min_signal=1e-3,
                              max_samples_for_focused_fit=500, max_ram_used=80):
        """
                Full training orchestration for one round: encode inputs for
                both models (`sequence_encoding`/`text_encoder`/tf-idf), build
                the hybrid raw+tfidf+dot-product feature matrix (with a checksum
                -based fallback for any all-zero rows, so a text that shares no
                vocabulary with the current tf-idf fit still gets a
                deterministic non-zero feature row instead of a dead one),
                conditionally train the transformer.Transformer (`_should_train_transformer`)
                and fold its pooled features into the hybrid matrix, then
                initialize and train `self.model3` (mlp.MLP) and finally the LSTM
                branch (`lstm_setup_inference`). If training isn't warranted this
                round, instead (re)builds model shells sized from the given data
                and loads previously-saved weights from `self.storage` for pure
                inference.
        """
        unsuitable = False
        max_ram_used = self.max_ram_allowed if max_ram_used is None else max_ram_used

        if X_provided is not None:
            X_raw = X_provided

        if y_true is None:
            if datasets is not None:
                _, y_true = self.input_encoding(datasets)
            else:
                raise ValueError('[!] y_true samples is None and datasets is not provided! Cannot proceed with Full Training!')
                
        y_true = self._sanitize_string_chars(y_true)
        X_raw = self._sanitize_string_chars(X_raw)
        X_provided = self._sanitize_string_chars(X_provided)

        if isinstance(X_raw, (str, np.str_)):
            clean_str = str(X_raw).replace('[', '').replace(']', '').replace('...', '').strip()
            X_raw = np.fromstring(clean_str, sep=' ')

        if isinstance(X_raw, np.ndarray) and np.issubdtype(X_raw.dtype, np.character):
            # catches arrays filled with string text
            clean_str = ' '.join(X_raw.astype(str).flatten()).replace('[', '').replace(']', '')
            skip_values = {"...", "nan", "null"}
            X_raw = np.fromiter((v for v in clean_str.split() if v not in skip_values), dtype=float)
        
        if datasets is not None:       
            self.text_encoder(datasets)

        if X_provided is None:
            sequence_inputs = self.sequence_encoding(datasets, label_map=label_map)
        else:
            sequence_inputs = self._features_to_sequence(X_provided, d_model=self.transformer_d_model)

        unsuitable_training = self.training_necessary_condition(sequence_inputs, X_raw)
        lr = self.model2.transformer_lr if self.model2 else self.transformer_lr

        if not unsuitable_training:
            print(f'🚀 Training transformer.Transformer with {len(sequence_inputs)} Samples: ')

            x_conditional_anisotropy = self.anisotropy_measurement(sequence_inputs)
            s_conditional_anisotropy = self.anisotropy_measurement(X_raw)

            AME_x = self.AME_Encoder(X_raw)
            AME_s = self.AME_Encoder(sequence_inputs)
            AMR_x = 1.0 / (1.0 + np.exp(-AME_x))
            AMR_s = 1.0 / (1.0 + np.exp(-AME_s))

            AMR_ratio = AMR_x / (AMR_s + min_signal)
            anisotropy_ratio = x_conditional_anisotropy / (s_conditional_anisotropy + min_signal)

            dynamic_complex_environment = (anisotropy_ratio < 0.5 and 
                                             AMR_ratio < 0.5)                           
                                               
            if dynamic_complex_environment: 
                print('[+] Dynamic Backward for transformer.Transformer Initiated')
                mode = 'dynamic_backward'
            else:
                print('[=] Fixed Backward for transformer.Transformer initiated')
                mode = 'fixed_backward'

            if X_provided is None and datasets is not None and rules is not None:
                X_raw_generation, y, n_classes, input_dim = self.mlp_training_features(rules, datasets)
            else:
                if X_provided is None:
                    raise ValueError('[!] X_provided is None but the auto-generation '
                                    'branch conditions were not met — cannot proceed Training and prediction!')

                X_provided = np.asarray(X_provided)
                if X_provided.ndim == 1:
                    X_provided = X_provided.reshape(-1, 1)

                X_raw_generation = X_provided
                y = y_true

                y_arr = np.asarray(y_true) if y_true is not None else None
                n_classes = (y_arr.shape[1] if y_arr is not None and y_arr.ndim > 1
                            else len(np.unique(y_arr)) if y_arr is not None else 0)

                # input_dim should reflect the REAL feature count, with a
                # sanity check against hybrid_X's expected width, not silently
                # trust whatever shape X_provided happens to have arrived in
                input_dim = X_provided.shape[1] if X_provided.ndim > 1 else 1

                if input_dim == 1 and X_provided.shape[0] > 1:
                    print(f'[⚠️] input_dim resolved to 1 from X_provided.shape={X_provided.shape} '
                        f'— this usually means X_provided was already narrowed '
                        f'upstream (e.g. by a string-parsing gate or prior '
                        f'truncation) BEFORE reaching this point. If your real '
                        f'feature count should be larger, trace X_provided '
                        f'backward from here to find where it got reduced to '
                        f'a single column.')

            if isinstance(X_raw_generation[0], str):
                X_raw_features = self.tfidf.transform(X_raw_generation).toarray()
                X_raw_features = self._refit_sparse_data(X_raw_features, X_raw_generation)
            elif X_provided is not None:
                X_raw_features = self._refit_sparse_data(X_provided, X_raw_generation) 
            else:
                X_raw_features = np.asarray(X_raw_generation) 
            
            row_sums = X_raw_features.sum(axis=1)
            weak_rows = np.where(row_sums < min_signal)[0]
            weak_ratio = len(weak_rows) / len(X_raw_features)

            print(f'[>] Zero ratio in samples: {weak_ratio * 100}%')
            if weak_ratio > 0.3:  # more than 30% zero rows means vocab mismatch
                if isinstance(X_raw_generation[0], str):
                    print(f'[= ! =] High zero-row ratio ({weak_ratio:.0%}), refitting on current batch')
                    self.tfidf.fit(X_raw_generation)
                    X_raw_features = self.tfidf.transform(X_raw_generation).toarray()  
            
            zero_rows = np.where(X_raw_features.sum(axis=1) == 0)[0]
            if len(zero_rows) > 0:
                print(f'[!] {len(zero_rows)} zero rows detected, applying checksum fallback')
                for i in zero_rows:
                    text = X_raw[i] if isinstance(X_raw[i], str) else str(X_raw[i])
                    checksum = int(hashlib.md5(text.encode()).hexdigest(), 16)

                    # distribute checksum signal across feature dims
                    rng = np.random.default_rng(checksum)
                    X_raw_features[i] = rng.uniform(0.01, 0.1, size=X_raw_features.shape[1])  

            X_features = X_raw_features.copy()
            if isinstance(X_raw_features, list):
                X_features = np.asarray(X_raw_features)
            if isinstance(X_raw, list):
                X_raw = np.asarray(X_raw) 

            # hybrid features by dot product of raw features and extracted features from transformer, this allows the mlp.MLP to learn from both the original feature space and the transformer-extracted feature space
            # potentially improving its ability to capture complex patterns in the given first data   
            try:
                if len(X_features.shape) < 2: 
                    X_features = X_features.reshape(1, -1)
                if len(X_raw.shape) < 2:
                    X_raw = X_raw.reshape(1, -1)
                                    
                X_raw = X_raw[:X_features.shape[0], :X_features.shape[1]]

                hybrid_X = np.dot(X_raw, X_features.T)
            except:
                if len(X_features.shape) < 2: 
                    X_features = X_features.reshape(1, -1)
                if len(X_raw.shape) < 2:
                    X_raw = X_raw.reshape(1, -1)

                subnet_X_feature = X_features[:X_raw.shape[1], :X_raw.shape[0]]
                subnet_X_raw = X_raw[:, :subnet_X_feature.shape[0]]
                hybrid_X = np.dot(subnet_X_raw, subnet_X_feature)
            
            hybrid_X = np.concatenate([X_raw, X_features, hybrid_X], axis=-1)
            if isinstance(X_raw_generation[0], str):
                self.initialize_fitting(X_raw_generation)    
            else:
                self.vocab_size = 1

            onehot_validation = self._validate_onehot(y)
            if onehot_validation:
                if n_classes != len(label_map):
                    n_classes = len(label_map)
                if n_classes > np.max(y):
                    y = np.eye(n_classes)[np.asarray(y)]
                else:
                    print('[⚠️] Warning: Proper Y onehot encoding fails, modifying number of classes to exactly match Y values to do one final one hot encoding...')
                    try:
                        n_classes = np.max(y) + 1
                        y = np.eye(n_classes)[np.asarray(y)]
                    except:
                        print('[!] Error: One hot encoding failed, please check your Y values and label map for consistency, passing raw Y and skipping mlp.MLP Training...')
                        unsuitable = True
                        y = y.copy()  # fallback to raw y if one-hot fails

            if self.model2 is None:
                self.model2 = transformer.Transformer(
                    vocab_size=self.vocab_size,
                    d_model=self.transformer_d_model,
                    n_heads=self.transformer_heads,
                    num_classes=n_classes
                )
                
            should_train_transformer, reason = self._should_train_transformer(sequence_inputs, X_raw, min_seq_len=3, min_anisotropy=0.35, min_samples=10, ram_headroom_mb=max_ram_used)                  
            if self.use_transformer and should_train_transformer:
                print(f'[=] transformer.Transformer Training allowed, Reason: {reason}')
                self.model2.train(sequence_inputs, y_true, epochs=self.transformer_training_epochs, mode=mode, lr=lr, embedded=True, batch_size=batch_size)

            if should_train_transformer:      
                transformer_features = self.transformer_pooled_features(sequence_inputs)
                hybrid_X = np.concatenate([hybrid_X, transformer_features], axis=-1)  

            X = self.shape_adaptation(hybrid_X, input_dim)      
            self.initialize_model_(X, input_dim, n_classes)
            if not unsuitable:
                self.model3.train(X, y, epochs=self.mlp_training_epochs, lr=self.mlp_lr, max_samples_for_focused_fit=max_samples_for_focused_fit)
            else:
                print('[->] mlp.MLP Training skipped due to unproper Y samples.')

            self.lstm_setup_inference(X, y, input_ids=sequence_inputs) 
            if self.lstm_engine:
                self.storage.save_weights(self.memory_name, model_type='Pipeline') 
                
            print('🎉 All Model Trained!')
        else:
            print(f'[=] No suitable condition for training!')
            print('[>] Loading Weights for prediction...')

            num_classes = self._get_num_classes(label_map=label_map) if label_map else (y_true.shape[1] if y_true.ndim > 1 else len(np.unique(y_true)))

            if X_provided is not None:
                X = X_provided
            else:
                X = X_raw
                
            if y_true is not None:
                n_classes = y_true.shape[1] if y_true.ndim > 1 else len(np.unique(y_true))
            else:
                n_classes = X.shape[1] if X.ndim > 1 else 1

            input_dim = X.shape[1] if X.ndim > 1 else 1
            self.initialize_model_(X, input_dim, n_classes)

            self.model2 = transformer.Transformer(
                vocab_size=1,
                d_model=self.transformer_d_model,
                n_heads=self.transformer_heads,
                num_classes=num_classes
            )

            self.storage.load_weights(self.memory_name)

            pass



    def transformer_input_encoding(self, titles):
        """
                Encode `titles` (list of strings or `(text, ...)` tuples) into a
                `(n, max_seq_len)` token-id array using the current `self.vocab`,
                for use in transformer.Transformer-based probability calibration. Returns an
                empty list if no vocabulary exists yet.
        """
        if hasattr(self, 'vocab') and self.vocab:
            print("🔄 Using transformer.Transformer for probability calibration")
            input_ids_list = []
            for title in titles:
                if isinstance(title, tuple):
                    title = title[0]
                    
                ids = self.encode(title, self.vocab)
                input_ids_list.append(np.array(ids))
                
                input_ids = np.array(input_ids_list)
                return input_ids

        else:
            print('[-] Cant get sufficient data!')
            return []


    def train(self, X, y_raw):
        """
                Minimal standalone mlp.MLP-only training entry point: fits/
                transforms `X` through tf-idf, one-hot encodes `y_raw` via
                `initialize_model_encoding`, and trains. Does not touch the
                transformer.Transformer or LSTM branches — see `transformer_utilities`/`train`
                (the fuller pipeline-wide training path) for that.
        """
        self.initialize_fitting(X)            
        X_tfidf = self.tfidf.transform(X).toarray()
        self.X = X_tfidf.copy()

        print(f"\n🚀 Separate Modular mlp.MLP Pipeline:")
        print(f" Samples: {len(self.X)}")

        y_true = self.initialize_model_encoding(self.X, y_raw)      
        print('✅ Done Training mlp.MLP Model! ')





