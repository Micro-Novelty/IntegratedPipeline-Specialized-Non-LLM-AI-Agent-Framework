"""
mlp.py
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
from . import layers
from . import prediction_manager
from . import transformer
from . import weight_shaping

class MLP:
    """
    A from-scratch, fully-connected feed-forward network (a stack of
    `layers.Dense` layers ending in `layers.SoftmaxOutput`) with two custom mechanisms
    layered on top of a standard MLP:

    1. **Performance-gated layers ("focused" vs. "standard" pipelines).**
       The network actually holds *two* independent stacks of layers:
       `self.layers` (the normal/standard pipeline, used by `forward` /
       `backward` / `train`) and `self.feed_layers` (the "focused"
       pipeline, used by `focused_forward` / `focused_backward`). Both
       pipelines route every layer's `forward`/`backward` call through a
       scalar `performance_score` (from `performance_calculation`),
       which each `layers.Dense` layer can use to modulate its own behavior
       (e.g. scaling updates or activations by confidence -- see the
       `layers.Dense` layer for the exact mechanism). `train` decides which
       pipeline to use per-run via `focused_fit_condition`: the focused
       pipeline only kicks in when a `feed_layers` stack exists *and*
       the data looks sufficiently anisotropic/hard-to-abstract (high
       AMR/anisotropy) *and* the dataset is small enough
       (`len(X) < max_samples_for_focused_fit`) -- the idea being that a
       separate, performance-score-driven pipeline is only worth the
       overhead when data is scarce and geometrically complex, and
       falls back to the plain pipeline otherwise (or automatically, if
       the focused pipeline's loss goes non-finite mid-training).

    2. **`performance_score`: an AME/anisotropy/error-history-driven
       confidence signal.** Rather than every layer normalizing/scaling
       activations by a fixed constant, `performance_calculation`
       combines: (a) `AME * anisotropy` ("gate_uncertain" -- how
       geometrically hard the data is), (b) `1 - AME` ("gate_certainty"
       -- the flip side), and (c), when available, the network's own
       running `error_counts`/`pred_counts` history (how often each
       class has been mispredicted / how often each class has been
       predicted) folded in as `standard_low_error_mean` and
       `standard_pred_quality`. The resulting scalar is threaded through
       every layer's forward/backward call as `performance_score`, so
       layers can (in principle) adapt their behavior to both the
       current batch's geometry *and* the network's recent track record.

    3. **`continuous_predictive_correction`: online, per-class
       probability recalibration.** Given a `prediction_manager.PipelinePredictionManager`-
       style `manager` holding running `error_counts`/`pred_counts`/
       `decay`/`label_map`, this method down-weights predicted
       probabilities for classes with a history of being wrong (a
       activations.sigmoid-shaped `small_reputation` multiplier derived from each
       class's error rate), re-normalizes the probability vector, and
       keeps `pred_counts`/`error_counts` shape-synced with the current
       label space -- effectively a lightweight, non-gradient-based
       "trust calibration" layer applied after the network's raw output,
       driven by the same AME/anisotropy `performance_score` used
       elsewhere in the class.

    `AME_Encoder` / `anisotropy_measurement` here are MLP-local
    re-implementations of the same heuristics defined in
    `weight_shaping.GeometricWeightShaping` and `transformer.Transformer` (see those docstrings for
    the underlying math) -- kept as separate copies so this class has no
    hard dependency on the other two, at the cost of near-duplicated
    logic.
    """

    def __init__(self):
        """Initialize empty standard (`self.layers`) and focused
        (`self.feed_layers`) layer stacks, a default learning rate, the
        (initially unset) error/prediction-count history used by
        `performance_calculation`/`continuous_predictive_correction`,
        and the output `layers.SoftmaxOutput` layer."""
        self.layers = []
        self.layers2 = []
        self.lr = 0.1
        self.feed_layers = []

        self.error_counts = None
        self.pred_counts = None
        self.error_decay = None

        self.temp_AME_sample = 0
        self.temp_anisotropy_sample = 0

        self.softmax = layers.SoftmaxOutput()
      

    def feed_add(self, layer):
        """Append a layer to the "focused" pipeline (`self.feed_layers`),
        used by `focused_forward`/`focused_backward`."""
        self.feed_layers.append(layer)

    def add(self, layer):
        """Append a layer to the standard pipeline (`self.layers`), used
        by `forward`/`backward`."""
        self.layers.append(layer)

    def focused_forward(self, x, AME=None, anisotropy=None):
        """
        Forward pass through the "focused" pipeline (`self.feed_layers`)
        instead of the standard one -- see the class docstring for when
        `train` selects this path. Computes `performance_score` once up
        front and threads it through every layer's `forward` call, then
        applies the shared `layers.SoftmaxOutput`.
        """
        performance_score = self.performance_calculation(x, AME=AME, anisotropy=anisotropy) 

        for layer in self.feed_layers:
            x = layer.forward(np.asarray(x, dtype=np.float64), performance_score)
            
        return self.softmax.forward(x) 

    def k_fold_split(self, X, y, k=5, seed=42, min_fold_size=2):
        """
        Generator yielding `(X_train, y_train, X_val, y_val)` for each of
        `k` folds (Fisher-Yates-shuffled via a seeded `Generator`),
        automatically reducing `k` when the dataset is too small to give
        every fold at least `min_fold_size` samples.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        n = len(X)

        # adapt k downward for small datasets, never produce
        # folds smaller than min_fold_size
        effective_k = min(k, max(2, n // min_fold_size))
        if effective_k < k:
            print(f'[⚠️] k_fold_split: reduced k from {k} to {effective_k} '
                f'for n={n} samples, to avoid folds smaller than '
                f'{min_fold_size} samples')

        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        folds = np.array_split(idx, effective_k)

        for i in range(effective_k):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(effective_k) if j != i])
            yield X[train_idx], y[train_idx], X[val_idx], y[val_idx]


    def confusion_matrix(self, y_true, y_pred, num_classes):
        """Build a `(num_classes, num_classes)` confusion matrix from
        `y_true`/`y_pred`, accepting either one-hot/probability arrays
        (argmax'd first) or already-integer class-label arrays. Labels
        that fall outside `[0, num_classes)` are skipped with a
        warning rather than raising."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        cm = np.zeros((num_classes, num_classes), dtype=int)

        # Convert to label arrays: use argmax only if input is one-hot/prob (2D+),
        # otherwise need to treat as already being class labels.
        true_labels = np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true
        pred_labels = np.argmax(y_pred, axis=1) if y_pred.ndim > 1 else y_pred

        for t, p in zip(true_labels, pred_labels):
            if 0 <= t < num_classes and 0 <= p < num_classes:
                cm[t, p] += 1
            else:
                print(f'[⚠️] confusion_matrix: label out of range '
                    f'(true={t}, pred={p}, num_classes={num_classes}) — skipped')
        return cm
            
    def performance_calculation(self, x, AME=None, anisotropy=None):
        """
        Compute the scalar `performance_score` threaded through every
        layer's forward/backward call (see the class docstring's
        "performance_score" section for the intuition).

        `AME`/`anisotropy` are computed from `x` via `AME_Encoder` /
        `anisotropy_measurement` if not supplied. The score combines:
        `gate_uncertain = AME * anisotropy`, `gate_certainty = 1 - AME`,
        plus (if the network has an error/prediction-count history from
        `continuous_predictive_correction`) `standard_low_error_mean`
        (inverse of mean historical error rate) and
        `standard_pred_quality` (mean historical prediction confidence).
        Falls back to a fixed `0.15` if the combination is non-finite.
        """
        eps = 1e-5
        standard_low_error_mean = eps
        standard_pred_quality = eps

        if AME is None and anisotropy is None:
            AME = self.AME_Encoder(x)     
            anisotropy = self.anisotropy_measurement(x)

        gate_uncertain = (AME * anisotropy) + eps / 4
        gate_certainty = (1.0 - AME) + eps / 4
        if self.error_counts is not None and self.pred_counts is not None:
            standard_low_error_mean = (1.0 - np.mean(self.error_counts) + eps / 4)
            standard_pred_quality = np.mean(self.pred_counts) + eps / 4

        performance_score = (gate_uncertain + gate_certainty + 
                               standard_low_error_mean + standard_pred_quality)

        if np.isnan(performance_score) or np.isinf(performance_score):
            performance_score = 0.15

        return performance_score

    def forward(self, x, y=None, AME=None, anisotropy=None, condition=None):
        eps = 1e-5
        performance_score = self.performance_calculation(x, AME=AME, anisotropy=anisotropy)

        for layer in self.layers:
            x = layer.forward(x, performance_score)

        output = self.softmax.forward(x)
        if not condition == 'training' and y is not None:
            if y.shape == output.shape:
                acc = np.mean(np.argmax(output, axis=1) == np.argmax(y, axis=1))
                loss = activations.Loss.categorical_crossentropy(y, output)
                print(f"[=] MLP Forward validation score: | loss: {loss:.4f} | Acc: {acc:.2%}")

        return output
        
    def _sanitize_string_chars(self, x):
        if isinstance(x, (str, np.str_)):
            clean_str = str(x).replace('[', '').replace(']', '').replace('...', '').strip()
            x = np.fromstring(clean_str, sep=' ')

        if isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.character):
            # catches arrays filled with string text
            clean_str = ' '.join(x.astype(str).flatten()).replace('[', '').replace(']', '')
            skip_values = {"...", "NaN", "null"}
            x = np.fromiter((v for v in clean_str.split() if v not in skip_values), dtype=float)

        return x



    def continuous_predictive_correction(self, manager, prob, predicted_index, AME=None, anisotropy=None):
        eps = 1e-5

        error_counts = manager.error_counts
        pred_counts = manager.pred_counts
        decay = manager.decay
        label_map = manager.label_map

        self.error_counts = error_counts
        self.pred_counts = pred_counts
        self.error_decay = decay

        if prob is None:
            print('[!] Probabilities is None! returning the probabilities...')
            return prob

        try:
            self.pred_counts[predicted_index] += 1.0
            n_classes = len(label_map)

            performance_score = self.performance_calculation(prob, AME=AME, anisotropy=anisotropy)
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
                        small_reputation    = 1.0 / (1.0 + error_rate)
                        weighted_reputation = small_reputation * performance_score

                        reputation_growth = 1.0 / (1.0 + np.exp(-performance_score))
                        compounding_factor = weighted_reputation * reputation_growth

                        if isinstance(error_rate, (list, np.ndarray)):
                            if len(error_rate) != len(prob):
                                weighted_reputation = 1.0 / (1.0 + np.mean(error_rate))

                        if c < len(prob):
                            prob[c]  *= small_reputation
                        if c < len(self.pred_counts):
                            self.pred_counts[c] *= compounding_factor
                    
            prob_sum = prob.sum()
            if prob_sum > 1e-8:
                prob /= prob_sum

            # re adapt shape of pred_counts and error_counts if they don't match prob shape
            if self.pred_counts.shape != prob.shape:
                self.pred_counts = np.zeros_like(prob)
                self.pred_counts *= decay
            if self.error_counts.shape != prob.shape:
                self.error_counts = np.zeros_like(prob)
                self.error_counts *= decay

        except Exception as e:
            print(f'[!] Cant check and calibrate probs based on penalty due to: {e}')    
            traceback.print_exc()
        
        return prob 


    def focused_backward(self, grad, lr, AME, anisotropy):
        grad = self.softmax.backward(grad)
        perf_score = self.performance_calculation(grad, AME=AME, anisotropy=anisotropy)

        for layer in reversed(self.feed_layers):
            grad, key_grad = layer.backward(grad, lr, perf_score)
        return grad, key_grad

    def backward(self, grad, lr):
        grad = self.softmax.backward(grad)
        perf_score = self.performance_calculation(grad, AME=self.temp_AME_sample, anisotropy=self.temp_anisotropy_sample)

        for layer in reversed(self.layers):
            grad, key_grad = layer.backward(grad, lr, perf_score)
        return grad, key_grad
            
    def predict(self, X, y, epochs=1000, verbose=True):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = activations.Loss.categorical_crossentropy(y, y_pred)    		
            if verbose and epoch % 100 == 0:
                acc = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))
                self.acc2.append(acc)
                print(f"[=] Epoch {epoch} | loss:{loss:.4f} | Acc: {acc:.2f}")

	   
    def prediction(self, X):
        y_pred = self.forward(X)   
        return y_pred      
     
         
    def AME_Encoder(self, x):
        X = np.asarray(x)

        if len(X) == 0:
            print('[!] X size is 0, AME Will be replaced by minimum confidence threshold')
            return 0.0

        if _OPT_AVAILABLE and np.asarray(X).ndim == 2:
            return optimized_ame_encoder(np.asarray(X, dtype=np.float64))     


        if x.shape[1] == 1:
            x = x.T
            x= x.flatten()

        try:
            gradient = np.gradient(x, axis=-1)
        except:
            x = np.reshape(x, (x.shape[0], -1))
            gradient = np.gradient(x, axis=-1, edge_order=1)

        grad_energy = np.mean(np.linalg.norm(gradient, axis=-1))       
        X_mag = np.mean(np.linalg.norm(X, axis=-1))

        AME = np.log1p(X_mag) * np.log1p(grad_energy) 
        return AME


    def anisotropy_measurement(self, x):
        eps = 1e-5
        if _OPT_AVAILABLE:
            x = np.asarray(x)            
            x = x.reshape(x.shape[0], -1)
            return optimized_anisotropy(np.asarray(x, dtype=np.float64))

        try:
            gradient = np.gradient(x)
        except:
            subnet = x[:min(10, x.shape[0]), :min(10, x.shape[1])]
            gradient = np.gradient(subnet.flatten())

        val = [np.linalg.norm(v) for v in gradient]
        anisotropy = np.std(val) + eps/ np.mean(val) + eps
        return anisotropy

    def adapt_predict_shape(self, y_pred, y_true):
        try:
            y_pred_arr = np.asarray(y_pred)
            y_arr      = np.asarray(y_true)

            # normalize to 2D
            if y_pred_arr.ndim == 1:
                y_pred_arr = y_pred_arr[np.newaxis, :]
            if y_arr.ndim == 1:
                y_arr = y_arr[np.newaxis, :]

            # align batch and class dims — same approach as activations.Loss
            min_batch = min(y_pred_arr.shape[0], y_arr.shape[0])
            min_class = min(y_pred_arr.shape[1], y_arr.shape[1])

            y_pred_aligned = y_pred_arr[:min_batch, :min_class]
            y_aligned      = y_arr[:min_batch, :min_class]

            if y_pred_aligned.size == 0 or y_aligned.size == 0:
                print(f'[!] Cannot compute accuracy — empty after alignment')
                acc = 0.0
            else:
                preds = np.argmax(y_pred_aligned, axis=1)
                true  = np.argmax(y_aligned, axis=1)
                acc   = float(np.mean(preds == true))

            return y_pred_aligned, y_aligned

        except Exception as e:
            print(f'[!] Cant adapt shape of Y sample arrays due to: {e}')
            return y_pred, y_true

    
    def train(self, X, y, epochs=1000, lr=0.01, verbose=True, max_samples_for_focused_fit=500):
        X = self._sanitize_string_chars(X)
        y = self._sanitize_string_chars(y)
        focused_fit_condition = False
        parameters = sum(w.size for w in self.layers[0].W) + sum(b.size for b in self.layers[0].b)

        AME = self.AME_Encoder(X)     
        AMR = 1.0 / (1.0 + np.exp(-float(AME)))

        anisotropy = self.anisotropy_measurement(X)
        self.temp_AMR_sample = AMR
        self.temp_anisotropy_sample = anisotropy

        focused_fit_condition = len(self.feed_layers) > 0 and anisotropy > 0.25 and AMR > 0.25 and len(X) < max_samples_for_focused_fit
        print(f'[+] Focused fit condition: {focused_fit_condition} || Anisotropy: {self.anisotropy_measurement(X):.4f} || AME: {self.AME_Encoder(X):.4f}')

        training_not_allowed = np.isnan(anisotropy) or np.isinf(anisotropy) or np.isnan(AME) or np.isinf(AME) or AME < 0.1
        if training_not_allowed:
            print(f'[!] MLP Training not allowed due to unsuitable data characteristics. Anisotropy: {anisotropy:.4f}, AME: {AME:.4f}')
        else:
            print(f'[+] MLP Training started with: {parameters} Parameters.')
            for epoch in range(epochs):
                if not focused_fit_condition:
                    y_pred = self.forward(X, y=y, AME=AME, anisotropy=anisotropy, condition='training')
                else:
                    y_pred = self.focused_forward(X, AME=AME, anisotropy=anisotropy)

                y_pred, y_true = self.adapt_predict_shape(y_pred, y)
                
                loss = activations.Loss.categorical_crossentropy(y_true, y_pred)
                grad = activations.Loss.softmax_crossentropy_derivative(y_true, y_pred)
                if focused_fit_condition:
                    _, key_grad = self.focused_backward(grad, self.lr, AME, anisotropy)
                else:
                    _, key_grad = self.backward(grad, self.lr)

				# Adam Optimizer used here after gradient was calculated and weight has changed, adam preserve gradient direction and renormalize it to 
				# be more suited to what the gradient has and contains and stabilize parameters update.
                try:
                    params = self.layers[0].opt.step(self.layers[0].params, key_grad, clip_norm=5.0)
                except:
                    continue
					
                if np.isnan(loss) or np.isinf(loss):
                    if focused_fit_condition:
                        focused_fit_condition = False
                
                if verbose and epoch % 100 == 0:
                    acc = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))
                    print(f"[=] Epoch {epoch} | activations.Loss: {loss:.4f} | Acc: {acc:.2f}")
              
# ─────────────────────────────────────────────
#  LSTM Cell
# ─────────────────────────────────────────────



