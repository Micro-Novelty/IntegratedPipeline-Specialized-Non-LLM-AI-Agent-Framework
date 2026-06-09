import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import random
from datetime import datetime, timedelta
import sqlite3
import json
import joblib
import ast
import re
import sys
import threading
import time
from collections import deque, defaultdict
import socket
import pickle
import hashlib
import ssl
import os
import asyncio
import queue
import uuid
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Tuple, Optional, Dict, List
from enum import IntEnum, Enum
import traceback
from concurrent.futures import TimeoutError as FutureTimeoutError
import secrets
import ipaddress
from functools import wraps
import hmac
import aiohttp
import psutil

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# nn.py
# Core neural network building blocks:
#   Activation  — static activation functions (relu, sigmoid, tanh, softmax, leaky_relu)
#   Loss        — static loss functions and their gradients
#   Dense       — a single fully-connected layer backed by GWS initialisation
#   SoftmaxOutput — thin output layer wrapper
# Depends on: geometry (GeometricWeightShaping)
# ---------------------------------------------------------------------------
# ─────────────────────────────────────────────
#  LSTM Cell
# ─────────────────────────────────────────────

class LSTMCell:
    """
    Single LSTM cell operating on one time-step.

    Gate layout (all concatenated into one weight matrix for speed):
        W shape: (4*hidden, input + hidden)
        b shape: (4*hidden,)

    Slice order: [forget | input | gate (candidate) | output]
    """

    def __init__(self, input_size: int, hidden_size: int, seed: int = 42):
        self.input_size  = input_size
        self.hidden_size = hidden_size
        np.random.seed(seed)

        # Xavier / Glorot init
        scale = np.sqrt(2.0 / (input_size + hidden_size))
        self.W = np.random.randn(4 * hidden_size, input_size + hidden_size) * scale
        self.b = np.zeros((4 * hidden_size,))

        # Output projection: hidden → output
        self.Wy = np.random.randn(hidden_size, hidden_size) * scale
        self.by = np.zeros((hidden_size,))

    # ── slicing helpers ──────────────────────
    def _f(self, v): return v[:self.hidden_size]
    def _i(self, v): return v[self.hidden_size:2*self.hidden_size]
    def _g(self, v): return v[2*self.hidden_size:3*self.hidden_size]
    def _o(self, v): return v[3*self.hidden_size:]


    # ── forward ──────────────────────────────
    def forward(self, x_seq: np.ndarray, h0=None, c0=None):
        """
        x_seq : (T, input_size)
        returns: hs (T, hidden), cs (T, hidden), cache (for BPTT)
        """
        T = x_seq.shape[0]
        H = self.hidden_size
        expected_input = self.input_size

        h = np.zeros(H) if h0 is None else h0.copy()
        c = np.zeros(H) if c0 is None else c0.copy()

        hs, cs = np.zeros((T, H)), np.zeros((T, H))
        cache  = []   # store everything needed for backward

        for t in range(T):
            x  = x_seq[t]
            if x.ndim == 0:
                x = x.reshape(1)                          # fix zero-dimensional
            if x.shape[0] < expected_input:
                x = np.pad(x, (0, expected_input - x.shape[0]))   # pad if too small
            elif x.shape[0] > expected_input:
                x = x[:expected_input]                    # truncate if too large
           
            xh = np.concatenate([x, h])           # (input+hidden,)

            z  = self.W @ xh + self.b             # (4H,)
            f  = sigmoid(self._f(z))              # forget
            i  = sigmoid(self._i(z))              # input
            g  = np.tanh(self._g(z))              # candidate
            o  = sigmoid(self._o(z))              # output

            c_new = f * c + i * g
            tanh_c = np.tanh(c_new)
            h_new = o * tanh_c

            cache.append((x, h, c, f, i, g, o, c_new, tanh_c, xh))
            h, c = h_new, c_new
            hs[t], cs[t] = h, c

        return hs, cs, cache

    # ── backward (BPTT) ──────────────────────
    def backward(self, dhs: np.ndarray, cache, dh_next=None, dc_next=None):
        """
        dhs : (T, hidden)  — gradient of loss w.r.t. each hidden state
        Returns: gradients dict + dx_seq
        """
        T = len(cache)
        H = self.hidden_size

        dW  = np.zeros_like(self.W)
        db  = np.zeros_like(self.b)
        dh  = np.zeros(H) if dh_next is None else dh_next.copy()
        dc  = np.zeros(H) if dc_next is None else dc_next.copy()
        dx_seq = np.zeros((T, self.input_size))

        for t in reversed(range(T)):
            x, h_prev, c_prev, f, i, g, o, c_new, tanh_c, xh = cache[t]

            dh_total = dhs[t] + dh   # gradient from loss + recurrence

            # output gate
            do     = dh_total * tanh_c
            dtanhc = dh_total * o

            # cell state
            dc_new = dtanhc * tanh_deriv(tanh_c) + dc

            # gates
            df = dc_new * c_prev
            di = dc_new * g
            dg = dc_new * i
            dc = dc_new * f           # flows back to previous cell state

            # pre-activation gradients
            df_pre = df * sigmoid_deriv(f)
            di_pre = di * sigmoid_deriv(i)
            dg_pre = dg * tanh_deriv(g)
            do_pre = do * sigmoid_deriv(o)

            dz = np.concatenate([df_pre, di_pre, dg_pre, do_pre])

            dW  += np.outer(dz, xh)
            db  += dz
            dxh  = self.W.T @ dz
            dx_seq[t] = dxh[:self.input_size]
            dh   = dxh[self.input_size:]

        return {"dW": dW, "db": db}, dx_seq, dh, dc


# ─────────────────────────────────────────────
#  LSTM Network (cell + linear output head)
# ─────────────────────────────────────────────

class LSTMNetwork:
    """
    One LSTM layer + a linear output projection.
    input_size  → hidden_size  → output_size
    """

    def __init__(self, pipeline, input_size, hidden_size, output_size, seed=0):
        self.cell = LSTMCell(input_size, hidden_size, seed)
        H = hidden_size      

        self.weight_shaper = GeometricWeightShaping(output_size, H)         
        self.Wy = None
        self.by = np.zeros(output_size)
        self.pipeline = pipeline

    def forward(self, x_seq):
        if self.Wy is None:
            self.Wy = self.weight_shaper.weight_shaping(x_seq)

        hs, cs, cache = self.cell.forward(x_seq)
        # linear projection at every timestep
        preds = hs @ self.Wy.T + self.by        # (T, output_size)
        return preds, hs, cs, cache

    def loss_mse(self, preds, targets, AMR):
        if preds.shape != targets.shape:
            if preds.shape[0] > targets.shape[0]:
                targets = targets[:preds.shape[0], :]
            if targets.shape[0] > preds.shape[0]:
                preds = preds[:targets.shape[0], :]
            else:
                preds = preds[:targets.shape[0], :targets.shape[1]]
        
        diff = preds - targets
        return AMR * np.mean(diff ** 2), diff / len(diff)

    def backward(self, dpreds, hs, cache):
        # gradient through linear head
        min_T    = min(dpreds.shape[0], hs.shape[0])
        dpreds   = dpreds[:min_T, :]    # (min_T, out)
        hs       = hs[:min_T, :]        # (min_T, hidden)

        dWy = dpreds.T @ hs                     # (out, hidden)
        dby = dpreds.sum(axis=0)
        dhs = dpreds @ self.Wy                  # (T, hidden)

        cell_grads, dx, _, _ = self.cell.backward(dhs, cache[:min_T])
        return cell_grads, {"dWy": dWy, "dby": dby}, dx

    def update(self, cell_grads, out_grads, lr=1e-3, clip=5.0):
        """SGD with gradient clipping."""
        def clip_and_step(param, grad):
            grad = np.clip(grad, -clip, clip)
            param -= lr * grad

        clip_and_step(self.cell.W,  cell_grads["dW"])
        clip_and_step(self.cell.b,  cell_grads["db"])
        clip_and_step(self.Wy,      out_grads["dWy"])
        clip_and_step(self.by,      out_grads["dby"])

    def train_step(self, x_seq, targets, lr=1e-3):
        if self.Wy is None:
            self.Wy = self.weight_shaper.weight_shaping(x_seq)

        preds, hs, cs, cache = self.forward(x_seq)

        AME = self.pipeline.AME_Encoder(x_seq)
        AMR = 1.0 / (1.0 + np.exp(-AME))

        loss, dloss = self.loss_mse(preds, targets, AMR)
        cell_grads, out_grads, _ = self.backward(dloss, hs, cache)
        self.update(cell_grads, out_grads, lr)

        return loss, preds


# ─────────────────────────────────────────────
#  LSTM Engine
# ─────────────────────────────────────────────

class LSTMEngine:
    """
    Wraps a trained LSTMNetwork and adds three confidence layers:

      Layer 1 — MC Dropout on hidden state (h)
                Perturbs h between timesteps — respects the cell's
                internal recurrence without touching W or b.
                Most faithful to this architecture.

      Layer 2 — Gate uncertainty
                Reads forget/input gate activations directly from cache.
                Low forget + high input = model is overwriting memory
                = structurally uncertain moment.

      Layer 3 — Prediction interval
                Built from validation residuals. Distribution-free,
                zero extra parameters, works on edge hardware.

    Usage:
        engine = LSTMEngine(model, dropout=0.1, n_samples=50)
        engine.calibrate(X_val, Y_val)
        result = engine.predict(x_seq, label_bins=None)
    """

    def __init__(self, pipeline: Any, model_network: LSTMNetwork, dropout: float = 0.1,
                 n_samples: int = 50):
        self.pipeline = pipeline
        self.model     = model_network
        self.dropout   = dropout
        self.n_samples = n_samples
        self.residual_std  = None   # set by calibrate()
        self.residual_mean = None

    # ── calibrate on validation set ──────────
    def calibrate(self, X_val, Y_val):
        """
        Collect residuals on clean (no-dropout) val predictions.
        Must be called before predict().
        """
        residuals = []
        for j in range(len(X_val)):
            preds, _, _, _ = self.model.forward(X_val[j])
            err = (preds[:, 0] - Y_val[j, :, 0])
            residuals.extend(err.tolist())

        residuals = np.array(residuals)
        self.residual_mean = residuals.mean()
        self.residual_std  = residuals.std()
        print(f"[=] Calibrated: residual μ={self.residual_mean:.4f} "
              f"[=] σ={self.residual_std:.4f}")

    # ── MC dropout forward ────────────────────
    def _mc_forward(self, x_seq: np.ndarray) -> Any:
        """
        One stochastic forward pass — dropout applied to h
        between every timestep, scaled to preserve expected value.
        """
        T = x_seq.shape[0]
        H = self.model.cell.hidden_size
        expected_input = self.model.cell.input_size
        p = self.dropout

        h = np.zeros(H)
        c = np.zeros(H)
        preds = []

        cell = self.model.cell

        for t in range(T):
            x  = x_seq[t]
            if x.ndim == 0:
                x = x.reshape(1)                          # fix zero-dimensional
            if x.shape[0] < expected_input:
                x = np.pad(x, (0, expected_input - x.shape[0]))   # pad if too small
            elif x.shape[0] > expected_input:
                x = x[:expected_input]                    # truncate if too large            
            xh = np.concatenate([x, h])
            z  = cell.W @ xh + cell.b

            # gate activations
            f  = sigmoid(cell._f(z))
            i  = sigmoid(cell._i(z))
            g  = np.tanh(cell._g(z))
            o  = sigmoid(cell._o(z))

            c      = f * c + i * g
            tanh_c = np.tanh(c)
            h      = o * tanh_c

            # ── dropout on h, inverted scaling ──
            mask = (np.random.rand(H) > p).astype(float) / (1.0 - p)
            h    = h * mask             # perturb hidden state only
                                        # cell state c is untouched —
                                        # preserves long-term memory
            if self.model.Wy is None:
                self.model.Wy = self.model.weight_shaper.weight_shaping(x_seq)

            pred = h @ self.model.Wy.T + self.model.by
            preds.append(pred[0])

        return np.array(preds)   # (T,)

    # ── gate uncertainty ──────────────────────
    def _gate_uncertainty(self, x_seq: np.ndarray, AMR: float) -> Any:
        """
        Structural uncertainty from gate activations.

        High uncertainty when:
          forget gate (f) is LOW  → model is erasing memory
          input gate  (i) is HIGH → model is overwriting with new info
          → transition moment, inherently harder to predict

        Returns per-timestep uncertainty in [0, 1].
        """
        _, _, cache = self.model.cell.forward(x_seq)
        gate_uncertainty = []
        for entry in cache:
            _, _, _, f, i, g, o, _, _, _ = entry
            # mean forget across hidden dims — low = erasing
            forget_instability = 1.0 - f.mean()
            # mean input — high = overwriting
            input_activity     = i.mean()
            # combined: both high = maximally uncertain transition
            u = AMR * forget_instability + AMR * input_activity
            gate_uncertainty.append(u)

        return np.array(gate_uncertainty)   # (T,)


    # empirical quantiles from actual residuals:
    def calibrate(self, X_val, Y_val):
        residuals = []
        for j in range(len(X_val)):
            preds, _, _, _ = self.model.forward(X_val[j])          
            min_T = min(preds.shape[0], Y_val[j].shape[0])
            err   = preds[:min_T, 0] - Y_val[j, :min_T, 0]         
            residuals.extend(err.tolist())

        residuals = np.array(residuals)
        self.residual_std  = residuals.std()
        self.residual_mean = residuals.mean()

        # store empirical quantiles instead of assuming normality
        self.quantiles = {
            0.90: (np.percentile(residuals, 5),
                np.percentile(residuals, 95)),
            0.95: (np.percentile(residuals, 2.5),
                np.percentile(residuals, 97.5)),
            0.99: (np.percentile(residuals, 0.5),
                np.percentile(residuals, 99.5)),
        }

    # interval to calculate prediction interval from MC mean + empirical quantiles
    def _interval(self, mc_mean, confidence_level):
        lo_bias, hi_bias = self.quantiles[confidence_level]
        return mc_mean + lo_bias, mc_mean + hi_bias


    # MC sample counting for label confidence (last timestep)
    def _label_confidence_empirical(self, mc_samples_last, label_bins):
        """
        mc_samples_last : (n_samples,) — raw MC draws at last timestep
        label_bins      : {"Good": (0, 35), "Moderate": (35, 75), ...}

        No distribution assumption — just count what fraction
        of actual MC samples land in each bin.
        """
        label_conf = {}
        n = len(mc_samples_last)

        for name, (lo, hi) in label_bins.items():
            hits = ((mc_samples_last >= lo) & (mc_samples_last < hi)).sum()
            label_conf[name] = hits / n

        return label_conf

    # LSTM training loop with confidence layers integrated into the loss and validation monitoring.
    def fit_stm(self, X, Y, epochs=50, hidden=32, lr=5e-3, seq_len=20, print_every=5):
        print("[= =] Training LSTM with confidence layers (MC dropout + gate uncertainty + prediction intervals)")
   
        model   = self.model
        AME = self.pipeline.AME_Encoder(X)
        AMR = 1.0 / (1.0 + np.exp(-AME))

        n_train = int(AMR * len(X))
        X_tr, Y_tr = X[:n_train], Y[:n_train]
        X_te, Y_te = X[n_train:], Y[n_train:]

        idx = np.arange(n_train)

        for epoch in range(1, epochs + 1):
            np.random.shuffle(idx)
            epoch_loss = 0.0
            for j in idx:
                loss, _ = model.train_step(X_tr[j], Y_tr[j], lr=lr)
                epoch_loss += loss
            epoch_loss /= n_train

            if epoch % print_every == 0 or epoch == 1:
                # validation
                val_loss = 0.0
                for j in range(len(X_te)):
                    preds, _, _, _ = model.forward(X_te[j])
                    if preds.shape != Y_te.shape:
                        if preds.shape[0] > Y_te.shape[0]:
                            Y_te = Y_te[:preds.shape[0], :]
                        if Y_te.shape[0] > preds.shape[0]:
                            preds = preds[:Y_te.shape[0], :]
                        else:
                            preds = preds[:Y_te.shape[0], :Y_te.shape[1]]

                    Y_te = Y_te[:preds.shape[0], :preds.shape[1]]
                    preds = preds[:Y_te.shape[0], :Y_te.shape[1]]   

                    val_loss += AMR * np.mean((preds - Y_te[j]) ** 2)
                val_loss /= len(X_te)
                print(f"[=] Epoch {epoch:>4}/{epochs}  "
                    f"[=] train_loss={epoch_loss:.6f}  val_loss={val_loss:.6f}")

        print("[=] Training complete!")
        print(f"[=] Final val loss: {val_loss:.6f}")

    def lstm_optimal_samples(self, engine, x_seq, tolerance=0.005, max_n=500):
        """
        Run increasing n_samples until std estimate stabilizes.
        Stable = change in std < tolerance between consecutive checks.
        """
        prev_std = None
        for n in range(10, max_n, 10):
            samples = np.stack([
                engine._mc_forward(x_seq) for _ in range(n)
            ])
            current_std = samples.std(axis=0).mean()
            if prev_std is not None:
                delta = abs(current_std - prev_std)
                print(f"  n={n:>4}  std={current_std:.5f}  delta={delta:.5f}")
                if delta < tolerance:
                    print(f"  → Converged at n={n}")
                    return n
            prev_std = current_std
        return max_n


    def derive_bins_from_data(self, y_values, n_bins=4, labels=None):
        """
        Use percentiles of actual data to set boundaries.
        Guarantees roughly equal sample count per bin —
        avoids empty bins on skewed distributions.
        """
        unique_vals = np.unique(y_values)

        # if binary or near-binary — skip percentiles entirely
        if len(unique_vals) <= 2:
            if labels is None:
                labels = ["Negative", "Positive"]
            return {
                labels[0]: (float(unique_vals[0]) - 1e-6,
                            float(unique_vals[0]) + 1e-6),
                labels[1]: (float(unique_vals[-1]) - 1e-6,
                            float(unique_vals[-1]) + 1e-6),
            }

        # if highly skewed — need to derive from non-dominant values
        dominant_val  = float(np.percentile(y_values, 50))
        dominant_frac = (y_values == dominant_val).mean()

        if dominant_frac > 0.5:
            # majority is one value — use non-dominant for percentiles
            non_dominant = y_values[y_values != dominant_val]
            percentiles  = np.linspace(0, 100, n_bins)
            boundaries   = np.percentile(non_dominant, percentiles)

            if labels is None:
                labels = ["Base"] + [f"Level_{i+1}" for i in range(n_bins-1)]

            bins = {"Base": (-1e-6, float(boundaries[0]))}
            for i in range(n_bins - 1):
                lo = boundaries[i]
                hi = boundaries[i+1] if i < n_bins-2 else boundaries[i+1]*1.5
                bins[labels[i+1]] = (float(lo), float(hi))
            return bins

        # normal case — standard percentile approach
        percentiles = np.linspace(0, 100, n_bins + 1)
        boundaries  = np.percentile(y_values, percentiles)
        if labels is None:
            labels = [f"Level_{i+1}" for i in range(n_bins)]
        bins = {}
        for i in range(n_bins):
            lo = boundaries[i]
            hi = boundaries[i+1] if i < n_bins-1 else boundaries[i+1]*1.5
            bins[labels[i]] = (float(lo), float(hi))
        return bins


    # ── main predict ─────────────────────────
    def predict(self, x_seq: np.ndarray,
                label_bins: dict = None,
                confidence_level: float = 0.90) -> Any:
        """
        Full confidence-aware prediction.

        Args:
            x_seq          : (T, input_size)
            label_bins     : optional dict defining label thresholds
                             e.g. {"Low": (-1, -0.33),
                                   "Mid": (-0.33, 0.33),
                                   "High": (0.33, 1.0)}
            confidence_level: for prediction interval (default 90%)

        Returns dict with:
            prediction     : point estimate (T,)
            mc_mean        : MC dropout mean (T,)
            mc_std         : MC dropout std  (T,)
            mc_confidence  : per-timestep confidence in [0,1]
            gate_uncertainty: structural uncertainty (T,)
            interval_low   : lower bound of prediction interval (T,)
            interval_high  : upper bound of prediction interval (T,)
            label_confidence: {label: probability} if label_bins given
            overall        : single scalar confidence for last timestep
        """
        assert self.residual_std is not None, \
            "Call calibrate(X_val, Y_val) before predict()"

        # ── point prediction ──────────────────
        preds_clean, _, _, _ = self.model.forward(x_seq)
        AME = self.pipeline.AME_Encoder(x_seq)  # geometric complexity scalar
        AMR = 1.0 / (1.0 + np.exp(-AME))  # abstract modelling rate
        point = preds_clean[:, 0]   # (T,)

        # ── MC dropout sampling ───────────────
        samples = np.stack([
            self._mc_forward(x_seq) for _ in range(self.n_samples)
        ])  # (n_samples, T)


        mc_mean = samples.mean(axis=0)   # (T,)
        mc_std  = samples.std(axis=0)    # (T,)

        # confidence: tight distribution = high confidence
        # normalize std by typical residual std so scale is meaningful
        normalized_std = mc_std / (self.residual_std + 1e-8)
        mc_confidence  = np.exp(-normalized_std)   # (T,) in (0,1]

        # ── gate uncertainty ──────────────────
        gate_unc = self._gate_uncertainty(x_seq, AMR)   # (T,)

        # ── prediction interval ───────────────
        total_std = np.sqrt(mc_std**2 + self.residual_std**2)        
        low, high = self._interval(mc_mean, confidence_level)

        # ── label confidence (last timestep) ──
        label_conf = None
        if label_bins is not None:
            label_conf = self._label_confidence_empirical(samples[:, -1], 
                          label_bins)

            # renormalize
            total_p = sum(label_conf.values()) + 1e-8
            label_conf = {k: v/total_p for k, v in label_conf.items()}

        # ── overall scalar confidence ─────────
        # weighted combination of MC confidence and gate stability
        gate_stability = 1.0 - gate_unc[-1]           # high = stable
        overall = AMR * mc_confidence[-1] + \
                  self.pipeline.confidence_threshold * gate_stability     

        return {
            "prediction"      : point,
            "mc_mean"         : mc_mean,
            "mc_std"          : mc_std,
            "mc_confidence"   : mc_confidence,
            "gate_uncertainty": gate_unc,
            "interval_low"    : low,
            "interval_high"   : high,
            "label_confidence": label_conf,
            "overall"         : overall,
        }
        
    # ─────────────────────────────────────────────
    #  Architecture summary helper
    # ─────────────────────────────────────────────

    def architectural_summary(self, model: LSTMNetwork):
        H = model.cell.hidden_size
        I = model.cell.input_size
        O = model.Wy.shape[0]
        W_params  = model.cell.W.size + model.cell.b.size
        Wy_params = model.Wy.size + model.by.size
        total     = W_params + Wy_params

        print("\n┌─────────────────────────────────────────┐")
        print("  │          LSTM Architecture Summary      │")
        print("  ├─────────────────────────────────────────┤")
        print(f" │  Input  size   : {I:<24}│")
        print(f" │  Hidden size   : {H:<24}│")
        print(f" │  Output size   : {O:<24}│")
        print(f" │  LSTM   params : {W_params:<24,}│")
        print(f" │  Linear params : {Wy_params:<24,}│")
        print(f" │  Total  params : {total:<24,}│")
        print("  └─────────────────────────────────────────┘")


from .geometry import GeometricWeightShaping

class Activation:
    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)

    @staticmethod
    def sigmoid(x):
        eps = 1e-5
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        eps = 1e-5
        s = Activation.sigmoid(x)
        return s * (1.0 - s)

    @staticmethod
    def softmax(x):
        # numerical stability
        if x.ndim > 1:
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        else:
            exp_x = np.exp(x - np.max(x, keepdims=True))
            return exp_x / np.sum(exp_x, keepdims=True)


class Loss:
    @staticmethod
    def categorical_crossentropy(y_true, y_pred):
        eps = 1e-9
        y_pred = np.clip(y_pred, eps, 1 - eps)
        try:
            loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        except:
            subnet_y_pred = y_pred[:, :y_true.shape[1]]
            subnet_y_true = y_true[:, :subnet_y_pred.shape[1]]

            loss = -np.mean(np.sum(subnet_y_true * np.log(subnet_y_pred + eps), axis=1))
        
        return loss

    @staticmethod
    def softmax_crossentropy_derivative(y_true, y_pred):
        try:
            cross_ent = (y_pred - y_true) / y_true.shape[0]
        except:
            subnet_y_pred = y_pred[:, :y_true.shape[1]]
            subnet_y_true = y_true[:, :subnet_y_pred.shape[1]]

            cross_ent = (subnet_y_pred - subnet_y_true) / subnet_y_true.shape[0]
        return cross_ent


class Dense:
    def __init__(self, x, input_size, output_size, activation=None):

        self.special_weight = GeometricWeightShaping(input_size, output_size)
        self.W = self.special_weight.weight_shaping(x)

        self.b = np.zeros((1, output_size))
        self.activation_name = activation
       

        if activation:
            self.activation = getattr(Activation, activation)
            self.activation_derivative = getattr(Activation, activation + "_derivative")
        else:
            self.activation = None
            self.activation_derivative = None

    def multi_modal_linear_transformation(self, x):
        if len(x.shape) > 1 and x.shape[1] != self.W.shape[0]:
            V1, V2 = x.shape[0], x.shape[1]            
            try:
                self.W = self.W[:V2, :]
            except:
                self.special_weight = GeometricWeightShaping(V2, V1)
                self.W = self.special_weight.weight_shaping(x)
        try:
            try:
                z = np.dot(x, self.W) + self.b
            except:
                subnet_W = self.W[:x.shape[1], :x.shape[0]]

                sub_z = np.dot(x, subnet_W)
                sub_b = self.b[:sub_z.shape[1], :sub_z.shape[0]]

                z = sub_z + sub_b

        except:
            try:
                subnet_W = self.W[:x.shape[1]:, :x.shape[0]]
                sub_z = np.dot(x, subnet_W)
            except:
                weight = self.W

                try:
                    subnet_x = x[:, :weight.shape[0]]
                    subnet_W = weight[:x.shape[1], :]                    
                except:
                    subnet_x = x[:weight.shape[0]]
                    subnet_W = weight[:x.shape[0]]

                sub_z = np.dot(subnet_x, subnet_W)

            try:
                subnet_B = self.b[:sub_z.shape[0], :sub_z.shape[1]]
            except:
                subnet_B = self.b[:sub_z.shape[0]]
                
            z = sub_z + subnet_B      

        return z


    def forward(self, x):
        self.x = x
        self.z = self.multi_modal_linear_transformation(x)

        if self.activation:
            self.a = self.activation(self.z)
        else:
            self.a = self.z

        return self.a


    def backward(self, da, lr):
        eps = 1e-5

        if self.activation_derivative: 
            dz = da * self.activation_derivative(self.z)
        else:
            dz = da

        dW = np.dot(self.x.T, dz)
        db = np.sum(dz, axis=0, keepdims=True)
        dx = np.dot(dz, self.W.T)

        self.W -= lr * dW 
        self.b -= lr * db 
      
        return dx


class SoftmaxOutput:
    def forward(self, x):
        self.out = Activation.softmax(x)
        return self.out

    def backward(self, dL_dZ):
        # gradient already computed as y_pred - y_true
        return dL_dZ


# enhanced MLP with focused forward and backward for better handling of data with varying geometric complexity,
# allowing it to complement the transformer module in the ensemble method.
# providing robust performance across a wider range of data complexities by dynamically adjusting its learning focus based on the data's geometric properties.
# focused forward and backward allows the MLP to adaptively concentrate on abundant layers during training, enhancing its ability to learn from data with varying geometric complexity for flexible applications.
# and providing a complementary learning dynamic when combined with the transformer in the ensemble.
# source of geometric weight research: https://github.com/Micro-Novelty/Specialized-MLP-for-noise-robustness
