# THIS IS THE SOURCE CODE OF ABSTRACTINTEGRATEDMODULE
# YOU ARE HEREBY GRANTED TO AUDIT, REVIEW, AND INITIATE PULL REQUESTS AND ISSUES
# LICENSE: MIT, PROVIDED.
# ──────────────────────────────────────────────────────────────
# Part of AbstractIntegratedPipeline — split for readability.
# See __init__.py for full import map.
# ──────────────────────────────────────────────────────────────

import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime, timedelta
import sqlite3
import json
import joblib
import ast
import re
import sys
import threading
import time
from collections import deque
import socket
import pickle
from collections import defaultdict
import hashlib
import ssl
import os
import asyncio
import queue
import threading
import uuid
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Tuple, Optional, Dict, List
from datetime import datetime, timedelta
from enum import IntEnum, Enum
from collections import deque
import traceback
from concurrent.futures import TimeoutError as FutureTimeoutError
import secrets
import ipaddress
from functools import wraps
import hmac
import aiohttp
import psutil
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)



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

    # _________ forward method for cell class _____________
    def forward(self, x_seq: np.ndarray, h0=None, c0=None):
        T              = x_seq.shape[0]
        H              = self.hidden_size
        expected_input = self.input_size

        h = np.zeros(H) if h0 is None else h0.copy()
        c = np.zeros(H) if c0 is None else c0.copy()

        hs    = np.zeros((T, H))
        cs    = np.zeros((T, H))
        cache = []

        # preallocate xh buffer once
        xh = np.empty(expected_input + H)

        for t in range(T):
            x = x_seq[t]
            if x.ndim == 0:
                x = x.reshape(1)
            if x.shape[0] < expected_input:
                x = np.pad(x, (0, expected_input - x.shape[0]))
            elif x.shape[0] > expected_input:
                x = x[:expected_input]

            # write into buffer, no allocation
            xh[:expected_input] = x
            xh[expected_input:] = h

            z      = self.W @ xh + self.b
            H1, H2, H3 = H, H * 2, H * 3

            # direct slices, no method calls
            f      = sigmoid(z[:H1])
            i      = sigmoid(z[H1:H2])
            g      = np.tanh(z[H2:H3])
            o      = sigmoid(z[H3:])

            c_new  = f * c + i * g
            tanh_c = np.tanh(c_new)
            h_new  = o * tanh_c

            # store copies — h/c will be overwritten next iteration 
            cache.append((x.copy(), h.copy(), c.copy(),
                        f, i, g, o, c_new, tanh_c, xh.copy()))
            h, c = h_new, c_new
            hs[t] = h
            cs[t] = c

        return hs, cs, cache

    # ________ backward method for Cell class __________
    def backward(self, dhs: np.ndarray, cache,
                dh_next=None, dc_next=None, T_limit=None):
        # T_limit avoids slicing cache list externally
        T = T_limit if T_limit is not None else len(cache)
        H = self.hidden_size

        dW     = np.zeros_like(self.W)
        db     = np.zeros_like(self.b)
        dh     = np.zeros(H) if dh_next is None else dh_next.copy()
        dc     = np.zeros(H) if dc_next is None else dc_next.copy()
        dx_seq = np.zeros((T, self.input_size))

        # preallocate dz buffer once
        dz     = np.empty(4 * H)
        H1, H2, H3 = H, H * 2, H * 3

        for t in reversed(range(T)):
            x, h_prev, c_prev, f, i, g, o, c_new, tanh_c, xh = cache[t]

            dh_total = dhs[t] + dh

            do     = dh_total * tanh_c
            dtanhc = dh_total * o
            dc_new = dtanhc * tanh_deriv(tanh_c) + dc

            df = dc_new * c_prev
            di = dc_new * g
            dg = dc_new * i
            dc = dc_new * f

            # write into preallocated dz buffer
            dz[:H1]  = df * sigmoid_deriv(f)
            dz[H1:H2] = di * sigmoid_deriv(i)
            dz[H2:H3] = dg * tanh_deriv(g)
            dz[H3:]   = do * sigmoid_deriv(o)

            # nplace accumulation, no intermediate allocation
            dW  += np.outer(dz, xh)   # unavoidable alloc but outer is C-level
            db  += dz

            dxh        = self.W.T @ dz
            dx_seq[t]  = dxh[:self.input_size]
            dh         = dxh[self.input_size:]

        return {"dW": dW, "db": db}, dx_seq, dh, dc


# ─────────────────────────────────────────────
#  LSTM Network (cell + linear output head)
# ─────────────────────────────────────────────
class LSTMNetwork:
    def __init__(self, pipeline, input_size, hidden_size, output_size, seed=0):
        self.cell         = LSTMCell(input_size, hidden_size, seed)
        self.weight_shaper = GeometricWeightShaping(output_size, hidden_size)
        self.Wy           = None
        self.by           = np.zeros(output_size)
        self.pipeline     = pipeline
        self._trained     = False

    # forward method to calculate proper weight for prediction and training.
    def forward(self, x_seq):
        # also Wy init only here, removed from train_step
        if self.Wy is None:
            self.Wy = self.weight_shaper.weight_shaping(x_seq)
        hs, cs, cache = self.cell.forward(x_seq)
        preds = hs @ self.Wy.T + self.by   # (T, output_size)
        return preds, hs, cs, cache

    # calculate loss of MSE (Mean squared error.)
    def loss_mse(self, preds, targets, AMR):
        # proper and correct shape alignment
        min_T = min(preds.shape[0], targets.shape[0])
        min_F = min(preds.shape[1], targets.shape[1])
        preds   = preds[:min_T, :min_F]
        targets = targets[:min_T, :min_F]

        diff = preds - targets
        loss = (1.0 - AMR) * np.mean(diff ** 2)
        dloss = diff / (min_T * min_F)   # FIX 3 — normalize by full element count
        return loss, dloss

    # backward method for the network to calculate proper weights with cell backward
    def backward(self, dpreds, hs, cache):
        min_T  = min(dpreds.shape[0], hs.shape[0])
        dpreds = dpreds[:min_T]
        hs     = hs[:min_T]

        dWy = dpreds.T @ hs
        dby = dpreds.sum(axis=0)
        dhs = dpreds @ self.Wy

        # pass min_T directly, need to avoid creating a sliced list
        cell_grads, dx, _, _ = self.cell.backward(dhs, cache, T_limit=min_T)
        return cell_grads, {"dWy": dWy, "dby": dby}, dx

    # update ensured proper gradient clipping
    def update(self, cell_grads, out_grads, lr=1e-3, clip=5.0):
        def clip_and_step(param, grad):
            np.clip(grad, -clip, clip, out=grad)  
            param -= lr * grad

        clip_and_step(self.cell.W, cell_grads["dW"])
        clip_and_step(self.cell.b, cell_grads["db"])
        clip_and_step(self.Wy,     out_grads["dWy"])
        clip_and_step(self.by,     out_grads["dby"])

    # train step for each LSTM fitting method 
    def train_step(self, x_seq, targets, lr=1e-3, AMR=None):
        # accept precomputed AMR 
        if AMR is None:
            AME = self.pipeline.AME_Encoder(x_seq)
            AMR = 1.0 / (1.0 + np.exp(-AME))

        preds, hs, cs, cache = self.forward(x_seq)
        loss, dloss          = self.loss_mse(preds, targets, AMR)
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
    def calibrate_residual(self, X_val, Y_val):
        if len(X_val) == 0:
            self.residual_mean = 0.0
            self.residual_std  = 1.0
            return

        confidence_errors = []

        for j in range(len(X_val)):
            preds, _, _, _ = self.model.forward(X_val[j])

            pred_vals = preds[:, 0] if preds.ndim > 1 else preds

            # get true class
            true_vals = Y_val[j, :, 0] if Y_val[j].ndim > 1 else Y_val[j]
            min_len   = min(len(pred_vals), len(true_vals))

            # sigmoid to get predicted probability
            pred_prob = 1.0 / (1.0 + np.exp(-pred_vals[:min_len]))
            true_bin  = true_vals[:min_len].astype(float)

            # confidence error — how wrong was the predicted probability
            # correct prediction   → low error
            # wrong prediction     → high error
            err = np.abs(pred_prob - true_bin)
            confidence_errors.extend(err.tolist())

        errors = np.array(confidence_errors)

        # IQR outlier removal
        q25, q75     = np.percentile(errors, [25, 75])
        iqr          = q75 - q25
        mask         = (errors >= q25 - 1.5 * iqr) & (errors <= q75 + 1.5 * iqr)
        clean        = errors[mask] if mask.sum() > 0 else errors

        self.residual_mean = float(clean.mean())
        self.residual_std  = float(max(clean.std(), 1e-6))

        # floor std on small n — can't trust calibration with few samples
        if len(X_val) < 20:
            self.residual_std = max(self.residual_std, 0.1)
            print(f'[!] Small val set ({len(X_val)} samples) — flooring σ to 0.1')

        self.calibration_coverage   = float(mask.mean())
        self.n_calibration_samples  = len(X_val)

        print(f"[=] Calibrated: residual μ={self.residual_mean:.4f} "
            f"σ={self.residual_std:.4f} "
            f"coverage={self.calibration_coverage:.1%} "
            f"n={self.n_calibration_samples}")


    # ── MC dropout forward ────────────────────
    def _mc_forward(self, x_seq: np.ndarray) -> np.ndarray:
        T            = x_seq.shape[0]
        H            = self.model.cell.hidden_size
        expected_input = self.model.cell.input_size
        p            = self.dropout
        cell         = self.model.cell
        W            = cell.W
        b            = cell.b
        H1, H2, H3  = H, H * 2, H * 3   # slice boundaries precomputed

        # Wy check once before loop
        if self.model.Wy is None:
            self.model.Wy = self.model.weight_shaper.weight_shaping(x_seq)
        Wy = self.model.Wy
        by = self.model.by

        h     = np.zeros(H)
        c     = np.zeros(H)
        xh    = np.empty(expected_input + H)  # preallocate concat buffer
        preds = np.empty(T)                   # preallocate output

        # precompute dropout scale factor
        inv_keep = 1.0 / (1.0 - p)

        for t in range(T):
            x = x_seq[t]

            # shape alignment
            if x.ndim == 0:
                x = x.reshape(1)
            if x.shape[0] < expected_input:
                x = np.pad(x, (0, expected_input - x.shape[0]))
            elif x.shape[0] > expected_input:
                x = x[:expected_input]

            # need to write into preallocated buffer instead of np.concatenate
            xh[:expected_input] = x
            xh[expected_input:] = h

            z = W @ xh + b                    # (4H,)

            # direct slices instead of method calls
            f      = sigmoid(z[:H1])
            i      = sigmoid(z[H1:H2])
            g      = np.tanh(z[H2:H3])
            o      = sigmoid(z[H3:])

            c      = f * c + i * g
            tanh_c = np.tanh(c)
            h      = o * tanh_c

            # precomputed inv_keep, inplace mask application
            mask = (np.random.rand(H) > p) * inv_keep
            h   *= mask

            preds[t] = (h @ Wy.T + by)[0]

        return preds   # (T,)

    # ── gate uncertainty for LSTM prediction──────────────────────
    def _gate_uncertainty(self, x_seq: np.ndarray, AMR: float) -> np.ndarray:
        """
        Structural uncertainty from gate activations.
        Vectorized — no Python loop over timesteps.
        """
        _, _, cache = self.model.cell.forward(x_seq)

        # cache[t] = (x, h_prev, c_prev, f, i, g, o, c_new, tanh_c, xh)
        # need to extract f and i directly as stacked arrays — shape (T, H)
        T = len(cache)
        
        if T == 0:
            return np.array([0.0])

        # vectorized extraction — one pass instead of T unpacks
        f_all = np.empty((T, cache[0][3].shape[0]))  # (T, H)
        i_all = np.empty((T, cache[0][4].shape[0]))  # (T, H)

        for t, entry in enumerate(cache):
            f_all[t] = entry[3]   # forget gate
            i_all[t] = entry[4]   # input gate

        # vectorized computation — no per-timestep Python arithmetic
        forget_instability = 1.0 - f_all.mean(axis=1)   # (T,)
        input_activity     = i_all.mean(axis=1)           # (T,)

        # precompute scalar factor once
        scale = 1.0 - AMR
        gate_uncertainty = scale * (forget_instability + input_activity)

        return np.clip(gate_uncertainty, 0.0, 1.0)      

    # empirical quantiles from actual residuals
    def calibrate(self, X_val, Y_val):
        if len(X_val) == 0:
            self.residual_mean = 0.0
            self.residual_std  = 1.0
            self.quantiles     = {}
            return

        all_errors = []
        for j in range(len(X_val)):
            preds, _, _, _ = self.model.forward(X_val[j])
            y         = Y_val[j]
            pred_vals = preds[:, 0] if preds.ndim > 1 else preds
            true_vals = y[:, 0]    if y.ndim > 1    else y
            min_T     = min(len(pred_vals), len(true_vals))
            all_errors.append(pred_vals[:min_T] - true_vals[:min_T])

        residuals = np.concatenate(all_errors)
        n         = len(residuals)

        self.residual_mean = float(residuals.mean())
        self.residual_std  = float(max(residuals.std(), 1e-6))

        # adapt confidence levels to sample size
        # small n → only compute what's statistically supportable
        if n < 20:
            # only 90% interval is reliable — use 10th/90th percentile
            # wide enough to be honest about uncertainty
            levels = [10.0, 90.0]
            p = np.percentile(residuals, levels)
            self.quantiles = {
                0.90: (float(p[0]), float(p[1])),
                0.95: (float(p[0]), float(p[1])),  # same as 90% — honest, not fake precision
                0.99: (float(p[0]), float(p[1])),
            }
            print(f'[!] n={n} too small for tail quantiles — '
                f'using 10th/90th for all intervals')

        elif n < 50:
            # 90% and 95% supportable, 99% not reliable
            levels = [2.5, 5.0, 95.0, 97.5]
            p = np.percentile(residuals, levels)
            self.quantiles = {
                0.90: (float(p[1]), float(p[2])),
                0.95: (float(p[0]), float(p[3])),
                0.99: (float(p[0]), float(p[3])),  # same as 95% — honest
            }
            print(f'[!] n={n} insufficient for 99% interval — '
                f'using 95% as proxy')

        else:
            # full precision justified
            levels = [0.5, 2.5, 5.0, 95.0, 97.5, 99.5]
            p = np.percentile(residuals, levels)
            self.quantiles = {
                0.90: (float(p[2]), float(p[3])),
                0.95: (float(p[1]), float(p[4])),
                0.99: (float(p[0]), float(p[5])),
            }

        # floor std on small n
        if n < 20:
            self.residual_std = max(self.residual_std, 0.1)

        print(f"[=] Calibrated: μ={self.residual_mean:.4f} "
            f"σ={self.residual_std:.4f} "
            f"n={n} "
            f"90%=[{self.quantiles[0.90][0]:.4f}, {self.quantiles[0.90][1]:.4f}]")

    # interval to calculate prediction interval from MC mean + empirical quantiles
    def _interval(self, mc_mean, confidence_level):
        # flatten mc_mean safely — handles scalar, 0-d array, or 1-d array
        mc_scalar = float(np.asarray(mc_mean).flat[0])

        if confidence_level not in self.quantiles:
            available = sorted(self.quantiles.keys())
            if not available:
                return mc_scalar - self.residual_std, mc_scalar + self.residual_std
            confidence_level = min(available, key=lambda k: abs(k - confidence_level))
            print(f'[!] Confidence level not found, using closest: {confidence_level}')

        lo_bias, hi_bias = self.quantiles[confidence_level]

        lo = mc_scalar + float(np.asarray(lo_bias).flat[0])
        hi = mc_scalar + float(np.asarray(hi_bias).flat[0])

        if lo > hi:
            lo, hi = hi, lo

        return lo, hi

    # MC sample counting for label confidence (last timestep)
    def _label_confidence_empirical(self, mc_samples_last, label_bins):
        """
        mc_samples_last : (n_samples,) — raw MC draws at last timestep
        label_bins      : {"Good": (0, 35), "Moderate": (35, 75), ...}
        """
        n = len(mc_samples_last)
        if n == 0:
            return {name: 0.0 for name in label_bins}

        names  = list(label_bins.keys())
        bounds = np.array(list(label_bins.values()))  # (n_bins, 2)

        # vectorized — broadcast (n_samples,) against (n_bins, 2)
        # samples shape: (1, n_samples), bounds shape: (n_bins, 1)
        samples = mc_samples_last[np.newaxis, :]          # (1, n_samples)
        lo      = bounds[:, 0, np.newaxis]                # (n_bins, 1)
        hi      = bounds[:, 1, np.newaxis]                # (n_bins, 1)

        hits = ((samples >= lo) & (samples < hi)).sum(axis=1)  # (n_bins,)
        probs = hits / n                                        # (n_bins,)

        return dict(zip(names, probs.tolist()))

    # LSTM training loop with confidence layers integrated into the loss and validation monitoring.
    def fit_stm(self, X, Y, epochs=50, hidden=32, lr=5e-3, seq_len=20, print_every=5):
        print("[= =] Training LSTM with confidence layers (MC dropout + gate uncertainty + prediction intervals)")
   
        model   = self.model
        AME = self.pipeline.AME_Encoder(X)
        AMR = 1.0 / (1.0 + np.exp(-AME))

        n_train = int((1.0 - AMR) * len(X))
        X_tr, Y_tr = X[:n_train], Y[:n_train]
        X_te, Y_te = X[n_train:], Y[n_train:]

        idx = np.arange(n_train)

        for epoch in range(1, epochs + 1):
            np.random.shuffle(idx)
            epoch_loss = 0.0
            for j in idx:
                loss, _ = model.train_step(X_tr[j], Y_tr[j], lr=lr, AMR=AMR)
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
        print('===== CALIBRATION METHOD =====')
        self.calibrate_residual(X_te, Y_te)

    # get optimal lstm samples amount for the model to process
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

    # derive local bins for flexibility in scarce dataset
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


    # ── main predict function for the whole network ─────────────────────────
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
        # compute interval for every timestep — always returns (T,) arrays
        interval_low  = np.empty(len(mc_mean))
        interval_high = np.empty(len(mc_mean))
        for t in range(len(mc_mean)):
            interval_low[t], interval_high[t] = self._interval(mc_mean[t], confidence_level)

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
        overall = (1.0 - AMR) * mc_confidence[-1] + \
                  self.pipeline.confidence_threshold * gate_stability     

        return {
            "prediction"      : point,
            "mc_mean"         : mc_mean,
            "mc_std"          : mc_std,
            "mc_confidence"   : mc_confidence,
            "gate_uncertainty": gate_unc,
            "interval_low"    : interval_low,
            "interval_high"   : interval_high,
            "label_confidence": label_conf,
            "overall"         : overall,
        }
        
    # ─────────────────────────────────────────────
    #  Architecture summary helper to visualize results
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




# geometric weight shaping provides the model with a robust geometric complexity alignment>
#  allowing it to better process data with varying geometric complexity, and providing a more stable training process in scarce data environment. 
# It can be used as a general weight initialization and shaping method for various models, especially in scenarios where data geometry is complex and data is scarce.

