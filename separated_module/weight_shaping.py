"""
weight_shaping.py
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
from . import mlp
from . import transformer

class GeometricWeightShaping:
    """
    Generates data-dependent ("shaped") weight/context matrices by analyzing
    the geometric structure of an input batch, rather than drawing weights
    from a fixed, data-agnostic distribution (e.g. plain Xavier/He init).

    The core idea: look at how the input data is spread out in feature
    space -- how anisotropic it is, how much of its variance concentrates
    in a few directions (spectral/eigenvalue structure), and how "surprising"
    it is to reconstruct (Abstract Modelling Error, AME) -- and use those
    signals to pick the *scale* of a random weight matrix. Data that is
    simple/low-complexity gets weights sampled from a narrower, more
    conservative range; data that is highly anisotropic or spectrally
    concentrated gets a differently-scaled range. This is meant to give
    downstream layers (transformer.Transformer/mlp.MLP projections, etc.) an initial
    weight distribution that is already "aligned" with the geometry of
    the data it will process, instead of starting from geometry-agnostic
    noise.

    High-level pipeline (see `weight_shaping` -> `abstract_weight_shaping`):
        1. Measure how anisotropic the input is (`anisotropy_measurement`).
        2. Measure how much abstraction/reconstruction error the data
           implies (`AME_Encoder`) and convert it into an "abstract
           modelling rate" (AMR) via a activations.sigmoid squashing.
        3. Look at the eigenvalue/energy spectrum of the (noise-augmented)
           covariance matrix to estimate an effective rank `k`
           (`eigenvalue_encoder`).
        4. Compare the spectral signature of the real data against a
           perturbed/noisy version of itself (`spectral_similarity`) to
           get a sense of how stable the spectrum is.
        5. Combine all of the above into a single scalar
           "abstraction efficiency", which is used as the upper bound of
           a uniform distribution that the final `(input_size, output_size)`
           weight matrix is drawn from.
        6. Normalize the resulting matrix to roughly [-1, 1].

    None of this uses gradient-based learning -- it's a heuristic,
    statistics-driven initializer/context generator that several other
    classes in this file (e.g. transformer.Transformer, mlp.MLP) call into when they want
    a "geometry-aware" starting point for their weights.

    Numerical safety: most formulas below add small epsilon terms and/or
    have NaN/Inf fallbacks, because ratios of near-zero eigenvalues,
    anisotropy, etc. can blow up in degenerate cases (e.g. near-constant
    input, single-sample batches). Where you see `if np.isnan(...) or
    np.isinf(...)`, that's a deliberate fallback to a safer, simpler
    formula rather than a bug.
    """

    def __init__(self, input_size, output_size):
        # Target shape of the weight/context matrix this instance will
        # eventually produce via `weight_shaping` / `abstract_weight_shaping`.
        self.input_size = input_size
        self.output_size = output_size


    def eigenvalue_encoder(self, x):
        """
        Estimate the effective spectral rank `k` of the input's covariance
        structure, and produce a random "floating point" perturbation
        matrix whose spread is derived from that spectral structure.

        Steps:
            - Compute AME (`AME_Encoder`) and squash it into AMR (abstract
              modelling rate) via a activations.sigmoid, giving a value in (0, 1).
            - Compute the average vector magnitude of the raw input.
            - Compute anisotropy of the raw input (`anisotropy_measurement`).
            - Augment the data with uniform "structured noise" of similar
              magnitude, stack it with the original data, and compute the
              covariance matrix of the combined set.
            - Eigendecompose that covariance matrix and sort eigenvalues
              descending; take the cumulative normalized energy
              (explained-variance curve).
            - `k` = number of leading eigenvalues needed to reach 90% of
              total energy (i.e. an effective-rank / PCA-style dimension
              estimate).
            - Combine `k`, anisotropy, energy-curve consistency, and AMR
              through a chain of ratio expressions (trA/trB/trC) to get a
              scalar that bounds a final uniform random matrix
              (`floating_point`), which mirrors the shape used for the
              stacked covariance input `X`.
            - If the trC chain degenerates to NaN/Inf (which can happen
              when denominators like `1 - trB**2` approach zero), fall
              back to progressively simpler formulas.

        Args:
            x: Input array-like, shape (n_samples, ...). Will be
               flattened to 2D (n_samples, n_features) if higher-dim.

        Returns:
            k (int): Estimated effective spectral rank (# components
                needed to explain ~90% of variance).
            floating_point (np.ndarray): Random matrix, same shape as the
                internal stacked `X`, sampled uniformly between
                `min(trC, 0)` and `max(trC, 0)`. Used downstream as a
                perturbation reference in `abstract_weight_shaping`.
            structured_noise (np.ndarray): The synthetic uniform noise
                that was stacked onto the raw data to compute covariance;
                returned so callers can reuse the exact same noise (e.g.
                in `spectral_similarity`) for a consistent comparison.
        """
        eps = 1e-5
        raw_X = np.asarray(x)
        AME = self.AME_Encoder(raw_X)
        AMR = 1.0 / (1.0 + np.exp(-AME)) + eps
        mag = np.mean(np.linalg.norm(raw_X, axis=-1))

        if raw_X.ndim > 2:
            raw_X = raw_X.reshape(raw_X.shape[0], -1)

        anisotropy = self.anisotropy_measurement(raw_X)

        # Synthetic noise at the same magnitude as the real data, used to
        # "stress test" the covariance/eigen structure rather than relying
        # on the (possibly too-small) real sample alone.
        structured_noise = np.random.uniform(0, mag, size=raw_X.shape)
        X = np.vstack((raw_X, structured_noise))
        if X.ndim == 2 and X.shape[1] == 1:
            # Single-feature case: vstack just duplicates rows, so instead
            # widen the feature dimension via hstack to actually mix in
            # the noise for a meaningful covariance matrix.
            X = np.hstack((raw_X, structured_noise))

        cov = np.cov(X, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]

        # Cumulative explained-variance ("energy") curve, then squashed
        # through a activations.sigmoid purely to measure how "jagged" vs. "smooth"
        # the growth curve is (energy_consistency = its std-dev).
        energy = np.cumsum(eigenvalues) / np.sum(eigenvalues)
        energy_sigmoid_growth = 1.0 / (1.0 + np.exp(-energy))
        energy_consistency = np.std(energy_sigmoid_growth)
        k = np.searchsorted(energy, 0.90) + 1     # +1 converts 0-based index to count

        # trA/trB/trC: a hand-tuned chain of ratios combining rank (k),
        # anisotropy, energy-curve consistency and AMR into one scalar
        # that will bound the random matrix below. Each stage divides by
        # a quantity that can approach zero, hence the eps terms and the
        # NaN/Inf fallback chain that follows.
        trA = k / (1.0 - anisotropy) + eps
        trB = (1/2 + energy_consistency) / (1.0 + trA**2)
        trC = (1/6 + AMR) / (1.0 - trB**2) + eps

        if np.isnan(trC) or np.isinf(trC):
            # Primary formula degenerated (e.g. trB ~= 1) -> fall back to
            # a simpler anisotropy-driven expression.
            trC = anisotropy * (trB**2 - 1.0) + eps
            if np.isnan(trC) or np.isinf(trC):
                # Still degenerate -> final, always-finite fallback.
                trC = (1.0 - AMR)

        min_val = min(trC, 0)
        max_val = max(trC, 0)
        floating_point = np.random.uniform(min_val, max_val, size=X.shape)
        return k, floating_point, structured_noise


    def spectral_signature(self, x, structured_noise, k=5):
        """
        Compute a compact "spectral signature" for `x`: the top-`k`
        normalized eigenvalues of its covariance matrix (optionally
        augmented with `structured_noise` when `x` is a single-feature
        column), i.e. a rough descriptor of how variance is distributed
        across the data's principal directions.

        This is deliberately robust to shape mismatches between `x` and
        `structured_noise` (1D vs 2D, differing row counts) since callers
        may pass noise generated for a different-shaped batch; rows are
        truncated to the common minimum rather than raising.

        Args:
            x: Input array-like, reshaped to 2D (n_samples, n_features).
            structured_noise: Extra noise features to concatenate onto
                `x` when `x` only has a single feature column (keeps the
                covariance matrix from being degenerate/rank-1).
            k (int): Number of leading eigenvalues to keep.

        Returns:
            np.ndarray of length up to `k`: the largest eigenvalues of
            the covariance matrix, normalized to sum to (approximately)
            1 (i.e. the fraction of variance each leading component
            explains). Returns zeros of length `k` (or less, if fewer
            eigenvalues exist) when the covariance can't be computed
            (too few samples, near-zero total variance, or a linear
            algebra failure).
        """
        raw_X = np.asarray(x, dtype=np.float64)

        if raw_X.ndim > 2:
            X = raw_X.reshape(raw_X.shape[0], -1)
        else:
            X = raw_X.reshape(raw_X.shape[0], -1)

        X = np.atleast_2d(X)

        if X.ndim == 2 and X.shape[1] == 1:
            # normalize structured_noise to 2D matching X's row count
            noise = np.asarray(structured_noise, dtype=np.float64)

            if noise.ndim == 1:
                # reshape to (n_samples, n_noise_features)
                # if noise length matches X's row count, treat as column vector
                if noise.shape[0] == X.shape[0]:
                    noise = noise.reshape(-1, 1)
                else:
                    # noise is a flat feature vector not aligned to X's rows —
                    # broadcast it across all rows instead of stacking blindly
                    noise = np.tile(noise.reshape(1, -1), (X.shape[0], 1))
            elif noise.ndim > 2:
                noise = noise.reshape(noise.shape[0], -1)

            # align row counts before hstack
            if noise.shape[0] != X.shape[0]:
                min_rows = min(noise.shape[0], X.shape[0])
                X     = X[:min_rows]
                noise = noise[:min_rows]
                print(f'[⚠️] spectral_signature: row count mismatch, '
                    f'truncated to {min_rows} rows')

            X = np.hstack((X, noise))

        # guard against degenerate covariance — need at least 2 samples
        if X.shape[0] < 2:
            print(f'[⚠️] spectral_signature: only {X.shape[0]} sample(s), '
                f'cannot compute covariance — returning zeros')
            return np.zeros(k)

        try:
            cov     = np.cov(X, rowvar=False, ddof=1)
            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.sort(eigvals)[::-1]
            eig_sum = eigvals.sum()
            if eig_sum <= 1e-8:
                return np.zeros(min(k, len(eigvals)))
            return eigvals[:k] / (eig_sum + 1e-8)
        except np.linalg.LinAlgError as e:
            print(f'[⚠️] spectral_signature: eigendecomposition failed: {e}')
            return np.zeros(k)


    def spectral_similarity(self, a, b, structured_noise):
        """
        Compare two inputs' spectral signatures (`spectral_signature`) and
        return a similarity score in (0, 1], where 1 means identical
        top-k eigenvalue distributions and values approach 0 as the
        signatures diverge.

        Implemented as `exp(-||sig(a) - sig(b)||)`, i.e. an RBF-style
        similarity over the two signature vectors (truncated to the
        shorter length if the two signatures differ, e.g. due to
        degenerate covariance producing fewer eigenvalues for one side).

        Args:
            a, b: The two inputs to compare (passed through
                `spectral_signature`).
            structured_noise: Noise passed along to `spectral_signature`
                for the single-feature augmentation case.

        Returns:
            float: similarity score in (0, 1].
        """
        sa = self.spectral_signature(a, structured_noise)
        sb = self.spectral_signature(b, structured_noise)
        if sa.shape != sb.shape:
            min_rows = min(sa.shape[0], sb.shape[0])

            sa = sa[:min_rows]
            sb = sb[:min_rows]

        return np.exp(-np.linalg.norm(sa - sb))

    # abstract modelling error provides the model how to better process weights when the data complexity has little geometric complexity
    def AME_Encoder(self, x):
        """
        Compute the "Abstract Modelling Error" (AME): a scalar heuristic
        for how much reconstruction/abstraction difficulty the input's
        local structure implies, based on the interplay between the
        magnitude of the raw data and the magnitude of its local gradient
        (rate of change).

        Intuition: `X_mag` captures "how big" the data is on average, and
        `mean_vector_mag` (gradient magnitude) captures "how fast it
        changes" locally. AME = log1p(X_mag) * log1p(gradient_mag) grows
        when both the data and its local variation are large -- i.e. data
        that is both large-scale and volatile is harder to abstract
        cleanly, hence a higher "error". A higher AME later gets squashed
        through a activations.sigmoid (AMR, elsewhere) to bias weight-shaping toward
        more conservative behavior; a lower AME means the data is
        "easier" and can tolerate less conservative abstraction.

        Uses the Cython-optimized `optimized_ame_encoder` when available
        and the input is already 2D; otherwise falls back to a NumPy
        `np.gradient`-based computation (which itself falls back further
        to a 10x10 sub-block if the gradient call fails on the full
        array, e.g. due to shape issues).

        Args:
            x: Input array-like.

        Returns:
            float: AME score. Returns 0.0 for empty input.
        """
        X = np.asarray(x)

        if len(X) == 0:
            print('[!] X size is 0, AME Will be replaced by minimum confidence threshold')
            return 0.0

        if _OPT_AVAILABLE and np.asarray(X).ndim == 2:
            return optimized_ame_encoder(np.asarray(X, dtype=np.float64))

        try:
            gradient = np.gradient(x)
        except:
            subnet = x[:min(10, x.shape[0]), :min(10, x.shape[1])]
            gradient = np.gradient(subnet.flatten())

        mean_vector_mag  = np.mean(np.linalg.norm(gradient, axis=-1))
        X_mag = np.mean(np.linalg.norm(X, axis=-1))
        # Regular AME Equations, higher AME provides capabilities for the model to experience errors during abstraction
        # Lower AME means lower chance for un optimal abstraction.

        AME =  np.log1p(X_mag) * np.log1p(mean_vector_mag)
        return AME

    # anisotropy provides the model the standard complexity of the data geometry, allowing it to know how complex the data needs to be processed.
    def anisotropy_measurement(self, x):
        """
        Measure how "anisotropic" (directionally uneven) the local
        variation of the input is, via the coefficient of variation
        (std / mean) of per-sample gradient magnitudes.

        A low value means the rate of change is roughly uniform across
        samples (isotropic); a high value means some samples change much
        faster than others (anisotropic), implying the data's geometry is
        more complex / uneven and should be processed accordingly by
        downstream weight shaping.

        Uses the Cython-optimized `optimized_anisotropy` when available
        (input flattened to 2D first); otherwise computes `np.gradient`
        directly (falling back to a 10x10 sub-block on failure, same
        pattern as `AME_Encoder`).

        Args:
            x: Input array-like.

        Returns:
            float: anisotropy score (always > 0 due to the `eps` term).
        """
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
        anisotropy = np.std(val) / np.mean(val) + eps

        return anisotropy

    # weight shaping provides directional context in which how the data should be processed in order to align with the data geometry
    def abstract_weight_shaping(self, x):
        """
        Combine anisotropy, AME/AMR, spectral rank (`k`), and spectral
        similarity into a single "abstraction efficiency" scalar, and use
        it as the upper bound of a `(input_size, output_size)` uniform
        random matrix -- i.e. the actual geometry-aware weight/context
        generator that `weight_shaping` wraps with normalization.

        Steps:
            1. Compute anisotropy and average magnitude of `x`.
            2. Run `eigenvalue_encoder` to get the effective rank `k`,
               a reference perturbation matrix (`floating_point`), and
               the noise used to build it.
            3. Compute AME -> AMR (activations.sigmoid-squashed AME, the "abstract
               modelling rate").
            4. Compute spectral similarity between `x` and the
               `floating_point` reference, and fold it (plus a 0.3
               baseline and anisotropy) into an "Abstraction Efficiency
               activations.Loss" term `AEL`.
            5. Compute `abstraction_efficiency` from `(k + AEL) * (1 -
               AMR)`. (Note: an earlier `(1.0 + AEL) * (1.0 - AMR)`
               expression is computed and then intentionally overwritten
               by the `k`-based version below it -- the `k`-based
               formula is the one actually used.)
            6. If that value is non-finite, fall back to `(1 - AMR) +
               eps`.
            7. Sample a `(input_size, output_size)` matrix uniformly in
               `[0, abstraction_efficiency]`.

        Args:
            x: Input array-like used to derive the geometry statistics.

        Returns:
            np.ndarray of shape (self.input_size, self.output_size):
            the raw (unnormalized) abstract context / weight matrix.
        """
        input_size = self.input_size
        output_size = self.output_size


        eps = 1e-5
        x = np.asarray(x)

        rng = np.random.default_rng()

        anisotropy = self.anisotropy_measurement(x)
        mag = np.mean(np.linalg.norm(x))

        k, floating_point, structured_noise = self.eigenvalue_encoder(x)
        AME = self.AME_Encoder(x)
        AMR = 1.0 / (1.0 + np.exp(-AME)) # abstract modelling rate

        spectral_similarity = self.spectral_similarity(x, floating_point, structured_noise)

        AEL = (0.3 + spectral_similarity + eps) * anisotropy

        scaled_anisotropy = anisotropy / (anisotropy + 1.0)

        abstraction_efficiency = (1.0 + AEL) * (1.0 - AMR) + eps
        # NOTE: the line above is superseded by the k-weighted version
        # below (kept intentionally for compatibility with the tuning
        # this formula was validated against) -- `k` replaces the
        # constant `1.0`, making effective spectral rank part of the
        # final bound.
        abstraction_efficiency = (k + AEL) * (1.0 - AMR) + eps

        if np.isnan(abstraction_efficiency) or np.isinf(abstraction_efficiency):
            abstraction_efficiency = (1 - AMR) + eps

        abstract_context = rng.uniform(0, abstraction_efficiency, size=(input_size, output_size))

        return abstract_context

    def weight_shaping(self, x, type=None):
        """
        Public entry point: sanitize `x`, derive a geometry-aware context
        matrix via `abstract_weight_shaping`, and normalize it to
        roughly [-1, 1] by dividing by its max absolute value.

        Sanitization performed before shaping:
            - Replaces NaN/Inf values in `x` with 0 / large finite bounds.
            - Converts list input to a NumPy array.
            - Flattens any input with more than 2 dims down to
              (n_samples, n_features).
            - If `x` has zero variance (constant input), replaces it with
              fresh uniform random noise so the downstream geometry
              measurements (which divide by std/anisotropy, etc.) don't
              degenerate.

        Args:
            x: Input array-like (or list) to base the weight shaping on.
            type: Unused parameter (kept for call-site compatibility /
                future extension).

        Returns:
            np.ndarray of shape (self.input_size, self.output_size),
            normalized so its largest-magnitude entry is ~1.
        """
        if np.isnan(x).any() or np.isinf(x).any():
            x = np.nan_to_num(x, nan=0.0, posinf=1e99, neginf=-1e99)

        if isinstance(x, list):
            x = np.asarray(x)

        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)

        if np.std(x) == 0:
            x = np.random.uniform(0, 1, size=x.shape)

        abstract_context = self.abstract_weight_shaping(x)
        abstract_context /= np.max(np.abs(abstract_context)) + 1e-5  # Normalized to [-1, 1]

        return abstract_context



# ________ UTILITY functions for activations and losses, can be used across different models and architectures _________



