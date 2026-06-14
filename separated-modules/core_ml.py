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



class GeometricWeightShaping:
    '''
    Data-adaptive weight initialisation based on the geometric properties of an
    input batch.

    Motivation
    ----------
    Standard He / Xavier initialisation scales weights by a fixed function of
    layer size.  In low-sample regimes (few dozen examples) this can be
    suboptimal because the true scale of the feature space varies wildly
    between tasks.  GWS instead measures three complementary properties of the
    *actual* input data and uses them as the upper bound for a uniform weight
    sampler:

    - Anisotropy   : directional spread of the input's gradient field.
                     High anisotropy → the data has varied geometry → wider
                     weight range helps the layer respond to diverse directions.
    - Eigenvalue encoding (trC, k) : compact scalar derived from the input's
                     covariance eigenspectrum.  k is the number of principal
                     components that capture 90% of variance; trC is a
                     three-stage cascade that compresses k and anisotropy into
                     a single magnitude.
    - AME          : Abstract Modelling Error — log-product of input magnitude
                     and gradient energy.  Measures raw "learning difficulty".

    Usage
    -----
    Instantiate with the layer's (input_size, output_size), then call
    weight_shaping(X) where X is the training batch.  Returns a float32
    matrix of shape (input_size, output_size) drawn from
    Uniform[0, efficient_distributed_energy].
    '''
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.floating_context = None

    def eigenvalue_encoder(self, x):
        # Encodes the geometric complexity of the input data into a scalar (trC) and a
        # principal component count (k). The scalar trC is later used as the upper bound
        # for the random floating-point context in abstract_weight_shaping.
        #
        # Step-by-step logic:
        #   1. Augment input with magnitude-scaled structured noise so the covariance
        #      matrix is never degenerate even on very small or homogeneous datasets.
        #   2. Run eigendecomposition on the augmented covariance, sort eigenvalues
        #      descending, then find k = the number of principal components that
        #      capture 90% of cumulative variance.  k is a compact measure of
        #      intrinsic dimensionality.
        #   3. Derive three chained scalars (trA → trB → trC) that compress k and the
        #      data anisotropy into a single weight-shaping magnitude.
        #      - trA  : scales k by directional variation; high anisotropy → large trA
        #      - trB  : dampens trA²; keeps the signal in a bounded range
        #      - trC  : final scalar — NOTE: trB² - 1.0 can equal zero when trB == ±1,
        #               causing division-by-zero (known fragility flagged in code review)
        eps = 1e-5
        raw_X = np.asarray(x)
        if raw_X.ndim > 2:
            raw_X = raw_X.reshape(raw_X.shape[0], -1)

        mag = np.mean(np.linalg.norm(raw_X, axis=-1))

        anisotropy = self.anisotropy_measurement(raw_X)

        # Augment data with noise proportional to its magnitude to avoid a singular
        # covariance matrix when the dataset is small or nearly constant.
        structured_noise = np.random.uniform(0, mag, size=raw_X.shape)
        X = np.vstack((raw_X, structured_noise))
        if X.ndim == 2 and X.shape[1] == 1:
            X = np.hstack((raw_X, structured_noise))  

        cov = np.cov(X, rowvar=False)

        # eigh is used instead of eig because cov is symmetric; it returns real eigenvalues
        # and is numerically more stable than the general eigensolver.
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]  # sort largest-first

        eigenvalues = eigenvalues[idx]
        # Cumulative explained variance ratio; searchsorted finds the elbow at 90 %.
        energy = np.cumsum(eigenvalues) / np.sum(eigenvalues)
        k = np.searchsorted(energy, 0.90) + 1     # +1 converts 0-based index to count

        # K_G: normalised inverse of k — small k (low-dim data) → K_G near 1,
        #       large k (high-dim data) → K_G near 0.
        K_G = 1.0 / (1.0 + k)
        mag_G = 1.0 / (1.0 + K_G)  # secondary magnitude dampener

        # Three-stage compression cascade that maps (k, anisotropy) → trC scalar.
        trA = k / (1.0 - anisotropy) + eps   # anisotropy close to 1 inflates trA
        trB = (1/2 + mag_G) / (1.0 + trA**2) # quadratic dampener keeps trB < 0.5
        # WARNING: trB² - 1.0 is negative for all typical trB values (|trB| < 1),
        # so trC ends up negative.  When trB == ±1 exactly this divides by zero.
        trC = (1/6 + K_G) / (trB**2 - 1.0)
        if np.isnan(trC) or np.isinf(trC):
            trC = anisotropy * (1.0 - mag_G) + eps

        floating_point = np.random.uniform(0, trC, size=x.shape)

        return k, floating_point, structured_noise


    def spectral_signature(self, x, structured_noise, k=5):
        '''
        Returns the top-k normalised eigenvalues of the input covariance matrix.
        These form a compact "fingerprint" of the dataset's principal variance
        structure.  Used by spectral_similarity to measure how geometrically
        close two arrays are.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
        k : number of leading eigenvalues to return (default 5)

        Returns
        -------
        ndarray of shape (k,) — each value in [0, 1], sums to ≤ 1.
        '''
        raw_X = np.asarray(x)
        if raw_X.ndim > 2:
            X = raw_X.reshape(raw_X.shape[0], -1)
        else:
            X = raw_X.reshape(raw_X.shape[0], -1)
            
        X = np.atleast_2d(X)
        if X.ndim == 2 and X.shape[1] == 1:
            X = np.hstack((X, structured_noise))

        cov = np.cov(X, rowvar=False)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.sort(eigvals)[::-1]
        return eigvals[:k] / (eigvals.sum() + 1e-8)

    def spectral_similarity(self, a, b, structured_noise):
        '''
        Measures how spectrally similar two arrays are by comparing their top-k
        eigenvalue signatures via the L2 norm of their difference, mapped through
        an exponential kernel so the result is in (0, 1].
        Value near 1 → arrays share similar geometric structure.
        Value near 0 → very different spectral profiles.
        Used in abstract_weight_shaping to gauge how much the real data
        "looks like" random noise drawn from the same eigenvalue range.
        '''
        sa = self.spectral_signature(a, structured_noise)
        sb = self.spectral_signature(b, structured_noise)
        return np.exp(-np.linalg.norm(sa - sb))

    # abstract modelling error provides the model how to better process weights when the data complexity has little geometric complexity
    def AME_Encoder(self, x):
        X = np.asarray(x)
        try:
            gradient = np.gradient(x)
        except:
            subnet = x[:min(10, x.shape[0]), :min(10, x.shape[1])]
            gradient = np.gradient(subnet.flatten())

        grad_energy = np.mean(np.linalg.norm(gradient, axis=-1))       
        X_mag = np.mean(np.linalg.norm(X, axis=-1))
        # Regular AME Equations, higher AME provides capabilities for the model to experience errors during abstraction
        # Lower AME means lower chance for un optimal abstraction.
        
        AME =  np.log1p(X_mag) * np.log1p(grad_energy) 
        return AME

    # anisotropy provides the model the standard complexity of the data geometry, allowing it to know how complex the data needs to be processed.
    def anisotropy_measurement(self, x):
        eps = 1e-5
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
        # Derives a data-adaptive random weight matrix whose range is governed by
        # the geometric complexity of the input batch x.
        #
        # Key scalars produced along the way:
        #   anisotropy  — directional spread of gradients across x (higher = more varied)
        #   trC, k      — eigenvalue-derived complexity scalar and intrinsic dimensionality
        #   AME         — Abstract Modelling Error: log-product of magnitude × gradient energy
        #   AEL         — Adaptive Energy Level: blends spectral similarity with anisotropy;
        #                 measures how much the data geometry resembles random noise
        #   AMR         — sigmoid-scaled AME; used as a soft gate between 0 and 1
        #   efficient_distributed_energy — the upper bound fed to the final uniform sampler;
        #                 equals k + AEL*(1 - AMR): dominated by intrinsic dimensionality
        #                 when the model rate (AMR) is high, shifts to AEL when AMR is low.
        #
        # The resulting weight matrix (shape: input_size × output_size) is drawn from
        # Uniform[0, efficient_distributed_energy], which gives the downstream Dense layer
        # a geometry-aware initialisation instead of a fixed scale like He/Xavier.
        input_size = self.input_size
        output_size = self.output_size

        rng = np.random.default_rng()

        anisotropy = self.anisotropy_measurement(x)
        mag = np.mean(np.linalg.norm(x))

        k, floating_point, structured_noise = self.eigenvalue_encoder(x)
        AME = self.AME_Encoder(x)
        AMR = 1.0 / (1.0 + np.exp(-AME))  # abstract modelling rate — sigmoid gate on AME

        # floating_point: noise draw bounded by trC; used only to compute spectral
        # similarity (how much the real data "looks like" noise geometrically).
        spectral_similarity = self.spectral_similarity(x, floating_point, structured_noise)

        # AEL rises when data is both spectrally noise-like and highly anisotropic.
        AEL = 0.3 + spectral_similarity * anisotropy       
        scaled_anisotropy = anisotropy / (anisotropy + 1.0)  # unused below; kept for potential future use

        # Upper bound of the weight distribution.
        # When data complexity is low (AMR → 1), the AEL term vanishes → bound ≈ k.
        # When data is geometrically rich (AMR → 0), AEL contributes more → wider init.
        efficient_distributed_energy = k + AEL * (1.0 - AMR)
        if np.isnan(efficient_distributed_energy) or np.isinf(efficient_distributed_energy):
            efficient_distributed_energy = (1 - AMR) + eps

        floating_context = rng.uniform(0, efficient_distributed_energy, size=(input_size, output_size)) 
        self.floating_context = floating_context

        return floating_context

    def weight_shaping(self, x, type=None):
        '''
        Public entry-point for GWS.  Validates the input array (converts lists,
        flattens >2-D tensors, replaces constant inputs with uniform noise to
        avoid zero-variance covariance) and then delegates to
        abstract_weight_shaping().

        Falls back to a ones matrix if abstract_weight_shaping produces NaN or
        non-finite values (e.g. due to the known trC division-by-zero edge case).

        Parameters
        ----------
        x    : array-like — the training batch for the layer being initialised.
        type : unused placeholder for future type-specific shaping strategies.

        Returns
        -------
        ndarray of shape (input_size, output_size) — geometry-scaled weights.
        '''
        if isinstance(x, list):
            x = np.asarray(x)

        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)

        if np.std(x) == 0:
            x = np.random.uniform(0, 1, size=x.shape)

        floating_context = self.abstract_weight_shaping(x)
        if np.isnan(floating_context).any() or not np.isfinite(floating_context).any():
            floating_context = np.ones_like(x)

        return floating_context



class Activation:
    '''
    Stateless collection of activation functions and their derivatives.
    All methods are @staticmethod so no instance is needed.

    Functions provided
    ------------------
    relu / relu_derivative     : Standard rectified linear unit.
    sigmoid / sigmoid_derivative : Logistic sigmoid; derivative re-uses
                                   the forward pass value for efficiency.
    softmax                    : Numerically stable (max-subtraction) softmax.
                                 Handles both 1-D vectors and 2-D batches.
    '''
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
    '''
    Stateless loss functions for multi-class classification.

    categorical_crossentropy
        Standard cross-entropy over one-hot targets.  Clips predictions to
        [eps, 1-eps] to avoid log(0).  Falls back to a slice of the smaller
        axis when y_true and y_pred have mismatched class counts (can occur
        during hybrid-feature training when the MLP and Transformer have
        different output dimensions).

    softmax_crossentropy_derivative
        Analytic gradient of softmax + cross-entropy combined:
        dL/dlogits = (softmax(logits) - y_true) / batch_size.
        Using this combined form is numerically more stable than computing the
        two gradients separately.
    '''
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


