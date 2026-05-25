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
# geometry.py
# GeometricWeightShaping (GWS): data-adaptive weight initialisation using
# PCA-derived complexity scalars (trC, k), anisotropy, spectral similarity,
# and the Abstract Modelling Error (AME).
# No local dependencies — pure numpy.
# ---------------------------------------------------------------------------

# geometric weight shaping provides the model with a robust geometric complexity
# alignment, allowing it to better process data with varying geometric complexity
# and providing a more stable training process in scarce data environments.
class GeometricWeightShaping:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.floating_context = None

    def eigenvalue_encoder(self, x):
        eps = 1e-5
        X = np.asarray(x)
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)

        mag = np.mean(np.linalg.norm(X, axis=-1))

        anisotropy = self.anisotropy_measurement(X)

        structured_noise = np.random.uniform(0, mag, size=X.shape)
        X = np.vstack((X, structured_noise))
        cov = np.cov(X, rowvar=False)

        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]

        eigenvalues = eigenvalues[idx]
        energy = np.cumsum(eigenvalues) / np.sum(eigenvalues)
        k = np.searchsorted(energy, 0.90) + 1     

        K_G = 1.0 / (1.0 + k)
        mag_G = 1.0 / (1.0 + K_G)

        trA = k / (1.0 - anisotropy) + eps  
        trB = (1/2 + mag_G) / (1.0 + trA**2)
        trC = (1/6 + K_G) / (trB**2 - 1.0)
        return trC, k


    def spectral_signature(self, x, k=5):
        X = np.asarray(x)
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        else:
            X = X.reshape(X.shape[0], -1)
        cov = np.cov(X, rowvar=False)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.sort(eigvals)[::-1]
        return eigvals[:k] / (eigvals.sum() + 1e-8)

    def spectral_similarity(self, a, b):
        sa = self.spectral_signature(a)
        sb = self.spectral_signature(b)
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
        input_size = self.input_size
        output_size = self.output_size

        rng = np.random.default_rng()

        anisotropy = self.anisotropy_measurement(x)
        mag = np.mean(np.linalg.norm(x))

        trC, k = self.eigenvalue_encoder(x)
        AME = self.AME_Encoder(x)

        floating_point = np.random.uniform(0, trC, size=x.shape)
        spectral_similarity = self.spectral_similarity(x, floating_point)

        AEL = 0.3 + spectral_similarity * anisotropy       
        scaled_anisotropy = anisotropy / (anisotropy + 1.0)
        AMR = 1.0 / (1.0 + np.exp(-AME)) # abstract modelling rate

        efficient_distributed_energy = k + AEL * (1.0 - AMR)
        floating_context = rng.uniform(0, efficient_distributed_energy, size=(input_size, output_size)) 
        self.floating_context = floating_context

        return floating_context

    def weight_shaping(self, x, type=None):
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