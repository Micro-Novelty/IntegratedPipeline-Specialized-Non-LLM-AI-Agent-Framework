"""
activations.py
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

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_deriv(s):          # s = sigmoid(x) already computed
    return s * (1.0 - s)

def tanh_deriv(t):             # t = tanh(x) already computed
    return 1.0 - t ** 2

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
        if _OPT_AVAILABLE:
            output = optimized_softmax_2d(np.asarray(x, dtype=np.float64))    
            return output

        # numerical stability
        if x.ndim > 1:
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            normalized = exp_x / np.sum(exp_x, axis=1, keepdims=True)
            return normalized

        else:
            exp_x = np.exp(x - np.max(x, keepdims=True))
            return exp_x / np.sum(exp_x, keepdims=True) 

class Loss:
    @staticmethod
    def categorical_crossentropy(y_true, y_pred):
        eps = 1e-5
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)

        # normalize to 2D
        if y_true.ndim == 1:
            y_true = y_true[np.newaxis, :]
        if y_pred.ndim == 1:
            y_pred = y_pred[np.newaxis, :]

        # align both batch dim (axis=0) and class dim (axis=1)
        min_batch = min(y_true.shape[0], y_pred.shape[0])
        min_class = min(y_true.shape[1], y_pred.shape[1])

        if y_true.shape != y_pred.shape:
            print(f'[!] Shape mismatch in crossentropy: '
                  f'y_true={y_true.shape} y_pred={y_pred.shape} '
                  f'— aligning to ({min_batch}, {min_class})')
            y_true = y_true[:min_batch, :min_class]
            y_pred = y_pred[:min_batch, :min_class]

        y_pred = np.clip(y_pred, eps, 1 - eps)

        # guard against empty result after alignment
        if y_true.size == 0 or y_pred.size == 0:
            print('[!] Empty arrays after alignment — returning safe default loss')
            return 1.0

        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

        # guard against NaN/Inf from degenerate alignment
        if not np.isfinite(loss):
            return 0.0

        return float(loss)

    @staticmethod
    def softmax_crossentropy_derivative(y_true, y_pred):
        eps = 1e-5
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)

        if y_true.ndim == 1:
            y_true = y_true[np.newaxis, :]
        if y_pred.ndim == 1:
            y_pred = y_pred[np.newaxis, :]

        # align both dimensions consistently
        min_batch = min(y_true.shape[0], y_pred.shape[0])
        min_class = min(y_true.shape[1], y_pred.shape[1])

        if y_true.shape != y_pred.shape:
            print(f'[!] Shape mismatch in crossentropy derivative: '
                  f'y_true={y_true.shape} y_pred={y_pred.shape} '
                  f'— aligning to ({min_batch}, {min_class})')
            y_true = y_true[:min_batch, :min_class]
            y_pred = y_pred[:min_batch, :min_class]

        if y_true.size == 0 or y_pred.size == 0:
            print('[!] Empty arrays after alignment — returning zero gradient')
            return np.zeros((1, 1))

        cross_ent = (y_pred - y_true) / y_true.shape[0]

        return cross_ent





