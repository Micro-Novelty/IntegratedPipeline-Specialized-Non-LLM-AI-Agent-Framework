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
from datetime import datetime, timedelta, timezone
from enum import IntEnum, Enum
from collections import deque
from collections import Counter
import traceback
from concurrent.futures import TimeoutError as FutureTimeoutError
import secrets
import ipaddress
from functools import wraps
import hmac
import aiohttp
import psutil
from sklearn.preprocessing import StandardScaler
import io
import concurrent.futures
import struct

from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
import stat

# Optimized Modules In Cython implementation.
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
    print('[=] Cython acceleration loaded ✅')
except ImportError as e:
    _OPT_AVAILABLE = False
    print(f'[=] Cython not available: {e}, using numpy fallback')

try:
    import abstract_weights_core as wc
    _RUST_MODULE_AVAILABLE = True
    print('[=] Rust weight storage loaded ✅')
except ImportError as e:
    _RUST_MODULE_AVAILABLE = False
    print(f'[=] Rust weight storage unavailable due to: {e}, using Python sqlite3 fallback')


# initial Setup logging for AgentDistributedInference and ModelStorage class logger and security logger
logger = logging.getLogger(__name__)
_integrated_pipeline_lock = threading.Lock()

class MessagePriority(Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3

class WrapperState(Enum):
    """Wrapper state machine."""
    UNINITIALIZED = "uninitialized"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"

@dataclass
class AsyncTask:
    """Track async tasks for proper cleanup."""
    id: str
    future: asyncio.Future
    created_at: float
    callback: Optional[Callable] = None
    timeout: float = 30.0

class TrustLevel(IntEnum):
    """Trust levels for peer agents"""
    UNTRUSTED = 0      # No trust - will be rejected
    BASIC = 1          # Basic trust - limited operations
    STANDARD = 2       # Standard trust - most operations
    HIGH = 3           # High trust - sensitive operations
    FULL = 4           # Full trust - administrative access


class RequestStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"

@dataclass
class AsyncRequest:
    """Track an async prediction request"""
    request_id: str
    texts: Any
    api_key: Optional[str]
    client_ip: Optional[str]
    callback: Optional[Callable] = None
    webhook_url: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    status: RequestStatus = RequestStatus.PENDING
    result: Optional[Dict] = None
    error: Optional[str] = None
    completed_at: Optional[float] = None
    
    @property
    def age(self) -> float:
        return time.time() - self.created_at


    @property
    def is_expired(self, timeout: int = 30) -> bool:
        return self.age > timeout

@dataclass
class SecureMessage:
    # Secure message wrapper
    id: str
    type: str
    payload: Any
    timestamp: float
    signature: str = ""

@dataclass
class Message:
    id: str
    type: str
    sender: str
    recipient: str
    payload: Any
    timestamp: datetime
    priority: MessagePriority = MessagePriority.NORMAL
    callback: Optional[Callable] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout: float = 30.0
    created_at: float = field(default_factory=time.time)
    trust: float = 1.0    

    @property
    def age(self) -> float:
        """Age of message in seconds."""
        return time.time() - self.created_at

    @property
    def proper_trust(self) -> bool:
        return self.trust > 0.3

    @property
    def degrade_trust(self) -> bool:
        self.trust = self.trust - 0.1 

    @property
    def is_expired(self) -> bool:
        """Check if message has expired."""
        expired = self.age > self.timeout
        if expired:
            self.degrade_trust()
        return expired
    
    # ============ COMPARISON METHODS FOR PRIORITY QUEUE ============
    
    def __lt__(self, other):
        """Less than comparison for priority queue."""
        if not isinstance(other, Message):
            return NotImplemented
        
        # Compare by priority value first (lower number = higher priority)
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        
        # If same priority, compare by creation time (older messages get processed first)
        return self.created_at < other.created_at
    
    def __le__(self, other):
        """Less than or equal."""
        if not isinstance(other, Message):
            return NotImplemented
        return self.__lt__(other) or self.__eq__(other)
    
    def __eq__(self, other):
        """Equality comparison."""
        if not isinstance(other, Message):
            return NotImplemented
        return self.id == other.id
    
    def __ne__(self, other):
        """Not equal."""
        if not isinstance(other, Message):
            return NotImplemented
        return not self.__eq__(other)
    
    def __gt__(self, other):
        """Greater than."""
        if not isinstance(other, Message):
            return NotImplemented
        return not self.__lt__(other) and not self.__eq__(other)
    
    def __ge__(self, other):
        """Greater than or equal."""
        if not isinstance(other, Message):
            return NotImplemented
        return not self.__lt__(other)
    
    def __hash__(self):
        """Message hashable"""
        return hash(self.id)


@dataclass
class SecurityConfig:
    """Security configuration for async manager"""
    max_text_length: int = 10000
    max_queue_size: int = 100
    max_pending_tasks: int = 50
    rate_limit_requests: int = 60  # per minute
    rate_limit_window: int = 60  # seconds
    request_timeout: float = 30.0
    max_concurrent: int = 10
    enable_auth: bool = True
    allowed_ips: List[str] = field(default_factory=list)  # Empty = allow all
    blocklisted_ips: List[str] = field(default_factory=list)
    require_api_key: bool = True
    api_key_rotation_days: int = 30

    # Admin-specific settings
    admin_bypass_rate_limit: bool = True  # Admins bypass rate limiting
    admin_bypass_ip_check: bool = False   # Admins still need IP whitelist
    enforce_admin_ip_whitelist: bool = True  # Separate admin IP whitelist
    admin_allowed_ips: List[str] = field(default_factory=list)  # Admin-specific IPs
    admin_rate_limit: int = 300  # Higher limit for admins (per minute)
    log_all_admin_actions: bool = True
    
    # Start protection (NOT authentication)
    min_start_interval: float = 5.0  # Seconds between start attempts
    max_consecutive_failures: int = 3  # Before circuit breaker
    max_cpu_percent: float = 99.0  # Don't start if CPU > 99%
    max_memory_percent: float = 95.0  # Don't start if memory > 95%
    min_disk_space_mb: int = 100  # Minimum 100MB free
    
    # Per-request security (REAL authentication)
    rate_limit_per_ip: bool = True
    
    # Optional: Bootstrap only for critical deployments
    require_bootstrap_auth: bool = False  # Default OFF for flexibility
    bootstrap_token_hash: Optional[str] = None  # Only if above is True    

class SecurityLevel(Enum):
    """Deployment security levels"""
    DEVELOPMENT = "dev"      # No security, max flexibility
    STAGING = "staging"      # API keys only
    PRODUCTION = "prod"      # API keys + rate limiting
    HARDENED = "hardened"    # Everything + bootstrap token

class SecurityError(Exception):
    pass

class AdminRole(Enum):
    ADMIN = "admin"
    OPERATOR = "operator"  # Can view but not modify
    AUDITOR = "auditor"     # Can only view audit log


class SingletonMeta(type):
    """Thread-safe singleton metaclass"""
    _instances: Dict[type, Any] = {}
    _lock: threading.Lock = threading.Lock()
    
    def __call__(cls, *args, **kwargs):
        # Fast path: instance already exists
        if cls in cls._instances:
            return cls._instances[cls]
        
        # Slow path: create instance with lock
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
            return cls._instances[cls]
    
    @classmethod
    def clear_instance(cls, target_class):
        """Clear singleton instance (useful for testing)"""
        with cls._lock:
            if target_class in cls._instances:
                del cls._instances[target_class]
    
    @classmethod
    def get_instance(cls, target_class):
        """Get instance without creating"""
        return cls._instances.get(target_class)


class Singleton(metaclass=SingletonMeta):
    """Base singleton class - inherit from this"""
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        # This is handled by metaclass, but for clarity
        return super().__new__(cls)
    
    def __init__(self, *args, **kwargs):
        if self._initialized:
            print(f"[===] Reusing existing {self.__class__.__name__} instance (id: {id(self)})")
            return
        self._initialized = True
        print(f"[===] Creating NEW {self.__class__.__name__} instance (id: {id(self)})")

# geometric weight shaping provides the model with a robust geometric complexity alignment>
#  allowing it to better process data with varying geometric complexity, and providing a more stable training process in scarce data environment. 
# It can be used as a general weight initialization and shaping method for various models, especially in scenarios where data geometry is complex and data is scarce.

class GeometricWeightShaping:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
    

    def eigenvalue_encoder(self, x):
        eps = 1e-5
        raw_X = np.asarray(x)
        AME = self.AME_Encoder(raw_X)  
        AMR = 1.0 / (1.0 + np.exp(-AME)) + eps
        mag = np.mean(np.linalg.norm(raw_X, axis=-1))

        if raw_X.ndim > 2:
            raw_X = raw_X.reshape(raw_X.shape[0], -1)

        anisotropy = self.anisotropy_measurement(raw_X)

        structured_noise = np.random.uniform(0, mag, size=raw_X.shape)
        X = np.vstack((raw_X, structured_noise))
        if X.ndim == 2 and X.shape[1] == 1:
            X = np.hstack((raw_X, structured_noise))

        cov = np.cov(X, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]

        energy = np.cumsum(eigenvalues) / np.sum(eigenvalues)
        energy_sigmoid_growth = 1.0 / (1.0 + np.exp(-energy))
        energy_consistency = np.std(energy_sigmoid_growth)
        k = np.searchsorted(energy, 0.90) + 1     # +1 converts 0-based index to count

        trA = k / (1.0 - anisotropy) + eps  
        trB = (1/2 + energy_consistency) / (1.0 + trA**2)
        trC = (1/6 + AMR) / (1.0 - trB**2) + eps

        if np.isnan(trC) or np.isinf(trC):
            trC = anisotropy * (trB**2 - 1.0) + eps
            if np.isnan(trC) or np.isinf(trC):
                trC = (1.0 - AMR)

        min_val = min(trC, 0) 
        max_val = max(trC, 0) 
        floating_point = np.random.uniform(min_val, max_val, size=X.shape) 
        return k, floating_point, structured_noise


    def spectral_signature(self, x, structured_noise, k=5):
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
        sa = self.spectral_signature(a, structured_noise)
        sb = self.spectral_signature(b, structured_noise)
        if sa.shape != sb.shape:
            min_rows = min(sa.shape[0], sb.shape[0])

            sa = sa[:min_rows]
            sb = sb[:min_rows]

        return np.exp(-np.linalg.norm(sa - sb))

    # abstract modelling error provides the model how to better process weights when the data complexity has little geometric complexity
    def AME_Encoder(self, x):
        X = np.asarray(x)
        if _OPT_AVAILABLE and np.asarray(X).ndim == 2:
            return optimized_ame_encoder(np.asarray(X, dtype=np.float64))     

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
        
        abstraction_efficiency = (1.0 + AEL) * (1.0 - AMR)

        abstraction_efficiency = k + AEL * (1.0 - AMR)
        if np.isnan(abstraction_efficiency) or np.isinf(abstraction_efficiency):
            abstraction_efficiency = (1 - AMR) + eps

        abstract_context = rng.uniform(0, abstraction_efficiency, size=(input_size, output_size)) 
        return abstract_context

    def weight_shaping(self, x, type=None):
        if np.isnan(x).any() or np.isinf(x).any():
            x = np.nan_to_num(x, nan=0.0, posinf=1e99, neginf=-1e99)     

        if isinstance(x, list):
            x = np.asarray(x)

        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)

        if np.std(x) == 0:
            x = np.random.uniform(0, 1, size=x.shape)

        abstract_context = self.abstract_weight_shaping(x)
        if np.isnan(abstract_context).any() or not np.isfinite(abstract_context).any():
            abstract_context = np.ones_like(x)

        return abstract_context



# ________ UTILITY functions for activations and losses, can be used across different models and architectures _________

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
            print(f'[!] Non-finite loss detected ({loss}) — returning safe default')
            return 1.0

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



class Transformer:
    def __init__(self, vocab_size, d_model=8, n_heads=2, num_classes=7, learning_rate=0.01, attn_dropout=0.0, ffn_dropout=0.0, weight_decay=1e-4):
        self.d_model = d_model  # Embedding dimension
        self.n_heads = n_heads
        self.attn_dropout_rate = attn_dropout
        self.ffn_dropout_rate  = ffn_dropout
        self.transformer_lr = learning_rate
        self.weight_decay = weight_decay

        self.token_embedding = np.random.randn(vocab_size, d_model) * 0.02
        
        # Positional embeddings (word order)
        self.pos_embedding = np.random.randn(100, d_model) * 0.02  
        
        # Multi-head attention parameters
        self.W_q = np.random.randn(n_heads, d_model, d_model // n_heads) * 0.02
        self.W_k = np.random.randn(n_heads, d_model, d_model // n_heads) * 0.02
        self.W_v = np.random.randn(n_heads, d_model, d_model // n_heads) * 0.02

        self.W_q_fixed = self.W_q.copy()
        self.W_k_fixed = self.W_k.copy()
        self.W_v_fixed = self.W_v.copy()

        self.W_o = np.random.randn(d_model, d_model) * 0.02
        self.encoded = False 
        self._attn_quality_step = 0
        self._forward_count = 0
        self._attn_scale = None

        # Feed-forward network
        self.ffn1 = np.random.randn(d_model, d_model * 4) * 0.02
        self.ffn2 = np.random.randn(d_model * 4, d_model) * 0.02
        
        # Layer norms
        self.ln1_scale = np.ones(d_model)
        self.ln1_shift = np.zeros(d_model)
        self.ln2_scale = np.ones(d_model)
        self.ln2_shift = np.zeros(d_model)
        
        # Output layer
        self.output = np.random.randn(d_model, num_classes) * 0.02
        self.output_bias = np.zeros(num_classes)
        
        self.cache = {}

    def _clear_forward_cache(self, keep_essential=True, max_age_forward_passes=5):
        """Clear cache entries older than max_age_forward_passes."""
        
        essential_keys = {'input_ids', 'mask', 'alpha', 'attn_weights', 'probs', 'logits'}
        oldest_to_keep = self._forward_count - max_age_forward_passes
        
        # Group cache keys by their forward pass number
        forward_keys = {}
        latest_keys = {}
        
        for key in list(self.cache.keys()):
            if '_f' in key:
                parts = key.split('_f')
                if len(parts) == 2:
                    base_key, fwd_num_str = parts
                    try:
                        fwd_num = int(fwd_num_str)
                        if fwd_num not in forward_keys:
                            forward_keys[fwd_num] = []
                        forward_keys[fwd_num].append(key)
                    except ValueError:
                        pass
            elif key.startswith('latest_'):
                latest_keys[key] = True
        
        # Delete old forward passes
        for fwd_num in list(forward_keys.keys()):
            if fwd_num < oldest_to_keep:
                for key in forward_keys[fwd_num]:
                    if key in self.cache:
                        del self.cache[key]
                del forward_keys[fwd_num]
        
        # Also delete latest pointers for essential keys that are too old
        if keep_essential:
            for latest_key in list(latest_keys.keys()):
                actual_key = latest_key.replace('latest_', '')
                # Check if we still have any version of this key from recent passes
                has_recent = False
                for fwd_num in forward_keys:
                    if fwd_num >= oldest_to_keep:
                        for key in forward_keys[fwd_num]:
                            if key.startswith(actual_key + '_f'):
                                has_recent = True
                                break
                    if has_recent:
                        break
                
                if not has_recent and actual_key in essential_keys:
                    # No recent versions of this essential key, delete the latest pointer
                    del self.cache[latest_key]

    def layer_norm(self, x, scale, shift):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return scale * (x - mean) / np.sqrt(var + 1e-5) + shift

    def apply_update(self, param, grad, lr):
        # L2 weight decay applied directly at update time
        # equivalent to: grad += weight_decay * param
        return param - lr * (grad + self.weight_decay * param)

    def dropout(self, x, rate=0.1, training=True, alpha=None):
        if not training or rate == 0.0:
            return x, None
        
        # If alpha provided, scale the effective drop rate by it
        # low alpha (early training, fixed attention) → very light dropout
        # high alpha (dynamic attention active)       → full dropout rate
        effective_rate = rate * alpha if alpha is not None else rate
        
        if effective_rate == 0.0:
            return x, None

        mask = (np.random.rand(*x.shape) > effective_rate).astype(np.float32)
        return x * mask / (1.0 - effective_rate), mask

    def _get_attn_scale(self, d_k):
        """Cache scale factor — d_k never changes at runtime."""
        if self._attn_scale is None:
            self._attn_scale = 1.0 / np.sqrt(d_k)
        return self._attn_scale



    def attention(self, Q, K, V, mask=None):
        d_k = Q.shape[-1]

        # cached scale, no sqrt per call
        scale = self._get_attn_scale(d_k)

        # np.einsum avoids explicit transpose + matmul allocation
        # 'bhqd,bhkd->bhqk' computes Q @ K.T without creating K.T
        scores = np.einsum('bhqd,bhkd->bhqk', Q, K) * scale

        # inplace clip
        np.clip(scores, -50, 50, out=scores)

        # inplace mask application
        if mask is not None:
            # mask: (B,1,1,T) broadcasts to (B,H,T,T)
            scores += (1.0 - mask) * -1e9   # inplace add instead of np.where

        # fused softmax with numerical stability inplace
        scores -= scores.max(axis=-1, keepdims=True)  # stability shift inplace
        np.exp(scores, out=scores)                     # inplace exp
        scores /= scores.sum(axis=-1, keepdims=True) + 1e-8  # inplace normalize
        weights = scores   

        output = np.matmul(weights, V)
        return output, weights


    def softmax(self, x):
        if _OPT_AVAILABLE:
            return optimized_softmax_2d(np.asarray(x, dtype=np.float64))    

        if x.ndim == 3:
            shifted = x - np.max(x, axis=-1, keepdims=True)
        else:
            shifted = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(shifted)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    

    def multi_head_attention(self, x, mask=None, alpha=None,
                  W_q_mix=None, W_k_mix=None, W_v_mix=None):
        batch_size, seq_len, d_model = x.shape
        if W_q_mix is None:
            one_minus = 1.0 - alpha
            W_q_mix = one_minus * self.W_q_fixed + alpha * self.W_q
            W_k_mix = one_minus * self.W_k_fixed + alpha * self.W_k
            W_v_mix = one_minus * self.W_v_fixed + alpha * self.W_v        

        # Optimized Project heads with Cython implementation.
        if _OPT_AVAILABLE:
            B, S, D = batch_size, seq_len, d_model
            M = D // self.n_heads 

            Q = optimized_project_heads(x, W_q_mix, B, S, self.n_heads, D, M)
            K = optimized_project_heads(x, W_k_mix, B, S, self.n_heads, D, M)
            V = optimized_project_heads(x, W_v_mix, B, S, self.n_heads, D, M)
        else:
            Q = np.einsum('bsd,hdm->bhsm', x, W_q_mix)
            K = np.einsum('bsd,hdm->bhsm', x, W_k_mix)
            V = np.einsum('bsd,hdm->bhsm', x, W_v_mix)    
        
        # Store for backward
        self.cache['Q'] = Q
        self.cache['K'] = K
        self.cache['V'] = V
        self.cache['x_attn_input'] = x
        
        # Attention
        attn_output, attn_weights = self.attention(Q, K, V, mask)
        self.cache['attn_weights'] = attn_weights
        self.cache['attn_output'] = attn_output
        
        # Concatenate heads
        attn_output = attn_output.transpose(0, 2,1, 3).reshape(batch_size, seq_len, -1)
        self.cache['attn_concat'] = attn_output
        
        # Final linear projection
        output = np.matmul(attn_output, self.W_o)
        self.cache['attn_out'] = output
        
        return output, attn_weights
        
    def _handle_indices(self, input_ids, dtype=None):
        if not dtype:
            dtype = np.int32

        try:
            try:
                ids = np.asarray(input_ids)
                return ids
            except:
                def flatten(x):
                    for item in x:
                        if isinstance(item, (list, tuple)):
                            yield from flatten(item)
                        else:
                            yield item

                if isinstance(input_ids, (list, tuple)):
                    flat_ids = list(flatten(input_ids))
                else:
                    flat_ids = input_ids  # already a flat array/tensor

                ids = np.asarray(flat_ids, dtype=dtype)
        except:
            flat_ids = self.pipeline._safe_to_2d_float(input_ids)
            ids = np.asarray(flat_ids, dtype=dtype)

        return ids

    def forward(self, input_ids, AME=None, _update_quality_matrix=None, embedded=False, pad_token_id=0, training=True, attn_dropout=0.1, ffn_dropout=0.1):
        self._forward_count += 1
            
        # Clean up old cache entries before storing new ones
        self._clear_forward_cache(keep_essential=True, max_age_forward_passes=5) 

        if embedded:
            x = self._handle_indices(input_ids, dtype=np.int64)
            if x.ndim == 2:
                x = x[np.newaxis, ...]
            batch_size, seq_len, _ = x.shape
            self.cache['embedded_input'] = x
            self.cache['input_ids'] = None
            mask = None
        else:
            input_ids = self._handle_indices(input_ids, dtype=np.int32)
            if input_ids.ndim == 1:
                input_ids = input_ids[np.newaxis, :]

            x = self.token_embedding[input_ids]
            x = x + self.pos_embedding[:x.shape[1]]
            batch_size, seq_len = input_ids.shape
            self.cache['embedded_input'] = None
            self.cache['input_ids'] = input_ids
            mask = self.padding_mask_utility(input_ids, pad_token_id)  # (B,1,1,T)            

        self.cache['mask'] = mask if not embedded else None
        self.cache['seq_len'] = seq_len
        self.cache['batch_size'] = batch_size
        self.cache['x_token'] = x
        self.cache['x_pos'] = x
        
        # Multi-head attention with residual     
        if AME is None:
            AME = self.AME_Encoder(x)

        alpha = 1.0 / (1.0 + np.exp(-float(AME)))
        if _update_quality_matrix is None:
            # input ids ranged 0 to 1.
            consistency = 1.0 / (1.0 + np.std(input_ids))
            alpha_rate = 1.0 / (1.0 + alpha)
            _update_quality_matrix = alpha_rate * (1.0 - ffn_dropout) * consistency

        _should_update = (
            not training or
            not hasattr(self, '_attn_quality_step') or
            self._attn_quality_step % max(1, int(1.0 / (_update_quality_matrix + 1e-6))) == 0
        )
        
        one_minus_alpha = 1.0 - alpha
        W_q_mix = one_minus_alpha * self.W_q_fixed + alpha * self.W_q
        W_k_mix = one_minus_alpha * self.W_k_fixed + alpha * self.W_k
        W_v_mix = one_minus_alpha * self.W_v_fixed + alpha * self.W_v

        attn_out, attn_weights = self.multi_head_attention(x, mask=mask, alpha=alpha,
                      W_q_mix=W_q_mix, W_k_mix=W_k_mix, W_v_mix=W_v_mix )
  
        current_alpha = self.cache.get('alpha', 0.0) 

        attn_out, attn_drop_mask = self.dropout(attn_out, rate=self.attn_dropout_rate, training=training, alpha=current_alpha)   
        self.cache['attn_drop_mask'] = attn_drop_mask  

        if training and hasattr(self, '_attn_quality_step'):
            self._attn_quality_step += 1
            if _should_update and self._attn_quality_step % 10 == 0:
                attn_quality = self.attention_quality_computing(attn_weights, AME=AME, mask=mask)
                self._cached_attn_quality = attn_quality
            else:
                attn_quality = getattr(self, '_cached_attn_quality', 0.5)
        else:
            self._attn_quality_step  = 0
            self._cached_attn_quality = alpha * (1.0 - current_alpha)
            attn_quality = self._cached_attn_quality 

        alpha = 0.95 * alpha + 0.05 * attn_quality 

        self.alpha = alpha
        self.cache['alpha'] = alpha  # store in cache     

        self.cache['x_ln1_input'] = x + attn_out
        x, ln1_mean, ln1_var = self.layer_norm_with_cache(
                self.cache['x_ln1_input'], self.ln1_scale, self.ln1_shift
            )   

        self.cache['x_after_ln1'] = x
        self.cache['ln1_mean']    = ln1_mean   # reused in backward
        self.cache['ln1_var']     = ln1_var    

        # Feed-forward with residual
        self.cache['ffn_input'] = x
        ffn_pre = np.matmul(x, self.ffn1)
        self.cache['ffn_pre'] = ffn_pre
        
        ffn_act = np.maximum(0, ffn_pre)  # ReLU
           
        ffn_act, ffn_drop_mask = self.dropout(ffn_act, rate=self.ffn_dropout_rate, training=training, alpha=current_alpha)
     
        self.cache['ffn_act'] = ffn_act
        self.cache['ffn_drop_mask'] = ffn_drop_mask   

        ffn_out = np.matmul(ffn_act, self.ffn2)
        self.cache['ffn_out'] = ffn_out
        
        self.cache['x_ln2_input'] = x + ffn_out
        x, ln2_mean, ln2_var = self.layer_norm_with_cache(
                self.cache['x_ln2_input'], self.ln2_scale, self.ln2_shift
            )        
        self.cache['x_after_ln2'] = x
        self.cache['ln2_mean']    = ln2_mean
        self.cache['ln2_var']     = ln2_var        
        
        if mask is not None:
            # Reshape mask to (B, T, 1) for broadcasting against (B, T, D)
            token_mask = mask[:, 0, 0, :, np.newaxis]        # (B, T, 1)
            x_masked   = x * token_mask                       # zero out padding
            lengths    = token_mask.sum(axis=1)               # (B, 1) valid token counts
            x_pooled   = x_masked.sum(axis=1) / (lengths + 1e-6)  # (B, D)
        else:
            x_pooled = np.mean(x, axis=1)

        self.cache['x_pooled'] = x_pooled
        
        # Output projection
        logits = np.matmul(x_pooled, self.output) + self.output_bias
        self.cache['logits'] = logits
        
        probs = self.softmax(logits)
        self.cache['probs'] = probs
        
        return probs, attn_weights

    def layer_norm_with_cache(self, x, scale, shift, eps=1e-5):
        """Layer norm that returns mean and var for backward reuse."""
        mean  = x.mean(axis=-1, keepdims=True)
        var   = x.var(axis=-1,  keepdims=True)
        x_hat = (x - mean) / np.sqrt(var + eps)
        return x_hat * scale + shift, mean, var 


    def layer_norm_backward(self, d_out, x, scale, shift,
               mean=None, var=None):
        eps = 1e-5
        if mean is None:
            mean = x.mean(axis=-1, keepdims=True)
        if var is None:
            var  = x.var(axis=-1,  keepdims=True)

        std = np.sqrt(var + eps)
        x_hat = (x - mean) / std
        
        N = x.shape[-1]
        dx_hat = d_out * scale
        dvar = np.sum(dx_hat * (x - mean) * -0.5 * std**-3, axis=-1, keepdims=True)
        dmean = np.sum(dx_hat * (-1.0 / std), axis=-1, keepdims=True)
        
        dx = (
        dx_hat / std
        + dvar * 2*(x-mean)/N
        + dmean / N
        )
        
        return dx
    
    # fixed attention backward allow the transformer to not update its Q, K, V projections, allowing much stable attention, while sacrificing flexibility.
    def fixed_attention_backward(self, d_logits, lr=0.01, max_norm=1.0):

        # Gradient for output layer
        d_output = d_logits
        alpha = self.cache.get('alpha', 1.0)

        d_Wo = np.dot(self.cache['x_pooled'].T, d_output)
        d_bo = np.sum(d_output, axis=0, keepdims=True)
        
        # Gradient for pooled features
        d_pooled = np.dot(d_output, self.output.T)
        
        # Expand pooled gradient to all positions
        d_x = np.repeat(d_pooled[:, np.newaxis, :] / self.cache['seq_len'], self.cache['seq_len'], axis=1)
        
        # Layer norm 2 gradient
        d_x = self.layer_norm_backward(d_x, self.cache['x_ln1_input'],
                                        self.ln1_scale, self.ln1_shift,
                                        mean=self.cache.get('ln1_mean'),
                                        var=self.cache.get('ln1_var'))
                                        
        # FFN gradients
        d_ffn = d_x
        
        # Gradient for FFN2
        d_ffn2 = np.sum(np.matmul(self.cache['ffn_act'].transpose(0, 2, 1), d_ffn), axis=0)
        
        # Gradient for FFN1 through ReLU
        d_ffn_act = np.matmul(d_ffn, self.ffn2.T)
        ffn_drop_mask = self.cache.get('ffn_drop_mask')
        if ffn_drop_mask is not None:
            d_ffn_act = d_ffn_act * ffn_drop_mask / (1.0 - self.ffn_dropout_rate)

        d_ffn_pre = d_ffn_act * (self.cache['ffn_pre'] >= 0)   # ReLU backward unchanged

        d_prev = np.matmul(d_ffn_pre, self.ffn1.T)
        d_ffn1 = np.sum(np.matmul(self.cache['ffn_input'].transpose(0, 2, 1), d_ffn_pre), axis=0)
        
        # Layer norm 1 gradient
        d_x = self.layer_norm_backward(d_x - self.cache['attn_out'], 
                                        self.cache['x_ln1_input'],
                                        self.ln1_scale, self.ln1_shift)
        
        d_ffn = d_x
        d_residual_ffn = d_ffn
        dx = d_prev + d_residual_ffn
        d_attn = dx

        # Gradient for attention output projection
        attn_drop_mask = self.cache.get('attn_drop_mask')
        if attn_drop_mask is not None:
            d_attn = d_attn * attn_drop_mask / (1.0 - self.attn_dropout_rate)

        d_Wo_attn = np.sum(np.matmul(self.cache['attn_concat'].transpose(0,2,1), d_attn), axis=0)        

        grads = {
                'output':  d_Wo,
                'ffn2':    d_ffn2,
                'ffn1':    d_ffn1,
                'W_o':     d_Wo_attn,
            }

        grads, norm = self.clip_gradients(grads, max_norm)        

        # Update weights
        self.output = self.apply_update(self.output, grads['output'], lr)
        self.ffn2   = self.apply_update(self.ffn2,   grads['ffn2'],   lr)
        self.ffn1   = self.apply_update(self.ffn1,   grads['ffn1'],   lr)
        self.W_o    = self.apply_update(self.W_o,    grads['W_o'],    lr)
        # output_bias intentionally excluded — biases don't get weight decay
            
        return d_x
    

    def dynamic_backward(self, d_logits, lr=0.01, max_norm=1.0):
        # Gradient for output layer
        d_output = d_logits
        alpha = self.cache.get('alpha', 1.0)

        d_Wo = np.dot(self.cache['x_pooled'].T, d_output)
        d_bo = np.sum(d_output, axis=0)
        
        # Gradient for pooled features
        d_pooled = np.dot(d_output, self.output.T)
        
        # Expand pooled gradient to all positions
        mask = self.cache['mask']  # (B, 1, 1, T)
        if mask is not None:
            token_mask = mask[:, 0, 0, :, np.newaxis]             # (B, T, 1)
            lengths    = token_mask.sum(axis=1, keepdims=True)    # (B, 1, 1)
            d_x        = (d_pooled[:, np.newaxis, :] / (lengths + 1e-6)) * token_mask
        else:
            d_x = np.repeat(d_pooled[:, np.newaxis, :] / self.cache['seq_len'], self.cache['seq_len'], axis=1)    

        # Layer norm 2 gradient
        d_x = self.layer_norm_backward(d_x, self.cache['x_ln2_input'],
                                self.ln2_scale, self.ln2_shift,
                                mean=self.cache.get('ln2_mean'),
                                var=self.cache.get('ln2_var'))

        # FFN gradients
        d_ffn = d_x
        
        # Gradient for FFN2
        d_ffn2 = np.sum(np.matmul(self.cache['ffn_act'].transpose(0, 2, 1), d_ffn), axis=0)
        
        # Gradient for FFN1 through ReLU
        d_ffn_act = np.matmul(d_ffn, self.ffn2.T)
        ffn_drop_mask = self.cache.get('ffn_drop_mask')
        if ffn_drop_mask is not None:
            d_ffn_act = d_ffn_act * ffn_drop_mask / (1.0 - self.ffn_dropout_rate)

        d_ffn_pre = d_ffn_act * (self.cache['ffn_pre'] >= 0)   # ReLU backward unchanged

        d_prev = np.matmul(d_ffn_pre, self.ffn1.T)
        d_ffn1 = np.sum(np.matmul(self.cache['ffn_input'].transpose(0, 2, 1), d_ffn_pre), axis=0)
        
        # Layer norm 1 gradient
        d_ln1 = self.layer_norm_backward(
            d_prev, self.cache['x_ln1_input'], self.ln1_scale, self.ln1_shift
        ) 

        d_residual = d_ln1
        d_attn = d_ln1
        dx = d_prev + d_residual

        # Gradient for attention output projection
        attn_drop_mask = self.cache.get('attn_drop_mask')
        if attn_drop_mask is not None:
            d_attn = d_attn * attn_drop_mask / (1.0 - self.attn_dropout_rate)

        d_Wo_attn = np.sum(np.matmul(self.cache['attn_concat'].transpose(0, 2, 1), d_attn), axis=0)

        d_attn_concat = np.matmul(d_attn, self.W_o.T)
        batch, seq_len, _ = d_attn_concat.shape
        d_head = self.n_heads
        d_dim = self.d_model // self.n_heads

        d_attn_heads = d_attn_concat.reshape(batch, seq_len, d_head, d_dim) .transpose(0, 2, 1, 3)      

        V = self.cache['V']
        K = self.cache['K']
        Q = self.cache['Q']
        weight = self.cache['attn_weights']
        
        d_V = np.matmul(weight.transpose(0, 1, 3, 2), d_attn_heads)
        d_weights = np.matmul(d_attn_heads, V.transpose(0, 1, 3, 2))

        d_scores = weight * (d_weights - np.sum(d_weights * weight, axis=-1, keepdims=True))
        d_scores /= np.sqrt(Q.shape[-1])

        d_Q = np.matmul(d_scores, K)
        d_K = np.matmul(d_scores.transpose(0, 1, 3, 2), Q)

        x = self.cache['x_attn_input']

        if _OPT_AVAILABLE:
            B, S = x.shape[0], x.shape[1]
            H, M = self.n_heads, self.d_model // self.n_heads

            d_W_q = optimized_qkv_weight_grad(x, d_Q, B, S, H, self.d_model, M)
            d_W_k = optimized_qkv_weight_grad(x, d_K, B, S, H, self.d_model, M)
            d_W_v = optimized_qkv_weight_grad(x, d_V, B, S, H, self.d_model, M)

            d_x_q = optimized_qkv_input_grad(d_Q, self.W_q, B, S, H, self.d_model, M)
            d_x_k = optimized_qkv_input_grad(d_K, self.W_k, B, S, H, self.d_model, M)
            d_x_v = optimized_qkv_input_grad(d_V, self.W_v, B, S, H, self.d_model, M)  
        else:          
            d_W_q = np.einsum('bsd, bhsm->hdm', x, d_Q)
            d_W_k = np.einsum('bsd, bhsm->hdm', x, d_K)
            d_W_v = np.einsum('bsd, bhsm->hdm', x, d_V)

            d_x_q = np.einsum('bhsm, hdm->bsd', d_Q, self.W_q)
            d_x_k = np.einsum('bhsm, hdm->bsd', d_K, self.W_k)
            d_x_v = np.einsum('bhsm, hdm->bsd', d_V, self.W_v)

        d_x_attn_input = d_x_q + d_x_k + d_x_v
        d_x_total = d_x_attn_input + d_residual

        input_ids = self.cache.get('input_ids')

        # Update weights
        grads = {
                'output': d_Wo,
                'ffn2':   d_ffn2,
                'ffn1':   d_ffn1,
                'W_o':    d_Wo_attn,
                'W_q':    alpha * d_W_q,   # already alpha-scaled, clip the combined thing
                'W_k':    alpha * d_W_k,
                'W_v':    alpha * d_W_v,
            }
        grads, norm = self.clip_gradients(grads, max_norm)  

        self.output = self.apply_update(self.output, grads['output'], lr)
        self.ffn2   = self.apply_update(self.ffn2,   grads['ffn2'],   lr)
        self.ffn1   = self.apply_update(self.ffn1,   grads['ffn1'],   lr)
        self.W_o    = self.apply_update(self.W_o,    grads['W_o'],    lr)
        self.W_q    = self.apply_update(self.W_q,    grads['W_q'],    lr)
        self.W_k    = self.apply_update(self.W_k,    grads['W_k'],    lr)
        self.W_v    = self.apply_update(self.W_v,    grads['W_v'],    lr)

        if input_ids is not None:
            emb_norm = np.linalg.norm(d_x_total)
            emb_coef = min(1.0, max_norm / (emb_norm + 1e-6))

            flat_ids   = input_ids.flatten()                          # (B*T,)
            flat_grads = d_x_total.reshape(-1, self.d_model) / self.cache['seq_len']  # (B*T, D)

            np.add.at(self.token_embedding, flat_ids, -lr * emb_coef * flat_grads)
            self.pos_embedding[:seq_len] -= lr * emb_coef * d_x_total.mean(axis=0)
        else:
            self.pos_embedding[:seq_len] -= lr * d_x_total.mean(axis=0)
            norm = d_x_total

        return norm

    def smoothing_labels_utility(self, y_true, smoothing=0.1):
        # y_true: (B, num_classes) one-hot
        try:
            num_classes = y_true.shape[1]
        except:
            y_true_2d = y_true.reshape(-1, 1)
            num_classes = y_true_2d.shape[1] 
            
        return y_true * (1.0 - smoothing) + smoothing / num_classes

    def learning_rate_warm_up(self, epoch, epochs, lr_base, schedule='cosine_warmup', warmup_frac=0.1):
        warmup_epochs = int(epochs * warmup_frac)
        
        if schedule == 'cosine_warmup':
            if epoch < warmup_epochs:
                # Linear warmup
                return lr_base * (epoch + 1) / warmup_epochs
            else:
                # Cosine decay after warmup
                progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
                return lr_base * 0.5 * (1 + np.cos(np.pi * progress))

        elif schedule == 'step':
            # Halve lr every 30% of training
            step = int(epochs * 0.3)
            return lr_base * (0.5 ** (epoch // step))

        elif schedule == 'constant':
            return lr_base

        return lr_base

    def padding_mask_utility(self, input_ids, pad_token_id=0):
        # input_ids: (B, T)
        # Returns: (B, 1, 1, T) — broadcast-ready for (B, heads, T_q, T_k)
        mask = (input_ids != pad_token_id).astype(np.float32)
        return mask[:, np.newaxis, np.newaxis, :]   # (B, 1, 1, T)
        
    def clip_gradients(self, grads: dict, max_norm: float = 1.0):
        #clip gradients function to prevent overflow.
        total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads.values()))
        clip_coef  = max_norm / (total_norm + 1e-6)

        if clip_coef < 1.0:
            for g in grads.values():
                g *= clip_coef    

        return grads, total_norm    


    def batch_padding_utility(self, sequences, pad_token_id=0):
        # sequences: list of 1-D np arrays of varying length
        max_len = max(len(s) for s in sequences)
        padded  = np.full((len(sequences), max_len), pad_token_id, dtype=np.int32)
        for i, s in enumerate(sequences):
            padded[i, :len(s)] = s
        return padded   # (B, T)

    def train(self, input_ids_list, y_true_list, epochs=100, mode=None,
            lr=0.01, embedded=False, max_norm=1.0,
            schedule='cosine_warmup', pad_token_id=0, batch_size=None):

        # Main Training function for Transformer
        losses  = []
        accs    = []
        y_true_smoothed_list = []        
        d_model = self.d_model

        input_ids_list = self._sanitize_string_chars(input_ids_list)
        y_true_list = self._sanitize_string_chars(y_true_list)

        # W_o geometric init — unchanged
        if not self.encoded:
            self.shaping = GeometricWeightShaping(d_model, d_model)
            shaping_input = input_ids_list
            if embedded:
                shaping_input = np.vstack([
                    x.reshape(-1, x.shape[-1]) if x.ndim >= 2 else x
                    for x in input_ids_list
                ])
            self.W_o     = self.shaping.weight_shaping(shaping_input)
            self.encoded = True

        # batch padding — unchanged
        if batch_size is not None and not embedded:
            input_ids_list = [
                self.batch_padding_utility(input_ids_list[i:i+batch_size], pad_token_id)
                for i in range(0, len(input_ids_list), batch_size)
            ]
            y_true_list = [
                np.stack(y_true_list[i:i+batch_size])
                for i in range(0, len(y_true_list), batch_size)
            ]

        AME = self.AME_Encoder(input_ids_list)

        # precompute LR schedule once
        lr_schedule = [
            self.learning_rate_warm_up(e, epochs, lr, schedule)
            for e in range(epochs)
        ]

        print(f"[==] Starting comprehensive training for {epochs} epochs with mode: {mode}, learning rate: {lr}, schedule: {schedule}")

        for y in y_true_list:
            y_arr = np.asarray(y, dtype=np.float64)

            # normalize to exactly 2D before smoothing
            if y_arr.ndim == 0:
                y_arr = y_arr.reshape(1, 1)
            elif y_arr.ndim == 1:
                y_arr = y_arr.reshape(1, -1)
            elif y_arr.ndim > 2:
                y_arr = y_arr.reshape(-1, y_arr.shape[-1])

            smoothed = self.smoothing_labels_utility(y_arr, smoothing=0.1)

            # guarantee 2D output regardless of what smoothing returns
            smoothed = np.asarray(smoothed, dtype=np.float64)
            if smoothed.ndim == 0:
                smoothed = smoothed.reshape(1, 1)
            elif smoothed.ndim == 1:
                smoothed = smoothed.reshape(1, -1)
            elif smoothed.ndim > 2:
                smoothed = smoothed.reshape(-1, smoothed.shape[-1])

            y_true_smoothed_list.append(smoothed)

        for epoch in range(epochs):
            epoch_losses = []
            epoch_accs   = []
        
            current_lr = lr_schedule[epoch]
            self.alpha = min(1.0, epoch / 100)

            for input_ids, y_true, y_true_smooth in zip(
                input_ids_list, y_true_list, y_true_smoothed_list
            ):
                if input_ids.ndim == 1:
                    input_ids    = input_ids[np.newaxis, :]
                if y_true.ndim == 1:
                    y_true       = y_true[np.newaxis, :]
                
                loss, acc = self.train_step(
                    input_ids, epoch, y_true,
                    current_lr, AME=AME, mode=mode,
                    embedded=embedded, max_norm=max_norm,
                    pad_token_id=pad_token_id,
                    y_true_smooth=y_true_smooth   # pass precomputed
                )
                epoch_losses.append(loss)
                epoch_accs.append(acc)

            avg_loss = float(np.mean(epoch_losses))
            avg_acc  = float(np.mean(epoch_accs))
            losses.append(avg_loss)
            accs.append(avg_acc)

            if epoch % 10 == 0:
                print(f"[=] Epoch {epoch} | loss: {avg_loss:.4f} | Acc: {avg_acc:.2%}")

        return losses, accs

    def train_step(self, input_ids, epoch, y_true, lr=0.01, AME=None,
                mode=None, embedded=False, max_norm=1.0,
                pad_token_id=0, y_true_smooth=None):

        y_true = np.asarray(y_true)
        if len(y_true.shape) < 2:
            y_true = y_true.reshape(-1, 1)

        if not embedded and input_ids.ndim == 1:
            input_ids = input_ids[np.newaxis, :]
        if y_true.ndim == 1:
            y_true = y_true[np.newaxis, :]

        probs, attn_weights = self.forward(
            input_ids, AME=AME, embedded=embedded,
            pad_token_id=pad_token_id, training=True,
            attn_dropout=self.attn_dropout_rate,
            ffn_dropout=self.ffn_dropout_rate
        )

        # comprehensive shape normalization before any use
        if y_true_smooth is not None:
            y_true_smooth = np.asarray(y_true_smooth, dtype=np.float64)

            # handle 0-d scalar — expand to (1, 1)
            if y_true_smooth.ndim == 0:
                print(f'[!] y_true_smooth was scalar ({float(y_true_smooth):.4f}) — recomputing')
                y_true_smooth = self.smoothing_labels_utility(y_true, smoothing=0.1)

            # squeeze extra leading dims safely
            while y_true_smooth.ndim > 2:
                y_true_smooth = y_true_smooth.squeeze(0)

            # ensure at least 2D
            if y_true_smooth.ndim == 1:
                y_true_smooth = y_true_smooth[np.newaxis, :]

            # final sanity — if still not 2D something is deeply wrong
            if y_true_smooth.ndim != 2:
                print(f'[!] y_true_smooth shape {y_true_smooth.shape} unrecoverable — recomputing')
                y_true_smooth = self.smoothing_labels_utility(y_true, smoothing=0.1)

        else:
            # compute fresh
            y_true_smooth = self.smoothing_labels_utility(y_true, smoothing=0.1)

        # shape alignment, y_true_smooth guaranteed 2D
        if y_true_smooth.shape[1] != probs.shape[1]:
            if y_true_smooth.shape[1] > probs.shape[1]:
                y_true_smooth = y_true_smooth[:, :probs.shape[1]]
                y_true        = y_true[:, :probs.shape[1]]
            else:
                pad           = probs.shape[1] - y_true_smooth.shape[1]
                y_true_smooth = np.pad(y_true_smooth, ((0, 0), (0, pad)))
                y_true        = np.pad(y_true,        ((0, 0), (0, pad)))

        loss     = -np.mean(np.sum(y_true_smooth * np.log(probs + 1e-8), axis=1))
        d_logits = (probs - y_true_smooth) / y_true_smooth.shape[0]

        if mode == 'fixed_backward':
            self.fixed_attention_backward(d_logits, lr, max_norm=max_norm)
        else:
            self.dynamic_backward(d_logits, lr, max_norm=max_norm)

        preds = np.argmax(probs, axis=1)
        true  = np.argmax(y_true, axis=1)
        acc   = float(np.mean(preds == true))

        return loss, acc
        
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

    def predict(self, input_ids, embedded=False):
        if not embedded and input_ids.ndim == 1:
            input_ids = input_ids.reshape(1, -1)

        AME = self.AME_Encoder(input_ids)
        probs, attn_weights = self.forward(input_ids, AME=AME, embedded=embedded, training=False, attn_dropout=0.0, ffn_dropout=0.0)
        preds = np.argmax(probs, axis=1)
        
        return preds, probs, attn_weights


    def AME_Encoder(self, x):
        x = self._sanitize_string_chars(x)

        # Optimized AME_Encoder for Transformer
        x = np.asarray(x)        
        if _OPT_AVAILABLE and np.asarray(x).ndim == 2:
            return optimized_ame_encoder(np.asarray(x, dtype=np.float64))     

        X = np.asarray(x)
        # Regular AME Equations, higher AME provides capabilities for the model to experience errors during abstraction
        # Lower AME means lower chance for un optimal abstraction.
        gradient = np.gradient(x, axis=-1)
        grad_energy = np.mean(np.linalg.norm(gradient, axis=-1))       
        X_mag = np.mean(np.linalg.norm(X, axis=-1))

        AME = np.log1p(X_mag) * np.log1p(grad_energy) 
        return AME


    def anisotropy_measurement(self, x):
        eps = 1e-5

        x = self._sanitize_string_chars(x)
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



    # attention quality computing provides the transformer a robust geometric complexity alignment scalar,
    #  this scalar can be used to compute alpha for a much stable forward pass in scarce data environment, allowing it to complement with AWE MLP below.
    def attention_quality_computing(self, attn_weights, AME=None, mask=None):
        eps = 1e-5
        batch, heads, seq_len, _ = attn_weights.shape

        if mask is not None:
            mask_expanded = np.broadcast_to(
                mask, (batch, heads, seq_len, seq_len)
            )
            # FIX 1 — inplace operations, ARM64 NEON works better on contiguous memory
            attn_weights = attn_weights * mask_expanded
            row_sums     = attn_weights.sum(axis=-1, keepdims=True) + eps
            attn_weights = attn_weights / row_sums

        # fuse AME and anisotropy into single gradient pass
        # instead of calling AME_Encoder + anisotropy_measurement separately
        # both call np.gradient internally — compute once, reuse
        unsuitable_shape_condition = (
            attn_weights.shape[0] % 2 != 0,
            attn_weights.shape[1] % 2 != 0,   
            attn_weights.shape[2] % 2 != 0, 
            attn_weights.shape[3] % 2 != 0,    
        )
        if unsuitable_shape_condition:
            return self.robust_attention_quality_computing(attn_weights, AME=AME, mask=mask)
        else:
            gradient = np.gradient(attn_weights)

        # AME inline — avoids second np.gradient call in AME_Encoder
        # gradient is already a list of arrays, one per dimension
        # stacked into single array for vectorized norm — ARM64 NEON friendly
        if AME is None:
            grad_stack  = np.stack([g.ravel() for g in gradient])  # (ndim, N)
            grad_norms  = np.linalg.norm(grad_stack, axis=1)       # (ndim,) — one NEON call
            grad_energy = grad_norms.mean()
            X_mag       = np.linalg.norm(attn_weights.ravel())  / attn_weights.size
            AME = np.log1p(X_mag) * np.log1p(grad_energy)

        AMR = 1.0 / (1.0 + np.exp(-float(AME)))

        # anisotropy inline — reuses grad_norms, no second gradient call
        anisotropy_val = grad_norms.std() / (grad_norms.mean() + eps)

        # fuse entropy + max + var into single pass over attn_weights
        # avoids 3 separate scans of same array
        flat        = attn_weights.reshape(batch * heads * seq_len, seq_len)

        # entropy — one pass
        log_w       = np.log(flat + eps)
        entropy     = -(flat * log_w).sum(axis=-1)            # (B*H*T,)
        norm_entropy = 1.0 - entropy.mean() / np.log(seq_len + eps)

        # max — same flat array
        avg_max     = flat.max(axis=-1).mean()

        # var — same flat array
        norm_var    = np.clip(flat.var() * seq_len, 0.0, 1.0)

        # quality score
        qualified     = (1.0 - AMR) + eps * anisotropy_val
        quality_score = (qualified * norm_entropy +
                        qualified * avg_max +
                        anisotropy_val * norm_var)

        return float(np.clip(quality_score, 0.0, 1.0)) 


    def robust_attention_quality_computing(self, attn_weights, AME=None, mask=None):
        eps = 1e-5
        eps = 1e-5
        batch, heads, seq_len, _ = attn_weights.shape    

        if AME is None:
            AME = self.AME_Encoder(attn_weights)

        anisotropy = self.anisotropy_measurement(attn_weights)

        entropy = -np.sum(attn_weights * np.log(attn_weights + eps), axis=-1)
        max_entropy = np.log(seq_len)
        norm_entropy = 1.0 - (np.mean(entropy) / max_entropy)

        max_attn = np.max(attn_weights, axis=-1)
        avg_max = np.mean(max_attn)

        var_attn = np.var(attn_weights)
        norm_var = np.clip(var_attn * seq_len, 0, 1)

        AMR = 1.0 / (1.0 + np.exp(-AME))  # abstract modelling rate
        qualified = (1.0 - AMR) + eps * anisotropy

        quality_score = qualified * norm_entropy + qualified * avg_max + anisotropy * norm_var
        dynamic_alpha = np.clip(quality_score, 0, 1.0)

        return dynamic_alpha



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

    def multi_modal_linear_transformation(self, x, perf_score):
        x = self._sanitize_string_chars(x)

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
                
            z = (sub_z + subnet_B) * perf_score  

        return z


    def forward(self, x, perf_score):

        x = self._sanitize_string_chars(x)
        self.x = x
        self.z = self.multi_modal_linear_transformation(x, perf_score)

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

class MLP:
    def __init__(self):
        self.layers = []
        self.layers2 = []
        self.lr = 0.1
        self.feed_layers = []

        self.error_counts = None
        self.pred_counts = None
        self.error_decay = None

        self.softmax = SoftmaxOutput()
      

    def feed_add(self, layer):
        self.feed_layers.append(layer)

    def add(self, layer):
        self.layers.append(layer)

    def focused_forward(self, x, AME=None, anisotropy=None):    
        performance_score = self.performance_calculation(x, AME=AME, anisotropy=anisotropy) 

        for layer in self.feed_layers:
            x = layer.forward(np.asarray(x, dtype=np.float64), performance_score)
            
        return self.softmax.forward(x) 

    def performance_calculation(self, x, AME=None, anisotropy=None):
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
            performance_score = 0.5

        return performance_score

    def forward(self, x, AME=None, anisotropy=None):
        eps = 1e-5
        performance_score = self.performance_calculation(x, AME=AME, anisotropy=anisotropy)

        for layer in self.layers:
            x = layer.forward(x, performance_score)

        output = self.softmax.forward(x)

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


    def _calibrate_gradient(self, grad, AME, anisotropy):
        std = np.std(grad)
        eps = 1e-5
        calibration = grad.copy()

        if std > 0.1:
            AEL = (1.0 - AME) * anisotropy # abstraction enviromental limit
            PRA = (1.0 - AEL) * std # possible reflected abstraction 

            calibration = grad * AEL * PRA
            calibration /= np.sum(calibration)

        calibration = np.asarray(calibration, dtype=np.float64)
        if np.isnan(calibration).any() or np.isinf(calibration).any():
            calibration = grad.copy()

        return calibration



    def continuous_predictive_correction(self, manager, prob, predicted_index):
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

            self.pred_counts = self.pred_counts[0] if isinstance(self.pred_counts[0], np.ndarray) and self.pred_counts.ndim > 1 else self.pred_counts

            for c in range(n_classes):
                if len(self.pred_counts) < c:
                    if isinstance(self.pred_counts[c], (int, float)) and self.pred_counts[c] > 0:
                        error_rate    = self.error_counts[c] / (self.pred_counts[c] + 1e-8)
                        # sigmoid-shaped dampening — never goes negative
                        # error_rate=0.0 → multiplier=1.0 (no change)
                        # error_rate=0.5 → multiplier≈0.67
                        # error_rate=1.0 → multiplier≈0.5
                        reputation    = 1.0 / (1.0 + error_rate)
                        if c < len(prob):
                            prob[c]  *= reputation  
                else:
                    self.pred_counts = np.zeros(n_classes, dtype=np.float64)
                    self.pred_counts[predicted_index] += 1.0
                    self.pred_counts = self.pred_counts[0] if isinstance(self.pred_counts[0], np.ndarray) and self.pred_counts.ndim > 1 else self.pred_counts

                    if isinstance(self.pred_counts[c], (int, float)) and self.pred_counts[c] > 0:
                        error_rate    = self.error_counts[c] / (self.pred_counts[c] + 1e-8)
                    
                        reputation    = 1.0 / (1.0 + error_rate)
                        if c < len(prob):
                            prob[c]  *= reputation
                    
            prob_sum = prob.sum()
            if prob_sum > 1e-8:
                prob /= prob_sum

            # re adapt shape of pred_counts and error_counts if they don't match prob shape
            if self.pred_counts.shape != prob.shape:
                self.pred_counts = np.zeros_like(prob)
            if self.error_counts.shape != prob.shape:
                self.error_counts = np.zeros_like(prob)

        except Exception as e:
            print(f'[!] Cant check and calibrate probs based on penalty due to: {e}')    
            traceback.print_exc()
        
        return prob 


    def focused_backward(self, grad, lr, AME, anisotropy):
        grad = self._calibrate_gradient(grad, AME, anisotropy)        
        grad = self.softmax.backward(grad)
        for layer in reversed(self.feed_layers):
            grad = layer.backward(grad, lr)
        return grad  

    def backward(self, grad, lr):
        grad = self.softmax.backward(grad)
        for layer in reversed(self.layers):
            grad = layer.backward(grad, lr)
        return grad
            
    def predict(self, X, y, epochs=1000, verbose=True):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = Loss.categorical_crossentropy(y, y_pred)    		
            if verbose and epoch % 100 == 0:
                acc = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))
                self.acc2.append(acc)
                print(f"[=] Epoch {epoch} | loss:{loss:.4f} | Acc: {acc:.2f}")

	   
    def prediction(self, X):
        y_pred = self.forward(X)   
        return y_pred      
     
         
    def AME_Encoder(self, x):
        X = np.asarray(x)
        if _OPT_AVAILABLE and np.asarray(X).ndim == 2:
            return optimized_ame_encoder(np.asarray(X, dtype=np.float64))     

        if x.shape[1] == 1:
            x = x.T
            x= x.flatten()

        gradient = np.gradient(x, axis=-1)
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
        anisotropy = np.std(val) + eps / np.mean(val) 
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

            # align batch and class dims — same approach as Loss
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

    def train(self, X, y, epochs=1000, lr=0.01, verbose=True):
        X = self._sanitize_string_chars(X)
        y = self._sanitize_string_chars(y)

        AME = self.AME_Encoder(X)     
        anisotropy = self.anisotropy_measurement(X)

        focused_fit_condition = len(self.feed_layers) > 0 and anisotropy > 0.25 and AME > 0.25
        print(f'[+] Focused fit condition: {focused_fit_condition} || Anisotropy: {self.anisotropy_measurement(X):.4f} || AME: {self.AME_Encoder(X):.4f}')

        for epoch in range(epochs):
            if not focused_fit_condition:
                y_pred = self.forward(X, AME=AME, anisotropy=anisotropy)
            else:
                y_pred = self.focused_forward(X, AME=AME, anisotropy=anisotropy)

            y_pred, y_true = self.adapt_predict_shape(y_pred, y)
            loss = Loss.categorical_crossentropy(y_true, y_pred)
            grad = Loss.softmax_crossentropy_derivative(y_true, y_pred)
            if focused_fit_condition:
                _ = self.focused_backward(grad, self.lr, AME, anisotropy)
            else:
                _ = self.backward(grad, self.lr)

            if verbose and epoch % 100 == 0:
                acc = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))
                print(f"[=] Epoch {epoch} | Loss: {loss:.4f} | Acc: {acc:.2f}")
              
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

    # _________ forward method for cell class _____________
    def forward(self, x_seq: np.ndarray, h0=None, c0=None):

        # Optimized LSTM Cell Implementation with Cython based Language.
        if _OPT_AVAILABLE:
            hs, cs, cache = optimized_lstm_cell_forward(
                np.ascontiguousarray(x_seq, dtype=np.float64),  # force float64
                np.ascontiguousarray(self.W,  dtype=np.float64),
                np.ascontiguousarray(self.b,  dtype=np.float64),
                self.input_size,
                self.hidden_size
            )
            return hs, cs, cache 

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
        # T_limit avoids slicing cache list externally Later.
        T = T_limit if T_limit is not None else len(cache)
        H = self.hidden_size

        dW     = np.zeros_like(self.W)
        db     = np.zeros_like(self.b)
        dh     = np.zeros(H) if dh_next is None else np.ascontiguousarray(dh_next.copy())
        dc     = np.zeros(H) if dc_next is None else np.ascontiguousarray(dc_next.copy())
        dx_seq = np.zeros((T, self.input_size))
        if _OPT_AVAILABLE:
            grads, dx_seq, dh, dc = optimized_lstm_cell_backward(
                np.ascontiguousarray(dhs, dtype=np.float64),
                cache,
                np.ascontiguousarray(self.W, dtype=np.float64),
                self.input_size,
                self.hidden_size,
                dh,
                dc,
                T
            )
            return grads, dx_seq, dh, dc

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

        eps = 1e-5
        T            = x_seq.shape[0]
        H            = self.model.cell.hidden_size
        expected_input = self.model.cell.input_size
        p            = self.dropout
        cell         = self.model.cell
        W            = cell.W
        b            = cell.b
        H1, H2, H3  = H, H * 2, H * 3   # slice boundaries precomputed

        # FIX 1 — Wy check once before loop, not T times inside
        if self.model.Wy is None:
            self.model.Wy = self.model.weight_shaper.weight_shaping(x_seq)
        Wy = self.model.Wy
        by = self.model.by

        h     = np.zeros(H)
        c     = np.zeros(H)
        xh    = np.empty(expected_input + H)  # preallocate concat buffer
        preds = np.empty(T)                   # preallocate output

        # precompute dropout scale factor
        inv_keep = 1.0 / (1.0 - p) + eps

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

            # FIX 4 — precomputed inv_keep, inplace mask application
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

        eps = 1e-3
        model   = self.model
        AME = self.pipeline.AME_Encoder(X)
        AMR = 1.0 / (1.0 + np.exp(-AME))

        n_train = int((1.0 - AMR) * len(X))
        if np.isnan(n_train) or np.isinf(n_train) or n_train <= eps:
            n_train = int(1.0 + self.pipeline.confidence_threshold * len(X))
            
        X_tr, Y_tr = X[:n_train], Y[:n_train]
        X_te, Y_te = X[n_train:], Y[n_train:]

        idx = np.arange(n_train)

        for epoch in range(1, epochs + 1):
            np.random.shuffle(idx)
            epoch_loss = 0.0
            for j in idx:
                loss, _ = model.train_step(X_tr[j], Y_tr[j], lr=lr, AMR=AMR)
                epoch_loss += loss
            epoch_loss /= n_train + eps

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
                    # calculates how much value loss when multiplied by model error rate 
                    # to gain how much the model can greatly applied its losses efficiently vs the possible error rate during prediction.
                    # Higher val_loss correlates to the model possible bad predicted capabilities in the future Training 
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

        if np.isnan(AMR) or np.isinf(AMR) or AMR <= 1e-10:
            AMR = self.pipeline.confidence_threshold + 1e-5

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
        print(f" │  Input  size   : {I:<24}                │")             
        print(f" │  Hidden size   : {H:<24}                │")
        print(f" │  Output size   : {O:<24}                │")
        print(f" │  LSTM   params : {W_params:<24,}        │")
        print(f" │  Linear params : {Wy_params:<24,}       │")
        print(f" │  Total  params : {total:<24,}           │")
        print("  └─────────────────────────────────────────┘")





# weighted ensemble predictor that dynamically adjusts the weights of the transformer and MLP based on the input data's geometric complexity and the attention quality>
# allowing it to leverage the strengths of both models for improved performance across a wider range of data complexities.
class WeightedEnsemblePredictor:
    def __init__(self, pipeline, distribution, memory_name):
        self.pipeline = pipeline
        self.storage = ModelStorage(pipeline, memory_name, db_path='activity_log.db')
        self.inference = distribution
        self.query_node = QueryNode(pipeline, memory_name, self.storage)

        self.transformer_weight = 0.5  # Initial equal weight
        self.mlp_weight = 0.5 # initial equal mlp weight
        self.calibration_history = []
        self.explainer = ExplainabilityModule(pipeline, self)
        self.memory_name = memory_name
        self.db_path = 'activity_log.db'

        self.error_counts = None
        self.pred_counts = None
        self.error_decay = 0.85

        self.self_attn_weights = None

        if not self.storage.memory_exists(self.memory_name, type='Transformer'):
            self.memory = {}
        else:
            self.memory = self.storage.memory_retrieval(self.memory_name, type_func='Transformer', verbose=True)

    def _get_lstm_probs(self, input_ids, X_mlp, label_bins=None, confidence_level=0.90):
        """
        Convert LSTMEngine prediction output into a probability
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
        memory_name = self.memory_name

        clean_X = self._sanitize_for_storage(X)
        clean_X2 = self._sanitize_for_storage(X2)
        clean_X3 = self._sanitize_for_storage(X3)
        clean_X4 = self._sanitize_for_storage(X4)

        self.memory['TA'] = clean_X, text, clean_X2, clean_X3, clean_X4

        self.storage.save_model_dict(memory_name, self.memory, type='Transformer', model_type='attention')

        print('🚀 Memory Probability Added!')




    def explainability_prediction_batch(self, texts, mlp_probs, trans_probs, attn_weights, show_explanation=False):
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
        if type == 'Transformer':
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
        results = []
        for text in texts:
            result = self.predict_single(text, mlp_probs, trans_probs, attn_weights, show_explanation)
            results.append(result)
            explanation = result['explanation']
            print(f'[+] Explanation: {explanation}')
        
        return results
    


    def predict_ensemble(self, input_ids, X_mlp, y_true, method='dynamic', embedded=False):
        label_bins=None
        if self.pipeline.cache and 'label_bins' in self.pipeline.cache:
            print('[=] label_bins cache found!')
            label_bins = self.pipeline.cache['label_bins']

        AME = self.pipeline.model2.AME_Encoder(input_ids)

        trans_probs, attn_weights = self.pipeline.model2.forward(input_ids, AME=AME, embedded=embedded)
        mlp_probs = self.pipeline.model3.forward(X_mlp)
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
            # Use attention to weight transformer vs MLP
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
            print('[-] Ensemble probs is invalid , using MLP probs as.')
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
        # normalize all inputs to guaranteed 2D float64
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
                        has_lstm
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
                mw = mlp_cf   * (1.0 + agreement) / 2.0

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
            print(f'[!] Cant do ensemble prediction due ensemble prediction due to: {e}, returning MLP probabilities')
            return mlp_probs.copy()


    
    def _attention_weighted_ensemble(self, trans_probs, mlp_probs, attn_weights):
        if attn_weights is None:
            return (trans_probs + mlp_probs) / 2
        
        batch_size = trans_probs.shape[0]
        ensemble = np.zeros_like(trans_probs)

        n_trans_classes = trans_probs.shape[1]        
        n_mlp_classes = mlp_probs.shape[1]
        
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
                anisotropy = self.anisotropy_measurement(attn)                
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
            
            # MLP gets the rest
            mlp_trust = 1.0 - trans_trust

            try:
                ensemble[i] = trans_trust * trans_row + mlp_trust * mlp_row
            except:
                ensemble = trans_trust * trans_row + mlp_trust * mlp_row
        
        return ensemble
    
    def _meta_ensemble(self, trans_probs, mlp_probs, attn_weights, X_mlp, 
                         lstm_probs=None, lstm_weight_hint=0.0):
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
                    np.max(trans_row),           # Transformer confidence
                    np.max(mlp_row),              # MLP confidence
                    np.std(trans_row),             # Transformer spread
                    np.std(mlp_row),               # MLP spread
                    1.0 if np.argmax(trans_row) == np.argmax(mlp_row) else 0.0,  # Agreement
                ]
            else:
                features = [
                    np.max(trans_row),           # Transformer confidence
                    np.max(mlp_row),              # MLP confidence
                    np.max(lstm_row),
                    np.std(trans_row),             # Transformer spread
                    np.std(mlp_row),               # MLP spread
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
            agreement = meta_features[i, 4]
            
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

        print(f"\n✅ Optimal weights: Transformer: {best_weight:.2f}, MLP={1-best_weight:.2f}")
        print(f"[-] Validation accuracy: {best_accuracy:.2%}")
        
        return best_weight

# Cross-session automation module that allows exporting and importing of sessions, syncing with another device, and listing available sessions for better management and continuity of work across different environments.
class CrossSessionAutomation:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def export_session(self, session_name=None):
        if session_name is None:
            session_name = f"session_{self.session_id}"
        
        session_data = {
            'session_id': self.session_id,
            'session_name': session_name,
            'timestamp': datetime.now().isoformat(),
            'memories': self.pipeline.memory.copy(),
        }
        
        filename = f"{session_name}.json"
        with open(filename, 'w') as f:
            json.dump(session_data, f, default=str)
        
        print(f"💾 Session exported to: {filename}")
        return filename
    
    def import_session(self, filename):
        with open(filename, 'r') as f:
            session_data = json.load(f)
        
        print(f"\n📥 Importing session: {session_data['session_name']}")
        print(f"   Created: {session_data['timestamp']}")
        print(f"   Memories: {len(session_data['memories'])}")
        
        # Merge memories
        for key, value in session_data['memories'].items():
            if key not in self.pipeline.memory:
                self.pipeline.memory[key] = value
        
        print(f"✅ Session imported! Total memories: {len(self.pipeline.memory)}")
    
    def sync_with_another_device(self, device_ip, port=5000):
        import socket
        import pickle
        
        # Export current session
        temp_file = self.export_session(f"sync_{self.session_id}")
        
        try:
            with self.ssl_context.wrap_socket(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
                s.connect((device_ip, port))
                with open(temp_file, 'rb') as f:
                    s.sendall(f.read())
                print(f"📡 Synced to {device_ip} || {port}")
                print('🚀 Succesfully sync and export memory session to another device! ')                 
        except Exception as e:
            print(f"❌ Sync failed: {e}")
            pass
        

    
    def list_sessions(self, name):
        import glob

        sessions = glob.glob(f"{name}*.json")
        
        print(f"\n📚 Available Sessions: {sessions}")
        if sessions:
            for session in sessions:
                with open(session, 'r') as f:
                    data = json.load(f)
                    print(f"   • {session}: {data['session_name']} ({len(data['memories'])} memories)")

        else:          
            print('[-] No available sessions! ')
        
        return sessions


# Explainability module that provides detailed explanations for predictions, allows learning from user feedback, and maintains a history of decisions for transparency and continuous improvement of the model.
class ExplainabilityModule:
    def __init__(self, pipeline, predictor):
        self.pipeline = pipeline
        self.decision_history = []     

        self.decision_history = []     
        
        self.uncertainty_threshold = 0.2
        self.pending_queries = []
        self.learned_from_feedback = []   
        self.feedback_buffer = []  # Store feedback for batch training
        self.buffer_size = 10  # Train after every 10 feedbacks

        self.supervised_learning = True


    def data_preparation(self, titles, labels):
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

        self.pipeline.initialize_fitting(raw)
        X_raw = self.pipeline.tfidf.transform(raw).toarray()
        X_raw = self._refit_sparse_data(X_raw, raw)

        return datasets, X_raw

    
    def draw_bar(self, value, max_width=20):
        value = max(0, min(1, value))  # Ensure value is between 0 and 1
        filled = int(value * max_width)
        return '█' * filled + '░' * (max_width - filled)

    def _learn_from_feedback(self, text, correct_label, wrong_result, batch_size=2):
        eps = 1e-5
        print(f"\n[📚] Learning: '{text}' → {correct_label}...")
        
        # 1. Convert to features
        X_intents = self.pipeline.tfidf.transform(self.pipeline.intents).toarray()
        X_input = self.pipeline.tfidf.transform([text]).toarray()
        X_raw = np.dot(X_intents, X_input.T).T

        if np.allclose(X_raw, 0.0) or self.pipeline.anisotropy_measurement(X_raw) < 0.3 or np.isnotfinite(X_raw).any():
            checksum = int(hashlib.md5(text.encode()).hexdigest(), 16) % 1000 / 10000
            X_raw[0, 0] = checksum + eps
            
        X = X_raw.copy()
    
        try:
            print('[🔄] Verifying if similar correct_label is already in pipeline supervised memory...')

            memory_key = f'supervised_memory'
            retrieved = [key for key, (corr_label) in self.pipeline.memory.items() if key.startswith(memory_key) and correct_label == corr_label]
            if retrieved:
                for retrieve in retrieved:
                    _, correct_label = self.pipeline.memory[retrieve]

                print(f'[✅] retrieved similar correct label: {correct_label}')
                print(f'[=] This proves consistency over time necessary for gradual supervised learning loop to provide transparency')
            else:
                print('[-] No similar matching correct label')
                print('[=] This suggests that the model has never learned this previous input in supervised learning')

        except Exception as e:
            print(f'[=] Cant save to and retrieve memory due to {e} error.')

        # 2. Convert label to one-hot
        label_idx = self.pipeline.intents[correct_label] if correct_label in self.pipeline.intents else 0
        if label_idx is None:
            label_idx = len(self.pipeline.intents)
            self.pipeline.intents[correct_label] = label_idx
        
        y_onehot = np.zeros((1, len(self.pipeline.intents)))
        y_onehot[0, label_idx] = 1
        if y_onehot.shape[1] != X.shape[1]:
            if y_onehot.shape[1] < X.shape[1]:
                y_onehot = np.pad(y_onehot, ((0, 0), (0, X.shape[1] - y_onehot.shape[1])), mode='constant')
            else:
                X = np.pad(X, ((0, 0), (0, y_onehot.shape[1] - X.shape[1])), mode='constant')
        
        # 3. IMMEDIATE TRAINING (single step with higher Learning Rate)
        anisotropy = self.pipeline.anisotropy_measurement(X)

        anisotropy_dist = 1.0 / (1.0 + np.exp(-anisotropy))
        deviation = 1.0 / (1.0 + np.std(X))
        AEL = (1.0 - deviation) * anisotropy_dist + eps

        old_lr = self.pipeline.mlp.lr
        self.pipeline.mlp.lr = 2 / (1.0 + AEL) # use stable learning rate that match the environment complexity for correction
        print(f"[=] Training MLP on corrected example with boosted LR: {self.pipeline.mlp.lr}...")

        # Train on this single example for a few epochs
        self.pipeline.focused_mlp.train(X, y_onehot, epochs=1000, lr=self.pipeline.mlp.lr, verbose=True)
        time.sleep(5)

        self.pipeline.mlp.lr = old_lr  # Restore old LR
        
        # 4. train transformer for efficient processing later tho.
        if self.pipeline.model2:
            input_ids = np.array([self.pipeline.encode(text, self.pipeline.vocab)])
            for _ in range(5):
                self.pipeline.model2.train_step(input_ids, 0, y_onehot, lr=0.1, mode='fixed_backward')
        
        # 5. Store in memory gate for fast retrieval
        self.pipeline.modular_prediction_saving(
            self.pipeline.encode(text, self.pipeline.vocab),
            X,
            correct_label
        )
        
        # 6. Add to buffer for batch consolidation later.
        self.feedback_buffer.append((X, y_onehot, text, correct_label))
        
        # 7. Batch train when buffer is full
        if len(self.feedback_buffer) >= self.buffer_size:
            print(f"\n[🔄] Buffer full with {len(self.feedback_buffer)} feedback examples. Starting batch training...")
            self._batch_train_from_feedback()
        
        print(f"[✅] Learned: '{text}' → {correct_label} (model weights updated)")

        supervised_memory = {
            'input': text,
            'label': correct_label,
            'original_prediction': wrong_result['final_label'], 
            'original_confidence': wrong_result['final_confidence'],
            'timestamp': datetime.now(),
            'learned': True

        }

        self.learned_from_feedback.append(supervised_memory)
        if hasattr(self.pipeline, 'memory'):
            print('[🔄] Applying correct label to pipelines memory')
            memory_key = f'supervised_memory'
            self.pipeline.memory[memory_key] = (X, correct_label)

        if len(self.learned_from_feedback) % 10 == 0:
            self.consolidate_supervised_memories(batch_size=batch_size)

        return X
    
    def _batch_train_from_feedback(self):
        if not self.feedback_buffer:
            return
        
        print(f"\n🔄 Batch training on {len(self.feedback_buffer)} feedback examples...")
        
        # Collect all feedback
        # Determine max dimensions

        max_x_dim = max(fb[0].shape[1] for fb in self.feedback_buffer)
        current_y_dim = len(self.pipeline.intents)
        
        # Collect all feedback with padding
        X_list = []
        y_list = []
        for fb in self.feedback_buffer:
            X = fb[0]
            y = fb[1]
            if X.shape[1] < max_x_dim:
                X = np.pad(X, ((0, 0), (0, max_x_dim - X.shape[1])), mode='constant')
            if y.shape[1] < current_y_dim:
                y = np.pad(y, ((0, 0), (0, current_y_dim - y.shape[1])), mode='constant')
            X_list.append(X)
            y_list.append(y)
        
        X_batch = np.vstack(X_list)
        y_batch = np.vstack(y_list)

        # Train MLP on batch
        old_lr = self.pipeline.mlp.lr
        self.pipeline.mlp.lr = old_lr * 2
        for epoch in range(20):
            y_pred = self.pipeline.mlp.forward(X_batch)
            loss = Loss.categorical_crossentropy(y_batch, y_pred)
            grad = Loss.softmax_crossentropy_derivative(y_batch, y_pred)
            self.pipeline.mlp.backward(grad, self.pipeline.mlp.lr)

        self.pipeline.mlp.lr = old_lr
        
        # Clear buffer
        self.feedback_buffer = []
        print("[✅] Batch training complete")

    
    def _ask_for_feedback(self, text, result, explanation):
        print("\n" + "="*60)
        print(f"[🤔] I'm confused about this detail: '{text}'")
        print(f"[=] I thought: {result['final_label']} ({result['final_confidence']:.1%})")
        if 'final_label' in result and 'final_confidence' in result:    
            confidence = result['final_confidence']
            bar = self.draw_bar(confidence)   
            print(f"[+] Confidence: [{bar}] {confidence:.1%}")  

        # Show top 3 predictions
        if 'details' in result and 'all_probs' in result['details']:
            probs = result['details']['all_probs']
            top3 = np.argsort(probs)[-3:][::-1]            
            print("\n[+] Top possibilities:")
            for idx in top3:
                label = self.pipeline.intents.get(idx, f"class_{idx}")
                print(f"[=]  • {label}: {probs[idx]:.1%}")
        
        print("\n[📚] Options:")
        print("  1. Enter correct label")
        print("  2. Skip")
        print("  3. Show explanation")
        print('  4. Get decision history')
        
        choice = input("\n [=] What is the correct label? (ex: break/work): ").strip()
        
        if choice.lower() == 'skip':
            return None
        elif choice.lower() == 'explain':
            print(explanation)
            return self._ask_for_feedback(text, result, explanation)
        elif choice == '4':
            history = self.get_decision_history(limit=10)  
            for entry in history:
                print(f" [?] {entry['input']} → {entry['label']}")
        elif choice:
            print('[==] Assigning Training for correct label: ')
            return choice
        return None

    def analyze_with_feedback(self, details, input_text, mlp_probs, trans_probs, attn_weights, explanation, batch_size=2, auto_ask=True):
        uncertain = self.pipeline.confidence_threshold

        input_ids = np.array([self.pipeline.encode(input_text, self.pipeline.vocab)])
        if isinstance(input_ids, list):
            input_ids = np.array(input_ids)

        if uncertain == 0.0:
            uncertain = self.uncertainty_threshold

        is_uncertain = details['final_confidence'] < uncertain
        
        if is_uncertain and self.supervised_learning:
            feedback = self._ask_for_feedback(input_text, details, explanation)
            if feedback:
                print(f"[📚] Received feedback: '{input_text}' should be '{feedback}'")
                print('[=] Supervised learning took many trials to get right. This is normal. Please be patient as the model updates continously each label request...')

                evaluated_input = self._learn_from_feedback(input_text, feedback, details, batch_size=2)
                self.supervised_learning = False  # Prevent infinite loop
                return False
        
        return False

    def _refit_sparse_data(self, X_features, texts, threshold=0.3):
        """Refit TF-IDF if zero-row ratio exceeds threshold."""
        X_features = np.asarray(X_features, dtype=np.float32)
        if X_features.ndim == 1:
            X_features = X_features.reshape(1, -1)        
            X_features = np.asarray(X_features)
                    
        zero_rows = np.where(X_features.sum(axis=1) == 0)[0]
        zero_ratio = len(zero_rows) / len(X_features)
        
        if zero_ratio > threshold:
            print(f'[!] {len(zero_rows)} zero rows ({zero_ratio:.0%}), refitting TF-IDF on current batch')
            self.tfidf.fit(texts)
            X_features = self.tfidf.transform(texts).toarray()
            
            # second pass — fill remaining zeros with checksum fingerprint
            zero_rows = np.where(X_features.sum(axis=1) == 0)[0]
            for i in zero_rows:
                text = texts[i] if isinstance(texts[i], str) else str(texts[i])
                checksum = int(hashlib.md5(text.encode()).hexdigest(), 16)
                rng = np.random.default_rng(checksum)
                X_features[i] = rng.uniform(0.01, 0.1, size=X_features.shape[1])
                print(f'[!] Row {i} still zero after refit, checksum fallback applied')
        
        return X_features

    def _get_lstm_explanation(self, lstm_result: dict) -> Any:
        """
        Extract readable signals from LSTMEngine.predict() output.
        lstm_result is the raw dict returned by engine.predict().
        """
        if lstm_result is None:
            return None

        mc_conf_last     = float(lstm_result['mc_confidence'][-1])
        gate_unc_last    = float(lstm_result['gate_uncertainty'][-1])
        overall          = float(lstm_result['overall'])
        interval_low     = float(lstm_result['interval_low'][-1])
        interval_high    = float(lstm_result['interval_high'][-1])
        mc_std_last      = float(lstm_result['mc_std'][-1])

        # gate stability — inverse of uncertainty, easier to read
        gate_stability   = 1.0 - gate_unc_last

        # dominant label from label_confidence if available
        label_conf       = lstm_result.get('label_confidence')
        dominant_label   = None
        dominant_prob    = 0.0
        if label_conf:
            dominant_label = max(label_conf, key=label_conf.get)
            dominant_prob  = label_conf[dominant_label]

        return {
            'mc_confidence'   : mc_conf_last,     # how tight MC dropout samples are
            'gate_stability'  : gate_stability,    # 1 = stable memory, 0 = actively overwriting
            'gate_uncertainty': gate_unc_last,     # raw gate signal
            'overall'         : overall,           # combined scalar
            'interval'        : (interval_low, interval_high),  # prediction interval
            'mc_std'          : mc_std_last,       # spread of MC samples
            'dominant_label'  : dominant_label,    # top label_bin if bins were passed
            'dominant_prob'   : dominant_prob,
            'label_confidence': label_conf
        }

    def consolidate_supervised_memories(self, batch_size=2):
        if not self.learned_from_feedback:
            return
        
        print(f"\n🔄 Consolidating {len(self.learned_from_feedback)} supervised memories...")
        
        # Extract all supervised examples
        texts = [m['input'] for m in self.learned_from_feedback]
        labels = [m['label'] for m in self.learned_from_feedback]

        dataset, _ = self.data_preparation(texts, labels)
        self.initialize_fitting(texts)

        X = self.tfidf.transform(texts).toarray()
        X = self._refit_sparse_data(X, texts)

        try:
            unique_labels = sorted(set(labels))
            label_to_idx  = {l: i for i, l in enumerate(unique_labels)}
            y_indices     = np.array([label_to_idx[l] for l in labels])

            n_classes = len(unique_labels)
            y_onehot  = np.zeros((len(y_indices), n_classes))
            y_onehot[np.arange(len(y_indices)), y_indices] = 1

            self.pipeline.model3.train(X, y_onehot, epochs=100, lr=0.01, verbose=True)
            
            print("[✅] Supervised memories consolidated!")
        except Exception as e:
            print(f"[❌] Error during memory consolidation: {e}")

    
    def get_uncertain_predictions(self, result):
        uncertain = []
        if result['final_confidence'] < self.uncertainty_threshold:
            uncertain.append({
                    'text': result['input_text'],
                    'prediction': result['final_label'],
                    'confidence': result['final_confidence'],
                    'attention_quality': result['attention_quality']
            })
        
        # Sort by most uncertain first
        uncertain.sort(key=lambda x: x['confidence'])
        
        print(f"\n🔍 Found {len(uncertain)} uncertain predictions:")
        for u in uncertain[:10]:
            print(f"   • '{u['text']}' → {u['prediction']} ({u['confidence']:.1%})")
        
        return uncertain
 

    def _get_prediction_details(self, input_text, mlp_probs, trans_probs, attn_weights, lstm_result=None, batch_size=2):
        show_details = self.pipeline.show_explainability_details
        if trans_probs.ndim == 1:
            trans_probs = trans_probs.reshape(1, -1)

        trans_pred = np.argmax(trans_probs[0])
        trans_conf = trans_probs[0][trans_pred]
        
        # Handle mlp_probs - ensure it's 2D
        try:
            if mlp_probs.ndim == 1:
                mlp_probs = mlp_probs.reshape(1, -1)
            mlp_pred = np.argmax(mlp_probs)
            mlp_conf = mlp_probs[mlp_pred]
        except:
            if isinstance(mlp_probs, float):
                mlp_pred = int(mlp_probs)
                mlp_conf = 0.15
            else:
                mlp_pred = np.argmax(mlp_probs[0])
                mlp_conf = mlp_probs[0][mlp_pred]

        if isinstance(mlp_conf, np.ndarray):
            mlp_conf = np.clip(np.mean(mlp_conf), 0, 1)
        if isinstance(trans_conf, np.ndarray):
            trans_conf = np.clip(np.mean(trans_conf), 0, 1)

        reverse_map = self.pipeline.reverse_map
        
        final_pred, final_conf = self._get_final_output(
            mlp_pred, mlp_conf, trans_pred, trans_conf, attn_weights
        )
        
        # Extract attention focus
        focus_words = self._get_attention_focus(attn_weights, input_text)
        
        # Extract geometric features
        geometric_features = self._get_geometric_features(input_text)

        details = {
            'input_text': input_text,
            'final_label': reverse_map.get(final_pred, f"class_{final_pred}"),
            'final_confidence': final_conf,
            'final_class': final_pred,
            'mlp': {
                'label': reverse_map.get(mlp_pred, f"class_{mlp_pred}"),
                'confidence': mlp_conf,
                'class': mlp_pred
            },
            'transformer': {
                'label': reverse_map.get(trans_pred, f"class_{trans_pred}"),
                'confidence': trans_conf,
                'class': trans_pred,
                'attention_words': focus_words,
                'attention_weights': attn_weights
            },
            'lstm': self._get_lstm_explanation(lstm_result) if lstm_result is not None else None,
            'geometric_features': geometric_features,
            'agreement': mlp_pred == trans_pred,
            'anisotropy': self._compute_anisotropy(attn_weights) if attn_weights is not None else None,
            'attention_quality': self._compute_attention_quality(attn_weights) if attn_weights is not None else None
        }

        explanation, confidence, comparison = self._generate_explanation(details)
        
        self.decision_history.append({
            'timestamp': datetime.now(),
            'input': input_text,
            'prediction': details['final_label'],
            'confidence': details['final_confidence'],
            'explanation': explanation,
            'details': details
        })
        
        if show_details:
            self._display_explanation(explanation)
            self._display_explanation(confidence)
            self._display_explanation(comparison)
            self.get_uncertain_predictions(details)
      
            if details['final_confidence'] < 0.15 and not self.pipeline.autonomous:
                self.analyze_with_feedback(details, input_text, mlp_probs, trans_probs, attn_weights, explanation, batch_size=2)
 
        confidence = self.explain_confidence(details)
        if final_conf:
            print('[||] Final confidence set to: ', final_conf)
            self.pipeline.final_conf_score = final_conf

        return details, explanation
    
    
    def _get_final_output(self, mlp_pred, mlp_conf, trans_pred, trans_conf, attn_weights):
        eps = 1e-5
        if isinstance(mlp_conf, np.ndarray):
            mlp_conf = np.clip(np.mean(mlp_conf), 0, 1)
        if isinstance(trans_conf, np.ndarray):
            trans_conf = np.clip(np.mean(trans_conf), 0, 1)

        if mlp_pred == trans_pred:
            final_pred = mlp_pred
            final_conf = max(mlp_conf, trans_conf)
        else:
            sliced_attention_weight = attn_weights[0]
            if isinstance(sliced_attention_weight, np.ndarray):
                sliced_attention_weight = sliced_attention_weight[:, 0]
                sliced_attention_weight = sliced_attention_weight[0]
               
            sliced_anisotropy = self.pipeline.anisotropy_measurement(sliced_attention_weight) 
            sigmoid_growth = 1.0 / (1.0 + np.exp(-sliced_attention_weight))
            attn_quality = self._compute_attention_quality(attn_weights)

            # Abstract attention transformation
            AAT = sigmoid_growth * (1.0 - sliced_anisotropy) + eps 
            # lower AAT means transformer is less reliable because abstraction is underserved/nonoptimal in this env.
            # Higher AAT means transformer is more focused and reliable and is near optimal.

            if mlp_conf > trans_conf:
                final_pred = mlp_pred
                final_conf = mlp_conf * (1.0 - trans_conf) * (1.0 - np.mean(AAT)) + eps
            else:
                final_pred = trans_pred
                final_conf = trans_conf * (1.0 - mlp_conf) * np.mean(AAT) + eps

            print('='*50)
            print('===== ABSTRACTION LAYER ======')
            print('='*50)
            print(f'[= ABSTRACTION =] Consistency of abstraction transformation: {np.std(AAT)}')
            print(f'[= ABSTRACTION =] Attention Quality: {attn_quality}')
            print(f'[= ABSTRACTION =] Sigmoid growth of Attention weight consistency: {np.std(sigmoid_growth)}')
            print('[=] Note: Very little Consistency meaning Transformer attention quality is Healthy and focused')

        if isinstance(final_conf, np.ndarray):
            final_conf = 1.0 / (1.0 + np.std(final_conf))
            # growth deviation of arrayed final confidence helped to distinguish noise from unnecessary distribution, 
            # with real covariance of distribution from the data.

        if np.isnan(final_conf).any() or np.isinf(final_conf).any():
            final_conf = self.pipeline.confidence_threshold

        return final_pred, final_conf
    
    def _get_attention_focus(self, attn_weights, text):
        if attn_weights is None or len(attn_weights) == 0:
            return text.split()[:3]
        
        words = text.lower().split()
        attn = attn_weights[0].mean(axis=0) if len(attn_weights[0].shape) > 1 else attn_weights[0]
        top_indices = np.argsort(attn)[-3:][::-1]
        if attn.ndim > 1:
            attn = attn.flatten()

        top_indices = np.argsort(attn)[-3:][::-1]
        
        focus_words = []
        for idx in top_indices:
            if hasattr(idx, 'item'):
                idx = idx.item()
            
            if isinstance(idx, (int, np.integer)) and idx >= 0 and idx < len(words):
                focus_words.append(words[idx])
        
        return focus_words if focus_words else words[:3]
    
    def _get_geometric_features(self, text):
        X_tfidf = self.pipeline.tfidf.transform([text]).toarray()
        
        if hasattr(self.pipeline, 'geometric_shaping'):
            anisotropy = self.pipeline.model2.geometric_shaping.anisotropy_measurement(X_tfidf)
            ame = self.pipeline.model2.geometric_shaping.AME_Encoder(X_tfidf)
        else:
            # Fallback: compute simple statistics
            anisotropy = np.std(X_tfidf) / (np.mean(X_tfidf) + 1e-8)
            ame = self.AME_Encoder(X_tfidf)
     
        # Extract dominant features
        feature_names = self.pipeline.tfidf.get_feature_names_out()
        non_zero = X_tfidf[0] > 0
        dominant_features = [feature_names[i] for i in np.where(non_zero)[0][:3]]
        
        return {
            'anisotropy': float(anisotropy),
            'AME': float(ame),
            'dominant_features': dominant_features,
            'feature_energy': float(np.sum(X_tfidf ** 2))
        }
    
    def AME_Encoder(self, x):
        X = np.asarray(x)
        if _OPT_AVAILABLE and np.asarray(X).ndim == 2:
            return optimized_ame_encoder(np.asarray(X, dtype=np.float64))     


        try:
            gradient = np.gradient(x, axis=-1)
        except:
            subset = x[:]
            gradient = np.gradient(subset)

        grad_energy = np.mean(np.linalg.norm(gradient, axis=-1))       
        X_mag = np.mean(np.linalg.norm(X, axis=-1))
        AME = np.log1p(X_mag) * np.log1p(grad_energy) 

        return AME

    def _compute_anisotropy(self, attn_weights):
        if attn_weights is None or len(attn_weights) == 0:
            return 0.5
        
        try:
    
            if hasattr(self.pipeline, 'anisotropy_measurement'):
                return self.pipeline.anisotropy_measurement(attn_weights.flatten())
            
            # Fallback calculation
            attn_flat = attn_weights.flatten()
            gradient = np.gradient(attn_flat)
            val = [np.linalg.norm(v) for v in gradient]
            return np.std(val) / (np.mean(val) + 1e-8)

        except:
            return 0.5
    
    def _compute_attention_quality(self, attn_weights):
        eps = 1e-5
        if attn_weights is None or len(attn_weights) == 0:
            return 0.5
        
        try:

            if hasattr(self.pipeline.model2, 'attention_quality_computing'):
                return self.pipeline.model2.attention_quality_computing(attn_weights)
            
            # Fallback calculation
            eps = 1e-5
            batch, heads, seq_len, _ = attn_weights.shape
            
            entropy = -np.sum(attn_weights * np.log(attn_weights + eps), axis=-1)
            max_entropy = np.log(seq_len)
            norm_entropy = 1.0 - (np.mean(entropy) / max_entropy)
            
            max_attn = np.max(attn_weights, axis=-1)
            avg_max = np.mean(max_attn)
            
            var_attn = np.var(attn_weights)
            norm_var = np.clip(var_attn * seq_len, 0, 1)
            
            AME = self.AME_Encoder(attn_weights)
            AMR = 1.0 / (1.0 + np.exp(-AME) + eps)

            quality = norm_entropy * (1.0 - AMR) + avg_max * AMR + norm_var * AMR
            return np.clip(quality, 0, 1)
        except:
            print("[-] Error occurred while computing attention quality.")
            AMR = 0.1
            if attn_weights is not None:
                print(f"[-] Attention weights shape: {attn_weights.shape}")
                AME = self.AME_Encoder(attn_weights)
                AMR = 1.0 / (1.0 + np.exp(-AME) + eps)            
            return AMR
    
    def _generate_explanation(self, details):
        parts = []
        
        # Final decision
        parts.append(f"📌 Decision: I think my prediction is: **{details['final_label']}**")
        parts.append(f"[=] Confidence Degree: {details['final_confidence']}\n")
        
        # MLP's geometric reasoning
        parts.append("🧠 Geometric MLP Reasoning:")
        parts.append(f"   • Detected Detail: {', '.join(details['geometric_features']['dominant_features'][:3])}")
        parts.append(f"   • Geometric complexity signature: {details['geometric_features']['anisotropy']:.3f}")
        parts.append(f"   • Energy: signature {details['geometric_features']['feature_energy']:.3f}")
        parts.append(f"   • Confidence Focus: {details['mlp']['confidence']:.1%} to → {details['mlp']['label']}")

        if details.get('lstm') is not None:
            lstm = details['lstm']
            parts.append("\n⏳ LSTM Memory Reasoning:")
            parts.append(f"   • MC Dropout Confidence: {lstm['mc_confidence']:.1%} "
                        f"(spread: ±{lstm['mc_std']:.4f})")
            parts.append(f"   • Gate Stability: {lstm['gate_stability']:.1%} "
                        f"({'stable memory' if lstm['gate_stability'] > 0.6 else 'actively rewriting memory — uncertain transition'})")
            parts.append(f"   • Prediction Interval: [{lstm['interval'][0]:.4f}, {lstm['interval'][1]:.4f}]")
            parts.append(f"   • Overall LSTM Confidence: {lstm['overall']:.1%}")
            if lstm['dominant_label']:
                parts.append(f"   • Strongest Sequence Signal: {lstm['dominant_label']} "
                            f"({lstm['dominant_prob']:.1%})") 

        # Transformer's contextual reasoning
        if self.pipeline.use_transformer:
            parts.append("\n🌀 Transformer Reasoning:")
            if details['transformer']['attention_words']:
                parts.append(f"   • Focused on: '{', '.join(details['transformer']['attention_words'])}'")
            parts.append(f"   • Attention quality: {details.get('attention_quality', 0.5)}")
            parts.append(f"   • Attention anisotropy: {details.get('anisotropy', 0.5):.3f}")
            parts.append(f"   • Confidence Focus: {details['transformer']['confidence']:.1%} to → {details['transformer']['label']}")

        # Agreement analysis
        lstm = details.get('lstm')
        if details['agreement']:
            parts.append("\n✅ Models Agreed:")
            parts.append("   Both geometric and contextual analysis point to the same conclusion")
            if lstm and lstm['gate_stability'] > 0.6:
                parts.append("[=+=] LSTM memory is stable — sequence history supports this decision")
            else:
                parts.append("[!] LSTM Uncertain - Sequence history does not supports this decision")
        else:
            if self.pipeline.use_transformer:
                parts.append("\n⚠️ Models Disagreed:")
                parts.append(f"   Geometric MLP Focusing on → {details['mlp']['label']} detail")
                parts.append(f"   Transformer Focusing on → {details['transformer']['label']} detail")
                if lstm:
                    stability_note = "reinforces" if lstm['gate_stability'] > 0.6 else "is uncertain about"
                    parts.append(f"   LSTM {stability_note} the sequence context "
                                f"(gate stability: {lstm['gate_stability']:.1%})")
                parts.append(f"   I weighted them with {details['final_confidence']:.1%} "
                            f"confident in {details['final_label']}")                
            else:
                parts.append("🌀 Supporting Argument From LSTM:")
                if lstm:
                    stability_note = "reinforces" if lstm['gate_stability'] > 0.6 else "is uncertain about"
                    parts.append(f"   LSTM {stability_note} the sequence context "
                                f"(gate stability: {lstm['gate_stability']:.1%})")                
                parts.append(f"   Geometric MLP Focusing on → {details['mlp']['label']} detail")
                parts.append(f"   I weighted them with {details['final_confidence']:.1%} confident in {details['final_label']}")

        # 5. Uncertainty assessment
        if details['final_confidence'] < 0.6:
            parts.append("\n🤔 Uncertainty Note:")
            parts.append(f"   I'm not very confident about this prediction || Confidence: {details['final_confidence']}")
            parts.append("   • This pattern is unusual in my training data")
            parts.append("   • More same examples would help me learn enough pattern")
        
        # 6. Geometric signature 
        parts.append("\n🔬 Geometric Signature:")
        parts.append(f"   • AME Signature: {details['geometric_features']['AME']:.4f}")
        parts.append(f"   • Anisotropy Signature: {details['geometric_features']['anisotropy']:.4f}")

        confidence = self.explain_confidence(details) 
        comparison = self.compare_decisions() 

        return '\n'.join(parts), confidence, comparison
    
    def _display_explanation(self, explanation):
        print("\n" + "="*80)
        print("🤖 AI EXPLANATION")
        print("="*80)
        print(explanation)
        print("="*80)
        pass
    
    def explain_decision(self, idx=-1):
        if abs(idx) <= len(self.decision_history):
            return self.decision_history[idx]['explanation']
        return "Decision not found"
    
    def compare_decisions(self, idx1=-1, idx2=-2):
        if len(self.decision_history) < 2:
            return "Need at least two decisions to compare"
        
        comparison = []
        d1 = self.decision_history[idx1]
        d2 = self.decision_history[idx2]
        
        comparison.append(f"🔄 Decision Comparison")
        comparison.append("====================================")
        
        comparison.append("[<] Earlier Decision:")
        comparison.append(f"[+] Input: {d1['input']}")
        comparison.append(f"[+] Detail Focus: {d1['prediction']} ({d1['confidence']:.1%})")
        
        comparison.append("🧠 Later Decision:")
        comparison.append(f"[=] Input: {d2['input']}")
        comparison.append(f"[=] Detail Focus: {d2['prediction']} ({d2['confidence']:.1%})")
        
        comparison.append("🔬 Learning Progress: ")
        comparison.append(f"• Confidence {'increased' if d2['confidence'] > d1['confidence'] else 'decreased'} from {d1['confidence']} to {d2['confidence']}")
        comparison.append(f"• The model is becoming {'more' if d2['confidence'] > d1['confidence'] else 'less'} certain")
        
        return '\n'.join(comparison)
    
    def explain_confidence(self, details):

        factors = []
        
        # Check MLP confidence
        if details['mlp']['confidence'] > 0.8:
            factors.append(f"✅ MLP is very confident ({details['mlp']['confidence']:.1%}) due to strong geometric patterns")
        elif details['mlp']['confidence'] < 0.5:
            factors.append(f"🤔 MLP is uncertain ({details['mlp']['confidence']:.1%}) due to ambiguous geometric patterns")
        
        # Check transformer confidence
        if details['transformer']['confidence'] > 0.8:
            factors.append(f"✅ Transformer is confident ({details['transformer']['confidence']}) with focused attention")
        elif details['transformer']['confidence'] < 0.5:
            factors.append(f"🤔 Transformer is uncertain ({details['transformer']['confidence']}) due to scattered attention")
        
        # Check agreement
        if details['agreement']:
            factors.append("[✅] Both Models agree, reinforcing confidence")
        else:
            factors.append("[⚠️] Both Models disagree, reducing overall confidence")
        
        # Attention quality
        if details.get('attention_quality', 0) > 0.7:
            factors.append(f"[✅] High attention quality ({details['attention_quality']}) indicates clear consistent patterns!")
        elif details.get('attention_quality', 0) < 0.3:
            factors.append(f"[-] Low Attention Quality! : ({details['attention_quality']}) Indicates noisy unnecessary patterns on seen data!")
        
        return '\n'.join(factors)
    
    def get_decision_history(self, limit=10):
        history = []
        for i, dec in enumerate(self.decision_history[-limit:]):
            history.append({
                'id': i,
                'timestamp': dec['timestamp'],
                'input': dec['input'],
                'prediction': dec['prediction'],
                'confidence': dec['confidence']
            })

            print('=== DECISION HISTORY REPORT ===')
            print(f'[=] ID: {history['id']}')
            print(f'[=] Timestamp: {history['timestamp']}')
            print(f'[=] Processed Input: {history['input']}')
            print(f'[=] Prediction: {history['prediction']}')
            print(f'[=] Confidence: {history['confidence']}')

        return history

# Model storage module that handles saving and loading of trained models, their versions, and associated metadata to a database for persistence and future retrieval.
class ModelStorage:
    def __init__(self, pipeline, memory_name, db_path='activity_log.db'):
        self.pipeline = pipeline
        self.db_path = db_path

        self.setup_storage_table()
        self.setup_explainable_table()
        self.setup_agent_table()
        self.setup_node_table()
        self.setup_weight_table()
        self.setup_accurate_cache_table()

        self.memory_name = memory_name

        if not self.memory_exists(self.memory_name, type='Peer'):
            self.id_history = []
        else:
            print(f'|| Found Matched ID from memory: {self.memory_name}!')
            self.id_history = self.load_agent_id(self.memory_name)

    def get_database_path(self):
        db_filename= self.db_path
        if getattr(sys, 'frozen', False):
            application_path = sys._MEIPASS
            print(f"[🔄] Running as EXE, temp path: {application_path}")
        else:
            application_path = os.path.dirname(os.path.abspath(__file__))
            print(f"[🔄] Running as script, path: {application_path}")
    
        db_path = os.path.join(application_path, db_filename)
        print(f"[🔄] Looking for database at: {db_path}")
        print(f"[✅] Database exists: {os.path.exists(db_path)}")
    
        return db_path

    def setup_explainable_table(self):
        try:
            try:
                db_path = self.get_database_path()            
                conn = sqlite3.connect(db_path)
            except:
                conn = sqlite3.connect(self.db_path)

            c = conn.cursor()
    
            c.execute('''CREATE TABLE IF NOT EXISTS model_attn_storage
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      memory_name TEXT,
                      model_type TEXT,
                      model_data TEXT,
                      is_active INTEGER DEFAULT 0,                      
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        
            conn.commit()
            conn.close()
            print('|| Update Attention Saved to database! ')

        except Exception as e:
            print(f'|| Cant Update Database: {e}')
            filepath = input('|| Insert Database filepath: ')
            if filepath:
                conn = sqlite3.connect(filepath)
            else:
                print('|| Skipping Database Modification...')
                pass
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS model_attn_storage
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      memory_name TEXT,
                      model_type TEXT,
                      model_data TEXT,
                      is_active INTEGER DEFAULT 0,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        
            conn.commit()
            conn.close()


    def setup_storage_table(self):
        try:
            try:
                db_path = self.get_database_path()            
                conn = sqlite3.connect(db_path)
            except:
                conn = sqlite3.connect(self.db_path)

            c = conn.cursor()
    
            c.execute('''CREATE TABLE IF NOT EXISTS model_storage
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      memory_name TEXT,
                      model_version TEXT,
                      model_type TEXT,
                      model_data TEXT,  -- JSON string for dict
                      model_binary BLOB,  -- For pickle files
                      trained_on TEXT,
                      metadata TEXT,  -- JSON for extra info
                      is_active INTEGER DEFAULT 0,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        
            conn.commit()
            conn.close()
            print('|| Update Saved to database! ')

        except Exception as e:
            print(f'|| Cant Update Database: {e}')
            filepath = input('|| Insert Database filepath: ')
            if filepath:
                conn = sqlite3.connect(filepath)
            else:
                print('|| Skipping Database Modification...')
                pass
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS model_storage
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      memory_name TEXT,
                      model_version TEXT,
                      model_type TEXT,
                      model_data TEXT,  -- JSON string for dict
                      model_binary BLOB,  -- For pickle files
                      trained_on TEXT,
                      metadata TEXT,  -- JSON for extra info
                      is_active INTEGER DEFAULT 0,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        
            conn.commit()
            conn.close()


    def get_database_path(self):
        db_filename= self.db_path
        if getattr(sys, 'frozen', False):
            application_path = sys._MEIPASS
            print(f"Running as EXE, temp path: {application_path}")
        else:
            application_path = os.path.dirname(os.path.abspath(__file__))
            print(f"Running as script, path: {application_path}")
    
        db_path = os.path.join(application_path, db_filename)
        print(f"Looking for database at: {db_path}")
        print(f"Database exists: {os.path.exists(db_path)}")
    
        return db_path


    def setup_explainable_table(self):
        try:
            try:
                db_path = self.get_database_path()            
                conn = sqlite3.connect(db_path)
            except:
                conn = sqlite3.connect(self.db_path)

            c = conn.cursor()
    
            c.execute('''CREATE TABLE IF NOT EXISTS model_attn_storage
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      memory_name TEXT,
                      model_type TEXT,
                      model_data TEXT,
                      is_active INTEGER DEFAULT 0,                      
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        
            conn.commit()
            conn.close()
            print('|| Update Attention Saved to database! ')

        except Exception as e:
            print(f'|| Cant Update Database: {e}')
            filepath = input('|| Insert Database filepath: ')
            if filepath:
                conn = sqlite3.connect(filepath)
            else:
                print('|| Skipping Database Modification...')
                pass
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS model_attn_storage
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      memory_name TEXT,
                      model_type TEXT,
                      model_data TEXT,
                      is_active INTEGER DEFAULT 0,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        
            conn.commit()
            conn.close()

    def setup_node_table(self):
        try:
            try:
                db_path = self.get_database_path()            
                conn = sqlite3.connect(db_path)
            except:
                conn = sqlite3.connect(self.db_path)

            c = conn.cursor()
    
            c.execute('''CREATE TABLE IF NOT EXISTS node_storage
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      memory_name TEXT,
                      model_type TEXT,
                      node_data TEXT,
                      node_id TEXT,
                      is_active INTEGER DEFAULT 0,                      
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        
            conn.commit()
            conn.close()
            print('|| Update Node Saved to database! ')

        except Exception as e:
            print(f'|| Cant Update Database: {e}')
            filepath = input('|| Insert Database filepath: ')
            if filepath:
                conn = sqlite3.connect(filepath)
            else:
                print('|| Skipping Database Modification...')
                pass
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS node_storage
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      memory_name TEXT,
                      model_type TEXT,
                      node_data TEXT,
                      node_id TEXT,
                      is_active INTEGER DEFAULT 0,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        
            conn.commit()
            conn.close()


    def setup_agent_table(self):
        try:
            try:
                db_path = self.get_database_path()            
                conn = sqlite3.connect(db_path)
            except:
                conn = sqlite3.connect(self.db_path)

            c = conn.cursor()
    
            c.execute('''CREATE TABLE IF NOT EXISTS agent_attn_storage
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      memory_name TEXT,
                      model_type TEXT,
                      model_attn_data TEXT,
                      model_target_pred TEXT,
                      agent_id TEXT,
                      is_active INTEGER DEFAULT 0,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''') 
        
            conn.commit()
            conn.close()
            print('|| Update Agent Saved to database! ')

        except Exception as e:
            print(f'|| Cant Update Database: {e}')
            filepath = input('|| Insert Database filepath: ')
            if filepath:
                conn = sqlite3.connect(filepath)
            else:
                print('|| Skipping Database Modification...')
                pass
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS agent_attn_storage
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      memory_name TEXT,
                      model_type TEXT,
                      model_attn_data TEXT,
                      model_target_pred TEXT,
                      agent_id TEXT,
                      is_active INTEGER DEFAULT 0,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        
        
            conn.commit()
            conn.close()


    def setup_weight_table(self):
        try:
            try:
                db_path = self.get_database_path()            
                conn = sqlite3.connect(db_path)
            except:
                conn = sqlite3.connect(self.db_path)

            c = conn.cursor()
    
            c.execute('''CREATE TABLE IF NOT EXISTS weight_storage
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      memory_name TEXT,
                      model_type TEXT,
                      weights TEXT,
                      is_active INTEGER DEFAULT 0,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''') 
        
            conn.commit()
            conn.close()
            print('|| Update Agent Saved to database! ')

        except Exception as e:
            print(f'|| Cant Update Database: {e}')
            filepath = input('|| Insert Database filepath: ')
            if filepath:
                conn = sqlite3.connect(filepath)
            else:
                print('|| Skipping Database Modification...')
                pass
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS weight_storage
                      (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      memory_name TEXT,
                      model_type TEXT,
                      weights TEXT,
                      is_active INTEGER DEFAULT 0,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''') 
        
        
            conn.commit()
            conn.close()


    def setup_accurate_cache_table(self):
        try:
            try:
                db_path = self.get_database_path()            
                conn = sqlite3.connect(db_path)
            except:
                conn = sqlite3.connect(self.db_path)

            c = conn.cursor()
    
            c.execute('''CREATE TABLE IF NOT EXISTS accurate_cache_storage
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      memory_name TEXT,
                      cache TEXT,
                      is_active INTEGER DEFAULT 0,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''') 
        
            conn.commit()
            conn.close()
            print('|| Update cached to database! ')

        except Exception as e:
            print(f'|| Cant Update Database: {e}')
            filepath = input('|| Insert Database filepath: ')
            if filepath:
                conn = sqlite3.connect(filepath)
            else:
                print('|| Skipping Database Modification...')
                pass
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS accurate_cache_storage
                      (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      memory_name TEXT,
                      cache TEXT,
                      is_active INTEGER DEFAULT 0,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''') 
        
        
            conn.commit()
            conn.close() 

    def save_accurate_cache_dict(self, memory_name, payload, model_type='Pipeline'):
        try:
            db_path = self.get_database_path()
            conn = sqlite3.connect(db_path)
        except:
            conn = sqlite3.connect(self.db_path)

        c = conn.cursor()

        cache = json.dumps(payload, default=str)

        try:
            c.execute("""
                INSERT INTO accurate_cache_storage
                (memory_name, cache, is_active)
                VALUES (?, ?, ?)
            """, (memory_name, cache, 1))
        
            c.execute("""
                UPDATE accurate_cache_storage
                SET is_active = 0 
                WHERE memory_name = ? AND id != last_insert_rowid()
            """, (memory_name,)) 

            conn.commit()
            conn.close()

            print('|| Accurate cache saved!')

        except Exception as e:
            print(f'[-] Cant save accurate cache memory due to: {e}') 
            pass      

   

    def save_model_dict(self, memory_name, model_dict, type=None, model_type='mlp'):
        try:
            db_path = self.get_database_path()            
            conn = sqlite3.connect(db_path)
        except:
            conn = sqlite3.connect(self.db_path)

        c = conn.cursor()

        model_dict = self.pipeline._sanitize_for_storage(model_dict)  
        serializable_dict = self._prepare_for_serialization(model_dict)
        model_json = json.dumps(serializable_dict, default=str)

        if _RUST_MODULE_AVAILABLE:
            try:
                wc.save_pipelines_dict(self.db_path, memory_name, 
                                    model_type, model_json)
                print('[=] Pipelines dictionary saved using Rust module!')
            except Exception as e:
                print(f'[!] Cant save Pipelines dictionary due to: {e}')
                
        else:
            if type == 'Transformer':
                try:
                    c.execute("""
                        INSERT INTO model_attn_storage 
                        (memory_name, model_type, model_data, is_active)
                        VALUES (?, ?, ?, ?)
                    """, (memory_name, model_type, model_json, 1))
            
                    c.execute("""
                        UPDATE model_attn_storage 
                        SET is_active = 0 
                        WHERE memory_name = ? AND id != last_insert_rowid()
                    """, (memory_name,))

                except Exception as e:
                    print(f'[-] Cant save model memory due to: {e}') 
                    pass             
            else:
                try:
                    c.execute("""
                        INSERT INTO model_storage
                        (memory_name, model_type, model_data, is_active)
                        VALUES (?, ?, ?, ?)
                    """, (memory_name, model_type, model_json, 1))
            
                    c.execute("""
                        UPDATE model_storage 
                        SET is_active = 0 
                        WHERE memory_name = ? AND id != last_insert_rowid()
                    """, (memory_name,)) 

                except Exception as e:
                    print(f'[-] Cant save model memory due to: {e}') 
                    pass          
            
        conn.commit()
        model_id = c.lastrowid        
        conn.close()
        
        print(f"✅ Memory '{memory_name}' saved as dict (ID: {model_id})")
        return model_id

    def _prepare_for_serialization(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: self._prepare_for_serialization(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._prepare_for_serialization(item) for item in obj]
        return obj
        


    def load_model_dict(self, memory_name):
        conn = None
        if _RUST_MODULE_AVAILABLE:
            try:
                num_classes = self.pipeline._get_num_classes() or 0
                cleaned_json = wc.load_and_validate_model_dict(
                    self.db_path, memory_name, num_classes
                )
                if cleaned_json is None:
                    return None
                data = json.loads(cleaned_json)
                # the heavy JSON parse + basic schema check already done in Rust
                validation = self._validate_and_repair(data, memory_name)
                return validation
            except Exception as e:
                print(f'[!] Rust load_model_dict failed: {e}')
        else:
            try:
                try:
                    conn = sqlite3.connect(self.db_path)
                except:
                    conn = sqlite3.connect(self.get_database_path())

                c = conn.cursor()
                c.execute("""
                    SELECT model_data FROM model_storage
                    WHERE memory_name = ? AND is_active = 1
                    ORDER BY id DESC LIMIT 1
                """, (memory_name,))

                result = c.fetchone()
                if not result:
                    return None

                data = json.loads(result[0])
                data = self._validate_and_repair(data, memory_name)
                return data   # actually return data

            except Exception as e:
                print(f'[!] Error loading model dict: {e}')
                return None
            finally:
                if conn:
                    conn.close()


    def _validate_and_repair(self, data, memory_name=None):
        """Validate loaded data and repair if corrupted."""

        if data is None:
            return {}

        # get num_classes safely without touching pipeline
        num_classes = 0
        try:
            # try pipeline first
            if hasattr(self, 'pipeline') and self.pipeline is not None:
                num_classes = self.pipeline._get_num_classes()
            # fallback — infer from data itself
            elif isinstance(data, dict):
                for v in data.values():
                    if isinstance(v, (list, np.ndarray)):
                        arr = np.asarray(v)
                        if arr.ndim == 1:
                            num_classes = len(arr)
                            break
        except Exception as e:
            print(f'[!] Could not determine num_classes: {e}')
            num_classes = 0

        # handle list data
        if isinstance(data, list) and len(data) > 0:
            if num_classes > 0 and len(data) != num_classes:
                print(f'[!] Shape mismatch: got {len(data)}, expected {num_classes} — repairing')
                return {}

            if all(isinstance(x, (int, float)) for x in data[:10]):
                print(f'[!] List appears to be probabilities, wrapping')
                return {'_cached_probs': np.array(data, dtype=np.float64)}

        # handle dict data
        if isinstance(data, dict):
            repaired = {}
            for key, value in data.items():

                # dynamic corruption check instead of hardcoded 133
                if isinstance(value, list):
                    arr = np.asarray(value)
                    if arr.ndim > 2:
                        print(f'[!] Corrupted value for key {key} '
                            f'(ndim={arr.ndim}), removing')
                        continue

                    if num_classes > 0 and arr.ndim == 1 and \
                    len(arr) not in (num_classes, num_classes * 2):
                        print(f'[!] Suspicious shape for key {key}: '
                            f'{arr.shape}, expected {num_classes} — removing')
                        continue

                # None values — skip silently
                if value is None:
                    continue

                repaired[key] = value

            return repaired

        return data


    def _convert_to_arrays(self, data):
        """
        Recursively convert data to numpy arrays where possible.
        Safe for ARM64 and handles all data types.
        """
        if data is None:
            return None
        
        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                converted = self._convert_value(value)
                if converted is not None:
                    result[key] = converted
            return result
        
        elif isinstance(data, (list, tuple)):
            return [self._convert_value(item) for item in data]
        
        else:
            return self._convert_value(data)



    def _convert_value(self, value):
        """
        Convert a single value to appropriate type.
        Returns original value if conversion fails.
        """
        if value is None:
            return None
        
        # Already numpy array - keep as is
        if isinstance(value, np.ndarray):
            return value
        
        # Handle lists recursively
        if isinstance(value, (list, tuple)):
            return [self._convert_value(item) for item in value]
        
        # Handle dicts recursively
        if isinstance(value, dict):
            return self._convert_to_arrays(value)
        
        # Handle string that might represent an array
        if isinstance(value, str):
            return self._parse_array_string(value)
        
        # Return as-is for other types (int, float, bool, etc.)
        return value


    def _parse_array_string(self, s):
        """
        Parse string representation of array back to numpy array.
        Returns original string if parsing fails.
        """
        if not isinstance(s, str) or not s:
            return s

        if _RUST_MODULE_AVAILABLE:
            try:
                data = wc.parse_array_string(self.db_path, s)
                print('[+] Data successfully parsed!')
                return data
            except Exception as e:
                print(f'[!] Data cant be parsed due to: {e}')
                s = s.replace('\n', '').replace('\r', '').replace('\t', '')
                s = ' '.join(s.split()).strip()
                
                if not s:
                    return s
                
                # parsing as JSON array first
                if s.startswith('[') and s.endswith(']'):
                    try:
                        parsed = json.loads(s)
                        if isinstance(parsed, list):
                            return np.array(parsed, dtype=np.float32)
                    except (json.JSONDecodeError, ValueError):
                        pass
                    
                    # Try parsing with ast.literal_eval
                    try:
                        parsed = ast.literal_eval(s)
                        if isinstance(parsed, (list, tuple)):
                            return np.array(parsed, dtype=np.float32)
                    except (ValueError, SyntaxError, TypeError):
                        pass
                
                # parsing space-separated numbers
                if re.fullmatch(r'[\[\]\s\d\.\,\-\+E]+', s):        
                    parts = s.replace('[', ' ').replace(']', ' ').split()
                    if parts:
                        try:
                            float_values = [float(x) for x in parts]
                            return np.array(float_values, dtype=np.float32)
                        except ValueError:
                            pass
                
                # Handle comma-separated values
                if ',' in s:
                    cleaned = s.replace('[', '').replace(']', '').strip()
                    parts = [p.strip() for p in cleaned.split(',') if p.strip()]
                    try:
                        float_values = [float(x) for x in parts]
                        return np.array(float_values, dtype=np.float32)
                    except ValueError:
                        pass  

                return s         
        else:
            # Clean the string
            s = s.replace('\n', '').replace('\r', '').replace('\t', '')
            s = ' '.join(s.split()).strip()
            
            if not s:
                return s
            
            # parsing as JSON array first
            if s.startswith('[') and s.endswith(']'):
                try:
                    parsed = json.loads(s)
                    if isinstance(parsed, list):
                        return np.array(parsed, dtype=np.float32)
                except (json.JSONDecodeError, ValueError):
                    pass
                
                # Try parsing with ast.literal_eval
                try:
                    parsed = ast.literal_eval(s)
                    if isinstance(parsed, (list, tuple)):
                        return np.array(parsed, dtype=np.float32)
                except (ValueError, SyntaxError, TypeError):
                    pass
            
            # parsing space-separated numbers
            if re.fullmatch(r'[\[\]\s\d\.\,\-\+E]+', s):        
                parts = s.replace('[', ' ').replace(']', ' ').split()
                if parts:
                    try:
                        float_values = [float(x) for x in parts]
                        return np.array(float_values, dtype=np.float32)
                    except ValueError:
                        pass
            
            # Handle comma-separated values
            if ',' in s:
                cleaned = s.replace('[', '').replace(']', '').strip()
                parts = [p.strip() for p in cleaned.split(',') if p.strip()]
                try:
                    float_values = [float(x) for x in parts]
                    return np.array(float_values, dtype=np.float32)
                except ValueError:
                    pass
        
        # Return original string if nothing worked
        return s


    def _convertables_utility(self, memory_name, data, data2, type_func=None, verbose=False):
        """
        Convert and display memory data safely.
        Returns tuple (result, result2) always for consistent return type.
        """
        name = memory_name
        
        # Initialize results
        result = None
        result2 = None
        
        # Convert data based on type_func
        if type_func == "TwoPass" and data2 is not None:
            print('|| Two pass utility converting.')
            result = self._convert_to_arrays(data)
            result2 = self._convert_to_arrays(data2)
        else:
            result = self._convert_to_arrays(data)
        
        # Verify result is a dictionary before calling .items()
        if verbose and result is not None:
            print(f"[=] Retrieved memory: {name}")
            
            # ✅ SAFE: Check if result is a dict before iterating
            if isinstance(result, dict):
                for key, value in result.items():
                    self._print_memory_value(key, value)
            else:
                print(f"[!] Result is not a dict: {type(result)}")
                print(f"[!] Result length: {len(result) if hasattr(result, '__len__') else 'N/A'}")
        
        # Handle TwoPass verbose output
        if verbose and data2 is not None and result2 is not None:
            print(f"[=] Retrieved secondary memory: {name}_secondary")
            if isinstance(result2, dict):
                for key, value in result2.items():
                    self._print_memory_value(key, value)
            else:
                print(f"[!] Secondary result is not a dict: {type(result2)}")
        
        # ✅ ALWAYS return consistent types
        if data2 is not None:
            return result, result2
        else:
            return result


    def _print_memory_value(self, key, value):
        # Helper method to print memory values safely
        if isinstance(value, list):
            print(f"  {key}: list of {len(value)} items")
            for i, v in enumerate(value[:5]):  # Limit to first 5 items
                if isinstance(v, np.ndarray):
                    print(f"    [{i}]: array shape {v.shape}")
                else:
                    print(f"    [{i}]: {type(v)}")
            if len(value) > 5:
                print(f"    ... and {len(value) - 5} more items")
        
        elif isinstance(value, np.ndarray):
            print(f"  {key}: array shape {value.shape}, dtype={value.dtype}")
        
        elif isinstance(value, dict):
            print(f"  {key}: dict with {len(value)} keys")
        
        else:
            print(f"  {key}: {type(value)}")


    def memory_retrieval(self, memory_name=None, type_func=None, verbose=False):  
        name = memory_name

        if type_func == 'Transformer':
            data = self.load_transformer_dict(name)
        elif type_func == 'Peer':
            id_history = self.id_history
          
            first_data, second_data = self.load_peer_request_dict(name, id_history) 
            result, result2 = self._convertables_utility(name, first_data, second_data, type_func='TwoPass', verbose=verbose)
            return result, result2
        elif type_func == 'Node':
            data = self._load_node_dict(name)            
        else:
           data = self.load_model_dict(memory_name)

        if data is None:
            print(f"[-] No memory found: {name}")
            return {}

        result = self._convertables_utility(name, data, None, type_func=type_func, verbose=verbose)

        return result


    def load_accurate_cache(self, memory_name):
        try:
            try:
                db_path = self.get_database_path()            
                conn = sqlite3.connect(db_path)
            except:
                conn = sqlite3.connect(self.db_path)

            c = conn.cursor()      
            
            c.execute("""
            SELECT weights FROM accurate_cache_storage 
            WHERE memory_name = ? AND is_active = 1
            """, (memory_name,))               
        
            result = c.fetchone()
            conn.close()
        
            if result:
                return json.loads(result[0])
        except Exception as e:
            print(f'[!] Error handling cache dict: {e}')
        return None


    def _load_weights(self, memory_name, type=None):
        try:
            try:
                db_path = self.get_database_path()            
                conn = sqlite3.connect(db_path)
            except:
                conn = sqlite3.connect(self.db_path)

            c = conn.cursor()
 
            c.execute("""
            SELECT weights FROM weight_storage 
            WHERE memory_name = ? AND is_active = 1
            """, (memory_name,))               
        
            result = c.fetchone()
            conn.close()
        
            if result:
                return json.loads(result[0])
        except Exception as e:
            print(f'[!] Error handling Weight dict: {e}')
        return None


    def weight_retrieval(self, memory_name=None, type=None, verbose=False):  
        name = memory_name

        data = self._load_weights(memory_name, type=type)
        if data is None:
            print(f"[-] No Saved Weight found: {name}")
            return None

        result = self._convertables_utility(name, data, None, type_func='firstpass', verbose=verbose)

        return result       

    def _load_node_dict(self, memory_name):
        try:
            try:
                db_path = self.get_database_path()            
                conn = sqlite3.connect(db_path)
            except:
                conn = sqlite3.connect(self.db_path)

            c = conn.cursor()
        
            c.execute("""
            SELECT node_data FROM node_storage 
            WHERE memory_name = ? AND is_active = 1
            """, (memory_name,))
        
            result = c.fetchone()
            conn.close()
        
            if result:
                return json.loads(result[0])
        except Exception as e:
            print(f'[!] Error handling node dict: {e}')
        return None

    def save_nodes_dict(self, memory_name, node_memory, node_id, model_type='Node'):
        try:
            db_path = self.get_database_path()
            conn = sqlite3.connect(db_path)
        except:
            conn = sqlite3.connect(self.db_path)

        c = conn.cursor()

        node_json = json.dumps(node_memory, default=str)

        try:
            c.execute("""
                INSERT INTO node_storage 
                (memory_name, model_type, node_data, node_id, is_active)
                VALUES (?, ?, ?, ?, ?)
            """, (memory_name, model_type, node_json, node_id, 1))
        
            c.execute("""
                UPDATE node_storage 
                SET is_active = 0 
                WHERE memory_name = ? AND id != last_insert_rowid()
            """, (memory_name,)) 

            conn.commit()
            conn.close()

            print('[||] Node data dictionary saved!')

        except Exception as e:
            print(f'[-] Cant save Node memory due to: {e}')
            pass  


    def save_weights(self, memory_name, model_type=None):
        """Save weights to database."""
        try:
            db_path = self.get_database_path()
            conn = sqlite3.connect(db_path)
        except:
            conn = sqlite3.connect(self.db_path)

        c = conn.cursor()

        weights = {
            'lstm_W'  : self.pipeline.network_model.cell.W.tolist(),
            'lstm_b'  : self.pipeline.network_model.cell.b.tolist(),
            'Wy' : self.pipeline.network_model.Wy.tolist() if self.pipeline.network_model.Wy is not None else None,
            'by' : self.pipeline.network_model.by.tolist(),
            'residual_mean': self.pipeline.lstm_engine.residual_mean,
            'residual_std' : self.pipeline.lstm_engine.residual_std,
            'quantiles'    : {str(k): list(v) 
                            for k, v in self.pipeline.lstm_engine.quantiles.items()} 
                            if self.pipeline.lstm_engine.quantiles else {},
            'n_samples'    : self.pipeline.lstm_engine.n_samples,
            'saved_at'     : datetime.now().isoformat(),
        }
        
        weight_json = json.dumps(weights, default=str)
        if _RUST_MODULE_AVAILABLE:
            try:
                wc.save_lstm_weights(self.db_path, memory_name, weight_json)
                print('[||] LSTM weights saved using Rust module !')
                return
            except Exception as e:
                print(f'[!] Rust save failed, falling back to Python: {e}')
        else:
            print('[=] Rust module unavailable, using python sqlite3.')

        try:
            c.execute("""
                INSERT INTO weight_storage 
                (memory_name, model_type, weights, is_active)
                VALUES (?, ?, ?, ?)
            """, (memory_name, model_type, weight_json, 1))
        
            c.execute("""
                UPDATE weight_storage 
                SET is_active = 0 
                WHERE memory_name = ? AND id != last_insert_rowid()
            """, (memory_name,)) 
            
            c.execute("""
                DELETE FROM weight_storage
                WHERE memory_name = ?
                AND model_type = 'Pipeline'
                AND is_active = 0
            """, (memory_name,))            

            conn.commit()
            conn.close()

            self.save_transformer_weights(memory_name)
            print('[||] All Weights dictionary saved!')

        except Exception as e:
            print(f'[-] Cant save Weights due to: {e}')
            pass          


    def save_transformer_weights(self, memory_name: str):
        """Save transformer weights as compressed binary blob."""
        tf = self.pipeline.model2

        try:
            db_path = self.get_database_path()
            conn = sqlite3.connect(db_path)
        except:
            conn = sqlite3.connect(self.db_path)

        c = conn.cursor()

        buf = io.BytesIO()
        np.savez_compressed(buf,
            token_embedding = tf.token_embedding,
            pos_embedding   = tf.pos_embedding,
            W_q             = tf.W_q,
            W_k             = tf.W_k,
            W_v             = tf.W_v,
            W_q_fixed       = tf.W_q_fixed,
            W_k_fixed       = tf.W_k_fixed,
            W_v_fixed       = tf.W_v_fixed,
            W_o             = tf.W_o,
            ffn1            = tf.ffn1,
            ffn2            = tf.ffn2,
            ln1_scale       = tf.ln1_scale,
            ln1_shift       = tf.ln1_shift,
            ln2_scale       = tf.ln2_scale,
            ln2_shift       = tf.ln2_shift,
            output          = tf.output,
            output_bias     = tf.output_bias
        )
        binary_data = buf.getvalue()
        if _RUST_MODULE_AVAILABLE:
            try:
                wc.save_transformer_weights(self.db_path, memory_name, binary_data)
                print('[||] LSTM weights saved using Rust module for flexibility!')
                return
            except Exception as e:
                print(f'[!] Rust save failed, falling back to Python: {e}')
        else:
            print('[=] Rust module unavailable, using python sqlite3.')

        try:
            c.execute("""
                INSERT INTO weight_storage
                (memory_name, model_type, weights, is_active)
                VALUES (?, ?, ?, ?)
            """, (memory_name, 'transformer', 
                sqlite3.Binary(binary_data), 1))

            c.execute("""
                UPDATE weight_storage SET is_active = 0
                WHERE memory_name = ?
                AND model_type = 'transformer'
                AND id != last_insert_rowid()
            """, (memory_name,))

            c.execute("""
                DELETE FROM weight_storage
                WHERE memory_name = ?
                AND model_type = 'transformer'
                AND is_active = 0
            """, (memory_name,))

            conn.commit()
            print('[||] Transformer weights saved!')

        except Exception as e:
            print(f'[!] Transformer weight save failed: {e}')
            conn.rollback()
        finally:
            conn.close()

    def load_transformer_weights(self, memory_name: str) -> bool:
        try:
            db_path = self.get_database_path()
            conn = sqlite3.connect(db_path)
        except:
            conn = sqlite3.connect(self.db_path)

        c = conn.cursor() 


        try:
            if _RUST_MODULE_AVAILABLE:
                try:
                    binary = wc.load_transformer_weights(self.db_path, memory_name) 

                    buf = io.BytesIO(bytes(binary))
                    data = np.load(buf, allow_pickle=False)

                    print('[+] Transformer weights data loaded using Rust module!')

                except Exception as e:
                    print(f'[=] Cant load Transformer weights: {e}, using python sqlite3 to handle weights.')
                    time.sleep(3)
                    c.execute("""
                        SELECT weights FROM weight_storage
                        WHERE memory_name = ? AND model_type = 'transformer' AND is_active = 1
                        ORDER BY id DESC LIMIT 1
                    """, (memory_name,))
                    row = c.fetchone()
                    if not row:
                        print(f'[=] No saved transformer weights for {memory_name}')
                        return False

                    buf = io.BytesIO(bytes(row[0]))
                    data = np.load(buf, allow_pickle=False)

            else:
                c.execute("""
                    SELECT weights FROM weight_storage
                    WHERE memory_name = ? AND model_type = 'transformer' AND is_active = 1
                    ORDER BY id DESC LIMIT 1
                """, (memory_name,))
                row = c.fetchone()
                if not row:
                    print(f'[=] No saved transformer weights for {memory_name}')
                    return False

                buf = io.BytesIO(bytes(row[0]))
                data = np.load(buf, allow_pickle=False)

            t = self.pipeline.model2
            t.token_embedding = data['token_embedding']
            t.pos_embedding   = data['pos_embedding']
            t.W_q             = data['W_q']
            t.W_k             = data['W_k']
            t.W_v             = data['W_v']
            t.W_q_fixed       = data['W_q_fixed']
            t.W_k_fixed       = data['W_k_fixed']
            t.W_v_fixed       = data['W_v_fixed']
            t.W_o             = data['W_o']
            t.ffn1            = data['ffn1']
            t.ffn2            = data['ffn2']
            t.ln1_scale       = data['ln1_scale']
            t.ln1_shift       = data['ln1_shift']
            t.ln2_scale       = data['ln2_scale']
            t.ln2_shift       = data['ln2_shift']
            t.output          = data['output']
            t.output_bias     = data['output_bias']

            print(f'[||] Transformer weights loaded!')
            return True

        except Exception as e:
            print(f'[!] Transformer weight load failed: {e}')
            return False
        finally:
            conn.close()


    def load_weights(self, memory_name):
        """Load weights from database. Returns True if found."""
        if _RUST_MODULE_AVAILABLE:
            try:
                result = wc.load_lstm_weights(self.db_path, memory_name)
                print('[+] LSTM weights loaded via Rust module')
            except Exception as e:
                result = self.weight_retrieval(memory_name)
                print(f'[=] Cant load LSTM Weights due to: {e}, using python sqlite3 as fallback.')
                  
        else:
            result = self.weight_retrieval(memory_name)           
    
        if not result:
            print(f'[=] No saved weights for {memory_name}')
            return False

        try:
            weights = result
            self.pipeline.network_model.cell.W  = np.array(weights['lstm_W'])
            self.pipeline.network_model.cell.b  = np.array(weights['lstm_b'])
            self.pipeline.network_model.Wy      = np.array(weights['Wy']) if weights['Wy'] else None
            self.pipeline.network_model.by      = np.array(weights['by'])
            self.pipeline.lstm_engine.residual_mean = weights.get('residual_mean', 0.0)
            self.pipeline.lstm_engine.residual_std  = weights.get('residual_std',  1.0)
            self.pipeline.lstm_engine.n_samples     = weights.get('n_samples', self.pipeline.lstm_engine.n_samples)
            self.pipeline.lstm_engine.quantiles     = {float(k): tuple(v) 
                                for k, v in weights.get('quantiles', {}).items()}

            tf_loaded = self.load_transformer_weights(memory_name)

            print(f'[=] Transformer weights loaded: {tf_loaded}')
            if tf_loaded:
                print(f'[=] All Weights loaded for {memory_name}  '
                    f'(saved at {weights.get("saved_at", "unknown")})')

        except Exception as e:
            print(f'[!] Cant load any Weights due to: {e}')


    def load_transformer_dict(self, memory_name):
        try:
            try:
                conn = sqlite3.connect(self.db_path)
            except:
                db_path = self.get_database_path()
                conn = sqlite3.connect(db_path)   

            if _RUST_MODULE_AVAILABLE:
                try:
                    data = wc.load_attention_dict(self.db_path, memory_name)
                    print('[+] Transformer attention loaded using Rust module!')
                    return data
                except Exception as e:
                    print(f'[=]: {e}, Loading transformer attention from python sqlite3...')
            else:     
                conn = sqlite3.connect(self.db_path)
                c = conn.cursor()
            
                c.execute("""
                SELECT model_data FROM model_attn_storage 
                WHERE memory_name = ? AND is_active = 1
                """, (memory_name,))
            
                result = c.fetchone()
                conn.close()
            
                if result:
                    return json.loads(result[0])
        except Exception as e:
            print(f'[!] Error handling attention dict: {e}')

        return None   

    def save_peer_needs_dict(self, memory_name, model_dict, target_pred, agent_id, model_type='Pipeline'):
        try:
            db_path = self.get_database_path()
            conn = sqlite3.connect(db_path)
        except:
            conn = sqlite3.connect(self.db_path)

        c = conn.cursor()

        model_json = json.dumps(model_dict, default=str)
        target_json = json.dumps(target_pred, default=str)
        agent_id_converted = json.dumps(agent_id, default=str)

        try:
            c.execute("""
                INSERT INTO agent_attn_storage 
                (memory_name, model_type, model_attn_data, model_target_pred, agent_id, is_active)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (memory_name, model_type, model_json, target_json, agent_id_converted, 1))
        
            c.execute("""
                UPDATE agent_attn_storage 
                SET is_active = 0 
                WHERE memory_name = ? AND id != last_insert_rowid()
            """, (memory_name,)) 

            conn.commit()
            conn.close()

            print('|| Peer data Needs dictionary saved!')

        except Exception as e:
            print(f'[-] Cant save model memory due to: {e}') 
            pass        


    def load_peer_request_dict(self, memory_name, agent_id):
        print(f'|| Peer request with Agent')
        try:
            try:
                db_path = self.get_database_path()
                conn = sqlite3.connect(db_path)   
            except:
                conn = sqlite3.connect(self.db_path)


            c = conn.cursor()
            placeholders = ",".join(["?"] * len(agent_id))

            query = f"""
            SELECT model_attn_data, model_target_pred FROM agent_attn_storage 
            WHERE memory_name = ? AND is_active = 1 AND agent_id NOT IN ({placeholders})
            """
            params = [memory_name] + agent_id
        
            c.execute(query, params)  
        
            result = c.fetchone()
            conn.close()
            print(f"|| Retrieved Peer Request memory: {memory_name} for agent_id: {agent_id}: result: {result}")
 
            if result:
                return json.loads(result[0]), json.loads(result[1])
            return None, None
        except Exception as e:
            print(f'|| Cant load peer request memory due to: {e}') 
            return None, None  

        

    def fix_corrupted_memory(self, memory_name):
        # Clear corrupted memory entries
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            # Deactivate corrupted entries
            c.execute("""
                UPDATE model_storage 
                SET is_active = 0 
                WHERE memory_name = ? AND is_active = 1
            """, (memory_name,))
            
            c.execute("""
                UPDATE model_attn_storage 
                SET is_active = 0 
                WHERE memory_name = ? AND is_active = 1
            """, (memory_name,))
            
            conn.commit()
            conn.close()
            
            print(f'[✅] Cleared corrupted memory for {memory_name}')
            return True
        except Exception as e:
            print(f'[!] Failed to clear memory: {e}')
            return False


    def load_agent_id(self, memory_name):
        try:

            try:
               db_path = self.get_database_path()
               conn = sqlite3.connect(db_path)  
            except:
                conn = sqlite3.connect(self.db_path)

            if _RUST_MODULE_AVAILABLE:
                try:
                    agent_data = wc.load_agent_id(self.db_path, memory_name)
                    print('[+] Got Agent ID Data from Database!')
                except Exception as e:
                    print(f'[!] Cant load agent ID data from DB: {e}')
                    return None    
            else:
                c = conn.cursor()
            
                c.execute("""
                SELECT agent_id FROM agent_attn_storage 
                WHERE memory_name = ? AND is_active = 1
                """, (memory_name,))
            
                result = c.fetchone()
                conn.close()

                print(f'[+] Retrieved Agent ID of {memory_name}: result: {result}')
            
                if result:
                    return json.loads(result[0])
        except Exception as e:
            print(f'[-] Error loading ID from database: {e}')

        return None        


    def memory_exists(self, memory_name, type=None):

        conn = None
        try:
            try:
                db_path = self.get_database_path()
                conn = sqlite3.connect(db_path)               
            except:
                conn = sqlite3.connect(self.db_path)

            if _RUST_MODULE_AVAILABLE:
                try:
                    exists = wc.verify_memory_exist(self.db_path, memory_name, type)
                    print(f'[=] Memory exist: {exists}')
                    return exists
                except:
                    pass
            else:
                if type == 'Transformer':
                    c = conn.cursor()
            
                    c.execute("""
                    SELECT 1 FROM model_attn_storage 
                    WHERE memory_name = ? AND is_active = 1
                    LIMIT 1
                    """, (memory_name,))
            
                    result = c.fetchone()
                    exists = result is not None
                    print(f"|| Retrieved Attention: {memory_name}")

                elif type == 'Peer':
                    c = conn.cursor()
            
                    c.execute("""
                    SELECT 1 FROM agent_attn_storage 
                    WHERE memory_name = ? AND is_active = 1
                    LIMIT 1
                    """, (memory_name,))
            
                    result = c.fetchone()
                    exists = result is not None
                    print(f"|| Retrieved Peer Memory: {memory_name}")

                elif type == 'Accurate-Cache':
                    c.execute("""
                    SELECT 1 FROM accurate_cache_storage
                    WHERE memory_name = ? and is_active = 1
                    LIMIT 1""", (memory_name, ))

                    result = c.fetchone()
                    exists = result is not None
                    print(f"|| Retrieved Accurate Fact Cache Memory for memory: {memory_name}")
                else:
                    c = conn.cursor()

                    c.execute("""
                    SELECT 1 FROM model_storage 
                    WHERE memory_name = ? AND is_active = 1
                    LIMIT 1
                    """, (memory_name,))
            
                    result = c.fetchone()
                    exists = result is not None
                    print(f"|| Retrieved Memory: {memory_name}")

                return exists
        
        except sqlite3.OperationalError as e:
            print(f"[!] Database error: {e}")
            return False
            
        except Exception as e:
            print(f"[!] Unexpected error in handling memory: {e}") 
            return False
        finally:
            if conn:
                conn.close()


    def save_model_binary(self, model_object, memory_name, model_type='mlp'):
        try:
            try:
                conn = sqlite3.connect(self.db_path)
            except:
               db_path = self.get_database_path()
               conn = sqlite3.connect(db_path)          
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
        
            model_binary = joblib.dumps(model_object)
        
            c.execute("""
            INSERT INTO model_storage 
            (memory_name, model_type, model_binary, is_active)
            VALUES (?, ?, ?, ?)
            """, (memory_name, model_type, model_binary, 1))
        
            # Deactivate other versions
            c.execute("""
            UPDATE model_storage 
            SET is_active = 0 
            WHERE memory_name = ? AND id != last_insert_rowid()
            """, (memory_name,))
        
            conn.commit()
            model_id = c.lastrowid
            print(f"✅ Memory '{memory_name}' saved as binary (ID: {model_id})")
        except Exception as e:
            logger.error(f"[-] Error handling: {e}")

        conn.close()

        return model_id
    
    def load_model_binary(self, memory_name):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute("""
            SELECT model_binary FROM model_storage 
            WHERE memory_name = ? AND is_active = 1
        """, (memory_name,))
        
        result = c.fetchone()
        conn.close()
        
        if result:
            return joblib.loads(result[0])
        return None
    
    def save_complete_pipeline(self, pipeline_name, pipeline_dict):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Convert entire pipeline to JSON (for dicts)
        pipeline_json = json.dumps(pipeline_dict, default=str)
        
        c.execute("""
            INSERT INTO model_storage 
            (pipeline_name, model_type, model_data, metadata, is_active)
            VALUES (?, ?, ?, ?, ?)
        """, (pipeline_name, 'pipeline', pipeline_json, 
               json.dumps({'components': list(pipeline_dict.keys())}), 1))
        
        conn.commit()
        model_id = c.lastrowid
        conn.close()
        
        print(f"✅ Integrated pipeline '{pipeline_name}' saved")
        return model_id



class AsyncMessageQueue:
    def __init__(self, max_size=1000, dead_letter_queue_size=100,
                latency_smoothing=0.2):   
        self.queue   = asyncio.PriorityQueue(maxsize=max_size)
        self.pending: Dict[str, asyncio.Future] = {}
        self.results: Dict[str, Any] = {}
        self.handlers: Dict[str, Callable] = {}
        self.dead_letter_queue: deque = deque(maxlen=dead_letter_queue_size)
        self._running = False

        self._worker_task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._start_lock = asyncio.Lock()   

        self._counter = 0
        self.alpha    = latency_smoothing   #EMA weight

        self._stats = {
            'messages_processed': 0,
            'messages_failed'   : 0,
            'messages_retried'  : 0,
            'messages_expired'  : 0,
            'avg_latency'       : 0.0,
            'messages_untrusted': 0
        }

    def register_handler(self, message_type: str, handler: Callable):
        self.handlers[message_type] = handler
        logger.info(f"[=] Registered handler for {message_type}")


    async def _ensure_started(self):
        """Single entry point for starting the worker """
        if self._running:
            return

        async with self._start_lock:
            if self._running:
                return
            self._running     = True
            self._worker_task = asyncio.create_task(self._worker())

            logger.info("[=] Async message queue worker started")
            await asyncio.sleep(0.1)
            if self._worker_task.done():
                exc = self._worker_task.exception()
                if exc:
                    logger.error(f"[=] Worker failed: {exc}")
                    self._running = False   
                    raise exc


    async def publish(self, message: Message) -> Any:
        await self._ensure_started()

        # plain increment
        self._counter += 1
        counter = self._counter
        
        if self._stats['avg_latency'] > 0.25:
            message.trust - 0.1

        if message.is_expired:
            logger.warning(f"[-] Message {message.id} already expired")
            raise TimeoutError(f"[-] Message {message.id} already expired")
            

        if not message.proper_trust:
            raise Warning(f"[!] Message is not properly Trusted!")

        future = asyncio.Future()
        self.pending[message.id] = future
        logger.debug(f"[=] Publishing message {message.id} type={message.type} "
                     f"priority={message.priority.name}")

        # 3-tuple, matches _worker's unpack exactly as is.
        await self.queue.put((message.priority.value, counter, message))

        try:
            result = await asyncio.wait_for(future, timeout=message.timeout)
            return result
        except asyncio.TimeoutError:
            self.pending.pop(message.id, None)
            self._stats['messages_expired'] += 1
            logger.warning(f"[-] Message {message.id} timed out after {message.timeout}s")
            raise
        except Exception as e:
            self.pending.pop(message.id, None)
            logger.error(f"[-] Error processing message {message.id}: {e}")
            raise

    async def publish_async(self, message: Message, callback: Optional[Callable] = None):
        """Fire and forget, uses consistent 3-tuple format."""
        await self._ensure_started()
        message.callback = callback

        self._counter += 1
        counter = self._counter
        await self.queue.put((message.priority.value, counter, message))

    async def _worker(self):
        while self._running:
            try:
                priority, counter, message = await asyncio.wait_for(
                    self.queue.get(), timeout=1.0
                )
                logger.debug(f"[=] Worker picked up {message.id} "
                            f"(counter={counter}, priority={priority})")

                start_time = time.time()

                if message.is_expired:
                    self._stats['messages_expired'] += 1
                    self._handle_orphaned_message(message)
                    continue

                if not message.proper_trust:
                    self._stats['messages_untrusted'] += 1

                    # treat as orphan
                    self._handle_orphaned_message(message)
                    continue                    

                if message.type in self.handlers:
                    try:
                        if asyncio.iscoroutinefunction(self.handlers[message.type]):
                            result = await self.handlers[message.type](message)
                        else:
                            result = self.handlers[message.type](message)

                        latency = time.time() - start_time
                        self._update_stats(latency, success=True)

                        # pop from pending on success, was leaking before
                        if message.id in self.pending:
                            future = self.pending.pop(message.id)
                            if not future.done():
                                future.set_result(result)
                        elif message.callback:
                            message.callback(result)

                    except Exception as e:
                        self._stats['messages_failed'] += 1
                        logger.error(f"[-] Handler failed for {message.type}: {e}\n"
                                    f"{traceback.format_exc()}")

                        if message.retry_count < message.max_retries:
                            message.retry_count += 1
                            self._stats['messages_retried'] += 1
                            # consistent 3-tuple on retry too
                            self._counter += 1
                            retry_counter = self._counter
                            await self.queue.put(
                                (message.priority.value, retry_counter, message)
                            )
                        else:
                            self._dead_letter_message(message, e)
                            # popped here.
                            if message.id in self.pending:
                                future = self.pending.pop(message.id)
                                if not future.done():
                                    future.set_exception(e)
                            elif message.callback:
                                message.callback(e)
                else:
                    logger.warning(f"[-] No handler for message type: {message.type}")
                    self._dead_letter_message(
                        message, Exception(f"[!] No handler for {message.type}")
                    )
                    # pop here too, unhandled message type leaked before
                    if message.id in self.pending:
                        future = self.pending.pop(message.id)
                        if not future.done():
                            future.set_exception(
                                Exception(f"No handler for {message.type}")
                            )

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                logger.info("[=] Worker task cancelled")
                break
            except Exception as e:
                logger.error(f"[-] Worker error: {e}\n{traceback.format_exc()}")
                await asyncio.sleep(0.1)


    def _update_stats(self, latency: float, success: bool):
        self._stats['messages_processed'] += 1
        if not success:
            self._stats['messages_failed'] += 1

        alpha = self.alpha
        self._stats['avg_latency'] = (
            alpha * latency + (1 - alpha) * self._stats['avg_latency']
        )


    def _dead_letter_message(self, message: Message, error: Exception):
        self.dead_letter_queue.append({
            'message'    : message,
            'error'      : str(error),
            'timestamp'  : datetime.now(),
            'retry_count': message.retry_count
        })
        logger.error(f"[=] Message {message.id} sent to DLQ after "
                    f"{message.retry_count} retries")

    def _handle_orphaned_message(self, message: Message):
        logger.warning(f"[=] Orphaned message {message.id} of type {message.type}")
        self.dead_letter_queue.append({
            'message'  : message,
            'error'    : 'Orphaned message - expired before processing',
            'timestamp': datetime.now()
        })
        # orphaned messages with pending futures also leaked before
        if message.id in self.pending:
            future = self.pending.pop(message.id)
            if not future.done():
                future.set_exception(TimeoutError(f"Message {message.id} expired"))

    def get_stats(self) -> Dict:
        return {
            **self._stats,
            'pending_count': len(self.pending),
            'queue_size'   : self.queue.qsize(),
            'dlq_size'     : len(self.dead_letter_queue),
            'is_running'   : self._running
        }

    async def start(self):
        """delegates to _ensure_started, single code path."""
        try:
            await self._ensure_started()
        except Exception as e:
            print(f'[!] Workers failed to start: {e}')

    
    async def stop(self, timeout: float = 5.0):
        logger.info("[=] Stopping message queue...")
        self._running = False
        if self._worker_task:
            try:
                await asyncio.wait_for(self._worker_task, timeout=timeout)
            except asyncio.TimeoutError:
                self._worker_task.cancel()
                logger.warning("[=] Worker task did not stop gracefully")

        # clean up any still-pending futures on shutdown
        for msg_id, future in list(self.pending.items()):
            if not future.done():
                future.set_exception(RuntimeError("Queue stopped"))
        self.pending.clear()

        logger.info("[=] Async message queue stopped")

    def get_dead_letter_queue(self) -> List[Dict]:
        if self.dead_letter_queue is not None:
            return list(self.dead_letter_queue)
        else:
            return []


class ThreadedMessageQueue:
    # Thread-based message queue for synchronous code.
    def __init__(self, max_size=1000, worker_threads=4):
        self.queue          = queue.Queue(maxsize=max_size)
        self.results        = {}
        self.handlers       = {}
        self._running       = False
        self._workers       = []
        self._worker_threads = worker_threads
        self._stats = {
            'messages_processed': 0,
            'messages_failed'   : 0,
            'active_workers'    : 0
        }
        self._lock = threading.Lock()


    def register_handler(self, message_type: str, handler: Callable):
        self.handlers[message_type] = handler
        logger.info(f"[=] Registered handler for {message_type}")


    def publish(self, message: Message, timeout: float = 30.0) -> Any:
        # threading.Event.
        done_event      = threading.Event()
        result_container = {'result': None, 'error': None}

        def callback_wrapper(res):
            # distinguish success from failure
            if isinstance(res, Exception):
                result_container['error'] = res
            else:
                result_container['result'] = res
            done_event.set()

        message.callback = callback_wrapper

        try:
            self.queue.put(message, timeout=timeout)
        except queue.Full:
            raise TimeoutError(f"[!] Queue full, could not enqueue message {message.id}")

        signaled = done_event.wait(timeout=timeout)

        if not signaled:
            with self._lock:
                self._stats['messages_failed'] += 1   # actually tracked now
            raise TimeoutError(f"[!] Message {message.id} timed out")

        if result_container['error'] is not None:
            with self._lock:
                self._stats['messages_failed'] += 1
            raise result_container['error']   # now actually raises

        with self._lock:
            self._stats['messages_processed'] += 1

        return result_container['result']

    def publish_async(self, message: Message, callback: Optional[Callable] = None):
        message.callback = callback
        try:
            self.queue.put(message, block=False)
            return True
        except queue.Full:
            logger.error(f"[=] Queue full, cannot publish message {message.id}")
            return False



    def _worker(self, worker_id: int, stop_event: threading.Event):
        logger.info(f'[=] Worker started: {worker_id}')

        while self._running and not stop_event.is_set():
            try:
                message = self.queue.get(timeout=1)

                if message.type in self.handlers:
                    try:
                        result = self.handlers[message.type](message)
                        if message.callback:
                            message.callback(result)
                    except Exception as e:
                        logger.error(f"[=] Worker {worker_id} handler failed: {e}")
                        with self._lock:
                            self._stats['messages_failed'] += 1
                        if message.callback:
                            # callback_wrapper 
                            message.callback(e)
                else:
                    logger.warning(f"[=] No handler for message type: {message.type}")

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[=] Worker {worker_id} error: {e}")
                # track that this worker degraded, without killing loop entirely
                with self._lock:
                    self._stats.setdefault('worker_errors', {})
                    self._stats['worker_errors'][worker_id] = \
                        self._stats['worker_errors'].get(worker_id, 0) + 1

        logger.info(f'[=] Worker {worker_id} exiting cleanly')
        with self._lock:
            self._stats['active_workers'] = max(0, self._stats['active_workers'] - 1)


    def start(self):
        if self._running:
            logger.warning('[=] ThreadedMessageQueue already running — ignoring duplicate start')
            return

        self._running   = True
        self._stop_event = threading.Event()   # allows prompt wake-up on stop
        self._workers   = []

        for i in range(self._worker_threads):
            thread = threading.Thread(
                target=self._worker, args=(i, self._stop_event), daemon=True
            )
            thread.start()
            self._workers.append(thread)

        with self._lock:
            self._stats['active_workers'] = len(self._workers)

        logger.info(f"[=] Threaded message queue started with {self._worker_threads} workers")

    def stop(self, timeout: float = 5.0):
        """
        completely rewritten. This is a threading-based class.
        """
        if not self._running:
            return

        logger.info("[=] Stopping threaded message queue...")
        self._running = False
        self._stop_event.set()   # signal immediately

        for worker in self._workers:
            worker.join(timeout=timeout)
            if worker.is_alive():
                logger.warning(
                    f'[!] Worker thread {worker.name} did not stop within '
                    f'{timeout}s — it will be abandoned as a daemon thread '
                    f'(Python cannot forcibly kill threads)'
                )

        self._workers.clear()   # clear references regardless,
                                # any stragglers are daemon threads that
                                # die automatically when the process exits

        with self._lock:
            self._stats['active_workers'] = 0

        logger.info("[=] Threaded message queue stopped")


    def get_stats(self) -> Dict:
        with self._lock:
            return {
                **self._stats,
                'queue_size': self.queue.qsize(),
                'workers'   : len(self._workers),
                'is_running': self._running
            }


# Integrated inference module that allows multiple agents to connect and share their predictions, attention maps, and confidence scores for ensemble decision making.
# while also providing security features like authentication, rate limiting, and message validation.
class AgentDistributedInference:
    def __init__(self, pipeline, storage, memory_name, port=5555, 
                 use_async=False, secret_key=None, 
                 ssl_cert_file=None, ssl_key_file=None, 
                 ssl_context=None, client_ssl_context=None,
                 shared_auth_token=None, predict_manager=None,
                 bind_host=None, security_level=None):      

        self.pipeline = pipeline
        self.memory_name = memory_name
        self.port = port
        self.storage = storage

        self.query_node = QueryNode(pipeline, memory_name, self.storage)        
        
        self.agent_comm_log = {}
        self.connections_log = {}
        self.connections = []  # List of connected sockets
        self.remote_agents = {}  # {agent_id: {'sock': sock, 'host': host, 'port': port, 'trust': 1.0}}
        
        self.running = False
        self.socket = None
        self.temporary_message = None
        self.temporary_agent_id = None  
        self._server_started = False  # explicit flag

        self.established_connections = set()  # Track established connections to prevent duplicates      

        self.next_agent_id = 1
        self.connection_timeout = 15

        # for security purposes
        # Security: Authentication token
        self.auth_token = shared_auth_token
        self.secret_key = shared_auth_token 

        # Security: Rate limiting
        self.max_connections_per_minute = 20
        self.connection_timestamps = deque(maxlen=20)
        self.max_requests_per_minute = 40
        self.request_timestamps = defaultdict(lambda: deque(maxlen=40))
        self.secret_key = secret_key

        # Security: Message validation
        self.max_message_size = 10 * 1024 * 1024  # 10MB limit

        # Security: Trusted agents
        self.trusted_agents = {}

        # Security: Audit log
        self.security_log = []        

        self.enable_ssl = True  # Set to False for basic P2P.
        # i provided basic cert file and key since there are other layered security other than ssl, and also due to infrequent external connections.
        self.ssl_cert_file = ssl_cert_file
        self.ssl_key_file = ssl_key_file
        self.ssl_context = ssl_context
        self.client_ssl_context = client_ssl_context

        if self.enable_ssl:
            print('[+] setting up SSL...')
            self._setup_ssl()

        self.allowed_ips = set()  # Add trusted IPs
        self.blocked_ips = set()  # Block malicious IPs

        self.bind_host = bind_host
        self.security_level = security_level or getattr(pipeline, 'security_level', None)

        # Message types
        self.MSG_TYPES = {
            'PREDICT_REQUEST': 1,
            'PREDICT_RESPONSE': 2,
            'MEMORY_SYNC_REQUEST': 3,
            'MEMORY_SYNC_RESPONSE': 4,
            'ENSEMBLE_VOTE_REQUEST': 5,
            'ENSEMBLE_VOTE_RESPONSE': 6,
            'FAILURE_REPORT': 7,
            'TRUST_UPDATE': 8,
            'AGENT_INFO': 9,
            'PING': 10,
            'PONG': 11,
            'DISCONNECT': 12
        }
        
        # message queue
        self.max_retries = 3
        self.retry_delay = 1.0
        self.message_timeout = 30.0 
        self.CHUNK_SIZE = 8192
        self.predict_manager = predict_manager
        self._health_check_interval = 30  # seconds

        self.use_async = use_async
        
        # Register message handlers
        print('[=++=] Initiating message Queue')
        self.message_queue = AsyncMessageQueue()
            
        self.message_queue.register_handler('predict_request', self._handle_predict_request_async)
        self.message_queue.register_handler('memory_sync', self._handle_memory_sync_async)
        self.message_queue.register_handler('ensemble_vote', self._handle_ensemble_vote_async)
        self.message_queue.register_handler('ping', self._handle_ping)
        self.message_queue.register_handler('status', self._handle_status)
                
        if self.use_async:
            self._start_health_checker()     

        # Queue for outgoing messages (buffered with retry)
        self.outgoing_queue = deque()
        self.queue_processor_thread = None
        self._last_health_check = time.time()       
            
        # Trust configuration
        self.min_trust_level_for_auto_add = TrustLevel.STANDARD
        self.trusted_agents = {}  # agent_id -> {'token': token, 'trust_level': TrustLevel, 'added_at': datetime} 
        self.highly_trusted_peer = []
        self.socket_owners = {}

        self.pending_requests = {}  # request_id -> Future
        self.request_lock = threading.Lock()        

    # ============ SECURITY FEATURES ============

    def _check_ip_access(self, ip: str) -> bool:
        """
        IP access check with security-level-aware default policy.

        Empty allowed_ips behavior:
        DEVELOPMENT/STAGING + bound to 127.0.0.1
            → loopback only, allow 127.x.x.x automatically
            → external IPs blocked implicitly by OS before reaching here

        PRODUCTION/HARDENED + bound to 0.0.0.0
            → allowed_ips MUST be populated, empty = deny all external
            → forces explicit configuration rather than accidental open access

        Explicit allowed_ips populated
            → always enforced regardless of security level
        """
        print(f'|| Checking IP access for: {ip}')

        # blocked list always takes priority
        if ip in self.blocked_ips:
            self._log_security_event('ip_blocked_access_denied', {'ip': ip})
            return False

        # loopback always allowed — needed for local agent communication
        if ip in ('127.0.0.1', '::1', 'localhost'):
            return True

        # explicit allowlist — always enforced when populated
        if self.allowed_ips:
            allowed = ip in self.allowed_ips
            if not allowed:
                self._log_security_event('ip_not_in_allowlist', {'ip': ip})
            return allowed

        # allowed_ips is empty — behavior depends on security level
        security_level = getattr(self, 'security_level', None)
        bind_host      = getattr(self, 'bind_host', '0.0.0.0')

        if security_level in (SecurityLevel.HARDENED, SecurityLevel.PRODUCTION):
            # PRODUCTION/HARDENED with empty allowlist and external binding
            # → deny external IPs, require explicit configuration
            print(f'[⚠️] IP {ip} denied here — allowed_ips is empty in '
                f'{security_level.value} mode. '
                f'Populate allowed_ips to permit external peers.')
            self._log_security_event('ip_denied_empty_allowlist', {
                'ip': ip,
                'security_level': security_level.value,
                'hint': 'populate allowed_ips or use DEVELOPMENT mode for local testing'
            })
            return False

        elif security_level in (SecurityLevel.DEVELOPMENT, SecurityLevel.STAGING):
            # DEVELOPMENT/STAGING bound to 127.0.0.1 — external IPs
            # shouldn't reach here, sometimes.
            if bind_host == '127.0.0.1':
                print(f'[⚠️] External IP {ip} reached local-only server '
                    f'— denying')
                return False

            # bound to 0.0.0.0 in dev mode.
            print(f'[⚠️] Allowing {ip} in {security_level.value} mode '
                f'with empty allowlist — not recommended for production')
            return True

        else:
            # no security level set — conservative default, deny external
            print(f'[⚠️] IP {ip} denied — no security level configured '
                f'and allowed_ips is empty')
            self._log_security_event('ip_denied_no_security_config', {'ip': ip})
            return False    

    def add_allowed_ip(self, ip):
        self.allowed_ips.add(ip)
        self._log_security_event('ip_allowed', {'ip': ip})

    def remove_allowed_ip(self, ip):
        self.allowed_ips.discard(ip)
        self._log_security_event('ip_removed_from_allow', {'ip': ip})

    def add_blocked_ip(self, ip):
        self.blocked_ips.add(ip)
        self._log_security_event('ip_blocked', {'ip': ip})

    def remove_blocked_ip(self, ip):
        self.blocked_ips.discard(ip)
        self._log_security_event('ip_removed_from_block', {'ip': ip})

    def _setup_ssl(self):
        try:
            if self.ssl_context and isinstance(self.ssl_context, ssl.SSLContext):
                # User passed a pre-built server context
                print("✅ Using user-provided SSL context.")

                if self.client_ssl_context and isinstance(self.client_ssl_context, ssl.SSLContext):
                    # User also provided client context — ideal case
                    print("✅ Using user-provided client SSL context.")

                elif self.ssl_cert_file and self.ssl_key_file:
                    # Build client context from provided cert files
                    client_ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
                    client_ctx.load_cert_chain(self.ssl_cert_file, self.ssl_key_file)
                    client_ctx.check_hostname = False
                    self.client_ssl_context = client_ctx
                    print("✅ Built client SSL context from provided cert files.")

                else:
                    # No client context and no cert files — can't do mTLS outbound
                    print("⚠️  No client cert available for outgoing connections. "
                        "Peers requiring mTLS will reject this agent.")
                    self.client_ssl_context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)

            elif self.ssl_cert_file and self.ssl_key_file:
                # User supplied cert files — resolve and load both contexts
                cert_path, key_path, ssl_contexts = self.resolve_and_load_ssl(
                    self.ssl_cert_file, self.ssl_key_file
                )
                if ssl_contexts:
                    self.ssl_cert_file      = cert_path
                    self.ssl_key_file       = key_path
                    self.ssl_context        = ssl_contexts["server"]
                    self.client_ssl_context = ssl_contexts["client"]
                else:
                    print("⚠️  Provided cert/key could not be loaded, falling back to self-signed.")
                    self._generate_self_signed_cert()

            else:
                # Nothing provided — generate self-signed fallback
                self._generate_self_signed_cert()

        except Exception as e:
            print(f"[!] SSL setup failed: {e}")
            self.enable_ssl = False

    def _generate_self_signed_cert(self):
        print(
            "⚠️  SSL running in self-signed fallback mode. "
            "Provides encryption but NOT production-grade identity verification. "
            "Supply real certs via ssl_cert_file/ssl_key_file for production use."
        )
        def _make_key():
            return rsa.generate_private_key(public_exponent=65537, key_size=2048)

        def _save_pem(path, data):
            with open(path, 'wb') as f:
                f.write(data)

        # ── Generate CA ──────
        ca_key = _make_key()
        ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, u"AbstractAgent-CA")])
        ca_cert = (
            x509.CertificateBuilder()
            .subject_name(ca_name)
            .issuer_name(ca_name)
            .public_key(ca_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.now(timezone.utc))
            .not_valid_after(datetime.now(timezone.utc) + timedelta(days=365))
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
            .sign(ca_key, hashes.SHA256())
        )
        ca_cert_pem = ca_cert.public_bytes(serialization.Encoding.PEM)
        ca_key_pem  = ca_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption()
        )
        _save_pem('ca.crt', ca_cert_pem)
        _save_pem('ca.key', ca_key_pem)
        # Lock down private key permissions on UNIX
        if os.name != 'nt':
            for key_file in ('ca.key', 'server.key', 'client.key'):
                try:
                    os.chmod(key_file, stat.S_IRUSR | stat.S_IWUSR)  # 600
                except OSError as e:
                    logger.warning(f"⚠️  Could not set permissions on {key_file}: {e}")

        # ─ Helper: sign a cert with the CA ──────────────────────────
        def _make_signed_cert(common_name: str, san_dns: str, cert_path: str, key_path: str):
            key = _make_key()
            name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])
            cert = (
                x509.CertificateBuilder()
                .subject_name(name)
                .issuer_name(ca_cert.subject)       # signed by CA, not self
                .public_key(key.public_key())
                .serial_number(x509.random_serial_number())
                .not_valid_before(datetime.now(timezone.utc))
                .not_valid_after(datetime.now(timezone.utc) + timedelta(days=365))
                .add_extension(
                    x509.SubjectAlternativeName([x509.DNSName(san_dns)]),
                    critical=False,
                )
                .add_extension(
                    x509.BasicConstraints(ca=False, path_length=None), critical=True
                )
                .sign(ca_key, hashes.SHA256())      # signed by CA key
            )
            _save_pem(cert_path, cert.public_bytes(serialization.Encoding.PEM))
            _save_pem(key_path,  key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption()
            ))
            return cert

        # ── Generate server.crt and client.crt ───────────────────────
        _make_signed_cert("server", "localhost", "server.crt", "server.key")
        _make_signed_cert("client", "localhost", "client.crt", "client.key")

        logger.info("✅ Generated CA, server, and client certificates.")

        # ── Build SSL contexts directly from generated files ──────────
        try:
            # Server: presents server.crt, trusts clients signed by the CA
            server_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            server_ctx.load_cert_chain('server.crt', 'server.key')
            server_ctx.load_verify_locations('ca.crt')
            server_ctx.verify_mode   = ssl.CERT_REQUIRED
            server_ctx.check_hostname = False

            # Client: presents client.crt, trusts servers signed by the CA
            client_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            client_ctx.load_cert_chain('client.crt', 'client.key')
            client_ctx.load_verify_locations('ca.crt')
            client_ctx.verify_mode   = ssl.CERT_REQUIRED
            client_ctx.check_hostname = False

            self.ssl_cert_file      = 'server.crt'
            self.ssl_key_file       = 'server.key'
            self.ssl_context        = server_ctx
            self.client_ssl_context = client_ctx

        except ssl.SSLError as e:
            logger.error(f"❌ SSL context build failed: {e}")

    def _generate_auth_token(self):
        return hashlib.sha256(os.urandom(32)).hexdigest()

    def _generate_secret_key(self):
        return hashlib.sha256(os.urandom(48)).hexdigest()

    def _log_security_event(self, event_type, details):
        self.security_log.append({
            'timestamp': datetime.now().isoformat(),
            'event': event_type,
            'details': details
        })
        if len(self.security_log) > 1000:
            self.security_log = self.security_log[-1000:]

    def _sanitize_input(self, text, amount=1000):
        if not isinstance(text, str):
            return str(text)
        sanitized = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
        return sanitized[:amount]

    def _sanitize_arrays_and_dicts(self, data, amount=1000):
        if isinstance(data, list):
            return [self._sanitize_input(item, amount) for item in data]
        elif isinstance(data, dict):
            return {key: self._sanitize_input(value, amount) for key, value in data.items()}
        else:
            return self._sanitize_input(data, amount)


    def _check_rate_limit(self, agent_id=None):
        start = time.time()
        if not agent_id:
            print('|| No agent ID provided for rate limiting, applying global connection limit.')
            return False

        print(f'|| Checking rate limit for agent: {agent_id}')
        now = time.time()
        self.connection_timestamps = [t for t in self.connection_timestamps if now - t < 10]
        recent_connections = len(self.connection_timestamps)
        if recent_connections > self.max_connections_per_minute:
            self._log_security_event('rate_limit_exceeded', {'type': 'connection', 'agent': agent_id})
            return False
        if agent_id:
            stale = [aid for aid, timestamps in self.request_timestamps.items() if not timestamps or now - timestamps[-1] >= 10]
            for aid in stale:
                del self.request_timestamps[aid]

            self.request_timestamps[agent_id] = [t for t in self.request_timestamps[agent_id] if now - t < 10]
            if time.time() - start > 5:
                print('|| Rate limit check timed out.')
                return False  

            recent_requests = len(self.request_timestamps[agent_id])
            if recent_requests > self.max_requests_per_minute:
                self._log_security_event('rate_limit_exceeded', {'type': 'request', 'agent': agent_id})
                return False
        return True

    def _sign_message(self, message):
        # Create HMAC signature - DOES NOT modify original message
 
        # Created a COPY of the message with timestamp
        signed_message = message.copy()  # ← IMPORTANT: Copy!
        
        # Ensure timestamp is float if present
        if 'timestamp' in signed_message and isinstance(signed_message['timestamp'], str):
            from datetime import datetime
            try:
                dt = datetime.fromisoformat(signed_message['timestamp'].replace('Z', '+00:00'))
                signed_message['timestamp'] = dt.timestamp()
            except:
                signed_message['timestamp'] = time.time()
    
        # Sort keys for consistent serialization
        sorted_message = {k: signed_message[k] for k in sorted(signed_message.keys())}

        message_bytes = json.dumps(sorted_message, sort_key=True, default=str).encode('utf-8')
      
        key = self.secret_key.encode() if isinstance(self.secret_key, str) else self.secret_key
        signature = hmac.new(key, message_bytes, hashlib.sha256).hexdigest()

        print(f'|| Signing message with: {len(message)} total of size')  
        logger.info(f"[=] Signing message: {len(message)}")
        return signature



    def resolve_and_load_ssl(self, cert_filename, key_filename):
        def find_file(filename):
            candidates = []

            if os.path.isabs(filename):
                candidates.append(filename)
            else:
                # current working directory
                candidates.append(os.path.join(os.getcwd(), filename))

                # script directory
                try:
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    candidates.append(os.path.join(script_dir, filename))
                    # 3b — certs/ subfolder next to script
                    candidates.append(os.path.join(script_dir, "certs", filename))
                except NameError:
                    pass

                # home directory
                candidates.append(os.path.join(os.path.expanduser('~'), filename))

                # common data folders
                home = os.path.expanduser('~')
                for folder in ['Downloads', 'Documents', 'Desktop', 'Data', 'data']:
                    candidates.append(os.path.join(home, folder, filename))

                # sys.path entries 
                for p in sys.path:
                    if p:
                        candidates.append(os.path.join(p, filename))

            filepath = None
            for candidate in candidates:
                if os.path.exists(candidate):
                    filepath = candidate
                    break

            if filepath is None:
                print(f"❌ Could not find '{filename}' in any of these locations:")
                for c in candidates[:6]:
                    print(f"   {c}")
                print(f"\n💡 Tip: place your SSL files in a certs/ folder next to your script, or pass the full path.")
                print(f"   {os.getcwd()}\\{filename}")
                print(f"   {os.path.expanduser('~')}\\Downloads\\{filename}")
                return None

            print(f"✅ Found '{filename}' at: {filepath}")
            return filepath

        # Resolve both files
        cert_path = find_file(cert_filename)
        key_path  = find_file(key_filename)

        if not cert_path or not key_path:
            print("❌ SSL load aborted — one or more files not found.")
            return None, None, None

        # Load SSL context
        try:
            # Server context (for accepting incoming peer connections)
            server_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            server_ctx.load_cert_chain(cert_path, key_path)
            server_ctx.load_verify_locations(cert_path)  # trust self-signed cert
            server_ctx.verify_mode = ssl.CERT_REQUIRED
            server_ctx.check_hostname = False
            print("✅ Server SSL context loaded.")

            # Client context (for outgoing peer connections)
            client_ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
            client_ctx.load_cert_chain(cert_path, key_path)
            client_ctx.load_verify_locations(cert_path)  # trust self-signed cert
            client_ctx.verify_mode = ssl.CERT_REQUIRED
            client_ctx.check_hostname = False
            print("✅ Client SSL context loaded.")

            return cert_path, key_path, {"server": server_ctx, "client": client_ctx}

        except ssl.SSLError as e:
            print(f"❌ SSL error while loading cert/key: {e}")
            return cert_path, key_path, None
        except Exception as e:
            print(f"❌ Unexpected error loading SSL: {e}")
            return cert_path, key_path, None


    def _verify_signature(self, message, signature):
        # Verify signature - with timestamp in message
        
        # Create a copy without the signature field
        print(f'|| Verifying message signature total: {len(message)}')
        temp_msg = {k: v for k, v in message.items() if k != 'signature'}

        if 'timestamp' in temp_msg and isinstance(temp_msg['timestamp'], str):
            try:
                dt = datetime.fromisoformat(temp_msg['timestamp'].replace('Z', '+00:00'))
                temp_msg['timestamp'] = dt.timestamp()
            except:
                temp_msg['timestamp'] = time.time()
          
        # Sort keys for consistent serialization
        sorted_msg = {k: temp_msg[k] for k in sorted(temp_msg.keys())}
          
        message_bytes = pickle.dumps(sorted_msg, protocol=pickle.HIGHEST_PROTOCOL)
         
        key = self.secret_key.encode() if isinstance(self.secret_key, str) else self.secret_key
        expected = hmac.new(key, message_bytes, hashlib.sha256).hexdigest()
        
        result = hmac.compare_digest(expected, signature)

        print(f'[=] Comparing result...')
        print(f'|| Signature verification result: {result}')
        logger.info(f"[-] Signature verification result: {result}")

        return result

   
    def add_trusted_agent(self, agent_id, agent_token):
        if agent_id == 'local':
            print(f"[❌] Cannot add 'local' as trusted agent")
            return

        self.trusted_agents[agent_id] = {'token': agent_token, 'added_at': datetime.now()}
        self._log_security_event('trusted_agent_added', {'agent_id': agent_id})

    def _authenticate_agent(self, token, agent_id):
        print(f'|| Authenticating agent: {agent_id}...')
        logger.info(f"[==] Authenticating agent: {agent_id}...")

        if agent_id in self.highly_trusted_peer:
            print('[=+=] Agent is authenticated and already verified')
            return True

        elif token == self.auth_token:
            print(f"[=✅=] Agent {agent_id} authenticated with SHARED SECRET (FULL trust)")
            
            # Add to trusted list with FULL trust if not exists
            if agent_id not in self.trusted_agents:
                self._add_trusted_agent(agent_id, token, TrustLevel.FULL, source="shared_secret")
            else:
                # Update trust level if higher
                current_level = self.trusted_agents[agent_id].get('trust_level', TrustLevel.BASIC)
                if TrustLevel.FULL > current_level:
                    self.trusted_agents[agent_id]['trust_level'] = TrustLevel.FULL
                    print(f"[=] Upgraded trust level to FULL")
                    self.highly_trusted_peer.append(agent_id)

        elif agent_id in self.trusted_agents:
            stored_token = self.trusted_agents[agent_id]['token']
            stored_trust = self.trusted_agents[agent_id].get('trust_level', TrustLevel.BASIC)
            
            if stored_token == token:
                print(f"[✅] Agent {agent_id} authenticated with {stored_trust.name} trust")
                self.highly_trusted_peer.append(agent_id)
                return True
            else:
                print(f"[❌] Token mismatch for {agent_id}")
                return False

        else:  
            auto_add_threshold = getattr(self, 'min_trust_level_for_auto_add', TrustLevel.STANDARD)
        
            print(f"[-] Agent {agent_id} not in trusted list")
            print(f"[=/=] Auto-add threshold: {auto_add_threshold.name}")
            
            # Only auto-add if you have high trust in the network
            if auto_add_threshold == TrustLevel.FULL:
                # In high-security mode, don't auto-add
                print(f"[-] Auto-add disabled (requires manual approval)")
                return False
            else:
                # Auto-add with BASIC trust
                print(f"[+] Auto-adding agent {agent_id} with BASIC trust")
                self._add_trusted_agent(agent_id, token, TrustLevel.BASIC, source="auto_discovery")
                return True

            print('[==] Agent is not authenticated! ')
            return False

    def _add_trusted_agent(self, agent_id, token, trust_level=TrustLevel.STANDARD, source="manual"):
        """Add a trusted agent with specified trust level"""
        if agent_id == 'local':
            print(f"[❌] Cannot add 'local' as trusted agent")
            return

        self.trusted_agents[agent_id] = {
            'token': token,
            'trust_level': trust_level,
            'added_at': datetime.now(),
            'added_by': source,
            'last_seen': datetime.now(),
            'successful_connections': 0,
            'failed_connections': 0
        }
        
        self._log_security_event('trusted_agent_added', {
            'agent_id': agent_id,
            'trust_level': trust_level.name,
            'source': source
        })
        
        print(f"✅ Added trusted agent: {agent_id} (trust: {trust_level.name})")




    def _get_bind_host(self) -> str:
        """
        verifying for host binding IP.

        DEVELOPMENT → 127.0.0.1  loop back only, safest default,
                                no external exposure even on dev machines
        STAGING     → 127.0.0.1  still local, test P2P on same machine
        PRODUCTION  → 0.0.0.0    multi-machine P2P, SSL should be enabled
        HARDENED    → 0.0.0.0    multi-machine, SSL enforced separately

        """
        # explicit override always wins
        if hasattr(self, 'bind_host') and self.bind_host:
            return self.bind_host

        # derive from security level if available
        if hasattr(self, 'security_level'):
            if self.security_level in (SecurityLevel.DEVELOPMENT,
                                    SecurityLevel.STAGING):
                return '127.0.0.1'
            else:
                return '0.0.0.0'

        # no security level set — default to loopback, safest choice for local usage here.
        return '127.0.0.1'

    # ============ SERVER METHODS ============
    def start_server(self):

        print('[!] Inspect this Information Carefully:')
        self._validate_security_config()

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        bind_host = self._get_bind_host()
        self.socket.bind(('0.0.0.0', self.port))

        self.socket.settimeout(1.0)
        self.socket.listen(5)
        self.running = True
        logger.info(f"[=] Server started on port {self.port} with SSL={'enabled' if self.enable_ssl else 'disabled'}")

        print(f"[🤖] Agent listening on port {self.port}")
        if bind_host == '0.0.0.0' and not self.enable_ssl:
            logger.warning(
                '[⚠️] SECURITY WARNING: Server bound to 0.0.0.0 without SSL. '
                'All network interfaces are exposed. Use security_level=PRODUCTION '
                'or higher, or set enable_ssl=True for external deployments.'
            )
            print('[⚠️] SECURITY WARNING: Bound to all interfaces without SSL '
              '— suitable for local P2P only')    

        # Start accepting connections in background
        accept_thread = threading.Thread(target=self._accept_connections, daemon=True)
        accept_thread.start()

        self._server_started = True
        logger.info("[=] Server started and accepting connections...")
        
        return self.socket


    def _validate_security_config(self):
        """This function Warns about dangerous security configurations at startup."""
        security_level = getattr(self, 'security_level', None)
        bind_host      = getattr(self, 'bind_host', '0.0.0.0')

        warnings = []

        if bind_host == '0.0.0.0' and not self.enable_ssl:
            warnings.append(
                'Bound to 0.0.0.0 without SSL — all interfaces exposed unencrypted'
            )

        if not self.allowed_ips and bind_host == '0.0.0.0':
            if security_level in (SecurityLevel.PRODUCTION, SecurityLevel.HARDENED):
                warnings.append(
                    f'allowed_ips is empty in {security_level.value} mode '
                    f'with 0.0.0.0 binding — external peers will be denied. '
                    f'Add trusted peer IPs via add_allowed_ip()'
                )
            else:
                warnings.append(
                    'allowed_ips is empty — all external IPs currently permitted. '
                    'Consider populating allowed_ips for production deployments.'
                )

        if not self.secret_key:
            warnings.append('[!] No secret_key configured — HMAC signing disabled')

        for w in warnings:
            print(f'[⚠️] SECURITY CONFIG: {w}')
            logger.warning(f'[⚠️] SECURITY CONFIG: {w}')
            self._log_security_event('security_config_warning', {'warning': w})


    def _accept_connections(self):
        while self.running:
            try:
                client, addr = self.socket.accept()
                client.settimeout(self.connection_timeout)
                host = addr[0]
                port = addr[1]

                if host in ['127.0.0.1', 'localhost'] and port == self.port:
                    print(f"[❌] Rejected self-connection from {host}:{port}")
                    client.close()
                    continue
    

                if not self._check_ip_access(host):
                    print(f"[-] Connection attempt from blocked IP: {host}")
                    self._log_security_event('connection_blocked', {'ip': host})
                    client.close()
                    return

                print(f"📡 Connected to agent at {addr}")
                auth_msg = self._receive_message(client)
                if not auth_msg:
                    print(f"[-] No authentication message from {addr}")
                    client.close()
                    continue
                                        

                if not self._authenticate_agent(auth_msg.get('token', ''), f"{addr[0]}:{addr[1]}"):
                    print(f"[-] Authentication failed for agent with address: {addr}")
                    self._log_security_event('authentication_failed', {'agent': f"{addr[0]}:{addr[1]}"})
                    self.report_failure(id(self), 'authentication', reason=f'Failed authentication from {addr}')
                    client.close()
                    return

                # Send agent info to identify
                self._send_agent_info(client)
                
                # Start handler thread
                thread = threading.Thread(target=self._handle_client, args=(client, addr))
                thread.daemon = True
                thread.start()

            except socket.timeout:
                continue 

            except Exception as e:
                if self.running:
                    print(f"[-] Accept error: {e}")
                    traceback.print_exc()
                    self.report_failure(id(self), 'processing', reason=f'{e}')
                                        
                break
    
    def _send_agent_info(self, client):
        info = {
            'type': self.MSG_TYPES['AGENT_INFO'],
            'agent_id': id(self),
            'agent_name': self.memory_name,
            'token': self.auth_token,
            'capabilities': ['prediction', 'memory_sync', 'ensemble'],
            'timestamp': time.time()
        }
        self._send_message(client, info)
        print(f"[==] Sent agent info for authentication")
        logger.info("[==] Sent agent info for authentication")


    def stop_server(self):
        self.running = False   
        # Close all connections
        for conn in self.connections:
            try:    
                self._send_message(conn, {'type': self.MSG_TYPES['DISCONNECT']})

                conn.shutdown(socket.SHUT_RDWR)
                conn.close()
            except Exception as e:
                print(f'[= ERROR =] Socket cant be shutdown due to: {e}')
                pass

        self.connections.clear()
        if self.socket:
            try:
                self.socket.close()  
            except Exception as e:
                print(f'[= ERROR =] Socket cant be closed due to: {e}')
                pass

        print("[🛑] Server stopped")
    
    # ============ CLIENT METHODS ============
    def _is_duplicate_connection(self, host, port):
        # Check if this connection attempt is a duplicate in later flow
        for agent_id, info in self.remote_agents.items():
            if info.get('host') == host and info.get('port') == port:
                return True
        return False    


    def connect_to_agent(self, host, port):
        """
        Connect to a peer agent with proper authentication flow.
        """
        if host == 'local':
            print(f"[❌] Cannot connect to 'local'")
            return None 

        if host in ['127.0.0.1', 'localhost', '0.0.0.0']:
            # Check if this is our own port
            if port == self.port or port == 0:
                print(f"[❌] Rejecting self-connection attempt to {host}:{port}")
                return None

        agent_id = f"{host}:{port}"
        print(f'🔗 Attempting to connect to agent: {agent_id}')
        
        # Generate a unique ID for this connection attempt
        connection_id = str(uuid.uuid4())[:8]
        
        try:
            # ========== SECURITY CHECKS ==========
            # Rate limiting 
            if not self._check_rate_limit(agent_id):
                print(f'[❌] Rate limit exceeded for {agent_id}')
                self._log_security_event('rate_limit_exceeded', 
                                        {'type': 'connection_attempt', 'agent': agent_id})
                self.report_failure(agent_id, 'connection_attempt', reason=f'Rate limit exceeded for {agent_id}')

                return None
            
            # IP access check
            if not self._check_ip_access(host):
                print(f"[-] Connection attempt to blocked IP: {host}")
                self._log_security_event('connection_blocked', {'ip': host})
                return None

            # Socket creation
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)    
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)  # 1MB
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024 * 1024)  # 1MB      
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            if hasattr(socket, 'TCP_KEEPIDLE'):
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 10)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 5)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3)

           
            sock.settimeout(10)

            print(f"[connect_to_agent() SOCKET CREATED] id={id(sock)}")            
            sock.settimeout(self.connection_timeout)
            print(f'[==] Connecting to {host}:{port}...')
            sock.connect((host, port))

            if self.enable_ssl and self.ssl_context:
                sock = self.ssl_context.wrap_socket(sock, server_hostname=host)
                print('[==] Socket Connected with SSL Provided!')
            
            # ========== SEND AUTHENTICATION FIRST ==========
            # Send agent info and token BEFORE receiving
            auth_message = {
                'type': self.MSG_TYPES['AGENT_INFO'],
                'agent_id': id(self),
                'agent_name': getattr(self, 'memory_name', 'unknown'),
                'token': self.auth_token,  # Your authentication token
                'timestamp': time.time()
            }
            
            if not self._send_message(sock, auth_message):
                print(f"[-] Failed to send authentication to {host}:{port}")
                self._log_security_event('authentication_failed', {'agent': agent_id})
                sock.close()
                return None
            
            print(f'[=?=] Authentication sent')
            
            # ========== RECEIVE PEER INFO ==========
            info = self._receive_message(sock)
            
            if not info:
                print(f"[-] No response from {host}:{port}")
                sock.close()
                return None
            
            # Authenticate the peer
            if not self._authenticate_agent(info.get('token', ''), agent_id):
                print(f"[-] Authentication failed for agent {host}:{port}")
                self._log_security_event('authentication_failed', {'agent': agent_id})
                sock.close()
                return None
            
            # ========== ESTABLISH PEER RELATIONSHIP ==========
            if info.get('type') == self.MSG_TYPES['AGENT_INFO']:
                remote_id = info.get('agent_id', agent_id)
                
                query_result = self.query_node._establish_peer_nodes(remote_id)
                
                if not query_result:
                    print(f'[❌] Connection to peer {remote_id} denied by query node.')
                    self.report_failure(id(self), 'peer_establishment', reason=f'Connection to peer {remote_id} denied')
                    sock.close()
                    return None

                print('[===] Connection to peer is permitted')
                
                # Store the connection
                self.remote_agents[remote_id] = {
                    'sock': sock,
                    'host': host,
                    'port': port,
                    'trust': 1.0,
                    'last_seen': datetime.now(),
                    'failures': 0,
                    'connection_id': connection_id
                }
                self.connections.append(sock)
                
                print(f"[=✅=] Connected to agent {remote_id} at {host}:{port}")
                if self.running:
                    print('[=+=] server is still listening for messages!')
                return sock
            else:
                print(f"[❌] Invalid agent response from {host}:{port}")
                self.report_failure(id(self), 'authentication', reason=f'Failed authentication from {host}:{port}')                
                sock.close()
                return None
                
        except socket.timeout:
            print(f"[❌] Connection timeout to {host}:{port}")
            return None
        except ConnectionRefusedError:
            print(f"[❌] Connection refused by {host}:{port} - server not running?")
            return None
        except Exception as e:
            print(f"[❌] Failed to connect to {host}:{port}: {e}")
            import traceback
            traceback.print_exc()
            return None

    
    def disconnect_agent(self, agent_id):
        if agent_id in self.remote_agents:
            try:
                self._send_message(self.remote_agents[agent_id]['sock'], 
                                  {'type': self.MSG_TYPES['DISCONNECT']})
                self.remote_agents[agent_id]['sock'].close()

                print(f'[===] Removing Agent id: {agent_id}')
                del self.remote_agents[agent_id]
            except:
                pass
            print(f"🔌 Disconnected from agent {agent_id}")

    def _sanitize_structured(self, data, amount=1000):
        """Recursively sanitize strings inside structures"""
        if isinstance(data, str):
            return self._sanitize_input(data, amount)
        elif isinstance(data, list):
            return [self._sanitize_structured(item, amount) for item in data]
        elif isinstance(data, tuple):
            return tuple(self._sanitize_structured(item, amount) for item in data)
        elif isinstance(data, dict):
            return {key: self._sanitize_structured(value, amount) for key, value in data.items()}
        else:
            return data

    # ============ asynchronous queue setup ============
    async def _handle_predict_request_async(self, message):
        # Async handler for prediction requests
        payload = message.payload

        # Initialize variables
        text = None
        test_titles = None
        label_map = None
        rules = None
        X = None
        y = None
        
        # ✅ Check payload (which is a dict), not the message itself
        if isinstance(payload, dict):
            if 'test_titles' in payload:
                test_titles = payload.get('test_titles')
                label_map = payload.get('label_map')
                rules = payload.get('rules')
                X = payload.get('X')
                y = payload.get('y')
                
                # Sanitize if needed
                if test_titles:
                    test_titles = self._sanitize_structured(test_titles)
                if label_map:
                    label_map = self._sanitize_structured(label_map)
                if rules:
                    rules = self._sanitize_structured(rules)
                if X is not None:
                    X = self._sanitize_structured(X)
                if y is not None:
                    y = self._sanitize_structured(y)

                if X is None or y is None:
                    print('[=] Got necessary titles, label_map and rules.')
                else:
                    print('[=] Got necessary titles, label_map, rules and X And Y samples.')
            else:
                text = payload.get('text')
                if text:
                    text = self._sanitize_input(text)
                print(f'[=] Got text: {text}')
        else:
            # Fallback: maybe payload is the text directly
            text = str(payload) if payload else None
        
        if not text and not test_titles:
            print('[===] ERROR: No text or test_titles in message payload!')
            return {'type': self.MSG_TYPES['PREDICT_RESPONSE'], 'error': 'No text or test_titles provided'}
        
        # Run the actual prediction
        print(f'[=] Initiating prediction method')
        try:
            if test_titles is not None:
                print('[=] initiating Advanced prediction method...')
                if not self.pipeline.autonomous:
                    self.pipeline.autonomous = True
                    self.pipeline.ensemble.explainer.supervised_learning = False

                if self.predict_manager is not None:
                    result = await asyncio.to_thread(
                        self.predict_manager.advanced_prediction_method,
                        test_titles, label_map, rules, X=X, y=y,
                        show_proba=True,
                        use_transformer=self.pipeline.use_transformer
                    )
                    # Handle tuple return (result, chosen_label, confidence)
                    if isinstance(result, tuple) and len(result) == 3:
                        _, chosen_label, confidence = result
                    else:
                        chosen_label = result.get('prediction', 'unknown')
                        confidence = result.get('confidence', 0)
                    
                    return {
                        'type': self.MSG_TYPES['PREDICT_RESPONSE'],
                        'prediction': chosen_label,
                        'confidence': confidence,
                        'success': True
                    }

                else:
                    print('[=] Initaiting basic prediction...')
                    result = await asyncio.to_thread(self.pipeline.predict_single, text)
            
                    return {
                        'type': self.MSG_TYPES['PREDICT_RESPONSE'],
                        'prediction': result.get('prediction'),
                        'confidence': result.get('confidence'),
                        'probabilities': result.get('probabilities', []),
                        'agent_id': id(self),
                        'success': True
                    }                    
            else:

                print('[=] Basic prediction method')
                result = await asyncio.to_thread(self.pipeline.predict_single, text)
                
                return {
                    'type': self.MSG_TYPES['PREDICT_RESPONSE'],
                    'prediction': result.get('prediction'),
                    'confidence': result.get('confidence'),
                    'probabilities': result.get('probabilities', []),
                    'agent_id': id(self),
                    'success': True
                }
                
        except Exception as e:
            logger.info(f'[==] error in async method predict request: {e}')
            return {'type': self.MSG_TYPES['PREDICT_RESPONSE'], 'error': str(e), 'success': False}    
        

    async def _async_method_handle_predict_request_(self, message, sender_id, method='basic_prediction', predict_manager=None):
        # Handle prediction request async-ly
        text = None
        test_titles = None
        label_map = None
        rules = None
            
        if 'test_titles' in message:
            test_titles = message.get('test_titles')
            label_map = message.get('label_map')
            rules = message.get('rules')
            X = message.get('X')
            y = message.get('y')

            test_titles = self._sanitize_input(test_titles)
            label_map = self._sanitize_input(label_map)
            rules = self._sanitize_input(rules)
        else:
            text = message.get('text')
            text = self._sanitize_input(text) 

        if not text:
            print('[===] ERROR: No matched configuration in message for prediction!')
            return {'type': self.MSG_TYPES['PREDICT_RESPONSE'], 'error': 'No text provided'}
        
        # Run the actual prediction in thread pool (since predict_single is sync)
        print(f'[=] Initiating prediction method: {method}')
        try:
            print('[=] Advanced prediction method')
            if method != 'basic_prediction' or predict_manager:
                result = await asyncio.to_thread(
                    predict_manager.advanced_prediction_method,
                    test_titles, label_map, rules, X=X, y=y, show_proba=False, use_transformer=self.pipeline.use_transformer
                )
            else:
                print('[=] basic prediction method')
                result = await asyncio.to_thread(self.pipeline.predict_single, text)
            
            return {
                'type': self.MSG_TYPES['PREDICT_RESPONSE'],
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'probabilities': result.get('probabilities', []),
                'agent_id': id(self)
            }
        except Exception as e:
            logger.info(f'[==] error in async method predict request: {e}')
            return {'type': self.MSG_TYPES['PREDICT_RESPONSE'], 'error': str(e)}


    async def _handle_memory_sync_async(self, message):
        # Safe handler for memory sync.
        try:
            logger.info(f"[=] Processing memory sync from {message.sender}")
            return await self._handle_memory_sync_request(message, message.sender)
        except Exception as e:
            logger.error(f"[❌] Memory sync failed: {e}")
            return {'error': str(e), 'status': 'failed'}
    
    async def _handle_ensemble_vote_async(self, message):
        # Safe handler for ensemble voting.
        try:
            logger.info(f"[=] Processing ensemble vote from {message.sender}")
            return await self._handle_ensemble_vote_request(message, message.sender)
        except Exception as e:
            logger.error(f"[❌] Ensemble vote failed: {e}")
            return {'error': str(e), 'status': 'failed'}
    
    async def _handle_ping(self, message):
        # Simple ping handler for health checks.
        return {'pong': True, 'timestamp': time.time(), 'agent_id': self.agent_id}
    
    async def _handle_status(self, message):
        # Status handler for monitoring.
        return {
            'status': 'healthy',
            'queue_stats': self.message_queue.get_stats(),
            'connected_agents': len(self.remote_agents),
            'memory_size': len(self.pipeline.memory),
            'uptime': time.time() - self.start_time if hasattr(self, 'start_time') else 0
        }   


    def request_prediction(self, agent_id: Any, text: Any, timeout: float = 30.0) -> Any:
        # Unified prediction request - works with both sync and async modes.
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Run the async version and wait for result
            result = loop.run_until_complete(
                self.request_prediction_async(agent_id, text, timeout)
            )
            return result
        finally:
            loop.close()

    async def request_advanced_prediction_async(self, manager: Any, use_transformer: bool=False, agent_id: str=None, test_titles: List[tuple]=None, label_map: Dict[str, int]=None, rules: List[tuple]=None, X: np.ndarray=None, y: np.ndarray=None, timeout: float = 30.0, callback: Optional[Callable] = None):
        # Asynchronous prediction request
        # Local bypass - NO QUEUE
        try:
            if agent_id == 'local':
                logger.info(f"[=] Local request - direct execution")
                # Run sync prediction in thread pool
                result = await asyncio.to_thread(manager.advanced_prediction_method, test_titles, label_map, rules, X=X, y=y, show_proba=True, use_transformer=use_transformer)
                logger.info(f"[=] Local result: {result[1]} || confidence: {result[2]}")
                return result  

            msg_id = str(uuid.uuid4())
            message = Message(
                id=msg_id,
                type='predict_request',
                sender=self.temporary_agent_id,
                recipient=agent_id,
                payload={'test_titles': test_titles, 'label_map': label_map, 'rules': rules, 'X':X, 'y':y},
                timestamp=datetime.now(),
                timeout=timeout,
                callback=callback,
                max_retries=self.max_retries
            )

            logger.info(f"[=] Remote request - publishing to queue")
            response = await self.message_queue.publish(message)
            logger.info(f"[=] Queue response type: {type(response)}")

            # Extract prediction from response if needed
            if isinstance(response, dict) and 'prediction' in response:
                return response
            elif isinstance(response, dict) and 'result' in response:
                return response['result']
            else:
                return response  
        except Exception as e:
            print(f'[!] Cannot request advanced prediction async: {e}') 
            response = {'prediction': None, 'result': None} 
            return response
        

    
    def request_prediction_direct(self, agent_id, text, timeout=5):
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        
        # Create future for response
        future = asyncio.Future()
        with self.request_lock:
            self.pending_requests[request_id] = future
        
        # Send message with request_id
        message = {
            'type': 1,
            'text': text,
            'token': self.auth_token,
            'request_id': request_id,  # ← Include in message!
            'timestamp': time.time()
        }
        
        sock = self.remote_agents[agent_id]['sock']
        self._send_message(sock, message)
        
        # Wait for response with timeout
        try:
            return future.result(timeout=timeout)
        finally:
            with self.request_lock:
                self.pending_requests.pop(request_id, None)


    async def request_prediction_async(self, agent_id: Any, text: Any, timeout: float = 30.0, callback: Optional[Callable] = None):
        # Asynchronous prediction request
        # # Local bypass
        if agent_id == 'local':
            return await asyncio.to_thread(self.pipeline.predict_single, text)  

        if agent_id not in self.remote_agents:
            print(f"[❌] No connection to {agent_id}")
            return None
        
        sock = self.remote_agents[agent_id]['sock']
        
        # Create prediction request
        message = {
            'type': self.MSG_TYPES['PREDICT_REQUEST'],
            'text': text,
            'token': self.auth_token,
            'requester': id(self)
        }
        
        try:
            # Send via existing socket
            self._send_message(sock, message)
            
            # Wait for response
            response = self._receive_message(sock)
            
            if response and response.get('type') == self.MSG_TYPES['PREDICT_RESPONSE']:
                return response
            return None
            
        except Exception as e:
            print(f"[❌] Prediction request failed: {e}")
            return None

    def request_prediction_batch(self, agent_id: str, texts, timeout: float = 30.0) -> List[Any]:
        # Batch async prediction requests (parallelized)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(texts)) as executor:
            futures = [
                executor.submit(self.request_prediction, agent_id, text, timeout)
                for text in texts
            ]
            results = [f.result(timeout=timeout) for f in futures]
        
        return results
    
    
    def start_queue_processor(self):
        # Start background queue processor
        self.queue_processor_thread = threading.Thread(target=self._process_outgoing_queue, daemon=True)
        self.queue_processor_thread.start()
    
    def _process_outgoing_queue(self):
        # Process queued outgoing messages
        while self.running:
            if self.outgoing_queue:
                msg = self.outgoing_queue.popleft()
                try:
                    self._send_message(msg['sock'], msg['message'])
                    if msg.get('callback'):
                        msg['callback'](True)
                except Exception as e:
                    if msg.get('callback'):
                        msg['callback'](e)
                    # Retry logic
                    if msg.get('retry_count', 0) < msg.get('max_retries', 3):
                        msg['retry_count'] = msg.get('retry_count', 0) + 1
                        self.outgoing_queue.append(msg)
            else:
                time.sleep(0.01)     

    
    def _start_health_checker(self):
        # Start background health checker for async mode.
        def health_check_loop():
            for _ in range(self._health_check_interval * 10):
                if not self.running:
                    return
                time.sleep(0.1)

            if self.running:
                self._check_agent_health()
        
        self._health_thread = threading.Thread(target=health_check_loop, daemon=True)
        self._health_thread.start()
    
    def _check_health(self):
        # Check health of all connected agents.
        stats = self.message_queue.get_stats()
        logger.debug(f"[=] Queue stats: {stats}")
        
        # Check for stuck messages
        if stats.get('pending_count', 0) > 100:
            logger.warning(f"[=] High pending count: {stats['pending_count']}")
        
        # Ping all agents
        for agent_id in list(self.remote_agents.keys()):
            try:
                result = self.broadcast_ping()
                if agent_id not in result or result[agent_id].get('error'):
                    logger.warning(f"[=] Agent {agent_id} not responding")
            except Exception as e:
                logger.warning(f"[=] Health check failed for {agent_id}: {e}")
    
    def get_queue_stats(self) -> Dict:
        # Get message queue statistics.
        return self.message_queue.get_status()
    
    def get_dead_letter_queue(self) -> List[Dict]:
        # Get failed messages for inspection.
        if hasattr(self.message_queue, 'get_dead_letter_queue'):
            return self.message_queue.get_dead_letter_queue()
        return []
    
    def stop(self):
        # Graceful shutdown.
        logger.info("[=] Shutting down AgentDistributedInference...")
        self.running = False

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.call_soon_threadsafe(
                    lambda: asyncio.ensure_future(self.message_queue.stop())
                )
            else:
                loop.run_until_complete(self.message_queue.stop())
        except Exception as e:
            logger.warning(f"[=] Message queue stop warning: {e}")
        
        logger.info("[=] Shutdown complete")

    # ============ MESSAGE HANDLING ============


    def _send_message(self, sock, message):
        # Send message with signature and DOES NOT modify original
        if sock is None:
            print(f"[==] Send error: socket is None")
            return False
            
        # ✅ Check if socket is still connected
        try:
            sock.getpeername()
        except (socket.error, OSError, AttributeError) as e:
            print(f"[==] Send error: socket is dead - {e}")
            # Remove dead socket from remote_agents
            self._remove_dead_socket(sock)
            return False     

        try:
            msg_to_send = message.copy()  # ← Important: Copy!
            
            # Add signature to the copy
            msg_to_send['signature'] = self._sign_message(msg_to_send)

            sorted_msg = {k: msg_to_send[k] for k in sorted(msg_to_send.keys())}

            print(f'[==] Sending message, Total: {len(sorted_msg)}')   
            data = json.dumps(sorted_msg, default=str).encode('utf-8')
            sock.sendall(len(data).to_bytes(4, 'big'))
            bytes_sent = 0
            while bytes_sent < len(data):
                chunk = data[bytes_sent:bytes_sent + self.CHUNK_SIZE]
                sock.sendall(chunk)
                bytes_sent += len(chunk)
                # Small delay to prevent buffer overflow
                if len(chunk) == self.CHUNK_SIZE:
                    time.sleep(0.001)
              
            print(f'[==] Message sent successfully')
            logger.info(f"[=] Message sent successfully: {sorted_msg}")
            return True
        except Exception as e:
            print(f"[==] Send error: {e}")
            traceback.print_exc()
            self._remove_dead_socket(sock)
            return False


    def _remove_dead_socket(self, sock):
        """Remove dead socket from remote_agents"""
        for agent_id, info in list(self.remote_agents.items()):
            if info.get('sock') == sock:
                print(f"[=] Removing dead connection to {agent_id}")
                del self.remote_agents[agent_id]
                break 

    def _receive_message(self, sock):
        try:
            print(f'[==] Server status: {self.running}')
            print(f'[=] Sock status: {sock}')

            if sock is None:
                print('[=] Sock is None !')
                return None

            try:
                data_len = sock.recv(4)
            except:
                data_len = sock.recv(10)

            print(f'[==] Data length received: {data_len}')
            if not data_len:
                print('[=] received empty message.')
                return None
            
            msg_len = int.from_bytes(data_len, 'big')
            if msg_len > self.max_message_size:
                print('[=] message size exceeds maximum to be handled')
                self.log_security_event('message_too_large', {'size': msg_len})
                return None

            data = b''
       
            while len(data) < msg_len:
                remaining = msg_len - len(data)
                chunk_size = min(self.CHUNK_SIZE, remaining)
                chunk = sock.recv(chunk_size)
                if not chunk:
                    print(f'[=] Connection closed while receiving')
                    return None
                data += chunk            
           
            try:
                message = json.loads(data.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                logger.warning(f'[=] Invalid JSON from peer: {e}')
                self._log_security_event('invalid_json', {})
                return None
        
            if "signature" in message:
                msg_for_verify = {k: v for k, v in message.items() if k != 'signature'}

                if not self._verify_signature(msg_for_verify, message['signature']):
                    logger.warning(f"[=] Invalid message signature from agent {self.temporary_agent_id}")
                    self._log_security_event('invalid_signature', {'agent_id': self.temporary_agent_id})
                    return None

            print('[= Message received]')
            return message

        except socket.timeout:
            print('[-] Socket timeout')
            return None
        except Exception as e:
            logger.error(f"[=] Receive error: {e}")
            traceback.print_exc()
            return None
    
    def _handle_client(self, client, addr):
        agent_id = f"{addr[0]}:{addr[1]}"
        self.temporary_agent_id = agent_id
        
        # Register this thread as the owner of this socket
        self.socket_owners[client] = threading.current_thread().name

        if addr[0] in ['127.0.0.1', 'localhost', 'local'] and addr[1] == self.port:
            print(f"[❌] Client is self, ignoring")
            client.close()
            return        
            
        if self._is_duplicate_connection(addr[0], addr[1]):
            print(f"[⚠️] Duplicate connection from {addr[0]}:{addr[1]}, rejecting")
            client.close()
            return
            
        # ✅ Prevent multiple connections from same host
        for existing_id, info in list(self.remote_agents.items()):
            if info.get('host') == addr[0]:
                print(f"[❌] Already have connection from {addr[0]}, rejecting new connection")
                client.close()
                return            

        while self.running:
            try:
                if 'request_id' in message:
                    continue 


                if not self._check_rate_limit(agent_id):
                    self._send_message(client, {'type': 'error', 'message': 'Rate limit exceeded'})
                    logger.info(f"[=##=] Rate limit exceeded for agent {agent_id}, request reduced.")
                    time.sleep(5)  # Sleep briefly to mitigate rapid retries
                    continue

                if message.get('type') == 2:  # PREDICT_RESPONSE
                    continue  # Skip, we'll read it in request_prediction_method

                message = self._receive_message(client) 
                self.temporary_message = message
                if message is None:
                    print('[-] Message is None.')
                    continue

                request_id = message.get('request_id')
                if request_id and request_id in self.pending_requests:
                    with self.request_lock:
                        future = self.pending_requests.get(request_id)
                        if future and not future.done():
                            future.set_result(message) 

                response = self._process_message(message, agent_id)
    
                print(f'[=~=] Got Response from client with address: {addr[0]}:{addr[1]}')
                if response:
                    print(f'[=] Sending response to client: {client}')
                    self._send_message(client, response)
                    logger.info(f'[=] Succesfully send response to client: {client}')
                else:
                    print("[SERVER] No response to send - ")
                    self._send_message(client, {'type': 'ack', 'status': 'ok'})

            except Exception as e:
                print(f"[=] Handler error for {agent_id}: {e}")
                break
        
        # Cleanup on disconnect
        if agent_id in self.remote_agents:
            print(f'[===] Removing Agent id: {agent_id}')            
            del self.remote_agents[agent_id]
        if client in self.connections:
            self.connections.remove(client)

        client.close()
        print(f"📡 Disconnected from {agent_id}")
    

    def _process_message(self, message, sender_id):
        # Process incoming messages based on type
        msg_type = message.get('type')
        
        if msg_type == self.MSG_TYPES['PREDICT_REQUEST']:
            return self._handle_predict_request(message, sender_id)

        elif msg_type == self.MSG_TYPES['MEMORY_SYNC_REQUEST']:
            return self._handle_memory_sync_request(message, sender_id)
        
        elif msg_type == self.MSG_TYPES['ENSEMBLE_VOTE_REQUEST']:
            return self._handle_ensemble_vote_request(message, sender_id)
        
        elif msg_type == self.MSG_TYPES['FAILURE_REPORT']:
            return self._handle_failure_report(message, sender_id)
        
        elif msg_type == self.MSG_TYPES['TRUST_UPDATE']:
            return self._handle_trust_update(message, sender_id)
        
        elif msg_type == self.MSG_TYPES['PING']:
            return {'type': self.MSG_TYPES['PONG'], 'timestamp': time.time()}
        
        elif msg_type == self.MSG_TYPES['DISCONNECT']:
            return None
        
        return {'type': 'ack', 'status': 'ok'}


    # ====== HANDLE PREDICTION AND UNCERTAINTY CALIBRATION ====== 
    def _check_trust_level(self, agent_id, required_trust=TrustLevel.STANDARD):
        # Check if agent has sufficient trust level for operation
        
        if agent_id not in self.trusted_agents:
            print(f"[-] Agent {agent_id} not trusted")
            return False
        
        agent_trust = self.trusted_agents[agent_id].get('trust_level', TrustLevel.BASIC)
        
        if agent_trust >= required_trust:
            return True
        else:
            print(f"[-] Agent {agent_id} trust level {agent_trust.name} < required {required_trust.name}")
            return False


    def _handle_peer_agent_request(self, probs, self_attn_weights, input_ids, type=None, agreement=False):
        memory_exist = self.sync_with_local_peer(self.memory_name)   
        established_connection = self.query_node._establish_peer_nodes(self.temporary_agent_id)

        if established_connection:
            print(f'[||] Connection established and permitted with peer agent: {self.temporary_agent_id}')
            try:
                if memory_exist and type == 'DevicePeer':
                    target_preds, attn_weights = self.pipeline.storage.memory_retrieval(self.memory_name, type_func="Peer", verbose=False)
                    
                else:
                    # external peer communicates via socket
                    if type == "ExternalPeer":
                        try:
                            target_preds, attn_weights = self.get_external_peer_message()
                            if target_preds is None:
                                print('[-] Cant get viable components needed for processing request, returning regular probs...')
                                return probs

                        except Exception as e:
                            print(f'[-] No valid in device peer memory id found in database for memory name: {self.memory_name} and error: {e}')
                            return probs
                    else:
                        print('[-] Invalid type..., returning regular probs...')
                        return probs

                if not agreement:
                    probs = self.handle_peer_uncertainty(probs, target_preds, self_attn_weights, attn_weights, input_ids)
                else:
                    try:
                        probs = self.process_peer_request(probs, target_preds, self_attn_weights, input_ids)
    
                    except Exception as e:
                       print(f"[-] Error processing request: {e}, returning regular probs")

            except Exception as e:
                print(f'[-] Error handling request... {e}, returning regular probs')
                self.report_failure(id(self), 'processing', reason=f'{e}')                        

            print(f'[||] Successfully calibrate probs with previous Peer using database!')
            self.save_to_local_peer(self.memory_name, probs)
        else:
            print(f'[-] Connection to peer agent {self.temporary_agent_id} is not permitted, returning regular probs...')

        return probs


    def _calibrate_peer_probs(self, probs, target_preds, self_attn_weights, attn_weights, input_ids, AEL):
        eps = 1e-5
        calibrated = probs.copy()

        try:
            n_classes = probs.shape[1]
        except:
            n_classes = probs.shape[0]

        batch_size = len(target_preds)
        anisotropy = self.pipeline.anisotropy_measurement(attn_weights)  


        if isinstance(attn_weights, (str, np.str_)):
            clean_str = str(attn_weights).replace('[', '').replace(']', '').replace('...', '')
            attn_weights = np.fromstring(clean_str, sep=' ')
        elif isinstance(attn_weights, np.ndarray) and np.issubdtype(attn_weights.dtype, np.character):
            # catches arrays filled with string text
            clean_str = ' '.join(attn_weights.astype(str).flatten()).replace('[', '').replace(']', '').replace('...', '')
            attn_weights = np.fromiter(
                    (x for x in clean_str.split() if x != "..."), dtype=float
                ) 
        else:
            # Ensure standard float array if it was integers or objects
            attn_weights = np.asarray(attn_weights, dtype=float)

        if isinstance(target_preds, (str, np.str_)):
            clean_str = str(target_preds).replace('[', '').replace(']', '').replace('...', '')
            target_preds = np.fromstring(clean_str, sep=" ")

        elif isinstance(target_preds, np.ndarray) and np.issubdtype(target_preds.dtype, np.character):
            # catches arrays filled with string text
            clean_str = ' '.join(target_preds.astype(str).flatten()).replace('[', '').replace(']', '').replace('...', '')
            target_preds = np.fromiter(
                    (x for x in clean_str.split() if x != "..."), dtype=float
                ) 
        else:
            # Ensure standard float array if it was integers or objects
            target_preds = np.asarray(target_preds, dtype=float)

        target_preds = np.asarray(target_preds, dtype=np.float32)
    
        for i in range(batch_size):

            mlp_target = target_preds[i] if target_preds.ndim > 1 and target_preds.shape[0] > i else target_preds
            attn_target = attn_weights[i] if attn_weights.ndim > 1 and attn_weights.shape[0] > i else attn_weights
       
            if self_attn_weights is not None and i < len(attn_weights):
                attn = self_attn_weights[i]

                attn_quality = np.std(attn) if attn.size > 0.0 else AEL
                target_attention_quality = np.std(attn_target) if attn.size > 0.0 else AEL

                try:
                    target_attn_indices = np.argmax(attn_weights)
                    target_mlp_indices = np.argmax(mlp_target)
                except:
                    target_attn_indices = np.argmax(attn_weights, axis=1)
                    target_mlp_indices = np.argmax(mlp_target, axis=1)                    

                consensus = np.allclose(target_mlp_indices, target_attn_indices, atol=eps)

                justified = (1.0 - AEL) + (1.0 - attn_quality) * consensus
                boost = justified * anisotropy + eps

            else:
                attn_quality = 1.0 / (1.0 + np.exp(-self_attn_weights[i]))

                target_attn_indices = np.argmax(attn_weights, axis=1)
                target_prob_indices = np.argmax(probs, axis=1)

                consensus = np.allclose(target_prob_indices, target_attn_indices, atol=eps)

                justified = (1.0 - AEL) * consensus + eps
                boost = (1.0 + justified) * attn_quality + eps

            abstract_error_quality_score = (1.0 - attn_quality) * anisotropy + eps
            self.query_node.peer_trust = (1.0 - abstract_error_quality_score) + boost * justified 

            try:
                calibrated[i, mlp_target] = min(calibrated[i, mlp_target] * (1.5 * (1.0 - abstract_error_quality_score)), 0.95)
            except:
                return calibrated

            calibrated[i] /= calibrated[i].sum()


        return calibrated        
            

    def handle_peer_uncertainty(self, probs, target_preds, self_attn_weights, attn_weights, input_ids):
        try:
            embedded = False
            if isinstance(input_ids, (list, np.ndarray)):
                embedded = True

            if self_attn_weights is None:
                _, _, self_attn_weights = self.pipeline.model2.predict(input_ids, embedded=embedded)  


            if isinstance(attn_weights, tuple):
                attn_weights = attn_weights[0]
            if isinstance(self_attn_weights, tuple):
                self_attn_weights = self_attn_weights[0]

            if isinstance(self_attn_weights, str):
                self_attn_weights = np.array(self_attn_weights) 
                self_attn_weights = self_attn_weights[0]             
             
            if isinstance(attn_weights, str):
                attn_weights = np.array(attn_weights)
        
            batch_similarity = self.pipeline.cosine_similarity(attn_weights, self_attn_weights)

            anisotropy = self.pipeline.anisotropy_measurement(attn_weights)
            AME = self.pipeline.AME_Encoder(attn_weights)
            AMR = 1.0 / (1.0 + np.exp(-AME))

            weighted_quality_rate = (1.0 - AMR) * anisotropy
            
            print(f'[=] Batch similarity: {batch_similarity} With quality rate of attention: {weighted_quality_rate}')
            if weighted_quality_rate > 0.75 and batch_similarity > 0.75:
                return self.process_peer_request(probs, target_preds, attn_weights, input_ids)
            else:
                print('[!] Low uncertainty, normalizing with local agent data...')

                AEL = self.pipeline.confidence_threshold + weighted_quality_rate + (1.0 - AMR) * anisotropy
                calibrated = self._calibrate_peer_probs(probs, target_preds, self_attn_weights, attn_weights, input_ids, AEL)
                return calibrated

        except Exception as e:
            print(f"[= =] Error in uncertainty handling: {e}")
            traceback.print_exc()
            return probs


    def process_peer_request(self, probs, target_preds, attn_weights, input_ids):
        if probs is not None and target_preds is not None and attn_weights is not None and input_ids is not None:
            try:
                response_probs = self.pipeline._calibrate_probs(probs, target_preds, attn_weights, input_ids)
                return response_probs
            except Exception as e:
                print(f"[-] Error in peer request_processing: {e}")
                return probs
        else:
            print('[=] Cannot process peer request due to incomplete Missing samples, returning regular probs!')
            return probs
        
            

    # ============ REQUEST HANDLERS ============
    def get_external_peer_message(self):
        message = self.temporary_message
        if not message:
            print('[-] No viable messages')
            return None, None

        try:
            attn_weights = message.get('attn_weights')
            target_preds = message.get('target_preds')
            if not attn_weights:
                print('|| Invalid format of message, may be a Nonetype object...')
                return None, None
            return attn_weights, target_preds

        except Exception as e:
            print(f'[-] Cant get external peer message: {e}')
            return None, None
         
    
    def _handle_predict_request(self, message, sender_id, method='basic_prediction'):
        if not self._check_trust_level(sender_id, TrustLevel.STANDARD):
            return {'type': 'error', 'message': 'Insufficient trust level'}           
                         
        if method == 'basic_prediction' and self.predict_manager is None:
            text = message.get('text')
            if not text:
                return {'type': self.MSG_TYPES['PREDICT_RESPONSE'], 'error': 'No text provided'}
            
            text = self._sanitize_input(text)
            if not self._check_rate_limit(sender_id):
                return {'type': self.MSG_TYPES['PREDICT_RESPONSE'], 'error': 'Rate limit exceeded'}

            try:
                result = self.pipeline.predict_single(text)
            
                # Log the interaction
                self._log_interaction(sender_id, 'prediction', result['confidence'])
                
                return {
                    'type': self.MSG_TYPES['PREDICT_RESPONSE'],
                    'prediction': result['prediction'],
                    'confidence': result['confidence'],
                    'probabilities': result.get('probabilities', []),
                    'agent_id': id(self)
                }
            except Exception as e:
                print(f"[-] Prediction error: {e}")
                return {'type': self.MSG_TYPES['PREDICT_RESPONSE'], 'error': str(e)}

        else:
            titles = message.get('test_titles')
            label_map = message.get('label_map')
            rules = message.get('rules')
            X = message.get('X')
            y = message.get('y')
            if not titles and label_map and rules:
                return {'type': self.MSG_TYPES['PREDICT_RESPONSE'], 'error': 'No test titles provided'}
            
            titles = self._sanitize_arrays_and_dicts(titles)
            label_map = self._sanitize_arrays_and_dicts(label_map)
            rules = self._sanitize_arrays_and_dicts(rules)
            X = self._sanitize_arrays_and_dicts(X)
            y = self._sanitize_arrays_and_dicts(y)

            if not self._check_rate_limit(sender_id):
                return {'type': self.MSG_TYPES['PREDICT_RESPONSE'], 'error': 'Rate limit exceeded'}

            try:
                result, chosen_label, confidence = self.predict_manager.advanced_prediction_method(titles, label_map, rules, X=X, y=y, show_proba=True, use_transformer=self.pipeline.use_transformer)
            
                # Log the interaction
                self._log_interaction(sender_id, 'prediction', confidence)
                
                return {
                    'type': self.MSG_TYPES['PREDICT_RESPONSE'],
                    'prediction': chosen_label,
                    'confidence': confidence,
                    'probabilities': result,
                    'agent_id': id(self)
                }

            except Exception as e:
                print(f"[-] Advanced prediction error: {e}")
                return {'type': self.MSG_TYPES['PREDICT_RESPONSE'], 'error': str(e)}

    def _handle_memory_sync_request(self, message, sender_id):
        memory_name = message.get('memory_name')
        if not memory_name:
            return {'type': self.MSG_TYPES['MEMORY_SYNC_RESPONSE'], 'error': 'No memory name'}
        
        try:
            # For local peer (database)
            if message.get('peer_type') == 'local':
                memory_data = self.pipeline.storage.load_model_dict(memory_name)
            else:
                # For external peer
                memory_data = self.pipeline.memory.get(memory_name, {})
            
            return {
                'type': self.MSG_TYPES['MEMORY_SYNC_RESPONSE'],
                'memory_name': memory_name,
                'data': memory_data,
                'timestamp': time.time()
            }
        except Exception as e:
            return {'type': self.MSG_TYPES['MEMORY_SYNC_RESPONSE'], 'error': str(e)}



    def _handle_ensemble_vote_request(self, message, sender_id):
        # Handle ensemble vote request from another agent
        text = message.get('text')
        if not text:
            return {'type': self.MSG_TYPES['ENSEMBLE_VOTE_RESPONSE'], 'error': 'No text provided'}
        
        try:
            result = self.pipeline.predict_single(text)
            
            return result['prediction'], {
                'type': self.MSG_TYPES['ENSEMBLE_VOTE_RESPONSE'],
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'agent_id': id(self),
                'trust_score': self.remote_agents.get(sender_id, {}).get('trust', 1.0)
            }
        except Exception as e:
            return None, {'type': self.MSG_TYPES['ENSEMBLE_VOTE_RESPONSE'], 'error': str(e)}
    
    def _handle_failure_report(self, message, sender_id):
        # Handle failure report from another agent

        failed_agent = message.get('failed_agent')
        task_type = message.get('task_type')
        failure_reason = message.get('reason', 'unknown')
        
        # Update trust for the failed agent
        if failed_agent in self.remote_agents:
            self.remote_agents[failed_agent]['failures'] += 1
            self.remote_agents[failed_agent]['trust'] = max(
                0.1, 
                1.0 - (self.remote_agents[failed_agent]['failures'] / 10)
            )
        
        # Log the failure
        self._log_interaction(failed_agent, 'failure', confidence=0, details={
            'task_type': task_type,
            'reason': failure_reason,
            'reported_by': sender_id
        })
        
        return {'type': 'ack', 'status': 'failure_recorded'}


    
    def _handle_trust_update(self, message, sender_id):
        # Handle trust score update
        target_agent = message.get('target_agent')
        new_trust = message.get('trust_score')

        self.query_node.peer_trust = new_trust
        
        if target_agent in self.remote_agents:
            self.remote_agents[target_agent]['trust'] = new_trust
        
        return {'type': 'ack', 'status': 'trust_updated'}


    # ============ REQUEST SENDING METHODS ============       
    def request_prediction_method(self, agent_id, text, timeout=5):
        if agent_id == 'local':
            result = self.pipeline.predict_single(text)
            return result

        if agent_id not in self.remote_agents:
            print(f"Agent {agent_id} not connected")
            return None
        
        sock = self.remote_agents[agent_id]['sock']
        request_id = str(uuid.uuid4())[:8]
            
        message = {
            'type': self.MSG_TYPES['PREDICT_REQUEST'],  # 1
            'text': text,
            'request_id': request_id,  # ← Add request ID!
            'token': self.auth_token,
            'timestamp': time.time()
        }
                
        try:
            sock.settimeout(timeout)
            self._send_message(sock, message)
            response = self._receive_message(sock)
            sock.settimeout(None)
            
            if response and response.get('type') == self.MSG_TYPES['PREDICT_RESPONSE']:
                return response
            return None
        except Exception as e:
            print(f"Request failed for {agent_id}: {e}")
            return None
    
    def request_ensemble_vote(self, agent_id, text, timeout=5):
        if agent_id not in self.remote_agents:
            return None
        
        sock = self.remote_agents[agent_id]['sock']
        message = {
            'type': self.MSG_TYPES['ENSEMBLE_VOTE_REQUEST'],
            'text': text
        }
        
        try:
            sock.settimeout(timeout)
            self._send_message(sock, message)
            response = self._receive_message(sock)
            sock.settimeout(None)
            
            if response and response.get('type') == self.MSG_TYPES['ENSEMBLE_VOTE_RESPONSE']:
                return response['prediction'], response['text']
            return None, None
        except Exception as e:
            print(f"Vote request failed: {e}")
            return None, None
    
    def sync_memory_with_agent(self, agent_id, memory_name, timeout=10):
        if agent_id not in self.remote_agents:
            return None
        
        sock = self.remote_agents[agent_id]['sock']
        message = {
            'type': self.MSG_TYPES['MEMORY_SYNC_REQUEST'],
            'memory_name': memory_name,
            'peer_type': 'external'
        }
        
        try:
            sock.settimeout(timeout)
            self._send_message(sock, message)
            response = self._receive_message(sock)
            sock.settimeout(None)
            
            if response and response.get('type') == self.MSG_TYPES['MEMORY_SYNC_RESPONSE']:
                return response.get('data', {})
            return None
        except Exception as e:
            print(f"Memory sync failed: {e}")
            return None
    
    def report_failure(self, agent_id, task_type, reason="unknown"):
        report = {
            'type': self.MSG_TYPES['FAILURE_REPORT'],
            'failed_agent': agent_id,
            'task_type': task_type,
            'reason': reason,
            'timestamp': time.time()
        }
        
        # Send to all other agents
        for other_id, agent_info in list(self.remote_agents.items()):
            if other_id != agent_id:
                self._send_message(agent_info['sock'], report)
    
    def broadcast_ping(self):
        # Check which agents are still alive
        alive_agents = []
        for agent_id, agent_info in list(self.remote_agents.items()):
            try:
                sock = agent_info['sock']
                self._send_message(sock, {'type': self.MSG_TYPES['PING']})
                response = self._receive_message(sock)
                if response and response.get('type') == self.MSG_TYPES['PONG']:
                    alive_agents.append(agent_id)
                    agent_info['last_seen'] = datetime.now()
                else:
                    # Agent dead, remove
                    print(f'[===] Removing Agent id: {agent_id}')                    
                    del self.remote_agents[agent_id]
            except:
                print(f'[===] Removing Agent id: {agent_id}')                
                del self.remote_agents[agent_id]
        
        return alive_agents
    
    # ============ LOCAL PEER (DATABASE) METHODS ============
    
    def sync_with_local_peer(self, memory_name):
        try:
            memory_exist = self.pipeline.storage.memory_exists(self.memory_name, type='Peer')
            if memory_exist:
                memory_data = self.pipeline.storage.memory_retrieval(self.pipeline.memory_name, type_func="Peer", verbose=False)
                print(f'|| Retrieved memory, Samples: {len(memory_data)}')

            try:
                if memory_exist and memory_data:
                    # Merge with current memory
                    print('[=] Syncing with local peer memory data...')
                    try:
                        for key, value in memory_data.items():
                            if key not in self.pipeline.memory:
                                self.pipeline.memory[key] = value
                    except Exception as e:
                        print(f'|| Using sync memory function because of {e} problem in regular syncing using value in items.')
                        agent_id = self.temporary_agent_id
                        self.sync_memory_with_agent(agent_id, memory_name)

                    print(f"✅ Synced with local peer: {len(memory_data)} memories")
            except:
                print(f'[-] Failed converting and syncing with peer, but memory exist is assured.')
            memory_exist = True
            return memory_exist

        except Exception as e:
            print(f"Local peer sync failed: {e}")
            memory_exist = False

        print(f'|| Memory Exist: {memory_exist}')
        
        return memory_exist
    
    def save_to_local_peer(self, memory_name, data):
        try:
            self.pipeline.storage.save_model_dict(memory_name, data)
            print(f"✅ Saved local peer presence: {memory_name}")
            return True
        except Exception as e:
            print(f"Save to local peer failed: {e}")
            return False
    
    # ============ UTILITY METHODS ============
    
    def _log_interaction(self, agent_id, interaction_type, confidence, details=None):
        if agent_id not in self.agent_comm_log:
            self.agent_comm_log[agent_id] = []
        
        self.agent_comm_log[agent_id].append({
            'timestamp': datetime.now(),
            'type': interaction_type,
            'confidence': confidence,
            'details': details
        })
    
    def get_agent_status(self):
        status = {}
        for agent_id, info in list(self.remote_agents.items()):
            status[agent_id] = {
                'connected': True,
                'trust': info['trust'],
                'failures': info['failures'],
                'last_seen': info['last_seen'].isoformat(),
                'host': info['host'],
                'port': info['port']
            }
        return status
    
    def get_communication_log(self, agent_id=None, limit=50):
        # Get communication log for an agent
        if agent_id:
            return self.agent_comm_log.get(agent_id, [])[-limit:]
        
        # Return all logs
        return self.agent_comm_log
    
    def print_network_status(self):
        print("\n" + "="*60)
        print("🤖 == AGENT NETWORK STATUS ==")
        print("="*60)
        print(f"[=] Local Agent: {self.memory_name}")
        print(f"[=] Port: {self.port}")
        print(f"[=] Connected Agents: {len(self.remote_agents)}")

        agent_id = self.temporary_agent_id
        comm_log = self.get_communication_log(agent_id)
        
        for agent_id, info in self.remote_agents.items():
            print(f"\n  📡 {agent_id}")
            print(f"     Trust: {info['trust']:.2f}")
            print(f"     Failures: {info['failures']}")
            print(f"     Last seen: {info['last_seen'].strftime('%H:%M:%S')}")
            print(f"     Agent Communication Log: {comm_log}")
        
        print("="*60)

# The QueryNode class manages the connection and interaction with other nodes (agents) in the network. It handles node identification, agreement evaluation, safety checks, and maintains a memory of connected nodes. 
# The class allows for flexible interactions while ensuring the safety and integrity of the Master node.
class QueryNode:
    def __init__(self, pipeline, memory_name, storage):
        self.master_node = pipeline
        self.memory_name = memory_name
        self.storage = storage
        self.agreement = False

        if not self.storage.memory_exists(self.memory_name, type='Node'):
            print(f"|| Creating new memory for Nodes population: {memory_name}!")
            self.nodes = {}
        else:
            print(f'|| Found Matched Memory for Nodes : {memory_name}!')
            self.nodes = self.storage.memory_retrieval(self.memory_name, type_func='Node', verbose=True)

        self.master_nodes_id = 0
        self.safety_check_value = 0.0
        self.node_id = 0
        self.peer_trust = 1.0

        self.permission = False

    def _add_node(self, node):
        node_id = id(node)

        self.nodes[node_id] = node
        print(f"✅ Node {node_id} added to QueryNode")

        return node_id

    def _save_node_memory(self, node):
        try:
            node_id = id(node)
            self.master_node.storage.save_nodes_dict(self.memory_name, self.nodes, node_id, model_type='Node')

            print(f"[💾] Node {node_id} memory saved to storage!")
            return True
        except Exception as e:
            print(f"[-] Error saving node memory: {e}")
            return False


    def _evaluate_node_agreement(self, node):
        print(f"[=] Evaluating node {id(node)} || agreement: {self.agreement} || Master Node memory: {self.memory_name}")
        self.agreement_threshold = (self.master_node.confidence_threshold + self.master_node.final_conf_score * self.master_node.temperature)

        if self.agreement or self.agreement_threshold > self.master_node.confidence_threshold:
            print(f"[✅] Node {id(node)} is in agreement with the Master node")
            return True

        print(f"[-] Node {id(node)} is NOT in agreement with the Master node")
        return False


    def _connect_with_node(self, node):
        self.agreement = self.master_node.agreement

        if not self._identify_node(node):
            node_id = self._add_node(node)

        node_id = id(node)
        agreement = self._evaluate_node_agreement(node)
        safety = self._node_safety_check(node)

        # stable Node is established if either agreement is met or safety check is passed, allowing for some flexibility in interactions while still protecting the Master node from harmful interactions
        if safety or agreement:
            print(f"[🔗] Node {node_id} successfully connected to the Master node")
            self.permission = True
        else:
            print(f"[⚠️] Node {node_id} connection failed due to Disagreement")
            self.permission = False

        print('== Connection Evaluation Summary ==')
        print(f'[=] Node {node_id}')
        print(f'[=] agreement: {agreement}')
        print(f'[=] safety: {safety} || permission: {self.permission}')
 
        return self.permission

    def _connect_with_peer(self, node):
        self.agreement = self.master_node.agreement

        if not self._identify_node(node):
            node_id = self._add_node(node)

        node_id = id(node)
        agreement = self._evaluate_node_agreement(node)
        safety = self._node_safety_check(node)

        # peer agreement is optional to allow for more flexible interactions, but safety check is still enforced to protect the Master node from harmful interactions
        if safety or agreement:
            print(f"[🔗] Peer with ID: {node_id} successfully connected to the Master node")
            self.permission = True
        else:
            print(f"[⚠️] Peer with ID: {node_id} connection failed due to Disagreement")
            self.permission = False
 
        return self.permission        

    def _identify_node(self, node):
        eps = 1e-5
        print(f"[||] Identifying node {id(node)} with Master node memory: {self.memory_name}")
        identified_nodes = [(nid, n) for nid, n in self.nodes.items() if n == node]
        if identified_nodes:
            for node in identified_nodes:
                print(f"✅ Node {id(node)} is already identified with the Master node")
                self.safety_check_value = (self.master_node.final_conf_score + self.master_node.temperature) + eps
                return True
        else:
            print(f"[-] Node {id(node)} is NOT identified with the Master node")
            self.safety_check_value = (1.0 - self.master_node.final_conf_score + self.master_node.temperature) + eps
            return False


    def _node_safety_check(self, node):
        print(f"[🛡️] Performing safety check for node {id(node)} with safety value: {self.safety_check_value}")
        if self.safety_check_value > self.master_node.confidence_threshold:
            print(f"✅ Node {id(node)} passed the safety check")
            return True
        else:
            print(f"[-] Node {id(node)} failed the safety check")
            if self.safety_check_value < (self.master_node.confidence_threshold / 2):
                print(f"[⚠️] Node {id(node)} is considered useless and will be removed")
                removed = self._remove_node(node)
                return removed

            return False

    def _remove_node(self, node):
        node_id = id(node)
        if node in self.nodes:
            del self.nodes[node_id]
            print(f"[🗑️] Node {node_id} removed from Nodes population")
            return True
        else:
            print(f"[-] Node {node_id} not found in Nodes population")
            return False

    def _node_activation(self, Node):
        try:
            if self.permission:
                print(f"🚀 Node {id(Node)} is now active with the Master node")
                return True
            else:
                print(f"[-] Node {id(Node)} cannot be activated due to lack of permission")
                return False
        except Exception as e:
            print(f"[-] Error during node activation: {e}")
            return False

    def _identify_peer_trust(self, peer):
        print(f"[=] Identifying peer node {id(peer)} trustworthiness with Master node memory: {self.memory_name}")
        if self.peer_trust > self.master_node.confidence_threshold:
            print(f"[✅] Peer node {id(peer)} is identified as trustworthy with trust score: {self.peer_trust:.2f}")
            return True
        else:
            print(f"[-] Peer node {id(peer)} is NOT identified as trustworthy with trust score: {self.peer_trust:.2f}")
            return False

    def _establish_peer_nodes(self, peer):
        print(f"[=] Establishing peer node connection with peer with memory: {self.memory_name}")
        if self._connect_with_peer(peer) and self._identify_peer_trust(peer):
            print(f"[✅] Peer node {id(peer)} is now connected and can interact with the Master node")
        else:
            print(f"[-] Peer node {id(peer)} cannot interact with the Master node due to failed agreement")

        activation = self._node_activation(peer)
        saved = self._save_node_memory(peer)
        return activation

    def _establish_node_connection(self, node):
        if self._connect_with_node(node):
            print(f"[✅] Node {id(node)} is now connected and can interact with the Master node")
        else:
            print(f"[-] Node {id(node)} cannot interact with the Master node due to failed agreement")

        activation = self._node_activation(node)
        saved = self._save_node_memory(node)
        return activation




# The AutoBatcherAutomation class manages the batching of incoming prediction requests to optimize processing efficiency. 
# It collects requests over a short time window or until a maximum batch size is reached, then processes them together through the pipeline. This allows for improved throughput while still providing timely responses to individual requests.
class AutoBatcherAutomation:
    def __init__(self, pipeline, max_batch_size=32, max_wait_ms=50):
        self.pipeline       = pipeline
        self.max_batch_size = max_batch_size
        self.max_wait_ms    = max_wait_ms

        self.request_queue  = deque()
        self.processing     = False
        self.results        = {}
        self.result_events  = {}   # per-request Event 
        self.next_id        = 0

        self._state_lock = threading.Lock()   # guards processing flag + next_id

    def add_request(self, text, callback=None):
        with self._state_lock:
            request_id    = self.next_id
            self.next_id += 1

            event = threading.Event()
            self.result_events[request_id] = event

            self.request_queue.append({
                'id'       : request_id,
                'text'     : text,
                'callback' : callback,
                'timestamp': time.time()
            })

            # check-and-set happens atomically under the same lock
            should_start = not self.processing
            if should_start:
                self.processing = True

        if should_start:
            self._start_processing()

        result = self.get_result(request_id)
        self.cleanup_stale()

        return request_id

    def _start_processing(self):
        thread = threading.Thread(target=self._process_batches, daemon=True)
        thread.start()

    def _process_batches(self):
        try:
            while True:
                with self._state_lock:
                    if not self.request_queue:
                        break

                time.sleep(self.max_wait_ms / 1000)

                batch = []
                with self._state_lock:
                    while self.request_queue and len(batch) < self.max_batch_size:
                        batch.append(self.request_queue.popleft())

                if batch:
                    # never let one bad batch kill the worker permanently here.
                    try:
                        self._process_batch(batch)
                    except Exception as e:
                        logger.error(f'[!] Batch processing failed: {e}')
                        # deliver the failure to every waiter in this batch
                        # instead of leaving them hanging forever
                        for req in batch:
                            self._deliver_result(req, None, error=e)
        finally:
            # always reset processing, even if something above
            # raised unexpectedly
            with self._state_lock:
                self.processing = False

            # need catch the case where requests arrived after the
            # while-loop's last empty check but before processing=False landed
            with self._state_lock:
                still_pending = bool(self.request_queue)
                if still_pending and not self.processing:
                    self.processing = True
                    restart = True
                else:
                    restart = False
            if restart:
                self._start_processing()

    def _process_batch(self, batch):
        texts   = [req['text'] for req in batch]
        results = self.pipeline.prediction_batch(texts)

        for i, req in enumerate(batch):
            result = results[i] if i < len(results) else None
            self._deliver_result(req, result)

    def _deliver_result(self, req, result, error=None):
        """Single delivery path — callback or stored result, always signals."""
        if req['callback']:
            try:
                req['callback'](result if error is None else error)
            except Exception as cb_err:
                logger.error(f'[!] Callback failed for request {req["id"]}: {cb_err}')
        else:
            with self._state_lock:
                self.results[req['id']] = result if error is None else None

        # signal the waiting event 
        event = self.result_events.get(req['id'])
        if event:
            event.set()

    def get_result(self, request_id, timeout=5):
        event = self.result_events.get(request_id)
        if event is None:
            return None

        # blocks efficiently 
        signaled = event.wait(timeout=timeout)

        with self._state_lock:
            # always need clean up
            result = self.results.pop(request_id, None)
            self.result_events.pop(request_id, None)

        return result if signaled else None

    def cleanup_stale(self, max_age_seconds=300):
        """
        periodic sweep for requests that were never collected
        via get_result (e.g. caller crashed or forgot to call it).
        Call this periodically from a health check loop.
        """
        with self._state_lock:
            now = time.time()
            stale_ids = [
                rid for rid, event in self.result_events.items()
                if event.is_set()   # already delivered but never collected
            ]
            for rid in stale_ids:
                self.results.pop(rid, None)
                self.result_events.pop(rid, None)
            if stale_ids:
                logger.info(f'[=] Cleaned up {len(stale_ids)} stale results')


# The IntegratedPipeline class serves as the central component that integrates all the different modules and functionalities of the system. 
# It manages the overall workflow, including data processing, model training, prediction, memory management, and interactions with other agents.
class IntegratedPipeline:
    def __init__(self, memory_name='agent_memory', 
                  use_async=False, agent_port=None, 
                  ssl_cert_file=None, ssl_key_file=None, 
                  ssl_context=None, client_ssl_context=None,
                  secret_key=None, 
                  shared_auth_token=None, predict_manager=None,
                  bind_host=None, security_level=None):
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

        self.mlp_training_epochs = 2000
        self.transformer_training_epochs = 100

        # Main component setup
        self.standard_scaler = StandardScaler()
        self.tfidf = TfidfVectorizer(max_features=70)
        # LSTM __init__ setup
        self.network_model = LSTMNetwork(self, input_size=self.input_size, hidden_size=self.hidden, output_size=self.output_size)
        self.scrapper_model = LSTMEngine(self, self.network_model, dropout=self.dropout_rate, n_samples=50)        
        self.lstm_engine = None
        self.lstm_n_samples = 0

        self.storage = ModelStorage(self, memory_name, db_path='activity_log.db')
        self.distribution = AgentDistributedInference(self, self.storage, memory_name, port=self.port, 
                                                         use_async=use_async, secret_key=self.secret_key, 
                                                         ssl_cert_file=ssl_cert_file, ssl_key_file=ssl_key_file, 
                                                         ssl_context=self.ssl_context, client_ssl_context=self.client_ssl_context,
                                                         shared_auth_token=self.shared_auth_token, predict_manager=self.manager,
                                                         bind_host=bind_host, security_level=security_level)        
        self.ensemble = WeightedEnsemblePredictor(self, self.distribution, memory_name)        
        self.session_automation = CrossSessionAutomation(self)
        self.batcher = AutoBatcherAutomation(self)
        self.query_node = QueryNode(self, memory_name, self.storage)
        self.accurate_cache_lookup = AccurateAnswerCache(self, similarity_threshold=0.85, max_size=self.max_size)

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

        self.mlp = MLP()
        self.focused_mlp = MLP()

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
        self.froze_learning = False 
        self._cache_save_count = None
        self._prob_save_count = None

        self.temperature = 1.0
        self.transformer_lr = 0.1
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
        self.tfidf.fit_transform(text).toarray()
        vocab_size = len(self.tfidf.get_feature_names_out())
        self.vocab_size = vocab_size
        

    def initialize_model_encoding(self, X, y_raw):
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
        automatic_change = self.automatic_parameterization(input_dim, num_classes)

        automatic_change = self.automatic_parameterization(input_dim, num_classes)

        layer1= Dense(X, input_dim, automatic_change, activation="relu")
        layer2 = Dense(X, automatic_change, num_classes, activation='relu')
        
        abundant_layer = int(automatic_change * 10)
        first_feed_layer = Dense(X, input_dim, abundant_layer, activation="relu")
        sec_feed_layer = Dense(X, abundant_layer, num_classes, activation="relu")

        self.model3 = MLP() 

        self.model3.add(layer1)
        self.model3.add(layer2)

        self.model3.feed_add(first_feed_layer)   
        self.model3.feed_add(sec_feed_layer)                

        
    def automatic_parameterization(self, input_size, num_classes):
        parameters = input_size * num_classes / 2
        parameters = int(parameters)
        return parameters


    def automatic_dense_layer(self, X, input_dim, num_classes):
        vocab_size = self.vocab_size 

        automatic_change = self.automatic_parameterization(input_dim, num_classes)

        layer1= Dense(X, input_dim, automatic_change, activation="relu")
        layer2 = Dense(X, automatic_change, num_classes, activation="relu")

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
        texts   = [d[0] for d in datasets]
        intents = [d[1] for d in datasets]

        intent_to_id = {intent: i for i, intent in enumerate(sorted(set(intents)))}
        batch_classes = len(intent_to_id)

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

        self.model2 = Transformer(
            vocab_size=len(self.vocab),
            d_model=32,
            n_heads=4,
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
        # Broadcast message to all connected peers.
        if not self.distribution:
            raise RuntimeError("Distributed inference not initialized")
        
        return await asyncio.to_thread(
            self.distribution.broadcast,
            message_type, payload, timeout=10
        )
    
    def get_network_status(self):
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
        self.memory['MW'] = X2, output # MLP Weight
        try:
            print('🚀 Memory Prediction Added!')
            self.storage.save_model_dict(memory_name, self.memory, type='Pipeline', model_type='prediction')
        except Exception as e:
            print(f'[!] Cant save model memory: {e}')


    def modular_probability_saving(self, X, X2, prob):
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
                print('[!] Cant get number of classes from transformer and MLP model!')
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
                print('[!] Cant get number of classes from transformer and MLP model!')
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
                print(f'[=] MLP Memory length: {len(cache_mlp_memory)}')
                print(f'[=] Transformer Memory length: {len(cache_trans_memory)}')

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
            
            # Prepare MLP input
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
        try:
            id_req = self.batcher.add_request(text, callback)
            result = self.batcher.get_result(id_req, timeout=10)
            self.batcher.cleanup_stale()
            return id_req

        except Exception as e:
            print(f'[=] error in automatic batcher: {e}')
            return None
    

    def predict_single(self, text):
        result = [None]
        
        def callback(r):
            result[0] = r
        
        self.predict_async(text, callback)
        
        # Wait for result
        while result[0] is None:
            time.sleep(0.001)
        
        return result[0]
 

    def _batch_hybrid_prediction(self, batch_input_ids, batch_X_raw, y_true, embedded=True):
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
                fresh_probs = self.model3.forward(fresh_X_raw)
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
                    probs_i        = fresh_probs[i].copy()
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


    def hybrid_prediction(self, rules, input_ids, dataset, X=None, y=None, use_embedded=True):
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
                        time.sleep(10)

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
                        time.sleep(10)

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
        if isinstance(X, str) or isinstance(X[0], str):
            self.initialize_fitting(X)            
            X_tfidf = self.tfidf.transform(X).toarray() 

        logits = self.mlp.prediction(X_tfidf)
        return logits

   
    def predict_proba(self, input_ids, X, type='Hybrid', embedded=False):
        eps = 1e-5
        probs_memory = self.model_probability_gate(input_ids, X)
        if isinstance(self.storage.id_history, list) or isinstance(self.storage.id_history, np.ndarray) and not self.agent_id in self.storage.id_history:
            self.storage.id_history.append(self.agent_id)
            id_history = self.storage.id_history
        else:
            self.temporary_id.append(self.agent_id)
            id_history = self.temporary_id

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
                    probs[i, target] = min(probs[i, target] * 1.2, 0.95)
                    probs[i] /= probs[i].sum()

                    
            if not agreement and probs_memory is not None:
                self.storage.save_peer_needs_dict(self.memory_name, probs_memory, mlp_pred, id_history)   
            else:
                self.storage.save_peer_needs_dict(self.memory_name, probs, mlp_pred, id_history)

            self.modular_probability_saving(input_ids, X, probs)
            print('🚀 Memory Added!')
            return probs
        else:
            print(f'[=] MLP Based classification method. Transformer usage permission: {self.use_transformer}')
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
            print(f'[!] {len(zero_rows)} zero rows ({zero_ratio:.0%}), refitting TF-IDF on current batch')
            self.tfidf.fit(texts)
            X_features = self.tfidf.transform(texts).toarray()
            
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
                calibrated[i, mlp_target_int] = min(
                    calibrated[i, mlp_target_int] * (1.5 * (1.0 - abstract_score)), 0.95
                )

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


    def utility_MLP_set(self, X, y):
        if not self.autonomous:
            try:
                joblib.dump(self.mlp, 'analyzer_model.pkl')
                joblib.dump(self.model2, 'analyzer_agent.pkl')
                print("🎉 Model trained and saved!")            
            except Exception as e:
                print(f'|| Failed to joblib dump file! : {e}, User Manual filepath suggestion needed...')

                permission = input('|| Insert Filepath? [Y/N]: ')
                if permission == 'Y':
                    suggested_path = input('|| Filepath suggestion: ')
                    if suggested_path:
                        self.safe_pickle_save_with_feedback(self.mlp, suggested_path)
                        self.safe_pickle_save_with_feedback(self.model2, suggested_path)
                        print(" || Model saved!")                     
                    else:
                        print('|| Failed to dump Your model! ')
                        pass
                else:
                    print('|| Failed to dump Your model! ')
                    pass
         
        else:
            pass


    def auto_generate_labels_from_texts(self, rules, texts):
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


    def mlp_training_features(self, rules, dataset):
        print("\n[🔄] Preparing MLP data from dataset format")

        if isinstance(dataset[0], tuple) and len(dataset[0]) == 2:
            print('[=] Dataset Type 1: [(features, label), ...]')
            features_list = []
            labels_list   = []
            for item in dataset:
                features, label = item
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

        unique_labels = sorted(set(y_raw))
        label_to_idx  = {l: i for i, l in enumerate(unique_labels)}
        y_indices     = np.array([label_to_idx[l] for l in y_raw])

        n_classes = len(unique_labels)
        y_onehot  = np.zeros((len(y_indices), n_classes))
        y_onehot[np.arange(len(y_indices)), y_indices] = 1

        if isinstance(X_mlp, np.ndarray) and X_mlp.ndim > 1:
            input_dim = X_mlp.shape[1]
        elif isinstance(X_mlp, np.ndarray):
            input_dim = 1   # 1D array — single feature per sample
        else:
            input_dim = len(X_mlp[0]) if len(X_mlp) > 0 else 0

        print(f"\n✅ MLP data ready:")
        print(f"[=] X shape: {X_mlp.shape if isinstance(X_mlp, np.ndarray) else len(X_mlp)}")
        print(f"[=] input_dim: {input_dim}")
        print(f"[=] y shape: {y_onehot.shape}")
        print(f"[=] Classes: {label_to_idx}")

        return X_mlp, y_onehot, n_classes, input_dim


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
                    # RAGGED — pad to uniform length 
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
            X = np.atleast_2d(X).T  # Returns shape back to (12, 1) safely

        # Handle cropping for a 2D array
        if X.ndim == 2:
            rows, cols = X.shape
            new_rows = rows - 1 if (rows % 2 != 0 and rows > 1) else rows
            new_cols = cols - 1 if (cols % 2 != 0 and cols > 1) else cols
            
            X = X[:new_rows, :new_cols]

        return X 

    def AME_Encoder(self, x):
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

            if X.shape[1] == 1:
                gradient = np.gradient(X, axis=0)  # Calculate vertically instead of horizontally
            else:
                gradient = np.gradient(X, axis=-1) # Calculate horizontally

            grad_energy = np.mean(np.linalg.norm(gradient, axis=-1))       
            X_mag = np.mean(np.linalg.norm(X, axis=-1))

            AME = np.log1p(X_mag) * np.log1p(grad_energy) 

            if AME <= eps:
                AME = (1.0 - self.confidence_threshold) + eps
            
        except Exception as e:
            print(f'[!] Cant calculate AME from samples due to: {e}, using normalized value...')
            AME = (1.0 - self.confidence_threshold) + eps

        return AME

    def feature_generation(self, rules, dataset):
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
        AccurateAnswerCache before reshaping.
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
        Pull verified-correct entries from AccurateAnswerCache to use as
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

    def lstm_setup_inference(self, raw_X, raw_Y):
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

        n_train = int(0.8 * len(X)) # 80% of the data training is used for training
        X_val   = X[n_train:]
        Y_val   = Y[n_train:]
        n_display = len(X_val)
        if n_display > 3:
            n_display = 3

        print('[= FIT =] Fitting Short Term Memory...')
        n_samples = scrapper_engine.lstm_optimal_samples(scrapper_engine, X_val[0])
        self.lstm_n_samples = n_samples

        label_bins = scrapper_engine.derive_bins_from_data(
            Y_val.ravel(),
            n_bins=4,
            labels=["Low", "Moderate", "High", "Extreme"]
        )
        # {"Low": (5.0, 42.3), "Moderate": (42.3, 68.1),
        #  "High": (68.1, 98.7), "Extreme": (98.7, 219.0)}

        # build and calibrate engine
        self.lstm_engine = LSTMEngine(self, self.network_model, 
                                       dropout=self.dropout_rate, n_samples=self.lstm_n_samples)
        
        engine = self.lstm_engine

        self.lstm_engine.fit_stm(X, Y)
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
            actual = Y_val[j, -1, 0]
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
        eps = 1e-5
        final_conf = self.final_conf_score
        confidence_threshold = self.confidence_threshold
        unsuitable_training = False

        probs = self.model_memory_gate(input_ids, x)

        anisotropy = self.anisotropy_measurement(input_ids)
        AME = self.AME_Encoder(input_ids)
        AMR = 1.0 / (1.0 + np.exp(-AME)) 

        LMR = 1.0 / (1.0 + np.exp(-AMR)) # logistic modelling rate
        ALR = 1.0 / (1.0 + np.exp(-anisotropy)) # anisotropic logistic rate
        AAC = (1.0 - ALR) + (1.0 - LMR) + eps # anisotropic abstract coefficient

        self.confidence_threshold = anisotropy * AAC 
        if np.isnan(self.confidence_threshold) or np.isinf(self.confidence_threshold):
            self.confidence_threshold = 0.45

        print(f'[||] Confidence threshold set to: {self.confidence_threshold} || Final Confidence Score: {final_conf}')

        # training is wasteful if the model processes little abstraction divergence without necessary context on a similar environment
        # AMR is guaranteed to give sufficient ratio on how modelling error error could be sufficient enough to guarantee the model successful training 
        # (not too high that it shows unstability, not too low that it shows rigidity), high anisotropy correlates to a much complex non linearity that the model will have a hard time adjusting
        # higher LMR than AAC means the model is likely to be in a regime where training could lead to overfitting or divergence due to insufficient modelling capacity relative to the complexity of the data, especially if the confidence score is also low, indicating that the model is not currently confident in its predictions and may not benefit from further training on this data.
        unsuitable_tolerance = probs is not None or LMR > AAC 
        unsuitable_conditions = anisotropy > 0.85 or final_conf > confidence_threshold or self.froze_learning
        unsuitable_peer_request = probs is not None and self.peer_assistance_threshold > self.confidence_threshold

        if unsuitable_tolerance or unsuitable_conditions or unsuitable_peer_request:
            print(f'[==] Unsuitable training condition detected! Tolerance: {unsuitable_tolerance} || Unsuitable Conditions: {unsuitable_conditions}')
            print(f'[==] Peer assistance condition: {unsuitable_peer_request} || Peer assistance threshold: {self.peer_assistance_threshold}')
            unsuitable_training = True

        print('== Training Condition evaluation == ')
        print(f'[==] Unsuitable training condition Evaluation || Unsuitable Tolerance: {unsuitable_tolerance} || Unsuitable Conditions: {unsuitable_conditions} || Unsuitable Peer Assistance: {unsuitable_peer_request}')
        print('[==] Final Decision on Training: ' + ('Unsuitable' if unsuitable_training else 'Suitable') + ' for training.')

        return unsuitable_training

    def sequence_encoding(self, datasets=None, label_map=None, max_len=32):
        input_sequences = []

        if datasets:
            for item in datasets:
                if not self.model2:
                    intents = [d[1] for d in datasets]
                    intent_to_id = {intent:i for i, intent in enumerate(sorted(set(intents)))}
                    num_classes = self._get_num_classes(label_map=label_map)                
                    self.model2 = Transformer(
                        vocab_size=len(self.vocab),
                        d_model=32,
                        n_heads=4,
                        num_classes=num_classes
                    ) 
                    
                self._sync_vocab_to_embedding()
                text = item[0] if isinstance(item, tuple) else item

                token_ids = self.encode(text, self.vocab, max_len=max_len)
                token_embs = self.model2.token_embedding[token_ids]         # (max_len, d_model)
                pos_embs = self.model2.pos_embedding[:max_len]              # (max_len, d_model)

                sequence_input = token_embs + pos_embs
                input_sequences.append(sequence_input)
            return np.stack(input_sequences)  # shape: (batch, max_len, d_model)
        else:
            raise Warning('[!] Dataset is None! make sure you provide a dataset or create it Automatically!')
            
            return None

    def transformer_pooled_features(self, sequence_inputs):
        # mean/max/std pooling over sequence dimension
        mean_pool = np.mean(sequence_inputs, axis=1)
        max_pool = np.max(sequence_inputs, axis=1)
        std_pool = np.std(sequence_inputs, axis=1)
        return np.concatenate([mean_pool, max_pool, std_pool], axis=-1)

    def _features_to_sequence(self, X_provided, d_model=None, min_seq_len=2):
        """
        Convert a flat (n_samples, n_features) array into a pseudo-sequence
        (n_samples, T, d_model) that the Transformer can process with
        embedded=True.

        Strategy: split the feature dimension into d_model-sized windows.
        Each window becomes one timestep. This lets the Transformer's
        attention mechanism find relationships BETWEEN feature windows,
        not just within them — genuinely useful when X_provided has
        structured groups of features (e.g. sensor readings, TF-IDF blocks).

        Args:
            X_provided: (n_samples, n_features) array
            d_model: target feature width per timestep. Defaults to
                    self.model2.d_model if a transformer already exists,
                    otherwise a sensible default based on n_features.
            min_seq_len: minimum number of timesteps to produce — if
                        n_features is small, pad timesteps rather than
                        collapsing to T=1 (attention needs T>=2 to be useful)
        """
        if isinstance(X_provided, (str, np.str_)):
            clean_str = str(X_provided).replace('[', '').replace(']', '').replace('...', '').strip()
            X_provided = np.fromstring(clean_str, sep=' ')

        if isinstance(X_provided, np.ndarray) and np.issubdtype(X_provided.dtype, np.character):
            # catches arrays filled with string text
            clean_str = ' '.join(X_provided.astype(str).flatten()).replace('[', '').replace(']', '')
            skip_values = {"...", "NaN", "null"}
            X_provided = np.fromiter((v for v in clean_str.split() if v not in skip_values), dtype=float)

        X_provided = np.asarray(X_provided, dtype=np.float64)
        if X_provided.ndim == 1:
            X_provided = X_provided.reshape(1, -1)

        n_samples, n_features = X_provided.shape

        # resolve d_model — reuse existing transformer's width if available
        if d_model is None:
            if hasattr(self, 'model2') and self.model2 is not None:
                d_model = self.model2.d_model
            else:
                # pick a d_model that gives a reasonable T for this feature count
                # aim for T in range [4, 16] as a sane default sequence length
                d_model = max(1, n_features // 8)

        # compute how many timesteps this produces
        T = int(np.ceil(n_features / d_model))

        # ensure minimum sequence length — pad with extra empty timesteps
        # if the feature count is too small to naturally produce enough T
        if T < min_seq_len:
            T = min_seq_len

        padded_width = T * d_model
        pad_amount   = padded_width - n_features

        if pad_amount > 0:
            X_padded = np.pad(
                X_provided, ((0, 0), (0, pad_amount)),
                mode='constant', constant_values=0.0
            )
        else:
            X_padded = X_provided[:, :padded_width]

        sequence_inputs = X_padded.reshape(n_samples, T, d_model)

        print(f'[=] Converted X_provided {X_provided.shape} → '
            f'sequence_inputs {sequence_inputs.shape} '
            f'(d_model={d_model}, T={T}, padded {pad_amount} values)')

        return sequence_inputs

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

    def transformer_utilities(self, X_provided= None, X_raw=None, y_true=None, rules=None, datasets=None, label_map=None, batch_size=2, min_signal=1e-3):
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
            skip_values = {"...", "NaN", "null"}
            X_raw = np.fromiter((v for v in clean_str.split() if v not in skip_values), dtype=float)
        
        if datasets is not None:       
            self.text_encoder(datasets)

        if X_provided is None:
            sequence_inputs = self.sequence_encoding(datasets, label_map=label_map)
        else:
            sequence_inputs = self._features_to_sequence(X_provided)

        unsuitable_training = self.training_necessary_condition(sequence_inputs, X_raw)
        lr = self.model2.transformer_lr if self.model2 else self.transformer_lr

        if not unsuitable_training:
            print(f'🚀 Training Transformer with {len(sequence_inputs)} Samples: ')

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
                print('[+] Dynamic Backward for Transformer Initiated')
                mode = 'dynamic_backward'
            else:
                print('[=] Fixed Backward for Transformer initiated')
                mode = 'fixed_backward'

            if self.use_transformer:
                self.model2.train(sequence_inputs, y_true, epochs=self.transformer_training_epochs, mode=mode, lr=lr, embedded=True, batch_size=batch_size)
                
            if X_provided is None and datasets is not None and rules is not None:
                X_raw_generation, y, n_classes, input_dim = self.mlp_training_features(rules, datasets)
            else:
                X_raw_generation = X_provided
                y          = y_true
                n_classes  = y_true.shape[1] if y_true.ndim > 1 else len(np.unique(y_true))
                input_dim  = X_provided.shape[1] if X_provided.ndim > 1 else 1

            X_raw_features = self.tfidf.transform(X_raw_generation).toarray()
            X_raw_features = self._refit_sparse_data(X_raw_features, X_raw_generation)      
            
            row_sums = X_raw_features.sum(axis=1)
            weak_rows = np.where(row_sums < min_signal)[0]
            weak_ratio = len(weak_rows) / len(X_raw_features)

            print(f'[!] Zero ratio in samples: {weak_ratio * 100}')
            if weak_ratio > 0.3:  # more than 30% zero rows means vocab mismatch
                print(f'[= ! =] High zero-row ratio ({weak_ratio:.0%}), refitting TF-IDF on current batch')
                self.tfidf.fit(X_raw_generation)
                X_raw_features = self.tfidf.transform(X_raw_generation).toarray()  

            transformer_features = self.transformer_pooled_features(sequence_inputs)
            X_raw_features = np.concatenate([X_raw_features, transformer_features], axis=-1)
            
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

            # hybrid features by dot product of raw features and extracted features from transformer, this allows the MLP to learn from both the original feature space and the transformer-extracted feature space
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
            self.initialize_fitting(X_raw_generation)            
            X = self.shape_adaptation(hybrid_X, input_dim)
                
            self.lstm_setup_inference(X, y) 
            self.initialize_model_(X, input_dim, n_classes)
            self.model3.train(X, y, epochs=self.mlp_training_epochs, lr=0.1)

            if self.lstm_engine:
                self.storage.save_weights(self.memory_name, model_type='Pipeline') 
                
            print('🎉 All Model Trained!')
        else:
            print(f'[=] No suitable condition for training!')
            print('[=] Saving Weights for prediction')

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

            self.model2 = Transformer(
                vocab_size=1,
                d_model=32,
                n_heads=4,
                num_classes=num_classes
            )

            self.storage.load_weights(self.memory_name)

            pass



    def transformer_input_encoding(self, titles):
        if hasattr(self, 'vocab') and self.vocab:
            print("🔄 Using Transformer for probability calibration")
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
        self.initialize_fitting(X)            
        X_tfidf = self.tfidf.transform(X).toarray()
        self.X = X_tfidf.copy()

        print(f"\n🚀 Separate Modular MLP Pipeline:")
        print(f" Samples: {len(self.X)}")

        y_true = self.initialize_model_encoding(self.X, y_raw)
        self.utility_MLP_set(self.X, y_true)       
        print('✅ Done Training MLP Model! ')

class AccurateAnswerCache:
    def __init__(self, pipeline, similarity_threshold=0.85, max_size=500):
        self.pipeline = pipeline
        self.memory_name = self.pipeline.memory_name
        self.similarity_threshold = similarity_threshold
        self.max_size = max_size
        self.max_threshold = 0.7

        self.memory_exist = self.pipeline.storage.memory_exists(self.memory_name, type='Accurate-cache')
        if self.memory_exist:
            self.cache = self.pipeline.storage.load_accurate_cache(self.memory_name)
        else:
            self.cache = {}

        self.exact_hash_index = {}   # O(1) exact match lookup

    def _flatten_indices(self, input_ids):
        print('[=] Handling Input indices for inhomogenous shape checks.')
        try:
            try:
                ids = np.asarray(input_ids).ravel()
                return ids
            except:
                def flatten(x):
                    for item in x:
                        if isinstance(item, (list, tuple)):
                            yield from flatten(item)
                        else:
                            yield item

                if isinstance(input_ids, (list, tuple)):
                    flat_ids = list(flatten(input_ids))
                else:
                    flat_ids = input_ids  # already a flat array/tensor

                ids = np.asarray(flat_ids).ravel()
        except:
            flat_ids = self.pipeline._safe_to_2d_float(input_ids)
            ids = np.asarray(flat_ids).ravel()

        return ids

    def _adapt_ids_shape(self, ids_a, ids_b):
        try:
            flat_ids_a = self.pipeline._safe_to_2d_float(ids_a)
            flat_ids_b = self.pipeline._safe_to_2d_float(ids_b)

            if flat_ids_a.shape != flat_ids_b.shape:
                min_rows = min(flat_ids_a.shape[0], flat_ids_b.shape[0])
                min_cols = min(flat_ids_a.shape[1], flat_ids_b.shape[1])
                flat_ids_a = flat_ids_a[:min_rows, :min_cols]
                flat_ids_b = flat_ids_b[:min_rows, :min_cols]                          

            return flat_ids_a, flat_ids_b
        except Exception as e:
            print(f'[!] cannot adapt indices shape due to: {e}')
            return ids_a, ids_b
            

        
    def add_verified(self, x_mlp, input_ids, prediction, confidence, index,
                     source='user_confirmed'):

        try:
            key = self._make_key(x_mlp)

            # hash input_ids for fast exact-match 
            ids_hash = self._hash_ids(input_ids) if input_ids is not None else None

            flat_ids = self._flatten_indices(input_ids)
            entry = {
                'x_mlp'      : np.asarray(x_mlp, dtype=np.float64).ravel(),
                'input_ids'  : flat_ids if flat_ids is not None else None,
                'ids_hash'   : ids_hash,
                'prediction' : prediction,
                'confidence' : float(confidence),
                'index'      : int(index) if index is not None else None,
                'source'     : source,
                'hit_count'  : 0,
                'added_at'   : datetime.now().isoformat(),
                'last_hit'   : None
            }

            self.cache[key] = entry

            # maintain O(1) exact match index
            if ids_hash is not None:
                self.exact_hash_index[ids_hash] = key

            if len(self.cache) > self.max_size:
                self._evict_lru()

            print(f'[💎] Verified answer cached: {prediction} (source={source})')

            if self.cache[key]['source'] != 'automatic_verified' and not source.startswith('automatic'):
                self.pipeline.storage.save_accurate_cache_dict(self.memory_name, self.cache)
            
        except Exception as e:
            print(f'[!] Failed to add samples and answer to Answer cache due to: {e}')


    def lookup(self, x_mlp, input_ids=None):
        try:
            confidence_threshold = self.pipeline.confidence_threshold
            if confidence_threshold <= 0.5:
                confidence_treshold = self.max_threshold

            if not self.cache:
                return None
            
            AME = self.pipeline.AME_Encoder(x_mlp)
            anisotropy = self.pipeline.anisotropy_measurement(x_mlp)
            if np.isinf(AME) or np.isnan(AME):
                AME = (1.0 - self.pipeline.confidence_threshold)
            if np.isinf(anisotropy) or np.isnan(anisotropy):
                anisotropy = 0.15

            # FAST PATH — O(1) hash lookup 
            if input_ids is not None:
                ids_hash = self._hash_ids(input_ids)
                if ids_hash in self.exact_hash_index:
                    key   = self.exact_hash_index[ids_hash]
                    entry = self.cache.get(key)
                    if isinstance(entry, dict) and entry['source'] != 'automatic_verified' and 
                          not entry['source'].startswith('automatic'):
                          
                        entry['hit_count'] += 1
                        entry['last_hit']   = datetime.now().isoformat()
                        return {
                            'prediction' : entry['prediction'],
                            'confidence' : entry['confidence'],
                            'index'      : entry['index'],
                            'similarity' : 1.0,
                            'source'     : entry['source'],
                            'hit_count'  : entry['hit_count'],
                            'match_type' : 'exact_ids'
                        }  

                    if entry is not None and isinstance(entry, dict):
                        entry['hit_count'] += 1
                        entry['last_hit']   = datetime.now().isoformat()
                        return {
                            'prediction' : entry['prediction'],
                            'confidence' : entry['confidence'],
                            'index'      : entry['index'],
                            'similarity' : 1.0,
                            'source'     : entry['source'],
                            'hit_count'  : entry['hit_count'],
                            'match_type' : 'exact_ids'
                        }

            # SIMILARITY PATH — same as before, x_mlp + input_ids combined
            x_mlp = np.asarray(x_mlp, dtype=np.float64).ravel()
            best_match = None
            best_combined_sim = 0.0

            for entry in self.cache.values():
                mlp_sim = self.pipeline.cosine_robust_similarity(x_mlp, entry['x_mlp'])

                seq_sim = 1.0
                if input_ids is not None and entry['input_ids'] is not None:
                    ids_a = self._flatten_indices(input_ids)
                    ids_b = entry['input_ids'].ravel()

                    if ids_a.shape != ids_b.shape:
                        ids_a, ids_b = self._adapt_ids_shape(ids_a, ids_b)

                    min_len = min(len(ids_a), len(ids_b))
                    if min_len > 0:
                        seq_sim = float(np.mean(ids_a[:min_len] == ids_b[:min_len]))

                combined_env_sim = mlp_sim * confidence_threshold + seq_sim * anisotropy
                deterministic_modelling_sim = mlp_sim * 0.7 + seq_sim * AME

                combinatorial_absolute_factor = (
                    deterministic_modelling_sim > best_combined_sim and 
                    AME < 0.5 and 
                    entry['confidence'] > 0.5
                )
                dynamic_environmental_factor = (
                    combined_env_sim > best_combined_sim and 
                    AME > 0.5 and
                    entry['confidence'] > confidence_threshold
                )

                if combinatorial_absolute_factor:
                    best_combined_sim = deterministic_modelling_sim
                else:
                    best_combined_sim = combined_env_sim

                best_match = entry

            if best_match and best_combined_sim >= self.similarity_threshold:
                best_match['hit_count'] += 1
                best_match['last_hit']   = datetime.now().isoformat()
                return {
                    'prediction' : best_match['prediction'],
                    'confidence' : best_match['confidence'],
                    'index'      : best_match['index'],
                    'similarity' : float(best_combined_sim),
                    'source'     : best_match['source'],
                    'hit_count'  : best_match['hit_count'],
                    'match_type' : 'feature_similarity'
                }
                
            return None

        except Exception as e:
            print(f'[!] Cant search for Correct answer in cache due to: {e}')
            return None

    def _hash_ids(self, input_ids):
        try:
            print('[=] Creating a hash ID ')
            ids = self._flatten_indices(input_ids)
            return hashlib.md5(ids.tobytes()).hexdigest()
        except Exception as e:
            print(f'[!] cannot create a Hash ID for samples due to: {e}')
            return None

    def _make_key(self, x_mlp):
        try:
            def flatten(x):
                for item in x:
                    if isinstance(item, (list, tuple)):
                        yield from flatten(item)
                    else:
                        yield item

            if isinstance(x_mlp, (list, tuple)):
                flat_x = list(flatten(x_mlp))
            else:
                flat_x = x_mlp  # already a flat array/tensor

            x = np.asarray(flat_x, dtype='<i4').ravel()       
        except:
            flat_x = self.pipeline._safe_to_2d_float(x)
            x = np.asarray(flat_x, dtype='<i4').ravel()

        return hashlib.md5(x.tobytes()).hexdigest()

    def _evict_lru(self):
        try:
            if not self.cache:
                return
            lru_key = min(
                self.cache.keys(),
                key=lambda k: (self.cache[k]['hit_count'], self.cache[k]['added_at'])
            )
            entry = self.cache[lru_key]
            if entry.get('ids_hash'):
                self.exact_hash_index.pop(entry['ids_hash'], None)
            del self.cache[lru_key]
        except Exception as e:
            print(f'[!] cannot delete certain cache due to: {e}')

    

class RateLimiter:
    """
    Token bucket rate limiter.

    Supports both:
      - a single shared bucket (original behavior, backward compatible)
      - independent per-key buckets (e.g. per peer IP/agent_id),
        so one noisy peer cannot starve rate-limit capacity from others
    """

    def __init__(self, requests_per_minute: int = 60, per_key: bool = False,
                max_keys: int = 1000):
        #validate config, refuse a limiter that can never refill
        if requests_per_minute <= 0:
            raise ValueError(
                f"[-] requests_per_minute must be > 0, got {requests_per_minute}"
            )

        self.requests_per_minute = requests_per_minute
        self.per_key = per_key
        self.max_keys = max_keys
        self._lock = threading.Lock()

        if per_key:
            # one bucket per key, so peers don't share capacity here
            self._buckets: Dict[str, dict] = {}
        else:
            self.tokens      = float(requests_per_minute)
            self.last_refill = time.time()

    def acquire(self, key: str = None) -> bool:
        with self._lock:
            if self.per_key:
                return self._acquire_keyed(key or "_default")
            return self._acquire_global()

    def _acquire_global(self) -> bool:
        now     = time.time()
        elapsed = now - self.last_refill

        # guarded against clock skew producing negative elapsed
        if elapsed < 0:
            logger.warning(
                f'[!] RateLimiter: system clock moved backward by '
                f'{-elapsed:.3f}s — ignoring this interval for refill'
            )
            elapsed = 0.0

        new_tokens   = elapsed * (self.requests_per_minute / 60.0)
        self.tokens  = min(self.requests_per_minute, self.tokens + new_tokens)
        self.last_refill = now

        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False

    def _acquire_keyed(self, key: str) -> bool:
        now = time.time()

        if key not in self._buckets:
            # cap number of tracked keys to prevent unbounded
            # growth from an attacker cycling through many fake peer IDs
            if len(self._buckets) >= self.max_keys:
                self._evict_oldest_bucket()

            self._buckets[key] = {
                'tokens': float(self.requests_per_minute),
                'last_refill': now,
            }

        bucket  = self._buckets[key]
        elapsed = now - bucket['last_refill']

        if elapsed < 0:
            logger.warning(
                f'[!] RateLimiter[{key}]: clock moved backward by '
                f'{-elapsed:.3f}s — ignoring interval'
            )
            elapsed = 0.0

        new_tokens = elapsed * (self.requests_per_minute / 60.0)
        bucket['tokens'] = min(self.requests_per_minute, bucket['tokens'] + new_tokens)
        bucket['last_refill'] = now

        if bucket['tokens'] >= 1:
            bucket['tokens'] -= 1
            return True
        return False

    def _evict_oldest_bucket(self):
        """Evict the least-recently-refilled bucket to bound memory use."""
        if not self._buckets:
            return
        oldest_key = min(
            self._buckets.keys(),
            key=lambda k: self._buckets[k]['last_refill']
        )
        del self._buckets[oldest_key]

    def get_wait_time(self, key: str = None) -> float:
        """
        Seconds until at least 1 token will be available.
        Useful for callers that want to back off.
        """
        with self._lock:
            if self.per_key:
                bucket = self._buckets.get(key or "_default")
                tokens = bucket['tokens'] if bucket else self.requests_per_minute
            else:
                tokens = self.tokens

            if tokens >= 1:
                return 0.0
            tokens_needed = 1 - tokens
            return tokens_needed / (self.requests_per_minute / 60.0)

    def get_stats(self) -> Dict:
        """Visibility into limiter state"""
        with self._lock:
            if self.per_key:
                return {
                    'mode'          : 'per_key',
                    'tracked_keys'  : len(self._buckets),
                    'max_keys'      : self.max_keys,
                    'requests_per_minute': self.requests_per_minute,
                }
            return {
                'mode'          : 'global',
                'current_tokens': round(self.tokens, 2),
                'requests_per_minute': self.requests_per_minute,
            }


class InputSanitizer:
    """Sanitize and validate inputs."""

    # comprehensive control-char stripping including
    # newline/CR, since this class explicitly exists to protect
    # logging/serialization from injection
    _CONTROL_CHARS_PATTERN = re.compile(
        r'[\x00-\x1f\x7f]'   
    )

    # OPT-IN for genuinely
    # multi-line legitimate text (e.g. free-form descriptions).
    _CONTROL_CHARS_ALLOW_WHITESPACE = re.compile(
        r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]'   
    )

    @staticmethod
    def sanitize_text(text: str, max_length: int = 10000,
                      allow_newlines: bool = False) -> str:
        if not isinstance(text, str):
            raise SecurityError("[-] Input must be a string")

        # check length BEFORE any transformation, so length
        # limits reflect what was actually submitted.
        if len(text) > max_length:
            raise SecurityError(f"[-] Text exceeds maximum length of {max_length}")

        if allow_newlines:
            text = InputSanitizer._CONTROL_CHARS_ALLOW_WHITESPACE.sub('', text)
            # even when newlines are allowed, escape them for log safety
        else:
            text = InputSanitizer._CONTROL_CHARS_PATTERN.sub('', text)

        escape_patterns = [
            r'\\x[0-9a-fA-F]{2}',   # literal "\x41" style escapes
            r'\\u[0-9a-fA-F]{4}',   # literal "\u0041" style escapes
        ]
        for pattern in escape_patterns:
            if re.search(pattern, text):
                logger.warning(
                    f'[!] sanitize_text: literal escape sequence pattern '
                    f'detected and neutralized (pattern={pattern})'
                )
                text = re.sub(pattern, '?', text)

        return text.strip()

    @staticmethod
    def validate_batch_size(size: int, max_batch: int = 100) -> bool:
        # validated type before comparison to avoid a confusing
        # TypeError if a non-int slips through
        if not isinstance(size, int):
            raise SecurityError(f"[-] Batch size must be an integer, got {type(size).__name__}")
        if size <= 0 or size > max_batch:
            raise SecurityError(f"[-] Batch size must be between 1 and {max_batch}")
        return True


class ApiKeyManager:
    """Manage API keys with rotation and bounded storage."""

    MIN_KEY_LENGTH = 16   # minimum acceptable length for caller-supplied keys

    def __init__(self, rotation_days: int = 30, max_keys: int = 10000):
        self.keys: Dict[str, dict] = {}
        self.rotation_days = rotation_days
        self.max_keys = max_keys  
        self._lock = threading.Lock()

    def generate_key(self, metadata: dict = None, key_value: str = None) -> str:
        if key_value:
            # reject obviously weak caller-supplied keys
            if not isinstance(key_value, str) or len(key_value) < self.MIN_KEY_LENGTH:
                raise SecurityError(
                    f"[-] Provided key must be a string of at least "
                    f"{self.MIN_KEY_LENGTH} characters"
                )
            raw_key = key_value
        else:
            raw_key = secrets.token_urlsafe(32)

        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

        with self._lock:
            # warn on collision 
            if key_hash in self.keys and self.keys[key_hash].get('is_active'):
                logger.warning(
                    f'[!] generate_key: key hash collision with an '
                    f'already-active key — overwriting metadata'
                )

            # evict oldest inactive entries if at capacity,
            if len(self.keys) >= self.max_keys:
                self._evict_oldest_inactive()

            self.keys[key_hash] = {
                'created_at': datetime.now(),
                'last_used' : None,
                'metadata'  : metadata or {},
                'is_active' : True
            }

        return raw_key

    def validate_key(self, api_key: str) -> bool:
        # explicit type check before .encode(), avoids an
        # uncontrolled AttributeError on non-string input
        if not api_key or not isinstance(api_key, str):
            return False

        try:
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        except (UnicodeEncodeError, AttributeError):
            return False

        with self._lock:
            key_info = self.keys.get(key_hash)
            if not key_info or not key_info.get('is_active', False):
                return False

            age = (datetime.now() - key_info['created_at']).days
            if age >= self.rotation_days:
                key_info['is_active'] = False
                return False

            key_info['last_used'] = datetime.now()
            return True

    def revoke_key(self, api_key: str) -> bool:
        if not api_key or not isinstance(api_key, str):
            return False

        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        with self._lock:
            if key_hash in self.keys:
                self.keys[key_hash]['is_active'] = False
                return True
        return False

    def cleanup_expired(self, grace_period_days: int = 7):
        """
        periodic sweep to actually remove long-inactive keys.
        """
        with self._lock:
            now = datetime.now()
            to_remove = [
                key_hash for key_hash, info in self.keys.items()
                if not info.get('is_active', False)
                and (now - info['created_at']).days >= (self.rotation_days + grace_period_days)
            ]
            for key_hash in to_remove:
                del self.keys[key_hash]

            if to_remove:
                logger.info(f'[=] Cleaned up {len(to_remove)} expired API key entries')

            return len(to_remove)

    def _evict_oldest_inactive(self):
        """Evict the oldest inactive key to make room, called under lock."""
        inactive = [
            (h, info) for h, info in self.keys.items()
            if not info.get('is_active', False)
        ]
        if inactive:
            oldest_hash = min(inactive, key=lambda x: x[1]['created_at'])[0]
            del self.keys[oldest_hash]
        else:
            # no inactive keys to evict and still at capacity 
            logger.warning(
                f'[!] ApiKeyManager at max_keys={self.max_keys} capacity '
                f'with no inactive keys to evict — consider raising max_keys '
                f'or auditing why so many keys remain active'
            )

    def get_stats(self) -> Dict:
        """Visibility into key store health — same pattern as WorkerPool.get_health()."""
        with self._lock:
            active   = sum(1 for k in self.keys.values() if k.get('is_active'))
            inactive = len(self.keys) - active
            return {
                'total_keys'   : len(self.keys),
                'active_keys'  : active,
                'inactive_keys': inactive,
                'at_capacity'  : len(self.keys) >= self.max_keys,
            }

class AsyncResultQueue:
    """
    Complete result queue with integrated processor.
    Handles callbacks, webhooks, WebSocket, storage, and streaming.
    """
    
    def __init__(self, max_size: int = 1000, cleanup_interval: int = 60):
        self._requests: Dict[str, AsyncRequest] = {}
        self._pending_queue: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        self._completion_queue: asyncio.Queue = asyncio.Queue()
        self._result_futures: Dict[str, asyncio.Future] = {}
        self._lock = asyncio.Lock()
        self._cleanup_interval = cleanup_interval
        self._running = False
        
        # Result processor components
        self._cleanup_task: Optional[asyncio.Task] = None
        self._processor_task: Optional[asyncio.Task] = None
        
        # Optional features
        self._webhook_url: Optional[str] = None
        self._websocket_clients: List = []  # Store WebSocket connections
        self._storage_enabled: bool = False
        self._storage_path: Optional[str] = None
        self._streaming_queue: Optional[asyncio.Queue] = None
        
        # Metrics
        self._metrics = {
            'total_completed': 0,
            'total_failed': 0,
            'total_callbacks': 0,
            'total_webhooks': 0,
            'avg_processing_time': 0.0
        }
    
    # ============ INITIALIZATION ============
    
    async def start(self, 
                   webhook_url: str = None,
                   storage_path: str = None,
                   enable_streaming: bool = False):
        """
        Start the result queue with optional features.
        
        Args:
            webhook_url: Send results to this URL when complete
            storage_path: Save results to disk
            enable_streaming: Enable result streaming queue
        """
        self._running = True
        self._webhook_url = webhook_url
        self._storage_enabled = bool(storage_path)
        self._storage_path = storage_path
        
        if enable_streaming:
            self._streaming_queue = asyncio.Queue()
        
        # Start background tasks
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._processor_task = asyncio.create_task(self._result_processor())
        
         
        logger.info("✅ AsyncResultQueue started")
        


    async def stop(self):
        """Stop the result queue"""
        self._running = False
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
        if self._processor_task:
            self._processor_task.cancel()
        
        await asyncio.gather(
            self._cleanup_task, 
            self._processor_task, 
            return_exceptions=True
        )
        
        logger.info("✅ AsyncResultQueue stopped")
    
    # ============ REQUEST SUBMISSION ============
    
    async def submit(self, 
                    texts, 
                    api_key: str = None, 
                    client_ip: str = None,
                    callback: Callable = None,
                    webhook_url: str = None) -> str:
        """
        Submit a prediction request.
        
        Args:
            texts: List of input texts to predict
            api_key: API key for authentication
            client_ip: Client IP address
            callback: Optional callback function
            webhook_url: Optional webhook for this specific request
            
        Returns:
            request_id for tracking
        """
        request_id = str(uuid.uuid4())
        
        async with self._lock:
            request = AsyncRequest(
                request_id=request_id,
                texts=texts,
                api_key=api_key,
                client_ip=client_ip,
                callback=callback,
                webhook_url=webhook_url
            )
            self._requests[request_id] = request
            
            # Create future for awaiting result
            future = asyncio.Future()
            self._result_futures[request_id] = future
            
            # Add to processing queue
            await self._pending_queue.put(request)
            
        logger.debug(f"[=] Submitted request {request_id}: {texts}")
        return request_id
    
    async def wait_for_result(self, request_id: str, timeout: int = 30) -> Dict:
        """
        Wait for a specific request to complete.
        
        Args:
            request_id: ID from submit()
            timeout: Maximum wait time in seconds
            
        Returns:
            Prediction result dictionary
        """
        async with self._lock:
            future = self._result_futures.get(request_id)
            if not future:
                request = self._requests.get(request_id)
                if request and request.status == RequestStatus.COMPLETED:
                    return request.result
                raise ValueError(f"[=] Unknown request_id: {request_id}")
        
        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            await self._mark_failed(request_id, "Request timeout")
            raise
        finally:
            async with self._lock:
                self._result_futures.pop(request_id, None)
    
    async def complete(self, request_id: str, result: Dict):
        """
        Mark a request as completed with result.
        Called by worker when prediction is done.
        """
        async with self._lock:
            request = self._requests.get(request_id)
            if not request:
                logger.warning(f"[=] Request {request_id} not found")
                return
            
            request.status = RequestStatus.COMPLETED
            request.result = result
            request.completed_at = time.time()
            
            # Calculate processing time for metrics
            processing_time = request.completed_at - request.created_at
            alpha = 0.1
            self._metrics['avg_processing_time'] = (
                alpha * processing_time + 
                (1 - alpha) * self._metrics['avg_processing_time']
            )
            
            # Add to completion queue for processor
            await self._completion_queue.put(request)
            
            logger.debug(f"[=] Request {request_id} completed in {processing_time:.2f}s")
    
    async def _mark_failed(self, request_id: str, error: str):
        """Mark a request as failed"""
        async with self._lock:
            request = self._requests.get(request_id)
            if request:
                request.status = RequestStatus.FAILED
                request.error = error
                request.completed_at = time.time()
                await self._completion_queue.put(request)
            
            future = self._result_futures.get(request_id)
            if future and not future.done():
                future.set_exception(Exception(error))
    
    async def get_pending(self) -> Optional[AsyncRequest]:
        """Get next pending request (blocks)"""
        try:
            return await self._pending_queue.get()
        except asyncio.CancelledError:
            return None
    
    # ============ THE MAIN RESULT PROCESSOR ============
    
    async def _result_processor(self):
        """
        Process results as they complete.
        Handles: Callbacks, Webhooks, WebSocket, Storage, Streaming
        """
        logger.info("[=] Result processor started")
        
        while self._running:
            try:
                # Wait for a completed request
                completed_request = await asyncio.wait_for(
                    self._completion_queue.get(), 
                    timeout=1.0
                )
                
                if not completed_request:
                    continue
                
                request_id = completed_request.request_id
                result = completed_request.result
                error = completed_request.error
                is_success = error is None
                
                # Update metrics
                if is_success:
                    self._metrics['total_completed'] += 1
                else:
                    self._metrics['total_failed'] += 1
                
                # ============ 1. EXECUTE CALLBACK ============
                if completed_request.callback:
                    try:
                        self._metrics['total_callbacks'] += 1
                        
                        # Support both sync and async callbacks
                        if asyncio.iscoroutinefunction(completed_request.callback):
                            # Async callback
                            await completed_request.callback(request_id, result, error)
                        else:
                            # Sync callback - run in thread pool
                            await asyncio.to_thread(
                                completed_request.callback,
                                request_id, result, error
                            )
                        logger.debug(f"[=] Callback executed for {request_id}")
                        
                    except Exception as e:
                        logger.error(f"[=] Callback failed for {request_id}: {e}")
                
                # ============ 2. RESOLVE WAITING FUTURE ============
                future = self._result_futures.get(request_id)
                if future and not future.done():
                    if error:
                        future.set_exception(Exception(error))
                    else:
                        future.set_result(result)
                    self._result_futures.pop(request_id, None)
                
                # ============ 3. WEBHOOK NOTIFICATION ============
                webhook_url = completed_request.webhook_url or self._webhook_url
                if webhook_url and is_success:
                    await self._send_webhook(webhook_url, {
                        'request_id': request_id,
                        'status': 'success',
                        'result': result,
                        'timestamp': completed_request.completed_at
                    })
                    self._metrics['total_webhooks'] += 1
                
                # ============ 4. WEBSOCKET PUSH ============
                if self._websocket_clients:
                    await self._broadcast_websocket({
                        'type': 'prediction_complete',
                        'request_id': request_id,
                        'result': result,
                        'error': error,
                        'processing_time': completed_request.completed_at - completed_request.created_at
                    })
                
                # ============ 5. PERSISTENT STORAGE ============
                if self._storage_enabled and is_success:
                    await self._store_result(request_id, result, completed_request)
                
                # ============ 6. STREAMING TO RESPONSE QUEUE ============
                if self._streaming_queue:
                    await self._streaming_queue.put({
                        'request_id': request_id,
                        'result': result,
                        'error': error,
                        'completed_at': completed_request.completed_at
                    })
                
                # ============ 7. LOG COMPLETION ============
                logger.info(
                    f"[=] Request {request_id} processed - "
                    f"[=] Success: {is_success}, "
                    f"[=] Time: {completed_request.completed_at - completed_request.created_at:.2f}s"
                )
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[=] Result processor error: {e}")
                await asyncio.sleep(0.1)
        
        logger.info("[=] Result processor stopped")
    
    # ============ WEBHOOK HANDLER ============
    
    async def _send_webhook(self, url: str, data: dict):
        """Send result to webhook URL"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, 
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        logger.debug(f"[=] Webhook sent to {url}")
                    else:
                        logger.warning(f"[=] Webhook failed: {response.status}")
        except Exception as e:
            logger.error(f"[=] Webhook error: {e}")
    
    # ============ WEBSOCKET HANDLER ============
    
    async def register_websocket(self, websocket):
        """Register a WebSocket client for real-time updates"""
        self._websocket_clients.append(websocket)
        
        # Remove when closed
        try:
            await websocket.wait_closed()
        finally:
            self._websocket_clients.remove(websocket)
    
    async def _broadcast_websocket(self, message: dict):
        """Broadcast message to all connected WebSocket clients"""
        disconnected = []
        
        for client in self._websocket_clients:
            try:
                await client.send_json(message)
            except:
                disconnected.append(client)
        
        # Clean up disconnected clients
        for client in disconnected:
            self._websocket_clients.remove(client)
    
    # ============ PERSISTENT STORAGE ============
    
    async def _store_result(self, request_id: str, result: dict, request: AsyncRequest):
        # Store result to disk
        if not self._storage_path:
            return
        
        try:
            import aiofiles
            import os
            
            os.makedirs(self._storage_path, exist_ok=True)
            
            filepath = os.path.join(self._storage_path, f"{request_id}.json")
            
            data = {
                'request_id': request_id,
                'text': request.text,
                'result': result,
                'created_at': request.created_at,
                'completed_at': request.completed_at,
                'processing_time': request.completed_at - request.created_at
            }
            
            async with aiofiles.open(filepath, 'w') as f:
                await f.write(json.dumps(data, indent=2))
                
            logger.debug(f"[=] Result stored: {filepath}")
            
        except Exception as e:
            logger.error(f"[=] Storage failed: {e}")
    
    # ============ RESULT STREAMING ============
    
    async def get_result_stream(self) -> asyncio.Queue:
        """
        Get a queue that receives results as they complete.
        Useful for streaming responses to clients.
        """
        if not self._streaming_queue:
            self._streaming_queue = asyncio.Queue()
        return self._streaming_queue
    
    # ============ CLEANUP ============
    
    async def _cleanup_loop(self):
        # Periodically clean up expired and old requests
        while self._running:
            await asyncio.sleep(self._cleanup_interval)
            
            async with self._lock:
                now = time.time()
                
                # Find expired pending requests
                expired = [
                    req_id for req_id, req in self._requests.items()
                    if req.is_expired and req.status in (RequestStatus.PENDING, RequestStatus.PROCESSING)
                ]
                
                for req_id in expired:
                    request = self._requests[req_id]
                    request.status = RequestStatus.TIMEOUT
                    request.completed_at = now
                    request.error = "Request expired"
                    await self._completion_queue.put(request)
                    
                    # Clean up future
                    future = self._result_futures.pop(req_id, None)
                    if future and not future.done():
                        future.set_exception(TimeoutError(f"Request {req_id} expired"))
                
                # Remove completed/failed/timeout requests older than 1 hour
                old_cutoff = now - 3600
                to_remove = [
                    req_id for req_id, req in self._requests.items()
                    if req.completed_at and req.completed_at < old_cutoff
                ]
                for req_id in to_remove:
                    self._requests.pop(req_id, None)
                
                if expired:
                    logger.info(f"[=] Cleaned up {len(expired)} expired requests")
    
    async def _cleanup_request(self, request_id: str):
        # Clean up a single completed request after processing
        # Schedule for removal after delay
        async def _delayed_remove():
            await asyncio.sleep(3600)  # Keep for 1 hour
            async with self._lock:
                self._requests.pop(request_id, None)
        
        asyncio.create_task(_delayed_remove())
    
    # ============ UTILITY METHODS ============
    
    def _update_metrics(self, request: AsyncRequest):
        # Update metrics after processing
        # Metrics already updated in complete() and _result_processor
        pass
    
    def get_status(self, request_id: str) -> Optional[RequestStatus]:
        # Get status of a request
        request = self._requests.get(request_id)
        return request.status if request else None
    
    def get_result(self, request_id: str) -> Optional[Dict]:
        # Get result if completed
        request = self._requests.get(request_id)
        if request and request.status == RequestStatus.COMPLETED:
            return request.result
        return None
    
    def get_metrics(self) -> Dict:
        # Get queue metrics
        return {
            **self._metrics,
            'pending_count': self._pending_queue.qsize(),
            'completion_count': self._completion_queue.qsize(),
            'total_requests': len(self._requests),
            'pending': sum(1 for r in self._requests.values() if r.status == RequestStatus.PENDING),
            'processing': sum(1 for r in self._requests.values() if r.status == RequestStatus.PROCESSING),
            'completed': sum(1 for r in self._requests.values() if r.status == RequestStatus.COMPLETED),
            'failed': sum(1 for r in self._requests.values() if r.status == RequestStatus.FAILED),
            'timeout': sum(1 for r in self._requests.values() if r.status == RequestStatus.TIMEOUT)
        }

class WorkerPool:
    """Worker pool that processes requests from the result queue."""

    def __init__(self, result_queue: AsyncResultQueue, num_workers: int = 4,
                max_consecutive_errors: int = 10):
        self.result_queue = result_queue
        self.num_workers   = num_workers
        self._workers: List[asyncio.Task] = []
        self._running       = False
        self._start_lock    = asyncio.Lock()   # FIX 1 — prevent double-start
        self.max_consecutive_errors = max_consecutive_errors

        # per-worker health tracking — FIX 4
        self._worker_error_counts = [0] * num_workers
        self._worker_last_active  = [0.0] * num_workers

    async def start(self, predict_func):
        # guard against double-start leaking orphaned tasks
        async with self._start_lock:
            if self._running:
                logger.warning('[=] WorkerPool.start() called while already '
                              'running — ignoring duplicate start')
                return

            self._running = True
            self._worker_error_counts = [0] * self.num_workers
            self._worker_last_active  = [time.time()] * self.num_workers

            self._workers = [
                asyncio.create_task(self._worker(predict_func, worker_idx=i))
                for i in range(self.num_workers)
            ]
            await self.result_queue.start()
            logger.info(f'[=] WorkerPool started with {self.num_workers} workers')

    async def stop(self):
        self._running = False
        await self.result_queue.stop()

        for worker in self._workers:
            worker.cancel()

        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)

        self._workers.clear()   # always clear, no stale references remain
        logger.info('[=] WorkerPool stopped, all workers cleared')

    async def _worker(self, predict_func, worker_idx: int):
        consecutive_errors = 0

        while self._running:
            try:
                try:
                    request = await asyncio.wait_for(
                        self.result_queue.get_pending(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                if not request:
                    continue

                self._worker_last_active[worker_idx] = time.time()

                async with self.result_queue._lock:
                    request.status = RequestStatus.PROCESSING

                try:
                    result = await asyncio.wait_for(
                        asyncio.to_thread(
                            predict_func,
                            texts=request.texts,
                            api_key=request.api_key,
                            client_ip=request.client_ip
                        ),
                        timeout=30.0
                    )
                    await self.result_queue.complete(request.request_id, result)
                    consecutive_errors = 0   

                except asyncio.TimeoutError:
                    # log that the underlying thread may still be
                    # running; can't force-kill it, but need at least
                    # that this is a known limitation rather than silent
                    logger.warning(
                        f'[!] Worker {worker_idx}: request {request.request_id} '
                        f'timed out after 30s. The underlying OS thread may '
                        f'still be running in the background (Python threads '
                        f'cannot be forcibly cancelled) — consider making '
                        f'predict_func more responsive to avoid thread buildup.'
                    )

                    await self.result_queue._mark_failed(request.request_id, 'timeout')

                except asyncio.CancelledError:
                    await self.result_queue._mark_failed(request.request_id, 'cancelled')
                    break

                except Exception as e:
                    await self.result_queue._mark_failed(request.request_id, str(e))
                    consecutive_errors = 0   # per-request errors don't count
                                             # toward the outer circuit breaker

            except asyncio.CancelledError:
                break

            except Exception as e:
                consecutive_errors += 1
                self._worker_error_counts[worker_idx] = consecutive_errors

                logger.error(f'[=] Worker {worker_idx} error '
                           f'({consecutive_errors}/{self.max_consecutive_errors}): {e}')

                # circuit breaker with exponential backoff
                if consecutive_errors >= self.max_consecutive_errors:
                    logger.error(
                        f'[!!] Worker {worker_idx} exceeded '
                        f'{self.max_consecutive_errors} consecutive errors — '
                        f'stopping this worker to avoid runaway resource use. '
                        f'Pool now running with reduced capacity.'
                    )
                    break   

                backoff = min(0.1 * (2 ** consecutive_errors), 5.0)
                await asyncio.sleep(backoff)

    def get_health(self) -> Dict:
        """visibility into worker pool health"""
        now = time.time()
        return {
            'num_workers'       : self.num_workers,
            'active_workers'    : sum(1 for t in self._workers if not t.done()),
            'dead_workers'      : sum(1 for t in self._workers if t.done()),
            'worker_error_counts': list(self._worker_error_counts),
            'worker_idle_seconds': [
                round(now - t, 1) for t in self._worker_last_active
            ],
            'is_running'        : self._running,
        }


class PipelineAsyncManager:
    """
    Robust wrapper for using async features in synchronous code.
    
    Features:
    - Automatic retry on failure
    - Task tracking and cleanup
    - Graceful shutdown with timeout
    - Health monitoring
    - Queue for pending requests
    - Thread-safe operations
    - Security Layers
    """
    
    def __init__(self, pipeline, prediction_manager, config: SecurityConfig=None, security_level: SecurityLevel = SecurityLevel.STAGING, state_file: str = None, api_key: Any=None, max_workers=4, task_timeout=30, max_retries=3):
        self.pipeline = pipeline
        self.prediction_manager = prediction_manager
        self.max_workers = max_workers
        self.default_timeout = task_timeout
        self.max_retries = max_retries
        
        self.security_level = security_level
        self.config = self._get_config_for_level(security_level) or SecurityConfig()

        # Security components
        self.rate_limiter = RateLimiter(self.config.rate_limit_requests)
        self.sanitizer = InputSanitizer()
        self.api_key_manager = ApiKeyManager(self.config.api_key_rotation_days) 
        
        # State management
        self._state = WrapperState.UNINITIALIZED
        self._lock = threading.RLock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
     
        # Task management
        self._pending_tasks: Dict[str, AsyncTask] = {}
        self._task_counter = 0
        self._task_lock = threading.Lock()
        
        # Queue for requests (when at capacity)
        self._request_queue: queue.Queue = queue.Queue(maxsize=1000)
        self._queue_worker: Optional[threading.Thread] = None
        
        # Rate limiting per IP
        self._ip_rate_limiters: Dict[str, RateLimiter] = defaultdict(
            lambda: RateLimiter(self.config.rate_limit_requests)
        ) 

        # Health monitoring
        self._health_check_interval = 30
        self._health_thread: Optional[threading.Thread] = None
        self._last_heartbeat: float = 0
        
        # Audit log
        self._audit_log: List[Dict] = []
        self._max_audit_log = 1000 

        # Statistics
        self._stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'timed_out_requests': 0,
            'avg_response_time': 0.0,
            'queue_size': 0
        }
        # Generate initial API key if enabled
        if self.config.require_api_key:
            self._default_api_key = api_key if api_key else self.api_key_manager.generate_key({'type': 'default'})
            print(f"[🔑] Default API Key: {self._default_api_key}")
            print("[!] Store this key securely - it won't be shown again!")

        self.state_file = state_file if state_file is not None else 'security_state.json'
        self.admin_keys: Dict[str, dict] = {}  # admin_token -> {role, created_at}
        self._load_state()
        self._bootstrap_token_hash = None
        self._bootstrap_token_file = "bootstrap.token"  

        # Generate initial admin key if none exists
        if not self.admin_keys or self.config.require_bootsrap_auth:
            self._initialize_bootstrap_security()

        self._start_count = 0
        self._last_start_time = 0
        self._failed_starts = 0
        self._pending_start = None
                 
        
    @property
    def state(self) -> str:
        # Get current wrapper state.
        return self._state.value
    
    @property
    def is_running(self) -> bool:
        # Check if wrapper is running.
        return self._state == WrapperState.RUNNING

    # ======= Security and Utility Methods =======

    def _get_config_for_level(self, level: SecurityLevel) -> SecurityConfig:
        # appropriate security config for deployment level
        
        if level == SecurityLevel.DEVELOPMENT:
            return SecurityConfig(
                require_api_key=False,
                rate_limit_requests=1000,
                min_start_interval=0,
                require_bootstrap_auth=False
            )
        
        elif level == SecurityLevel.STAGING:
            return SecurityConfig(
                require_api_key=True,
                rate_limit_requests=120,
                min_start_interval=2.0,
                require_bootstrap_auth=False
            )
        
        elif level == SecurityLevel.PRODUCTION:
            return SecurityConfig(
                require_api_key=True,
                rate_limit_requests=60,
                min_start_interval=5.0,
                max_consecutive_failures=3,
                require_bootstrap_auth=False  # Still off for auto-restart
            )
        
        elif level == SecurityLevel.HARDENED:
            return SecurityConfig(
                require_api_key=True,
                rate_limit_requests=30,
                min_start_interval=10.0,
                max_consecutive_failures=2,
                require_bootstrap_auth=True,  # Only for hardened
                bootstrap_token_hash=os.environ.get('BOOTSTRAP_TOKEN_HASH')
            )
     

    def _load_state(self):
        # Load persisted state
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                state = json.load(f)
                # Restore API keys and IP lists
                self.config.allowed_ips = state.get('allowed_ips', [])
                self.config.blocklisted_ips = state.get('blocklisted_ips', [])
                # Restore API keys (needs careful handling)
    
    def _save_state(self):
        # Save state to disk
        with open(self.state_file, 'w') as f:
            json.dump({
                'allowed_ips': self.config.allowed_ips,
                'blocklisted_ips': self.config.blocklisted_ips,
                'last_saved': datetime.now().isoformat()
            }, f)

    def _audit(self, event_type: str, details: dict, client_ip: str = None):
        # Log security events
        entry = {
            'timestamp': datetime.now().isoformat(),
            'event': event_type,
            'details': details,
            'ip': client_ip
        }
        self._audit_log.append(entry)
        
        # Trim log
        if len(self._audit_log) > self._max_audit_log:
            self._audit_log = self._audit_log[-self._max_audit_log:]
        
        # Log critical events
        if event_type in ['auth_failure', 'security_block', 'rate_limit_exceeded']:
            print(f"[=⚠️ SECURITY=] {event_type}: {details}")  
            
        with open('security_audit.log', 'a') as f:
            f.write(json.dumps(entry) + '\n')
    

    def _verify_admin(self, admin_token: str, required_role: AdminRole = AdminRole.ADMIN) -> bool:
        # Verify admin token and role
        if not admin_token:
            return False
        
        token_hash = hashlib.sha256(admin_token.encode()).hexdigest()
        admin_info = self.admin_keys.get(token_hash)
        
        if not admin_info:
            return False
        
        # Check role hierarchy
        role_hierarchy = {
            AdminRole.ADMIN: 3,
            AdminRole.OPERATOR: 2,
            AdminRole.AUDITOR: 1
        }
        
        return role_hierarchy.get(admin_info['role'], 0) >= role_hierarchy.get(required_role, 0)
    
    def _check_ip_allowed(self, client_ip: str, is_admin: bool = False) -> bool:
        # IP checking with CIDR support
        # Check global blocklist first (applies to everyone)
        if client_ip in self.config.blocklisted_ips:
            self._audit('security_block', {'reason': 'blocklisted_ip', 'ip': client_ip}, client_ip)
            return False
        
        # Admin-specific IP whitelist
        if is_admin and self.config.enforce_admin_ip_whitelist:
            if self.config.admin_allowed_ips:
                # Check if admin IP is allowed
                if not self._ip_in_list(client_ip, self.config.admin_allowed_ips):
                    self._audit('security_block', 
                               {'reason': 'admin_ip_not_allowed', 'ip': client_ip}, 
                               client_ip)
                    return False
            return True
        
        # Regular user IP whitelist
        if self.config.allowed_ips:
            return self._ip_in_list(client_ip, self.config.allowed_ips)
        
        return True
        
    def _ip_in_list(self, ip: str, ip_list: List[str]) -> bool:
        # Check if IP matches any entry in list (supports CIDR)
        try:
            client = ipaddress.ip_address(ip)
            for allowed in ip_list:
                if '/' in allowed:
                    network = ipaddress.ip_network(allowed, strict=False)
                    if client in network:
                        return True
                elif ip == allowed:
                    return True
        except ValueError:
            pass
        return False    

    def _check_rate_limit(self, client_ip: str = None, is_admin: bool = False) -> bool:
        # Check rate limit for IP or global
        if is_admin and self.config.admin_bypass_rate_limit:
            # Admins use separate, higher limit or no limit
            if self.config.admin_rate_limit < 999:  # If limit is set
                limiter = self.admin_rate_limiter
            else:
                return True  # No rate limit for admins
        else:
            limiter = self._ip_rate_limiters.get(client_ip, self.rate_limiter)
        
        allowed = limiter.acquire()
        if not allowed:
            self._stats['rate_limiter_blocks'] += 1
            self._audit('rate_limit_exceeded', 
                       {'ip': client_ip or 'global', 'is_admin': is_admin}, 
                       client_ip)
        
        return allowed
        
    def _is_admin_token(self, api_key: str) -> bool:
        # Check if an API key is actually an admin token
        if not api_key:
            return False
        
        token_hash = hashlib.sha256(api_key.encode()).hexdigest()
        return token_hash in self.admin_keys
    
    def _authenticate(self, api_key: str, client_ip: str = None, is_admin: bool = False) -> bool:
        # Enhanced authentication - handles both API keys and admin tokens
        if not self.config.require_api_key:
            return True

        validation = api_key == self._default_api_key
        if self._default_api_key and validation:
            return True
        
        if not api_key:
            if 'auth_failures' in self._stats:
                self._stats['auth_failures'] += 1
                self._audit('auth_failure', {'reason': 'missing_api_key'}, client_ip)
            else:
                self._stats['auth_failures'] = 0
                self._stats['auth_failures'] += 1
                self._audit('auth_failure', {'reason': 'missing_api_key'}, client_ip)                
            return False
        
        # Check if it's an admin token first
        if self._is_admin_token(api_key):
            # Admin tokens are always valid (but may have other restrictions)
            return True
        
        # Regular API key validation
        valid = self.api_key_manager.validate_key(api_key)
        if not valid:
            if 'auth_failures' in self._stats:
                self._stats['auth_failures'] += 1
                self._audit('auth_failure', {'reason': 'invalid_api_key'}, client_ip)
            else:
                self._stats['auth_failures'] = 0
                self._stats['auth_failures'] += 1
                self._audit('auth_failure', {'reason': 'invalid_api_key'}, client_ip)                

        return valid if valid else validation

    # ======= Core Wrapper Methods =======
    def start(self, timeout: float = 5.0, method: str = None, bootstrap_token: str = None) -> bool:
        """
        Start the async event loop and workers.
        
        Args:
            timeout: Maximum time to wait for startup
            method: The prediction method to use
            bootstrap_token: Token for initial authentication

        Returns:
            True if started successfully, False otherwise
        """

        # Only check bootstrap token in HARDENED mode
        if self.security_level == SecurityLevel.HARDENED:
            if not self._validate_bootstrap_token(bootstrap_token):
                logger.error("[=] Bootstrap token required for HARDENED security level")
                return False

        with self._lock:
            if self._state in (WrapperState.RUNNING, WrapperState.STARTING):
                logger.warning(f"[=] AsyncWrapper already in state: {self._state}")
                return True
                
            self._state = WrapperState.STARTING 

            # Prevent rapid restart attacks (crash looping)
            now = time.time()
            if now - self._last_start_time < self.config.min_start_interval:
                logger.warning(f"[-] Start too frequent - need {self.config.min_start_interval}s between starts")
                self._audit('start_throttled', {'interval': now - self._last_start_time})
                return False
            
            # Track failed starts for circuit breaker
            if self._state == WrapperState.ERROR:
                self._failed_starts += 1
                if self._failed_starts > self.config.max_consecutive_failures:
                    logger.error(f"[-] Too many failed starts ({self._failed_starts}) - circuit open")
                    self._audit('circuit_open', {'failures': self._failed_starts})
                    return False
            else:
                self._failed_starts = 0  # Reset on success          
            
        try:
            print(f"[=] Starting PipelineAsyncManager with method: {method or 'default'}")
            # Start event loop thread
            self._start_with_limits(timeout, method=method)
            
            with self._lock:
                self._state = WrapperState.RUNNING
                self._last_heartbeat = time.time()
            
            logger.info(f"[=] PipelineAsyncManager started successfully (workers={self.max_workers})")
            return True
            
        except Exception as e:
            logger.error(f"[-] Failed to start manager: {e}")
            with self._lock:
                self._state = WrapperState.ERROR
            return False
            
    def _start_with_limits(self, timeout: float, method: str = None):
        # Start with resource limits to prevent abuse
        
        # Check system resources before starting
        try:
            import psutil
            
            # CPU limit
            if psutil.cpu_percent(interval=1) > self.config.max_cpu_percent:
                raise RuntimeError(f"[=] System CPU too high ({psutil.cpu_percent()}%)")
            
            # Memory limit  
            memory = psutil.virtual_memory()
            if memory.percent > self.config.max_memory_percent:
                raise RuntimeError(f"[=] System memory too high ({memory.percent}%)")
            
            # Disk space for logs
            disk = psutil.disk_usage('/')
            if disk.free < self.config.min_disk_space_mb * 1024 * 1024:
                raise RuntimeError(f"[=] Insufficient disk space ({disk.free / 1024 / 1024:.0f}MB)")
            
            # Proceed with normal startup
            self._thread = threading.Thread(
                target=self._run_event_loop,
                name="AsyncLoopThread",
                daemon=True
            )
            self._thread.start()
            
            # Wait for loop to start
            start_time = time.time()
            while self._loop is None and (time.time() - start_time) < timeout:
                time.sleep(0.01)

            print(f"[=] Event loop started in {time.time() - start_time:.2f}s")
            if self._loop is None:
                raise RuntimeError("[-] Event loop failed to start")
            
            self._start_queue_worker(method=method)
            self._start_health_monitor()
        except Exception as e:
            logger.error(f"[-] Startup failed due to resource limits: {e}")
            with self._lock:
                self._state = WrapperState.ERROR
                self._last_heartbeat = time.time()


    def _run_event_loop(self):
        # Run the async event loop in background thread.
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_forever()
        except Exception as e:
            logger.error(f"[-] Event loop crashed: {e}")
        finally:
            self._loop = None
    
    def _start_queue_worker(self, method):
        # Start worker thread for processing request queue.
        def process_queue():
            while self.is_running:
                try:
                    # Get request with timeout to allow checking state
                    request = self._request_queue.get(timeout=1.0)
                    self._submit_request(request, method=method)
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"[-] Queue worker error: {e}")
        
        self._queue_worker = threading.Thread(
            target=process_queue,
            name="QueueWorker",
            daemon=True
        )
        self._queue_worker.start()
    
    def _start_health_monitor(self):
        # Start health monitoring thread.
        def monitor():
            while self.is_running:
                time.sleep(self._health_check_interval)
                logger.info("[.] Checking health...")
                self._check_health()
        
        self._health_thread = threading.Thread(
            target=monitor,
            name="HealthMonitor",
            daemon=True
        )
        self._health_thread.start()
    
    def _check_health(self):
        # Check health of the wrapper and its components.
        now = time.time()
        
        # Check event loop responsiveness
        if self._loop and self._loop.is_running():
            self._last_heartbeat = now
        elif self._state == WrapperState.RUNNING:
            logger.warning("[-] Event loop not responding, attempting recovery")
            self._recover()
        
        # Check for stuck tasks
        with self._task_lock:
            stuck_tasks = [
                task for task in self._pending_tasks.values()
                if (now - task.created_at) > task.timeout * 2
            ]
            
            for task in stuck_tasks:
                logger.warning(f"[-] Cancelling stuck task {task.id}")
                task.future.cancel()
                del self._pending_tasks[task.id]
        
        # Update stats
        with self._lock:
            self._stats['queue_size'] = self._request_queue.qsize()
    
    def _recover(self):
        # Attempt to recover from failure.
        logger.warning("[==] Attempting recovery...")
        with self._lock:
            if self._state != WrapperState.RUNNING:
                return
        
        try:
            # Stop current loop
            if self._loop and self._loop.is_running():
                self._loop.call_soon_threadsafe(self._loop.stop)
            
            # Restart
            self.stop()
            time.sleep(1)
            self.start()
            
        except Exception as e:
            logger.error(f"[-] Recovery failed: {e}")
            with self._lock:
                self._state = WrapperState.ERROR
    
    def predict(self, texts, timeout: float = None, retries: int = None, api_key: str = None, client_ip: str = None, method: str = 'advanced') -> Any:
        """
        Synchronous prediction with layered security retry logic.
        
        Args:
            texts: List of input texts to predict
            timeout: Timeout in seconds (default: self.default_timeout)
            retries: Number of retries on failure (default: self.max_retries)
            method: Prediction method to use (default: 'advanced')

        Returns:
            Prediction result dictionary
        """
            
        try:
            if not self.pipeline.autonomous: 
                print('[=+=] Initiating Autonomous prediction handling...')
                self.pipeline.autonomous = True
                self.pipeline.ensemble.explainer.supervised_learning = False

            # Security checks
            is_admin = self._is_admin_token(api_key)
            if not self._check_ip_allowed(client_ip or 'unknown', is_admin=is_admin):
                raise SecurityError("[==] IP not allowed")
            
            if not self._check_rate_limit(client_ip, is_admin=is_admin):
                raise SecurityError("[==] Rate limit exceeded")
            
            if not self._authenticate(api_key, client_ip, is_admin=is_admin):
                raise SecurityError("[==] Authentication failed")
                
            # Input validation
            try:
                try:
                    if not 'test_titles' in texts and 'label_map' in texts and 'rules' in texts and 'X' in texts and 'y' in texts:
                        for i in range(len(texts)):
                            if isinstance(texts[i], tuple):
                                texts[i] = texts[i][0]  # Extract text from tuple if needed
                                sanitized_texts = self.sanitizer.sanitize_text(texts[i], self.config.max_text_length)
                            else:
                                sanitized_texts = self.sanitizer.sanitize_text(texts[i], self.config.max_text_length)
                    else:
                        sanitized_texts = texts  # Will handle advanced case separately

                except (IndexError, TypeError):
                    # partial sanitization failure, try to sanitize first text if possible and proceed with original texts
                    sanitized_texts = self.sanitizer.sanitize_text(texts[0][0], self.config.max_text_length)

                # texts validated and sanitized at this point, can proceed with original texts for prediction
                if sanitized_texts is None:
                    raise SecurityError("[==] Input validation failed - empty text")

            except SecurityError as e:
                self._audit('input_rejected', {'reason': str(e), 'original_length': len(texts)}, client_ip)
                raise
            
            # Check pending tasks limit
            with self._task_lock:
                if len(self._pending_tasks) >= self.config.max_pending_tasks:
                    self._audit('resource_limit', {'reason': 'max_pending_tasks'}, client_ip)
                    raise SecurityError("[--] Server at capacity - too many pending requests")      

            if not self.is_running:
                if not self.start():
                    raise RuntimeError("[-] AsyncWrapper not running and failed to start")
            
            timeout = timeout or self.default_timeout
            retries = retries or self.max_retries
            
            with self._lock:
                self._stats['total_requests'] += 1
            
            start_time = time.time()
            
            for attempt in range(retries):
                try:
                    # Submit request and wait for result
                    if method != 'advanced':
                        if isinstance(texts, tuple):
                            texts = texts[0]
                            
                        result = self._predict_sync(texts, timeout)

                    elif method == 'singlepass':
                        if isinstance(texts, tuple):
                            texts = texts[0]
                        result = self._predict_single_sync(texts, timeout)
                    else:
                        if 'test_titles' in texts and 'label_map' in texts and 'rules' in texts and 'X' in texts and 'y' in texts:
                            result = self._advanced_predict_sync(texts['test_titles'], texts['label_map'], texts['rules'], texts['X'], texts['y'], texts.get('agent_id', 'default'), texts.get('use_transformer', False), timeout)
                        else:
                            if isinstance(texts, tuple):
                                texts = texts[0]

                            result = self._predict_sync(texts, timeout)

                    # Update success stats
                    elapsed = time.time() - start_time
                    with self._lock:
                        self._stats['successful_requests'] += 1
                        # Update moving average
                        alpha = 0.1
                        self._stats['avg_response_time'] = (
                            alpha * elapsed + 
                            (1 - alpha) * self._stats['avg_response_time']
                        )
                    
                    return result
                    
                except FutureTimeoutError:
                    logger.warning(f"[-] Request timed out (attempt {attempt + 1}/{retries})")
                    self._stats['timed_out_requests'] += 1
                    self._audit('prediction_failed', {'error': 'Request timed out', 'text_preview': texts}, client_ip)                

                    if attempt == retries - 1:
                        raise TimeoutError(f"[-] Prediction timed out after {timeout}s")
                        
                except Exception as e:
                    logger.warning(f"[-] Request failed (attempt {attempt + 1}/{retries}): {e}")
                    traceback.print_exc()
                    self._stats['failed_requests'] += 1
                    self._audit('prediction_failed', {'error': str(e), 'text_preview': texts}, client_ip)                
                    if attempt == retries - 1:
                        raise
            
            # Should never reach here
            raise RuntimeError("[-] Unexpected error in retry loop")
        except Exception as e:
            print(f'[-] Error in predict function: {e}')
            traceback.print_exc()


    def _predict_sync(self, text: Any, timeout: float) -> Any:
        # Internal synchronous prediction.

        if not self._loop or not self._loop.is_running():
            raise RuntimeError("[-] Event loop not available")

    
        future = asyncio.run_coroutine_threadsafe(
            self._predict_with_timeout(text, timeout),
            self._loop
        )
        
        # Track task for cleanup
        task_id = self._add_task(future, timeout)
        
        try:
            result = future.result(timeout=timeout + 1)
            return result
        finally:
            self._remove_task(task_id)

    async def _predict_with_timeout(self, text: Any, timeout: float) -> Any:
        # Async prediction with timeout.
        try:
            return await asyncio.wait_for(
                self.pipeline.predict_async_await(text),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            raise FutureTimeoutError(f"[-] Prediction timed out after {timeout}s")
    


    def _advanced_predict_sync(self, test_titles, label_map, rules, X=None, y=None, agent_id: str=None, use_transformer: bool=False, timeout: float = 30.0) -> Any:
        # Internal synchronous prediction.

        if not self._loop or not self._loop.is_running():
            raise RuntimeError("[-] Event loop not available")

    
        future = asyncio.run_coroutine_threadsafe(
            self.advanced_predict_async_await(test_titles, label_map, rules, X=X, y=y, use_transformer=use_transformer, agent_id=agent_id, timeout=timeout),
            self._loop
        )
        
        # Track task for cleanup
        task_id = self._add_task(future, timeout)
        
        try:
            result = future.result(timeout=timeout + 1)
            return result
        finally:
            self._remove_task(task_id)
    
    
    async def advanced_predict_async_await(self, test_titles: list[tuple], label_map: dict, rules: list[tuple], X: np.ndarray=None, y: np.ndarray=None, use_transformer: bool=False, agent_id: str=None, timeout: float = 30.0):
        # Async advanced prediction with await support. 
        try:
            return await asyncio.wait_for(
                self.pipeline.distribution.request_advanced_prediction_async(
                    self.prediction_manager,
                    use_transformer=use_transformer,
                    agent_id=agent_id,
                    test_titles=test_titles,
                    label_map=label_map,
                    rules=rules,
                    X=X, y=y,
                    timeout=timeout
                ),
                timeout=timeout+5
            )

        except asyncio.TimeoutError:
            raise FutureTimeoutError(f"[-] Advanced prediction timed out after {timeout}s")
        except Exception as e:
            print(f'[!] Error in asynchronous advanced prediction: {e}')
                     
    # ============ ADMIN FUNCTIONS (with authentication) ============
    
    def _initialize_bootstrap_security(self):
        # Initialize bootstrap security on first startup
        if self.config.require_bootstrap_auth:
            # Check if bootstrap token exists
            if os.path.exists(self._bootstrap_token_file):
                with open(self._bootstrap_token_file, 'r') as f:
                    self._bootstrap_token_hash = f.read().strip()
            else:
                # Generate first-time bootstrap token
                new_token = secrets.token_urlsafe(32)
                token_hash = hashlib.sha256(new_token.encode()).hexdigest()
                
                with open(self._bootstrap_token_file, 'w') as f:
                    f.write(token_hash)
                
                print("\n" + "="*60)
                print("🔐 FIRST TIME BOOTSTRAP TOKEN GENERATED")
                print("="*60)
                print(f"[=] TOKEN: {new_token}")
                print("\n⚠️  SAVE THIS TOKEN SECURELY - YOU WILL NEED IT TO START THE SERVICE")
                print("="*60 + "\n")
                
                self._bootstrap_token_hash = token_hash
    
    def _validate_bootstrap_token(self, provided_token: str) -> bool:
        # Validate the bootstrap token for service startup
        if not self.config.require_bootstrap_auth:
            return True
        
        if not provided_token:
            logger.error("[=] Bootstrap token required but not provided")
            return False
        
        if not self._bootstrap_token_hash:
            logger.error("[=] No bootstrap token configured")
            return False
        
        provided_hash = hashlib.sha256(provided_token.encode()).hexdigest()
        
        # Use constant-time comparison to prevent timing attacks
        return secrets.compare_digest(provided_hash, self._bootstrap_token_hash)
        
    def regenerate_bootstrap_token(self, current_token: str, admin_token: str = None) -> str:
        # Regenerate bootstrap token (requires current token or admin)
        
        # Allow either current bootstrap token OR admin token
        if not (self._validate_bootstrap_token(current_token) or 
                self._verify_admin(admin_token)):
            raise SecurityError("Valid bootstrap token or admin token required")
        
        new_token = secrets.token_urlsafe(32)
        new_hash = hashlib.sha256(new_token.encode()).hexdigest()
        
        # Save new token
        with open(self._bootstrap_token_file, 'w') as f:
            f.write(new_hash)
        
        self._bootstrap_token_hash = new_hash
        
        self._audit('bootstrap_token_regenerated', {
            'by_admin': bool(admin_token),
            'by_bootstrap': bool(current_token)
        })

        return new_token



    def generate_api_key(self, metadata: dict = None, admin_token: str = None) -> str:
        # Generate a new API key - requires admin token
        if not self._verify_admin(admin_token, AdminRole.ADMIN):
            self._audit('unauthorized_admin_access', {'action': 'generate_api_key'}, 'admin')
            raise SecurityError("Admin authentication required")
        
        api_key = self.api_key_manager.generate_key(metadata)
        self._audit('api_key_generated', {'metadata': metadata}, 'admin')
        self._save_state()
        return api_key
    
    def revoke_api_key(self, api_key: str, admin_token: str = None) -> bool:
        # Revoke an API key - requires admin token
        if not self._verify_admin(admin_token, AdminRole.ADMIN):
            self._audit('unauthorized_admin_access', {'action': 'revoke_api_key'}, 'admin')
            raise SecurityError("Admin authentication required")
        
        result = self.api_key_manager.revoke_key(api_key)
        if result:
            self._audit('api_key_revoked', {}, 'admin')
            self._save_state()
        return result
    
    def add_allowed_ip(self, ip: str, admin_token: str = None):
        # Add IP to whitelist - supports CIDR (requires operator+)
        if not self._verify_admin(admin_token, AdminRole.OPERATOR):
            self._audit('unauthorized_admin_access', {'action': 'add_allowed_ip'}, 'admin')
            raise SecurityError("Operator authentication required")
        
        # Validate CIDR or IP format
        try:
            if '/' in ip:
                ipaddress.ip_network(ip, strict=False)
            else:
                ipaddress.ip_address(ip)
        except ValueError:
            raise SecurityError(f"Invalid IP or CIDR format: {ip}")
        
        if ip not in self.config.allowed_ips:
            self.config.allowed_ips.append(ip)
            self._audit('ip_whitelisted', {'ip': ip}, 'admin')
            self._save_state()
    
    def remove_allowed_ip(self, ip: str, admin_token: str = None):
        # Remove IP from whitelist - requires admin
        if not self._verify_admin(admin_token, AdminRole.ADMIN):
            raise SecurityError("Admin authentication required")
        
        if ip in self.config.allowed_ips:
            self.config.allowed_ips.remove(ip)
            self._audit('ip_removed_from_whitelist', {'ip': ip}, 'admin')
            self._save_state()
    
    def block_ip(self, ip: str, admin_token: str = None):
        # Block an IP address - requires operator+
        if not self._verify_admin(admin_token, AdminRole.OPERATOR):
            raise SecurityError("Operator authentication required")
        
        # Validate IP format
        try:
            ipaddress.ip_address(ip)
        except ValueError:
            raise SecurityError(f"Invalid IP format: {ip}")
        
        if ip not in self.config.blocklisted_ips:
            self.config.blocklisted_ips.append(ip)
            self._audit('ip_blocked', {'ip': ip}, 'admin')
            self._save_state()
    
    def unblock_ip(self, ip: str, admin_token: str = None):
        # Unblock an IP address - requires admin
        if not self._verify_admin(admin_token, AdminRole.ADMIN):
            raise SecurityError("Admin authentication required")
        
        if ip in self.config.blocklisted_ips:
            self.config.blocklisted_ips.remove(ip)
            self._audit('ip_unblocked', {'ip': ip}, 'admin')
            self._save_state()
    
    def get_audit_log(self, limit: int = 100, admin_token: str = None) -> List[Dict]:
        # Get recent audit log entries - requires auditor+
        if not self._verify_admin(admin_token, AdminRole.AUDITOR):
            raise SecurityError("Auditor authentication required")
        
        return self._audit_log[-limit:]
    
    def list_api_keys(self, admin_token: str = None) -> List[Dict]:
        # List all API keys (without revealing full keys) - requires admin
        if not self._verify_admin(admin_token, AdminRole.ADMIN):
            raise SecurityError("Admin authentication required")
        
        return [
            {
                'key_hash': k[:8] + '...',  # Partial hash only
                'created_at': v['created_at'].isoformat(),
                'last_used': v.get('last_used', '').isoformat() if v.get('last_used') else None,
                'metadata': v.get('metadata', {}),
                'is_active': v.get('is_active', True)
            }
            for k, v in self.api_key_manager.keys.items()
        ]
    
    def create_admin_token(self, role: AdminRole = AdminRole.OPERATOR, admin_token: str = None) -> str:
        # Create a new admin token - requires existing admin token
        if not self._verify_admin(admin_token, AdminRole.ADMIN):
            raise SecurityError("Admin authentication required")
        
        new_token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(new_token.encode()).hexdigest()
        
        self.admin_keys[token_hash] = {
            'role': role,
            'created_at': datetime.now(),
            'created_by': hashlib.sha256(admin_token.encode()).hexdigest()[:8]
        }
        
        self._audit('admin_token_created', {'role': role.value}, 'admin')
        self._save_state()
        
        return new_token
    
    def revoke_admin_token(self, token_to_revoke: str, admin_token: str = None):
        # Revoke an admin token - requires admin
        if not self._verify_admin(admin_token, AdminRole.ADMIN):
            raise SecurityError("Admin authentication required")
        
        token_hash = hashlib.sha256(token_to_revoke.encode()).hexdigest()
        if token_hash in self.admin_keys:
            del self.admin_keys[token_hash]
            self._audit('admin_token_revoked', {}, 'admin')
            self._save_state()


    def advanced_batch_prediction(self, test_titles, label_map, rules,
                                X=None, y=None, api_key=None, client_ip=None):
        try:
            eps = 1e-5
            attn_weights     = None
            final_idx        = None
            attn_weight_rate = None
            confidence       = None    
            chosen_label     = None
            performance_score = self.pipeline.performance_result

            reverse_label_map = {v: k for k, v in label_map.items()}
            n_classes         = len(label_map)

            if 'attn_weights' in self.pipeline.model2.cache:
                attn_weights = self.pipeline.model2.cache['attn_weights']
            if attn_weights is not None:
                attn_weight_growth = 1.0 / (1.0 + np.exp(-attn_weights[0]))
                attn_weight_rate   = float(np.std(attn_weight_growth))

            texts           = [text[0] for text in test_titles]
            expected_labels = [text[1] for text in test_titles]
            predicted_output = []

            text_payload = {
                "test_titles"    : test_titles,
                "label_map"      : label_map,
                "rules"          : rules,
                "X"              : X,
                "y"              : y,
                "use_transformer": True
            }

            if not self.pipeline.intents:
                result     = self.predict(
                    text_payload,
                    timeout=self.pipeline.timeout,
                    retries=None,
                    api_key=api_key,
                    client_ip=client_ip,
                )
                confidence = None   # no confidence from this path
            else:
                result, chosen_label, confidence = self.advanced_prediction_method(
                    self.prediction_manager, test_titles, label_map, rules,
                    X=X, y=y, method='Transformer_included'
                )
                try:
                    if isinstance(result, list) and len(result) > 0:
                        final_idx = result[0].get('index')
                    elif isinstance(result, dict):
                        final_idx = result.get('index')
                except (KeyError, IndexError):
                    final_idx = None

            print(f'[=+=] Local single advanced prediction: {result.get("predicted", "N/A") if isinstance(result, dict) else result} With confidence: {confidence if confidence is not None else "N/A"}')
            print('[=] Initiating batch prediction...')

            results = self.predict_batch(
                texts=texts,
                timeout=self.pipeline.timeout,
                api_key=api_key
            )


            # class reputation track: how many times each class
            # was predicted wrong vs total predictions so far
            # error_rate[c] ∈ [0,1] — higher = this class is unreliable recently
            error_counts = np.zeros(n_classes, dtype=np.float64)
            pred_counts  = np.zeros(n_classes, dtype=np.float64)
            decay        = self.pipeline.error_decay   # how fast old errors fade — tunable

            print("========📊 PREDICTION RESULTS============")

            for idx, (text, expected, probs) in enumerate(
                zip(texts, expected_labels, results)
            ):
                probs = np.asarray(probs, dtype=np.float64).copy()

                # ── calibrate with single-prediction context ──────────────
                if (confidence is not None and
                    attn_weight_rate is not None and
                    final_idx is not None):

                    if 0 <= final_idx < len(probs):
                        boost = (1.0 - attn_weight_rate) * confidence
                        probs[final_idx] = min(probs[final_idx] + boost, 0.95)
                    else:
                        boost = (1.0 - attn_weight_rate) * confidence
                        probs = np.clip(probs + boost, 0.0, 0.95)

                # ── apply class reputation penalty ────────────────────────
                # ounded reputation-based penalty
                # classes with high recent error rate get their probs dampened
                for c in range(n_classes):
                    if pred_counts[c] > 0:
                        error_rate    = error_counts[c] / (pred_counts[c] + 1e-8)
                        # sigmoid-shaped dampening — never goes negative
                        # error_rate=0.0 → multiplier=1.0 (no change)
                        # error_rate=0.5 → multiplier≈0.67
                        # error_rate=1.0 → multiplier≈0.5
                        reputation    = 1.0 / (1.0 + error_rate)
                        probs[c]     *= reputation

                # renormalize after reputation dampening
                prob_sum = probs.sum()
                if prob_sum > 1e-8:
                    probs /= prob_sum

                predicted_index = int(np.argmax(probs))
                predicted_label = reverse_label_map.get(
                    predicted_index, f"class_{predicted_index}"
                )
                pred_conf = float(probs[predicted_index])
                if pred_conf > 0.8:
                    pred_conf = (pred_conf + performance_score) + eps / 2 

                top_3_indices = np.argsort(probs)[-3:][::-1]
                top_3 = [
                    (reverse_label_map.get(int(i), f"class_{i}"), float(probs[i]))
                    for i in top_3_indices
                ]

                print(f"\n📌 Input: '{text}'")
                print(f"   [=] Expected: {expected}")
                print(f"   🎯 Predicted: {predicted_label} ({pred_conf:.1%})")
                print(f"   🔍 Top possibilities:")
                for label, conf in top_3:
                    bar = '█' * int(conf * 20)
                    print(f"[•]   {label:<25} {bar} {conf:.1%}")

                print('===== COMPARISON MATCHING =====')
                is_correct = predicted_label == expected

                # update class reputation — decay old counts first..
                error_counts *= decay
                pred_counts  *= decay

                pred_counts[predicted_index] += 1.0
                if not is_correct:
                    print(f"[❌] INCORRECT (expected: {expected})")
                    error_counts[predicted_index] += 1.0
                else:
                    print(f"[✅] CORRECT!")

                predicted_output.append(
                    f'{text} -> {predicted_label} With {pred_conf:.1%} Confidence'
                )

            # batch accuracy summary
            correct = sum(
                1 for text, exp in zip(texts, expected_labels)
                for po in [predicted_output]
                if f'-> {exp}' in po
            )
            print(f"\n[=] Batch complete")
            print(f"[=] Stats: {self.get_stats()}")
            print('[=] Returning predicted output.')
            return predicted_output

        except Exception as e:
            print(f'[=] Error in advanced batch prediction: {e}')
            traceback.print_exc()
            return []
            
    
    def predict_batch(self, texts: List[str], timeout: float = None, api_key: Any=None) -> List[Dict[str, Any]]:
        """
        Synchronous batch prediction.
        
        Args:
            texts: List of input texts
            timeout: Timeout per request
            
        Returns:
            List of prediction results
        """
        if not self.is_running:
            if not self.start():
                raise RuntimeError("[-] Wrapper not running and failed to start")
        
        results = []
        for text in texts:
            result = self.predict(text, timeout, None, api_key)
            results.append(result)
        return results

    def single_pass_predict_batch(self, texts, timeout, api_key=None):
        """
        Single-pass batch prediction without ensemble or advanced features.
        """
        if not self.is_running:
            if not self.start():
                raise RuntimeError("[-] Wrapper not running and failed to start")   

        results = []     
        for text in texts:
            result = self.predict(text, timeout, api_key, method='singlepass')
            results.append(result)
        return results

    def advanced_prediction_method(self, manager, test_titles, label_map, rules, X=None, y=None, method='Transformer_included'):
        # starting PredictionManager for advanced prediction
        try:
            if method == 'Transformer_included':
                print('== PREDICTION 1: (advanced predictions with expected labels transformer included)')
                result, chosen_label, confidence = manager.advanced_prediction_method(
                test_titles,  # Titles with expected labels
                label_map,
                rules,
                X=X, y=y,
                show_proba=True,
                top_k=4,
                use_transformer=self.pipeline.use_transformer,
                return_attention=False,
            
                )   
            else:
                print('== PREDICTION 2: (titles only without transformer) ==')
                result, chosen_label, confidence = manager.advanced_prediction_method(
                [t[0] for t in test_titles],  # Just titles
                label_map,
                rules,
                X=X, y=y,
                show_proba=True
                )
        except Exception as e:
            logger.error(f"[-] Advanced prediction failed: {e}")
            result = None
            confidence = 0.0
            chosen_label = None

        return result, chosen_label, confidence


    def predict_async(self, text: Any, test_titles: list[tuple]= None, label_map: dict=None, rules: list[tuple]=None, callback: Optional[Callable] = None) -> str:
        """
        Asynchronous prediction (fire and forget).
        
        Args:
            text: Input text to predict
            callback: Optional callback for result
            
        Returns:
            Request ID for tracking
        """
        if not self.is_running:
            self.start()
        
        with self._task_lock:
            request_id = f"[=] req_{self._task_counter}"
            self._task_counter += 1
        
        # Queue the request (non-blocking)
        try:
            print('[===] request Queued... ')
            self._request_queue.put_nowait({
                'id': request_id,
                'text': text,
                'test_titles': test_titles,
                'label_map': label_map,
                'rules': rules,
                'callback': callback
            })
        except queue.Full:
            logger.warning(f"[=] Request queue full, rejecting request {request_id}")
            if callback:
                callback({'error': 'queue_full', 'success': False})
            return request_id
        
        return request_id
    
    def _submit_request(self, request: Dict, method=None):
        # Submit a request from the queue to the event loop.
        if not self._loop or not self._loop.is_running():
            logger.error("[-] Cannot submit request: event loop not available")
            if request.get('callback'):
                request['callback']({'error': 'event_loop_unavailable', 'success': False})
            return

        if method != 'advanced':
            future = asyncio.run_coroutine_threadsafe(
                self._predict_with_timeout(request['text'], self.default_timeout),
                self._loop
            )
        else:
            future = asyncio.run_coroutine_threadsafe(
                self.advanced_predict_async_await(
                    test_titles=request['test_titles'],
                    label_map=request['label_map'],
                    rules=request['rules'],
                    X=request['X'],
                    y=request['y'],
                    timeout=self.default_timeout
                ),
                self._loop
            )
        
        task_id = self._add_task(future, self.default_timeout)
        
        def on_completion(fut):
            self._remove_task(task_id)
            try:
                result = fut.result()
                if request.get('callback'):
                    request['callback'](result)
            except Exception as e:
                logger.error(f"[-] Async request failed: {e}")
                if request.get('callback'):
                    request['callback']({'error': str(e), 'success': False})
        
        future.add_done_callback(on_completion)
    
    def _add_task(self, future: asyncio.Future, timeout: float) -> str:
        # Track a pending task.
        with self._task_lock:
            task_id = f"[=] task_{self._task_counter}"
            self._task_counter += 1
            self._pending_tasks[task_id] = AsyncTask(
                id=task_id,
                future=future,
                created_at=time.time(),
                timeout=timeout
            )
        return task_id
    
    def _remove_task(self, task_id: str):
        # Remove a completed task.
        with self._task_lock:
            self._pending_tasks.pop(task_id, None)
    
    def get_stats(self) -> Any:
        # Get wrapper statistics.
        with self._lock:
            stats = self._stats.copy()
            stats['state'] = self.state
            stats['pending_tasks'] = len(self._pending_tasks)
            stats['queue_size'] = self._request_queue.qsize()
            stats['loop_running'] = self._loop and self._loop.is_running()
            stats['uptime'] = time.time() - self._last_heartbeat if self._last_heartbeat else 0
        return stats
    
    def wait_for_idle(self, timeout: float = 30.0) -> bool:
        """
        Wait for all pending tasks to complete.
        
        Args:
            timeout: Maximum time to wait
            
        Returns:
            True if idle, False if timeout
        """
        start_time = time.time()
        while (self._pending_tasks or self._request_queue.qsize() > 0) and \
              (time.time() - start_time) < timeout:
            time.sleep(0.1)
        
        return len(self._pending_tasks) == 0 and self._request_queue.qsize() == 0
    
    def stop(self, timeout: float = 10.0, force: bool = False) -> bool:
        '''
        Gracefully stop the wrapper.
        
        Args:
            timeout: Maximum time to wait for pending tasks
            force: Force stop even if tasks pending
            
        Returns:
            True if stopped successfully
        '''

        with self._lock:
            if self._state in (WrapperState.STOPPING, WrapperState.STOPPED):
                return True
            
            self._state = WrapperState.STOPPING
        
        logger.info("[-] Stopping PipelineAsyncManager...")
        
        # Wait for pending tasks if not forcing
        if not force:
            self.wait_for_idle(timeout)
        
        # Stop event loop
        if self._loop and self._loop.is_running():
            # Cancel all pending tasks
            for task in self._pending_tasks.values():
                task.future.cancel()
            
            # Stop the loop
            self._loop.call_soon_threadsafe(self._loop.stop)
        
        # Wait for thread to finish
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        
        # Wait for queue worker
        if self._queue_worker and self._queue_worker.is_alive():
            self._queue_worker.join(timeout=timeout)
        
        # Wait for health monitor
        if self._health_thread and self._health_thread.is_alive():
            self._health_thread.join(timeout=timeout)
        
        with self._lock:
            self._state = WrapperState.STOPPED
            self._loop = None
            self._thread = None
            self._queue_worker = None
            self._health_thread = None
        print('[=] PipelineAsync Wrapper stopped')
        logger.info("[-] PipelineAsync Wrapper stopped")
        return True
    
    def __enter__(self):
        # Context manager entry.
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Context manager exit.
        self.stop()
    
    def __del__(self):
        # Destructor for cleanup.
        try:
        # Only attempt cleanup if we have the attribute
            if hasattr(self, '_state'):
                if self._state not in (WrapperState.STOPPED, WrapperState.UNINITIALIZED):
                    # Use force=True since we're in destructor
                    self.stop(force=True)
        except AttributeError:
            # Object is already partially destroyed - nothing to clean
            pass
        except Exception as e:
            logger.debug(f"[=] Cleanup error in __del__: {e}")



class PipelinePredictionManager:
    def __init__(self, pipeline, label_csv='labels.csv', target_title='title', label='label'):
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
        try:
            dataset = None
            X_gen = None
            use_embedded = False
            attn_weights = None
            trans_probs = None
            mlp_probs = None

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
                lstm_probs, _ = self.pipeline.ensemble._get_lstm_probs(input_ids, X_gen, label_bins=label_bins)      
            else:
                lstm_probs = None


            if use_transformer and hasattr(self.pipeline, 'vocab') and self.pipeline.vocab:
                print("[🔄] Using Transformer for probability calibration")

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
                print("[⚡] Using MLP only for predictions")
                trans_probs = None
        
            if not hasattr(self.pipeline, 'tfidf') or self.pipeline.tfidf is None:
                self.pipeline.initialize_fitting(titles)
            
            # Prepare texts for MLP
            if titles is not None and len(titles) > 0:
                if isinstance(titles[0], tuple):
                    mlp_titles = [t[0] for t in titles]
                else:
                    mlp_titles = titles
                    
                X_tfidf = self.pipeline.tfidf.transform(mlp_titles).toarray() 
            else:
                X_tfidf = X

            # Forward pass through MLP
            if hasattr(self.pipeline.mlp, 'predict_proba'):
                mlp_probs = self.pipeline.mlp.predict_proba(X_tfidf)
            else:
                # Fallback if predict_proba not available
                logits = self.pipeline.mlp.forward(X_tfidf)
                mlp_probs = self.pipeline._softmax(logits)
                
            # Validate all MLP predictions at once
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
                    
                    # MLP prediction
                    mlp_class_idx = mlp_pred_indices[i]
                    mlp_class_idx = min(mlp_class_idx, num_classes - 1)  # Clamped to valid range
                    
                    mlp_confidence = mlp_probs[i][mlp_class_idx]
                    mlp_label = reverse_map.get(mlp_class_idx, f"unknown_{mlp_class_idx}")
                    anisotropy = self.pipeline.anisotropy_measurement(input_ids)
                    anisotropic_rate = 1.0 / (1.0 + np.exp(-anisotropy)) if anisotropy is not None else 1.0

                    # Transformer prediction (if available)
                    if trans_probs is not None:
                        if trans_probs.shape[0] > i:
                            trans_probs_i = trans_probs[i]
                        else:
                            trans_probs_i = trans_probs[-1]  # fallback to last if mismatch
                        
                        trans_class_idx = np.argmax(trans_probs_i)
                        trans_confidence = trans_probs_i[trans_class_idx]
                        trans_label = reverse_map.get(trans_class_idx, f"unknown_{trans_class_idx}")
                        
                        # Calibrated probabilities (blend of MLP and Transformer)
                        if use_transformer:
                            # Boost MLP's prediction in transformer probabilities
                            calibrated = trans_probs_i.copy()
                            try:
                                calibrated[mlp_class_idx] = max(calibrated[mlp_class_idx], anisotropic_rate)
                                calibrated /= calibrated.sum()
                            except Exception as e:
                                calibrated = self.pipeline._calibrate_probs(mlp_probs, mlp_pred_indices, attn_weights, input_ids)
            
                            final_probs = calibrated
                            final_class_idx = mlp_class_idx  # Trust MLP's class decision
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
                print("🎯 HYBRID PREDICTION RESULTS (MLP + Transformer)")
                print("="*70)
                
                correct_count = 0
                for result in results:
                    print(f"\n📌 '{result['title']}'")
                    
                    if result.get('expected'):
                        status = "✓" if result['predicted'] == result['expected'] else "✗"
                        print(f"   Expected: {result['expected']} {status}")
                    
                    print(f"   🎯 FINAL PREDICTION: {result['predicted']} ({result['confidence']:.1%})")
                    print(f"   ⚡ MLP: {result['mlp_prediction']} ({result['mlp_confidence']:.1%})")
                    
                    if result.get('transformer_prediction'):
                        arrow = "⬆️" if result['transformer_confidence'] > result['mlp_confidence'] else "⬇️"
                        print(f"   🌀 Transformer: {result['transformer_prediction']} ({result['transformer_confidence']:.1%}) {arrow}")
                    
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

                try:
                    joblib.dump(self.pipeline, 'modular_agent.pkl')
                    print('💾  Model saved!')
                except Exception as e:
                    print(f'|| Failed to joblib dump file! : {e}, User Manual filepath suggestion needed...')

                    try:
                        permission = input('|| Insert Filepath? [Y/N]: ')
                        if permission == 'Y':
                            suggested_path = input('|| Filepath suggestion: ')
                            if suggested_path:
                                self.pipeline.safe_pickle_save_with_feedback(self.pipeline, suggested_path)
                                print('💾  Model saved!')                
                            else:
                                print('|| Failed to dump Your model! ')
                                pass
                        else:
                            print('|| Failed to dump Your model! ')
                            pass  
                    except EOFError as e:
                        print('[!] EOF Error from reading a line!')

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
                print("\n[⚠️] Disagreement detected between MLP and Transformer predictions. Using calibrated probabilities for final decision.")
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



    def robust_prediction(self, pipeline, titles=None, label_map=None, rules=None, X=None, X_raw=None, y=None, show_proba=True, top_k=3, batch_size=2):
        self.pipeline.titles = titles
        self.pipeline.labels = label_map   

        try:

            if titles is not None and rules is not None:
                print(f"[🔍] Preparing data for {len(titles)} titles with {len(rules)} length of rules.")
                if X is None and y is None or X is None or y is None:
                    print('[🔄] Creating automatic X samples because X Samples is not provided manually.')
                    datasets, X_gen = self.pipeline.data_preparation(titles, label_map)  
                    _, y, _, _ = self.pipeline.mlp_training_features(rules, datasets)                  
                else:
                    datasets, _ = self.pipeline.data_preparation(titles, label_map)

            if X_gen is not None:
                self.pipeline.transformer_utilities(X_provided=X, X_raw=X_gen, y_true=y, rules=rules, datasets=dataset, label_map=label_map, batch_size=batch_size)
            else:
                self.pipeline.transformer_utilities(X_provided=X, X_raw=X, y_true=y, rules=rules, datasets=dataset, label_map=label_map, batch_size=batch_size)

            reverse_map = {v: k for k, v in label_map.items()}

            if datasets is not None:
                input_datasets = self.pipeline.transformer_input_encoding(datasets)
            else:
                input_datasets = self.pipeline._features_to_sequence(X)

            pred_probs = self.pipeline.predict_proba(input_datasets, X_raw, type='Hybrid')[0]
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
                        confidence = final_probs[i][class_idx]
                    else:
                        confidence = float(final_probs[i])  # Single value
                        
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
        """
        reverse_map = reverse_map or {}
        num_classes = num_classes or mlp_probs.shape[1]
        models_agree = False

        # ── MLP layer ─────────────────────────────────────────────
        mlp_class_idx  = int(np.argmax(mlp_probs[i]))
        is_valid_index = 0 <= mlp_class_idx < num_classes

        if not is_valid_index:
            return {
                'predicted'  : None,
                'confidence' : 0.0,
                'mlp_class'  : None,
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

        # ── Transformer ───────────────────────────────────────
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
            # no transformer available — MLP/LSTM only
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
            'predicted_idx'     : final_class_idx,
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

    def advanced_prediction_method(self, titles=None, label_map=None, rules=None,
                                X=None, y=None,
                                show_proba=False, top_k=3, 
                                use_transformer=True, return_attention=False,
                                save_results=True, batch_size=2):
        try:
            # ____ init temporary layer ____
            eps = 1e-5
            trans_probs = None
            attn_weights = None
            sequence_ids = None

            input_ids = None
            anisotropy = None
            final_class_idx = None

            AME = None
            use_embedded = False
            dataset = None

            X_gen = None
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

            if titles is not None and rules is not None:
                print(f"[🔍] Preparing data for {len(titles)} titles with {len(rules)} length of rules.")
                if X is None and y is None or X is None or y is None:
                    print('[🔄] Creating automatic X samples because X is not provided manually.')
                    dataset, X_gen = self.pipeline.data_preparation(titles, label_map)  
                    _, y, _, _ = self.pipeline.mlp_training_features(rules, dataset)                  
                else:
                    dataset, _ = self.pipeline.data_preparation(titles, label_map)

            if X_gen is not None:
                self.pipeline.transformer_utilities(X_provided=X, X_raw=X_gen, y_true=y, rules=rules, datasets=dataset, label_map=label_map, batch_size=batch_size)
            else:
                self.pipeline.transformer_utilities(X_provided=X, X_raw=X, y_true=y, rules=rules, datasets=dataset, label_map=label_map, batch_size=batch_size)

            if dataset is not None:
                input_ids, _ = self.pipeline.input_encoding(dataset)
            else:
                input_ids = self.pipeline._features_to_sequence(X)

            if use_transformer and hasattr(self.pipeline, 'vocab') and self.pipeline.vocab:
                print("\n[🔄] Running Transformer prediction method (Transformer)")
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
                        anisotropy = self.pipeline.anisotropy_measurement(X)
                    else:
                        anisotropy = self.pipeline.anisotropy_measurement(sequence_ids)

                    # Get transformer predictions with attention
                    print(f"[⚡] anisotropy rate detected on input: {anisotropy:.1%}.")                
                    use_embedded = True
                    
                    AME = self.pipeline.model2.AME_Encoder(sequence_ids)                
                    trans_probs, attn_weights = self.pipeline.model2.forward(sequence_ids, AME=AME, embedded=use_embedded)

                else:
                    if X is not None:
                        anisotropy = self.pipeline.anisotropy_measurement(X)
                    else:
                        anisotropy = self.pipeline.anisotropy_measurement(input_ids)

                    # Get transformer predictions with attention
                    print(f"[⚡] anisotropy rate detected on input: {anisotropy:.1%}.")                
                    use_embedded = True

                    AME = self.pipeline.model2.AME_Encoder(input_ids) 

                    trans_probs, attn_weights = self.pipeline.model2.forward(input_ids, AME=AME, embedded=use_embedded)
                    
            else:
                print("\n[⚡] Running MLP-only predictions")
                print("[⚡] Note: Transformer not available, so Transformer results will be replaced with MLP results.")

                anisotropy = self.pipeline.anisotropy_measurement(X) if X is not None else self.pipeline.confidence_threshold

                try:
                    AME = self.pipeline.model2.AME_Encoder(X) if X is not None else self.pipeline.confidence_threshold
                except:
                    AME = self.pipeline.confidence_threshold


            if X is None or len(X) == 0 or isinstance(X, int) or (isinstance(X, np.ndarray) and X.size == 0):
                # Get MLP predictions
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

                else:
                    raise ValueError("[!] No valid input data (X samples) provided for MLP predictions!")
                    

            # MLP forward pass  
            if hasattr(self.pipeline.mlp, 'predict_proba'):
                mlp_probs = self.pipeline.mlp.predict_proba(X)
            else:
                logits = self.pipeline.mlp.forward(X)
                mlp_probs = self.pipeline._softmax(logits)
            
             # Validate all MLP predictions at once
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
            else:
                print('[=] No verified output from cache available that matched samples, starting advanced prediction...')
                if self.pipeline.use_transformer:
                    if isinstance(input_ids, (list, np.ndarray)):
                        use_embedded = True

                    target_probs = self.pipeline.predict_proba(input_ids, X, type='Hybrid', embedded=use_embedded)
                else:
                    target_probs = mlp_probs
           
                target_probs = target_probs[:mlp_probs.shape[0], :mlp_probs.shape[1]] 
                target_probs = self.pipeline.model3.continuous_predictive_correction(self, target_probs, mlp_pred_indices)  
            
            target_pred_indices = np.argmax(target_probs, axis=1)    

            if self.pipeline.cache and 'label_bins' in self.pipeline.cache:
                print('[=] label_bins cache found!')
                label_bins = self.pipeline.cache['label_bins']
                lstm_probs, _ = self.pipeline.ensemble._get_lstm_probs(input_ids, X, label_bins=label_bins)      
            else:
                lstm_probs = None

            need_ensemble_method = (
                anisotropy > 0.3 and 
                AME is not None and 
                AME > 0.3 and
                np.mean(self.error_counts) > 0.3 
            )
            
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
                    
                    # MLP prediction     
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
                
                    # Transformer prediction and blending
                    if trans_probs is not None and attn_weights is not None:
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
                            # Blend predictions (MLP decides class, transformer calibrates confidence)
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
                                print(f"[🔄] MLP chosen for sample {i} due to highest confidence: {mlp_confidence:.1%}")
                            elif trans_confidence > lstm_confidence:
                                final_probs = trans_probs[i] if i < len(mlp_probs) else trans_probs[0]
                                final_class_idx = trans_class_idx
                                final_confidence = trans_confidence
                                print(f"[🔄] Transformer chosen for sample {i} due to highest confidence: {trans_confidence:.1%}")
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
                                    print(f"[🔄] No model chosen for sample {i} due to low confidence: MLP={mlp_confidence:.1%}, Transformer={trans_confidence:.1%}, LSTM={lstm_confidence:.1%}")

                            agreement = mlp_class_idx == trans_class_idx

                    else:
                        final_probs = mlp_probs[i]

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
                        'index': int(final_class_idx),
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
                        
                        # MLP top predictions
                        mlp_probs_i = mlp_probs[i][:num_classes] if num_classes > 0 else mlp_probs[i]
                        mlp_top = np.argsort(mlp_probs_i)[-top_k:][::-1]
                        result['mlp_top'] = [
                            {
                                'label': reverse_map.get(idx, f"unknown_{idx}"),
                                'confidence': float(mlp_probs_i[idx])
                            }
                            for idx in mlp_top if idx in reverse_map
                        ]
                        
                        # Transformer top predictions
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
                n_samples = mlp_probs.shape[0]

                lstm_pred_indices = np.argmax(lstm_probs, axis=1) if lstm_probs is not None else None
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
            
            chosen_label = results[0]['predicted'] if results else None
            confidence = results[0]['confidence'] if results else None
            if isinstance(chosen_label, int) or isinstance(chosen_label, np.integer):
                chosen_label = str(chosen_label)
                
            if isinstance(confidence, (np.ndarray, list)):
                confidence = np.mean(confidence)

            print(f"\n[🎯] Initial chosen label for input: {chosen_label} || Confidence: {confidence:.1%}")
            time.sleep(3)

            if results[0].get('models_agree', True) and confidence > self.pipeline.confidence_threshold and not chosen_label.startswith("unknown"):
                print(f"\n[🎯] Proper Confidence of Final chosen label for input: {chosen_label} || Confidence: {confidence:.1%}")
                return results, chosen_label, confidence
            
            # Only recalibrate if models disagreed
            elif results and not results[0].get('models_agree', True) or not self.pipeline.agreement:
                need_peer_condition = not results[0].get('models_agree', True) and self.pipeline.peer_assistance_threshold > 0.3
                print("\n[⚠️] Disagreement detected between MLP and Transformer predictions. Using calibrated probabilities for final decision.")
                if not self.pipeline.autonomous and need_peer_condition:
                    print('|| Uncertain advanced prediction, requesting peer assistance if allowed...')
                    final_probs = self.pipeline._handle_distributed_connections(final_probs, attn_weights, input_ids, agreement) 

                    final_idx = final_probs[0].argmax()
                    original_idx = final_idx

                    if final_idx > len(reverse_map):
                        final_idx = int(np.argmax(final_probs[:len(reverse_map)-1]))
                        print(f"[⚠️] Clamping {final_idx} → {final_idx}")               
                    final_idx = int(final_idx)  

                    chosen_label = reverse_map.get(final_idx, f"unknown_{final_idx}")
                    try:
                        print(final_probs)
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
                    attn_quality = 0.05
                    anisotropy = self.anisotropy_measurement(input_ids) if hasattr(self.pipeline, 'anisotropy_measurement') else 0.5

                mean_entropy = np.mean(normalized_entropy)

                use_robust_prediction = (
                anisotropy < 0.3 or
                mean_entropy > 0.5 or  # High uncertainty
                results[0].get('confidence', 0) < 0.4 or  # Low confidence
                not results[0].get('models_agree', True) or  # Disagreement
                attn_quality < 0.4
                )

                if use_robust_prediction:
                    print("\n[⚡] Condition is poorly unviable to handle agreement. Using robust prediction method for better reliability.")
                    predicted_label, confidence = self.robust_prediction(self.pipeline, titles=titles, label_map=label_map, X_raw=X, y=y, show_proba=show_proba, top_k=top_k)
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

            if isinstance(chosen_label, str) and chosen_label.startswith("unknown") or float(confidence) < self.pipeline.confidence_threshold:
                if chosen_label.startswith("unknown"):
                    chosen_label = 'Unknown'
                    confidence = 1.0 - confidence  # Invert confidence for unknown class

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
                        results[0]['index'] = final_idx
                else:
                    print('[!] No prediction in results cache are found!')

            except Exception as e:
                print(f'[!] Error initiating second prediction in Advanced prediction method: {e} ')

                results[0]['sec_predicted'] = chosen_label
                results[0]['sec_confidence'] = confidence
                results[0]['sec_index'] = final_idx

                time.sleep(5)

        except Exception as e:
            print(f"[!] Error in advanced prediction method: {e}, Initiating regular prediction method...")
            traceback.print_exc()            
            try:
                results = self.regular_prediction_method(titles=titles, label_map=label_map, rules=rules, X=X, y=y, show_proba=False, top_k=3, batch_size=2, use_transformer=True)
                chosen_label = results[0]['predicted']
                confidence = results[0]['confidence']
            except Exception as error:
                print(f'[= ! =] Error in all prediction method: {error}')
                traceback.print_exc()
                results, chosen_label, confidence = None, None, 0.0
                time.sleep(5)

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
        
        # delete pipelines cache
        print('[🔍] Pipelines Cache Cleaned!')
        self.pipeline.cache.clear()

        return results, chosen_label, confidence


    def calibration_penalized_check(self, final_probs, predicted_index):
        # update class reputation.

        decay = self.pipeline.error_decay
        self.error_counts *= decay
        self.pred_counts  *= decay

        if final_probs is None:
            print('[!] Warning final probabilities is None! returning the probabilities...')
            return final_probs

        try:

            self.pred_counts[predicted_index] += 1.0
            n_classes = len(self.label_map)

            self.pred_counts = self.pred_counts[0] if isinstance(self.pred_counts[0], np.ndarray) and self.pred_counts.ndim > 1 else self.pred_counts

            for c in range(n_classes):
                if len(self.pred_counts) < c:
                    if isinstance(self.pred_counts[c], (int, float)) and self.pred_counts[c] > 0:
                        error_rate    = self.error_counts[c] / (self.pred_counts[c] + 1e-8)
                        # sigmoid-shaped dampening — never goes negative
                        # error_rate=0.0 → multiplier=1.0 (no change)
                        # error_rate=0.5 → multiplier≈0.67
                        # error_rate=1.0 → multiplier≈0.5
                        reputation    = 1.0 / (1.0 + error_rate)
                        if c < len(final_probs):
                            final_probs[c]  *= reputation  
                else:
                    self.pred_counts = np.zeros(n_classes, dtype=np.float64)
                    self.pred_counts[predicted_index] += 1.0
                    self.pred_counts = self.pred_counts[0] if isinstance(self.pred_counts[0], np.ndarray) and self.pred_counts.ndim > 1 else self.pred_counts

                    if isinstance(self.pred_counts[c], (int, float)) and self.pred_counts[c] > 0:
                        error_rate    = self.error_counts[c] / (self.pred_counts[c] + 1e-8)
                        # sigmoid-shaped dampening — never goes negative
                        # error_rate=0.0 → multiplier=1.0 (no change)
                        # error_rate=0.5 → multiplier≈0.67
                        # error_rate=1.0 → multiplier≈0.5
                        reputation    = 1.0 / (1.0 + error_rate)
                        if c < len(final_probs):
                            final_probs[c]  *= reputation

                    
            prob_sum = final_probs.sum()
            if prob_sum > 1e-8:
                final_probs /= prob_sum

            # re adapt shape of pred_counts and error_counts if they don't match prob shape
            if self.pred_counts.shape != final_probs.shape:
                self.pred_counts = np.zeros_like(final_probs)
            if self.error_counts.shape != final_probs.shape:
                self.error_counts = np.zeros_like(final_probs)

        except Exception as e:
            print(f'[!] Cant check and calibrate probs based on penalty due to: {e}') 
            traceback.print_exc()   
        
        return final_probs 

        
    def display_hybrid_results(self, payload, predicted_index, results, top_k=3, verbose=False):
        print("\n" + "="*80)
        print("[🎯] == PREDICTION RESULTS == ")
        print("="*80)

        correct = 0
        sec_correct = 0
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
                    if result['sec_predicted'] == result['expected']:
                        self.pipeline.accurate_cache_lookup.add_verified(
                            X_samples, input_ids, 
                            result['sec_predicted'], result['sec_confidence'], result['sec_index'],
                            source='automatic_verified')

                        sec_correct += 1
                    else:
                        if isinstance(self.error_counts[predicted_index], (int, float)):
                            self.error_counts[predicted_index] += 1.0

                if result['predicted'] == result['expected']:
                    if result['sec_index'] is None:
                        result['sec_index'] = None

                    self.pipeline.accurate_cache_lookup.add_verified(
                        X_samples, input_ids, 
                        result['predicted'], result['confidence'], result['index'],
                        source='automatic_verified')      

                    correct += 1
                else:
                    if isinstance(self.error_counts[predicted_index], (int, float)):
                        self.error_counts[predicted_index] += 1.0
    
            
            # Agreement indicator
            agree_symbol = "✓" if result.get('models_agree', True) else "⚠️"
            print(f"[=] {agree_symbol} FINAL: {result['predicted']} ({result['confidence']:.1%})")

            # MLP vs Transformer
            print(f"      ├─ [⚡] MLP: {result['mlp_prediction']} ({result['mlp_confidence']:.1%})")
            if result.get('transformer_prediction'):
                arrow = "⬆️" if result['transformer_confidence'] > result['mlp_confidence'] else "⬇️"
                print(f"      └─ [🌀] Transformer: {result['transformer_prediction']} ({result['transformer_confidence']:.1%}) {arrow}")
            
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

class ConsecutivePeerAgent:
    """
    Robust PeerAgent with security layer.
    Used as fallback when main system fails during P2P.
    """
    
    def __init__(self, peer_id: str, port: int, secret_key: str, 
                 manager=None, pipeline=None):
        self.peer_id = peer_id
        self.port = port
        self.secret_key = secret_key
        self.manager = manager  # PipelinePredictionManager
        self.pipeline = pipeline  # IntegratedPipeline
        
        self.connected_peers: Dict[str, Dict] = {}
        self.server_socket: Optional[socket.socket] = None
        self.running = False
        self._lock = threading.RLock()
        
        # Security
        self.allowed_ips = {'127.0.0.1'}
        self.max_message_size = 10 * 1024 * 1024  # 10MB
        
        # Statistics
        self.stats = {
            'predictions': 0,
            'peer_requests': 0,
            'errors': 0
        }
    
    def _sign_message(self, message: dict) -> str:
        """Sign message with HMAC"""
        msg_copy = {k: v for k, v in message.items() if k != 'signature'}
        sorted_msg = {k: msg_copy[k] for k in sorted(msg_copy.keys())}
        msg_bytes = json.dumps(sorted_msg, default=str).encode('utf-8')
        key = self.secret_key.encode()
        return hmac.new(key, msg_bytes, hashlib.sha256).hexdigest()
    
    def _verify_signature(self, message: dict, signature: str) -> bool:
        # Verify message signature
        expected = self._sign_message({k: v for k, v in message.items() if k != 'signature'})

        print(f'[ConsecutivePeerAgent] Comparing Signature and verifying...')
        return hmac.compare_digest(expected, signature)
    
    def _send_message(self, sock: socket.socket, message: dict) -> bool:
        """Send signed message"""
        try:
            if sock is None:
                print('[=] Sock is None !')  
                return None

            msg_copy = message.copy()
            msg_copy['timestamp'] = time.time()
            msg_copy['signature'] = self._sign_message(msg_copy)   

            data = json.dumps(msg_copy, default=str).encode('utf-8')

            sock.sendall(len(data).to_bytes(4, 'big'))
            sock.sendall(data)
            return True
        except Exception as e:
            print(f"[ConsecutivePeerAgent] Send error: {e}")
            return False
    
    def _receive_message(self, sock: socket.socket) -> Optional[dict]:
        """Receive and verify message"""
        try:
            data_len = sock.recv(4)

            print(f'[ConsecutivePeerAgent] Got data length: {data_len}')
            if not data_len:
                return None
            
            msg_len = int.from_bytes(data_len, 'big')
            if msg_len > self.max_message_size:
                return None
            
            data = b''
            while len(data) < msg_len:
                chunk = sock.recv(min(4096, msg_len - len(data)))
                if not chunk:
                    return None
                data += chunk

            try:
                message = json.loads(data.decode('utf-8'))
                print(f'[ConsecutivePeerAgent] Received a message!')
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                logger.warning(f'[=] Invalid JSON from peer: {e}')
                self._log_security_event('invalid_json', {})
                return None

            if 'signature' in message:
                signature = message.pop('signature')
                if not self._verify_signature(message, signature):
                    print(f"[ConsecutivePeerAgent] Invalid signature authentication from message, Message Ignored.")
                    return None
                    
                message['signature'] = signature

            return message
            
        except Exception as e:
            print(f"[ConsecutivePeerAgent] Receive error: {e}")
            return None


    async def predict_local(self, text: Any=None) -> Dict:
        """Predict using local model (advanced prediction)"""
        try:
            # Use your existing advanced prediction method
            if self.manager:
                # For single text, wrap in list
                if 'test_titles' in text:
                    test_titles = text['test_titles']
                    label_map = text['label_map']
                    rules = text['rules']
                    X = text['X']
                    y = text['y']
                    result, chosen_label, confidence = self.manager.advanced_prediction_method(
                        test_titles,
                        label_map,
                        rules,
                        X=X, y=y,
                        show_proba=False,
                        use_transformer=self.pipeline.use_transformer
                    )                    
                else:
                    chosen_label = self.pipeline.predict_single(text)
                    confidence = self.pipeline.confidence_threshold # doubt on simple predictions

                return {
                    'text': text,
                    'prediction': chosen_label,
                    'confidence': confidence,
                    'source': 'local'
                }

            elif not self.manager and self.pipeline:
                result = self.pipeline.predict_single(text)
                return {
                    'text': text,
                    'prediction': result.get('prediction', 'unknown'),
                    'confidence': result.get('confidence', 0.5),
                    'source': 'local'
                }
            else:
                # Fallback simple prediction
                return {
                    'text': text,
                    'prediction': 'unknown',
                    'confidence': 0.5,
                    'source': 'local'
                }
        except Exception as e:
            print(f"[ConsecutivePeerAgent] Local prediction error: {e}")
            return {
                'text': text,
                'prediction': 'error',
                'confidence': 0.0,
                'source': 'local',
                'error': str(e)
            }



    async def request_peer_prediction(self, peer_host: Any, peer_port: int, text: Any, timeout: float = 5.0) -> Optional[Dict]:
        """Request prediction from peer - ONLY sends text!"""
        
        peer_key = f"{peer_host}:{peer_port}"
        
        with self._lock:
            # Check if already connected
            if peer_key not in self.connected_peers:
                # Create new connection
                try:
                    if self.pipeline.distribution.client_ssl_context:
                        print('[+] Prediction Request is Initiated with SSL.')
                        sock = self.pipeline.distribution.client_ssl_context.wrap_socket(
                            socket.socket(socket.AF_INET, socket.SOCK_STREAM),
                            server_hostname=peer_host
                        )
                    else:
                        print('[!] Prediction Request is Initiated without Any SSL!')
                        client_ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
                        client_ctx.check_hostname = False
                        client_ctx.verify_mode = ssl.CERT_NONE  
                        sock = client_ctx.wrap_socket(
                            socket.socket(socket.AF_INET, socket.SOCK_STREAM),
                            server_hostname=peer_host
                        )
        
                    sock.settimeout(timeout)
                    sock.bind(('127.0.0.1', 0))

                    sock.connect((peer_host, peer_port))

                    if peer_host in ['127.0.0.1', 'localhost', 'local'] and peer_port == self.port:
                        print(f"[❌] Requesting to self, ignoring request...")
                        sock.close()
                        return  
                    
                    # Authenticate
                    
                    auth_msg = {
                    'type': 'auth',
                    'peer_id': self.peer_id,
                    'token': self.secret_key
                    }


                    if not self._send_message(sock, auth_msg):
                        print(f"[ConsecutivePeerAgent] Failed to send auth to {peer_key}")                        
                        sock.close()
                        return None   
                    else:
                        print('[ConsecutivePeerAgent] Successfully send Authentication message')                        

                    response = self._receive_message(sock)

                    print(f'[ConsecutivePeerAgent] Got Authentication response from peer!')
                    if not response or response.get('status') != 'ok':
                        sock.close()
                        print('[ConsecutivePeerAgent] Socket is closed!')
                        return None
                    else:
                        print(f'[ConsecutivePeerAgent] Received Response from peer')
                    
                    self.connected_peers[peer_key] = {
                        'sock': sock,
                        'host': peer_host,
                        'port': peer_port,
                        'last_seen': time.time()
                    }
                except Exception as e:
                    print(f"[ConsecutivePeerAgent] Connection to {peer_key} failed: {e}")
                    return None
            
            sock = self.connected_peers[peer_key]['sock']
        
        # Send prediction request 
        try:
            request = {
                'type': 'predict',
                'text': text,
                'peer_id': self.peer_id
            }
            
            sock.settimeout(timeout)
            if not self._send_message(sock, request):
                print('[ConsecutivePeerAgent] Send Prediction request Message Failed!')
                return None
            else:
                print('[ConsecutivePeerAgent] Prediction request Message send successful ')
            
            response = self._receive_message(sock)
            sock.settimeout(None)
            print(f'[ConsecutivePeerAgent] Got Prediction response from peer with address: {peer_host}:{peer_port}')

            if response and response.get('type') == 'predict_response':
                self.stats['peer_requests'] += 1

    
                return {
                    'text': text,
                    'prediction': response.get('prediction'),
                    'confidence': response.get('confidence', 0.0),
                    'source': f"peer_{peer_host}:{peer_port}"
                }
            
            return None
            
        except Exception as e:
            print(f"[ConsecutivePeerAgent] Peer request error: {e}")
            # Clean up dead connection
            with self._lock:
                if peer_key in self.connected_peers:
                    try:
                        self.connected_peers[peer_key]['sock'].close()
                    except:
                        pass
                    del self.connected_peers[peer_key]
            return None
    
    async def ensemble_predict(self, peer_addresses: List[Tuple[str, int]],  text: Any=None,
                                confidence_threshold: float = 0.6) -> Dict:
        """
        Ensemble prediction: local first, then ask peers if confidence is low.
        """
        print(f"[ConsecutivePeerAgent] Starting ensemble prediction with port {self.port}!")
        print(f'[ConsecutivePeerAgent] Peer Addresses: {peer_addresses}')
        
        # Step 1: Local prediction
        local_result = await self.predict_local(text)
        print(f"[ConsecutivePeerAgent] Local: {local_result['prediction']} ({local_result['confidence']:.1%})")
        
        best_result = local_result
        
        # Step 2: If low confidence, ask peers
        if local_result['confidence'] < confidence_threshold and peer_addresses or peer_addresses:
            if local_result['confidence'] < confidence_threshold:
                print(f"[ConsecutivePeerAgent] Low confidence, asking {len(peer_addresses)} peers...")
            else:
                print(f'[ConsecutivePeerAgent] Verifying answer.., asking {len(peer_addresses)} peers...')
            
            peer_results = []
            for host, port in peer_addresses:
                result = await self.request_peer_prediction(host, port, text, timeout=60)
                if result:
                    peer_results.append(result)

                    print(f"[ConsecutivePeerAgent] Peer {host}:{port}: {result['prediction']} ({result['confidence']:.1%})")
                    print(f'[==] Local result: {local_result['prediction']} With Confidence: {local_result['confidence']}')
                    print(f'[==] Peer result: {result['prediction']} With Confidence: {result['confidence']}')

            if peer_results:
                best_peer = max(peer_results, key=lambda x: x['confidence'])                                   
                if best_peer['confidence'] > local_result['confidence']:
                    best_result = best_peer
                    print(f"[ConsecutivePeerAgent] Using peer result: {best_peer['prediction']} || Confidence: ({best_peer['confidence']:.1%})")
        else:
            print('[ConsecutivePeerAgent] Skipping Ensemble prediction... Peer address is None or empty')
            time.sleep(5)
            return best_result

        self.stats['predictions'] += 1
        return best_result
    

    def start_server(self):
        """Start server to accept peer connections"""
        
        def server_loop():
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind(('0.0.0.0', self.port))
            self.server_socket.settimeout(1.0)           
            self.server_socket.listen(5)

            if self.pipeline.distribution.enable_ssl and self.pipeline.distribution.ssl_context:
                self.pipeline.distribution.ssl_context.check_hostname = False
                self.server_socket = self.pipeline.distribution.ssl_context.wrap_socket(self.server_socket, server_side=True)

            self.running = True
            
            print(f"[ConsecutivePeerAgent] Server listening on port {self.port}!")
            
            while self.running:
                try:
                    client, addr = self.server_socket.accept()
                    
                    # Check IP
                    if addr[0] not in self.allowed_ips:
                        print(f"[ConsecutivePeerAgent] Rejected connection from {addr}")
                        client.close()
                        continue
                    
                    # Handle in thread
                    thread = threading.Thread(target=self._handle_client, args=(client, addr))
                    thread.daemon = True
                    thread.start()
                except socket.timeout:
                    continue                         
                except Exception as e:
                    if self.running:
                            print(f"[ConsecutivePeerAgent] Server error: {e}")
            try:
                self.server_socket.close()
            except:
                pass
            
        print("[ConsecutivePeerAgent] Server Successfully Stopped listening !")
        
        thread = threading.Thread(target=server_loop, daemon=True)
        thread.start()


    def _handle_client(self, client, addr):
        # Handle incoming peer connection
        print(f"[ConsecutivePeerAgent] Client connected from {addr}")
        
        try:
            # Authenticate
            if addr[0] in ['127.0.0.1', 'localhost', 'local'] and addr[1] == self.port:
                print(f"[❌] Client is self, ignoring...")
                client.close()
                return   

            auth_msg = self._receive_message(client)

            if not auth_msg:
                print(f"[ConsecutivePeerAgent] No authentication message from peer {addr}")
                client.close()
                return


            if not auth_msg or auth_msg.get('type') != 'auth':
                print(f"[ConsecutivePeerAgent] Auth failed from {addr}")
                client.close()
                return
            
            if auth_msg.get('token') != self.secret_key:
                print(f"[ConsecutivePeerAgent] Invalid token from {addr}")
                client.close()
                return
            
            # Send auth response
            self._send_message(client, {'type': 'auth_response', 'status': 'ok'})
            
            # Handle prediction requests
            while self.running:
                message = self._receive_message(client)
                if message is None:
                    break
                
                if message.get('type') == 'predict':
                    text = message.get('text', '')
                    print(f"[ConsecutivePeerAgent] Received prediction request!")
                    
                    # Use local prediction
                    result = asyncio.run(self.predict_local(text))
                    
                    response = {
                        'type': 'predict_response',
                        'prediction': result['prediction'],
                        'confidence': result['confidence']
                    }
                    self._send_message(client, response)
                    
                elif message.get('type') == 'ping':
                    self._send_message(client, {'type': 'pong'})
                    
        except Exception as e:
            print(f"[ConsecutivePeerAgent] Client handler error: {e}")
        finally:
            client.close()
            print(f"[ConsecutivePeerAgent] Client disconnected from {addr}")


    def stop_server(self):
        self.running = False

        print('[ConsecutivePeerAgent] Initiating Server shutdown...')   
        # Close all peer connections
        try:
            with self._lock:
                for key, info in self.connected_peers.items():
                    try:
                        info['sock'].shutdown(socket.SHUT_RDWR)
                        info['sock'].close()
                    except:
                        pass
                
                self.connected_peers.clear()
                if self.server_socket:
                    try:
                        self.server_socket.close()  
                    except Exception as e:
                        print(f'[ConsecutivePeerAgent] Cant close socket: {e}')
                        pass  
                                
                print('[ConsecutivePeerAgent] Server Successfully Stopped listening !')

        except Exception as e:
            print(f'[ConsecutivePeerAgent] Error closing socket: {e}')
            pass


    def get_stats(self) -> Dict:
        # Get statistics
        return {
            **self.stats,
            'connected_peers': len(self.connected_peers)
        }

    
class CohesiveAgentDeployment:
    """
    Safe deployment wrapper for Async Manager with external peer support.
    Handles graceful shutdown, error recovery, and peer connections.
    """
    
    def __init__(self,
                 pipeline: IntegratedPipeline,
                 memory_name: str,
                 filename: str,
                 target_title: str,
                 label_name: str,
                 security_level: str = "PRODUCTION",
                 enable_peers: bool = True,
                 trusted_networks: list = None,
                 secret_key: str = None,
                 peer_discovery_port: int = 5555,
                 shared_auth_token: str = None,
                 predict_manager: Any=None,
                 peer_config: Any='peer_config.json',
                 consecutive_peer_config: Any=None
                 ):
        self.pipeline = pipeline 
        self.pipeline.autonomous = True 
  
        # Initialize prediction manager
        self.manager = PipelinePredictionManager(
            self.pipeline,
            label_csv=filename,
            target_title=target_title,
            label=label_name
        )

        self._peer_agent = ConsecutivePeerAgent(
            peer_id=self.pipeline.memory_name,
            port=peer_discovery_port + 100,
            secret_key=secret_key,
            manager=self.manager,
            pipeline=self.pipeline
        ) 

        # Map security level string to enum
        self.security_map = {
            "DEVELOPMENT": SecurityLevel.DEVELOPMENT,
            "STAGING": SecurityLevel.STAGING,
            "PRODUCTION": SecurityLevel.PRODUCTION,
            "HARDENED": SecurityLevel.HARDENED
        }

        self.resolved_level = self.security_map.get(security_level, SecurityLevel.PRODUCTION)

        # this propagate security_level to pipeline AND distribution
        # so _check_ip_access and _get_bind_host can use it
        self.pipeline.security_level = self.resolved_level
        self.pipeline.distribution.security_level = self.resolved_level   # AgentDistributedInference
     
        # Create Async Manager with security
        self.async_manager = PipelineAsyncManager(
            pipeline=self.pipeline,
            prediction_manager=self.manager,
            security_level=self.security_map.get(security_level, SecurityLevel.PRODUCTION),
            api_key=shared_auth_token,
            max_workers=4,
            task_timeout=30,
            max_retries=3
        )

        self.peer_config_name = peer_config
        self.consecutive_peer_config = consecutive_peer_config
        if shared_auth_token:
            # Set for distribution (peer authentication)
            self.pipeline.distribution.auth_token = shared_auth_token
            self.pipeline.distribution.secret_key = shared_auth_token
            
            # Set for async manager (API key for predictions)
            self.async_manager._default_api_key = shared_auth_token
            self.async_manager.api_key_manager.keys = {}  # Reset
            self.async_manager.api_key_manager.generate_key(
                {'type': 'shared', 'source': 'cluster'},
                key_value=shared_auth_token  # Need to modify generate_key to accept value
            )
            
            print(f"[🔑] Using shared auth token for entire cluster") 

        self.discovery_enabled = True
        self.discovery = True                        # used in _broadcast_discovery while loop
        self.peer_discovery_broadcast = True         # ADD — this is what gates all discovery
        self.discovery_broadcast_only_trusted_network = True

        self.enable_peers = enable_peers
        self.peer_discovery_port = peer_discovery_port
        self._shutdown_event = asyncio.Event()
        self._peer_tasks = []
        self._known_peers = {}
        self.identified_peers = []

        self.attempt = 0
        self.max_attempts = 3
        
        self.result_queue = AsyncResultQueue(max_size=1000)
        self.worker_pool = WorkerPool(self.result_queue, num_workers=4) 
        
        # Discovery security settings
        self.discovery_secret = os.environ.get('DISCOVERY_SECRET', 'default_secret_change_me')
        self.discovery_enabled = True
        self.discovery_broadcast_only_trusted_network = True
        self.trusted_networks = trusted_networks  # Only respond to these networks
        self.discovery_rate_limit = 5  # Max 5 discovery responses per minute per IP
        self._discovery_requests = defaultdict(list)  # Track request rates
        self.local_ips = self._get_local_ips()  # Get local IPs for discovery filtering
        self._connecting_to = set()
        self.consecutive_peer_config = consecutive_peer_config if consecutive_peer_config else "consecutive_peers.json"

    def _get_local_ips(self) -> List[str]:
        # Get all local IP addresses for this machine
        ips = set()
        try:
            # Get hostname IP
            ips.add(socket.gethostbyname(socket.gethostname()))
            
            # Get all network interfaces
            hostname = socket.gethostname()
            for ip in socket.gethostbyname_ex(hostname)[2]:
                ips.add(ip)
            
            # Add localhost
            ips.add('127.0.0.1')
            
        except Exception as e:
            logger.warning(f"[-] Could not get local IPs: {e}")
            ips.add('127.0.0.1')
        
        return list(ips)
        
    def _is_trusted_network(self, client_ip: str) -> bool:
        # if client IP is from trusted network
        import ipaddress
        
        try:
            client = ipaddress.ip_address(client_ip)
            for network in self.trusted_networks:
                if client in ipaddress.ip_network(network):
                    return True
        except:
            pass
        return False
    
    def _check_discovery_rate_limit(self, client_ip: str) -> bool:
        # Rate limit discovery requests
        now = time.time()
        # Clean old requests
        self._discovery_requests[client_ip] = [
            t for t in self._discovery_requests[client_ip] 
            if now - t < 60  # Keep last minute
        ]
        
        if len(self._discovery_requests[client_ip]) >= self.discovery_rate_limit:
            logger.warning(f"[=] Discovery rate limit exceeded for {client_ip}")
            return False
        
        self._discovery_requests[client_ip].append(now)
        return True
    
    def _create_discovery_response(self) -> dict:
        # a secure discovery response (minimal info)
        return {
            'type': 'DISCOVERY_RESPONSE',
            'version': '1.0',
            'port': self.peer_discovery_port,
            'requires_auth': True,  # Don't reveal agent_id or capabilities
            'timestamp': time.time()
        }    


    async def start(self, bootstrap_token: str = None, skip_discovery: bool=False):
        # Start the agent with all components
        
        logger.info("[🚀] Starting Safe Agent Deployment...")
        
        # 1. Start Async Manager
        success = self.async_manager.start(bootstrap_token=bootstrap_token)
        if not success:
            raise RuntimeError("[-] Failed to start Async Manager")
        
        logger.info("[✅] Async Manager started")
        
        # 2. Start distributed inference (for peer connections)
        if self.enable_peers:
            # Start the server to listen for peer connections
            self.pipeline.distribution.start_server()
            logger.info(f"[✅] Peer server listening on port {self.peer_discovery_port}")
            
            # Start peer discovery if needed    
            await self._start_peer_discovery()   

            asyncio.create_task(self._health_monitor())
           
        # Start result queue and workers
        await self.result_queue.start()
        await self.worker_pool.start(self._prediction_worker)
        
        # 4. Start health monitoring loop
        asyncio.create_task(self._health_monitor())

        logger.info("[🎉] Agent fully operational!")
        self._print_status()
        
        return True
        
    async def _prediction_worker(self, texts: list, api_key: str = None, client_ip: str = None) -> dict:
        # Worker function for processing predictions
        # This runs in a thread pool via asyncio.to_thread
        return self.async_manager.predict(
            texts=texts,
            timeout=self.pipeline.timeout,
            retries=None,
            api_key=api_key,
            client_ip=client_ip,
            method='advanced'
        )

    async def _start_peer_discovery(self):
        # Discover and connect to peer agents safely
        
        #  Connect to known peers from config file
        known_peers = self._load_known_peers()
        
        for peer_host, peer_port in known_peers:
            try:
                try:
                    await self._connect_to_peer(peer_host, peer_port)
                except:
                    if self.peer_discovery_broadcast:
                        await self._discover_local_peers()
        
                    if self.peer_discovery_broadcast:
                        self._discovery_task = asyncio.create_task(self._broadcast_discovery())    

            except Exception as e:
                logger.error(f"[❌] Peer connection error {peer_host}:{peer_port} - {e}")
    

    def _load_known_peers(self):
        # Load known peers from config file

        print(f'[==] Loading known peers from config: {self.peer_config_name}')
        config_file = self.peer_config_name
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
                return config.get('known_peers', [])
        
        # Default peers (can be replaced with other IPs)
        return [
            ('127.0.0.1', 5555),  # Example peer
            ('127.0.0.1', 5556)
        ]
        
    async def _discover_local_peers(self):
        # Discover peers on local network via port scanning
        logger.info("🔍 Scanning for local peers...")
        
        # Scan common ports
        for port in range(self.peer_discovery_port, self.peer_discovery_port + 5):
            if port == self.peer_discovery_port:
                continue  # Skip self
                
            for ip in self.local_ips[:3]:  # Limit to first few IPs to avoid long scan
                if ip == '127.0.0.1':
                    continue
                    
                await self._connect_to_peer(ip, port)
    
    async def _broadcast_discovery(self):
        # broadcast discovery message to find peers on network
        logger.info("📡 Starting broadcast discovery...")
        
        while not self._shutdown_event.is_set() and self.discovery:
            try:
                # UDP broadcast socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)         
                print(f"[broadcast_discovery() SOCKET CREATED] id={id(sock)}")                
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                sock.settimeout(2)
                
                # Broadcast discovery message
                discovery_msg = json.dumps({
                    'type': 'DISCOVERY',
                    'agent_id': id(self.pipeline.distribution),
                    'port': self.peer_discovery_port,
                    'timestamp': time.time()
                }).encode()
                
                # Adding signature to prevent spoofing
                signature = self._sign_message(discovery_msg)
                discovery_msg['signature'] = signature

                sock.sendto(discovery_msg, ('<broadcast>', self.peer_discovery_port))
                
                # Listen for responses
                try:
                    data, addr = sock.recvfrom(1024)
                    client_ip = addr[0]
                    client_port = addr[1]

                    if client_ip in ['127.0.0.1', 'localhost']:
                        if client_port == self.peer_discovery_port:
                            print(f"[=] Ignoring self-discovery response")
                            continue  

                    # Security checks before processing response
                    if not self._is_trusted_network(client_ip):
                        logger.debug(f"[==] Ignoring discovery from untrusted network: {client_ip}")
                        continue
                    
                    if not self._check_discovery_rate_limit(client_ip):
                        continue    

                    response = json.loads(data.decode())
                    # Verify signature
                    if not self._verify_signature(response):
                        logger.warning(f"[=-=] Invalid discovery response signature from {client_ip}")
                        continue   

                    if response.get('type') == 'DISCOVERY_RESPONSE':
                        logger.info(f"✅ Received discovery response from {client_ip}")
                        peer_host = addr[0]
                        peer_port = response.get('port')
                        await self._connect_to_peer(peer_host, peer_port)
                except socket.timeout:
                    pass
                
                sock.close()
                
            except Exception as e:
                logger.debug(f"Broadcast discovery error: {e}")
            
            # Wait before next broadcast
            await asyncio.sleep(60)
    
    def _sign_message(self, message: dict) -> str:
        # Sign message with HMAC to prevent spoofing
       
        # Sort keys for consistent serialization
        message_str = json.dumps(message, sort_keys=True)
        return hmac.new(
            self.discovery_secret.encode(),
            message_str.encode(),
            hashlib.sha256
        ).hexdigest()  
        
    def _verify_signature(self, message: dict) -> bool:
        # Verify message signature
        if 'signature' not in message:
            return False
        
        signature = message.pop('signature')
        expected = self._sign_message(message)
        message['signature'] = signature
        
        return hmac.compare_digest(signature, expected)


    async def _connect_to_peer(self, host: str, port: int) -> bool:
        # Connect to a peer agent
        try:
            # Check if already connected    
            # Store peer info for reconnection
            peer_key = f"{host}:{port}"

            #  ✅ Prevent multiple simultaneous connection attempts to same peer
            if peer_key in self._connecting_to:
                print(f"[⚠️] Already connecting to {peer_key}, skipping")
                return False     

            self._connecting_to.add(peer_key)  

            for agent_id, info in self.pipeline.distribution.remote_agents.items():
                if info.get('host') == host and info.get('port') == port:
                    logger.debug(f"[=+=] Already connected to {host}:{port}")
                    return True
            
            logger.info(f"🔗 Connecting to peer {host}:{port}")
            
            # Use the distribution system to connect
            sock = self.pipeline.distribution.connect_to_agent(host, port)
            
            if sock:
                logger.info(f"✅ Connected to peer {host}:{port}")

                self._known_peers[peer_key] = {
                    'host': host,
                    'port': port,
                    'sock': sock,
                    'last_seen': datetime.now(),
                    'connected': True
                }
                
                # Start background task to handle peer messages
                task = asyncio.create_task(
                    self._handle_peer_communication(host, port, sock)
                )
                self._peer_tasks.append(task)
                return True
            else:
                logger.warning(f"[❌] Failed to connect to {host}:{port}")
                return False
                
        except Exception as e:
            logger.error(f"[-] Peer connection error {host}:{port} - {e}")
            return False
            


    async def _handle_peer_communication(self, peer_host: str, peer_port: int, sock):
        # Handle bidirectional communication with a peer
        logger.info(f"📡 Peer communication active for {peer_host}:{peer_port}")
        
        try:
            while not self._shutdown_event.is_set():
                # The distribution system handles message receiving internally
                # This task just monitors connection health
                await asyncio.sleep(5)
                
                # Send heartbeat to check connection
                try:
                    # self.pipeline.distribution._send_message(
                        # sock, {'type': 'PING', 'timestamp': time.time()}
                   # )
                   sock.getpeername()
                   print(f'[==] Peer name: {sock.getpeername()}')
                except:
                    logger.warning(f"[-] Peer {peer_host}:{peer_port} disconnected")
                    break
                
        except asyncio.CancelledError:
            logger.info(f"[-] Peer communication cancelled for {peer_host}:{peer_port}")
        except Exception as e:
            logger.error(f"[-] Peer communication error: {e}")
        finally:
            # Update peer status
            peer_key = f"{peer_host}:{peer_port}"
            if peer_key in self._known_peers:
                self._known_peers[peer_key]['connected'] = False
            sock.close()

    
    async def _peer_health_monitor(self):
        # Monitor peer health and reconnect if needed
        logger.info("[💓] Peer health monitor started")
        
        while not self._shutdown_event.is_set():
            await asyncio.sleep(30)
            
            try:
                # Ping all connected peers
                alive_agents = self.pipeline.distribution.broadcast_ping()
                logger.info(f"[=+=] Connected peers: {len(alive_agents)}")
                
                # Reconnect to known peers that went offline
                for peer_key, peer_info in self._known_peers.items():
                    if not peer_info.get('connected', False):
                        logger.info(f"[==] Attempting to reconnect to {peer_key}")
                        await self._connect_to_peer(peer_info['host'], peer_info['port'])
                        
            except Exception as e:
                logger.error(f"[❌] Peer health monitor error: {e}")
       
    
    async def _health_monitor(self):
        # Background health monitoring
        while not self._shutdown_event.is_set():
            await asyncio.sleep(30)
            
            try:
                stats = self.async_manager.get_stats()
                logger.info(f"[==] Health Check - Stats: {stats}")
                
                # Check if we need to reconnect peers
                if self.enable_peers:
                    alive_agents = self.pipeline.distribution.broadcast_ping()
                    logger.info(f"[=+=] Connected peers: {len(alive_agents)}")
                    
            except Exception as e:
                logger.error(f"[❌] Health monitor error: {e}")
                
    def save_peer_config(self, peers: List[tuple]):
        """Save peer configuration to file"""
        config = {'known_peers': peers}
        with open('peer_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"[==] Saved {len(peers)} peers to config")    

        
    def _print_status(self):
        print("\n" + "="*70)
        print("=== 🤖 COHESIVE INTEGRATED PIPELINE - STATUS ===")
        print("="*70)
        print(f"📊 State: {self.async_manager.state}")
        print(f"🔒 Security Level: {self.async_manager.security_level.value}")
        print(f"🌐 Peers Enabled: {self.enable_peers}")
        print(f"🔗 Connected Peers: {len(self.pipeline.distribution.remote_agents)}")
        print(f"📡 Peer Port: {self.peer_discovery_port}")
        print(f"🖥️  Local IPs: {', '.join(self.local_ips)}")
        print(f"⏳ Queue Size: {self.async_manager._stats['queue_size']}")
        print(f"🔑 API Key Required: {self.async_manager.config.require_api_key}")
        if self.async_manager.config.require_api_key:
            print(f"🔑 Default API Key: {getattr(self.async_manager, '_default_api_key', 'N/A')[:20]}...")
        
        # Show connected peers
        if self.pipeline.distribution.remote_agents:
            print("\n📡 Connected Peers:")
            for agent_id, info in self.pipeline.distribution.remote_agents.items():
                print(f"   → {info.get('host', 'unknown')}:{info.get('port', 'unknown')} (trust: {info.get('trust', 1.0):.2f})")
        
        print("="*70)
    
    def get_peers_status(self) -> Dict:
        """Get detailed status of all peers"""
        return {
            'connected_peers': len(self.pipeline.distribution.remote_agents),
            'known_peers': self._known_peers,
            'remote_agents': {
                agent_id: {
                    'host': info.get('host'),
                    'port': info.get('port'),
                    'trust': info.get('trust', 1.0)
                }
                for agent_id, info in self.pipeline.distribution.remote_agents.items()
            }
        }
    

    # ============ PREDICTION METHODS ============
    async def multi_modal_peer_ensemble_prediction(self, texts, api_key: str = None, method: str = 'advanced', disable_sync: bool=False) -> Any:
        """
        Robust prediction: try main system first, fallback to SecurePeerAgent.
        """
        try:
            # Try main prediction with timeout
            if not self.pipeline.autonomous:
                print('[==] Initiating Autonomous ensemble prediction...')
                self.pipeline.ensemble.explainer.supervised_learning = False
                self.pipeline.autonomous = True

            result = await asyncio.wait_for(
                self.predict_with_peers(texts, api_key, method, disable_sync=disable_sync),
                timeout=self.pipeline.timeout
            )
            
            # Check if result is valid
            if result and result.get('confidence', 0) > self.pipeline.confidence_threshold and result.get('peer_count') > 0:
                return result
            
            # Low confidence, try fallback
            print("[=] Initiating Consecutive peer ensemble...")
            return await self.predict_with_peer_consecutive(texts, api_key, method)
            
        except (asyncio.TimeoutError, Exception) as e:
            print(f"[=] Main prediction failed: {e}, using consecutive ensemble...")
            return await self.predict_with_peer_consecutive(texts, api_key, method)

    def _load_consecutive_known_peers(self):
        """Load peers for fallback using different ports"""
        config_file = self.consecutive_peer_config
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
                return config.get('known_peers', [])
        
        return [
            ('127.0.0.1', 5656),
            ('127.0.0.1', 5655)
        ]
    
    async def predict_with_peer_consecutive(self, texts, api_key: str = None, method: str = 'advanced') -> dict:
        """
        Fallback prediction using SecurePeerAgent when main system fails.
        """
        print("[=] Using Secure Peer Agent fallback...")
        

        if not self._peer_agent.running:
            self._peer_agent.start_server()
        
        # Extract text
        # Get peer addresses from config
        peer_addresses = self._load_consecutive_known_peers()
        print(f'[===] Peer addresses: {peer_addresses}')
        
        # Ensemble prediction
        result = await self._peer_agent.ensemble_predict(
            peer_addresses=peer_addresses,
            text=texts,           
            confidence_threshold=self.pipeline.confidence_threshold
        )

        
        return {
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'source': result.get('source', 'unknown'),
            'fallback': True
        }


    async def predict_with_peers(self, texts, api_key: str = None, method: str = 'advanced', disable_sync: bool=False) -> dict:
        """
        Simple peer prediction: Connect to peers first, then get predictions.
        """
        print("[=+=] Starting peer-augmented prediction")
        
        try:
            if not disable_sync:
                local_result = self.predict_sync(texts, api_key, method=method)
                print(f'[==] Local prediction Result: {local_result.get("prediction")} ({local_result.get("confidence", 0):.1%})')
            else:
                local_result = {'prediction': None, 'confidence': 0.0}

            connection = await self._ensure_peer_connections(api_key)

            print(f'[=] Peer connection ensured: {connection}')
            await asyncio.sleep(0.3)    

            peers = []
            for agent_id, info in list(self.pipeline.distribution.remote_agents.items()):
                if agent_id != 'local' and str(agent_id) != str(id(self)):
                    sock = info.get('sock')
                    if sock:
                        try:
                            sock.getpeername()
                            peers.append(agent_id)
                            print('[=+=] Socket is alive!')
                        except Exception as e:
                            print('[=] Socket is not available')
                            pass
                    else:
                        print('[=] No socket is available')
                else:
                    print(f'[=^=] peer in sight: {self.pipeline.distribution.remote_agents}')
            
            print(f'[=+=] Connected peers: {len(peers)}') 

            confidence_threshold = getattr(self.pipeline, 'confidence_threshold', 0.6)
            if not peers or local_result.get('confidence', 0) >= confidence_threshold:
                print(f'[==] Using local result (confidence: {local_result.get("confidence", 0):.1%})')
                return local_result
            
            print(f'[=/=] Asking {len(peers)} peers...')
            
            peer_results = []
            for agent_id in peers:
                try:
                    result = await self._ask_peer_simple(agent_id, texts)
                    if result:
                        peer_results.append(result)
                        print(f'[/==] Peer {agent_id} result: {result.get("prediction")} ({result.get("confidence", 0):.1%})')
                except Exception as e:
                    print(f'[/=-] Peer {agent_id} failed: {e}')
            
            if peer_results:
                best = max(peer_results, key=lambda x: x.get('confidence', 0))
                best_conf = best.get('confidence', 0)
                local_conf = local_result.get('confidence', 0)
                
                print(f'[==] Local: {local_conf:.1%}, Best peer: {best_conf:.1%}')
                
                if best_conf > local_conf:
                    print(f'[/==] Using peer result: {best.get("prediction")}')
                    return best
            
            return local_result
            
        except Exception as e:
            print(f"[=] Peer prediction failed: {e}")
            traceback.print_exc()
            return self.predict_sync(texts, api_key, method='basic')
            
    async def _ask_peer_simple(self, agent_id, texts):
        """
        Simple request to a single peer.
        """
        info = self.pipeline.distribution.remote_agents.get(agent_id)
        if not info:
            return None
        
        sock = info.get('sock')
        if not sock:
            return None
        
        # Prepare message
        print('[==] Preparing Message...')
        if isinstance(texts, dict) and 'test_titles' in texts:
            message = {
                'type': self.pipeline.distribution.MSG_TYPES['PREDICT_REQUEST'],

                'payload': {
                    'test_titles': texts.get('test_titles'),
                    'label_map': texts.get('label_map'),
                    'rules': texts.get('rules'),
                    'use_transformer': texts.get('use_transformer', True)
                },
                'token': self.get_api_key()
            }
        else:
            text = texts[0] if isinstance(texts, list) else str(texts)
            message = {
                'type': self.pipeline.distribution.MSG_TYPES['PREDICT_REQUEST'],
                'text': text,
                'token': self.get_api_key(),
                'timestamp': time.time()
            }
        
        try:
            sock.settimeout(10)
            # Add this before sending
            try:
                sock.getpeername()  # Test if socket is still alive
                print('[=] Socket still present!')
            except:
                print(f"[=] Socket to {agent_id} is dead")
                return None   

            self.pipeline.distribution._send_message(sock, message)

            print('[==] Successfully send prediction message!')
            response = self.pipeline.distribution._receive_message(sock)
            sock.settimeout(20)
            
            if response and response.get('type') == 2:
                print(f'[=+=] Got response from peer: {response}')
                return {
                    'prediction': response.get('prediction'),
                    'confidence': response.get('confidence', 0)
                }
            else:
                print('[-] No response from peer.')
            return None
            
        except Exception as e:
            print(f'[=] Error asking peer {agent_id}: {e}')
            return None


    def _is_server_listening(self) -> bool:
        # if the server is actually listening on its port
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) 
        sock.settimeout(1)
        try:
            result = sock.connect_ex(('127.0.0.1', self.peer_discovery_port))
            sock.close()
            listening = result == 0
            print('[=+=] Server is listening!')
            return listening
        except:
            return False

    async def _ensure_peer_connections(self, api_key: str = None):
        """
        Robust peer connection manager - prevents duplicate connections and WinError.
        """
        print("[=] Ensuring peer connections...")
        
        # ✅ Step 1: Clean up dead connections first
        dead_connections = []
        for agent_id, info in list(self.pipeline.distribution.remote_agents.items()):
            if agent_id == 'local':
                continue
            
            sock = info.get('sock')
            if sock is None:
                dead_connections.append(agent_id)
                continue
            
            # Test if socket is still alive
            try:
                sock.getpeername()
            except:
                print(f"[=] Dead connection detected: {agent_id}")
                dead_connections.append(agent_id)
        
        # Remove dead connections
        for agent_id in dead_connections:
            print(f"[=] Removing dead connection: {agent_id}")
            try:
                del self.pipeline.distribution.remote_agents[agent_id]
            except:
                pass
        
        # ✅ Step 2: Load known peers from config
        known_peers = self._load_known_peers()
        
        if not known_peers:
            print("[=] No known peers configured")
            return False
        
        # ✅ Step 3: Try each peer once, no retry loops
        successful = False
        
        for host, port in known_peers:
            peer_key = f"{host}:{port}"
            
            # Skip self
            if host in ['127.0.0.1', 'localhost'] and port == self.peer_discovery_port:
                print(f"[=] Skipping self: {peer_key}")
                continue
            
            # Check if already connected (and alive)
            already_connected = False
            for agent_id, info in self.pipeline.distribution.remote_agents.items():
                if info.get('host') == host and info.get('port') == port:
                    sock = info.get('sock')
                    if sock:
                        try:
                            sock.getpeername()
                            print(f"[=] Already connected to {peer_key}")
                            already_connected = True
                            successful = True
                            break
                        except:
                            # Socket dead, will reconnect
                            pass
            
            if already_connected:
                continue
            
            # ✅ Step 4: Single connection attempt (NO RETRY)
            print(f"[=] Connecting to {peer_key}...")
            
            try:
                # Use add_peer with timeout
                result = await self._connect_single_attempt(host, port, api_key)
                
                if result:
                    print(f"[=] ✅ Connected to {peer_key}")
                    successful = True
                else:
                    print(f"[=] ❌ Failed to connect to {peer_key}")
                    
            except Exception as e:
                print(f"[=] ❌ Error connecting to {peer_key}: {e}")
        
        return successful


    async def _connect_single_attempt(self, host, port, api_key, timeout=5):
        """
        Single connection attempt - no retries, no loops.
        """
        try:
            # Check if already connected (one more time)
            for agent_id, info in self.pipeline.distribution.remote_agents.items():
                if info.get('host') == host and info.get('port') == port:
                    sock = info.get('sock')
                    if sock:
                        try:
                            sock.getpeername()
                            return True
                        except:
                            pass
            
            # Single connection attempt with timeout
            result = await asyncio.wait_for(
                asyncio.to_thread(self.add_peer, host, port, api_key),
                timeout=timeout
            )
            
            # Verify connection is alive
            await asyncio.sleep(0.1)  # Give it a moment
            
            for agent_id, info in self.pipeline.distribution.remote_agents.items():
                if info.get('host') == host and info.get('port') == port:
                    sock = info.get('sock')
                    if sock:
                        try:
                            sock.getpeername()
                            return True
                        except:
                            pass
            
            return result
            
        except asyncio.TimeoutError:
            print(f"[=] Connection timeout to {host}:{port}")
            return False
        except Exception as e:
            print(f"[=] Connection error to {host}:{port}: {e}")
            return False


    async def _request_peer_prediction_async(self, agent_id, texts):
        """Async peer prediction request"""
        try:
            # Use async version
            return await self.pipeline.distribution.request_prediction_async(agent_id, texts, timeout=5)
        except Exception as e:
            logger.warning(f"[=-] Peer {agent_id} failed: {e}")
            return None

    def _ensemble_predictions(self, local: dict, peers: list) -> dict:
        # Combine predictions from multiple agents
        try:
            print('[=+=] Initiating Ensemble weighting with: {peers} peers total')
            votes = defaultdict(float)
            votes[local.get('prediction', 'unknown')] += local.get('confidence', 0)
            
            for peer in peers:
                if peer and isinstance(peer, dict):
                    votes[peer.get('prediction', 'unknown')] += peer.get('confidence', 0)
            
            best_pred = max(votes.items(), key=lambda x: x[1])
            
            total_weight = len(peers) + 1
            return {
                'prediction': best_pred[0],
                'confidence': min(best_pred[1] / total_weight, 1.0),
                'local_prediction': local.get('prediction'),
                'local_confidence': local.get('confidence'),
                'peer_count': len(peers),
                'ensemble_votes': dict(votes)
            }
        except Exception as e:
            print(f'[-] Error in ensemble weighting; {e}, returning local prediction with 0.0 confidence')
            return {
                'prediction': local,
                'confidence': 0.0,
                'peer_count': 0.0,
            }
     

    async def predict_batch_async(self, texts: List[str], api_key: str = None, client_ip: str = None) -> List[dict]:
        """
        Batch async predictions - runs in parallel!
        """
        tasks = [
            self.predict_async(text, api_key, client_ip)
            for text in texts
        ]
        
        # Run all predictions concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        output = []
        for text, result in zip(texts, results):
            if isinstance(result, Exception):
                output.append({
                    'text': text,
                    'prediction': 'error',
                    'confidence': 0.0,
                    'error': str(result)
                })
            else:
                output.append({
                    'text': text,
                    'prediction': result.get('prediction'),
                    'confidence': result.get('confidence', 0),
                    **result
                })
        
        return output 


    def predict_sync(self, texts: Any, api_key: str = None, client_ip: str = None, method: str = 'advanced') -> dict:
        """
        Synchronous prediction with security.
        Use this for simple, blocking calls.
        """
        # ✅ Direct prediction without async queue
        print('[==] Initiating predict sync...')
        try:
            if method == 'advanced':
                test_titles = texts['test_titles']
                label_map = texts['label_map']
                rules = texts['rules']
                X = texts['X']
                y = texts['y']

                result, chosen_label, confidence = self.manager.advanced_prediction_method(
                    test_titles,
                    label_map,
                    rules,
                    X=X, y=y,
                    show_proba=True,
                    use_transformer=self.pipeline.use_transformer
                )
                return {
                    'prediction': chosen_label,
                    'confidence': confidence,
                    'result': result
                }
            else:
                # Basic prediction
                text = texts[0] if isinstance(texts, list) and texts else str(texts)
                result = self.pipeline.predict_single(text)
                return result
                        

        except Exception as e:
            logger.error(f"[-] Prediction failed: {e}")
            print(f"[-] Prediction failed: {e}")
            traceback.print_exc()
            return {
                'prediction': 'error',
                'confidence': 0.0,
                'error': str(e)
            }
    
    async def predict_async(self, texts, api_key: str = None, client_ip: str = None) -> dict:
        """
        Asynchronous prediction.
        Use this for non-blocking operations.
        """
        try:
            # Submit request to queue
            request_id = await self.result_queue.submit(
                texts=texts,
                api_key=api_key,
                client_ip=client_ip,
            )
            
            # Wait for result with timeout
            result = await self.result_queue.wait_for_result(
                request_id=request_id,
                timeout=30
            )
            
            return result
            
        except TimeoutError:
            logger.error(f"[-] Async prediction timed out for: {texts}")
            return {
                'prediction': 'timeout',
                'confidence': 0.0,
                'error': 'Request timeout'
            }
        except Exception as e:
            logger.error(f"[-] Async prediction failed: {e}")
            traceback.print_exc()
            return {
                'prediction': 'error',
                'confidence': 0.0,
                'error': str(e)
            }
            
    def get_queue_stats(self) -> Dict:
        # Get result queue statistics
        logger.info("[=] Fetching result queue stats...")
        return self.result_queue.get_status(request_id=None)

     
    # ============ PEER MANAGEMENT ============
    
    def add_peer(self, host: str, port: int, api_key: str = None):
        # Manually add a peer connection
        if not api_key:
            agent_id = f"{host}:{port}"
            if hasattr(self.pipeline.distribution, 'peer_tokens'):
                api_key = self.pipeline.distribution.peer_tokens.get(agent_id)
        else:
            self.pipeline.distribution.add_trusted_agent(f"{host}:{port}", api_key)
        
        # Connecting
        sock = self.pipeline.distribution.connect_to_agent(host, port)
        if host in ['127.0.0.1', 'localhost', '0.0.0.0']:
            if port == self.pipeline.distribution.port or port == 0:
                print(f"[❌] Cannot add self as peer ({host}:{port})")
                return False        

        if sock:
            # Store in known peers
            peer_key = f"{host}:{port}"
            self._known_peers[peer_key] = {
                'host': host,
                'port': port,
                'sock': sock,
                'last_seen': datetime.now(),
                'connected': True
            }
            
            # Start communication task
            task = asyncio.create_task(
                self._handle_peer_communication(host, port, sock)
            )
            self._peer_tasks.append(task)
            
            logger.info(f"✅ Manually added peer {host}:{port}")
            return True
        
        logger.error(f"[-] Failed to add peer {host}:{port}")
        return False
    
    def remove_peer(self, host: str, port: int):
        # Remove a peer connection
        peer_key = f"{host}:{port}"
        
        # Find and disconnect
        for agent_id, info in list(self.pipeline.distribution.remote_agents.items()):
            if info.get('host') == host and info.get('port') == port:
                self.pipeline.distribution.disconnect_agent(agent_id)
                break
        
        # Remove from known peers
        if peer_key in self._known_peers:
            del self._known_peers[peer_key]
        
        logger.info(f"[-] Removed peer {host}:{port}")
    
    def list_peers(self) -> List[Dict]:
        # List all connected peers
        peers = []
        for agent_id, info in self.pipeline.distribution.remote_agents.items():
            if agent_id == 'local':
                continue

            if info.get('port') == 0 or info.get('port') == self.pipeline.distribution.port:
                continue
            if info.get('host') in ['localhost', '127.0.0.1', '0.0.0.0']:
                if info.get('port') == self.pipeline.distribution.port:
                    continue        

            peers.append({
                'agent_id': agent_id,
                'host': info.get('host'),
                'port': info.get('port'),
                'trust': info.get('trust', 1.0),
                'last_seen': info.get('last_seen', datetime.now()).isoformat()
            })

        return peers 

    async def _connect_with_smart_retry(self, agent, host, port, api_key, max_retries=3, delay=1):
        """
        Smart connection with retry - STOPS once connected.
        """
        
        for attempt in range(max_retries):
            # ✅ Check if already connected BEFORE attempting
            existing_peers = agent.list_peers()
            for peer in existing_peers:
                if peer.get('host') == host and peer.get('port') == port:
                    print(f"[/==] Already connected to {host}:{port}, skipping retry")
                    return True
            
            print(f"[/==] Attempt {attempt + 1}/{max_retries}: Connecting to {host}:{port}...")
            
            try:
                # Try to connect
                if asyncio.iscoroutinefunction(agent.add_peer):
                    result = await agent.add_peer(host, port, api_key)
                else:
                    result = agent.add_peer(host, port, api_key)
                
                if result:
                    # ✅ Verify connection was successful
                    await asyncio.sleep(0.5)  # Give it a moment
                    existing_peers = agent.list_peers()
                    for peer in existing_peers:
                        if peer.get('host') == host and peer.get('port') == port:
                            print(f"[✅] Successfully connected on attempt {attempt + 1}")
                            return True
                    
                    print(f"[⚠️] Connection reported success but peer not found")
                    return True
                    
            except Exception as e:
                print(f"[=/] Attempt {attempt + 1} failed: {e}")
            
            # Don't retry if already connected
            if attempt < max_retries - 1:
                # Check again before waiting
                existing_peers = agent.list_peers()
                if any(p.get('host') == host and p.get('port') == port for p in existing_peers):
                    print(f"[=+=] Already connected, stopping retries")
                    return True
                
                print(f"[===] Waiting {delay}s before retry...")
                await asyncio.sleep(delay)
                delay *= 1.5
        
        return False


    # ============ SHUTDOWN ============
    
    async def shutdown(self):
        # Graceful shutdown of all components
        logger.info("🛑 Shutting down agent...")
        
        # signal shutdown to all loops
        self._shutdown_event.set()

        # stop worker pool.
        if hasattr(self, 'worker_pool'):
            await self.worker_pool.stop()

        if hasattr(self, 'result_queue'):
            await self.result_queue.stop()   

        # cancel peer tasks
        if self._peer_tasks:
            for task in self._peer_tasks:
                task.cancel()
            await asyncio.gather(*self._peer_tasks, return_exceptions=True)
        
        # stop peer agent server
        if hasattr(self, '_peer_agent'):
            self._peer_agent.stop_server()

        # stop distribution server
        if self.enable_peers:
            self.pipeline.distribution.stop_server()

        # FIX 1 — offload blocking stop() to thread so event loop stays free
        print('[=] Stopping Asynchronous manager setup...')
        await asyncio.get_event_loop().run_in_executor(
            None, 
            lambda: self.async_manager.stop(timeout=5, force=True)
        )

        await asyncio.sleep(0.5)
        print('✅ Agent shutdown complete')

        logger.info("✅ Agent shutdown complete")
    
    def get_api_key(self) -> str:
        # Get the default API key (for client distribution)
        return getattr(self.async_manager, '_default_api_key', None)
    


# ============ EXAMPLE: SECURE PEER-TO-PEER CLUSTER ============
async def run_secure_agent_cluster(pipeline,test_titles, label_map, rules, X=None, y=None, agent_id=None, filename=None, title_name=None, label_name=None, manager=None):
    """
    Run multiple agents that securely communicate.
    Stops retrying once connected successfully.
    """
    print("\n" + "="*60)
    print("=== SECURE PEER-TO-PEER CLUSTER ===")
    print("="*60)
    
    # Set discovery secret (in production, use environment variable)
    secret_key = 'my-ultra-safe-secret-key-for-authentication'

    # Agent 1 - Primary (Port 5555)
    agent1 = CohesiveAgentDeployment(
        pipeline=pipeline,
        memory_name="agent_primary",
        filename=filename,
        target_title=title_name,
        label_name=label_name,
        security_level="PRODUCTION",
        enable_peers=True,
        trusted_networks=['127.0.0.1/32', '192.168.1.0/24'],
        peer_discovery_port=5555,
        secret_key=secret_key,
        shared_auth_token=secret_key,
        predict_manager=manager
    )
    
    # Agent 2 - Secondary (Port 5556)
    agent2 = CohesiveAgentDeployment(
        pipeline=pipeline,
        memory_name="agent_secondary",
        filename=filename,
        target_title=title_name,
        label_name=label_name,
        security_level="PRODUCTION",
        enable_peers=True,
        trusted_networks=['127.0.0.1/32', '192.168.1.0/24'],
        peer_discovery_port=5556,
        secret_key=secret_key,
        shared_auth_token=secret_key,
        predict_manager=manager
    )
    
    try:
        # Start both agents
        print("\n🚀 Starting Agent 1...")
        await agent1.start()
        print("✅ Agent 1 started on port 5555")
        
        print("\n🚀 Starting Agent 2...")
        await agent2.start()
        print("✅ Agent 2 started on port 5556")
        
        # Give servers time to fully bind
        await asyncio.sleep(2)
        
        # Get API keys
        api_key = agent1.get_api_key()
        print(f"\n🔑 Using API Key: {api_key[:20]}...")
        
        texts = {"test_titles": test_titles, "label_map": label_map, "rules": rules, "X":X, "y":y, "use_transformer": True, "agent_id": agent_id}

        # Make prediction with peer ensemble
        # Peer Connection will be ensured successful during P2P 
        result = await agent1.multi_modal_peer_ensemble_prediction(
            texts=texts,
            api_key=api_key,
            method='advanced',
            disable_sync=True
        )    

        result2 = await agent2.multi_modal_peer_ensemble_prediction(
            texts=texts,
            api_key=api_key,
            method='advanced',
            disable_sync=True
        )      
        
        print(f"\n📊 Ensemble Result for Agent 1:")
        print(f"   Prediction: {result.get('prediction', 'N/A')}")
        print(f"   Confidence: {result.get('confidence', 0):.2%}")

        print(f"   Second Prediction: {result2.get('prediction', 'N/A')}")
        print(f"   Second Confidence: {result2.get('confidence', 0):.2%}")

        # Keep running briefly
        print("\n⏳ Cluster stable. Waiting 30 seconds before shutdown...")
        await asyncio.sleep(30)
        agent1._peer_agent.stop_server()
        agent2._peer_agent.stop_server()
        
    except Exception as e:
        print(f"\n❌ Error in cluster: {e}")
        traceback.print_exc()
        
    print("\n🛑 Shutting down cluster...")
    await agent1.shutdown()
    await agent2.shutdown()
    print("✅ Cluster shutdown complete")




async def example_async_with_result_queue(pipeline, test_titles, label_map, rules, X=None, y=None,agent_id=None, filename=None, title_name=None, label_name=None):
    # Example using the proper result queue
    
    agent = CohesiveAgentDeployment(
        pipeline=pipeline,
        memory_name="test_agent",
        filename=filename,
        target_title=title_name,
        label_name=label_name,
        security_level="DEVELOPMENT",
        enable_peers=False
    )
    
    await agent.start()
    
    api_key = agent.get_api_key()
    payloads = {"test_titles": test_titles, "label_map": label_map, "rules": rules, "X":X, "y":y, "use_transformer": True, "agent_id": agent_id}
    
    # Single async prediction
    print('[==] Single sync prediction: (using single text: "Opening Thesis.docx")')
    sync_result = agent.predict_sync(
        texts=payloads,
        api_key=api_key,
        client_ip="127.0.0.1",
        method='advanced'
    )

    print(f"[=] Sync Result: {sync_result}")


    print("[==] Single async prediction: (using single text: Opening Thesis.docx)")
    result = await agent.predict_async(
        texts=payloads,
        api_key=api_key,
        client_ip="127.0.0.1",
    )
    print(f"[=] Result: {result.get('prediction')} ({result.get('confidence', 0)}")
    
    # Batch async predictions (parallel!)
    print("\n[=] Batch async predictions (parallel):")
    texts = [
        "Watching YouTube",
        "Programming in VS Code",
        "Checking Slack messages",
        "Reading documentation",
        "Taking a break"
    ]
    
    start_time = time.time()
    results = await agent.predict_batch_async(texts, timeout=60, api_key=api_key)
    elapsed = time.time() - start_time
    
    for result in results:
        print(f"[=] '{result['text']}' → {result['prediction']} ({result['confidence']:.1%})")
    
    print(f"\n[=] Completed {len(texts)} predictions in {elapsed:.2f}s")
    
    # Get queue stats
    stats = agent.get_queue_stats()
    print(f"[=] Queue stats: {stats}")
    
    await agent.shutdown()




def initiate_cohesive_agent_deployment_test(pipeline, test_titles, label_map, rules, X, y, agent_id, filename, title_name, label_name, manager):
    print("\n" + "="*60)
    print("🔮 = TESTING COHESIVE AGENT DEPLOYMENT WITH ASYNC MANAGER = ")

    print('Test 1 of Multi agent cluster')
    asyncio.run(run_secure_agent_cluster(pipeline=pipeline, test_titles=test_titles, label_map=label_map, rules=rules, X=X, y=y, agent_id=agent_id, filename=filename, title_name=title_name, label_name=label_name, manager=manager))
      
    print("\n1. Basic async with result queue")
    asyncio.run(example_async_with_result_queue(pipeline=pipeline, test_titles=test_titles, label_map=label_map, rules=rules, X=X,y=y, agent_id=agent_id, filename=filename, title_name=title_name, label_name=label_name))
    

# async manager setup examples
def initiate_prediction_usage(pipeline, manager, predict_wrapper, test_titles, label_map, rules, X, y):
    """Basic synchronous usage."""
    # Use context manager (auto start/stop)
    api_key = 'my-ultra-safe-secret-key-for-authentication'

    with predict_wrapper as wrapper:
        print('[==] Initiating regular prediction')
        texts = {'test_titles': test_titles, 'label_map': label_map, 'rules': rules, "X":X, "y":y,'use_transformer': True}
        regular_predict = wrapper.predict(
        texts=texts, 
        timeout=pipeline.timeout,
        retries=None,
        api_key=api_key,
        client_ip=None)

        print('[==] Initiating advanced batch prediction')
        predicted_output = wrapper.advanced_batch_prediction(test_titles, label_map, rules, X=None, y=None, api_key=api_key, client_ip=None)


def initiate_with_retries(pipeline, manager, wrapper, test_titles, label_map, rules, X, y):
    """Example with retry logic."""
    
    try:
        # Will retry up to 5 times
        texts = {"test_titles": test_titles, "label_map": label_map, "rules": rules, "X":X, 'y':y, "use_transformer": True}
        result = wrapper.predict(texts, timeout=60, retries=None, api_key=None)
        advanced_result, chosen_label, confidence = wrapper.advanced_prediction_method(manager, test_titles, label_map, rules, X=X, y=y, method='Transformer_included')
        print(f"[=] Result after retries: {result}")
        print(f"[=] Advanced Result: {chosen_label} || ({confidence:.1%})")

    except Exception as e:
        print(f"[!] Failed after retries: {e}")
    finally:
        wrapper.stop()


def initiate_graceful_shutdown(pipeline, wrapper):
    """Example showing graceful shutdown."""
   
    # Submit many async requests
    for i in range(10):
        wrapper.predict_async(f"[=] Request {i}")
    
    # Wait for idle with timeout
    if wrapper.wait_for_idle(timeout=30):
        print("[+] All requests completed")
    else:
        print("[!] Some requests still pending")
    
    # Graceful shutdown
    wrapper.stop()

def AsyncWrappertest(pipeline, prediction_manager, test_titles, label_map, rules, X, y):
    print("\n" + "="*60)
    print("🔮 = TESTING ASYNCHRONOUS PREDICTION WRAPPER = ")
    print("="*60)

    api_key = 'my-ultra-safe-secret-key-for-authentication'

    config = SecurityConfig(
            max_text_length=10000,
            max_queue_size=100,
            rate_limit_requests=60,  # 60 per minute
            require_api_key=True,
            max_pending_tasks=50,
            request_timeout=30.0,

            # Start with no IP restrictions, add via admin API
            allowed_ips=[],
            blocklisted_ips=[],
            require_bootstrap_auth = False
        )

    wrapper = PipelineAsyncManager(pipeline, 
              prediction_manager, 
              config=config, 
              state_file=None, 
              security_level=SecurityLevel.PRODUCTION,
              api_key=api_key, 
              max_workers=4, 
              task_timeout=30, 
              max_retries=3 )

    wrapper.start(method='Transformer_included', bootstrap_token=None)
    
    logging.basicConfig(level=logging.INFO)
    
    # Run examples
    initiate_prediction_usage(pipeline, prediction_manager, wrapper, test_titles, label_map, rules, X, y)
    initiate_with_retries(pipeline, prediction_manager, wrapper, test_titles, label_map, rules, X, y)
    initiate_graceful_shutdown(pipeline, wrapper)

    print("\n✅ Asynchronous prediction wrapper test completed successfully.")


def PermissiveTest():
    print("\n" + "="*60)
    print("🔮 = TESTING HYBRID PREDICTION SYSTEM = ")
    print("="*60)

    print("📖 Loading labels from text file with CSV format...")
    filename = input('|| Insert Filename (press N to skip): ')
    title = input('|| Insert Title name you have in your file (press N to skip): ')
    label = input('|| Insert Label name you have in your file (press N to skip): ')
    agent_id = input('|| Insert Agent ID for distributed inference (press N to skip): ')

    print('📖 Need to insert custom memory Name for the AI')
    file = input('|| Insert Memory name: ')
    print('📖 Need to insert custom SSL certificate and key files for secure communication')
    print('[=] Important for secure external-device Peer to peer between Agents (optional)')

    cert_file = input('|| Insert SSL certificate file (press N to skip): ')
    key_file = input('|| Insert SSL key file (press N to skip): ')
    if cert_file != 'N':
        cert_file = cert_file
    else:
        cert_file = None
    if key_file != 'N':
        key_file = key_file
    else:
        key_file = None

    if file:
        pipeline = IntegratedPipeline(file, use_async=True, agent_port=5001, ssl_cert_file=cert_file, ssl_key_file=key_file, bind_host='127.0.0.1', security_level='PRODUCTION')
    else:
        print('|| Using original csv_file.pkl file as fallback...')
        pipeline = IntegratedPipeline('csv_file.pkl', use_async=True, agent_port=5001, ssl_cert_file=cert_file, ssl_key_file=key_file, bind_host='127.0.0.1', security_level='PRODUCTION')

    manager = PipelinePredictionManager(pipeline, label_csv='ManualsTraining.txt', target_title='window_title', label='label')

    pipeline.distribution.predict_manager = manager
    if agent_id == 'N':
        agent_id = 'local'

    if filename and title and label and filename != 'N':
        titles, y_raw, label_map = manager.load_labels_from_csv(filename, title, label)
        print(f"✅ Loaded {len(titles)} labeled examples")
    else:
        print('|| Fallback to Original given files...')
        titles, y_raw, label_map = manager.load_labels_from_csv('ManualsTraining.txt', 'window_title', 'label')
        print(f"✅ Loaded {len(titles)} labeled examples")


    print('== Training Model... ==')
    loss_history = pipeline.train(titles, y_raw)

    test_titles = [
    ("Opening Thesis.docx", "slight_work"),
    ("Watching YouTube and Google Chrome", "distracted"),
    ("Watching Slack", "communication"),
    ("Programming in Visual Studio Code", "focused_work"),
    ("Watching netflix.com - Chrome", "break"),
    ]
    rules = [
        # === WORK / PRODUCTIVITY ===
        (r'code|programming|develop|debug|compile|script', 'focused_work'),
        (r'vscode|visual_studio|ide|terminal|shell', 'focused_work'),
        (r'notion|evernote|onenote|notes|todo|task', 'productive'),
        (r'slack|teams|discord|zoom|meeting|call', 'communication'),
        (r'email|gmail|outlook|inbox|mail', 'communication'),
        
        # === ENTERTAINMENT ===
        (r'youtube|netflix|twitch|stream|video', 'entertainment'),
        (r'music|spotify|soundcloud|audio|player', 'entertainment'),
        (r'game|gaming|steam|epic|play', 'gaming'),
        (r'facebook|instagram|tiktok|social|post', 'social_media'),
        
        # === BROWSING ===
        (r'chrome|firefox|edge|safari|browser', 'browsing'),
        (r'google|search|wiki|wiki|article', 'information'),
        (r'stackoverflow|github|docs|documentation', 'research'),
        
        # === FILE MANAGEMENT ===
        (r'download|folder|file|document|pdf', 'file_work'),
        (r'dropbox|onedrive|google_drive|cloud', 'cloud_storage'),
        (r'zip|rar|extract|compress|archive', 'file_management'),
        
        # === SYSTEM / DEV ===
        (r'terminal|cmd|powershell|bash|shell', 'system_work'),
        (r'docker|kubernetes|container|deploy', 'devops'),
        (r'git|commit|push|pull|branch|merge', 'version_control'),
        (r'test|unit|debug|error|exception', 'testing'),
        
        # === DATA / ANALYSIS ===
        (r'excel|spreadsheet|sheet|csv|table', 'data_work'),
        (r'python|r|sql|query|database', 'data_analysis'),
        (r'chart|graph|visualization|dashboard|plot', 'visualization'),
        
        # === COMMUNICATION ===
        (r'whatsapp|telegram|signal|messenger', 'messaging'),
        (r'zoom|meet|webex|video_call', 'video_call'),
        (r'calendar|schedule|event|meeting|appointment', 'scheduling'),
        
        # === CREATIVE ===
        (r'photoshop|illustrator|figma|design|canvas', 'creative'),
        (r'premiere|final_cut|video_edit|render', 'video_editing'),
        (r'blender|3d|model|render|animation', '3d_work'),
        
        # === LEARNING ===
        (r'coursera|udemy|edx|course|learn', 'learning'),
        (r'book|ebook|reader|pdf|document', 'reading'),
        (r'podcast|audiobook|listen|lecture', 'audio_learning'),
        
        # === UTILITY ===
        (r'calculator|converter|tool|utility', 'utility'),
        (r'weather|clock|timer|alarm|reminder', 'utility'),
        (r'translate|language|dictionary|translate', 'utility'),
        
        # === RARITY PATTERNS ===
        (r'common|not_common|twitch|debian|watch', 'very abundant'),
        (r'bit-common|pycharm|unix|code|programming|python|java', 'bit-abundant'),
        (r'medium|discord|teams|zoom|linux_mint|message', 'abundant'),
        (r'rare|pdf|word|macOS|ubuntu|document', 'not abundant'),
        (r'ultra|firefox|edge|browser|unix|web', 'medium rare'),
        (r'ultra_rare|music|linux|Home_linux_router', 'bit-rare'),
        (r'medium-rare|steam|red_hat_enterprise_linux|play|windows', 'very rare'),
        (r'rarer|oracle|system|config|server_linux_router', 'absolute rare'),
    ]

    running = True
    X, y = None, None
    while running:
        permission = input('|| Allow Hybrid prediction test? [Y/N]: ')

        if permission == 'Y' or permission == 'y':
            print('== TEST 1: (titles only without transformer) ==')
            advanced_result = manager.advanced_prediction_method(
            [t[0] for t in test_titles],
            label_map,
            rules,
            X=X, y=y,
            show_proba=True
            )
            time.sleep(5)
        
            print('== TEST 2: (advanced predictions with expected labels and also use transformer)')
            advanced_results = manager.advanced_prediction_method(
            test_titles,  # Titles with expected labels
            label_map,
            rules,
            X=X, y=y,
            show_proba=True,
            top_k=4,
            use_transformer=True,
            return_attention=True
        
            )
        
            print("\n📊 COMPARISON: MLP-only vs Hybrid")
            mlp_only = manager.regular_prediction_method(
            [t[0] for t in test_titles],
            label_map,
            rules,
            X=X, y=y,
            use_transformer=False
            )
        
            hybrid = manager.regular_prediction_method(
            test_titles,
            label_map,
            rules,
            X=X, y=y,
            use_transformer=True       
            )
            print('== CompletePipeline Successfully tested! ==')

        permission_continue = input('|| Do you want to test the Asynchronous wrapper for multiple predictions? [Y/N]: ')
        if permission_continue == 'Y' or permission_continue == 'y':
            AsyncWrappertest(pipeline, manager, test_titles, label_map, rules, X, y)
            print('== Asynchronous wrapper Successfully tested! ==')

        cohesive_permission = input('|| Do you want to test the Cohesive Agent Deployment with Async Manager? [Y/N]: ')
        if cohesive_permission == 'Y' or cohesive_permission == 'y':
            if not (filename and title and label and filename != 'N'):
                print('[=] Searching fallback filename: ManualsTraining.txt, window_title, label')
                initiate_cohesive_agent_deployment_test(pipeline, test_titles, label_map, rules, X, y, agent_id, 'ManualsTraining.txt', 'window_title', 'label', manager)
            else:
                initiate_cohesive_agent_deployment_test(pipeline, test_titles , label_map, rules, X, y, agent_id, filename, title, label, manager)
            print('== Cohesive Agent Deployment Successfully tested! ==')

        else:
            running = False
            print('|| Program Prediction test aborted!')
            pass


if __name__ == "__main__":
    try:
        PermissiveTest()
    except Exception as e:
        print(f'|| Program Crashed...,  Error: {e}')
        traceback.print_exc()
        pass


