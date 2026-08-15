import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
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
import glob
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



