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



class MessagePriority(Enum):
    '''
    Priority levels for messages in the async/threaded queues.
    Lower integer value = higher urgency.  Messages with equal priority are
    processed in FIFO order (oldest creation timestamp first).
    '''
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
    '''
    Lightweight authenticated envelope for inter-agent socket messages.

    Fields
    ------
    id        : Unique message identifier (UUID string recommended).
    type      : Logical message type string (e.g. 'predict_request').
    payload   : Arbitrary data to transmit.  Must be pickle-serialisable.
    timestamp : Unix epoch float; used by the receiver for replay-attack
                detection (reject messages older than a configurable window).
    signature : HMAC-SHA256 hex digest computed over the sorted, pickled
                payload by AgentDistributedInference._sign_message().
                Empty string means the message has not yet been signed.
    '''
    # Secure message wrapper
    id: str
    type: str
    payload: Any
    timestamp: float
    signature: str = ""

@dataclass
class Message:
    '''
    Full-featured message object used by both AsyncMessageQueue and
    ThreadedMessageQueue.

    Routing
    -------
    sender / recipient : Logical agent identifiers (string).
    type               : Handler dispatch key.  Each queue registers handlers
                         keyed on this string (e.g. 'predict', 'sync_memory').

    Reliability
    -----------
    retry_count / max_retries : A failed handler is re-queued up to max_retries
                                times before the message is forwarded to the
                                dead-letter queue.
    timeout / is_expired      : Messages older than `timeout` seconds are
                                dropped silently with a warning log.

    Priority ordering
    -----------------
    Comparison operators (__lt__ etc.) allow Python's heapq / PriorityQueue to
    sort messages.  Within the same priority level, older messages win (FIFO).
    '''
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
    
    @property
    def age(self) -> float:
        """Age of message in seconds."""
        return time.time() - self.created_at
    
    @property
    def is_expired(self) -> bool:
        """Check if message has expired."""
        return self.age > self.timeout
    
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
        """Make Message hashable (useful for sets/dicts)."""
        return hash(self.id)


@dataclass
class SecurityConfig:
    """
    Security configuration for PipelineAsyncManager.

    Rate limiting
    -------------
    rate_limit_requests   : Max requests allowed per `rate_limit_window` seconds
                            for a regular (non-admin) caller.
    admin_rate_limit      : Separate, higher limit for admin-token callers.
    admin_bypass_rate_limit : When True, admin callers skip the bucket entirely
                              unless admin_rate_limit < 999.

    IP access control
    -----------------
    allowed_ips           : CIDR-capable whitelist.  Empty list = allow all IPs.
    blocklisted_ips       : Explicit deny list, checked before whitelist.
    admin_allowed_ips     : Separate whitelist for admin callers when
                            enforce_admin_ip_whitelist is True.

    Resource guards (NOT authentication)
    -------------------------------------
    max_cpu_percent / max_memory_percent / min_disk_space_mb:
        PipelineAsyncManager._start_with_limits() refuses to start if any
        system resource threshold is exceeded, preventing runaway launches.
    min_start_interval    : Minimum seconds between successive start() calls;
                            blocks rapid restart / crash-loop attacks.
    max_consecutive_failures : Circuit-breaker threshold; after this many
                            failed start attempts the breaker opens.

    Bootstrap auth (optional, only enforced in HARDENED security level)
    --------------------------------------------------------------------
    require_bootstrap_auth : When True, start() demands a one-time bootstrap
                             token whose SHA-256 hash matches bootstrap_token_hash.
    """
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
        # Double-checked locking pattern:
        #   Fast path (no lock) — if the instance already exists, return immediately.
        #   Slow path (with lock) — only one thread can create the instance; a second
        #   check inside the lock guards against two threads both passing the fast path
        #   before either acquires the lock.
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

# ________ UTILITY functions for activations and losses, can be used across different models and architectures, targeted for LSTM. _________

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_deriv(s):          # s = sigmoid(x) already computed
    return s * (1.0 - s)

def tanh_deriv(t):             # t = tanh(x) already computed
    return 1.0 - t ** 2


