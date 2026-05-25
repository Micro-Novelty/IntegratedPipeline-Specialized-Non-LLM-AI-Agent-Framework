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
# primitives.py
# Core enums, lightweight dataclasses, and the thread-safe Singleton base.
# Every other module imports from here; this module has no local dependencies.
# ---------------------------------------------------------------------------

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


@dataclass
class SecureMessage:
    # Secure message wrapper
    id: str
    type: str
    payload: Any
    timestamp: float
    signature: str = ""

@dataclass


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