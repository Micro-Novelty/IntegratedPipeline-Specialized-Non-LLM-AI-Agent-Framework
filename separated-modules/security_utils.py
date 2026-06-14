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



class RateLimiter:
    '''
    Token-bucket rate limiter (thread-safe).

    Algorithm
    ---------
    The bucket starts full (tokens = requests_per_minute).  Each acquire()
    call refills the bucket proportionally to the elapsed time since the last
    refill (continuous refill model), then attempts to consume one token.

    acquire() returns True if a token was available (request allowed) or
    False if the bucket was empty (request denied).  The internal lock
    ensures correctness under concurrent access from multiple threads.

    Parameters
    ----------
    requests_per_minute : Maximum request rate.  Default 60 (1/sec on average).
    '''
    # Token bucket rate limiter
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.tokens = requests_per_minute
        self.last_refill = time.time()
        self._lock = threading.Lock()
    
    def acquire(self) -> bool:
        with self._lock:
            now = time.time()
            # Refill tokens
            elapsed = now - self.last_refill
            new_tokens = elapsed * (self.requests_per_minute / 60)
            self.tokens = min(self.requests_per_minute, self.tokens + new_tokens)
            self.last_refill = now
            
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False

class InputSanitizer:
    '''
    Static text sanitisation and validation utilities.

    sanitize_text(text, max_length)
        Strips ASCII control characters (except tab/newline), rejects inputs
        longer than max_length, and removes common injection-style escape
        sequences (hex \\xNN, unicode \\uNNNN) from the string.  Intended to
        be called before passing user-supplied text to the model or logging it.

    Raises SecurityError (a subclass of Exception defined in this module) on
    type violations or length overflows so callers can catch it uniformly.
    '''
    # Sanitize and validate inputs
    
    @staticmethod
    def sanitize_text(text: str, max_length: int = 10000) -> str:
        if not isinstance(text, str):
            raise SecurityError("[-] Input must be a string")
        
        # RemovING null bytes and control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        
        # Limit length
        if len(text) > max_length:
            raise SecurityError(f"[-] Text exceeds maximum length of {max_length}")
        
        # Remove potential injection patterns (for logging/serialization)
        dangerous_patterns = [
            r'[\x00-\x08\x0b\x0c\x0e-\x1f]',  # Control chars
            r'\\x[0-9a-fA-F]{2}',  # Hex escapes
            r'\\u[0-9a-fA-F]{4}',  # Unicode escapes
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, text):
                # Log but don't block - just escape
                text = re.sub(pattern, '?', text)
        
        return text.strip()
    
    @staticmethod
    def validate_batch_size(size: int, max_batch: int = 100) -> bool:
        if size <= 0 or size > max_batch:
            raise SecurityError(f"[-] Batch size must be between 1 and {max_batch}")
        return True

class ApiKeyManager:
    '''
    Thread-safe API key lifecycle manager with automatic rotation.

    Key storage
    -----------
    Keys are never stored in plain text.  Only the SHA-256 hash of the raw
    key value is kept in self.keys, so a DB/memory dump does not reveal active
    keys.

    Rotation
    --------
    validate_key() automatically deactivates keys that are older than
    rotation_days (default 30).  Callers receive False on the same request
    that triggers deactivation, so they can prompt the user to regenerate.

    Methods
    -------
    generate_key(metadata, key_value)
        Creates a new key.  If key_value is provided (e.g. a user-supplied
        secret) it is hashed and registered; otherwise a cryptographically
        random URL-safe 32-byte token is generated via secrets.token_urlsafe.
    validate_key(api_key) → bool
        Checks existence, activation status, and age.
    revoke_key(api_key) → bool
        Deactivates a key immediately.
    '''
    # Manage API keys with rotation
    
    def __init__(self, rotation_days: int = 30):
        self.keys: Dict[str, dict] = {}  # key_hash -> {created_at, last_used, metadata}
        self.rotation_days = rotation_days
        self._lock = threading.Lock()
    
    def generate_key(self, metadata: dict = None, key_value: str = None) -> str:
        # Generate a new API key
        if key_value:
            # Use provided key value
            raw_key = key_value
        else:
            # Generate random key
            raw_key = secrets.token_urlsafe(32)    

        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        
        with self._lock:
            self.keys[key_hash] = {
                'created_at': datetime.now(),
                'last_used': None,
                'metadata': metadata or {},
                'is_active': True
            }
        
        return raw_key
    
    def validate_key(self, api_key: str) -> bool:
        # Validate an API key
        if not api_key:
            return False
        
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        with self._lock:
            key_info = self.keys.get(key_hash)
            if not key_info or not key_info.get('is_active', False):
                return False
            
            # Check rotation
            age = (datetime.now() - key_info['created_at']).days
            if age >= self.rotation_days:
                key_info['is_active'] = False
                return False
            
            key_info['last_used'] = datetime.now()
            return True
    
    def revoke_key(self, api_key: str) -> bool:
        # Revoke an API key
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        with self._lock:
            if key_hash in self.keys:
                self.keys[key_hash]['is_active'] = False
                return True
        return False

