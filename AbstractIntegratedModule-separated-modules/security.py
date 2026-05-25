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
# security.py
# Security primitives: configuration dataclass, level enum, error type,
# role enum, token-bucket rate limiter, input sanitizer, and API key manager.
# Depends on: primitives (SecurityError is self-contained here)
# ---------------------------------------------------------------------------

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


class RateLimiter:
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