"""
caching_security.py
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
from . import async_manager

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
                    if isinstance(entry, dict) and entry['source'] != 'automatic_verified' and not entry['source'].startswith('automatic'):
                          
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

                combined_env_sim = ((mlp_sim + seq_sim) / 2) * anisotropy
                deterministic_modelling_sim = ((mlp_sim + seq_sim) / 2) * AME 

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
    # logging/serialization from injection from unknown source.
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
            # even when newlines are allowed, escape them for log safety in here although
        else:
            text = InputSanitizer._CONTROL_CHARS_PATTERN.sub('', text)

        escape_patterns = [
            r'\\x[0-9a-fA-F]{2}',   # literal "\x41" style escapes.
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
        """Visibility into key store health — same pattern as async_manager.WorkerPool.get_health()."""
        with self._lock:
            active   = sum(1 for k in self.keys.values() if k.get('is_active'))
            inactive = len(self.keys) - active
            return {
                'total_keys'   : len(self.keys),
                'active_keys'  : active,
                'inactive_keys': inactive,
                'at_capacity'  : len(self.keys) >= self.max_keys,
            }



