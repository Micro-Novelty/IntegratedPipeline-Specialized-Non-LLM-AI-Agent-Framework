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
# async_manager.py
# PipelineAsyncManager  — asyncio-based HTTP-serving wrapper around IntegratedPipeline.
#                         Handles rate limiting, IP allow/block lists, API key auth,
#                         bootstrap protection, and admin bypass rules.
# PipelinePredictionManager — synchronous batch prediction manager with thread-pool
#                         execution and result caching.
# Depends on: security (all security primitives), messaging (AsyncResultQueue, WorkerPool),
#             primitives (WrapperState, AsyncTask), mlp, transformer
# IntegratedPipeline is injected at runtime (passed in __init__) to avoid a circular
# import — pipeline.py must not import async_manager.py.
# ---------------------------------------------------------------------------
from .security import (SecurityConfig, SecurityLevel, SecurityError, AdminRole,
                        RateLimiter, InputSanitizer, ApiKeyManager)
from .messaging import AsyncResultQueue, WorkerPool
from .primitives import WrapperState, AsyncTask
from .mlp import MLP
from .transformer import Transformer

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
                    if not 'test_titles' in texts and 'label_map' in texts and 'rules' in texts:
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
                    else:
                        if 'test_titles' in texts and 'label_map' in texts and 'rules' in texts:
                            result = self._advanced_predict_sync(texts['test_titles'], texts['label_map'], texts['rules'], texts.get('agent_id', 'default'), texts.get('use_transformer', False), timeout)
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


    def _predict_sync(self, text: str, timeout: float) -> Any:
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

    def _advanced_predict_sync(self, test_titles, label_map, rules, agent_id: str, use_transformer: bool=False, timeout: float = 30.0) -> Any:
        # Internal synchronous prediction.

        if not self._loop or not self._loop.is_running():
            raise RuntimeError("[-] Event loop not available")

    
        future = asyncio.run_coroutine_threadsafe(
            self.advanced_predict_async_await(test_titles, label_map, rules, use_transformer, agent_id, timeout),
            self._loop
        )
        
        # Track task for cleanup
        task_id = self._add_task(future, timeout)
        
        try:
            result = future.result(timeout=timeout + 1)
            return result
        finally:
            self._remove_task(task_id)
    
    async def _predict_with_timeout(self, text: str, timeout: float) -> Any:
        # Async prediction with timeout.
        try:
            return await asyncio.wait_for(
                self.pipeline.predict_async_await(text),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            raise FutureTimeoutError(f"[-] Prediction timed out after {timeout}s")
    
    async def advanced_predict_async_await(self, test_titles: list[tuple], label_map: dict, rules: list[tuple], use_transformer: bool=False, agent_id: str=None, timeout: float = 30.0):
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
                    timeout=timeout
                ),
                timeout=timeout+5
            )

        except asyncio.TimeoutError:
            raise FutureTimeoutError(f"[-] Advanced prediction timed out after {timeout}s")
            
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


    def advanced_batch_prediction(self, test_titles, label_map, rules, api_key, client_ip):
        try:
            reverse_label_map = {v: k for k, v in label_map.items()}

            # Batch predictions
            texts = [text[0] for text in test_titles]
            expected_labels = [text[1] for text in test_titles]  # Your ground truth
            predicted_output = []

            text = {"test_titles": test_titles, "label_map": label_map, "rules": rules, "use_transformer": True}

            if not self.pipeline.intents:
                advanced_result = self.predict(
                text, 
                timeout=40,
                retries=None,
                api_key=api_key,
                client_ip=client_ip,
                ) # regular predict to populate pipelines data
            else:
                advanced_result = self.advanced_prediction_method(self.prediction_manager
                , test_titles, label_map, rules, method='Transformer_included')

            print(f'[=+=] Local advanced Prediction result: {advanced_result}')

            print('[=] Initiating Batch prediction for multiple texts...')
            results = self.predict_batch(
                texts=texts,
                timeout=60,
                api_key=api_key)

            print("📊 PREDICTION RESULTS")
           
            for text, expected, probs in zip(texts, expected_labels, results):
                # probs is an array of 8 probability values
                predicted_index = np.argmax(probs)  # Get index with highest probability
                predicted_label = reverse_label_map.get(predicted_index, f"class_{predicted_index}")
                confidence = probs[predicted_index]
                
                # Get top 3 predictions for more insight
                top_3_indices = np.argsort(probs)[-3:][::-1]
                top_3 = [(reverse_label_map.get(idx, f"class_{idx}"), probs[idx]) for idx in top_3_indices]
                
                print(f"\n📌 Input: '{text}'")
                print(f"   [=] Expected: {expected}")
                print(f"   🎯 Predicted: {predicted_label} ({confidence:.1%})")
                
                # Show top 3 possibilities
                print(f"   🔍 Top possibilities:")
                for label, conf in top_3:
                    bar = '█' * int(conf * 20)
                    print(f"[•]   {label:<25} {bar} {conf:.1%}")
                
                # Match expected vs predicted
                print('===== COMPARISON MATCHING =====')
                if predicted_label == expected:
                    print(f"[✅] CORRECT LABEL!")
                else:
                    print(f"[❌] INCORRECT LABEL (expected: {expected})")

                predicted_output.append(predicted_label)

            # Get stats
            print(f"[=] Stats: {self.get_stats()}")        
            return predicted_output 

        except Exception as e:
            print(f'[=] Error in advanced batch prediction: {e}')
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

    def advanced_prediction_method(self, manager, test_titles, label_map, rules, method='Transformer_included'):
        # starting PredictionManager for advanced prediction
        try:
            if method == 'Transformer_included':
                print('== PREDICTION 1: (advanced predictions with expected labels transformer included)')
                result, chosen_label, confidence = manager.advanced_prediction_method(
                test_titles,  # Titles with expected labels
                label_map,
                rules,
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
                show_proba=True
                )
        except Exception as e:
            logger.error(f"[-] Advanced prediction failed: {e}")
            result = None
            confidence = 0.0
            chosen_label = None

        return result, chosen_label, confidence


    def predict_async(self, text: str, test_titles: list[tuple]= None, label_map: dict=None, rules: list[tuple]=None, callback: Optional[Callable] = None) -> str:
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
        
        logger.info("[-] PipelineAsyncWrapper stopped")
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
            self.titles, self.y_raw, self.label_map = [], [], {}

        print(f"✅ Loaded {len(self.titles)} labeled examples")

    def load_labels_from_csv(self, filename, target_title, label):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(script_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"❌ File not found: {filepath}")
            return [], [], {}
        
        # Read CSV file
        df = pd.read_csv(filepath)
        
        print(f"✅ Loaded CSV with columns: {list(df.columns)}")
        
        # Extract titles and labels
        titles = df[target_title].tolist()
        string_labels = df[label].tolist()
        
        # Remove quotes if they're still there
        titles = [t.strip('"') for t in titles]
        
        print(f"📊 Found {len(titles)} examples")
        print(f"📊 Labels: {set(string_labels)}")
        
        # Create numeric labels
        unique_labels = sorted(set(string_labels))
        label_map = {label: i for i, label in enumerate(unique_labels)}
        y = [label_map[label] for label in string_labels]
        
        return titles, y, label_map



    def regular_prediction_method(self, titles, label_map, rules, show_proba=False, top_k=3, use_transformer=True):
        try:
            print(f"\n[🚀] Regular Prediction for labels with {len(titles)} titles...")
            self.pipeline.titles = titles
            self.pipeline.labels = label_map

            reverse_map = {v: k for k, v in label_map.items()}
            num_classes = len(label_map)

            dataset, X = self.pipeline.data_preparation(titles, label_map)      
            _, y, _, _ = self.pipeline.mlp_training_features(rules, dataset)
            self.pipeline.transformer_utilities(rules, dataset, X) 
            input_ids, _ = self.pipeline.input_encoding(dataset)


            if use_transformer and hasattr(self.pipeline, 'vocab') and self.pipeline.vocab:
                print("[🔄] Using Transformer for probability calibration")
            
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
            
            # Get transformer probabilities
                trans_probs, attn_weights = self.pipeline.model2.forward(input_ids)
            else:
                print("⚡ Using MLP only for predictions")
                trans_probs = None
        
            if not hasattr(self.pipeline, 'tfidf') or self.pipeline.tfidf is None:
                self.pipeline.initialize_fitting(titles)
            
            # Prepare texts for MLP
            if isinstance(titles[0], tuple):
                mlp_titles = [t[0] for t in titles]
            else:
                mlp_titles = titles
                
            X_tfidf = self.pipeline.tfidf.transform(mlp_titles).toarray()            
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
                num_classes = mlp_probs.shape[1] if mlp_probs.ndim > 1 else len(mlp_probs)

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
            
            # Display results
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

            verbose = False
            if float(results[0]['confidence']) < self.pipeline.confidence_threshold:
                verbose = True
            
            self.display_hybrid_results(results, top_k, verbose=verbose)


            # Use results directly - they already contain calibrated predictions
            chosen_label = results[0]['predicted'] if results else None
            confidence = results[0]['confidence'] if results else None

            if isinstance(chosen_label, int) or isinstance(chosen_label, np.integer):
                chosen_label = str(chosen_label)
                
            # Only recalibrate if models disagreed AND we have valid results
            if results and not results[0].get('models_agree', True):
                print("\n[⚠️] Disagreement detected between MLP and Transformer predictions. Using calibrated probabilities for final decision.")
                calibrated_probs = self.pipeline.hybrid_prediction(input_ids, X)
                
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
              
        except Exception as e:
            print(f"[=] Error during prediction: {e}")
            results = []

        return results

    def hybrid_model_prediction(self, datasets, X_raw):
        self.pipeline.transformer_utilities(datasets, X_raw)
        input_datasets = self.pipeline.transformer_input_encoding([i[0] for i in datasets])

        probs = self.model.predict_proba(input_datasets, X_raw, type='Hybrid')[0]
        pred = self.model.hybrid_prediction(input_datasets, X_raw)

        return probs, pred

    def robust_prediction(self, pipeline, titles, label_map, show_proba=True, top_k=3):
        self.pipeline.titles = titles
        self.pipeline.labels = label_map   

        try:

            datasets, X_raw = self.pipeline.data_preparation(titles, label_map)
            reverse_map = {v: k for k, v in label_map.items()}
            
            self.pipeline.transformer_utilities(datasets, X_raw)
            input_datasets = self.pipeline.transformer_input_encoding(datasets)
            pred_probs = self.pipeline.predict_proba(input_datasets, X_raw, type='Hybrid')[0]
            pred_result = self.pipeline.hybrid_prediction(input_datasets, X_raw)

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
        
            n_samples = len(titles)
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

        except Exception as e:
            print(f"[=] Error during robust prediction: {e}")
            predicted = None
            predicted_confidence = None
        return predicted, predicted_confidence
        
    def calculate_entropy(self, probs):
        return -np.sum(probs * np.log(probs + 1e-10), axis=-1)


    def advanced_prediction_method(self, titles, label_map, rules,
                                show_proba=False, top_k=3, 
                                use_transformer=True,
                                return_attention=False,
                                save_results=True):
        try:
            eps = 1e-5
            trans_probs = None
            attn_weights = None
            sequence_ids = None

            print("\n[🚀] Starting Advanced Hybrid Prediction Method")

            reverse_map = {v: k for k, v in label_map.items()}
            num_classes = len(label_map)

            self.pipeline.titles = titles
            self.pipeline.labels = label_map
         
            dataset, X = self.pipeline.data_preparation(titles, label_map)    
            _, y, _, _ = self.pipeline.mlp_training_features(rules, dataset)
            self.pipeline.transformer_utilities(rules, dataset, X)
            input_ids, _ = self.pipeline.input_encoding(dataset)
          
            if use_transformer and hasattr(self.pipeline, 'vocab') and self.pipeline.vocab:
                use_embedded = False
                print("\n[🔄] Running dual predictions (MLP + Transformer)")
            
                input_ids_list = []
                for title in titles:
                    if isinstance(title, tuple):
                        title = title[0]
                    ids = self.pipeline.encode(title, self.pipeline.vocab)
                    input_ids_list.append(np.array(ids))
                
                input_ids = np.array(input_ids_list)
                
                # Get transformer predictions with attention
                if self.pipeline.anisotropy_measurement(input_ids) < self.pipeline.confidence_threshold:
                    print("⚡ Low anisotropy detected on input, relying on sequence encoding for input...")
                    sequence_ids = self.pipeline.sequence_encoding(dataset)
                    use_embedded = True
                    trans_probs, attn_weights = self.pipeline.model2.forward(sequence_ids, embedded=use_embedded)
                else:
                    print("⚡ Anisotropy above threshold, using standard input encoding for transformer...")
                    trans_probs, attn_weights = self.pipeline.model2.forward(input_ids, embedded=use_embedded)    

            else:
                print("\n⚡ Running MLP-only predictions")
                print("⚡ Note: Transformer not available, so Transformer results will be replaced with MLP results.")

            if X is None or len(X) == 0 or isinstance(X, int) or (isinstance(X, np.ndarray) and X.size == 0):
                # Get MLP predictions
                if isinstance(titles[0], tuple):
                    mlp_titles = [t[0] for t in titles]
                else:
                    mlp_titles = titles
                
                if not hasattr(self.pipeline, 'tfidf') or self.pipeline.tfidf is None:
                    self.pipeline.initialize_fitting(mlp_titles)
                                    
                X = self.pipeline.tfidf.transform(mlp_titles).toarray()

            # MLP forward pass
            if hasattr(self.pipeline.mlp, 'predict_proba'):
                mlp_probs = self.pipeline.mlp.predict_proba(X)
            else:
                logits = self.pipeline.mlp.forward(X)
                mlp_probs = self.pipeline._softmax(logits)
            
             # Validate all MLP predictions at once
            mlp_pred_indices = np.argmax(mlp_probs, axis=1)
            if num_classes <= 0:
                num_classes = mlp_probs.shape[1] if mlp_probs.ndim > 1 else len(mlp_probs[0])

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
            target_probs = self.pipeline.predict_proba(input_ids, X, type='Hybrid', embedded=True)
            target_probs = target_probs[:mlp_probs.shape[0], :mlp_probs.shape[1]] 
            target_pred_indices = np.argmax(target_probs, axis=1)          

            results = []
            attention_data = [] if return_attention else None

            for i, title in enumerate(titles):
                # Parse input
                if isinstance(title, tuple):
                    display_title = title[0]
                    expected_label = title[1] if len(title) > 1 else None
                else:
                    display_title = title
                    expected_label = None
                
                # MLP prediction                 
                mlp_class_idx = mlp_pred_indices[i]
                mlp_class_idx = min(mlp_class_idx, num_classes - 1)  # Clamped to valid range
                if mlp_class_idx < 0 or mlp_class_idx >= num_classes:
                    mlp_class_idx = 0  # Safe default              

                mlp_confidence = mlp_probs[i][mlp_class_idx]
                mlp_label = reverse_map.get(mlp_class_idx, f"unknown_{mlp_class_idx}")

                target_confidence = mlp_confidence
                target_probs = mlp_probs
                target_pred_indices = mlp_pred_indices
                target_class_idx = mlp_class_idx
            
                if float(mlp_confidence) < self.pipeline.confidence_threshold:
                    target_class_idx = target_pred_indices[i]
                    target_confidence = target_probs[i][target_class_idx]
                
                # Transformer prediction and blending
                if trans_probs is not None:
                    trans_probs_i = trans_probs[i]
                    trans_class_idx = np.argmax(trans_probs_i)
                    if isinstance(trans_probs_i, float):
                        trans_confidence = target_confidence
                    else:
                        trans_confidence = trans_probs_i[trans_class_idx]

                    trans_label = reverse_map.get(trans_class_idx, f"unknown_{trans_class_idx}")

                    calibration = self.pipeline._calibrate_probs(target_probs, target_pred_indices, attn_weights, input_ids)
                    # Blend predictions (MLP decides class, transformer calibrates confidence)
                    mlp_weight = mlp_confidence / (target_confidence + trans_confidence + eps)
                    trans_weight = trans_confidence / (target_confidence + trans_confidence + eps)
                        
                    calibration_weighting = calibration[target_class_idx] if target_class_idx < len(calibration) else 0.0
                        
                    # Weighted blend: calibration_weighting * calibrated + (1-weight) * mlp
                    final_probs = mlp_weight * target_probs[i][:len(calibration)] + trans_weight * calibration[i][:len(calibration)]
                 
                    final_class_idx = target_class_idx

                    try:
                        final_confidence = final_probs[final_class_idx]
                    except IndexError:
                        final_confidence = np.max(final_probs) if isinstance(final_probs, np.ndarray) else final_probs

                    if isinstance(final_confidence, np.ndarray):
                        final_confidence = np.max(final_confidence)

                    # Calculate agreement
                    agreement = mlp_class_idx == trans_class_idx
                else:
                    final_probs = mlp_probs[i]
                    final_class_idx = mlp_class_idx
                    final_confidence = mlp_confidence[0] if isinstance(mlp_confidence, np.ndarray) else mlp_confidence
                    if isinstance(final_confidence, np.ndarray) or isinstance(final_confidence, list):
                        final_confidence = np.max(final_confidence)

                    trans_label = None
                    trans_confidence = None
                    agreement = True
                
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
                    'models_agree': bool(agreement)
                }
                
                if trans_label is not None:
                    result['transformer_prediction'] = trans_label
                    result['transformer_confidence'] = float(trans_confidence)
                
                # Add top-k predictions
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
                    if trans_probs:
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
            
            # Display results
            verbose = False
            if float(results[0]['confidence']) < self.pipeline.confidence_threshold:
                verbose = True
            
            self.display_hybrid_results(results, top_k, verbose=verbose)
    
            chosen_label = results[0]['predicted'] if results else None
            confidence = results[0]['confidence'] if results else None
            if isinstance(chosen_label, int) or isinstance(chosen_label, np.integer):
                chosen_label = str(chosen_label)

            print(f"\n[🎯] Initial chosen label for input: {chosen_label} || Confidence: {confidence:.1%}")
            time.sleep(3)

            if results[0].get('models_agree', True) and confidence > self.pipeline.confidence_threshold and not chosen_label.startswith("unknown"):
                print(f"\n[🎯] Proper Confidence of Final chosen label for input: {chosen_label} || Confidence: {confidence:.1%}")
                return results, chosen_label, confidence
            
            # Only recalibrate if models disagreed
            elif results and not results[0].get('models_agree', True) or not self.pipeline.agreement:
                need_peer_condition = not results[0].get('models_agree', True) and self.pipeline.peer_assistance_threshold > 0.3
                print("\n[⚠️] Disagreement detected between MLP and Transformer predictions. Using calibrated probabilities for final decision.")
                if need_peer_condition:
                    print('|| Uncertain advanced prediction, requesting peer assistance if allowed...')
                    final_probs = self.pipeline._handle_distributed_connections(final_probs, attn_weights, input_ids, agreement)   

                elif not results[0].get('models_agree', True) and confidence > self.pipeline.confidence_threshold:
                    if final_confidence is not None and confidence < self.pipeline.confidence_threshold:
                        print("\n[⚠️] High confidence detected, but both models don't agree. Using calibrated probabilities for final decision to ensure robustness.")
                        final_probs = self.pipeline.hybrid_prediction(rules, input_ids, dataset)
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
                    if self.pipeline.use_transformer:
                        print("\n[⚠️] Uncertain confidence and disagreement detected. Using ensemble method for final decision.")
                        input_forward = sequence_ids if sequence_ids is not None else input_ids
                        final_probs, details = self.pipeline.ensemble.predict_ensemble(input_forward, X, y, method='dynamic', embedded=True)
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
                        input_forward = sequence_ids if sequence_ids is not None else input_ids
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
                    predicted_label, confidence = self.robust_prediction(self.pipeline, titles, label_map, show_proba=show_proba, top_k=top_k)
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
                    try:
                        confidence = float(final_probs[0][final_idx])   
                    except:
                        confidence = float(final_probs[0][len(reverse_map)-1]) if isinstance(final_probs[0], (float, int)) else 0.0             
                            
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
                                                  
            if isinstance(chosen_label, str) and chosen_label.startswith("unknown") or float(confidence) < self.pipeline.confidence_threshold:
                if chosen_label.startswith("unknown"):
                    chosen_label = 'Unknown'
                    confidence = 1.0 - confidence  # Invert confidence for unknown class

                print(f"\n[⚠️] Final prediction is {chosen_label} with uncertain confidence: {confidence:.1%}. Consider more consistent data for the model to learn from.")
            else:
                print(f"\n[🎯] Final chosen label for input: {chosen_label} || Confidence: {confidence:.1%}")  

        except Exception as e:
            print(f"[-] Error in advanced prediction method: {e}")
            traceback.print_exc()
            results, chosen_label, confidence = None, None, 0.0
        
        return results, chosen_label, confidence
        
        
    def display_hybrid_results(self, results, top_k=3, verbose=False):
        print("\n" + "="*80)
        print("[🎯] == PREDICTION RESULTS == ")
        print("="*80)
        
        correct = 0
        total_with_expected = 0
        
        for idx, result in enumerate(results):
            print(f"\n{idx+1}. 📌 '{result['title']}'")
            
            if result.get('expected'):
                total_with_expected += 1
                status = "[✅]" if result['predicted'] == result['expected'] else "❌"
                print(f"[=] Expected: {result['expected']} {status}")
                if result['predicted'] == result['expected']:
                    correct += 1
            
            # Agreement indicator
            agree_symbol = "✓" if result.get('models_agree', True) else "⚠️"
            print(f"   {agree_symbol} FINAL: {result['predicted']} ({result['confidence']:.1%})")
            
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