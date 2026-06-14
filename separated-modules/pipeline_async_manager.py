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
        self.timeout = 120
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

    
    async def _predict_with_timeout(self, text: Any, timeout: float) -> Any:
        # Async prediction with timeout.
        try:
            return await asyncio.wait_for(
                self.pipeline.predict_async_await(text),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            raise FutureTimeoutError(f"[-] Prediction timed out after {timeout}s")
    
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


    def advanced_batch_prediction(self, test_titles, label_map, rules, X=None, y=None, api_key=None, client_ip=None):
        try:
            reverse_label_map = {v: k for k, v in label_map.items()}

            # Batch predictions
            texts = [text[0] for text in test_titles]
            expected_labels = [text[1] for text in test_titles]  # Your ground truth
            predicted_output = []

            text = {"test_titles": test_titles, "label_map": label_map, "rules": rules, "X":X, "y":y, "use_transformer": True}

            if not self.pipeline.intents:
                advanced_result = self.predict(
                text, 
                timeout=self.timeout,
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



