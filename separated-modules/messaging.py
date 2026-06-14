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



class Message:
    # Enhanced message with better tracking and retry logic for async processing
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
        # Age of message in seconds.
        return time.time() - self.created_at
    
    @property
    def is_expired(self) -> bool:
        # Check if message has expired.
        return self.age > self.timeout

class AsyncMessageQueue:
    # async message queue handler.
    
    def __init__(self, max_size=1000, dead_letter_queue_size=100):
        self.queue = asyncio.PriorityQueue(maxsize=max_size)
        self.pending: Dict[str, asyncio.Future] = {}
        self.results: Dict[str, Any] = {}
        self.handlers: Dict[str, Callable] = {}
        self.dead_letter_queue: deque = deque(maxlen=dead_letter_queue_size)
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None   
        self._counter_lock = asyncio.Lock() 
        self._start_lock = asyncio.Lock()            
        self._counter = 0
        self._stats = {
            'messages_processed': 0,
            'messages_failed': 0,
            'messages_retried': 0,
            'messages_expired': 0,
            'avg_latency': 0.0
        }
        
    def register_handler(self, message_type: str, handler: Callable):
        # Register a handler for specific message type
        self.handlers[message_type] = handler
      
        logger.info(f"[=] Registered handler for {message_type}")
    
    async def _ensure_started(self):
        """Start the worker if not already running"""
        if self._running:
            return
        
        async with self._start_lock:
            if self._running:
                return
            
            self._running = True
            self._worker_task = asyncio.create_task(self._worker())
            logger.info("[=] Async message queue worker started")
            
            # Give worker a moment to initialize
            await asyncio.sleep(0.1)
            
            if self._worker_task.done():
                exc = self._worker_task.exception()
                if exc:
                    logger.error(f"[=] Worker failed: {exc}")
                    raise exc
    

    async def publish(self, message: Message) -> Any:
        # Publish a message and wait for response.
        # Generate unique counter
        await self._ensure_started()

        async with self._counter_lock:
            self._counter += 1
            counter = self._counter
        
        if message.is_expired:
            logger.warning(f"[-] Message {message.id} already expired")
            raise TimeoutError(f"[-] Message {message.id} already expired")
            
        future = asyncio.Future()
        self.pending[message.id] = future
        logger.info(f"[=] Publishing message, ID: {message.id} of type {message.type} with priority {message.priority.name}")

        print('=== Queue Status ===')
        logger.info(f"[QueueStatus] Queue size before put: {self.queue.qsize()}")
        logger.info(f"[QueueStatus] Queue maxsize: {self.queue.maxsize}")
        logger.info(f"[QueueStatus] Queue full: {self.queue.full()}")
        logger.info(f"[QueueStatus] Worker running: {self._worker_task is not None and not self._worker_task.done()}")
                
        # Use priority queue (lower number = higher priority)
        await self.queue.put((message.priority.value, counter, message))
        logger.debug(f"[=] Message {message.id} enqueued, waiting for response...")
        
        try:
            logger.debug(f"[=] Awaiting response for message {message.id} with timeout {message.timeout}s")
            result = await asyncio.wait_for(future, timeout=message.timeout)
            logger.debug(f"[=] Received response for message {message.id}: {result}")
            return result
        except asyncio.TimeoutError:
            self.pending.pop(message.id, None)
            self._stats['messages_expired'] += 1
            logger.warning(f"[-] Message {message.id} timed out after {message.timeout}s")
            raise
        except Exception as e:
            self.pending.pop(message.id, None)
            logger.error(f"[-] Error occurred while processing message {message.id}: {e}")
            raise
    
    async def publish_async(self, message: Message, callback: Optional[Callable] = None):
        # Publish without waiting (fire and forget).
        message.callback = callback
        await self.queue.put((message.priority.value, message))
    
    async def _worker(self):
        # Background worker processing messages.
        while self._running:
            try:
                priority, counter, message = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                logger.info(f"[=] Worker picked up message {message.id} (counter: {counter}, priority: {priority})") 

                start_time = time.time()
                
                if message.is_expired:
                    self._stats['messages_expired'] += 1
                    logger.warning(f"[-] Dropping expired message {message.id}")
                    self._handle_orphaned_message(message)
                    continue
                
                if message.type in self.handlers:
                    try:
                        # Execute handler
                        if asyncio.iscoroutinefunction(self.handlers[message.type]):
                            result = await self.handlers[message.type](message)
                        else:
                            result = self.handlers[message.type](message)
                        
                        # Calculate latency
                        latency = time.time() - start_time
                        self._update_stats(latency, success=True)
                        
                        # Handle response
                        if message.id in self.pending:
                            self.pending[message.id].set_result(result)
                        elif message.callback:
                            message.callback(result)
                            
                    except Exception as e:
                        self._stats['messages_failed'] += 1
                        logger.error(f"[-] Handler failed for {message.type}: {e}\n{traceback.format_exc()}")
                        
                        if message.retry_count < message.max_retries:
                            message.retry_count += 1
                            self._stats['messages_retried'] += 1
                            logger.info(f"[-] Retrying message {message.id} (attempt {message.retry_count})")
                            await self.queue.put((message.priority.value, message))
                        else:
                            self._dead_letter_message(message, e)
                            if message.id in self.pending:
                                self.pending[message.id].set_exception(e)
                            elif message.callback:
                                message.callback(e)
                else:
                    logger.warning(f"[-] No handler for message type: {message.type}")
                    self._dead_letter_message(message, Exception(f"No handler for {message.type}"))
                    
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                logger.info("[=] Worker task cancelled")
                break
            except Exception as e:
                logger.error(f"[-] Worker error: {e}\n{traceback.format_exc()}")
                await asyncio.sleep(0.1)
    
    def _update_stats(self, latency: float, success: bool):
        # Update queue statistics
        self._stats['messages_processed'] += 1
        if not success:
            self._stats['messages_failed'] += 1
        
        # Exponential moving average of message latency.
        # alpha = 0.1 means the current measurement contributes 10 % to the running average,
        # providing a smoothed latency estimate that is robust to spikes without requiring
        # a fixed-size history window.
        alpha = 0.1  # Smoothing factor
        self._stats['avg_latency'] = alpha * latency + (1 - alpha) * self._stats['avg_latency']
    
    def _dead_letter_message(self, message: Message, error: Exception):
        # Send failed message to dead letter queue
        self.dead_letter_queue.append({
            'message': message,
            'error': str(error),
            'timestamp': datetime.now(),
            'retry_count': message.retry_count
        })
        logger.error(f"[=] Message {message.id} sent to DLQ after {message.retry_count} retries")
    
    def _handle_orphaned_message(self, message: Message):
        # Handle orphaned messages (no pending future, no callback).
        logger.warning(f"[=] Orphaned message {message.id} of type {message.type}")
        # Could store for manual inspection
        self.dead_letter_queue.append({
            'message': message,
            'error': 'Orphaned message - no handler or callback',
            'timestamp': datetime.now()
        })
    
    def get_stats(self) -> Dict:
        # Get queue statistics.
        return {
            **self._stats,
            'pending_count': len(self.pending),
            'queue_size': self.queue.qsize(),
            'dlq_size': len(self.dead_letter_queue),
            'is_running': self._running
        }
    

    
    async def start(self):
        # Start the async worker (async version).
        if self._running:
            logger.warning("[=] Queue already running")
            return
        
        self._running = True
        self._worker_task = asyncio.create_task(self._worker())
        logger.info("[=] Async message queue started")
          
    
    async def stop(self, timeout: float = 5.0):
        # Stop the message queue worker gracefully.
        logger.info("[=] Stopping message queue...")
        self._running = False
        if self._worker_task:
            try:
                await asyncio.wait_for(self._worker_task, timeout=timeout)
            except asyncio.TimeoutError:
                self._worker_task.cancel()
                logger.warning("[=] Worker task did not stop gracefully")
        logger.info("[=] Async message queue stopped")
    
    def get_dead_letter_queue(self) -> List[Dict]:
        # Get copy of dead letter queue for inspection.
        return list(self.dead_letter_queue)


class ThreadedMessageQueue:
    # Thread-based message queue for synchronous code
    
    def __init__(self, max_size=1000, worker_threads=4):
        self.queue = queue.Queue(maxsize=max_size)
        self.results = {}
        self.handlers = {}
        self._running = False
        self._workers = []
        self._worker_threads = worker_threads
        self._stats = {
            'messages_processed': 0,
            'messages_failed': 0,
            'active_workers': 0
        }
        self._lock = threading.Lock()
        
    def register_handler(self, message_type: str, handler: Callable):
        # Register a handler for specific message type
        self.handlers[message_type] = handler
        logger.info(f"[=] Registered handler for {message_type}")
    
    def publish(self, message: Message, timeout: float = 30.0) -> Any:
        # Publish a message and wait for response
        result_container = {'result': None, 'error': None, 'ready': False}
        
        def callback_wrapper(res):
            result_container['result'] = res
            result_container['ready'] = True
        
        message.callback = callback_wrapper
        self.queue.put(message)
        
        start_time = time.time()
        while not result_container['ready'] and (time.time() - start_time) < timeout:
            time.sleep(0.01)
        
        if not result_container['ready']:
            raise TimeoutError(f"Message {message.id} timed out")
        
        if result_container['error']:
            raise result_container['error']
        
        with self._lock:
            self._stats['messages_processed'] += 1
        
        return result_container['result']
    
    def publish_async(self, message: Message, callback: Optional[Callable] = None):
        # Publish without waiting
        message.callback = callback
        try:
            self.queue.put(message, block=False)
            return True
        except queue.Full:
            logger.error(f"[=] Queue full, cannot publish message {message.id}")
            return False
    
    def _worker(self, worker_id: int):
        # Worker thread processing messages.

        print(f'[=] Worker started: {worker_id}')
        while self._running:
            try:
                message = self.queue.get(timeout=1)
           
                if message.type in self.handlers:
                    try:
                        result = self.handlers[message.type](message)
                        if message.callback:
                            message.callback(result)
                    except Exception as e:
                        logger.error(f"[=] Worker {worker_id} handler failed: {e}")
                        if message.callback:
                            message.callback(e)
                else:
                    logger.warning(f"[=] No handler for message type: {message.type}")
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[=] Worker {worker_id} error: {e}")
    
    def start(self):
        # Start worker threads.
        if self._running:
            return
            
        self._running = True
        for i in range(self._worker_threads):
            thread = threading.Thread(target=self._worker, args=(i,), daemon=True)
            thread.start()
            self._workers.append(thread)
        
        with self._lock:
            self._stats['active_workers'] = len(self._workers)
        
        logger.info(f"[=] Threaded message queue started with {self._worker_threads} workers")
    
    async def stop(self, timeout: float = 5.0):
        # Stop worker threads gracefully.
        self._running = False
        # cancel and wait for all workers to exit cleanly
        for worker in self._workers:
            worker.cancel()

        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)

        self._workers.clear()

        logger.info("[=] Threaded message queue stopped")
    

    def get_stats(self) -> Dict:
        # Get queue statistics.
        with self._lock:
            return {
                **self._stats,
                'queue_size': self.queue.qsize(),
                'workers': len(self._workers),
                'is_running': self._running
            }

# Integrated inference module that allows multiple agents to connect and share their predictions, attention maps, and confidence scores for ensemble decision making.
# while also providing security features like authentication, rate limiting, and message validation.
