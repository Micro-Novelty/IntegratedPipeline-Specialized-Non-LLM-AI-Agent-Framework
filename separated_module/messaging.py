"""
messaging.py
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


