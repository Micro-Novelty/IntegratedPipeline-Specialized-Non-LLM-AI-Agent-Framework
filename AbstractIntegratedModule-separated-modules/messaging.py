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
# messaging.py
# Async and threaded message queues, request/result tracking.
# Depends on: primitives (MessagePriority, RequestStatus, AsyncRequest)
# ---------------------------------------------------------------------------
from .primitives import MessagePriority, RequestStatus, AsyncRequest

# Message dataclass used by AsyncMessageQueue / ThreadedMessageQueue
@dataclass
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
        
        # Update moving average
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
    
    def stop(self, timeout: float = 5.0):
        # Stop worker threads gracefully.
        self._running = False
        for thread in self._workers:
            thread.join(timeout=timeout)
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


class AsyncResultQueue:
    """
    Complete result queue with integrated processor.
    Handles callbacks, webhooks, WebSocket, storage, and streaming.
    """
    
    def __init__(self, max_size: int = 1000, cleanup_interval: int = 60):
        self._requests: Dict[str, AsyncRequest] = {}
        self._pending_queue: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        self._completion_queue: asyncio.Queue = asyncio.Queue()
        self._result_futures: Dict[str, asyncio.Future] = {}
        self._lock = asyncio.Lock()
        self._cleanup_interval = cleanup_interval
        self._running = False
        
        # Result processor components
        self._cleanup_task: Optional[asyncio.Task] = None
        self._processor_task: Optional[asyncio.Task] = None
        
        # Optional features
        self._webhook_url: Optional[str] = None
        self._websocket_clients: List = []  # Store WebSocket connections
        self._storage_enabled: bool = False
        self._storage_path: Optional[str] = None
        self._streaming_queue: Optional[asyncio.Queue] = None
        
        # Metrics
        self._metrics = {
            'total_completed': 0,
            'total_failed': 0,
            'total_callbacks': 0,
            'total_webhooks': 0,
            'avg_processing_time': 0.0
        }
    
    # ============ INITIALIZATION ============
    
    async def start(self, 
                   webhook_url: str = None,
                   storage_path: str = None,
                   enable_streaming: bool = False):
        """
        Start the result queue with optional features.
        
        Args:
            webhook_url: Send results to this URL when complete
            storage_path: Save results to disk
            enable_streaming: Enable result streaming queue
        """
        self._running = True
        self._webhook_url = webhook_url
        self._storage_enabled = bool(storage_path)
        self._storage_path = storage_path
        
        if enable_streaming:
            self._streaming_queue = asyncio.Queue()
        
        # Start background tasks
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._processor_task = asyncio.create_task(self._result_processor())
        
         
        logger.info("✅ AsyncResultQueue started")
        


    async def stop(self):
        """Stop the result queue"""
        self._running = False
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
        if self._processor_task:
            self._processor_task.cancel()
        
        await asyncio.gather(
            self._cleanup_task, 
            self._processor_task, 
            return_exceptions=True
        )
        
        logger.info("✅ AsyncResultQueue stopped")
    
    # ============ REQUEST SUBMISSION ============
    
    async def submit(self, 
                    texts, 
                    api_key: str = None, 
                    client_ip: str = None,
                    callback: Callable = None,
                    webhook_url: str = None) -> str:
        """
        Submit a prediction request.
        
        Args:
            texts: List of input texts to predict
            api_key: API key for authentication
            client_ip: Client IP address
            callback: Optional callback function
            webhook_url: Optional webhook for this specific request
            
        Returns:
            request_id for tracking
        """
        request_id = str(uuid.uuid4())
        
        async with self._lock:
            request = AsyncRequest(
                request_id=request_id,
                texts=texts,
                api_key=api_key,
                client_ip=client_ip,
                callback=callback,
                webhook_url=webhook_url
            )
            self._requests[request_id] = request
            
            # Create future for awaiting result
            future = asyncio.Future()
            self._result_futures[request_id] = future
            
            # Add to processing queue
            await self._pending_queue.put(request)
            
        logger.debug(f"[=] Submitted request {request_id}: {texts}")
        return request_id
    
    async def wait_for_result(self, request_id: str, timeout: int = 30) -> Dict:
        """
        Wait for a specific request to complete.
        
        Args:
            request_id: ID from submit()
            timeout: Maximum wait time in seconds
            
        Returns:
            Prediction result dictionary
        """
        async with self._lock:
            future = self._result_futures.get(request_id)
            if not future:
                request = self._requests.get(request_id)
                if request and request.status == RequestStatus.COMPLETED:
                    return request.result
                raise ValueError(f"[=] Unknown request_id: {request_id}")
        
        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            await self._mark_failed(request_id, "Request timeout")
            raise
        finally:
            async with self._lock:
                self._result_futures.pop(request_id, None)
    
    async def complete(self, request_id: str, result: Dict):
        """
        Mark a request as completed with result.
        Called by worker when prediction is done.
        """
        async with self._lock:
            request = self._requests.get(request_id)
            if not request:
                logger.warning(f"[=] Request {request_id} not found")
                return
            
            request.status = RequestStatus.COMPLETED
            request.result = result
            request.completed_at = time.time()
            
            # Calculate processing time for metrics
            processing_time = request.completed_at - request.created_at
            alpha = 0.1
            self._metrics['avg_processing_time'] = (
                alpha * processing_time + 
                (1 - alpha) * self._metrics['avg_processing_time']
            )
            
            # Add to completion queue for processor
            await self._completion_queue.put(request)
            
            logger.debug(f"[=] Request {request_id} completed in {processing_time:.2f}s")
    
    async def _mark_failed(self, request_id: str, error: str):
        """Mark a request as failed"""
        async with self._lock:
            request = self._requests.get(request_id)
            if request:
                request.status = RequestStatus.FAILED
                request.error = error
                request.completed_at = time.time()
                await self._completion_queue.put(request)
            
            future = self._result_futures.get(request_id)
            if future and not future.done():
                future.set_exception(Exception(error))
    
    async def get_pending(self) -> Optional[AsyncRequest]:
        """Get next pending request (blocks)"""
        try:
            return await self._pending_queue.get()
        except asyncio.CancelledError:
            return None
    
    # ============ THE MAIN RESULT PROCESSOR ============
    
    async def _result_processor(self):
        """
        Process results as they complete.
        Handles: Callbacks, Webhooks, WebSocket, Storage, Streaming
        """
        logger.info("[=] Result processor started")
        
        while self._running:
            try:
                # Wait for a completed request
                completed_request = await asyncio.wait_for(
                    self._completion_queue.get(), 
                    timeout=1.0
                )
                
                if not completed_request:
                    continue
                
                request_id = completed_request.request_id
                result = completed_request.result
                error = completed_request.error
                is_success = error is None
                
                # Update metrics
                if is_success:
                    self._metrics['total_completed'] += 1
                else:
                    self._metrics['total_failed'] += 1
                
                # ============ 1. EXECUTE CALLBACK ============
                if completed_request.callback:
                    try:
                        self._metrics['total_callbacks'] += 1
                        
                        # Support both sync and async callbacks
                        if asyncio.iscoroutinefunction(completed_request.callback):
                            # Async callback
                            await completed_request.callback(request_id, result, error)
                        else:
                            # Sync callback - run in thread pool
                            await asyncio.to_thread(
                                completed_request.callback,
                                request_id, result, error
                            )
                        logger.debug(f"[=] Callback executed for {request_id}")
                        
                    except Exception as e:
                        logger.error(f"[=] Callback failed for {request_id}: {e}")
                
                # ============ 2. RESOLVE WAITING FUTURE ============
                future = self._result_futures.get(request_id)
                if future and not future.done():
                    if error:
                        future.set_exception(Exception(error))
                    else:
                        future.set_result(result)
                    self._result_futures.pop(request_id, None)
                
                # ============ 3. WEBHOOK NOTIFICATION ============
                webhook_url = completed_request.webhook_url or self._webhook_url
                if webhook_url and is_success:
                    await self._send_webhook(webhook_url, {
                        'request_id': request_id,
                        'status': 'success',
                        'result': result,
                        'timestamp': completed_request.completed_at
                    })
                    self._metrics['total_webhooks'] += 1
                
                # ============ 4. WEBSOCKET PUSH ============
                if self._websocket_clients:
                    await self._broadcast_websocket({
                        'type': 'prediction_complete',
                        'request_id': request_id,
                        'result': result,
                        'error': error,
                        'processing_time': completed_request.completed_at - completed_request.created_at
                    })
                
                # ============ 5. PERSISTENT STORAGE ============
                if self._storage_enabled and is_success:
                    await self._store_result(request_id, result, completed_request)
                
                # ============ 6. STREAMING TO RESPONSE QUEUE ============
                if self._streaming_queue:
                    await self._streaming_queue.put({
                        'request_id': request_id,
                        'result': result,
                        'error': error,
                        'completed_at': completed_request.completed_at
                    })
                
                # ============ 7. LOG COMPLETION ============
                logger.info(
                    f"[=] Request {request_id} processed - "
                    f"[=] Success: {is_success}, "
                    f"[=] Time: {completed_request.completed_at - completed_request.created_at:.2f}s"
                )
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[=] Result processor error: {e}")
                await asyncio.sleep(0.1)
        
        logger.info("[=] Result processor stopped")
    
    # ============ WEBHOOK HANDLER ============
    
    async def _send_webhook(self, url: str, data: dict):
        """Send result to webhook URL"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, 
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        logger.debug(f"[=] Webhook sent to {url}")
                    else:
                        logger.warning(f"[=] Webhook failed: {response.status}")
        except Exception as e:
            logger.error(f"[=] Webhook error: {e}")
    
    # ============ WEBSOCKET HANDLER ============
    
    async def register_websocket(self, websocket):
        """Register a WebSocket client for real-time updates"""
        self._websocket_clients.append(websocket)
        
        # Remove when closed
        try:
            await websocket.wait_closed()
        finally:
            self._websocket_clients.remove(websocket)
    
    async def _broadcast_websocket(self, message: dict):
        """Broadcast message to all connected WebSocket clients"""
        disconnected = []
        
        for client in self._websocket_clients:
            try:
                await client.send_json(message)
            except:
                disconnected.append(client)
        
        # Clean up disconnected clients
        for client in disconnected:
            self._websocket_clients.remove(client)
    
    # ============ PERSISTENT STORAGE ============
    
    async def _store_result(self, request_id: str, result: dict, request: AsyncRequest):
        # Store result to disk
        if not self._storage_path:
            return
        
        try:
            import aiofiles
            import os
            
            os.makedirs(self._storage_path, exist_ok=True)
            
            filepath = os.path.join(self._storage_path, f"{request_id}.json")
            
            data = {
                'request_id': request_id,
                'text': request.text,
                'result': result,
                'created_at': request.created_at,
                'completed_at': request.completed_at,
                'processing_time': request.completed_at - request.created_at
            }
            
            async with aiofiles.open(filepath, 'w') as f:
                await f.write(json.dumps(data, indent=2))
                
            logger.debug(f"[=] Result stored: {filepath}")
            
        except Exception as e:
            logger.error(f"[=] Storage failed: {e}")
    
    # ============ RESULT STREAMING ============
    
    async def get_result_stream(self) -> asyncio.Queue:
        """
        Get a queue that receives results as they complete.
        Useful for streaming responses to clients.
        """
        if not self._streaming_queue:
            self._streaming_queue = asyncio.Queue()
        return self._streaming_queue
    
    # ============ CLEANUP ============
    
    async def _cleanup_loop(self):
        # Periodically clean up expired and old requests
        while self._running:
            await asyncio.sleep(self._cleanup_interval)
            
            async with self._lock:
                now = time.time()
                
                # Find expired pending requests
                expired = [
                    req_id for req_id, req in self._requests.items()
                    if req.is_expired and req.status in (RequestStatus.PENDING, RequestStatus.PROCESSING)
                ]
                
                for req_id in expired:
                    request = self._requests[req_id]
                    request.status = RequestStatus.TIMEOUT
                    request.completed_at = now
                    request.error = "Request expired"
                    await self._completion_queue.put(request)
                    
                    # Clean up future
                    future = self._result_futures.pop(req_id, None)
                    if future and not future.done():
                        future.set_exception(TimeoutError(f"Request {req_id} expired"))
                
                # Remove completed/failed/timeout requests older than 1 hour
                old_cutoff = now - 3600
                to_remove = [
                    req_id for req_id, req in self._requests.items()
                    if req.completed_at and req.completed_at < old_cutoff
                ]
                for req_id in to_remove:
                    self._requests.pop(req_id, None)
                
                if expired:
                    logger.info(f"[=] Cleaned up {len(expired)} expired requests")
    
    async def _cleanup_request(self, request_id: str):
        # Clean up a single completed request after processing
        # Schedule for removal after delay
        async def _delayed_remove():
            await asyncio.sleep(3600)  # Keep for 1 hour
            async with self._lock:
                self._requests.pop(request_id, None)
        
        asyncio.create_task(_delayed_remove())
    
    # ============ UTILITY METHODS ============
    
    def _update_metrics(self, request: AsyncRequest):
        # Update metrics after processing
        # Metrics already updated in complete() and _result_processor
        pass
    
    def get_status(self, request_id: str) -> Optional[RequestStatus]:
        # Get status of a request
        request = self._requests.get(request_id)
        return request.status if request else None
    
    def get_result(self, request_id: str) -> Optional[Dict]:
        # Get result if completed
        request = self._requests.get(request_id)
        if request and request.status == RequestStatus.COMPLETED:
            return request.result
        return None
    
    def get_metrics(self) -> Dict:
        # Get queue metrics
        return {
            **self._metrics,
            'pending_count': self._pending_queue.qsize(),
            'completion_count': self._completion_queue.qsize(),
            'total_requests': len(self._requests),
            'pending': sum(1 for r in self._requests.values() if r.status == RequestStatus.PENDING),
            'processing': sum(1 for r in self._requests.values() if r.status == RequestStatus.PROCESSING),
            'completed': sum(1 for r in self._requests.values() if r.status == RequestStatus.COMPLETED),
            'failed': sum(1 for r in self._requests.values() if r.status == RequestStatus.FAILED),
            'timeout': sum(1 for r in self._requests.values() if r.status == RequestStatus.TIMEOUT)
        }


class WorkerPool:
    """
    Worker pool that processes requests from the result queue.
    """
    
    def __init__(self, result_queue: AsyncResultQueue, num_workers: int = 4):
        self.result_queue = result_queue
        self.num_workers = num_workers
        self._workers: List[asyncio.Task] = []
        self._running = False
    
    async def start(self, predict_func):
        # Start worker pool with prediction function
        self._running = True
        self._workers = [
            asyncio.create_task(self._worker(predict_func))
            for _ in range(self.num_workers)
        ]

        await self.result_queue.start()
        
    async def stop(self):
        # Stop all workers
        self._running = False
        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
    
    async def _worker(self, predict_func):
        # Worker that processes requests
        while self._running:
            try:
                # Get next pending request
                request = await self.result_queue.get_pending()
                if not request:
                    continue
                
                # Mark as processing
                async with self.result_queue._lock:
                    request.status = RequestStatus.PROCESSING
                
                try:
                    # Execute prediction (run sync function in thread pool)
                    result = await asyncio.to_thread(
                        predict_func,
                        texts=request.texts,
                        api_key=request.api_key,
                        client_ip=request.client_ip
                    )
                    
                    # Mark as completed
                    await self.result_queue.complete(request.request_id, result)
                    
                except Exception as e:
                    # Mark as failed
                    await self.result_queue._mark_failed(request.request_id, str(e))
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[=] Worker error: {e}")
                await asyncio.sleep(0.1)