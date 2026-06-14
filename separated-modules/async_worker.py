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
        await self.result_queue.stop()

        # cancel and wait for all workers to exit cleanly
        for worker in self._workers:
            worker.cancel()

        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)

        self._workers.clear()

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

