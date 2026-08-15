"""
automation.py
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
from . import pipeline

class CrossSessionAutomation:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def export_session(self, session_name=None):
        if session_name is None:
            session_name = f"session_{self.session_id}"
        
        session_data = {
            'session_id': self.session_id,
            'session_name': session_name,
            'timestamp': datetime.now().isoformat(),
            'memories': self.pipeline.memory.copy(),
        }
        
        filename = f"{session_name}.json"
        with open(filename, 'w') as f:
            json.dump(session_data, f, default=str)
        
        print(f"💾 Session exported to: {filename}")
        return filename
    
    def import_session(self, filename):
        with open(filename, 'r') as f:
            session_data = json.load(f)
        
        print(f"\n📥 Importing session: {session_data['session_name']}")
        print(f"   Created: {session_data['timestamp']}")
        print(f"   Memories: {len(session_data['memories'])}")
        
        # Merge memories
        for key, value in session_data['memories'].items():
            if key not in self.pipeline.memory:
                self.pipeline.memory[key] = value
        
        print(f"✅ Session imported! Total memories: {len(self.pipeline.memory)}")
    
    def sync_with_another_device(self, device_ip, port=5000):
        # Export current session
        temp_file = self.export_session(f"sync_{self.session_id}")
        
        try:
            with self.ssl_context.wrap_socket(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
                s.connect((device_ip, port))
                with open(temp_file, 'rb') as f:
                    s.sendall(f.read())
                print(f"📡 Synced to {device_ip} || {port}")
                print('🚀 Succesfully sync and export memory session to another device! ')                 
        except Exception as e:
            print(f"❌ Sync failed: {e}")
            pass
        

    
    def list_sessions(self, name):
        sessions = glob.glob(f"{name}*.json")
        
        print(f"\n📚 Available Sessions: {sessions}")
        if sessions:
            for session in sessions:
                with open(session, 'r') as f:
                    data = json.load(f)
                    print(f"   • {session}: {data['session_name']} ({len(data['memories'])} memories)")

        else:          
            print('[-] No available sessions! ')
        
        return sessions


# Explainability module that provides detailed explanations for predictions, allows learning from user feedback, and maintains a history of decisions for transparency and continuous improvement of the model.


class AutoBatcherAutomation:
    def __init__(self, pipeline, max_batch_size=32, max_wait_ms=50):
        self.pipeline       = pipeline
        self.max_batch_size = max_batch_size
        self.max_wait_ms    = max_wait_ms

        self.request_queue  = deque()
        self.processing     = False
        self.results        = {}
        self.result_events  = {}   # per-request Event 
        self.next_id        = 0

        self._state_lock = threading.Lock()   # guards processing flag + next_id

    def add_request(self, text, callback=None):
        with self._state_lock:
            request_id    = self.next_id
            self.next_id += 1

            event = threading.Event()
            self.result_events[request_id] = event

            self.request_queue.append({
                'id'       : request_id,
                'text'     : text,
                'callback' : callback,
                'timestamp': time.time()
            })

            # check-and-set happens atomically under the same lock
            should_start = not self.processing
            if should_start:
                self.processing = True

        if should_start:
            self._start_processing()

        result = self.get_result(request_id)
        self.cleanup_stale()

        return request_id

    def _start_processing(self):
        thread = threading.Thread(target=self._process_batches, daemon=True)
        thread.start()

    def _process_batches(self):
        try:
            while True:
                with self._state_lock:
                    if not self.request_queue:
                        break

                time.sleep(self.max_wait_ms / 1000)

                batch = []
                with self._state_lock:
                    while self.request_queue and len(batch) < self.max_batch_size:
                        batch.append(self.request_queue.popleft())

                if batch:
                    # never let one bad batch kill the worker permanently here.
                    try:
                        self._process_batch(batch)
                    except Exception as e:
                        logger.error(f'[!] Batch processing failed: {e}')
                        # deliver the failure to every waiter in this batch
                        # instead of leaving them hanging forever
                        for req in batch:
                            self._deliver_result(req, None, error=e)
        finally:
            # always reset processing, even if something above
            # raised unexpectedly
            with self._state_lock:
                self.processing = False

            # need catch the case where requests arrived after the
            # while-loop's last empty check but before processing=False landed
            with self._state_lock:
                still_pending = bool(self.request_queue)
                if still_pending and not self.processing:
                    self.processing = True
                    restart = True
                else:
                    restart = False
            if restart:
                self._start_processing()

    def _process_batch(self, batch):
        texts   = [req['text'] for req in batch]
        results = self.pipeline.prediction_batch(texts)

        for i, req in enumerate(batch):
            result = results[i] if i < len(results) else None
            self._deliver_result(req, result)

    def _deliver_result(self, req, result, error=None):
        """Single delivery path — callback or stored result, always signals."""
        if req['callback']:
            try:
                req['callback'](result if error is None else error)
            except Exception as cb_err:
                logger.error(f'[!] Callback failed for request {req["id"]}: {cb_err}')
        else:
            with self._state_lock:
                self.results[req['id']] = result if error is None else None

        # signal the waiting event 
        event = self.result_events.get(req['id'])
        if event:
            event.set()

    def get_result(self, request_id, timeout=5):
        event = self.result_events.get(request_id)
        if event is None:
            return None

        # blocks efficiently 
        signaled = event.wait(timeout=timeout)

        with self._state_lock:
            # always need clean up
            result = self.results.pop(request_id, None)
            self.result_events.pop(request_id, None)

        return result if signaled else None

    def cleanup_stale(self, max_age_seconds=300):
        """
        periodic sweep for requests that were never collected
        via get_result (e.g. caller crashed or forgot to call it).
        Call this periodically from a health check loop.
        """
        with self._state_lock:
            now = time.time()
            stale_ids = [
                rid for rid, event in self.result_events.items()
                if event.is_set()   # already delivered but never collected
            ]
            for rid in stale_ids:
                self.results.pop(rid, None)
                self.result_events.pop(rid, None)
            if stale_ids:
                logger.info(f'[=] Cleaned up {len(stale_ids)} stale results')


# The pipeline.IntegratedPipeline class serves as the central component that integrates all the different modules and functionalities of the system. 
# It manages the overall workflow, including data processing, model training, prediction, memory management, and interactions with other agents.


