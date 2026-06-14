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



class IntegratedPipeline:
    '''
    Top-level pipeline that wires together all subsystems.

    Component map
    -------------
    self.mlp            : MLP — primary TF-IDF-based classifier.
    self.focused_mlp    : MLP — secondary "focused" MLP for fine-tuning.
    self.model2         : Transformer — token-sequence-based classifier.
    self.model3         : MLP — trained on hybrid (TF-IDF + Transformer) features.
    self.tfidf          : TfidfVectorizer(max_features=70) — shared feature extractor.
    self.storage        : ModelStorage — SQLite persistence.
    self.distribution   : AgentDistributedInference — P2P networking.
    self.batcher        : AutoBatcherAutomation — dynamic batching.
    self.query_node     : QueryNode — peer node registry.

    Singleton behaviour
    -------------------
    The class checks for self._singleton_initialized to prevent double-init
    when the same instance is passed across modules.  The singleton is NOT
    enforced at the metaclass level, so multiple independent instances can
    exist (each with a different memory_name).

    Training flow
    -------------
    1. train(X, y_raw)          — fits TF-IDF, encodes labels, trains MLP.
    2. transformer_utilities()  — (called internally) checks training necessity,
                                  trains Transformer, builds hybrid features,
                                  trains model3 on the concatenated feature space.

    Prediction flow
    ---------------
    predict_single(text)        — returns a dict with prediction, confidence,
                                  probabilities, and optional attention weights.
    prediction_batch(texts)     — calls predict_single in a list comprehension.
    _calibrate_probs()          — post-hoc confidence calibration using
                                  attention quality, anisotropy, and peer data.

    Memory / persistence
    --------------------
    On construction the pipeline checks whether a saved model exists for
    memory_name.  If found it loads weights and validates them via
    is_memory_corrupted().  Predictions are written back through
    modular_prediction_saving() / modular_probability_saving().

    Parameters
    ----------
    memory_name       : Logical name for DB scoping and model identity.
    use_async         : If True, sets up an AsyncMessageQueue for non-blocking
                        distributed operations.
    agent_port        : TCP port for AgentDistributedInference (None = disabled).
    ssl_cert_file     : TLS certificate for encrypted agent comms.
    ssl_key_file      : TLS private key for encrypted agent comms.
    secret_key        : HMAC key for message signing.
    shared_auth_token : Shared secret for peer agent authentication.
    predict_manager   : Optional PipelinePredictionManager reference (set after
                        construction to avoid circular initialisation).
    cache = to store lstm memory or bins.
    '''
    def __init__(self, memory_name, use_async, agent_port=None, ssl_cert_file=None, ssl_key_file=None, secret_key=None, shared_auth_token=None, predict_manager=None):
        super().__init__()

        if hasattr(self, '_singleton_initialized'):
            print(f"[===] IntegratedPipeline already initialized, reusing...")
            return
        
        self._singleton_initialized = True
        
        # Stored initialization params for debugging later
        self._init_params = {
            'memory_name': memory_name,
            'port': agent_port,
            'secret_key': secret_key,
            'ssl_cert_file': ssl_cert_file,
            'ssl_key_file': ssl_key_file,
            'shared_auth_token': shared_auth_token
        }  

        self.ssl_cert_file = ssl_cert_file
        self.ssl_key_file = ssl_key_file
        self.secret_key = secret_key
        self._use_async = use_async
        self.port = agent_port
        self.shared_auth_token = shared_auth_token
        self.manager = None

        self.standard_scaler = StandardScaler()
        self.tfidf = TfidfVectorizer(max_features=70)
        self.storage = ModelStorage(self, memory_name, db_path='activity_log.db')
        self.distribution = AgentDistributedInference(self, self.storage, memory_name, port=self.port, use_async=use_async, secret_key=self.secret_key, ssl_cert_file=ssl_cert_file, ssl_key_file=ssl_key_file, shared_auth_token=self.shared_auth_token, predict_manager=self.manager)        
        self.ensemble = WeightedEnsemblePredictor(self, self.distribution, memory_name)        
        self.session_automation = CrossSessionAutomation(self)
        self.batcher = AutoBatcherAutomation(self)
        self.query_node = QueryNode(self, memory_name, self.storage)
        
        self._agent_mode = 'single'
        self._agent_port = int(agent_port) if int(agent_port) is not None else 5000
        self._use_async = use_async

        print(f'[= PORT =] IntegratedPipeline initialized on port {self._agent_port}')
         
        # Queue for managing async operations
        self._async_tasks = set()
        self._loop = None  

        self.mlp = MLP()
        self.focused_mlp = MLP()

        self.X = None
        self.vocab_size = None
        self.model2 = None
        self.model3 = None
        self.texts = None
        self.intents = None
        self.role_bot = None
        self.batch_timer = None
        self.reverse_map = None
        self.rules = None
        self.titles = None
        self.labels = None

        self.use_transformer = False
        self.agreement = False
        self.external_peer_enabled = False
        self.autonomous = False 
        self.show_explainability_details = True   
        self.froze_learning = False  

        self.temperature = 1.0
        self.memory_name = memory_name

        self.pending_batch = []
        self.temporary_id = []

        self.input_size = 1
        self.hidden = 32
        self.output_size = 1
        self.dropout_rate = 0.1

        self.final_conf_score = 0.0
        self.timeout = 120
        self.confidence_threshold = 0.45  
        self.peer_assistance_threshold = 0.0              
        self.agent_id = random.randint(0, 10000)

        self.vocab = {}
        self.cache = {}

        # LSTM __init__ setup for lstm architectures
        self.network_model = LSTMNetwork(self, input_size=self.input_size, hidden_size=self.hidden, output_size=self.output_size)
        self.scrapper_model = LSTMEngine(self, self.network_model, dropout=self.dropout_rate, n_samples=50)        
        self.lstm_engine = None
        self.lstm_n_samples = 0
     
        if not self.storage.memory_exists(memory_name, type='Pipeline'):
            self.memory = {}
        else:
            print(f'|| Found Matched Memory: {memory_name}!')
            self.memory = self.storage.memory_retrieval(memory_name, type_func='Pipeline', verbose=True)

            is_corrupted, reason, suggested_fix = self.is_memory_corrupted(self.memory)
            if is_corrupted:
                print(f'[⚠️] MEMORY CORRUPTION DETECTED!')
                print(f'    Reason: {reason}')
                print(f'    Suggestion: {suggested_fix}')
                
                # Auto-fix based on severity
                if '133' in reason:
                    print(f'[!] Hybrid feature detected - clearing all memory entries')
                    self.memory = {}
                    # Also clear from storage
                    self.storage.fix_corrupted_memory(memory_name)
                elif 'deserialization' in reason.lower():
                    print(f'[!] Deserialization error - resetting memory')
                    self.memory = {}
                else:
                    print(f'[!] Keeping memory but will validate on access')
            else:
                print(f'[✅] Memory validation passed: {reason}')              

        if use_async:
            self._setup_async_queue()
            
        self.distribution.remote_agents['local'] = {
            'sock': None,  # No socket needed for local
            'host': 'localhost',
            'port': self.port,
            'trust': 1.0,
            'last_seen': datetime.now(),
            'failures': 0
        }

    def is_memory_corrupted(self, memory, num_classes: int = None) -> tuple:
        """
        Robust memory corruption detection.
        
        Returns:
            (is_corrupted: bool, reason: str, suggested_fix: str)
        """
        if num_classes is None:
            num_classes = self._get_num_classes()
        
        # Case 1: Memory is None
        if memory is None:
            return (True, "Memory is None", "Initialize new memory dict")
        
        # Case 2: Memory is numpy array
        if isinstance(memory, np.ndarray):
            # Valid probability array
            if memory.ndim == 1 and memory.shape[0] == num_classes:
                return (False, "Valid probability array", None)
            
            # Valid probability matrix (batch)
            if memory.ndim == 2 and memory.shape[1] == num_classes:
                return (False, "Valid probability matrix", None)
            
            # Hybrid features - NOT corrupted, just wrong type
            if memory.shape[0] != self._get_num_classes():
                return (True, f"[=] Hybrid feature array (shape {memory.shape}) stored as memory", 
                    "Clear and retrain, or convert to probability array")
            
            # Other shapes
            return (True, f"[=] Unexpected array shape: {memory.shape}", 
                "Clear memory and retrain model")
        
        # Case 3: Memory is list
        if isinstance(memory, list):
            if not memory:
                return (True, "[=] Empty list", "Initialize new memory dict")
            
            # List of probabilities
            if len(memory) == num_classes:
                # Check if all elements are numbers
                if all(isinstance(x, (int, float)) for x in memory[:min(5, len(memory))]):
                    return (False, "[=] Valid probability list", None)
            
            # List of hybrid features 
            if len(memory) != self._get_num_classes():
                return (True, f"[=] Hybrid feature list with length: {len(memory)} stored as memory", 
                    "Clear memory and retrain")
            
            # List of tuples (likely valid memory entries)
            if all(isinstance(item, (tuple, list)) and len(item) >= 2 for item in memory[:min(5, len(memory))]):
                return (False, "[=] Valid memory entries list", None)
            
            # Check for numpy arrays in list
            if any(isinstance(item, np.ndarray) for item in memory):
                arrays = [item.shape for item in memory if isinstance(item, np.ndarray)]
                if any(shape[0] == num_classes for shape in arrays):
                    return (False, "[=] Valid with numpy arrays", None)
            
            return (True, f"[=] Suspicious list length: {len(memory)}", 
                "Inspect memory contents")
        
        # Case 4: Memory is dict (expected format)
        if isinstance(memory, dict):
            # Empty dict is fine (no memory yet)
            if not memory:
                return (False, "[=] Empty dict (no memory)", None)
            
            # Check for valid keys
            valid_keys = {'TW', 'MW', 'TP', 'MP', 'TA', 'local', '_cached_probs', '_data'}
            
            for key, value in memory.items():
                # Skip valid keys
                if key in valid_keys or key.startswith(('TW', 'MW', 'TP', 'MP', 'TA')):
                    continue
                
                # Suspicious key
                if isinstance(key, (int, float)):
                    return (True, f"[=] Dict has numeric key: {key}", 
                        "Likely deserialization error, clear memory")
                
                if len(str(key)) > 100:
                    return (True, f"[=] Dict has very long key: {len(str(key))} chars", 
                        "Possible corruption, clear memory")
            
            # Check values in dict
            for key, value in memory.items():
                # Check for corrupted array values
                if isinstance(value, np.ndarray):
                    if value.shape != self._get_num_classes:
                        return (True, f"[=] Array with shape {value.shape} in dict key '{key}'", 
                            "Hybrid feature stored incorrectly, clear entry")
                    
                    if value.shape == (0,):
                        return (True, f"[=] Empty array in dict key '{key}'", 
                            "Corrupted array, clear entry")
                
                # Check for corrupted list values
                if isinstance(value, list) and len(value) == 133:
                    return (True, f"[=] List with length 133 in dict key '{key}'", 
                        "Hybrid feature stored incorrectly, clear entry")
            
            return (False, "Valid dict structure", None)
        
        # Case 5: Memory is other type
        return (True, f"Unexpected memory type: {type(memory)}", 
            "Clear and reinitialize memory")


    def initialize_fitting(self, text):
        self.tfidf.fit_transform(text).toarray()
        vocab_size = len(self.tfidf.get_feature_names_out())
        self.vocab_size = vocab_size
        

    def initialize_model_encoding(self, X, y_raw):
        vocab_size = self.vocab_size

        num_classes = len(np.unique(y_raw))
        y_onehot = np.zeros((len(y_raw), num_classes))
        y_onehot[np.arange(len(y_raw)), y_raw] = 1

        automatic_change = self.automatic_parameterization(vocab_size, num_classes)
        self.embedding_dim = automatic_change
        layer1, layer2 = self.automatic_dense_layer(X, vocab_size, num_classes)

        model = self.mlp  

        model.add(layer1)
        model.add(layer2)

        return y_onehot



    def initialize_model_(self, X, input_dim, num_classes):
        automatic_change = self.automatic_parameterization(input_dim, num_classes)

        automatic_change = self.automatic_parameterization(input_dim, num_classes)

        layer1= Dense(X, input_dim, automatic_change, activation="relu")
        layer2 = Dense(X, automatic_change, num_classes, activation='relu')

        abundant_layer = automatic_change * 10
        first_feed_layer = Dense(X, input_dim, abundant_layer, activation="relu")
        sec_feed_layer = Dense(X, abundant_layer, num_classes, activation="relu")

        self.model3 = MLP() 

        self.model3.add(layer1)
        self.model3.add(layer2)

        self.focused_mlp.feed_add(first_feed_layer)   
        self.focused_mlp.feed_add(sec_feed_layer)                

        
    def automatic_parameterization(self, input_size, num_classes):
        parameters = input_size * num_classes / 2
        parameters = int(parameters)
        return parameters


    def automatic_dense_layer(self, X, input_dim, num_classes):
        vocab_size = self.vocab_size 

        automatic_change = self.automatic_parameterization(input_dim, num_classes)

        layer1= Dense(X, input_dim, automatic_change, activation="relu")
        layer2 = Dense(X, automatic_change, num_classes, activation="relu")

        return layer1, layer2


    def text_encoder(self, texts):
        vocab = self.vocab
        idx = 0
        for item in texts:
            texts = item[0] if isinstance(item, tuple) else item
            for word in texts.split():
                if word not in vocab:
                    vocab[word] = idx
                    idx += 1


    def encode(self, sentence, vocab, max_len=6):        
        tokens = sentence.split()
        ids = [vocab.get(w, 0) for w in tokens]
        while len(ids) < max_len:
            ids.append(0)
        
        return ids[:max_len]



    def input_encoding(self, datasets):
        texts = [d[0] for d in datasets]
        intents = [d[1] for d in datasets]
        intent_to_id = {intent:i for i, intent in enumerate(sorted(set(intents)))}
        num_classes = len(intent_to_id)
        labels = [intent_to_id[i] for i in intents]

        reverse_map = {}
        for i in range(len(intents)):
            reverse_map[i] = intents[i]

        self.texts = texts
        self.intents = intents
        self.reverse_map = reverse_map

        self.model2 = Transformer(
            vocab_size=len(self.vocab),
            d_model=32,
            n_heads=4,
            num_classes=num_classes
        )   

        y_true = np.zeros((len(labels), num_classes))
        for i,l in enumerate(labels):
            y_true[i,l] = 1
            
        input_ids_list = []

        input_ids_list = []
        for text in texts:
            input_ids_list.append(np.array(self.encode(text, self.vocab)))  

        return input_ids_list, y_true


    def cosine_robust_similarity(self, a, b):
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b[0])
    
        try:
            dot_product = np.dot(a, b)
        except:
            try: #detect inhomogenous shape
                dot_product = np.dot(a.flatten(), b[:a.flatten().shape[0]])
            except:
                try:
                    dot_product = np.dot(a[:b.shape[0]], b.flatten()[:a.shape[0]])
                except:
                    print('[-] No similarity due to inhomogenous shapes and failed attempts to find subsets, returning low similarity score.')
                    return 0.1
    
        dot_product = np.dot(a, b)
        cosine = dot_product / (norm_a * norm_b)
    
        return np.clip(cosine, -1.0, 1.0)

    # ===== async setup ======
    def _setup_async_queue(self):
        # Setup async queue handlers
        if not self.distribution or not self.distribution.use_async:
            return
            
        # Register custom handlers
        self.distribution.message_queue.register_handler(
            'custom_prediction', 
            self._handle_custom_prediction_async
        )
        self.distribution.message_queue.register_handler(
            'model_update',
            self._handle_model_update_async
        )
        
        # Start queue processor if not already running
        self.distribution.start_queue_processor()
        
        # Start health checker for async mode
        if self.distribution.use_async:
            self.distribution._start_health_checker()
            
        print("✅ Async message queue initialized")
    
    async def _handle_custom_prediction_async(self, message):
        # Handle custom prediction requests asynchronously.
        try:
            text = message.payload.get('text', '')
            result = self.predict_single(text)
            return {
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'success': True
            }
        except Exception as e:
            return {
                'prediction': None,
                'confidence': 0.0,
                'success': False,
                'error': str(e)
            }
    
    async def _handle_model_update_async(self, message):
        # Handle model update requests asynchronously.
        try:
            weights = message.payload.get('weights')
            if weights:
                self.update_weights(weights)
                return {'success': True, 'message': 'Model updated'}
            return {'success': False, 'message': 'No weights provided'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    # ============ Async Prediction Methods ============
    
    def predict_async(self, text: Any, callback=None, timeout=30):
        # Async prediction with callback support.
        if not self.distribution or not self.distribution.use_async:
            # Fallback to sync prediction
            result = self.predict_single(text)
            if callback:
                callback(result)
            return result
        
        return self.distribution.request_prediction_async(
            agent_id='local',
            text=text,
            callback=callback
        )
    
    async def predict_async_await(self, text, timeout=30):
        # Async prediction with await support.
        if not self.distribution or not self.distribution.use_async:
            # Fallback to sync prediction (run in thread)
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, self.predict_single, text
            )   
            
        try:
            result = await asyncio.wait_for(
                self.distribution.request_prediction_async('local', text, timeout=timeout),
                timeout=timeout + 5
            )
            return result
        except asyncio.TimeoutError:
            raise TimeoutError(f"[-] Prediction timed out after {timeout + 5}s")
        except Exception as e:
            logger.error(f"[-] Async prediction failed: {e}")
            traceback.print_exc()
            raise

    
    async def predict_batch_async(self, texts: List[str], callback=None):
        # Batch async predictions.
        if not self.distribution or not self.distribution.use_async:
            # Fallback to sync batch prediction
            results = self.prediction_batch(texts)
            if callback:
                callback(results)
            return results

        return await self.distribution.request_batch_prediction_async(
            agent_id='local',
            texts=texts,
            callback=callback
        )


    # ============ Distributed Agent Methods ============
    
    def connect_peers(self, peer_addresses: List[tuple]):
        # Connect to multiple peer agents.
        if not self.distribution:
            self.init_distributed()
        
        results = []
        for host, port in peer_addresses:
            try:
                sock = self.distribution.connect_to_agent(host, port)
                results.append({'host': host, 'port': port, 'success': sock is not None})
            except Exception as e:
                results.append({'host': host, 'port': port, 'success': False, 'error': str(e)})
        
        return results
    
    async def broadcast_to_peers(self, message_type: str, payload: dict):
        # Broadcast message to all connected peers.
        if not self.distribution:
            raise RuntimeError("Distributed inference not initialized")
        
        return await asyncio.to_thread(
            self.distribution.broadcast,
            message_type, payload, timeout=10
        )
    
    def get_network_status(self):
        # Get current network status.
        if not self.distribution:
            return {'status': 'not_initialized'}
        
        return {
            'status': 'active',
            'connected_agents': len(self.distribution.remote_agents),
            'queue_stats': self.distribution.get_queue_stats(),
            'mode': 'async' if self.distribution.use_async else 'sync'
        }
    
    # ============ Lifecycle Management ============

    async def shutdown_async(self):
        # Graceful shutdown of async components.
        print("🛑 Shutting down async components...")
        
        # Cancel all pending async tasks
        for task in self._async_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self._async_tasks:
            await asyncio.gather(*self._async_tasks, return_exceptions=True)
        
        # Shutdown distributed inference
        if self.distribution:
            self.distribution.stop()
        
        print("✅ Async shutdown complete")
    
    async def shutdown(self):
        if self.distribution:
            self.distribution.stop()      # sync call
            self.distribution.stop_server()
            
            
            if hasattr(self, '_shutdown_event'):
                self._shutdown_event.set()

            # cancel async tasks
            if self._async_tasks:
                for task in self._async_tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*self._async_tasks, return_exceptions=True)




    def cosine_similarity(self, a, b):
        eps = 1e-5
        b = b[0]

        if isinstance(a, (str, np.str_)):
            clean_str = str(a).replace('[', '').replace(']', '')
            a = np.fromstring(clean_str, sep=' ')
        elif isinstance(a, np.ndarray) and np.issubdtype(a.dtype, np.character):
            # catches arrays filled with string text
            clean_str = ' '.join(a.astype(str).flatten()).replace('[', '').replace(']', '')
            a = np.fromiter(
                    (x for x in clean_str.split() if x != "..."), dtype=float
                )               
        else:
            # Ensure standard float array if it was integers or objects
            a = np.asarray(a, dtype=float)

        # Handle variable b
        if isinstance(b, (str, np.str_)):
            clean_str = str(b).replace('[', '').replace(']', '')
            b = np.fromstring(clean_str, sep=' ')
        elif isinstance(b, np.ndarray) and np.issubdtype(b.dtype, np.character):
            clean_str = ' '.join(b.astype(str).flatten()).replace('[', '').replace(']', '')
            b = np.fromstring(clean_str, sep=' ')
        else:
            b = np.asarray(b, dtype=float)

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
    
        if len(a.shape) > 1 and len(b.shape) > 1 and a.shape[1] != b.shape[0]:
            a = np.asarray(a)
            b = np.asarray(b)
            subset_a = a[0, :b.shape[0]]
            subset_b = a[:subset_a.shape[0], 0]

            try:
                try:
                    dot_product = np.dot(subset_a, subset_b)
                except:
                    dot_product = np.dot(subset_a[0, :min(subset_a.shape[1], subset_b.shape[0])], subset_b[:min(subset_a.shape[1], subset_b.shape[0])])
            except:
                try:
                    dot_product = np.dot(a.flatten(), b[:a.flatten().shape[0]])
                except:
                    try:
                        dot_product = np.dot(a[:b.shape[0]], b.flatten()[:a.shape[0]])
                    except:
                        print('[-] No similarity due to inhomogenous shapes and failed attempts to find subsets, returning low similarity score.')
                        return 0.1
       

        else:
            subset_a = a[:a.shape[0]]
            subset_b = b[:subset_a.shape[0]]

            try:
                dot_product = np.dot(subset_a, subset_b) 
            except:
                try:
                    subset_b_2 = subset_b[:subset_a.shape[1], :subset_a.shape[0]]  
                    dot_product = np.dot(subset_a, subset_b_2)

                except:
                    print('[-] No similarity due to inhomogenous shapes and failed attempts to find subsets, returning low similarity score.')
  
                    return 0.0         

        cosine = dot_product / (norm_a * norm_b)
        cosine = np.clip(cosine, -1.0, 1.0)

        return cosine  

    def anisotropy_measurement(self, x):
        eps = 1e-5
        if isinstance(x, list):
            print(f'[=] Converting list to array with shape: {len(x)}')
            x = np.array(x)
            x = x.reshape(x.shape[0], -1)  # Flatten if necessary
        
        try:
            try:
                grads = np.gradient(x)
                # Stack gradients into a single array of vectors and find the norm of each
                # automatically handles multi-dimensional arrays (e.g., 2D, 3D volumes)
                stacked_grads = np.stack(grads, axis=-1)
                norms = np.linalg.norm(stacked_grads, axis=-1)
                
                # Safely filter out potential NaNs or infs (common at array boundaries)
                valid_norms = norms[np.isfinite(norms)]
                
                if len(valid_norms) == 0:
                    return 0.0 # Return zero or appropriate default if no valid values exist
                    
                # Calculate statistics using the clean data
                std_val = np.std(valid_norms)
                mean_val = np.mean(valid_norms)
                
                anisotropy = std_val / (mean_val + eps)    
                if np.isnan(anisotropy) or np.isinf(anisotropy):
                    anisotropy = self.pipeline.confidence_threshold
            except:
                try:
                    gradient = np.gradient(x)
                except:
                    subnet = x[:min(10, x.shape[0]), :min(10, x.shape[1])]
                    gradient = np.gradient(subnet.flatten())

                val = [np.linalg.norm(v) for v in gradient]
                anisotropy = np.std(val) / np.mean(val) + eps 
        except Exception as e:
            print(f'[!] Cant calculate anisotropy due to: {e}')   
            anisotropy = self.confidence_threshold

        if np.isnan(anisotropy) or np.isinf(anisotropy):
            anisotropy = self.confidence_threshold

        return anisotropy


    def modular_prediction_saving(self, X, X2, output):
        memory_name = self.memory_name
        if self.memory is None:
            self.memory = {}

        elif not isinstance(self.memory, dict):
            print(f'[!] Warning: memory was {type(self.memory)}, converting to dict')
            # Try to convert or reset
            if isinstance(self.memory, np.ndarray):
                # If memory was an array, changing type - log it
                print(f'[!] Converting memory from array to dict, old shape: {self.memory.shape}')
            self.memory = {}

        self.memory['TW'] = X, output # transformers Weight
        self.memory['MW'] = X2, output # MLP Weight
        try:
            print('🚀 Memory Prediction Added!')
            self.storage.save_model_dict(memory_name, self.memory, type='Pipeline', model_type='prediction')
        except Exception as e:
            print(f'[!] Cant save model memory: {e}')


    def modular_probability_saving(self, X, X2, prob):
        memory_name = self.memory_name

        if self.memory is None:
            self.memory = {}

        elif not isinstance(self.memory, dict):
            print(f'[!] Warning: memory was {type(self.memory)}, converting to dict')
            # Try to convert or reset
            if isinstance(self.memory, np.ndarray):
                # If memory was an array, changing type - log it
                print(f'[!] Converting memory from array to dict, old shape: {self.memory.shape}')
            self.memory = {}
    
        self.memory['TP'] = (X, prob)
        self.memory['MP'] = (X2, prob)
        
        # Save to database
        try:
            self.storage.save_model_dict(memory_name, self.memory, type='Pipeline', model_type='probs')
            print('🚀 Memory Probability Added!')
        except Exception as e:
            print(f'[!] Failed to save memory: {e}')
       
    def _cross_session_availability(self):
        try:
            print('=== CROSS SESSION MEMORY AVAILABILITY AND TRANSFER ===')
            print('1. Export memory session')
            print('2. Import memory session')
            print('3. Sync with another device')
            print('4. List sessions')

            chosen = input('[=] Choose Options [1/2/3/4]: ')
            if chosen != '3':
                filename = input('[=] Insert filename name to export or import or list session (ex. name: memory_session): ')

            if chosen == '1':
                if filename:
                    print(f'[+] Exporting memory session with name {filename}...')
                    self.session_automation.export_session(filename)
                    print(f'🚀 {filename} successfully exported as json!')
                else:
                    print('[-] Invalid filename!')
                    pass

            elif chosen == '2':
                if isinstance(filename, str):
                    json_converted = filename + '.json'
                    print(f'[=] Importing memory session, filename: {json_converted}...')
                    self.session_automation.import_session(json_converted)
                    print(f'[+] Successfully imported {json_converted}! ')
                else:
                    print('[-] Invalid filename!')
                    pass

            elif chosen == '3':
                ip_number = input('[=] Insert ip number of your device for syncing: ')  
                if ip_number:
                    print('[=] Syncing with device to export memory session...')  
                    self.session_automation.sync_with_another_device(ip_number, port=5000) 
                else:
                    print('[-] Invalid filename!')
                    pass

            elif chosen == '4':
                print('[=] Listing sessions...')                
                print('[!] Note: you must put the common name of memory sessions you have, (ex: memory_sessions)')
                print('if your most memory sessions you have contains memory_ name in front, insert it in the input.')
                self.session_automation.list_sessions(filename)
            else:
                print('[-] Invalid options! ')
                pass

        except Exception as e:
            print(f'[-] Warning! Error detected during session availability: {e}')
            pass


    def model_memory_gate(self, x, x2):
        memory = self.memory
        is_corrupted, reason, _ = self.is_memory_corrupted(memory)
               

        if isinstance(memory, np.ndarray):
            if self._get_num_classes() and memory.shape[-1] == self._get_num_classes():
                if is_corrupted:
                    print('[!] Memory corruption detected, Trying possible conversion to extract memory...')
                    print(f'[REASON]: {reason}')
                else:
                    print('[+] Memory is a direct array, converting...')
                               
                num_classes = self._get_num_classes()                 
                if memory.ndim == 1:                   
                    print(f'[+] Memory is flat array shape {memory.shape}, converting to 2D...')
                    converted_memory = memory.reshape(1, -1)
                    
                    if converted_memory.shape[-1] == num_classes:
                        print(f'[+] Memory converted to shape {converted_memory.shape}, returning it.')
                        return converted_memory
                    else:
                        print(f'[!] Shape mismatch! Got {converted_memory.shape[-1]} classes, expected {num_classes}')
                        return None
                
                # Case 2: Already 2D array (1, 8) or (N, 8)
                elif memory.ndim == 2:
                    if memory.shape[-1] == num_classes:
                        print(f'[+] Memory already in correct shape {memory.shape}, returning it.')
                        return memory
                    else:
                        print(f'[!] Memory shape mismatch! shape: {memory.shape}, expected classes: {num_classes}')
                        return None
                
                # Case 3: Higher dimensional array
                else:
                    print(f'[!] Memory has unexpected dimensions: {memory.ndim}D, shape: {memory.shape}, No matching memory!')
                    return None

            elif not self._get_num_classes():
                print('[!] Cant get number of classes from transformer and MLP model!')
                return None
            else:
                print(f'[!] Memory shape mismatch! shape: {memory.shape}, No matching memory! ')
                return None

        if isinstance(memory, list):
            print('[==] Memory is a list, checking contents...')
            if len(memory) > 0 and isinstance(memory[0], np.ndarray):
                print('[+] Got probability features inside list!')
                return memory[0].copy()
            else:
                if self._gate_from_list(memory, x, x2):
                    return self._gate_from_list(memory, x, x2)
                else:
                    print('[!] No matching item from memory!')
                    return None

        if isinstance(memory, dict):
            for key, value in list(memory.items()):
                is_val_corrupted, reason, _ = self.is_memory_corrupted(value)
                if is_val_corrupted:
                    print(f'[!] Removing corrupted entry {key}: {reason}')
                    del memory[key] 

            cache_trans_memory = [key for key, (inp) in memory.items() if key.startswith('TW') and (isinstance(inp, np.ndarray) or isinstance(inp, list)) and self.cosine_robust_similarity(x, inp) >= 0.9]
            cache_mlp_memory =  [key for key, (inp2) in memory.items() if key.startswith('MW') and (isinstance(inp2, np.ndarray) or isinstance(inp2, list)) and self.cosine_similarity(x2, inp2) >= 0.9]

            if cache_mlp_memory and cache_trans_memory:
                for memo in cache_trans_memory:
                    _, out = memory[memo]
                    
                for memo2  in cache_mlp_memory:
                    _, out = memory[memo2]

                if isinstance(out, str):
                    out = np.array([float(x) for x in out.strip('[]').split(',')])  # Convert string to numpy array

                output = out.copy()
                return output      
            else:
                if cache_mlp_memory:
                    print('[+] Found matching memory from mlp past memory!')                
                    for memo in cache_mlp_memory:
                        _, out = memory[memo] 
                        if isinstance(out, str):
                            out = np.array([float(x) for x in out.strip('[]').split(',')])  # Convert string to numpy array

                    output = out.copy() 
                    return output 

                elif cache_trans_memory:
                    print('[+] Found matching memory from transformer past memory!')                
                    for memo in cache_trans_memory:
                        _, out = memory[memo] 
                        if isinstance(out, str):
                            out = np.array([float(x) for x in out.strip('[]').split(',')])  # Convert string to numpy array

                    output = out.copy() 
                    return output

                else:
                    print('🔄 No Matching Memory!')
                    return None

            
        else:
            print('[!] No matching memory types!')
            return None

    def _gate_from_list(self, memory: list, x, x2) -> Optional[np.ndarray]:
        # xtract from list memory (fallback for corrupted storage)

        print('[=] Extracting from list memory.. handling possible corruption of data...')
        for item in memory:
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                stored_x = item[0]
                stored_out = item[1]
                
                # Check similarity
                print('[=] checking similarity from stored X data and stored output...')
                if self.cosine_robust_similarity(x, stored_x) >= 0.9:
                    return self._to_numpy_array(stored_out)

        print('[=] Cant get item from memory, possible dangerous data Corruption!')
        return None


    def _to_numpy_array(self, value) -> Optional[np.ndarray]:
        """
        Convert various return types to numpy array.
        Handles: array, list, tuple, string, scalar.
        """
        if value is None:
            print('[!] Value is None!')
            return None
        
        # check here numpy array
        if isinstance(value, np.ndarray):
            return value.copy() if value.size > 0 else None
        
        # List or tuple of numbers
        if isinstance(value, (list, tuple)):
            try:
                arr = np.array(value, dtype=np.float32)
                if arr.ndim == 0:
                    arr = arr.reshape(1)
                return arr
            except:
                return None
        
        # String (from database JSON)
        if isinstance(value, str):
            try:
                # JSON first
                if value.startswith('['):
                    parsed = json.loads(value)
                    return np.array(parsed, dtype=np.float32)
                # space-separated
                parts = value.strip('[]').split()
                if parts:
                    return np.array([float(p) for p in parts], dtype=np.float32)
            except:
                pass
            return None
        
        # Scalar number
        if isinstance(value, (int, float)):
            return np.array([value], dtype=np.float32)
        
        print(f'[!] Cannot convert to array: {type(value)}')
        return None


    def _get_num_classes(self) -> int:
        # Get number of classes from model first
        if hasattr(self, 'model2') and self.model2:
            return self.model2.output.shape[1]
        elif hasattr(self, 'mlp') and self.mlp.layers:
            return self.mlp.layers[-1].b.shape[1]
        return None  


    def model_probability_gate(self, x, x2):
        memory = self.memory
        is_corrupted, reason, _ = self.is_memory_corrupted(memory)

        if isinstance(memory, np.ndarray):
            if self._get_num_classes() and not memory.shape[-1] == self._get_num_classes():
                if is_corrupted:
                    print('[!] Memory corruption detected, Trying possible conversion to extract memory...')
                    print(f'[REASON]: {reason}')
                else:
                    print('[+] Memory is a direct probability, converting...')

                num_classes = self._get_num_classes()       

                if memory.ndim == 1:                  
                    print(f'[+] Memory is flat array shape {memory.shape}, converting to 2D...')
                    converted_memory = memory.reshape(1, -1)
                    
                    if converted_memory.shape[-1] == num_classes:
                        print(f'[+] Memory converted to shape {converted_memory.shape}, returning it.')
                        return converted_memory
                    else:
                        print(f'[!] Shape mismatch! Got {converted_memory.shape[-1]} classes, expected {num_classes}')
                        return None
                
                # Case 2: Already 2D array (1, 8) or (N, 8)
                elif memory.ndim == 2:
                    if memory.shape[-1] == num_classes:
                        print(f'[+] Memory already in correct shape {memory.shape}, returning it.')
                        return memory
                    else:
                        print(f'[!] Memory shape mismatch! shape: {memory.shape}, expected classes: {num_classes}')
                        return None
                
                # Case 3: Higher dimensional array
                else:
                    print(f'[!] Memory has unexpected dimensions: {memory.ndim}D, shape: {memory.shape}, No matching memory!')
                    return None

            elif not self._get_num_classes():
                print('[!] Cant get number of classes from transformer and MLP model!')
                return None
            else:
                print(f'[!] Memory shape mismatch! shape: {memory.shape}, No matching memory! ')
                return None

        if isinstance(memory, list):
            print('[==] Memory is a list, checking contents...')
            if len(memory) > 0 and isinstance(memory[0], np.ndarray):
                print('[+] Got probability features inside list!')
                return memory[0].copy()
            else:
                self._gate_from_list(memory, x, x2)

        if isinstance(memory, dict):
            for key, value in list(memory.items()):
                is_val_corrupted, reason, _ = self.is_memory_corrupted(value)
                if is_val_corrupted:
                    print(f'[!] Removing corrupted entry {key}: {reason}')
                    del memory[key] 

            cache_trans_memory = [key for key, (inp) in memory.items() if key.startswith('TP') and (isinstance(inp, np.ndarray) or isinstance(inp, list)) and self.cosine_robust_similarity(x, inp) >= 0.95]
            cache_mlp_memory =  [key for key, (inp2) in memory.items() if key.startswith('MP') and (isinstance(inp2, np.ndarray) or isinstance(inp2, list)) and self.cosine_similarity(x2, inp2) >= 0.95]

            if cache_mlp_memory and cache_trans_memory:
                for memo in cache_trans_memory:
                    _, out = memory[memo]

                for memo2  in cache_mlp_memory:
                    _, out = memory[memo2]

                if isinstance(out, str):
                    out = np.array([float(x) for x in out.strip('[]').split(',')])  # Convert string to numpy array

                output = out.copy()
                return output      
            else:
                print('🔄 No Matching Probability!')
                return None
        else:
            print('[!] No matching memory types!')
            return None

    def prediction_batch(self, texts):

        self.initialize_fitting(texts)
        if not texts:
            return []
        
        # Prepare batch inputs

        input_ids_list = []
        X_raw_list = []

        for text in texts:
            # Prepare transformer input
            input_ids = np.array([self.encode(text, self.vocab)])
            input_ids_list.append(input_ids)
            
            # Prepare MLP input
            if not hasattr(self, 'tfidf') or self.tfidf is None:
                self.initialize_fitting([text])
            X_raw = self.tfidf.transform([text]).toarray()
            X_raw_list.append(X_raw)
        
        # Stack into batches
        batch_input_ids = np.vstack(input_ids_list)  # (batch_size, seq_len)
        batch_X_raw = np.vstack(X_raw_list) # (batch_size, features)

        if self.labels is None and self.titles is None:
            _, y_true = self.input_encoding(list(zip(texts, texts)))  # No-value labels for y_true
        else:
            dataset, _ = self.data_preparation(self.titles, self.labels)
            y_true = self.input_encoding(dataset)
        
        # Run batch prediction through your existing logic
        return self._batch_prediction_core(batch_input_ids, batch_X_raw, y_true)



    def _batch_model_memory_gate(self, batch_input_ids, batch_X_raw):
        batch_probs = [None] * len(batch_input_ids)
        
        for i in range(len(batch_input_ids)):
            probs = self.model_memory_gate(
                batch_input_ids[i:i+1], 
                batch_X_raw[i:i+1]
            )
            if probs is not None:
                arr = np.array(probs)
                if arr.ndim > 1:
                    arr = arr[0]
                elif arr.ndim == 0:
                    arr = arr.reshape(1)
                batch_probs[i] = arr.copy()
        
        return batch_probs


    def _refit_sparse_data(self, X_features, texts, threshold=0.3):
        """Refit TF-IDF if zero-row ratio exceeds threshold."""
        X_features = np.asarray(X_features, dtype=np.float32)
        if X_features.ndim == 1:
            X_features = X_features.reshape(1, -1)        
            X_features = np.asarray(X_features)
                    
        zero_rows = np.where(X_features.sum(axis=1) == 0)[0]
        zero_ratio = len(zero_rows) / len(X_features)
        
        if zero_ratio > threshold:
            print(f'[!] {len(zero_rows)} zero rows ({zero_ratio:.0%}), refitting TF-IDF on current batch')
            self.tfidf.fit(texts)
            X_features = self.tfidf.transform(texts).toarray()
            
            # second pass — fill remaining zeros with checksum fingerprint
            zero_rows = np.where(X_features.sum(axis=1) == 0)[0]
            for i in zero_rows:
                text = texts[i] if isinstance(texts[i], str) else str(texts[i])
                checksum = int(hashlib.md5(text.encode()).hexdigest(), 16)
                rng = np.random.default_rng(checksum)
                X_features[i] = rng.uniform(0.01, 0.1, size=X_features.shape[1])
                print(f'[!] Row {i} still zero after refit, checksum fallback applied')
        
        return X_features 


    def _batch_prediction_core(self, batch_input_ids: np.ndarray, batch_X: np.ndarray, 
                                batch_size: Any = None, show_progress: bool = True) -> np.ndarray:
        """
        Robust batch prediction with dynamic shape handling.
        """
        if len(batch_input_ids) == 0:
            return np.array([])
        
        # Auto-calculate optimal batch size if not provided
        if batch_size is None:
            batch_size = self._calculate_optimal_batch_size(batch_input_ids, batch_X)
        
        if isinstance(batch_size, (list, np.ndarray)):
            try:
                try:
                    batch_size = int(batch_size[0]) if len(batch_size) > 0 else 32
                except:
                    batch_size = int(batch_size[0][0])
            except:
                batch_size = int(batch_size)
        else:
            if isinstance(batch_size, (tuple, list, np.ndarray)):
                batch_size = batch_size[0]
                batch_size = len(batch_size)
            else:
                batch_size = int(batch_size)
            
        n_samples = len(batch_input_ids)
        chunks = []
        if isinstance(batch_size, (tuple, list)):
            batch_size = batch_size[0] 

        print(f'[=] Total samples: {n_samples}, using batch size: {batch_size}, total batches: {((n_samples - 1) // batch_size) + 1}')
        
        for i in range(0, n_samples, batch_size):
            chunk = (
                batch_input_ids[i:i + batch_size],
                batch_X[i:i + batch_size]
            )
            chunks.append(chunk)
        
        # ✅ Determine number of classes dynamically from first successful chunk
        num_classes = None
        batch_probs = None
        
        for chunk_idx, (chunk_ids, chunk_X) in enumerate(chunks):
            if show_progress:
                start_idx = chunk_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                print(f"\r📊 Processing batch {chunk_idx + 1}/{len(chunks)} (samples {start_idx}-{end_idx})...", end="")
            
            try:
                # Process chunk with memory gate
                chunk_probs = self._process_batch_chunk(chunk_ids, chunk_X)
                
                # ✅ Handle numpy array conversion
                if isinstance(chunk_probs, list):
                    chunk_probs = np.array(chunk_probs)
                
                # ✅ Determine number of classes from first successful chunk
                if num_classes is None:
                    if chunk_probs.ndim == 1:
                        num_classes = 1
                    else:
                        num_classes = chunk_probs.shape[1] if chunk_probs.ndim > 1 else len(chunk_probs)
                    
                    # Initialize results array
                    batch_probs = np.zeros((n_samples, num_classes))
                    print(f'\n[=] Detected {num_classes} classes from chunk {chunk_idx + 1}')
                
                # ✅ Handle dimension mismatches
                if chunk_probs.ndim == 1:
                    chunk_probs = chunk_probs.reshape(-1, 1)
                
                # ✅ If chunk has different number of classes, pad or trim
                if chunk_probs.shape[1] != num_classes:
                    print(f'[=] Shape mismatch: chunk has {chunk_probs.shape[1]} classes, expected {num_classes}')
                    
                    if chunk_probs.shape[1] > num_classes:
                        # Trim extra classes
                        chunk_probs = chunk_probs[:, :num_classes]
                        print(f'[=] Trimmed to {chunk_probs.shape[1]} classes')
                    else:
                        # Pad missing classes
                        padded = np.zeros((chunk_probs.shape[0], num_classes))
                        padded[:, :chunk_probs.shape[1]] = chunk_probs
                        chunk_probs = padded
                        print(f'[=] Padded to {chunk_probs.shape[1]} classes')
                
                # Place in results
                start_idx = chunk_idx * batch_size
                end_idx = start_idx + len(chunk_probs)
                batch_probs[start_idx:end_idx] = chunk_probs
                
            except Exception as e:
                print(f"\n⚠️ Chunk {chunk_idx + 1} failed: {e}")
                traceback.print_exc()
                
                # Fill failed chunk with zeros
                start_idx = chunk_idx * batch_size
                end_idx = start_idx + len(chunk_ids)
                
                if batch_probs is None:
                    # If first chunk failed, initialize with default
                    num_classes = self._get_num_classes()
                    batch_probs = np.zeros((n_samples, num_classes))
                
                batch_probs[start_idx:end_idx] = 0
        
        if show_progress:
            print(f"\r✅ Batch complete: {n_samples} samples processed                    ")
        
        return batch_probs if batch_probs is not None else np.array([])

    def _process_batch_chunk(self, chunk_ids: np.ndarray, chunk_X: np.ndarray) -> np.ndarray:
        """
        Process a single chunk - core batch logic with memory gate.
        """
       
        chunk_probs = self._batch_model_memory_gate(chunk_ids, chunk_X)
        
        needs_fresh = [i for i, p in enumerate(chunk_probs) if p is None]
        
        if needs_fresh:
            # Extract samples that need fresh prediction
            fresh_ids = chunk_ids[needs_fresh]
            fresh_X = chunk_X[needs_fresh]
            
            # Get fresh predictions from ensemble
            fresh_probs, _ = self.ensemble.predict_ensemble(
                fresh_ids, fresh_X, 
                np.zeros((len(fresh_ids), self._get_num_classes())),
                method='dynamic', embedded=False
            )
            
            # Store fresh predictions
            for i, fresh_idx in enumerate(needs_fresh):
                chunk_probs[fresh_idx] = fresh_probs[i]
                
                # Cache to memory (first 2 only to avoid spam)
                if fresh_idx < 2:
                    self.modular_prediction_saving(
                        fresh_ids[i:i+1],
                        fresh_X[i:i+1],
                        fresh_probs[i:i+1]
                    )
        
        # Convert list to array
        return np.array([p if p is not None else np.zeros(self._get_num_classes()) 
                        for p in chunk_probs])



    def _calculate_optimal_batch_size(self, batch_input_ids: np.ndarray, batch_X: Any=None) -> int:
        """
        Calculate optimal batch size based on available memory.
        """
        try:
            import psutil
            # Estimate memory per sample
            sample_size = batch_input_ids[0].nbytes + batch_X[0].nbytes if hasattr(batch_X, '__len__') else 1024
            available_memory = psutil.virtual_memory().available
            max_samples = int(available_memory * 0.1 / sample_size)  # Use 10% of memory
            return min(64, max(8, max_samples))
        except:
            # Fallback to conservative batch size
            return 32

    def _get_num_classes(self) -> int:
        if hasattr(self, 'model2') and self.model2 is not None:
            if hasattr(self.model2, 'output'):
                return self.model2.output.shape[1]
        
        if hasattr(self, 'mlp') and self.mlp is not None:
            if hasattr(self.mlp, 'layers') and len(self.mlp.layers) > 0:
                last_layer = self.mlp.layers[-1]
                if hasattr(last_layer, 'b'):
                    return last_layer.b.shape[1]
        
        if hasattr(self, 'reverse_map') and self.reverse_map:
            return len(self.reverse_map)
        
        if hasattr(self, 'label_map') and self.label_map:
            return len(self.label_map)
        
        # infer from any cached probability
        if isinstance(self.memory, dict):
            for value in self.memory.values():
                if isinstance(value, np.ndarray) and value.ndim == 1:
                    return value.shape[0]
                if isinstance(value, list) and len(value) > 0:
                    if all(isinstance(x, (int, float)) for x in value[:5]):
                        return len(value)
        
        print('[!] Could not determine num_classes, defaulting to 0')
        return 0
    
  
    def predict_async(self, text, callback=None):
        try:
            return self.batcher.add_request(text, callback)
        except Exception as e:
            print(f'[=] error in automatic batcher: {e}')
            return None
    

    def predict_single(self, text):
        result = [None]
        
        def callback(r):
            result[0] = r
        
        self.predict_async(text, callback)
        
        # Wait for result
        while result[0] is None:
            time.sleep(0.001)
        
        return result[0]
 

    def _batch_hybrid_prediction(self, batch_input_ids, batch_X_raw, y_true):
        print('[+] Initiating hybrid prediction batching...')
        idx_total = 0
        batch_probs = self._batch_model_memory_gate(batch_input_ids, batch_X_raw)
        
        needs_prediction = [i for i, p in enumerate(batch_probs) if p is None]
        
        if needs_prediction:
            fresh_input_ids = batch_input_ids[needs_prediction]
            fresh_X_raw = batch_X_raw[needs_prediction]
            
            # Batch ensemble
            fresh_probs, details = self.ensemble.predict_ensemble(
                fresh_input_ids, fresh_X_raw, y_true, method='dynamic', embedded=False
            )
            
            for i, idx in enumerate(needs_prediction):
                batch_probs[idx] = fresh_probs[i]
                idx_total += 1
                if idx_total < 2:
                    self.modular_prediction_saving(
                        fresh_input_ids[i:i+1], 
                        fresh_X_raw[i:i+1], 
                        fresh_probs[i:i+1]
                    )

        valid_probs = []
        for p in batch_probs:
            if p is None:
                # Use zeros as fallback for None values
                valid_probs.append(np.zeros_like(fresh_probs[0]))
            elif isinstance(p, list):
                valid_probs.append(np.array(p))
            else:
                valid_probs.append(p)
        
        try:
            try:
                return np.array(valid_probs)
            except:
                return valid_probs
        except Exception as e:
            print(f'[-] Error converting batch probabilities to array: {e}')
            return valid_probs
    

    def _batch_predict_proba(self, batch_input_ids, batch_X, type='Hybrid'):
        batch_size = len(batch_input_ids)
        idx_total = 0
        
        # Batch memory gate
        batch_probs = [None] * batch_size
        for i in range(batch_size):
            probs = self.model_probability_gate(
                batch_input_ids[i:i+1], 
                batch_X[i:i+1]
            )
            if probs is not None:
                batch_probs[i] = probs[0]
        
        # Find which need fresh prediction
        needs_prediction = [i for i, p in enumerate(batch_probs) if p is None]
        
        if needs_prediction:
            fresh_input_ids = batch_input_ids[needs_prediction]
            fresh_X = batch_X[needs_prediction]
            
            # Batch transformer and MLP
            transformer_pred, fresh_probs, attn_weights = self.model2.predict(fresh_input_ids)
            mlp_pred = self.mlp.forward(fresh_X)
            
            mlp_pred_indices = np.argmax(mlp_pred, axis=1)
            trans_pred_indices = np.argmax(fresh_probs, axis=1)
            
            # Calibrate batch
            for i, idx in enumerate(needs_prediction):
                idx_total += 1
                if mlp_pred_indices[i] != trans_pred_indices[i]:
                    calibrated = self._calibrate_probs(
                        fresh_probs[i:i+1], 
                        [mlp_pred_indices[i]], 
                        attn_weights[i:i+1] if attn_weights is not None else None,
                        fresh_input_ids[i:i+1]
                    )
                    batch_probs[idx] = calibrated[0]
                else:
                    # Models agree
                    probs_i = fresh_probs[i].copy()
                    target = mlp_pred_indices[i]
                    probs_i[target] = min(probs_i[target] * 1.2, 0.95)
                    probs_i /= probs_i.sum()
                    batch_probs[idx] = probs_i
                
                # Save to memory
                if idx_total < 2:
                    self.modular_probability_saving(
                        fresh_input_ids[i:i+1], 
                        fresh_X[i:i+1], 
                        np.array([batch_probs[idx]])
                    )
      
        return np.array(batch_probs)



    def hybrid_prediction(self, rules, input_ids, dataset):
        X, y, _, _ = self.feature_generation(rules, dataset) 
        if len(input_ids.shape) == 2 and input_ids.shape[0] > 1:
            # this is batch mode version
            return self._batch_hybrid_prediction(input_ids, X, y)

        probs = self.model_memory_gate(input_ids, X)

        if probs is None or not self.agreement:
            if not self.autonomous:
                print('= Prediction Method needed: ')
                print('[1]. dynamic')
                print('[2]. meta')
                print('[3]. attention')

                print('[-] Autonomous prediction give the model full control of its dynamic prediction, without any user input.')
                choose_method = input('[=] Autonomous prediction initiated? [Y/N] (press N to insert manual prediction method): ')

                if choose_method == 'Y':
                    self.autonomous = True
                    probs, details = self.ensemble.predict_ensemble(input_ids, X, y, method='dynamic', embedded=True) 

                else:
                    method = input('|| Choose one method (ex: dynamic): ')
                    if method:
                        probs, details = self.ensemble.predict_ensemble(input_ids, X, y, method=method, embedded=True)
                    else:
                        print('|| Invalid Method.. returning to dynamic prediction..')
                        probs, details = self.ensemble.predict_ensemble(input_ids, X, y, method='dynamic', embedded=True)    
            else:
                print('[+] Autonomous dynamic prediction: ')
                probs, details = self.ensemble.predict_ensemble(input_ids, X, y, method='dynamic', embedded=True) 

            self.modular_prediction_saving(input_ids, X, probs)
            print('🚀 Memory Added!')

        return probs

    def _handle_distributed_connections(self, probs, self_attn_weights, input_ids, agreement):
        print('=== AGENT DISTRIBUTIED INFERENCE HANDLING ===')
        print('1. Handle local In-device Peer')
        print('2. Handle external-device Peer')

        program = None
        if not self.autonomous:
            program = input('[=] Pick your choice [1/2] (choose N to skip): ')

        if program == '1' or self.autonomous:
            print('=== IN-DEVICE PEER REQUEST INITIATED ===')
            probs = self.distribution._handle_peer_agent_request(probs, self_attn_weights, input_ids, type='DevicePeer', agreement=agreement)
            if self.distribution.query_node.peer_trust < self.confidence_threshold:
                print('[-] Peer trust is low, broadcasting ping to check for better peers...')
                alive_agents = self.distribution.broadcast_ping()

                if alive_agents:
                    print(f'[+] Alive agents: {alive_agents} identified, enabling external peer connections for better assistance...')
                    self.external_peer_enabled = bool(alive_agents)
                    self.autonomous = False

        elif program == '2' or self.external_peer_enabled:
            print('=== EXTERNAL PEER REQUEST INITIATED ===')
            ip_number = input('[=] Insert IP Number to connect with peer: ')

            if ip_number:
                try:
                    distributed_a = self.distribution
                    
                    distributed_a.start_server()
                    if distributed_a.connect_to_agent(ip_number, 5555):
                        print(f'[=+=] Successfully connected to external peer at {ip_number}!')
                        time.sleep(10)

                        print('=== EXTERNAL PEER REQUEST INITIATED ===')
                        print('[1]. Request prediction')
                        print('[2]. Handle Peer uncertainty')
                        sec_program = input('[=] Pick your choice [1/2]: ')    

                        if sec_program == '1':
                            for intent in self.intents:
                                result = distributed_a.request_prediction_method(self, intent)
                                print(f"[+] Remote prediction: {result}")
                    
                                # Ensemble vote across all agents
                                votes = []
                                list_probs = []

                            # Check network status
                            print('=== Checking network status with broadcast ping... ===')
                            alive_agents = distributed_a.broadcast_ping()  
                            if alive_agents:
                                print(f'[+] Alive agents: {alive_agents} identified, requesting ensemble votes...')                          
                                for agent_id in distributed_a.remote_agents:
                                    peer_probs, vote = distributed_a.request_ensemble_vote(agent_id, intent)
                                    if vote:
                                        votes.append(vote)
                                        list_probs.append(peer_probs)
                                        
                                        for vote in votes:
                                            print(f'[+] Prediction: {vote['prediction']}')
                                            print(f'[+] Confidence: {vote['confidence']}')
                                            print(f'[+] Trust: {vote['trust_score']}')
                                            if vote['confidence'] > self.confidence_threshold:
                                                for i in range(len(probs)):
                                                    probs = probs[i] * (1.0 + vote['trust_score'] * vote['confidence'])
                                                probs = probs.copy() / np.sum(probs) # Normalize after adjustment
                                else:
                                    print(f'[-] No alive agents found, Total: {alive_agents} Agent found. Using local prediction only.')
                                    probs = self.distribution._handle_peer_agent_request(probs, self_attn_weights, input_ids, type='ExternalPeer', agreement=agreement)

                            distributed_a.print_network_status()

                        elif sec_program == '2':
                            probs = self.distribution._handle_peer_agent_request(probs, self_attn_weights, input_ids, type='ExternalPeer', agreement=agreement)     
                    else:
                        print(f'[-] No Peer agents found, Using local prediction only.')
                        probs = self.distribution._handle_peer_agent_request(probs, self_attn_weights, input_ids, type='DevicePeer', agreement=agreement) 
                        time.sleep(10)

                except Exception as e:
                    print(f'[-] Error establishing connections: {e}, returning previous probs.')
                    self.distribution.report_failure(id(self), 'processing', reason=f'{e}')

            else:
                print(f'[-] Invalid Choice... returning previous probs.')
                self.distribution.report_failure(id(self), 'processing', reason="InvalidChoice")                        

        elif program == 'N':
            print('|| Skipping Peer connections, returning previous probs')
        else:
            print('[-] Invalid choice! returning previous probs')
            
        return probs



    def mlp_predict(self, X):
        if isinstance(X, str) or isinstance(X[0], str):
            self.initialize_fitting(X)            
            X_tfidf = self.tfidf.transform(X).toarray() 

        logits = self.mlp.prediction(X_tfidf)
        return logits

    
    def predict_proba(self, input_ids, X, type='Hybrid', embedded=False):
        eps = 1e-5
        probs_memory = self.model_probability_gate(input_ids, X)
        if isinstance(self.storage.id_history, list) or isinstance(self.storage.id_history, np.ndarray) and not self.agent_id in self.storage.id_history:
            self.storage.id_history.append(self.agent_id)
            id_history = self.storage.id_history
        else:
            self.temporary_id.append(self.agent_id)
            id_history = self.temporary_id

        if self.use_transformer:
            is_batch = len(input_ids.shape) == 2 and input_ids.shape[0] > 1
        else:
            is_batch = False

        AME = self.AME_Encoder(input_ids)
        AMR = 1.0 / (1.0 + np.exp(-AME))
        
        if is_batch:
            return self._batch_predict_proba(input_ids, X, type) 

        if type == 'Hybrid' and self.use_transformer:
            print('[=] Hybrid based classification method.')
            transformer_pred, probs, attn_weights = self.model2.predict(input_ids, embedded=embedded)
            mlp_pred = self.mlp.forward(X)

            if mlp_pred.ndim == 1:
                mlp_pred = mlp_pred.reshape(1, -1)
            # Ensure transformer_pred is 2D
            if transformer_pred.ndim == 1:
                transformer_pred = transformer_pred.reshape(1, -1)

            mlp_pred_indices = np.argmax(mlp_pred, axis=1)
            trans_pred_indices = np.argmax(transformer_pred, axis=1)
           
            if probs_memory is None:
                agreement = np.allclose(mlp_pred_indices, trans_pred_indices, rtol=eps)
          
            else:
                # memory agreement must match previously detected learned patterns, contextual transformer prediction isnot needed
                try:
                    probs_memory_ = np.argmax(probs_memory) 
                except:
                    probs_memory_= np.argmax(probs_memory, axis=1)

                agreement = np.allclose(mlp_pred_indices, probs_memory_, rtol=eps)

            need_peer_condition = not agreement and probs_memory is None
            self.agreement = agreement

            if not agreement:
                self.peer_assistance_threshold += 0.1                                     
                # if both pattern are still conflicting, use contextual relations for sorting regularization.
                if need_peer_condition:
                    print('|| Uncertain prediction, requesting peer assistance if allowed...')
                    probs = self._handle_distributed_connections(probs, attn_weights, input_ids, agreement)
                else:
                    need_calibration_condition = not agreement and self.final_conf_score > self.confidence_threshold
                    if need_calibration_condition:
                        print('[||] Uncertain prediction, but memory exist, skipping peer assistance and calibrating with attention because of high confidence...')                        
                        probs = self._calibrate_probs(probs, mlp_pred_indices, attn_weights, input_ids)       
                    else:
                        print('[-] Uncertain prediction, needing local peer assistance...')                        
                        probs = self._handle_distributed_connections(probs, attn_weights, input_ids, agreement)

            else:
                self.peer_assistance_threshold -= 0.2               
                print('[+] Both Models agree, Normalizing prediction with confidence boost...')
                for i, target in enumerate(mlp_pred_indices):
                    probs[i, target] = min(probs[i, target] * 1.2, 0.95)
                    probs[i] /= probs[i].sum()

                    
            if not agreement and probs_memory is not None:
                self.storage.save_peer_needs_dict(self.memory_name, probs_memory, mlp_pred, id_history)   
            else:
                self.storage.save_peer_needs_dict(self.memory_name, probs, mlp_pred, id_history)

            self.modular_probability_saving(input_ids, X, probs)
            print('🚀 Memory Added!')
            return probs
        else:
            print(f'[=] MLP Based classification method. Transformer usage permission: {self.use_transformer}')
            logits = self.mlp.forward(X)

            if probs_memory is not None:
                mlp_pred_indices = np.argmax(logits, axis=1)
                probs_memo = np.array(probs_memory)
                try:
                    probs_memory_ = np.argmax(probs_memory) 
                except:
                    probs_memory_ = np.argmax(probs_memory, axis=1)

                agreement = np.allclose(mlp_pred_indices, probs_memory_, rtol=eps)
                
                if not agreement:
                    # if both pattern are still conflicting, used latest prediction
                    probs = logits.copy()       
                else:
                    for i, target in enumerate(mlp_pred_indices):
                        probs_memo[i, target] = min(probs_memo[i, target] * AMR, 0.95)
                        probs_memo[i] /= probs_memo[i].sum()

            return logits

    def data_preparation(self, titles, labels):
        datasets = []
        raw = []
        for title in titles:
            tupled_title = (str(title))
            datasets.append(tupled_title)
            raw.append(str(title))

        for label in labels:
            tupled_label = (str(label))
            datasets.append(tupled_label)
            raw.append(str(label))

        self.initialize_fitting(raw)
        X_raw = self.tfidf.transform(raw).toarray()
        X_raw = self._refit_sparse_data(X_raw, raw)

        return datasets, X_raw

    
    def _calibrate_probs(self, probs, target_preds, attn_weights, input_ids):
        calibrated = probs.copy()
        if isinstance(input_ids, list):
            input_ids = np.array(input_ids)

        if len(probs.shape) > 1:
            n_classes = probs.shape[1]
        else:
            n_classes = probs.shape[0]

        batch_size = len(target_preds)
        eps = 1e-5

        for i in range(batch_size):
            mlp_target = target_preds[i]

            if attn_weights is None:
                anisotropy = self.anisotropy_measurement(mlp_target)     
            else:
                anisotropy = self.anisotropy_measurement(attn_weights[i])
            if attn_weights is not None and i < len(attn_weights):
                attn = attn_weights[i]
    
                score_quality = np.std(attn) if attn.size > 0 else self.confidence_threshold
                abstract_score = self.confidence_threshold + score_quality * anisotropy

            else:
                if attn_weights is not None:
                    score_quality = 1.0 / (1.0 + np.exp(-attn_weights[i]))
                else:
                    score_quality = 1.0 / (1.0 + np.exp(-mlp_target))

                abstract_score = (1.0 - score_quality) + eps

            self.temperature = (1.0 - abstract_score) + score_quality * anisotropy
            if isinstance(self.temperature, np.ndarray):
                self.temperature = np.clip(np.mean(self.temperature), 1e-5, 5.0)

            try:
                calibrated[i, mlp_target] = min(calibrated[i, mlp_target] * (1.5 * (1.0 - abstract_score)), 0.95)
            except:
                return calibrated

            calibrated[i] /= calibrated[i].sum()
        return calibrated
    
    def _softmax(self, x):
        temp = self.temperature

        try:
            if len(x.shape) > 1:
                x_dip = x / temp
                exp_x = np.exp(x_dip - np.max(x_dip, axis=1, keepdims=True))
                softmax = exp_x / np.sum(exp_x, axis=1, keepdims=True)     
            else:
                x_dip = x / temp
                exp_x = np.exp(x_dip - np.max(x_dip))
                softmax = exp_x / np.sum(exp_x)
        except:
            x_dip = x / temp
            exp_x = np.exp(x_dip - np.max(x_dip, axis=1, keepdims=True))


        softmax = exp_x / np.sum(exp_x, axis=1, keepdims=True)     
        return softmax


    def validate_writable_path(self, path):
        try:
            path = os.path.expanduser(path)
        
            directory = os.path.dirname(path) or '.'
        
            if not os.path.exists(directory):
                return False, f"Directory does not exist: {directory}"
        
            if not os.access(directory, os.W_OK):
                return False, f"No write permission for directory: {directory}"
        
            if os.path.exists(path):
                if not os.access(path, os.W_OK):
                    return False, f"File exists but is not writable: {path}"
        
            test_file = os.path.join(directory, f".test_write_{os.getpid()}")
            try:
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
            except Exception as e:
                return False, f"Write test failed: {e}"
        
            return True, "Path is writable"
        
        except Exception as e:
            return False, f"Validation error: {e}"

    def safe_pickle_save_with_feedback(self, data, suggested_path):
        print("\n" + "="*50)
        if suggested_path:
            print(f"|| Suggested path: {suggested_path}")
        
        user_path = input("|| Enter path to save pickle file (or 'cancel' to skip): ").strip()
        
        if user_path.lower() == 'cancel':
            print("|| Save cancelled.")
            pass
        
        user_path = user_path.strip('"').strip("'")
        user_path = os.path.expanduser(user_path)
        
        if os.path.isdir(user_path):
            from datetime import datetime
            default_filename = f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            user_path = os.path.join(user_path, default_filename)
            print(f"|| Using filename: {default_filename}")
        
        valid, message = self.validate_writable_path(user_path)
        
        if valid:
            try:
                os.makedirs(os.path.dirname(user_path), exist_ok=True)
                
                with open(user_path, 'wb') as f:
                    joblib.dump(data, f)
                
                print(f"✓ Successfully saved to: {user_path}")
                pass
                
            except PermissionError as e:
                print(f"✗ Permission denied: {e}")
                print("|| Try a different location (like your Desktop or Documents folder)")
                pass
            except Exception as e:
                print(f"✗ Save failed: {e}")
                pass

        else:
            print(f"✗ Invalid path: {message}")
            print("Tips:")
            print("  - Use a path in your home directory: ~/Documents/myfile.pkl")
            print("  - Make sure the directory exists and is writable")
            print("  - Try saving to Desktop or Documents folder")
            pass


    def utility_MLP_set(self, X, y):
        print('🚀 Training MLP Pipeline: ')
        self.mlp.train(X, y, epochs=1000, lr=0.1)

        try:
            joblib.dump(self.mlp, 'analyzer_model.pkl')
            joblib.dump(self.model2, 'analyzer_agent.pkl')
            print("🎉 Model trained and saved!")            
        except Exception as e:
            print(f'|| Failed to joblib dump file! : {e}, User Manual filepath suggestion needed...')

            permission = input('|| Insert Filepath? [Y/N]: ')
            if permission == 'Y':
                suggested_path = input('|| Filepath suggestion: ')
                if suggested_path:
                    self.safe_pickle_save_with_feedback(self.mlp, suggested_path)
                    self.safe_pickle_save_with_feedback(self.model2, suggested_path)
                    print("🎉 Model trained and saved!")                     
                else:
                    print('|| Failed to dump Your model! ')
                    pass
            else:
                print('|| Failed to dump Your model! ')
                pass

        print("🎉 Model trained!")


    def auto_generate_labels_from_texts(self, rules, texts):
        import re
        y_raw = []
        self.rules = rules
     
        for text in texts:
            text_lower = text.lower()
            matched = False
            for pattern, label in rules:
                if re.search(pattern, text_lower):
                    y_raw.append(label)
                    matched = True
                    break
        
            if not matched:
                y_raw.append('other')
    
        from collections import Counter
        print("\n📊 Auto-generated label distribution:")
        for label, count in Counter(y_raw).items():
            print(f"   {label}: {count} ({count/len(texts)*100:.1f}%)")
    
        return y_raw



    def mlp_training_features(self, rules, dataset):
        print("\n🔄 Preparing MLP data from dataset format")
    
        if isinstance(dataset[0], tuple) and len(dataset[0]) == 2:
            # Format: [(features, label), ...]
            features_list = []
            labels_list = []
            print('Dataset Type 1: [(value), (value)]')

            for item in dataset:
                features, label = item
                features_list.append(features)
                labels_list.append(label)
        
            X_mlp = np.array(features_list)
            y_raw = np.array(labels_list)
        
        elif isinstance(dataset[0], (list, np.ndarray)) and len(dataset[0]) > 1:
            print('Dataset Type 2')
            X_mlp = np.array([item[:-1] for item in dataset])
            y_raw = np.array([item[-1] for item in dataset])

        else: 
            print('Dataset type 3')     
            X_mlp = dataset.copy()   
            y_raw = self.auto_generate_labels_from_texts(rules, dataset)         

        unique_labels = sorted(set(y_raw))
        label_to_idx = {l: i for i, l in enumerate(unique_labels)}
        y_indices = np.array([label_to_idx[l] for l in y_raw])
    
        n_classes = len(unique_labels)
        y_onehot = np.zeros((len(y_indices), n_classes))
        y_onehot[np.arange(len(y_indices)), y_indices] = 1

        if isinstance(X_mlp, np.ndarray):
            input_dim = X_mlp.shape[0]           
        else:
            input_dim = len(X_mlp)

        print(f"\n✅ MLP data ready:")
        print(f"   X shape: {input_dim}")
        print(f"   y shape: {y_onehot.shape}")
        print(f"   Classes: {label_to_idx}")     
        return X_mlp, y_onehot, n_classes, input_dim        

    def shape_adaptation(self, X, inp):
        tuple_ver = (inp, inp)
        if X.shape != tuple_ver:
            X = X[:inp, :inp]

        return X

    def AME_Encoder(self, x):
        # function that  handles abstraction modelling rate to predict model prediction capabilities.
        X = np.asarray(x)

        if isinstance(X, (str, np.str_)):
            clean_str = str(X).replace('[', '').replace(']', '')
            X = np.fromstring(clean_str, sep=' ')
        if isinstance(X, np.ndarray) and np.issubdtype(X.dtype, np.character):
            # catches arrays filled with string text
            clean_str = ' '.join(X.astype(str).flatten()).replace('[', '').replace(']', '')
            X = np.fromiter(
                    (x for x in clean_str.split() if x != "..."), dtype=float
                )               
        X = np.squeeze(X)
        if X.ndim == 1:
            X = np.atleast_2d(X).T  # Returns shape back to (12, 1) safely

        # Handle cropping for a 2D array
        if X.ndim == 2:
            rows, cols = X.shape
            new_rows = rows - 1 if (rows % 2 != 0 and rows > 1) else rows
            new_cols = cols - 1 if (cols % 2 != 0 and cols > 1) else cols
            
            X = X[:new_rows, :new_cols]

        if X.shape[1] == 1:
            gradient = np.gradient(X, axis=0)  # Calculate vertically instead of horizontally
        else:
            gradient = np.gradient(X, axis=-1) # Calculate horizontally
        grad_energy = np.mean(np.linalg.norm(gradient, axis=-1))       
        X_mag = np.mean(np.linalg.norm(X, axis=-1))

        AME = np.log1p(X_mag) * np.log1p(grad_energy) 

        if AME == 0.0:
            eps = 1e-5
            AME = AME + eps

        return AME

    def feature_generation(self, rules, dataset):
        X_raw, y, n_classes, input_dim = self.mlp_training_features(rules, dataset)
            
        self.initialize_fitting(X_raw)            
        X_tfidf = self.tfidf.transform(X_raw).toarray()
        X = X_tfidf.copy() 

        X = self.shape_adaptation(X, input_dim)  

        return X, y, input_dim, n_classes       


    # necessary functions to reduce wasteful training in similar scarce environment
    def training_necessary_condition(self, input_ids, x):
        eps = 1e-5
        final_conf = self.final_conf_score
        confidence_threshold = self.confidence_threshold
        unsuitable_training = False

        probs = self.model_memory_gate(input_ids, x)

        anisotropy = self.anisotropy_measurement(input_ids)
        AME = self.AME_Encoder(input_ids)
        AMR = 1.0 / (1.0 + np.exp(-AME)) 

        LMR = 1.0 / (1.0 + np.exp(-AMR)) # logistic modelling rate
        ALR = 1.0 / (1.0 + np.exp(-anisotropy)) # anisotropic logistic rate
        AAC = (1.0 - ALR) + (1.0 - LMR) + eps # anisotropic abstract coefficient

        self.confidence_threshold = anisotropy * AAC 
        if np.isnan(self.confidence_threshold) or np.isinf(self.confidence_threshold):
            self.confidence_threshold = 0.45

        print(f'[||] Confidence threshold set to: {self.confidence_threshold} || Final Confidence Score: {final_conf}')

        # training is wasteful if the model processes little abstraction divergence without necessary context on a similar environment
        # AMR is guaranteed to give sufficient ratio on how modelling error error could be sufficient enough to guarantee the model successful training 
        # (not too high that it shows unstability, not too low that it shows rigidity), high anisotropy correlates to a much complex non linearity that the model will have a hard time adjusting
        # higher LMR than AAC means the model is likely to be in a regime where training could lead to overfitting or divergence due to insufficient modelling capacity relative to the complexity of the data, especially if the confidence score is also low, indicating that the model is not currently confident in its predictions and may not benefit from further training on this data.
        unsuitable_tolerance = probs is not None or LMR > AAC 
        unsuitable_conditions = anisotropy > 0.85 or final_conf > confidence_threshold or self.froze_learning
        unsuitable_peer_request = probs is not None and self.peer_assistance_threshold > self.confidence_threshold

        if unsuitable_tolerance or unsuitable_conditions or unsuitable_peer_request:
            print(f'[==] Unsuitable training condition detected! Tolerance: {unsuitable_tolerance} || Unsuitable Conditions: {unsuitable_conditions}')
            print(f'[==] Peer assistance condition: {unsuitable_peer_request} || Peer assistance threshold: {self.peer_assistance_threshold}')
            unsuitable_training = True

        print('== Training Condition evaluation == ')
        print(f'[==] Unsuitable training condition Evaluation || Unsuitable Tolerance: {unsuitable_tolerance} || Unsuitable Conditions: {unsuitable_conditions} || Unsuitable Peer Assistance: {unsuitable_peer_request}')
        print('[==] Final Decision on Training: ' + ('Unsuitable' if unsuitable_training else 'Suitable') + ' for training.')

        return unsuitable_training

    def sequence_encoding(self, datasets, max_len=32):
        input_sequences = []
        for item in datasets:
            if not self.model2:
                intents = [d[1] for d in datasets]
                intent_to_id = {intent:i for i, intent in enumerate(sorted(set(intents)))}
                num_classes = len(intent_to_id)                
                self.model2 = Transformer(
                    vocab_size=len(self.vocab),
                    d_model=32,
                    n_heads=4,
                    num_classes=num_classes
                ) 

            text = item[0] if isinstance(item, tuple) else item
            token_ids = self.encode(text, self.vocab, max_len=max_len)
            token_embs = self.model2.token_embedding[token_ids]         # (max_len, d_model)
            pos_embs = self.model2.pos_embedding[:max_len]              # (max_len, d_model)
            sequence_input = token_embs + pos_embs
            input_sequences.append(sequence_input)
        return np.stack(input_sequences)  # shape: (batch, max_len, d_model)

    def transformer_pooled_features(self, sequence_inputs):
        # mean/max/std pooling over sequence dimension
        mean_pool = np.mean(sequence_inputs, axis=1)
        max_pool = np.max(sequence_inputs, axis=1)
        std_pool = np.std(sequence_inputs, axis=1)
        return np.concatenate([mean_pool, max_pool, std_pool], axis=-1)


    def _set_lstm_samples(self, X, Y):
        X = np.array(X)[..., np.newaxis]
        Y = np.array(Y)[..., np.newaxis]

        print('[=] Successfully set up LSTM Samples:')
        print(f'[=] X.shape: {X.shape}')
        print(f'[=] Y.shape: {Y.shape}')

        return X, Y

    def lstm_setup_inference(self, raw_X, raw_Y):
        print("\n" + "=" * 55)
        print("===== LSTM SETUP INFERENCE =====")
        print('[=] LSTM Setup is initiated for Longer short term memory.')

        scaler_y = self.standard_scaler 
        scrapper_engine = self.scrapper_model

        # build dataset for calibration
        AME = self.AME_Encoder(raw_X)  # geometric complexity scalar
        AMR = 1.0 / (1.0 + np.exp(-AME))  # abstract modelling rate

        X, Y = self._set_lstm_samples(raw_X, raw_Y)
        n_train = int(0.8 * len(X)) # 80% of the data training is used for training
        X_val   = X[n_train:]
        Y_val   = Y[n_train:]
        n_display = len(X_val)
        if n_display > 3:
            n_display = 3

        print('[= FIT =] Fitting Short Term Memory...')
        n_samples = scrapper_engine.lstm_optimal_samples(scrapper_engine, X_val[0])
        self.lstm_n_samples = n_samples

        label_bins = scrapper_engine.derive_bins_from_data(
            Y_val.ravel(),
            n_bins=4,
            labels=["Low", "Moderate", "High", "Extreme"]
        )
        # {"Low": (5.0, 42.3), "Moderate": (42.3, 68.1),
        #  "High": (68.1, 98.7), "Extreme": (98.7, 219.0)}

        # build and calibrate engine
        self.lstm_engine = LSTMEngine(self, self.network_model, 
                                       dropout=self.dropout_rate, n_samples=self.lstm_n_samples)
        
        engine = self.lstm_engine

        self.lstm_engine.fit_stm(X, Y)
        engine.calibrate(X_val, Y_val)

        print("\n[= LSTM INSIGHT =] Per-sample confidence report:")
        print(f"[*] {'#':<4} {'Predicted':>10} {'Actual':>10} "
            f"[*] {'Confidence':>12} {'Gate Uncert':>13} "
            f"[*] {'90% Interval':>20}  Label Confidence")
        print("  " + "─" * 100)

        # n display to see the prediction result over specific batch
        for j in range(n_display):
            result = engine.predict(X_val[j], label_bins=label_bins)

            p      = result["prediction"][-1]
            actual = Y_val[j, -1, 0]
            conf   = result["overall"]
            gate_u = result["gate_uncertainty"][-1]
            lo     = result["interval_low"][-1]
            hi     = result["interval_high"][-1]
            lc     = result["label_confidence"]

            best_label = max(lc, key=lc.get)
            label_str  = "  ".join(f"{k}={v:.0%}" for k,v in lc.items())

            print(f"  {j:<4} {p:>+10.4f} {actual:>+10.4f} "
                f"{conf:>12.1%} {gate_u:>13.3f} "
                f"  [{lo:+.3f}, {hi:+.3f}]"
                f"  {label_str}")

        print('[=+=] ==== STATUS REPORT ====')
        print("\n[=] Confidence breakdown for sample 0:")
        r = engine.predict(X_val[0], label_bins=label_bins)
        print(f"    MC mean (last step)   : {r['mc_mean'][-1]:+.4f}")
        print(f"    MC std  (last step)   : {r['mc_std'][-1]:.4f}  "
            f"← tight = certain")
        print(f"    Gate uncertainty      : {r['gate_uncertainty'][-1]:.4f}  "
            f"← low = stable memory")
        print(f"    MC confidence         : {r['mc_confidence'][-1]:.1%}")
        print(f"    Overall confidence    : {r['overall']:.1%}")
        print(f"    90% interval          : "
            f"[{r['interval_low'][-1]:+.4f}, {r['interval_high'][-1]:+.4f}]")
        print(f"    Label confidence      : {r['label_confidence']}")

        self.cache['lstm_result'] = result
        self.cache['label_bins'] = label_bins


    def transformer_utilities(self, rules, datasets, X_raw, y_true=None, batch_size=2, min_signal=1e-3):
        self.text_encoder(datasets)
        if y_true is None:
            _, y_true = self.input_encoding(datasets)

        sequence_inputs = self.sequence_encoding(datasets)
        unsuitable_training = self.training_necessary_condition(sequence_inputs, X_raw)
        lr = self.model2.transformer_lr if self.model2 else self.transformer_lr

        if not unsuitable_training:
            print(f'🚀 Training Transformer with {len(sequence_inputs)} Samples: ')
            conditional_anisotropy = self.anisotropy_measurement(sequence_inputs)
            if conditional_anisotropy >= self.confidence_threshold: 
                print('[+] Dynamic Backward')
                mode = 'dynamic_backward'
            else:
                print('[-] Fixed Backward')
                mode = 'fixed_backward'

            if self.use_transformer:
                self.model2.train(sequence_inputs, y_true, epochs=100, mode=mode, lr=lr, embedded=True, batch_size=batch_size)

            X_raw_generation, y, n_classes, input_dim = self.mlp_training_features(rules, datasets)
            X_raw_features = self.tfidf.transform(X_raw_generation).toarray()
            X_raw_features = self._refit_sparse_data(X_raw_features, X_raw_generation)      
            
            row_sums = X_raw_features.sum(axis=1)
            weak_rows = np.where(row_sums < min_signal)[0]
            weak_ratio = len(weak_rows) / len(X_raw_features)

            print(f'[!] Zero ratio in samples: {weak_ratio * 100}')
            if weak_ratio > 0.3:  # more than 30% zero rows means vocab mismatch
                print(f'[= ! =] High zero-row ratio ({weak_ratio:.0%}), refitting TF-IDF on current batch')
                self.tfidf.fit(X_raw_generation)
                X_raw_features = self.tfidf.transform(X_raw_generation).toarray()  

            transformer_features = self.transformer_pooled_features(sequence_inputs)
            X_raw_features = np.concatenate([X_raw_features, transformer_features], axis=-1)
            
            zero_rows = np.where(X_raw_features.sum(axis=1) == 0)[0]
            if len(zero_rows) > 0:
                print(f'[!] {len(zero_rows)} zero rows detected, applying checksum fallback')
                for i in zero_rows:
                    text = X_raw[i] if isinstance(X_raw[i], str) else str(X_raw[i])
                    checksum = int(hashlib.md5(text.encode()).hexdigest(), 16)
                    # distribute checksum signal across feature dims
                    rng = np.random.default_rng(checksum)
                    X_raw_features[i] = rng.uniform(0.01, 0.1, size=X_raw_features.shape[1])  

            X_features = X_raw_features.copy()
            if isinstance(X_raw_features, list):
                X_features = np.asarray(X_raw_features)
            if isinstance(X_raw, list):
                X_raw = np.asarray(X_raw) 

            # hybrid features by dot product of raw features and extracted features from transformer, this allows the MLP to learn from both the original feature space and the transformer-extracted feature space
            # potentially improving its ability to capture complex patterns in the given first data   
            try:
                if len(X_features.shape) < 2: 
                    X_features = X_features.reshape(1, -1)
                if len(X_raw.shape) < 2:
                    X_raw = X_raw.reshape(1, -1)
                                    
                X_raw = X_raw[:X_features.shape[0], :X_features.shape[1]]

                hybrid_X = np.dot(X_raw, X_features.T)
            except:
                if len(X_features.shape) < 2: 
                    X_features = X_features.reshape(1, -1)
                if len(X_raw.shape) < 2:
                    X_raw = X_raw.reshape(1, -1)

                subnet_X_feature = X_features[:X_raw.shape[1], :X_raw.shape[0]]
                subnet_X_raw = X_raw[:, :subnet_X_feature.shape[0]]
                hybrid_X = np.dot(subnet_X_raw, subnet_X_feature)
            
            hybrid_X = np.concatenate([X_raw, X_features, hybrid_X], axis=-1)
            self.initialize_fitting(X_raw_generation)            
            X = self.shape_adaptation(hybrid_X, input_dim)
                
            self.lstm_setup_inference(X, y) 
            self.initialize_model_(X, input_dim, n_classes)
            self.model3.train(X, y, epochs=1000, lr=0.1)

            if self.lstm_engine:
                self.storage.save_weights(self.memory_name, model_type='Pipeline') 

            print('🎉 All Model Trained!')
        else:
            print(f'[=] No suitable condition for training!')
            print('[=] Saving Weights for prediction')
            self.storage.load_weights(self.memory_name)

            pass

    def transformer_input_encoding(self, titles):
        if hasattr(self, 'vocab') and self.vocab:
            print("🔄 Using Transformer for probability calibration")
            input_ids_list = []
            for title in titles:
                if isinstance(title, tuple):
                    title = title[0]
                    
                ids = self.encode(title, self.vocab)
                input_ids_list.append(np.array(ids))
                
                input_ids = np.array(input_ids_list)
                return input_ids

        else:
            print('[-] Cant get sufficient data!')
            return []


    def train(self, X, y_raw):
        self.initialize_fitting(X)            
        X_tfidf = self.tfidf.transform(X).toarray()
        self.X = X_tfidf.copy()

        print(f"\n🚀 Separate Modular MLP Pipeline:")
        print(f" Samples: {len(self.X)}")

        y_true = self.initialize_model_encoding(self.X, y_raw)
        self.utility_MLP_set(self.X, y_true)       
        print('✅ Done Training MLP Model! ')
         
