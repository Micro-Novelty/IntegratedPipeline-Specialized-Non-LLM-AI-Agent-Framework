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
# inference.py
# QueryNode       — single-request orchestrator; drives MLP + Transformer inference,
#                   resolves ensemble predictions, and hits the memory cache.
# AutoBatcherAutomation — collects individual requests into mini-batches and
#                   dispatches them to IntegratedPipeline.
# AgentDistributedInference — multi-agent peer network; manages trust levels,
#                   encrypted sockets, and distributed prediction consensus.
# Depends on: primitives (TrustLevel), messaging (AsyncMessageQueue, Message),
#             geometry, mlp, transformer, ensemble, storage
# IntegratedPipeline is imported lazily inside AutoBatcherAutomation to avoid
# a circular import (pipeline.py imports QueryNode / AutoBatcherAutomation).
# ---------------------------------------------------------------------------
from .primitives import TrustLevel
from .messaging import AsyncMessageQueue, Message
from .geometry import GeometricWeightShaping

class QueryNode:
    def __init__(self, pipeline, memory_name, storage):
        self.master_node = pipeline
        self.memory_name = memory_name
        self.storage = storage
        self.agreement = False

        if not self.storage.memory_exists(self.memory_name, type='Node'):
            print(f"|| Creating new memory for Nodes population: {memory_name}!")
            self.nodes = {}
        else:
            print(f'|| Found Matched Memory for Nodes : {memory_name}!')
            self.nodes = self.storage.memory_retrieval(self.memory_name, type_func='Node', verbose=True)

        self.master_nodes_id = 0
        self.safety_check_value = 0.0
        self.node_id = 0
        self.peer_trust = 1.0

        self.permission = False

    def _add_node(self, node):
        node_id = id(node)

        self.nodes[node_id] = node
        print(f"✅ Node {node_id} added to QueryNode")

        return node_id

    def _save_node_memory(self, node):
        try:
            node_id = id(node)
            self.master_node.storage.save_nodes_dict(self.memory_name, self.nodes, node_id, model_type='Node')

            print(f"[💾] Node {node_id} memory saved to storage!")
            return True
        except Exception as e:
            print(f"[-] Error saving node memory: {e}")
            return False


    def _evaluate_node_agreement(self, node):
        print(f"[=] Evaluating node {id(node)} || agreement: {self.agreement} || Master Node memory: {self.memory_name}")
        self.agreement_threshold = (self.master_node.confidence_threshold + self.master_node.final_conf_score * self.master_node.temperature)

        if self.agreement or self.agreement_threshold > self.master_node.confidence_threshold:
            print(f"[✅] Node {id(node)} is in agreement with the Master node")
            return True

        print(f"[-] Node {id(node)} is NOT in agreement with the Master node")
        return False


    def _connect_with_node(self, node):
        self.agreement = self.master_node.agreement

        if not self._identify_node(node):
            node_id = self._add_node(node)

        node_id = id(node)
        agreement = self._evaluate_node_agreement(node)
        safety = self._node_safety_check(node)

        # stable Node is established if either agreement is met or safety check is passed, allowing for some flexibility in interactions while still protecting the Master node from harmful interactions
        if safety or agreement:
            print(f"[🔗] Node {node_id} successfully connected to the Master node")
            self.permission = True
        else:
            print(f"[⚠️] Node {node_id} connection failed due to Disagreement")
            self.permission = False

        print('== Connection Evaluation Summary ==')
        print(f'[=] Node {node_id}')
        print(f'[=] agreement: {agreement}')
        print(f'[=] safety: {safety} || permission: {self.permission}')
 
        return self.permission

    def _connect_with_peer(self, node):
        self.agreement = self.master_node.agreement

        if not self._identify_node(node):
            node_id = self._add_node(node)

        node_id = id(node)
        agreement = self._evaluate_node_agreement(node)
        safety = self._node_safety_check(node)

        # peer agreement is optional to allow for more flexible interactions, but safety check is still enforced to protect the Master node from harmful interactions
        if safety or agreement:
            print(f"[🔗] Peer with ID: {node_id} successfully connected to the Master node")
            self.permission = True
        else:
            print(f"[⚠️] Peer with ID: {node_id} connection failed due to Disagreement")
            self.permission = False
 
        return self.permission        

    def _identify_node(self, node):
        eps = 1e-5
        print(f"[||] Identifying node {id(node)} with Master node memory: {self.memory_name}")
        identified_nodes = [(nid, n) for nid, n in self.nodes.items() if n == node]
        if identified_nodes:
            for node in identified_nodes:
                print(f"✅ Node {id(node)} is already identified with the Master node")
                self.safety_check_value = (self.master_node.final_conf_score + self.master_node.temperature) + eps
                return True
        else:
            print(f"[-] Node {id(node)} is NOT identified with the Master node")
            self.safety_check_value = (1.0 - self.master_node.final_conf_score + self.master_node.temperature) + eps
            return False


    def _node_safety_check(self, node):
        print(f"[🛡️] Performing safety check for node {id(node)} with safety value: {self.safety_check_value}")
        if self.safety_check_value > self.master_node.confidence_threshold:
            print(f"✅ Node {id(node)} passed the safety check")
            return True
        else:
            print(f"[-] Node {id(node)} failed the safety check")
            if self.safety_check_value < (self.master_node.confidence_threshold / 2):
                print(f"[⚠️] Node {id(node)} is considered useless and will be removed")
                removed = self._remove_node(node)
                return removed

            return False

    def _remove_node(self, node):
        node_id = id(node)
        if node in self.nodes:
            del self.nodes[node_id]
            print(f"[🗑️] Node {node_id} removed from Nodes population")
            return True
        else:
            print(f"[-] Node {node_id} not found in Nodes population")
            return False

    def _node_activation(self, Node):
        try:
            if self.permission:
                print(f"🚀 Node {id(Node)} is now active with the Master node")
                return True
            else:
                print(f"[-] Node {id(Node)} cannot be activated due to lack of permission")
                return False
        except Exception as e:
            print(f"[-] Error during node activation: {e}")
            return False

    def _identify_peer_trust(self, peer):
        print(f"[=] Identifying peer node {id(peer)} trustworthiness with Master node memory: {self.memory_name}")
        if self.peer_trust > self.master_node.confidence_threshold:
            print(f"[✅] Peer node {id(peer)} is identified as trustworthy with trust score: {self.peer_trust:.2f}")
            return True
        else:
            print(f"[-] Peer node {id(peer)} is NOT identified as trustworthy with trust score: {self.peer_trust:.2f}")
            return False

    def _establish_peer_nodes(self, peer):
        print(f"[=] Establishing peer node connection with peer with memory: {self.memory_name}")
        if self._connect_with_peer(peer) and self._identify_peer_trust(peer):
            print(f"[✅] Peer node {id(peer)} is now connected and can interact with the Master node")
        else:
            print(f"[-] Peer node {id(peer)} cannot interact with the Master node due to failed agreement")

        activation = self._node_activation(peer)
        saved = self._save_node_memory(peer)
        return activation

    def _establish_node_connection(self, node):
        if self._connect_with_node(node):
            print(f"[✅] Node {id(node)} is now connected and can interact with the Master node")
        else:
            print(f"[-] Node {id(node)} cannot interact with the Master node due to failed agreement")

        activation = self._node_activation(node)
        saved = self._save_node_memory(node)
        return activation




# The AutoBatcherAutomation class manages the batching of incoming prediction requests to optimize processing efficiency. 
# It collects requests over a short time window or until a maximum batch size is reached, then processes them together through the pipeline. This allows for improved throughput while still providing timely responses to individual requests.


class AutoBatcherAutomation:
    def __init__(self, pipeline, max_batch_size=32, max_wait_ms=50):
        self.pipeline = pipeline
        self.dataset = None
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        
        self.request_queue = deque()
        self.processing = False
        self.results = {}
        self.next_id = 0
    
    def add_request(self, text, callback=None):
        request_id = self.next_id
        self.next_id += 1
        
        self.request_queue.append({
            'id': request_id,
            'text': text,
            'callback': callback,
            'timestamp': time.time()
        })
        
        if not self.processing:
            self._start_processing()
        
        return request_id
    
    def _start_processing(self):
        self.processing = True
        thread = threading.Thread(target=self._process_batches, daemon=True)
        thread.start()
    
    def _process_batches(self):
        while self.request_queue:
            # Wait for more requests or max wait time
            time.sleep(self.max_wait_ms / 1000)
            dataset = self.dataset
            
            # Collect batch
            batch = []
            while self.request_queue and len(batch) < self.max_batch_size:
                batch.append(self.request_queue.popleft())
            
            if batch:
                self._process_batch(batch)
        
        self.processing = False
    
    def _process_batch(self, batch):
        texts = [req['text'] for req in batch]
        
        results = self.pipeline.prediction_batch(texts)
        
        # Send results back
        for i, req in enumerate(batch):
            result = results[i] if i < len(results) else None
            if req['callback']:
                req['callback'](result)
            else:
                self.results[req['id']] = result
    
    def get_result(self, request_id, timeout=5):
        start = time.time()
        while request_id not in self.results:
            if time.time() - start > timeout:
                return None
            time.sleep(0.01)
        return self.results.pop(request_id)


# The IntegratedPipeline class serves as the central component that integrates all the different modules and functionalities of the system. 
# It manages the overall workflow, including data processing, model training, prediction, memory management, and interactions with other agents.


class AgentDistributedInference:
    def __init__(self, pipeline, storage, memory_name, port=5555, use_async=False, secret_key=None, ssl_cert_file=None, ssl_key_file=None, shared_auth_token=None, predict_manager=None):
        super().__init__()
        
        # Only initialized once
        if hasattr(self, '_singleton_initialized'):
            print(f"[===] AgentDistributedInference already initialized, reusing...")
            return
        
        self._singleton_initialized = True
        
        # Stored initialization params for debugging later
        self._init_params = {
            'memory_name': memory_name,
            'port': port,
            'use_async': use_async,
            'secret_key': secret_key,
            'ssl_cert_file': ssl_cert_file,
            'ssl_key_file': ssl_key_file,
            'shared_auth_token': shared_auth_token
        }  

        self.pipeline = pipeline
        self.memory_name = memory_name
        self.port = port
        self.storage = storage

        self.query_node = QueryNode(pipeline, memory_name, self.storage)        
        
        self.agent_comm_log = {}
        self.connections_log = {}
        self.connections = []  # List of connected sockets
        self.remote_agents = {}  # {agent_id: {'sock': sock, 'host': host, 'port': port, 'trust': 1.0}}
        
        self.running = False
        self.socket = None
        self.temporary_message = None
        self.temporary_agent_id = None  
        self.established_connections = set()  # Track established connections to prevent duplicates      

        self.next_agent_id = 1
        self.connection_timeout = 15

        # for security purposes
        # Security: Authentication token
        self.auth_token = shared_auth_token
        self.secret_key = shared_auth_token 

        # Security: Rate limiting
        self.max_connections_per_minute = 20
        self.connection_timestamps = deque(maxlen=20)
        self.max_requests_per_minute = 40
        self.request_timestamps = defaultdict(lambda: deque(maxlen=40))
        self.secret_key = secret_key

        # Security: Message validation
        self.max_message_size = 10 * 1024 * 1024  # 10MB limit

        # Security: Trusted agents
        self.trusted_agents = {}

        # Security: Audit log
        self.security_log = []        

        self.enable_ssl = True  # Set to True to enable SSL encryption
        # i provided basic cert file and key since there are other layered security other than ssl, and also due to infrequent external connections.
        self.ssl_cert_file = ssl_cert_file
        self.ssl_key_file = ssl_key_file
        self.ssl_context = None

        if self.enable_ssl:
            self._setup_ssl()

        self.allowed_ips = set()  # Add trusted IPs
        self.blocked_ips = set()  # Block malicious IPs

        # Message types
        self.MSG_TYPES = {
            'PREDICT_REQUEST': 1,
            'PREDICT_RESPONSE': 2,
            'MEMORY_SYNC_REQUEST': 3,
            'MEMORY_SYNC_RESPONSE': 4,
            'ENSEMBLE_VOTE_REQUEST': 5,
            'ENSEMBLE_VOTE_RESPONSE': 6,
            'FAILURE_REPORT': 7,
            'TRUST_UPDATE': 8,
            'AGENT_INFO': 9,
            'PING': 10,
            'PONG': 11,
            'DISCONNECT': 12
        }
        
        # message queue
        self.max_retries = 3
        self.retry_delay = 1.0
        self.message_timeout = 30.0 
        self.CHUNK_SIZE = 8192
        self.predict_manager = predict_manager

        self.use_async = use_async
        
        # Register message handlers
        print('[=++=] Initiating message Queue')
        self.message_queue = AsyncMessageQueue()
            
        self.message_queue.register_handler('predict_request', self._handle_predict_request_async)
        self.message_queue.register_handler('memory_sync', self._handle_memory_sync_async)
        self.message_queue.register_handler('ensemble_vote', self._handle_ensemble_vote_async)
        self.message_queue.register_handler('ping', self._handle_ping)
        self.message_queue.register_handler('status', self._handle_status)
                
           
        # Queue for outgoing messages (buffered with retry)
        self.outgoing_queue = deque()
        self.queue_processor_thread = None
        self._health_check_interval = 30  # seconds
        self._last_health_check = time.time()
      
        # Start health checker if async
        if use_async:
            self._start_health_checker()        
            
        # Trust configuration
        self.min_trust_level_for_auto_add = TrustLevel.STANDARD
        self.trusted_agents = {}  # agent_id -> {'token': token, 'trust_level': TrustLevel, 'added_at': datetime} 
        self.highly_trusted_peer = []
        self.socket_owners = {}
        
        self.pending_requests = {}  # request_id -> Future
        self.request_lock = threading.Lock()        
     

    # ============ SECURITY FEATURES ============

    def _check_ip_access(self, ip):
        print(f'|| Checking IP access for: {ip}')
        if ip in self.blocked_ips:
            return False
        if self.allowed_ips and ip not in self.allowed_ips:
            return False
        return True

    def add_allowed_ip(self, ip):
        self.allowed_ips.add(ip)
        self._log_security_event('ip_allowed', {'ip': ip})

    def remove_allowed_ip(self, ip):
        self.allowed_ips.discard(ip)
        self._log_security_event('ip_removed_from_allow', {'ip': ip})

    def add_blocked_ip(self, ip):
        self.blocked_ips.add(ip)
        self._log_security_event('ip_blocked', {'ip': ip})

    def remove_blocked_ip(self, ip):
        self.blocked_ips.discard(ip)
        self._log_security_event('ip_removed_from_block', {'ip': ip})

    def _setup_ssl(self):
        # Setup SSL context for encrypted connections
        try:
            self.ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)      
            if self.ssl_cert_file and self.ssl_key_file:
                self.ssl_context.load_cert_chain(self.ssl_cert_file, self.ssl_key_file)
            else:
                # Generate self-signed certificate for first layer security
                self._generate_self_signed_cert()
                
        except Exception as e:
            print(f"SSL setup failed: {e}")
            self.enable_ssl = False

    def _generate_self_signed_cert(self):
        # Generate self-signed certificate
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        import datetime
        
        # Generate private key
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        
        # Creatingcertificate
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, u"localhost"),
        ])
        
        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            issuer
        ).public_key(
            private_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.datetime.utcnow()
        ).not_valid_after(
            datetime.datetime.utcnow() + datetime.timedelta(days=365)
        ).add_extension(
            x509.SubjectAlternativeName([x509.DNSName(u"localhost")]),
            critical=False,
        ).sign(private_key, hashes.SHA256())
        
        # Saving certificate and key
        cert_pem = cert.public_bytes(serialization.Encoding.PEM)
        key_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        with open('server.crt', 'wb') as f:
            f.write(cert_pem)
        with open('server.key', 'wb') as f:
            f.write(key_pem)
            
        self.ssl_cert_file = 'server.crt'
        self.ssl_key_file = 'server.key'
        self.ssl_context.load_cert_chain(self.ssl_cert_file, self.ssl_key_file)


    def _generate_auth_token(self):
        return hashlib.sha256(os.urandom(32)).hexdigest()

    def _generate_secret_key(self):
        return hashlib.sha256(os.urandom(48)).hexdigest()

    def _log_security_event(self, event_type, details):
        self.security_log.append({
            'timestamp': datetime.now().isoformat(),
            'event': event_type,
            'details': details
        })
        if len(self.security_log) > 1000:
            self.security_log = self.security_log[-1000:]

    def _sanitize_input(self, text, amount=1000):
        if not isinstance(text, str):
            return str(text)
        sanitized = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
        return sanitized[:amount]

    def _sanitize_arrays_and_dicts(self, data, amount=1000):
        if isinstance(data, list):
            return [self._sanitize_input(item, amount) for item in data]
        elif isinstance(data, dict):
            return {key: self._sanitize_input(value, amount) for key, value in data.items()}
        else:
            return self._sanitize_input(data, amount)


    def _check_rate_limit(self, agent_id=None):
        start = time.time()
        if not agent_id:
            print('|| No agent ID provided for rate limiting, applying global connection limit.')
            return False

        print(f'|| Checking rate limit for agent: {agent_id}')
        now = time.time()
        self.connection_timestamps = [t for t in self.connection_timestamps if now - t < 10]
        recent_connections = len(self.connection_timestamps)
        if recent_connections > self.max_connections_per_minute:
            self._log_security_event('rate_limit_exceeded', {'type': 'connection', 'agent': agent_id})
            return False
        if agent_id:
            stale = [aid for aid, timestamps in self.request_timestamps.items() if not timestamps or now - timestamps[-1] >= 10]
            for aid in stale:
                del self.request_timestamps[aid]

            self.request_timestamps[agent_id] = [t for t in self.request_timestamps[agent_id] if now - t < 10]
            if time.time() - start > 5:
                print('|| Rate limit check timed out.')
                return False  

            recent_requests = len(self.request_timestamps[agent_id])
            if recent_requests > self.max_requests_per_minute:
                self._log_security_event('rate_limit_exceeded', {'type': 'request', 'agent': agent_id})
                return False
        return True

    def _sign_message(self, message):
        # Create HMAC signature - DOES NOT modify original message
 
        # Created a COPY of the message with timestamp
        signed_message = message.copy()  # ← IMPORTANT: Copy!
        
        # Ensure timestamp is float if present
        if 'timestamp' in signed_message and isinstance(signed_message['timestamp'], str):
            from datetime import datetime
            try:
                dt = datetime.fromisoformat(signed_message['timestamp'].replace('Z', '+00:00'))
                signed_message['timestamp'] = dt.timestamp()
            except:
                signed_message['timestamp'] = time.time()
    
        # Sort keys for consistent serialization
        sorted_message = {k: signed_message[k] for k in sorted(signed_message.keys())}
      
        message_bytes = pickle.dumps(sorted_message, protocol=pickle.HIGHEST_PROTOCOL)
      
        key = self.secret_key.encode() if isinstance(self.secret_key, str) else self.secret_key
        signature = hmac.new(key, message_bytes, hashlib.sha256).hexdigest()

        print(f'|| Signing message with: {len(message)} total of size, with signature: {signature}')  
        logger.info(f"[=] Signing message: {len(message)} with signature: {signature}")
        return signature



    def _verify_signature(self, message, signature):
        # Verify signature - with timestamp in message
        
        # Create a copy without the signature field
        print(f'|| Verifying message signature total: {len(message)} with signature: {signature}')
        temp_msg = {k: v for k, v in message.items() if k != 'signature'}

        if 'timestamp' in temp_msg and isinstance(temp_msg['timestamp'], str):
            try:
                dt = datetime.fromisoformat(temp_msg['timestamp'].replace('Z', '+00:00'))
                temp_msg['timestamp'] = dt.timestamp()
            except:
                temp_msg['timestamp'] = time.time()
          
        # Sort keys for consistent serialization
        sorted_msg = {k: temp_msg[k] for k in sorted(temp_msg.keys())}
          
        message_bytes = pickle.dumps(sorted_msg, protocol=pickle.HIGHEST_PROTOCOL)
         
        key = self.secret_key.encode() if isinstance(self.secret_key, str) else self.secret_key
        expected = hmac.new(key, message_bytes, hashlib.sha256).hexdigest()
        
        result = hmac.compare_digest(expected, signature)

        print(f'[=] Comparing result: {expected} || {signature}')
        print(f'|| Signature verification result: {result}')
        logger.info(f"[-] Signature verification result: {result}")

        return result

   
    def add_trusted_agent(self, agent_id, agent_token):
        if agent_id == 'local':
            print(f"[❌] Cannot add 'local' as trusted agent")
            return

        self.trusted_agents[agent_id] = {'token': agent_token, 'added_at': datetime.now()}
        self._log_security_event('trusted_agent_added', {'agent_id': agent_id})

    def _authenticate_agent(self, token, agent_id):
        print(f'|| Authenticating agent: {agent_id} with token: {token}')
        logger.info(f"[==] Authenticating agent: {agent_id} with token: {token}")

        if agent_id in self.highly_trusted_peer:
            print('[=+=] Agent is authenticated and already verified')
            return True

        elif token == self.auth_token:
            print(f"[=✅=] Agent {agent_id} authenticated with SHARED SECRET (FULL trust)")
            
            # Add to trusted list with FULL trust if not exists
            if agent_id not in self.trusted_agents:
                self._add_trusted_agent(agent_id, token, TrustLevel.FULL, source="shared_secret")
            else:
                # Update trust level if higher
                current_level = self.trusted_agents[agent_id].get('trust_level', TrustLevel.BASIC)
                if TrustLevel.FULL > current_level:
                    self.trusted_agents[agent_id]['trust_level'] = TrustLevel.FULL
                    print(f"[=] Upgraded trust level to FULL")
                    self.highly_trusted_peer.append(agent_id)

        elif agent_id in self.trusted_agents:
            stored_token = self.trusted_agents[agent_id]['token']
            stored_trust = self.trusted_agents[agent_id].get('trust_level', TrustLevel.BASIC)
            
            if stored_token == token:
                print(f"[✅] Agent {agent_id} authenticated with {stored_trust.name} trust")
                self.highly_trusted_peer.append(agent_id)
                return True
            else:
                print(f"[❌] Token mismatch for {agent_id}")
                return False

        else:  
            auto_add_threshold = getattr(self, 'min_trust_level_for_auto_add', TrustLevel.STANDARD)
        
            print(f"[-] Agent {agent_id} not in trusted list")
            print(f"[=/=] Auto-add threshold: {auto_add_threshold.name}")
            
            # Only auto-add if you have high trust in the network
            if auto_add_threshold == TrustLevel.FULL:
                # In high-security mode, don't auto-add
                print(f"[-] Auto-add disabled (requires manual approval)")
                return False
            else:
                # Auto-add with BASIC trust
                print(f"[+] Auto-adding agent {agent_id} with BASIC trust")
                self._add_trusted_agent(agent_id, token, TrustLevel.BASIC, source="auto_discovery")
                return True

            print('[==] Agent is not authenticated! ')
            return False

    def _add_trusted_agent(self, agent_id, token, trust_level=TrustLevel.STANDARD, source="manual"):
        """Add a trusted agent with specified trust level"""
        if agent_id == 'local':
            print(f"[❌] Cannot add 'local' as trusted agent")
            return

        self.trusted_agents[agent_id] = {
            'token': token,
            'trust_level': trust_level,
            'added_at': datetime.now(),
            'added_by': source,
            'last_seen': datetime.now(),
            'successful_connections': 0,
            'failed_connections': 0
        }
        
        self._log_security_event('trusted_agent_added', {
            'agent_id': agent_id,
            'trust_level': trust_level.name,
            'source': source
        })
        
        print(f"✅ Added trusted agent: {agent_id} (trust: {trust_level.name})")

    # ============ SERVER METHODS ============
    def start_server(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(('0.0.0.0', self.port))
        self.socket.listen(5)
        self.running = True
        logger.info(f"[=] Server started on port {self.port} with SSL={'enabled' if self.enable_ssl else 'disabled'}")

        if self.enable_ssl and self.ssl_context:
            self.socket = self.ssl_context.wrap_socket(self.socket, server_side=True)

        print(f"[🤖] Agent listening on port {self.port}")
        
        # Start accepting connections in background
        accept_thread = threading.Thread(target=self._accept_connections, daemon=True)
        accept_thread.start()
        logger.info("[=] Server started and accepting connections...")
        
        return self.socket
    
    def _accept_connections(self):
        while self.running:
            try:
                client, addr = self.socket.accept()
                client.settimeout(self.connection_timeout)
                host = addr[0]
                port = addr[1]

                if host in ['127.0.0.1', 'localhost'] and port == self.port:
                    print(f"[❌] Rejected self-connection from {host}:{port}")
                    client.close()
                    continue
    

                if not self._check_ip_access(host):
                    print(f"[-] Connection attempt from blocked IP: {host}")
                    self._log_security_event('connection_blocked', {'ip': host})
                    client.close()
                    return

                print(f"📡 Connected to agent at {addr}")
                auth_msg = self._receive_message(client)
                if not auth_msg:
                    print(f"[-] No authentication message from {addr}")
                    client.close()
                    continue
                                        

                if not self._authenticate_agent(auth_msg.get('token', ''), f"{addr[0]}:{addr[1]}"):
                    print(f"[-] Authentication failed for agent with address: {addr}")
                    self._log_security_event('authentication_failed', {'agent': f"{addr[0]}:{addr[1]}"})
                    self.report_failure(id(self), 'authentication', reason=f'Failed authentication from {addr}')
                    client.close()
                    return

                # Send agent info to identify
                self._send_agent_info(client)
                
                # Start handler thread
                thread = threading.Thread(target=self._handle_client, args=(client, addr))
                thread.daemon = True
                thread.start()
                
            except Exception as e:
                if self.running:
                    print(f"[-] Accept error: {e}")
                    traceback.print_exc()
                    self.report_failure(id(self), 'processing', reason=f'{e}')
                                        
                break
    
    def _send_agent_info(self, client):
        info = {
            'type': self.MSG_TYPES['AGENT_INFO'],
            'agent_id': id(self),
            'agent_name': self.memory_name,
            'token': self.auth_token,
            'capabilities': ['prediction', 'memory_sync', 'ensemble'],
            'timestamp': time.time()
        }
        self._send_message(client, info)
        print(f"[==] Sent agent info for authentication")
        logger.info("[==] Sent agent info for authentication")


    def stop_server(self):
        self.running = False
        if self.socket:
            self.socket.close()
        
        # Close all connections
        for conn in self.connections:
            try:
                self._send_message(conn, {'type': self.MSG_TYPES['DISCONNECT']})
                conn.close()
            except:
                pass
        
        print("[🛑] Server stopped")
    
    # ============ CLIENT METHODS ============
    def _is_duplicate_connection(self, host, port):
        # Check if this connection attempt is a duplicate in later flow
        for agent_id, info in self.remote_agents.items():
            if info.get('host') == host and info.get('port') == port:
                return True
        return False    


    def connect_to_agent(self, host, port):
        """
        Connect to a peer agent with proper authentication flow.
        """
        if host == 'local':
            print(f"[❌] Cannot connect to 'local'")
            return None 

        if host in ['127.0.0.1', 'localhost', '0.0.0.0']:
            # Check if this is our own port
            if port == self.port or port == 0:
                print(f"[❌] Rejecting self-connection attempt to {host}:{port}")
                return None

        agent_id = f"{host}:{port}"
        print(f'🔗 Attempting to connect to agent: {agent_id}')
        
        # Generate a unique ID for this connection attempt
        connection_id = str(uuid.uuid4())[:8]
        
        try:
            # ========== SECURITY CHECKS ==========
            # Rate limiting 
            if not self._check_rate_limit(agent_id):
                print(f'[❌] Rate limit exceeded for {agent_id}')
                self._log_security_event('rate_limit_exceeded', 
                                        {'type': 'connection_attempt', 'agent': agent_id})
                self.report_failure(agent_id, 'connection_attempt', reason=f'Rate limit exceeded for {agent_id}')

                return None
            
            # IP access check
            if not self._check_ip_access(host):
                print(f"[-] Connection attempt to blocked IP: {host}")
                self._log_security_event('connection_blocked', {'ip': host})
                return None
            
            # ========== CREATE SOCKET ==========
            if self.enable_ssl and self.ssl_context:
                sock = self.ssl_context.wrap_socket(socket.socket(socket.AF_INET, socket.SOCK_STREAM))
            else:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)  # 1MB
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024 * 1024)  # 1MB      
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            sock.settimeout(10)

            print(f"[connect_to_agent() SOCKET CREATED] id={id(sock)}")            
            sock.settimeout(self.connection_timeout)
            print(f'[==] Connecting to {host}:{port}...')
            sock.connect((host, port))
            print(f'[==] Socket connected')
            
            # ========== SEND AUTHENTICATION FIRST ==========
            # Send agent info and token BEFORE receiving
            auth_message = {
                'type': self.MSG_TYPES['AGENT_INFO'],
                'agent_id': id(self),
                'agent_name': getattr(self, 'memory_name', 'unknown'),
                'token': self.auth_token,  # Your authentication token
                'timestamp': time.time()
            }
            
            if not self._send_message(sock, auth_message):
                print(f"[-] Failed to send authentication to {host}:{port}")
                self._log_security_event('authentication_failed', {'agent': agent_id})
                sock.close()
                return None
            
            print(f'[=?=] Authentication sent')
            
            # ========== RECEIVE PEER INFO ==========
            info = self._receive_message(sock)
            
            if not info:
                print(f"[-] No response from {host}:{port}")
                sock.close()
                return None
            
            # Authenticate the peer
            if not self._authenticate_agent(info.get('token', ''), agent_id):
                print(f"[-] Authentication failed for agent {host}:{port}")
                self._log_security_event('authentication_failed', {'agent': agent_id})
                sock.close()
                return None
            
            # ========== ESTABLISH PEER RELATIONSHIP ==========
            if info.get('type') == self.MSG_TYPES['AGENT_INFO']:
                remote_id = info.get('agent_id', agent_id)
                
                query_result = self.query_node._establish_peer_nodes(remote_id)
                
                if not query_result:
                    print(f'[❌] Connection to peer {remote_id} denied by query node.')
                    self.report_failure(id(self), 'peer_establishment', reason=f'Connection to peer {remote_id} denied')
                    sock.close()
                    return None

                print('[===] Connection to peer is permitted')
                
                # Store the connection
                self.remote_agents[remote_id] = {
                    'sock': sock,
                    'host': host,
                    'port': port,
                    'trust': 1.0,
                    'last_seen': datetime.now(),
                    'failures': 0,
                    'connection_id': connection_id
                }
                self.connections.append(sock)
                
                print(f"[=✅=] Connected to agent {remote_id} at {host}:{port}")
                if self.running:
                    print('[=+=] server is still listening for messages!')
                return sock
            else:
                print(f"[❌] Invalid agent response from {host}:{port}")
                self.report_failure(id(self), 'authentication', reason=f'Failed authentication from {host}:{port}')                
                sock.close()
                return None
                
        except socket.timeout:
            print(f"[❌] Connection timeout to {host}:{port}")
            return None
        except ConnectionRefusedError:
            print(f"[❌] Connection refused by {host}:{port} - server not running?")
            return None
        except Exception as e:
            print(f"[❌] Failed to connect to {host}:{port}: {e}")
            import traceback
            traceback.print_exc()
            return None

    
    def disconnect_agent(self, agent_id):
        if agent_id in self.remote_agents:
            try:
                self._send_message(self.remote_agents[agent_id]['sock'], 
                                  {'type': self.MSG_TYPES['DISCONNECT']})
                self.remote_agents[agent_id]['sock'].close()

                print(f'[===] Removing Agent id: {agent_id}')
                del self.remote_agents[agent_id]
            except:
                pass
            print(f"🔌 Disconnected from agent {agent_id}")

    def _sanitize_structured(self, data, amount=1000):
        """Recursively sanitize strings inside structures"""
        if isinstance(data, str):
            return self._sanitize_input(data, amount)
        elif isinstance(data, list):
            return [self._sanitize_structured(item, amount) for item in data]
        elif isinstance(data, tuple):
            return tuple(self._sanitize_structured(item, amount) for item in data)
        elif isinstance(data, dict):
            return {key: self._sanitize_structured(value, amount) for key, value in data.items()}
        else:
            return data

    # ============ asynchronous queue setup ============
    async def _handle_predict_request_async(self, message):
        # Async handler for prediction requests
        payload = message.payload

        # Initialize variables
        text = None
        test_titles = None
        label_map = None
        rules = None
        
        # ✅ Check payload (which is a dict), not the message itself
        if isinstance(payload, dict):
            if 'test_titles' in payload:
                test_titles = payload.get('test_titles')
                label_map = payload.get('label_map')
                rules = payload.get('rules')
                
                # Sanitize if needed
                if test_titles:
                    test_titles = self._sanitize_structured(test_titles)
                if label_map:
                    label_map = self._sanitize_structured(label_map)
                if rules:
                    rules = self._sanitize_structured(rules)

                print('[=] Got necessary titles, label_map and rules')

            else:
                text = payload.get('text')
                if text:
                    text = self._sanitize_input(text)
                print(f'[=] Got text: {text}')
        else:
            # Fallback: maybe payload is the text directly
            text = str(payload) if payload else None
        
        if not text and not test_titles:
            print('[===] ERROR: No text or test_titles in message payload!')
            return {'type': self.MSG_TYPES['PREDICT_RESPONSE'], 'error': 'No text or test_titles provided'}
        
        # Run the actual prediction
        print(f'[=] Initiating prediction method')
        try:
            if test_titles is not None:
                print('[=] initiating Advanced prediction method...')
                if not self.pipeline.autonomous:
                    self.pipeline.autonomous = True
                    self.pipeline.ensemble.explainer.supervised_learning = False

                if self.predict_manager is not None:
                    result = await asyncio.to_thread(
                        self.predict_manager.advanced_prediction_method,
                        test_titles, label_map, rules,
                        show_proba=True,
                        use_transformer=self.pipeline.use_transformer
                    )
                    # Handle tuple return (result, chosen_label, confidence)
                    if isinstance(result, tuple) and len(result) == 3:
                        _, chosen_label, confidence = result
                    else:
                        chosen_label = result.get('prediction', 'unknown')
                        confidence = result.get('confidence', 0)
                    
                    return {
                        'type': self.MSG_TYPES['PREDICT_RESPONSE'],
                        'prediction': chosen_label,
                        'confidence': confidence,
                        'success': True
                    }

                else:
                    print('[=] Initaiting basic prediction...')
                    result = await asyncio.to_thread(self.pipeline.predict_single, text)
            
                    return {
                        'type': self.MSG_TYPES['PREDICT_RESPONSE'],
                        'prediction': result.get('prediction'),
                        'confidence': result.get('confidence'),
                        'probabilities': result.get('probabilities', []),
                        'agent_id': id(self),
                        'success': True
                    }                    
            else:

                print('[=] Basic prediction method')
                result = await asyncio.to_thread(self.pipeline.predict_single, text)
                
                return {
                    'type': self.MSG_TYPES['PREDICT_RESPONSE'],
                    'prediction': result.get('prediction'),
                    'confidence': result.get('confidence'),
                    'probabilities': result.get('probabilities', []),
                    'agent_id': id(self),
                    'success': True
                }
                
        except Exception as e:
            logger.info(f'[==] error in async method predict request: {e}')
            return {'type': self.MSG_TYPES['PREDICT_RESPONSE'], 'error': str(e), 'success': False}    
        

    async def _async_method_handle_predict_request_(self, message, sender_id, method='basic_prediction', predict_manager=None):
        # Handle prediction request async-ly
        text = None
        test_titles = None
        label_map = None
        rules = None
            
        if 'test_titles' in message:
            test_titles = message.get('test_titles')
            label_map = message.get('label_map')
            rules = message.get('rules')

            test_titles = self._sanitize_input(test_titles)
            label_map = self._sanitize_input(label_map)
            rules = self._sanitize_input(rules)
        else:
            text = message.get('text')
            text = self._sanitize_input(text) 

        if not text:
            print('[===] ERROR: No matched configuration in message for prediction!')
            return {'type': self.MSG_TYPES['PREDICT_RESPONSE'], 'error': 'No text provided'}
        
        # Run the actual prediction in thread pool (since predict_single is sync)
        print(f'[=] Initiating prediction method: {method}')
        try:
            print('[=] Advanced prediction method')
            if method != 'basic_prediction' or predict_manager:
                result = await asyncio.to_thread(
                    predict_manager.advanced_prediction_method,
                    test_titles, label_map, rules, show_proba=False, use_transformer=self.pipeline.use_transformer
                )
            else:
                print('[=] basic prediction method')
                result = await asyncio.to_thread(self.pipeline.predict_single, text)
            
            return {
                'type': self.MSG_TYPES['PREDICT_RESPONSE'],
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'probabilities': result.get('probabilities', []),
                'agent_id': id(self)
            }
        except Exception as e:
            logger.info(f'[==] error in async method predict request: {e}')
            return {'type': self.MSG_TYPES['PREDICT_RESPONSE'], 'error': str(e)}


    async def _handle_memory_sync_async(self, message):
        # Safe handler for memory sync.
        try:
            logger.info(f"[=] Processing memory sync from {message.sender}")
            return await self._handle_memory_sync_request(message, message.sender)
        except Exception as e:
            logger.error(f"[❌] Memory sync failed: {e}")
            return {'error': str(e), 'status': 'failed'}
    
    async def _handle_ensemble_vote_async(self, message):
        # Safe handler for ensemble voting.
        try:
            logger.info(f"[=] Processing ensemble vote from {message.sender}")
            return await self._handle_ensemble_vote_request(message, message.sender)
        except Exception as e:
            logger.error(f"[❌] Ensemble vote failed: {e}")
            return {'error': str(e), 'status': 'failed'}
    
    async def _handle_ping(self, message):
        # Simple ping handler for health checks.
        return {'pong': True, 'timestamp': time.time(), 'agent_id': self.agent_id}
    
    async def _handle_status(self, message):
        # Status handler for monitoring.
        return {
            'status': 'healthy',
            'queue_stats': self.message_queue.get_stats(),
            'connected_agents': len(self.remote_agents),
            'memory_size': len(self.pipeline.memory),
            'uptime': time.time() - self.start_time if hasattr(self, 'start_time') else 0
        }   


    def request_prediction(self, agent_id: str, text: str, timeout: float = 30.0) -> Any:
        # Unified prediction request - works with both sync and async modes.
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Run the async version and wait for result
            result = loop.run_until_complete(
                self.request_prediction_async(agent_id, text, timeout)
            )
            return result
        finally:
            loop.close()

    async def request_advanced_prediction_async(self, manager: Any, use_transformer: bool=False, agent_id: str=None, test_titles: List[tuple]=None, label_map: Dict[str, int]=None, rules: List[tuple]=None, timeout: float = 30.0, callback: Optional[Callable] = None):
        # Asynchronous prediction request
        # Local bypass - NO QUEUE
        if agent_id == 'local':
            logger.info(f"[=] Local request - direct execution")
            # Run sync prediction in thread pool
            result = await asyncio.to_thread(manager.advanced_prediction_method, test_titles, label_map, rules, show_proba=True, use_transformer=use_transformer)
            logger.info(f"[=] Local result: {result[1]} || confidence: {result[2]}")
            return result  

        msg_id = str(uuid.uuid4())
        message = Message(
            id=msg_id,
            type='predict_request',
            sender=self.temporary_agent_id,
            recipient=agent_id,
            payload={'test_titles': test_titles, 'label_map': label_map, 'rules': rules},
            timestamp=datetime.now(),
            timeout=timeout,
            callback=callback,
            max_retries=self.max_retries
        )

        logger.info(f"[=] Remote request - publishing to queue")
        response = await self.message_queue.publish(message)
        logger.info(f"[=] Queue response type: {type(response)}")

        # Extract prediction from response if needed
        if isinstance(response, dict) and 'prediction' in response:
            return response
        elif isinstance(response, dict) and 'result' in response:
            return response['result']
        else:
            return response    
    

    
    def request_prediction_direct(self, agent_id, text, timeout=5):
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        
        # Create future for response
        future = asyncio.Future()
        with self.request_lock:
            self.pending_requests[request_id] = future
        
        # Send message with request_id
        message = {
            'type': 1,
            'text': text,
            'token': self.auth_token,
            'request_id': request_id,  # ← Include in message!
            'timestamp': time.time()
        }
        
        sock = self.remote_agents[agent_id]['sock']
        self._send_message(sock, message)
        
        # Wait for response with timeout
        try:
            return future.result(timeout=timeout)
        finally:
            with self.request_lock:
                self.pending_requests.pop(request_id, None)


    async def request_prediction_async(self, agent_id: str, text: str, timeout: float = 30.0, callback: Optional[Callable] = None):
        # Asynchronous prediction request
        # # Local bypass
        if agent_id == 'local':
            return await asyncio.to_thread(self.pipeline.predict_single, text)  

        if agent_id not in self.remote_agents:
            print(f"[❌] No connection to {agent_id}")
            return None
        
        sock = self.remote_agents[agent_id]['sock']
        
        # Create prediction request
        message = {
            'type': self.MSG_TYPES['PREDICT_REQUEST'],
            'text': text,
            'token': self.auth_token,
            'requester': id(self)
        }
        
        try:
            # Send via existing socket
            self._send_message(sock, message)
            
            # Wait for response
            response = self._receive_message(sock)
            
            if response and response.get('type') == self.MSG_TYPES['PREDICT_RESPONSE']:
                return response
            return None
            
        except Exception as e:
            print(f"[❌] Prediction request failed: {e}")
            return None

    def request_prediction_batch(self, agent_id: str, texts, timeout: float = 30.0) -> List[Any]:
        # Batch async prediction requests (parallelized)
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(texts)) as executor:
            futures = [
                executor.submit(self.request_prediction, agent_id, text, timeout)
                for text in texts
            ]
            results = [f.result(timeout=timeout) for f in futures]
        
        return results
    
    
    def start_queue_processor(self):
        # Start background queue processor
        self.queue_processor_thread = threading.Thread(target=self._process_outgoing_queue, daemon=True)
        self.queue_processor_thread.start()
    
    def _process_outgoing_queue(self):
        # Process queued outgoing messages
        while self.running:
            if self.outgoing_queue:
                msg = self.outgoing_queue.popleft()
                try:
                    self._send_message(msg['sock'], msg['message'])
                    if msg.get('callback'):
                        msg['callback'](True)
                except Exception as e:
                    if msg.get('callback'):
                        msg['callback'](e)
                    # Retry logic
                    if msg.get('retry_count', 0) < msg.get('max_retries', 3):
                        msg['retry_count'] = msg.get('retry_count', 0) + 1
                        self.outgoing_queue.append(msg)
            else:
                time.sleep(0.01)     

    
    def _start_health_checker(self):
        # Start background health checker for async mode.
        def health_check_loop():
            while self.running:
                time.sleep(self._health_check_interval)
                self._check_health()
        
        self._health_thread = threading.Thread(target=health_check_loop, daemon=True)
        self._health_thread.start()
    
    def _check_health(self):
        # Check health of all connected agents.
        stats = self.message_queue.get_status()
        logger.debug(f"[=] Queue stats: {stats}")
        
        # Check for stuck messages
        if stats.get('pending_count', 0) > 100:
            logger.warning(f"[=] High pending count: {stats['pending_count']}")
        
        # Ping all agents
        for agent_id in list(self.remote_agents.keys()):
            try:
                result = self.broadcast('ping', {}, timeout=5.0)
                if agent_id not in result or result[agent_id].get('error'):
                    logger.warning(f"[=] Agent {agent_id} not responding")
            except Exception as e:
                logger.warning(f"[=] Health check failed for {agent_id}: {e}")
    
    def get_queue_stats(self) -> Dict:
        # Get message queue statistics.
        return self.message_queue.get_status()
    
    def get_dead_letter_queue(self) -> List[Dict]:
        # Get failed messages for inspection.
        if hasattr(self.message_queue, 'get_dead_letter_queue'):
            return self.message_queue.get_dead_letter_queue()
        return []
    
    def stop(self):
        # Graceful shutdown.
        logger.info("[=] Shutting down AgentDistributedInference...")
        self.running = False
        asyncio.create_task(self.message_queue.stop())
        logger.info("[=] Shutdown complete")

    # ============ MESSAGE HANDLING ============


    def _send_message(self, sock, message):
        # Send message with signature and DOES NOT modify original
        if sock is None:
            print(f"[==] Send error: socket is None")
            return False
            
        # ✅ Check if socket is still connected
        try:
            sock.getpeername()
        except (socket.error, OSError, AttributeError) as e:
            print(f"[==] Send error: socket is dead - {e}")
            # Remove dead socket from remote_agents
            self._remove_dead_socket(sock)
            return False     

        try:
            msg_to_send = message.copy()  # ← Important: Copy!
            
            # Add signature to the copy
            msg_to_send['signature'] = self._sign_message(msg_to_send)

            sorted_msg = {k: msg_to_send[k] for k in sorted(msg_to_send.keys())}

            print(f'[==] Sending message, Total: {len(sorted_msg)}')   
            data = pickle.dumps(sorted_msg, protocol=pickle.HIGHEST_PROTOCOL)
            sock.send(len(data).to_bytes(4, 'big'))
            bytes_sent = 0
            while bytes_sent < len(data):
                chunk = data[bytes_sent:bytes_sent + self.CHUNK_SIZE]
                sock.send(chunk)
                bytes_sent += len(chunk)
                # Small delay to prevent buffer overflow
                if len(chunk) == self.CHUNK_SIZE:
                    time.sleep(0.001)
              
            print(f'[==] Message sent successfully')
            logger.info(f"[=] Message sent successfully: {sorted_msg}")
            return True
        except Exception as e:
            print(f"[==] Send error: {e}")
            traceback.print_exc()
            self._remove_dead_socket(sock)
            return False


    def _remove_dead_socket(self, sock):
        """Remove dead socket from remote_agents"""
        for agent_id, info in list(self.remote_agents.items()):
            if info.get('sock') == sock:
                print(f"[=] Removing dead connection to {agent_id}")
                del self.remote_agents[agent_id]
                break 

    def _receive_message(self, sock):
        try:
            print(f'[==] Server status: {self.running}')
            print(f'[=] Sock status: {sock}')

            if sock is None:
                print('[=] Sock is None !')
                return None

            try:
                data_len = sock.recv(4)
            except:
                data_len = sock.recv(10)

            print(f'[==] Data length received: {data_len}')
            if not data_len:
                print('[=] received empty message.')
                return None
            
            msg_len = int.from_bytes(data_len, 'big')
            if msg_len > self.max_message_size:
                print('[=] message size exceeds maximum to be handled')
                self.log_security_event('message_too_large', {'size': msg_len})
                return None

            data = b''
       
            while len(data) < msg_len:
                remaining = msg_len - len(data)
                chunk_size = min(self.CHUNK_SIZE, remaining)
                chunk = sock.recv(chunk_size)
                if not chunk:
                    print(f'[=] Connection closed while receiving')
                    return None
                data += chunk            
           
            message = pickle.loads(data)
        
            if "signature" in message:
                msg_for_verify = {k: v for k, v in message.items() if k != 'signature'}

                if not self._verify_signature(msg_for_verify, message['signature']):
                    logger.warning(f"[=] Invalid message signature from agent {self.temporary_agent_id}")
                    self._log_security_event('invalid_signature', {'agent_id': self.temporary_agent_id})
                    return None

            print('[= Message received]')
            return message

        except socket.timeout:
            print('[-] Socket timeout')
            return None
        except Exception as e:
            logger.error(f"[=] Receive error: {e}")
            traceback.print_exc()
            return None
    
    def _handle_client(self, client, addr):
        agent_id = f"{addr[0]}:{addr[1]}"
        self.temporary_agent_id = agent_id
        
        # Register this thread as the owner of this socket
        self.socket_owners[client] = threading.current_thread().name

        if addr[0] in ['127.0.0.1', 'localhost', 'local'] and addr[1] == self.port:
            print(f"[❌] Client is self, ignoring")
            client.close()
            return        
            
        if self._is_duplicate_connection(addr[0], addr[1]):
            print(f"[⚠️] Duplicate connection from {addr[0]}:{addr[1]}, rejecting")
            client.close()
            return
            
        # ✅ Prevent multiple connections from same host
        for existing_id, info in list(self.remote_agents.items()):
            if info.get('host') == addr[0]:
                print(f"[❌] Already have connection from {addr[0]}, rejecting new connection")
                client.close()
                return            

        while self.running:
            try:
                if 'request_id' in message:
                    continue 


                if not self._check_rate_limit(agent_id):
                    self._send_message(client, {'type': 'error', 'message': 'Rate limit exceeded'})
                    logger.info(f"[=##=] Rate limit exceeded for agent {agent_id}, request reduced.")
                    time.sleep(5)  # Sleep briefly to mitigate rapid retries
                    continue

                if message.get('type') == 2:  # PREDICT_RESPONSE
                    continue  # Skip, we'll read it in request_prediction_method

                message = self._receive_message(client) 
                self.temporary_message = message
                if message is None:
                    print('[-] Message is None.')
                    continue

                request_id = message.get('request_id')
                if request_id and request_id in self.pending_requests:
                    with self.request_lock:
                        future = self.pending_requests.get(request_id)
                        if future and not future.done():
                            future.set_result(message) 

                response = self._process_message(message, agent_id)
    
                print(f'[=~=] Got Response from client with address: {addr[0]}:{addr[1]}')
                if response:
                    print(f'[=] Sending response to client: {client}')
                    self._send_message(client, response)
                    logger.info(f'[=] Succesfully send response to client: {client}')
                else:
                    print("[SERVER] No response to send - ")
                    self._send_message(client, {'type': 'ack', 'status': 'ok'})

            except Exception as e:
                print(f"[=] Handler error for {agent_id}: {e}")
                break
        
        # Cleanup on disconnect
        if agent_id in self.remote_agents:
            print(f'[===] Removing Agent id: {agent_id}')            
            del self.remote_agents[agent_id]
        if client in self.connections:
            self.connections.remove(client)

        client.close()
        print(f"📡 Disconnected from {agent_id}")
    

    def _process_message(self, message, sender_id):
        # Process incoming messages based on type
        msg_type = message.get('type')
        
        if msg_type == self.MSG_TYPES['PREDICT_REQUEST']:
            return self._handle_predict_request(message, sender_id)

        elif msg_type == self.MSG_TYPES['MEMORY_SYNC_REQUEST']:
            return self._handle_memory_sync_request(message, sender_id)
        
        elif msg_type == self.MSG_TYPES['ENSEMBLE_VOTE_REQUEST']:
            return self._handle_ensemble_vote_request(message, sender_id)
        
        elif msg_type == self.MSG_TYPES['FAILURE_REPORT']:
            return self._handle_failure_report(message, sender_id)
        
        elif msg_type == self.MSG_TYPES['TRUST_UPDATE']:
            return self._handle_trust_update(message, sender_id)
        
        elif msg_type == self.MSG_TYPES['PING']:
            return {'type': self.MSG_TYPES['PONG'], 'timestamp': time.time()}
        
        elif msg_type == self.MSG_TYPES['DISCONNECT']:
            return None
        
        return {'type': 'ack', 'status': 'ok'}


    # ====== HANDLE PREDICTION AND UNCERTAINTY CALIBRATION ====== 
    def _check_trust_level(self, agent_id, required_trust=TrustLevel.STANDARD):
        # Check if agent has sufficient trust level for operation
        
        if agent_id not in self.trusted_agents:
            print(f"[-] Agent {agent_id} not trusted")
            return False
        
        agent_trust = self.trusted_agents[agent_id].get('trust_level', TrustLevel.BASIC)
        
        if agent_trust >= required_trust:
            return True
        else:
            print(f"[-] Agent {agent_id} trust level {agent_trust.name} < required {required_trust.name}")
            return False


    def _handle_peer_agent_request(self, probs, self_attn_weights, input_ids, type=None, agreement=False):
        memory_exist = self.sync_with_local_peer(self.memory_name)   
        established_connection = self.query_node._establish_peer_nodes(self.temporary_agent_id)

        if established_connection:
            print(f'[||] Connection established and permitted with peer agent: {self.temporary_agent_id}')
            try:
                if memory_exist and type == 'DevicePeer':
                    target_preds, attn_weights = self.pipeline.storage.memory_retrieval(self.memory_name, type_func="Peer", verbose=False)
                    
                else:
                    # external peer communicates via socket
                    if type == "ExternalPeer":
                        try:
                            target_preds, attn_weights = self.get_external_peer_message()
                            if target_preds is None:
                                print('[-] Cant get viable components needed for processing request, returning regular probs...')
                                return probs

                        except Exception as e:
                            print(f'[-] No valid in device peer memory id found in database for memory name: {self.memory_name} and error: {e}')
                            return probs
                    else:
                        print('[-] Invalid type..., returning regular probs...')
                        return probs

                if not agreement:
                    probs = self.handle_peer_uncertainty(probs, target_preds, self_attn_weights, attn_weights, input_ids)
                else:
                    try:
                        probs = self.process_peer_request(target_preds, self_attn_weights, attn_weights)
    
                    except Exception as e:
                       print(f"[-] Error processing request: {e}, returning regular probs")

            except Exception as e:
                print(f'[-] Error handling request... {e}, returning regular probs')
                self.report_failure(id(self), 'processing', reason=f'{e}')                        

            print(f'[||] Successfully calibrate probs with previous Peer using database!')
            self.save_to_local_peer(self.memory_name, probs)
        else:
            print(f'[-] Connection to peer agent {self.temporary_agent_id} failed or not permitted, returning regular probs...')

        return probs


    def _calibrate_peer_probs(self, probs, target_preds, self_attn_weights, attn_weights, input_ids, AEL):
        calibrated = probs.copy()
        try:
            n_classes = probs.shape[1]
        except:
            n_classes = probs.shape[0]

        batch_size = len(target_preds)
        anisotropy = self.pipeline.anisotropy_measurement(attn_weights)    
        eps = 1e-5
  
        for i in range(batch_size):
            mlp_target = target_preds[i]
            attn_target = attn_weights[i]
            if self_attn_weights is not None and i < len(attn_weights):
                attn = self_attn_weights[i]

                attn_quality = np.std(attn) if attn.size > 0.0 else AEL
                target_attention_quality = np.std(attn_target) if attn.size > 0.0 else AEL

                try:
                    target_attn_indices = np.argmax(attn_weights)
                    target_mlp_indices = np.argmax(mlp_target)
                except:
                    target_attn_indices = np.argmax(attn_weights, axis=1)
                    target_mlp_indices = np.argmax(mlp_target, axis=1)                    

                consensus = np.allclose(target_mlp_indices, target_attn_indices, atol=eps)

                justified = (1.0 - AEL) + (1.0 - attn_quality) * consensus
                boost = justified * anisotropy + eps

            else:
                attn_quality = 1.0 / (1.0 + np.exp(-self_attn_weights[i]))

                target_attn_indices = np.argmax(attn_weights, axis=1)
                target_prob_indices = np.argmax(probs, axis=1)

                consensus = np.allclose(target_prob_indices, target_attn_indices, atol=eps)

                justified = (1.0 - AEL) + attn_quality * consensus
                boost = (1.0 - justified) * anisotropy + eps

            quality_temperature = (boost + 1.0 - AEL) + (1.0 - attn_quality) * anisotropy + eps
            self.query_node.peer_trust = quality_temperature + justified * anisotropy

            try:
                calibrated[i, mlp_target] = min(calibrated[i, mlp_target] * (1.5 * quality_temperature), 0.95)
            except:
                return calibrated

            calibrated[i] /= calibrated[i].sum()


        return calibrated        
            

    def handle_peer_uncertainty(self, probs, target_preds, self_attn_weights, attn_weights, input_ids):
        try:
            if self_attn_weights is None:
                _, _, self_attn_weights = self.pipeline.model2.predict(input_ids)                
            batch_similarity = self.pipeline.cosine_similarity(attn_weights, self_attn_weights)
        
            anisotropy = self.pipeline.anisotropy_measurement(attn_weights)
            AME = self.pipeline.AME_Encoder(attn_weights)
            AMR = 1.0 / (1.0 + np.exp(-AME))
            weighted_similarity = batch_similarity * (1.0 - AMR) * anisotropy

            if weighted_similarity > 0.75:
                return self.process_peer_request(probs, target_preds, attn_weights, input_ids)
            else:
                print('[-] Low uncertainty, normalizing with local agent data...')

                AEL = 0.3 + weighted_similarity * anisotropy
                calibrated = self._calibrate_peer_probs(probs, target_preds, self_attn_weights, attn_weights, input_ids, AEL)
                return calibrated

        except Exception as e:
            print(f"[-] Error in uncertainty handling: {e}")
            return probs


    def process_peer_request(self, probs, target_preds, attn_weights, input_ids):
        try:
            response_probs = self.pipeline.pipeline._calibrate_probs(probs, target_preds, attn_weights, input_ids)
            return response_probs
        except Exception as e:
            print(f"[-] Error in peer request_processing: {e}")
            return None
            

    # ============ REQUEST HANDLERS ============
    def get_external_peer_message(self):
        message = self.temporary_message
        if not message:
            print('[-] No viable messages')
            return None, None

        try:
            attn_weights = message.get('attn_weights')
            target_preds = message.get('target_preds')
            if not attn_weights:
                print('|| Invalid format of message, may be a Nonetype object...')
                return None, None
            return attn_weights, target_preds

        except Exception as e:
            print(f'[-] Cant get external peer message: {e}')
            return None, None
         
    
    def _handle_predict_request(self, message, sender_id, method='basic_prediction'):
        if not self._check_trust_level(sender_id, TrustLevel.STANDARD):
            return {'type': 'error', 'message': 'Insufficient trust level'}           
                         
        if method == 'basic_prediction' and self.predict_manager is None:
            text = message.get('text')
            if not text:
                return {'type': self.MSG_TYPES['PREDICT_RESPONSE'], 'error': 'No text provided'}
            
            text = self._sanitize_input(text)
            if not self._check_rate_limit(sender_id):
                return {'type': self.MSG_TYPES['PREDICT_RESPONSE'], 'error': 'Rate limit exceeded'}

            try:
                result = self.pipeline.predict_single(text)
            
                # Log the interaction
                self._log_interaction(sender_id, 'prediction', result['confidence'])
                
                return {
                    'type': self.MSG_TYPES['PREDICT_RESPONSE'],
                    'prediction': result['prediction'],
                    'confidence': result['confidence'],
                    'probabilities': result.get('probabilities', []),
                    'agent_id': id(self)
                }
            except Exception as e:
                print(f"[-] Prediction error: {e}")
                return {'type': self.MSG_TYPES['PREDICT_RESPONSE'], 'error': str(e)}

        else:
            titles = message.get('test_titles')
            label_map = message.get('label_map')
            rules = message.get('rules')
            if not titles and label_map and rules:
                return {'type': self.MSG_TYPES['PREDICT_RESPONSE'], 'error': 'No test titles provided'}
            
            titles = self._sanitize_arrays_and_dicts(titles)
            label_map = self._sanitize_arrays_and_dicts(label_map)
            rules = self._sanitize_arrays_and_dicts(rules)

            if not self._check_rate_limit(sender_id):
                return {'type': self.MSG_TYPES['PREDICT_RESPONSE'], 'error': 'Rate limit exceeded'}

            try:
                result, chosen_label, confidence = self.predict_manager.advanced_prediction_method(titles, label_map, rules, show_proba=True, use_transformer=self.pipeline.use_transformer)
            
                # Log the interaction
                self._log_interaction(sender_id, 'prediction', confidence)
                
                return {
                    'type': self.MSG_TYPES['PREDICT_RESPONSE'],
                    'prediction': chosen_label,
                    'confidence': confidence,
                    'probabilities': result,
                    'agent_id': id(self)
                }

            except Exception as e:
                print(f"[-] Advanced prediction error: {e}")
                return {'type': self.MSG_TYPES['PREDICT_RESPONSE'], 'error': str(e)}

    def _handle_memory_sync_request(self, message, sender_id):
        memory_name = message.get('memory_name')
        if not memory_name:
            return {'type': self.MSG_TYPES['MEMORY_SYNC_RESPONSE'], 'error': 'No memory name'}
        
        try:
            # For local peer (database)
            if message.get('peer_type') == 'local':
                memory_data = self.pipeline.storage.load_model_dict(memory_name)
            else:
                # For external peer
                memory_data = self.pipeline.memory.get(memory_name, {})
            
            return {
                'type': self.MSG_TYPES['MEMORY_SYNC_RESPONSE'],
                'memory_name': memory_name,
                'data': memory_data,
                'timestamp': time.time()
            }
        except Exception as e:
            return {'type': self.MSG_TYPES['MEMORY_SYNC_RESPONSE'], 'error': str(e)}



    def _handle_ensemble_vote_request(self, message, sender_id):
        # Handle ensemble vote request from another agent
        text = message.get('text')
        if not text:
            return {'type': self.MSG_TYPES['ENSEMBLE_VOTE_RESPONSE'], 'error': 'No text provided'}
        
        try:
            result = self.pipeline.predict_single(text)
            
            return result['prediction'], {
                'type': self.MSG_TYPES['ENSEMBLE_VOTE_RESPONSE'],
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'agent_id': id(self),
                'trust_score': self.remote_agents.get(sender_id, {}).get('trust', 1.0)
            }
        except Exception as e:
            return None, {'type': self.MSG_TYPES['ENSEMBLE_VOTE_RESPONSE'], 'error': str(e)}
    
    def _handle_failure_report(self, message, sender_id):
        # Handle failure report from another agent

        failed_agent = message.get('failed_agent')
        task_type = message.get('task_type')
        failure_reason = message.get('reason', 'unknown')
        
        # Update trust for the failed agent
        if failed_agent in self.remote_agents:
            self.remote_agents[failed_agent]['failures'] += 1
            self.remote_agents[failed_agent]['trust'] = max(
                0.1, 
                1.0 - (self.remote_agents[failed_agent]['failures'] / 10)
            )
        
        # Log the failure
        self._log_interaction(failed_agent, 'failure', confidence=0, details={
            'task_type': task_type,
            'reason': failure_reason,
            'reported_by': sender_id
        })
        
        return {'type': 'ack', 'status': 'failure_recorded'}


    
    def _handle_trust_update(self, message, sender_id):
        # Handle trust score update
        target_agent = message.get('target_agent')
        new_trust = message.get('trust_score')

        self.query_node.peer_trust = new_trust
        
        if target_agent in self.remote_agents:
            self.remote_agents[target_agent]['trust'] = new_trust
        
        return {'type': 'ack', 'status': 'trust_updated'}


    # ============ REQUEST SENDING METHODS ============       
    def request_prediction_method(self, agent_id, text, timeout=5):
        if agent_id == 'local':
            result = self.pipeline.predict_single(text)
            return result

        if agent_id not in self.remote_agents:
            print(f"Agent {agent_id} not connected")
            return None
        
        sock = self.remote_agents[agent_id]['sock']
        request_id = str(uuid.uuid4())[:8]
            
        message = {
            'type': self.MSG_TYPES['PREDICT_REQUEST'],  # 1
            'text': text,
            'request_id': request_id,  # ← Add request ID!
            'token': self.auth_token,
            'timestamp': time.time()
        }
                
        try:
            sock.settimeout(timeout)
            self._send_message(sock, message)
            response = self._receive_message(sock)
            sock.settimeout(None)
            
            if response and response.get('type') == self.MSG_TYPES['PREDICT_RESPONSE']:
                return response
            return None
        except Exception as e:
            print(f"Request failed for {agent_id}: {e}")
            return None
    
    def request_ensemble_vote(self, agent_id, text, timeout=5):
        if agent_id not in self.remote_agents:
            return None
        
        sock = self.remote_agents[agent_id]['sock']
        message = {
            'type': self.MSG_TYPES['ENSEMBLE_VOTE_REQUEST'],
            'text': text
        }
        
        try:
            sock.settimeout(timeout)
            self._send_message(sock, message)
            response = self._receive_message(sock)
            sock.settimeout(None)
            
            if response and response.get('type') == self.MSG_TYPES['ENSEMBLE_VOTE_RESPONSE']:
                return response['prediction'], response['text']
            return None, None
        except Exception as e:
            print(f"Vote request failed: {e}")
            return None, None
    
    def sync_memory_with_agent(self, agent_id, memory_name, timeout=10):
        if agent_id not in self.remote_agents:
            return None
        
        sock = self.remote_agents[agent_id]['sock']
        message = {
            'type': self.MSG_TYPES['MEMORY_SYNC_REQUEST'],
            'memory_name': memory_name,
            'peer_type': 'external'
        }
        
        try:
            sock.settimeout(timeout)
            self._send_message(sock, message)
            response = self._receive_message(sock)
            sock.settimeout(None)
            
            if response and response.get('type') == self.MSG_TYPES['MEMORY_SYNC_RESPONSE']:
                return response.get('data', {})
            return None
        except Exception as e:
            print(f"Memory sync failed: {e}")
            return None
    
    def report_failure(self, agent_id, task_type, reason="unknown"):
        report = {
            'type': self.MSG_TYPES['FAILURE_REPORT'],
            'failed_agent': agent_id,
            'task_type': task_type,
            'reason': reason,
            'timestamp': time.time()
        }
        
        # Send to all other agents
        for other_id, agent_info in list(self.remote_agents.items()):
            if other_id != agent_id:
                self._send_message(agent_info['sock'], report)
    
    def broadcast_ping(self):
        # Check which agents are still alive
        alive_agents = []
        for agent_id, agent_info in list(self.remote_agents.items()):
            try:
                sock = agent_info['sock']
                self._send_message(sock, {'type': self.MSG_TYPES['PING']})
                response = self._receive_message(sock)
                if response and response.get('type') == self.MSG_TYPES['PONG']:
                    alive_agents.append(agent_id)
                    agent_info['last_seen'] = datetime.now()
                else:
                    # Agent dead, remove
                    print(f'[===] Removing Agent id: {agent_id}')                    
                    del self.remote_agents[agent_id]
            except:
                print(f'[===] Removing Agent id: {agent_id}')                
                del self.remote_agents[agent_id]
        
        return alive_agents
    
    # ============ LOCAL PEER (DATABASE) METHODS ============
    
    def sync_with_local_peer(self, memory_name):
        try:
            memory_exist = self.pipeline.storage.memory_exists(self.memory_name, type='Peer')
            if memory_exist:
                memory_data = self.pipeline.storage.memory_retrieval(self.pipeline.memory_name, type_func="Peer", verbose=False)
                print(f'|| Retrieved memory, Samples: {len(memory_data)}')

            try:
                if memory_exist and memory_data:
                    # Merge with current memory
                    print('[=] Syncing with local peer memory data...')
                    try:
                        for key, value in memory_data.items():
                            if key not in self.pipeline.memory:
                                self.pipeline.memory[key] = value
                    except Exception as e:
                        print(f'|| Using sync memory function because of {e} problem in regular syncing using value in items.')
                        agent_id = self.temporary_agent_id
                        self.sync_memory_with_agent(agent_id, memory_name)

                    print(f"✅ Synced with local peer: {len(memory_data)} memories")
            except:
                print(f'[-] Failed converting and syncing with peer, but memory exist is assured.')
            memory_exist = True
            return memory_exist

        except Exception as e:
            print(f"Local peer sync failed: {e}")
            memory_exist = False

        print(f'|| Memory Exist: {memory_exist}')
        
        return memory_exist
    
    def save_to_local_peer(self, memory_name, data):
        try:
            self.pipeline.storage.save_model_dict(memory_name, data)
            print(f"✅ Saved local peer presence: {memory_name}")
            return True
        except Exception as e:
            print(f"Save to local peer failed: {e}")
            return False
    
    # ============ UTILITY METHODS ============
    
    def _log_interaction(self, agent_id, interaction_type, confidence, details=None):
        if agent_id not in self.agent_comm_log:
            self.agent_comm_log[agent_id] = []
        
        self.agent_comm_log[agent_id].append({
            'timestamp': datetime.now(),
            'type': interaction_type,
            'confidence': confidence,
            'details': details
        })
    
    def get_agent_status(self):
        status = {}
        for agent_id, info in list(self.remote_agents.items()):
            status[agent_id] = {
                'connected': True,
                'trust': info['trust'],
                'failures': info['failures'],
                'last_seen': info['last_seen'].isoformat(),
                'host': info['host'],
                'port': info['port']
            }
        return status
    
    def get_communication_log(self, agent_id=None, limit=50):
        # Get communication log for an agent
        if agent_id:
            return self.agent_comm_log.get(agent_id, [])[-limit:]
        
        # Return all logs
        return self.agent_comm_log
    
    def print_network_status(self):
        print("\n" + "="*60)
        print("🤖 == AGENT NETWORK STATUS ==")
        print("="*60)
        print(f"[=] Local Agent: {self.memory_name}")
        print(f"[=] Port: {self.port}")
        print(f"[=] Connected Agents: {len(self.remote_agents)}")

        agent_id = self.temporary_agent_id
        comm_log = self.get_communication_log(agent_id)
        
        for agent_id, info in self.remote_agents.items():
            print(f"\n  📡 {agent_id}")
            print(f"     Trust: {info['trust']:.2f}")
            print(f"     Failures: {info['failures']}")
            print(f"     Last seen: {info['last_seen'].strftime('%H:%M:%S')}")
            print(f"     Agent Communication Log: {comm_log}")
        
        print("="*60)

# The QueryNode class manages the connection and interaction with other nodes (agents) in the network. It handles node identification, agreement evaluation, safety checks, and maintains a memory of connected nodes. 
# The class allows for flexible interactions while ensuring the safety and integrity of the Master node.