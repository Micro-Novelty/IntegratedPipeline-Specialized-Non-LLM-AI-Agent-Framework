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
# agents.py
# ConsecutivePeerAgent    — a single agent node that wraps IntegratedPipeline,
#                           stores predictions, and participates in peer-to-peer
#                           inference sharing via Message passing.
# CohesiveAgentDeployment — orchestrates a fleet of ConsecutivePeerAgents,
#                           manages the PipelineAsyncManager HTTP server,
#                           WorkerPool, and AsyncResultQueue for the full
#                           production deployment stack.
# Depends on: pipeline, async_manager, messaging, security, mlp, transformer
# ---------------------------------------------------------------------------
from .pipeline import IntegratedPipeline
from .async_manager import PipelineAsyncManager, PipelinePredictionManager
from .messaging import Message, AsyncResultQueue, WorkerPool
from .security import SecurityConfig, SecurityLevel
from .mlp import MLP
from .transformer import Transformer

class ConsecutivePeerAgent:
    """
    Robust PeerAgent with security layer.
    Used as fallback when main system fails.
    """
    
    def __init__(self, peer_id: str, port: int, secret_key: str, 
                 manager=None, pipeline=None):
        self.peer_id = peer_id
        self.port = port
        self.secret_key = secret_key
        self.manager = manager  # PipelinePredictionManager
        self.pipeline = pipeline  # IntegratedPipeline
        
        self.connected_peers: Dict[str, Dict] = {}
        self.server_socket: Optional[socket.socket] = None
        self.running = False
        self._lock = threading.RLock()
        
        # Security
        self.allowed_ips = {'127.0.0.1'}
        self.max_message_size = 10 * 1024 * 1024  # 10MB
        
        # Statistics
        self.stats = {
            'predictions': 0,
            'peer_requests': 0,
            'errors': 0
        }
    
    def _sign_message(self, message: dict) -> str:
        """Sign message with HMAC"""
        msg_copy = {k: v for k, v in message.items() if k != 'signature'}
        sorted_msg = {k: msg_copy[k] for k in sorted(msg_copy.keys())}
        msg_bytes = pickle.dumps(sorted_msg, protocol=4)
        key = self.secret_key.encode()
        return hmac.new(key, msg_bytes, hashlib.sha256).hexdigest()
    
    def _verify_signature(self, message: dict, signature: str) -> bool:
        # Verify message signature
        expected = self._sign_message({k: v for k, v in message.items() if k != 'signature'})

        print(f'[ConsecutivePeerAgent] Comparing Signature {signature} with {expected}')
        return hmac.compare_digest(expected, signature)
    
    def _send_message(self, sock: socket.socket, message: dict) -> bool:
        """Send signed message"""
        try:
            if sock is None:
                print('[=] Sock is None !')  
                return None

            msg_copy = message.copy()
            msg_copy['timestamp'] = time.time()
            msg_copy['signature'] = self._sign_message(msg_copy)   

            data = pickle.dumps(msg_copy)
            sock.send(len(data).to_bytes(4, 'big'))
            sock.send(data)
            return True
        except Exception as e:
            print(f"[ConsecutivePeerAgent] Send error: {e}")
            return False
    
    def _receive_message(self, sock: socket.socket) -> Optional[dict]:
        """Receive and verify message"""
        try:
            data_len = sock.recv(4)

            print(f'[ConsecutivePeerAgent] Got data length: {data_len}')
            if not data_len:
                return None
            
            msg_len = int.from_bytes(data_len, 'big')
            if msg_len > self.max_message_size:
                return None
            
            data = b''
            while len(data) < msg_len:
                chunk = sock.recv(min(4096, msg_len - len(data)))
                if not chunk:
                    return None
                data += chunk
            
            message = pickle.loads(data)

            print(f'[ConsecutivePeerAgent] Received a message!')
            if 'signature' in message:
                signature = message.pop('signature')
                if not self._verify_signature(message, signature):
                    print(f"[ConsecutivePeerAgent] Invalid signature authentication from message, Message Ignored.")
                    return None

                message['signature'] = signature
            
            return message
            
        except Exception as e:
            print(f"[ConsecutivePeerAgent] Receive error: {e}")
            return None


    async def predict_local(self, text: Any=None) -> Dict:
        """Predict using local model (advanced prediction)"""
        try:
            # Use your existing advanced prediction method
            if self.manager:
                # For single text, wrap in list
                if 'test_titles' in text:
                    test_titles = text['test_titles']
                    label_map = text['label_map']
                    rules = text['rules']
                    result, chosen_label, confidence = self.manager.advanced_prediction_method(
                        test_titles,
                        label_map,
                        rules,
                        show_proba=False,
                        use_transformer=self.pipeline.use_transformer
                    )                    
                else:
                    chosen_label = self.pipeline.predict_single(text)
                    confidence = self.pipeline.confidence_threshold # doubt on simple predictions

                return {
                    'text': text,
                    'prediction': chosen_label,
                    'confidence': confidence,
                    'source': 'local'
                }

            elif not self.manager and self.pipeline:
                result = self.pipeline.predict_single(text)
                return {
                    'text': text,
                    'prediction': result.get('prediction', 'unknown'),
                    'confidence': result.get('confidence', 0.5),
                    'source': 'local'
                }
            else:
                # Fallback simple prediction
                return {
                    'text': text,
                    'prediction': 'unknown',
                    'confidence': 0.5,
                    'source': 'local'
                }
        except Exception as e:
            print(f"[ConsecutivePeerAgent] Local prediction error: {e}")
            return {
                'text': text,
                'prediction': 'error',
                'confidence': 0.0,
                'source': 'local',
                'error': str(e)
            }
    
    async def request_peer_prediction(self, peer_host: str, peer_port: int, text: str, timeout: float = 5.0) -> Optional[Dict]:
        """Request prediction from peer - ONLY sends text!"""
        
        peer_key = f"{peer_host}:{peer_port}"
        
        with self._lock:
            # Check if already connected
            if peer_key not in self.connected_peers:
                # Create new connection
                try:
                    if self.pipeline.distribution.enable_ssl and self.pipeline.distribution.ssl_context:
                        sock = self.pipeline.distribution.ssl_context.wrap_socket(socket.socket(socket.AF_INET, socket.SOCK_STREAM))
                    else:
                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                
                    sock.settimeout(timeout)
                    sock.bind(('127.0.0.1', 0))

                    sock.connect((peer_host, peer_port))

                    if peer_host in ['127.0.0.1', 'localhost', 'local'] and peer_port == self.port:
                        print(f"[❌] Requesting to self, ignoring request...")
                        sock.close()
                        return  
                    
                    # Authenticate
                    
                    auth_msg = {
                    'type': 'auth',
                    'peer_id': self.peer_id,
                    'token': self.secret_key
                    }


                    if not self._send_message(sock, auth_msg):
                        print(f"[ConsecutivePeerAgent] Failed to send auth to {peer_key}")                        
                        sock.close()
                        return None   
                    else:
                        print('[ConsecutivePeerAgent] Successfully send Authentication message')                        

                    response = self._receive_message(sock)

                    print(f'[ConsecutivePeerAgent] Got Authentication response from peer: {response} ')
                    if not response or response.get('status') != 'ok':
                        sock.close()
                        print('[ConsecutivePeerAgent] Socket is closed!')
                        return None
                    else:
                        print(f'[ConsecutivePeerAgent] Received Response from peer')
                    
                    self.connected_peers[peer_key] = {
                        'sock': sock,
                        'host': peer_host,
                        'port': peer_port,
                        'last_seen': time.time()
                    }
                except Exception as e:
                    print(f"[ConsecutivePeerAgent] Connection to {peer_key} failed: {e}")
                    return None
            
            sock = self.connected_peers[peer_key]['sock']
        
        # Send prediction request 
        try:
            request = {
                'type': 'predict',
                'text': text,
                'peer_id': self.peer_id
            }
            
            sock.settimeout(timeout)
            if not self._send_message(sock, request):
                print('[ConsecutivePeerAgent] Send Prediction request Message Failed!')
                return None
            else:
                print('[ConsecutivePeerAgent] Prediction request Message send successful ')
            
            response = self._receive_message(sock)
            sock.settimeout(None)
            print(f'[ConsecutivePeerAgent] Got Prediction response from peer with address: {peer_host}:{peer_port}')

            if response and response.get('type') == 'predict_response':
                self.stats['peer_requests'] += 1

    
                return {
                    'text': text,
                    'prediction': response.get('prediction'),
                    'confidence': response.get('confidence', 0.0),
                    'source': f"peer_{peer_host}:{peer_port}"
                }
            
            return None
            
        except Exception as e:
            print(f"[ConsecutivePeerAgent] Peer request error: {e}")
            # Clean up dead connection
            with self._lock:
                if peer_key in self.connected_peers:
                    try:
                        self.connected_peers[peer_key]['sock'].close()
                    except:
                        pass
                    del self.connected_peers[peer_key]
            return None
    
    async def ensemble_predict(self, peer_addresses: List[Tuple[str, int]],  text: Any=None,
                                confidence_threshold: float = 0.6) -> Dict:
        """
        Ensemble prediction: local first, then ask peers if confidence is low.
        """
        print(f"[ConsecutivePeerAgent] Starting ensemble prediction with port {self.port}!")
        
        # Step 1: Local prediction
        local_result = await self.predict_local(text)
        print(f"[ConsecutivePeerAgent] Local: {local_result['prediction']} ({local_result['confidence']:.1%})")
        
        best_result = local_result
        
        # Step 2: If low confidence, ask peers
        if local_result['confidence'] < confidence_threshold and peer_addresses:
            print(f"[ConsecutivePeerAgent] Low confidence, asking {len(peer_addresses)} peers...")
            
            peer_results = []
            for host, port in peer_addresses:
                result = await self.request_peer_prediction(host, port, text, timeout=60)
                if result:
                    peer_results.append(result)

                    print(f"[ConsecutivePeerAgent] Peer {host}:{port}: {result['prediction']} ({result['confidence']:.1%})")
                    print(f'[==] Local result: {local_result['prediction']} With Confidence: {local_result['confidence']}')
                    print(f'[==] Peer result: {result['prediction']} With Confidence: {result['confidence']}')

            if peer_results:
                best_peer = max(peer_results, key=lambda x: x['confidence'])                                   
                if best_peer['confidence'] > local_result['confidence']:
                    best_result = best_peer
                    print(f"[ConsecutivePeerAgent] Using peer result: {best_peer['prediction']} || Confidence: ({best_peer['confidence']:.1%})")
        
        self.stats['predictions'] += 1
        return best_result
    
    def start_server(self):
        """Start server to accept peer connections"""
        
        def server_loop():
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind(('0.0.0.0', self.port))
            if self.pipeline.distribution.enable_ssl and self.pipeline.distribution.ssl_context:
                self.server_socket = self.pipeline.distribution.ssl_context.wrap_socket(self.server_socket, server_side=True)

            self.server_socket.listen(5)
            self.running = True
            
            print(f"[ConsecutivePeerAgent] Server listening on port {self.port}!")
            
            while self.running:
                try:
                    client, addr = self.server_socket.accept()
                    
                    # Check IP
                    if addr[0] not in self.allowed_ips:
                        print(f"[ConsecutivePeerAgent] Rejected connection from {addr}")
                        client.close()
                        continue
                    
                    # Handle in thread
                    thread = threading.Thread(target=self._handle_client, args=(client, addr))
                    thread.daemon = True
                    thread.start()
                    
                except Exception as e:
                    if self.running:
                        print(f"[ConsecutivePeerAgent] Server error: {e}")
        
        thread = threading.Thread(target=server_loop, daemon=True)
        thread.start()
    
    def _handle_client(self, client, addr):
        # Handle incoming peer connection
        print(f"[ConsecutivePeerAgent] Client connected from {addr}")
        
        try:
            # Authenticate
            if addr[0] in ['127.0.0.1', 'localhost', 'local'] and addr[1] == self.port:
                print(f"[❌] Client is self, ignoring...")
                client.close()
                return   

            auth_msg = self._receive_message(client)

            if not auth_msg:
                print(f"[ConsecutivePeerAgent] No authentication message from peer {addr}")
                client.close()
                return


            if not auth_msg or auth_msg.get('type') != 'auth':
                print(f"[ConsecutivePeerAgent] Auth failed from {addr}")
                client.close()
                return
            
            if auth_msg.get('token') != self.secret_key:
                print(f"[ConsecutivePeerAgent] Invalid token from {addr}")
                client.close()
                return
            
            # Send auth response
            self._send_message(client, {'type': 'auth_response', 'status': 'ok'})
            
            # Handle prediction requests
            while self.running:
                message = self._receive_message(client)
                if message is None:
                    break
                
                if message.get('type') == 'predict':
                    text = message.get('text', '')
                    print(f"[ConsecutivePeerAgent] Received prediction request!")
                    
                    # Use local prediction
                    result = asyncio.run(self.predict_local(text))
                    
                    response = {
                        'type': 'predict_response',
                        'prediction': result['prediction'],
                        'confidence': result['confidence']
                    }
                    self._send_message(client, response)
                    
                elif message.get('type') == 'ping':
                    self._send_message(client, {'type': 'pong'})
                    
        except Exception as e:
            print(f"[ConsecutivePeerAgent] Client handler error: {e}")
        finally:
            client.close()
            print(f"[ConsecutivePeerAgent] Client disconnected from {addr}")


    def stop_server(self):
        self.running = False

        print('[ConsecutivePeerAgent] Server shutdown initiated...')
        if self.server_socket:
            self.server_socket.close()
        
        # Close all peer connections
        with self._lock:
            for key, info in self.connected_peers.items():
                try:
                    info['sock'].close()
                except:
                    pass
            
            self.connected_peers.clear()
            print('[ConsecutivePeerAgent] Server Successfully Stopped listening !')
    
    def get_stats(self) -> Dict:
        # Get statistics
        return {
            **self.stats,
            'connected_peers': len(self.connected_peers)
        }


class CohesiveAgentDeployment:
    """
    Safe deployment wrapper for Async Manager with external peer support.
    Handles graceful shutdown, error recovery, and peer connections.
    """
    
    def __init__(self, 
                 memory_name: str,
                 filename: str,
                 target_title: str,
                 label_name: str,
                 security_level: str = "PRODUCTION",
                 enable_peers: bool = True,
                 trusted_networks: list = None,
                 secret_key: str = None,
                 peer_discovery_port: int = 5555,
                 shared_auth_token: str = None,
                 predict_manager: Any=None,
                 peer_config: Any='peer_config.json',
                 consecutive_peer_config: Any=None
                 ):
        super().__init__()

        if hasattr(self, '_singleton_initialized'):
            print(f"[===] CohesiveAgentDeployment already initialized, reusing...")
            return
        
        self._singleton_initialized = True
        
       
        self._init_params = {
            'memory_name': memory_name,
            'port': peer_discovery_port,
            'secret_key': secret_key,
            'trusted_networks':trusted_networks,
            'shared_auth_token': shared_auth_token
        }  
        
        self.pipeline = IntegratedPipeline(
            memory_name=memory_name,
            use_async=True,
            agent_port=peer_discovery_port,
            ssl_cert_file=None,  
            ssl_key_file=None,
            secret_key=secret_key,
            shared_auth_token=shared_auth_token,
            predict_manager=predict_manager
        )
        
        # Initialize prediction manager
        self.manager = PipelinePredictionManager(
            self.pipeline,
            label_csv=filename,
            target_title=target_title,
            label=label_name
        )

        self._peer_agent = ConsecutivePeerAgent(
            peer_id=self.pipeline.memory_name,
            port=peer_discovery_port + 100,  # Different port to avoid conflict
            secret_key=secret_key,
            manager=self.manager,
            pipeline=self.pipeline
        ) 

        # Map security level string to enum
        security_map = {
            "DEVELOPMENT": SecurityLevel.DEVELOPMENT,
            "STAGING": SecurityLevel.STAGING,
            "PRODUCTION": SecurityLevel.PRODUCTION,
            "HARDENED": SecurityLevel.HARDENED
        }
        
        # Create Async Manager with security
        self.async_manager = PipelineAsyncManager(
            pipeline=self.pipeline,
            prediction_manager=self.manager,
            security_level=security_map.get(security_level, SecurityLevel.PRODUCTION),
            api_key=shared_auth_token,
            max_workers=4,
            task_timeout=30,
            max_retries=3
        )

        self.peer_config_name = peer_config
        self.consecutive_peer_config = consecutive_peer_config
        if shared_auth_token:
            # Set for distribution (peer authentication)
            self.pipeline.distribution.auth_token = shared_auth_token
            self.pipeline.distribution.secret_key = shared_auth_token
            
            # Set for async manager (API key for predictions)
            self.async_manager._default_api_key = shared_auth_token
            self.async_manager.api_key_manager.keys = {}  # Reset
            self.async_manager.api_key_manager.generate_key(
                {'type': 'shared', 'source': 'cluster'},
                key_value=shared_auth_token  # Need to modify generate_key to accept value
            )
            
            print(f"[🔑] Using shared auth token for entire cluster") 


        self.enable_peers = enable_peers
        self.peer_discovery_port = peer_discovery_port
        self._shutdown_event = asyncio.Event()
        self._peer_tasks = []
        self._known_peers = {}
        self.identified_peers = []

        self.attempt = 0
        self.max_attempts = 3
        
        self.result_queue = AsyncResultQueue(max_size=1000)
        self.worker_pool = WorkerPool(self.result_queue, num_workers=4) 
        
        # Discovery security settings
        self.discovery_secret = os.environ.get('DISCOVERY_SECRET', 'default_secret_change_me')
        self.discovery_enabled = True
        self.discovery_broadcast_only_trusted_network = True
        self.trusted_networks = trusted_networks  # Only respond to these networks
        self.discovery_rate_limit = 5  # Max 5 discovery responses per minute per IP
        self._discovery_requests = defaultdict(list)  # Track request rates
        self.local_ips = self._get_local_ips()  # Get local IPs for discovery filtering
        self._connecting_to = set()
        self.consecutive_peer_config = consecutive_peer_config if consecutive_peer_config else "consecutive_peers.json"

    def _get_local_ips(self) -> List[str]:
        # Get all local IP addresses for this machine
        ips = set()
        try:
            # Get hostname IP
            ips.add(socket.gethostbyname(socket.gethostname()))
            
            # Get all network interfaces
            hostname = socket.gethostname()
            for ip in socket.gethostbyname_ex(hostname)[2]:
                ips.add(ip)
            
            # Add localhost
            ips.add('127.0.0.1')
            
        except Exception as e:
            logger.warning(f"[-] Could not get local IPs: {e}")
            ips.add('127.0.0.1')
        
        return list(ips)
        
    def _is_trusted_network(self, client_ip: str) -> bool:
        # if client IP is from trusted network
        import ipaddress
        
        try:
            client = ipaddress.ip_address(client_ip)
            for network in self.trusted_networks:
                if client in ipaddress.ip_network(network):
                    return True
        except:
            pass
        return False
    
    def _check_discovery_rate_limit(self, client_ip: str) -> bool:
        # Rate limit discovery requests
        now = time.time()
        # Clean old requests
        self._discovery_requests[client_ip] = [
            t for t in self._discovery_requests[client_ip] 
            if now - t < 60  # Keep last minute
        ]
        
        if len(self._discovery_requests[client_ip]) >= self.discovery_rate_limit:
            logger.warning(f"[=] Discovery rate limit exceeded for {client_ip}")
            return False
        
        self._discovery_requests[client_ip].append(now)
        return True
    
    def _create_discovery_response(self) -> dict:
        # a secure discovery response (minimal info)
        return {
            'type': 'DISCOVERY_RESPONSE',
            'version': '1.0',
            'port': self.peer_discovery_port,
            'requires_auth': True,  # Don't reveal agent_id or capabilities
            'timestamp': time.time()
        }    


    async def start(self, bootstrap_token: str = None):
        # Start the agent with all components
        
        logger.info("[🚀] Starting Safe Agent Deployment...")
        
        # 1. Start Async Manager
        success = self.async_manager.start(bootstrap_token=bootstrap_token)
        if not success:
            raise RuntimeError("[-] Failed to start Async Manager")
        
        logger.info("[✅] Async Manager started")
        
        # 2. Start distributed inference (for peer connections)
        if self.enable_peers:
            # Start the server to listen for peer connections
            self.pipeline.distribution.start_server()
            logger.info(f"[✅] Peer server listening on port {self.peer_discovery_port}")
            
            # Start peer discovery if needed
            await self._start_peer_discovery()
        
            asyncio.create_task(self._health_monitor())
           
        # Start result queue and workers
        await self.result_queue.start()
        await self.worker_pool.start(self._prediction_worker)
        
        # 4. Start health monitoring loop
        asyncio.create_task(self._health_monitor())

        logger.info("[🎉] Agent fully operational!")
        self._print_status()
        
        return True
        
    async def _prediction_worker(self, texts: list, api_key: str = None, client_ip: str = None) -> dict:
        # Worker function for processing predictions
        # This runs in a thread pool via asyncio.to_thread
        return self.async_manager.predict(
            texts=texts,
            timeout=40,
            retries=None,
            api_key=api_key,
            client_ip=client_ip,
            method='advanced'
        )

    async def _start_peer_discovery(self):
        # Discover and connect to peer agents safely
        
        #  Connect to known peers from config file
        known_peers = self._load_known_peers()
        
        for peer_host, peer_port in known_peers:
            try:
                try:
                    await self._connect_to_peer(peer_host, peer_port)
                except:
                    if self.peer_discovery_broadcast:
                        await self._discover_local_peers()
        
                    if self.peer_discovery_broadcast:
                        self._discovery_task = asyncio.create_task(self._broadcast_discovery())    

            except Exception as e:
                logger.error(f"[❌] Peer connection error {peer_host}:{peer_port} - {e}")
    
    def _load_known_peers(self):
        # Load known peers from config file

        print(f'[==] Loading known peers from config: {self.peer_config_name}')
        config_file = self.peer_config_name
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
                return config.get('known_peers', [])
        
        # Default peers (can be replaced with other IPs)
        return [
            ('127.0.0.1', 5555),  # Example peer
            ('127.0.0.1', 5556)
        ]
        
    async def _discover_local_peers(self):
        # Discover peers on local network via port scanning
        logger.info("🔍 Scanning for local peers...")
        
        # Scan common ports
        for port in range(self.peer_discovery_port, self.peer_discovery_port + 5):
            if port == self.peer_discovery_port:
                continue  # Skip self
                
            for ip in self.local_ips[:3]:  # Limit to first few IPs to avoid long scan
                if ip == '127.0.0.1':
                    continue
                    
                await self._connect_to_peer(ip, port)
    
    async def _broadcast_discovery(self):
        # broadcast discovery message to find peers on network
        logger.info("📡 Starting broadcast discovery...")
        
        while not self._shutdown_event.is_set() and self.discovery:
            try:
                # UDP broadcast socket
                if self.pipeline.distribution.enable_ssl and self.pipeline.distribution.ssl_context:
                    sock = self.pipeline.distribution.ssl_context.wrap_socket(socket.socket(socket.AF_INET, socket.SOCK_DGRAM))
                else:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                            
                print(f"[broadcast_discovery() SOCKET CREATED] id={id(sock)}")                
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                sock.settimeout(2)
                
                # Broadcast discovery message
                discovery_msg = json.dumps({
                    'type': 'DISCOVERY',
                    'agent_id': id(self.pipeline.distribution),
                    'port': self.peer_discovery_port,
                    'timestamp': time.time()
                }).encode()
                
                # Adding signature to prevent spoofing
                signature = self._sign_message(discovery_msg)
                discovery_msg['signature'] = signature

                sock.sendto(discovery_msg, ('<broadcast>', self.peer_discovery_port))
                
                # Listen for responses
                try:
                    data, addr = sock.recvfrom(1024)
                    client_ip = addr[0]
                    client_port = addr[1]

                    if client_ip in ['127.0.0.1', 'localhost']:
                        if client_port == self.peer_discovery_port:
                            print(f"[=] Ignoring self-discovery response")
                            continue  

                    # Security checks before processing response
                    if not self._is_trusted_network(client_ip):
                        logger.debug(f"[==] Ignoring discovery from untrusted network: {client_ip}")
                        continue
                    
                    if not self._check_discovery_rate_limit(client_ip):
                        continue    

                    response = json.loads(data.decode())
                    # Verify signature
                    if not self._verify_signature(response):
                        logger.warning(f"[=-=] Invalid discovery response signature from {client_ip}")
                        continue   

                    if response.get('type') == 'DISCOVERY_RESPONSE':
                        logger.info(f"✅ Received discovery response from {client_ip}")
                        peer_host = addr[0]
                        peer_port = response.get('port')
                        await self._connect_to_peer(peer_host, peer_port)
                except socket.timeout:
                    pass
                
                sock.close()
                
            except Exception as e:
                logger.debug(f"Broadcast discovery error: {e}")
            
            # Wait before next broadcast
            await asyncio.sleep(60)
    
    def _sign_message(self, message: dict) -> str:
        # Sign message with HMAC to prevent spoofing
       
        # Sort keys for consistent serialization
        message_str = json.dumps(message, sort_keys=True)
        return hmac.new(
            self.discovery_secret.encode(),
            message_str.encode(),
            hashlib.sha256
        ).hexdigest()  
        
    def _verify_signature(self, message: dict) -> bool:
        # Verify message signature
        if 'signature' not in message:
            return False
        
        signature = message.pop('signature')
        expected = self._sign_message(message)
        message['signature'] = signature
        
        return hmac.compare_digest(signature, expected)


    async def _connect_to_peer(self, host: str, port: int) -> bool:
        # Connect to a peer agent
        try:
            # Check if already connected    
            # Store peer info for reconnection
            peer_key = f"{host}:{port}"

            #  ✅ Prevent multiple simultaneous connection attempts to same peer
            if peer_key in self._connecting_to:
                print(f"[⚠️] Already connecting to {peer_key}, skipping")
                return False     

            self._connecting_to.add(peer_key)  

            for agent_id, info in self.pipeline.distribution.remote_agents.items():
                if info.get('host') == host and info.get('port') == port:
                    logger.debug(f"[=+=] Already connected to {host}:{port}")
                    return True
            
            logger.info(f"🔗 Connecting to peer {host}:{port}")
            
            # Use the distribution system to connect
            sock = self.pipeline.distribution.connect_to_agent(host, port)
            
            if sock:
                logger.info(f"✅ Connected to peer {host}:{port}")

                self._known_peers[peer_key] = {
                    'host': host,
                    'port': port,
                    'sock': sock,
                    'last_seen': datetime.now(),
                    'connected': True
                }
                
                # Start background task to handle peer messages
                task = asyncio.create_task(
                    self._handle_peer_communication(host, port, sock)
                )
                self._peer_tasks.append(task)
                return True
            else:
                logger.warning(f"[❌] Failed to connect to {host}:{port}")
                return False
                
        except Exception as e:
            logger.error(f"[-] Peer connection error {host}:{port} - {e}")
            return False
            


    async def _handle_peer_communication(self, peer_host: str, peer_port: int, sock):
        # Handle bidirectional communication with a peer
        logger.info(f"📡 Peer communication active for {peer_host}:{peer_port}")
        
        try:
            while not self._shutdown_event.is_set():
                # The distribution system handles message receiving internally
                # This task just monitors connection health
                await asyncio.sleep(5)
                
                # Send heartbeat to check connection
                try:
                    # self.pipeline.distribution._send_message(
                        # sock, {'type': 'PING', 'timestamp': time.time()}
                   # )
                   sock.getpeername()
                   print(f'[==] Peer name: {sock.getpeername()}')
                except:
                    logger.warning(f"[-] Peer {peer_host}:{peer_port} disconnected")
                    break
                
        except asyncio.CancelledError:
            logger.info(f"[-] Peer communication cancelled for {peer_host}:{peer_port}")
        except Exception as e:
            logger.error(f"[-] Peer communication error: {e}")
        finally:
            # Update peer status
            peer_key = f"{peer_host}:{peer_port}"
            if peer_key in self._known_peers:
                self._known_peers[peer_key]['connected'] = False
            sock.close()

    
    async def _peer_health_monitor(self):
        # Monitor peer health and reconnect if needed
        logger.info("[💓] Peer health monitor started")
        
        while not self._shutdown_event.is_set():
            await asyncio.sleep(30)
            
            try:
                # Ping all connected peers
                alive_agents = self.pipeline.distribution.broadcast_ping()
                logger.info(f"[=+=] Connected peers: {len(alive_agents)}")
                
                # Reconnect to known peers that went offline
                for peer_key, peer_info in self._known_peers.items():
                    if not peer_info.get('connected', False):
                        logger.info(f"[==] Attempting to reconnect to {peer_key}")
                        await self._connect_to_peer(peer_info['host'], peer_info['port'])
                        
            except Exception as e:
                logger.error(f"[❌] Peer health monitor error: {e}")
       
    
    async def _health_monitor(self):
        # Background health monitoring
        while not self._shutdown_event.is_set():
            await asyncio.sleep(30)
            
            try:
                stats = self.async_manager.get_stats()
                logger.info(f"[==] Health Check - Stats: {stats}")
                
                # Check if we need to reconnect peers
                if self.enable_peers:
                    alive_agents = self.pipeline.distribution.broadcast_ping()
                    logger.info(f"[=+=] Connected peers: {len(alive_agents)}")
                    
            except Exception as e:
                logger.error(f"[❌] Health monitor error: {e}")
                
    def save_peer_config(self, peers: List[tuple]):
        """Save peer configuration to file"""
        config = {'known_peers': peers}
        with open('peer_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"[==] Saved {len(peers)} peers to config")    

        
    def _print_status(self):
        print("\n" + "="*70)
        print("=== 🤖 COHESIVE INTEGRATED PIPELINE - STATUS ===")
        print("="*70)
        print(f"📊 State: {self.async_manager.state}")
        print(f"🔒 Security Level: {self.async_manager.security_level.value}")
        print(f"🌐 Peers Enabled: {self.enable_peers}")
        print(f"🔗 Connected Peers: {len(self.pipeline.distribution.remote_agents)}")
        print(f"📡 Peer Port: {self.peer_discovery_port}")
        print(f"🖥️  Local IPs: {', '.join(self.local_ips)}")
        print(f"⏳ Queue Size: {self.async_manager._stats['queue_size']}")
        print(f"🔑 API Key Required: {self.async_manager.config.require_api_key}")
        if self.async_manager.config.require_api_key:
            print(f"🔑 Default API Key: {getattr(self.async_manager, '_default_api_key', 'N/A')[:20]}...")
        
        # Show connected peers
        if self.pipeline.distribution.remote_agents:
            print("\n📡 Connected Peers:")
            for agent_id, info in self.pipeline.distribution.remote_agents.items():
                print(f"   → {info.get('host', 'unknown')}:{info.get('port', 'unknown')} (trust: {info.get('trust', 1.0):.2f})")
        
        print("="*70)
    
    def get_peers_status(self) -> Dict:
        """Get detailed status of all peers"""
        return {
            'connected_peers': len(self.pipeline.distribution.remote_agents),
            'known_peers': self._known_peers,
            'remote_agents': {
                agent_id: {
                    'host': info.get('host'),
                    'port': info.get('port'),
                    'trust': info.get('trust', 1.0)
                }
                for agent_id, info in self.pipeline.distribution.remote_agents.items()
            }
        }
    

    # ============ PREDICTION METHODS ============
    async def multi_modal_peer_ensemble_prediction(self, texts, api_key: str = None, method: str = 'advanced', disable_sync: bool=False) -> dict:
        """
        Robust prediction: try main system first, fallback to SecurePeerAgent.
        """
        try:
            # Try main prediction with timeout
            if not self.pipeline.autonomous:
                print('[==] Initiating Autonomous ensemble prediction...')
                self.pipeline.ensemble.explainer.supervised_learning = False
                self.pipeline.autonomous = True

            result = await asyncio.wait_for(
                self.predict_with_peers(texts, api_key, method, disable_sync=disable_sync),
                timeout=60.0
            )
            
            # Check if result is valid
            if result and result.get('confidence', 0) > self.pipeline.confidence_threshold and result.get('peer_count') > 0:
                return result
            
            # Low confidence, try fallback
            print("[=] Initiating Consecutive peer ensemble...")
            return await self.predict_with_peer_consecutive(texts, api_key, method)
            
        except (asyncio.TimeoutError, Exception) as e:
            print(f"[=] Main prediction failed: {e}, using consecutive ensemble...")
            return await self.predict_with_peer_consecutive(texts, api_key, method)

    def _load_consecutive_known_peers(self):
        """Load peers for fallback using different ports"""
        config_file = self.consecutive_peer_config
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
                return config.get('known_peers', [])
        
        return [
            ('127.0.0.1', 5656),
            ('127.0.0.1', 5655)
        ]
    
    async def predict_with_peer_consecutive(self, texts, api_key: str = None, method: str = 'advanced') -> dict:
        """
        Fallback prediction using SecurePeerAgent when main system fails.
        """
        print("[=] Using Secure Peer Agent fallback...")
        

        if not self._peer_agent.running:
            self._peer_agent.start_server()
        
        # Extract text
        # Get peer addresses from config
        peer_addresses = self._load_consecutive_known_peers()
        print(f'[===] Peer addresses: {peer_addresses}')
        
        # Ensemble prediction
        result = await self._peer_agent.ensemble_predict(
            peer_addresses=peer_addresses,
            text=texts,           
            confidence_threshold=self.pipeline.confidence_threshold
        )

        
        return {
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'source': result.get('source', 'unknown'),
            'fallback': True
        }


    async def predict_with_peers(self, texts, api_key: str = None, method: str = 'advanced', disable_sync: bool=False) -> dict:
        """
        Simple peer prediction: Connect to peers first, then get predictions.
        """
        print("[=+=] Starting peer-augmented prediction")
        
        try:
            if not disable_sync:
                local_result = self.predict_sync(texts, api_key, method=method)
                print(f'[==] Local prediction Result: {local_result.get("prediction")} ({local_result.get("confidence", 0):.1%})')
            else:
                local_result = {'prediction': None, 'confidence': 0.0}

            connection = await self._ensure_peer_connections(api_key)

            print(f'[=] Peer connection ensured: {connection}')
                     
            peers = []
            for agent_id, info in list(self.pipeline.distribution.remote_agents.items()):
                if agent_id != 'local' and str(agent_id) != str(id(self)):
                    sock = info.get('sock')
                    if sock:
                        try:
                            sock.getpeername()
                            peers.append(agent_id)
                            print('[=+=] Socket is alive!')
                        except Exception as e:
                            print('[=] Socket is not available')
                            pass
                    else:
                        print('[=] No socket is available')
                else:
                    print(f'[=^=] peer in sight: {self.pipeline.distribution.remote_agents}')
            
            print(f'[=+=] Connected peers: {len(peers)}') 

            confidence_threshold = getattr(self.pipeline, 'confidence_threshold', 0.6)
            if not peers or local_result.get('confidence', 0) >= confidence_threshold:
                print(f'[==] Using local result (confidence: {local_result.get("confidence", 0):.1%})')
                return local_result
            
            print(f'[=/=] Asking {len(peers)} peers...')
            
            peer_results = []
            for agent_id in peers:
                try:
                    result = await self._ask_peer_simple(agent_id, texts)
                    if result:
                        peer_results.append(result)
                        print(f'[/==] Peer {agent_id} result: {result.get("prediction")} ({result.get("confidence", 0):.1%})')
                except Exception as e:
                    print(f'[/=-] Peer {agent_id} failed: {e}')
            
            if peer_results:
                best = max(peer_results, key=lambda x: x.get('confidence', 0))
                best_conf = best.get('confidence', 0)
                local_conf = local_result.get('confidence', 0)
                
                print(f'[==] Local: {local_conf:.1%}, Best peer: {best_conf:.1%}')
                
                if best_conf > local_conf:
                    print(f'[/==] Using peer result: {best.get("prediction")}')
                    return best
            
            return local_result
            
        except Exception as e:
            print(f"[=] Peer prediction failed: {e}")
            traceback.print_exc()
            return self.predict_sync(texts, api_key, method='basic')
            
    async def _ask_peer_simple(self, agent_id, texts):
        """
        Simple request to a single peer.
        """
        info = self.pipeline.distribution.remote_agents.get(agent_id)
        if not info:
            return None
        
        sock = info.get('sock')
        if not sock:
            return None
        
        # Prepare message
        print('[==] Preparing Message...')
        if isinstance(texts, dict) and 'test_titles' in texts:
            message = {
                'type': self.pipeline.distribution.MSG_TYPES['PREDICT_REQUEST'],

                'payload': {
                    'test_titles': texts.get('test_titles'),
                    'label_map': texts.get('label_map'),
                    'rules': texts.get('rules'),
                    'use_transformer': texts.get('use_transformer', True)
                },
                'token': self.get_api_key()
            }
        else:
            text = texts[0] if isinstance(texts, list) else str(texts)
            message = {
                'type': self.pipeline.distribution.MSG_TYPES['PREDICT_REQUEST'],
                'text': text,
                'token': self.get_api_key(),
                'timestamp': time.time()
            }
        
        try:
            sock.settimeout(10)
            # Add this before sending
            try:
                sock.getpeername()  # Test if socket is still alive
                print('[=] Socket still present!')
            except:
                print(f"[=] Socket to {agent_id} is dead")
                return None   

            self.pipeline.distribution._send_message(sock, message)

            print('[==] Successfully send prediction message!')
            response = self.pipeline.distribution._receive_message(sock)
            sock.settimeout(20)
            
            if response and response.get('type') == 2:
                print(f'[=+=] Got response from peer: {response}')
                return {
                    'prediction': response.get('prediction'),
                    'confidence': response.get('confidence', 0)
                }
            else:
                print('[-] No response from peer.')
            return None
            
        except Exception as e:
            print(f'[=] Error asking peer {agent_id}: {e}')
            return None


    def _is_server_listening(self) -> bool:
        # if the server is actually listening on its port
        if self.pipeline.distribution.enable_ssl and self.pipeline.distribution.ssl_context:
            sock = self.pipeline.distribution.ssl_context.wrap_socket(socket.socket(socket.AF_INET, socket.SOCK_STREAM))
        else:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                
        sock.settimeout(1)
        try:
            result = sock.connect_ex(('127.0.0.1', self.peer_discovery_port))
            sock.close()
            listening = result == 0
            print('[=+=] Server is listening!')
            return listening
        except:
            return False

    async def _ensure_peer_connections(self, api_key: str = None):
        """
        Robust peer connection manager - prevents duplicate connections and WinError.
        """
        print("[=] Ensuring peer connections...")
        
        # ✅ Step 1: Clean up dead connections first
        dead_connections = []
        for agent_id, info in list(self.pipeline.distribution.remote_agents.items()):
            if agent_id == 'local':
                continue
            
            sock = info.get('sock')
            if sock is None:
                dead_connections.append(agent_id)
                continue
            
            # Test if socket is still alive
            try:
                sock.getpeername()
            except:
                print(f"[=] Dead connection detected: {agent_id}")
                dead_connections.append(agent_id)
        
        # Remove dead connections
        for agent_id in dead_connections:
            print(f"[=] Removing dead connection: {agent_id}")
            try:
                del self.pipeline.distribution.remote_agents[agent_id]
            except:
                pass
        
        # ✅ Step 2: Load known peers from config
        known_peers = self._load_known_peers()
        
        if not known_peers:
            print("[=] No known peers configured")
            return False
        
        # ✅ Step 3: Try each peer once, no retry loops
        successful = False
        
        for host, port in known_peers:
            peer_key = f"{host}:{port}"
            
            # Skip self
            if host in ['127.0.0.1', 'localhost'] and port == self.peer_discovery_port:
                print(f"[=] Skipping self: {peer_key}")
                continue
            
            # Check if already connected (and alive)
            already_connected = False
            for agent_id, info in self.pipeline.distribution.remote_agents.items():
                if info.get('host') == host and info.get('port') == port:
                    sock = info.get('sock')
                    if sock:
                        try:
                            sock.getpeername()
                            print(f"[=] Already connected to {peer_key}")
                            already_connected = True
                            successful = True
                            break
                        except:
                            # Socket dead, will reconnect
                            pass
            
            if already_connected:
                continue
            
            # ✅ Step 4: Single connection attempt (NO RETRY)
            print(f"[=] Connecting to {peer_key}...")
            
            try:
                # Use add_peer with timeout
                result = await self._connect_single_attempt(host, port, api_key)
                
                if result:
                    print(f"[=] ✅ Connected to {peer_key}")
                    successful = True
                else:
                    print(f"[=] ❌ Failed to connect to {peer_key}")
                    
            except Exception as e:
                print(f"[=] ❌ Error connecting to {peer_key}: {e}")
        
        return successful


    async def _connect_single_attempt(self, host, port, api_key, timeout=5):
        """
        Single connection attempt - no retries, no loops.
        """
        try:
            # Check if already connected (one more time)
            for agent_id, info in self.pipeline.distribution.remote_agents.items():
                if info.get('host') == host and info.get('port') == port:
                    sock = info.get('sock')
                    if sock:
                        try:
                            sock.getpeername()
                            return True
                        except:
                            pass
            
            # Single connection attempt with timeout
            result = await asyncio.wait_for(
                asyncio.to_thread(self.add_peer, host, port, api_key),
                timeout=timeout
            )
            
            # Verify connection is alive
            await asyncio.sleep(0.1)  # Give it a moment
            
            for agent_id, info in self.pipeline.distribution.remote_agents.items():
                if info.get('host') == host and info.get('port') == port:
                    sock = info.get('sock')
                    if sock:
                        try:
                            sock.getpeername()
                            return True
                        except:
                            pass
            
            return result
            
        except asyncio.TimeoutError:
            print(f"[=] Connection timeout to {host}:{port}")
            return False
        except Exception as e:
            print(f"[=] Connection error to {host}:{port}: {e}")
            return False


    async def _request_peer_prediction_async(self, agent_id, texts):
        """Async peer prediction request"""
        try:
            # Use async version
            return await self.pipeline.distribution.request_prediction_async(agent_id, texts, timeout=5)
        except Exception as e:
            logger.warning(f"[=-] Peer {agent_id} failed: {e}")
            return None

    def _ensemble_predictions(self, local: dict, peers: list) -> dict:
        # Combine predictions from multiple agents
        try:
            print('[=+=] Initiating Ensemble weighting with: {peers} peers total')
            votes = defaultdict(float)
            votes[local.get('prediction', 'unknown')] += local.get('confidence', 0)
            
            for peer in peers:
                if peer and isinstance(peer, dict):
                    votes[peer.get('prediction', 'unknown')] += peer.get('confidence', 0)
            
            best_pred = max(votes.items(), key=lambda x: x[1])
            
            total_weight = len(peers) + 1
            return {
                'prediction': best_pred[0],
                'confidence': min(best_pred[1] / total_weight, 1.0),
                'local_prediction': local.get('prediction'),
                'local_confidence': local.get('confidence'),
                'peer_count': len(peers),
                'ensemble_votes': dict(votes)
            }
        except Exception as e:
            print(f'[-] Error in ensemble weighting; {e}, returning local prediction with 0.0 confidence')
            return {
                'prediction': local,
                'confidence': 0.0,
                'peer_count': 0.0,
            }
     

    async def predict_batch_async(self, texts: List[str], api_key: str = None, client_ip: str = None) -> List[dict]:
        """
        Batch async predictions - runs in parallel!
        """
        tasks = [
            self.predict_async(text, api_key, client_ip)
            for text in texts
        ]
        
        # Run all predictions concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        output = []
        for text, result in zip(texts, results):
            if isinstance(result, Exception):
                output.append({
                    'text': text,
                    'prediction': 'error',
                    'confidence': 0.0,
                    'error': str(result)
                })
            else:
                output.append({
                    'text': text,
                    'prediction': result.get('prediction'),
                    'confidence': result.get('confidence', 0),
                    **result
                })
        
        return output 


    def predict_sync(self, texts: Any, api_key: str = None, client_ip: str = None, method: str = 'advanced') -> dict:
        """
        Synchronous prediction with security.
        Use this for simple, blocking calls.
        """
        # ✅ Direct prediction without async queue
        print('[==] Initiating predict sync...')
        try:
            if method == 'advanced':
                test_titles = texts['test_titles']
                label_map = texts['label_map']
                rules = texts['rules']

                result, chosen_label, confidence = self.manager.advanced_prediction_method(
                    test_titles,
                    label_map,
                    rules,
                    show_proba=True,
                    use_transformer=self.pipeline.use_transformer
                )
                return {
                    'prediction': chosen_label,
                    'confidence': confidence,
                    'result': result
                }
            else:
                # Basic prediction
                text = texts[0] if isinstance(texts, list) and texts else str(texts)
                result = self.pipeline.predict_single(text)
                return result
                        

        except Exception as e:
            logger.error(f"[-] Prediction failed: {e}")
            print(f"[-] Prediction failed: {e}")
            traceback.print_exc()
            return {
                'prediction': 'error',
                'confidence': 0.0,
                'error': str(e)
            }
    
    async def predict_async(self, texts, api_key: str = None, client_ip: str = None) -> dict:
        """
        Asynchronous prediction.
        Use this for non-blocking operations.
        """
        try:
            # Submit request to queue
            request_id = await self.result_queue.submit(
                texts=texts,
                api_key=api_key,
                client_ip=client_ip,
            )
            
            # Wait for result with timeout
            result = await self.result_queue.wait_for_result(
                request_id=request_id,
                timeout=30
            )
            
            return result
            
        except TimeoutError:
            logger.error(f"[-] Async prediction timed out for: {texts}")
            return {
                'prediction': 'timeout',
                'confidence': 0.0,
                'error': 'Request timeout'
            }
        except Exception as e:
            logger.error(f"[-] Async prediction failed: {e}")
            traceback.print_exc()
            return {
                'prediction': 'error',
                'confidence': 0.0,
                'error': str(e)
            }
            
    def get_queue_stats(self) -> Dict:
        # Get result queue statistics
        logger.info("[=] Fetching result queue stats...")
        return self.result_queue.get_status(request_id=None)

     
    # ============ PEER MANAGEMENT ============
    
    def add_peer(self, host: str, port: int, api_key: str = None):
        # Manually add a peer connection
        if not api_key:
            agent_id = f"{host}:{port}"
            if hasattr(self.pipeline.distribution, 'peer_tokens'):
                api_key = self.pipeline.distribution.peer_tokens.get(agent_id)
        else:
            self.pipeline.distribution.add_trusted_agent(f"{host}:{port}", api_key)
        
        # Connecting
        sock = self.pipeline.distribution.connect_to_agent(host, port)
        if host in ['127.0.0.1', 'localhost', '0.0.0.0']:
            if port == self.pipeline.distribution.port or port == 0:
                print(f"[❌] Cannot add self as peer ({host}:{port})")
                return False        

        if sock:
            # Store in known peers
            peer_key = f"{host}:{port}"
            self._known_peers[peer_key] = {
                'host': host,
                'port': port,
                'sock': sock,
                'last_seen': datetime.now(),
                'connected': True
            }
            
            # Start communication task
            task = asyncio.create_task(
                self._handle_peer_communication(host, port, sock)
            )
            self._peer_tasks.append(task)
            
            logger.info(f"✅ Manually added peer {host}:{port}")
            return True
        
        logger.error(f"[-] Failed to add peer {host}:{port}")
        return False
    
    def remove_peer(self, host: str, port: int):
        # Remove a peer connection
        peer_key = f"{host}:{port}"
        
        # Find and disconnect
        for agent_id, info in list(self.pipeline.distribution.remote_agents.items()):
            if info.get('host') == host and info.get('port') == port:
                self.pipeline.distribution.disconnect_agent(agent_id)
                break
        
        # Remove from known peers
        if peer_key in self._known_peers:
            del self._known_peers[peer_key]
        
        logger.info(f"[-] Removed peer {host}:{port}")
    
    def list_peers(self) -> List[Dict]:
        # List all connected peers
        peers = []
        for agent_id, info in self.pipeline.distribution.remote_agents.items():
            if agent_id == 'local':
                continue

            if info.get('port') == 0 or info.get('port') == self.pipeline.distribution.port:
                continue
            if info.get('host') in ['localhost', '127.0.0.1', '0.0.0.0']:
                if info.get('port') == self.pipeline.distribution.port:
                    continue        

            peers.append({
                'agent_id': agent_id,
                'host': info.get('host'),
                'port': info.get('port'),
                'trust': info.get('trust', 1.0),
                'last_seen': info.get('last_seen', datetime.now()).isoformat()
            })

        return peers 

    async def _connect_with_smart_retry(self, agent, host, port, api_key, max_retries=3, delay=1):
        """
        Smart connection with retry - STOPS once connected.
        """
        
        for attempt in range(max_retries):
            # ✅ Check if already connected BEFORE attempting
            existing_peers = agent.list_peers()
            for peer in existing_peers:
                if peer.get('host') == host and peer.get('port') == port:
                    print(f"[/==] Already connected to {host}:{port}, skipping retry")
                    return True
            
            print(f"[/==] Attempt {attempt + 1}/{max_retries}: Connecting to {host}:{port}...")
            
            try:
                # Try to connect
                if asyncio.iscoroutinefunction(agent.add_peer):
                    result = await agent.add_peer(host, port, api_key)
                else:
                    result = agent.add_peer(host, port, api_key)
                
                if result:
                    # ✅ Verify connection was successful
                    await asyncio.sleep(0.5)  # Give it a moment
                    existing_peers = agent.list_peers()
                    for peer in existing_peers:
                        if peer.get('host') == host and peer.get('port') == port:
                            print(f"[✅] Successfully connected on attempt {attempt + 1}")
                            return True
                    
                    print(f"[⚠️] Connection reported success but peer not found")
                    return True
                    
            except Exception as e:
                print(f"[=/] Attempt {attempt + 1} failed: {e}")
            
            # Don't retry if already connected
            if attempt < max_retries - 1:
                # Check again before waiting
                existing_peers = agent.list_peers()
                if any(p.get('host') == host and p.get('port') == port for p in existing_peers):
                    print(f"[=+=] Already connected, stopping retries")
                    return True
                
                print(f"[===] Waiting {delay}s before retry...")
                await asyncio.sleep(delay)
                delay *= 1.5
        
        return False


    # ============ SHUTDOWN ============
    
    async def shutdown(self):
        # Graceful shutdown of all components
        logger.info("🛑 Shutting down agent...")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Cancel peer tasks
        for task in self._peer_tasks:
            task.cancel()
        
        # Stop distribution server
        if self.enable_peers:
            self.pipeline.distribution.stop_server()
        
        # Stop async manager
        self.async_manager.stop(timeout=10, force=False)
        
        # Wait for cleanup
        await asyncio.sleep(2)
        
        logger.info("✅ Agent shutdown complete")
    
    def get_api_key(self) -> str:
        # Get the default API key (for client distribution)
        return getattr(self.async_manager, '_default_api_key', None)
    
 

# ============ EXAMPLE: SECURE PEER-TO-PEER CLUSTER ============


async def run_secure_agent_cluster(test_titles, label_map, rules, agent_id, filename, title_name, label_name, manager):
    """
    Run multiple agents that securely communicate.
    Stops retrying once connected successfully.
    """
    print("\n" + "="*60)
    print("=== SECURE PEER-TO-PEER CLUSTER ===")
    print("="*60)
    
    # Set discovery secret (in production, use environment variable)
    secret_key = 'my-ultra-safe-secret-key-for-authentication'

    # Agent 1 - Primary (Port 5555)
    agent1 = CohesiveAgentDeployment(
        memory_name="agent_primary",
        filename=filename,
        target_title=title_name,
        label_name=label_name,
        security_level="PRODUCTION",
        enable_peers=True,
        trusted_networks=['127.0.0.1/32', '192.168.1.0/24'],
        peer_discovery_port=5555,
        secret_key=secret_key,
        shared_auth_token=secret_key,
        predict_manager=manager
    )
    
    # Agent 2 - Secondary (Port 5556)
    agent2 = CohesiveAgentDeployment(
        memory_name="agent_secondary",
        filename=filename,
        target_title=title_name,
        label_name=label_name,
        security_level="PRODUCTION",
        enable_peers=True,
        trusted_networks=['127.0.0.1/32', '192.168.1.0/24'],
        peer_discovery_port=5556,
        secret_key=secret_key,
        shared_auth_token=secret_key,
        predict_manager=manager
    )
    
    try:
        # Start both agents
        print("\n🚀 Starting Agent 1...")
        await agent1.start()
        print("✅ Agent 1 started on port 5555")
        
        print("\n🚀 Starting Agent 2...")
        await agent2.start()
        print("✅ Agent 2 started on port 5556")
        
        # Give servers time to fully bind
        await asyncio.sleep(2)
        
        # Get API keys
        api_key = agent1.get_api_key()
        print(f"\n🔑 Using API Key: {api_key[:20]}...")
        
        texts = {"test_titles": test_titles, "label_map": label_map, "rules": rules, "use_transformer": True, "agent_id": agent_id}

        # Make prediction with peer ensemble
        # Peer Connection will be ensured successful during P2P 
        result = await agent1.multi_modal_peer_ensemble_prediction(
            texts=texts,
            api_key=api_key,
            method='advanced',
            disable_sync=True
        )    

        result2 = await agent2.multi_modal_peer_ensemble_prediction(
            texts=texts,
            api_key=api_key,
            method='advanced',
            disable_sync=True
        )      
        
        print(f"\n📊 Ensemble Result for Agent 1:")
        print(f"   Prediction: {result.get('prediction', 'N/A')}")
        print(f"   Confidence: {result.get('confidence', 0):.2%}")

        print(f"   Second Prediction: {result2.get('prediction', 'N/A')}")
        print(f"   Second Confidence: {result2.get('confidence', 0):.2%}")

        # Keep running briefly
        print("\n⏳ Cluster stable. Waiting 5 seconds before shutdown...")
        await asyncio.sleep(5)
        agent2._peer_agent.stop_server()
        
    except Exception as e:
        print(f"\n❌ Error in cluster: {e}")
        traceback.print_exc()
        
    finally:
        print("\n🛑 Shutting down cluster...")
        await agent1.shutdown()
        await agent2.shutdown()
        print("✅ Cluster shutdown complete")




async def example_async_with_result_queue(test_titles, label_map, rules, agent_id, filename, title_name, label_name):
    # Example using the proper result queue
    
    agent = CohesiveAgentDeployment(
        memory_name="test_agent",
        filename=filename,
        target_title=title_name,
        label_name=label_name,
        security_level="DEVELOPMENT",
        enable_peers=False
    )
    
    await agent.start()
    
    api_key = agent.get_api_key()
    payloads = {"test_titles": test_titles, "label_map": label_map, "rules": rules, "use_transformer": True, "agent_id": agent_id}
    
    # Single async prediction
    print('[==] Single sync prediction: (using single text: "Opening Thesis.docx")')
    sync_result = agent.predict_sync(
        texts=payloads,
        api_key=api_key,
        client_ip="127.0.0.1",
        method='advanced'
    )

    print(f"[=] Sync Result: {sync_result}")


    print("[==] Single async prediction: (using single text: Opening Thesis.docx)")
    result = await agent.predict_async(
        texts=payloads,
        api_key=api_key,
        client_ip="127.0.0.1",
    )
    print(f"[=] Result: {result.get('prediction')} ({result.get('confidence', 0)}")
    
    # Batch async predictions (parallel!)
    print("\n[=] Batch async predictions (parallel):")
    texts = [
        "Watching YouTube",
        "Programming in VS Code",
        "Checking Slack messages",
        "Reading documentation",
        "Taking a break"
    ]
    
    start_time = time.time()
    results = await agent.predict_batch_async(texts, timeout=60, api_key=api_key)
    elapsed = time.time() - start_time
    
    for result in results:
        print(f"[=] '{result['text']}' → {result['prediction']} ({result['confidence']:.1%})")
    
    print(f"\n[=] Completed {len(texts)} predictions in {elapsed:.2f}s")
    
    # Get queue stats
    stats = agent.get_queue_stats()
    print(f"[=] Queue stats: {stats}")
    
    await agent.shutdown()




def initiate_cohesive_agent_deployment_test(test_titles, label_map, rules, agent_id, filename, title_name, label_name, manager):
    print("\n" + "="*60)
    print("🔮 = TESTING COHESIVE AGENT DEPLOYMENT WITH ASYNC MANAGER = ")

    print('Test 1 of Multi agent cluster')
    asyncio.run(run_secure_agent_cluster(test_titles=test_titles, label_map=label_map, rules=rules, agent_id=agent_id, filename=filename, title_name=title_name, label_name=label_name, manager=manager))
      
    print("\n1. Basic async with result queue")
    asyncio.run(example_async_with_result_queue(test_titles=test_titles, label_map=label_map, rules=rules, agent_id=agent_id, filename=filename, title_name=title_name, label_name=label_name))
    

# async manager setup examples
def initiate_prediction_usage(pipeline, manager, predict_wrapper, test_titles, label_map, rules):
    """Basic synchronous usage."""
    # Use context manager (auto start/stop)
    api_key = 'my-ultra-safe-secret-key-for-authentication'

    with predict_wrapper as wrapper:
        print('[==] Initiating regular prediction')
        texts = {'test_titles': test_titles, 'label_map': label_map, 'rules': rules, 'use_transformer': True}
        regular_predict = wrapper.predict(
        texts=texts, 
        timeout=40,
        retries=None,
        api_key=api_key,
        client_ip=None)

        print('[==] Initiating advanced batch prediction')
        predicted_output = wrapper.advanced_batch_prediction(test_titles, label_map, rules, api_key, client_ip=None)


def initiate_with_retries(pipeline, manager, wrapper, test_titles, label_map, rules):
    """Example with retry logic."""
    
    try:
        # Will retry up to 5 times
        texts = {"test_titles": test_titles, "label_map": label_map, "rules": rules, "use_transformer": True}
        result = wrapper.predict(texts, timeout=60, retries=None, api_key=None)
        advanced_result, chosen_label, confidence = wrapper.advanced_prediction_method(manager, test_titles, label_map, rules, method='Transformer_included')
        print(f"[=] Result after retries: {result}")
        print(f"[=] Advanced Result: {chosen_label} || ({confidence:.1%})")

    except Exception as e:
        print(f"[!] Failed after retries: {e}")
    finally:
        wrapper.stop()


def initiate_graceful_shutdown(pipeline, wrapper):
    """Example showing graceful shutdown."""
   
    # Submit many async requests
    for i in range(10):
        wrapper.predict_async(f"[=] Request {i}")
    
    # Wait for idle with timeout
    if wrapper.wait_for_idle(timeout=30):
        print("[+] All requests completed")
    else:
        print("[!] Some requests still pending")
    
    # Graceful shutdown
    wrapper.stop(timeout=10)

def AsyncWrappertest(pipeline, prediction_manager, test_titles, label_map, rules):
    print("\n" + "="*60)
    print("🔮 = TESTING ASYNCHRONOUS PREDICTION WRAPPER = ")
    print("="*60)

    api_key = 'my-ultra-safe-secret-key-for-authentication'

    config = SecurityConfig(
            max_text_length=10000,
            max_queue_size=100,
            rate_limit_requests=60,  # 60 per minute
            require_api_key=True,
            max_pending_tasks=50,
            request_timeout=30.0,

            # Start with no IP restrictions, add via admin API
            allowed_ips=[],
            blocklisted_ips=[],
            require_bootstrap_auth = False
        )

    wrapper = PipelineAsyncManager(pipeline, 
              prediction_manager, 
              config=config, 
              state_file=None, 
              security_level=SecurityLevel.PRODUCTION,
              api_key=api_key, 
              max_workers=4, 
              task_timeout=30, 
              max_retries=3 )

    wrapper.start(method='Transformer_included', bootstrap_token=None)
    
    logging.basicConfig(level=logging.INFO)
    
    # Run examples
    initiate_prediction_usage(pipeline, prediction_manager, wrapper, test_titles, label_map, rules)
    initiate_with_retries(pipeline, prediction_manager, wrapper, test_titles, label_map, rules)
    initiate_graceful_shutdown(pipeline, wrapper)

    print("\n✅ Asynchronous prediction wrapper test completed successfully.")


def PermissiveTest():
    print("\n" + "="*60)
    print("🔮 = TESTING HYBRID PREDICTION SYSTEM = ")
    print("="*60)

    print("📖 Loading labels from text file with CSV format...")
    filename = input('|| Insert Filename (press N to skip): ')
    title = input('|| Insert Title name you have in your file (press N to skip): ')
    label = input('|| Insert Label name you have in your file (press N to skip): ')
    agent_id = input('|| Insert Agent ID for distributed inference (press N to skip): ')

    print('📖 Need to insert custom memory Name for the AI')
    file = input('|| Insert Memory name: ')
    print('📖 Need to insert custom SSL certificate and key files for secure communication')
    print('[=] Important for secure external-device Peer to peer between Agents (optional)')

    cert_file = input('|| Insert SSL certificate file (press N to skip): ')
    key_file = input('|| Insert SSL key file (press N to skip): ')
    if cert_file != 'N':
        cert_file = cert_file
    else:
        cert_file = None
    if key_file != 'N':
        key_file = key_file
    else:
        key_file = None

    if file:
        pipeline = IntegratedPipeline(file, use_async=True, agent_port=5001, ssl_cert_file=cert_file, ssl_key_file=key_file)
    else:
        print('|| Using original csv_file.pkl file as fallback...')
        pipeline = IntegratedPipeline('csv_file.pkl', use_async=True, agent_port=5001, ssl_cert_file=cert_file, ssl_key_file=key_file)

    manager = PipelinePredictionManager(pipeline, label_csv='ManualsTraining.txt', target_title='window_title', label='label')

    pipeline.distribution.predict_manager = manager
    if agent_id == 'N':
        agent_id = 'local'

    if filename and title and label and filename != 'N':
        titles, y_raw, label_map = manager.load_labels_from_csv(filename, title, label)
        print(f"✅ Loaded {len(titles)} labeled examples")
    else:
        print('|| Fallback to Original given files...')
        titles, y_raw, label_map = manager.load_labels_from_csv('ManualsTraining.txt', 'window_title', 'label')
        print(f"✅ Loaded {len(titles)} labeled examples")


    print('== Training Model... ==')
    loss_history = pipeline.train(titles, y_raw)

    test_titles = [
    ("Opening Thesis.docx", "slight_work"),
    ("Watching YouTube and Google Chrome", "distracted"),
    ("Watching Slack", "communication"),
    ("Programming in Visual Studio Code", "focused_work"),
    ("Watching netflix.com - Chrome", "break"),
    ]
    rules = [
        # === WORK / PRODUCTIVITY ===
        (r'code|programming|develop|debug|compile|script', 'focused_work'),
        (r'vscode|visual_studio|ide|terminal|shell', 'focused_work'),
        (r'notion|evernote|onenote|notes|todo|task', 'productive'),
        (r'slack|teams|discord|zoom|meeting|call', 'communication'),
        (r'email|gmail|outlook|inbox|mail', 'communication'),
        
        # === ENTERTAINMENT ===
        (r'youtube|netflix|twitch|stream|video', 'entertainment'),
        (r'music|spotify|soundcloud|audio|player', 'entertainment'),
        (r'game|gaming|steam|epic|play', 'gaming'),
        (r'facebook|instagram|tiktok|social|post', 'social_media'),
        
        # === BROWSING ===
        (r'chrome|firefox|edge|safari|browser', 'browsing'),
        (r'google|search|wiki|wiki|article', 'information'),
        (r'stackoverflow|github|docs|documentation', 'research'),
        
        # === FILE MANAGEMENT ===
        (r'download|folder|file|document|pdf', 'file_work'),
        (r'dropbox|onedrive|google_drive|cloud', 'cloud_storage'),
        (r'zip|rar|extract|compress|archive', 'file_management'),
        
        # === SYSTEM / DEV ===
        (r'terminal|cmd|powershell|bash|shell', 'system_work'),
        (r'docker|kubernetes|container|deploy', 'devops'),
        (r'git|commit|push|pull|branch|merge', 'version_control'),
        (r'test|unit|debug|error|exception', 'testing'),
        
        # === DATA / ANALYSIS ===
        (r'excel|spreadsheet|sheet|csv|table', 'data_work'),
        (r'python|r|sql|query|database', 'data_analysis'),
        (r'chart|graph|visualization|dashboard|plot', 'visualization'),
        
        # === COMMUNICATION ===
        (r'whatsapp|telegram|signal|messenger', 'messaging'),
        (r'zoom|meet|webex|video_call', 'video_call'),
        (r'calendar|schedule|event|meeting|appointment', 'scheduling'),
        
        # === CREATIVE ===
        (r'photoshop|illustrator|figma|design|canvas', 'creative'),
        (r'premiere|final_cut|video_edit|render', 'video_editing'),
        (r'blender|3d|model|render|animation', '3d_work'),
        
        # === LEARNING ===
        (r'coursera|udemy|edx|course|learn', 'learning'),
        (r'book|ebook|reader|pdf|document', 'reading'),
        (r'podcast|audiobook|listen|lecture', 'audio_learning'),
        
        # === UTILITY ===
        (r'calculator|converter|tool|utility', 'utility'),
        (r'weather|clock|timer|alarm|reminder', 'utility'),
        (r'translate|language|dictionary|translate', 'utility'),
        
        # === RARITY PATTERNS ===
        (r'common|not_common|twitch|debian|watch', 'very abundant'),
        (r'bit-common|pycharm|unix|code|programming|python|java', 'bit-abundant'),
        (r'medium|discord|teams|zoom|linux_mint|message', 'abundant'),
        (r'rare|pdf|word|macOS|ubuntu|document', 'not abundant'),
        (r'ultra|firefox|edge|browser|unix|web', 'medium rare'),
        (r'ultra_rare|music|linux|Home_linux_router', 'bit-rare'),
        (r'medium-rare|steam|red_hat_enterprise_linux|play|windows', 'very rare'),
        (r'rarer|oracle|system|config|server_linux_router', 'absolute rare'),
    ]

    running = True
    while running:
        permission = input('|| Allow Hybrid prediction test? [Y/N]: ')

        if permission == 'Y' or permission == 'y':
            print('== TEST 1: (titles only without transformer) ==')
            advanced_result = manager.advanced_prediction_method(
            [t[0] for t in test_titles],  # Just titles
            label_map,
            rules,
            show_proba=True
            )
            time.sleep(5)
        
            print('== TEST 2: (advanced predictions with expected labels and also use transformer)')
            advanced_results = manager.advanced_prediction_method(
            test_titles,  # Titles with expected labels
            label_map,
            rules,
            show_proba=True,
            top_k=4,
            use_transformer=True,
            return_attention=True
        
            )
        
            print("\n📊 COMPARISON: MLP-only vs Hybrid")
            mlp_only = manager.regular_prediction_method(
            [t[0] for t in test_titles],
            label_map,
            rules,
            use_transformer=False
            )
        
            hybrid = manager.regular_prediction_method(
            [t[0] for t in test_titles],
            label_map,
            rules,
            use_transformer=True       
            )
            print('== CompletePipeline Successfully tested! ==')

        permission_continue = input('|| Do you want to test the Asynchronous wrapper for multiple predictions? [Y/N]: ')
        if permission_continue == 'Y' or permission_continue == 'y':
            AsyncWrappertest(pipeline, manager, test_titles, label_map, rules)
            print('== Asynchronous wrapper Successfully tested! ==')

        cohesive_permission = input('|| Do you want to test the Cohesive Agent Deployment with Async Manager? [Y/N]: ')
        if cohesive_permission == 'Y' or cohesive_permission == 'y':
            if not (filename and title and label and filename != 'N'):
                print('[=] Searching fallback filename: ManualsTraining.txt, window_title, label')
                initiate_cohesive_agent_deployment_test(test_titles, label_map, rules, agent_id, 'ManualsTraining.txt', 'window_title', 'label', manager)
            else:
                initiate_cohesive_agent_deployment_test(test_titles, label_map, rules, agent_id, filename, title, label, manager)
            print('== Cohesive Agent Deployment Successfully tested! ==')

        else:
            running = False
            print('|| Program Prediction test aborted!')
            pass


if __name__ == "__main__":
    try:
        PermissiveTest()
    except Exception as e:
        print(f'|| Program Crashed...,  Error: {e}')
        traceback.print_exc()
        pass