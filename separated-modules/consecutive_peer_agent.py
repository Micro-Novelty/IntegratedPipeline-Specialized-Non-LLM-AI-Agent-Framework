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



class ConsecutivePeerAgent:
    """
    Lightweight, security-hardened peer agent used as a fallback when the main
    AgentDistributedInference system cannot be initialised or has failed.

    Compared to AgentDistributedInference, this class is intentionally simpler:
    it handles a single peer connection at a time and does not implement async
    queuing or TrustLevel tiers.  Its primary use-case is in
    CohesiveAgentDeployment where a chain of agents needs to exchange
    predictions sequentially.

    Security features
    -----------------
    - All messages are HMAC-SHA256 signed (same sign/verify pattern as
      AgentDistributedInference).
    - A 10 MB message size cap prevents memory exhaustion.
    - allowed_ips whitelist (default: localhost only).
    - Messages with an invalid signature are silently dropped.

    Parameters
    ----------
    peer_id    : Unique string identifier for this agent.
    port       : TCP port this agent listens on.
    secret_key : HMAC key shared with all trusted peers.
    manager    : Optional PipelinePredictionManager for prediction requests.
    pipeline   : Optional IntegratedPipeline for direct model access.
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

        print(f'[ConsecutivePeerAgent] Comparing Signature and verifying...')
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
    
    async def request_peer_prediction(self, peer_host: Any, peer_port: int, text: Any, timeout: float = 5.0) -> Optional[Dict]:
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
        if local_result['confidence'] < confidence_threshold and peer_addresses or peer_addresses:
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
        else:
            print('[ConsecutivePeerAgent] Skipping Ensemble prediction... Peer address is None or empty')
            time.sleep(5)
            return best_result


        self.stats['predictions'] += 1
        return best_result
    
    def start_server(self):
        """Start server to accept peer connections"""
        
        def server_loop():
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind(('0.0.0.0', self.port))

            self.server_socket.settimeout(1.0)           
            self.server_socket.listen(5)

            if self.pipeline.distribution.enable_ssl and self.pipeline.distribution.ssl_context:
                self.server_socket = self.pipeline.distribution.ssl_context.wrap_socket(self.server_socket, server_side=True)

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

                except socket.timeout:
                    continue    

                except Exception as e:
                    if self.running:
                        print(f"[ConsecutivePeerAgent] Server error: {e}")
    
            try:
                self.server_socket.close()
            except:
                pass

        print("[ConsecutivePeerAgent] Server Successfully Stopped listening !")
        
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

        print('[ConsecutivePeerAgent] Initiating Server shutdown...')   
        # Close all peer connections
        try:
            with self._lock:
                for key, info in self.connected_peers.items():
                    try:
                        info['sock'].shutdown(socket.SHUT_RDWR)
                        info['sock'].close()
                    except:
                        pass
                
                self.connected_peers.clear()
                if self.server_socket:
                    try:
                        self.server_socket.close()  
                    except Exception as e:
                        print(f'[ConsecutivePeerAgent] Cant close socket: {e}')
                        pass  
                                
                print('[ConsecutivePeerAgent] Server Successfully Stopped listening !')

        except Exception as e:
            print(f'[ConsecutivePeerAgent] Error closing socket: {e}')
            pass


    def get_stats(self) -> Dict:
        # Get statistics
        return {
            **self.stats,
            'connected_peers': len(self.connected_peers)
        }

    
