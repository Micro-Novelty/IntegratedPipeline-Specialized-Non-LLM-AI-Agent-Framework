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



class CohesiveAgentDeployment:
    """
    Production-grade deployment wrapper that assembles and manages the full
    system stack in a single object.

    What it wires together
    ----------------------
    1. IntegratedPipeline          — core ML pipeline (training + inference).
    2. PipelinePredictionManager   — dataset loading and high-level prediction.
    3. PipelineAsyncManager        — async event loop, API-key auth, rate limits.
    4. AgentDistributedInference   — P2P peer networking (via the pipeline).
    5. ConsecutivePeerAgent        — fallback peer agent for direct connections.

    Lifecycle
    ---------
    __init__  : Constructs all components, loads training data, and initialises
                security (API key, bootstrap token if HARDENED level).
    train()   : Trains the pipeline on the loaded dataset.
    start()   : Starts the async manager and (if enable_peers) the peer server.
    predict() : Routes through the async manager if running, otherwise falls
                back to synchronous prediction.
    stop()    : Gracefully shuts down async manager and peer connections.

    Singleton behaviour
    -------------------
    _singleton_initialized guard prevents double-construction when the same
    instance is reused across call sites.

    Parameters
    ----------
    memory_name            : Logical DB scope for all model artefacts.
    filename               : Path to the label CSV file.
    target_title           : Column name for text inputs in the CSV.
    label_name             : Column name for class labels in the CSV.
    security_level         : One of "DEVELOPMENT", "STAGING", "PRODUCTION",
                             "HARDENED" (maps to SecurityLevel enum).
    enable_peers           : Whether to start the peer discovery server.
    trusted_networks       : List of CIDR strings for the IP allowlist.
    secret_key             : HMAC key for peer message signing.
    peer_discovery_port    : TCP port for AgentDistributedInference server.
    shared_auth_token      : Shared secret for peer authentication.
    predict_manager        : External PipelinePredictionManager (optional;
                             one is created internally if not provided).
    peer_config            : Path to a JSON file with pre-configured peer
                             addresses (host/port pairs).
    consecutive_peer_config: Optional config for ConsecutivePeerAgent setup.
    """
    
    def __init__(self, 
                 pipeline: IntegratedPipeline,    
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
        self.pipeline = pipeline
        self.pipeline.autonomous = True
        
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
        self.timeout = 120
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
            timeout=self.timeout,
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
        # Graceful shutdown of all components
        logger.info("🛑 Shutting down agent...")
        
        # signal shutdown to all loops
        self._shutdown_event.set()

        # stop worker pool.
        if hasattr(self, 'worker_pool'):
            await self.worker_pool.stop()

        if hasattr(self, 'result_queue'):
            await self.result_queue.stop()   

        # cancel peer tasks
        if self._peer_tasks:
            for task in self._peer_tasks:
                task.cancel()
            await asyncio.gather(*self._peer_tasks, return_exceptions=True)
        
        # stop peer agent server
        if hasattr(self, '_peer_agent'):
            self._peer_agent.stop_server()

        # stop distribution server
        if self.enable_peers:
            self.pipeline.distribution.stop_server()

        # FIX 1 — offload blocking stop() to thread so event loop stays free
        print('[=] Stopping Asynchronous manager setup...')
        await asyncio.get_event_loop().run_in_executor(
            None, 
            lambda: self.async_manager.stop(timeout=5, force=True)
        )

        await asyncio.sleep(0.5)
        print('✅ Agent shutdown complete')

        logger.info("✅ Agent shutdown complete")
    

    def get_api_key(self) -> str:
        # Get the default API key (for client distribution)
        return getattr(self.async_manager, '_default_api_key', None)
    
 

