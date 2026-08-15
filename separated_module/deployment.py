"""
deployment.py
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
from . import async_manager
from . import distributed_inference
from . import peer_agent
from . import pipeline
from . import prediction_manager

class CohesiveAgentDeployment:
    """
    Production deployment wrapper that turns one `pipeline.IntegratedPipeline`
    into a networked, secure, concurrently-serving prediction agent.

    Where `pipeline.IntegratedPipeline` is the model/prediction "brain",
    `CohesiveAgentDeployment` is the operational shell around it: it
    owns the security-aware task manager (`async_manager`, a
    `async_manager.PipelineAsyncManager`) that actually serves prediction requests
    with API-key/security-level enforcement, a bounded async result
    queue + fixed-size worker pool for concurrency
    (`result_queue`/`worker_pool`), and two independent peer-discovery
    /connection layers:
      - the PRIMARY path, riding on `pipeline.distribution`
        (`distributed_inference.AgentDistributedInference`) — signed UDP broadcast discovery
        (`_broadcast_discovery`), config-file peers
        (`_load_known_peers`), and per-peer keepalive tasks
        (`_handle_peer_communication`);
      - a FALLBACK path via `_peer_agent` (`peer_agent.ConsecutivePeerAgent`,
        listening on `peer_discovery_port + 100`) used when the
        primary path times out or returns low confidence
        (`predict_with_peer_consecutive`).

    Broadly, its methods fall into a few groups:
      - **Lifecycle**: `start` (boots the async manager, P2P server,
        discovery, result queue/workers, and health monitor) and
        `shutdown` (tears all of that down in reverse, offloading the
        blocking `async_manager.stop()` call to a thread so the event
        loop stays responsive).
      - **Discovery**: signed, rate-limited, trusted-network-gated UDP
        broadcast discovery (`_broadcast_discovery`,
        `_sign_message`/`_verify_signature`,
        `_is_trusted_network`/`_check_discovery_rate_limit`,
        `_create_discovery_response`) plus local port-scanning
        (`_discover_local_peers`) and config-file peers
        (`_load_known_peers`).
      - **Peer connection management**: connecting
        (`_connect_to_peer`/`add_peer`/`_connect_single_attempt`/
        `_connect_with_smart_retry`), listing/removing
        (`list_peers`/`remove_peer`/`get_peers_status`), and ensuring a
        known-peers list is actually reachable before a prediction
        needs them (`_ensure_peer_connections`).
      - **Prediction**: `predict_sync`/`predict_async`/
        `predict_batch_async` for local-only predictions through
        `self.manager` (a `prediction_manager.PipelinePredictionManager`), and the
        peer-augmented variants (`predict_with_peers`,
        `multi_modal_peer_ensemble_prediction`,
        `predict_with_peer_consecutive`, `_ask_peer_simple`) that
        compare local confidence against peers' and pick the best
        result.
      - **Status/diagnostics**: `_print_status` and `get_peers_status`
        for a human-readable or structured snapshot of connections,
        queue depth, and security configuration.

    Security posture is driven by `security_level` (mapped to a
    `SecurityLevel` enum and propagated onto both `self.pipeline` and
    `self.pipeline.distribution` so every access-control check in the
    stack agrees on it), and, when `shared_auth_token` is supplied,
    that single token is wired up as both the P2P layer's
    signing/auth secret and a generated API key shared by the whole
    agent cluster — simplifying multi-agent deployments where every
    node should trust every other node equally.
    """
    
    def __init__(self,
                 pipeline: pipeline.IntegratedPipeline,
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
        """
                Wire up one deployable agent around a shared `pipeline`: a
                prediction manager (`prediction_manager.PipelinePredictionManager`), a fallback
                peer agent (`peer_agent.ConsecutivePeerAgent`) used when the primary P2P
                path fails, the security-aware async task manager
                (`async_manager.PipelineAsyncManager`), and a bounded result queue + worker
                pool for serving predictions concurrently. Also propagates the
                resolved `SecurityLevel` onto both `pipeline` and
                `pipeline.distribution` so downstream access-control checks
                (`_check_ip_access`, `_get_bind_host`) see a consistent value,
                and — if a `shared_auth_token` is given — configures it as both
                the P2P auth token and a generated API key shared across the
                whole cluster.

                Args:
                    pipeline: The `pipeline.IntegratedPipeline` this agent serves;
                        `pipeline.autonomous` is forced to True.
                    memory_name: Currently unused directly here (the pipeline's
                        own `memory_name` is what's actually used for P2P peer
                        identity) — accepted for API/config-file symmetry.
                    filename, target_title, label_name: Forwarded to
                        `prediction_manager.PipelinePredictionManager` to describe the labeled
                        dataset this agent predicts against.
                    security_level: One of `DEVELOPMENT`/`STAGING`/`PRODUCTION`/
                        `HARDENED`; unrecognized strings default to PRODUCTION.
                    enable_peers: If False, peer discovery/connections and the
                        P2P server are skipped entirely in `start()`.
                    trusted_networks: CIDR ranges allowed to receive discovery
                        responses (see `_is_trusted_network`).
                    secret_key: HMAC key used to sign/verify UDP discovery
                        broadcasts (`_sign_message`/`_verify_signature`).
                    peer_discovery_port: Base port for peer discovery; the
                        fallback `peer_agent.ConsecutivePeerAgent` listens on this port
                        +100.
                    shared_auth_token: If given, becomes the single API
                        key/auth token for the whole agent cluster (see above).
                    predict_manager: Currently unused directly in this
                        constructor — accepted for API symmetry with
                        `pipeline.IntegratedPipeline.__init__`.
                    peer_config, consecutive_peer_config: Paths to the JSON
                        files listing known peers for the primary and fallback
                        discovery paths respectively (`_load_known_peers`/
                        `_load_consecutive_known_peers`).
        """
        self.pipeline = pipeline 
        self.pipeline.autonomous = True 
  
        # Initialize prediction manager
        self.manager = prediction_manager.PipelinePredictionManager(
            self.pipeline,
            label_csv=filename,
            target_title=target_title,
            label=label_name
        )

        self._peer_agent = peer_agent.ConsecutivePeerAgent(
            peer_id=self.pipeline.memory_name,
            port=peer_discovery_port + 100,
            secret_key=secret_key,
            manager=self.manager,
            pipeline=self.pipeline
        ) 

        # Map security level string to enum
        self.security_map = {
            "DEVELOPMENT": SecurityLevel.DEVELOPMENT,
            "STAGING": SecurityLevel.STAGING,
            "PRODUCTION": SecurityLevel.PRODUCTION,
            "HARDENED": SecurityLevel.HARDENED
        }

        self.resolved_level = self.security_map.get(security_level, SecurityLevel.PRODUCTION)

        # this propagate security_level to pipeline AND distribution
        # so _check_ip_access and _get_bind_host can use it
        self.pipeline.security_level = self.resolved_level
        self.pipeline.distribution.security_level = self.resolved_level   # distributed_inference.AgentDistributedInference
     
        # Create Async Manager with security
        self.async_manager = async_manager.PipelineAsyncManager(
            pipeline=self.pipeline,
            prediction_manager=self.manager,
            security_level=self.security_map.get(security_level, SecurityLevel.PRODUCTION),
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

        self.discovery_enabled = True
        self.discovery = True                        # used in _broadcast_discovery while loop
        self.peer_discovery_broadcast = True         # ADD — this is what gates all discovery
        self.discovery_broadcast_only_trusted_network = True

        self.enable_peers = enable_peers
        self.peer_discovery_port = peer_discovery_port
        self._shutdown_event = asyncio.Event()
        self._peer_tasks = []
        self._known_peers = {}
        self.identified_peers = []

        self.attempt = 0
        self.max_attempts = 3
        
        self.result_queue = async_manager.AsyncResultQueue(max_size=1000)
        self.worker_pool = async_manager.WorkerPool(self.result_queue, num_workers=4) 
        
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
        """
                Best-effort collection of this machine's local IP addresses
                (hostname resolution + all interfaces via
                `socket.gethostbyname_ex`, plus `127.0.0.1`), used to recognize
                and skip self-discovery during peer scanning. Falls back to
                `['127.0.0.1']` if lookup fails.
        """
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
        """
                Check whether `client_ip` falls inside any CIDR range in
                `self.trusted_networks`. Used to decide whether an incoming
                discovery request/response should be processed at all — untrusted
                senders are ignored outright rather than rate-limited.
        """
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
        """
                Sliding-window (60s) rate limiter for discovery requests per
                source IP, capped at `self.discovery_rate_limit`. Records this
                request's timestamp as a side effect when allowed.
        """
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
        """
                Build the minimal payload sent back in response to a discovery
                probe — deliberately excludes this agent's id/capabilities and
                sets `requires_auth: True`, so a bare discovery response leaks
                as little as possible about the agent to an unauthenticated
                prober.
        """
        # a secure discovery response (minimal info)
        return {
            'type': 'DISCOVERY_RESPONSE',
            'version': '1.0',
            'port': self.peer_discovery_port,
            'requires_auth': True,  # Don't reveal agent_id or capabilities
            'timestamp': time.time()
        }    


    async def start(self, bootstrap_token: str = None, skip_discovery: bool=False):
        """
        Bring the whole agent online: start `self.async_manager`
        (raising if it fails to start), start the P2P server and peer
        discovery when `self.enable_peers` is set, start the result queue
        and worker pool (backed by `_prediction_worker`), and kick off the
        background `_health_monitor` loop. Prints a status summary
        (`_print_status`) once everything is up.
        """
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
        """
        Worker callback handed to `self.worker_pool.start`: runs a
        prediction for `texts` via `self.async_manager.predict` (method
        `'advanced'`) using the pipeline's configured timeout. Executed in
        a thread via `asyncio.to_thread` by the worker pool, so it's safe
        for this to block.
        """
        # Worker function for processing predictions
        # runs in a thread pool via asyncio.to_thread
        return self.async_manager.predict(
            texts=texts,
            timeout=self.pipeline.timeout,
            retries=None,
            api_key=api_key,
            client_ip=client_ip,
            method='advanced'
        )

    async def _start_peer_discovery(self):
        """
        Peer-discovery bootstrap run once at startup: connects to every
        peer listed in `_load_known_peers`, and for any that fail, falls
        back to local-network scanning (`_discover_local_peers`) and/or
        starts the background UDP broadcast-discovery loop
        (`_broadcast_discovery`) if `self.peer_discovery_broadcast` is set.
        """
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
        """
                Load `(host, port)` peer pairs from `self.peer_config_name`
                (JSON, under a `known_peers` key), or fall back to two
                hardcoded localhost defaults if the config file doesn't exist.
        """
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
        """
        Best-effort local-network peer discovery by port-scanning: tries
        `_connect_to_peer` against a small range of ports around
        `self.peer_discovery_port` on each of this machine's first few
        local IPs, skipping this agent's own port/loopback address.
        """
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
        """
        Long-running loop (runs until `self._shutdown_event` is set) that
        periodically sends a signed UDP broadcast `'DISCOVERY'` message and
        listens briefly for responses. Ignores self-originated responses
        and anything from an untrusted network or over the per-IP rate
        limit (`_is_trusted_network`/`_check_discovery_rate_limit`), and
        verifies each response's HMAC signature (`_verify_signature`)
        before connecting to the responding peer (`_connect_to_peer`).
        Sleeps 60s between broadcast attempts.
        """
        # broadcast discovery message to find peers on network
        logger.info("📡 Starting broadcast discovery...")
        
        while not self._shutdown_event.is_set() and self.discovery:
            try:
                # UDP broadcast socket
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
        """
                HMAC-SHA256 sign a discovery message (after sorting its keys for
                a deterministic serialization) using `self.discovery_secret`, so
                `_verify_signature` can later confirm a discovery broadcast/
                response wasn't spoofed.
        """
        # Sign message with HMAC to prevent spoofing
       
        # Sort keys for consistent serialization
        message_str = json.dumps(message, sort_keys=True)
        return hmac.new(
            self.discovery_secret.encode(),
            message_str.encode(),
            hashlib.sha256
        ).hexdigest()  
        
    def _verify_signature(self, message: dict) -> bool:
        """
                Verify a discovery message's `'signature'` field against a
                freshly recomputed HMAC over the rest of the message (via
                `_sign_message`), using a constant-time comparison
                (`hmac.compare_digest`) to avoid timing side-channels. Returns
                False if no signature is present.
        """
        # Verify message signature
        if 'signature' not in message:
            return False
        
        signature = message.pop('signature')
        expected = self._sign_message(message)
        message['signature'] = signature
        
        return hmac.compare_digest(signature, expected)


    async def _connect_to_peer(self, host: str, port: int) -> bool:
        """
        Establish (or reuse) a connection to a peer at `host:port`: skips
        if already connecting or already connected, otherwise connects via
        `self.pipeline.distribution.connect_to_agent`, records the peer in
        `self._known_peers`, and spawns a background task
        (`_handle_peer_communication`) to keep it alive. Returns True on a
        new or pre-existing connection, False on failure.
        """
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
        """
        Background per-peer keepalive loop: every 5s, sends a `PING`
        message over `sock` to confirm the peer is still reachable; marks
        the peer disconnected (in `self._known_peers`) and closes the
        socket once the ping fails or the task is cancelled/shutdown is
        signaled.
        """
        # Handle bidirectional communication with a peer
        logger.info(f"📡 Peer communication active for {peer_host}:{peer_port}")
        
        try:
            while not self._shutdown_event.is_set():
                # The distribution system handles message receiving internally
                # This task just monitors connection health
                await asyncio.sleep(5)
                
                # Send heartbeat to check connection
                try:
                    self.pipeline.distribution._send_message(
                        sock, {'type': 'PING', 'timestamp': time.time()}
                    )
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

    
    
    async def _health_monitor(self):
        """
        Background loop (runs until `self._shutdown_event` is set) that
        every 30s logs `self.async_manager.get_stats()` and, when peers are
        enabled, pings the network (`distribution.broadcast_ping`) to check
        how many peers are still alive — a lightweight health signal
        separate from the per-connection `_handle_peer_communication`
        pings.
        """
        # Background health monitoring
        print("[💓] Peer health monitor started")
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
        """
                Print a human-readable snapshot of this agent's current state:
                async-manager state/security level, peer-enablement and count,
                listening port, local IPs, queue size, whether an API key is
                required (with a truncated preview of the default key), and a
                per-peer host:port/trust listing.
        """
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
    async def multi_modal_peer_ensemble_prediction(self, texts, api_key: str = None, method: str = 'advanced', disable_sync: bool=False) -> Any:
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
                timeout=self.pipeline.timeout
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
            await asyncio.sleep(0.3)    

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
        """
                Quick liveness probe for the local peer-discovery port: opens a
                throwaway UDP socket and attempts `connect_ex` against
                `127.0.0.1:peer_discovery_port`, treating a zero return code as
                "listening". Best-effort only — returns False on any exception
                rather than raising.
        """
        # if the server is actually listening on its port
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) 
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
            
            # ✅ 4: Single connection attempt (NO RETRYS)
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
                X = texts['X']
                y = texts['y']

                result, chosen_label, confidence = self.manager.advanced_prediction_method(
                    test_titles,
                    label_map,
                    rules,
                    X=X, y=y,
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
        """
                Convenience pass-through to `self.result_queue.get_status()` for
                overall queue statistics (not a specific request).
        """
        # Get result queue statistics
        logger.info("[=] Fetching result queue stats...")
        return self.result_queue.get_status(request_id=None)

     
    # ============ PEER MANAGEMENT ============
    
    def add_peer(self, host: str, port: int, api_key: str = None):
        """
                Manually connect to a peer at `host:port` and register it in
                both `self._known_peers` and (via
                `self.pipeline.distribution.connect_to_agent`) the distribution
                layer's socket table, then spin up a background task
                (`_handle_peer_communication`) to keep the connection's health
                monitored. Refuses to add this agent's own address as a peer.
                If `api_key` isn't given, attempts to look one up from
                `distribution.peer_tokens`; if it is given, registers it via
                `distribution.add_trusted_agent` first.

                Returns True if the connection was established, False otherwise.
        """
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
        """
                Disconnect and forget a peer at `host:port`: finds the matching
                entry in `pipeline.distribution.remote_agents` and disconnects it,
                then removes the corresponding entry from `self._known_peers`.
        """
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
        """
                List currently-registered remote agents as plain dicts
                (`agent_id`, `host`, `port`, `trust`, `last_seen`), filtering out
                the synthetic `'local'` entry and anything that looks like this
                agent talking to itself (port 0, or this agent's own
                host:port).
        """
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
        """
        Graceful full shutdown: signals `_shutdown_event`, stops the
        worker pool and result queue, cancels and awaits all peer tasks,
        stops the fallback peer agent and the distribution server, and
        finally stops `self.async_manager` (offloaded to a thread via
        `run_in_executor` since its `stop()` is blocking) with a 5s timeout
        and forced termination.
        """
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
        """
                Return the default API key configured for this agent (set from
                `shared_auth_token` in `__init__`, or generated by
                `async_manager.PipelineAsyncManager` otherwise) — used when this agent acts as
                a client making requests to peers.
        """
        # Get the default API key (for client distribution)
        return getattr(self.async_manager, '_default_api_key', None)
    


# ============ EXAMPLE: SECURE PEER-TO-PEER CLUSTER ============
async def run_secure_agent_cluster(pipeline,test_titles, label_map, rules, X=None, y=None, agent_id=None, filename=None, title_name=None, label_name=None, manager=None):
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
        pipeline=pipeline,
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
        pipeline=pipeline,
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
        
        texts = {"test_titles": test_titles, "label_map": label_map, "rules": rules, "X":X, "y":y, "use_transformer": True, "agent_id": agent_id}

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
        print("\n⏳ Cluster stable. Waiting 30 seconds before shutdown...")
        await asyncio.sleep(30)
        agent1._peer_agent.stop_server()
        agent2._peer_agent.stop_server()
        
    except Exception as e:
        print(f"\n❌ Error in cluster: {e}")
        traceback.print_exc()
        
    print("\n🛑 Shutting down cluster...")
    await agent1.shutdown()
    await agent2.shutdown()
    print("✅ Cluster shutdown complete")




async def example_async_with_result_queue(pipeline, test_titles, label_map, rules, X=None, y=None,agent_id=None, filename=None, title_name=None, label_name=None):
    # Example using the proper result queue
    
    agent = CohesiveAgentDeployment(
        pipeline=pipeline,
        memory_name="test_agent",
        filename=filename,
        target_title=title_name,
        label_name=label_name,
        security_level="DEVELOPMENT",
        enable_peers=False
    )
    
    await agent.start()
    
    api_key = agent.get_api_key()
    payloads = {"test_titles": test_titles, "label_map": label_map, "rules": rules, "X":X, "y":y, "use_transformer": True, "agent_id": agent_id}
    
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






