"""
distributed_inference.py
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
from . import ensemble
from . import messaging
from . import model_storage
from . import pipeline
from . import query_node

class AgentDistributedInference:
    """
    The peer-to-peer (P2P) networking and security layer for the pipeline.

    This class turns a single pipeline instance into a node in a small
    trust-based mesh network of "agent" processes that can ask each other
    for help with predictions, sync memory, and vote on ensemble decisions.
    It is intentionally self-contained: it owns socket setup (with optional
    TLS), authentication (HMAC-signed messages + shared/per-agent tokens),
    rate limiting, IP allow/block lists, an outgoing message queue with
    retry, and the actual request/response protocol used between agents.

    High-level responsibilities (see the `# ===` section markers in the
    method list below for the actual grouping used in the source):
      - **Security**: IP allow/block lists (`_check_ip_access`,
        `add_allowed_ip`/`add_blocked_ip`), SSL context setup (`_setup_ssl`,
        `_generate_self_signed_cert`), message signing/verification
        (`_sign_message`, `_verify_signature`), input sanitization
        (`_sanitize_input`, `_sanitize_arrays_and_dicts`,
        `_sanitize_structured`), and rate limiting (`_check_rate_limit`).
      - **Connection lifecycle**: starting/stopping the listening server
        (`start_server`, `_accept_connections`, `stop_server`), connecting
        out to peers (`connect_to_agent`), and tearing connections down
        (`disconnect_agent`, `_remove_dead_socket`).
      - **Wire protocol**: framing/encoding messages for the socket layer
        (`_send_message`, `_receive_message`, `_encode_arrays_for_wire`,
        `_decode_arrays_from_wire`), and dispatching received messages to
        the right handler (`_handle_client`, `_process_message`).
      - **Peer requests**: the actual "ask a peer for a prediction / vote /
        memory sync" request-response calls (`request_prediction`,
        `request_prediction_batch`, `request_ensemble_vote`,
        `sync_memory_with_agent`) and their server-side handlers
        (`_handle_predict_request`, `_handle_ensemble_vote_request`,
        `_handle_memory_sync_request`, `_handle_failure_report`,
        `_handle_trust_update`).
      - **Ambiguity escalation**: when the local ensemble
        (`ensemble.WeightedEnsemblePredictor`) can't confidently resolve a
        prediction, it calls into `_handle_peer_agent_request` /
        `handle_peer_uncertainty` here to ask a trusted peer (or another
        local device) and calibrate the returned probabilities against the
        local ones (`_calibrate_peer_probs`).
      - **Async mode**: an optional `messaging.AsyncMessageQueue`-backed path
        (`use_async=True`) that registers handlers for the same message
        types but processes them off a queue instead of inline per-socket.

    This class is deliberately defensive: almost every network-facing
    method wraps its work in try/except and degrades to logging + returning
    a safe default (None/False/original input) rather than raising, since a
    single flaky peer should never be able to crash the local pipeline.

    Args (constructor):
        pipeline: The parent `pipeline.IntegratedPipeline`, used to reach shared
            config and, indirectly, the ensemble predictor for producing
            local predictions to answer peer requests with.
        storage (model_storage.ModelStorage): Shared persistence layer, used to build the
            local `query_node.QueryNode` and for peer memory sync.
        memory_name (str): Namespace used for this agent's memory
            (attention memory, sync state, etc.).
        port (int): TCP port this agent listens on for incoming peer
            connections. Default 5555.
        use_async (bool): If True, uses the `messaging.AsyncMessageQueue` handler path
            and starts a background health checker.
        secret_key: HMAC key used to sign/verify outgoing/incoming messages
            (see `_sign_message`/`_verify_signature`).
        ssl_cert_file, ssl_key_file: Paths to a PEM cert/key pair for TLS;
            if omitted, a self-signed pair can be generated
            (`_generate_self_signed_cert`).
        ssl_context, client_ssl_context: Pre-built `ssl.SSLContext` objects
            to use instead of having this class construct its own.
        shared_auth_token: A token accepted from any peer that presents it
            (as opposed to a per-agent token registered via
            `add_trusted_agent`).
        predict_manager: Optional external prediction manager used to
            answer incoming predict requests when the pipeline itself isn't
            the source of truth.
        bind_host: Interface to bind the listening socket to; if None,
            resolved via `_get_bind_host()` based on `security_level`.
        security_level: Overrides `pipeline.security_level` for this agent
            (affects default IP-access policy and bind-host selection).

    Attributes (selected — most are set up directly in `__init__`, grouped
    by purpose in the source):
        MSG_TYPES (dict): Enum-like mapping of message type name -> int
            code used on the wire (PREDICT_REQUEST, PING, TRUST_UPDATE, ...).
        remote_agents (dict): `{agent_id: {'sock', 'host', 'port', 'trust'}}`
            — the live registry of connected peers.
        trusted_agents (dict): `{agent_id: {'token', 'trust_level',
            'added_at'}}` — agents allowed to authenticate beyond the
            shared token.
        outgoing_queue (deque): Buffered outgoing messages processed by
            `_process_outgoing_queue` with retry (`max_retries`,
            `retry_delay`).
        message_queue (messaging.AsyncMessageQueue): Handler registry used in async
            mode; also used to register the sync-mode message type names
            even when `use_async=False` isn't strictly required, so both
            modes share one dispatch table.
        pending_requests (dict): `{request_id: Future}` used to correlate
            an outgoing request with its eventual async response, guarded
            by `request_lock`.
    """

    def __init__(self, pipeline, storage, memory_name, port=5555, 
                 use_async=False, secret_key=None, 
                 ssl_cert_file=None, ssl_key_file=None, 
                 ssl_context=None, client_ssl_context=None,
                 shared_auth_token=None, predict_manager=None,
                 bind_host=None, security_level=None):      

        self.pipeline = pipeline
        self.memory_name = memory_name
        self.port = port
        self.storage = storage

        self.query_node = query_node.QueryNode(pipeline, memory_name, self.storage)        
        
        self.agent_comm_log = {}
        self.connections_log = {}
        self.connections = []  # List of connected sockets
        self.remote_agents = {}  # {agent_id: {'sock': sock, 'host': host, 'port': port, 'trust': 1.0}}
        
        self.running = False
        self.socket = None
        self.temporary_message = None
        self.temporary_agent_id = None  
        self._server_started = False  # explicit flag

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

        self.enable_ssl = True  # Set to False for basic P2P.
        # i provided basic cert file and key since there are other layered security other than ssl, and also due to infrequent external connections.
        self.ssl_cert_file = ssl_cert_file
        self.ssl_key_file = ssl_key_file
        self.ssl_context = ssl_context
        self.client_ssl_context = client_ssl_context

        if self.enable_ssl:
            print('[+] setting up SSL...')
            self._setup_ssl()

        self.allowed_ips = set()  # Add trusted IPs
        self.blocked_ips = set()  # Block malicious IPs

        self.bind_host = bind_host
        self.security_level = security_level or getattr(pipeline, 'security_level', None)

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
        self._health_check_interval = 30  # seconds

        self.use_async = use_async
        
        # Register message handlers
        print('[=++=] Initiating message Queue')
        self.message_queue = messaging.AsyncMessageQueue()
            
        self.message_queue.register_handler('predict_request', self._handle_predict_request_async)
        self.message_queue.register_handler('memory_sync', self._handle_memory_sync_async)
        self.message_queue.register_handler('ensemble_vote', self._handle_ensemble_vote_async)
        self.message_queue.register_handler('ping', self._handle_ping)
        self.message_queue.register_handler('status', self._handle_status)
                
        if self.use_async:
            self._start_health_checker()     

        # Queue for outgoing messages (buffered with retry)
        self.outgoing_queue = deque()
        self.queue_processor_thread = None
        self._last_health_check = time.time()       
            
        # Trust configuration
        self.min_trust_level_for_auto_add = TrustLevel.STANDARD
        self.trusted_agents = {}  # agent_id -> {'token': token, 'trust_level': TrustLevel, 'added_at': datetime} 
        self.highly_trusted_peer = []
        self.socket_owners = {}

        self.pending_requests = {}  # request_id -> Future
        self.request_lock = threading.Lock()        

    # ============ SECURITY FEATURES ============

    def _check_ip_access(self, ip: str) -> bool:
        """
        IP access check with security-level-aware default policy.

        Empty allowed_ips behavior:
        DEVELOPMENT/STAGING + bound to 127.0.0.1
            → loopback only, allow 127.x.x.x automatically
            → external IPs blocked implicitly by OS before reaching here

        PRODUCTION/HARDENED + bound to 0.0.0.0
            → allowed_ips MUST be populated, empty = deny all external
            → forces explicit configuration rather than accidental open access

        Explicit allowed_ips populated
            → always enforced regardless of security level
        """
        print(f'|| Checking IP access for: {ip}')

        # blocked list always takes priority
        if ip in self.blocked_ips:
            self._log_security_event('ip_blocked_access_denied', {'ip': ip})
            return False

        # loopback always allowed — needed for local agent communication
        if ip in ('127.0.0.1', '::1', 'localhost'):
            return True

        # explicit allowlist — always enforced when populated
        if self.allowed_ips:
            allowed = ip in self.allowed_ips
            if not allowed:
                self._log_security_event('ip_not_in_allowlist', {'ip': ip})
            return allowed

        # allowed_ips is empty — behavior depends on security level
        security_level = getattr(self, 'security_level', None)
        bind_host      = getattr(self, 'bind_host', '0.0.0.0')

        if security_level in (SecurityLevel.HARDENED, SecurityLevel.PRODUCTION):
            # PRODUCTION/HARDENED with empty allowlist and external binding
            # → deny external IPs, require explicit configuration
            print(f'[⚠️] IP {ip} denied here — allowed_ips is empty in '
                f'{security_level.value} mode. '
                f'Populate allowed_ips to permit external peers.')
            self._log_security_event('ip_denied_empty_allowlist', {
                'ip': ip,
                'security_level': security_level.value,
                'hint': 'populate allowed_ips or use DEVELOPMENT mode for local testing'
            })
            return False

        elif security_level in (SecurityLevel.DEVELOPMENT, SecurityLevel.STAGING):
            # DEVELOPMENT/STAGING bound to 127.0.0.1 — external IPs
            # shouldn't reach here, sometimes.
            if bind_host == '127.0.0.1':
                print(f'[⚠️] External IP {ip} reached local-only server '
                    f'— denying')
                return False

            # bound to 0.0.0.0 in dev mode.
            print(f'[⚠️] Allowing {ip} in {security_level.value} mode '
                f'with empty allowlist — not recommended for production')
            return True

        else:
            # no security level set — conservative default, deny external
            print(f'[⚠️] IP {ip} denied — no security level configured '
                f'and allowed_ips is empty')
            self._log_security_event('ip_denied_no_security_config', {'ip': ip})
            return False    

    def add_allowed_ip(self, ip):
        """Add an IP to the allow-list (see `_check_ip_access`)."""
        self.allowed_ips.add(ip)
        self._log_security_event('ip_allowed', {'ip': ip})

    def remove_allowed_ip(self, ip):
        """Remove an IP from the allow-list."""
        self.allowed_ips.discard(ip)
        self._log_security_event('ip_removed_from_allow', {'ip': ip})

    def add_blocked_ip(self, ip):
        """Add an IP to the block-list; always denied regardless of allow-list state."""
        self.blocked_ips.add(ip)
        self._log_security_event('ip_blocked', {'ip': ip})

    def remove_blocked_ip(self, ip):
        """Remove an IP from the block-list."""
        self.blocked_ips.discard(ip)
        self._log_security_event('ip_removed_from_block', {'ip': ip})

    def _setup_ssl(self):
        """
        Resolve this agent's SSL/TLS configuration for both the listening
        (server) socket and outgoing (client) connections, in priority
        order: caller-supplied `ssl_context`/`client_ssl_context` objects,
        then caller-supplied cert/key file paths (via
        `resolve_and_load_ssl`), then a generated self-signed CA + cert as
        a last resort (`_generate_self_signed_cert`).

        On any failure, disables SSL entirely (`self.enable_ssl = False`)
        rather than leaving the agent in a half-configured state.
        """
        try:
            if self.ssl_context and isinstance(self.ssl_context, ssl.SSLContext):
                # User passed a pre-built server context
                print("✅ Using user-provided SSL context.")

                if self.client_ssl_context and isinstance(self.client_ssl_context, ssl.SSLContext):
                    # User also provided client context — ideal case
                    print("✅ Using user-provided client SSL context.")

                elif self.ssl_cert_file and self.ssl_key_file:
                    # Build client context from provided cert files
                    client_ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
                    client_ctx.load_cert_chain(self.ssl_cert_file, self.ssl_key_file)
                    client_ctx.check_hostname = False
                    self.client_ssl_context = client_ctx
                    print("✅ Built client SSL context from provided cert files.")

                else:
                    # No client context and no cert files — can't do mTLS outbound
                    print("⚠️  No client cert available for outgoing connections. "
                        "Peers requiring mTLS will reject this agent.")
                    self.client_ssl_context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)

            elif self.ssl_cert_file and self.ssl_key_file:
                # User supplied cert files — resolve and load both contexts
                cert_path, key_path, ssl_contexts = self.resolve_and_load_ssl(
                    self.ssl_cert_file, self.ssl_key_file
                )
                if ssl_contexts:
                    self.ssl_cert_file      = cert_path
                    self.ssl_key_file       = key_path
                    self.ssl_context        = ssl_contexts["server"]
                    self.client_ssl_context = ssl_contexts["client"]
                else:
                    print("⚠️  Provided cert/key could not be loaded, falling back to self-signed.")
                    self._generate_self_signed_cert()

            else:
                # Nothing provided — generate self-signed fallback
                self._generate_self_signed_cert()

        except Exception as e:
            print(f"[!] SSL setup failed: {e}")
            self.enable_ssl = False

    def _generate_self_signed_cert(self):
        """
        Fallback path when no real certificates are supplied: generates an
        in-memory self-signed CA and leaf certificate (RSA-2048) purely to
        get TLS *encryption* working for local/staging P2P testing.

        This provides confidentiality on the wire but NOT real identity
        verification — do not rely on this for production deployments
        across untrusted networks; supply real certs via
        `ssl_cert_file`/`ssl_key_file` instead.
        """
        print(
            "⚠️  SSL running in self-signed fallback mode. "
            "Provides encryption but NOT production-grade identity verification. "
            "Supply real certs via ssl_cert_file/ssl_key_file for production use."
        )
        def _make_key():
            return rsa.generate_private_key(public_exponent=65537, key_size=2048)

        def _save_pem(path, data):
            with open(path, 'wb') as f:
                f.write(data)

        # ── Generate CA ──────
        ca_key = _make_key()
        ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, u"AbstractAgent-CA")])
        ca_cert = (
            x509.CertificateBuilder()
            .subject_name(ca_name)
            .issuer_name(ca_name)
            .public_key(ca_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.now(timezone.utc))
            .not_valid_after(datetime.now(timezone.utc) + timedelta(days=365))
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
            .sign(ca_key, hashes.SHA256())
        )
        ca_cert_pem = ca_cert.public_bytes(serialization.Encoding.PEM)
        ca_key_pem  = ca_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption()
        )
        _save_pem('ca.crt', ca_cert_pem)
        _save_pem('ca.key', ca_key_pem)
        # Lock down private key permissions on UNIX
        if os.name != 'nt':
            for key_file in ('ca.key', 'server.key', 'client.key'):
                try:
                    os.chmod(key_file, stat.S_IRUSR | stat.S_IWUSR)  # 600
                except OSError as e:
                    logger.warning(f"⚠️  Could not set permissions on {key_file}: {e}")

        # ─ Helper: sign a cert with the CA ──────────────────────────
        def _make_signed_cert(common_name: str, san_dns: str, cert_path: str, key_path: str):
            key = _make_key()
            name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])
            cert = (
                x509.CertificateBuilder()
                .subject_name(name)
                .issuer_name(ca_cert.subject)       # signed by CA, not self
                .public_key(key.public_key())
                .serial_number(x509.random_serial_number())
                .not_valid_before(datetime.now(timezone.utc))
                .not_valid_after(datetime.now(timezone.utc) + timedelta(days=365))
                .add_extension(
                    x509.SubjectAlternativeName([x509.DNSName(san_dns)]),
                    critical=False,
                )
                .add_extension(
                    x509.BasicConstraints(ca=False, path_length=None), critical=True
                )
                .sign(ca_key, hashes.SHA256())      # signed by CA key
            )
            _save_pem(cert_path, cert.public_bytes(serialization.Encoding.PEM))
            _save_pem(key_path,  key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption()
            ))
            return cert

        # ── Generate server.crt and client.crt ───────────────────────
        _make_signed_cert("server", "localhost", "server.crt", "server.key")
        _make_signed_cert("client", "localhost", "client.crt", "client.key")

        logger.info("✅ Generated CA, server, and client certificates.")

        # ── Build SSL contexts directly from generated files ──────────
        try:
            # Server: presents server.crt, trusts clients signed by the CA
            server_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            server_ctx.load_cert_chain('server.crt', 'server.key')
            server_ctx.load_verify_locations('ca.crt')
            server_ctx.verify_mode   = ssl.CERT_REQUIRED
            server_ctx.check_hostname = False

            # Client: presents client.crt, trusts servers signed by the CA
            client_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            client_ctx.load_cert_chain('client.crt', 'client.key')
            client_ctx.load_verify_locations('ca.crt')
            client_ctx.verify_mode   = ssl.CERT_REQUIRED
            client_ctx.check_hostname = False

            self.ssl_cert_file      = 'server.crt'
            self.ssl_key_file       = 'server.key'
            self.ssl_context        = server_ctx
            self.client_ssl_context = client_ctx

        except ssl.SSLError as e:
            logger.error(f"❌ SSL context build failed: {e}")

    def _generate_auth_token(self):
        """Generate a fresh random authentication token (used when none is supplied)."""
        return hashlib.sha256(os.urandom(32)).hexdigest()

    def _generate_secret_key(self):
        """Generate a fresh random HMAC secret key (used for message signing when none is supplied)."""
        return hashlib.sha256(os.urandom(48)).hexdigest()

    def _log_security_event(self, event_type, details):
        """Append a timestamped entry to `self.security_log` for auditing (IP blocks, auth failures, etc.)."""
        self.security_log.append({
            'timestamp': datetime.now().isoformat(),
            'event': event_type,
            'details': details
        })
        if len(self.security_log) > 1000:
            self.security_log = self.security_log[-1000:]
            
    def _looks_numeric(self, data):
        """Best-effort check for whether `data` is a number or numeric-looking string, used by the sanitizers to decide whether to coerce vs. reject a value."""
        try:
            arr = np.asarray(data)
            return np.issubdtype(arr.dtype, np.number)
        except Exception:
            return False

    def _sanitize_input(self, text, amount=1000):
        """
        Defensive cleanup for a single scalar/string field coming off the
        wire: truncates to `amount` characters and strips characters that
        could enable injection into downstream storage/logging.

        Args:
            text: Raw incoming value.
            amount: Max length to keep.

        Returns:
            The sanitized value (same type where practical).
        """
        if not isinstance(text, str):
            return str(text)
        sanitized = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
        return sanitized[:amount]

    def _sanitize_arrays_and_dicts(self, data, amount=100000):
        """
        Recursively sanitize a numeric array/dict payload from a peer
        message, bounding size (`amount`) and coercing/rejecting
        non-numeric entries so a malicious peer can't smuggle arbitrary
        objects into the pipeline's numpy-heavy code paths.
        """
        if isinstance(data, list) and self._looks_numeric(data):
            return [self._sanitize_input(item, amount) for item in data]
        elif isinstance(data, dict):
            return {key: self._sanitize_input(value, amount) for key, value in data.items()}
        else:
            return self._sanitize_input(data, amount)


    def _check_rate_limit(self, agent_id=None):
        """
        Sliding-window rate limiter: checks the connection-level window
        (`connection_timestamps`, `max_connections_per_minute`) and, when
        an `agent_id` is given, the per-agent request window
        (`request_timestamps[agent_id]`, `max_requests_per_minute`).

        Args:
            agent_id: If provided, also enforces the per-agent request
                limit in addition to the global connection limit.

        Returns:
            bool — True if the caller is within limits and may proceed.
        """
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
        """
        Compute an HMAC-SHA256 signature over the message payload using
        `self.secret_key`, so the receiver can verify the message wasn't
        tampered with in transit (see `_verify_signature`).
        """
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

        message_bytes = json.dumps(sorted_message, sort_keys=True, default=str).encode('utf-8')
      
        key = self.secret_key.encode() if isinstance(self.secret_key, str) else self.secret_key
        signature = hmac.new(key, message_bytes, hashlib.sha256).hexdigest()

        print(f'|| Signing message with: {len(message)} total of size')  
        logger.info(f"[=] Signing message: {len(message)}")
        return signature



    def resolve_and_load_ssl(self, cert_filename, key_filename):
        """
        Locate and load a cert/key pair from disk (resolving relative
        paths as needed) and build both the server-side and client-side
        `ssl.SSLContext` objects from them.

        Args:
            cert_filename, key_filename: Paths (absolute or relative) to
                the PEM cert and private key files.

        Returns:
            `(cert_path, key_path, {"server": SSLContext, "client":
            SSLContext})` on success, or a falsy `ssl_contexts` value the
            caller (`_setup_ssl`) treats as failure and falls back from.
        """
        def find_file(filename):
            candidates = []

            if os.path.isabs(filename):
                candidates.append(filename)
            else:
                # current working directory
                candidates.append(os.path.join(os.getcwd(), filename))

                # script directory
                try:
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    candidates.append(os.path.join(script_dir, filename))
                    # 3b — certs/ subfolder next to script
                    candidates.append(os.path.join(script_dir, "certs", filename))
                except NameError:
                    pass

                # home directory
                candidates.append(os.path.join(os.path.expanduser('~'), filename))

                # common data folders
                home = os.path.expanduser('~')
                for folder in ['Downloads', 'Documents', 'Desktop', 'Data', 'data']:
                    candidates.append(os.path.join(home, folder, filename))

                # sys.path entries 
                for p in sys.path:
                    if p:
                        candidates.append(os.path.join(p, filename))

            filepath = None
            for candidate in candidates:
                if os.path.exists(candidate):
                    filepath = candidate
                    break

            if filepath is None:
                print(f"❌ Could not find '{filename}' in any of these locations:")
                for c in candidates[:6]:
                    print(f"   {c}")
                print(f"\n💡 Tip: place your SSL files in a certs/ folder next to your script, or pass the full path.")
                print(f"   {os.getcwd()}\\{filename}")
                print(f"   {os.path.expanduser('~')}\\Downloads\\{filename}")
                return None

            print(f"✅ Found '{filename}' at: {filepath}")
            return filepath

        # Resolve both files
        cert_path = find_file(cert_filename)
        key_path  = find_file(key_filename)

        if not cert_path or not key_path:
            print("❌ SSL load aborted — one or more files not found.")
            return None, None, None

        # Load SSL context
        try:
            # Server context (for accepting incoming peer connections)
            server_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            server_ctx.load_cert_chain(cert_path, key_path)
            server_ctx.load_verify_locations(cert_path)  # trust self-signed cert
            server_ctx.verify_mode = ssl.CERT_REQUIRED
            server_ctx.check_hostname = False
            print("✅ Server SSL context loaded.")

            # Client context (for outgoing peer connections)
            client_ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
            client_ctx.load_cert_chain(cert_path, key_path)
            client_ctx.load_verify_locations(cert_path)  # trust self-signed cert
            client_ctx.verify_mode = ssl.CERT_REQUIRED
            client_ctx.check_hostname = False
            print("✅ Client SSL context loaded.")

            return cert_path, key_path, {"server": server_ctx, "client": client_ctx}

        except ssl.SSLError as e:
            print(f"❌ SSL error while loading cert/key: {e}")
            return cert_path, key_path, None
        except Exception as e:
            print(f"❌ Unexpected error loading SSL: {e}")
            return cert_path, key_path, None


    def _verify_signature(self, message, signature):
        """
        Recompute the HMAC over `message` and compare it (constant-time)
        against the provided `signature`, rejecting the message if they
        don't match. Companion to `_sign_message`.
        """
        # Verify signature - with timestamp in message
        
        # Create a copy without the signature field
        print(f'|| Verifying message signature total: {len(message)}')
        temp_msg = {k: v for k, v in message.items() if k != 'signature'}

        if 'timestamp' in temp_msg and isinstance(temp_msg['timestamp'], str):
            try:
                dt = datetime.fromisoformat(temp_msg['timestamp'].replace('Z', '+00:00'))
                temp_msg['timestamp'] = dt.timestamp()
            except:
                temp_msg['timestamp'] = time.time()
          
        # Sort keys for consistent serialization
        sorted_msg = {k: temp_msg[k] for k in sorted(temp_msg.keys())}
          
        message_bytes = json.dumps(sorted_msg, sort_keys=True, default=str).encode('utf-8')
         
        key = self.secret_key.encode() if isinstance(self.secret_key, str) else self.secret_key
        expected = hmac.new(key, message_bytes, hashlib.sha256).hexdigest()
        
        result = hmac.compare_digest(expected, signature)

        print(f'[=] Comparing result...')
        print(f'|| Signature verification result: {result}')
        logger.info(f"[-] Signature verification result: {result}")

        return result

   
    def add_trusted_agent(self, agent_id, agent_token):
        """Public helper to register an externally-known agent + token as trusted; delegates to `_add_trusted_agent` with the default trust level."""
        if agent_id == 'local':
            print(f"[❌] Cannot add 'local' as trusted agent")
            return

        self.trusted_agents[agent_id] = {'token': agent_token, 'added_at': datetime.now()}
        self._log_security_event('trusted_agent_added', {'agent_id': agent_id})

    def _authenticate_agent(self, token, agent_id):
        """
        Validate an incoming peer's credentials: accepts either the
        process-wide `shared_auth_token`/`secret_key`, or a token
        previously registered for this specific `agent_id` via
        `add_trusted_agent`/`_add_trusted_agent`.

        Args:
            token: The credential presented by the connecting agent.
            agent_id: The claimed identity of the connecting agent.

        Returns:
            bool — True if authentication succeeds.
        """
        print(f'|| Authenticating agent: {agent_id}...')
        logger.info(f"[==] Authenticating agent: {agent_id}...")

        if agent_id in self.highly_trusted_peer:
            print('[=+=] Agent is authenticated and already verified')
            return True

        elif token == self.auth_token:
            print(f"[=✅=] Agent {agent_id} authenticated with SHARED SECRET (FULL trust)")
            
            # Add to trusted list with FULL trust if not exists
            if agent_id not in self.trusted_agents:
                self._add_trusted_agent(agent_id, token, TrustLevel.FULL, source="shared_secret")
                return True
            else:
                # Update trust level if higher
                current_level = self.trusted_agents[agent_id].get('trust_level', TrustLevel.BASIC)
                if TrustLevel.FULL > current_level:
                    self.trusted_agents[agent_id]['trust_level'] = TrustLevel.FULL
                    print(f"[=] Upgraded trust level to FULL")
                    self.highly_trusted_peer.append(agent_id)
                    return True

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




    def _get_bind_host(self) -> str:
        """
        verifying for host binding IP.

        DEVELOPMENT → 127.0.0.1  loop back only, safest default,
                                no external exposure even on dev machines
        STAGING     → 127.0.0.1  still local, test P2P on same machine
        PRODUCTION  → 0.0.0.0    multi-machine P2P, SSL should be enabled
        HARDENED    → 0.0.0.0    multi-machine, SSL enforced separately

        """
        # explicit override always wins
        if hasattr(self, 'bind_host') and self.bind_host:
            return self.bind_host

        # derive from security level if available
        if hasattr(self, 'security_level'):
            if self.security_level in (SecurityLevel.DEVELOPMENT,
                                    SecurityLevel.STAGING):
                return '127.0.0.1'
            else:
                return '0.0.0.0'

        # no security level set — default to loopback, safest choice for local usage here.
        return '127.0.0.1'

    # ============ SERVER METHODS ============
    def start_server(self):
        """
        Bring up the listening TCP (optionally TLS-wrapped) socket and
        start the background accept-loop thread (`_accept_connections`).
        Validates the security configuration first (`_validate_security_config`)
        and is a no-op if the server is already started (`_server_started`).
        """

        print('[!] Inspect this Information Carefully:')
        self._validate_security_config()

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        bind_host = self._get_bind_host()
        self.socket.bind((bind_host, self.port))

        self.socket.settimeout(1.0)
        self.socket.listen(5)
        self.running = True
        logger.info(f"[=] Server started on port {self.port} with SSL={'enabled' if self.enable_ssl else 'disabled'}")

        print(f"[🤖] Agent listening on port {self.port}")
        if bind_host == '0.0.0.0' and not self.enable_ssl:
            logger.warning(
                '[⚠️] SECURITY WARNING: Server bound to 0.0.0.0 without SSL. '
                'All network interfaces are exposed. Use security_level=PRODUCTION '
                'or higher, or set enable_ssl=True for external deployments.'
            )
            print('[⚠️] SECURITY WARNING: Bound to all interfaces without SSL '
              '— suitable for local P2P only')    

        # Start accepting connections in background
        accept_thread = threading.Thread(target=self._accept_connections, daemon=True)
        accept_thread.start()

        self._server_started = True
        logger.info("[=] Server started and accepting connections...")
        
        return self.socket


    def _validate_security_config(self):
        """This function Warns about dangerous security configurations at startup."""
        security_level = getattr(self, 'security_level', None)
        bind_host      = getattr(self, 'bind_host', '0.0.0.0')

        warnings = []

        if bind_host == '0.0.0.0' and not self.enable_ssl:
            warnings.append(
                'Bound to 0.0.0.0 without SSL — all interfaces exposed unencrypted'
            )

        if not self.allowed_ips and bind_host == '0.0.0.0':
            if security_level in (SecurityLevel.PRODUCTION, SecurityLevel.HARDENED):
                warnings.append(
                    f'allowed_ips is empty in {security_level.value} mode '
                    f'with 0.0.0.0 binding — external peers will be denied. '
                    f'Add trusted peer IPs via add_allowed_ip()'
                )
            else:
                warnings.append(
                    'allowed_ips is empty — all external IPs currently permitted. '
                    'Consider populating allowed_ips for production deployments.'
                )

        if not self.secret_key:
            warnings.append('[!] No secret_key configured — HMAC signing disabled')

        for w in warnings:
            print(f'[⚠️] SECURITY CONFIG: {w}')
            logger.warning(f'[⚠️] SECURITY CONFIG: {w}')
            self._log_security_event('security_config_warning', {'warning': w})


    def _accept_connections(self):
        """
        Background loop (run in its own thread) that accepts incoming
        socket connections, applies `_check_ip_access` and rate limiting
        up front, completes the TLS handshake if SSL is enabled, and hands
        each accepted client off to `_handle_client` (typically on its own
        thread) for the authentication + message loop.
        """
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
                    continue

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
                    continue

                # Send agent info to identify
                self._send_agent_info(client)
                
                # Start handler thread
                thread = threading.Thread(target=self._handle_client, args=(client, addr))
                thread.daemon = True
                thread.start()

            except socket.timeout:
                continue 

            except Exception as e:
                if self.running:
                    print(f"[-] Accept error: {e}")
                    traceback.print_exc()
                    self.report_failure(id(self), 'processing', reason=f'{e}')
                                        
                break
    
    def _send_agent_info(self, client):
        """Send this agent's identity/capability info to a newly connected client as the first message on a connection."""
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
        """
        Gracefully shut down: flips `running` off, closes the listening
        socket and all active peer connections, and stops the queue
        processor / health-checker threads if they were started.
        """
        self.running = False   
        # Close all connections
        for conn in self.connections:
            try:    
                self._send_message(conn, {'type': self.MSG_TYPES['DISCONNECT']})

                conn.shutdown(socket.SHUT_RDWR)
                conn.close()
            except Exception as e:
                print(f'[= ERROR =] Socket cant be shutdown due to: {e}')
                pass

        self.connections.clear()
        if self.socket:
            try:
                self.socket.close()  
            except Exception as e:
                print(f'[= ERROR =] Socket cant be closed due to: {e}')
                pass

        print("[🛑] Server stopped")
    
    # ============ CLIENT METHODS ============
    def _is_duplicate_connection(self, host, port):
        """Check `established_connections` to avoid opening a second redundant socket to a peer we're already connected to at (host, port)."""
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

            # Socket creation
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)    
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)  # 1MB
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024 * 1024)  # 1MB      
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            if hasattr(socket, 'TCP_KEEPIDLE'):
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 10)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 5)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3)

           
            sock.settimeout(10)

            print(f"[connect_to_agent() SOCKET CREATED] id={id(sock)}")            
            sock.settimeout(self.connection_timeout)
            print(f'[==] Connecting to {host}:{port}...')
            sock.connect((host, port))

            if self.enable_ssl and self.ssl_context:
                sock = self.ssl_context.wrap_socket(sock, server_hostname=host)
                print('[==] Socket Connected with SSL Provided!')
            
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
        """Cleanly close and remove a peer connection by `agent_id`, cleaning up `remote_agents`/`connections`/`established_connections` bookkeeping."""
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
        X = None
        y = None
        
        # ✅ Check payload (which is a dict), not the message itself
        if isinstance(payload, dict):
            if 'test_titles' in payload:
                test_titles = payload.get('test_titles')
                label_map = payload.get('label_map')
                rules = payload.get('rules')
                X = payload.get('X')
                y = payload.get('y')
                
                # Sanitize if needed
                if test_titles:
                    test_titles = self._sanitize_structured(test_titles)
                if label_map:
                    label_map = self._sanitize_structured(label_map)
                if rules:
                    rules = self._sanitize_structured(rules)
                if X is not None:
                    X = self._sanitize_structured(X)
                if y is not None:
                    y = self._sanitize_structured(y)

                if X is None or y is None:
                    print('[=] Got necessary titles, label_map and rules.')
                else:
                    print('[=] Got necessary titles, label_map, rules and X And Y samples.')
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
                        test_titles, label_map, rules, X=X, y=y,
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
            X = message.get('X')
            y = message.get('y')

            print(f'[DEBUG] X immediately after deserialization: '
                f'shape={np.asarray(X).shape if X is not None else None}')


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
                    test_titles, label_map, rules, X=X, y=y, show_proba=False, use_transformer=self.pipeline.use_transformer
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


    def request_prediction(self, agent_id: Any, text: Any, timeout: float = 30.0) -> Any:
        """
        Ask a specific peer (`agent_id`) to run a prediction on `text` and
        wait (up to `timeout` seconds) for the response.

        Args:
            agent_id: Target peer's identity (must be in `remote_agents`).
            text: Input to have the peer predict on.
            timeout: Max seconds to wait for a response before giving up.

        Returns:
            The peer's prediction response payload, or None/an error
            indicator if the request times out or the peer is unreachable.
        """
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

    async def request_advanced_prediction_async(self, manager: Any, use_transformer: bool=False, agent_id: str=None, test_titles: List[tuple]=None, label_map: Dict[str, int]=None, rules: List[tuple]=None, X: np.ndarray=None, y: np.ndarray=None, timeout: float = 30.0, callback: Optional[Callable] = None):
        # Asynchronous prediction request
        # Local bypass - NO QUEUE
        try:
            if agent_id == 'local':
                logger.info(f"[=] Local request - direct execution")
                # Run sync prediction in thread pool
                result = await asyncio.to_thread(manager.advanced_prediction_method, test_titles, label_map, rules, X=X, y=y, show_proba=True, use_transformer=use_transformer)
                logger.info(f"[=] Local result: {result[1]} || confidence: {result[2]}")
                return result  

            msg_id = str(uuid.uuid4())
            message = Message(
                id=msg_id,
                type='predict_request',
                sender=self.temporary_agent_id,
                recipient=agent_id,
                payload={'test_titles': test_titles, 'label_map': label_map, 'rules': rules, 'X':X, 'y':y},
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
        except Exception as e:
            print(f'[!] Cannot request advanced prediction async: {e}') 
            response = {'prediction': None, 'result': None} 
            return response
        

    
    def request_prediction_direct(self, agent_id, text, timeout=5):
        """
        Lower-level variant of `request_prediction` that sends the request
        and blocks on the socket directly (shorter default timeout),
        bypassing the queue/Future-based async path — used for quick,
        latency-sensitive lookups.
        """
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


    async def request_prediction_async(self, agent_id: Any, text: Any, timeout: float = 30.0, callback: Optional[Callable] = None):
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
        """
        Request predictions for multiple `texts` from one peer in a single
        round trip, returning a list of per-text results (or per-text
        errors) rather than requiring one round trip per text.
        """
        # Batch async prediction requests (parallelized)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(texts)) as executor:
            futures = [
                executor.submit(self.request_prediction, agent_id, text, timeout)
                for text in texts
            ]
            results = [f.result(timeout=timeout) for f in futures]
        
        return results
    
    
    def start_queue_processor(self):
        """Start the background thread that drains `outgoing_queue` via `_process_outgoing_queue`, if not already running."""
        # Start background queue processor
        self.queue_processor_thread = threading.Thread(target=self._process_outgoing_queue, daemon=True)
        self.queue_processor_thread.start()
    
    def _process_outgoing_queue(self):
        """
        Background loop that pops messages off `outgoing_queue` and sends
        them, retrying up to `max_retries` times with `retry_delay` backoff
        on failure before giving up on a given message.
        """
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
        """Start the background thread that periodically calls `_check_health` every `_health_check_interval` seconds (async mode only)."""
        # Start background health checker for async mode.
        def health_check_loop():
            for _ in range(self._health_check_interval * 10):
                if not self.running:
                    return
                time.sleep(0.1)

            if self.running:
                self._check_agent_health()
        
        self._health_thread = threading.Thread(target=health_check_loop, daemon=True)
        self._health_thread.start()
    
    def _check_health(self):
        """Ping connected peers / verify sockets are still alive, pruning dead connections and updating `_last_health_check`."""
        # Check health of all connected agents.
        stats = self.message_queue.get_stats()
        logger.debug(f"[=] Queue stats: {stats}")
        
        # Check for stuck messages
        if stats.get('pending_count', 0) > 100:
            logger.warning(f"[=] High pending count: {stats['pending_count']}")
        
        # Ping all agents
        for agent_id in list(self.remote_agents.keys()):
            try:
                result = self.broadcast_ping()
                if agent_id not in result or result[agent_id].get('error'):
                    logger.warning(f"[=] Agent {agent_id} not responding")
            except Exception as e:
                logger.warning(f"[=] Health check failed for {agent_id}: {e}")
    
    def get_queue_stats(self) -> Dict:
        """Return a snapshot dict describing the current outgoing queue depth and related counters, for monitoring."""
        # Get message queue statistics.
        return self.message_queue.get_status()
    
    def get_dead_letter_queue(self) -> List[Dict]:
        """Return messages that exhausted their retries in `_process_outgoing_queue` and were never successfully delivered."""
        # Get failed messages for inspection.
        if hasattr(self.message_queue, 'get_dead_letter_queue'):
            return self.message_queue.get_dead_letter_queue()
        return []
    
    def stop(self):
        """Public shutdown entry point: stops the server, queue processor, and health checker, and closes all sockets."""
        # Graceful shutdown.
        logger.info("[=] Shutting down AgentDistributedInference...")
        self.running = False

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.call_soon_threadsafe(
                    lambda: asyncio.ensure_future(self.message_queue.stop())
                )
            else:
                loop.run_until_complete(self.message_queue.stop())
        except Exception as e:
            logger.warning(f"[=] Message queue stop warning: {e}")
        
        logger.info("[=] Shutdown complete")

    # ============ MESSAGE HANDLING ============

    def _encode_arrays_for_wire(self, obj):
        """
        Recursively replace numpy arrays with a shape-preserving dict
        BEFORE json.dumps runs, so 2D+ structure survives the wire.
        """
        if isinstance(obj, np.ndarray):
            return {
                '__ndarray__': True,
                'data': obj.ravel().tolist(),
                'shape': list(obj.shape),
                'dtype': str(obj.dtype),
            }
        if isinstance(obj, dict):
            return {k: self._encode_arrays_for_wire(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._encode_arrays_for_wire(v) for v in obj]
        return obj


    def _decode_arrays_from_wire(self, obj):
        """Reverse of _encode_arrays_for_wire — reconstructs real shape."""
        if isinstance(obj, dict):
            if obj.get('__ndarray__'):
                data  = np.array(obj['data'], dtype=obj.get('dtype', 'float64'))
                shape = tuple(obj['shape'])
                return data.reshape(shape)
            return {k: self._decode_arrays_from_wire(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._decode_arrays_from_wire(v) for v in obj]
        return obj


    def _send_message(self, sock, message):
        """
        Serialize (`_encode_arrays_for_wire` + JSON), sign
        (`_sign_message`), length-prefix, and write a message to `sock` in
        `CHUNK_SIZE` chunks. Raises/logs and removes the socket via
        `_remove_dead_socket` on a broken pipe.
        """
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

            msg_to_send = self._encode_arrays_for_wire(msg_to_send)
            # Add signature to the copy
            msg_to_send['signature'] = self._sign_message(msg_to_send)

            sorted_msg = {k: msg_to_send[k] for k in sorted(msg_to_send.keys())}

            print(f'[==] Sending message, Total: {len(sorted_msg)}')   
            data = json.dumps(sorted_msg, default=str).encode('utf-8')
            sock.sendall(len(data).to_bytes(4, 'big'))
            bytes_sent = 0
            while bytes_sent < len(data):
                chunk = data[bytes_sent:bytes_sent + self.CHUNK_SIZE]
                sock.sendall(chunk)
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
        """
        Read a length-prefixed, signed message off `sock` (in
        `CHUNK_SIZE` reads), verify its signature (`_verify_signature`),
        enforce `max_message_size`, and decode it (JSON +
        `_decode_arrays_from_wire`) back into a Python object.

        Returns:
            The decoded message dict, or None if the read fails, the
            signature is invalid, or the message exceeds the size limit.
        """
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
           
            try:
                message = json.loads(data.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                logger.warning(f'[=] Invalid JSON from peer: {e}')
                self._log_security_event('invalid_json', {})
                return None
                
            message = self._decode_arrays_from_wire(message)
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
        """
        Per-connection server-side loop (typically run in its own thread):
        authenticates the client, then repeatedly calls `_receive_message`
        and dispatches each one to `_process_message` until the connection
        closes or an unrecoverable error occurs.
        """
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
                    continue  # Skipped

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
        """
        Route a decoded incoming message to the correct handler based on
        its `MSG_TYPES` code (predict request, memory sync, ensemble vote,
        failure report, trust update, ping, etc.), after checking the
        sender's trust level where required (`_check_trust_level`).
        """
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
        """
        Verify that `agent_id`'s recorded trust level (from
        `trusted_agents`) meets or exceeds `required_trust` before allowing
        a privileged operation to proceed.
        """
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
        """
        Entry point used by `ensemble.WeightedEnsemblePredictor.explain_past_memory`
        when the local ensemble can't confidently resolve a prediction on
        its own: attempts to sync with a local peer first
        (`sync_with_local_peer`), and if that doesn't yield a usable
        target, tries to reach an actual peer agent (device or external)
        depending on `type`, then calibrates whatever probabilities come
        back against the local ones (`_calibrate_peer_probs`) before
        returning.

        Args:
            probs: The local (ambiguous) probability distribution to
                improve on.
            self_attn_weights: The local attention weights for this input,
                forwarded so the peer can reason about the same context.
            input_ids: The input being predicted on.
            type: 'DevicePeer' to look for another local peer via shared
                storage, or 'ExternalPeer' to go out over the socket layer.
            agreement: Whether a node agreement has already been
                established for this exchange (affects downstream
                explainability side effects at the call site).

        Returns:
            A (possibly peer-calibrated) probability array; falls back to
            the original `probs` if no peer help could be obtained.
        """
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
                        probs = self.process_peer_request(probs, target_preds, self_attn_weights, input_ids)
    
                    except Exception as e:
                       print(f"[-] Error processing request: {e}, returning regular probs")

            except Exception as e:
                print(f'[-] Error handling request... {e}, returning regular probs')
                self.report_failure(id(self), 'processing', reason=f'{e}')                        

            print(f'[||] Successfully calibrate probs with previous Peer using database!')
            self.save_to_local_peer(self.memory_name, probs)
        else:
            print(f'[-] Connection to peer agent {self.temporary_agent_id} is not permitted, returning regular probs...')

        return probs


    def _calibrate_peer_probs(self, probs, target_preds, self_attn_weights, attn_weights, input_ids, AEL):
        """
        Blend/reconcile the local prediction with a peer's suggested
        target prediction and attention weights, using agreement between
        the two as a trust signal (similar in spirit to
        `ensemble.WeightedEnsemblePredictor._dynamic_weighted_ensemble` but scoped
        to a single peer's input rather than multiple local models).

        Args:
            probs: Local probability distribution.
            target_preds: The peer-provided target prediction(s) to
                reconcile against.
            self_attn_weights, attn_weights: Local vs. peer attention
                weights, used to gauge how comparable the two contexts are.
            input_ids: The input being predicted on.
            AEL: Peer-supplied auxiliary evidence/label information used in
                the calibration (upstream-provided; consumed as-is here).

        Returns:
            The calibrated probability array to use as the final result.
        """
        eps = 1e-5
        calibrated = probs.copy()

        try:
            n_classes = probs.shape[1]
        except:
            n_classes = probs.shape[0]

        batch_size = len(target_preds)
        anisotropy = self.pipeline.anisotropy_measurement(attn_weights)  


        if isinstance(attn_weights, (str, np.str_)):
            clean_str = str(attn_weights).replace('[', '').replace(']', '').replace('...', '')
            attn_weights = np.fromstring(clean_str, sep=' ')
        elif isinstance(attn_weights, np.ndarray) and np.issubdtype(attn_weights.dtype, np.character):
            # catches arrays filled with string text
            clean_str = ' '.join(attn_weights.astype(str).flatten()).replace('[', '').replace(']', '').replace('...', '')
            attn_weights = np.fromiter(
                    (x for x in clean_str.split() if x != "..."), dtype=float
                ) 
        else:
            # Ensure standard float array if it was integers or objects
            attn_weights = np.asarray(attn_weights, dtype=float)

        if isinstance(target_preds, (str, np.str_)):
            clean_str = str(target_preds).replace('[', '').replace(']', '').replace('...', '')
            target_preds = np.fromstring(clean_str, sep=" ")

        elif isinstance(target_preds, np.ndarray) and np.issubdtype(target_preds.dtype, np.character):
            # catches arrays filled with string text
            clean_str = ' '.join(target_preds.astype(str).flatten()).replace('[', '').replace(']', '').replace('...', '')
            target_preds = np.fromiter(
                    (x for x in clean_str.split() if x != "..."), dtype=float
                ) 
        else:
            # Ensure standard float array if it was integers or objects
            target_preds = np.asarray(target_preds, dtype=float)

        target_preds = np.asarray(target_preds, dtype=np.float32)
    
        for i in range(batch_size):

            mlp_target = target_preds[i] if target_preds.ndim > 1 and target_preds.shape[0] > i else target_preds
            attn_target = attn_weights[i] if attn_weights.ndim > 1 and attn_weights.shape[0] > i else attn_weights
       
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

                justified = (1.0 - AEL) * consensus + eps
                boost = (1.0 + justified) * attn_quality + eps

            abstract_error_quality_score = (1.0 - attn_quality) * anisotropy + eps
            self.query_node.peer_trust = (1.0 - abstract_error_quality_score) + boost * justified 

            try:
                calibrated[i, mlp_target] = min(calibrated[i, mlp_target] * (1.5 * (1.0 - abstract_error_quality_score)), 0.95)
            except:
                return calibrated

            calibrated[i] /= calibrated[i].sum()


        return calibrated        
            

    def handle_peer_uncertainty(self, probs, target_preds, self_attn_weights, attn_weights, input_ids):
        """
        Server-side counterpart invoked when a peer reports that *it* is
        uncertain and wants this agent's opinion: reconciles the peer's
        probabilities/target predictions with this agent's own view using
        the same calibration logic as `_calibrate_peer_probs`, then returns
        the combined result to send back.
        """
        try:
            embedded = False
            if isinstance(input_ids, (list, np.ndarray)):
                embedded = True

            if self_attn_weights is None:
                _, _, self_attn_weights = self.pipeline.model2.predict(input_ids, embedded=embedded)  


            if isinstance(attn_weights, tuple):
                attn_weights = attn_weights[0]
            if isinstance(self_attn_weights, tuple):
                self_attn_weights = self_attn_weights[0]

            if isinstance(self_attn_weights, str):
                self_attn_weights = np.array(self_attn_weights) 
                self_attn_weights = self_attn_weights[0]             
             
            if isinstance(attn_weights, str):
                attn_weights = np.array(attn_weights)
        
            batch_similarity = self.pipeline.cosine_similarity(attn_weights, self_attn_weights)

            anisotropy = self.pipeline.anisotropy_measurement(attn_weights)
            AME = self.pipeline.AME_Encoder(attn_weights)
            AMR = 1.0 / (1.0 + np.exp(-AME))

            weighted_quality_rate = (1.0 - AMR) * anisotropy
            
            print(f'[=] Batch similarity: {batch_similarity} With quality rate of attention: {weighted_quality_rate}')
            if weighted_quality_rate > 0.75 and batch_similarity > 0.75:
                return self.process_peer_request(probs, target_preds, attn_weights, input_ids)
            else:
                print('[!] Low uncertainty, normalizing with local agent data...')

                AEL = self.pipeline.confidence_threshold + weighted_quality_rate + (1.0 - AMR) * anisotropy
                calibrated = self._calibrate_peer_probs(probs, target_preds, self_attn_weights, attn_weights, input_ids, AEL)
                return calibrated

        except Exception as e:
            print(f"[= =] Error in uncertainty handling: {e}")
            traceback.print_exc()
            return probs


    def process_peer_request(self, probs, target_preds, attn_weights, input_ids):
        """
        Handle an incoming request from a peer needing a second opinion:
        produces this agent's own view given the peer's `probs`/
        `target_preds`/`attn_weights`/`input_ids` and returns it (used from
        the message-handling path rather than being a client-initiated
        call).
        """
        if probs is not None and target_preds is not None and attn_weights is not None and input_ids is not None:
            try:
                response_probs = self.pipeline._calibrate_probs(probs, target_preds, attn_weights, input_ids)
                return response_probs
            except Exception as e:
                print(f"[-] Error in peer request_processing: {e}")
                return probs
        else:
            print('[=] Cannot process peer request due to incomplete Missing samples, returning regular probs!')
            return probs
        
            

    # ============ REQUEST HANDLERS ============
    def get_external_peer_message(self):
        """Block waiting for/pull the next inbound message from an external (socket-connected) peer, used by `_handle_peer_agent_request` for the 'ExternalPeer' path."""
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
        """
        Server-side handler for an incoming PREDICT_REQUEST message: runs
        the local prediction (optionally via `predict_manager` when
        `method != 'basic_prediction'`) and sends a PREDICT_RESPONSE back
        to the requester.
        """
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
            X = message.get('X')
            y = message.get('y')

            print(f'[DEBUG] X immediately after deserialization: '
                f'shape={np.asarray(X).shape if X is not None else None}')

            if not titles and label_map and rules:
                return {'type': self.MSG_TYPES['PREDICT_RESPONSE'], 'error': 'No test titles provided'}
            
            titles = self._sanitize_arrays_and_dicts(titles)
            label_map = self._sanitize_arrays_and_dicts(label_map)
            rules = self._sanitize_arrays_and_dicts(rules)
            X = self._sanitize_arrays_and_dicts(X)
            y = self._sanitize_arrays_and_dicts(y)

            if not self._check_rate_limit(sender_id):
                return {'type': self.MSG_TYPES['PREDICT_RESPONSE'], 'error': 'Rate limit exceeded'}

            try:
                result, chosen_label, confidence = self.predict_manager.advanced_prediction_method(titles, label_map, rules, X=X, y=y, show_proba=True, use_transformer=self.pipeline.use_transformer)
            
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
        """Server-side handler for MEMORY_SYNC_REQUEST: reads the requested memory via `storage` and responds with a MEMORY_SYNC_RESPONSE."""
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
        """Server-side handler for ENSEMBLE_VOTE_REQUEST: computes this agent's vote/prediction and responds with an ENSEMBLE_VOTE_RESPONSE."""
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
        """Server-side handler for FAILURE_REPORT: logs a peer-reported failure (see `report_failure`) for later inspection/trust adjustment."""
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
        """Server-side handler for TRUST_UPDATE: applies a peer-initiated change to a trust level recorded in `trusted_agents`, subject to permission checks."""
        # Handle trust score update
        target_agent = message.get('target_agent')
        new_trust = message.get('trust_score')

        self.query_node.peer_trust = new_trust
        
        if target_agent in self.remote_agents:
            self.remote_agents[target_agent]['trust'] = new_trust
        
        return {'type': 'ack', 'status': 'trust_updated'}


    # ============ REQUEST SENDING METHODS ============       
    def request_prediction_method(self, agent_id, text, timeout=5):
        """Convenience wrapper around `request_prediction`/`_process_message` for requesting a peer prediction and unwrapping just the method result."""
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
        """Send an ENSEMBLE_VOTE_REQUEST to `agent_id` for `text` and wait (up to `timeout`) for its ENSEMBLE_VOTE_RESPONSE."""
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
            print(f"[-] Vote request failed: {e}")
            return None, None
    
    def sync_memory_with_agent(self, agent_id, memory_name, timeout=10):
        """Send a MEMORY_SYNC_REQUEST for `memory_name` to `agent_id` and wait (up to `timeout`) for the synced memory payload."""
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
        """
        Record a failure for `agent_id`/`task_type` with a human-readable
        `reason`, both locally (for trust-level bookkeeping) and, where
        applicable, forwarded to the peer as a FAILURE_REPORT message.
        """
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
        """Send a PING to every connected peer in `remote_agents`, used as a liveness check outside of the automatic health-checker loop."""
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
        """
        Look for another local process/pipeline sharing the same
        `memory_name` via shared storage (rather than the network), and
        pull its most recent prediction context if present — the
        'DevicePeer' path used by `_handle_peer_agent_request` before
        falling back to a real network peer.
        """
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
        """Persist `data` under `memory_name` via `storage` so another local process sharing the same memory namespace can pick it up (`sync_with_local_peer`)."""
        try:
            self.pipeline.storage.save_model_dict(memory_name, data)
            print(f"✅ Saved local peer presence: {memory_name}")
            return True
        except Exception as e:
            print(f"Save to local peer failed: {e}")
            return False
    
    # ============ UTILITY METHODS ============
    
    def _log_interaction(self, agent_id, interaction_type, confidence, details=None):
        """Append a record of a peer interaction (agent, type, confidence, extra details) to `agent_comm_log` for later auditing/inspection."""
        if agent_id not in self.agent_comm_log:
            self.agent_comm_log[agent_id] = []
        
        self.agent_comm_log[agent_id].append({
            'timestamp': datetime.now(),
            'type': interaction_type,
            'confidence': confidence,
            'details': details
        })
    
    def get_agent_status(self):
        """Return a summary dict of this agent's current state: connected peers, trust levels, queue depth, server running state, etc."""
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
        """Return up to `limit` recent entries from `agent_comm_log`, optionally filtered to a single `agent_id`."""
        # Get communication log for an agent
        if agent_id:
            return self.agent_comm_log.get(agent_id, [])[-limit:]
        
        # Return all logs
        return self.agent_comm_log
    
    def print_network_status(self):
        """Pretty-print a human-readable snapshot of connected peers, trust levels, and queue/health stats to stdout."""
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

# The query_node.QueryNode class manages the connection and interaction with other nodes (agents) in the network. It handles node identification, agreement evaluation, safety checks, and maintains a memory of connected nodes. 
# The class allows for flexible interactions while ensuring the safety and integrity of the Master node.


