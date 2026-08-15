"""
peer_agent.py
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
from . import distributed_inference
from . import pipeline
from . import prediction_manager

class ConsecutivePeerAgent:
    """
    Lightweight, self-contained P2P peer used as the *fallback* transport for
    distributed/ensemble prediction when the primary P2P stack
    (`distributed_inference.AgentDistributedInference`, `self.pipeline.distribution`) is
    unavailable, misconfigured, or the caller just wants a simpler
    "ask my peers" mechanism.

    Wire protocol
    --------------
    Every message sent over the socket is a length-prefixed JSON blob:

        [4 bytes big-endian length][UTF-8 JSON payload]

    Before sending, any `numpy.ndarray` values inside the message are
    replaced with a shape-preserving dict via `_encode_arrays_for_wire`
    (JSON has no native array/dtype concept), and on receipt they are
    restored with `_decode_arrays_from_wire`. Every outgoing message is
    HMAC-SHA256 signed with `secret_key` (`_sign_message`) and every
    incoming message that carries a `signature` field is verified
    (`_verify_signature`) using `hmac.compare_digest` (constant-time, to
    avoid timing side-channel attacks on the signature check).

    Connection lifecycle
    ---------------------
    1. `start_server()` spins up a background thread that `accept()`s
       incoming TCP connections (optionally wrapped in TLS via
       `self.pipeline.distribution.ssl_context`), filters by `allowed_ips`,
       and hands each connection to `_handle_client` on its own thread.
    2. A peer that wants a prediction calls `request_peer_prediction`,
       which lazily opens+authenticates a socket (caching it in
       `connected_peers`) and then sends a `predict` request.
    3. `_handle_client` mirrors this on the server side: it expects an
       `auth` message first (checked against the shared `secret_key`),
       replies with `auth_response`, and then services `predict`/`ping`
       messages in a loop until the peer disconnects.
    4. `ensemble_predict` is the main entry point used by callers: it runs
       a local prediction first and only fans out to peers when local
       confidence is below `confidence_threshold` (or when peer results are
       explicitly wanted for cross-checking).

    Security notes
    ---------------
    - Authentication is a shared-secret token compared with plain `==`
      inside `_handle_client` (not constant-time) but message integrity is
      protected via the constant-time-verified HMAC signature above.
    - `allowed_ips` defaults to localhost-only; production deployments are
      expected to populate it (or rely on the TLS layer) before opening the
      server to a real network.
    - `max_message_size` bounds inbound message size to guard against
      memory-exhaustion from a malicious/broken peer.

    Attributes:
        peer_id (str): Identifier this agent announces to peers.
        port (int): TCP port the server listens on.
        secret_key (str): Shared HMAC/auth secret for this peer group.
        manager (prediction_manager.PipelinePredictionManager | None): Optional prediction
            manager used for label-aware local predictions.
        pipeline (pipeline.IntegratedPipeline | None): Owning pipeline, used both for
            plain `predict_single` fallback and for reaching the SSL
            context configured on `pipeline.distribution`.
        connected_peers (Dict[str, Dict]): Live outbound connections keyed
            by `"host:port"`, each entry holding the socket and metadata.
        server_socket (socket.socket | None): Listening socket once
            `start_server()` has run.
        running (bool): Server loop flag; set False by `stop_server()`.
        stats (Dict): Running counters (`predictions`, `peer_requests`,
            `errors`) exposed via `get_stats()`.
    """
    
    def __init__(self, peer_id: str, port: int, secret_key: str, 
                 manager=None, pipeline=None):
        self.peer_id = peer_id
        self.port = port
        self.secret_key = secret_key
        self.manager = manager  # prediction_manager.PipelinePredictionManager
        self.pipeline = pipeline  # pipeline.IntegratedPipeline
        
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
        """
        Compute an HMAC-SHA256 signature over a message.

        The message is copied (dropping any existing `signature` key so the
        signature never signs itself), its keys are sorted so the signed
        byte representation is deterministic regardless of dict insertion
        order, then JSON-serialized and signed with `secret_key`.

        Args:
            message (dict): Message payload to sign (may already include
                other metadata such as `timestamp`, but not `signature`).

        Returns:
            str: Hex-encoded HMAC-SHA256 digest.
        """
        msg_copy = {k: v for k, v in message.items() if k != 'signature'}
        sorted_msg = {k: msg_copy[k] for k in sorted(msg_copy.keys())}
        msg_bytes = json.dumps(sorted_msg, default=str).encode('utf-8')
        key = self.secret_key.encode()
        return hmac.new(key, msg_bytes, hashlib.sha256).hexdigest()
    
    def _verify_signature(self, message: dict, signature: str) -> bool:
        """
        Verify a received message's HMAC signature.

        Recomputes the expected signature over `message` (with any
        pre-existing `signature` key excluded, matching how `_sign_message`
        was originally applied) and compares it to `signature` using
        `hmac.compare_digest` to avoid timing-based signature forgery.

        Args:
            message (dict): The message body as received (signature field
                already popped out by the caller).
            signature (str): Hex-encoded signature to check against.

        Returns:
            bool: True if the signature matches, False otherwise.
        """
        # Verify message signature
        expected = self._sign_message({k: v for k, v in message.items() if k != 'signature'})

        print(f'[ConsecutivePeerAgent] Comparing Signature and verifying...')
        return hmac.compare_digest(expected, signature)
    
    def _send_message(self, sock: socket.socket, message: dict) -> bool:
        """
        Frame, sign, and send a single message over an open socket.

        Pipeline: encode any numpy arrays for JSON-safety
        (`_encode_arrays_for_wire`) -> stamp a `timestamp` -> sign the
        message (`_sign_message`) -> JSON-serialize -> prefix with a 4-byte
        big-endian length header -> write both to the socket.

        Args:
            sock (socket.socket): Destination socket (may be TLS-wrapped).
            message (dict): Message payload to send. Not mutated in place;
                a copy is signed/sent.

        Returns:
            bool | None: True on success, False if sending raised an
            exception, or None if `sock` was None (caller should treat
            both falsy results as failure).
        """
        try:
            if sock is None:
                print('[=] Sock is None !')  
                return None

            msg_copy = message.copy()

            msg_copy = self._encode_arrays_for_wire(msg_copy)
            msg_copy['timestamp'] = time.time()
            msg_copy['signature'] = self._sign_message(msg_copy)   

            data = json.dumps(msg_copy, default=str).encode('utf-8')

            sock.sendall(len(data).to_bytes(4, 'big'))
            sock.sendall(data)
            return True
        except Exception as e:
            print(f"[ConsecutivePeerAgent] Send error: {e}")
            return False
            
    def _encode_arrays_for_wire(self, obj):
        """
        Recursively replace numpy arrays with a shape-preserving dict
        BEFORE json.dumps runs, so 2D+ structure survives the wire.

        Each ndarray becomes `{'__ndarray__': True, 'data': <flat list>,
        'shape': [...], 'dtype': '<numpy dtype string>'}`; dicts/lists/
        tuples are walked recursively so arrays nested at any depth are
        caught, and any other value is passed through unchanged.

        Args:
            obj: Arbitrary (possibly nested) message value.

        Returns:
            The same structure with all `numpy.ndarray` instances replaced
            by their JSON-serializable encoding.
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
        """
        Reverse of `_encode_arrays_for_wire` — reconstructs real shape.

        Walks the decoded JSON structure and, wherever an
        `{'__ndarray__': True, ...}` marker dict is found, rebuilds the
        original `numpy.ndarray` (dtype + shape) from its flat `data` list.

        Args:
            obj: Decoded JSON value (dict/list/scalar) as produced by
                `json.loads`.

        Returns:
            The same structure with ndarray-marker dicts restored to real
            `numpy.ndarray` objects.
        """
        if isinstance(obj, dict):
            if obj.get('__ndarray__'):
                data  = np.array(obj['data'], dtype=obj.get('dtype', 'float64'))
                shape = tuple(obj['shape'])
                return data.reshape(shape)
            return {k: self._decode_arrays_from_wire(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._decode_arrays_from_wire(v) for v in obj]
        return obj 

    def _receive_message(self, sock: socket.socket) -> Optional[dict]:
        """
        Read one length-prefixed JSON message from a socket and verify it.

        Reads the 4-byte length header, rejects messages larger than
        `max_message_size` (anti-DoS), then reads exactly that many bytes
        in a recv loop (handling short reads/partial TCP segments). The
        payload is JSON-decoded, run through `_decode_arrays_from_wire` to
        restore any numpy arrays, and — if it carries a `signature` — that
        signature is checked with `_verify_signature`; a bad signature
        causes the message to be silently dropped (returns None) rather
        than raising, so the connection can continue or be closed by the
        caller as appropriate.

        Args:
            sock (socket.socket): Source socket to read from.

        Returns:
            Optional[dict]: The decoded message, or None if the connection
            was closed, the message was malformed/oversized, or signature
            verification failed.
        """
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

            try:
                message = json.loads(data.decode('utf-8'))
                print(f'[ConsecutivePeerAgent] Received a message!')
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                logger.warning(f'[=] Invalid JSON from peer: {e}')
                self._log_security_event('invalid_json', {})
                return None

            message = self._decode_arrays_from_wire(message)
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
        """
        Produce a prediction from local models only (no peer fan-out).

        Dispatch depends on what's wired up and what `text` looks like:
        - If `self.manager` (a `prediction_manager.PipelinePredictionManager`) is set and
          `text` is the rich dict form (`{'test_titles', 'label_map',
          'rules', 'X', 'y'}`), delegates to
          `manager.advanced_prediction_method` for a label-aware,
          rule-assisted prediction.
        - If `self.manager` is set but `text` is plain input, falls back to
          `self.pipeline.predict_single(text)` and uses the pipeline's
          configured `confidence_threshold` as a conservative confidence
          estimate (since a raw label has no probability attached).
        - If there's no `manager` but there is a `pipeline`, calls
          `pipeline.predict_single` and reads `prediction`/`confidence`
          out of its result dict.
        - If neither is available, returns an `'unknown'` placeholder
          prediction with 0.5 confidence.

        Any exception during prediction is caught and turned into an
        `'error'` result rather than propagating, so a single bad
        prediction can't take down the peer's request/response loop.

        Args:
            text: The input to predict on — either raw model input or the
                structured dict described above.

        Returns:
            Dict: `{'text', 'prediction', 'confidence', 'source': 'local'}`,
            plus `'error'` when an exception was caught.
        """
        try:
            # Use your existing advanced prediction method
            if self.manager:
                # For single text, wrap in list
                if 'test_titles' in text:
                    test_titles = text['test_titles']
                    label_map = text['label_map']
                    rules = text['rules']
                    X = text['X']
                    y = text['y']
                    result, chosen_label, confidence = self.manager.advanced_prediction_method(
                        test_titles,
                        label_map,
                        rules,
                        X=X, y=y,
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
        """
        Ask a single remote peer for a prediction, connecting/authenticating
        lazily and reusing the connection on subsequent calls.

        Behavior:
        - Looks up `"host:port"` in `connected_peers`; if not already
          connected, opens a new TCP socket (TLS-wrapped using
          `pipeline.distribution.client_ssl_context` if available,
          otherwise an unverified TLS context as a best-effort fallback),
          binds an ephemeral local port, and connects with `timeout`.
        - Refuses to connect to itself (same host + own `self.port`) to
          avoid a degenerate self-request loop.
        - Sends an `auth` message containing `peer_id` + `secret_key` and
          waits for an `{'status': 'ok'}` response before caching the
          connection in `connected_peers`.
        - Sends a `predict` request with the raw `text` (only the text is
          sent to the peer — no local model state), waits for a
          `predict_response`, and increments `stats['peer_requests']` on
          success.
        - On any connection/send/receive error, cleans up and evicts the
          (now presumed-dead) cached connection for this peer so the next
          call will reconnect from scratch, and returns None.

        Args:
            peer_host: Peer's host/IP.
            peer_port (int): Peer's listening port.
            text: Input to send for prediction (sent as-is; keep it
                JSON/array-serializable).
            timeout (float): Socket timeout in seconds for connect/send/
                receive.

        Returns:
            Optional[Dict]: `{'text', 'prediction', 'confidence',
            'source': 'peer_<host>:<port>'}` on success, or None if the
            connection/auth/request failed or self-connection was detected.
        """
        
        peer_key = f"{peer_host}:{peer_port}"
        
        with self._lock:
            # Check if already connected
            if peer_key not in self.connected_peers:
                # Create new connection
                try:
                    if self.pipeline.distribution.client_ssl_context:
                        print('[+] Prediction Request is Initiated with SSL.')
                        sock = self.pipeline.distribution.client_ssl_context.wrap_socket(
                            socket.socket(socket.AF_INET, socket.SOCK_STREAM),
                            server_hostname=peer_host
                        )
                    else:
                        print('[!] Prediction Request is Initiated without Any SSL!')
                        client_ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
                        client_ctx.check_hostname = False
                        client_ctx.verify_mode = ssl.CERT_NONE  
                        sock = client_ctx.wrap_socket(
                            socket.socket(socket.AF_INET, socket.SOCK_STREAM),
                            server_hostname=peer_host
                        )
        
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

                    print(f'[ConsecutivePeerAgent] Got Authentication response from peer!')
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
        Top-level P2P ensemble entry point: predict locally first, then
        consult peers when local confidence is low (or to cross-check).

        Flow:
        1. Run `predict_local(text)` and treat it as the current best
           result.
        2. If local confidence is below `confidence_threshold` (or peer
           addresses were supplied for verification even at higher
           confidence), query every address in `peer_addresses` via
           `request_peer_prediction`, collecting whichever ones respond.
        3. If any peer responded:
           - If the collected results are plain dicts, the *highest
             single-peer confidence* result's prediction is taken as-is
             but reported with a *combined* confidence normalized by
             `len(peer_addresses) + 1` (a simple, conservative way to
             avoid overstating certainty just because more peers agreed to
             answer, rather than because they agreed on the label) — note
             this branch returns the raw `best_peer` dict directly.
           - Otherwise, only replaces `best_result` with a peer's result
             if that peer's confidence strictly exceeds the local one.
        4. If no peer addresses were given at all, returns the local result
           immediately without touching `stats['predictions']` again below.

        `stats['predictions']` is incremented once per call that reaches
        the peer-querying branch (whether or not a peer improved on the
        local result).

        Args:
            peer_addresses (List[Tuple[str, int]]): `(host, port)` pairs of
                peers to consult if needed.
            text: Input to predict on, forwarded to both local and peer
                prediction paths.
            confidence_threshold (float): Local confidence below which
                peers are consulted for a potentially better answer.

        Returns:
            Dict: A prediction result dict (shape matches `predict_local`/
            `request_peer_prediction` output) representing the best answer
            found.
        """
        print(f"[ConsecutivePeerAgent] Starting ensemble prediction with port {self.port}!")
        print(f'[ConsecutivePeerAgent] Peer Addresses: {peer_addresses}')
        
        # Step 1: Local prediction
        local_result = await self.predict_local(text)
        print(f"[ConsecutivePeerAgent] Local: {local_result['prediction']} ({local_result['confidence']:.1%})")
        
        best_result = local_result
        
        # Step 2: If low confidence, asking peers
        if local_result['confidence'] < confidence_threshold and peer_addresses or peer_addresses:
            if local_result['confidence'] < confidence_threshold:
                print(f"[ConsecutivePeerAgent] Low confidence, asking {len(peer_addresses)} peers...")
            else:
                print(f'[ConsecutivePeerAgent] Verifying answer.., asking {len(peer_addresses)} peers...')
            
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
                if isinstance(peer_results[0], dict):
                    total_weight = len(peer_addresses) + 1
                    best_result = best_peer.get('prediction') 
                    best_confidence = min(best_peer.get('confidence') / total_weight, 1.0)
                    
                    print(f"[ConsecutivePeerAgent] Using Ensemble prediction result is: {best_result} || With Confidence: ({best_confidence:.1%})")
                    self.stats['predictions'] += 1

                    return best_peer  

                if best_peer['confidence'] > local_result['confidence']:
                    best_result = best_peer
                    print(f"[ConsecutivePeerAgent] Using peer result: {best_peer['prediction']} || Confidence: ({best_peer['confidence']:.1%})")
        else:
            print('[ConsecutivePeerAgent] Skipping Ensemble prediction... Peer address is None or empty')
            return best_result

        self.stats['predictions'] += 1
        return best_result
    

    def start_server(self):
        """
        Start the P2P listener in a background daemon thread.

        The inner `server_loop`:
        - Creates a `SO_REUSEADDR` TCP socket bound to `0.0.0.0:self.port`
          with a 1-second `accept()` timeout (so the loop can periodically
          re-check `self.running` and exit cleanly on `stop_server()`).
        - Wraps the socket in TLS server mode if
          `pipeline.distribution.enable_ssl` and an `ssl_context` are
          configured.
        - For each accepted connection, rejects IPs not in `allowed_ips`
          and otherwise spins up a daemon thread running `_handle_client`
          so multiple peers can be served concurrently.
        - On loop exit, closes the listening socket.

        Note: this method itself returns immediately after starting the
        thread; the "Server Successfully Stopped" print right after
        `thread.start()` reflects source ordering rather than actual
        shutdown (it does not block on the server loop).
        """
        
        def server_loop():
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind(('0.0.0.0', self.port))
            self.server_socket.settimeout(1.0)           
            self.server_socket.listen(5)

            if self.pipeline.distribution.enable_ssl and self.pipeline.distribution.ssl_context:
                self.pipeline.distribution.ssl_context.check_hostname = False
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
        """
        Per-connection handler run on its own daemon thread by
        `start_server`'s accept loop.

        Protocol handled:
        1. Refuses a connection that appears to be the server talking to
           itself (same loopback address + own port).
        2. Requires the first message to be `{'type': 'auth', 'token': ...}`
           with `token == self.secret_key`; anything else (missing
           message, wrong type, wrong token) closes the connection
           immediately without a response.
        3. On successful auth, replies `{'type': 'auth_response',
           'status': 'ok'}` and then loops reading messages:
           - `predict`: runs `predict_local` (via `asyncio.run`, since this
             handler itself is synchronous/threaded rather than async) and
             replies with a `predict_response` carrying the prediction and
             confidence.
           - `ping`: replies `pong` (simple liveness check).
           - Loop exits when `_receive_message` returns None (peer closed
             the connection or sent something unparseable).
        4. The socket is always closed in a `finally` block, whether the
           loop exited normally or an exception was raised.

        Args:
            client (socket.socket): The accepted per-peer socket.
            addr: The peer's `(host, port)` address tuple from `accept()`.
        """
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
        """
        Gracefully shut down the P2P agent: stop the accept loop, close
        every cached outbound peer connection (`connected_peers`), and
        close the listening socket.

        Sets `self.running = False` first so the server thread's next loop
        iteration (or timeout) exits on its own; socket-close failures are
        swallowed (best-effort cleanup) since the process is shutting down
        this component regardless.
        """
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
        """
        Snapshot of this agent's activity counters.

        Returns:
            Dict: A copy of `self.stats` (`predictions`, `peer_requests`,
            `errors`) plus a `connected_peers` count reflecting the current
            size of `self.connected_peers`.
        """
        # Get statistics
        return {
            **self.stats,
            'connected_peers': len(self.connected_peers)
        }

    


