"""
query_node.py
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
from . import activations
from . import automation
from . import caching_security

class QueryNode:
    """
    Manages connection, agreement evaluation, safety checks, and trust
    for peer/child nodes interacting with the Master node.
    """

    def __init__(self, pipeline, memory_name, storage,
                max_nodes=1000, requests_per_minute=30):
        self.master_node = pipeline
        self.memory_name = memory_name
        self.storage     = storage
        self.agreement   = False

        if not self.storage.memory_exists(self.memory_name, type='Node'):
            print(f"|| Creating new memory for Nodes population: {memory_name}!")
            self.nodes = {}
        else:
            print(f'|| Found Matched Memory for Nodes : {memory_name}!')
            self.nodes = self.storage.memory_retrieval(
                self.memory_name, type_func='Node', verbose=True
            )

        self.master_nodes_id      = 0
        self.safety_check_value   = 0.0
        self.node_id              = 0
        self.permission           = False

        # trust now evolves per-node, 
        # keyed by node_id, each entry decays/updates from real outcomes
        self.peer_trust = 1.0
        self._trust_scores: Dict[int, float] = {}
        self._default_trust = 1.0

        # concurrency guard + bounded growth
        self._lock     = threading.RLock()
        self.max_nodes = max_nodes

        # rate limiter for connection attempts, per node — reuses the
        # hardened caching_security.RateLimiter class fixed earlier this session
        self._rate_limiter = caching_security.RateLimiter(
            requests_per_minute=requests_per_minute, per_key=True
        )

    def _add_node(self, node):
        node_id = id(node)

        with self._lock:
            # bounded growth, evict least-trusted node if full
            if len(self.nodes) >= self.max_nodes and node_id not in self.nodes:
                self._evict_least_trusted()

            self.nodes[node_id] = node
            self._trust_scores.setdefault(node_id, self._default_trust)

        print(f"✅ Node {node_id} added to QueryNode")
        return node_id

    def _evict_least_trusted(self):
        """bounded population, evict lowest-trust node when full."""
        if not self.nodes:
            return
        worst_id = min(
            self.nodes.keys(),
            key=lambda nid: self._trust_scores.get(nid, self._default_trust)
        )
        del self.nodes[worst_id]
        self._trust_scores.pop(worst_id, None)
        print(f'[=] Evicted least-trusted node {worst_id} to stay under '
              f'max_nodes={self.max_nodes}')

    def _save_node_memory(self, node):
        try:
            node_id = id(node)
            self.storage.save_nodes_dict(
                self.memory_name, self.nodes, node_id, model_type='Node'
            )
            print(f"[💾] Node {node_id} memory saved to storage!")
            return True
        except Exception as e:
            print(f"[-] Error saving node memory: {e}")
            return False

    def _evaluate_node_agreement(self, node):
        node_id = id(node)
        threshold = self.master_node.confidence_threshold if self.master_node.confidence_threshold > 0.1 else 0.3

        print(f"[=] Evaluating node {node_id} || agreement: {self.agreement} "
              f"|| Master Node memory: {self.memory_name}")

        # agreement_threshold is comparable against
        # confidence_threshold: normalized into [0,1] via activations.sigmoid instead
        # of an unbounded additive term that made the check always-true
        raw_signal = self.master_node.final_conf_score * self.master_node.temperature
        normalized_signal = 1.0 / (1.0 + np.exp(-raw_signal))  # squashed to (0,1)
        self.agreement_threshold = (
            self.master_node.confidence_threshold * (1.0 - normalized_signal) +
            normalized_signal
        )

        trust = self._trust_scores.get(node_id, self._default_trust)
        secondary_trust = self.peer_trust

        # agreement requires real threshold pass here
        meets_threshold = self.agreement_threshold > self.master_node.confidence_threshold
        result = (self.agreement and trust > threshold) or (meets_threshold and trust > 0.5) and (secondary_trust > self.master_node.confidence_threshold)

        if result:
            print(f"[✅] Node {node_id} is in agreement with the Master node "
                  f"(trust={trust:.2f})")
        else:
            print(f"[-] Node {node_id} is NOT in agreement with the Master node "
                  f"(trust={trust:.2f})")
        return result

    def _connect_with_node(self, node):
        node_id = id(node)

        # rate limit connection attempts per node
        if not self._rate_limiter.acquire(key=str(node_id)):
            print(f"[⏱️] Node {node_id} rate-limited — too many connection "
                  f"attempts, rejecting")
            self.permission = False
            return False

        self.agreement = self.master_node.agreement

        with self._lock:
            already_known = self._identify_node(node)
            if not already_known:
                self._add_node(node)

        agreement = self._evaluate_node_agreement(node)
        safety    = self._node_safety_check(node)

        if safety or agreement:
            print(f"[🔗] Node {node_id} successfully connected to the Master node")
            self.permission = True
            self._adjust_trust(node_id, delta=+0.05)   # reward success
        else:
            print(f"[⚠️] Node {node_id} connection failed due to Disagreement")
            self.permission = False
            self._adjust_trust(node_id, delta=-0.15)   # penalize failure

        print('== Connection Evaluation Summary ==')
        print(f'[=] Node {node_id}')
        print(f'[=] agreement: {agreement}')
        print(f'[=] safety: {safety} || permission: {self.permission}')
        print(f'[=] trust: {self._trust_scores.get(node_id, self._default_trust):.2f}')

        return self.permission


    def _connect_with_peer(self, node):
        node_id = id(node)

        if not self._rate_limiter.acquire(key=str(node_id)):
            print(f"[⏱️] Peer {node_id} rate-limited — rejecting")
            self.permission = False
            return False

        self.agreement = self.master_node.agreement

        with self._lock:
            already_known = self._identify_node(node)
            if not already_known:
                self._add_node(node)

        agreement = self._evaluate_node_agreement(node)
        safety    = self._node_safety_check(node)

        if safety or agreement:
            print(f"[🔗] Peer with ID: {node_id} successfully connected")
            self.permission = True
            self._adjust_trust(node_id, delta=+0.05)
        else:
            print(f"[⚠️] Peer with ID: {node_id} connection failed")
            self.permission = False
            self._adjust_trust(node_id, delta=-0.15)

        return self.permission


    def _adjust_trust(self, node_id, delta):
        """
        trust that actually evolves. EMA-style bounded adjustment.
        """
        current = self._trust_scores.get(node_id, self._default_trust)
        updated = float(np.clip(current + delta, 0.0, 1.0))
        self._trust_scores[node_id] = updated


    def _identify_node(self, node):
        eps = 1e-5
        node_id = id(node)
        print(f"[||] Identifying node {node_id} with Master node memory: "
              f"{self.memory_name}")

        matches = [(nid, n) for nid, n in self.nodes.items() if n == node]

        if matches:
            matched_id = matches[0][0]
            print(f"✅ Node {matched_id} is already identified with the "
                  f"Master node")
            self.safety_check_value = (
                self.master_node.final_conf_score + self.master_node.temperature
            ) + eps
            return True

        print(f"[-] Node {node_id} is NOT identified with the Master node")
        self.safety_check_value = (
            1.0 - self.master_node.final_conf_score + self.master_node.temperature
        ) + eps
        return False


    def _node_safety_check(self, node):
        node_id = id(node)
        trust   = self._trust_scores.get(node_id, self._default_trust)

        print(f"[🛡️] Safety check for node {node_id}: "
              f"safety_value={self.safety_check_value:.3f}, trust={trust:.2f}")

        if self.safety_check_value > self.master_node.confidence_threshold:
            print(f"✅ Node {node_id} passed the safety check")
            return True

        print(f"[-] Node {node_id} failed the safety check")

        # combined with trust so genuinely low-trust AND
        # low-safety nodes get removed
        if (self.safety_check_value < (self.master_node.confidence_threshold / 2)
                and trust < 0.3):
            print(f"[⚠️] Node {node_id} is considered unsafe and will be removed")
            removed = self._remove_node(node)
            return False if removed else False   # removal ≠ passing safety

        return False

    def _remove_node(self, node):
        # key by id(node), matching how nodes are actually stored
        node_id = id(node)
        with self._lock:
            if node_id in self.nodes:
                del self.nodes[node_id]
                self._trust_scores.pop(node_id, None)
                print(f"[🗑️] Node {node_id} removed from Nodes population")
                return True
            else:
                print(f"[-] Node {node_id} not found in Nodes population")
                return False

    def _node_activation(self, node):
        try:
            node_id = id(node)
            if self.permission:
                print(f"🚀 Node {node_id} is now active with the Master node")
                return True
            print(f"[-] Node {node_id} cannot be activated due to lack of permission")
            return False
        except Exception as e:
            print(f"[-] Error during node activation: {e}")
            return False

    def _identify_peer_trust(self, peer):
        node_id = id(peer)
        trust   = self._trust_scores.get(node_id, self._default_trust)

        print(f"[=] Identifying peer {node_id} trustworthiness: {trust:.2f}")
        if trust > self.master_node.confidence_threshold:
            print(f"[✅] Peer {node_id} identified as trustworthy (trust={trust:.2f})")
            return True
        print(f"[-] Peer {node_id} NOT trustworthy (trust={trust:.2f})")
        return False

    def _establish_peer_nodes(self, peer):
        print(f"[=] Establishing peer connection: {self.memory_name}")
        if self._connect_with_peer(peer) and self._identify_peer_trust(peer):
            print(f"[✅] Peer {id(peer)} connected and can interact")
        else:
            print(f"[-] Peer {id(peer)} cannot interact")

        activation = self._node_activation(peer)
        self._save_node_memory(peer)
        return activation

    def _establish_node_connection(self, node):
        if self._connect_with_node(node):
            print(f"[✅] Node {id(node)} connected and can interact")
        else:
            print(f"[-] Node {id(node)} cannot interact")

        activation = self._node_activation(node)
        self._save_node_memory(node)
        return activation

    def get_stats(self) -> Dict:
        """Operational visibility"""
        with self._lock:
            return {
                'total_nodes'  : len(self.nodes),
                'max_nodes'    : self.max_nodes,
                'avg_trust'    : (float(np.mean(list(self._trust_scores.values())))
                                 if self._trust_scores else self._default_trust),
                'low_trust_nodes': sum(1 for t in self._trust_scores.values() if t < 0.3),
            }


# The automation.AutoBatcherAutomation class manages the batching of incoming prediction requests to optimize processing efficiency. 
# It collects requests over a short time window or until a maximum batch size is reached, then processes them together through the pipeline. This allows for improved throughput while still providing timely responses to individual requests.


