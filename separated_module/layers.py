"""
layers.py
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
from . import mlp
from . import weight_shaping

class Dense:
    def __init__(self, x, input_size, output_size, activation=None):

        self.special_weight = weight_shaping.GeometricWeightShaping(input_size, output_size)
        self.W = self.special_weight.weight_shaping(x)

        self.b = np.zeros((1, output_size))
        self.activation_name = activation
       

        if activation:
            self.activation = getattr(activations.Activation, activation)
            self.activation_derivative = getattr(activations.Activation, activation + "_derivative")
        else:
            self.activation = None
            self.activation_derivative = None

    def _sanitize_string_chars(self, x):
        if isinstance(x, (str, np.str_)):
            clean_str = str(x).replace('[', '').replace(']', '').replace('...', '').strip()
            x = np.fromstring(clean_str, sep=' ')

        if isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.character):
            # catches arrays filled with string text
            clean_str = ' '.join(x.astype(str).flatten()).replace('[', '').replace(']', '')
            skip_values = {"...", "NaN", "null"}
            x = np.fromiter((v for v in clean_str.split() if v not in skip_values), dtype=float)

        return x

    def multi_modal_linear_transformation(self, x, perf_score):
        x = self._sanitize_string_chars(x)

        if len(x.shape) > 1 and x.shape[1] != self.W.shape[0]:
            V1, V2 = x.shape[0], x.shape[1]            
            try:
                self.W = self.W[:V2, :]
            except:
                self.special_weight = weight_shaping.GeometricWeightShaping(V2, V1)
                self.W = self.special_weight.weight_shaping(x)

        try:
            try:
                z = np.dot(x, self.W) + self.b
            except:
                subnet_W = self.W[:x.shape[1], :x.shape[0]]

                sub_z = np.dot(x, subnet_W)
                sub_b = self.b[:sub_z.shape[1], :sub_z.shape[0]]

                z = sub_z + sub_b

        except:
            try:
                subnet_W = self.W[:x.shape[1]:, :x.shape[0]]
                sub_z = np.dot(x, subnet_W)
            except:
                weight = self.W

                try:
                    subnet_x = x[:, :weight.shape[0]]
                    subnet_W = weight[:x.shape[1], :]                    
                except:
                    subnet_x = x[:weight.shape[0]]
                    subnet_W = weight[:x.shape[0]]

                sub_z = np.dot(subnet_x, subnet_W)

            try:
                subnet_B = self.b[:sub_z.shape[0], :sub_z.shape[1]]
            except:
                subnet_B = self.b[:sub_z.shape[0]]
                
            z = (sub_z + subnet_B) * perf_score 
        
        return z


    def forward(self, x, perf_score):

        x = self._sanitize_string_chars(x)
        self.x = x
        self.z = self.multi_modal_linear_transformation(x, perf_score)

        if self.activation:
            self.a = self.activation(self.z)
        else:
            self.a = self.z

        return self.a


    def backward(self, da, lr, perf_score, clip_value=1.0):
        eps = 1e-5
        batch_size = self.x.shape[0]

        if self.activation_derivative:
            dz = da * self.activation_derivative(self.z)
        else:
            dz = da

        dW = np.dot(self.x.T, dz) / batch_size
        db = np.sum(dz, axis=0, keepdims=True) / batch_size
        dx = np.dot(dz, self.W.T)

        # global norm clipping.
        # Preserves gradient DIRECTION, only scales magnitude down when
        # the total norm exceeds clip_value — avoided the frozen-equilibrium
        # pattern that per-element clipping can produce.
        total_norm = np.sqrt(np.sum(dW ** 2) + np.sum(db ** 2))
        if total_norm > clip_value:
            scale = clip_value / (total_norm + eps)
            dW *= scale
            db *= scale

        self.W -= (lr * dW) 
        self.b -= (lr * db) + (1.0 + perf_score) * 1e-5  # small bias regularization

        return dx
        

    	
    	
class SoftmaxOutput:
    def forward(self, x):
        self.out = activations.Activation.softmax(x)

        return self.out

    def backward(self, dL_dZ):
        # gradient already computed as y_pred - y_true
        return dL_dZ


# enhanced mlp.MLP with focused forward and backward for better handling of data with varying geometric complexity,
# allowing it to complement the transformer module in the ensemble method.
# providing robust performance across a wider range of data complexities by dynamically adjusting its learning focus based on the data's geometric properties.
# focused forward and backward allows the mlp.MLP to adaptively concentrate on abundant layers during training, enhancing its ability to learn from data with varying geometric complexity for flexible applications.
# and providing a complementary learning dynamic when combined with the transformer in the ensemble.
# source of geometric weight research: https://github.com/Micro-Novelty/Specialized-mlp.MLP-for-noise-robustness



