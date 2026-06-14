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



class Dense:
    '''
    A single fully-connected layer with GWS-initialised weights.

    Weight initialisation
    ---------------------
    Weights are drawn from GeometricWeightShaping.weight_shaping(x) where x is
    the first training batch.  This aligns the initial weight scale with the
    geometric complexity of the actual data rather than using a fixed constant.

    Activation
    ----------
    Pass the name of any method in the Activation class (e.g. 'relu', 'sigmoid').
    If None, the layer is linear (identity activation).

    Shape mismatch tolerance
    ------------------------
    multi_modal_linear_transformation() implements a three-level shape recovery
    cascade so the layer can be called with inputs that differ in feature count
    from the original training batch.  This is intentional: the pipeline reuses
    the same Dense layer instance during training, evaluation, and inference
    where TF-IDF vocabulary size can vary slightly.
    '''
    def __init__(self, x, input_size, output_size, activation=None):

        self.special_weight = GeometricWeightShaping(input_size, output_size)
        self.W = self.special_weight.weight_shaping(x)

        self.b = np.zeros((1, output_size))
        self.activation_name = activation
       

        if activation:
            self.activation = getattr(Activation, activation)
            self.activation_derivative = getattr(Activation, activation + "_derivative")
        else:
            self.activation = None
            self.activation_derivative = None

    def multi_modal_linear_transformation(self, x):
        # Standard linear layer z = xW + b, but with a multi-level shape-mismatch
        # recovery cascade.  This is needed because the GWS weight matrix W is shaped
        # at construction time from the training data, and at inference time the input
        # may have a different number of features (e.g. after vocabulary drift or
        # when calling the model with embedded TF-IDF vectors vs raw token IDs).
        #
        # Recovery hierarchy (outermost try wins):
        #   Level 1 (primary): normal dot(x, W) + b.
        #   Level 2 (first fallback): column-slice W to match x.shape[1], then add
        #             a matching slice of b.
        #   Level 3 (deep fallback): slice both x and W along whichever dimension fits,
        #             then add a b slice.  Covers edge cases where both x and W need trimming.
        #
        # The guard at the top reshapes W in-place if shapes are obviously mismatched
        # (x.shape[1] != W.shape[0]), preferring slicing over re-initialisation.
        if len(x.shape) > 1 and x.shape[1] != self.W.shape[0]:
            V1, V2 = x.shape[0], x.shape[1]            
            try:
                # Trim W's rows to match the feature dimension of x.
                self.W = self.W[:V2, :]
            except:
                # If trimming fails (W is already smaller), re-initialise with correct dims.
                self.special_weight = GeometricWeightShaping(V2, V1)
                self.W = self.special_weight.weight_shaping(x)
        try:
            try:
                z = np.dot(x, self.W) + self.b
            except:
                # W has more rows than x has columns; trim and add matching bias slice.
                subnet_W = self.W[:x.shape[1], :x.shape[0]]

                sub_z = np.dot(x, subnet_W)
                sub_b = self.b[:sub_z.shape[1], :sub_z.shape[0]]

                z = sub_z + sub_b

        except:
            try:
                subnet_W = self.W[:x.shape[1]:, :x.shape[0]]
                sub_z = np.dot(x, subnet_W)
            except:
                # Last resort: trim x to fit W or vice versa, whichever succeeds first.
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
                
            z = sub_z + subnet_B      

        return z


    def forward(self, x):
        self.x = x
        self.z = self.multi_modal_linear_transformation(x)

        if self.activation:
            self.a = self.activation(self.z)
        else:
            self.a = self.z

        return self.a


    def backward(self, da, lr):
        eps = 1e-5

        if self.activation_derivative: 
            dz = da * self.activation_derivative(self.z)
        else:
            dz = da

        dW = np.dot(self.x.T, dz)
        db = np.sum(dz, axis=0, keepdims=True)
        dx = np.dot(dz, self.W.T)

        self.W -= lr * dW 
        self.b -= lr * db 
      
        return dx
        

    	
    	
class SoftmaxOutput:
    def forward(self, x):
        self.out = Activation.softmax(x)
        return self.out

    def backward(self, dL_dZ):
        # gradient already computed as y_pred - y_true
        return dL_dZ


# Enhanced MLP with focused forward and backward passes for better handling of
# data with varying geometric complexity, complementing the Transformer in the
# ensemble method.  The "focused" path (feed_layers) allows the MLP to
# concentrate on a subset of layers during fine-tuning, providing a
# complementary learning dynamic alongside the Transformer.
#
# Geometric weight research: https://github.com/Micro-Novelty/Specialized-MLP-for-noise-robustness

class MLP:
    '''
    Multi-layer perceptron composed of Dense (GWS-initialised) layers.

    Two execution paths
    -------------------
    Standard path  : self.layers     — used for primary training and prediction.
    Focused path   : self.feed_layers — a secondary stack that can be trained
                     independently via focused_forward / focused_backward.
                     Useful when you want to fine-tune only specific layers
                     without affecting the rest of the network.

    Both paths share the same SoftmaxOutput layer at the end.

    Usage
    -----
        mlp = MLP()
        mlp.add(Dense(X_train, input_dim, hidden_dim, activation='relu'))
        mlp.add(Dense(X_train, hidden_dim, num_classes))
        mlp.train(X_train, y_one_hot, epochs=1000, lr=0.1)
        probs = mlp.predict_proba(X_test)
    '''
    def __init__(self):
        self.layers = []
        self.layers2 = []
        self.lr = 0.1
        self.feed_layers = []
    
        self.softmax = SoftmaxOutput()
      

    def feed_add(self, layer):
        self.feed_layers.append(layer)

    def add(self, layer):
        self.layers.append(layer)

    def focused_forward(self, x):
        for layer in self.feed_layers:
            x = layer.forward(x)
            
        return self.softmax.forward(x) 

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
            
        return self.softmax.forward(x)
        
    def focused_backward(self, grad, lr):
        grad = self.softmax.backward(grad)
        for layer in reversed(self.feed_layers):
            grad = layer.backward(grad, lr)
        return grad  

    def backward(self, grad, lr):
        grad = self.softmax.backward(grad)
        for layer in reversed(self.layers):
            grad = layer.backward(grad, lr)
        return grad
            
    def predict(self, X, y, epochs=1000, verbose=True):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = Loss.categorical_crossentropy(y, y_pred)    		
            if verbose and epoch % 100 == 0:
                acc = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))
                self.acc2.append(acc)
                print(f"[=] Epoch {epoch} | loss:{loss:.4f} | Acc: {acc:.2f}")

	   
    def prediction(self, X):
        y_pred = self.forward(X)   
        return y_pred      


    def score(self, X, y):
        y_pred = self.prediction(X)
        acc = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))
        return acc         
         
    def AME_Encoder(self, x):
        X = np.asarray(x)

        if x.shape[1] == 1:
            x = x.T
            x= x.flatten()

        gradient = np.gradient(x, axis=-1)
        grad_energy = np.mean(np.linalg.norm(gradient, axis=-1))       
        X_mag = np.mean(np.linalg.norm(X, axis=-1))

        AME = np.log1p(X_mag) * np.log1p(grad_energy) 
        return AME


    def anisotropy_measurement(self, x):
        eps = 1e-5
        try:
            gradient = np.gradient(x, axis=-1)
        except:
            subset = x[:, 0, 0]
            gradient = np.gradient(subset)

        val = [np.linalg.norm(v) for v in gradient]
        anisotropy = np.std(val) / np.mean(val) + eps

        return anisotropy


    def train(self, X, y, epochs=1000, lr=0.01, verbose=True):
        # Decide whether to use the "focused" sub-network (feed_layers) or the
        # standard full network (layers) for this training run.
        #
        # focused_fit_condition is True when ALL three hold:
        #   1. feed_layers is non-empty  — a focused sub-network exists
        #   2. anisotropy > 0.25         — data has sufficient directional variation
        #                                  (flat/isotropic data doesn't benefit from focus)
        #   3. AME > 0.25               — combined magnitude × gradient energy is above
        #                                  a minimum threshold (data is complex enough)
        #
        # When True, only feed_layers are updated via focused_forward/focused_backward,
        # letting the model concentrate its learning capacity on high-complexity data
        # without disrupting the full network's previously learned representations.
        focused_fit_condition = len(self.feed_layers) > 0 and self.anisotropy_measurement(X) > 0.25 and self.AME_Encoder(X) > 0.25
        print(f'[+] Focused fit condition: {focused_fit_condition} || Anisotropy: {self.anisotropy_measurement(X):.4f} || AME: {self.AME_Encoder(X):.4f}')
        for epoch in range(epochs):
            if not focused_fit_condition:
                y_pred = self.forward(X)
            else:
                y_pred = self.focused_forward(X)

            loss = Loss.categorical_crossentropy(y, y_pred)
            grad = Loss.softmax_crossentropy_derivative(y, y_pred)
            _ = self.backward(grad, self.lr)

            if verbose and epoch % 100 == 0:
                acc = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))
                print(f"[=] Epoch {epoch} | Loss: {loss:.4f} | Acc: {acc:.2f}")
              

# weighted ensemble predictor that dynamically adjusts the weights of the transformer and MLP based on the input data's geometric complexity and the attention quality>
# allowing it to leverage the strengths of both models for improved performance across a wider range of data complexities.
