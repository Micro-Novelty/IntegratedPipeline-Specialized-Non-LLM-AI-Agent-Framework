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
# mlp.py
# Multi-Layer Perceptron built from Dense + SoftmaxOutput layers.
# Provides standard forward/backward, a focused sub-network path
# (feed_layers), and GWS-derived geometry helpers inherited from the base class.
# Depends on: geometry (GWS mixin), nn (Loss, SoftmaxOutput, Dense)
# ---------------------------------------------------------------------------
from .geometry import GeometricWeightShaping
from .nn import Loss, Dense, SoftmaxOutput, Activation

class MLP:
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