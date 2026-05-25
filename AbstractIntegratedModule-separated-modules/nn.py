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
# nn.py
# Core neural network building blocks:
#   Activation  — static activation functions (relu, sigmoid, tanh, softmax, leaky_relu)
#   Loss        — static loss functions and their gradients
#   Dense       — a single fully-connected layer backed by GWS initialisation
#   SoftmaxOutput — thin output layer wrapper
# Depends on: geometry (GeometricWeightShaping)
# ---------------------------------------------------------------------------
from .geometry import GeometricWeightShaping

class Activation:
    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)

    @staticmethod
    def sigmoid(x):
        eps = 1e-5
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        eps = 1e-5
        s = Activation.sigmoid(x)
        return s * (1.0 - s)

    @staticmethod
    def softmax(x):
        # numerical stability
        if x.ndim > 1:
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        else:
            exp_x = np.exp(x - np.max(x, keepdims=True))
            return exp_x / np.sum(exp_x, keepdims=True)


class Loss:
    @staticmethod
    def categorical_crossentropy(y_true, y_pred):
        eps = 1e-9
        y_pred = np.clip(y_pred, eps, 1 - eps)
        try:
            loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        except:
            subnet_y_pred = y_pred[:, :y_true.shape[1]]
            subnet_y_true = y_true[:, :subnet_y_pred.shape[1]]

            loss = -np.mean(np.sum(subnet_y_true * np.log(subnet_y_pred + eps), axis=1))
        
        return loss

    @staticmethod
    def softmax_crossentropy_derivative(y_true, y_pred):
        try:
            cross_ent = (y_pred - y_true) / y_true.shape[0]
        except:
            subnet_y_pred = y_pred[:, :y_true.shape[1]]
            subnet_y_true = y_true[:, :subnet_y_pred.shape[1]]

            cross_ent = (subnet_y_pred - subnet_y_true) / subnet_y_true.shape[0]
        return cross_ent


class Dense:
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
        if len(x.shape) > 1 and x.shape[1] != self.W.shape[0]:
            V1, V2 = x.shape[0], x.shape[1]            
            try:
                self.W = self.W[:V2, :]
            except:
                self.special_weight = GeometricWeightShaping(V2, V1)
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


# enhanced MLP with focused forward and backward for better handling of data with varying geometric complexity,
# allowing it to complement the transformer module in the ensemble method.
# providing robust performance across a wider range of data complexities by dynamically adjusting its learning focus based on the data's geometric properties.
# focused forward and backward allows the MLP to adaptively concentrate on abundant layers during training, enhancing its ability to learn from data with varying geometric complexity for flexible applications.
# and providing a complementary learning dynamic when combined with the transformer in the ensemble.
# source of geometric weight research: https://github.com/Micro-Novelty/Specialized-MLP-for-noise-robustness