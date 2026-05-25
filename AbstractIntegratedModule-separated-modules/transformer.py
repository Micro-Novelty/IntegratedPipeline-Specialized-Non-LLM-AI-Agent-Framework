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
# transformer.py
# Single-layer Transformer with:
#   - GWS-initialised Q/K/V/O projection matrices
#   - Frozen↔learned projection blending via a geometry-derived alpha
#   - Manual forward + backward pass (no autograd framework)
#   - Scaled dot-product multi-head attention
# Depends on: geometry (GWS), nn (Loss), mlp (MLP for base GWS mixin methods)
# Note: MLP is imported lazily inside methods to avoid a circular import with mlp.py
# ---------------------------------------------------------------------------
from .geometry import GeometricWeightShaping
from .nn import Loss

class Transformer:
    def __init__(self, vocab_size, d_model=32, n_heads=4, num_classes=7):
        self.d_model = d_model  # Embedding dimension
        self.n_heads = n_heads

        self.token_embedding = np.random.randn(vocab_size, d_model) * 0.02
        
        # Positional embeddings (word order)
        self.pos_embedding = np.random.randn(100, d_model) * 0.02  
        
        # Multi-head attention parameters
        self.W_q = np.random.randn(n_heads, d_model, d_model // n_heads) * 0.02
        self.W_k = np.random.randn(n_heads, d_model, d_model // n_heads) * 0.02
        self.W_v = np.random.randn(n_heads, d_model, d_model // n_heads) * 0.02

        self.W_q_fixed = self.W_q.copy()
        self.W_k_fixed = self.W_k.copy()
        self.W_v_fixed = self.W_v.copy()

        self.W_o = np.random.randn(d_model, d_model) * 0.02
        self.encoded = False 

        # Feed-forward network
        self.ffn1 = np.random.randn(d_model, d_model * 4) * 0.02
        self.ffn2 = np.random.randn(d_model * 4, d_model) * 0.02
        
        # Layer norms
        self.ln1_scale = np.ones(d_model)
        self.ln1_shift = np.zeros(d_model)
        self.ln2_scale = np.ones(d_model)
        self.ln2_shift = np.zeros(d_model)
        
        # Output layer
        self.output = np.random.randn(d_model, num_classes) * 0.02
        self.output_bias = np.zeros(num_classes)
        
        self.cache = {}
        
    def layer_norm(self, x, scale, shift):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return scale * (x - mean) / np.sqrt(var + 1e-5) + shift
    
    def softmax(self, x):
        if x.ndim == 3:
            shifted = x - np.max(x, axis=-1, keepdims=True)
        else:
            shifted = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(shifted)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def attention(self, Q, K, V, mask=None):
        d_k = Q.shape[-1]
        scores = np.matmul(Q, K.transpose(0,1,3,2)) / np.sqrt(d_k)
        scores = np.clip(scores, -50, 50)
        
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)
        
        weights = self.softmax(scores)
        output = np.matmul(weights, V)
        return output, weights
    

    def multi_head_attention(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape
        try:
            alpha = self.alpha  # between 0 and 1
        except:
            AME = self.AME_Encoder(x)
            AMR = 1.0 / (1.0 + np.exp(-AME))
            alpha = AMR
            self.alpha = alpha

        W_q_mix = (1 - alpha) * self.W_q_fixed + alpha * self.W_q
        W_k_mix = (1 - alpha) * self.W_k_fixed + alpha * self.W_k
        W_v_mix = (1 - alpha) * self.W_v_fixed + alpha * self.W_v
        
        Q = np.einsum('bsd,hdm->bhsm', x, W_q_mix)
        K = np.einsum('bsd,hdm->bhsm', x, W_k_mix)
        V = np.einsum('bsd,hdm->bhsm', x, W_v_mix)         
        
        # Store for backward
        self.cache['Q'] = Q
        self.cache['K'] = K
        self.cache['V'] = V
        self.cache['x_attn_input'] = x
        
        # Attention
        attn_output, attn_weights = self.attention(Q, K, V, mask)
        self.cache['attn_weights'] = attn_weights
        self.cache['attn_output'] = attn_output
        
        # Concatenate heads
        attn_output = attn_output.transpose(0, 2,1, 3).reshape(batch_size, seq_len, -1)
        self.cache['attn_concat'] = attn_output
        
        # Final linear projection
        output = np.matmul(attn_output, self.W_o)
        self.cache['attn_out'] = output
        
        return output, attn_weights
    

    def forward(self, input_ids, embedded=False):
        if embedded:
            x = np.asarray(input_ids)
            if x.ndim == 2:
                x = x[np.newaxis, ...]
            batch_size, seq_len, _ = x.shape
            self.cache['embedded_input'] = x
            self.cache['input_ids'] = None
        else:
            if input_ids.ndim == 1:
                input_ids = input_ids.reshape(1, -1)
            x = self.token_embedding[input_ids]
            x = x + self.pos_embedding[:x.shape[1]]
            batch_size, seq_len = input_ids.shape
            self.cache['embedded_input'] = None
            self.cache['input_ids'] = input_ids

        self.cache['seq_len'] = seq_len
        self.cache['batch_size'] = batch_size
        self.cache['x_token'] = x
        self.cache['x_pos'] = x
        
        # Multi-head attention with residual
        attn_out, attn_weights = self.multi_head_attention(x)
        self.alpha = 0.95 * self.alpha + 0.05 * self.attention_quality_computing(attn_weights)

        self.cache['x_ln1_input'] = x + attn_out
        x = self.layer_norm(x + attn_out, self.ln1_scale, self.ln1_shift)
        self.cache['x_after_ln1'] = x
        
        # Feed-forward with residual
        self.cache['ffn_input'] = x
        ffn_pre = np.matmul(x, self.ffn1)
        self.cache['ffn_pre'] = ffn_pre
        
        ffn_act = np.maximum(0, ffn_pre)  # ReLU
        self.cache['ffn_act'] = ffn_act
        
        ffn_out = np.matmul(ffn_act, self.ffn2)
        self.cache['ffn_out'] = ffn_out
        
        self.cache['x_ln2_input'] = x + ffn_out
        x = self.layer_norm(x + ffn_out, self.ln2_scale, self.ln2_shift)
        self.cache['x_after_ln2'] = x
        
        x_pooled = np.mean(x, axis=1)  # (batch, d_model)
        self.cache['x_pooled'] = x_pooled
        
        # Output projection
        logits = np.matmul(x_pooled, self.output) + self.output_bias
        self.cache['logits'] = logits
        
        probs = self.softmax(logits)
        self.cache['probs'] = probs
        
        return probs, attn_weights
    

    def layer_norm_backward(self, d_out, x, scale, shift):
        eps = 1e-5
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        
        std = np.sqrt(var + eps)
        x_hat = (x - mean) / std
        
        N = x.shape[-1]
        dx_hat = d_out * scale
        dvar = np.sum(dx_hat * (x - mean) * -0.5 * std**-3, axis=-1, keepdims=True)
        dmean = (
        np.sum(dx_hat * -1/std, axis=-1, keepdims=True)
        + dvar * np.mean(-2*(x-mean), axis=-1, keepdims=True)
        )
        
        dx = (
        dx_hat / std
        + dvar * 2*(x-mean)/N
        + dmean / N
        )
        
        return dx
    
    # fixed attention backward allow the transformer to not update its Q, K, V projections, allowing much stable attention, while sacrificing flexibility.
    def fixed_attention_backward(self, d_logits, lr=0.001):

        # Gradient for output layer
        d_output = d_logits
        d_Wo = np.dot(self.cache['x_pooled'].T, d_output)
        d_bo = np.sum(d_output, axis=0, keepdims=True)
        
        # Gradient for pooled features
        d_pooled = np.dot(d_output, self.output.T)
        
        # Expand pooled gradient to all positions
        d_x = np.repeat(d_pooled[:, np.newaxis, :] / self.cache['seq_len'], self.cache['seq_len'], axis=1)
        
        # Layer norm 2 gradient
        d_x = self.layer_norm_backward(d_x, self.cache['x_ln2_input'], 
                                        self.ln2_scale, self.ln2_shift)
        
        # FFN gradients
        d_ffn = d_x
        
        # Gradient for FFN2
        d_ffn2 = np.sum(np.matmul(self.cache['ffn_act'].transpose(0, 2, 1), d_ffn), axis=0)
        
        # Gradient for FFN1 through ReLU
        d_ffn_act = np.matmul(d_ffn, self.ffn2.T)
        d_ffn_pre = d_ffn_act 
        d_ffn_pre[self.cache['ffn_pre'] <= 0] = 0
        d_prev = np.matmul(d_ffn_pre, self.ffn1.T)
        d_ffn1 = np.sum(np.matmul(self.cache['ffn_input'].transpose(0, 2, 1), d_ffn_pre), axis=0)
        
        # Layer norm 1 gradient
        d_x = self.layer_norm_backward(d_x - self.cache['attn_out'], 
                                        self.cache['x_ln1_input'],
                                        self.ln1_scale, self.ln1_shift)
        
        d_ffn = d_x
        d_residual_ffn = d_ffn
        dx = d_prev + d_residual_ffn
        d_attn = dx

        # Gradient for attention output projection
        d_Wo_attn = np.sum(np.matmul(self.cache['attn_concat'].transpose(0, 2, 1), d_attn), axis=0)
        
        # Update weights
        self.output -= lr * d_Wo
        self.output_bias -= lr * d_bo.squeeze()
        self.ffn2 -= lr * d_ffn2
        self.ffn1 -= lr * d_ffn1
        self.W_o -= lr * d_Wo_attn
        
     
        return d_x
    

    def dynamic_backward(self, d_logits, lr=0.001):

        # Gradient for output layer
        d_output = d_logits
        d_Wo = np.dot(self.cache['x_pooled'].T, d_output)
        d_bo = np.sum(d_output, axis=0)
        
        # Gradient for pooled features
        d_pooled = np.dot(d_output, self.output.T)
        
        # Expand pooled gradient to all positions
        d_x = np.repeat(d_pooled[:, np.newaxis, :] / self.cache['seq_len'], self.cache['seq_len'], axis=1)
        
        # Layer norm 2 gradient
        d_x = self.layer_norm_backward(d_x, self.cache['x_ln2_input'], 
                                        self.ln2_scale, self.ln2_shift)
        
        # FFN gradients
        d_ffn = d_x
        
        # Gradient for FFN2
        d_ffn2 = np.sum(np.matmul(self.cache['ffn_act'].transpose(0, 2, 1), d_ffn), axis=0)
        
        # Gradient for FFN1 through ReLU
        d_ffn_act = np.matmul(d_ffn, self.ffn2.T)

        d_ffn_pre = d_ffn_act * (self.cache['ffn_pre'] >= 0)
        d_prev = np.matmul(d_ffn_pre, self.ffn1.T)
        d_ffn1 = np.sum(np.matmul(self.cache['ffn_input'].transpose(0, 2, 1), d_ffn_pre), axis=0)
        
        # Layer norm 1 gradient
        d_ln1 = self.layer_norm_backward(d_x, self.cache['x_ln1_input'], self.ln1_scale, self.ln1_shift)
        
        d_residual = d_ln1
        d_attn = d_ln1
        dx = d_prev + d_residual

        # Gradient for attention output projection
        d_Wo_attn = np.sum(np.matmul(self.cache['attn_concat'].transpose(0, 2, 1), d_attn), axis=0)

        d_attn_concat = np.matmul(d_attn, self.W_o.T)
        batch, seq_len, _ = d_attn_concat.shape
        d_head = self.n_heads
        d_dim = self.d_model // self.n_heads

        d_attn_heads = d_attn_concat.reshape(batch, seq_len, d_head, d_dim) .transpose(0, 2, 1, 3)      

        V = self.cache['V']
        K = self.cache['K']
        Q = self.cache['Q']
        weight = self.cache['attn_weights']
        
        d_V = np.matmul(weight.transpose(0, 1, 3, 2), d_attn_heads)
        d_weights = np.matmul(d_attn_heads, V.transpose(0, 1, 3, 2))

        d_scores = weight * (d_weights - np.sum(d_weights * weight, axis=-1, keepdims=True))
        d_scores /= np.sqrt(Q.shape[-1])

        d_Q = np.matmul(d_scores, K)
        d_K = np.matmul(d_scores.transpose(0, 1, 3, 2), Q)

        x = self.cache['x_attn_input']

        d_W_q = np.einsum('bsd, bhsm->hdm', x, d_Q)
        d_W_k = np.einsum('bsd, bhsm->hdm', x, d_K)
        d_W_v = np.einsum('bsd, bhsm->hdm', x, d_V)

        d_x_q = np.einsum('bhsm, hdm->bsd', d_Q, self.W_q)
        d_x_k = np.einsum('bhsm, hdm->bsd', d_K, self.W_k)
        d_x_v = np.einsum('bhsm, hdm->bsd', d_V, self.W_v)

        d_x_attn_input = d_x_q + d_x_k + d_x_v
        d_x_total = d_x_attn_input + d_residual

        input_ids = self.cache.get('input_ids')

        if input_ids is not None:
            for b in range(input_ids.shape[0]):
                for t in range(input_ids.shape[1]):
                    idx = int(input_ids[b, t])
                    self.token_embedding[idx] -= lr * d_x_total[b, t] / self.cache['seq_len']

        # Update weights
        self.output -= lr * d_Wo
        self.output_bias -= lr * d_bo.squeeze()
        self.ffn2 -= lr * d_ffn2
        self.ffn1 -= lr * d_ffn1
        self.W_o -= lr * d_Wo_attn

        alpha = self.alpha

        self.W_q -= lr * alpha * d_W_q
        self.W_k -= lr * alpha * d_W_k
        self.W_v -= lr * alpha * d_W_v

        self.pos_embedding[:seq_len] -= lr * d_x_total.mean(axis=0)

     
        return d_x_total
    

    def train_step(self, input_ids, epoch, y_true, lr=0.001, mode=None, embedded=False):
        probs, attn_weights = self.forward(input_ids, embedded=embedded)
        
        # Loss (cross-entropy)
        if y_true.shape[0] and y_true.shape[1] != probs.shape[0] and probs.shape[1]:
            if y_true.shape[1] > probs.shape[1]:
                y_true = y_true[:, :probs.shape[1]]
            else:
                y_true = np.pad(y_true, ((0, 0), (0, probs.shape[1] - y_true.shape[1])), mode='constant')

        loss = -np.mean(np.sum(y_true * np.log(probs + 1e-8), axis=1))
        
        # Gradient of loss w.r.t. logits
        d_logits = (probs - y_true) / y_true.shape[0]
        
        # Backward pass
        if mode == 'fixed_backward':
            self.fixed_attention_backward(d_logits, lr)
        else:
            self.dynamic_backward(d_logits, lr)
        
        # Accuracy
        preds = np.argmax(probs, axis=1)
        true = np.argmax(y_true, axis=1)
        acc = np.mean(preds == true)
        
        return loss, acc


    def train(self, input_ids_list, y_true_list, epochs=100, mode=None, lr=0.001, embedded=False):
        losses = []
        accs = []
        d_model = self.d_model

        # Geometric initialized W_o provides the transformer better geometric alignment with the data geometric complexity
        # only works with W_o, else caused fragility and overfitting. shows that geometric alignment is only supportive in fixed projection embedding inside transformer like W_o.
        if not self.encoded:         
            self.shaping = GeometricWeightShaping(d_model, d_model)
            shaping_input = input_ids_list
            if embedded:
                shaping_input = np.vstack([x.reshape(1, -1) if x.ndim > 2 else x for x in input_ids_list])
            self.W_o = self.shaping.weight_shaping(shaping_input)
            self.encoded = True

        for epoch in range(epochs):
            epoch_losses = []
            epoch_accs = []
            self.alpha = min(1.0, epoch / 100) 

            for input_ids, y_true in zip(input_ids_list, y_true_list):
                  
                if input_ids.ndim == 1:
                    input_ids = input_ids.reshape(1, -1)
                if y_true.ndim == 1:
                    y_true = y_true.reshape(1, -1)
                    
                loss, acc = self.train_step(input_ids, epoch, y_true, lr, mode, embedded=embedded)
                epoch_losses.append(loss)
                epoch_accs.append(acc)
            
            avg_loss = np.mean(epoch_losses)
            avg_acc = np.mean(epoch_accs)
            losses.append(avg_loss)
            accs.append(avg_acc)
            
            if epoch % 10 == 0:
                print(f"[=] Epoch {epoch} | loss: {avg_loss:.4f} | Acc: {avg_acc:.2%}")
        
        return losses, accs
    

    def predict(self, input_ids, embedded=False):
        if not embedded and input_ids.ndim == 1:
            input_ids = input_ids.reshape(1, -1)
        
        probs, attn_weights = self.forward(input_ids, embedded=embedded)
        preds = np.argmax(probs, axis=1)
        
        return preds, probs, attn_weights


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

    # attention quality computing provides the transformer a robust geometric complexity alignment scalar,
    #  this scalar can be used to compute alpha for a much stable forward pass in scarce data environment, allowing it to complement with AWE MLP below.
    def attention_quality_computing(self, attn_weights):
        eps = 1e-5
        eps = 1e-5
        batch, heads, seq_len, _ = attn_weights.shape
        AME = self.AME_Encoder(attn_weights)
        anisotropy = self.anisotropy_measurement(attn_weights)

        entropy = -np.sum(attn_weights * np.log(attn_weights + eps), axis=-1)
        max_entropy = np.log(seq_len)
        norm_entropy = 1.0 - (np.mean(entropy) / max_entropy)

        max_attn = np.max(attn_weights, axis=-1)
        avg_max = np.mean(max_attn)

        var_attn = np.var(attn_weights)
        norm_var = np.clip(var_attn * seq_len, 0, 1)

        AMR = 1.0 / (1.0 + np.exp(-AME))  # abstract modelling rate
        qualified = (1.0 - AMR) + eps * anisotropy

        quality_score = qualified * norm_entropy + qualified * avg_max + anisotropy * norm_var
        dynamic_alpha = np.clip(quality_score, 0, 1.0)

        return dynamic_alpha