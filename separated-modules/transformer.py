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

# More Newer Transformer version
# This class is used for direct comparison only, not used by Any Class.
class OptimizedTransformer:
    def __init__(self, vocab_size, d_model=8, n_heads=2, num_classes=7, learning_rate=0.01, attn_dropout=0.0, ffn_dropout=0.0, weight_decay=1e-4):
        self.d_model = d_model  # Embedding dimension
        self.n_heads = n_heads
        self.attn_dropout_rate = attn_dropout
        self.ffn_dropout_rate  = ffn_dropout
        self.transformer_lr = learning_rate
        self.weight_decay = weight_decay

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
        self._attn_quality_step = 0
        self._forward_count = 0
        self._attn_scale = None

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

    def _clear_forward_cache(self, keep_essential=True, max_age_forward_passes=5):
        """Clear cache entries older than max_age_forward_passes."""
        
        essential_keys = {'input_ids', 'mask', 'alpha', 'attn_weights', 'probs', 'logits'}
        oldest_to_keep = self._forward_count - max_age_forward_passes
        
        # Group cache keys by their forward pass number
        forward_keys = {}
        latest_keys = {}
        
        for key in list(self.cache.keys()):
            if '_f' in key:
                parts = key.split('_f')
                if len(parts) == 2:
                    base_key, fwd_num_str = parts
                    try:
                        fwd_num = int(fwd_num_str)
                        if fwd_num not in forward_keys:
                            forward_keys[fwd_num] = []
                        forward_keys[fwd_num].append(key)
                    except ValueError:
                        pass
            elif key.startswith('latest_'):
                latest_keys[key] = True
        
        # Delete old forward passes
        for fwd_num in list(forward_keys.keys()):
            if fwd_num < oldest_to_keep:
                for key in forward_keys[fwd_num]:
                    if key in self.cache:
                        del self.cache[key]
                del forward_keys[fwd_num]
        
        # Also delete latest pointers for essential keys that are too old
        if keep_essential:
            for latest_key in list(latest_keys.keys()):
                actual_key = latest_key.replace('latest_', '')
                # Check if we still have any version of this key from recent passes
                has_recent = False
                for fwd_num in forward_keys:
                    if fwd_num >= oldest_to_keep:
                        for key in forward_keys[fwd_num]:
                            if key.startswith(actual_key + '_f'):
                                has_recent = True
                                break
                    if has_recent:
                        break
                
                if not has_recent and actual_key in essential_keys:
                    # No recent versions of this essential key, delete the latest pointer
                    del self.cache[latest_key]

    def layer_norm(self, x, scale, shift):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return scale * (x - mean) / np.sqrt(var + 1e-5) + shift

    def apply_update(self, param, grad, lr):
        # L2 weight decay applied directly at update time
        # equivalent to: grad += weight_decay * param
        return param - lr * (grad + self.weight_decay * param)

    def dropout(self, x, rate=0.1, training=True, alpha=None):
        if not training or rate == 0.0:
            return x, None
        
        # If alpha provided, scale the effective drop rate by it
        # low alpha (early training, fixed attention) → very light dropout
        # high alpha (dynamic attention active)       → full dropout rate
        effective_rate = rate * alpha if alpha is not None else rate
        
        if effective_rate == 0.0:
            return x, None

        mask = (np.random.rand(*x.shape) > effective_rate).astype(np.float32)
        return x * mask / (1.0 - effective_rate), mask

    def _get_attn_scale(self, d_k):
        """Cache scale factor — d_k never changes at runtime."""
        if self._attn_scale is None:
            self._attn_scale = 1.0 / np.sqrt(d_k)
        return self._attn_scale



    def attention(self, Q, K, V, mask=None):
        d_k = Q.shape[-1]

        # cached scale, no sqrt per call
        scale = self._get_attn_scale(d_k)

        # np.einsum avoids explicit transpose + matmul allocation
        # 'bhqd,bhkd->bhqk' computes Q @ K.T without creating K.T
        scores = np.einsum('bhqd,bhkd->bhqk', Q, K) * scale

        # inplace clip
        np.clip(scores, -50, 50, out=scores)

        # inplace mask application
        if mask is not None:
            # mask: (B,1,1,T) broadcasts to (B,H,T,T)
            scores += (1.0 - mask) * -1e9   # inplace add instead of np.where

        # fused softmax with numerical stability inplace
        scores -= scores.max(axis=-1, keepdims=True)  # stability shift inplace
        np.exp(scores, out=scores)                     # inplace exp
        scores /= scores.sum(axis=-1, keepdims=True) + 1e-8  # inplace normalize
        weights = scores   

        output = np.matmul(weights, V)
        return output, weights


    def softmax(self, x):
        if x.ndim == 3:
            shifted = x - np.max(x, axis=-1, keepdims=True)
        else:
            shifted = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(shifted)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    

    def multi_head_attention(self, x, mask=None, alpha=None,
                  W_q_mix=None, W_k_mix=None, W_v_mix=None):
        batch_size, seq_len, d_model = x.shape
        if W_q_mix is None:
            one_minus = 1.0 - alpha
            W_q_mix = one_minus * self.W_q_fixed + alpha * self.W_q
            W_k_mix = one_minus * self.W_k_fixed + alpha * self.W_k
            W_v_mix = one_minus * self.W_v_fixed + alpha * self.W_v        

        # Optimized Project heads with Cython implementation.
        if _OPT_AVAILABLE:
            B, S, D = batch_size, seq_len, d_model
            M = D // self.n_heads 

            Q = optimized_project_heads(x, W_q_mix, B, S, self.n_heads, D, M)
            K = optimized_project_heads(x, W_k_mix, B, S, self.n_heads, D, M)
            V = optimized_project_heads(x, W_v_mix, B, S, self.n_heads, D, M)
        else:
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
    

    def forward(self, input_ids, AME=None, _update_quality_matrix=None, embedded=False, pad_token_id=0, training=True, attn_dropout=0.1, ffn_dropout=0.1):
        self._forward_count += 1
            
        # Clean up old cache entries before storing new ones
        self._clear_forward_cache(keep_essential=True, max_age_forward_passes=5) 

        if embedded:
            x = np.asarray(input_ids)
            if x.ndim == 2:
                x = x[np.newaxis, ...]
            batch_size, seq_len, _ = x.shape
            self.cache['embedded_input'] = x
            self.cache['input_ids'] = None
            mask = None
        else:
            input_ids = np.asarray(input_ids, dtype=np.int32)
            if input_ids.ndim == 1:
                input_ids = input_ids[np.newaxis, :]

            x = self.token_embedding[input_ids]
            x = x + self.pos_embedding[:x.shape[1]]
            batch_size, seq_len = input_ids.shape
            self.cache['embedded_input'] = None
            self.cache['input_ids'] = input_ids
            mask = self.padding_mask_utility(input_ids, pad_token_id)  # (B,1,1,T)            

        self.cache['mask'] = mask if not embedded else None
        self.cache['seq_len'] = seq_len
        self.cache['batch_size'] = batch_size
        self.cache['x_token'] = x
        self.cache['x_pos'] = x
        
        # Multi-head attention with residual     
        if AME is None:
            AME = self.AME_Encoder(x)

        alpha = 1.0 / (1.0 + np.exp(-float(AME)))
        if _update_quality_matrix is None:
            # input ids ranged 0 to 1.
            consistency = 1.0 / (1.0 + np.std(input_ids))
            alpha_rate = 1.0 / (1.0 + alpha)
            _update_quality_matrix = alpha_rate * (1.0 - ffn_dropout) * consistency

        _should_update = (
            not training or
            not hasattr(self, '_attn_quality_step') or
            self._attn_quality_step % max(1, int(1.0 / (_update_quality_matrix + 1e-6))) == 0
        )
        
        one_minus_alpha = 1.0 - alpha
        W_q_mix = one_minus_alpha * self.W_q_fixed + alpha * self.W_q
        W_k_mix = one_minus_alpha * self.W_k_fixed + alpha * self.W_k
        W_v_mix = one_minus_alpha * self.W_v_fixed + alpha * self.W_v

        attn_out, attn_weights = self.multi_head_attention(x, mask=mask, alpha=alpha,
                      W_q_mix=W_q_mix, W_k_mix=W_k_mix, W_v_mix=W_v_mix )
  
        current_alpha = self.cache.get('alpha', 0.0) 

        attn_out, attn_drop_mask = self.dropout(attn_out, rate=self.attn_dropout_rate, training=training, alpha=current_alpha)   
        self.cache['attn_drop_mask'] = attn_drop_mask  

        if training and hasattr(self, '_attn_quality_step'):
            self._attn_quality_step += 1
            if _should_update and self._attn_quality_step % 10 == 0:
                attn_quality = self.attention_quality_computing(attn_weights, AME=AME, mask=mask)
                self._cached_attn_quality = attn_quality
            else:
                attn_quality = getattr(self, '_cached_attn_quality', 0.5)
        else:
            self._attn_quality_step  = 0
            self._cached_attn_quality = alpha * (1.0 - current_alpha)
            attn_quality = self._cached_attn_quality 

        alpha = 0.95 * alpha + 0.05 * attn_quality 

        self.alpha = alpha
        self.cache['alpha'] = alpha  # store in cache     

        self.cache['x_ln1_input'] = x + attn_out
        x, ln1_mean, ln1_var = self.layer_norm_with_cache(
                self.cache['x_ln1_input'], self.ln1_scale, self.ln1_shift
            )   

        self.cache['x_after_ln1'] = x
        self.cache['ln1_mean']    = ln1_mean   # reused in backward
        self.cache['ln1_var']     = ln1_var    

        # Feed-forward with residual
        self.cache['ffn_input'] = x
        ffn_pre = np.matmul(x, self.ffn1)
        self.cache['ffn_pre'] = ffn_pre
        
        ffn_act = np.maximum(0, ffn_pre)  # ReLU
           
        ffn_act, ffn_drop_mask = self.dropout(ffn_act, rate=self.ffn_dropout_rate, training=training, alpha=current_alpha)
     
        self.cache['ffn_act'] = ffn_act
        self.cache['ffn_drop_mask'] = ffn_drop_mask   

        ffn_out = np.matmul(ffn_act, self.ffn2)
        self.cache['ffn_out'] = ffn_out
        
        self.cache['x_ln2_input'] = x + ffn_out
        x, ln2_mean, ln2_var = self.layer_norm_with_cache(
                self.cache['x_ln2_input'], self.ln2_scale, self.ln2_shift
            )        
        self.cache['x_after_ln2'] = x
        self.cache['ln2_mean']    = ln2_mean
        self.cache['ln2_var']     = ln2_var        
        
        if mask is not None:
            # Reshape mask to (B, T, 1) for broadcasting against (B, T, D)
            token_mask = mask[:, 0, 0, :, np.newaxis]        # (B, T, 1)
            x_masked   = x * token_mask                       # zero out padding
            lengths    = token_mask.sum(axis=1)               # (B, 1) valid token counts
            x_pooled   = x_masked.sum(axis=1) / (lengths + 1e-6)  # (B, D)
        else:
            x_pooled = np.mean(x, axis=1)

        self.cache['x_pooled'] = x_pooled
        
        # Output projection
        logits = np.matmul(x_pooled, self.output) + self.output_bias
        self.cache['logits'] = logits
        
        probs = self.softmax(logits)
        self.cache['probs'] = probs
        
        return probs, attn_weights

    def layer_norm_with_cache(self, x, scale, shift, eps=1e-5):
        """Layer norm that returns mean and var for backward reuse."""
        mean  = x.mean(axis=-1, keepdims=True)
        var   = x.var(axis=-1,  keepdims=True)
        x_hat = (x - mean) / np.sqrt(var + eps)
        return x_hat * scale + shift, mean, var 


    def layer_norm_backward(self, d_out, x, scale, shift,
               mean=None, var=None):
        eps = 1e-5
        if mean is None:
            mean = x.mean(axis=-1, keepdims=True)
        if var is None:
            var  = x.var(axis=-1,  keepdims=True)

        std = np.sqrt(var + eps)
        x_hat = (x - mean) / std
        
        N = x.shape[-1]
        dx_hat = d_out * scale
        dvar = np.sum(dx_hat * (x - mean) * -0.5 * std**-3, axis=-1, keepdims=True)
        dmean = np.sum(dx_hat * (-1.0 / std), axis=-1, keepdims=True)
        
        dx = (
        dx_hat / std
        + dvar * 2*(x-mean)/N
        + dmean / N
        )
        
        return dx
    
    # fixed attention backward allow the transformer to not update its Q, K, V projections, allowing much stable attention, while sacrificing flexibility.
    def fixed_attention_backward(self, d_logits, lr=0.01, max_norm=1.0):

        # Gradient for output layer
        d_output = d_logits
        alpha = self.cache.get('alpha', 1.0)

        d_Wo = np.dot(self.cache['x_pooled'].T, d_output)
        d_bo = np.sum(d_output, axis=0, keepdims=True)
        
        # Gradient for pooled features
        d_pooled = np.dot(d_output, self.output.T)
        
        # Expand pooled gradient to all positions
        d_x = np.repeat(d_pooled[:, np.newaxis, :] / self.cache['seq_len'], self.cache['seq_len'], axis=1)
        
        # Layer norm 2 gradient
        d_x = self.layer_norm_backward(d_x, self.cache['x_ln1_input'],
                                        self.ln1_scale, self.ln1_shift,
                                        mean=self.cache.get('ln1_mean'),
                                        var=self.cache.get('ln1_var'))
                                        
        # FFN gradients
        d_ffn = d_x
        
        # Gradient for FFN2
        d_ffn2 = np.sum(np.matmul(self.cache['ffn_act'].transpose(0, 2, 1), d_ffn), axis=0)
        
        # Gradient for FFN1 through ReLU
        d_ffn_act = np.matmul(d_ffn, self.ffn2.T)
        ffn_drop_mask = self.cache.get('ffn_drop_mask')
        if ffn_drop_mask is not None:
            d_ffn_act = d_ffn_act * ffn_drop_mask / (1.0 - self.ffn_dropout_rate)

        d_ffn_pre = d_ffn_act * (self.cache['ffn_pre'] >= 0)   # ReLU backward unchanged

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
        attn_drop_mask = self.cache.get('attn_drop_mask')
        if attn_drop_mask is not None:
            d_attn = d_attn * attn_drop_mask / (1.0 - self.attn_dropout_rate)

        d_Wo_attn = np.sum(np.matmul(self.cache['attn_concat'].transpose(0,2,1), d_attn), axis=0)        

        grads = {
                'output':  d_Wo,
                'ffn2':    d_ffn2,
                'ffn1':    d_ffn1,
                'W_o':     d_Wo_attn,
            }

        grads, norm = self.clip_gradients(grads, max_norm)        

        # Update weights
        self.output = self.apply_update(self.output, grads['output'], lr)
        self.ffn2   = self.apply_update(self.ffn2,   grads['ffn2'],   lr)
        self.ffn1   = self.apply_update(self.ffn1,   grads['ffn1'],   lr)
        self.W_o    = self.apply_update(self.W_o,    grads['W_o'],    lr)
        # output_bias intentionally excluded — biases don't get weight decay
            
        return d_x
    

    def dynamic_backward(self, d_logits, lr=0.01, max_norm=1.0):
        # Gradient for output layer
        d_output = d_logits
        alpha = self.cache.get('alpha', 1.0)

        d_Wo = np.dot(self.cache['x_pooled'].T, d_output)
        d_bo = np.sum(d_output, axis=0)
        
        # Gradient for pooled features
        d_pooled = np.dot(d_output, self.output.T)
        
        # Expand pooled gradient to all positions
        mask = self.cache['mask']  # (B, 1, 1, T)
        if mask is not None:
            token_mask = mask[:, 0, 0, :, np.newaxis]             # (B, T, 1)
            lengths    = token_mask.sum(axis=1, keepdims=True)    # (B, 1, 1)
            d_x        = (d_pooled[:, np.newaxis, :] / (lengths + 1e-6)) * token_mask
        else:
            d_x = np.repeat(d_pooled[:, np.newaxis, :] / self.cache['seq_len'], self.cache['seq_len'], axis=1)    

        # Layer norm 2 gradient
        d_x = self.layer_norm_backward(d_x, self.cache['x_ln2_input'],
                                self.ln2_scale, self.ln2_shift,
                                mean=self.cache.get('ln2_mean'),
                                var=self.cache.get('ln2_var'))

        # FFN gradients
        d_ffn = d_x
        
        # Gradient for FFN2
        d_ffn2 = np.sum(np.matmul(self.cache['ffn_act'].transpose(0, 2, 1), d_ffn), axis=0)
        
        # Gradient for FFN1 through ReLU
        d_ffn_act = np.matmul(d_ffn, self.ffn2.T)
        ffn_drop_mask = self.cache.get('ffn_drop_mask')
        if ffn_drop_mask is not None:
            d_ffn_act = d_ffn_act * ffn_drop_mask / (1.0 - self.ffn_dropout_rate)

        d_ffn_pre = d_ffn_act * (self.cache['ffn_pre'] >= 0)   # ReLU backward unchanged

        d_prev = np.matmul(d_ffn_pre, self.ffn1.T)
        d_ffn1 = np.sum(np.matmul(self.cache['ffn_input'].transpose(0, 2, 1), d_ffn_pre), axis=0)
        
        # Layer norm 1 gradient
        d_ln1 = self.layer_norm_backward(
            d_prev, self.cache['x_ln1_input'], self.ln1_scale, self.ln1_shift
        ) 

        d_residual = d_ln1
        d_attn = d_ln1
        dx = d_prev + d_residual

        # Gradient for attention output projection
        attn_drop_mask = self.cache.get('attn_drop_mask')
        if attn_drop_mask is not None:
            d_attn = d_attn * attn_drop_mask / (1.0 - self.attn_dropout_rate)

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
            flat_ids = input_ids.flatten()          # (B*T,)
            flat_grads = d_x_total.reshape(-1, self.d_model) / self.cache['seq_len']
            np.add.at(self.token_embedding, flat_ids, -lr * flat_grads)

        # Update weights
        grads = {
                'output': d_Wo,
                'ffn2':   d_ffn2,
                'ffn1':   d_ffn1,
                'W_o':    d_Wo_attn,
                'W_q':    alpha * d_W_q,   # already alpha-scaled, clip the combined thing
                'W_k':    alpha * d_W_k,
                'W_v':    alpha * d_W_v,
            }
        grads, norm = self.clip_gradients(grads, max_norm)  

        self.output = self.apply_update(self.output, grads['output'], lr)
        self.ffn2   = self.apply_update(self.ffn2,   grads['ffn2'],   lr)
        self.ffn1   = self.apply_update(self.ffn1,   grads['ffn1'],   lr)
        self.W_o    = self.apply_update(self.W_o,    grads['W_o'],    lr)
        self.W_q    = self.apply_update(self.W_q,    grads['W_q'],    lr)
        self.W_k    = self.apply_update(self.W_k,    grads['W_k'],    lr)
        self.W_v    = self.apply_update(self.W_v,    grads['W_v'],    lr)

        if input_ids is not None:
            emb_norm = np.linalg.norm(d_x_total)
            emb_coef = min(1.0, max_norm / (emb_norm + 1e-6))

            flat_ids   = input_ids.flatten()                          # (B*T,)
            flat_grads = d_x_total.reshape(-1, self.d_model) / self.cache['seq_len']  # (B*T, D)

            np.add.at(self.token_embedding, flat_ids, -lr * emb_coef * flat_grads)
            self.pos_embedding[:seq_len] -= lr * emb_coef * d_x_total.mean(axis=0)
        else:
            self.pos_embedding[:seq_len] -= lr * d_x_total.mean(axis=0)
            norm = d_x_total

        return norm

    def smoothing_labels_utility(self, y_true, smoothing=0.1):
        # y_true: (B, num_classes) one-hot
        try:
            num_classes = y_true.shape[1]
        except:
            y_true_2d = y_true.reshape(-1, 1)
            num_classes = y_true_2d.shape[1] 
            
        return y_true * (1.0 - smoothing) + smoothing / num_classes

    def learning_rate_warm_up(self, epoch, epochs, lr_base, schedule='cosine_warmup', warmup_frac=0.1):
        warmup_epochs = int(epochs * warmup_frac)
        
        if schedule == 'cosine_warmup':
            if epoch < warmup_epochs:
                # Linear warmup
                return lr_base * (epoch + 1) / warmup_epochs
            else:
                # Cosine decay after warmup
                progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
                return lr_base * 0.5 * (1 + np.cos(np.pi * progress))

        elif schedule == 'step':
            # Halve lr every 30% of training
            step = int(epochs * 0.3)
            return lr_base * (0.5 ** (epoch // step))

        elif schedule == 'constant':
            return lr_base

        return lr_base

    def padding_mask_utility(self, input_ids, pad_token_id=0):
        # input_ids: (B, T)
        # Returns: (B, 1, 1, T) — broadcast-ready for (B, heads, T_q, T_k)
        mask = (input_ids != pad_token_id).astype(np.float32)
        return mask[:, np.newaxis, np.newaxis, :]   # (B, 1, 1, T)
        
    def clip_gradients(self, grads: dict, max_norm: float = 1.0):
        #clip gradients function to prevent overflow.
        total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads.values()))
        clip_coef  = max_norm / (total_norm + 1e-6)

        if clip_coef < 1.0:
            for g in grads.values():
                g *= clip_coef    

        return grads, total_norm    

    def batch_padding_utility(self, sequences, pad_token_id=0):
        # sequences: list of 1-D np arrays of varying length
        max_len = max(len(s) for s in sequences)
        padded  = np.full((len(sequences), max_len), pad_token_id, dtype=np.int32)
        for i, s in enumerate(sequences):
            padded[i, :len(s)] = s
        return padded   # (B, T)

    def train(self, input_ids_list, y_true_list, epochs=100, mode=None,
            lr=0.01, embedded=False, max_norm=1.0,
            schedule='cosine_warmup', pad_token_id=0, batch_size=None):

        # Main Training function for Transformer
        losses  = []
        accs    = []
        y_true_smoothed_list = []        
        d_model = self.d_model

        # W_o geometric init — unchanged
        if not self.encoded:
            self.shaping = GeometricWeightShaping(d_model, d_model)
            shaping_input = input_ids_list
            if embedded:
                shaping_input = np.vstack([
                    x.reshape(-1, x.shape[-1]) if x.ndim >= 2 else x
                    for x in input_ids_list
                ])
            self.W_o     = self.shaping.weight_shaping(shaping_input)
            self.encoded = True

        # batch padding — unchanged
        if batch_size is not None and not embedded:
            input_ids_list = [
                self.batch_padding_utility(input_ids_list[i:i+batch_size], pad_token_id)
                for i in range(0, len(input_ids_list), batch_size)
            ]
            y_true_list = [
                np.stack(y_true_list[i:i+batch_size])
                for i in range(0, len(y_true_list), batch_size)
            ]

        AME = self.AME_Encoder(input_ids_list)



        # precompute LR schedule once
        lr_schedule = [
            self.learning_rate_warm_up(e, epochs, lr, schedule)
            for e in range(epochs)
        ]

        print(f"[==] Starting comprehensive training for {epochs} epochs with mode: {mode}, learning rate: {lr}, schedule: {schedule}")

        for y in y_true_list:
            y_arr = np.asarray(y)
            if y_arr.ndim == 1:
                y_arr = y_arr[np.newaxis, :]
            smoothed = self.smoothing_labels_utility(y_arr, smoothing=0.1)
            # guarantee 2D output
            while smoothed.ndim > 2:
                smoothed = smoothed.squeeze(0)
            y_true_smoothed_list.append(smoothed)

        for epoch in range(epochs):
            epoch_losses = []
            epoch_accs   = []

            # FIX 2 — lookup instead of compute
            current_lr = lr_schedule[epoch]
            self.alpha = min(1.0, epoch / 100)

            # FIX 1 — pass precomputed smooth labels
            for input_ids, y_true, y_true_smooth in zip(
                input_ids_list, y_true_list, y_true_smoothed_list
            ):
                if input_ids.ndim == 1:
                    input_ids    = input_ids[np.newaxis, :]
                if y_true.ndim == 1:
                    y_true       = y_true[np.newaxis, :]
                
                loss, acc = self.train_step(
                    input_ids, epoch, y_true,
                    current_lr, AME=AME, mode=mode,
                    embedded=embedded, max_norm=max_norm,
                    pad_token_id=pad_token_id,
                    y_true_smooth=y_true_smooth   # pass precomputed
                )
                epoch_losses.append(loss)
                epoch_accs.append(acc)

            avg_loss = float(np.mean(epoch_losses))
            avg_acc  = float(np.mean(epoch_accs))
            losses.append(avg_loss)
            accs.append(avg_acc)

            if epoch % 10 == 0:
                print(f"[=] Epoch {epoch} | loss: {avg_loss:.4f} | Acc: {avg_acc:.2%}")

        return losses, accs


    def train_step(self, input_ids, epoch, y_true, lr=0.01, AME=None,
                mode=None, embedded=False, max_norm=1.0,
                pad_token_id=0, y_true_smooth=None):   

        y_true = np.asarray(y_true)
        if len(y_true.shape) < 2:
            y_true = y_true.reshape(-1, 1)

        if not embedded and input_ids.ndim == 1:
            input_ids = input_ids[np.newaxis, :]
        if y_true.ndim == 1:
            y_true = y_true[np.newaxis, :]

        probs, attn_weights = self.forward(
            input_ids, AME=AME, embedded=embedded,
            pad_token_id=pad_token_id, training=True,
            attn_dropout=self.attn_dropout_rate,
            ffn_dropout=self.ffn_dropout_rate
        )

        # only compute if not precomputed
        if y_true_smooth is not None:
            y_true_smooth = np.asarray(y_true_smooth)
            # squeeze out any extra leading dimensions
            while y_true_smooth.ndim > 2:
                y_true_smooth = y_true_smooth.squeeze(0)
            # ensure at least 2D
            if y_true_smooth.ndim == 1:
                y_true_smooth = y_true_smooth[np.newaxis, :]

        # shape alignment 
        if y_true_smooth.shape[0] and y_true_smooth.shape[1] != probs.shape[1]:
            if y_true_smooth.shape[1] > probs.shape[1]:
                y_true_smooth = y_true_smooth[:, :probs.shape[1]]
                y_true        = y_true[:, :probs.shape[1]]
            else:
                pad = probs.shape[1] - y_true_smooth.shape[1]
                y_true_smooth = np.pad(y_true_smooth, ((0,0),(0,pad)))
                y_true        = np.pad(y_true,        ((0,0),(0,pad)))

        loss     = -np.mean(np.sum(y_true_smooth * np.log(probs + 1e-8), axis=1))
        d_logits = (probs - y_true_smooth) / y_true_smooth.shape[0]

        if mode == 'fixed_backward':
            self.fixed_attention_backward(d_logits, lr, max_norm=max_norm)
        else:
            self.dynamic_backward(d_logits, lr, max_norm=max_norm)

        preds = np.argmax(probs, axis=1)
        true  = np.argmax(y_true, axis=1)
        acc   = np.mean(preds == true)

        return loss, acc

    

    def predict(self, input_ids, embedded=False):
        if not embedded and input_ids.ndim == 1:
            input_ids = input_ids.reshape(1, -1)

        AME = self.AME_Encoder(input_ids)
        probs, attn_weights = self.forward(input_ids, AME=AME, embedded=embedded, training=False, attn_dropout=0.0, ffn_dropout=0.0)
        preds = np.argmax(probs, axis=1)
        
        return preds, probs, attn_weights


    def AME_Encoder(self, x):
        # Optimized AME_Encoder for Transformer
        if _OPT_AVAILABLE and np.asarray(x).ndim == 2:
            return optimized_ame_encoder(np.asarray(x, dtype=np.float64))     

        X = np.asarray(x)
        # Regular AME Equations, higher AME provides capabilities for the model to experience errors during abstraction
        # Lower AME means lower chance for un optimal abstraction.
        
        gradient = np.gradient(x, axis=-1)
        grad_energy = np.mean(np.linalg.norm(gradient, axis=-1))       
        X_mag = np.mean(np.linalg.norm(X, axis=-1))

        AME = np.log1p(X_mag) * np.log1p(grad_energy) 
        return AME


    def anisotropy_measurement(self, x):
        eps = 1e-5
        try:
            gradient = np.gradient(x)
        except:
            subnet = x[:min(10, x.shape[0]), :min(10, x.shape[1])]
            gradient = np.gradient(subnet.flatten())

        val = [np.linalg.norm(v) for v in gradient]
        anisotropy = np.std(val) / np.mean(val) + eps 

        return anisotropy

    # attention quality computing provides the transformer a robust geometric complexity alignment scalar,
    #  this scalar can be used to compute alpha for a much stable forward pass in scarce data environment, allowing it to complement with AWE MLP below.
    def attention_quality_computing(self, attn_weights, AME=None, mask=None):
        eps = 1e-5
        batch, heads, seq_len, _ = attn_weights.shape

        if mask is not None:
            mask_expanded = np.broadcast_to(
                mask, (batch, heads, seq_len, seq_len)
            )
            # FIX 1 — inplace operations, ARM64 NEON works better on contiguous memory
            attn_weights = attn_weights * mask_expanded
            row_sums     = attn_weights.sum(axis=-1, keepdims=True) + eps
            attn_weights = attn_weights / row_sums

        # FIX 2 — fuse AME and anisotropy into single gradient pass
        # instead of calling AME_Encoder + anisotropy_measurement separately
        # both call np.gradient internally — compute once, reuse
        unsuitable_shape_condition = (
            attn_weights.shape[0] % 2 != 0,
            attn_weights.shape[1] % 2 != 0,   
            attn_weights.shape[2] % 2 != 0, 
            attn_weights.shape[3] % 2 != 0,    
        )
        if unsuitable_shape_condition:
            return self.robust_attention_quality_computing(attn_weights, AME=AME, mask=mask)
        else:
            gradient = np.gradient(attn_weights)

        # AME inline — avoids second np.gradient call in AME_Encoder
        # gradient is already a list of arrays, one per dimension
        # stack into single array for vectorized norm — ARM64 NEON friendly
        if AME is None:
            grad_stack  = np.stack([g.ravel() for g in gradient])  # (ndim, N)
            grad_norms  = np.linalg.norm(grad_stack, axis=1)       # (ndim,) — one NEON call
            grad_energy = grad_norms.mean()
            X_mag       = np.linalg.norm(attn_weights.ravel())  / attn_weights.size
            AME = np.log1p(X_mag) * np.log1p(grad_energy)

        AMR = 1.0 / (1.0 + np.exp(-float(AME)))

        # anisotropy inline — reuses grad_norms, no second gradient call
        anisotropy_val = grad_norms.std() / (grad_norms.mean() + eps)

        # FIX 3 — fuse entropy + max + var into single pass over attn_weights
        # avoids 3 separate scans of same array
        flat        = attn_weights.reshape(batch * heads * seq_len, seq_len)

        # entropy — one pass
        log_w       = np.log(flat + eps)
        entropy     = -(flat * log_w).sum(axis=-1)            # (B*H*T,)
        norm_entropy = 1.0 - entropy.mean() / np.log(seq_len + eps)

        # max — same flat array
        avg_max     = flat.max(axis=-1).mean()

        # var — same flat array
        norm_var    = np.clip(flat.var() * seq_len, 0.0, 1.0)

        # quality score
        qualified     = (1.0 - AMR) + eps * anisotropy_val
        quality_score = (qualified * norm_entropy +
                        qualified * avg_max +
                        anisotropy_val * norm_var)

        return float(np.clip(quality_score, 0.0, 1.0)) 

    def robust_attention_quality_computing(self, attn_weights, AME=None, mask=None):
        eps = 1e-5
        eps = 1e-5
        batch, heads, seq_len, _ = attn_weights.shape    

        if AME is None:
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


class Transformer:
    """
    A single-block Transformer classifier designed for low-sample environments.
 
    Key design philosophy:
    - Dynamic alpha gating: blends fixed (stable) and learned (flexible) attention
      projections based on data complexity, measured by AME (Abstract Modelling Energy).
      Early in training alpha ~ 0, so attention stays fixed and gradients are stable.
      As training progresses alpha → 1, allowing full dynamic attention.
    - GWS (Geometric Weight Shaping): initializes W_o using data geometry rather than
      random noise, giving the output projection a head start aligned with the input
      distribution's geometric complexity.
    - All improvements (gradient clipping, lr scheduling, padding masks, alpha-aware
      dropout, weight decay, label smoothing) are coordinated around the alpha warmup
      so they don't fight each other during early training.
    """
 
    def __init__(self, vocab_size, d_model=8, n_heads=2, num_classes=7,
                 learning_rate=0.01, attn_dropout=0.0, ffn_dropout=0.0, weight_decay=1e-4):
        """
        Args:
            vocab_size:     Number of unique tokens in the vocabulary.
            d_model:        Embedding dimension. Keep small (8-32) for low-sample data
                            to avoid overparameterization.
            n_heads:        Number of attention heads. d_model must be divisible by n_heads.
            num_classes:    Number of output classes for classification.
            learning_rate:  Base learning rate — actual lr is scheduled via get_lr().
            attn_dropout:   Dropout rate on attention output. Scaled by alpha at runtime,
                            so effective rate = attn_dropout * alpha.
            ffn_dropout:    Dropout rate on FFN activations (after ReLU). Also alpha-scaled.
            weight_decay:   L2 regularization coefficient applied at each weight update.
                            Does NOT apply to biases, embeddings, or layer norm params.
        """
        self.d_model = d_model
        self.n_heads = n_heads
        self.attn_dropout_rate = attn_dropout
        self.ffn_dropout_rate  = ffn_dropout
        self.transformer_lr    = learning_rate
        self.weight_decay      = weight_decay
 
        # Token embedding table: maps token ids → d_model vectors
        # Small init scale (0.02) prevents large logits at start
        self.token_embedding = np.random.randn(vocab_size, d_model) * 0.02
 
        # Positional embeddings: adds word-order signal to token embeddings
        # Supports sequences up to length 100
        self.pos_embedding = np.random.randn(100, d_model) * 0.02
 
        # Multi-head attention projections: shape (n_heads, d_model, d_model // n_heads)
        # Each head gets its own Q/K/V slice of dimension d_model // n_heads
        self.W_q = np.random.randn(n_heads, d_model, d_model // n_heads) * 0.02
        self.W_k = np.random.randn(n_heads, d_model, d_model // n_heads) * 0.02
        self.W_v = np.random.randn(n_heads, d_model, d_model // n_heads) * 0.02
 
        # Fixed copies of QKV projections — used for alpha blending.
        # When alpha=0: W_q_mix = W_q_fixed (no updates flow to QKV).
        # When alpha=1: W_q_mix = W_q (fully dynamic).
        # This gives stability in early training without permanently sacrificing flexibility.
        self.W_q_fixed = self.W_q.copy()
        self.W_k_fixed = self.W_k.copy()
        self.W_v_fixed = self.W_v.copy()
 
        # Output projection: concatenated head outputs → d_model
        # This is the only weight initialized by GWS — geometric init only helps
        # in fixed output projections, not in the QKV matrices.
        self.W_o = np.random.randn(d_model, d_model) * 0.02
 
        # Flag to ensure GWS shaping only runs once at the start of train()
        self.encoded = False
 
        # Feed-forward network: two-layer MLP with ReLU, hidden dim = d_model * 4
        # FFN provides non-linear transformation after attention
        self.ffn1 = np.random.randn(d_model, d_model * 4) * 0.02
        self.ffn2 = np.random.randn(d_model * 4, d_model) * 0.02
 
        # Layer norm parameters: scale (gamma) and shift (beta)
        # ln1 normalizes after attention + residual
        # ln2 normalizes after FFN + residual
        self.ln1_scale = np.ones(d_model)
        self.ln1_shift = np.zeros(d_model)
        self.ln2_scale = np.ones(d_model)
        self.ln2_shift = np.zeros(d_model)
 
        # Classification head: pooled d_model vector → num_classes logits
        self.output      = np.random.randn(d_model, num_classes) * 0.02
        self.output_bias = np.zeros(num_classes)
 
        # Intermediate activations stored here during forward pass for use in backward
        self.cache = {}
 
 
    # ─────────────────────────────────────────────
    # Core math utilities
    # ─────────────────────────────────────────────
 
    def layer_norm(self, x, scale, shift):
        """
        Layer normalization: normalizes across the last dimension (d_model).
        Stabilizes training by keeping activations at unit variance.
        eps=1e-5 prevents division by zero on zero-variance inputs.
        """
        mean = np.mean(x, axis=-1, keepdims=True)
        var  = np.var(x,  axis=-1, keepdims=True)
        return scale * (x - mean) / np.sqrt(var + 1e-5) + shift
 
 
    def apply_update(self, param, grad, lr):
        """
        L2 weight decay update: param -= lr * (grad + decay * param)
        Equivalent to adding decay * param to the gradient before stepping.
        Shrinks weights toward zero each step, discouraging large weights.
        NOTE: never call this on biases, embeddings, or layer norm params.
        """
        return param - lr * (grad + self.weight_decay * param)
 
 
    def dropout(self, x, rate=0.1, training=True, alpha=None):
        """
        Inverted dropout with alpha-aware effective rate.
 
        Standard dropout randomly zeros activations during training.
        Inverted scaling (divide by 1-rate) means no scaling is needed at inference.
 
        Alpha scaling: effective_rate = rate * alpha
        - When alpha ~ 0 (early training, fixed attention): near-zero dropout
          → clean gradients when the model is most fragile
        - When alpha ~ 1 (late training, dynamic attention): full dropout rate
          → regularization kicks in when the model has enough capacity to benefit
 
        Returns (dropped_x, mask) during training, (x, None) at inference.
        """
        if not training or rate == 0.0:
            return x, None
 
        effective_rate = rate * alpha if alpha is not None else rate
 
        if effective_rate == 0.0:
            return x, None
 
        mask = (np.random.rand(*x.shape) > effective_rate).astype(np.float32)
        return x * mask / (1.0 - effective_rate), mask
 
 
    def softmax(self, x):
        """
        Numerically stable softmax: subtracts max before exp to prevent overflow.
        Works on any ndim — keepdims ensures correct broadcasting.
        """
        shifted = x - np.max(x, axis=-1, keepdims=True)
        exp_x   = np.exp(shifted)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
 
 
    # ─────────────────────────────────────────────
    # Attention
    # ─────────────────────────────────────────────
 
    def attention(self, Q, K, V, mask=None):
        """
        Scaled dot-product attention.
 
        scores = QK^T / sqrt(d_k)  — scaling prevents softmax saturation
                                      in high-dimensional spaces
        mask:  (B, 1, 1, T) padding mask — positions where mask==0 get -1e9
               so they become ~0 after softmax (padding tokens ignored)
        clip:  scores clipped to [-50, 50] for numerical stability before softmax
 
        Returns attention output (B, heads, T, d_k) and weights (B, heads, T, T).
        """
        d_k    = Q.shape[-1]
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
        scores = np.clip(scores, -50, 50)
 
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)
 
        weights = self.softmax(scores)
        output  = np.matmul(weights, V)
        return output, weights
 
 
    def multi_head_attention(self, x, mask=None, alpha=None):
        """
        Multi-head attention with dynamic alpha blending.
 
        Alpha blending: W_mix = (1 - alpha) * W_fixed + alpha * W_learned
        - alpha=0: uses frozen initial projections → stable, no QKV gradient flow
        - alpha=1: uses fully learned projections → maximum flexibility
        - intermediate: smooth interpolation between the two
 
        Heads allow the model to attend to different representation subspaces
        simultaneously. Each head operates on a d_model // n_heads slice.
 
        Flow: x → Q,K,V projections → scaled dot-product attention
              → concatenate heads → W_o projection → output
        """
        batch_size, seq_len, d_model = x.shape
 
        # Blend fixed and learned projections based on current alpha
        W_q_mix = (1 - alpha) * self.W_q_fixed + alpha * self.W_q
        W_k_mix = (1 - alpha) * self.W_k_fixed + alpha * self.W_k
        W_v_mix = (1 - alpha) * self.W_v_fixed + alpha * self.W_v
 
        # Project input to Q, K, V for all heads simultaneously via einsum
        # 'bsd,hdm->bhsm': batch, seq, d_model × heads, d_model, head_dim
        Q = np.einsum('bsd,hdm->bhsm', x, W_q_mix)
        K = np.einsum('bsd,hdm->bhsm', x, W_k_mix)
        V = np.einsum('bsd,hdm->bhsm', x, W_v_mix)
 
        # Cache Q, K, V for backward pass gradient computation
        self.cache['Q'] = Q
        self.cache['K'] = K
        self.cache['V'] = V
        self.cache['x_attn_input'] = x
 
        attn_output, attn_weights = self.attention(Q, K, V, mask)
        self.cache['attn_weights'] = attn_weights
        self.cache['attn_output']  = attn_output
 
        # Merge heads: (B, heads, T, head_dim) → (B, T, d_model)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        self.cache['attn_concat'] = attn_output
 
        # Final linear projection mixes information across heads
        output = np.matmul(attn_output, self.W_o)
        self.cache['attn_out'] = output
 
        return output, attn_weights
 
 
    # ─────────────────────────────────────────────
    # Forward pass
    # ─────────────────────────────────────────────
 
    def forward(self, input_ids, embedded=False, pad_token_id=0,
                training=True, attn_dropout=0.1, ffn_dropout=0.1):
        """
        Full forward pass: tokens → probabilities.
 
        Two input modes:
        - embedded=False (default): input_ids are integer token indices (B, T)
          → looked up in token_embedding, positional embedding added
        - embedded=True: input_ids are already float embeddings (B, T, D)
          → skip embedding lookup, no padding mask available
 
        Alpha computation:
        1. AME_Encoder measures input complexity → raw alpha signal
        2. Sigmoid maps to (0, 1)
        3. After attention, EMA blends with attention_quality_computing score:
           alpha = 0.95 * alpha + 0.05 * quality_score
           This makes alpha track both input complexity AND attention quality.
 
        Masked mean pooling:
        - Averages only over valid (non-padding) token positions
        - Prevents padding zeros from diluting the pooled representation
        - Backward pass must use the same mask and lengths for correct gradients
        """
        if embedded:
            x = np.asarray(input_ids, dtype=np.float32)
            if x.ndim == 2:
                x = x[np.newaxis, ...]
            batch_size, seq_len, _ = x.shape
            self.cache['embedded_input'] = x
            self.cache['input_ids']      = None
            mask = None
        else:
            input_ids = np.asarray(input_ids, dtype=np.int32)  # guard against float input
            if input_ids.ndim == 1:
                input_ids = input_ids[np.newaxis, :]
 
            x         = self.token_embedding[input_ids]          # (B, T, D)
            x         = x + self.pos_embedding[:x.shape[1]]      # add positional signal
            batch_size, seq_len = input_ids.shape
            self.cache['embedded_input'] = None
            self.cache['input_ids']      = input_ids
            mask = self.padding_mask_utility(input_ids, pad_token_id)  # (B, 1, 1, T)
 
        self.cache['mask']       = mask if not embedded else None
        self.cache['seq_len']    = seq_len
        self.cache['batch_size'] = batch_size
        self.cache['x_token']    = x
        self.cache['x_pos']      = x
 
        # Compute alpha from input complexity before attention
        # AME measures gradient energy × magnitude — proxy for data complexity
        AME   = self.AME_Encoder(x)
        alpha = 1.0 / (1.0 + np.exp(-AME))  # sigmoid → (0, 1)
 
        attn_out, attn_weights = self.multi_head_attention(x, mask=mask, alpha=alpha)
 
        # Apply alpha-scaled dropout to attention output
        # current_alpha from cache (may differ from freshly computed alpha above
        # during the first step when cache is empty)
        current_alpha = self.cache.get('alpha', 0.0)
        attn_out, attn_drop_mask = self.dropout(
            attn_out, rate=self.attn_dropout_rate,
            training=training, alpha=current_alpha
        )
        self.cache['attn_drop_mask'] = attn_drop_mask
 
        # Refine alpha with attention quality signal (EMA update)
        # quality_computing returns a scalar in (0,1) based on attention entropy,
        # max attention, and anisotropy — higher = more structured attention
        alpha = 0.95 * alpha + 0.05 * self.attention_quality_computing(attn_weights, mask=mask)
        self.alpha        = alpha
        self.cache['alpha'] = alpha
 
        # Attention sub-layer: residual connection + layer norm
        self.cache['x_ln1_input'] = x + attn_out
        x = self.layer_norm(x + attn_out, self.ln1_scale, self.ln1_shift)
        self.cache['x_after_ln1'] = x
 
        # FFN sub-layer: two linear layers with ReLU activation
        self.cache['ffn_input'] = x
        ffn_pre  = np.matmul(x, self.ffn1)
        self.cache['ffn_pre'] = ffn_pre
 
        ffn_act = np.maximum(0, ffn_pre)   # ReLU: zero out negative pre-activations
 
        # Apply alpha-scaled dropout to FFN activations (after ReLU, before ffn2)
        ffn_act, ffn_drop_mask = self.dropout(
            ffn_act, rate=self.ffn_dropout_rate,
            training=training, alpha=current_alpha
        )
        self.cache['ffn_act']      = ffn_act
        self.cache['ffn_drop_mask'] = ffn_drop_mask
 
        ffn_out = np.matmul(ffn_act, self.ffn2)
        self.cache['ffn_out'] = ffn_out
 
        # FFN sub-layer: residual connection + layer norm
        self.cache['x_ln2_input'] = x + ffn_out
        x = self.layer_norm(x + ffn_out, self.ln2_scale, self.ln2_shift)
        self.cache['x_after_ln2'] = x
 
        # Masked mean pooling: average across valid token positions only
        # Padding positions (mask=0) are zeroed out before summing
        if mask is not None:
            token_mask = mask[:, 0, 0, :, np.newaxis]              # (B, T, 1)
            x_masked   = x * token_mask                            # zero padding
            lengths    = token_mask.sum(axis=1)                    # (B, 1) valid counts
            x_pooled   = x_masked.sum(axis=1) / (lengths + 1e-6)  # (B, D)
        else:
            x_pooled = np.mean(x, axis=1)                         # (B, D)
 
        self.cache['x_pooled'] = x_pooled
 
        # Classification head: pooled vector → class logits → probabilities
        logits = np.matmul(x_pooled, self.output) + self.output_bias
        self.cache['logits'] = logits
 
        probs = self.softmax(logits)
        self.cache['probs'] = probs
 
        return probs, attn_weights
 
 
    # ─────────────────────────────────────────────
    # Backward passes
    # ─────────────────────────────────────────────
 
    def layer_norm_backward(self, d_out, x, scale, shift):
        """
        Analytic backward through layer normalization.
 
        Computes dx given upstream gradient d_out and the original pre-norm input x.
        Three gradient terms:
        1. dx_hat / std          — direct path through normalization
        2. dvar * 2(x-mean)/N   — path through variance
        3. dmean / N             — path through mean
 
        Note: the dvar * mean(-2*(x-mean)) term found in some implementations
        is always zero (mean of deviations from the mean = 0) and is omitted here.
        """
        eps  = 1e-5
        mean = np.mean(x, axis=-1, keepdims=True)
        var  = np.var(x,  axis=-1, keepdims=True)
        std  = np.sqrt(var + eps)
        N    = x.shape[-1]
 
        dx_hat = d_out * scale
        dvar   = np.sum(dx_hat * (x - mean) * -0.5 * std**-3, axis=-1, keepdims=True)
        dmean  = np.sum(dx_hat * (-1.0 / std), axis=-1, keepdims=True)
 
        dx = dx_hat / std + dvar * 2 * (x - mean) / N + dmean / N
        return dx
 
 
    def fixed_attention_backward(self, d_logits, lr=0.01, max_norm=1.0):
        """
        Backward pass that does NOT update Q, K, V projections.
 
        Used when mode='fixed_backward'. Gradients stop at W_o — the QKV
        projections remain frozen at their initial values. This sacrifices
        flexibility for maximum stability, useful when data is very scarce
        or when you want the attention pattern to stay fixed while only
        the FFN and output head adapt.
 
        Gradient flow (top to bottom):
        logits → output head → pooling → layer_norm_2 → FFN → layer_norm_1
        → attention output (W_o only, no further)
        """
        d_output = d_logits
 
        # Output head gradients
        d_Wo     = np.dot(self.cache['x_pooled'].T, d_output)
        d_bo     = np.sum(d_output, axis=0, keepdims=True)
        d_pooled = np.dot(d_output, self.output.T)
 
        # Expand pooled gradient back to all token positions uniformly
        d_x = np.repeat(
            d_pooled[:, np.newaxis, :] / self.cache['seq_len'],
            self.cache['seq_len'], axis=1
        )
 
        # Layer norm 2 backward
        d_x = self.layer_norm_backward(d_x, self.cache['x_ln2_input'],
                                        self.ln2_scale, self.ln2_shift)
 
        # FFN backward
        d_ffn  = d_x
        d_ffn2 = np.sum(np.matmul(self.cache['ffn_act'].transpose(0, 2, 1), d_ffn), axis=0)
 
        d_ffn_act = np.matmul(d_ffn, self.ffn2.T)
 
        # FFN dropout backward: re-apply the same mask used in forward,
        # then rescale by 1/(1-rate) to undo the inverted scaling
        ffn_drop_mask = self.cache.get('ffn_drop_mask')
        if ffn_drop_mask is not None:
            d_ffn_act = d_ffn_act * ffn_drop_mask / (1.0 - self.ffn_dropout_rate)
 
        # ReLU backward: zero gradient where pre-activation was negative
        d_ffn_pre = d_ffn_act * (self.cache['ffn_pre'] >= 0)
 
        d_prev = np.matmul(d_ffn_pre, self.ffn1.T)
        d_ffn1 = np.sum(np.matmul(self.cache['ffn_input'].transpose(0, 2, 1), d_ffn_pre), axis=0)
 
        # Layer norm 1 backward
        # Note: subtracts attn_out because d_x currently includes the FFN residual path
        d_x = self.layer_norm_backward(
            d_x - self.cache['attn_out'],
            self.cache['x_ln1_input'],
            self.ln1_scale, self.ln1_shift
        )
 
        d_ffn          = d_x
        d_residual_ffn = d_ffn
        dx             = d_prev + d_residual_ffn
        d_attn         = dx
 
        # Attention dropout backward
        attn_drop_mask = self.cache.get('attn_drop_mask')
        if attn_drop_mask is not None:
            d_attn = d_attn * attn_drop_mask / (1.0 - self.attn_dropout_rate)
 
        # W_o gradient only — gradient stops here (no QKV updates)
        d_Wo_attn = np.sum(
            np.matmul(self.cache['attn_concat'].transpose(0, 2, 1), d_attn), axis=0
        )
 
        # Collect, clip, then apply updates with weight decay
        grads = {
            'output': d_Wo,
            'ffn2':   d_ffn2,
            'ffn1':   d_ffn1,
            'W_o':    d_Wo_attn,
        }
        grads, norm = self.clip_gradients(grads, max_norm)
 
        self.output = self.apply_update(self.output, grads['output'], lr)
        self.output_bias -= lr * d_bo.squeeze()   # bias: no weight decay
        self.ffn2   = self.apply_update(self.ffn2, grads['ffn2'], lr)
        self.ffn1   = self.apply_update(self.ffn1, grads['ffn1'], lr)
        self.W_o    = self.apply_update(self.W_o,  grads['W_o'],  lr)
 
        return d_x
 
 
    def dynamic_backward(self, d_logits, lr=0.01, max_norm=1.0):
        """
        Full backward pass including Q, K, V projection updates.
 
        Used when mode='dynamic_backward' (default). QKV gradients are
        alpha-scaled before clipping — when alpha is low, QKV updates are
        small, preserving stability during early training.
 
        Token embedding updates use np.add.at (scatter-add) rather than
        fancy indexing to correctly accumulate gradients for repeated tokens.
        Embeddings and positional embeddings are clipped separately from the
        main gradient dict because they are variable-shape scatter updates.
 
        Gradient flow (top to bottom):
        logits → output head → masked pooling → layer_norm_2 → FFN
        → layer_norm_1 → W_o → attention heads → Q, K, V projections
        → token embeddings + positional embeddings
        """
        d_output = d_logits
        alpha    = self.cache.get('alpha', 1.0)
 
        # Output head gradients
        d_Wo     = np.dot(self.cache['x_pooled'].T, d_output)
        d_bo     = np.sum(d_output, axis=0)
        d_pooled = np.dot(d_output, self.output.T)
 
        # Masked pooling backward: gradient flows only to valid (non-padding) positions
        # Divides by per-sequence valid token count, matching the forward mean
        mask = self.cache['mask']
        if mask is not None:
            token_mask = mask[:, 0, 0, :, np.newaxis]              # (B, T, 1)
            lengths    = token_mask.sum(axis=1, keepdims=True)     # (B, 1, 1)
            d_x        = (d_pooled[:, np.newaxis, :] / (lengths + 1e-6)) * token_mask
        else:
            d_x = np.repeat(
                d_pooled[:, np.newaxis, :] / self.cache['seq_len'],
                self.cache['seq_len'], axis=1
            )
 
        # Layer norm 2 backward
        d_x = self.layer_norm_backward(d_x, self.cache['x_ln2_input'],
                                        self.ln2_scale, self.ln2_shift)
 
        # FFN backward
        d_ffn  = d_x
        d_ffn2 = np.sum(np.matmul(self.cache['ffn_act'].transpose(0, 2, 1), d_ffn), axis=0)
 
        d_ffn_act = np.matmul(d_ffn, self.ffn2.T)
 
        # FFN dropout backward
        ffn_drop_mask = self.cache.get('ffn_drop_mask')
        if ffn_drop_mask is not None:
            d_ffn_act = d_ffn_act * ffn_drop_mask / (1.0 - self.ffn_dropout_rate)
 
        # ReLU backward
        d_ffn_pre = d_ffn_act * (self.cache['ffn_pre'] >= 0)
 
        d_prev = np.matmul(d_ffn_pre, self.ffn1.T)
        d_ffn1 = np.sum(np.matmul(self.cache['ffn_input'].transpose(0, 2, 1), d_ffn_pre), axis=0)
 
        # Layer norm 1 backward
        d_ln1     = self.layer_norm_backward(d_x, self.cache['x_ln1_input'],
                                              self.ln1_scale, self.ln1_shift)
        d_residual = d_ln1
        d_attn     = d_ln1
        dx         = d_prev + d_residual
 
        # Attention dropout backward
        attn_drop_mask = self.cache.get('attn_drop_mask')
        if attn_drop_mask is not None:
            d_attn = d_attn * attn_drop_mask / (1.0 - self.attn_dropout_rate)
 
        # W_o gradient
        d_Wo_attn     = np.sum(
            np.matmul(self.cache['attn_concat'].transpose(0, 2, 1), d_attn), axis=0
        )
 
        # Attention backward: split concatenated head gradient back into per-head tensors
        d_attn_concat = np.matmul(d_attn, self.W_o.T)
        batch, seq_len, _ = d_attn_concat.shape
        d_head = self.n_heads
        d_dim  = self.d_model // self.n_heads
 
        # Reshape (B, T, D) → (B, heads, T, head_dim) to match Q/K/V layout
        d_attn_heads = d_attn_concat.reshape(batch, seq_len, d_head, d_dim).transpose(0, 2, 1, 3)
 
        V      = self.cache['V']
        K      = self.cache['K']
        Q      = self.cache['Q']
        weight = self.cache['attn_weights']
 
        # Gradient through softmax attention weights
        d_V       = np.matmul(weight.transpose(0, 1, 3, 2), d_attn_heads)
        d_weights = np.matmul(d_attn_heads, V.transpose(0, 1, 3, 2))
 
        # Softmax backward: d_scores = weights * (d_weights - sum(d_weights * weights))
        d_scores  = weight * (d_weights - np.sum(d_weights * weight, axis=-1, keepdims=True))
        d_scores /= np.sqrt(Q.shape[-1])  # undo the sqrt(d_k) scaling
 
        d_Q = np.matmul(d_scores, K)
        d_K = np.matmul(d_scores.transpose(0, 1, 3, 2), Q)
 
        x = self.cache['x_attn_input']
 
        # QKV projection gradients via einsum (inverse of forward einsum)
        d_W_q = np.einsum('bsd,bhsm->hdm', x, d_Q)
        d_W_k = np.einsum('bsd,bhsm->hdm', x, d_K)
        d_W_v = np.einsum('bsd,bhsm->hdm', x, d_V)
 
        # Input gradients from Q, K, V paths (sum all three contributions)
        d_x_q = np.einsum('bhsm,hdm->bsd', d_Q, self.W_q)
        d_x_k = np.einsum('bhsm,hdm->bsd', d_K, self.W_k)
        d_x_v = np.einsum('bhsm,hdm->bsd', d_V, self.W_v)
 
        d_x_attn_input = d_x_q + d_x_k + d_x_v
        d_x_total      = d_x_attn_input + d_residual
 
        # Collect main weight gradients, alpha-scale QKV, clip globally
        grads = {
            'output': d_Wo,
            'ffn2':   d_ffn2,
            'ffn1':   d_ffn1,
            'W_o':    d_Wo_attn,
            'W_q':    alpha * d_W_q,   # alpha-scaled: small updates when attention is mostly fixed
            'W_k':    alpha * d_W_k,
            'W_v':    alpha * d_W_v,
        }
        grads, norm = self.clip_gradients(grads, max_norm)
 
        self.output = self.apply_update(self.output, grads['output'], lr)
        self.output_bias -= lr * d_bo.squeeze()   # bias: no weight decay
        self.ffn2   = self.apply_update(self.ffn2, grads['ffn2'], lr)
        self.ffn1   = self.apply_update(self.ffn1, grads['ffn1'], lr)
        self.W_o    = self.apply_update(self.W_o,  grads['W_o'],  lr)
        self.W_q    = self.apply_update(self.W_q,  grads['W_q'],  lr)
        self.W_k    = self.apply_update(self.W_k,  grads['W_k'],  lr)
        self.W_v    = self.apply_update(self.W_v,  grads['W_v'],  lr)
 
        input_ids = self.cache.get('input_ids')
 
        if input_ids is not None:
            # Embedding and positional embedding updates clipped separately
            # (variable shape, scatter update — can't go in the main grads dict)
            emb_norm  = np.linalg.norm(d_x_total)
            emb_coef  = min(1.0, max_norm / (emb_norm + 1e-6))
 
            flat_ids   = input_ids.flatten()                              # (B*T,)
            flat_grads = d_x_total.reshape(-1, self.d_model) / self.cache['seq_len']  # (B*T, D)
 
            # np.add.at correctly accumulates gradients for repeated token ids
            # (fancy indexing would silently drop duplicates)
            np.add.at(self.token_embedding, flat_ids, -lr * emb_coef * flat_grads)
            self.pos_embedding[:seq_len] -= lr * emb_coef * d_x_total.mean(axis=0)
        else:
            self.pos_embedding[:seq_len] -= lr * d_x_total.mean(axis=0)
            norm = d_x_total
 
        return norm
 
 
    # ─────────────────────────────────────────────
    # Training utilities
    # ─────────────────────────────────────────────
 
    def smoothing_labels_utility(self, y_true, smoothing=0.1):
        """
        Label smoothing: softens one-hot targets to prevent overconfidence.
 
        Hard target: [0, 0, 1, 0, 0, 0, 0]
        Smoothed:    [0.014, 0.014, 0.914, 0.014, 0.014, 0.014, 0.014]
 
        Formula: y_smooth = y * (1 - smoothing) + smoothing / num_classes
        The model is penalized less for wrong classes and less rewarded for
        being maximally confident on the correct class. Helps generalization
        especially on small datasets where the model tends to overfit hard labels.
        """
        num_classes = y_true.shape[1]
        return y_true * (1.0 - smoothing) + smoothing / num_classes
 
 
    def learning_rate_warm_up(self, epoch, epochs, lr_base,
                               schedule='cosine_warmup', warmup_frac=0.1):
        """
        Learning rate scheduling with optional linear warmup.
 
        cosine_warmup (recommended):
        - Linear warmup for first warmup_frac * epochs epochs
        - Cosine decay from peak back toward 0 for the remainder
        - Coordinates naturally with alpha warmup: low lr while alpha is low,
          peak lr as alpha reaches ~0.1-0.3, then both decay together
 
        step:
        - Halves lr every 30% of training — simpler, less smooth
 
        constant:
        - No scheduling — useful for debugging or very short runs
        """
        warmup_epochs = int(epochs * warmup_frac)
 
        if schedule == 'cosine_warmup':
            if epoch < warmup_epochs:
                return lr_base * (epoch + 1) / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
                return lr_base * 0.5 * (1 + np.cos(np.pi * progress))
 
        elif schedule == 'step':
            step = int(epochs * 0.3)
            return lr_base * (0.5 ** (epoch // step))
 
        elif schedule == 'constant':
            return lr_base
 
        return lr_base
 
 
    def padding_mask_utility(self, input_ids, pad_token_id=0):
        """
        Creates a broadcast-ready padding mask from token ids.
 
        Positions where input_ids == pad_token_id are marked 0 (ignore),
        all other positions are marked 1 (attend).
 
        Output shape (B, 1, 1, T) broadcasts correctly against attention
        scores of shape (B, heads, T_query, T_key) — the 1s expand automatically
        across the heads and query dimensions without copying memory.
        """
        mask = (input_ids != pad_token_id).astype(np.float32)
        return mask[:, np.newaxis, np.newaxis, :]   # (B, 1, 1, T)
 
 
    def clip_gradients(self, grads: dict, max_norm: float = 1.0):
        """
        Global gradient norm clipping.
 
        Computes the L2 norm across ALL gradients in the dict combined,
        then scales all gradients down proportionally if the norm exceeds max_norm.
        Only scales DOWN — never amplifies small gradients.
 
        Global (not per-parameter) clipping is important here because
        alpha-scaled QKV gradients are already small; per-parameter clipping
        would amplify them relative to the FFN gradients.
 
        Returns (clipped_grads, total_norm) — monitor total_norm in training
        logs to detect gradient explosions or overly aggressive clipping.
        """
        total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads.values()))
        clip_coef  = max_norm / (total_norm + 1e-6)
 
        if clip_coef < 1.0:
            grads = {k: g * clip_coef for k, g in grads.items()}
 
        return grads, total_norm
 
 
    def batch_padding_utility(self, sequences, pad_token_id=0):
        """
        Pads a list of variable-length token sequences to the same length.
 
        Finds the longest sequence in the batch, then right-pads all shorter
        sequences with pad_token_id. The padding mask in forward() will
        ensure these positions are ignored during attention and pooling.
 
        Args:
            sequences:    list of 1-D int arrays of varying lengths
            pad_token_id: token id used for padding (default 0)
        Returns:
            padded array of shape (B, max_len) dtype int32
        """
        max_len = max(len(s) for s in sequences)
        padded  = np.full((len(sequences), max_len), pad_token_id, dtype=np.int32)
        for i, s in enumerate(sequences):
            padded[i, :len(s)] = s
        return padded   # (B, T)
 
 
    def train_step(self, input_ids, epoch, y_true, lr=0.01, mode=None,
                   embedded=False, max_norm=1.0, pad_token_id=0):
        """
        Single training step: forward → loss → backward → weight update.
 
        Label smoothing is applied to y_true before loss computation,
        but accuracy is measured against the original hard labels so the
        metric remains interpretable throughout training.
 
        mode='fixed_backward': only FFN, W_o, and output head update
        mode=None (default):   full dynamic_backward including QKV and embeddings
        """
        if not embedded and input_ids.ndim == 1:
            input_ids = input_ids[np.newaxis, :]
        if y_true.ndim == 1:
            y_true = y_true[np.newaxis, :]
 
        probs, attn_weights = self.forward(
            input_ids, embedded=embedded, pad_token_id=pad_token_id,
            training=True, attn_dropout=self.attn_dropout_rate,
            ffn_dropout=self.ffn_dropout_rate
        )
 
        # Smooth labels for loss and gradient computation
        y_true_smooth = self.smoothing_labels_utility(y_true, smoothing=0.1)
 
        # Shape alignment guard (handles edge cases in multi-class setups)
        if y_true_smooth.shape[0] and y_true_smooth.shape[1] != probs.shape[0] and probs.shape[1]:
            if y_true_smooth.shape[1] > probs.shape[1]:
                y_true_smooth = y_true_smooth[:, :probs.shape[1]]
            else:
                y_true_smooth = np.pad(y_true_smooth, ((0,0),(0, probs.shape[1]-y_true_smooth.shape[1])), mode='constant')
 
        if y_true.shape[0] and y_true.shape[1] != probs.shape[0] and probs.shape[1]:
            if y_true.shape[1] > probs.shape[1]:
                y_true = y_true[:, :probs.shape[1]]
            else:
                y_true = np.pad(y_true, ((0,0),(0, probs.shape[1]-y_true.shape[1])), mode='constant')
 
        # Cross-entropy loss on smoothed labels
        loss = -np.mean(np.sum(y_true_smooth * np.log(probs + 1e-8), axis=1))
 
        # Gradient of cross-entropy loss w.r.t. logits: (probs - y_true) / batch_size
        d_logits = (probs - y_true_smooth) / y_true_smooth.shape[0]
 
        if mode == 'fixed_backward':
            self.fixed_attention_backward(d_logits, lr, max_norm=max_norm)
        else:
            self.dynamic_backward(d_logits, lr, max_norm=max_norm)
 
        # Accuracy uses hard labels (not smoothed) for interpretable reporting
        preds = np.argmax(probs, axis=1)
        true  = np.argmax(y_true, axis=1)
        acc   = np.mean(preds == true)
 
        return loss, acc
 
 
    def train(self, input_ids_list, y_true_list, epochs=100, mode=None,
              lr=0.01, embedded=False, max_norm=1.0, schedule='cosine_warmup',
              pad_token_id=0, batch_size=None):
        """
        Full training loop with GWS initialization, lr scheduling, and optional batching.
 
        GWS (Geometric Weight Shaping):
        - Runs once at the start via self.shaping.weight_shaping()
        - Replaces the random W_o init with one derived from the data's geometry
        - self.encoded=True prevents re-running on subsequent train() calls
        - Note: W_o drifts during training — GWS effect is strongest early on
 
        Batching:
        - batch_size=None: original behavior, one sample per step
        - batch_size=N: sequences are grouped and padded before training starts
          All padding happens once upfront, not per-epoch
 
        Alpha warmup: self.alpha = min(1.0, epoch / 100)
        - Runs in parallel with lr schedule
        - Both start low and rise together, then lr decays while alpha stays at 1.0
        """
        losses, accs = [], []
        d_model = self.d_model
 
        # GWS: geometric initialization of W_o from data structure
        if not self.encoded:
            self.shaping = GeometricWeightShaping(d_model, d_model)
            if embedded:
                shaping_input = np.vstack([
                    x.reshape(-1, x.shape[-1]) if x.ndim >= 2 else x
                    for x in input_ids_list
                ])
            else:
                shaping_input = input_ids_list
            self.W_o      = self.shaping.weight_shaping(shaping_input)
            self.encoded  = True
 
        # Pre-pad into batches once before training (not per-epoch)
        if batch_size is not None and not embedded:
            input_ids_list = [
                self.batch_padding_utility(input_ids_list[i:i+batch_size], pad_token_id)
                for i in range(0, len(input_ids_list), batch_size)
            ]
            y_true_list = [
                np.stack(y_true_list[i:i+batch_size])
                for i in range(0, len(y_true_list), batch_size)
            ]
 
        print(f"[==] Starting comprehensive training for {epochs} epochs "
              f"with mode: {mode}, learning rate: {lr}, schedule: {schedule}")
 
        for epoch in range(epochs):
            epoch_losses, epoch_accs = [], []
            current_lr   = self.learning_rate_warm_up(epoch, epochs, lr, schedule)
            self.alpha   = min(1.0, epoch / 100)   # alpha warmup: 0 → 1 over 100 epochs
 
            for input_ids, y_true in zip(input_ids_list, y_true_list):
                if input_ids.ndim == 1:
                    input_ids = input_ids[np.newaxis, :]
                if y_true.ndim == 1:
                    y_true = y_true[np.newaxis, :]
 
                loss, acc = self.train_step(
                    input_ids, epoch, y_true, current_lr, mode,
                    embedded=embedded, max_norm=max_norm, pad_token_id=pad_token_id
                )
                epoch_losses.append(loss)
                epoch_accs.append(acc)
 
            avg_loss = np.mean(epoch_losses)
            avg_acc  = np.mean(epoch_accs)
            losses.append(avg_loss)
            accs.append(avg_acc)
 
            if epoch % 10 == 0:
                print(f"[=] Epoch {epoch} | loss: {avg_loss:.4f} | Acc: {avg_acc:.2%}"
                      f" | lr: {current_lr:.6f} | alpha: {self.alpha:.3f}")
 
        return losses, accs
 
 
    def predict(self, input_ids, embedded=False):
        """
        Inference pass — dropout disabled, gradients not computed.
 
        input_ids cast to int32 defensively in case upstream processing
        returned float arrays (e.g. from np.stack on mixed-type lists).
        """
        if not embedded:
            input_ids = np.asarray(input_ids, dtype=np.int32)
            if input_ids.ndim == 1:
                input_ids = input_ids[np.newaxis, :]
 
        probs, attn_weights = self.forward(
            input_ids, embedded=embedded,
            training=False, attn_dropout=0.0, ffn_dropout=0.0
        )
        preds = np.argmax(probs, axis=1)
        return preds, probs, attn_weights
 
 
    # ─────────────────────────────────────────────
    # Alpha / complexity measurement
    # ─────────────────────────────────────────────
 
    def AME_Encoder(self, x):
        """
        Abstract Modelling Energy (AME): measures input complexity.
 
        Combines two signals:
        - X_mag: mean L2 norm of the input — how large the activations are
        - grad_energy: mean norm of the gradient along the last axis
                       — how rapidly the signal changes across the sequence
 
        AME = log(1 + X_mag) * log(1 + grad_energy)
 
        log1p keeps the value stable near zero and prevents explosion.
        The product means BOTH magnitude AND variation must be high for
        a large AME — prevents either signal alone from dominating.
 
        Used as the pre-sigmoid input for alpha: high AME → alpha closer to 1
        (complex input → allow more dynamic attention adaptation).
        """
        X          = np.asarray(x)
        gradient   = np.gradient(x, axis=-1)
        grad_energy = np.mean(np.linalg.norm(gradient, axis=-1))
        X_mag      = np.mean(np.linalg.norm(X, axis=-1))
        return np.log1p(X_mag) * np.log1p(grad_energy)
 
 
    def anisotropy_measurement(self, x):
        """
        Measures directional variation (anisotropy) in the input.
 
        Anisotropy = std(gradient_norms) / mean(gradient_norms)
 
        High anisotropy: gradient norms vary a lot across positions
                         → structured, position-dependent signal
        Low anisotropy:  gradient norms are uniform → flat or isotropic signal
 
        Used as a secondary signal in attention_quality_computing to weight
        the contribution of attention variance to the quality score.
        eps prevents division by zero when all gradient norms are identical.
        """
        eps = 1e-5
        try:
            gradient = np.gradient(x, axis=-1)
        except:
            subset   = x[:, 0, 0]
            gradient = np.gradient(subset)
 
        val        = [np.linalg.norm(v) for v in gradient]
        anisotropy = np.std(val) / np.mean(val) + eps
        return anisotropy
 
 
    def attention_quality_computing(self, attn_weights, mask=None):
        """
        Computes a quality score (0-1) for the current attention distribution.
        Used to refine alpha via EMA in forward(): alpha = 0.95*alpha + 0.05*quality
 
        Four components combined into a single scalar:
 
        1. norm_entropy: 1 - normalized_entropy
           High score when attention is sharp (concentrated on few tokens).
           Low score when attention is uniform (spread across all tokens).
           Uniform attention = model hasn't learned meaningful patterns yet.
 
        2. avg_max: mean of per-position attention max values
           High when each query position has one clearly dominant key.
           Complements entropy — both should be high for focused attention.
 
        3. anisotropy * norm_var: attention variance weighted by anisotropy
           Rewards attention patterns that vary meaningfully across positions.
           Prevents reward for random high-variance (noise) attention.
 
        4. AMR (Abstract Modelling Rate): sigmoid(AME) of the attention weights
           qualified = (1 - AMR) + eps * anisotropy
           Acts as a global scaling factor — when attention weights themselves
           show high complexity (high AME), the quality score is down-weighted,
           preventing reward for chaotic attention.
 
        Mask handling: padding positions are zeroed and rows renormalized
        before computing any statistics, so padding doesn't inflate entropy
        or distort the quality estimate.
        """
        eps = 1e-5
        batch, heads, seq_len, _ = attn_weights.shape
 
        if mask is not None:
            mask_expanded = np.broadcast_to(mask, (batch, heads, seq_len, seq_len))
            attn_weights  = attn_weights * mask_expanded
            row_sums      = attn_weights.sum(axis=-1, keepdims=True) + eps
            attn_weights  = attn_weights / row_sums   # renormalize over valid tokens
 
        AME        = self.AME_Encoder(attn_weights)
        anisotropy = self.anisotropy_measurement(attn_weights)
 
        entropy      = -np.sum(attn_weights * np.log(attn_weights + eps), axis=-1)
        max_entropy  = np.log(seq_len)
        norm_entropy = 1.0 - (np.mean(entropy) / max_entropy)
 
        max_attn = np.max(attn_weights, axis=-1)
        avg_max  = np.mean(max_attn)
 
        var_attn = np.var(attn_weights)
        norm_var = np.clip(var_attn * seq_len, 0, 1)
 
        AMR       = 1.0 / (1.0 + np.exp(-AME))
        qualified = (1.0 - AMR) + eps * anisotropy
 
        quality_score = qualified * norm_entropy + qualified * avg_max + anisotropy * norm_var
        return np.clip(quality_score, 0, 1.0)


