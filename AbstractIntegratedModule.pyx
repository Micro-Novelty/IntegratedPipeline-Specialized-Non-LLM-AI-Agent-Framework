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


class GeometricWeightShaping:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.floating_context = None

    def eigenvalue_encoder(self, x):
        eps = 1e-5
        X = np.asarray(x)
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)

        mag = np.mean(np.linalg.norm(X, axis=-1))

        anisotropy = self.anisotropy_measurement(X)

        structured_noise = np.random.uniform(0, mag, size=X.shape)
        X = np.vstack((X, structured_noise))
        cov = np.cov(X, rowvar=False)

        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]

        eigenvalues = eigenvalues[idx]
        energy = np.cumsum(eigenvalues) / np.sum(eigenvalues)
        k = np.searchsorted(energy, 0.90) + 1     

        K_G = 1.0 / (1.0 + k)
        mag_G = 1.0 / (1.0 + K_G)

        trA = k / (1.0 - anisotropy) + eps  
        trB = (1/2 + mag_G) / (1.0 + trA**2)
        trC = (1/6 + K_G) / (trB**2 - 1.0)
        return trC, k


    def spectral_signature(self, x, k=5):
        X = np.asarray(x)
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        else:
            X = X.reshape(X.shape[0], -1)
        cov = np.cov(X, rowvar=False)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.sort(eigvals)[::-1]
        return eigvals[:k] / (eigvals.sum() + 1e-8)

    def spectral_similarity(self, a, b):
        sa = self.spectral_signature(a)
        sb = self.spectral_signature(b)
        return np.exp(-np.linalg.norm(sa - sb))

    # abstract modelling error provides the model how to better process weights when the data complexity has little geometric complexity
    def AME_Encoder(self, x):
        X = np.asarray(x)
        try:
            gradient = np.gradient(x)
        except:
            subnet = x[:min(10, x.shape[0]), :min(10, x.shape[1])]
            gradient = np.gradient(subnet.flatten())

        grad_energy = np.mean(np.linalg.norm(gradient, axis=-1))       
        X_mag = np.mean(np.linalg.norm(X, axis=-1))

        AME =  np.log1p(X_mag) * np.log1p(grad_energy) 
        return AME

    # anisotropy provides the model the standard complexity of the data geometry, allowing it to know how complex the data needs to be processed.
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

    # weight shaping provides directional context in which how the data should be processed in order to align with the data geometry
    def abstract_weight_shaping(self, x):
        input_size = self.input_size
        output_size = self.output_size

        rng = np.random.default_rng()

        anisotropy = self.anisotropy_measurement(x)
        mag = np.mean(np.linalg.norm(x))

        trC, k = self.eigenvalue_encoder(x)
        AME = self.AME_Encoder(x)

        floating_point = np.random.uniform(0, trC, size=x.shape)
        spectral_similarity = self.spectral_similarity(x, floating_point)

        AEL = 0.3 + spectral_similarity * anisotropy       
        scaled_anisotropy = anisotropy / (anisotropy + 1.0)
        AMR = 1.0 / (1.0 + np.exp(-AME)) # abstract modelling rate

        efficient_distributed_energy = k + AEL * (1.0 - AMR)
        floating_context = rng.uniform(0, efficient_distributed_energy, size=(input_size, output_size)) 
        self.floating_context = floating_context

        return floating_context

    def weight_shaping(self, x, type=None):
        if isinstance(x, list):
            x = np.asarray(x)

        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)

        if np.std(x) == 0:
            x = np.random.uniform(0, 1, size=x.shape)

        floating_context = self.abstract_weight_shaping(x)
        if np.isnan(floating_context).any() or not np.isfinite(floating_context).any():
            floating_context = np.ones_like(x)

        return floating_context



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
class WeightedEnsemblePredictor:
    def __init__(self, pipeline, memory_name):
        self.pipeline = pipeline
        self.storage = ModelStorage(memory_name, db_path='activity_log.db')
        self.inference = AgentDistributedInference(pipeline, self.storage, memory_name, port=5001, ssl_cert_file=pipeline.ssl_cert_file, ssl_key_file=pipeline.ssl_key_file)
        self.query_node = QueryNode(pipeline, memory_name, self.storage)

        self.transformer_weight = 0.5  # Initial equal weight
        self.mlp_weight = 0.5 # initial equal mlp weight
        self.calibration_history = []
        self.explainer = ExplainabilityModule(pipeline, self)
        self.memory_name = memory_name
        self.db_path = 'activity_log.db'

        self.self_attn_weights = None

        if not self.storage.memory_exists(self.memory_name, type='Transformer'):
            self.memory = {}
        else:
            self.memory = self.storage.memory_retrieval(self.memory_name, type_func='Transformer', verbose=True)


    def attention_memory_gate(self, probs, x):
        memory = self.memory
        cache_attn_memory = [key for key, (_, inp, _, _, _) in memory.items() if key.startswith('TA') and self.pipeline.cosine_similarity(x, inp) >= 0.85]

        if cache_attn_memory:
            print('[+] Found matching attention memory!')
            for memo in cache_attn_memory:
                texts, _, x2, x3, x4 = memory[memo]

            return texts, x2, x3, x4

        else:
            print('🔄 NoMatching Attention Weights!')
            if self.self_attn_weights is not None:
                print('|| Using current attention weights because of no matches found.')
                attn_weights = self.self_attn_weights
                return None, None, None, attn_weights

            return None, None, None, None

           

    def modular_attention_saving(self, text, X, X2, X3, X4):
        memory_name = self.memory_name

        self.memory['TA'] = X, text, X2, X3, X4

        self.storage.save_model_dict(memory_name, self.memory, type='Transformer', model_type='attention')

        print('🚀 Memory Probability Added!')




    def explainability_prediction_batch(self, texts, mlp_probs, trans_probs, attn_weights, show_explanation=False):
        results = []
        for i, text in enumerate(texts):
            text_mlp_probs = mlp_probs[i] if i < len(mlp_probs) else mlp_probs
            text_trans_probs = trans_probs[i] if i < len(trans_probs) else trans_probs
            text_attn_weights = attn_weights[i] if i < len(attn_weights) else attn_weights

            result = self.predict_single(text, text_mlp_probs, text_trans_probs, text_attn_weights, show_explanation)
            results.append(result)
            explanation = result['explanation']
            print(f'[+] Explanation: {explanation}')
        
        return results
    

    def credibility_summarized_prediction(self, input_ids, mlp_probs, trans_probs, attn_weights, type=None):
        if type == 'Transformer':
            texts, mlp_probs, trans_probs, attn_weights = self.attention_memory_gate(input_ids)
            if not texts:
                texts = self.pipeline.texts
        else:
            texts = self.pipeline.texts

        results = self.explainability_prediction_batch(texts, mlp_probs, trans_probs, attn_weights)

        # Calculate summary
        predictions = [r['prediction'] for r in results]
        confidences = [r['confidence'] for r in results]
        
        from collections import Counter
        distribution = Counter(predictions)
        
        print("\n📊 Batch Summary:")
        print(f"   Total: {len(results)} predictions")
        print(f"   Avg Confidence: {np.mean(confidences):.1%}")
        print(f"   Distribution: {dict(distribution)}")
        
        self.modular_attention_saving(input_ids, texts, mlp_probs, trans_probs, attn_weights )


    def explain_past_memory(self, probs, input_ids):
        _, mlp_probs, trans_probs, attn_weights = self.attention_memory_gate(probs, input_ids)
        self.self_attn_weights = attn_weights

        self_attn_weights = self.self_attn_weights

        if trans_probs is not None:
            print('[+] Attention memory retrieved! ')
            method = 'memory_retrieval'

            self.credibility_summarized_prediction(input_ids, mlp_probs, trans_probs, attn_weights, type='pipeline')
            ensemble_probs = self._dynamic_weighted_ensemble(
                trans_probs, mlp_probs, attn_weights, input_ids
            )

            return ensemble_probs
        else:
            print("[-] Ambiguity present, Requesting peer assistance... ")
        
            try:
                probs = self.inference._handle_peer_agent_request(probs, self_attn_weights, input_ids, type='DevicePeer', agreement=False)
                return probs
            except Exception as e:
                print(f'|| Error initiating peer request: {e}, returning regular probs.')
                self.inference.report_failure(id(self), 'processing', reason=f'{e}')                        

                return probs


    def predict_single(self, text, mlp_probs, trans_probs, attn_weights, show_explanation=True):
        result, explanation = self.explainer._get_prediction_details(text, mlp_probs, trans_probs, attn_weights)
        return {
            'prediction': result['final_label'],
            'confidence': result['final_confidence'],
            'explanation': explanation,
            'details': result
        }


    def explainability_prediction_batch(self, texts, mlp_probs, trans_probs, attn_weights, show_explanation=False):
        results = []
        for text in texts:
            result = self.predict_single(text, mlp_probs, trans_probs, attn_weights, show_explanation)
            results.append(result)
            explanation = result['explanation']
            print(f'[+] Explanation: {explanation}')
        
        return results
    

     

    def predict_ensemble(self, input_ids, X_mlp, y_true, method='dynamic', embedded=False):
        trans_probs, attn_weights = self.pipeline.model2.forward(input_ids, embedded=embedded)
        mlp_probs = self.pipeline.mlp.forward(X_mlp)
        established_agreement = self.query_node._establish_node_connection("PredictEnsemble")
        
        if method == 'equal':
            # Simple average
            ensemble_probs = (trans_probs + mlp_probs) / 2
            
        elif method == 'confidence':
            # Weight by confidence (max probability)
            trans_conf = np.max(trans_probs, axis=1, keepdims=True)
            mlp_conf = np.max(mlp_probs, axis=1, keepdims=True)
            
            # Normalize weights
            total_conf = trans_conf + mlp_conf + 1e-8
            trans_weight = trans_conf / total_conf
            mlp_weight = mlp_conf / total_conf
            
            ensemble_probs = trans_weight * trans_probs + mlp_weight * mlp_probs
            
        elif method == 'dynamic':
            # Dynamic weighting based on agreement and attention
            ensemble_probs = self._dynamic_weighted_ensemble(
                trans_probs, mlp_probs, attn_weights, input_ids
            )
            
        elif method == 'attention':
            # Use attention to weight transformer vs MLP
            ensemble_probs = self._attention_weighted_ensemble(
                trans_probs, mlp_probs, attn_weights
            )
            
        elif method == 'meta':
            # Meta-learner that decides weights
            ensemble_probs = self._meta_ensemble(
                trans_probs, mlp_probs, attn_weights, X_mlp
            )
        
        elif method == 'calibration':
            calibrated_weight = self.calibrate_weights(input_ids, X_mlp, y_true, step=3)  
            ensemble_probs = self._attention_weighted_ensemble(
                trans_probs, mlp_probs, calibrated_weight
            )
                                    
        else:
            print(f"Unknown method: {method}")

        
        if established_agreement and self.pipeline.show_explainability_details:
            print('[✅] Agreement established, generating explainability features.')
            try:
                print('=== COMPLETE EXPLAINABILITY PREDICTION ==') 
                self.credibility_summarized_prediction(input_ids, mlp_probs, trans_probs, attn_weights, type='pipeline')
            except Exception as e:
                print(f'[-] Cant get explainability features! : {e}')
        else:
            print('[-] No agreement established, skipping explainability features.')


        try:
            ensemble_probs = ensemble_probs / ensemble_probs.sum(axis=1, keepdims=True)
        except:
            ensemble_probs = ensemble_probs / ensemble_probs.sum()

        if ensemble_probs is None or np.isnan(ensemble_probs).any() or np.isinf(ensemble_probs).any():
            print('[-] Ensemble probs is invalid , using MLP probs as.')
            return mlp_probs, {
            'transformer': trans_probs,
            'mlp': mlp_probs,
            'ensemble': mlp_probs,
            'method': None              
            }   

        return ensemble_probs, {
            'transformer': trans_probs,
            'mlp': mlp_probs,
            'ensemble': ensemble_probs,
            'method': method
        }
        
    def anisotropy_measurement(self, x):
        eps = 1e-5
        gradient = np.gradient(x)

        val = [np.linalg.norm(v) for v in gradient]
        anisotropy = np.std(val) / np.mean(val) + eps
        return anisotropy

    def _dynamic_weighted_ensemble(self, trans_probs, mlp_probs, attn_weights, input_ids):
        batch_size = trans_probs.shape[0]
        try:
            n_trans_classes = trans_probs.shape[1]
            n_mlp_classes = mlp_probs.shape[1]
        except:
            n_trans_classes = trans_probs.shape[-1]
            n_mlp_classes = mlp_probs.shape[-1]        

        n_classes = max(n_trans_classes, n_mlp_classes)
        
        print(f"🔄 Aligning classes: {n_trans_classes} and {n_mlp_classes} → {n_classes}")
        ensemble = np.zeros((batch_size, n_classes))
        for i in range(batch_size):
            trans_row = np.zeros(n_classes)
            mlp_row = np.zeros(n_classes)
            
            trans_row[:n_trans_classes] = trans_probs[i]
            mlp_row[:n_mlp_classes] = mlp_probs[i]
            
            trans_row = trans_row / (trans_row.sum() + 1e-8)
            mlp_row = mlp_row / (mlp_row.sum() + 1e-8)
            
            trans_pred = np.argmax(trans_probs[i])
            mlp_pred = np.argmax(mlp_probs[i])
            agreement = 1.0 if trans_pred == mlp_pred else 0.3

            if attn_weights is not None and i < len(attn_weights):
                print('🔄 Sophisticated confidence assembling')
                attn = attn_weights[i]
                anisotropy = self.anisotropy_measurement(attn) 

                attn_focus = np.std(attn) if attn.size > 0 else 0.5
                attn_growth = 1.0 / (1.0 + np.exp(-attn_focus))
                attn_limit = (1.0 - attn_focus + attn_growth) * anisotropy

                trans_conf_factor = attn_growth + attn_limit * attn_focus 
            else:
                attn_growth = 1.0 / (1.0 + np.exp(-attn_weights))
                anisotropy = self.anisotropy_measurement(attn_weights)
                trans_conf_factor = attn_growth * anisotropy

            mlp_entropy = -np.sum(mlp_probs[i] * np.log(mlp_probs[i] + 1e-8))
            mlp_conf_factor = 1.0 / (1.0 + mlp_entropy)  # Lower entropy = higher confidence
            
            trans_weight = trans_conf_factor * (1.0 + agreement) / 2
            mlp_weight = mlp_conf_factor * (1.0 + agreement) / 2
            
            # Normalizing
            total = trans_weight + mlp_weight + 1e-8
            trans_weight /= total
            mlp_weight /= total
                    
            ensemble[i] = trans_weight * trans_row + mlp_weight * mlp_row
    
        return ensemble
    
    def _attention_weighted_ensemble(self, trans_probs, mlp_probs, attn_weights):

        if attn_weights is None:
            return (trans_probs + mlp_probs) / 2
        
        batch_size = trans_probs.shape[0]
        ensemble = np.zeros_like(trans_probs)

        n_trans_classes = trans_probs.shape[1]        
        n_mlp_classes = mlp_probs.shape[1]
        n_classes = max(n_trans_classes, n_mlp_classes)

        for i in range(batch_size):
            trans_row = np.zeros(n_classes)
            mlp_row = np.zeros(n_classes)
            
            trans_row[:n_trans_classes] = trans_probs[i]
            mlp_row[:n_mlp_classes] = mlp_probs[i]
            
            trans_row = trans_row / (trans_row.sum() + 1e-8)
            mlp_row = mlp_row / (mlp_row.sum() + 1e-8)

            if i < len(attn_weights):
                attn = attn_weights[i]
                anisotropy = self.anisotropy_measurement(attn)                
                # Attention entropy: lower entropy = more focused = trust transformer more
                if attn.size > 0:
                    attn_flat = attn.flatten()
                    attn_entropy = -np.sum(attn_flat * np.log(attn_flat + 1e-8)) / np.log(len(attn_flat))
                    trans_trust = 1.0 - attn_entropy  # 0 to 1
                else:
                    attn_focus = 1.0 / (1.0 + np.exp(-attn))
                    trans_trust = attn_focus * anisotropy
            else:
                attn_focus = 1.0 / (1.0 + np.exp(-attn))
                attn_limit = 1.0 - np.exp(-attn_focus)
                trans_trust = attn_limit * (1.0 - anisotropy)
            
            # MLP gets the rest
            mlp_trust = 1.0 - trans_trust

            try:
                ensemble[i] = trans_trust * trans_row + mlp_trust * mlp_row
            except:
                ensemble = trans_trust * trans_row + mlp_trust * mlp_row
        
        return ensemble
    
    def _meta_ensemble(self, trans_probs, mlp_probs, attn_weights, X_mlp):
        batch_size = trans_probs.shape[0]
        n_classes = trans_probs.shape[1]

        n_trans_classes = trans_probs.shape[1]        
        n_mlp_classes = mlp_probs.shape[1]
        n_classes = max(n_trans_classes, n_mlp_classes)

        # Create meta features
        meta_features = []
        for i in range(batch_size):
            trans_row = np.zeros(n_classes)
            mlp_row = np.zeros(n_classes)
            
            trans_row[:n_trans_classes] = trans_probs[i]
            mlp_row[:n_mlp_classes] = mlp_probs[i]
            
            trans_row = trans_row / (trans_row.sum() + 1e-8)
            mlp_row = mlp_row / (mlp_row.sum() + 1e-8)

            features = [
                np.max(trans_row),           # Transformer confidence
                np.max(mlp_row),              # MLP confidence
                np.std(trans_row),             # Transformer spread
                np.std(mlp_row),               # MLP spread
                1.0 if np.argmax(trans_row) == np.argmax(mlp_row) else 0.0,  # Agreement
            ]
            
            # Add attention stats if available
            if attn_weights is not None and i < len(attn_weights):
                attn = attn_weights[i]
                if attn.size > 0:
                    features.append(np.std(attn))
                    features.append(np.max(attn))
                else:
                    features.extend([0.5, 0.5])
            else:
                features.extend([0.5, 0.5])
            
            meta_features.append(features)
        
        meta_features = np.array(meta_features)  
        ensemble = np.zeros_like(trans_probs)
        
        for i in range(batch_size):
            # Calculate weight based on meta features
            trans_conf = meta_features[i, 0]
            mlp_conf = meta_features[i, 1]
            agreement = meta_features[i, 4]
            
            # Boost weight when models agree
            base_weight = 0.5 + 0.3 * agreement
            
            # Adjust based on relative confidence
            if trans_conf > mlp_conf:
                trans_weight = base_weight
                mlp_weight = 1.0 - base_weight
            else:
                trans_weight = 1.0 - base_weight
                mlp_weight = base_weight

            try:
                ensemble[i] = trans_weight * trans_row + mlp_weight * mlp_row
            except:
                ensemble = trans_weight * trans_row + mlp_weight * mlp_row                
        
        return ensemble
    
    def calibrate_weights(self, input_ids, X_mlp, y_true, step=3):
        print("\n🔧 Calibrating ensemble weights...")
        
        best_weight = 0.5
        best_accuracy = 0
        
        # Try different weights
        for w in np.linspace(0, 1, 11):
            self.transformer_weight = w
            self.mlp_weight = 1 - w
            
            correct = 0
            total = 0
            for i in range(step):
                trans_probs, _ = self.pipeline.model2.forward(input_ids)
                mlp_probs = self.pipeline.mlp.forward(X_mlp)
                
                ensemble = w * trans_probs + (1-w) * mlp_probs
                preds = np.argmax(ensemble, axis=1)
                true = np.argmax(y_true, axis=1)
                
                correct += np.sum(preds == true)
                total += len(preds)
            
            accuracy = correct / total
            print(f" || Weight: {w:.1f}: Accuracy: {accuracy:.2%}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_weight = w

        self.transformer_weight = best_weight
        self.mlp_weight = 1.0 - best_weight

        print(f"\n✅ Optimal weights: Transformer: {best_weight:.2f}, MLP={1-best_weight:.2f}")
        print(f"[-] Validation accuracy: {best_accuracy:.2%}")
        
        return best_weight

# Cross-session automation module that allows exporting and importing of sessions, syncing with another device, and listing available sessions for better management and continuity of work across different environments.
class CrossSessionAutomation:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def export_session(self, session_name=None):
        if session_name is None:
            session_name = f"session_{self.session_id}"
        
        session_data = {
            'session_id': self.session_id,
            'session_name': session_name,
            'timestamp': datetime.now().isoformat(),
            'memories': self.pipeline.memory.copy(),
        }
        
        filename = f"{session_name}.json"
        with open(filename, 'w') as f:
            json.dump(session_data, f, default=str)
        
        print(f"💾 Session exported to: {filename}")
        return filename
    
    def import_session(self, filename):
        with open(filename, 'r') as f:
            session_data = json.load(f)
        
        print(f"\n📥 Importing session: {session_data['session_name']}")
        print(f"   Created: {session_data['timestamp']}")
        print(f"   Memories: {len(session_data['memories'])}")
        
        # Merge memories
        for key, value in session_data['memories'].items():
            if key not in self.pipeline.memory:
                self.pipeline.memory[key] = value
        
        print(f"✅ Session imported! Total memories: {len(self.pipeline.memory)}")
    
    def sync_with_another_device(self, device_ip, port=5000):
        import socket
        import pickle
        
        # Export current session
        temp_file = self.export_session(f"sync_{self.session_id}")
        
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((device_ip, port))
                with open(temp_file, 'rb') as f:
                    s.sendall(f.read())
                print(f"📡 Synced to {device_ip} || {port}")
                print('🚀 Succesfully sync and export memory session to another device! ')                 
        except Exception as e:
            print(f"❌ Sync failed: {e}")
            pass
        

    
    def list_sessions(self, name):
        import glob

        sessions = glob.glob(f"{name}*.json")
        
        print(f"\n📚 Available Sessions: {sessions}")
        if sessions:
            for session in sessions:
                with open(session, 'r') as f:
                    data = json.load(f)
                    print(f"   • {session}: {data['session_name']} ({len(data['memories'])} memories)")

        else:          
            print('[-] No available sessions! ')
        
        return sessions


# Explainability module that provides detailed explanations for predictions, allows learning from user feedback, and maintains a history of decisions for transparency and continuous improvement of the model.
class ExplainabilityModule:
    def __init__(self, pipeline, predictor):
        self.pipeline = pipeline
        self.decision_history = []     

        self.decision_history = []     
        
        self.uncertainty_threshold = 0.2
        self.pending_queries = []
        self.learned_from_feedback = []   
        self.feedback_buffer = []  # Store feedback for batch training
        self.buffer_size = 10  # Train after every 10 feedbacks

        self.auto_ask = True


    def data_preparation(self, titles, labels):
        datasets = []
        raw = []
        for title in titles:
            tupled_title = (str(title))
            datasets.append(tupled_title)
            raw.append(str(title))

        for label in labels:
            tupled_label = (str(label))
            datasets.append(tupled_label)
            raw.append(str(label))

        self.pipeline.initialize_fitting(raw)
        X_raw = self.pipeline.tfidf.transform(raw).toarray()

        return datasets, X_raw

    
    def draw_bar(self, value, max_width=20):
        value = max(0, min(1, value))  # Ensure value is between 0 and 1
        filled = int(value * max_width)
        return '█' * filled + '░' * (max_width - filled)

    def _learn_from_feedback(self, text, correct_label, wrong_result):
        eps = 1e-5
        print(f"\n[📚] Learning: '{text}' → {correct_label}...")
        
        # 1. Convert to features
        X_intents = self.pipeline.tfidf.transform(self.pipeline.intents).toarray()
        X_input = self.pipeline.tfidf.transform([text]).toarray()
        X_raw = np.dot(X_intents, X_input.T).T

        if np.allclose(X_raw, 0.0) or self.pipeline.anisotropy_measurement(X_raw) < 0.3 or np.isnotfinite(X_raw).any():
            checksum = int(hashlib.md5(text.encode()).hexdigest(), 16) % 1000 / 10000
            X_raw[0, 0] = checksum + eps
            
        X = X_raw.copy()
    
        try:
            print('[🔄] Verifying if similar correct_label is already in pipeline supervised memory...')

            memory_key = f'supervised_memory'
            retrieved = [key for key, (corr_label) in self.pipeline.memory.items() if key.startswith(memory_key) and correct_label == corr_label]
            if retrieved:
                for retrieve in retrieved:
                    _, correct_label = self.pipeline.memory[retrieve]

                print(f'[✅] retrieved similar correct label: {correct_label}')
                print(f'[=] This proves consistency over time necessary for gradual supervised learning loop to provide transparency')
            else:
                print('[-] No similar matching correct label')
                print('[=] This suggests that the model has never learned this previous input in supervised learning')

        except Exception as e:
            print(f'[=] Cant save to and retrieve memory due to {e} error.')

        # 2. Convert label to one-hot
        label_idx = self.pipeline.intents[correct_label] if correct_label in self.pipeline.intents else 0
        if label_idx is None:
            label_idx = len(self.pipeline.intents)
            self.pipeline.intents[correct_label] = label_idx
        
        y_onehot = np.zeros((1, len(self.pipeline.intents)))
        y_onehot[0, label_idx] = 1
        if y_onehot.shape[1] != X.shape[1]:
            if y_onehot.shape[1] < X.shape[1]:
                y_onehot = np.pad(y_onehot, ((0, 0), (0, X.shape[1] - y_onehot.shape[1])), mode='constant')
            else:
                X = np.pad(X, ((0, 0), (0, y_onehot.shape[1] - X.shape[1])), mode='constant')
        
        # 3. IMMEDIATE TRAINING (single step with higher Learning Rate)
        anisotropy = self.pipeline.anisotropy_measurement(X)

        anisotropy_dist = 1.0 / (1.0 + np.exp(-anisotropy))
        deviation = 1.0 / (1.0 + np.std(X))
        AEL = (1.0 - deviation) * anisotropy_dist + eps

        old_lr = self.pipeline.mlp.lr
        self.pipeline.mlp.lr = 2 / (1.0 + AEL) # use stable learning rate that match the environment complexity for correction
        print(f"[=] Training MLP on corrected example with boosted LR: {self.pipeline.mlp.lr}...")

        # Train on this single example for a few epochs
        self.pipeline.focused_mlp.train(X, y_onehot, epochs=1000, lr=self.pipeline.mlp.lr, verbose=True)
        time.sleep(5)

        self.pipeline.mlp.lr = old_lr  # Restore LR
        
        # 4. train transformer for efficient processing later tho.
        if self.pipeline.model2:
            input_ids = np.array([self.pipeline.encode(text, self.pipeline.vocab)])
            for _ in range(5):
                self.pipeline.model2.train_step(input_ids, 0, y_onehot, lr=0.1, mode='fixed_backward')
        
        # 5. Store in memory gate for fast retrieval
        self.pipeline.modular_prediction_saving(
            self.pipeline.encode(text, self.pipeline.vocab),
            X,
            correct_label
        )
        
        # 6. Add to buffer for batch consolidation later.
        self.feedback_buffer.append((X, y_onehot, text, correct_label))
        
        # 7. Batch train when buffer is full
        if len(self.feedback_buffer) >= self.buffer_size:
            print(f"\n[🔄] Buffer full with {len(self.feedback_buffer)} feedback examples. Starting batch training...")
            self._batch_train_from_feedback()
        
        print(f"[✅] Learned: '{text}' → {correct_label} (model weights updated)")

        supervised_memory = {
            'input': text,
            'label': correct_label,
            'original_prediction': wrong_result['final_label'], 
            'original_confidence': wrong_result['final_confidence'],
            'timestamp': datetime.now(),
            'learned': True

        }

        self.learned_from_feedback.append(supervised_memory)
        if hasattr(self.pipeline, 'memory'):
            print('[🔄] Applying correct label to pipelines memory')
            memory_key = f'supervised_memory'
            self.pipeline.memory[memory_key] = (X, correct_label)

        if len(self.learned_from_feedback) % 10 == 0:
            self.consolidate_supervised_memories()

        return X
    
    def _batch_train_from_feedback(self):
        if not self.feedback_buffer:
            return
        
        print(f"\n🔄 Batch training on {len(self.feedback_buffer)} feedback examples...")
        
        # Collect all feedback
        # Determine max dimensions

        max_x_dim = max(fb[0].shape[1] for fb in self.feedback_buffer)
        current_y_dim = len(self.pipeline.intents)
        
        # Collect all feedback with padding
        X_list = []
        y_list = []
        for fb in self.feedback_buffer:
            X = fb[0]
            y = fb[1]
            if X.shape[1] < max_x_dim:
                X = np.pad(X, ((0, 0), (0, max_x_dim - X.shape[1])), mode='constant')
            if y.shape[1] < current_y_dim:
                y = np.pad(y, ((0, 0), (0, current_y_dim - y.shape[1])), mode='constant')
            X_list.append(X)
            y_list.append(y)
        
        X_batch = np.vstack(X_list)
        y_batch = np.vstack(y_list)

        # Train MLP on batch
        old_lr = self.pipeline.mlp.lr
        self.pipeline.mlp.lr = old_lr * 2
        for epoch in range(20):
            y_pred = self.pipeline.mlp.forward(X_batch)
            loss = Loss.categorical_crossentropy(y_batch, y_pred)
            grad = Loss.softmax_crossentropy_derivative(y_batch, y_pred)
            self.pipeline.mlp.backward(grad, self.pipeline.mlp.lr)

        self.pipeline.mlp.lr = old_lr
        
        # Clear buffer
        self.feedback_buffer = []
        print("[✅] Batch training complete")

    
    def _ask_for_feedback(self, text, result, explanation):
        print("\n" + "="*60)
        print(f"[🤔] I'm confused about this detail: '{text}'")
        print(f"[=] I thought: {result['final_label']} ({result['final_confidence']:.1%})")
        if 'final_label' in result and 'final_confidence' in result:    
            confidence = result['final_confidence']
            bar = self.draw_bar(confidence)   
            print(f"[+] Confidence: [{bar}] {confidence:.1%}")  

        # Show top 3 predictions
        if 'details' in result and 'all_probs' in result['details']:
            probs = result['details']['all_probs']
            top3 = np.argsort(probs)[-3:][::-1]            
            print("\n[+] Top possibilities:")
            for idx in top3:
                label = self.pipeline.intents.get(idx, f"class_{idx}")
                print(f"[=]  • {label}: {probs[idx]:.1%}")
        
        print("\n[📚] Options:")
        print("  1. Enter correct label")
        print("  2. Skip")
        print("  3. Show explanation")
        print('  4. Get decision history')
        
        choice = input("\n [=] What is the correct label? (ex: break/work): ").strip()
        
        if choice.lower() == 'skip':
            return None
        elif choice.lower() == 'explain':
            print(explanation)
            return self._ask_for_feedback(text, result, explanation)
        elif choice == '4':
            history = self.get_decision_history(limit=10)  
            for entry in history:
                print(f" [?] {entry['input']} → {entry['label']}")
        elif choice:
            print('[==] Assigning Training for correct label: ')
            return choice
        return None

    def analyze_with_feedback(self, details, input_text, mlp_probs, trans_probs, attn_weights, explanation, auto_ask=True):
        uncertain = self.pipeline.confidence_threshold

        input_ids = np.array([self.pipeline.encode(input_text, self.pipeline.vocab)])
        if isinstance(input_ids, list):
            input_ids = np.array(input_ids)

        if uncertain == 0.0:
            uncertain = self.uncertainty_threshold

        is_uncertain = details['final_confidence'] < uncertain
        
        if is_uncertain and self.auto_ask:
            feedback = self._ask_for_feedback(input_text, details, explanation)
            if feedback:
                print(f"[📚] Received feedback: '{input_text}' should be '{feedback}'")
                print('[=] Supervised learning took many trials to get right. This is normal. Please be patient as the model updates continously each label request...')

                evaluated_input = self._learn_from_feedback(input_text, feedback, details)
                self.auto_ask = False  # Prevent infinite loop
                return False
        
        return False
    
    def consolidate_supervised_memories(self):
        if not self.learned_from_feedback:
            return
        
        print(f"\n🔄 Consolidating {len(self.learned_from_feedback)} supervised memories...")
        
        # Extract all supervised examples
        texts = [m['input'] for m in self.learned_from_feedback]
        labels = [m['label'] for m in self.learned_from_feedback]

        dataset, _ = self.data_preparation(texts, labels)

        self.initialize_fitting(labels)            
        X = self.tfidf.transform(labels).toarray()   

        self.pipeline.transformer_utilities(dataset, X)
        
        print("✅ Supervised memories consolidated!")
    
    def get_uncertain_predictions(self, result):
        uncertain = []
        if result['final_confidence'] < self.uncertainty_threshold:
            uncertain.append({
                    'text': result['input_text'],
                    'prediction': result['final_label'],
                    'confidence': result['final_confidence'],
                    'attention_quality': result['attention_quality']
            })
        
        # Sort by most uncertain first
        uncertain.sort(key=lambda x: x['confidence'])
        
        print(f"\n🔍 Found {len(uncertain)} uncertain predictions:")
        for u in uncertain[:10]:
            print(f"   • '{u['text']}' → {u['prediction']} ({u['confidence']:.1%})")
        
        return uncertain
 

    def _get_prediction_details(self, input_text, mlp_probs, trans_probs, attn_weights):
        show_details = self.pipeline.show_explainability_details
        if trans_probs.ndim == 1:
            trans_probs = trans_probs.reshape(1, -1)
        trans_pred = np.argmax(trans_probs[0])
        trans_conf = trans_probs[0][trans_pred]
        
        # Handle mlp_probs - ensure it's 2D
        try:
            if mlp_probs.ndim == 1:
                mlp_probs = mlp_probs.reshape(1, -1)
            mlp_pred = np.argmax(mlp_probs)
            mlp_conf = mlp_probs[mlp_pred]
        except:
            if isinstance(mlp_probs, float):
                mlp_pred = int(mlp_probs)
                mlp_conf = 0.15
            else:
                mlp_pred = np.argmax(mlp_probs[0])
                mlp_conf = mlp_probs[0][mlp_pred]

        if isinstance(mlp_conf, np.ndarray):
            mlp_conf = np.clip(np.mean(mlp_conf), 0, 1)
        if isinstance(trans_conf, np.ndarray):
            trans_conf = np.clip(np.mean(trans_conf), 0, 1)

        reverse_map = self.pipeline.reverse_map
        
        final_pred, final_conf = self._get_final_output(
            mlp_pred, mlp_conf, trans_pred, trans_conf, attn_weights
        )
        
        # Extract attention focus
        focus_words = self._get_attention_focus(attn_weights, input_text)
        
        # Extract geometric features
        geometric_features = self._get_geometric_features(input_text)

        details = {
            'input_text': input_text,
            'final_label': reverse_map.get(final_pred, f"class_{final_pred}"),
            'final_confidence': final_conf,
            'final_class': final_pred,
            'mlp': {
                'label': reverse_map.get(mlp_pred, f"class_{mlp_pred}"),
                'confidence': mlp_conf,
                'class': mlp_pred
            },
            'transformer': {
                'label': reverse_map.get(trans_pred, f"class_{trans_pred}"),
                'confidence': trans_conf,
                'class': trans_pred,
                'attention_words': focus_words,
                'attention_weights': attn_weights
            },
            'geometric_features': geometric_features,
            'agreement': mlp_pred == trans_pred,
            'anisotropy': self._compute_anisotropy(attn_weights) if attn_weights is not None else None,
            'attention_quality': self._compute_attention_quality(attn_weights) if attn_weights is not None else None
        }

        explanation, confidence, comparison = self._generate_explanation(details)
        
        self.decision_history.append({
            'timestamp': datetime.now(),
            'input': input_text,
            'prediction': details['final_label'],
            'confidence': details['final_confidence'],
            'explanation': explanation,
            'details': details
        })
        
        if show_details:
            self._display_explanation(explanation)
            self._display_explanation(confidence)
            self._display_explanation(comparison)
            self.get_uncertain_predictions(details)
      
            if details['final_confidence'] < 0.15 and not self.pipeline.autonomous:
                self.analyze_with_feedback(details, input_text, mlp_probs, trans_probs, attn_weights, explanation)
 
        confidence = self.explain_confidence(details)
        if final_conf:
            print('[||] Final confidence set to: ', final_conf)
            self.pipeline.final_conf_score = final_conf

        return details, explanation
    
    
    def _get_final_output(self, mlp_pred, mlp_conf, trans_pred, trans_conf, attn_weights):
        eps = 1e-5
        if isinstance(mlp_conf, np.ndarray):
            mlp_conf = np.clip(np.mean(mlp_conf), 0, 1)
        if isinstance(trans_conf, np.ndarray):
            mlp_conf = np.clip(np.mean(mlp_conf), 0, 1)

        if mlp_pred == trans_pred:
            final_pred = mlp_pred
            final_conf = max(mlp_conf, trans_conf)
        else:
            sliced_anisotropy = self.pipeline.anisotropy_measurement(attn_weights[0]) 
            deviation = 1.0 / (1.0 + np.std(attn_weights))
            attn_quality = self._compute_attention_quality(attn_weights)

            # abstract attention transformation
            AAT = deviation * (1.0 - sliced_anisotropy) + eps

            if mlp_conf > trans_conf:
                final_pred = mlp_pred
                final_conf = mlp_conf * (1.0 - trans_conf) * (1.0 - AAT) + eps
            else:
                final_pred = trans_pred
                final_conf = trans_conf * (1.0 - mlp_conf) * AAT + eps

        return final_pred, final_conf
    
    def _get_attention_focus(self, attn_weights, text):
        if attn_weights is None or len(attn_weights) == 0:
            return text.split()[:3]
        
        words = text.lower().split()
        attn = attn_weights[0].mean(axis=0) if len(attn_weights[0].shape) > 1 else attn_weights[0]
        top_indices = np.argsort(attn)[-3:][::-1]
        if attn.ndim > 1:
            attn = attn.flatten()

        top_indices = np.argsort(attn)[-3:][::-1]
        
        focus_words = []
        for idx in top_indices:
            if hasattr(idx, 'item'):
                idx = idx.item()
            
            if isinstance(idx, (int, np.integer)) and idx >= 0 and idx < len(words):
                focus_words.append(words[idx])
        
        return focus_words if focus_words else words[:3]
    
    def _get_geometric_features(self, text):
        X_tfidf = self.pipeline.tfidf.transform([text]).toarray()
        
        if hasattr(self.pipeline, 'geometric_shaping'):
            anisotropy = self.pipeline.model2.geometric_shaping.anisotropy_measurement(X_tfidf)
            ame = self.pipeline.model2.geometric_shaping.AME_Encoder(X_tfidf)
        else:
            # Fallback: compute simple statistics
            anisotropy = np.std(X_tfidf) / (np.mean(X_tfidf) + 1e-8)
            ame = self.AME_Encoder(X_tfidf)
     
        # Extract dominant features
        feature_names = self.pipeline.tfidf.get_feature_names_out()
        non_zero = X_tfidf[0] > 0
        dominant_features = [feature_names[i] for i in np.where(non_zero)[0][:3]]
        
        return {
            'anisotropy': float(anisotropy),
            'AME': float(ame),
            'dominant_features': dominant_features,
            'feature_energy': float(np.sum(X_tfidf ** 2))
        }
    
    def AME_Encoder(self, x):
        X = np.asarray(x)

        try:
            gradient = np.gradient(x, axis=-1)
        except:
            subset = x[:]
            gradient = np.gradient(subset)

        grad_energy = np.mean(np.linalg.norm(gradient, axis=-1))       
        X_mag = np.mean(np.linalg.norm(X, axis=-1))
        AME = np.log1p(X_mag) * np.log1p(grad_energy) 

        return AME

    def _compute_anisotropy(self, attn_weights):
        if attn_weights is None or len(attn_weights) == 0:
            return 0.5
        
        try:
    
            if hasattr(self.pipeline, 'anisotropy_measurement'):
                return self.pipeline.anisotropy_measurement(attn_weights.flatten())
            
            # Fallback calculation
            attn_flat = attn_weights.flatten()
            gradient = np.gradient(attn_flat)
            val = [np.linalg.norm(v) for v in gradient]
            return np.std(val) / (np.mean(val) + 1e-8)

        except:
            return 0.5
    
    def _compute_attention_quality(self, attn_weights):
        eps = 1e-5
        if attn_weights is None or len(attn_weights) == 0:
            return 0.5
        
        try:

            if hasattr(self.pipeline.model2, 'attention_quality_computing'):
                return self.pipeline.model2.attention_quality_computing(attn_weights)
            
            # Fallback calculation
            eps = 1e-5
            batch, heads, seq_len, _ = attn_weights.shape
            
            entropy = -np.sum(attn_weights * np.log(attn_weights + eps), axis=-1)
            max_entropy = np.log(seq_len)
            norm_entropy = 1.0 - (np.mean(entropy) / max_entropy)
            
            max_attn = np.max(attn_weights, axis=-1)
            avg_max = np.mean(max_attn)
            
            var_attn = np.var(attn_weights)
            norm_var = np.clip(var_attn * seq_len, 0, 1)
            
            AME = self.AME_Encoder(attn_weights)
            AMR = 1.0 / (1.0 + np.exp(-AME) + eps)

            quality = norm_entropy * (1.0 - AMR) + avg_max * AMR + norm_var * AMR
            return np.clip(quality, 0, 1)
        except:
            print("[-] Error occurred while computing attention quality.")
            AMR = 0.1
            if attn_weights is not None:
                print(f"[-] Attention weights shape: {attn_weights.shape}")
                AME = self.AME_Encoder(attn_weights)
                AMR = 1.0 / (1.0 + np.exp(-AME) + eps)            
            return AMR
    
    def _generate_explanation(self, details):
        parts = []
        
        # 1. Final decision
        parts.append(f"📌 Decision: I think my prediction is: **{details['final_label']}**")
        parts.append(f"   (Confidence Degree: {details['final_confidence']})\n")
        
        # 2. MLP's geometric reasoning
        parts.append("🧠 Geometric MLP Reasoning:")
        parts.append(f"   • Detected Detail: {', '.join(details['geometric_features']['dominant_features'][:3])}")
        parts.append(f"   • Geometric complexity signature: {details['geometric_features']['anisotropy']:.3f}")
        parts.append(f"   • Energy: signature {details['geometric_features']['feature_energy']:.3f}")
        parts.append(f"   • Confidence Focus: {details['mlp']['confidence']:.1%} to → {details['mlp']['label']}")
        
        # 3. Transformer's contextual reasoning
        parts.append("\n🌀 Transformer Reasoning:")
        if details['transformer']['attention_words']:
            parts.append(f"   • Focused on: '{', '.join(details['transformer']['attention_words'])}'")
        parts.append(f"   • Attention quality: {details.get('attention_quality', 0.5)}")
        parts.append(f"   • Attention anisotropy: {details.get('anisotropy', 0.5):.3f}")
        parts.append(f"   • Confidence Focus: {details['transformer']['confidence']:.1%} to → {details['transformer']['label']}")
        
        # 4. Agreement analysis
        if details['agreement']:
            parts.append("\n✅ Models Agreed:")
            parts.append("   Both geometric and contextual analysis point to the same conclusion")
        else:
            parts.append("\n⚠️ Models Disagreed:")
            parts.append(f"   Geometric MLP Focusing on → {details['mlp']['label']} detail")
            parts.append(f"   Transformer Focusing on → {details['transformer']['label']} detail")
            parts.append(f"   I weighted them with {details['final_confidence']:.1%} confident in {details['final_label']}")
        
        # 5. Uncertainty assessment
        if details['final_confidence'] < 0.6:
            parts.append("\n🤔 Uncertainty Note:")
            parts.append(f"   I'm not very confident about this prediction || Confidence: {details['final_confidence']}")
            parts.append("   • This pattern is unusual in my training data")
            parts.append("   • More same examples would help me learn enough pattern")
        
        # 6. Geometric signature 
        parts.append("\n🔬 Geometric Signature:")
        parts.append(f"   • AME Signature: {details['geometric_features']['AME']:.4f}")
        parts.append(f"   • Anisotropy Signature: {details['geometric_features']['anisotropy']:.4f}")

        confidence = self.explain_confidence(details) 
        comparison = self.compare_decisions() 

        return '\n'.join(parts), confidence, comparison
    
    def _display_explanation(self, explanation):
        print("\n" + "="*80)
        print("🤖 AI EXPLANATION")
        print("="*80)
        print(explanation)
        print("="*80)
        pass
    
    def explain_decision(self, idx=-1):
        if abs(idx) <= len(self.decision_history):
            return self.decision_history[idx]['explanation']
        return "Decision not found"
    
    def compare_decisions(self, idx1=-1, idx2=-2):
        if len(self.decision_history) < 2:
            return "Need at least two decisions to compare"
        
        comparison = []
        d1 = self.decision_history[idx1]
        d2 = self.decision_history[idx2]
        
        comparison.append(f"🔄 Decision Comparison")
        comparison.append("====================================")
        
        comparison.append("[<] Earlier Decision:")
        comparison.append(f"[+] Input: {d1['input']}")
        comparison.append(f"[+] Detail Focus: {d1['prediction']} ({d1['confidence']:.1%})")
        
        comparison.append("🧠 Later Decision:")
        comparison.append(f"[=] Input: {d2['input']}")
        comparison.append(f"[=] Detail Focus: {d2['prediction']} ({d2['confidence']:.1%})")
        
        comparison.append("🔬 Learning Progress: ")
        comparison.append(f"• Confidence {'increased' if d2['confidence'] > d1['confidence'] else 'decreased'} from {d1['confidence']} to {d2['confidence']}")
        comparison.append(f"• The model is becoming {'more' if d2['confidence'] > d1['confidence'] else 'less'} certain")
        
        return '\n'.join(comparison)
    
    def explain_confidence(self, details):

        factors = []
        
        # Check MLP confidence
        if details['mlp']['confidence'] > 0.8:
            factors.append(f"✅ MLP is very confident ({details['mlp']['confidence']:.1%}) due to strong geometric patterns")
        elif details['mlp']['confidence'] < 0.5:
            factors.append(f"🤔 MLP is uncertain ({details['mlp']['confidence']:.1%}) due to ambiguous geometric patterns")
        
        # Check transformer confidence
        if details['transformer']['confidence'] > 0.8:
            factors.append(f"✅ Transformer is confident ({details['transformer']['confidence']}) with focused attention")
        elif details['transformer']['confidence'] < 0.5:
            factors.append(f"🤔 Transformer is uncertain ({details['transformer']['confidence']}) due to scattered attention")
        
        # Check agreement
        if details['agreement']:
            factors.append("[✅] Both Models agree, reinforcing confidence")
        else:
            factors.append("[⚠️] Both Models disagree, reducing overall confidence")
        
        # Attention quality
        if details.get('attention_quality', 0) > 0.7:
            factors.append(f"[✅] High attention quality ({details['attention_quality']}) indicates clear consistent patterns!")
        elif details.get('attention_quality', 0) < 0.3:
            factors.append(f"[-] Low Attention Quality! : ({details['attention_quality']}) Indicates noisy unnecessary patterns on seen data!")
        
        return '\n'.join(factors)
    
    def get_decision_history(self, limit=10):
        history = []
        for i, dec in enumerate(self.decision_history[-limit:]):
            history.append({
                'id': i,
                'timestamp': dec['timestamp'],
                'input': dec['input'],
                'prediction': dec['prediction'],
                'confidence': dec['confidence']
            })

            print('=== DECISION HISTORY REPORT ===')
            print(f'[=] ID: {history['id']}')
            print(f'[=] Timestamp: {history['timestamp']}')
            print(f'[=] Processed Input: {history['input']}')
            print(f'[=] Prediction: {history['prediction']}')
            print(f'[=] Confidence: {history['confidence']}')

        return history

# Model storage module that handles saving and loading of trained models, their versions, and associated metadata to a database for persistence and future retrieval.
class ModelStorage:
    def __init__(self, memory_name, db_path='activity_log.db'):
        self.db_path = db_path

        self.setup_storage_table()
        self.setup_explainable_table()
        self.setup_agent_table()
        self.setup_node_table()

        self.memory_name = memory_name

        if not self.memory_exists(self.memory_name, type='Peer'):
            self.id_history = []
        else:
            print(f'|| Found Matched ID from memory: {self.memory_name}!')
            self.id_history = self.load_agent_id(self.memory_name)

    def get_database_path(self):
        db_filename= self.db_path
        if getattr(sys, 'frozen', False):
            application_path = sys._MEIPASS
            print(f"Running as EXE, temp path: {application_path}")
        else:
            application_path = os.path.dirname(os.path.abspath(__file__))
            print(f"Running as script, path: {application_path}")
    
        db_path = os.path.join(application_path, db_filename)
        print(f"Looking for database at: {db_path}")
        print(f"Database exists: {os.path.exists(db_path)}")
    
        return db_path

    def setup_explainable_table(self):
        try:
            try:
                db_path = self.get_database_path()            
                conn = sqlite3.connect(db_path)
            except:
                conn = sqlite3.connect(self.db_path)

            c = conn.cursor()
    
            c.execute('''CREATE TABLE IF NOT EXISTS model_attn_storage
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      memory_name TEXT,
                      model_type TEXT,
                      model_data TEXT,
                      is_active INTEGER DEFAULT 0,                      
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        
            conn.commit()
            conn.close()
            print('|| Update Attention Saved to database! ')

        except Exception as e:
            print(f'|| Cant Update Database: {e}')
            filepath = input('|| Insert Database filepath: ')
            if filepath:
                conn = sqlite3.connect(filepath)
            else:
                print('|| Skipping Database Modification...')
                pass
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS model_attn_storage
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      memory_name TEXT,
                      model_type TEXT,
                      model_data TEXT,
                      is_active INTEGER DEFAULT 0,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        
            conn.commit()
            conn.close()



    def setup_storage_table(self):
        try:
            try:
                db_path = self.get_database_path()            
                conn = sqlite3.connect(db_path)
            except:
                conn = sqlite3.connect(self.db_path)

            c = conn.cursor()
    
            c.execute('''CREATE TABLE IF NOT EXISTS model_storage
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      memory_name TEXT,
                      model_version TEXT,
                      model_type TEXT,
                      model_data TEXT,  -- JSON string for dict
                      model_binary BLOB,  -- For pickle files
                      trained_on TEXT,
                      metadata TEXT,  -- JSON for extra info
                      is_active INTEGER DEFAULT 0,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        
            conn.commit()
            conn.close()
            print('|| Update Saved to database! ')

        except Exception as e:
            print(f'|| Cant Update Database: {e}')
            filepath = input('|| Insert Database filepath: ')
            if filepath:
                conn = sqlite3.connect(filepath)
            else:
                print('|| Skipping Database Modification...')
                pass
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS model_storage
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      memory_name TEXT,
                      model_version TEXT,
                      model_type TEXT,
                      model_data TEXT,  -- JSON string for dict
                      model_binary BLOB,  -- For pickle files
                      trained_on TEXT,
                      metadata TEXT,  -- JSON for extra info
                      is_active INTEGER DEFAULT 0,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        
            conn.commit()
            conn.close()

    def get_database_path(self):
        db_filename= self.db_path
        if getattr(sys, 'frozen', False):
            application_path = sys._MEIPASS
            print(f"Running as EXE, temp path: {application_path}")
        else:
            application_path = os.path.dirname(os.path.abspath(__file__))
            print(f"Running as script, path: {application_path}")
    
        db_path = os.path.join(application_path, db_filename)
        print(f"Looking for database at: {db_path}")
        print(f"Database exists: {os.path.exists(db_path)}")
    
        return db_path

    def setup_explainable_table(self):
        try:
            try:
                db_path = self.get_database_path()            
                conn = sqlite3.connect(db_path)
            except:
                conn = sqlite3.connect(self.db_path)

            c = conn.cursor()
    
            c.execute('''CREATE TABLE IF NOT EXISTS model_attn_storage
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      memory_name TEXT,
                      model_type TEXT,
                      model_data TEXT,
                      is_active INTEGER DEFAULT 0,                      
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        
            conn.commit()
            conn.close()
            print('|| Update Attention Saved to database! ')

        except Exception as e:
            print(f'|| Cant Update Database: {e}')
            filepath = input('|| Insert Database filepath: ')
            if filepath:
                conn = sqlite3.connect(filepath)
            else:
                print('|| Skipping Database Modification...')
                pass
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS model_attn_storage
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      memory_name TEXT,
                      model_type TEXT,
                      model_data TEXT,
                      is_active INTEGER DEFAULT 0,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        
            conn.commit()
            conn.close()

    def setup_node_table(self):
        try:
            try:
                db_path = self.get_database_path()            
                conn = sqlite3.connect(db_path)
            except:
                conn = sqlite3.connect(self.db_path)

            c = conn.cursor()
    
            c.execute('''CREATE TABLE IF NOT EXISTS node_storage
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      memory_name TEXT,
                      model_type TEXT,
                      node_data TEXT,
                      node_id TEXT,
                      is_active INTEGER DEFAULT 0,                      
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        
            conn.commit()
            conn.close()
            print('|| Update Node Saved to database! ')

        except Exception as e:
            print(f'|| Cant Update Database: {e}')
            filepath = input('|| Insert Database filepath: ')
            if filepath:
                conn = sqlite3.connect(filepath)
            else:
                print('|| Skipping Database Modification...')
                pass
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS node_storage
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      memory_name TEXT,
                      model_type TEXT,
                      node_data TEXT,
                      node_id TEXT,
                      is_active INTEGER DEFAULT 0,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        
            conn.commit()
            conn.close()


    def setup_agent_table(self):
        try:
            try:
                db_path = self.get_database_path()            
                conn = sqlite3.connect(db_path)
            except:
                conn = sqlite3.connect(self.db_path)

            c = conn.cursor()
    
            c.execute('''CREATE TABLE IF NOT EXISTS agent_attn_storage
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      memory_name TEXT,
                      model_type TEXT,
                      model_attn_data TEXT,
                      model_target_pred TEXT,
                      agent_id TEXT,
                      is_active INTEGER DEFAULT 0,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''') 
        
            conn.commit()
            conn.close()
            print('|| Update Agent Saved to database! ')

        except Exception as e:
            print(f'|| Cant Update Database: {e}')
            filepath = input('|| Insert Database filepath: ')
            if filepath:
                conn = sqlite3.connect(filepath)
            else:
                print('|| Skipping Database Modification...')
                pass
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS agent_attn_storage
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      memory_name TEXT,
                      model_type TEXT,
                      model_attn_data TEXT,
                      model_target_pred TEXT,
                      agent_id TEXT,
                      is_active INTEGER DEFAULT 0,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        
        
            conn.commit()
            conn.close()


    def save_model_dict(self, memory_name, model_dict, type=None, model_type='mlp'):
        try:
            db_path = self.get_database_path()            
            conn = sqlite3.connect(db_path)
        except:
            conn = sqlite3.connect(self.db_path)
                    
        c = conn.cursor()

        model_json = json.dumps(model_dict, default=str)
        if type == 'Transformer':
            try:
                c.execute("""
                    INSERT INTO model_attn_storage 
                    (memory_name, model_type, model_data, is_active)
                    VALUES (?, ?, ?, ?)
                """, (memory_name, model_type, model_json, 1))
        
                c.execute("""
                    UPDATE model_attn_storage 
                    SET is_active = 0 
                    WHERE memory_name = ? AND id != last_insert_rowid()
                """, (memory_name,))

            except Exception as e:
                print(f'[-] Cant save model memory due to: {e}') 
                pass             
        else:
            try:
                c.execute("""
                    INSERT INTO model_storage
                    (memory_name, model_type, model_data, is_active)
                    VALUES (?, ?, ?, ?)
                """, (memory_name, model_type, model_json, 1))
        
                c.execute("""
                    UPDATE model_storage 
                    SET is_active = 0 
                    WHERE memory_name = ? AND id != last_insert_rowid()
                """, (memory_name,)) 

            except Exception as e:
                print(f'[-] Cant save model memory due to: {e}') 
                pass          
        
        conn.commit()
        model_id = c.lastrowid        
        conn.close()
        
        print(f"✅ Memory '{memory_name}' saved as dict (ID: {model_id})")
        return model_id


    def _convert_to_arrays(self, data):
    
        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                result[key] = self._convert_value(value)
            return result
        elif isinstance(data, list):
            return [self._convert_value(item) for item in data]
        else:
            return self._convert_value(data)
    
    def _convert_value(self, value):
        if isinstance(value, np.ndarray):
            return value
        
        if isinstance(value, list):
            return [self._convert_value(item) for item in value]
        
        # Handle nested dicts
        if isinstance(value, dict):
            return self._convert_to_arrays(value)
        
        if isinstance(value, str):
            return self._parse_array_string(value)
        
        return value
    
    def _parse_array_string(self, s):
        if not isinstance(s, str):
            return s
        
        # Clean the string
        s = s.replace('\n', '').replace('\r', '').replace('\t', '')
        s = ' '.join(s.split())
        s = s.strip()
        
        if not s:
            return s
        
        no_brackets = s.replace('[', ' ').replace(']', ' ')
        parts = no_brackets.split()
        
        if parts:
            try:
                # Try to convert all parts to float
                float_values = [float(x) for x in parts]
                return np.array(float_values)
            except ValueError:
                pass
        
        if ',' in s:
            try:
                # Clean the string for JSON
                cleaned = s.replace('[', '[').replace(']', ']')
                parsed = json.loads(cleaned)
                return np.array(parsed)
            except:
                try:
                    parsed = ast.literal_eval(s)
                    return np.array(parsed)
                except:
                    pass
        
        if s.startswith('[[') and s.endswith(']]'):
            inner = s[2:-2]
            inner_parts = inner.split()
            try:
                return np.array([float(x) for x in inner_parts])
            except:
                pass
        
        if s.startswith('[') and s.endswith(']'):
            inner = s[1:-1]
            if ',' in inner:
                try:
                    return np.array([float(x.strip()) for x in inner.split(',') if x.strip()])
                except:
                    pass

        if ('[' in s or ']' in s) and '...' in s:
            s = re.sub(r'\s*\.\.\.\s*', ' ', s)

            s = re.sub(r'\[\s*\.\.\.\s*\]', '[]', s)
            s = re.sub(r'\[\s*\.\.\.', '[', s)
            s = re.sub(r'\.\.\.\s*\]', ']', s)

            s = s.replace('[',' ').replace(']',' ')
            parts = s.split()
            if parts:
                try:
                    return np.array([float(x) for x in parts])
                except:
                    print(f'|| Cant parse: {s}')
                    return s

        print(f"Warning: Could not parse: {s[:500]}...")
        return s
    
    def _convertables_utility(self, memory_name, data, data2, type_func=None, verbose=False):
        name = memory_name        
        if type_func == "TwoPass" and data2 is not None:
            print('|| Two pass utility converting.')
            result = self._convert_to_arrays(data)
            result2 = self._convert_to_arrays(data2)
        else:
            result = self._convert_to_arrays(data)

        
        if verbose and result is not None and data2 is None:
            print(f"Retrieved memory: {name}")
            for key, value in result.items():
                if isinstance(value, list):
                    print(f"  {key}: list of {len(value)} items")
                    for i, v in enumerate(value):
                        if isinstance(v, np.ndarray):
                            print(f"    [{i}]: array shape {v.shape}")
                        else:
                            print(f"    [{i}]: {type(v)}")
                elif isinstance(value, np.ndarray):
                    print(f"  {key}: array shape {value.shape}")
                else:
                    print(f"  data: {key}: {type(value)}")
                    
            return result

        elif verbose and data2 is not None:
            print(f"Retrieved memory: {name}")
            for key, value in result.items():
                if isinstance(value, list):
                    print(f"  {key}: list of {len(value)} items")
                    for i, v in enumerate(value):
                        if isinstance(v, np.ndarray):
                            print(f"    [{i}]: array shape {v.shape}")
                        else:
                            print(f"    [{i}]: {type(v)}")
                elif isinstance(value, np.ndarray):
                    print(f"  {key}: array shape {value.shape}")
                else:
                    print(f"  data: {key}: {type(value)}")
                    
            return result, result2 

        elif not verbose and data2 is not None:
            print("[-] Memory retrieved but not displayed (verbose=False) or error during processing")   
            return result, result2  

        else:
            print('|| Invalid memory type') 
            return None, None   

    def memory_retrieval(self, memory_name=None, type_func=None, verbose=False):  
        name = memory_name

        if type_func == 'Transformer':
            data = self.load_transformer_dict(name)
        elif type_func == 'Peer':
            id_history = self.id_history
          
            first_data, second_data = self.load_peer_request_dict(name, id_history) 
            result, result2 = self._convertables_utility(name, first_data, second_data, type_func='TwoPass', verbose=verbose)
            return result, result2
        elif type_func == 'Node':
            data = self._load_node_dict(name)            
        else:
            data = self.load_model_dict(name)
        
        if data is None:
            print(f"[-] No memory found: {name}")
            return {}

        result = self._convertables_utility(name, data, None, type_func=type_func, verbose=verbose)

        return result

    def _load_node_dict(self, memory_name):
        try:
            try:
                db_path = self.get_database_path()            
                conn = sqlite3.connect(db_path)
            except:
                conn = sqlite3.connect(self.db_path)

            c = conn.cursor()
        
            c.execute("""
            SELECT node_data FROM node_storage 
            WHERE memory_name = ? AND is_active = 1
            """, (memory_name,))
        
            result = c.fetchone()
            conn.close()
        
            if result:
                return json.loads(result[0])
        except Exception as e:
            print(f'Error handling node dict: {e}')
        return None

    def save_nodes_dict(self, memory_name, node_memory, node_id, model_type='Node'):
        try:
            db_path = self.get_database_path()
            conn = sqlite3.connect(db_path)
        except:
            conn = sqlite3.connect(self.db_path)

        c = conn.cursor()

        node_json = json.dumps(node_memory, default=str)

        try:
            c.execute("""
                INSERT INTO node_storage 
                (memory_name, model_type, node_data, node_id, is_active)
                VALUES (?, ?, ?, ?, ?)
            """, (memory_name, model_type, node_json, node_id, 1))
        
            c.execute("""
                UPDATE node_storage 
                SET is_active = 0 
                WHERE memory_name = ? AND id != last_insert_rowid()
            """, (memory_name,)) 

            conn.commit()
            conn.close()

            print('[||] Node data dictionary saved!')

        except Exception as e:
            print(f'[-] Cant save Node memory due to: {e}')
            pass        



    def load_transformer_dict(self, memory_name):
        try:
            try:
                conn = sqlite3.connect(self.db_path)
            except:
                db_path = self.get_database_path()
                conn = sqlite3.connect(db_path)   
                     
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
        
            c.execute("""
            SELECT model_data FROM model_attn_storage 
            WHERE memory_name = ? AND is_active = 1
            """, (memory_name,))
        
            result = c.fetchone()
            conn.close()
        
            if result:
                return json.loads(result[0])
        except Exception as e:
            print(f'Error handling attention dict: {e}')

        return None   

    def save_peer_needs_dict(self, memory_name, model_dict, target_pred, agent_id, model_type='Pipeline'):
        try:
            db_path = self.get_database_path()
            conn = sqlite3.connect(db_path)
        except:
            conn = sqlite3.connect(self.db_path)

        c = conn.cursor()

        model_json = json.dumps(model_dict, default=str)
        target_json = json.dumps(target_pred, default=str)
        agent_id_converted = json.dumps(agent_id, default=str)

        try:
            c.execute("""
                INSERT INTO agent_attn_storage 
                (memory_name, model_type, model_attn_data, model_target_pred, agent_id, is_active)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (memory_name, model_type, model_json, target_json, agent_id_converted, 1))
        
            c.execute("""
                UPDATE agent_attn_storage 
                SET is_active = 0 
                WHERE memory_name = ? AND id != last_insert_rowid()
            """, (memory_name,)) 

            conn.commit()
            conn.close()

            print('|| Peer data Needs dictionary saved!')

        except Exception as e:
            print(f'[-] Cant save model memory due to: {e}') 
            pass        

    def load_peer_request_dict(self, memory_name, agent_id):
        print(f'|| Peer request with Agent')
        try:
            try:
                db_path = self.get_database_path()
                conn = sqlite3.connect(db_path)   
            except:
                conn = sqlite3.connect(self.db_path)

            c = conn.cursor()
            placeholders = ",".join(["?"] * len(agent_id))

            query = f"""
            SELECT model_attn_data, model_target_pred FROM agent_attn_storage 
            WHERE memory_name = ? AND is_active = 1 AND agent_id NOT IN ({placeholders})
            """
            params = [memory_name] + agent_id
        
            c.execute(query, params)  
        
            result = c.fetchone()
            conn.close()
            print(f"|| Retrieved Peer Request memory: {memory_name} for agent_id: {agent_id}: result: {result}")
 
            if result:
                return json.loads(result[0]), json.loads(result[1])
            return None, None
        except Exception as e:
            print(f'|| Cant load peer request memory due to: {e}') 
            return None, None  


    def load_model_dict(self, memory_name):
        try:
            try:
                conn = sqlite3.connect(self.db_path)
            except:
               db_path = self.get_database_path()
               conn = sqlite3.connect(db_path)  
            c = conn.cursor()
        
            c.execute("""
            SELECT model_data FROM model_storage 
            WHERE memory_name = ? AND is_active = 1
            """, (memory_name,))
        
            result = c.fetchone()
            conn.close()
        
            if result:
                return json.loads(result[0])
        except Exception as e:
            print(f'Error handling model dict: {e}')
        return None
        
    def load_agent_id(self, memory_name):
        try:

            try:
               db_path = self.get_database_path()
               conn = sqlite3.connect(db_path)  
            except:
                conn = sqlite3.connect(self.db_path)

            c = conn.cursor()
        
            c.execute("""
            SELECT agent_id FROM agent_attn_storage 
            WHERE memory_name = ? AND is_active = 1
            """, (memory_name,))
        
            result = c.fetchone()
            conn.close()

            print(f'[+] Retrieved Agent ID of {memory_name}: result: {result}')
        
            if result:
                return json.loads(result[0])
        except Exception as e:
            print(f'[-] Error handling ID: {e}')

        return None        

    def memory_exists(self, memory_name, type=None):

        conn = None
        try:
            try:
                db_path = self.get_database_path()
                conn = sqlite3.connect(db_path)               
            except:
                conn = sqlite3.connect(self.db_path)
   

            if type == 'Transformer':
                c = conn.cursor()
        
                c.execute("""
                SELECT 1 FROM model_attn_storage 
                WHERE memory_name = ? AND is_active = 1
                LIMIT 1
                """, (memory_name,))
        
                result = c.fetchone()
                exists = result is not None
                print(f"|| Retrieved Attention: {memory_name}")

            elif type == 'Peer':
                c = conn.cursor()
        
                c.execute("""
                SELECT 1 FROM agent_attn_storage 
                WHERE memory_name = ? AND is_active = 1
                LIMIT 1
                """, (memory_name,))
        
                result = c.fetchone()
                exists = result is not None
                print(f"|| Retrieved Peer Memory: {memory_name}")

            else:
                c = conn.cursor()

                c.execute("""
                SELECT 1 FROM model_storage 
                WHERE memory_name = ? AND is_active = 1
                LIMIT 1
                """, (memory_name,))
        
                result = c.fetchone()
                exists = result is not None
                print(f"|| Retrieved Memory: {memory_name}")

            return exists
        
        except sqlite3.OperationalError as e:
            print(f"Database error: {e}")
            return False
            
        except Exception as e:
            print(f"Unexpected error: {e}") 
            return False
        finally:
            if conn:
                conn.close()


    def save_model_binary(self, model_object, memory_name, model_type='mlp'):
        try:
            try:
                conn = sqlite3.connect(self.db_path)
            except:
               db_path = self.get_database_path()
               conn = sqlite3.connect(db_path)          
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
        
            model_binary = joblib.dumps(model_object)
        
            c.execute("""
            INSERT INTO model_storage 
            (memory_name, model_type, model_binary, is_active)
            VALUES (?, ?, ?, ?)
            """, (memory_name, model_type, model_binary, 1))
        
            # Deactivate other versions
            c.execute("""
            UPDATE model_storage 
            SET is_active = 0 
            WHERE memory_name = ? AND id != last_insert_rowid()
            """, (memory_name,))
        
            conn.commit()
            model_id = c.lastrowid
            print(f"✅ Memory '{memory_name}' saved as binary (ID: {model_id})")
        except Exception as e:
            print(f'[-] Error handling: {e}')

        conn.close()

        return model_id
    
    def load_model_binary(self, memory_name):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute("""
            SELECT model_binary FROM model_storage 
            WHERE memory_name = ? AND is_active = 1
        """, (memory_name,))
        
        result = c.fetchone()
        conn.close()
        
        if result:
            return joblib.loads(result[0])
        return None
    
    def save_complete_pipeline(self, pipeline_name, pipeline_dict):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Convert entire pipeline to JSON (for dicts)
        pipeline_json = json.dumps(pipeline_dict, default=str)
        
        c.execute("""
            INSERT INTO model_storage 
            (pipeline_name, model_type, model_data, metadata, is_active)
            VALUES (?, ?, ?, ?, ?)
        """, (pipeline_name, 'pipeline', pipeline_json, 
               json.dumps({'components': list(pipeline_dict.keys())}), 1))
        
        conn.commit()
        model_id = c.lastrowid
        conn.close()
        
        print(f"✅ Integrated pipeline '{pipeline_name}' saved")
        return model_id

# Integrated inference module that allows multiple agents to connect and share their predictions, attention maps, and confidence scores for ensemble decision making.
# while also providing security features like authentication, rate limiting, and message validation.
class AgentDistributedInference:
    def __init__(self, pipeline, storage, memory_name, port=5555, ssl_cert_file=None, ssl_key_file=None):
        self.pipeline = pipeline
        self.memory_name = memory_name
        self.port = port
        self.storage = storage

        self.query_node = QueryNode(pipeline, memory_name, self.storage)        
        
        self.agent_comm_log = {}
        self.connections_log = {}
        self.connections = []  # List of connected sockets
        self.remote_agents = {}  # {agent_id: {'sock': sock, 'host': host, 'port': port, 'trust': 1.0}}
        
        self.running = False
        self.socket = None
        self.temporary_message = None
        self.temporary_agent_id = None        

        self.next_agent_id = 1
        self.connection_timeout = 30

        # for security purposes
        # Security: Authentication token
        self.auth_token = self._generate_auth_token()
        self.secret_key = self._generate_secret_key()

        # Security: Rate limiting
        self.max_connections_per_minute = 30
        self.connection_timestamps = deque(maxlen=30)
        self.max_requests_per_minute = 100
        self.request_timestamps = defaultdict(lambda: deque(maxlen=100))

        # Security: Message validation
        self.max_message_size = 10 * 1024 * 1024  # 10MB limit

        # Security: Trusted agents
        self.trusted_agents = {}

        # Security: Audit log
        self.security_log = []        

        self.enable_ssl = True  # Set to True to enable SSL encryption
        # i provided basic cert file and key since there are other layered security other than ssl, and also due to infrequent external connections.
        self.ssl_cert_file = ssl_cert_file
        self.ssl_key_file = ssl_key_file
        self.ssl_context = None

        if self.enable_ssl:
            self._setup_ssl()

        self.allowed_ips = set()  # Add trusted IPs
        self.blocked_ips = set()  # Block malicious IPs

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

    # ============ SECURITY FEATURES ============

    def _check_ip_access(self, ip):
        if ip in self.blocked_ips:
            return False
        if self.allowed_ips and ip not in self.allowed_ips:
            return False
        return True

    def add_allowed_ip(self, ip):
        self.allowed_ips.add(ip)
        self._log_security_event('ip_allowed', {'ip': ip})

    def remove_allowed_ip(self, ip):
        self.allowed_ips.discard(ip)
        self._log_security_event('ip_removed_from_allow', {'ip': ip})

    def add_blocked_ip(self, ip):
        self.blocked_ips.add(ip)
        self._log_security_event('ip_blocked', {'ip': ip})

    def remove_blocked_ip(self, ip):
        self.blocked_ips.discard(ip)
        self._log_security_event('ip_removed_from_block', {'ip': ip})

    def _setup_ssl(self):
        # Setup SSL context for encrypted connections
        try:
            self.ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            self.ssl_context.check_hostname = False
            self.ssl_context.verify_mode = ssl.CERT_NONE  # For self-signed certs
            
            if self.ssl_cert_file and self.ssl_key_file:
                self.ssl_context.load_cert_chain(self.ssl_cert_file, self.ssl_key_file)
            else:
                # Generate self-signed certificate for first layer security
                self._generate_self_signed_cert()
                
        except Exception as e:
            print(f"SSL setup failed: {e}")
            self.enable_ssl = False

    def _generate_self_signed_cert(self):
        # Generate self-signed certificate
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        import datetime
        
        # Generate private key
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        
        # Creatingcertificate
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, u"localhost"),
        ])
        
        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            issuer
        ).public_key(
            private_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.datetime.utcnow()
        ).not_valid_after(
            datetime.datetime.utcnow() + datetime.timedelta(days=365)
        ).add_extension(
            x509.SubjectAlternativeName([x509.DNSName(u"localhost")]),
            critical=False,
        ).sign(private_key, hashes.SHA256())
        
        # Saving certificate and key
        cert_pem = cert.public_bytes(serialization.Encoding.PEM)
        key_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        with open('server.crt', 'wb') as f:
            f.write(cert_pem)
        with open('server.key', 'wb') as f:
            f.write(key_pem)
            
        self.ssl_cert_file = 'server.crt'
        self.ssl_key_file = 'server.key'
        self.ssl_context.load_cert_chain(self.ssl_cert_file, self.ssl_key_file)


    def _generate_auth_token(self):
        return hashlib.sha256(os.urandom(32)).hexdigest()

    def _generate_secret_key(self):
        return hashlib.sha256(os.urandom(48)).hexdigest()

    def _log_security_event(self, event_type, details):
        self.security_log.append({
            'timestamp': datetime.now().isoformat(),
            'event': event_type,
            'details': details
        })
        if len(self.security_log) > 1000:
            self.security_log = self.security_log[-1000:]

    def _sanitize_input(self, text):
        if not isinstance(text, str):
            return str(text)
        sanitized = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
        return sanitized[:10000]

    def _check_rate_limit(self, agent_id=None):
        now = time.time()
        self.connection_timestamps.append(now)
        recent_connections = sum(1 for t in self.connection_timestamps if now - t < 60)
        if recent_connections > self.max_connections_per_minute:
            self._log_security_event('rate_limit_exceeded', {'type': 'connection', 'agent': agent_id})
            return False
        if agent_id:
            self.request_timestamps[agent_id].append(now)
            recent_requests = sum(1 for t in self.request_timestamps[agent_id] if now - t < 60)
            if recent_requests > self.max_requests_per_minute:
                self._log_security_event('rate_limit_exceeded', {'type': 'request', 'agent': agent_id})
                return False
        return True

    def _sign_message(self, message):
        # Create HMAC signature for message integrity        
        import hmac
        message['timestamp'] = time.time()
        message_bytes = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
        key = self.secret_key.encode() if isinstance(self.secret_key, str) else self.secret_key
        return hmac.new(key, message_bytes, hashlib.sha256).hexdigest()

    def _verify_signature(self, message, signature):
        #Verify HMAC signature 
        import hmac
        temp_msg = {k: v for k, v in message.items() if k != 'signature'}
        message_bytes = pickle.dumps(temp_msg, protocol=pickle.HIGHEST_PROTOCOL)
        key = self.secret_key.encode() if isinstance(self.secret_key, str) else self.secret_key
        expected = hmac.new(key, message_bytes, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, signature)

    def add_trusted_agent(self, agent_id, agent_token):
        self.trusted_agents[agent_id] = {'token': agent_token, 'added_at': datetime.now()}
        self._log_security_event('trusted_agent_added', {'agent_id': agent_id})

    def _authenticate_agent(self, token, agent_id):
        if agent_id in self.trusted_agents:
            return self.trusted_agents[agent_id]['token'] == token
        return token == self.auth_token 

    # ============ SERVER METHODS ============
    
    def start_server(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(('0.0.0.0', self.port))
        self.socket.listen(5)
        self.running = True

        if self.enable_ssl and self.ssl_context:
            self.socket = self.ssl_context.wrap_socket(self.socket, server_side=True)

        print(f"[🤖] Agent listening on port {self.port}")
        
        # Start accepting connections in background
        accept_thread = threading.Thread(target=self._accept_connections, daemon=True)
        accept_thread.start()
        
        return self.socket
    
    def _accept_connections(self):
        while self.running:
            try:
                client, addr = self.socket.accept()
                client.settimeout(self.connection_timeout)
                host = addr[0]
                if not self._check_ip_access(host):
                    print(f"[-] Connection attempt from blocked IP: {host}")
                    self._log_security_event('connection_blocked', {'ip': host})
                    client.close()
                    return

                print(f"📡 Connected to agent at {addr}")
                auth_msg = self._receive_message(client)
                if not auth_msg or not self._authenticate_agent(auth_msg.get('token', ''), f"{addr[0]}:{addr[1]}"):
                    print(f"[-] Authentication failed for agent with address: {addr}")
                    self._log_security_event('authentication_failed', {'agent': f"{addr[0]}:{addr[1]}"})
                    self.report_failure(id(self), 'authentication', reason=f'Failed authentication from {addr}')
                    client.close()
                    return

                # Send agent info to identify
                self._send_agent_info(client)
                
                # Start handler thread
                thread = threading.Thread(target=self._handle_client, args=(client, addr))
                thread.daemon = True
                thread.start()
                
            except Exception as e:
                if self.running:
                    print(f"Accept error: {e}")
                    self.inference.report_failure(id(self), 'processing', reason=f'{e}')
                                        
                break
    
    def _send_agent_info(self, client):
        info = {
            'type': self.MSG_TYPES['AGENT_INFO'],
            'agent_id': id(self),
            'agent_name': self.memory_name,
            'capabilities': ['prediction', 'memory_sync', 'ensemble']
        }
        self._send_message(client, info)
    
    def stop_server(self):
        self.running = False
        if self.socket:
            self.socket.close()
        
        # Close all connections
        for conn in self.connections:
            try:
                self._send_message(conn, {'type': self.MSG_TYPES['DISCONNECT']})
                conn.close()
            except:
                pass
        
        print("[🛑] Server stopped")
    
    # ============ CLIENT METHODS ============
    
    def connect_to_agent(self, host, port, agent_id=None):
        established_connection = self.query_node._establish_peer_nodes(self.temporary_agent_id)
        try:
            if not self._check_rate_limit(agent_id):
                print(f'[❌] Rate limit exceeded for agent {agent_id}, connection attempt reduced.')
                self._log_security_event('rate_limit_exceeded', {'type': 'connection_attempt', 'agent': agent_id})
                time.sleep(1)  # Sleep briefly to mitigate rapid retries
                return None

            client, addr = self.socket.accept()
            agent_id = f"{addr[0]}:{addr[1]}"

            if not self._check_ip_access(addr[0]):
                print(f"[-] Connection attempt from blocked IP: {addr[0]}")
                self._log_security_event('connection_blocked', {'ip': addr[0]})
                client.close()
                return None

            auth_msg = self._receive_message(client)
            if not auth_msg or not self._authenticate_agent(auth_msg.get('token', ''), agent_id):
                print(f"[-] Authentication failed for agent with address: {addr}")
                self._log_security_event('authentication_failed', {'agent': agent_id})
                self.report_failure(id(self), 'authentication', reason=f'Failed authentication from {addr}')
                client.close()
                return None

            if established_connection:
                print(f'[||] Connection established and permitted with peer agent: {self.temporary_agent_id}')
                if self.enable_ssl:
                    # Use SSL socket for client connections
                    context = ssl.create_default_context()
                    context.check_hostname = False
                    context.verify_mode = ssl.CERT_NONE
                    sock = context.wrap_socket(socket.socket(socket.AF_INET, socket.SOCK_STREAM))
                else:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    
                sock.connect((host, port))
                sock.settimeout(5.0)
            
                # Receive agent info
                info = self._receive_message(sock)
            
                if info and info.get('type') == self.MSG_TYPES['AGENT_INFO']:
                    remote_id = info.get('agent_id', f"{host}:{port}")
                    self.remote_agents[remote_id] = {
                    'sock': sock,
                    'host': host,
                    'port': port,
                    'trust': 1.0,
                    'last_seen': datetime.now(),
                    'failures': 0
                    }
                    self.connections.append(sock)
                    print(f"🔗 Connected to agent {remote_id} at {host}:{port}")
                    return sock
                else:
                    print(f"[❌] Invalid agent response from {host}:{port}")
                    sock.close()
                    return None
            else:
                print(f'[❌] Connection to peer agent {self.temporary_agent_id} denied by query node.')
                return None
                
        except Exception as e:
            print(f"[❌] Failed to connect to {host}:{port}: {e}")
            return None
    
    def disconnect_agent(self, agent_id):
        if agent_id in self.remote_agents:
            try:
                self._send_message(self.remote_agents[agent_id]['sock'], 
                                  {'type': self.MSG_TYPES['DISCONNECT']})
                self.remote_agents[agent_id]['sock'].close()
                del self.remote_agents[agent_id]
            except:
                pass
            print(f"🔌 Disconnected from agent {agent_id}")
    
    # ============ MESSAGE HANDLING ============
    
    def _send_message(self, sock, message):
        try:
            message['signature'] = self._sign_message(message)
            data = pickle.dumps(message)
            sock.send(len(data).to_bytes(4, 'big'))
            sock.send(data)
            return True
        except Exception as e:
            print(f"Send error: {e}")
            return False
    
    def _receive_message(self, sock):
        try:
            data_len = sock.recv(4)
            if not data_len:
                return None
            
            msg_len = int.from_bytes(data_len, 'big')
            if msg_len > self.max_message_size:
                self.log_security_event('message_too_large', {'size': msg_len})
                return None

            data = b''
            
            while len(data) < msg_len:
                chunk = sock.recv(min(4096, msg_len - len(data)))
                if not chunk:
                    return None
                data += chunk

            message = pickle.loads(data)
            if "signature" in message:
                if not self._verify_signature(message, message['signature']):
                    print(f"[-] Invalid message signature from agent {self.temporary_agent_id}")
                    self._log_security_event('invalid_signature', {'agent_id': self.temporary_agent_id})
                    return None

            return message

        except socket.timeout:
            return None
        except Exception as e:
            print(f"Receive error: {e}")
            return None
    
    def _handle_client(self, client, addr):
        agent_id = f"{addr[0]}:{addr[1]}"
        self.temporary_agent_id = agent_id
        
        while self.running:
            try:
                if not self._check_rate_limit(agent_id):
                    self._send_message(client, {'type': 'error', 'message': 'Rate limit exceeded'})
                    print(f'[❌] Rate limit exceeded for agent {agent_id}, request reduced.')
                    time.sleep(3)  # Sleep briefly to mitigate rapid retries
                    continue

                message = self._receive_message(client)
                self.temporary_message = message
                if message is None:
                    break
                
                response = self._process_message(message, agent_id)
                if response:
                    self._send_message(client, response)
                    
            except Exception as e:
                print(f"Handler error for {agent_id}: {e}")
                break
        
        # Cleanup on disconnect
        if agent_id in self.remote_agents:
            del self.remote_agents[agent_id]
        if client in self.connections:
            self.connections.remove(client)
        client.close()
        print(f"📡 Disconnected from {agent_id}")
    
    def _process_message(self, message, sender_id):
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

    def _handle_peer_agent_request(self, probs, self_attn_weights, input_ids, type=None, agreement=False):
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
                        probs = self.process_peer_request(target_preds, self_attn_weights, attn_weights)
    
                    except Exception as e:
                       print(f"[-] Error processing request: {e}, returning regular probs")

            except Exception as e:
                print(f'[-] Error handling request... {e}, returning regular probs')
                self.report_failure(id(self), 'processing', reason=f'{e}')                        

            print(f'[||] Successfully calibrate probs with previous Peer using database!')
            self.save_to_local_peer(self.memory_name, probs)
        else:
            print(f'[-] Connection to peer agent {self.temporary_agent_id} failed or not permitted, returning regular probs...')

        return probs


    def _calibrate_peer_probs(self, probs, target_preds, self_attn_weights, attn_weights, input_ids, AEL):
        calibrated = probs.copy()
        try:
            n_classes = probs.shape[1]
        except:
            n_classes = probs.shape[0]

        batch_size = len(target_preds)
        anisotropy = self.pipeline.anisotropy_measurement(attn_weights)    
        eps = 1e-5
  
        for i in range(batch_size):
            mlp_target = target_preds[i]
            attn_target = attn_weights[i]
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

                justified = (1.0 - AEL) + attn_quality * consensus
                boost = (1.0 - justified) * anisotropy + eps

            quality_temperature = (boost + 1.0 - AEL) + (1.0 - attn_quality) * anisotropy + eps
            self.query_node.peer_trust = quality_temperature + justified * anisotropy

            try:
                calibrated[i, mlp_target] = min(calibrated[i, mlp_target] * (1.5 * quality_temperature), 0.95)
            except:
                return calibrated

            calibrated[i] /= calibrated[i].sum()


        return calibrated        
            

    def handle_peer_uncertainty(self, probs, target_preds, self_attn_weights, attn_weights, input_ids):
        try:
            if self_attn_weights is None:
                _, _, self_attn_weights = self.pipeline.model2.predict(input_ids)                
            batch_similarity = self.pipeline.cosine_similarity(attn_weights, self_attn_weights)
        
            anisotropy = self.pipeline.anisotropy_measurement(attn_weights)
            AME = self.pipeline.AME_Encoder(attn_weights)
            AMR = 1.0 / (1.0 + np.exp(-AME))
            weighted_similarity = batch_similarity * (1.0 - AMR) * anisotropy

            if weighted_similarity > 0.75:
                return self.process_peer_request(probs, target_preds, attn_weights, input_ids)
            else:
                print('[-] Low uncertainty, normalizing with local agent data...')

                AEL = 0.3 + weighted_similarity * anisotropy
                calibrated = self._calibrate_peer_probs(probs, target_preds, self_attn_weights, attn_weights, input_ids, AEL)
                return calibrated

        except Exception as e:
            print(f"[-] Error in uncertainty handling: {e}")
            return probs


    def process_peer_request(self, probs, target_preds, attn_weights, input_ids):
        try:
            response_probs = self.pipeline.pipeline._calibrate_probs(probs, target_preds, attn_weights, input_ids)
            return response_probs
        except Exception as e:
            print(f"Error in peer request_processing: {e}")
            return None
            

    # ============ REQUEST HANDLERS ============
    def get_external_peer_message(self):
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
         
    
    def _handle_predict_request(self, message, sender_id):
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
            print(f"Prediction error: {e}")
            return {'type': self.MSG_TYPES['PREDICT_RESPONSE'], 'error': str(e)}
    

    def _handle_memory_sync_request(self, message, sender_id):
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
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {'type': self.MSG_TYPES['MEMORY_SYNC_RESPONSE'], 'error': str(e)}



    def _handle_ensemble_vote_request(self, message, sender_id):
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
        # Handle trust score update
        target_agent = message.get('target_agent')
        new_trust = message.get('trust_score')

        self.query_node.peer_trust = new_trust
        
        if target_agent in self.remote_agents:
            self.remote_agents[target_agent]['trust'] = new_trust
        
        return {'type': 'ack', 'status': 'trust_updated'}


    # ============ REQUEST SENDING METHODS ============       
    def request_prediction(self, agent_id, text, timeout=5):
        if agent_id not in self.remote_agents:
            print(f"Agent {agent_id} not connected")
            return None
        
        sock = self.remote_agents[agent_id]['sock']
        message = {
            'type': self.MSG_TYPES['PREDICT_REQUEST'],
            'text': text,
            'requester': id(self)
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
            print(f"Vote request failed: {e}")
            return None, None
    
    def sync_memory_with_agent(self, agent_id, memory_name, timeout=10):
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
        report = {
            'type': self.MSG_TYPES['FAILURE_REPORT'],
            'failed_agent': agent_id,
            'task_type': task_type,
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        }
        
        # Send to all other agents
        for other_id, agent_info in self.remote_agents.items():
            if other_id != agent_id:
                self._send_message(agent_info['sock'], report)
    
    def broadcast_ping(self):
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
                    del self.remote_agents[agent_id]
            except:
                del self.remote_agents[agent_id]
        
        return alive_agents
    
    # ============ LOCAL PEER (DATABASE) METHODS ============
    
    def sync_with_local_peer(self, memory_name):
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
        try:
            self.pipeline.storage.save_model_dict(memory_name, data)
            print(f"✅ Saved local peer presence: {memory_name}")
            return True
        except Exception as e:
            print(f"Save to local peer failed: {e}")
            return False
    
    # ============ UTILITY METHODS ============
    
    def _log_interaction(self, agent_id, interaction_type, confidence, details=None):
        if agent_id not in self.agent_comm_log:
            self.agent_comm_log[agent_id] = []
        
        self.agent_comm_log[agent_id].append({
            'timestamp': datetime.now(),
            'type': interaction_type,
            'confidence': confidence,
            'details': details
        })
    
    def get_agent_status(self):
        status = {}
        for agent_id, info in self.remote_agents.items():
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
        # Get communication log for an agent
        if agent_id:
            return self.agent_comm_log.get(agent_id, [])[-limit:]
        
        # Return all logs
        return self.agent_comm_log
    
    def print_network_status(self):
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

# The QueryNode class manages the connection and interaction with other nodes (agents) in the network. It handles node identification, agreement evaluation, safety checks, and maintains a memory of connected nodes. 
# The class allows for flexible interactions while ensuring the safety and integrity of the Master node.
class QueryNode:
    def __init__(self, pipeline, memory_name, storage):
        self.master_node = pipeline
        self.memory_name = memory_name
        self.storage = storage
        self.agreement = False

        if not self.storage.memory_exists(self.memory_name, type='Node'):
            print(f"|| Creating new memory for Nodes population: {memory_name}!")
            self.nodes = {}
        else:
            print(f'|| Found Matched Memory for Nodes : {memory_name}!')
            self.nodes = self.storage.memory_retrieval(self.memory_name, type_func='Node', verbose=True)

        self.master_nodes_id = 0
        self.safety_check_value = 0.0
        self.node_id = 0
        self.peer_trust = 1.0

        self.permission = False

    def _add_node(self, node):
        node_id = id(node)

        self.nodes[node_id] = node
        print(f"✅ Node {node_id} added to QueryNode")

        return node_id

    def _save_node_memory(self, node):
        try:
            node_id = id(node)
            self.master_node.storage.save_nodes_dict(self.memory_name, self.nodes, node_id, model_type='Node')

            print(f"[💾] Node {node_id} memory saved to storage!")
            return True
        except Exception as e:
            print(f"[-] Error saving node memory: {e}")
            return False


    def _evaluate_node_agreement(self, node):
        print(f"[=] Evaluating node {id(node)} || agreement: {self.agreement} || Master Node memory: {self.memory_name}")
        self.agreement_threshold = (self.master_node.confidence_threshold + self.master_node.final_conf_score * self.master_node.temperature)

        if self.agreement or self.agreement_threshold > self.master_node.confidence_threshold:
            print(f"[✅] Node {id(node)} is in agreement with the Master node")
            return True

        print(f"[-] Node {id(node)} is NOT in agreement with the Master node")
        return False


    def _connect_with_node(self, node):
        self.agreement = self.master_node.agreement

        if not self._identify_node(node):
            node_id = self._add_node(node)

        node_id = id(node)
        agreement = self._evaluate_node_agreement(node)
        safety = self._node_safety_check(node)

        # stable Node is established if either agreement is met or safety check is passed, allowing for some flexibility in interactions while still protecting the Master node from harmful interactions
        if safety or agreement:
            print(f"[🔗] Node {node_id} successfully connected to the Master node")
            self.permission = True
        else:
            print(f"[⚠️] Node {node_id} connection failed due to Disagreement")
            self.permission = False

        print('== Connection Evaluation Summary ==')
        print(f'[=] Node {node_id}')
        print(f'[=] agreement: {agreement}')
        print(f'[=] safety: {safety} || permission: {self.permission}')
 
        return self.permission

    def _connect_with_peer(self, node):
        self.agreement = self.master_node.agreement

        if not self._identify_node(node):
            node_id = self._add_node(node)

        node_id = id(node)
        agreement = self._evaluate_node_agreement(node)
        safety = self._node_safety_check(node)

        # peer agreement is optional to allow for more flexible interactions, but safety check is still enforced to protect the Master node from harmful interactions
        if safety or agreement:
            print(f"[🔗] Peer with ID: {node_id} successfully connected to the Master node")
            self.permission = True
        else:
            print(f"[⚠️] Peer with ID: {node_id} connection failed due to Disagreement")
            self.permission = False
 
        return self.permission        

    def _identify_node(self, node):
        eps = 1e-5
        print(f"[||] Identifying node {id(node)} with Master node memory: {self.memory_name}")
        identified_nodes = [(nid, n) for nid, n in self.nodes.items() if n == node]
        if identified_nodes:
            for node in identified_nodes:
                print(f"✅ Node {id(node)} is already identified with the Master node")
                self.safety_check_value = (self.master_node.final_conf_score + self.master_node.temperature) + eps
                return True
        else:
            print(f"[-] Node {id(node)} is NOT identified with the Master node")
            self.safety_check_value = (1.0 - self.master_node.final_conf_score + self.master_node.temperature) + eps
            return False


    def _node_safety_check(self, node):
        print(f"[🛡️] Performing safety check for node {id(node)} with safety value: {self.safety_check_value}")
        if self.safety_check_value > self.master_node.confidence_threshold:
            print(f"✅ Node {id(node)} passed the safety check")
            return True
        else:
            print(f"[-] Node {id(node)} failed the safety check")
            if self.safety_check_value < (self.master_node.confidence_threshold / 2):
                print(f"[⚠️] Node {id(node)} is considered useless and will be removed")
                removed = self._remove_node(node)
                return removed

            return False

    def _remove_node(self, node):
        node_id = id(node)
        if node in self.nodes:
            del self.nodes[node_id]
            print(f"[🗑️] Node {node_id} removed from Nodes population")
            return True
        else:
            print(f"[-] Node {node_id} not found in Nodes population")
            return False

    def _node_activation(self, Node):
        try:
            if self.permission:
                print(f"🚀 Node {id(Node)} is now active with the Master node")
                return True
            else:
                print(f"[-] Node {id(Node)} cannot be activated due to lack of permission")
                return False
        except Exception as e:
            print(f"[-] Error during node activation: {e}")
            return False

    def _identify_peer_trust(self, peer):
        print(f"[=] Identifying peer node {id(peer)} trustworthiness with Master node memory: {self.memory_name}")
        if self.peer_trust > self.master_node.confidence_threshold:
            print(f"[✅] Peer node {id(peer)} is identified as trustworthy with trust score: {self.peer_trust:.2f}")
            return True
        else:
            print(f"[-] Peer node {id(peer)} is NOT identified as trustworthy with trust score: {self.peer_trust:.2f}")
            return False

    def _establish_peer_nodes(self, peer):
        print(f"[=] Establishing peer node connection with peer with memory: {self.memory_name}")
        if self._connect_with_peer(peer) and self._identify_peer_trust(peer):
            print(f"[✅] Peer node {id(peer)} is now connected and can interact with the Master node")
        else:
            print(f"[-] Peer node {id(peer)} cannot interact with the Master node due to failed agreement")

        activation = self._node_activation(peer)
        saved = self._save_node_memory(peer)
        return activation

    def _establish_node_connection(self, node):
        if self._connect_with_node(node):
            print(f"[✅] Node {id(node)} is now connected and can interact with the Master node")
        else:
            print(f"[-] Node {id(node)} cannot interact with the Master node due to failed agreement")

        activation = self._node_activation(node)
        saved = self._save_node_memory(node)
        return activation




# The AutoBatcherAutomation class manages the batching of incoming prediction requests to optimize processing efficiency. 
# It collects requests over a short time window or until a maximum batch size is reached, then processes them together through the pipeline. This allows for improved throughput while still providing timely responses to individual requests.
class AutoBatcherAutomation:
    def __init__(self, pipeline, max_batch_size=32, max_wait_ms=50):
        self.pipeline = pipeline
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        
        self.request_queue = deque()
        self.processing = False
        self.results = {}
        self.next_id = 0
    
    def add_request(self, text, callback=None):
        request_id = self.next_id
        self.next_id += 1
        
        self.request_queue.append({
            'id': request_id,
            'text': text,
            'callback': callback,
            'timestamp': time.time()
        })
        
        if not self.processing:
            self._start_processing()
        
        return request_id
    
    def _start_processing(self):
        self.processing = True
        thread = threading.Thread(target=self._process_batches, daemon=True)
        thread.start()
    
    def _process_batches(self):
        while self.request_queue:
            # Wait for more requests or max wait time
            time.sleep(self.max_wait_ms / 1000)
            
            # Collect batch
            batch = []
            while self.request_queue and len(batch) < self.max_batch_size:
                batch.append(self.request_queue.popleft())
            
            if batch:
                self._process_batch(batch)
        
        self.processing = False
    
    def _process_batch(self, batch):
        texts = [req['text'] for req in batch]
        
        results = self.pipeline.predict_batch(texts)
        
        # Send results back
        for i, req in enumerate(batch):
            result = results[i] if i < len(results) else None
            if req['callback']:
                req['callback'](result)
            else:
                self.results[req['id']] = result
    
    def get_result(self, request_id, timeout=5):
        start = time.time()
        while request_id not in self.results:
            if time.time() - start > timeout:
                return None
            time.sleep(0.01)
        return self.results.pop(request_id)


# The IntegratedPipeline class serves as the central component that integrates all the different modules and functionalities of the system. 
# It manages the overall workflow, including data processing, model training, prediction, memory management, and interactions with other agents.
class IntegratedPipeline:
    def __init__(self, memory_name, ssl_cert_file=None, ssl_key_file=None):
        self.ssl_cert_file = ssl_cert_file
        self.ssl_key_file = ssl_key_file

        self.tfidf = TfidfVectorizer(max_features=70)
        self.ensemble = WeightedEnsemblePredictor(self, memory_name)
        self.storage = ModelStorage(memory_name, db_path='activity_log.db')
        self.session_automation = CrossSessionAutomation(self)
        self.batcher = AutoBatcherAutomation(self)
        self.distribution = AgentDistributedInference(self, self.storage, memory_name, port=5001, ssl_cert_file=ssl_cert_file, ssl_key_file=ssl_key_file)
        self.query_node = QueryNode(self, memory_name, self.storage)
        self.mlp = MLP()
        self.focused_mlp = MLP()

        self.X = None
        self.vocab_size = None
        self.model2 = None
        self.model3 = None
        self.texts = None
        self.intents = None
        self.role_bot = None
        self.batch_timer = None
        self.reverse_map = None

        self.use_transformer_for_proba = False
        self.agreement = False
        self.external_peer_enabled = False
        self.autonomous = False 
        self.show_explainability_details = True     

        self.temperature = 1.0
        self.memory_name = memory_name

        self.pending_batch = []
        self.temporary_id = []

        self.final_conf_score = 0.0
        self.confidence_threshold = 0.45  
        self.peer_assistance_threshold = 0.0              
        self.agent_id = random.randint(0, 10000)

        self.vocab = {}
     
        if not self.storage.memory_exists(memory_name, type='Pipeline'):
            self.memory = {}
        else:
            print(f'|| Found Matched Memory: {memory_name}!')
            self.memory = self.storage.memory_retrieval(memory_name, type_func='Pipeline', verbose=True)


    def initialize_fitting(self, text):
        self.tfidf.fit_transform(text).toarray()
        vocab_size = len(self.tfidf.get_feature_names_out())
        self.vocab_size = vocab_size
        

    def initialize_model_encoding(self, X, y_raw):
        vocab_size = self.vocab_size

        num_classes = len(np.unique(y_raw))
        y_onehot = np.zeros((len(y_raw), num_classes))
        y_onehot[np.arange(len(y_raw)), y_raw] = 1

        automatic_change = self.automatic_parameterization(vocab_size, num_classes)
        self.embedding_dim = automatic_change
        layer1, layer2 = self.automatic_dense_layer(X, vocab_size, num_classes)

        model = self.mlp  

        model.add(layer1)
        model.add(layer2)

        return y_onehot



    def initialize_model_(self, X, input_dim, num_classes):
        automatic_change = self.automatic_parameterization(input_dim, num_classes)

        automatic_change = self.automatic_parameterization(input_dim, num_classes)

        layer1= Dense(X, input_dim, automatic_change, activation="relu")
        layer2 = Dense(X, automatic_change, num_classes, activation='relu')

        abundant_layer = automatic_change * 10
        first_feed_layer = Dense(X, input_dim, abundant_layer, activation="relu")
        sec_feed_layer = Dense(X, abundant_layer, num_classes, activation="relu")

        self.model3 = MLP() 

        self.model3.add(layer1)
        self.model3.add(layer2)

        self.focused_mlp.feed_add(first_feed_layer)   
        self.focused_mlp.feed_add(sec_feed_layer)                

        
    def automatic_parameterization(self, input_size, num_classes):
        parameters = input_size * num_classes / 2
        parameters = int(parameters)
        return parameters


    def automatic_dense_layer(self, X, input_dim, num_classes):
        vocab_size = self.vocab_size 

        automatic_change = self.automatic_parameterization(input_dim, num_classes)

        layer1= Dense(X, input_dim, automatic_change, activation="relu")
        layer2 = Dense(X, automatic_change, num_classes, activation="relu")

        return layer1, layer2


    def text_encoder(self, texts):
        vocab = self.vocab
        idx = 0
        for item in texts:
            texts = item[0] if isinstance(item, tuple) else item
            for word in texts.split():
                if word not in vocab:
                    vocab[word] = idx
                    idx += 1


    def encode(self, sentence, vocab, max_len=6):        
        tokens = sentence.split()
        ids = [vocab.get(w, 0) for w in tokens]
        while len(ids) < max_len:
            ids.append(0)
        
        return ids[:max_len]



    def input_encoding(self, datasets):
        texts = [d[0] for d in datasets]
        intents = [d[1] for d in datasets]
        intent_to_id = {intent:i for i, intent in enumerate(sorted(set(intents)))}
        num_classes = len(intent_to_id)
        labels = [intent_to_id[i] for i in intents]

        reverse_map = {}
        for i in range(len(intents)):
            reverse_map[i] = intents[i]

        self.texts = texts
        self.intents = intents
        self.reverse_map = reverse_map

        self.model2 = Transformer(
            vocab_size=len(self.vocab),
            d_model=32,
            n_heads=4,
            num_classes=num_classes
        )   

        y_true = np.zeros((len(labels), num_classes))
        for i,l in enumerate(labels):
            y_true[i,l] = 1
            
        input_ids_list = []

        input_ids_list = []
        for text in texts:
            input_ids_list.append(np.array(self.encode(text, self.vocab)))  

        return input_ids_list, y_true


    def cosine_robust_similarity(self, a, b):
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b[0])
    
        try:
            dot_product = np.dot(a, b)
        except:
            try: #detect inhomogenous shape
                dot_product = np.dot(a.flatten(), b[:a.flatten().shape[0]])
            except:
                try:
                    dot_product = np.dot(a[:b.shape[0]], b.flatten()[:a.shape[0]])
                except:
                    print('[-] No similarity due to inhomogenous shapes and failed attempts to find subsets, returning low similarity score.')
                    return 0.1
    
        dot_product = np.dot(a, b)
        cosine = dot_product / (norm_a * norm_b)
    
        return np.clip(cosine, -1.0, 1.0)


    def cosine_similarity(self, a, b):
        eps = 1e-5
        b = b[0]

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
    
        if len(a.shape) > 1 and len(b.shape) > 1 and a.shape[1] != b.shape[0]:
            a = np.asarray(a)
            b = np.asarray(b)
            subset_a = a[0, :b.shape[0]]
            subset_b = a[:subset_a.shape[0], 0]

            try:
                try:
                    dot_product = np.dot(subset_a, subset_b)
                except:
                    dot_product = np.dot(subset_a[0, :min(subset_a.shape[1], subset_b.shape[0])], subset_b[:min(subset_a.shape[1], subset_b.shape[0])])
            except:
                try:
                    dot_product = np.dot(a.flatten(), b[:a.flatten().shape[0]])
                except:
                    try:
                        dot_product = np.dot(a[:b.shape[0]], b.flatten()[:a.shape[0]])
                    except:
                        print('[-] No similarity due to inhomogenous shapes and failed attempts to find subsets, returning low similarity score.')
                        return 0.1
       

        else:
            subset_a = a[:a.shape[0]]
            subset_b = b[:subset_a.shape[0]]

            try:
                dot_product = np.dot(subset_a, subset_b) 
            except:
                try:
                    subset_b_2 = subset_b[:subset_a.shape[1], :subset_a.shape[0]]  
                    dot_product = np.dot(subset_a, subset_b_2)

                except:
                    print('[-] No similarity due to inhomogenous shapes and failed attempts to find subsets, returning low similarity score.')
  
                    return 0.0         

        cosine = dot_product / (norm_a * norm_b)
        cosine = np.clip(cosine, -1.0, 1.0)

        return cosine  

    def anisotropy_measurement(self, x):
        eps = 1e-5
        if isinstance(x, list):
            print(f'[=] Converting list to array with shape: {len(x)}')
            x = np.array(x)
            x = x.reshape(x.shape[0], -1)  # Flatten if necessary
        try:
            gradient = np.gradient(x)
        except:
            subnet = x[:min(10, x.shape[0]), :min(10, x.shape[1])]
            gradient = np.gradient(subnet.flatten())

        val = [np.linalg.norm(v) for v in gradient]
        anisotropy = np.std(val) / np.mean(val) + eps
        return anisotropy 

    def modular_prediction_saving(self, X, X2, output):
        memory_name = self.memory_name
        self.memory['TW'] = X, output # transformers W
        self.memory['MW'] = X2, output # MLP W

        self.storage.save_model_dict(memory_name, self.memory, type='Pipeline', model_type='prediction')
        print('🚀 Memory Prediction Added!')


    def modular_probability_saving(self, X, X2, prob):
        memory_name = self.memory_name

        self.memory['TP'] = X, prob  # transformers prob
        self.memory['MP'] = X2, prob # MLP prob

        self.storage.save_model_dict(memory_name, self.memory, type='Pipeline', model_type='probs')
        print('🚀 Memory Probability Added!')


    def _cross_session_availability(self):
        try:
            print('=== CROSS SESSION MEMORY AVAILABILITY AND TRANSFER ===')
            print('1. Export memory session')
            print('2. Import memory session')
            print('3. Sync with another device')
            print('4. List sessions')

            chosen = input('[=] Choose Options [1/2/3/4]: ')
            if chosen != '3':
                filename = input('[=] Insert filename name to export or import or list session (ex. name: memory_session): ')

            if chosen == '1':
                if filename:
                    print(f'[+] Exporting memory session with name {filename}...')
                    self.session_automation.export_session(filename)
                    print(f'🚀 {filename} successfully exported as json!')
                else:
                    print('[-] Invalid filename!')
                    pass

            elif chosen == '2':
                if isinstance(filename, str):
                    json_converted = filename + '.json'
                    print(f'[=] Importing memory session, filename: {json_converted}...')
                    self.session_automation.import_session(json_converted)
                    print(f'[+] Successfully imported {json_converted}! ')
                else:
                    print('[-] Invalid filename!')
                    pass

            elif chosen == '3':
                ip_number = input('[=] Insert ip number of your device for syncing: ')  
                if ip_number:
                    print('[=] Syncing with device to export memory session...')  
                    self.session_automation.sync_with_another_device(ip_number, port=5000) 
                else:
                    print('[-] Invalid filename!')
                    pass

            elif chosen == '4':
                print('[=] Listing sessions...')                
                print('[!] Note: you must put the common name of memory sessions you have, (ex: memory_sessions)')
                print('if your most memory sessions you have contains memory_ name in front, insert it in the input.')
                self.session_automation.list_sessions(filename)
            else:
                print('[-] Invalid options! ')
                pass

        except Exception as e:
            print(f'[-] Warning! Error detected during session availability: {e}')
            pass


    def model_memory_gate(self, x, x2):
        memory = self.memory
        cache_trans_memory = [key for key, (inp) in memory.items() if key.startswith('TW') and self.cosine_robust_similarity(x, inp) >= 0.9]
        cache_mlp_memory =  [key for key, (inp2) in memory.items() if key.startswith('MW') and self.cosine_similarity(x2, inp2) >= 0.9]

        if cache_mlp_memory and cache_trans_memory:
            for memo in cache_trans_memory:
                _, out = memory[memo]

            for memo2  in cache_mlp_memory:
                _, out = memory[memo2]

            output = out.copy()
            return output      
        else:
            if cache_mlp_memory:
                print('[+] Found matching memory from mlp past memory!')                
                for memo in cache_mlp_memory:
                    _, out = memory[memo] 

                output = out.copy() 
                return output 

            elif cache_trans_memory:
                print('[+] Found matching memory from transformer past memory!')                
                for memo in cache_trans_memory:
                    _, out = memory[memo] 

                output = out.copy() 
                return output

            else:
                print('🔄 No Matching Memory!')
                return None


    def model_probability_gate(self, x, x2):
        memory = self.memory
        cache_trans_memory = [key for key, (inp) in memory.items() if key.startswith('TP') and self.cosine_robust_similarity(x, inp) >= 0.95]
        cache_mlp_memory =  [key for key, (inp2) in memory.items() if key.startswith('MP') and self.cosine_similarity(x2, inp2) >= 0.95]

        if cache_mlp_memory and cache_trans_memory:
            for memo in cache_trans_memory:
                _, out = memory[memo]

            for memo2  in cache_mlp_memory:
                _, out = memory[memo2]

            output = out.copy()
            return output      
        else:
            print('🔄 No Matching Probability!')
            return None

    def prediction_batch(self, texts):
        if not texts:
            return []
        
        # Prepare batch inputs
        input_ids_list = []
        X_raw_list = []
        
        for text in texts:
            # Prepare transformer input
            input_ids = np.array([self.encode(text, self.vocab)])
            input_ids_list.append(input_ids)
            
            # Prepare MLP input
            if not hasattr(self, 'tfidf') or self.tfidf is None:
                self.initialize_fitting([text])
            X_raw = self.tfidf.transform([text]).toarray()
            X_raw_list.append(X_raw)
        
        # Stack into batches
        batch_input_ids = np.vstack(input_ids_list)  # (batch_size, seq_len)
        batch_X_raw = np.vstack(X_raw_list)          # (batch_size, features)
        
        # Run batch prediction through your existing logic
        return self._batch_prediction_core(batch_input_ids, batch_X_raw)


    def _batch_model_memory_gate(self, batch_input_ids, batch_X_raw):
        batch_probs = [None] * len(batch_input_ids)
        
        for i in range(len(batch_input_ids)):
            probs = self.model_memory_gate(
                batch_input_ids[i:i+1], 
                batch_X_raw[i:i+1]
            )
            if probs is not None:
                arr = np.array(probs)
                if arr.ndim > 1:
                    arr = arr[0]
                elif arr.ndim == 0:
                    arr = arr.reshape(1)
                batch_probs[i] = arr.copy()
        
        return batch_probs


    def _batch_prediction_core(self, batch_input_ids, batch_X_raw):
        idx_total = 0
        batch_size = len(batch_input_ids)
        
        # Check memory gate for all samples (batch version)
        batch_probs = self._batch_model_memory_gate(batch_input_ids, batch_X_raw)
        
        # Find which indices need fresh prediction
        needs_prediction = [i for i, p in enumerate(batch_probs) if p is None]
        
        if needs_prediction:
            # Extract samples that need prediction
            fresh_input_ids = batch_input_ids[needs_prediction]
            fresh_X_raw = batch_X_raw[needs_prediction]
            
            # Run ensemble on all fresh samples at once
            fresh_probs, details = self.ensemble.predict_ensemble(
                fresh_input_ids, fresh_X_raw, method='dynamic'
            )
            
            # Store in memory
            for i, idx in enumerate(needs_prediction):
                batch_probs[idx] = fresh_probs[i]
                idx_total += 1
                if idx_total < 2:
                    self.modular_prediction_saving(
                        fresh_input_ids[i:i+1], 
                        fresh_X_raw[i:i+1], 
                        fresh_probs[i:i+1]
                    )
        return np.array(batch_probs)
    
  
    def predict_async(self, text, callback=None):
        return self.batcher.add_request(text, callback)
    

    def predict_single(self, text):
        result = [None]
        
        def callback(r):
            result[0] = r
        
        self.predict_async(text, callback)
        
        # Wait for result
        while result[0] is None:
            time.sleep(0.001)
        
        return result[0]
 

    def _batch_hybrid_prediction(self, batch_input_ids, batch_X_raw, y_true):
        print('[+] Initiating hybrid prediction batching...')
        idx_total = 0
        batch_probs = self._batch_model_memory_gate(batch_input_ids, batch_X_raw)
        
        needs_prediction = [i for i, p in enumerate(batch_probs) if p is None]
        
        if needs_prediction:
            fresh_input_ids = batch_input_ids[needs_prediction]
            fresh_X_raw = batch_X_raw[needs_prediction]
            
            # Batch ensemble
            fresh_probs, details = self.ensemble.predict_ensemble(
                fresh_input_ids, fresh_X_raw, y_true, method='dynamic'
            )
            
            for i, idx in enumerate(needs_prediction):
                batch_probs[idx] = fresh_probs[i]
                idx_total += 1
                if idx_total < 2:
                    self.modular_prediction_saving(
                        fresh_input_ids[i:i+1], 
                        fresh_X_raw[i:i+1], 
                        fresh_probs[i:i+1]
                    )
        valid_probs = []
        for p in batch_probs:
            if p is None:
                # Use zeros as fallback for None values
                valid_probs.append(np.zeros_like(fresh_probs[0]))
            elif isinstance(p, list):
                valid_probs.append(np.array(p))
            else:
                valid_probs.append(p)
        
        try:
            try:
                return np.array(valid_probs)
            except:
                return valid_probs
        except Exception as e:
            print(f'[-] Error converting batch probabilities to array: {e}')
            return valid_probs
    

    def _batch_predict_proba(self, batch_input_ids, batch_X, type='Hybrid'):
        batch_size = len(batch_input_ids)
        idx_total = 0
        
        # Batch memory gate
        batch_probs = [None] * batch_size
        for i in range(batch_size):
            probs = self.model_probability_gate(
                batch_input_ids[i:i+1], 
                batch_X[i:i+1]
            )
            if probs is not None:
                batch_probs[i] = probs[0]
        
        # Find which need fresh prediction
        needs_prediction = [i for i, p in enumerate(batch_probs) if p is None]
        
        if needs_prediction:
            fresh_input_ids = batch_input_ids[needs_prediction]
            fresh_X = batch_X[needs_prediction]
            
            # Batch transformer and MLP
            transformer_pred, fresh_probs, attn_weights = self.model2.predict(fresh_input_ids)
            mlp_pred = self.mlp.forward(fresh_X)
            
            mlp_pred_indices = np.argmax(mlp_pred, axis=1)
            trans_pred_indices = np.argmax(fresh_probs, axis=1)
            
            # Calibrate batch
            for i, idx in enumerate(needs_prediction):
                idx_total += 1
                if mlp_pred_indices[i] != trans_pred_indices[i]:
                    calibrated = self._calibrate_probs(
                        fresh_probs[i:i+1], 
                        [mlp_pred_indices[i]], 
                        attn_weights[i:i+1] if attn_weights is not None else None,
                        fresh_input_ids[i:i+1]
                    )
                    batch_probs[idx] = calibrated[0]
                else:
                    # Models agree
                    probs_i = fresh_probs[i].copy()
                    target = mlp_pred_indices[i]
                    probs_i[target] = min(probs_i[target] * 1.2, 0.95)
                    probs_i /= probs_i.sum()
                    batch_probs[idx] = probs_i
                
                # Save to memory
                if idx_total < 2:
                    self.modular_probability_saving(
                        fresh_input_ids[i:i+1], 
                        fresh_X[i:i+1], 
                        np.array([batch_probs[idx]])
                    )
      
        return np.array(batch_probs)



    def hybrid_prediction(self, rules, input_ids, dataset):
        X, y, _, _ = self.feature_generation(rules, dataset) 
        if len(input_ids.shape) == 2 and input_ids.shape[0] > 1:
            # this is batch mode version
            return self._batch_hybrid_prediction(input_ids, X, y)

        probs = self.model_memory_gate(input_ids, X)

        if probs is None or not self.agreement:
            if not self.autonomous:
                print('= Prediction Method needed: ')
                print('[1]. dynamic')
                print('[2]. meta')
                print('[3]. attention')

                print('[-] Autonomous prediction give the model full control of its dynamic prediction, without any user input.')
                choose_method = input('[=] Autonomous prediction initiated? [Y/N] (press N to insert manual prediction method): ')

                if choose_method == 'Y':
                    self.autonomous = True
                    probs, details = self.ensemble.predict_ensemble(input_ids, X, y, method='dynamic') 

                else:
                    method = input('|| Choose one method (ex: dynamic): ')
                    if method:
                        probs, details = self.ensemble.predict_ensemble(input_ids, X, y, method=method)
                    else:
                        print('|| Invalid Method.. returning to dynamic prediction..')
                        probs, details = self.ensemble.predict_ensemble(input_ids, X, y, method='dynamic')    
            else:
                print('[+] Autonomous dynamic prediction: ')
                probs, details = self.ensemble.predict_ensemble(input_ids, X, y, method='dynamic') 

            self.modular_prediction_saving(input_ids, X, probs)
            print('🚀 Memory Added!')

        return probs

    def _handle_distributed_connections(self, probs, self_attn_weights, input_ids, agreement):
        print('=== AGENT DISTRIBUTIED INFERENCE HANDLING ===')
        print('1. Handle local In-device Peer')
        print('2. Handle external-device Peer')

        if not self.autonomous:
            program = input('[=] Pick your choice [1/2] (choose N to skip): ')

        if self.autonomous or program == '1':
            print('=== IN-DEVICE PEER REQUEST INITIATED ===')
            probs = self.distribution._handle_peer_agent_request(probs, self_attn_weights, input_ids, type='DevicePeer', agreement=agreement)
            if self.distribution.query_node.peer_trust < self.confidence_threshold:
                print('[-] Peer trust is low, broadcasting ping to check for better peers...')
                alive_agents = self.distribution.broadcast_ping()

                if alive_agents:
                    print(f'[+] Alive agents: {alive_agents} identified, enabling external peer connections for better assistance...')
                    self.external_peer_enabled = bool(alive_agents)
                    self.autonomous = False

        elif self.external_peer_enabled and program == '2':
            print('=== EXTERNAL PEER REQUEST INITIATED ===')
            ip_number = input('[=] Insert IP Number to connect with peer: ')
            if self.role_bot is None:
                self.role_bot = input('[=] Start server or connect to agent? [Server/Connect]: ')

            if ip_number and self.role_bot:
                try:
                    distributed_a = self.distribution
                    
                    if self.role_bot == 'Server':
                        distributed_a.start_server()
                    else:
                        distributed_a.connect_to_agent(ip_number, 5555)   

                    print('=== EXTERNAL PEER REQUEST INITIATED ===')
                    print('[1]. Request prediction')
                    print('[2]. Handle Peer uncertainty')
                    sec_program = input('[=] Pick your choice [1/2]: ')    

                    if sec_program == '1':
                        for intent in self.intents:
                            result = distributed_a.request_prediction(self, intent)
                            print(f"[+] Remote prediction: {result}")
                
                            # Ensemble vote across all agents
                            votes = []
                            list_probs = []

                        # Check network status
                        print('=== Checking network status with broadcast ping... ===')
                        alive_agents = distributed_a.broadcast_ping()  
                        if alive_agents:
                            print(f'[+] Alive agents: {alive_agents} identified, requesting ensemble votes...')                          
                            for agent_id in distributed_a.remote_agents:
                                peer_probs, vote = distributed_a.request_ensemble_vote(agent_id, intent)
                                if vote:
                                    votes.append(vote)
                                    list_probs.append(peer_probs)
                                    
                                    for vote in votes:
                                        print(f'[+] Prediction: {vote['prediction']}')
                                        print(f'[+] Confidence: {vote['confidence']}')
                                        print(f'[+] Trust: {vote['trust_score']}')
                                        if vote['confidence'] > self.confidence_threshold:
                                            for i in range(len(probs)):
                                                probs = probs[i] * (1.0 + vote['trust_score'] * vote['confidence'])
                                            probs = probs.copy() / np.sum(probs) # Normalize after adjustment
                            else:
                                print(f'[-] No alive agents found, Total: {alive_agents} Agent found. Using local prediction only.')
                                probs = self.distribution._handle_peer_agent_request(probs, self_attn_weights, input_ids, type='ExternalPeer', agreement=agreement)

                        distributed_a.print_network_status()

                    elif sec_program == '2':
                        probs = self.distribution._handle_peer_agent_request(probs, self_attn_weights, input_ids, type='ExternalPeer', agreement=agreement)     

                except Exception as e:
                    print(f'[-] Error establishing connections: {e}, returning previous probs.')
                    self.inference.report_failure(id(self), 'processing', reason=f'{e}')                        

            else:
                print(f'[-] Invalid Choice... returning previous probs.')
                self.inference.report_failure(id(self), 'processing', reason="InvalidChoice")                        

        elif program == 'N':
            print('|| Skipping Peer connections, returning previous probs')
        else:
            print('[-] Invalid choice! returning previous probs')
            
        return probs



    def mlp_predict(self, X):
        if isinstance(X, str) or isinstance(X[0], str):
            self.initialize_fitting(X)            
            X_tfidf = self.tfidf.transform(X).toarray() 

        logits = self.mlp.prediction(X_tfidf)
        return logits

    
    def predict_proba(self, input_ids, X, type='Hybrid', embedded=False):
        eps = 1e-5
        probs_memory = self.model_probability_gate(input_ids, X)
        if isinstance(self.storage.id_history, list) or isinstance(self.storage.id_history, np.ndarray) and not self.agent_id in self.storage.id_history:
            self.storage.id_history.append(self.agent_id)
            id_history = self.storage.id_history
        else:
            self.temporary_id.append(self.agent_id)
            id_history = self.temporary_id

        is_batch = len(input_ids.shape) == 2 and input_ids.shape[0] > 1
        AME = self.AME_Encoder(input_ids)
        AMR = 1.0 / (1.0 + np.exp(-AME))
        
        if is_batch:
            return self._batch_predict_proba(input_ids, X, type) 

        if type == 'Hybrid':
            print('[=] Hybrid based classification method.')
            transformer_pred, probs, attn_weights = self.model2.predict(input_ids, embedded=embedded)
            mlp_pred = self.mlp.forward(X)

            if mlp_pred.ndim == 1:
                mlp_pred = mlp_pred.reshape(1, -1)
            # Ensure transformer_pred is 2D
            if transformer_pred.ndim == 1:
                transformer_pred = transformer_pred.reshape(1, -1)

            mlp_pred_indices = np.argmax(mlp_pred, axis=1)
            trans_pred_indices = np.argmax(transformer_pred, axis=1)
           
            if probs_memory is None:
                agreement = np.allclose(mlp_pred_indices, trans_pred_indices, rtol=eps)
          
            else:
                # memory agreement must match previously detected learned patterns, contextual transformer prediction isnot needed
                try:
                    probs_memory_ = np.argmax(probs_memory) 
                except:
                    probs_memory_= np.argmax(probs_memory, axis=1)

                agreement = np.allclose(mlp_pred_indices, probs_memory_, rtol=eps)

            need_peer_condition = not agreement and probs_memory is None
            self.agreement = agreement

            if not agreement:
                self.peer_assistance_threshold += 0.1                                     
                # if both pattern are still conflicting, use contextual relations for sorting regularization.
                if need_peer_condition:
                    print('|| Uncertain prediction, requesting peer assistance if allowed...')
                    probs = self._handle_distributed_connections(probs, attn_weights, input_ids, agreement)
                else:
                    need_calibration_condition = not agreement and self.final_conf_score > self.confidence_threshold
                    if need_calibration_condition:
                        print('[||] Uncertain prediction, but memory exist, skipping peer assistance and calibrating with attention because of high confidence...')                        
                        probs = self._calibrate_probs(probs, mlp_pred_indices, attn_weights, input_ids)       
                    else:
                        print('[-] Uncertain prediction, needing local peer assistance...')                        
                        probs = self._handle_distributed_connections(probs, attn_weights, input_ids, agreement)

            else:
                self.peer_assistance_threshold -= 0.2               
                print('[+] Both Models agree, Normalizing prediction with confidence boost...')
                for i, target in enumerate(mlp_pred_indices):
                    probs[i, target] = min(probs[i, target] * 1.2, 0.95)
                    probs[i] /= probs[i].sum()

                    
            if not agreement and probs_memory is not None:
                self.storage.save_peer_needs_dict(self.memory_name, probs_memory, mlp_pred, id_history)   
            else:
                self.storage.save_peer_needs_dict(self.memory_name, probs, mlp_pred, id_history)

            self.modular_probability_saving(input_ids, X, probs)
            print('🚀 Memory Added!')
            return probs
        else:
            print('[=] MLP Based classification method.')
            logits = self.mlp.forward(X)

            if probs_memory is not None:
                mlp_pred_indices = np.argmax(logits, axis=1)
                probs_memo = np.array(probs_memory)
                try:
                    probs_memory_ = np.argmax(probs_memory) 
                except:
                    probs_memory_ = np.argmax(probs_memory, axis=1)

                agreement = np.allclose(mlp_pred_indices, probs_memory_, rtol=eps)
                
                if not agreement:
                    # if both pattern are still conflicting, used latest prediction
                    probs = logits.copy()       
                else:
                    for i, target in enumerate(mlp_pred_indices):
                        probs_memo[i, target] = min(probs_memo[i, target] * AMR, 0.95)
                        probs_memo[i] /= probs_memo[i].sum()

            return logits

    def data_preparation(self, titles, labels):
        datasets = []
        raw = []
        for title in titles:
            tupled_title = (str(title))
            datasets.append(tupled_title)
            raw.append(str(title))

        for label in labels:
            tupled_label = (str(label))
            datasets.append(tupled_label)
            raw.append(str(label))

        self.initialize_fitting(raw)
        X_raw = self.tfidf.transform(raw).toarray()

        return datasets, X_raw

    
    def _calibrate_probs(self, probs, target_preds, attn_weights, input_ids):
        calibrated = probs.copy()
        if isinstance(input_ids, list):
            input_ids = np.array(input_ids)

        if len(probs.shape) > 1:
            n_classes = probs.shape[1]
        else:
            n_classes = probs.shape[0]

        batch_size = len(target_preds)
        eps = 1e-5

        for i in range(batch_size):
            mlp_target = target_preds[i]
            if attn_weights is not None and i < len(attn_weights):
                attn = attn_weights[i]
                anisotropy = self.anisotropy_measurement(attn)     

                attn_quality = np.std(attn) if attn.size > 0 else 0.5
                boost = 0.5 + attn_quality

            else:
                attn_quality = 1.0 / (1.0 + np.exp(-attn))
                boost = (1.0 - attn_quality) + eps

            self.temperature = boost + (1.0 / (1.0 + attn_quality)) * anisotropy
            if isinstance(self.temperature, np.ndarray):
                self.temperature = np.clip(np.mean(self.temperature), 1e-5, 5.0)

            try:
                calibrated[i, mlp_target] = min(calibrated[i, mlp_target] * (1.5 * boost), 0.95)
            except:
                return calibrated

            calibrated[i] /= calibrated[i].sum()
        return calibrated
    
    def _softmax(self, x):
        temp = self.temperature

        try:
            if len(x.shape) > 1:
                x_dip = x / temp
                exp_x = np.exp(x_dip - np.max(x_dip, axis=1, keepdims=True))
                softmax = exp_x / np.sum(exp_x, axis=1, keepdims=True)     
            else:
                x_dip = x / temp
                exp_x = np.exp(x_dip - np.max(x_dip))
                softmax = exp_x / np.sum(exp_x)
        except:
            x_dip = x / temp
            exp_x = np.exp(x_dip - np.max(x_dip, axis=1, keepdims=True))


        softmax = exp_x / np.sum(exp_x, axis=1, keepdims=True)     
        return softmax


    def validate_writable_path(self, path):
        try:
            path = os.path.expanduser(path)
        
            directory = os.path.dirname(path) or '.'
        
            if not os.path.exists(directory):
                return False, f"Directory does not exist: {directory}"
        
            if not os.access(directory, os.W_OK):
                return False, f"No write permission for directory: {directory}"
        
            if os.path.exists(path):
                if not os.access(path, os.W_OK):
                    return False, f"File exists but is not writable: {path}"
        
            test_file = os.path.join(directory, f".test_write_{os.getpid()}")
            try:
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
            except Exception as e:
                return False, f"Write test failed: {e}"
        
            return True, "Path is writable"
        
        except Exception as e:
            return False, f"Validation error: {e}"

    def safe_pickle_save_with_feedback(self, data, suggested_path):
        print("\n" + "="*50)
        if suggested_path:
            print(f"|| Suggested path: {suggested_path}")
        
        user_path = input("|| Enter path to save pickle file (or 'cancel' to skip): ").strip()
        
        if user_path.lower() == 'cancel':
            print("|| Save cancelled.")
            pass
        
        user_path = user_path.strip('"').strip("'")
        user_path = os.path.expanduser(user_path)
        
        if os.path.isdir(user_path):
            from datetime import datetime
            default_filename = f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            user_path = os.path.join(user_path, default_filename)
            print(f"|| Using filename: {default_filename}")
        
        valid, message = self.validate_writable_path(user_path)
        
        if valid:
            try:
                os.makedirs(os.path.dirname(user_path), exist_ok=True)
                
                with open(user_path, 'wb') as f:
                    joblib.dump(data, f)
                
                print(f"✓ Successfully saved to: {user_path}")
                pass
                
            except PermissionError as e:
                print(f"✗ Permission denied: {e}")
                print("|| Try a different location (like your Desktop or Documents folder)")
                pass
            except Exception as e:
                print(f"✗ Save failed: {e}")
                pass

        else:
            print(f"✗ Invalid path: {message}")
            print("Tips:")
            print("  - Use a path in your home directory: ~/Documents/myfile.pkl")
            print("  - Make sure the directory exists and is writable")
            print("  - Try saving to Desktop or Documents folder")
            pass


    def utility_MLP_set(self, X, y):
        print('🚀 Training MLP Pipeline: ')
        self.mlp.train(X, y, epochs=1000, lr=0.1)

        try:
            joblib.dump(self.mlp, 'analyzer_model.pkl')
            joblib.dump(self.model2, 'analyzer_agent.pkl')
            print("🎉 Model trained and saved!")            
        except Exception as e:
            print(f'|| Failed to joblib dump file! : {e}, User Manual filepath suggestion needed...')

            permission = input('|| Insert Filepath? [Y/N]: ')
            if permission == 'Y':
                suggested_path = input('|| Filepath suggestion: ')
                if suggested_path:
                    self.safe_pickle_save_with_feedback(self.mlp, suggested_path)
                    self.safe_pickle_save_with_feedback(self.model2, suggested_path)
                    print("🎉 Model trained and saved!")                     
                else:
                    print('|| Failed to dump Your model! ')
                    pass
            else:
                print('|| Failed to dump Your model! ')
                pass

        print("🎉 Model trained!")


    def auto_generate_labels_from_texts(self, rules, texts):
        import re
        y_raw = []
     
        for text in texts:
            text_lower = text.lower()
            matched = False
            for pattern, label in rules:
                if re.search(pattern, text_lower):
                    y_raw.append(label)
                    matched = True
                    break
        
            if not matched:
                y_raw.append('other')
    
        from collections import Counter
        print("\n📊 Auto-generated label distribution:")
        for label, count in Counter(y_raw).items():
            print(f"   {label}: {count} ({count/len(texts)*100:.1f}%)")
    
        return y_raw



    def mlp_training_features(self, rules, dataset):
        print("\n🔄 Preparing MLP data from dataset format")
    
        if isinstance(dataset[0], tuple) and len(dataset[0]) == 2:
            # Format: [(features, label), ...]
            features_list = []
            labels_list = []
            print('Dataset Type 1: [(value), (value)]')

            for item in dataset:
                features, label = item
                features_list.append(features)
                labels_list.append(label)
        
            X_mlp = np.array(features_list)
            y_raw = np.array(labels_list)
        
        elif isinstance(dataset[0], (list, np.ndarray)) and len(dataset[0]) > 1:
            print('Dataset Type 2')
            X_mlp = np.array([item[:-1] for item in dataset])
            y_raw = np.array([item[-1] for item in dataset])

        else: 
            print('Dataset type 3')     
            X_mlp = dataset.copy()   
            y_raw = self.auto_generate_labels_from_texts(rules, dataset)         

        unique_labels = sorted(set(y_raw))
        label_to_idx = {l: i for i, l in enumerate(unique_labels)}
        y_indices = np.array([label_to_idx[l] for l in y_raw])
    
        n_classes = len(unique_labels)
        y_onehot = np.zeros((len(y_indices), n_classes))
        y_onehot[np.arange(len(y_indices)), y_indices] = 1

        if isinstance(X_mlp, np.ndarray):
            input_dim = X_mlp.shape[0]           
        else:
            input_dim = len(X_mlp)

        print(f"\n✅ MLP data ready:")
        print(f"   X shape: {input_dim}")
        print(f"   y shape: {y_onehot.shape}")
        print(f"   Classes: {label_to_idx}")     
        return X_mlp, y_onehot, n_classes, input_dim        

    def shape_adaptation(self, X, inp):
        tuple_ver = (inp, inp)
        if X.shape != tuple_ver:
            X = X[:inp, :inp]

        return X

    def AME_Encoder(self, x):
        X = np.asarray(x)

        gradient = np.gradient(x, axis=-1)
        grad_energy = np.mean(np.linalg.norm(gradient, axis=-1))       
        X_mag = np.mean(np.linalg.norm(X, axis=-1))

        AME = np.log1p(X_mag) * np.log1p(grad_energy) 

        if AME == 0.0:
            eps = 1e-5
            AME = AME + eps
        return AME

    def feature_generation(self, rules, dataset):
        X_raw, y, n_classes, input_dim = self.mlp_training_features(rules, dataset)
            
        self.initialize_fitting(X_raw)            
        X_tfidf = self.tfidf.transform(X_raw).toarray()
        X = X_tfidf.copy() 

        X = self.shape_adaptation(X, input_dim)  

        return X, y, input_dim, n_classes       


    # necessary functions to reduce wasteful training in similar scarce environment
    def training_necessary_condition(self, input_ids, x):
        eps = 1e-5
        final_conf = self.final_conf_score
        confidence_threshold = self.confidence_threshold
        unsuitable_training = False

        probs = self.model_memory_gate(input_ids, x)

        anisotropy = self.anisotropy_measurement(input_ids)
        AME = self.AME_Encoder(input_ids)
        AMR = 1.0 / (1.0 + np.exp(-AME)) 

        LMR = 1.0 / (1.0 + np.exp(-AMR)) # logistic modelling rate
        ALR = 1.0 / (1.0 + np.exp(-anisotropy)) # anisotropic logistic rate
        AAC = (1.0 - ALR) + (1.0 - LMR) + eps # anisotropic abstract coefficient

        self.confidence_threshold = anisotropy * AAC 
        if np.isnan(self.confidence_threshold) or np.isinf(self.confidence_threshold):
            self.confidence_threshold = 0.45

        print(f'[||] Confidence threshold set to: {self.confidence_threshold} || Final Confidence Score: {final_conf}')

        # training is wasteful if the model processes little abstraction divergence without necessary context on a similar environment
        # AMR is guaranteed to give sufficient ratio on how modelling error error could be sufficient enough to guarantee the model successful training 
        # (not too high that it shows unstability, not too low that it shows rigidity), high anisotropy correlates to a much complex non linearity that the model will have a hard time adjusting
        # higher LMR than AAC means the model is likely to be in a regime where training could lead to overfitting or divergence due to insufficient modelling capacity relative to the complexity of the data, especially if the confidence score is also low, indicating that the model is not currently confident in its predictions and may not benefit from further training on this data.
        unsuitable_tolerance = probs is not None or LMR > AAC 
        unsuitable_conditions = anisotropy > 0.85 or final_conf > confidence_threshold
        unsuitable_peer_request = probs is not None and self.peer_assistance_threshold > self.confidence_threshold

        if unsuitable_tolerance or unsuitable_conditions or unsuitable_peer_request:
            print(f'[==] Unsuitable training condition detected! Tolerance: {unsuitable_tolerance} || Unsuitable Conditions: {unsuitable_conditions}')
            print(f'[==] Peer assistance condition: {unsuitable_peer_request} || Peer assistance threshold: {self.peer_assistance_threshold}')
            unsuitable_training = True

        print('== Training Condition evaluation == ')
        print(f'[==] Unsuitable training condition Evaluation || Unsuitable Tolerance: {unsuitable_tolerance} || Unsuitable Conditions: {unsuitable_conditions} || Unsuitable Peer Assistance: {unsuitable_peer_request}')
        print('[==] Final Decision on Training: ' + ('Unsuitable' if unsuitable_training else 'Suitable') + ' for training.')

        return unsuitable_training

    def sequence_encoding(self, datasets, max_len=32):
        input_sequences = []
        for item in datasets:
            if not self.model2:
                intents = [d[1] for d in datasets]
                intent_to_id = {intent:i for i, intent in enumerate(sorted(set(intents)))}
                num_classes = len(intent_to_id)                
                self.model2 = Transformer(
                    vocab_size=len(self.vocab),
                    d_model=32,
                    n_heads=4,
                    num_classes=num_classes
                ) 

            text = item[0] if isinstance(item, tuple) else item
            token_ids = self.encode(text, self.vocab, max_len=max_len)
            token_embs = self.model2.token_embedding[token_ids]         # (max_len, d_model)
            pos_embs = self.model2.pos_embedding[:max_len]              # (max_len, d_model)
            sequence_input = token_embs + pos_embs
            input_sequences.append(sequence_input)
        return np.stack(input_sequences)  # shape: (batch, max_len, d_model)

    def transformer_pooled_features(self, sequence_inputs):
        # mean/max/std pooling over sequence dimension
        mean_pool = np.mean(sequence_inputs, axis=1)
        max_pool = np.max(sequence_inputs, axis=1)
        std_pool = np.std(sequence_inputs, axis=1)
        return np.concatenate([mean_pool, max_pool, std_pool], axis=-1)

    def transformer_utilities(self, rules, datasets, X_raw):
        self.text_encoder(datasets)
        _, y_true = self.input_encoding(datasets)
        sequence_inputs = self.sequence_encoding(datasets)
        unsuitable_training = self.training_necessary_condition(sequence_inputs, X_raw)

        if not unsuitable_training:
            print(f'🚀 Training Transformer with {len(sequence_inputs)} Samples: ')
            conditional_anisotropy = self.anisotropy_measurement(sequence_inputs)
            if conditional_anisotropy >= self.confidence_threshold: 
                lr = 1e-4
                print('[+] Dynamic Backward')
                mode = 'dynamic_backward'
            else:
                lr = 0.1
                print('[-] Fixed Backward')
                mode = 'fixed_backward'

            self.model2.train(sequence_inputs, y_true, epochs=100, mode=mode, lr=lr, embedded=True)
            X_raw_generation, y, n_classes, input_dim = self.mlp_training_features(rules, datasets)
            X_raw_features = self.tfidf.transform(X_raw_generation).toarray()
            transformer_features = self.transformer_pooled_features(sequence_inputs)
            X_raw_features = np.concatenate([X_raw_features, transformer_features], axis=-1)

            if X_raw_features.var(axis=0).mean() < 0.2:
                for i in range(len(X_raw_features)):
                    for j in range(len(X_raw_features[i])):
                        if np.isnan(X_raw_features[i, j]) or np.isinf(X_raw_features[i, j]):
                            checksum = int(hashlib.md5(X_raw[i].encode()).hexdigest(), 16) % 1000 / 10000
                            X_raw_features[i, j] = checksum
            
            X_features = X_raw_features.copy()
            if isinstance(X_raw_features, list):
                X_features = np.asarray(X_raw_features)
            if isinstance(X_raw, list):
                X_raw = np.asarray(X_raw) 

            # hybrid features by dot product of raw features and extracted features from transformer, this allows the MLP to learn from both the original feature space and the transformer-extracted feature space
            # potentially improving its ability to capture complex patterns in the given first data   
            try:
                if len(X_features.shape) < 2: 
                    X_features = X_features.reshape(1, -1)
                if len(X_raw.shape) < 2:
                    X_raw = X_raw.reshape(1, -1)
                                    
                X_raw = X_raw[:X_features.shape[0], :X_features.shape[1]]

                hybrid_X = np.dot(X_raw, X_features)
            except:
                if len(X_features.shape) < 2: 
                    X_features = X_features.reshape(1, -1)
                if len(X_raw.shape) < 2:
                    X_raw = X_raw.reshape(1, -1)

                subnet_X_feature = X_features[:X_raw.shape[1], :X_raw.shape[0]]
                subnet_X_raw = X_raw[:, :subnet_X_feature.shape[0]]
                hybrid_X = np.dot(subnet_X_raw, subnet_X_feature)
            
            hybrid_X = np.concatenate([X_raw, X_features, hybrid_X], axis=-1)
            self.initialize_fitting(X_raw_generation)            
            X = self.shape_adaptation(hybrid_X, input_dim)

            self.initialize_model_(X, input_dim, n_classes)
            self.model3.train(X, y, epochs=1000, lr=0.1)
            print('🎉 All Model Trained!')
        else:
            print(f'[=] No suitable condition for training!')
            pass



    def transformer_input_encoding(self, titles):
        if hasattr(self, 'vocab') and self.vocab:
            print("🔄 Using Transformer for probability calibration")
            input_ids_list = []
            for title in titles:
                if isinstance(title, tuple):
                    title = title[0]
                    
                ids = self.encode(title, self.vocab)
                input_ids_list.append(np.array(ids))
                
                input_ids = np.array(input_ids_list)
                return input_ids

        else:
            print('[-] Cant get sufficient data!')
            return []


    def train(self, X, y_raw):
        self.initialize_fitting(X)            
        X_tfidf = self.tfidf.transform(X).toarray()
        self.X = X_tfidf.copy()

        print(f"\n🚀 Separate Modular MLP Pipeline:")
        print(f" Samples: {len(self.X)}")

        y_true = self.initialize_model_encoding(self.X, y_raw)
        self.utility_MLP_set(self.X, y_true)       
        print('✅ Done Training MLP Model! ')
         
    

class PipelinePredictionManager:
    def __init__(self, pipeline, label_csv='labels.csv', target_title='title', label='label'):
        self.pipeline = pipeline
        try:
            print("📖 Loading labels from text file...")
            self.titles, self.y_raw, self.label_map = self.load_labels_from_csv(label_csv, target_title, label)
        except Exception as e:
            print(f"Error loading labels: {e}")
            self.titles, self.y_raw, self.label_map = [], [], {}

        print(f"✅ Loaded {len(self.titles)} labeled examples")

    def load_labels_from_csv(self, filename, target_title, label):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(script_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"❌ File not found: {filepath}")
            return [], [], {}
        
        # Read CSV file
        df = pd.read_csv(filepath)
        
        print(f"✅ Loaded CSV with columns: {list(df.columns)}")
        
        # Extract titles and labels
        titles = df[target_title].tolist()
        string_labels = df[label].tolist()
        
        # Remove quotes if they're still there
        titles = [t.strip('"') for t in titles]
        
        print(f"📊 Found {len(titles)} examples")
        print(f"📊 Labels: {set(string_labels)}")
        
        # Create numeric labels
        unique_labels = sorted(set(string_labels))
        label_map = {label: i for i, label in enumerate(unique_labels)}
        y = [label_map[label] for label in string_labels]
        
        return titles, y, label_map



    def regular_prediction_method(self, titles, label_map, rules, show_proba=False, top_k=3, use_transformer=True):
        try:
            print(f"\n[🚀] Regular Prediction for labels with {len(titles)} titles...")
            reverse_map = {v: k for k, v in label_map.items()}
            num_classes = len(label_map)

            dataset, X = self.pipeline.data_preparation(titles, label_map)      
            _, y, _, _ = self.pipeline.mlp_training_features(rules, dataset)
            self.pipeline.transformer_utilities(rules, dataset, X) 
            input_ids, _ = self.pipeline.input_encoding(dataset)


            if use_transformer and hasattr(self.pipeline, 'vocab') and self.pipeline.vocab:
                print("[🔄] Using Transformer for probability calibration")
            
                # Encode titles for transformer
                input_ids_list = []
                for title in titles:
                    # Handle both string and tuple inputs
                    if isinstance(title, tuple):
                        title = title[0]
                    # Encode to token IDs using pipeline's vocabulary
                ids = self.pipeline.encode(title, self.pipeline.vocab)
                input_ids_list.append(np.array(ids))
            
                input_ids = np.array(input_ids_list)
            
            # Get transformer probabilities
                trans_probs, attn_weights = self.pipeline.model2.forward(input_ids)
            else:
                print("⚡ Using MLP only for predictions")
                trans_probs = None
        
            if not hasattr(self.pipeline, 'tfidf') or self.pipeline.tfidf is None:
                self.pipeline.initialize_fitting(titles)
            
            # Prepare texts for MLP
            if isinstance(titles[0], tuple):
                mlp_titles = [t[0] for t in titles]
            else:
                mlp_titles = titles
                
            X_tfidf = self.pipeline.tfidf.transform(mlp_titles).toarray()            
            # Forward pass through MLP
            if hasattr(self.pipeline.mlp, 'predict_proba'):
                mlp_probs = self.pipeline.mlp.predict_proba(X_tfidf)
            else:
                # Fallback if predict_proba not available
                logits = self.pipeline.mlp.forward(X_tfidf)
                mlp_probs = self.pipeline._softmax(logits)
                
            # Validate all MLP predictions at once
            mlp_pred_indices = np.argmax(mlp_probs, axis=1)
            if num_classes <= 0:
                num_classes = mlp_probs.shape[1] if mlp_probs.ndim > 1 else len(mlp_probs)

            valid_mask = mlp_pred_indices < num_classes
            if not np.all(valid_mask):
                invalid_count = np.sum(~valid_mask)
                # Replace invalid indices with argmax within valid range
                for i in range(len(mlp_pred_indices)):
                    valid_probs = mlp_probs[i][:num_classes] if num_classes > 0 else mlp_probs[i]
                    if len(valid_probs) > 0:
                        mlp_pred_indices[i] = int(np.argmax(valid_probs))
                    else:
                        mlp_pred_indices[i] = 0  # Default to first class   
                             
            results = []
            for i, title in enumerate(titles):
                # Handle tuple inputs
                if isinstance(title, tuple):
                    display_title = title[0]
                    expected_label = title[1] if len(title) > 1 else None
                else:
                    display_title = title
                    expected_label = None
                
                # MLP prediction
                mlp_class_idx = mlp_pred_indices[i]
                mlp_class_idx = min(mlp_class_idx, num_classes - 1)  # Clamped to valid range
                   
                mlp_confidence = mlp_probs[i][mlp_class_idx]
                mlp_label = reverse_map.get(mlp_class_idx, f"unknown_{mlp_class_idx}")
                anisotropy = self.pipeline.anisotropy_measurement(input_ids)
                anisotropic_rate = 1.0 / (1.0 + np.exp(-anisotropy)) if anisotropy is not None else 1.0

                # Transformer prediction (if available)
                if trans_probs is not None:
                    if trans_probs.shape[0] > i:
                        trans_probs_i = trans_probs[i]
                    else:
                        trans_probs_i = trans_probs[-1]  # fallback to last if mismatch
                    
                    trans_class_idx = np.argmax(trans_probs_i)
                    trans_confidence = trans_probs_i[trans_class_idx]
                    trans_label = reverse_map.get(trans_class_idx, f"unknown_{trans_class_idx}")
                    
                    # Calibrated probabilities (blend of MLP and Transformer)
                    if use_transformer:
                        # Boost MLP's prediction in transformer probabilities
                        calibrated = trans_probs_i.copy()
                        try:
                            calibrated[mlp_class_idx] = max(calibrated[mlp_class_idx], anisotropic_rate)
                            calibrated /= calibrated.sum()
                        except Exception as e:
                            calibrated = self.pipeline._calibrate_probs(mlp_probs, mlp_pred_indices, attn_weights, input_ids)
        
                        final_probs = calibrated
                        final_class_idx = mlp_class_idx  # Trust MLP's class decision
                        try:
                            final_confidence = final_probs[final_class_idx]
                        except IndexError:
                            final_confidence = np.max(final_probs) if isinstance(final_probs, np.ndarray) else final_probs

                        if isinstance(final_confidence, np.ndarray):
                            final_confidence = np.max(final_confidence)
                                          
                    else:
                        final_probs = mlp_probs[i]
                        final_class_idx = mlp_class_idx
                        final_confidence = mlp_confidence
                else:
                    final_probs = mlp_probs[i]
                    final_class_idx = mlp_class_idx
                    final_confidence = mlp_confidence
                    trans_label = reverse_map.get(mlp_class_idx, f"unknown_{mlp_class_idx}")
                    trans_confidence = mlp_confidence
                
                final_label = reverse_map.get(final_class_idx, f"unknown_{final_class_idx}")
                
                result = {
                    'title': display_title,
                    'expected': expected_label,
                    'predicted': final_label,
                    'confidence': final_confidence,
                    'index': final_class_idx,
                    'mlp_prediction': mlp_label,
                    'mlp_confidence': mlp_confidence,
                }
                
                if trans_label is not None:
                    result['transformer_prediction'] = trans_label
                    result['transformer_confidence'] = trans_confidence

                agreement = trans_label  == mlp_label
                
                
                # Include top-k predictions if requested
                if show_proba:
                    top_indices = np.argsort(final_probs)[-top_k:][::-1]
                    top_predictions = []
                    for idx in top_indices:
                        if idx in reverse_map:
                            top_predictions.append({
                                'label': reverse_map[idx],
                                'confidence': final_probs[idx]
                            })
                        else:
                            top_predictions.append({
                                'label': f"unknown_{idx}",
                                'confidence': final_probs[idx]
                            })
                    result['top_predictions'] = top_predictions
                    
                    mlp_top_indices = np.argsort(mlp_probs[i])[-top_k:][::-1]
                    mlp_top = []
                    for idx in mlp_top_indices:
                        if idx in reverse_map:
                            mlp_top.append({
                                'label': reverse_map[idx],
                                'confidence': mlp_probs[i][idx]
                            })
                    result['mlp_top_predictions'] = mlp_top
                    
                    if trans_probs is not None:
                        trans_top_indices = np.argsort(trans_probs[i])[-top_k:][::-1]
                        trans_top = []
                        for idx in trans_top_indices:
                            if idx in reverse_map:
                                trans_top.append({
                                    'label': reverse_map[idx],
                                    'confidence': trans_probs[i][idx]
                                })
                        result['transformer_top_predictions'] = trans_top
                
                results.append(result)
            
            # Display results
            print("\n" + "="*70)
            print("🎯 HYBRID PREDICTION RESULTS (MLP + Transformer)")
            print("="*70)
            
            correct_count = 0
            for result in results:
                print(f"\n📌 '{result['title']}'")
                
                if result.get('expected'):
                    status = "✓" if result['predicted'] == result['expected'] else "✗"
                    print(f"   Expected: {result['expected']} {status}")
                
                print(f"   🎯 FINAL PREDICTION: {result['predicted']} ({result['confidence']:.1%})")
                print(f"   ⚡ MLP: {result['mlp_prediction']} ({result['mlp_confidence']:.1%})")
                
                if result.get('transformer_prediction'):
                    arrow = "⬆️" if result['transformer_confidence'] > result['mlp_confidence'] else "⬇️"
                    print(f"   🌀 Transformer: {result['transformer_prediction']} ({result['transformer_confidence']:.1%}) {arrow}")
                
                if show_proba and 'top_predictions' in result:
                    print("\n   🔍 Top possibilities (calibrated):")
                    for j, pred in enumerate(result['top_predictions'][:top_k], 1):
                        bar = '█' * int(pred['confidence'] * 20)
                        print(f"      {j}. {pred['label']:20s} {bar} {pred['confidence']:.1%}")
                
                if result.get('expected') and result['predicted'] == result['expected']:
                    correct_count += 1
            
            if results and results[0].get('expected'):
                accuracy = correct_count / len(results)
                print(f"\n📊 Accuracy: {correct_count}/{len(results)} = {accuracy:.1%}")

            try:
                joblib.dump(self.pipeline, 'modular_agent.pkl')
                print('💾  Model saved!')
            except Exception as e:
                print(f'|| Failed to joblib dump file! : {e}, User Manual filepath suggestion needed...')

                permission = input('|| Insert Filepath? [Y/N]: ')
                if permission == 'Y':
                    suggested_path = input('|| Filepath suggestion: ')
                    if suggested_path:
                        self.pipeline.safe_pickle_save_with_feedback(self.pipeline, suggested_path)
                        print('💾  Model saved!')                
                    else:
                        print('|| Failed to dump Your model! ')
                        pass
                else:
                    print('|| Failed to dump Your model! ')
                    pass  

            verbose = False
            if results[0]['confidence'] < self.pipeline.confidence_threshold:
                verbose = True
            
            self.display_hybrid_results(results, top_k, verbose=verbose)


            # Use results directly - they already contain calibrated predictions
            chosen_label = results[0]['predicted'] if results else None
            confidence = results[0]['confidence'] if results else None

            if isinstance(chosen_label, int) or isinstance(chosen_label, np.integer):
                chosen_label = str(chosen_label)
                
            # Only recalibrate if models disagreed AND we have valid results
            if results and not results[0].get('models_agree', True):
                print("\n[⚠️] Disagreement detected between MLP and Transformer predictions. Using calibrated probabilities for final decision.")
                calibrated_probs = self.pipeline.hybrid_prediction(input_ids, X)
                
                if calibrated_probs is not None and len(calibrated_probs) > 0:
                    final_idx = int(np.argmax(calibrated_probs[:num_classes]))
                    if final_idx > len(reverse_map):
                        final_idx = int(np.argmax(calibrated_probs[:len(reverse_map)-1]))
                        print(f"[⚠️] Clamping {final_idx} → {final_idx}") 

                    final_idx = int(min(final_idx, num_classes - 1))  # Ensure index is within valid range
                    chosen_label = reverse_map.get(final_idx, f"unknown_{final_idx}")
                    try:
                        confidence = float(calibrated_probs[0][final_idx])   
                    except:
                        confidence = float(calibrated_probs[0][len(reverse_map)-1]) if isinstance(calibrated_probs[0], (float, int)) else 0.0             
                            
            if isinstance(chosen_label, str) and chosen_label.startswith("unknown") or confidence < self.pipeline.confidence_threshold:
                print(f"\n[⚠️] Final prediction is {chosen_label} with uncertain confidence: {confidence:.1%}. Consider collecting more data or adjusting the model.")
            else:
                print(f"\n[🎯] Final chosen label for input: {chosen_label} || Confidence: {confidence:.1%}")  
              
        except Exception as e:
            print(f"[=] Error during prediction: {e}")
            results = []

        return results

    def hybrid_model_prediction(self, datasets, X_raw):
        self.pipeline.transformer_utilities(datasets, X_raw)
        input_datasets = self.pipeline.transformer_input_encoding([i[0] for i in datasets])

        probs = self.model.predict_proba(input_datasets, X_raw, type='Hybrid')[0]
        pred = self.model.hybrid_prediction(input_datasets, X_raw)

        return probs, pred

    def robust_prediction(self, pipeline, titles, label_map, show_proba=True, top_k=3):
        try:
            datasets, X_raw = self.pipeline.data_preparation(titles, label_map)
            reverse_map = {v: k for k, v in label_map.items()}
            
            self.pipeline.transformer_utilities(datasets, X_raw)
            input_datasets = self.pipeline.transformer_input_encoding(datasets)
            pred_probs = self.pipeline.predict_proba(input_datasets, X_raw, type='Hybrid')[0]
            pred_result = self.pipeline.hybrid_prediction(input_datasets, X_raw)

            print("\n[🔍] Prediction result structure:")
            print(f"[=] Type: {type(pred_result)}")
            print(f"[=] Length: {len(pred_result) if isinstance(pred_result, tuple) else 1}")

            if isinstance(pred_result, tuple):
                if len(pred_result) == 3:
                    pred_indices = pred_result[0]
                    hybrid_probs = pred_result[1]  # Use different variable name
                    attn_weights = pred_result[2]
                    print("✅ Extracted: indices, probs, attention")
                elif len(pred_result) == 2:
                    pred_indices = pred_result[0]
                    hybrid_probs = pred_result[1]  # Use different variable name
                    print("✅ Extracted: indices, probs")
                else:
                    print(f"⚠️ Unknown tuple format with {len(pred_result)} elements")
                    pred_indices = pred_result[0]
                    hybrid_probs = pred_result[1] if len(pred_result) > 1 else None
            else:
                pred_indices = pred_result
                hybrid_probs = None
                print("✅ Single value return")
        
            # Use hybrid_probs if available, otherwise use pred_probs
            final_probs = hybrid_probs if hybrid_probs is not None else pred_probs
        
            if isinstance(pred_indices, (list, tuple)) and len(pred_indices) > 0:
                if isinstance(pred_indices[0], (np.ndarray, list)):

                    pred_indices = np.array([p[0] if isinstance(p, (np.ndarray, list)) else p 
                                        for p in pred_indices])
                else:
                    pred_indices = np.array(pred_indices)
            elif isinstance(pred_indices, np.ndarray):
                if pred_indices.ndim > 1:
                    pred_indices = pred_indices.flatten()
            else:
                pred_indices = np.array([pred_indices])
        
            print(f"\n[📊] Processed predictions:")
            print(f"[=] pred_indices shape: {pred_indices.shape}")
            print(f"[=] pred_indices: {pred_indices}")
        
            if final_probs is not None:
                print(f"[=] final_probs shape: {final_probs.shape if hasattr(final_probs, 'shape') else 'unknown'}")
        
            if final_probs is not None and isinstance(final_probs, np.ndarray) and final_probs.ndim == 1:
                final_probs = final_probs.reshape(1, -1)
        
            n_samples = len(titles)
            if len(pred_indices) < n_samples:
                print(f"[⚠️] Padding predictions from {len(pred_indices)} to {n_samples}")
                last_idx = pred_indices[-1] if len(pred_indices) > 0 else 0
                pred_indices = np.pad(pred_indices, (0, n_samples - len(pred_indices)), 
                                mode='constant', constant_values=last_idx)

            results = []
            best_idx = -1
            best_confidence = -1
            
            # Determine rows and cols from final_probs
            if final_probs is not None and hasattr(final_probs, 'shape'):
                rows = final_probs.shape[0]
                cols = final_probs.shape[1] if len(final_probs.shape) > 1 else 1
            elif final_probs is not None:
                rows = len(final_probs)
                cols = len(final_probs[0]) if rows > 0 and hasattr(final_probs[0], '__len__') else 1
            else:
                rows, cols = 0, 0
            
            for i in range(n_samples):
                class_idx = int(pred_indices[i]) if i < len(pred_indices) else 0
                    
            if final_probs is not None and i < rows and class_idx < cols:
                if hasattr(final_probs, 'shape'):
                    confidence = final_probs[i, class_idx]
                else:
                    if isinstance(final_probs[i], (list, np.ndarray)):
                        confidence = final_probs[i][class_idx]
                    else:
                        confidence = float(final_probs[i])  # Single value
                        
                if confidence > best_confidence:
                    best_idx = i
                    best_confidence = confidence
                    
            for i, title in enumerate(titles):
                if i < len(pred_indices):
                    class_idx = int(pred_indices[i])
                else:
                    class_idx = 0
                    
                # Get confidence from final_probs
                if final_probs is not None and i < rows and class_idx < cols:
                    if hasattr(final_probs, 'shape'):
                        confidence = final_probs[i, class_idx]
                    else:  # list
                        if isinstance(final_probs[i], (list, np.ndarray)):
                            confidence = final_probs[i][class_idx]
                        else:
                            confidence = float(final_probs[i])  # Single value
                else:
                    # Fallback: use max probability instead of min
                    if final_probs is not None and i < len(final_probs):
                        if isinstance(final_probs[i], (list, np.ndarray)):
                            confidence = max(final_probs[i])
                        else:
                            confidence = float(final_probs[i])
                    else:
                        confidence = 0.0
            
                label = reverse_map.get(class_idx, f"unknown_{class_idx}")


                result = {
                'title': title,
                'predicted': label,
                'confidence': confidence,
                'index': class_idx,
                'is_best': (i == best_idx)
                }
                
                if show_proba and i < rows and cols > 1:
                    if hasattr(final_probs, 'shape'):
                        probs_row = final_probs[i]
                    else:
                        if isinstance(final_probs[i], (list, np.ndarray)):
                            probs_row = np.array(final_probs[i])
                        else:
                            probs_row = np.array([final_probs[i]])
                
                    if len(probs_row) > 1:
                        top_indices = np.argsort(probs_row)[-top_k:][::-1]
                        top_predictions = []
                        for idx in top_indices:
                            if idx in reverse_map:
                                top_predictions.append({
                                'label': reverse_map[idx],
                                'confidence': float(probs_row[idx])
                                })
                        result['top_predictions'] = top_predictions
            
                results.append(result)
        
            print("\n" + "="*70)
            print("[🎯] LABEL PREDICTIONS")
            print("="*70)
        
            for i, result in enumerate(results):
                print(f"\n[📌] Label: {i+1}. '{result['title']}'")
            
                best_marker = "[🏆] BEST" if result.get('is_best') else ""
                print(f"   → {result['predicted']} ({result['confidence']}){best_marker}")
            
                if show_proba and 'top_predictions' in result:
                    print(" [  Top possibilities:")
                    for j, pred in enumerate(result['top_predictions'][:top_k], 1):
                        bar = '█' * int(pred['confidence'] * 20)
                        print(f"      {j}. {pred['label']} {bar} {pred['confidence']} %")
            
            # Return the best result (not inside loop)
            best_idx = int(np.argmax(final_probs[:, pred_indices] if final_probs is not None and hasattr(final_probs, 'shape') else [r['confidence'] for r in results]))
            if best_idx >= 0:
                best_result = results[best_idx]
                if isinstance(best_result['predicted'], str) and best_result['predicted'].startswith("unknown") or best_result['confidence'] < self.pipeline.confidence_threshold:
                    print(f"\n[⚠️] Final prediction is {best_result['predicted']} with uncertain confidence. Consider collecting more data or adjusting the model.")
                else:
                    print(f"\n✨ Most confident: '{best_result['title']}' → {best_result['predicted']} ({best_result['confidence']:.1%})")
                return best_result['predicted'], best_result['confidence'], best_result['confidence']
            elif results:
                # Fallback: return first result if no best found
                predicted = results[0]['predicted']
                predicted_confidence = results[0]['confidence']
                if isinstance(predicted, str) and predicted.startswith("unknown") and predicted_confidence < self.pipeline.confidence_threshold:
                    print(f"\n[⚠️] Final prediction is {predicted} with uncertain confidence: {predicted_confidence:.1%}. Consider more consistent data for the model to learn from.")
                else:
                    print(f"\n[🎯] Final chosen label for input: {predicted} || Confidence: {predicted_confidence:.1%}")  
                
                return predicted, predicted_confidence

        except Exception as e:
            print(f"[=] Error during robust prediction: {e}")
            predicted = None
            predicted_confidence = None
        return predicted, predicted_confidence
        
    def calculate_entropy(self, probs):
        return -np.sum(probs * np.log(probs + 1e-10), axis=-1)


    def advanced_prediction_method(self, titles, label_map, rules,
                                show_proba=False, top_k=3, 
                                use_transformer=True,
                                return_attention=False,
                                save_results=True):
        try:
            eps = 1e-5
            trans_probs = None
            attn_weights = None
            sequence_ids = None

            print("\n[🚀] Starting Advanced Hybrid Prediction Method")

            reverse_map = {v: k for k, v in label_map.items()}
            num_classes = len(label_map)
         
            dataset, X = self.pipeline.data_preparation(titles, label_map)    
            _, y, _, _ = self.pipeline.mlp_training_features(rules, dataset)
            self.pipeline.transformer_utilities(rules, dataset, X)
            input_ids, _ = self.pipeline.input_encoding(dataset)
          
            if use_transformer and hasattr(self.pipeline, 'vocab') and self.pipeline.vocab:
                use_embedded = False
                print("\n[🔄] Running dual predictions (MLP + Transformer)")
            
                input_ids_list = []
                for title in titles:
                    if isinstance(title, tuple):
                        title = title[0]
                    ids = self.pipeline.encode(title, self.pipeline.vocab)
                    input_ids_list.append(np.array(ids))
                
                input_ids = np.array(input_ids_list)
                
                # Get transformer predictions with attention
                if self.pipeline.anisotropy_measurement(input_ids) < self.pipeline.confidence_threshold:
                    print("⚡ Low anisotropy detected on input, relying on sequence encoding for input...")
                    sequence_ids = self.pipeline.sequence_encoding(dataset)
                    use_embedded = True
                    trans_probs, attn_weights = self.pipeline.model2.forward(sequence_ids, embedded=use_embedded)
                else:
                    print("⚡ Anisotropy above threshold, using standard input encoding for transformer...")
                    trans_probs, attn_weights = self.pipeline.model2.forward(input_ids, embedded=use_embedded)    

            else:
                print("\n⚡ Running MLP-only predictions")
                print("⚡ Note: Transformer not available, so Transformer results will be replaced with MLP results.")

            if X is None or len(X) == 0 or isinstance(X, int) or (isinstance(X, np.ndarray) and X.size == 0):
                # Get MLP predictions
                if isinstance(titles[0], tuple):
                    mlp_titles = [t[0] for t in titles]
                else:
                    mlp_titles = titles
                
                if not hasattr(self.pipeline, 'tfidf') or self.pipeline.tfidf is None:
                    self.pipeline.initialize_fitting(mlp_titles)
                                    
                X = self.pipeline.tfidf.transform(mlp_titles).toarray()

            # MLP forward pass
            if hasattr(self.pipeline.mlp, 'predict_proba'):
                mlp_probs = self.pipeline.mlp.predict_proba(X)
            else:
                logits = self.pipeline.mlp.forward(X)
                mlp_probs = self.pipeline._softmax(logits)
            
             # Validate all MLP predictions at once
            mlp_pred_indices = np.argmax(mlp_probs, axis=1)
            if num_classes <= 0:
                num_classes = mlp_probs.shape[1] if mlp_probs.ndim > 1 else len(mlp_probs[0])

            valid_mask = mlp_pred_indices < num_classes
            if not np.all(valid_mask):
                invalid_count = np.sum(~valid_mask)
                # Replace invalid indices with argmax within valid range
                for i in range(len(mlp_pred_indices)):
                    valid_probs = mlp_probs[i][:num_classes] if num_classes > 0 else mlp_probs[i]
                    if len(valid_probs) > 0:
                        mlp_pred_indices[i] = int(np.argmax(valid_probs))
                    else:
                        mlp_pred_indices[i] = 0  # Default to first class  

            if sequence_ids is not None:
                print("\n[🔍] Using sequence encoding for transformer input due to low anisotropy.")
                input_ids = sequence_ids.copy()
            target_probs = self.pipeline.predict_proba(input_ids, X, type='Hybrid', embedded=True)
            target_probs = target_probs[:mlp_probs.shape[0], :mlp_probs.shape[1]] 
            target_pred_indices = np.argmax(target_probs, axis=1)          

            results = []
            attention_data = [] if return_attention else None

            for i, title in enumerate(titles):
                # Parse input
                if isinstance(title, tuple):
                    display_title = title[0]
                    expected_label = title[1] if len(title) > 1 else None
                else:
                    display_title = title
                    expected_label = None
                
                # MLP prediction                 
                mlp_class_idx = mlp_pred_indices[i]
                mlp_class_idx = min(mlp_class_idx, num_classes - 1)  # Clamped to valid range
                if mlp_class_idx < 0 or mlp_class_idx >= num_classes:
                    mlp_class_idx = 0  # Safe default              

                mlp_confidence = mlp_probs[i][mlp_class_idx]
                mlp_label = reverse_map.get(mlp_class_idx, f"unknown_{mlp_class_idx}")

                target_confidence = mlp_confidence
                target_probs = mlp_probs
                target_pred_indices = mlp_pred_indices
                target_class_idx = mlp_class_idx
            
                if mlp_confidence < self.pipeline.confidence_threshold:
                    target_class_idx = target_pred_indices[i]
                    target_confidence = target_probs[i][target_class_idx]
                
                # Transformer prediction and blending
                if trans_probs is not None:
                    trans_probs_i = trans_probs[i]
                    trans_class_idx = np.argmax(trans_probs_i)
                    if isinstance(trans_probs_i, float):
                        trans_confidence = target_confidence
                    else:
                        trans_confidence = trans_probs_i[trans_class_idx]

                    trans_label = reverse_map.get(trans_class_idx, f"unknown_{trans_class_idx}")

                    calibration = self.pipeline._calibrate_probs(target_probs, target_pred_indices, attn_weights, input_ids)
                    # Blend predictions (MLP decides class, transformer calibrates confidence)
                    mlp_weight = mlp_confidence / (target_confidence + trans_confidence + eps)
                    trans_weight = trans_confidence / (target_confidence + trans_confidence + eps)
                        
                    calibration_weighting = calibration[target_class_idx] if target_class_idx < len(calibration) else 0.0
                        
                    # Weighted blend: calibration_weighting * calibrated + (1-weight) * mlp
                    final_probs = mlp_weight * target_probs[i][:len(calibration)] + trans_weight * calibration[i][:len(calibration)]
                 
                    final_class_idx = target_class_idx

                    try:
                        final_confidence = final_probs[final_class_idx]
                    except IndexError:
                        final_confidence = np.max(final_probs) if isinstance(final_probs, np.ndarray) else final_probs

                    if isinstance(final_confidence, np.ndarray):
                        final_confidence = np.max(final_confidence)

                    # Calculate agreement
                    agreement = mlp_class_idx == trans_class_idx
                else:
                    final_probs = mlp_probs[i]
                    final_class_idx = mlp_class_idx
                    final_confidence = mlp_confidence[0] if isinstance(mlp_confidence, np.ndarray) else mlp_confidence
                    if isinstance(final_confidence, np.ndarray) or isinstance(final_confidence, list):
                        final_confidence = np.max(final_confidence)

                    trans_label = None
                    trans_confidence = None
                    agreement = True
                
                final_label = reverse_map.get(final_class_idx, f"unknown_{final_class_idx}")
                # Build result
                result = {
                    'title': display_title,
                    'expected': expected_label,
                    'predicted': final_label,
                    'confidence': float(final_confidence),
                    'index': int(final_class_idx),
                    'mlp_prediction': mlp_label,
                    'mlp_confidence': float(mlp_confidence),
                    'models_agree': bool(agreement)
                }
                
                if trans_label is not None:
                    result['transformer_prediction'] = trans_label
                    result['transformer_confidence'] = float(trans_confidence)
                
                # Add top-k predictions
                final_probs = final_probs[:num_classes] if num_classes > 0 else final_probs
                if show_proba:
                    top_indices = np.argsort(final_probs)[-top_k:][::-1]
                    result['top_predictions'] = [
                        {
                            'label': reverse_map.get(idx, f"unknown_{idx}"),
                            'confidence': float(final_probs[idx])
                        }
                        for idx in top_indices if idx in reverse_map
                    ]
                    
                    # MLP top predictions
                    mlp_probs_i = mlp_probs[i][:num_classes] if num_classes > 0 else mlp_probs[i]
                    mlp_top = np.argsort(mlp_probs_i)[-top_k:][::-1]
                    result['mlp_top'] = [
                        {
                            'label': reverse_map.get(idx, f"unknown_{idx}"),
                            'confidence': float(mlp_probs_i[idx])
                        }
                        for idx in mlp_top if idx in reverse_map
                    ]
                    
                    # Transformer top predictions
                    if trans_probs.ndim > 1:
                        trans_probs = trans_probs[i][:num_classes] if num_classes > 0 else trans_probs[i]
                    else:
                        trans_probs = trans_probs.copy()
                    if trans_probs is not None:
                        trans_top = np.argsort(trans_probs)[-top_k:][::-1]
                        result['transformer_top'] = [
                            {
                                'label': reverse_map.get(idx, f"unknown_{idx}"),
                                'confidence': float(trans_probs[idx])
                            }
                            for idx in trans_top if idx in reverse_map
                        ]
                
                results.append(result)
                
                # Collect attention data if requested
                if return_attention and attn_weights is not None:
                    attention_data.append({
                        'title': display_title,
                        'attention': attn_weights[i].tolist() if i < len(attn_weights) else None
                    })
            
            # Display results
            verbose = False
            if results[0]['confidence'] < self.pipeline.confidence_threshold:
                verbose = True
            
            self.display_hybrid_results(results, top_k, verbose=verbose)
    
            chosen_label = results[0]['predicted'] if results else None
            confidence = results[0]['confidence'] if results else None
            if isinstance(chosen_label, int) or isinstance(chosen_label, np.integer):
                chosen_label = str(chosen_label)

            print(f"\n[🎯] Initial chosen label for input: {chosen_label} || Confidence: {confidence:.1%}")
            time.sleep(3)

            if results[0].get('models_agree', True) and confidence > self.pipeline.confidence_threshold and not chosen_label.startswith("unknown"):
                print(f"\n[🎯] Proper Confidence of Final chosen label for input: {chosen_label} || Confidence: {confidence:.1%}")
                return results, chosen_label, confidence
            
            # Only recalibrate if models disagreed
            elif results and not results[0].get('models_agree', True) or not self.pipeline.agreement:
                need_peer_condition = not results[0].get('models_agree', True) and self.pipeline.peer_assistance_threshold > 0.3
                print("\n[⚠️] Disagreement detected between MLP and Transformer predictions. Using calibrated probabilities for final decision.")
                if need_peer_condition:
                    print('|| Uncertain advanced prediction, requesting peer assistance if allowed...')
                    final_probs = self.pipeline._handle_distributed_connections(final_probs, attn_weights, input_ids, agreement)   

                elif not results[0].get('models_agree', True) and confidence > self.pipeline.confidence_threshold:
                    if final_confidence is not None and confidence < self.pipeline.confidence_threshold:
                        print("\n[⚠️] High confidence detected, but both models don't agree. Using calibrated probabilities for final decision to ensure robustness.")
                        final_probs = self.pipeline.hybrid_prediction(rules, input_ids, dataset)
                        final_idx = final_probs[0].argmax()
                        original_idx = final_idx

                        if final_idx > len(reverse_map):
                            final_idx = int(np.argmax(final_probs[:len(reverse_map)-1]))
                            print(f"[⚠️] Clamping {final_idx} → {final_idx}")               
                        final_idx = int(final_idx)  

                    chosen_label = reverse_map.get(final_idx, f"unknown_{final_idx}")
                    try:
                        confidence = float(final_probs[0][final_idx])   
                    except:
                        confidence = float(final_probs[0][len(reverse_map)-1]) if isinstance(final_probs[0], (float, int)) else 0.0             
                else:
                    print("\n[⚠️] Uncertain confidence and disagreement detected. Using ensemble method for final decision.")
                    input_forward = sequence_ids if sequence_ids is not None else input_ids
                    final_probs, details = self.pipeline.ensemble.predict_ensemble(input_forward, X, y, method='dynamic', embedded=True)
                    final_idx = final_probs[0].argmax()
                    original_idx = final_idx 

                    if final_idx > len(reverse_map):
                        final_idx = int(np.argmax(final_probs[:len(reverse_map)-1]))
                        print(f"[⚠️] Clamping {final_idx} → {final_idx}")               
                    final_idx = int(final_idx)          

                    chosen_label = reverse_map.get(final_idx, f"unknown_{final_idx}")
                    try:
                        confidence = float(final_probs[0][final_idx])   
                    except:
                        confidence = float(final_probs[0][len(reverse_map)-1]) if isinstance(final_probs[0], (float, int)) else 0.0             
                            
            elif confidence < self.pipeline.confidence_threshold and not self.pipeline.agreement and not results[0].get('models_agree', True):
                if trans_probs is not None:
                    prob_entropy = self.calculate_entropy(final_probs)
                    normalized_entropy = prob_entropy / np.log(prob_entropy.shape[-1]) if prob_entropy.shape[-1] > 1 else 0
                    attn_quality = 1.0 / (1.0 + np.exp(-attn_weights.mean()) + eps) if attn_weights is not None else 0.5
                    anisotropy = self.pipeline.anisotropy_measurement(attn_weights.mean() if attn_weights is not None else 0.5)

                else:
                    normalized_entropy = self.calculate_entropy(input_ids)  # Max entropy for uniform distribution
                    attn_quality = 0.05
                    anisotropy = self.anisotropy_measurement(input_ids) if hasattr(self.pipeline, 'anisotropy_measurement') else 0.5

                mean_entropy = np.mean(normalized_entropy)

                use_robust_prediction = (
                anisotropy < 0.3 or
                mean_entropy > 0.5 or  # High uncertainty
                results[0].get('confidence', 0) < 0.4 or  # Low confidence
                not results[0].get('models_agree', True) or  # Disagreement
                attn_quality < 0.4
                )

                if use_robust_prediction:
                    print("\n[⚡] Condition is poorly unviable to handle agreement. Using robust prediction method for better reliability.")
                    predicted_label, confidence = self.robust_prediction(self.pipeline, titles, label_map, show_proba=show_proba, top_k=top_k)
                    if predicted_label is not None:
                        print(f"\n[🎯] Robust prediction result: {predicted_label} with confidence {confidence:.1%}")
                        return _, predicted_label, confidence  
                else:
                    final_idx = final_probs[0].argmax()
                    original_idx = final_idx

                    if final_idx > len(reverse_map):
                        final_idx = int(np.argmax(final_probs[:len(reverse_map)-1]))
                        print(f"[⚠️] Clamping {final_idx} → {final_idx}")                      
                    final_idx = int(final_idx) 

                    chosen_label = reverse_map.get(final_idx, f"unknown_{final_idx}")
                    try:
                        confidence = float(final_probs[0][final_idx])   
                    except:
                        confidence = float(final_probs[0][len(reverse_map)-1]) if isinstance(final_probs[0], (float, int)) else 0.0             
                            
            else:
                print("\n[🎯] Using initial Regular final prediction as final decision.")
                final_idx = final_probs[0].argmax()

                if final_idx > len(reverse_map):
                    final_idx = int(np.argmax(final_probs[:len(reverse_map)-1]))
                    print(f"[⚠️] Clamping {final_idx} → {final_idx}")                    
                final_idx = int(final_idx)

                chosen_label = reverse_map.get(final_idx, f"unknown_{final_idx}")
                try:
                    confidence = float(final_probs[0][final_idx])   
                except:
                    confidence = float(final_probs[0][len(reverse_map)-1]) if isinstance(final_probs[0], (float, int)) else 0.0             
                                                  
            if isinstance(chosen_label, str) and chosen_label.startswith("unknown") or confidence < self.pipeline.confidence_threshold:
                if chosen_label.startswith("unknown"):
                    chosen_label = 'Unknown'
                    confidence = 1.0 - confidence  # Invert confidence for unknown class

                print(f"\n[⚠️] Final prediction is {chosen_label} with uncertain confidence: {confidence:.1%}. Consider more consistent data for the model to learn from.")
            else:
                print(f"\n[🎯] Final chosen label for input: {chosen_label} || Confidence: {confidence:.1%}")  

        except Exception as e:
            print(f"[-] Error in advanced prediction method: {e}")
            results, chosen_label, confidence = None, None, 0.0
        
        return results, chosen_label, confidence
        
        
    def display_hybrid_results(self, results, top_k=3, verbose=False):
        print("\n" + "="*80)
        print("[🎯] == PREDICTION RESULTS == ")
        print("="*80)
        
        correct = 0
        total_with_expected = 0
        
        for idx, result in enumerate(results):
            print(f"\n{idx+1}. 📌 '{result['title']}'")
            
            if result.get('expected'):
                total_with_expected += 1
                status = "[✅]" if result['predicted'] == result['expected'] else "❌"
                print(f"   Expected: {result['expected']} {status}")
                if result['predicted'] == result['expected']:
                    correct += 1
            
            # Agreement indicator
            agree_symbol = "✓" if result.get('models_agree', True) else "⚠️"
            print(f"   {agree_symbol} FINAL: {result['predicted']} ({result['confidence']:.1%})")
            
            # MLP vs Transformer
            print(f"      ├─ [⚡] MLP: {result['mlp_prediction']} ({result['mlp_confidence']:.1%})")
            if result.get('transformer_prediction'):
                arrow = "⬆️" if result['transformer_confidence'] > result['mlp_confidence'] else "⬇️"
                print(f"      └─ [🌀] Transformer: {result['transformer_prediction']} ({result['transformer_confidence']:.1%}) {arrow}")
            
            # Top predictions
            if 'top_predictions' in result:
                print(f"\n [🔍] Top {top_k} possibilities:")
                for j, pred in enumerate(result['top_predictions'][:top_k], 1):
                    bar = '█' * int(pred['confidence'] * 20)
                    print(f"         {j}. {pred['label']:20s} {bar} {pred['confidence']:.1%}")
        
        if total_with_expected > 0:
            accuracy = correct / total_with_expected
            print(f"\n📊 Accuracy: {correct}/{total_with_expected} = {accuracy:.1%}")




def PermissiveTest():
    print("\n" + "="*60)
    print("🔮 = TESTING HYBRID PREDICTION SYSTEM = ")
    print("="*60)

    print("📖 Loading labels from text file with CSV format...")
    filename = input('|| Insert Filename (press N to skip): ')
    title = input('|| Insert Title name you have in your file (press N to skip): ')
    label = input('|| Insert Label name you have in your file (press N to skip): ')

    print('📖 Need to insert custom memory Name for the AI')
    file = input('|| Insert Memory name: ')
    print('📖 Need to insert custom SSL certificate and key files for secure communication')
    print('[=] Important for secure external-device Peer to peer between Agents (optional)')

    cert_file = input('|| Insert SSL certificate file (press N to skip): ')
    key_file = input('|| Insert SSL key file (press N to skip): ')
    if cert_file != 'N':
        cert_file = cert_file
    else:
        cert_file = None
    if key_file != 'N':
        key_file = key_file
    else:
        key_file = None

    if file:
        pipeline = IntegratedPipeline(file, ssl_cert_file=cert_file, ssl_key_file=key_file)
    else:
        print('|| Using original csv_file.pkl file as fallback...')
        pipeline = IntegratedPipeline('csv_file.pkl', ssl_cert_file=cert_file, ssl_key_file=key_file)

    manager = PipelinePredictionManager(pipeline, label_csv='ManualsTraining.txt', target_title='window_title', label='label')

    if filename and title and label and filename != 'N':
        titles, y_raw, label_map = manager.load_labels_from_csv(filename, title, label)
        print(f"✅ Loaded {len(titles)} labeled examples")
    else:
        print('|| Fallback to Original given files...')
        titles, y_raw, label_map = manager.load_labels_from_csv('ManualsTraining.txt', 'window_title', 'label')
        print(f"✅ Loaded {len(titles)} labeled examples")


    print('== Training Model... ==')
    loss_history = pipeline.train(titles, y_raw)

    test_titles = [
    ("Opening Thesis.docx", "slight_work"),
    ("Watching YouTube and Google Chrome", "distracted"),
    ("Watching Slack", "communication"),
    ("Programming in Visual Studio Code", "focused_work"),
    ("Watching netflix.com - Chrome", "break"),
    ]
    rules = [
        # === WORK / PRODUCTIVITY ===
        (r'code|programming|develop|debug|compile|script', 'focused_work'),
        (r'vscode|visual_studio|ide|terminal|shell', 'focused_work'),
        (r'notion|evernote|onenote|notes|todo|task', 'productive'),
        (r'slack|teams|discord|zoom|meeting|call', 'communication'),
        (r'email|gmail|outlook|inbox|mail', 'communication'),
        
        # === ENTERTAINMENT ===
        (r'youtube|netflix|twitch|stream|video', 'entertainment'),
        (r'music|spotify|soundcloud|audio|player', 'entertainment'),
        (r'game|gaming|steam|epic|play', 'gaming'),
        (r'facebook|instagram|tiktok|social|post', 'social_media'),
        
        # === BROWSING ===
        (r'chrome|firefox|edge|safari|browser', 'browsing'),
        (r'google|search|wiki|wiki|article', 'information'),
        (r'stackoverflow|github|docs|documentation', 'research'),
        
        # === FILE MANAGEMENT ===
        (r'download|folder|file|document|pdf', 'file_work'),
        (r'dropbox|onedrive|google_drive|cloud', 'cloud_storage'),
        (r'zip|rar|extract|compress|archive', 'file_management'),
        
        # === SYSTEM / DEV ===
        (r'terminal|cmd|powershell|bash|shell', 'system_work'),
        (r'docker|kubernetes|container|deploy', 'devops'),
        (r'git|commit|push|pull|branch|merge', 'version_control'),
        (r'test|unit|debug|error|exception', 'testing'),
        
        # === DATA / ANALYSIS ===
        (r'excel|spreadsheet|sheet|csv|table', 'data_work'),
        (r'python|r|sql|query|database', 'data_analysis'),
        (r'chart|graph|visualization|dashboard|plot', 'visualization'),
        
        # === COMMUNICATION ===
        (r'whatsapp|telegram|signal|messenger', 'messaging'),
        (r'zoom|meet|webex|video_call', 'video_call'),
        (r'calendar|schedule|event|meeting|appointment', 'scheduling'),
        
        # === CREATIVE ===
        (r'photoshop|illustrator|figma|design|canvas', 'creative'),
        (r'premiere|final_cut|video_edit|render', 'video_editing'),
        (r'blender|3d|model|render|animation', '3d_work'),
        
        # === LEARNING ===
        (r'coursera|udemy|edx|course|learn', 'learning'),
        (r'book|ebook|reader|pdf|document', 'reading'),
        (r'podcast|audiobook|listen|lecture', 'audio_learning'),
        
        # === UTILITY ===
        (r'calculator|converter|tool|utility', 'utility'),
        (r'weather|clock|timer|alarm|reminder', 'utility'),
        (r'translate|language|dictionary|translate', 'utility'),
        
        # === RARITY PATTERNS ===
        (r'common|not_common|twitch|debian|watch', 'very abundant'),
        (r'bit-common|pycharm|unix|code|programming|python|java', 'bit-abundant'),
        (r'medium|discord|teams|zoom|linux_mint|message', 'abundant'),
        (r'rare|pdf|word|macOS|ubuntu|document', 'not abundant'),
        (r'ultra|firefox|edge|browser|unix|web', 'medium rare'),
        (r'ultra_rare|music|linux|Home_linux_router', 'bit-rare'),
        (r'medium-rare|steam|red_hat_enterprise_linux|play|windows', 'very rare'),
        (r'rarer|oracle|system|config|server_linux_router', 'absolute rare'),
    ]

    running = True
    while running:
        permission = input('|| Allow Hybrid prediction test? [Y/N]: ')

        if permission == 'Y' or permission == 'y':
            print('== TEST 1: (titles only without transformer) ==')
            advanced_result = manager.advanced_prediction_method(
            [t[0] for t in test_titles],  # Just titles
            label_map,
            rules,
            show_proba=True
            )
            time.sleep(5)
        
            print('== TEST 2: (advanced predictions with expected labels and also use transformer)')
            advanced_results = manager.advanced_prediction_method(
            test_titles,  # Titles with expected labels
            label_map,
            rules,
            show_proba=True,
            top_k=4,
            use_transformer=True,
            return_attention=True
        
            )
        
            print("\n📊 COMPARISON: MLP-only vs Hybrid")
            mlp_only = manager.regular_prediction_method(
            [t[0] for t in test_titles],
            label_map,
            rules,
            use_transformer=False
            )
        
            hybrid = manager.regular_prediction_method(
            [t[0] for t in test_titles],
            label_map,
            rules,
            use_transformer=True       
            )
            print('== CompletePipeline Successfully tested! ==')
            
        else:
            running = False
            print('|| Program Prediction test aborted!')
            pass


if __name__ == "__main__":
    try:
        PermissiveTest()
    except Exception as e:
        print(f'|| Program Crashed...,  Error: {e}')
        pass

