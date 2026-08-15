"""
explainability.py
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
from . import lstm
from . import mlp
from . import transformer

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

        self.supervised_learning = True


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
        X_raw = self._refit_sparse_data(X_raw, raw)

        return datasets, X_raw

    
    def draw_bar(self, value, max_width=20):
        value = max(0, min(1, value))  # Ensure value is between 0 and 1
        filled = int(value * max_width)
        return '█' * filled + '░' * (max_width - filled)


    def _learn_from_feedback(self, text, correct_label, wrong_result, batch_size=2):
        eps = 1e-5
        min_signal = 1e-3
        lr = 0.1

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
        print(f"[=] Training mlp.MLP on corrected example with boosted LR: {self.pipeline.mlp.lr}...")
        
        # 4. train transformer for efficient processing later tho.
        sequence_inputs = self.pipeline._features_to_sequence(X, d_model=self.pipeline.transformer_d_model) 
        should_train_transformer, _ = self.pipeline._should_train_transformer(sequence_inputs, X, min_seq_len=3, min_anisotropy=0.35, min_samples=10, ram_headroom_mb=80)   
        if self.pipeline.model2 and self.pipeline.use_transformer and should_train_transformer:
            transformer_features = self.pipeline.transformer_pooled_features(sequence_inputs)
            X_features = np.concatenate([X, transformer_features], axis=-1)
            x_conditional_anisotropy = self.anisotropy_measurement(sequence_inputs)
            s_conditional_anisotropy = self.anisotropy_measurement(X_features)

            AME_x = self.pipeline.AME_Encoder(X_raw)
            AME_s = self.pipeline.AME_Encoder(sequence_inputs)
            AMR_x = 1.0 / (1.0 + np.exp(-AME_x))
            AMR_s = 1.0 / (1.0 + np.exp(-AME_s))

            AMR_ratio = AMR_x / (AMR_s + min_signal)
            AMR_ratio = AMR_x / (AMR_s + min_signal)
            anisotropy_ratio = x_conditional_anisotropy / (s_conditional_anisotropy + min_signal)
            dynamic_complex_environment = (anisotropy_ratio < 0.5 and 
                                             AMR_ratio < 0.5)                                                         
            if dynamic_complex_environment: 
                print('[+] Dynamic Backward for transformer.Transformer Initiated')
                mode = 'dynamic_backward'
            else:
                print('[=] Fixed Backward for transformer.Transformer initiated')
                mode = 'fixed_backward'

            self.pipeline.model2.train(sequence_inputs, y_onehot, epochs=self.pipeline.transformer_training_epochs, mode=mode, lr=lr, embedded=True, batch_size=2)
        
        # Train on this single example for a few epochs
        self.pipeline.model3.train(X_features, y_onehot, epochs=1000, lr=self.pipeline.mlp.lr, verbose=True, max_samples_for_focused_fit=200)
        self.pipeline.model3.lr = old_lr  # Restore old LR
        # 5. Store in memory gate for fast retrieval
        self.pipeline.modular_prediction_saving(
            self.pipeline.encode(text, self.pipeline.vocab),
            X_features,
            correct_label
        )
        # 6. Add to buffer for batch consolidation later.
        self.feedback_buffer.append((X_features, y_onehot, text, correct_label))
        
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
            self.pipeline.memory[memory_key] = (X_features, correct_label)
        if len(self.learned_from_feedback) % 10 == 0:
            self.consolidate_supervised_memories(batch_size=batch_size)

        return X_features
    
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

        # Train mlp.MLP on batch
        old_lr = self.pipeline.model3.lr
        self.pipeline.model3.lr = old_lr * 2
        self.pipeline.focused_mlp.train(X_batch, y_batch, epochs=1000, lr=self.pipeline.model3.lr, verbose=True, max_samples_for_focused_fit=200)

        self.pipeline.model3.lr = old_lr
        
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
            print(f"Explanation: {explanation}")
            return self._ask_for_feedback(text, result, explanation)
        elif choice == '4':
            history = self.get_decision_history(limit=10)  
            for entry in history:
                print(f" [?] {entry['input']} → {entry['label']}")
        elif choice:
            print('[==] Assigning Training for correct label: ')
            return choice
        return None

    def analyze_with_feedback(self, details, input_text, mlp_probs, trans_probs, attn_weights, explanation, batch_size=2, auto_ask=True):
        uncertain = self.pipeline.confidence_threshold

        input_ids = np.array([self.pipeline.encode(input_text, self.pipeline.vocab)])
        if isinstance(input_ids, list):
            input_ids = np.array(input_ids)

        if uncertain == 0.0:
            uncertain = self.uncertainty_threshold

        is_uncertain = details['final_confidence'] < uncertain
        
        if is_uncertain and self.supervised_learning:
            feedback = self._ask_for_feedback(input_text, details, explanation)
            if feedback:
                print(f"[📚] Received feedback: '{input_text}' should be '{feedback}'")
                print('[=] Supervised learning took many trials to get right. This is normal. Please be patient as the model updates continously each label request...')

                evaluated_input = self._learn_from_feedback(input_text, feedback, details, batch_size=2)
                self.supervised_learning = False  # Prevent infinite loop
                return False
        
        return False

    def _refit_sparse_data(self, X_features, texts, threshold=0.3):
        """Refit TF-IDF if zero-row ratio exceeds threshold."""
        X_features = np.asarray(X_features, dtype=np.float32)
        if X_features.ndim == 1:
            X_features = X_features.reshape(1, -1)        
            X_features = np.asarray(X_features)
                    
        zero_rows = np.where(X_features.sum(axis=1) == 0)[0]
        zero_ratio = len(zero_rows) / len(X_features)
        
        if zero_ratio > threshold:
            print(f'[!] {len(zero_rows)} zero rows ({zero_ratio:.0%}), refitting on current batch')
            if isinstance(texts, str):
                self.tfidf.fit([texts])
                X_features = self.tfidf.transform([texts]).toarray()
            elif isinstance(texts[0], str):
                self.tfidf.fit([texts[0]])
                X_features = self.tfidf.transform([texts[0]]).toarray()
            else:
                X_features = X_features
            
            # second pass — fill remaining zeros with checksum fingerprint
            zero_rows = np.where(X_features.sum(axis=1) == 0)[0]
            for i in zero_rows:
                text = texts[i] if isinstance(texts[i], str) else str(texts[i])
                checksum = int(hashlib.md5(text.encode()).hexdigest(), 16)
                rng = np.random.default_rng(checksum)
                X_features[i] = rng.uniform(0.01, 0.1, size=X_features.shape[1])
                print(f'[!] Row {i} still zero after refit, checksum fallback applied')
        
        return X_features

    def _get_lstm_explanation(self, lstm_result: dict) -> Any:
        """
        Extract readable signals from lstm.LSTMEngine.predict() output.
        lstm_result is the raw dict returned by engine.predict().
        """
        if lstm_result is None:
            return None

        mc_conf_last     = float(lstm_result['mc_confidence'][-1])
        gate_unc_last    = float(lstm_result['gate_uncertainty'][-1])
        overall          = float(lstm_result['overall'])
        interval_low     = float(lstm_result['interval_low'][-1])
        interval_high    = float(lstm_result['interval_high'][-1])
        mc_std_last      = float(lstm_result['mc_std'][-1])

        # gate stability — inverse of uncertainty, easier to read
        gate_stability   = 1.0 - gate_unc_last

        # dominant label from label_confidence if available
        label_conf       = lstm_result.get('label_confidence')
        dominant_label   = None
        dominant_prob    = 0.0
        if label_conf:
            dominant_label = max(label_conf, key=label_conf.get)
            dominant_prob  = label_conf[dominant_label]

        return {
            'mc_confidence'   : mc_conf_last,     # how tight MC dropout samples are
            'gate_stability'  : gate_stability,    # 1 = stable memory, 0 = actively overwriting
            'gate_uncertainty': gate_unc_last,     # raw gate signal
            'overall'         : overall,           # combined scalar
            'interval'        : (interval_low, interval_high),  # prediction interval
            'mc_std'          : mc_std_last,       # spread of MC samples
            'dominant_label'  : dominant_label,    # top label_bin if bins were passed
            'dominant_prob'   : dominant_prob,
            'label_confidence': label_conf
        }

    def consolidate_supervised_memories(self, batch_size=2):
        if not self.learned_from_feedback:
            return
        
        print(f"\n[🔄] Consolidating {len(self.learned_from_feedback)} supervised memories...")
        
        # Extract all supervised examples
        texts = [m['input'] for m in self.learned_from_feedback]
        labels = [m['label'] for m in self.learned_from_feedback]

        dataset, _ = self.data_preparation(texts, labels)
        self.initialize_fitting(texts)

        X = self.tfidf.transform(texts).toarray()
        X = self._refit_sparse_data(X, texts)

        try:
            unique_labels = sorted(set(labels))
            label_to_idx  = {l: i for i, l in enumerate(unique_labels)}
            y_indices     = np.array([label_to_idx[l] for l in labels])

            n_classes = len(unique_labels)
            y_onehot  = np.zeros((len(y_indices), n_classes))
            y_onehot[np.arange(len(y_indices)), y_indices] = 1

            self.pipeline.model3.train(X, y_onehot, epochs=100, lr=0.01, verbose=True)
            
            print("[✅] Supervised memories consolidated!")
        except Exception as e:
            print(f"[❌] Error during memory consolidation: {e}")

    
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
 

    def _get_prediction_details(self, input_text, mlp_probs, trans_probs, attn_weights, lstm_result=None, batch_size=2):
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
            'lstm': self._get_lstm_explanation(lstm_result) if lstm_result is not None else None,
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
                self.analyze_with_feedback(details, input_text, mlp_probs, trans_probs, attn_weights, explanation, batch_size=2)
 
        confidence = self.explain_confidence(details)
        if final_conf:
            print('[||] Final confidence set to: ', final_conf)
            self.pipeline.final_conf_score = final_conf

        return details, explanation
    
    
    def _get_final_output(self, mlp_pred, mlp_conf, trans_pred, trans_conf, attn_weights):
        eps = 1e-5
        consensus_conf = 0.0 

        if isinstance(mlp_conf, np.ndarray):
            mlp_conf = np.clip(np.mean(mlp_conf), 0, 1)
        if isinstance(trans_conf, np.ndarray):
            trans_conf = np.clip(np.mean(trans_conf), 0, 1)

        if mlp_pred == trans_pred:
            final_pred = mlp_pred
            consensus_conf = max(mlp_conf, trans_conf)
        else:
            sliced_attention_weight = attn_weights[0]
            if isinstance(sliced_attention_weight, np.ndarray):
                sliced_attention_weight = sliced_attention_weight[:, 0]
                sliced_attention_weight = sliced_attention_weight[0]
               
            sliced_anisotropy = self.pipeline.anisotropy_measurement(sliced_attention_weight) 
            sigmoid_growth = 1.0 / (1.0 + np.exp(-sliced_attention_weight))
            attn_quality = self._compute_attention_quality(attn_weights)

            # Abstract attention transformation
            AAT = sigmoid_growth * (1.0 - sliced_anisotropy) + eps 
            # lower AAT means transformer is less reliable because abstraction is underserved/nonoptimal in this env.
            # Higher AAT means transformer is more focused and reliable and is near optimal.

            if mlp_conf > trans_conf:
                final_pred = mlp_pred
                consensus_conf = mlp_conf * (1.0 - trans_conf) * (1.0 - np.mean(AAT)) + eps
            else:
                final_pred = trans_pred
                consensus_conf = trans_conf * (1.0 - mlp_conf) * np.mean(AAT) + eps

            print('='*50)
            print('===== ABSTRACTION LAYER ======')
            print('='*50)
            print(f'[= ABSTRACTION =] Consistency of abstraction transformation: {np.std(AAT)}')
            print(f'[= ABSTRACTION =] Attention Quality: {attn_quality}')
            print(f'[= ABSTRACTION =] Sigmoid growth of Attention weight consistency: {np.std(sigmoid_growth)}')
            print('[=] Note: Very little Consistency meaning transformer.Transformer attention quality is Healthy and focused')

        if isinstance(consensus_conf, np.ndarray):
            consensus_conf = 1.0 / (1.0 + np.exp(-consensus_conf))
            # Apply a activations.sigmoid transformation to ensure the confidence is between 0 and 1

        # averaged all confidences to get the final confidence.
        final_conf = mlp_conf + trans_conf + consensus_conf / 3
        if np.isnan(final_conf).any() or np.isinf(final_conf).any():
            final_conf = self.pipeline.confidence_threshold

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
        if _OPT_AVAILABLE and np.asarray(X).ndim == 2:
            return optimized_ame_encoder(np.asarray(X, dtype=np.float64))     
            
        if len(X) == 0:
            print('[!] X size is 0, AME Will be replaced by minimum confidence threshold')
            return self.pipeline.confidence_threshold
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
            return self.pipeline.confidence_threshold
        
        try:
    
            if hasattr(self.pipeline, 'anisotropy_measurement'):
                return self.pipeline.anisotropy_measurement(attn_weights.flatten())
            
            # Fallback calculation
            attn_flat = attn_weights.flatten()
            gradient = np.gradient(attn_flat)
            val = [np.linalg.norm(v) for v in gradient]
            return np.std(val) / (np.mean(val) + 1e-8)

        except:
            return self.pipeline.confidence_threshold
    
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
        except Exception as e:
            print(f"[-] Error occurred while computing attention quality: {e}")
            AMR = self.pipeline.confidence_threshold
            if attn_weights is not None:
                print(f"[-] Attention weights shape: {attn_weights.shape}")
                AME = self.AME_Encoder(attn_weights)
                AMR = 1.0 / (1.0 + np.exp(-AME) + eps)            
            return AMR
    
    def _generate_explanation(self, details):
        parts = []
        
        # Final decision
        parts.append(f"📌 Decision: I think my prediction is: **{details['final_label']}**")
        parts.append(f"[=] Confidence Degree: {details['final_confidence']}\n")
        
        # mlp.MLP's geometric reasoning
        parts.append("🧠 Geometric mlp.MLP Reasoning:")
        parts.append(f"   • Detected Detail: {', '.join(details['geometric_features']['dominant_features'][:3])}")
        parts.append(f"   • Geometric complexity signature: {details['geometric_features']['anisotropy']:.3f}")
        parts.append(f"   • Energy: signature {details['geometric_features']['feature_energy']:.3f}")
        parts.append(f"   • Confidence Focus: {details['mlp']['confidence']:.1%} to → {details['mlp']['label']}")

        if details.get('lstm') is not None:
            lstm = details['lstm']
            parts.append("\n⏳ LSTM Memory Reasoning:")
            parts.append(f"   • MC Dropout Confidence: {lstm['mc_confidence']:.1%} "
                        f"(spread: ±{lstm['mc_std']:.4f})")
            parts.append(f"   • Gate Stability: {lstm['gate_stability']:.1%} "
                        f"({'stable memory' if lstm['gate_stability'] > 0.6 else 'actively rewriting memory — uncertain transition'})")
            parts.append(f"   • Prediction Interval: [{lstm['interval'][0]:.4f}, {lstm['interval'][1]:.4f}]")
            parts.append(f"   • Overall LSTM Confidence: {lstm['overall']:.1%}")
            if lstm['dominant_label']:
                parts.append(f"   • Strongest Sequence Signal: {lstm['dominant_label']} "
                            f"({lstm['dominant_prob']:.1%})") 

        # transformer.Transformer's contextual reasoning
        if self.pipeline.use_transformer:
            parts.append("\n🌀 transformer.Transformer Reasoning:")
            if details['transformer']['attention_words']:
                parts.append(f"   • Focused on: '{', '.join(details['transformer']['attention_words'])}'")
            parts.append(f"   • Attention quality: {details.get('attention_quality', 0.5)}")
            parts.append(f"   • Attention anisotropy: {details.get('anisotropy', 0.5):.3f}")
            parts.append(f"   • Confidence Focus: {details['transformer']['confidence']:.1%} to → {details['transformer']['label']}")

        # Agreement analysis
        lstm = details.get('lstm')
        if details['agreement']:
            parts.append("\n✅ Models Agreed:")
            parts.append("   Both geometric and contextual analysis point to the same conclusion")
            if lstm and lstm['gate_stability'] > 0.6:
                parts.append("[=+=] LSTM memory is stable — sequence history supports this decision")
            else:
                parts.append("[!] LSTM Uncertain - Sequence history does not supports this decision")
        else:
            if self.pipeline.use_transformer:
                parts.append("\n⚠️ Models Disagreed:")
                parts.append(f"   Geometric mlp.MLP Focusing on → {details['mlp']['label']} detail")
                parts.append(f"   transformer.Transformer Focusing on → {details['transformer']['label']} detail")
                if lstm:
                    stability_note = "reinforces" if lstm['gate_stability'] > 0.6 else "is uncertain about"
                    parts.append(f"   LSTM {stability_note} the sequence context "
                                f"(gate stability: {lstm['gate_stability']:.1%})")
                parts.append(f"   I weighted them with {details['final_confidence']:.1%} "
                            f"confident in {details['final_label']}")                
            else:
                parts.append("🌀 Supporting Argument From LSTM:")
                if lstm:
                    stability_note = "reinforces" if lstm['gate_stability'] > 0.6 else "is uncertain about"
                    parts.append(f"   LSTM {stability_note} the sequence context "
                                f"(gate stability: {lstm['gate_stability']:.1%})")                
                parts.append(f"   Geometric mlp.MLP Focusing on → {details['mlp']['label']} detail")
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
        
        comparison.append(f"🔄 Ensemble Decision Comparison")
        comparison.append("====================================")
        
        comparison.append("[<] Ensemble Earlier Decision:")
        comparison.append(f"[+] Input: {d1['input']}")
        comparison.append(f"[+] Detail Focus: {d1['prediction']} ({d1['confidence']:.1%})")
        
        comparison.append("🧠 Ensemble Later Decision:")
        comparison.append(f"[=] Input: {d2['input']}")
        comparison.append(f"[=] Detail Focus: {d2['prediction']} ({d2['confidence']:.1%})")
        
        comparison.append("🔬 Ensemble Learning Progress: ")
        comparison.append(f"• Confidence {'increased' if d2['confidence'] > d1['confidence'] else 'decreased'} from {d1['confidence']:.1%} to {d2['confidence']:.1%}")
        comparison.append(f"• The model is becoming {'more' if d2['confidence'] > d1['confidence'] else 'less'} certain")
        
        return '\n'.join(comparison)
    
    def explain_confidence(self, details):

        factors = []
        
        # Check mlp.MLP confidence
        if details['mlp']['confidence'] > 0.8:
            factors.append(f"✅ mlp.MLP is very confident ({details['mlp']['confidence']:.1%}) due to strong geometric patterns")
        elif details['mlp']['confidence'] < 0.5:
            factors.append(f"🤔 mlp.MLP is uncertain ({details['mlp']['confidence']:.1%}) due to ambiguous geometric patterns")
        
        # Check transformer confidence
        if details['transformer']['confidence'] > 0.8:
            factors.append(f"✅ transformer.Transformer is confident ({details['transformer']['confidence']:.1%}) with focused attention")
        elif details['transformer']['confidence'] < 0.5:
            factors.append(f"🤔 transformer.Transformer is uncertain with uncertainty up to: ({details['transformer']['confidence']:.1%}), due to scattered attention")
        
        # Check agreement
        if details['agreement']:
            factors.append("[✅] Both Models agree, reinforcing confidence")
        else:
            factors.append("[⚠️] Both Models disagree, reducing overall confidence")
        
        # Attention quality
        if details.get('attention_quality', 0) > 0.7:
            factors.append(f"[✅] High attention quality: ({details['attention_quality']:.1%}) indicates clear consistent patterns!")
        elif details.get('attention_quality', 0) < 0.3:
            factors.append(f"[-] Low Attention Quality: ({details['attention_quality']:.1%}) Indicates inconsistent and ambiguous patterns on seen data!")
        
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


