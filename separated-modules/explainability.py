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



class ExplainabilityModule:
    '''
    Generates human-readable explanations for pipeline predictions and
    implements an optional supervised-learning feedback loop.

    Explanation output
    ------------------
    explain() returns two values:
      details     — dict with keys: final_label, final_confidence, mlp (pred+conf),
                    transformer (pred+conf), attention_focus (top-3 words),
                    geometric_features (anisotropy, AME, dominant TF-IDF terms),
                    attention_quality.
      explanation — formatted multi-line string printed in a "reasoning chain"
                    style (MLP geometric reasoning → Transformer context → final
                    decision with confidence bar).

    Supervised feedback loop
    ------------------------
    When self.supervised_learning = True and a prediction falls below
    uncertainty_threshold (default 0.2), the module asks the user for the
    correct label via _ask_for_feedback(), then calls _learn_from_feedback()
    to perform a one-shot online weight update on the MLP.

    Feedback is buffered (buffer_size = 10) before triggering a batch backward
    pass, reducing oscillation from single-sample updates.

    consolidate_supervised_memories() can be called explicitly to merge all
    buffered feedback into a full training cycle.

    Parameters
    ----------
    pipeline   : IntegratedPipeline instance (used for feature extraction and
                 model forward passes).
    predictor  : WeightedEnsemblePredictor instance (provides ensemble weights
                 and peer calibration access).
    '''
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

        # Derive a geometry-aware learning rate for the correction step.
        # anisotropy_dist: sigmoid of anisotropy — saturates to 1 for strongly directional data.
        # deviation:       inverse of std; near 1 when features are tightly clustered (low spread).
        # AEL (Adaptive Error Level): high when data is variable (low deviation) AND anisotropic.
        #   AEL → 1  ⟹  corrective LR = 2/(1+1) = 1.0  (fast correction on complex data)
        #   AEL → 0  ⟹  corrective LR = 2/(1+0) = 2.0  (even faster on flat/simple data)
        # This intentionally boosts the correction LR above the normal training LR so
        # a single wrong prediction can be overridden quickly without many epochs.
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
        
        if is_uncertain and self.supervised_learning:
            feedback = self._ask_for_feedback(input_text, details, explanation)
            if feedback:
                print(f"[📚] Received feedback: '{input_text}' should be '{feedback}'")
                print('[=] Supervised learning took many trials to get right. This is normal. Please be patient as the model updates continously each label request...')

                evaluated_input = self._learn_from_feedback(input_text, feedback, details)
                self.supervised_learning = False  # Prevent infinite loop
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
        X = self._refit_sparse_data(X, texts)
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


    def _refit_sparse_data(self, X_features, texts, threshold=0.3):
        """Refit TF-IDF if zero-row ratio exceeds threshold."""
        X_features = np.asarray(X_features, dtype=np.float32)
        if X_features.ndim == 1:
            X_features = X_features.reshape(1, -1)        
            X_features = np.asarray(X_features)
                    
        zero_rows = np.where(X_features.sum(axis=1) == 0)[0]
        zero_ratio = len(zero_rows) / len(X_features)
        
        if zero_ratio > threshold:
            print(f'[!] {len(zero_rows)} zero rows ({zero_ratio:.0%}), refitting TF-IDF on current batch')
            self.tfidf.fit(texts)
            X_features = self.tfidf.transform(texts).toarray()
            
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
        Extract readable signals from LSTMEngine.predict() output.
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


    
    def _get_final_output(self, mlp_pred, mlp_conf, trans_pred, trans_conf, attn_weights):
        # Resolves the final prediction when the two models disagree.
        # When they agree, the higher-confidence model's score is taken directly.
        # When they disagree, an "Abstract Attention Transformation" (AAT) scalar
        # is computed to determine which model to trust more:
        #
        #   sliced_anisotropy — directional variation in the first attention slice;
        #                        high → attention is non-uniform / informative.
        #   deviation         — 1/(1 + std(attn_weights)); near 1 when attention is tightly
        #                        concentrated, near 0 when it is spread out.
        #   attn_quality      — overall quality score from attention_quality_computing.
        #   AAT               — deviation * (1 - sliced_anisotropy):
        #                        high when attention is concentrated (low anisotropy) AND
        #                        tightly distributed (low spread); this configuration favours
        #                        the transformer's focused contextual prediction.
        #
        # Confidence blending on disagreement:
        #   If MLP wins:         final_conf = mlp_conf * (1 - trans_conf) * (1 - AAT)
        #       → lower AAT (diffuse attention) → MLP gets more room to dominate.
        #   If Transformer wins: final_conf = trans_conf * (1 - mlp_conf) * AAT
        #       → higher AAT (focused attention) → transformer earns a larger share.
        eps = 1e-5
        if isinstance(mlp_conf, np.ndarray):
            mlp_conf = np.clip(np.mean(mlp_conf), 0, 1)
        if isinstance(trans_conf, np.ndarray):
            trans_conf = np.clip(np.mean(trans_conf), 0, 1)

        if mlp_pred == trans_pred:
            final_pred = mlp_pred
            final_conf = max(mlp_conf, trans_conf)
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
                final_conf = mlp_conf * (1.0 - trans_conf) * (1.0 - np.mean(AAT)) + eps
            else:
                final_pred = trans_pred
                final_conf = trans_conf * (1.0 - mlp_conf) * np.mean(AAT) + eps

            print('='*50)
            print('===== ABSTRACTION LAYER ======')
            print('='*50)
            print(f'[= ABSTRACTION =] Consistency of abstraction transformation: {np.std(AAT)}')
            print(f'[= ABSTRACTION =] Attention Quality: {attn_quality}')
            print(f'[= ABSTRACTION =] Sigmoid growth of Attention weight consistency: {np.std(sigmoid_growth)}')
            print('[=] Note: Very little Consistency meaning Transformer attention quality is Healthy and focused')

        if isinstance(final_conf, np.ndarray):
            final_conf = 1.0 / (1.0 + np.std(final_conf))
            # growth deviation of arrayed final confidence helped to distinguish noise from unnecessary distribution, 
            # with real covariance of distribution from the data.

        if np.isnan(final_conf).any() or np.isinf(final_conf).any():
            final_conf = self.pipeline.confidence_threshold

        return final_pred, final_conf
    
    def _get_attention_focus(self, attn_weights, text):
        # get focus of attention words in transformer
        if attn_weights is None or len(attn_weights) == 0:
            return text.split()[:3]
        
        words = text.lower().split()
        attn = attn_weights[0].mean(axis=0) if len(attn_weights[0].shape) > 1 else attn_weights[0]
        top_indices = np.argsort(attn)[-3:][::-1]
        if attn.ndim > 1:
            attn = attn.flatten()

        # top most focused words
        top_indices = np.argsort(attn)[-3:][::-1]
        
        focus_words = []
        for idx in top_indices:
            if hasattr(idx, 'item'):
                idx = idx.item()
            
            if isinstance(idx, (int, np.integer)) and idx >= 0 and idx < len(words):
                focus_words.append(words[idx])
        
        return focus_words if focus_words else words[:3]
    
    def _get_geometric_features(self, text):
        # get geometric features of MLP and transformer to explain its decisions.
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
        # function that calculates abstract modelling rate
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
        # function to compute anisotropy
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
        
        # Final decision
        parts.append(f"📌 Decision: I think my prediction is: **{details['final_label']}**")
        parts.append(f"[=] Confidence Degree: {details['final_confidence']}\n")
        
        # MLP's geometric reasoning
        parts.append("🧠 Geometric MLP Reasoning:")
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

        # Transformer's contextual reasoning
        if self.pipeline.use_transformer:
            parts.append("\n🌀 Transformer Reasoning:")
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
                parts.append(f"   Geometric MLP Focusing on → {details['mlp']['label']} detail")
                parts.append(f"   Transformer Focusing on → {details['transformer']['label']} detail")
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
                parts.append(f"   Geometric MLP Focusing on → {details['mlp']['label']} detail")
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
