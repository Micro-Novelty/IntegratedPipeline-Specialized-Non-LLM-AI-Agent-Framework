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



class WeightedEnsemblePredictor:
    '''
    Combines MLP and Transformer predictions using dynamically tuned weights,
    with an attention-based memory cache and an integrated ExplainabilityModule.

    Ensemble weights
    ----------------
    transformer_weight + mlp_weight = 1.0 (maintained by the caller).
    Initial split is 50/50.  find_optimal_ensemble_weights() searches over a
    grid of weight values and picks the split that maximises accuracy on a
    provided validation set.

    Attention memory gate
    ---------------------
    attention_memory_gate() checks self.memory for a previously seen input
    whose TF-IDF cosine similarity to the current input is ≥ 0.85.  If found,
    the cached attention outputs are returned directly (skipping the forward
    pass), acting as a lightweight associative memory for repeated inputs.

    Explainability
    --------------
    self.explainer (ExplainabilityModule) is used to generate human-readable
    prediction explanations that surface the dominant TF-IDF features,
    attention focus words, and the MLP vs Transformer arbitration logic.

    Parameters
    ----------
    pipeline      : IntegratedPipeline instance.
    distribution  : AgentDistributedInference instance (for peer calibration).
    memory_name   : DB scope for storing attention snapshots.
    '''
    def __init__(self, pipeline, distribution, memory_name):
        self.pipeline = pipeline
        self.storage = ModelStorage(pipeline, memory_name, db_path='activity_log.db')
        self.inference = distribution
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


    def _get_lstm_probs(self, input_ids, X_mlp, label_bins=None, confidence_level=0.90):
        """
        Convert LSTMEngine prediction output into a probability
        distribution compatible with trans_probs and mlp_probs
        for use in _dynamic_weighted_ensemble.

        Returns:
            lstm_probs : np.ndarray (batch_size, n_classes) or None if engine not ready
            lstm_weight_hint : float — calibrated confidence scalar for weighting
        """
        engine = self.pipeline.lstm_engine

        # guard — engine must be calibrated before predict() is callable
        if engine is None or engine.residual_std is None:
            print('[-] LSTM engine not ready, skipping LSTM probs')
            return None, 0.0

        # X_mlp is (batch, features) — LSTM expects (T, input_size) per sample
        # treating each feature vector as a single timestep sequence
        if X_mlp.ndim == 1:
            X_mlp = X_mlp.reshape(1, -1)

        batch_size = X_mlp.shape[0]
        n_classes = self.pipeline.model2.output.shape[1]  # align with transformer output

        lstm_probs = np.zeros((batch_size, n_classes))
        overall_confidences = []

        for i in range(batch_size):
            # shape each sample into (T=1, input_size) — single timestep
            x_seq = X_mlp[i].reshape(1, -1)

            try:
                result = engine.predict(
                    x_seq,
                    label_bins=label_bins,
                )
            except AssertionError:
                # calibrate() not called yet — skip gracefully
                print(f'[-] LSTM engine not calibrated for sample {i}, skipping')
                return None, 0.0

            # result['prediction'] is (T,) — take last timestep as scalar score
            raw_score = float(result['prediction'][-1])
            overall_confidences.append(result['overall'])

            # convert scalar score to class probabilities
            # label_confidence gives {label: prob} if label_bins were passed
            if result['label_confidence'] is not None:
                # map label_bins order to class indices
                label_probs = list(result['label_confidence'].values())
                n_label = len(label_probs)

                row = np.zeros(n_classes)
                row[:min(n_label, n_classes)] = label_probs[:n_classes]

                # renormalize in case n_label != n_classes
                row_sum = row.sum()
                if row_sum > 0:
                    row /= row_sum
                else:
                    row[0] = 1.0  # fallback — assign all mass to class 0

            else:
                # no label_bins — use mc_confidence at last timestep as a
                # soft signal: spread probability mass using mc_mean as logit
                mc_mean_last = float(result['mc_mean'][-1])
                mc_conf_last = float(result['mc_confidence'][-1])

                # build a soft peaked distribution using the score as a logit
                logits = np.full(n_classes, -mc_conf_last)
                target_class = int(np.clip(round(raw_score), 0, n_classes - 1))
                logits[target_class] = mc_conf_last

                # softmax
                logits -= logits.max()
                row = np.exp(logits)
                row /= row.sum()

            lstm_probs[i] = row

        # weight hint — average overall confidence across batch
        # lower residual_std = engine is well calibrated = higher weight earned
        mean_overall = float(np.mean(overall_confidences))
        lstm_weight_hint = mean_overall / (1.0 + engine.residual_std)

        return lstm_probs, lstm_weight_hint

    def attention_memory_gate(self, probs, x):
        # Fast-path cache lookup: checks whether a previously seen input (stored under
        # prefix 'TA' in self.memory) is geometrically similar to the current input x.
        # Similarity is measured by cosine similarity ≥ 0.85 (tight threshold to avoid
        # false hits on unrelated inputs that happen to share some features).
        #
        # If a match is found, the cached attention outputs (texts, x2, x3, x4) are
        # returned directly, skipping a full forward pass through the transformer.
        # This also acts as a continual memory mechanism: the pipeline "remembers"
        # past attention patterns and reuses them for similar future inputs.
        #
        # Cache miss path:
        #   - If self_attn_weights was set by a prior call, return it as a warm fallback.
        #   - Otherwise return (None, None, None, None) signalling a full inference needed.
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

    def predict_single(self, text, mlp_probs, trans_probs, attn_weights, show_explanation=True, batch_size=2):
        # small batch size is used to prevent memory overflow in explanation module when processing large attention weights, as it computes detailed explanations that can be memory intensive.
        cache = self.pipeline.cache
        if cache is not None and "lstm_result" in cache:
            lstm_result = self.pipeline.cache['lstm_result']
            result, explanation = self.explainer._get_prediction_details(text, mlp_probs, trans_probs, attn_weights, lstm_result=lstm_result, batch_size=batch_size)
        else:
            result, explanation = self.explainer._get_prediction_details(text, mlp_probs, trans_probs, attn_weights, batch_size=batch_size)            
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

    def _dynamic_weighted_ensemble(self, trans_probs, mlp_probs, attn_weights, input_ids,
                                    lstm_weight_hint):
        # Per-sample dynamic weighting of Transformer and MLP predictions.
        # paired with LSTM proof of credibility and prediction boost to calibrate ensemble weight
        # Unlike the static self.transformer_weight / self.mlp_weight used in
        # calibrate_weights(), this method derives weights on-the-fly from three signals:
        #
        #   trans_conf_factor  — derived from attention statistics:
        #                         attn_focus    = std of the attention map (0 = flat, high = peaked)
        #                         attn_growth   = sigmoid(attn_focus) — bounded confidence signal
        #                         attn_limit    = (1 - attn_focus + attn_growth) * anisotropy
        #                         factor        = attn_growth + attn_limit * attn_focus
        #                        Intuitively: the transformer earns more weight when its attention
        #                        is peaked (focused) AND the distribution is geometrically varied.
        #
        #   mlp_conf_factor    — derived from MLP output entropy:
        #                         lower entropy → sharper distribution → higher confidence → higher weight.
        #                         formula: 1 / (1 + entropy)
        #
        #   agreement          — 1.0 if both models predict the same class, else 0.3.
        #                        Acts as a confidence multiplier: agreement boosts both weights
        #                        proportionally, disagreement dampens the overall contribution.
        #
        # Both factors are multiplied by (1 + agreement) / 2, then normalised so they sum to 1.
        # The final ensemble for sample i is: trans_weight * trans_row + mlp_weight * mlp_row + lstm_weight * lstm_row
        batch_size = trans_probs.shape[0]
      
        try:
            n_trans_classes = trans_probs.shape[1]
            n_mlp_classes = mlp_probs.shape[1]
            n_lstm_classes = lstm_probs.shape[1]
        except:
            n_trans_classes = trans_probs.shape[-1]
            n_mlp_classes = mlp_probs.shape[-1]   
            n_lstm_classes = lstm_probs.shape[-1]     

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

            if lstm_probs is not None:
                lstm_row = np.zeros(n_classes)
                lstm_row[:n_lstm_classes] = lstm_probs[i]
                lstm_row = lstm_row / (lstm_row.sum() + 1e-8)

                # lstm_weight_hint already encodes residual_std confidence
                lstm_pred   = np.argmax(lstm_probs[i])
                lstm_agreement = 1.0 if lstm_pred == trans_pred or lstm_pred == mlp_pred else self.pipeline.confidence_threshold
                lstm_weight = lstm_weight_hint * (1.0 + lstm_agreement) / 2

                total = trans_weight + mlp_weight + lstm_weight + 1e-8
                trans_weight /= total
                mlp_weight   /= total
                lstm_weight  /= total

                ensemble[i] = trans_weight * trans_row + mlp_weight * mlp_row + lstm_weight * lstm_row
            else:
                # original two-model path — unchanged
                total = trans_weight + mlp_weight + 1e-8
                trans_weight /= total
                mlp_weight   /= total
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
        # Second-level ("stacking") ensemble.  Instead of computing weights from raw
        # attention or entropy signals, it builds a meta-feature vector for each sample
        # that summarises both models' outputs and their relationship, then derives
        # sample-specific weights from those features.
        #
        # Meta-features per sample (up to 7 values):
        #   [0] max(trans_row)             — transformer peak confidence
        #   [1] max(mlp_row)               — MLP peak confidence
        #   [2] std(trans_row)             — transformer output spread (uncertainty proxy)
        #   [3] std(mlp_row)               — MLP output spread
        #   [4] 1.0 if both agree, else 0  — inter-model agreement flag
        #   [5] std(attn[i])               — attention map spread (if available)
        #   [6] max(attn[i])               — peak attention value (if available)
        #
        # Weight derivation:
        #   base_weight = (threshold_feature + AME_sigmoid) * agreement  → lower base_weight (disagree) higher base_weight (agree) 
        #   the agree or disagreement happens before multiplication with agreement, meaning the subtitution provides a clear signal of data
        #   while multiplication with agreement means absolute,it will either absolutely agree or disgree, this part is used for amplification.
        #   Whichever model has higher confidence gets base_weight;
        #   the other gets 1 - base_weight.
        #
        # NOTE: there is a scoping bug here — trans_row / mlp_row from the loop above
        # are used outside the loop in the weight application (line ~1582).  On the last
        # iteration they hold values for sample batch_size-1, but for earlier iterations
        # the wrong row is applied.  Flagged in code review.
        batch_size = trans_probs.shape[0]
        n_classes = trans_probs.shape[1]
        threshold_feature = 0.1 + self.pipeline.confidence_threshold

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
            
            # Re-normalise after zero-padding to maintain valid probability distributions.
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
                    features.extend([threshold_feature, threshold_feature])
            else:
                features.extend([threshold_feature, threshold_feature])
            
            meta_features.append(features)
        
        meta_features = np.array(meta_features) 
        featured_AME = self.pipeline.AME_Encoder(meta_features) 
        AME_sigmoid = 1.0 / (1.0 + np.exp(-featured_AME))

        ensemble = np.zeros_like(trans_probs)
        
        for i in range(batch_size):
            # Calculate weight based on meta features
            trans_conf = meta_features[i, 0]
            mlp_conf = meta_features[i, 1]
            agreement = meta_features[i, 4]
            
            # Boost weight when models agree
            base_weight = threshold_feature + AME_sigmoid * agreement
            
            
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

        # Weight calibration methods
        best_weight = 0.1 + self.pipeline.confidence_threshold
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

# Cross-session automation module: export/import sessions, sync with another device,
# and list available sessions for continuity across environments.
