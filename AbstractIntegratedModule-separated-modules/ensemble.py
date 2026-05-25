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
# ensemble.py
# WeightedEnsemblePredictor: combines MLP and Transformer outputs using
# three strategies — static calibrated weights, dynamic per-sample confidence
# weighting, and meta-learning (stacking).  Also owns the cosine-similarity
# memory cache lookup (attention_memory_gate) backed by ModelStorage.
# Depends on: geometry, mlp, transformer, storage, explainability
# QueryNode is imported lazily inside methods (see inference.py) to break
# the circular dependency:  ensemble → inference → pipeline → ensemble.
# ---------------------------------------------------------------------------
from .geometry import GeometricWeightShaping
from .mlp import MLP
from .transformer import Transformer
from .storage import ModelStorage

class WeightedEnsemblePredictor:
    def __init__(self, pipeline, distribution, memory_name):
        self.pipeline = pipeline
        self.storage = ModelStorage(memory_name, db_path='activity_log.db')
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