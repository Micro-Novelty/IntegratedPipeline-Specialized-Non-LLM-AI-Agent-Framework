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



class PipelinePredictionManager:
    '''
    High-level prediction and evaluation helper that sits on top of
    IntegratedPipeline.

    Responsibilities
    ----------------
    1. Dataset loading     : load_labels_from_csv() reads a delimited file,
                             builds a string→integer label_map, and returns
                             (titles, y_numeric, label_map).

    2. Regular prediction  : regular_prediction_method() runs MLP + optional
                             Transformer in parallel, validates output indices,
                             and prints a ranked probability display.

    3. Advanced prediction : advanced_prediction_method() adds:
                             - Hybrid probability fusion (MLP + Transformer)
                             - AAT-based arbitration on disagreement
                             - Optional top-k output with attention weights
                             - Optional result saving to ModelStorage

    4. Robust prediction   : robust_prediction_method() (internal) handles
                             edge cases (batch padding, fallback confidences,
                             "best" result selection) for production use.

    The manager does NOT own the model weights — it delegates all forward
    passes to self.pipeline.

    Parameters
    ----------
    pipeline     : IntegratedPipeline instance (must be trained before
                   calling prediction methods).
    label_csv    : Path to training CSV (loaded at construction time).
    target_title : Column name for text inputs in the CSV.
    label        : Column name for class labels in the CSV.
    '''
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


    def regular_prediction_method(self, titles, label_map, rules, X=None, y=None, show_proba=False, top_k=3, batch_size=2,use_transformer=True):
        try:
            print(f"\n[🚀] Regular Prediction for labels with {len(titles)} titles...")
            self.pipeline.titles = titles
            self.pipeline.labels = label_map

            reverse_map = {v: k for k, v in label_map.items()}
            num_classes = len(label_map)

            if X is None and y is None or X is None or y is None:
                print('[🔄] Creating automatic X samples because X is not provided manually.')
                dataset, X = self.pipeline.data_preparation(titles, label_map)  
                _, y, _, _ = self.pipeline.mlp_training_features(rules, dataset)                  
            else:
                dataset, _ = self.pipeline.data_preparation(titles, label_map)

            self.pipeline.transformer_utilities(rules, dataset, X, batch_size=batch_size) 
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
            if float(results[0]['confidence']) < self.pipeline.confidence_threshold:
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
                calibrated_probs = self.pipeline.hybrid_prediction(input_ids, X, batch_size=batch_size)
                
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
                            
            if isinstance(chosen_label, str) and chosen_label.startswith("unknown") or float(confidence) < self.pipeline.confidence_threshold:
                print(f"\n[⚠️] Final prediction is {chosen_label} with uncertain confidence: {confidence:.1%}. Consider collecting more data or adjusting the model.")
            else:
                print(f"\n[🎯] Final chosen label for input: {chosen_label} || Confidence: {confidence:.1%}")  

            if results and results[0]['confidence'] > self.pipeline.confidence_threshold:
                results[0]['predicted'] = chosen_label
                results[0]['confidence'] = confidence

        except Exception as e:
            print(f"[=] Error during prediction: {e}")
            results = []

        return results

    def hybrid_model_prediction(self, datasets, X_raw, batch_size=2):
        self.pipeline.transformer_utilities(datasets, X_raw, batch_size=batch_size)
        input_datasets = self.pipeline.transformer_input_encoding([i[0] for i in datasets])

        probs = self.model.predict_proba(input_datasets, X_raw, type='Hybrid')[0]
        pred = self.model.hybrid_prediction(input_datasets, X_raw)

        return probs, pred

    def robust_prediction(self, pipeline, titles, label_map, X_raw=None, y=None, show_proba=True, top_k=3, batch_size=2):
        self.pipeline.titles = titles
        self.pipeline.labels = label_map   

        try:

            if X_raw is None and y is None or X_raw is None or y is None:
                print('[🔄] Creating automatic X samples because X is not provided manually.')
                datasets, X_raw = self.pipeline.data_preparation(titles, label_map)  
            else:
                datasets, _ = self.pipeline.data_preparation(titles, label_map)

            reverse_map = {v: k for k, v in label_map.items()}
            
            self.pipeline.transformer_utilities(datasets, X_raw, y_true=y, batch_size=batch_size)
            input_datasets = self.pipeline.transformer_input_encoding(datasets)
            pred_probs = self.pipeline.predict_proba(input_datasets, X_raw, type='Hybrid')[0]
            pred_result = self.pipeline.hybrid_prediction(input_datasets, X_raw, batch_size=batch_size)

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
                                X=None, y=None,
                                show_proba=False, top_k=3, 
                                use_transformer=True, return_attention=False,
                                save_results=True, batch_size=2):
        try:
            eps = 1e-5
            trans_probs = None
            attn_weights = None
            sequence_ids = None

            print("\n[🚀] Starting Advanced Hybrid Prediction Method")

            reverse_map = {v: k for k, v in label_map.items()}
            num_classes = len(label_map)

            self.pipeline.titles = titles
            self.pipeline.labels = label_map

            if X is None and y is None or X is None or y is None:
                print('[🔄] Creating automatic X samples because X is not provided manually.')
                dataset, X = self.pipeline.data_preparation(titles, label_map)  
                _, y, _, _ = self.pipeline.mlp_training_features(rules, dataset)                  
            else:
                dataset, _ = self.pipeline.data_preparation(titles, label_map)

            self.pipeline.transformer_utilities(rules, dataset, X, y_true=y, batch_size=batch_size)
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
                sequence_ids = self.pipeline.sequence_encoding(dataset)
                if X is not None:
                    anisotropy = self.pipeline.anisotropy_measurement(X)
                else:
                    anisotropy = self.pipeline.anisotropy_measurement(sequence_ids)

                # Get transformer predictions with attention
                print(f"[⚡] anisotropy rate detected on input: {anisotropy:.1%}.")                
                use_embedded = True
                trans_probs, attn_weights = self.pipeline.model2.forward(sequence_ids, embedded=use_embedded)

            else:
                print("\n[⚡] Running MLP-only predictions")
                print("[⚡] Note: Transformer not available, so Transformer results will be replaced with MLP results.")

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

            if self.pipeline.cache and 'label_bins' in self.pipeline.cache:
                print('[=] label_bins cache found!')
                label_bins = self.pipeline.cache['label_bins']
                lstm_probs, _ = self.pipeline.ensemble._get_lstm_probs(input_ids, X, label_bins=label_bins)      
            else:
                lstm_probs = None

            results = []
            attention_data = [] if return_attention else None

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

                if lstm_probs is not None:
                    lstm_pred_indices = np.argmax(lstm_probs, axis=1)
                    lstm_class_idx = lstm_pred_indices[i]              
                    lstm_confidence = lstm_probs[i][lstm_class_idx]
                else:
                    lstm_confidence = None

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
                    if lstm_confidence is not None:
                        lstm_weight = lstm_confidence / (target_confidence + lstm_confidence + eps)
                        
                    calibration_weighting = calibration[target_class_idx] if target_class_idx < len(calibration) else 0.0
                        
                    # Weighted blend: calibration_weighting * calibrated + (1-weight) * mlp
                    if lstm_weight is not None:
                        final_probs = mlp_weight * target_probs[i][:len(calibration)] + trans_weight * calibration[i][:len(calibration)] + lstm_weight * calibration[i][:len(calibration)]
                    else:
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
                    'models_agree': bool(agreement),
                    'sec_predicted': None,
                    'sec_confidence': 0.0,
                    'sec_index': None,
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
                    if trans_probs is not None:
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
            if float(results[0]['confidence']) < self.pipeline.confidence_threshold:
                verbose = True
            
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
                if not self.pipeline.autonomous and need_peer_condition:
                    print('|| Uncertain advanced prediction, requesting peer assistance if allowed...')
                    final_probs = self.pipeline._handle_distributed_connections(final_probs, attn_weights, input_ids, agreement)   
                    final_idx = final_probs[0].argmax()
                    original_idx = final_idx

                    if final_idx > len(reverse_map):
                        final_idx = int(np.argmax(final_probs[:len(reverse_map)-1]))
                        print(f"[⚠️] Clamping {final_idx} → {final_idx}")               
                    final_idx = int(final_idx)  

                    chosen_label = reverse_map.get(final_idx, f"unknown_{final_idx}")
                    try:
                        print(final_probs)
                        confidence = float(final_probs[final_idx])   
                    except:
                        confidence = float(final_probs[0][len(reverse_map)-1]) if isinstance(final_probs[0], (float, int)) else 0.0      
                        
                elif self.pipeline.autonomous and need_peer_condition and attn_weights is not None:
                    if agreement is None:
                        agreement = False

                    print('[||] Iniating local peer output search in database for best output...')
                    final_probs = self.pipeline.distribution._handle_peer_agent_request(final_probs, attn_weights, input_ids, type='DevicePeer', agreement=agreement)
                    final_idx = final_probs[0].argmax()
                    original_idx = final_idx

                    if final_idx > len(reverse_map):
                        final_idx = int(np.argmax(final_probs[:len(reverse_map)-1]))
                        print(f"[⚠️] Clamping {final_idx} → {final_idx}")               
                    final_idx = int(final_idx)  

                    chosen_label = reverse_map.get(final_idx, f"unknown_{final_idx}")
                    try:
                        confidence = float(final_probs[final_idx])   
                    except:
                        confidence = float(final_probs[0][len(reverse_map)-1]) if isinstance(final_probs[0], (float, int)) else 0.0      


                elif not results[0].get('models_agree', True) and confidence > self.pipeline.confidence_threshold:
                    if final_confidence is not None and confidence < self.pipeline.confidence_threshold:
                        print("\n[⚠️] Low confidence detected, but both models don't agree. Using calibrated probabilities for final decision to ensure robustness.")
                        final_probs = self.pipeline.hybrid_prediction(rules, input_ids, dataset)
                        final_idx = final_probs[0].argmax()
                        original_idx = final_idx

                        if final_idx > len(reverse_map):
                            final_idx = int(np.argmax(final_probs[:len(reverse_map)-1]))
                            print(f"[⚠️] Clamping {final_idx} → {final_idx}")               
                        final_idx = int(final_idx)  
                    else:
                        print('[🎯] Stable confidence established, But both Models doesnt Agree, Re-evaluating...')   
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
                    if self.pipeline.use_transformer:
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
                    else:
                        input_forward = sequence_ids if sequence_ids is not None else input_ids
                        final_idx = final_probs[0].argmax() if final_probs is not None else target_probs[0].argmax()

                        original_idx = final_idx 

                        if final_idx > len(reverse_map):
                            final_idx = int(np.argmax(final_probs[:len(reverse_map)-1]))
                            print(f"[⚠️] Clamping {final_idx} → {final_idx}")               
                        final_idx = int(final_idx)          

                        chosen_label = reverse_map.get(final_idx, f"unknown_{final_idx}")
                        if final_probs is None:
                            final_probs = target_probs.copy()

                        try:
                            try:
                                confidence = float(final_probs[0][final_idx])   
                            except:
                                confidence = float(final_probs[0][len(reverse_map)-1]) if isinstance(final_probs[0], (float, int)) else 0.0             
                        except:
                            try:
                                confidence = float(final_probs[final_idx]) 
                            except:
                                confidence = self.pipeline.confidence_threshold

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
                    predicted_label, confidence = self.robust_prediction(self.pipeline, titles, label_map, X_raw=X, y=y, show_proba=show_proba, top_k=top_k)
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
                                                  
            if isinstance(chosen_label, str) and chosen_label.startswith("unknown") or float(confidence) < self.pipeline.confidence_threshold:
                if chosen_label.startswith("unknown"):
                    chosen_label = 'Unknown'
                    confidence = 1.0 - confidence  # Invert confidence for unknown class

                print(f"\n[⚠️] Final prediction is {chosen_label} with uncertain confidence: {confidence:.1%}. Consider more consistent data for the model to learn from.")
            else:
                print(f"\n[🎯] Final chosen label for input: {chosen_label} || Confidence: {confidence:.1%}")  

            try:
                consecutive_probs = self.pipeline.distribution._handle_peer_agent_request(target_probs, attn_weights, input_ids, type='DevicePeer', agreement=agreement)
                sec_final_idx = consecutive_probs[0].argmax()

                if sec_final_idx > len(reverse_map):
                    sec_final_idx = int(np.argmax(consecutive_probs[:len(reverse_map)-1]))
                    print(f"[⚠️] Clamping {sec_final_idx} → {sec_final_idx}")                    
                sec_final_idx = int(sec_final_idx)

                sec_chosen_label = reverse_map.get(sec_final_idx, f"unknown_{sec_final_idx}")
                try:
                    sec_confidence = float(consecutive_probs[0][sec_final_idx])   
                except:
                    sec_confidence = float(consecutive_probs[0][len(reverse_map)-1]) if isinstance(consecutive_probs[0], (float, int)) else self.pipeline.confidence_threshold         

                print('========== Second Prediction Initiative ==========')
                print(f'[⚡] My Second Prediction: {sec_chosen_label}') 
                print(f'[⚡] Confidence: {sec_confidence:.1%}')  

                results[0]['sec_predicted'] = sec_chosen_label
                results[0]['sec_confidence'] = sec_confidence
                results[0]['sec_index'] = sec_final_idx

                if confidence > results[0]['confidence']:
                    results[0]['predicted'] = chosen_label
                    results[0]['confidence'] = confidence
                    results[0]['index'] = final_idx
            except Exception as e:
                print(f'[!] Error initiating second prediction in Advanced prediction method: {e} ')

                results[0]['sec_predicted'] = chosen_label
                results[0]['sec_confidence'] = confidence
                results[0]['sec_index'] = final_idx

                time.sleep(5)

        except Exception as e:
            print(f"[!] Error in advanced prediction method: {e}, Initiating regular prediction method...")
            try:
                results = self.regular_prediction_method(titles, label_map, rules, X=X, y=y, show_proba=False, top_k=3, batch_size=2,use_transformer=True)
                chosen_label = results[0]['predicted']
                confidence = results[0]['confidence']
            except Exception as error:
                print(f'[= ! =] Error in all prediction method: {error}')
                traceback.print_exc()
                results, chosen_label, confidence = None, None, 0.0
                time.sleep(5)

        print('[=] Displaying Results....')     
        correct, sec_correct = self.display_hybrid_results(results, top_k, verbose=True)
        if sec_chosen_label and sec_correct > correct:
            print(f'[⚡] Second Prediction: {sec_chosen_label} has higher accuracies, relying on: {sec_chosen_label} as final label.')
            chosen_label = sec_chosen_label # overrides previous chosen label if accuracy is higher
        if self.pipeline.autonomous and sec_confidence > self.pipeline.confidence_threshold:
            print(f'[⚡] Autonomous Prediction used second predicted label: {sec_chosen_label}')
            chosen_label = sec_chosen_label

        return results, chosen_label, confidence
        
        
    def display_hybrid_results(self, results, top_k=3, verbose=False):
        print("\n" + "="*80)
        print("[🎯] == PREDICTION RESULTS == ")
        print("="*80)
        
        correct = 0
        sec_correct = 0
        total_with_expected = 0
        
        for idx, result in enumerate(results):
            print(f"\n{idx+1}. 📌 '{result['title']}'")
            
            if result.get('expected'):
                total_with_expected += 1
                status = ": ✅" if result['predicted'] == result['expected'] else ": ❌"
                sec_status = ": ✅" if result['sec_predicted'] == result['expected'] else ": ❌"
                print(f"[=] First Expectation: {result['expected']} || Model Answer: {status}")
                print(f"[=] Second Expectation: {result['expected']} || Model Answer: {sec_status}")    

                if result['predicted'] == result['expected']:
                    correct += 1
                if result['sec_predicted'] == result['expected']:
                    sec_correct += 1
            
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

        return correct, sec_correct


