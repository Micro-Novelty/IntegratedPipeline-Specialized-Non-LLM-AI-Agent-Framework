"""
model_storage.py
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
from . import transformer

class ModelStorage:
    def __init__(self, pipeline, memory_name, db_path='activity_log.db'):
        self.pipeline = pipeline
        self.db_path = db_path

        self.setup_storage_table()
        self.setup_explainable_table()
        self.setup_agent_table()
        self.setup_node_table()
        self.setup_weight_table()
        self.setup_accurate_cache_table()

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
            print(f"[🔄] Running as EXE, temp path: {application_path}")
        else:
            application_path = os.path.dirname(os.path.abspath(__file__))
            print(f"[🔄] Running as script, path: {application_path}")
    
        db_path = os.path.join(application_path, db_filename)
        print(f"[🔄] Looking for database at: {db_path}")
        print(f"[✅] Database exists: {os.path.exists(db_path)}")
    
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


    def setup_weight_table(self):
        try:
            try:
                db_path = self.get_database_path()            
                conn = sqlite3.connect(db_path)
            except:
                conn = sqlite3.connect(self.db_path)

            c = conn.cursor()
    
            c.execute('''CREATE TABLE IF NOT EXISTS weight_storage
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      memory_name TEXT,
                      model_type TEXT,
                      weights TEXT,
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
            c.execute('''CREATE TABLE IF NOT EXISTS weight_storage
                      (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      memory_name TEXT,
                      model_type TEXT,
                      weights TEXT,
                      is_active INTEGER DEFAULT 0,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''') 
        
        
            conn.commit()
            conn.close()


    def setup_accurate_cache_table(self):
        try:
            try:
                db_path = self.get_database_path()            
                conn = sqlite3.connect(db_path)
            except:
                conn = sqlite3.connect(self.db_path)

            c = conn.cursor()
    
            c.execute('''CREATE TABLE IF NOT EXISTS accurate_cache_storage
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      memory_name TEXT,
                      cache TEXT,
                      is_active INTEGER DEFAULT 0,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''') 
        
            conn.commit()
            conn.close()
            print('|| Update cached to database! ')

        except Exception as e:
            print(f'|| Cant Update Database: {e}')
            filepath = input('|| Insert Database filepath: ')
            if filepath:
                conn = sqlite3.connect(filepath)
            else:
                print('|| Skipping Database Modification...')
                pass
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS accurate_cache_storage
                      (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      memory_name TEXT,
                      cache TEXT,
                      is_active INTEGER DEFAULT 0,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''') 
        
        
            conn.commit()
            conn.close() 

    def save_accurate_cache_dict(self, memory_name, payload, model_type='Pipeline'):
        try:
            db_path = self.get_database_path()
            conn = sqlite3.connect(db_path)
        except:
            conn = sqlite3.connect(self.db_path)

        c = conn.cursor()

        cache = json.dumps(payload, default=str)

        try:
            c.execute("""
                INSERT INTO accurate_cache_storage
                (memory_name, cache, is_active)
                VALUES (?, ?, ?)
            """, (memory_name, cache, 1))
        
            c.execute("""
                UPDATE accurate_cache_storage
                SET is_active = 0 
                WHERE memory_name = ? AND id != last_insert_rowid()
            """, (memory_name,)) 

            conn.commit()
            conn.close()

            print('|| Accurate cache saved!')

        except Exception as e:
            print(f'[-] Cant save accurate cache memory due to: {e}') 
            pass      

   

    def save_model_dict(self, memory_name, model_dict, type=None, model_type='mlp'):
        try:
            db_path = self.get_database_path()            
            conn = sqlite3.connect(db_path)
        except:
            conn = sqlite3.connect(self.db_path)

        c = conn.cursor()

        model_dict = self.pipeline._sanitize_for_storage(model_dict)  
        serializable_dict = self._prepare_for_serialization(model_dict)
        model_json = json.dumps(serializable_dict, default=str)

        if _RUST_MODULE_AVAILABLE:
            try:
                wc.save_pipelines_dict(self.db_path, memory_name, 
                                    model_type, model_json)
                print('[=] Pipelines dictionary saved using Rust module!')
            except Exception as e:
                print(f'[!] Cant save Pipelines dictionary due to: {e}')
                
        else:
            if type == 'transformer.Transformer':
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

    def _prepare_for_serialization(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: self._prepare_for_serialization(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._prepare_for_serialization(item) for item in obj]
        return obj
        


    def load_model_dict(self, memory_name):
        conn = None
        if _RUST_MODULE_AVAILABLE:
            try:
                num_classes = self.pipeline._get_num_classes() or 0
                cleaned_json = wc.load_and_validate_model_dict(
                    self.db_path, memory_name, num_classes
                )
                if cleaned_json is None:
                    return None
                data = json.loads(cleaned_json)
                # the heavy JSON parse + basic schema check already done in Rust
                validation = self._validate_and_repair(data, memory_name)
                return validation
            except Exception as e:
                print(f'[!] Rust load_model_dict failed: {e}')
        else:
            try:
                try:
                    conn = sqlite3.connect(self.db_path)
                except:
                    conn = sqlite3.connect(self.get_database_path())

                c = conn.cursor()
                c.execute("""
                    SELECT model_data FROM model_storage
                    WHERE memory_name = ? AND is_active = 1
                    ORDER BY id DESC LIMIT 1
                """, (memory_name,))

                result = c.fetchone()
                if not result:
                    return None

                data = json.loads(result[0])
                data = self._validate_and_repair(data, memory_name)
                return data   # actually return data

            except Exception as e:
                print(f'[!] Error loading model dict: {e}')
                return None
            finally:
                if conn:
                    conn.close()


    def _validate_and_repair(self, data, memory_name=None):
        """Validate loaded data and repair if corrupted."""

        if data is None:
            return {}

        # get num_classes safely without touching pipeline
        num_classes = 0
        try:
            # try pipeline first
            if hasattr(self, 'pipeline') and self.pipeline is not None:
                num_classes = self.pipeline._get_num_classes()
            # fallback — infer from data itself
            elif isinstance(data, dict):
                for v in data.values():
                    if isinstance(v, (list, np.ndarray)):
                        arr = np.asarray(v)
                        if arr.ndim == 1:
                            num_classes = len(arr)
                            break
        except Exception as e:
            print(f'[!] Could not determine num_classes: {e}')
            num_classes = 0

        # handle list data
        if isinstance(data, list) and len(data) > 0:
            if num_classes > 0 and len(data) != num_classes:
                print(f'[!] Shape mismatch: got {len(data)}, expected {num_classes} — repairing')
                return {}

            if all(isinstance(x, (int, float)) for x in data[:10]):
                print(f'[!] List appears to be probabilities, wrapping')
                return {'_cached_probs': np.array(data, dtype=np.float64)}

        # handle dict data
        if isinstance(data, dict):
            repaired = {}
            for key, value in data.items():

                # dynamic corruption check 
                if isinstance(value, list):
                    arr = np.asarray(value)
                    if arr.ndim > 2:
                        print(f'[!] Corrupted value for key {key} '
                            f'(ndim={arr.ndim}), removing')
                        continue

                    if num_classes > 0 and arr.ndim == 1 and \
                    len(arr) not in (num_classes, num_classes * 2):
                        print(f'[!] Suspicious shape for key {key}: '
                            f'{arr.shape}, expected {num_classes} — removing')
                        continue

                # None values — skipped silently
                if value is None:
                    continue

                repaired[key] = value

            return repaired

        return data


    def _convert_to_arrays(self, data):
        """
        Recursively convert data to numpy arrays where possible.
        Safe for ARM64 and handles all data types.
        """
        if data is None:
            return None
        
        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                converted = self._convert_value(value)
                if converted is not None:
                    result[key] = converted
            return result
        
        elif isinstance(data, (list, tuple)):
            return [self._convert_value(item) for item in data]
        
        else:
            return self._convert_value(data)



    def _convert_value(self, value):
        """
        Convert a single value to appropriate type.
        Returns original value if conversion fails.
        """
        if value is None:
            return None
        
        # Already numpy array - keep as is
        if isinstance(value, np.ndarray):
            return value
        
        # Handle lists recursively
        if isinstance(value, (list, tuple)):
            return [self._convert_value(item) for item in value]
        
        # Handle dicts recursively
        if isinstance(value, dict):
            return self._convert_to_arrays(value)
        
        # Handle string that might represent an array
        if isinstance(value, str):
            return self._parse_array_string(value)
        
        # Return as-is for other types (int, float, bool, etc.)
        return value


    def _parse_array_string(self, s):
        """
        Parse string representation of array back to numpy array.
        Returns original string if parsing fails.
        """
        if not isinstance(s, str) or not s:
            return s

        if _RUST_MODULE_AVAILABLE:
            try:
                data = wc.parse_array_string(self.db_path, s)
                print('[+] Data successfully parsed!')
                return data
            except Exception as e:
                print(f'[!] Data cant be parsed due to: {e}')
                s = s.replace('\n', '').replace('\r', '').replace('\t', '')
                s = ' '.join(s.split()).strip()
                
                if not s:
                    return s
                
                # parsing as JSON array first
                if s.startswith('[') and s.endswith(']'):
                    try:
                        parsed = json.loads(s)
                        if isinstance(parsed, list):
                            return np.array(parsed, dtype=np.float32)
                    except (json.JSONDecodeError, ValueError):
                        pass
                    
                    # Try parsing with ast.literal_eval
                    try:
                        parsed = ast.literal_eval(s)
                        if isinstance(parsed, (list, tuple)):
                            return np.array(parsed, dtype=np.float32)
                    except (ValueError, SyntaxError, TypeError):
                        pass
                
                # parsing space-separated numbers
                if re.fullmatch(r'[\[\]\s\d\.\,\-\+E]+', s):        
                    parts = s.replace('[', ' ').replace(']', ' ').split()
                    if parts:
                        try:
                            float_values = [float(x) for x in parts]
                            return np.array(float_values, dtype=np.float32)
                        except ValueError:
                            pass
                
                # Handle comma-separated values
                if ',' in s:
                    cleaned = s.replace('[', '').replace(']', '').strip()
                    parts = [p.strip() for p in cleaned.split(',') if p.strip()]
                    try:
                        float_values = [float(x) for x in parts]
                        return np.array(float_values, dtype=np.float32)
                    except ValueError:
                        pass  

                return s         
        else:
            # Clean the string
            s = s.replace('\n', '').replace('\r', '').replace('\t', '')
            s = ' '.join(s.split()).strip()
            
            if not s:
                return s
            
            # parsing as JSON array first
            if s.startswith('[') and s.endswith(']'):
                try:
                    parsed = json.loads(s)
                    if isinstance(parsed, list):
                        return np.array(parsed, dtype=np.float32)
                except (json.JSONDecodeError, ValueError):
                    pass
                
                # Try parsing with ast.literal_eval
                try:
                    parsed = ast.literal_eval(s)
                    if isinstance(parsed, (list, tuple)):
                        return np.array(parsed, dtype=np.float32)
                except (ValueError, SyntaxError, TypeError):
                    pass
            
            # parsing space-separated numbers
            if re.fullmatch(r'[\[\]\s\d\.\,\-\+E]+', s):        
                parts = s.replace('[', ' ').replace(']', ' ').split()
                if parts:
                    try:
                        float_values = [float(x) for x in parts]
                        return np.array(float_values, dtype=np.float32)
                    except ValueError:
                        pass
            
            # Handle comma-separated values
            if ',' in s:
                cleaned = s.replace('[', '').replace(']', '').strip()
                parts = [p.strip() for p in cleaned.split(',') if p.strip()]
                try:
                    float_values = [float(x) for x in parts]
                    return np.array(float_values, dtype=np.float32)
                except ValueError:
                    pass
        
        # Return original string if nothing worked
        return s


    def _convertables_utility(self, memory_name, data, data2, type_func=None, verbose=False):
        """
        Convert and display memory data safely.
        Returns tuple (result, result2) always for consistent return type.
        """
        name = memory_name
        
        # Initialize results
        result = None
        result2 = None
        
        # Convert data based on type_func
        if type_func == "TwoPass" and data2 is not None:
            print('|| Two pass utility converting.')
            result = self._convert_to_arrays(data)
            result2 = self._convert_to_arrays(data2)
        else:
            result = self._convert_to_arrays(data)
        
        # Verify result is a dictionary before calling .items()
        if verbose and result is not None:
            print(f"[=] Retrieved memory: {name}")
            
            # ✅ SAFE: Check if result is a dict before iterating
            if isinstance(result, dict):
                for key, value in result.items():
                    self._print_memory_value(key, value)
            else:
                print(f"[!] Result is not a dict: {type(result)}")
                print(f"[!] Result length: {len(result) if hasattr(result, '__len__') else 'N/A'}")
        
        # Handle TwoPass verbose output
        if verbose and data2 is not None and result2 is not None:
            print(f"[=] Retrieved secondary memory: {name}_secondary")
            if isinstance(result2, dict):
                for key, value in result2.items():
                    self._print_memory_value(key, value)
            else:
                print(f"[!] Secondary result is not a dict: {type(result2)}")
        
        # ✅ ALWAYS return consistent types
        if data2 is not None:
            return result, result2
        else:
            return result


    def _print_memory_value(self, key, value):
        # Helper method to print memory values safely
        if isinstance(value, list):
            print(f"  {key}: list of {len(value)} items")
            for i, v in enumerate(value[:5]):  # Limit to first 5 items
                if isinstance(v, np.ndarray):
                    print(f"    [{i}]: array shape {v.shape}")
                else:
                    print(f"    [{i}]: {type(v)}")
            if len(value) > 5:
                print(f"    ... and {len(value) - 5} more items")
        
        elif isinstance(value, np.ndarray):
            print(f"  {key}: array shape {value.shape}, dtype={value.dtype}")
        
        elif isinstance(value, dict):
            print(f"  {key}: dict with {len(value)} keys")
        
        else:
            print(f"  {key}: {type(value)}")


    def memory_retrieval(self, memory_name=None, type_func=None, verbose=False):  
        name = memory_name

        if type_func == 'transformer.Transformer':
            data = self.load_transformer_dict(name)
        elif type_func == 'Peer':
            id_history = self.id_history
          
            first_data, second_data = self.load_peer_request_dict(name, id_history) 
            result, result2 = self._convertables_utility(name, first_data, second_data, type_func='TwoPass', verbose=verbose)
            return result, result2
        elif type_func == 'Node':
            data = self._load_node_dict(name)            
        else:
           data = self.load_model_dict(memory_name)

        if data is None:
            print(f"[-] No memory found: {name}")
            return {}

        result = self._convertables_utility(name, data, None, type_func=type_func, verbose=verbose)

        return result


    def load_accurate_cache(self, memory_name):
        try:
            try:
                db_path = self.get_database_path()            
                conn = sqlite3.connect(db_path)
            except:
                conn = sqlite3.connect(self.db_path)

            c = conn.cursor()      
            
            c.execute("""
            SELECT weights FROM accurate_cache_storage 
            WHERE memory_name = ? AND is_active = 1
            """, (memory_name,))               
        
            result = c.fetchone()
            conn.close()
        
            if result:
                return json.loads(result[0])
        except Exception as e:
            print(f'[!] Error handling cache dict: {e}')
        return None


    def _load_weights(self, memory_name, type=None):
        try:
            try:
                db_path = self.get_database_path()            
                conn = sqlite3.connect(db_path)
            except:
                conn = sqlite3.connect(self.db_path)

            c = conn.cursor()
 
            c.execute("""
            SELECT weights FROM weight_storage 
            WHERE memory_name = ? AND is_active = 1
            """, (memory_name,))               
        
            result = c.fetchone()
            conn.close()
        
            if result:
                return json.loads(result[0])
        except Exception as e:
            print(f'[!] Error handling Weight dict: {e}')
        return None


    def weight_retrieval(self, memory_name=None, type=None, verbose=False):  
        name = memory_name

        data = self._load_weights(memory_name, type=type)
        if data is None:
            print(f"[-] No Saved Weight found: {name}")
            return None

        result = self._convertables_utility(name, data, None, type_func='firstpass', verbose=verbose)

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
            print(f'[!] Error handling node dict: {e}')
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


    def save_weights(self, memory_name, model_type=None):
        """Save weights to database."""
        try:
            db_path = self.get_database_path()
            conn = sqlite3.connect(db_path)
        except:
            conn = sqlite3.connect(self.db_path)

        c = conn.cursor()

        weights = {
            'lstm_W'  : self.pipeline.network_model.cell.W.tolist(),
            'lstm_b'  : self.pipeline.network_model.cell.b.tolist(),
            'Wy' : self.pipeline.network_model.Wy.tolist() if self.pipeline.network_model.Wy is not None else None,
            'by' : self.pipeline.network_model.by.tolist(),
            'residual_mean': self.pipeline.lstm_engine.residual_mean,
            'residual_std' : self.pipeline.lstm_engine.residual_std,
            'quantiles'    : {str(k): list(v) 
                            for k, v in self.pipeline.lstm_engine.quantiles.items()} 
                            if self.pipeline.lstm_engine.quantiles else {},
            'n_samples'    : self.pipeline.lstm_engine.n_samples,
            'saved_at'     : datetime.now().isoformat(),
        }
        
        weight_json = json.dumps(weights, default=str)
        if _RUST_MODULE_AVAILABLE:
            try:
                wc.save_lstm_weights(self.db_path, memory_name, weight_json)
                print('[||] LSTM weights saved using Rust module !')
                return
            except Exception as e:
                print(f'[!] Rust save failed, falling back to Python: {e}')
        else:
            print('[=] Rust module unavailable, using python sqlite3.')

        try:
            c.execute("""
                INSERT INTO weight_storage 
                (memory_name, model_type, weights, is_active)
                VALUES (?, ?, ?, ?)
            """, (memory_name, model_type, weight_json, 1))
        
            c.execute("""
                UPDATE weight_storage 
                SET is_active = 0 
                WHERE memory_name = ? AND id != last_insert_rowid()
            """, (memory_name,)) 
            
            c.execute("""
                DELETE FROM weight_storage
                WHERE memory_name = ?
                AND model_type = 'Pipeline'
                AND is_active = 0
            """, (memory_name,))            

            conn.commit()
            conn.close()

            self.save_transformer_weights(memory_name)
            print('[||] All Weights dictionary saved!')

        except Exception as e:
            print(f'[-] Cant save Weights due to: {e}')
            pass          


    def save_transformer_weights(self, memory_name: str):
        """Save transformer weights as compressed binary blob."""
        tf = self.pipeline.model2

        try:
            db_path = self.get_database_path()
            conn = sqlite3.connect(db_path)
        except:
            conn = sqlite3.connect(self.db_path)

        c = conn.cursor()

        buf = io.BytesIO()
        np.savez_compressed(buf,
            token_embedding = tf.token_embedding,
            pos_embedding   = tf.pos_embedding,
            W_q             = tf.W_q,
            W_k             = tf.W_k,
            W_v             = tf.W_v,
            W_q_fixed       = tf.W_q_fixed,
            W_k_fixed       = tf.W_k_fixed,
            W_v_fixed       = tf.W_v_fixed,
            W_o             = tf.W_o,
            ffn1            = tf.ffn1,
            ffn2            = tf.ffn2,
            ln1_scale       = tf.ln1_scale,
            ln1_shift       = tf.ln1_shift,
            ln2_scale       = tf.ln2_scale,
            ln2_shift       = tf.ln2_shift,
            output          = tf.output,
            output_bias     = tf.output_bias
        )
        binary_data = buf.getvalue()
        if _RUST_MODULE_AVAILABLE:
            try:
                wc.save_transformer_weights(self.db_path, memory_name, binary_data)
                print('[||] LSTM weights saved using Rust module for flexibility!')
                return
            except Exception as e:
                print(f'[!] Rust save failed, falling back to Python: {e}')
        else:
            print('[=] Rust module unavailable, using python sqlite3.')

        try:
            c.execute("""
                INSERT INTO weight_storage
                (memory_name, model_type, weights, is_active)
                VALUES (?, ?, ?, ?)
            """, (memory_name, 'transformer', 
                sqlite3.Binary(binary_data), 1))

            c.execute("""
                UPDATE weight_storage SET is_active = 0
                WHERE memory_name = ?
                AND model_type = 'transformer'
                AND id != last_insert_rowid()
            """, (memory_name,))

            c.execute("""
                DELETE FROM weight_storage
                WHERE memory_name = ?
                AND model_type = 'transformer'
                AND is_active = 0
            """, (memory_name,))

            conn.commit()
            print('[||] transformer.Transformer weights saved!')

        except Exception as e:
            print(f'[!] transformer.Transformer weight save failed: {e}')
            conn.rollback()
        finally:
            conn.close()

    def load_transformer_weights(self, memory_name: str) -> bool:
        try:
            db_path = self.get_database_path()
            conn = sqlite3.connect(db_path)
        except:
            conn = sqlite3.connect(self.db_path)

        c = conn.cursor() 


        try:
            if _RUST_MODULE_AVAILABLE:
                try:
                    binary = wc.load_transformer_weights(self.db_path, memory_name) 

                    buf = io.BytesIO(bytes(binary))
                    data = np.load(buf, allow_pickle=False)

                    print('[+] transformer.Transformer weights data loaded using Rust module!')

                except Exception as e:
                    print(f'[=] Cant load transformer.Transformer weights: {e}, using python sqlite3 to handle weights.')
                    c.execute("""
                        SELECT weights FROM weight_storage
                        WHERE memory_name = ? AND model_type = 'transformer' AND is_active = 1
                        ORDER BY id DESC LIMIT 1
                    """, (memory_name,))
                    row = c.fetchone()
                    if not row:
                        print(f'[=] No saved transformer weights for {memory_name}')
                        return False

                    buf = io.BytesIO(bytes(row[0]))
                    data = np.load(buf, allow_pickle=False)

            else:
                c.execute("""
                    SELECT weights FROM weight_storage
                    WHERE memory_name = ? AND model_type = 'transformer' AND is_active = 1
                    ORDER BY id DESC LIMIT 1
                """, (memory_name,))
                row = c.fetchone()
                if not row:
                    print(f'[=] No saved transformer weights for {memory_name}')
                    return False

                buf = io.BytesIO(bytes(row[0]))
                data = np.load(buf, allow_pickle=False)

            t = self.pipeline.model2
            t.token_embedding = data['token_embedding']
            t.pos_embedding   = data['pos_embedding']
            t.W_q             = data['W_q']
            t.W_k             = data['W_k']
            t.W_v             = data['W_v']
            t.W_q_fixed       = data['W_q_fixed']
            t.W_k_fixed       = data['W_k_fixed']
            t.W_v_fixed       = data['W_v_fixed']
            t.W_o             = data['W_o']
            t.ffn1            = data['ffn1']
            t.ffn2            = data['ffn2']
            t.ln1_scale       = data['ln1_scale']
            t.ln1_shift       = data['ln1_shift']
            t.ln2_scale       = data['ln2_scale']
            t.ln2_shift       = data['ln2_shift']
            t.output          = data['output']
            t.output_bias     = data['output_bias']

            print(f'[||] transformer.Transformer weights loaded!')
            return True

        except Exception as e:
            print(f'[!] transformer.Transformer weight load failed: {e}')
            return False
        finally:
            conn.close()


    def load_weights(self, memory_name):
        """Load weights from database. Returns True if found."""
        if _RUST_MODULE_AVAILABLE:
            try:
                result = wc.load_lstm_weights(self.db_path, memory_name)
                print('[+] LSTM weights loaded via Rust module')
            except Exception as e:
                result = self.weight_retrieval(memory_name)
                print(f'[=] Cant load LSTM Weights due to: {e}, using python sqlite3 as fallback.')
                  
        else:
            result = self.weight_retrieval(memory_name)           
    
        if not result:
            print(f'[=] No saved weights for {memory_name}')
            return False

        try:
            weights = result
            if isinstance(weights, str):
                weights = json.loads(weights)
            
            self.pipeline.network_model.cell.W  = np.array(weights.get('lstm_W'))
            self.pipeline.network_model.cell.b  = np.array(weights.get('lstm_b'))
            self.pipeline.network_model.Wy      = np.array(weights.get('Wy')) if weights.get('Wy') else None
            self.pipeline.network_model.by      = np.array(weights.get('by'))
            self.pipeline.lstm_engine.residual_mean = weights.get('residual_mean', 0.0)
            self.pipeline.lstm_engine.residual_std  = weights.get('residual_std',  1.0)
            self.pipeline.lstm_engine.n_samples     = weights.get('n_samples', self.pipeline.lstm_engine.n_samples)
            self.pipeline.lstm_engine.quantiles     = {float(k): tuple(v) 
                                for k, v in weights.get('quantiles', {}).items()}

            tf_loaded = self.load_transformer_weights(memory_name)

            print(f'[=] transformer.Transformer weights loaded: {tf_loaded}')
            if tf_loaded:
                print(f'[=] All Weights loaded for {memory_name}  '
                    f'(saved at {weights.get("saved_at", "unknown")})')

        except Exception as e:
            print(f'[!] Cant load any Weights due to: {e}')
            traceback.print_exc()


    def load_transformer_dict(self, memory_name):
        try:
            try:
                conn = sqlite3.connect(self.db_path)
            except:
                db_path = self.get_database_path()
                conn = sqlite3.connect(db_path)   

            if _RUST_MODULE_AVAILABLE:
                try:
                    data = wc.load_attention_dict(self.db_path, memory_name)
                    print('[+] transformer.Transformer attention loaded using Rust module!')
                    return data
                except Exception as e:
                    print(f'[=]: {e}, Loading transformer attention from python sqlite3...')
            else:     
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
            print(f'[!] Error handling attention dict: {e}')

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

        

    def fix_corrupted_memory(self, memory_name):
        # Clear corrupted memory entries
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            # Deactivate corrupted entries
            c.execute("""
                UPDATE model_storage 
                SET is_active = 0 
                WHERE memory_name = ? AND is_active = 1
            """, (memory_name,))
            
            c.execute("""
                UPDATE model_attn_storage 
                SET is_active = 0 
                WHERE memory_name = ? AND is_active = 1
            """, (memory_name,))
            
            conn.commit()
            conn.close()
            
            print(f'[✅] Cleared corrupted memory for {memory_name}')
            return True
        except Exception as e:
            print(f'[!] Failed to clear memory: {e}')
            return False


    def load_agent_id(self, memory_name):
        try:

            try:
               db_path = self.get_database_path()
               conn = sqlite3.connect(db_path)  
            except:
                conn = sqlite3.connect(self.db_path)

            if _RUST_MODULE_AVAILABLE:
                try:
                    agent_data = wc.load_agent_id(self.db_path, memory_name)
                    print('[+] Got Agent ID Data from Database!')
                except Exception as e:
                    print(f'[!] Cant load agent ID data from DB: {e}')
                    return None    
            else:
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
            print(f'[-] Error loading ID from database: {e}')

        return None        


    def memory_exists(self, memory_name, type=None):

        conn = None
        try:
            try:
                db_path = self.get_database_path()
                conn = sqlite3.connect(db_path)               
            except:
                conn = sqlite3.connect(self.db_path)

            if _RUST_MODULE_AVAILABLE:
                try:
                    exists = wc.verify_memory_exist(self.db_path, memory_name, type)
                    print(f'[=] Memory exist: {exists}')
                    return exists
                except:
                    pass
            else:
                if type == 'transformer.Transformer':
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

                elif type == 'Accurate-Cache':
                    c = conn.cursor()

                    c.execute("""
                    SELECT 1 FROM accurate_cache_storage
                    WHERE memory_name = ? and is_active = 1
                    LIMIT 1""", (memory_name, ))

                    result = c.fetchone()
                    exists = result is not None
                    print(f"|| Retrieved Accurate Fact Cache Memory for memory: {memory_name}")
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
            print(f"[!] Database error: {e}")
            return False
            
        except Exception as e:
            print(f"[!] Unexpected error in handling memory: {e}") 
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
            logger.error(f"[-] Error handling: {e}")

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





