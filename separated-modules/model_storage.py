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



class ModelStorage:
    '''
    SQLite-backed persistence layer for all model artefacts.

    Database tables
    ---------------
    model_storage      : MLP and pipeline weights stored as JSON strings.
    model_attn_storage : Transformer attention weights (JSON).
    node_storage       : QueryNode registry entries (JSON).
    agent_attn_storage : Per-agent attention snapshots and prediction targets.

    Versioning pattern (active-record)
    ------------------------------------
    Every save() inserts a new row marked is_active=1 and immediately sets
    is_active=0 on all older rows for the same memory_name.  Reads always
    query WHERE is_active=1, so only the most recent save is "live".
    This gives a simple append-only audit trail while keeping reads O(1).

    Numpy serialisation
    -------------------
    Numpy arrays are recursively converted to Python lists via
    _prepare_for_serialization() before json.dumps, and converted back via
    _convert_to_arrays() on load.  This avoids the need for pickle in the JSON
    tables, making stored data human-inspectable.

    EXE / frozen-app support
    ------------------------
    get_database_path() detects whether the process is a PyInstaller bundle
    (sys.frozen) and adjusts the DB path to sys._MEIPASS accordingly.

    Parameters
    ----------
    memory_name : Logical name used to scope all read/write operations.
    db_path     : SQLite file path (default: 'activity_log.db').
    '''
    def __init__(self, pipeline, memory_name, db_path='activity_log.db'):
        self.pipeline = pipeline        
        self.db_path = db_path

        self.setup_storage_table()
        self.setup_explainable_table()
        self.setup_agent_table()
        self.setup_node_table()
        self.setup_weight_table()

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
        # Persists a model's in-memory dict to SQLite using an "active record" versioning
        # pattern: each save inserts a new row marked is_active=1, then immediately
        # deactivates all other rows for the same memory_name via a secondary UPDATE.
        # This means only the most recent save is "live" — reads always fetch is_active=1.
        #
        # Two destination tables depending on the `type` argument:
        #   type == 'Transformer' → model_attn_storage  (stores attention-related weights)
        #   else                  → model_storage        (stores MLP / pipeline weights)
        #
        # numpy arrays inside model_dict are recursively converted to Python lists
        # by _prepare_for_serialization() before json.dumps, ensuring they round-trip
        # correctly when loaded back via _convert_to_arrays().
        try:
            db_path = self.get_database_path()            
            conn = sqlite3.connect(db_path)
        except:
            conn = sqlite3.connect(self.db_path)
                    
        c = conn.cursor()

        serializable_dict = self._prepare_for_serialization(model_dict)
        model_json = json.dumps(serializable_dict, default=str)
        if type == 'Transformer':
            try:
                c.execute("""
                    INSERT INTO model_attn_storage 
                    (memory_name, model_type, model_data, is_active)
                    VALUES (?, ?, ?, ?)
                """, (memory_name, model_type, model_json, 1))
        
                # Deactivate all other rows for this memory_name (soft-delete old versions).
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
        
                # Deactivate all other rows for this memory_name (soft-delete old versions).
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

    def setup_weight_table(self):
        # function that handles lstm weights
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
        # Attempts to recover a numpy array from a string representation that may have
        # been serialised in one of several formats (JSON, Python literal, space-separated,
        # or comma-separated).  This is necessary because model weights and probability
        # vectors are stored in SQLite as JSON strings and must be reconstructed precisely.
        #
        # Strategy order (first success wins):
        #   1. JSON array  — handles standard serialisation from json.dumps.
        #   2. ast.literal_eval — handles Python repr output, e.g. "[0.1, 0.2, ...]".
        #   3. Space/bracket-separated floats — covers numpy __str__ output like
        #      "[ 0.1  0.2  0.3]" (spaces instead of commas, optional brackets).
        #   4. Comma-separated floats — fallback for CSV-style strings.
        #
        # Returns the original string unchanged if all strategies fail, letting the
        # caller handle the type mismatch rather than silently producing garbage data.
        if not isinstance(s, str) or not s:
            return s
        
        # Normalise whitespace before parsing — strip newlines, tabs, collapse spaces.
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
            print(f"Retrieved memory: {name}")
            
            # ✅ SAFE: Check if result is a dict before iterating
            if isinstance(result, dict):
                for key, value in result.items():
                    self._print_memory_value(key, value)
            else:
                print(f"  Result is not a dict: {type(result)}")
                print(f"  Result length: {len(result) if hasattr(result, '__len__') else 'N/A'}")
        
        # Handle TwoPass verbose output
        if verbose and data2 is not None and result2 is not None:
            print(f"Retrieved secondary memory: {name}_secondary")
            if isinstance(result2, dict):
                for key, value in result2.items():
                    self._print_memory_value(key, value)
            else:
                print(f"  Secondary result is not a dict: {type(result2)}")
        
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
            print(f'[!] Error handling attention dict: {e}')

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
        # Retrieves a peer agent's stored prediction request from agent_attn_storage,
        # excluding rows whose agent_id matches any ID in the provided list.
        # The exclusion prevents an agent from retrieving its own previously stored
        # request, ensuring it only receives data from *other* agents in the network.
        #
        # The IN clause is constructed dynamically with one '?' placeholder per agent_id
        # entry, which is safe against SQL injection via parameterised queries.
        # Returns (model_attn_data, model_target_pred) parsed from JSON, or (None, None).
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
                data = json.loads(result[0])
                data = self._validate_and_repair(data)
        except Exception as e:
            print(f'Error handling model dict: {e}')
        return None
        
    def _validate_and_repair(self, data):
        """Validate loaded data and repair if corrupted"""
        
        # If data is an array with wrong shape
        if isinstance(data, list) and len(data) > 0:
            # Check if it's a probability array
            if len(data) != self.pipeline._get_num_classes():  
                print(f'[!] Detected corrupted memory with shape {len(data)}, repairing...')
                # Return empty dict to trigger retraining
                return {}
            
            # Try to convert list to dict format
            if all(isinstance(x, (int, float)) for x in data[:10]):
                print(f'[!] List appears to be probabilities, wrapping')
                return {'_cached_probs': np.array(data)}

        
        # If data is dict, ensure values are correct
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, list) and len(value) == 133:
                    print(f'[!] Corrupted value for key {key}, removing')
                    data[key] = None
        
        return data

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

            conn.commit()
            conn.close()

            print('[||] Weights dictionary saved!')

        except Exception as e:
            print(f'[-] Cant save Weights due to: {e}')
            pass          
       
    def load_weights(self, memory_name):
        """Load weights from database. Returns True if found."""
        result = self.weight_retrieval(memory_name)
    
        if not result:
            print(f'[=] No saved weights for {memory_name}')
            return False

        try:
            weights = result
            self.pipeline.network_model.cell.W  = np.array(weights['lstm_W'])
            self.pipeline.network_model.cell.b  = np.array(weights['lstm_b'])
            self.pipeline.network_model.Wy      = np.array(weights['Wy']) if weights['Wy'] else None
            self.pipeline.network_model.by      = np.array(weights['by'])
            self.pipeline.lstm_engine.residual_mean = weights.get('residual_mean', 0.0)
            self.pipeline.lstm_engine.residual_std  = weights.get('residual_std',  1.0)
            self.pipeline.lstm_engine.n_samples     = weights.get('n_samples', self.pipeline.lstm_engine.n_samples)
            self.pipeline.lstm_engine.quantiles     = {float(k): tuple(v) 
                                for k, v in weights.get('quantiles', {}).items()}
        
            print(f'[=] All Weights loaded for {memory_name} '
                f'(saved at {weights.get("saved_at", "unknown")})')
        except Exception as e:
            print(f'[!] Cant load any Weights due to: {e}')
            traceback.print_exc()

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

@dataclass
