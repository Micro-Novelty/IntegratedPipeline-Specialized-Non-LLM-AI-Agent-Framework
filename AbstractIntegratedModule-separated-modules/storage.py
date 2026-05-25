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
# storage.py
# SQLite-backed model persistence (ModelStorage) and cross-session state
# automation (CrossSessionAutomation).
# ModelStorage owns all DB schema creation, active-record versioning,
# cosine-similarity memory cache, and numpy ↔ JSON serialisation helpers.
# Depends on: geometry (GWS mixin for cosine_similarity / anisotropy helpers)
# ---------------------------------------------------------------------------
from .geometry import GeometricWeightShaping

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
            with self.ssl_context.wrap_socket(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
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

        serializable_dict = self._prepare_for_serialization(model_dict)
        model_json = json.dumps(serializable_dict, default=str)
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