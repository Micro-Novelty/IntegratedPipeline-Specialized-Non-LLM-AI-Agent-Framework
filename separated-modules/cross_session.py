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



class CrossSessionAutomation:
    '''
    Serialises and restores the full pipeline state across process restarts or
    different machines.

    A "session" is a JSON snapshot that includes:
      - Pipeline memory dict (TW/MW/TP/MP/TA keys and cached probabilities).
      - Model weights (TF-IDF vocab, MLP layer weights, Transformer weights).
      - Label map and vocabulary.
      - Metadata (timestamp, session name, pipeline memory_name).

    Methods
    -------
    export_session(session_name)
        Serialises the current pipeline state to a JSON file named
        <session_name>.json in the working directory.

    import_session(filepath)
        Loads a previously exported session from disk and restores all weights
        into the live pipeline instance.

    sync_with_device(host, port)
        Sends the current session over a raw TCP socket to a listening peer,
        enabling cross-machine model synchronisation without a shared DB.

    list_sessions()
        Returns the list of .json session files in the current directory.
    '''
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
