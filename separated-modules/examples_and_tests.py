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



# ============ EXAMPLE: SECURE PEER-TO-PEER CLUSTER ============

async def run_secure_agent_cluster(pipeline,test_titles, label_map, rules, X=None, y=None, agent_id=None, filename=None, title_name=None, label_name=None, manager=None):
    """
    Run multiple agents that securely communicate.
    Stops retrying once connected successfully.
    """
    print("\n" + "="*60)
    print("=== SECURE PEER-TO-PEER CLUSTER ===")
    print("="*60)
    
    # Set discovery secret (in production, use environment variable)
    secret_key = 'my-ultra-safe-secret-key-for-authentication'

    # Agent 1 - Primary (Port 5555)
    agent1 = CohesiveAgentDeployment(
        pipeline=pipeline,
        memory_name="agent_primary",
        filename=filename,
        target_title=title_name,
        label_name=label_name,
        security_level="PRODUCTION",
        enable_peers=True,
        trusted_networks=['127.0.0.1/32', '192.168.1.0/24'],
        peer_discovery_port=5555,
        secret_key=secret_key,
        shared_auth_token=secret_key,
        predict_manager=manager
    )
    
    # Agent 2 - Secondary (Port 5556)
    agent2 = CohesiveAgentDeployment(
        pipeline=pipeline,
        memory_name="agent_secondary",
        filename=filename,
        target_title=title_name,
        label_name=label_name,
        security_level="PRODUCTION",
        enable_peers=True,
        trusted_networks=['127.0.0.1/32', '192.168.1.0/24'],
        peer_discovery_port=5556,
        secret_key=secret_key,
        shared_auth_token=secret_key,
        predict_manager=manager
    )
    
    try:
        # Start both agents
        print("\n🚀 Starting Agent 1...")
        await agent1.start()
        print("✅ Agent 1 started on port 5555")
        
        print("\n🚀 Starting Agent 2...")
        await agent2.start()
        print("✅ Agent 2 started on port 5556")
        
        # Give servers time to fully bind
        await asyncio.sleep(2)
        
        # Get API keys
        api_key = agent1.get_api_key()
        print(f"\n🔑 Using API Key: {api_key[:20]}...")
        
        texts = {"test_titles": test_titles, "label_map": label_map, "rules": rules, "X": X, 'y':y, "use_transformer": True, "agent_id": agent_id}

        # Make prediction with peer ensemble
        # Peer Connection will be ensured successful during P2P 
        result = await agent1.multi_modal_peer_ensemble_prediction(
            texts=texts,
            api_key=api_key,
            method='advanced',
            disable_sync=True
        )    

        result2 = await agent2.multi_modal_peer_ensemble_prediction(
            texts=texts,
            api_key=api_key,
            method='advanced',
            disable_sync=True
        )      
        
        print(f"\n📊 Ensemble Result for Agent 1:")
        print(f"   Prediction: {result.get('prediction', 'N/A')}")
        print(f"   Confidence: {result.get('confidence', 0):.2%}")

        print(f"   Second Prediction: {result2.get('prediction', 'N/A')}")
        print(f"   Second Confidence: {result2.get('confidence', 0):.2%}")

        # Keep running briefly
        print("\n⏳ Cluster stable. Waiting 5 seconds before shutdown...")
        await asyncio.sleep(5)
        agent2._peer_agent.stop_server()
        
    except Exception as e:
        print(f"\n❌ Error in cluster: {e}")
        traceback.print_exc()
        
    finally:
        print("\n🛑 Shutting down cluster...")
        await agent1.shutdown()
        await agent2.shutdown()
        print("✅ Cluster shutdown complete")




async def example_async_with_result_queue(pipeline, test_titles, label_map, rules, agent_id, filename, title_name, label_name):
    # Example using the proper result queue
    
    agent = CohesiveAgentDeployment(
        memory_name="test_agent",
        filename=filename,
        target_title=title_name,
        label_name=label_name,
        security_level="DEVELOPMENT",
        enable_peers=False
    )
    
    await agent.start()
    
    api_key = agent.get_api_key()
    payloads = {"test_titles": test_titles, "label_map": label_map, "rules": rules, "use_transformer": True, "agent_id": agent_id}
    
    # Single async prediction
    print('[==] Single sync prediction: (using single text: "Opening Thesis.docx")')
    sync_result = agent.predict_sync(
        texts=payloads,
        api_key=api_key,
        client_ip="127.0.0.1",
        method='advanced'
    )

    print(f"[=] Sync Result: {sync_result}")


    print("[==] Single async prediction: (using single text: Opening Thesis.docx)")
    result = await agent.predict_async(
        texts=payloads,
        api_key=api_key,
        client_ip="127.0.0.1",
    )
    print(f"[=] Result: {result.get('prediction')} ({result.get('confidence', 0)}")
    
    # Batch async predictions (parallel!)
    print("\n[=] Batch async predictions (parallel):")
    texts = [
        "Watching YouTube",
        "Programming in VS Code",
        "Checking Slack messages",
        "Reading documentation",
        "Taking a break"
    ]
    
    start_time = time.time()
    results = await agent.predict_batch_async(texts, timeout=60, api_key=api_key)
    elapsed = time.time() - start_time
    
    for result in results:
        print(f"[=] '{result['text']}' → {result['prediction']} ({result['confidence']:.1%})")
    
    print(f"\n[=] Completed {len(texts)} predictions in {elapsed:.2f}s")
    
    # Get queue stats
    stats = agent.get_queue_stats()
    print(f"[=] Queue stats: {stats}")
    
    await agent.shutdown()




def initiate_cohesive_agent_deployment_test(pipeline, test_titles, label_map, rules, X, y, agent_id, filename, title_name, label_name, manager):
    print("\n" + "="*60)
    print("🔮 = TESTING COHESIVE AGENT DEPLOYMENT WITH ASYNC MANAGER = ")

    print('Test 1 of Multi agent cluster')
    asyncio.run(run_secure_agent_cluster(pipeline=pipeline, test_titles=test_titles, label_map=label_map, rules=rules, X=X, y=y, agent_id=agent_id, filename=filename, title_name=title_name, label_name=label_name, manager=manager))
      
    print("\n1. Basic async with result queue")
    asyncio.run(example_async_with_result_queue(pipeline=pipeline, test_titles=test_titles, label_map=label_map, rules=rules, X=X, y=y, agent_id=agent_id, filename=filename, title_name=title_name, label_name=label_name))
    

# async manager setup examples
def initiate_prediction_usage(pipeline, manager, predict_wrapper, test_titles, label_map, rules, X, y):
    """Basic synchronous usage."""
    # Use context manager (auto start/stop)
    api_key = 'my-ultra-safe-secret-key-for-authentication'

    with predict_wrapper as wrapper:
        print('[==] Initiating regular prediction')
        texts = {'test_titles': test_titles, 'label_map': label_map, 'rules': rules, 'X':X, 'y':y, 'use_transformer': True}
        regular_predict = wrapper.predict(
        texts=texts, 
        timeout=120,
        retries=None,
        api_key=api_key,
        client_ip=None)

        print('[==] Initiating advanced batch prediction')
        predicted_output = wrapper.advanced_batch_prediction(test_titles, label_map, rules, X=X, y=y, api_key=api_key, client_ip=None)


def initiate_with_retries(pipeline, manager, wrapper, test_titles, label_map, rules, X, y):
    """Example with retry logic."""
    
    try:
        # Will retry up to 5 times
        texts = {"test_titles": test_titles, "label_map": label_map, "rules": rules, "X":X, "y":y, "use_transformer": True}
        result = wrapper.predict(texts, timeout=60, retries=None, api_key=None)
        advanced_result, chosen_label, confidence = wrapper.advanced_prediction_method(manager, test_titles, label_map, rules, X=X, y=y, method='Transformer_included')
        print(f"[=] Result after retries: {result}")
        print(f"[=] Advanced Result: {chosen_label} || ({confidence:.1%})")

    except Exception as e:
        print(f"[!] Failed after retries: {e}")
    finally:
        wrapper.stop()


def initiate_graceful_shutdown(pipeline, wrapper):
    """Example showing graceful shutdown."""
   
    # Submit many async requests
    for i in range(10):
        wrapper.predict_async(f"[=] Request {i}")
    
    # Wait for idle with timeout
    if wrapper.wait_for_idle(timeout=30):
        print("[+] All requests completed")
    else:
        print("[!] Some requests still pending")
    
    # Graceful shutdown
    wrapper.stop(timeout=10)

def AsyncWrappertest(pipeline, prediction_manager, test_titles, label_map, rules, X, y):
    print("\n" + "="*60)
    print("🔮 = TESTING ASYNCHRONOUS PREDICTION WRAPPER = ")
    print("="*60)

    api_key = 'my-ultra-safe-secret-key-for-authentication'

    config = SecurityConfig(
            max_text_length=10000,
            max_queue_size=100,
            rate_limit_requests=60,  # 60 per minute
            require_api_key=True,
            max_pending_tasks=50,
            request_timeout=30.0,

            # Start with no IP restrictions, add via admin API
            allowed_ips=[],
            blocklisted_ips=[],
            require_bootstrap_auth = False
        )

    wrapper = PipelineAsyncManager(pipeline, 
              prediction_manager, 
              config=config, 
              state_file=None, 
              security_level=SecurityLevel.PRODUCTION,
              api_key=api_key, 
              max_workers=4, 
              task_timeout=30, 
              max_retries=3 )

    wrapper.start(method='Transformer_included', bootstrap_token=None)
    
    logging.basicConfig(level=logging.INFO)
    
    # Run examples
    initiate_prediction_usage(pipeline, prediction_manager, wrapper, test_titles, label_map, rules, X, y)
    initiate_with_retries(pipeline, prediction_manager, wrapper, test_titles, label_map, rules, X, y)
    initiate_graceful_shutdown(pipeline, wrapper)

    print("\n✅ Asynchronous prediction wrapper test completed successfully.")


def PermissiveTest():
    print("\n" + "="*60)
    print("🔮 = TESTING HYBRID PREDICTION SYSTEM = ")
    print("="*60)

    print("📖 Loading labels from text file with CSV format...")
    filename = input('|| Insert Filename (press N to skip): ')
    title = input('|| Insert Title name you have in your file (press N to skip): ')
    label = input('|| Insert Label name you have in your file (press N to skip): ')
    agent_id = input('|| Insert Agent ID for distributed inference (press N to skip): ')

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
        pipeline = IntegratedPipeline(file, use_async=True, agent_port=5001, ssl_cert_file=cert_file, ssl_key_file=key_file)
    else:
        print('|| Using original csv_file.pkl file as fallback...')
        pipeline = IntegratedPipeline('csv_file.pkl', use_async=True, agent_port=5001,ssl_cert_file=cert_file, ssl_key_file=key_file)

    manager = PipelinePredictionManager(pipeline, label_csv='ManualsTraining.txt', target_title='window_title', label='label')

    pipeline.distribution.predict_manager = manager
    if agent_id == 'N':
        agent_id = 'local'

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
    X, y = None, None
    while running:
        permission = input('|| Allow Hybrid prediction test? [Y/N]: ')

        if permission == 'Y' or permission == 'y':
            print('== TEST 1: (titles only without transformer) ==')
            advanced_result = manager.advanced_prediction_method(
            [t[0] for t in test_titles],  # Just titles
            label_map,
            rules,
            X=X, y=y,
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

        permission_continue = input('|| Do you want to test the Asynchronous wrapper for multiple predictions? [Y/N]: ')
        if permission_continue == 'Y' or permission_continue == 'y':
            AsyncWrappertest(pipeline, manager, test_titles, label_map, rules, X, y)
            print('== Asynchronous wrapper Successfully tested! ==')

        cohesive_permission = input('|| Do you want to test the Cohesive Agent Deployment with Async Manager? [Y/N]: ')
        if cohesive_permission == 'Y' or cohesive_permission == 'y':
            if not (filename and title and label and filename != 'N'):
                print('[=] Searching fallback filename: ManualsTraining.txt, window_title, label')
                initiate_cohesive_agent_deployment_test(pipeline,test_titles, label_map, rules, X, y, agent_id, 'ManualsTraining.txt', 'window_title', 'label', manager)
            else:
                initiate_cohesive_agent_deployment_test(pipeline, test_titles, label_map, rules, X, y, agent_id, filename, title, label, manager)
            print('== Cohesive Agent Deployment Successfully tested! ==')

        else:
            running = False
            print('|| Program Prediction test aborted!')
            pass


if __name__ == "__main__":
    try:
        PermissiveTest()
    except Exception as e:
        print(f'|| Program Crashed...,  Error: {e}')
        traceback.print_exc()
        pass


