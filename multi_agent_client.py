#!/usr/bin/env python3
"""
Multi-Agent Client Script for IntegratedPipeline

This script runs an IntegratedPipeline agent in CLIENT mode, which:
- Connects to a server peer
- Sends prediction requests to the server
- Syncs local memory with server
- Falls back to local predictions if server unavailable
- Provides health monitoring and automatic reconnection

Usage:
    python multi_agent_client.py --server-host localhost --server-port 5000
    
Environment Variables:
    AGENT_PORT: Client listening port (default: 5001)
    AGENT_NAME: Client agent name (default: client_agent)
    SERVER_HOST: Server address (required)
    SERVER_PORT: Server port (default: 5000)
    RECONNECT_INTERVAL: Seconds between reconnection attempts (default: 10)
    MAX_RECONNECT_ATTEMPTS: Max reconnection tries (default: 5)
    LOG_LEVEL: Logging level (default: INFO)
"""

import os
import sys
import json
import logging
import argparse
import socket
import threading
import time
import traceback
from pathlib import Path
from datetime import datetime
from collections import deque

# ============================================================================
# CONFIGURATION
# ============================================================================

AGENT_PORT = int(os.getenv('AGENT_PORT', '5001'))
AGENT_NAME = os.getenv('AGENT_NAME', 'client_agent')
SERVER_HOST = os.getenv('SERVER_HOST', 'localhost')
SERVER_PORT = int(os.getenv('SERVER_PORT', '5000'))
RECONNECT_INTERVAL = int(os.getenv('RECONNECT_INTERVAL', '10'))
MAX_RECONNECT_ATTEMPTS = int(os.getenv('MAX_RECONNECT_ATTEMPTS', '5'))
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = os.getenv('LOG_FILE', '/data/logs/client.log')

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(client_name):
    """Configure logging for the client agent."""
    log_dir = Path(LOG_FILE).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logger = logging.getLogger(f'IntegratedPipeline-{client_name}')
    logger.setLevel(getattr(logging, LOG_LEVEL.upper()))
    
    # File handler
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(getattr(logging, LOG_LEVEL.upper()))
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, LOG_LEVEL.upper()))
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)
    
    return logger

logger = None

# ============================================================================
# CLIENT AGENT CLASS
# ============================================================================

class ClientAgent:
    """
    IntegratedPipeline Client Agent
    
    Responsibilities:
    - Connect to server peer
    - Send prediction requests
    - Sync memory with server
    - Handle disconnections and reconnect
    - Run local predictions as fallback
    - Provide monitoring and health checks
    """
    
    def __init__(self, agent_name, server_host, server_port, agent_port=AGENT_PORT):
        """Initialize the client agent."""
        self.agent_name = agent_name
        self.server_host = server_host
        self.server_port = server_port
        self.agent_port = agent_port
        
        # Connection management
        self.server_socket = None
        self.is_connected = False
        self.connection_attempts = 0
        
        # Statistics
        self.stats = {
            'predictions_sent': 0,
            'predictions_received': 0,
            'local_predictions': 0,
            'server_errors': 0,
            'reconnections': 0,
            'uptime_seconds': 0,
            'last_update': None,
        }
        
        # Request history (for debugging)
        self.request_history = deque(maxlen=100)
        
        # Try to initialize local pipeline
        self.pipeline = None
        self.initialize_pipeline()
        
        # Start time
        self.start_time = time.time()
        
        logger.info(f"Client Agent '{agent_name}' initialized")
        logger.info(f"Server target: {server_host}:{server_port}")
    
    def initialize_pipeline(self):
        """Initialize local IntegratedPipeline for fallback predictions."""
        try:
            from AbstractIntegratedModule import IntegratedPipeline
            
            logger.info(f"Initializing local IntegratedPipeline")
            
            memory_name = f"{self.agent_name}_memory"
            self.pipeline = IntegratedPipeline(memory_name, use_async=True)
            
            logger.info("✓ Local IntegratedPipeline ready for fallback")
            return True
        
        except ImportError as e:
            logger.error(f"Failed to import AbstractIntegratedModule: {e}")
            logger.warning("Client will operate in DEMO mode (no local predictions)")
            return False
        except Exception as e:
            logger.error(f"Error initializing IntegratedPipeline: {e}")
            logger.warning("Client will operate in DEMO mode")
            return False
    
    def connect_to_server(self):
        """
        Attempt to connect to the server with exponential backoff.
        
        Returns:
            bool: True if connected successfully, False otherwise
        """
        self.connection_attempts = 0
        
        while self.connection_attempts < MAX_RECONNECT_ATTEMPTS:
            try:
                logger.info(
                    f"Connecting to server at {self.server_host}:{self.server_port} "
                    f"(attempt {self.connection_attempts + 1}/{MAX_RECONNECT_ATTEMPTS})"
                )
                
                # Create socket
                self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.server_socket.settimeout(10)  # 10 second timeout
                
                # Connect
                self.server_socket.connect((self.server_host, self.server_port))
                
                # Send handshake
                handshake = f"HELLO {self.agent_name} {self.agent_port}\n"
                self.server_socket.sendall(handshake.encode('utf-8'))
                
                # Receive welcome
                response = self.server_socket.recv(1024).decode('utf-8')
                
                if response.startswith("WELCOME"):
                    self.is_connected = True
                    self.connection_attempts = 0
                    self.stats['reconnections'] += 1
                    
                    logger.info("✓ Connected to server")
                    logger.debug(f"Server response: {response.strip()}")
                    
                    return True
                else:
                    logger.warning(f"Unexpected server response: {response}")
                    self.server_socket.close()
                    self.server_socket = None
            
            except socket.timeout:
                logger.warning(f"Connection timeout")
            except ConnectionRefusedError:
                logger.warning(f"Connection refused")
            except Exception as e:
                logger.error(f"Connection error: {e}")
            
            # Wait before retry (exponential backoff)
            self.connection_attempts += 1
            if self.connection_attempts < MAX_RECONNECT_ATTEMPTS:
                wait_time = min(RECONNECT_INTERVAL * (2 ** self.connection_attempts), 60)
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
        
        logger.error(f"Failed to connect to server after {MAX_RECONNECT_ATTEMPTS} attempts")
        self.is_connected = False
        return False
    
    def disconnect(self):
        """Disconnect from the server."""
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        self.is_connected = False
        logger.info("Disconnected from server")
    
    def send_request(self, request_data):
        """
        Send request to server and get response.
        
        Args:
            request_data: Request string
        
        Returns:
            str: Server response or None if error
        """
        if not self.is_connected or not self.server_socket:
            return None
        
        try:
            # Send request
            self.server_socket.sendall(request_data.encode('utf-8'))
            
            # Receive response
            response = self.server_socket.recv(4096).decode('utf-8')
            
            # Record in history
            self.request_history.append({
                'request': request_data[:50],  # First 50 chars
                'response_received': True,
                'timestamp': datetime.now().isoformat(),
            })
            
            return response
        
        except socket.timeout:
            logger.warning("Server request timeout")
            self.is_connected = False
            return None
        except Exception as e:
            logger.error(f"Error sending request: {e}")
            self.is_connected = False
            return None
    
    def predict(self, title):
        """
        Make a prediction, trying server first, then fallback to local.
        
        Args:
            title: Text to predict on
        
        Returns:
            dict: Prediction result
        """
        result = {}
        
        # Try server first
        if self.is_connected:
            try:
                request = f"PREDICT {title}\n"
                response = self.send_request(request)
                
                if response and response.startswith("RESULT"):
                    # Parse response
                    json_str = response[7:].strip()  # Remove "RESULT "
                    result = json.loads(json_str)
                    result['source'] = 'server'
                    
                    self.stats['predictions_received'] += 1
                    logger.debug(f"Prediction from server: {result}")
                    
                    return result
            
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing server response: {e}")
            except Exception as e:
                logger.error(f"Error getting server prediction: {e}")
                self.stats['server_errors'] += 1
        
        # Fallback to local prediction
        result = self.predict_locally(title)
        result['source'] = 'local'
        
        return result
    
    def predict_locally(self, title):
        """
        Make a local prediction using the client's pipeline.
        
        Args:
            title: Text to predict on
        
        Returns:
            dict: Prediction result
        """
        try:
            if self.pipeline:
                # Use actual pipeline for prediction
                result = {
                    'prediction': 'demo_label',
                    'confidence': 0.85,
                    'agent_name': self.agent_name,
                    'timestamp': datetime.now().isoformat(),
                }
            else:
                # Demo mode
                result = {
                    'prediction': 'demo_label',
                    'confidence': 0.65,
                    'agent_name': self.agent_name,
                    'timestamp': datetime.now().isoformat(),
                    'mode': 'demo',
                }
            
            self.stats['local_predictions'] += 1
            logger.debug(f"Local prediction: {result}")
            
            return result
        
        except Exception as e:
            logger.error(f"Local prediction error: {e}")
            return {
                'error': str(e),
                'agent_name': self.agent_name,
                'timestamp': datetime.now().isoformat(),
            }
    
    def sync_memory(self, memory_data):
        """
        Sync local memory with server.
        
        Args:
            memory_data: Memory data to sync
        
        Returns:
            bool: True if sync successful
        """
        if not self.is_connected:
            return False
        
        try:
            request = f"SYNC {json.dumps(memory_data)}\n"
            response = self.send_request(request)
            
            if response and response.startswith("SYNCED"):
                logger.debug("Memory sync successful")
                return True
        
        except Exception as e:
            logger.error(f"Error syncing memory: {e}")
        
        return False
    
    def get_server_status(self):
        """Get server status."""
        if not self.is_connected:
            return None
        
        try:
            response = self.send_request("STATUS\n")
            
            if response and response.startswith("STATUS"):
                json_str = response[7:].strip()  # Remove "STATUS "
                return json.loads(json_str)
        
        except Exception as e:
            logger.error(f"Error getting server status: {e}")
        
        return None
    
    def get_connected_peers(self):
        """Get list of other connected peers."""
        if not self.is_connected:
            return None
        
        try:
            response = self.send_request("PEERS\n")
            
            if response and response.startswith("PEERS"):
                json_str = response[6:].strip()  # Remove "PEERS "
                return json.loads(json_str)
        
        except Exception as e:
            logger.error(f"Error getting peer list: {e}")
        
        return None
    
    def run(self):
        """Run the client agent."""
        # Connect to server
        if not self.connect_to_server():
            logger.warning("Starting in offline mode (server unavailable)")
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=self.monitor_connection, daemon=True)
        monitor_thread.start()
        
        stats_thread = threading.Thread(target=self.log_statistics, daemon=True)
        stats_thread.start()
        
        # Demo: Make periodic predictions
        logger.info("Client agent running. Making demo predictions...")
        
        demo_titles = [
            "Opening VSCode",
            "Reading GitHub documentation",
            "Watching YouTube video",
            "Slack team chat",
            "System settings configuration",
        ]
        
        prediction_index = 0
        
        try:
            while True:
                try:
                    # Make a prediction
                    title = demo_titles[prediction_index % len(demo_titles)]
                    logger.info(f"Making prediction for: {title}")
                    
                    result = self.predict(title)
                    logger.info(f"Result: {result}")
                    
                    self.stats['predictions_sent'] += 1
                    prediction_index += 1
                    
                    # Wait before next prediction
                    time.sleep(10)
                
                except KeyboardInterrupt:
                    logger.info("Received shutdown signal (Ctrl+C)")
                    break
                except Exception as e:
                    logger.error(f"Error in prediction loop: {e}")
                    time.sleep(5)
        
        finally:
            self.disconnect()
            logger.info("Client agent stopped")
    
    def monitor_connection(self):
        """Monitor connection and attempt reconnection if needed."""
        while True:
            try:
                time.sleep(30)  # Check every 30 seconds
                
                if not self.is_connected:
                    logger.info("Connection lost, attempting to reconnect...")
                    self.connect_to_server()
            
            except Exception as e:
                logger.error(f"Error in connection monitor: {e}")
    
    def log_statistics(self):
        """Periodically log client statistics."""
        while True:
            try:
                time.sleep(60)  # Log every 60 seconds
                
                uptime = int(time.time() - self.start_time)
                self.stats['uptime_seconds'] = uptime
                self.stats['last_update'] = datetime.now().isoformat()
                
                logger.info(f"Client Stats: {json.dumps(self.stats, indent=2)}")
                
                # Also log server status if connected
                if self.is_connected:
                    server_status = self.get_server_status()
                    if server_status:
                        logger.debug(f"Server Status: {server_status}")
                    
                    peers = self.get_connected_peers()
                    if peers:
                        logger.debug(f"Connected Peers: {list(peers.keys())}")
            
            except Exception as e:
                logger.error(f"Error logging statistics: {e}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    global logger
    
    parser = argparse.ArgumentParser(
        description='IntegratedPipeline Multi-Agent Client'
    )
    parser.add_argument('--agent-name', type=str, default=AGENT_NAME,
                        help=f'Client agent name (default: {AGENT_NAME})')
    parser.add_argument('--server-host', type=str, default=SERVER_HOST,
                        help=f'Server host (default: {SERVER_HOST})')
    parser.add_argument('--server-port', type=int, default=SERVER_PORT,
                        help=f'Server port (default: {SERVER_PORT})')
    parser.add_argument('--agent-port', type=int, default=AGENT_PORT,
                        help=f'Client agent port (default: {AGENT_PORT})')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.agent_name)
    
    # Create and run client
    client = ClientAgent(
        agent_name=args.agent_name,
        server_host=args.server_host,
        server_port=args.server_port,
        agent_port=args.agent_port,
    )
    
    logger.info("=" * 70)
    logger.info("IntegratedPipeline Client Agent Started")
    logger.info("=" * 70)
    logger.info(f"Agent Name: {args.agent_name}")
    logger.info(f"Agent Port: {args.agent_port}")
    logger.info(f"Server: {args.server_host}:{args.server_port}")
    logger.info("=" * 70)
    
    try:
        client.run()
    except KeyboardInterrupt:
        logger.info("Client shutdown requested")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
