#!/usr/bin/env python3
# Note: you still need to edit ServerAgent class for more features and flexibility.
"""
Multi-Agent Server Script for IntegratedPipeline

This script runs an IntegratedPipeline agent in SERVER mode, which:
- Listens for incoming peer connections
- Manages peer-to-peer communication
- Coordinates predictions across multiple clients
- Maintains centralized memory and decision-making
- Provides health monitoring and logging

Usage:
    python multi_agent_server.py [--port 5000] [--host 0.0.0.0]
    
Environment Variables:
    AGENT_PORT: Server listening port (default: 5000)
    AGENT_HOST: Server binding address (default: 0.0.0.0)
    MEMORY_NAME: Memory database name (default: server_agent_memory)
    SSL_ENABLED: Enable SSL/TLS (default: false)
    SSL_CERT_FILE: Path to SSL certificate
    SSL_KEY_FILE: Path to SSL private key
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
from collections import defaultdict

# ============================================================================
# CONFIGURATION
# ============================================================================

# Get configuration from environment variables
AGENT_PORT = int(os.getenv('AGENT_PORT', '5000'))
AGENT_HOST = os.getenv('AGENT_HOST', '0.0.0.0')
MEMORY_NAME = os.getenv('MEMORY_NAME', 'server_agent_memory')
SSL_ENABLED = os.getenv('SSL_ENABLED', 'false').lower() == 'true'
SSL_CERT_FILE = os.getenv('SSL_CERT_FILE', None)
SSL_KEY_FILE = os.getenv('SSL_KEY_FILE', None)
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = os.getenv('LOG_FILE', '/data/logs/server.log')
MAX_PEERS = int(os.getenv('MAX_PEERS', '5'))
PEER_TIMEOUT = int(os.getenv('PEER_TIMEOUT', '30'))

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging():
    """Configure logging for the server agent."""
    # Create logs directory if it doesn't exist
    log_dir = Path(LOG_FILE).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create logger
    logger = logging.getLogger('IntegratedPipeline-Server')
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

logger = setup_logging()

# ============================================================================
# SERVER AGENT CLASS
# ============================================================================

class ServerAgent:
    """
    IntegratedPipeline Server Agent
    
    Responsibilities:
    - Listen for incoming peer connections
    - Manage peer registry and health monitoring
    - Coordinate predictions across peers
    - Maintain shared memory and decision history
    - Handle SSL/TLS security
    """
    
    def __init__(self, port=AGENT_PORT, host=AGENT_HOST, memory_name=MEMORY_NAME, example_rules):
        """Initialize the server agent."""
        self.port = port
        self.host = host
        self.memory_name = memory_name
        
        # Peer management
        self.peers = {}  # {peer_id: {host, port, last_seen, status}}
        self.peer_lock = threading.Lock()
        self.max_peers = MAX_PEERS
        self.peer_timeout = PEER_TIMEOUT 
        
        # Statistics
        self.stats = {
            'connections_received': 0,
            'predictions_processed': 0,
            'peers_connected': 0,
            'peers_failed': 0,
            'uptime_seconds': 0,
            'memory_usage_mb': 0,
            'last_update': None,
        }
        
        # Try to initialize IntegratedPipeline
        self.pipeline = None
        self.initialize_pipeline()

        self.label_map = self.pipeline.load_labels_from_csv('<your_filename>', '<your_csv_title>', '<title_label>')
        self.example_rules = example_rules

        # Start time
        self.start_time = time.time()
        
        logger.info(f"Server Agent initialized on {host}:{port}")
    
    def initialize_pipeline(self):
        """Initialize the IntegratedPipeline model."""
        try:
            from AbstractIntegratedModule import IntegratedPipeline, PipelinePredictionManager
            
            logger.info(f"Initializing IntegratedPipeline with memory: {self.memory_name}")
            
            # Create pipeline instance
            self.pipeline = IntegratedPipeline(self.memory_name)
            
            # Optional: Load training data if available
            # self.load_training_data()
            
            logger.info("✓ IntegratedPipeline initialized successfully")
            return True
            
        except ImportError as e:
            logger.error(f"Failed to import AbstractIntegratedModule: {e}")
            logger.warning("Server will run in DEMO mode (predictions won't use actual model)")
            return False
        except Exception as e:
            logger.error(f"Error initializing IntegratedPipeline: {e}")
            logger.warning("Server will run in DEMO mode")
            return False
    
    def start(self):
        """Start the server and begin listening for connections."""
        logger.info(f"Starting server on {self.host}:{self.port}...")
        
        # Create server socket
        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((self.host, self.port))
            server_socket.listen(self.max_peers)
            
            logger.info(f"✓ Server listening on {self.host}:{self.port}")
            
        except Exception as e:
            logger.error(f"Failed to create server socket: {e}")
            return False
        
        # Start monitoring threads
        monitor_thread = threading.Thread(target=self.monitor_peers, daemon=True)
        monitor_thread.start()
        
        stats_thread = threading.Thread(target=self.log_statistics, daemon=True)
        stats_thread.start()
        
        # Accept connections
        try:
            while True:
                try:
                    client_socket, client_address = server_socket.accept()
                    self.stats['connections_received'] += 1
                    
                    logger.info(f"New connection from {client_address}")
                    
                    # Handle client in separate thread
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_socket, client_address),
                        daemon=True
                    )
                    client_thread.start()
                    
                except KeyboardInterrupt:
                    logger.info("Received shutdown signal (Ctrl+C)")
                    break
                except Exception as e:
                    logger.error(f"Error accepting connection: {e}")
                    continue
        
        finally:
            server_socket.close()
            logger.info("Server socket closed")
    
    def handle_client(self, client_socket, client_address):
        """
        Handle communication with a connected client peer.
        
        Protocol:
        1. Client sends: HELLO <client_id> <client_port>
        2. Server responds: WELCOME <server_id> <status>
        3. Client can send: PREDICT <data> or SYNC <memory_data>
        4. Server responds with results
        """
        peer_id = None
        
        try:
            # Receive initial handshake
            handshake = client_socket.recv(1024).decode('utf-8')
            logger.debug(f"Received handshake: {handshake}")
            
            # Parse handshake
            if handshake.startswith('HELLO'):
                parts = handshake.split()
                if len(parts) >= 3:
                    peer_id = parts[1]
                    peer_port = parts[2]
                    
                    # Register peer
                    with self.peer_lock:
                        self.peers[peer_id] = {
                            'host': client_address[0],
                            'port': peer_port,
                            'last_seen': time.time(),
                            'status': 'connected',
                            'predictions_count': 0,
                        }
                        self.stats['peers_connected'] = len(self.peers)
                    
                    logger.info(f"Peer registered: {peer_id} ({client_address[0]}:{peer_port})")
                    
                    # Send welcome response
                    response = f"WELCOME {self.memory_name} OK\n"
                    client_socket.sendall(response.encode('utf-8'))
                    
                    # Handle client requests
                    self.handle_client_requests(client_socket, peer_id)
                else:
                    logger.warning(f"Invalid handshake format: {handshake}")
                    client_socket.sendall(b"ERROR Invalid handshake\n")
            else:
                logger.warning(f"Expected HELLO, got: {handshake}")
                client_socket.sendall(b"ERROR Expected HELLO\n")
        
        except Exception as e:
            logger.error(f"Error handling client: {e}")
            self.stats['peers_failed'] += 1
        
        finally:
            # Cleanup
            if peer_id and peer_id in self.peers:
                with self.peer_lock:
                    del self.peers[peer_id]
                    self.stats['peers_connected'] = len(self.peers)
            
            client_socket.close()
            logger.info(f"Client disconnected: {peer_id}")
    
    def handle_client_requests(self, client_socket, peer_id):
        """Handle incoming requests from connected client."""
        while True:
            try:
                # Receive request
                data = client_socket.recv(4096).decode('utf-8')
                
                if not data:
                    break  # Client disconnected
                
                # Update peer last seen
                with self.peer_lock:
                    if peer_id in self.peers:
                        self.peers[peer_id]['last_seen'] = time.time()
                
                # Process request
                response = self.process_request(data, peer_id)
                
                # Send response
                client_socket.sendall(response.encode('utf-8'))
                
            except socket.timeout:
                logger.warning(f"Timeout from peer {peer_id}")
                break
            except Exception as e:
                logger.error(f"Error in client request handling: {e}")
                break
    
    def process_request(self, request_data, peer_id):
        """
        Process incoming request from client.
        
        Request formats:
        - PREDICT <title> <label_map_json>
        - SYNC <memory_json>
        - STATUS
        - PEERS
        """
        try:
            parts = request_data.strip().split(maxsplit=1)
            command = parts[0] if parts else ""
            
            if command == "PREDICT":
                # Extract prediction request
                if len(parts) > 1:
                    title = parts[1]
                    result = self.predict(title, peer_id)
                    return f"RESULT {json.dumps(result)}\n"
                else:
                    return "ERROR Missing prediction data\n"
            
            elif command == "SYNC":
                # Sync memory with peer
                if len(parts) > 1:
                    memory_data = json.loads(parts[1])
                    self.sync_memory(memory_data, peer_id)
                    return "SYNCED OK\n"
                else:
                    return "ERROR Missing memory data\n"
            
            elif command == "STATUS":
                # Return server status
                status = {
                    'server_id': self.memory_name,
                    'uptime_seconds': int(time.time() - self.start_time),
                    'peers_connected': len(self.peers),
                    'stats': self.stats,
                }
                return f"STATUS {json.dumps(status)}\n"
            
            elif command == "PEERS":
                # Return connected peers
                with self.peer_lock:
                    peers_info = {
                        pid: {k: v for k, v in info.items() if k != 'last_seen'}
                        for pid, info in self.peers.items()
                    }
                return f"PEERS {json.dumps(peers_info)}\n"
            
            else:
                return f"ERROR Unknown command: {command}\n"
        
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return "ERROR Invalid JSON\n"
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return f"ERROR {str(e)}\n"
    
    def predict(self, title, peer_id):
        """
        Make a prediction using the pipeline.
        
        Args:
            title: Text to predict on
            peer_id: ID of requesting peer
        
        Returns:
            dict with prediction results
        """
        
        try:
            if self.pipeline:
                results, prediction, confidence = self.pipeline.advanced_prediction_method(title, self.label_map, self.example_rules, show_proba=True)
                result = {
                    'prediction': prediction,
                    'confidence': confidence,
                    'peer_id': peer_id,
                    'timestamp': datetime.now().isoformat(),
                }
            else:
                # Demo mode
                result = {
                    'prediction': 'demo_label',
                    'confidence': 0.75,
                    'peer_id': peer_id,
                    'timestamp': datetime.now().isoformat(),
                    'mode': 'demo',
                }
            
            # Update statistics
            self.stats['predictions_processed'] += 1
            with self.peer_lock:
                if peer_id in self.peers:
                    self.peers[peer_id]['predictions_count'] += 1
            
            return result
        
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {'error': str(e), 'peer_id': peer_id}
    
    def sync_memory(self, memory_data, peer_id):
        """Sync memory data from peer."""
        try:
            logger.debug(f"Syncing memory from {peer_id}")
            # Implement memory synchronization logic
            # For now, just log it
        except Exception as e:
            logger.error(f"Error syncing memory: {e}")
    
    def monitor_peers(self):
        """Periodically monitor peer health and remove stale connections."""
        while True:
            try:
                time.sleep(10)  # Check every 10 seconds
                
                current_time = time.time()
                with self.peer_lock:
                    stale_peers = [
                        peer_id for peer_id, info in self.peers.items()
                        if (current_time - info['last_seen']) > self.peer_timeout
                    ]
                    
                    for peer_id in stale_peers:
                        logger.warning(f"Removing stale peer: {peer_id}")
                        del self.peers[peer_id]
                        self.stats['peers_connected'] = len(self.peers)
            
            except Exception as e:
                logger.error(f"Error in peer monitoring: {e}")
    
    def log_statistics(self):
        """Periodically log server statistics."""
        while True:
            try:
                time.sleep(30)  # Log every 30 seconds
                
                uptime = int(time.time() - self.start_time)
                self.stats['uptime_seconds'] = uptime
                self.stats['last_update'] = datetime.now().isoformat()
                
                logger.info(f"Server Stats: {json.dumps(self.stats, indent=2)}")
            
            except Exception as e:
                logger.error(f"Error logging statistics: {e}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='IntegratedPipeline Multi-Agent Server'
    )
    parser.add_argument('--port', type=int, default=AGENT_PORT,
                        help=f'Server port (default: {AGENT_PORT})')
    parser.add_argument('--host', type=str, default=AGENT_HOST,
                        help=f'Server host (default: {AGENT_HOST})')
    parser.add_argument('--memory', type=str, default=MEMORY_NAME,
                        help=f'Memory database name (default: {MEMORY_NAME})')
    parser.add_argument('--max-peers', type=int, default=MAX_PEERS,
                        help=f'Maximum peers allowed (default: {MAX_PEERS})')
    
    args = parser.parse_args()

    example_rules = '<example-rules>'
    
    # Create and start server
    server = ServerAgent(
        port=args.port,
        host=args.host,
        memory_name=args.memory
        example_rules
    )
    
    logger.info("=" * 70)
    logger.info("IntegratedPipeline Server Agent Started")
    logger.info("=" * 70)
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Memory: {args.memory}")
    logger.info(f"Max Peers: {args.max_peers}")
    logger.info("=" * 70)
    
    try:
        server.start()
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
