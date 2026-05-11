#!/bin/bash
set -e

echo "=========================================="
echo "🤖 AbstractIntegratedModule Agent"
echo "=========================================="
echo "Mode: $AGENT_MODE"
echo "Memory: $MEMORY_NAME"
echo "Port: $AGENT_PORT"
echo "DB Path: $DB_PATH"
echo "=========================================="

# Create data directory if it doesn't exist
mkdir -p "$(dirname $DB_PATH)"

# Run based on mode
case "$AGENT_MODE" in
    server)
        echo "🚀 Starting in SERVER mode"
        exec python -c "
import AbstractIntegratedModule as aim
pipeline = aim.IntegratedPipeline('$MEMORY_NAME')
dist = aim.AgentDistributedInference(pipeline, pipeline.storage, '$MEMORY_NAME', port=$AGENT_PORT)
dist.start_server()
"
        ;;
    client)
        echo "🔗 Starting in CLIENT mode (connecting to $SERVER_HOST:$AGENT_PORT)"
        exec python -c "
import AbstractIntegratedModule as aim
pipeline = aim.IntegratedPipeline('$MEMORY_NAME')
dist = aim.AgentDistributedInference(pipeline, pipeline.storage, '$MEMORY_NAME', port=0)
dist.connect_to_agent('$SERVER_HOST', $AGENT_PORT)
print('✅ Connected to server. Press Ctrl+C to exit.')
import time
while True: time.sleep(1)
"
        ;;
    single)
        echo "🕹️ Starting in SINGLE mode"
        exec python main.py
        ;;
    *)
        echo "❌ Unknown mode: $AGENT_MODE"
        echo "Valid modes: server, client, single"
        exit 1
        ;;
esac
