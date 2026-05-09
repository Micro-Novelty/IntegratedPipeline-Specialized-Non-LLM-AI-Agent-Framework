#!/bin/bash
# Start multi-agent cluster
docker-compose up -d --scale agent-client=5
echo "Cluster started. 1 server + 5 clients running."
docker-compose logs -f
