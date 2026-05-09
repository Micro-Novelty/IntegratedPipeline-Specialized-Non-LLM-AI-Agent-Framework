# Quick start commands

# for single agent:
# Build image
docker build -t integrated-agent:latest .

# Run single agent
# docker run -it -v $(pwd)/data:/app/data integrated-agent:latest python #simpler start

# using activity_log.db (database file), 
# consider running the integratedPipeline in python first to automatically populate the data in the database.
# This provide interactive shell:
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/activity_log.db:/app/activity_log.db \
  integrated-agent:latest python

# Run single agent in background
docker run -d \
  --name ai-agent \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/activity_log.db:/app/activity_log.db \
  integrated-agent:latest

# start agents for P2P
docker-compose up -d

# All agents
docker-compose logs -f

# Specific agent
docker-compose logs -f agent-server
docker-compose logs -f agent-client-1
docker-compose logs -f agent-client-2

# scale to more clients
docker-compose up -d --scale agent-client=10

# check status
docker-compose ps

# stop all
docker-compose down

# stop and remove data
docker-compose down -v

# Remove all stopped containers
docker container prune

# Remove unused images
docker image prune
