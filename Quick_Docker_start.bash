# Quick start commands

# for single agent:
# Build image
docker build -t integrated-pipeline:latest .

# Run single agent
docker run -it -v $(pwd)/data:/app/data integrated-pipeline:latest python

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
