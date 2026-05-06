# 🐳 Docker Installation (Alternative Method)

Docker provides a containerized environment with all dependencies pre-installed. This is the **easiest way** to get started without dealing with system-specific binaries.

## Prerequisites
- **Docker** installed ([Download here](https://www.docker.com/products/docker-desktop))
- Approximately **500 MB** disk space for the image

## Quick Start with Docker

### 0. Clone repository
```bash
git clone https://github.com/Micro-Novelty/IntegratedPipeline-Continous-Learning-AI-Agent-library-framework.git
cd IntegratedPipeline-Continous-Learning-AI-Agent-library-framework
```

### 1. Build the Docker Image

```bash
# Clone the repository if you haven't
git clone https://github.com/Micro-Novelty/IntegratedPipeline-Continous-Learning-AI-Agent-library-framework.git
cd IntegratedPipeline-Continous-Learning-AI-Agent-library-framework
```

### 2. Build the Docker image
```bash
# Build image with tag
docker build -t integrated-pipeline:latest .

# Or with a specific version
docker build -t integrated-pipeline:v1.0.0 .
```

### 3. Run integratedPipeline
```bash
docker run -it --name integrated-agent integrated-pipeline:latest python
```
[=] For python shell:
```
from AbstractIntegratedModule import IntegratedPipeline, PipelinePredictionManager

model = IntegratedPipeline('agent_memory')
manager = PipelinePredictionManager(model)
print("✓ IntegratedPipeline initialized successfully!")
```

### 4. Run bash shell
```bash
# Access full container shell
docker run -it integrated-pipeline:latest bash

# Inside container:
python main.py
```

### 5. Run python script
```bash
# Mount current directory and run script
docker run -it -v $(pwd):/app/data integrated-pipeline:latest python /app/data/your_script.py
```

### 6. Start a container
```bash
docker run -it --name my-agent integrated-pipeline:latest

# list running containers:
docker ps -a

```

### 7. stop Container
```bash
docker stop my-agent
```

### 8. Image Management
```bash
docker images

# view image details:
docker inspect integrated-pipeline:latest

# remove images
docker rmi integrated-pipeline:latest

# rebuild image (no-cache)
docker build --no-cache -t integrated-pipeline:latest .
```

### 9. Volume Mounting
[=] Mount local directory for data access
```bash
# Linux/Mac
docker run -it -v $(pwd)/data:/app/data integrated-pipeline:latest python main.py

# Windows PowerShell
docker run -it -v ${PWD}/data:/app/data integrated-pipeline:latest python main.py

# Windows CMD
docker run -it -v %cd%/data:/app/data integrated-pipeline:latest python main.py
```

### View logs
```bash
# View container logs
docker logs my-agent

# Follow logs in real-time
docker logs -f my-agent

# Last 50 lines
docker logs --tail 50 my-agent
```
