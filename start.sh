#!/bin/bash
# Start single agent with defaults
docker run -it --rm \
  -v $(pwd)/data:/app/data \
  integrated-pipeline:latest python main.py
