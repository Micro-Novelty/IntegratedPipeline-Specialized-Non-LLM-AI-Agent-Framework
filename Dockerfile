# Stage 1: Builder (for multi-architecture selection)
FROM alpine:latest AS amd64
COPY AbstractIntegratedModule.cpython-39-x86_64-linux-gnu.so /libs/AbstractIntegratedModule.so

FROM alpine:latest AS arm64
COPY AbstractIntegratedModule.cpython-39-aarch64-linux-gnu.so /libs/AbstractIntegratedModule.so

# Stage 2: Runtime base
FROM python:3.13-slim AS runtime

# Install runtime dependencies (minimal - just Python)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash agentuser

# Set working directory
WORKDIR /app

# Copy the appropriate .so based on architecture
ARG TARGETARCH
COPY --from=amd64 /libs/AbstractIntegratedModule.so /app/ 2>/dev/null || true
COPY --from=arm64 /libs/AbstractIntegratedModule.so /app/ 2>/dev/null || true

# Copy application code
COPY main.py .
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Set ownership
RUN chown -R agentuser:agentuser /app

# Switch to non-root user
USER agentuser

# Environment variables (configurable)
ENV PYTHONUNBUFFERED=1
ENV MEMORY_NAME=default_agent
ENV AGENT_MODE=single
ENV AGENT_PORT=5555
ENV DB_PATH=/app/data/activity_log.db

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import AbstractIntegratedModule" || exit 1

# Entrypoint
ENTRYPOINT ["./entrypoint.sh"]

# Default command
CMD ["python", "main.py"]
