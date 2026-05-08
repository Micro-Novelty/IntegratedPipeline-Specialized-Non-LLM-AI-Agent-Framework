# Stage 1: Multi-architecture binary selection
FROM alpine:latest AS amd64
COPY AbstractIntegratedModule.cpython-39-x86_64-linux-gnu.so /libs/AbstractIntegratedModule.so

FROM alpine:latest AS arm64
COPY AbstractIntegratedModule.cpython-39-aarch64-linux-gnu.so /libs/AbstractIntegratedModule.so

# Stage 2: Final runtime
FROM python:3.13-slim

# Create non-root user
RUN useradd -m -u 1000 integratedpipelineuser

WORKDIR /app

# Copy the correct binary based on architecture
ARG TARGETARCH
COPY --from=amd64 /libs/AbstractIntegratedModule.so /app/AbstractIntegratedModule.so 2>/dev/null || true
COPY --from=arm64 /libs/AbstractIntegratedModule.so /app/AbstractIntegratedModule.so 2>/dev/null || true

# Fallback: copy any .so file if multi-arch fails
COPY *.so /app/ 2>/dev/null || true

# Copy your main script
COPY main.py /app/

# Set ownership
RUN chown -R integratedpipelineuser:integratedpipelineuser /app

USER integratedpipelineuser

CMD ["python", "main.py"]
