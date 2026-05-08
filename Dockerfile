# Stage 1: Builder (for compilation)
FROM python:3.13-slim AS builder

FROM python:3.13-slim

# Create non-root user
RUN useradd -m -u 1000 aweuser

WORKDIR /app

# Copy only what's needed
COPY --chown=aweuser:aweuser *.so .
COPY --chown=aweuser:aweuser main.py .

USER aweuser

# No pip install needed!
# test_installation.py could be changed 
CMD ["python", "test_installation.py"] 

# for multi agent support
# Stage 1: Copy pre-compiled libraries for different architectures
FROM alpine:latest AS amd64
COPY awe_mlp.cpython-39-x86_64-linux-gnu.so /libs/x86_64/

FROM alpine:latest AS arm64
COPY awe_mlp.cpython-39-aarch64-linux-gnu.so /libs/aarch64/

# Stage 2: Final runtime
FROM python:3.13-slim

# Detect architecture and copy correct .so
COPY --from=amd64 /libs/x86_64/* /app/ 2>/dev/null || true
COPY --from=arm64 /libs/aarch64/* /app/ 2>/dev/null || true

# Fallback to any .so file
COPY *.so . 2>/dev/null || true

COPY main.py .

CMD ["python", "test_installation.py"]
