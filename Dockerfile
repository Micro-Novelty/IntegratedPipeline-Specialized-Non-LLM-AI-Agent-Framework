# Stage 1: Builder (for compilation)
FROM python:3.13-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    python3-dev \
    cython3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy only requirements first (for caching)
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Copy your Cython source
COPY *.pyx .
COPY setup.py .

# Build your .so files
RUN python setup.py build_ext --inplace

# Stage 2: Runtime (smaller image)
FROM python:3.13-slim

# Create non-root user
RUN useradd --create-home --shell /bin/bash aweuser

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy built artifacts from builder
COPY --from=builder /build/*.so /app/
COPY --from=builder /root/.local /home/aweuser/.local

# Copy application code
COPY . .

# Set ownership to non-root user
RUN chown -R aweuser:aweuser /app

# Switch to non-root user
USER aweuser

# Python path
ENV PYTHONPATH=/app
ENV PATH=/home/aweuser/.local/bin:$PATH

# Run
# can use test_installation.py or main.py (your_script)
CMD ["python", "test_installation.py"] 

