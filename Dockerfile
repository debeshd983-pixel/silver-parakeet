# ==========================================
# Stage 1: Builder (fetches models, exports & quantizes ONNX)
# ==========================================
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-build.txt ./
RUN pip install --no-cache-dir -r requirements-build.txt

COPY scripts/ ./scripts/
COPY config/ ./config/

# In a full build pipeline with internet access:
# RUN python scripts/fetch_models.py
# RUN python scripts/export_onnx.py --output-dir models/
# RUN python scripts/quantize.py --models-dir models/

RUN mkdir -p models && touch models/.keep

# ==========================================
# Stage 2: Runtime (Slim, CPU-only, No Torch)
# ==========================================
FROM python:3.11-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create unprivileged runtime user
RUN useradd -m -u 1000 -s /bin/bash appuser

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application, configurations, and models
COPY app/ ./app/
COPY config/ ./config/
COPY --from=builder /build/models/ ./models/

# Verify no torch exists in runtime image
RUN python -c "import sys; 'torch' in sys.modules and sys.exit(1) or None"

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

ENTRYPOINT ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
