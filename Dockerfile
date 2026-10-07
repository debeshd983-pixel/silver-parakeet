# Single-stage runtime image.
#
# The INT8 checkpoint (~92 MB) ships inside the Python package, so there is no build
# stage, no Hugging Face fetch at build time, and no model download at container start.
# That also means the image build is fully reproducible offline.
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 -s /bin/bash appuser

COPY pyproject.toml README.md LICENSE ./
COPY sahu65/ ./sahu65/

# Install the package itself (pulls fastapi/onnxruntime/etc. from PyPI).
RUN pip install --no-cache-dir .

# Verify no torch leaked into the runtime image.
RUN python -c "import sys; sys.exit(1) if 'torch' in sys.modules else None"

# Verify the bundled checkpoint loads and reports a sane label mapping.
RUN python -c "import sahu65; i=sahu65.model_info(); assert i['loaded'] and i['ai_class_index']==0, i"

USER appuser

EXPOSE 8000

ENV MODEL_DIR=/app/sahu65/models \
    CONFIG_DIR=/app/sahu65/config \
    REQUIRE_REAL_MODEL=true \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Gate on /readyz, not /healthz. ONNX Runtime's ~10s session init holds the GIL, so the
# first HTTP response only lands once weights are resident. start-period must cover that.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/readyz || exit 1

ENTRYPOINT ["uvicorn", "sahu65.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]