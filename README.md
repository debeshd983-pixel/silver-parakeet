# AI-Generated Image (Deepfake) Detection Backend

A lightweight, accurate, CPU-only AI-generated image detection service built with **FastAPI**, **ONNX Runtime**, and calibrated ensemble fusion.

> **CRITICAL NOTICE:** The output is a **calibrated probability and heuristic signal, NOT proof.** See [MODEL_CARD.md](MODEL_CARD.md) for full limitations and ethical use guidelines.

---

## Key Features

- **Lightweight CPU Runtime:** Uses ONNX Runtime with INT8 dynamic quantization. **No PyTorch or Transformers** in the runtime environment.
- **Accurate Ensemble:** Fuses fine-tuned transformer classifier (S2) with frozen CLIP ViT-B/16 linear probe (S3) in logit space.
- **C2PA Provenance:** Inspects Content Credentials (S1) for verified cryptographic generation metadata.
- **Honest Abstain Band:** Explicit `inconclusive` zone ($T_{lo} \le p \le T_{hi}$) preventing false accusations.
- **Production-Ready:** Streamed file uploads with size caps, magic-byte sniffing, decompression bomb guards, structured JSON logging (zero image persistence), and rate limiting.

---

## Quickstart

### 1. Run Locally with Python 3.11

```bash
# Create venv and install runtime requirements
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Start the uvicorn server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

### 2. Run with Docker

```bash
# Build multi-stage image
docker build -t ai-image-detector:latest .

# Run container (memory capped at 1GB, 2 CPUs)
docker run -p 8000:8000 --cpus=2 --memory=1g ai-image-detector:latest
```

---

## API Reference

### `POST /v1/detect`

Accepts multipart form-data with an image file (`JPEG`, `PNG`, `WebP`).

**Example Request:**
```bash
curl -X POST http://localhost:8000/v1/detect \
  -F "file=@sample.jpg"
```

**Example Response (200 OK):**
```json
{
  "request_id": "9f1c7d2e4a8b",
  "verdict": "likely_ai",
  "ai_probability": 0.8845,
  "confidence": "high",
  "signals": {
    "c2pa": { "present": false, "ai_declared": null },
    "classifier": { "model": "Organika/sdxl-detector@657a8bf", "probability": 0.912 },
    "clip_probe": { "model": "openai/clip-vit-base-patch16@51a6c11", "probability": 0.814 }
  },
  "warnings": [],
  "model_version": "2026.10.0",
  "benchmark_id": "bench-a1b2c3d4e5f6",
  "latency_ms": 384.25
}
```

### Other Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/healthz` | `GET` | Liveness check (always 200) |
| `/readyz` | `GET` | Readiness check (200 once models loaded & warmed up; 503 while loading) |
| `/version` | `GET` | Active model IDs, commit revisions, thresholds, and fusion weights |

---

## Configuration

All configuration is handled via environment variables (see `app/config.py`):

| Variable | Default | Description |
|---|---|---|
| `MAX_UPLOAD_MB` | `10` | Hard cap on upload file size |
| `MAX_PIXELS` | `50000000` | Decompression bomb guard pixel cap |
| `MAX_CONCURRENT_INFERENCES`| `8` | Semaphore limit for concurrent CPU inference |
| `ORT_INTRA_THREADS` | `8` | Thread count for ONNX Runtime session |
| `ENABLE_C2PA` | `true` | Extract Content Credentials (falls back gracefully if unavailable) |
| `ENABLE_TTA` | `false` | Enable 5-crop test-time augmentation |
| `RATE_LIMIT` | `30/minute` | Rate limiter window per IP/key |
| `API_KEYS` | `""` | Comma-separated allowed keys for `X-API-Key` header |
| `ALLOWED_ORIGINS` | `""` | CORS allowed origins |

---

## Development & Build Pipeline

Build-time tools (benchmark creation, training probe, ONNX export, and quantization) require `requirements-build.txt`:

```bash
pip install -r requirements-build.txt

# 1. Build benchmark dataset
python scripts/build_benchmark.py

# 2. Fetch models by pinned commit hash
python scripts/fetch_models.py

# 3. Export to ONNX
python scripts/export_onnx.py

# 4. Quantize to INT8 dynamic
python scripts/quantize.py

# 5. Fit fusion and calibrate thresholds on calibration split
python scripts/fit_fusion.py

# 6. Evaluate once on held-out test split
python scripts/evaluate.py
```

---

## Running Tests

```bash
pytest
```
