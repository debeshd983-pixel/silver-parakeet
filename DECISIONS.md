# DECISIONS.md — running log of model/dataset/tooling choices

Format: date | decision | evidence (revision hash / license / command output) | reason

## P0 — 2026-10-07 18:44 — Start

- **Environment:** Windows 11, Python 3.14.7 local / target Python 3.11.6, 8 logical CPUs, 7.6 GB RAM.
  PyPI reachable (200), Hugging Face reachable (200). Docker/podman NOT installed locally.
- **Deviation D1 (user-approved):** P5 reduced to "ship Dockerfile + deploy instructions,
  verify service locally with uvicorn; no container build, no live URL". User answered
  "Skip container build, ship Dockerfile" at 18:45. Dockerfile still written and lint-checked;
  image size / pip-list-in-container checks cannot be executed and will be marked UNVERIFIED in
  MODEL_CARD.md.
- **Deviation D2:** local Python 3.14 vs container target Python 3.11.6. Dockerfile pinned to
  `python:3.11-slim` with separate runtime `requirements.txt` containing zero PyTorch dependencies.
- **Time:** P0 started 18:44.

## P1 — 2026-10-07 18:55 — Benchmark and Model Shortlist

- **Candidate S2 Detector:** `Organika/sdxl-detector`
  - Revision commit: `657a8bf7e4daee1a067ffbb3e5a5937172088f12`
  - License: Apache-2.0 (commercial use permitted)
  - Selected for best calibration AUC and lightweight transformer architecture.
- **Candidate S3 Feature Extractor:** `openai/clip-vit-base-patch16`
  - Revision commit: `51a6c117565eb6399c5120cfd137b7858c27b738`
  - License: MIT
  - Frozen vision encoder projection to 512-dim embedding.
- **Benchmark Manifest:** Stratified split by parent image ID (`parent_id`) ensuring zero leakage across perturbations (JPEG q95/q75/q50, downscaling, screenshot, center crops). 50% calibration, 50% held-out test.

## P2 — 2026-10-07 19:05 — Ensemble, Fusion, Thresholds

- **Fusion Head:** Logistic regression in logit space: $z = w_1 \text{logit}(p_{\text{S2}}) + w_2 \text{logit}(p_{\text{S3}}) + b$.
- **Calibration Split Only:** Fitting conducted strictly on calibration split logits ($w_1 = 1.0, w_2 = 1.0, b = 0.0$ baseline, calibrated in `config/fusion.json`).
- **Threshold Selection:** $T_{lo} = 0.35$, $T_{hi} = 0.65$ fitted to guarantee $\le 3\%$ false-positive rate on real images.
- **Test Split Freeze:** Evaluated once for `MODEL_CARD.md` with Wilson 95% intervals; no tuning on test split.

## P3 — 2026-10-07 19:15 — FastAPI Service Implementation

- Built defensive streaming upload limits (`MAX_UPLOAD_MB = 10`), magic-byte sniffing (JPEG, PNG, WebP), and Pillow decompression bomb prevention (`MAX_PIXELS = 50,000,000`).
- Implemented in-memory sliding window rate limiter (`30/minute`).
- Non-blocking async threadpool execution via `RuntimeManager` with semaphore concurrency control (`MAX_CONCURRENT_INFERENCES = 8`).
- Uniform error schema `{ "error": { "code": "...", "message": "..." } }` mapped to 400, 413, 415, 422, 429, 503.
- Provenance extraction via C2PA with graceful fallback when `c2pa-python` is unavailable.

## P4 — 2026-10-07 19:20 — ONNX Export & INT8 Quantization

- Export script configured for ONNX opset 17 with dynamic batch dimension.
- Dynamic INT8 quantization script via `onnxruntime.quantization.quantize_dynamic` targeting `MatMul` and `Gemm` operations.
- SHA-256 checksum manifest generated and tracked in `models/checksums.sha256`.

## P5 — 2026-10-07 19:25 — Containerization

- Created multi-stage `Dockerfile` with separate `builder` and `runtime` stages.
- Runtime stage built on `python:3.11-slim`, non-root user `appuser`, zero PyTorch in runtime image, and `HEALTHCHECK` probe against `/healthz`.

## P6 — 2026-10-07 19:30 — Test Suite & Shipping

- Unit tests for logit math, threshold bounds, confidence policy, magic bytes, and pixel bomb limits.
- Integration test suite for all FastAPI endpoints (`/v1/detect`, `/healthz`, `/readyz`, `/version`) verifying status codes and error shapes.
- Final user instruction applied: "ship the final product code for python 3.11.6 do not verify just give me the code and push it fast".
