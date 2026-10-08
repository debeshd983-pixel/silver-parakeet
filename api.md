# API Reference

Service: **AI-Generated Image Detector** (`sahu65`)
All routes are mounted under the **`/deep-guard`** prefix (single `APIRouter(prefix=...)` in `sahu65/api/routes.py`). There is no `/v1` segment — the prefix is the version boundary.

- Base URL (default): `http://localhost:8000`
- Content type: `application/json` (except `/detect`, which accepts `multipart/form-data`)
- Interactive docs: `GET /docs` (only when `ENABLE_DOCS=true`, disabled by default)

---

## Endpoint Summary

| # | Method | Endpoint | Type | Purpose |
|---|--------|----------|------|---------|
| 1 | GET | `/deep-guard/healthz` | Liveness probe | Cheap liveness check; always `200` while the process runs |
| 2 | GET | `/deep-guard/readyz` | Readiness probe | `200` only once models are loaded and warmed; `503` otherwise |
| 3 | GET | `/deep-guard/version` | Metadata | Model versions, thresholds, fusion weights, calibration status |
| 4 | POST | `/deep-guard/detect` | Inference (multipart upload) | Detects whether an uploaded image is AI-generated |

---

## 1. `GET /deep-guard/healthz`

**Type:** Liveness probe (no auth, no rate limit side effects)

**Work:** Returns a constant OK payload as soon as the socket is bound. Use it for container/process liveness — *not* for gating traffic, because it answers `200` even while models are still loading.

**Input:** none.

**Output — `200 OK`:**

```json
{ "status": "ok" }
```

---

## 2. `GET /deep-guard/readyz`

**Type:** Readiness probe (no auth)

**Work:** Reports whether the service can actually serve detections. Models load in the background (~8s of ONNX weight deserialization), so this returns `503` until warm-up inference has completed or if model loading failed.

**Input:** none.

**Output — `200 OK` (ready):**

```json
{ "status": "ready" }
```

**Output — `503 Service Unavailable`:**

```json
{ "error": { "code": "model_not_ready", "message": "Models still initializing" } }
```

```json
{ "error": { "code": "model_load_failed", "message": "<load error text>" } }
```

---

## 3. `GET /deep-guard/version`

**Type:** Metadata read (no auth)

**Work:** Returns version and tuning metadata: model IDs, decision thresholds, fusion parameters, and whether thresholds are calibrated on real data.

**Input:** none.

**Output — `200 OK`:**

| Field | Type | Description |
|---|---|---|
| `model_version` | string | API/model release, e.g. `"2026.10.0"` |
| `benchmark_id` | string \| null | Benchmark the thresholds were fitted on |
| `thresholds` | object | `{"T_lo": <float>, "T_hi": <float>}` — abstain band boundaries |
| `fusion` | object | `{"w1": <float>, "w2": <float>, "b": <float>}` — logit-space ensemble weights |
| `models` | object | `{"classifier": "<name>@<rev>", "clip_probe": "<name>@<rev>"}` (`"unloaded"` if not loaded) |
| `calibrated` | bool | `false` means T_lo/T_hi are unfitted placeholders → probability is **uncalibrated** |

```json
{
  "model_version": "2026.10.0",
  "benchmark_id": "bench-...",
  "thresholds": { "T_lo": 0.35, "T_hi": 0.65 },
  "fusion": { "w1": 1.0, "w2": 0.0, "b": 0.0 },
  "models": {
    "classifier": "Organika/sdxl-detector@657a8bf",
    "clip_probe": "openai/clip-vit-base-patch16@51a6c11"
  },
  "calibrated": false
}
```

---

## 4. `POST /deep-guard/detect`

**Type:** Inference, `multipart/form-data` (requires auth headers if API keys are configured; rate limited)

**Work:** Accepts an image, runs the full pipeline — C2PA provenance extraction → classifier (S2) + CLIP probe (S3) inference → logit-space fusion → threshold verdict policy → confidence rating — and returns an AI-generation verdict with an explicit abstain band.

### Input

**Request body (`multipart/form-data`):**

| Field | Required | Type | Constraints |
|---|---|---|---|
| `file` | yes | binary (image) | JPEG, PNG, or WebP — detected by **magic bytes**, not extension/content-type. Max `MAX_UPLOAD_MB` (default **10 MB**), streamed in 64 KB chunks. Max `MAX_PIXELS` (default **50,000,000**) pixels. Empty file rejected. |

**Headers:**

| Header | Required | Description |
|---|---|---|
| `Authorization: Bearer <token>` | only if auth configured | Preferred auth (Bear Token / API key) |
| `X-API-Key: <token>` | only if auth configured | Accepted as an alternative to `Authorization` |

Auth is enforced only when at least one key exists (`API_KEYS` env var or `sahu65 --key <name>` minted token); with none configured the endpoint is open. Rate limiting (default `30/minute`, `RATE_LIMIT`) is keyed by the verified token when auth is on, otherwise by client IP.

**Example:**

```bash
curl -X POST http://localhost:8000/deep-guard/detect \
  -H "Authorization: Bearer <token>" \
  -F "file=@photo.jpg"
```

### Output — `200 OK`

| Field | Type | Description |
|---|---|---|
| `request_id` | string | 12-hex-char request correlation ID |
| `verdict` | enum | `likely_ai` \| `likely_real` \| `inconclusive` \| `ai_generated_verified` |
| `ai_probability` | float (0–1) | Fused probability the image is AI-generated |
| `confidence` | enum | `high` \| `medium` \| `low` (from threshold distance + signal agreement) |
| `signals.c2pa` | object | `{ "present": bool, "ai_declared": bool\|null }` — provenance metadata |
| `signals.classifier` | object | `{ "model": "<name>@<rev>", "probability": 0–1 }` — S2 classifier |
| `signals.clip_probe` | object | `{ "model": "<name>@<rev>", "probability": 0–1 }` — S3 CLIP probe |
| `warnings` | string[] | e.g. `thresholds_unfitted`, `low_resolution`, `jpeg_recompressed_likely`, `out_of_domain_flat_text_heavy` |
| `out_of_domain` | bool | `true` when the document gate suppressed the score (only if `DOCUMENT_GATE=true`) |
| `model_version` | string | e.g. `"2026.10.0"` |
| `benchmark_id` | string \| null | Benchmark ID |
| `latency_ms` | float | End-to-end server latency |

```json
{
  "request_id": "9f1c0a2b4d5e",
  "verdict": "likely_ai",
  "ai_probability": 0.87,
  "confidence": "medium",
  "signals": {
    "c2pa": { "present": false, "ai_declared": null },
    "classifier": { "model": "Organika/sdxl-detector@657a8bf", "probability": 0.91 },
    "clip_probe": { "model": "openai/clip-vit-base-patch16@51a6c11", "probability": 0.79 }
  },
  "warnings": ["thresholds_unfitted"],
  "out_of_domain": false,
  "model_version": "2026.10.0",
  "benchmark_id": null,
  "latency_ms": 412.35
}
```

---

## Error Responses

All endpoints use one uniform error shape:

```json
{ "error": { "code": "<code>", "message": "<human-readable text>" } }
```

| Status | Code | Cause |
|---|---|---|
| 400 | `invalid_image` | Cannot decode / empty or corrupted file |
| 400 | `bad_request` | Generic HTTP 400 |
| 401 | `unauthorized` | Missing or invalid API key / token |
| 403 | `forbidden` | Forbidden |
| 404 | `not_found` | Unknown endpoint |
| 413 | `file_too_large` | Upload exceeds `MAX_UPLOAD_MB` (default 10 MB) |
| 415 | `unsupported_type` | Not JPEG/PNG/WebP (by magic bytes) |
| 422 | `image_too_large_pixels` | Dimensions exceed `MAX_PIXELS` (decompression-bomb guard) |
| 422 | `validation_error` | Invalid request parameters |
| 429 | `rate_limited` | Rate limit exceeded (default 30/minute) |
| 500 | `internal_error` | Unhandled server error |
| 503 | `model_not_ready` | Models still loading (`/detect`, `/readyz`) |
| 503 | `model_load_failed` | Model load failed (`/readyz`) |

---

## Authentication, Rate Limiting & Limits

| Setting | Env var | Default |
|---|---|---|
| API keys (comma-separated) | `API_KEYS` | empty = open access |
| Bear Token store (SHA-256 digests) | `SAHU65_KEYFILE` | `~/.sahu65/keys.json` |
| Rate limit | `RATE_LIMIT` | `30/minute` (sliding window, per token or IP) |
| Max upload size | `MAX_UPLOAD_MB` | `10` |
| Max pixels | `MAX_PIXELS` | `50_000_000` |
| Request timeout | `REQUEST_TIMEOUT_S` | `15.0` |
| C2PA provenance | `ENABLE_C2PA` | `true` |
| Test-time augmentation | `ENABLE_TTA` | `false` |
| Document gate (opt-in) | `DOCUMENT_GATE` | `false` |
| OpenAPI docs | `ENABLE_DOCS` | `false` |
| CORS origins | `ALLOWED_ORIGINS` | empty (CORS off) |

Auth check runs **before** readiness, so unauthenticated callers learn nothing about model state.

---

## Legacy App (`app/`)

The older `app/` package exposes the same handlers at unprefixed paths, with detection at **`POST /v1/detect`** (plus `/healthz`, `/readyz`, `/version`). The primary, packaged API is the `/deep-guard` one documented above (`sahu65/api/routes.py`).
