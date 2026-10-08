# Architecture: AI-Generated Image Detection Backend

> Status: design v1. Targets are goals to be verified by the benchmark, not promises.
> "Rectify" is interpreted as **detect and classify**. The service never modifies images.

---

## 1. Goals and non-goals

### Goals (the three requirements)

| Requirement | Concrete target |
|---|---|
| **Light** | Docker image < 1.5 GB, RAM < 1 GB at steady state, CPU-only, no torch at runtime |
| **Accurate** | Measured on our own held-out benchmark; FPR on real photos <= 3% for committed verdicts; accuracy >= 85% on committed verdicts; abstain rate <= 25% |
| **Deployable** | One Docker image, stateless, env-var config, runs on Cloud Run / Fly.io / Render / HF Spaces free or cheap tiers |

### Non-goals

- Fixing, de-fake-ing, or watermarking images.
- Video or audio detection.
- Training detectors from scratch.
- Heavy forensic methods (DIRE, diffusion reconstruction). They break "light".
- Persisting user images. The service is stateless and stores nothing.

---

## 2. System overview

```
                         +-----------------------------------------+
  client  --multipart--> |  FastAPI (uvicorn, 1-2 workers)         |
                         |                                         |
                         |  1. validate (type, size, pixels)       |
                         |  2. decode (Pillow, EXIF orient)        |
                         |  3. read provenance (C2PA / EXIF)       |
                         |  4. preprocess -> two tensors           |
                         |  5. run signals in threadpool           |
                         |       +--> Classifier  (ONNX int8)      |
                         |       +--> CLIP probe  (ONNX int8+head) |
                         |  6. fuse + calibrate (logit space)      |
                         |  7. apply thresholds -> verdict         |
                         +-------------------+---------------------+
                                             |
                            JSON response  <-+
```

All inference is in-process. There is no queue, database, or cache in v1.
That keeps the system small. If throughput demands it later, the clean
seam for scaling is horizontal replicas behind the platform load balancer.

---

## 3. Components

### 3.1 Signals

| # | Signal | Purpose | Failure mode |
|---|---|---|---|
| S1 | **Provenance (C2PA / EXIF)** via `c2pa-python` | Near-certain positive when a signed manifest declares AI generation | Absence means nothing. Most images carry no manifest, and stripping is trivial |
| S2 | **Fine-tuned image classifier** (a small Swin/ViT-style detector from Hugging Face, exported to ONNX) | Strong on generators it saw in training | Drops hard on unseen generators and after recompression |
| S3 | **CLIP linear probe** (frozen CLIP ViT-B/16 image encoder in ONNX + small logistic head), UnivFD-style | Better generalization to unseen generators | Lower peak accuracy; threshold is poorly calibrated without recalibration |

S2 and S3 fail on different images, which is the reason for the ensemble.

**Candidate S2 checkpoints** (verify availability, license, and size before choosing; do not assume these IDs are still valid):
`Organika/sdxl-detector`, `umm-maybe/AI-image-detector`, `prithivMLmods/Deep-Fake-Detector-v2-Model`.
Pick by benchmark result and license, not by README claims.

### 3.2 Fusion and calibration

Work in logit space, which is the standard way to combine detectors.

```
z = w1 * logit(p_S2) + w2 * logit(p_S3) + b
p_ai = sigmoid(z)
```

`w1, w2, b` are fitted by logistic regression on the **calibration split**
of the benchmark. This does fusion and calibration in one step (Platt-style).
Use isotonic regression only if the calibration set exceeds roughly 1,000 images;
below that it overfits.

Clip probabilities to [1e-4, 1 - 1e-4] before taking logits.

### 3.3 Verdict policy

| Condition | Verdict |
|---|---|
| Valid C2PA manifest declaring trained-algorithmic media | `ai_generated_verified` (p_ai = 0.99, fusion bypassed) |
| `p_ai >= T_hi` | `likely_ai` |
| `p_ai <= T_lo` | `likely_real` |
| otherwise | `inconclusive` |

`T_lo` and `T_hi` are **chosen on the calibration split** so that the false-positive
rate on real images (real flagged as AI) is <= 3%. They are stored in `config/thresholds.json`
with the benchmark hash that produced them. Initial placeholders: `T_lo = 0.35`, `T_hi = 0.65`.

A valid C2PA manifest that is **not** AI never produces `likely_real`.
It only appears as a signal, because provenance data can be real on an image that was later composited.

`confidence` is derived from distance to the nearest threshold plus signal agreement:
`high` if both signals agree and are far past the threshold, `medium` if they agree, `low` if they disagree.

### 3.4 Preprocessing

- Decode with Pillow, apply EXIF orientation, convert to RGB.
- **Read provenance before stripping anything.**
- S2: resize and normalize exactly as its model card specifies. Mismatched preprocessing is the most common silent accuracy bug.
- S3: CLIP preprocessing (center crop 224, CLIP mean/std). Optional TTA flag averages 5 crops, adding latency, off by default.
- Never upscale small images silently. If the shorter side is under 128 px, add the warning `low_resolution` and downgrade confidence.

---

## 4. API contract

All routes are mounted under the **`/deep-guard`** prefix (a single `APIRouter(prefix=...)`
in `sahu65/api/routes.py`). There is no `/v1` version segment; the prefix is the version
boundary.

### `POST /deep-guard/detect`

Multipart form, field `file`. Allowed types: JPEG, PNG, WebP (detected by content sniffing, not by extension or the client's header).

Response `200`:

```json
{
  "request_id": "9f1c...",
  "verdict": "likely_ai",
  "ai_probability": 0.87,
  "confidence": "medium",
  "signals": {
    "c2pa": { "present": false, "ai_declared": null },
    "classifier": { "model": "<name>@<revision>", "probability": 0.91 },
    "clip_probe": { "model": "<name>@<revision>", "probability": 0.79 }
  },
  "warnings": ["jpeg_recompressed_likely"],
  "model_version": "2026.10.0",
  "benchmark_id": "bench-<hash>",
  "latency_ms": 412
}
```

Errors use a uniform shape `{ "error": { "code": "...", "message": "..." } }`:

| Status | Code | Cause |
|---|---|---|
| 400 | `invalid_image` | Cannot decode |
| 413 | `file_too_large` | Over `MAX_UPLOAD_MB` |
| 415 | `unsupported_type` | Not JPEG/PNG/WebP |
| 422 | `image_too_large_pixels` | Over `MAX_PIXELS` (decompression bomb guard) |
| 429 | `rate_limited` | Rate limit hit |
| 503 | `model_not_ready` | Models still loading |

### Other endpoints

| Endpoint | Purpose |
|---|---|
| `GET /deep-guard/healthz` | Liveness, always cheap |
| `GET /deep-guard/readyz` | 200 only after models are loaded and a warm-up inference has run |
| `GET /deep-guard/version` | Model names, revisions, thresholds, benchmark ID |
| `GET /deep-guard/metrics` | Prometheus metrics (optional, behind `ENABLE_METRICS`) |

The OpenAPI docs at `/docs` are disabled in production by default.

---

## 5. Repository layout

```
deepfake-detector/
├── app/
│   ├── main.py              # app factory, lifespan (model load + warm-up)
│   ├── config.py            # pydantic-settings, all env vars
│   ├── schemas.py           # request/response models
│   ├── api/routes.py
│   ├── core/
│   │   ├── limits.py        # size/pixel/type validation, rate limit
│   │   └── logging.py       # structured JSON logs, no image data
│   └── services/
│       ├── preprocess.py
│       ├── provenance.py    # C2PA/EXIF
│       ├── classifier.py    # S2 ONNX session wrapper
│       ├── clip_probe.py    # S3 ONNX encoder + head
│       ├── ensemble.py      # fusion, thresholds, verdict, confidence
│       └── runtime.py       # onnxruntime session options, threadpool
├── config/
│   ├── thresholds.json
│   └── fusion.json          # w1, w2, b + benchmark ID
├── scripts/                 # build-time only, may use torch
│   ├── fetch_models.py
│   ├── export_onnx.py
│   ├── quantize.py
│   ├── build_benchmark.py
│   ├── evaluate.py
│   ├── fit_fusion.py
│   └── train_probe.py
├── tests/
├── benchmark/               # gitignored; manifest + hashes tracked, images not
├── models/                  # gitignored; fetched/exported at build
├── Dockerfile               # multi-stage
├── requirements.txt         # runtime only (no torch)
├── requirements-build.txt   # build-time (torch, transformers, optimum)
├── MODEL_CARD.md            # measured numbers, limits, licenses
└── README.md
```

**Key rule:** torch and transformers exist only in `requirements-build.txt` and the build stage.
The runtime image has onnxruntime, numpy, Pillow, FastAPI, and c2pa-python.
This is what keeps the image light.

---

## 6. Inference runtime

- **ONNX Runtime, CPUExecutionProvider.** Set `intra_op_num_threads` to the container's vCPU count, `inter_op_num_threads=1`, and enable graph optimization.
- **int8 dynamic quantization** for the transformer MatMul layers. Accept it only if the quantized model's accuracy on the calibration split is within 1 percentage point of fp32 and AUC drops by < 0.01. Otherwise ship fp32.
- **Concurrency:** `run_in_threadpool` around inference so the event loop stays free. Cap in-flight inferences with a semaphore (default = vCPU count), and queue the rest briefly before returning 429 or 503. Avoids memory blowups under load.
- **Warm-up:** run one dummy inference in the app lifespan before `/deep-guard/readyz` goes green, so the first real request is not slow.
- **Workers:** 1 uvicorn worker per container; scale with replicas. Multiple workers each load models and multiply RAM.

### Performance budget (targets, 2 vCPU, ~12 MP JPEG)

| Metric | Target |
|---|---|
| p50 latency | < 600 ms |
| p95 latency | < 1.5 s |
| RAM | < 1 GB |
| Cold start (to ready) | < 20 s |
| Docker image | < 1.5 GB |

If targets are missed, the order of fixes is: confirm int8 is in use, reduce decode cost (use `Image.draft()` for JPEG), cut TTA, then switch S3 to a smaller encoder.

---

## 7. Security and privacy

- Validate by **content sniffing** (magic bytes), not extension.
- Set `Image.MAX_IMAGE_PIXELS` and enforce `MAX_PIXELS` before full decode (use `Image.open` lazily and check `.size`).
- Hard request body limit enforced at the server level, not only after reading.
- Per-request timeout (default 15 s).
- Images are processed in memory and **never written to disk or logged**. Logs contain request ID, sizes, timings, verdict, and optionally a short SHA-256 prefix.
- Optional **Bear Token** auth on `/deep-guard/detect`: named keys minted with `sahu65 --key <name>` (stored as SHA-256 digests in `SAHU65_KEYFILE`, default `~/.sahu65/keys.json`) or plaintext `API_KEYS`, presented via `Authorization: Bearer` (preferred) or `X-API-Key`. Checked before readiness. Rate limiting by verified token or IP.
- CORS locked to `ALLOWED_ORIGINS`.
- Container runs as a non-root user with a read-only filesystem where the platform allows.
- Dependencies pinned; run `pip-audit` in CI.
- **Model supply chain:** pin Hugging Face models by commit revision hash, not `main`. Verify SHA-256 of ONNX files at startup.

---

## 8. Configuration (environment variables)

| Var | Default | Notes |
|---|---|---|
| `MAX_UPLOAD_MB` | 10 | |
| `MAX_PIXELS` | 50_000_000 | |
| `REQUEST_TIMEOUT_S` | 15 | |
| `MAX_CONCURRENT_INFERENCES` | vCPU count | |
| `ORT_INTRA_THREADS` | vCPU count | |
| `ENABLE_TTA` | false | |
| `ENABLE_C2PA` | true | Set false if native wheel is unavailable on the platform |
| `API_KEYS` | empty (open) | |
| `RATE_LIMIT` | `30/minute` | |
| `ALLOWED_ORIGINS` | empty | |
| `ENABLE_DOCS` | false | |
| `ENABLE_METRICS` | false | |
| `LOG_LEVEL` | INFO | |

---

## 9. Evaluation architecture

The benchmark is a first-class part of the system, not a one-off script.

**Composition (minimum ~500 images, aim for 1,000 if time allows):**

- **Real:** COCO val / Open Images subset, plus any rights-cleared phone photos. Include varied sizes.
- **AI:** images from at least 4 distinct generator families (for example SDXL, a Midjourney-style set, DALL-E-style, and a recent one such as Flux if obtainable). Use public datasets such as GenImage, Synthbuster, or Hugging Face AI-vs-real sets. **Verify each dataset's license and avoid using only CIFAKE** (32x32 images, unrepresentative).
- **Perturbed variants** of a subset: JPEG q95/q75/q50, 0.5x resize, resize + JPEG ("screenshot-like"), center crop.

**Splits:** 50% calibration (fit fusion weights + choose thresholds), 50% test (report only, never tune). Split by source image, so perturbed copies of one image never straddle the splits.

**Report (in `MODEL_CARD.md`):**

- Overall accuracy, AUC, and **false-positive rate on real photos**, each with Wilson 95% confidence intervals.
- Per-generator recall.
- Per-perturbation accuracy.
- Abstain rate and accuracy on committed verdicts.
- Reliability diagram or ECE for calibration.

Because of the small sample size, intervals will be wide. Say so in the model card.

---

## 10. Deployment

**Dockerfile (multi-stage):**

1. `builder` stage: install build requirements, fetch models by pinned revision, export to ONNX, quantize, verify, write SHA-256 manifest.
2. `runtime` stage: `python:3.11-slim`, install runtime requirements only, copy ONNX files + config, create non-root user, `HEALTHCHECK` against `/deep-guard/readyz`, start with `uvicorn --workers 1`.

**Target platforms**, in order of preference for this workload:

| Platform | Why | Watch out for |
|---|---|---|
| Google Cloud Run | Scale to zero, simple container deploy | Cold starts; set min instances = 1 if latency matters |
| Fly.io | Cheap always-on small VM | Memory sizing (use 1 GB+) |
| Render | Simple | Free tier sleeps |
| Hugging Face Spaces (Docker) | Free demo | Limited CPU |

CI (GitHub Actions): lint, unit tests, build image, run the smoke test against the built container, `pip-audit`. Deploy on tag.

---

## 11. Observability

- Structured JSON logs with request ID.
- Metrics: request count by verdict and status, latency histogram, in-flight gauge, model-load time.
- **Drift watch:** track the share of `inconclusive` verdicts and the mean `p_ai` over time. A rise in abstentions is the earliest sign a new generator is beating the models.

---

## 12. Key decisions and trade-offs

| Decision | Alternative | Why this choice |
|---|---|---|
| Ensemble of two detectors | Single best model | Different failure modes; more robust to new generators |
| Logit-space logistic fusion | Simple average | Calibrates and weights in one step with little data |
| Abstain band (`inconclusive`) | Binary verdict | Honest about uncertainty; protects against false accusations |
| ONNX Runtime, no torch at runtime | PyTorch serving | Smaller image, faster CPU inference |
| Stateless, no DB or queue | Async job queue | Meets "light"; replicas scale horizontally |
| C2PA as override, never as proof of "real" | Treat C2PA as real/fake oracle | Absence is meaningless; presence can be stale |
| Head-only fine-tuning if needed | Full fine-tune | Fits the time budget and keeps weights tiny |

---

## 13. Known limitations (must appear in README and model card)

1. No open detector generalizes across all generators; accuracy falls on generators released after the training data.
2. Heavy compression, screenshots, and resizing reduce accuracy.
3. Heavily edited, upscaled, or beauty-filtered real photos can be false positives.
4. Adversarial evasion is easy (small perturbations, re-encoding). The service is not a forensic proof.
5. Output is a probabilistic signal and must not be the sole basis for accusing anyone of fabricating media.
