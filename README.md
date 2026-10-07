# AI-Generated Image (Deepfake) Detection Backend

A lightweight, CPU-only AI-generated image detection service and Python package built with
**FastAPI**, **ONNX Runtime**, and calibrated ensemble fusion.

> **`WORKLOG.md`** records what was changed in this repository and why, including defects
> that were **not** fixed. `DECISIONS.md` has the chronological decision log; `MODEL_CARD.md`
> has measurements and limitations.

> **CRITICAL NOTICE:** The output is **NOT proof.** See [MODEL_CARD.md](MODEL_CARD.md) for full limitations and ethical use guidelines.

> ## READ THIS BEFORE USING IT ON DOCUMENTS OR IDs
>
> This service is a **natural-image** diffusion detector. It has **no measurable
> discriminative power on document images**: measured AUC **0.375** (worse than chance)
> on real vs. fake documents. Real photographs of documents — ID cards, certificates,
> forms — are flagged AI with ~99% confidence. This was verified not to be a tuning or
> preprocessing defect (four preprocessing pipelines were compared; all left real documents
> at 0.996–0.999).
>
> Document-image detection is an unresolved research problem as of 2026 (TextFake: no
> evaluated method exceeds 80% accuracy; best published result on AI-forged documents is
> 0.751 AUC). **Do not use this service for document, passport, or ID fraud decisions.**
>
> The returned probability is also **uncalibrated**: `T_lo`/`T_hi` are unfitted
> placeholders, so every response carries a `thresholds_unfitted` warning and `/deep-guard/version`
> reports `"calibrated": false`.

---

## Key Features

- **Lightweight CPU Runtime:** ONNX Runtime with INT8 dynamic quantization. **91.8 MB** of
  weights, **no PyTorch or Transformers** at runtime. Weights ship inside the wheel, so
  there is no download step and no cold-start stall from fetching a model.
- **Cold start: ~10 s.** Measured from `sahu65 serve`: TCP socket listening at ~1.5 s, first
  HTTP response at ~10 s, first successful detection at ~12 s. ONNX Runtime's session
  initialisation holds the GIL, so although the socket binds early the event loop cannot
  answer requests until the weights are resident. **Configure your orchestrator readiness
  probe with `start_period` >= 30 s** (the Dockerfile does). Calling `sahu65.warm()`
  ahead of time moves this cost off your first request.
- **Measured latency:** ~200 ms per image warm (1080p JPEG, 4 CPU threads).
- **Fails loudly:** a missing or unreadable checkpoint is a hard `/deep-guard/readyz` failure
  (`503 model_load_failed`), never a silent stub probability. Previously a missing model
  returned a constant `0.50` that looked like a working model.
- **Label mapping is data-driven:** the AI class index is resolved from the checkpoint's own
  `id2label` at load time, never hardcoded. Pinned by `tests/test_classifier.py`.
- **Honest Abstain Band:** explicit `inconclusive` zone ($T_{lo} \le p \le T_{hi}$).
- **Production-Ready:** streamed uploads with size caps, magic-byte sniffing, decompression
  bomb guards, structured JSON logging (zero image persistence), and rate limiting.

### Measured behaviour (2026-10-07)

Small, unbalanced sample, reported to document failure modes — **not an accuracy claim**.

| Input class | n | Correct verdict |
|---|---|---|
| AI-generated document | 4 | 4/4 |
| Real document | 2 | **0/2** |
| Natural photograph | 12 | 10/12 |

---

## Quickstart

### Install

```bash
pip install sahu65
```

### Python SDK — local, no server, no API key

```python
import sahu65

sahu65.warm()                        # optional: pre-load weights (~10s) so call 1 is fast
result = sahu65.detect("photo.jpg")  # path, bytes, or file-like object

print(result.verdict, result.ai_probability, result.confidence)
if "out_of_domain_flat_text_heavy" in result.warnings:
    ...  # detector was untrustworthy here - see MODEL_CARD.md
```

### Python SDK — hosted deployment

```python
from sahu65 import Client

client = Client(api_key="sk-...")            # or set SAHU65_API_KEY / SAHU65_BASE_URL
client.wait_until_ready(timeout_s=120)       # useful right after a cold start
result = client.detect("photo.jpg")
```

### CLI

```bash
sahu65 detect photo.jpg        # local inference
sahu65 detect photo.jpg --json # JSON output
sahu65 serve --port 8000       # HTTP API
sahu65 info                    # bundled checkpoint + threshold state
```

### Self-host the API

```bash
uvicorn sahu65.main:app --host 0.0.0.0 --port 8000 --workers 1
```

Authenticate with `X-API-Key` once `API_KEYS` is set. **Gate traffic on
`/deep-guard/readyz`, not `/deep-guard/healthz`** — `/deep-guard/readyz` is 503 until the
real weights are resident, and it reports `model_load_failed` if the checkpoint is missing.

### 2. Run with Docker

```bash
# Build multi-stage image
docker build -t ai-image-detector:latest .

# Run container (memory capped at 1GB, 2 CPUs)
docker run -p 8000:8000 --cpus=2 --memory=1g ai-image-detector:latest
```

---

## API Reference

All routes are mounted under the **`/deep-guard`** prefix.

| Endpoint | Method | Path |
|---|---|---|
| Detection | `POST` | `/deep-guard/detect` |
| Liveness | `GET` | `/deep-guard/healthz` |
| Readiness | `GET` | `/deep-guard/readyz` |
| Version | `GET` | `/deep-guard/version` |

### `POST /deep-guard/detect`

Accepts multipart form-data with an image file (`JPEG`, `PNG`, `WebP`).

**Example Request:**
```bash
curl -X POST http://localhost:8000/deep-guard/detect \
  -F "file=@sample.jpg"
```

**Example Response (200 OK):**
```json
{
  "request_id": "9f1c7d2e4a8b",
  "verdict": "likely_ai",
  "ai_probability": 0.9959,
  "confidence": "high",
  "signals": {
    "c2pa": { "present": false, "ai_declared": null },
    "classifier": { "model": "Organika/sdxl-detector@b37fede", "probability": 0.9959 },
    "clip_probe": { "model": "unloaded", "probability": 0.5 }
  },
  "warnings": ["jpeg_recompressed_likely", "thresholds_unfitted"],
  "model_version": "2026.10.0",
  "benchmark_id": null,
  "latency_ms": 305.2
}
```

Note `clip_probe` reports `0.5`: S3 is disabled (`w2 = 0.0`) because its head was fitted on
random noise. It does not affect the result. `thresholds_unfitted` means the probability is
an uncalibrated model score.

### Other Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/deep-guard/healthz` | `GET` | Liveness check. Note: it does **not** answer until weights are loaded (~10 s), because ONNX Runtime's session init holds the GIL. Useful as a "process is up" signal, not a "model is ready" one. |
| `/deep-guard/readyz` | `GET` | **Gate traffic on this.** 200 once models loaded & warmed; 503 `model_not_ready` while loading; 503 `model_load_failed` if the checkpoint is missing |
| `/deep-guard/version` | `GET` | Model IDs, revision, thresholds, fusion weights, and a `calibrated` flag |

---

## Configuration

All configuration is handled via environment variables (see `sahu65/config.py`):

| Variable | Default | Description |
|---|---|---|
| `MAX_UPLOAD_MB` | `10` | Hard cap on upload file size |
| `MAX_PIXELS` | `50000000` | Decompression bomb guard pixel cap |
| `MAX_CONCURRENT_INFERENCES`| `8` | Semaphore limit for concurrent CPU inference |
| `ORT_INTRA_THREADS` | `8` | Thread count for ONNX Runtime session |
| `ENABLE_C2PA` | `true` | Extract Content Credentials (falls back gracefully if unavailable) |
| `ENABLE_TTA` | `false` | Enable 5-crop test-time augmentation |
| `RATE_LIMIT` | `30/minute` | Rate limiter window per IP/key |
| `REQUIRE_REAL_MODEL` | `true` | Refuse to become ready if the S2 checkpoint is missing |
| `DOCUMENT_GATE` | `false` | **UNVALIDATED, opt-in.** Suppress confident verdicts to `inconclusive` on flat/text-heavy out-of-distribution input. See below. |
| `API_KEYS` | `""` | Comma-separated allowed keys for `X-API-Key` header |
| `ALLOWED_ORIGINS` | `""` | CORS allowed origins |

### `DOCUMENT_GATE` (opt-in, unvalidated)

The classifier has negative discriminative power on document images (AUC 0.375). When
`DOCUMENT_GATE=true`, any image the classifier scores ≥0.99 but which scores inconsistently
across 9 aspect-preserving tiles is downgraded to `inconclusive` with
`out_of_domain_flat_text_heavy` and `"out_of_domain": true`.

Measured on this repo's 18-image sample:

| Group | Gate OFF | Gate ON |
|---|---|---|
| AI documents | 4/4 | **3/4** (`uyntb.png` suppressed) |
| Real documents | **0/2** | **2/2** (both → `inconclusive`) |
| Natural photographs | 10/12 | **12/12** |

**These thresholds were fitted by inspecting those same 18 images.** They are not validated
on held-out data and will not necessarily hold. The gate trades recall on AI documents for
the elimination of confident false accusations on real ones.

---

## Licensing

The S2 classifier `Organika/sdxl-detector` is licensed **cc-by-nc-3.0 — non-commercial use
only**. (Earlier revisions of this repo recorded it as Apache-2.0; that was incorrect.)
See [MODEL_CARD.md](MODEL_CARD.md) §1.

---

## Publishing to PyPI

Publishing is automated by GitHub Actions. **You never handle the token locally.**

Every push to `main` runs the whole pipeline and then publishes: ruff → pytest →
checkpoint guard → build → `verify_dist.py` → `twine check` → **smoke-test the built
wheel in a clean venv** → upload to PyPI → poll PyPI until the version is live. Pull
requests run every step **except** the upload.

### One-time setup

1. Create a PyPI API token: <https://pypi.org/manage/account/token/>
   - **First upload:** `sahu65` does not exist on PyPI yet, so a project-scoped token
     cannot be created — scope it to your **entire account**, then replace it with a
     `sahu65`-scoped token after the first successful publish.
   - This is **not** your account password.
2. In the GitHub repo: **Settings → Secrets and variables → Actions → New repository
   secret**
   - Name: `PYPI_API_TOKEN` (exact spelling)
   - Value: the token (`pypi-...`)
3. *(Recommended)* Create a GitHub Environment named `pypi`
   (**Settings → Environments**). The publish job runs in it, so protection rules with
   required reviewers turn every upload into an approval. If that environment defines
   its own `PYPI_API_TOKEN`, it **overrides** the repository secret.

### Releasing a new version

PyPI releases are immutable: a push that does not change the version uploads nothing
(`skip-existing: true`), so the job stays green without re-publishing anything. To
actually ship a change, bump the version first — `scripts/bump_version.py` keeps
`pyproject.toml` and `sahu65/__init__.py` in sync:

```bash
python scripts/bump_version.py patch --dry-run   # preview: 1.0.0 -> 1.0.1
python scripts/bump_version.py patch             # or: minor | major | 1.2.3
git add -A && git commit -m "sahu65 1.0.1"
git push                                         # push to main -> build + publish
```

Watch it land: **Actions → CI → publish to PyPI**; the job finishes by confirming
`sahu65 <version>` is live at <https://pypi.org/project/sahu65/>.

### What the workflow does

`ci.yml` (every push/PR) → ruff, pytest, checkpoint guard, then build +
`verify_dist.py` + `twine check` + **smoke-test the built wheel in a clean venv**. On a
push to `main`/`master` a final `publish` job downloads those exact verified artifacts,
authenticates with `PYPI_API_TOKEN`, uploads with `skip-existing: true`, then polls
PyPI until the built version is visible. A missing secret fails the job with an
actionable error before anything is uploaded.

`scripts/verify_dist.py` is the guard that matters most. It fails the run *before* upload
if the checkpoint is missing from the wheel, if any artifact exceeds PyPI's 100 MiB
limit, or if required files are absent. It catches the exact failure that happened during
this build, where a `MANIFEST.in` exclude silently produced a 37 KB wheel containing no
model.

### Checkpoint size — a real constraint

The INT8 checkpoint is **96,234,407 bytes (91.78 MiB)**. GitHub hard-blocks any single
file at **100 MiB**, leaving only ~8 MiB of margin, and PyPI rejects uploads over 100 MiB
after compression is applied (current artifacts are 66.6 MB).

CI checks this explicitly. If a future re-quantization pushes the file past 100 MiB, the
fix is to stop committing it and instead:

- store it as a **GitHub Release asset** and fetch it in `scripts/fetch_models.py`, or
- enable **Git LFS** for `*.onnx`.

Do not simply delete it — `require_real_model=true` means the service refuses to become
ready without weights, so a missing checkpoint is a hard startup failure by design.

---

## Development & Build Pipeline

Build-time tools require `requirements-build.txt`:

```bash
pip install -r requirements-build.txt

# 1. Re-fetch the fp32 source weights (fp32 ONNX ships in the HF repo)
python scripts/fetch_models.py

# 2. Quantize to INT8 -> models/classifier.int8.onnx
python scripts/quantize.py
```

Steps 3–6 of the original pipeline (`export_onnx.py`, `fit_fusion.py`, `evaluate.py`) are
**not currently valid**: `build_benchmark.py` generates "AI" and "real" images from the same
random-ellipse procedure, `train_probe.py` fits the CLIP head on `np.random.randn`, and
`fit_fusion.py` / `evaluate.py` operate on `np.random.beta()` draws rather than model
outputs. Numbers produced by them describe random data, not this model. Do not use them to
set thresholds or report accuracy.

## Running Tests

```bash
pytest
ruff check app tests scripts
```

`tests/test_classifier.py` pins the AI/real label mapping so the historical inverted-index
bug cannot return silently.
