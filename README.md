# AI-Generated Image (Deepfake) Detection Backend

A lightweight, CPU-only AI-generated image detection service and Python package built with
**FastAPI**, **ONNX Runtime**, and a linear probe on a frozen CLIP encoder.

> **`WORKLOG.md`** records what changed and why, including defects that were **not** fixed.
> `DECISIONS.md` has the chronological decision log. `MODEL_CARD.md` has measurements,
> limitations and the full detector comparison.

> ## READ THIS BEFORE RELYING ON A VERDICT
>
> 1. **This is a signal, not proof.** A `likely_real` verdict does not establish that an
>    image is authentic.
> 2. **Do not use it on documents or IDs.** Document forgery detection is excluded from the
>    benchmark and unsupported — the previous model measured AUC 0.375, worse than chance.
>    See [MODEL_CARD.md](MODEL_CARD.md) §3.4.
> 3. **Recall is 31% at the shipped threshold.** That is a deliberate trade to keep false
>    accusations below 3%. Roughly one AI image in three is caught. Raising `T_hi` catches
>    more and accuses more real photographs — the trade-off table is in
>    [MODEL_CARD.md](MODEL_CARD.md) §3.3.

---

## Key Features

- **Lightweight CPU Runtime:** ONNX Runtime, INT8 dynamic quantization. **85.5 MB** of
  weights, **no PyTorch or Transformers at runtime**. Weights ship inside the wheel, so
  there is no download step and no cold-start stall from fetching a model.
- **One model, ~135 ms per image.** A single 512-d linear probe on a frozen CLIP ViT-B/16
  encoder. The previous two-model ensemble was removed: measured head to head, the probe
  beat the second model on its own (AUC 0.82 vs 0.64) and fusing them was *worse* than the
  probe alone. [MODEL_CARD.md](MODEL_CARD.md) §1.
- **Permissive licence.** The bundled encoder is CLIP ViT-B/16 (MIT per its upstream model
  card). The previous checkpoint was `cc-by-nc-3.0` — non-commercial, which was a real
  problem for a package published on PyPI.
- **Cold start: ~10 s.** Socket listening at ~1.5 s, first detection at ~12 s. ONNX
  Runtime's session init holds the GIL. **Set your orchestrator readiness probe
  `start_period` ≥ 30 s** (the Dockerfile does). `sahu65.warm()` moves the cost off your
  first request.
- **Fails loudly:** a missing or unreadable checkpoint is a hard `/deep-guard/readyz`
  failure (`503 model_load_failed`), never a silent constant probability.
- **Honest abstain band:** explicit `inconclusive` zone (`T_lo` ≤ p ≤ `T_hi`).
- **Plain-language explanations:** opt-in narration of the verdict by any LLM provider.
  See [Explanations](#explanations) below.
- **Production-Ready:** streamed uploads with size caps, magic-byte sniffing, decompression
  bomb guards, structured JSON logging (zero image persistence), rate limiting, and
  **Bear Token API-key auth** (`sahu65 --key <name>`; `Authorization: Bearer` or
  `X-API-Key`).

### Measured behaviour (benchmark `bench-8e0bd9e97e3e`, 831 images, out-of-fold)

| Metric | Measured |
|---|---|
| ROC AUC | **0.8213** |
| False-positive rate on real photos | **2.88%** (target ≤ 3%) |
| Recall on AI at that threshold | **30.96%** |
| Expected calibration error | **0.0440** (0.0761 before Platt scaling) |

Per-generator recall ranges from 40% to 100%. Real photographs come from two pipelines
(RAISE-1k, COCO val2017); AI images span 13 generator families. Per-family intervals are
roughly ±10 points at this sample size. Full breakdown and the FPR/recall trade-off:
[MODEL_CARD.md](MODEL_CARD.md) §3.

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

client = Client(api_key="bear_...")            # sahu65 --key <name>; or set SAHU65_API_KEY
client.wait_until_ready(timeout_s=120)         # useful right after a cold start
result = client.detect("photo.jpg")
```

### CLI

```bash
sahu65 detect photo.jpg                    # local inference
sahu65 detect photo.jpg --json             # JSON output
sahu65 detect photo.jpg --explain          # plain-language explanation (needs a provider key)
sahu65 serve --port 8000                   # HTTP API
sahu65 info                                # bundled checkpoint, thresholds, provider status
sahu65 --key myapp                         # mint a Bear Token (API key), printed once
sahu65 --list-keys                         # list minted token names (never the tokens)
```

### Self-host the API

```bash
uvicorn sahu65.main:app --host 0.0.0.0 --port 8000 --workers 1
```

**Gate traffic on `/deep-guard/readyz`, not `/deep-guard/healthz`** — `/readyz` is 503
until the weights are resident and reports `model_load_failed` if the checkpoint is missing.

### Bear Tokens (API keys)

`POST /deep-guard/detect` is open by default. The moment **one** key exists — minted with
the CLI or set via `API_KEYS` — every detect call must present it, otherwise the response
is `401 unauthorized` with `{"error": {"code": "unauthorized"}}`:

```bash
sahu65 --key myapp                          # prints the token ONCE; stores its SHA-256 only
curl -X POST http://localhost:8000/deep-guard/detect \
  -H "Authorization: Bearer bear_myapp_..." \
  -F "file=@sample.jpg"
```

- Tokens are stored in `~/.sahu65/keys.json` (override: `SAHU65_KEYFILE`) as **SHA-256
  digests only** — a leaked key file cannot be replayed.
- Re-running `sahu65 --key myapp` **rotates** the token; the previous one stops working.
- The store is re-read whenever the file changes, so a token minted while the server runs
  takes effect immediately.
- `API_KEYS=k1,k2` still works alongside minted tokens.
- SDK: `Client(api_key="bear_...")` or `export SAHU65_API_KEY=bear_...`.
- Auth is checked **before** readiness, so unauthenticated callers learn nothing about
  model state. `/healthz`, `/readyz` and `/version` stay open for probes.

### Docker

```bash
docker build -t ai-image-detector:latest .
docker run -p 8000:8000 --cpus=2 --memory=1g ai-image-detector:latest
```

---

## Explanations

Opt-in plain-language narration of a verdict, generated by any LLM provider. Supported out
of the box: **Google AI Studio (Gemini)** and **Groq**.

```bash
export GEMINI_API_KEY=...        # or GROQ_API_KEY=...
sahu65 detect photo.jpg --explain
```

```python
result = sahu65.detect("photo.jpg", explain_result=True)
print(result.explanation)
```

```bash
curl -X POST http://localhost:8000/deep-guard/detect?explain=true -F "file=@sample.jpg"
```

Three properties are enforced in code, not left to the prompt:

1. **The model narrates, it never decides.** `explain()` receives the finished verdict and
   the numbers behind it, and returns prose. No code path lets generated text alter
   `verdict` or `ai_probability`. If the verdict is `inconclusive`, an explanation claiming
   certainty is replaced with a fixed neutral phrasing.
2. **No image bytes leave the process.** Prompts are built from numbers and metadata only,
   so the promise that images are never written to disk or logged survives, and no user
   photo is handed to a third party.
3. **Fully optional and additive.** With no provider key the field is `null` and the
   detection is byte-for-byte unchanged. Local SDK use stays entirely offline. Both
   providers are called through the stdlib `urllib`, so neither adds a runtime dependency.

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | Google AI Studio |
| `GROQ_API_KEY` | Groq |
| `SAHU65_EXPLAIN_MODEL_GEMINI` / `SAHU65_EXPLAIN_MODEL_GROQ` | override the model name |

`/deep-guard/version` and `sahu65 info` report which provider is configured (name only,
never the key).

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

Accepts multipart form-data with an image file (`JPEG`, `PNG`, `WebP`). Query parameters:
`explain` (bool), `explain_provider` (`gemini` | `groq`).

```bash
curl -X POST http://localhost:8000/deep-guard/detect \
  -F "file=@sample.jpg"
```

**Example Response (200 OK):**

```json
{
  "request_id": "9f1c7d2e4a8b",
  "verdict": "likely_ai",
  "ai_probability": 0.9346,
  "confidence": "medium",
  "calibrated": true,
  "signals": {
    "c2pa": { "present": false, "ai_declared": null },
    "detector": {
      "model": "openai/clip-vit-base-patch16@57c2164",
      "probability": 0.9127
    }
  },
  "warnings": ["jpeg_recompressed_likely"],
  "explanation": null,
  "explanation_provider": null,
  "model_version": "2026.11.0",
  "benchmark_id": "bench-8e0bd9e97e3e",
  "latency_ms": 119.7
}
```

> **Breaking change in 2026.11.0:** `signals.classifier` and `signals.clip_probe` were
> replaced by a single `signals.detector`. Only one model runs now, and the old field names
> described a second one that is no longer shipped.

Errors use a uniform shape `{ "error": { "code": "...", "message": "..." } }`:

| Status | Code | Cause |
|---|---|---|
| 400 | `invalid_image` | Cannot decode |
| 401 | `unauthorized` | Bear Token required but absent/invalid |
| 413 | `file_too_large` | Over `MAX_UPLOAD_MB` |
| 415 | `unsupported_type` | Not JPEG/PNG/WebP |
| 422 | `image_too_large_pixels` | Over `MAX_PIXELS` (decompression bomb guard) |
| 429 | `rate_limited` | Rate limit hit |
| 503 | `model_not_ready` / `model_load_failed` | Loading, or weights missing |

---

## Configuration

All configuration is via environment variables (see `sahu65/config.py`):

| Variable | Default | Description |
|---|---|---|
| `MAX_UPLOAD_MB` | `10` | Hard cap on upload size |
| `MAX_PIXELS` | `50000000` | Decompression bomb guard |
| `MAX_CONCURRENT_INFERENCES` | `8` | Semaphore limit for concurrent inference |
| `ORT_INTRA_THREADS` | `8` | Threads per ONNX session |
| `ENABLE_C2PA` | `true` | Extract Content Credentials (degrades gracefully) |
| `RATE_LIMIT` | `30/minute` | Rate limiter window per IP/key |
| `REQUIRE_REAL_MODEL` | `true` | Refuse to become ready if weights are missing |
| `DOCUMENT_GATE` | `false` | **UNVALIDATED, opt-in.** Suppress confident verdicts to `inconclusive` on flat/text-heavy out-of-distribution input |
| `API_KEYS` | `""` | Comma-separated plaintext keys alongside minted Bear Tokens |
| `SAHU65_KEYFILE` | `~/.sahu65/keys.json` | Bear Token store — SHA-256 digests only |
| `ALLOWED_ORIGINS` | `""` | CORS allowed origins |
| `ENABLE_DOCS` | `false` | Serve `/docs` |
| `GEMINI_API_KEY` / `GROQ_API_KEY` | unset | Explanations (see above) |

> `ENABLE_TTA` is accepted for compatibility but ignored: the probe is served one image per
> inference because the encoder is dynamically quantized and batch-dependent. Batching it
> would silently change scores. See MODEL_CARD.md §3.5.

---

## Reproducing the measurements

Every number above comes from a real benchmark and is re-derivable:

```bash
pip install -r requirements.txt -r requirements-build.txt

# 1. Build the benchmark: 831 labelled images, 13 generator families, licences recorded
python scripts/build_benchmark.py --out benchmark

# 2. Export the CLIP encoder to INT8 ONNX (torch needed here, not at runtime)
python scripts/export_clip_onnx.py

# 3. Fit the probe head and emit out-of-fold scores
python scripts/train_clip_probe.py \
    --encoder models/clip_encoder.dyn.int8.onnx --embed-batch 1 \
    --emit-split-scores benchmark/scores_oof.csv --force

# 4. Fit thresholds under the FPR constraint
python scripts/fit_fusion.py --scores benchmark/scores_oof.csv --live-signal clip

# 5. Report on the held-out split with Wilson intervals
python scripts/evaluate.py --scores benchmark/scores_oof.csv --split test
```

`scripts/compare_detectors.py` reproduces the checkpoint comparison in MODEL_CARD.md §3.2.
`scripts/sweep_clip_quantization.py` reproduces the quantization results in §3.5.

---

## Publishing to PyPI

Publishing is automated by GitHub Actions. **You never handle the token locally.**

Every push to `main` runs the pipeline and publishes: ruff → pytest → checkpoint guard →
build → `verify_dist.py` → `twine check` → **smoke-test the built wheel in a clean venv** →
upload → poll PyPI until live. Pull requests run everything except the upload.

### One-time setup

1. Create a PyPI API token: <https://pypi.org/manage/account/token/>
2. In the GitHub repo: **Settings → Secrets and variables → Actions → New repository
   secret**, name `PYPI_API_TOKEN`, value the token.
3. *(Recommended)* Create a GitHub Environment named `pypi` so uploads require approval.

### Releasing a new version

PyPI releases are immutable: a push that does not change the version uploads nothing
(`skip-existing: true`). To ship a change, bump the version first —
`scripts/bump_version.py` keeps `pyproject.toml` and `sahu65/__init__.py` in sync:

```bash
python scripts/bump_version.py minor --dry-run
python scripts/bump_version.py minor
git add -A && git commit -m "sahu65 1.1.0"
git push                                         # push to main -> build + publish
```

`scripts/verify_dist.py` is the guard that matters most. It fails the run *before* upload if
the encoder is missing from the wheel, if any artifact exceeds PyPI's 100 MiB limit, or if
required files are absent. It catches the exact failure that happened during an earlier
build, where a `MANIFEST.in` exclude silently produced a 37 KB wheel containing no model.

---

## Development & Build Pipeline

```bash
pytest
ruff check sahu65 tests scripts
```

Runtime dependencies are pinned in `requirements.txt` (no torch). Build-time tools live in
`requirements-build.txt` (torch, transformers, onnx). Keeping torch out of the runtime is
what keeps the image and the wheel small.

> `scripts/build_benchmark.py`, `scripts/score_benchmark.py`,
> `scripts/train_clip_probe.py` and `scripts/fit_fusion.py` were rewritten in 2026.11.0.
> Earlier versions of `build_benchmark.py` and `fit_fusion.py` generated their "real" and
> "AI" images and their model predictions from random draws, so every metric they produced
> described a random number generator. Numbers from those versions were retracted in place
> in the previous MODEL_CARD and must not be cited.