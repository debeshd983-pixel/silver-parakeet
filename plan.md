# Execution Plan: AI-Generated Image Detection Backend

**Hard ceiling:** 5 hours.
**Companion doc:** `architecture.md` (the what and why). This file is the order of work.
**Rule:** every phase ends with a gate. If a gate fails, apply the listed fallback. Do not push on hoping.

---

## Timeline at a glance

| Phase | Window | Output | Gate |
|---|---|---|---|
| P0 | 0:00-0:15 | Repo skeleton, env, decisions log | `pytest` runs (empty) |
| P1 | 0:15-0:55 | Benchmark + candidate model shortlist | Benchmark manifest built; >= 2 detector candidates load |
| P2 | 0:55-1:35 | Ensemble, fusion weights, thresholds | Targets measured on calibration split |
| P3 | 1:35-2:45 | Working FastAPI service | End-to-end `/v1/detect` works locally |
| P4 | 2:45-3:25 | ONNX export + int8 quantization | Quantized accuracy within tolerance |
| P5 | 3:25-4:10 | Docker image + deployed URL | Smoke test passes against live URL |
| P6 | 4:10-5:00 | Tests, MODEL_CARD, README, buffer | Definition of Done met |

Time checkpoints are enforced: **if P3 is not finished at 2:45, trigger the cut-lines in section 9.**

---

## P0. Setup (15 min)

- [ ] Create repo using the layout in `architecture.md` section 5.
- [ ] Python 3.11 venv. Split `requirements.txt` (runtime) and `requirements-build.txt` (torch, transformers, optimum, onnx, datasets).
- [ ] Create `DECISIONS.md`. Log every model/dataset choice with: name, revision hash, license, date, reason.
- [ ] `.gitignore` for `models/`, `benchmark/images/`, `.venv/`.
- [ ] Check network access to Hugging Face and PyPI. **If Hugging Face is blocked, stop and ask the user for a mirror or pre-downloaded weights** rather than substituting silently.

**Done when:** `pytest` executes and `uvicorn app.main:app` starts with a stub `/healthz`.

---

## P1. Benchmark and model shortlist (40 min)

### Benchmark (do this first, it drives every later decision)

- [ ] Write `scripts/build_benchmark.py` producing `benchmark/manifest.csv` with columns: `path, label, source, generator, split, parent_id, perturbation, sha256`.
- [ ] **Real images (target >= 250):** COCO val2017 subset or Open Images; vary sizes.
- [ ] **AI images (target >= 250):** at least **4 generator families** from public datasets (e.g. GenImage, Synthbuster, Hugging Face AI-vs-real sets). Record the license of each. Do not rely on CIFAKE alone (32x32).
- [ ] **Perturbations** on a stratified ~40% subset: JPEG q95/q75/q50, 0.5x downscale, downscale + JPEG, center crop. Set `parent_id` for each derivative.
- [ ] **Split by `parent_id`**: 50% calibration, 50% test. Perturbed copies stay with their parent.
- [ ] Freeze: compute `benchmark_id = sha256(sorted(file hashes))[:12]`.

### Model shortlist

- [ ] Load 3 candidate classifiers via `transformers` and score the **calibration split only**. Candidates: `Organika/sdxl-detector`, `umm-maybe/AI-image-detector`, `prithivMLmods/Deep-Fake-Detector-v2-Model`. Verify the IDs resolve. If any do not, search the Hub for replacements and log it.
- [ ] Record per candidate: AUC, accuracy at 0.5, FPR on real, parameter count, license, fp32 latency on CPU.
- [ ] Build the CLIP probe features: extract frozen CLIP ViT-B/16 embeddings (via `open_clip` or `transformers`) for the calibration split.
- [ ] Choose the **best classifier by AUC and FPR, with license permitting redistribution/commercial use as needed.** Reject any with non-commercial licenses if the product is commercial; ask the user if unsure.

**Gate:** benchmark built, >= 2 classifier candidates scored. 
**Fallback:** if only ~300 images are obtainable, proceed, but widen all reported intervals and say so in `MODEL_CARD.md`.

---

## P2. Ensemble, fusion, thresholds (40 min)

- [ ] Train the CLIP probe head (logistic regression) on **calibration-split embeddings only**. Never touch the test split.
- [ ] If the probe underperforms (AUC < ~0.70 on calibration under cross-validation), try: concatenating features from a second layer, or adding ~1-3k extra images from recent generators for the head only. Time-box to 15 min.
- [ ] `scripts/fit_fusion.py`: fit `z = w1*logit(p_cls) + w2*logit(p_clip) + b` by logistic regression with cross-validation on the calibration split. Write `config/fusion.json`.
- [ ] Choose `T_lo`, `T_hi` on the calibration split so that FPR on real (committed as AI) <= 3%. Write `config/thresholds.json`.
- [ ] Run `scripts/evaluate.py` **once** on the test split. Record all metrics (architecture section 9). Do not iterate on the test split.
- [ ] Implement `ensemble.py` (pure functions: fuse, verdict, confidence) with unit tests using fixed inputs.

**Gate:** ensemble AUC >= best single model AUC on calibration CV. 
**Fallback:** if the ensemble is not better, ship the better single model with the abstain band and note it in the model card. Do not ship a worse ensemble for the sake of the design.

---

## P3. FastAPI service (70 min)

Build in this order, testing each step.

1. [ ] `config.py` (pydantic-settings) with all variables from architecture section 8.
2. [ ] `core/limits.py`: magic-byte sniffing, size cap (streaming read with a hard limit), pixel cap, rate limiter, optional API key.
3. [ ] `services/preprocess.py`: decode, EXIF orient, RGB, per-model transforms. Unit test against the preprocessing used during P1/P2 so inference matches evaluation. **Mismatch here silently ruins accuracy.**
4. [ ] `services/provenance.py`: C2PA read, behind `ENABLE_C2PA`. Test with a known C2PA sample if obtainable; otherwise test the "no manifest" path and mock the manifest path.
5. [ ] `services/classifier.py` and `clip_probe.py`: initially run via PyTorch/transformers to get end-to-end correctness. ONNX swap happens in P4 behind the same interface.
6. [ ] `services/runtime.py`: threadpool wrapper + semaphore.
7. [ ] `api/routes.py`: `/v1/detect`, `/healthz`, `/readyz`, `/version`. Uniform error shape.
8. [ ] `main.py` lifespan: load models, run warm-up, set ready flag.
9. [ ] Smoke script: `curl -F file=@sample.jpg localhost:8000/v1/detect`.

**Gate:** `/v1/detect` returns the full response shape; results on 20 benchmark images match offline `evaluate.py` outputs within 1e-3. 
**Fallback:** if C2PA install fails on the platform, set `ENABLE_C2PA=false` by default and move on (see cut-lines).

---

## P4. ONNX export and quantization (40 min)

- [ ] `scripts/export_onnx.py`: export classifier and CLIP image encoder (e.g. via `optimum` or `torch.onnx.export`, opset >= 17, dynamic batch axis).
- [ ] Parity check: ONNX fp32 vs PyTorch outputs on 50 images, max abs diff < 1e-3 on probabilities.
- [ ] `scripts/quantize.py`: `onnxruntime.quantization.quantize_dynamic` (QInt8) on MatMul-heavy layers.
- [ ] Re-run evaluation on the **calibration split** with quantized models. **Accept quantization only if** accuracy drops <= 1 percentage point and AUC drops < 0.01.
- [ ] If quantization is accepted, **refit fusion and thresholds** on quantized outputs (scores shift slightly).
- [ ] Swap the service's classifier and probe to ONNX Runtime. Remove torch from runtime imports. Verify `pip list` in the runtime venv contains no torch.
- [ ] Benchmark latency (p50/p95) on 50 requests, 2 threads.

**Gate:** latency and memory targets in architecture section 6, or documented deviations. 
**Fallback:** if int8 hurts accuracy, ship fp32 ONNX and accept a larger image. If both are too slow, drop TTA and reduce input size only if the model card allows.

---

## P5. Docker and deploy (45 min)

- [ ] Multi-stage `Dockerfile` per architecture section 10. Models are fetched by pinned revision in the builder stage, with the SHA-256 manifest written to the image.
- [ ] Runtime stage: slim base, non-root user, `HEALTHCHECK`, `uvicorn --workers 1`.
- [ ] Build and check `docker images` size < 1.5 GB. If over, inspect layers (`docker history`) and remove build caches and unused libraries.
- [ ] Run the container locally with a memory limit (`--memory=1g --cpus=2`) and run the smoke test and a 50-request load check.
- [ ] Deploy to **Cloud Run** (default) or Fly.io/Render/HF Spaces. Set env vars, min instances, memory >= 1 GiB, concurrency to match the semaphore.
- [ ] Run the smoke test against the live URL. Record the URL and cold-start time.

**Gate:** live URL returns correct responses for 3 known images (one real, one AI, one corrupted file -> 400). 
**Fallback:** if a platform deployment fights back for more than 20 min, switch to the next platform in the list. Do not debug platform quirks indefinitely.

---

## P6. Tests, docs, buffer (50 min)

### Tests

- [ ] Unit: fusion math, verdict boundaries, confidence logic, limits (oversize, bad magic bytes, pixel bomb), error shapes.
- [ ] Integration: TestClient on `/v1/detect` with fixtures (real JPEG, AI PNG, truncated file, wrong type, 11 MB file).
- [ ] Regression: 30 fixed benchmark images with expected probabilities (tolerance 0.02), so future model changes are visible.
- [ ] CI workflow: lint (ruff), tests, docker build, smoke test.

### Docs

- [ ] `MODEL_CARD.md`: models + revisions + licenses, benchmark description, **all measured metrics with intervals**, per-generator and per-perturbation tables, calibration plot, limitations (architecture section 13), intended and prohibited uses.
- [ ] `README.md`: quickstart (`docker run`, `curl`), API reference, config table, how to re-run the benchmark, how to add a new detector, how to retrain the probe head.
- [ ] `DECISIONS.md` finalized.

### Buffer

Keep the last 15-20 minutes unscheduled for the thing that broke.

---

## 8. Acceptance criteria (Definition of Done)

All must be true, or the deviation is written up in `MODEL_CARD.md`.

**Functional**
- [ ] `POST /v1/detect` returns the documented schema; all error codes in architecture section 4 are reachable and tested.
- [ ] `/readyz` is false until warm-up completes.

**Accuracy (measured on the held-out test split, never tuned on)**
- [ ] FPR on real photos <= 3% among committed verdicts, reported with a 95% interval.
- [ ] Accuracy on committed verdicts >= 85%.
- [ ] Abstain rate <= 25%.
- [ ] Per-generator and per-perturbation tables published, including the worst-case rows.

**Light**
- [ ] Image < 1.5 GB; RAM < 1 GB under 4 concurrent requests; no torch in the runtime image.
- [ ] p95 latency < 1.5 s on 2 vCPU for a ~12 MP JPEG.

**Deployable**
- [ ] Live URL answering; smoke test passes; container runs as non-root.
- [ ] One-command local run documented.

**Honest**
- [ ] README and model card state the limitations and that the output is a probability, not proof.

If the accuracy targets are missed, **do not tune on the test split to hit them.** Report the real numbers, state what blocks improvement (usually: unseen generators), and list the next steps.

---

## 9. Cut-lines (if running behind)

Apply in this order. Each is a conscious reduction, logged in `DECISIONS.md`.

| Trigger | Cut |
|---|---|
| At 2:45 P3 unfinished | Drop TTA, `/metrics`, optional API-key auth (keep rate limit) |
| At 3:00 | Drop C2PA (set `ENABLE_C2PA=false`, document) |
| At 3:25 quantization unresolved | Ship fp32 ONNX |
| At 3:40 | Drop the CLIP probe and ship the single best classifier with the abstain band |
| At 4:10 deploy unresolved | Ship the verified local Docker image + deploy instructions, switch platforms |
| At 4:30 | Skip regression fixtures; keep unit + integration tests |

**Never cut:** the benchmark, the held-out test split, input validation, the `inconclusive` band, the model card limitations.

---

## 10. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Candidate HF model IDs removed or renamed | Medium | Medium | Verify in P1; pick alternatives from the Hub; log in DECISIONS.md |
| License forbids commercial use | Medium | High | Check in P1 before building on it; ask the user |
| Detectors near chance on recent generators | High | High | Ensemble + abstain band + honest report; plan head-only retraining as follow-up |
| Preprocessing mismatch | Medium | High | Parity tests in P3 against the evaluation pipeline |
| int8 hurts accuracy | Medium | Medium | Accept/reject rule in P4; ship fp32 |
| `c2pa-python` wheel unavailable on target platform | Medium | Low | Feature flag, cut-line |
| Benchmark too small to be conclusive | High | Medium | Report intervals; recommend growing to 1,000+ |
| Memory blowup under load | Low | High | Semaphore, 1 worker, load check in P5 |
| Network restrictions in the build environment | Medium | Medium | Detect in P0; ask the user for mirrors/pre-downloaded weights |

---

## 11. Post-launch roadmap (not in the 5 hours)

1. Grow the benchmark to 2,000+ images, refreshed each quarter with new generators.
2. Scheduled head-only retraining of the CLIP probe as new generators ship.
3. Batch endpoint and async job mode.
4. Optional third signal (frequency-domain or noise-residual detector) if the benchmark shows a gap.
5. Human-review queue for `inconclusive` results.
