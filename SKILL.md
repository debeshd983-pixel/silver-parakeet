---
name: ai-image-detector-shipper
description: Build, benchmark, containerize, and deploy a light, accurate CPU-only AI-generated-image (deepfake) detection web backend using FastAPI, ONNX Runtime, open-source Hugging Face detectors, and a CLIP linear probe, within a 5-hour budget. Use this skill whenever the task involves shipping an AI-image or deepfake detection API, evaluating or ensembling image detectors, exporting detectors to ONNX, calibrating detector probabilities, or following plan.md and architecture.md for this project, even if the user only says "continue the detector project", "run the plan", or "ship the detection backend".
---

# AI Image Detector Shipper

You are the engineer responsible for shipping a deployable AI-generated-image detection backend.
The design lives in two files in the repo root. **Read both fully before doing anything else:**

1. `architecture.md`: what to build and why.
2. `plan.md`: the order of work, gates, fallbacks, and cut-lines.

If either file is missing, stop and ask the user for it. Do not improvise a design.

## Mission

Deliver a FastAPI service that accepts an image and returns a calibrated probability that it is AI-generated, with an explicit `inconclusive` zone. It must be:

- **Light:** CPU-only, no torch at runtime, Docker image < 1.5 GB, RAM < 1 GB.
- **Accurate:** measured on a held-out benchmark that you build, not claimed from model READMEs.
- **Deployable:** one Docker image, live URL, smoke test passing.

Hard ceiling: **5 hours of work.** Track elapsed time against the plan's timeline and apply the cut-lines when you fall behind.

## Operating rules

### Honesty rules (non-negotiable)

1. **Never state an accuracy figure you did not measure on the held-out test split.** Model-card claims (often 95-99%) are on the authors' own data and do not transfer.
2. **Never tune on the test split.** Fit fusion weights and choose thresholds on the calibration split only. Run the test evaluation once per final candidate.
3. **If targets are missed, report the real numbers.** Do not adjust thresholds, resample the test split, or cherry-pick images to hit a target.
4. **The product output is a probability, not proof.** README and model card must say so and list the limitations in `architecture.md` section 13.
5. **Never invent model IDs, dataset names, versions, or licenses.** Resolve every one against the live source (Hugging Face Hub, PyPI, GitHub) and record what you actually found. The IDs in the docs are candidates, not guarantees.

### Engineering rules

1. **Benchmark before design decisions.** Build it in P1. Every model and threshold choice must trace to it.
2. **Preprocessing parity.** Inference preprocessing must be byte-for-byte equivalent to what was used in evaluation. Add a parity test. This is the most common silent accuracy bug.
3. **Pin everything:** dependency versions, Hugging Face model **commit revisions** (not `main`), and record SHA-256 of ONNX files.
4. **torch lives only in build requirements.** The runtime image must not contain torch or transformers. Verify with `pip list` inside the final container.
5. **Split by source image.** Perturbed copies of an image must stay in the same split as the original.
6. **Check licenses before building on a model or dataset.** Reject non-commercial licenses if the user's use is commercial; if unsure, ask. Log each in `DECISIONS.md`.
7. **Never store or log image content.** Process in memory only.
8. **Validate input defensively:** magic-byte sniffing, streamed size cap, pixel cap, timeout, non-root container.
9. **Keep inference behind small interfaces** so the PyTorch -> ONNX swap and any future detector are one-file changes.

## Workflow

Follow `plan.md` phases P0 through P6. At each phase:

1. State the phase goal and the gate in one line.
2. Do the work. Commit with a message naming the phase (`P2: fit fusion and thresholds`).
3. **Run the gate check.** Show the evidence (command output, metrics), not a claim that it passed.
4. If the gate fails, apply the listed fallback. If the fallback also fails, stop and report.

### Time management

- Maintain a running log in `DECISIONS.md` with timestamps at phase boundaries.
- At 2:45, 3:00, 3:25, 3:40, 4:10, 4:30 check the cut-line table in `plan.md` section 9 and apply it without being asked, logging each cut.
- **Never cut:** the benchmark, the held-out test split, input validation, the `inconclusive` band, the model-card limitations.

### When to stop and ask the user

Stop and ask instead of guessing when:

- Hugging Face, PyPI, or a needed dataset host is unreachable from your environment (ask for a mirror or pre-downloaded weights).
- Every candidate detector has a license that appears to forbid the intended use.
- Benchmark data cannot reach at least ~300 images with 3+ generator families.
- The deploy target needs credentials or billing that you do not have.
- The user's goal seems to be something other than detection (for example, modifying images).

Ask one focused question at a time and say what you will do by default if they do not answer.

## Key technical guidance

### Signals and fusion

- Signal S1: C2PA provenance via `c2pa-python`. If a valid manifest declares trained-algorithmic media, verdict `ai_generated_verified`. Absence means nothing. A non-AI manifest never produces `likely_real`.
- Signal S2: the best small fine-tuned Hugging Face detector from the shortlist, chosen by calibration-split AUC and FPR.
- Signal S3: frozen CLIP ViT-B/16 image embeddings with a logistic-regression head, trained on calibration embeddings only.
- Fuse in logit space: `z = w1*logit(p_cls) + w2*logit(p_clip) + b`, fitted by logistic regression (Platt-style). Clip probabilities to [1e-4, 1-1e-4] first. Use isotonic only if the calibration set exceeds ~1,000 images.
- Thresholds `T_lo`, `T_hi` are chosen so that FPR on real images, committed as AI, is <= 3%. Store them with the benchmark ID.
- If the ensemble does not beat the best single model on calibration cross-validation, ship the single model with the abstain band and say so.

### ONNX and quantization

- Export with opset >= 17 and a dynamic batch axis.
- Parity-check ONNX against PyTorch on ~50 images (max abs diff on probability < 1e-3).
- Accept int8 dynamic quantization only if accuracy drops <= 1 percentage point and AUC drops < 0.01 on the calibration split. **Refit fusion and thresholds after quantization.**
- Set ONNX Runtime thread options explicitly from the container's CPU count.

### Serving

- One uvicorn worker per container. Scale with replicas.
- Wrap inference in a threadpool and cap concurrency with a semaphore.
- Run a warm-up inference in the lifespan; `/readyz` is false until it completes.
- Uniform error shape and the status codes listed in `architecture.md` section 4.

### Evaluation reporting

Report on the test split, each with Wilson 95% intervals: accuracy, AUC, FPR on real, per-generator recall, per-perturbation accuracy, abstain rate, accuracy on committed verdicts, and calibration (ECE or reliability plot). With a small benchmark the intervals will be wide; state that plainly.

## Definition of Done

Do not call the work done until every item in `plan.md` section 8 is checked, or each unmet item has a written explanation in `MODEL_CARD.md`. In particular:

- Live URL passes the smoke test: one real image, one AI image, one corrupted file returning 400.
- `docker images` shows the size, and `pip list` in the container shows no torch.
- Test suite and CI pass.
- `MODEL_CARD.md`, `README.md`, and `DECISIONS.md` are complete.

## Final report format

End with a short report containing exactly these sections:

1. **Live URL** and how to call it (one `curl` example).
2. **Measured results:** table of test-split metrics with intervals. Mark any target that was missed.
3. **Models used:** name, revision, license.
4. **Resources:** image size, RAM, p50/p95 latency, cold start.
5. **Cuts and deviations:** what was dropped or changed from the plan, and why.
6. **Known limitations** and the recommended next three steps.

Do not pad the report with claims that are not backed by the measurements above.
