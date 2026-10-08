# Model Card: AI-Generated Image Detector

## 1. Model Details

- **Model Name:** AI-Generated Image Detector
- **Version:** `2026.11.0`
- **Release Date:** November 2026
- **Architecture:** A single linear probe on a frozen CLIP ViT-B/16 image encoder.
  - **Encoder:** `openai/clip-vit-base-patch16` @ revision
    `57c216476eefef5ab752ec549e440a49ae4ae5f3`, INT8 dynamic-quantised ONNX
    (`models/clip_encoder.int8.onnx`, 85.5 MB). Licence: the Hub metadata declares none;
    the upstream model card states MIT. Confirm before a commercial release.
  - **Probe head:** 512 weights + bias (`models/clip_head.json`), fitted on real
    benchmark embeddings by `scripts/train_clip_probe.py` with L2 strength chosen by
    cross-validation. Fitted with torch at build time only; the runtime is ONNX Runtime.
  - **Calibration:** Platt scaling in logit space, `sigmoid(1.642 * logit(p) + 0.0877)`,
    fitted on out-of-fold benchmark scores.
  - **Preprocessing:** aspect-preserving — shortest side to 224 px, centre crop,
    CLIP mean/std. Never a squash to a square.
  - **Runtime:** ONNX Runtime, CPUExecutionProvider. No PyTorch, no Transformers at
    runtime. No second model.

### Why this replaced `Organika/sdxl-detector`

The previous release shipped `Organika/sdxl-detector` (Swin-T) with the CLIP probe
disabled, because the probe's head had been fitted on `np.random.randn`. Measured head to
head on benchmark `bench-8e0bd9e97e3e`:

| detector | AUC | false positives @ 50% recall | size | latency | licence |
|---|---|---|---|---|---|
| `Organika/sdxl-detector` (previous) | 0.6434 | 24.3% | 91.8 MB | 179 ms | **cc-by-nc-3.0** |
| **CLIP ViT-B/16 probe (shipped)** | **0.8213** | **6.5%** | **85.5 MB** | **135 ms** | MIT (upstream) |

The swap is smaller, faster and far more accurate, and it removes a non-commercial
restriction from a package published on PyPI. Three alternative checkpoints named in
`architecture.md` 3.1 were also benchmarked and all were worse — see 3.2.

Two properties of the previous model explain the gap. It saturated on the generator
families resembling its training data (dalle3, midjourney-v5 and stable-diffusion-xl all
scored a constant 1.000) and collapsed on the rest; the CLIP probe repairs exactly those
cases (glide 5.9% → 97.1%, dalle2 15.6% → 93.8%, FLUX.1-dev 13.2% → 78.9%).

**The ensemble was dropped.** `architecture.md` 3.1 justifies two signals on the grounds
that "S2 and S3 fail on different images". Measured, they do not: fusing the two was worse
than the probe alone at every mixing weight tried (best fused AUC 0.8173 against 0.8213
for the probe). Shipping the second model would have cost 92 MB and 179 ms for a worse
number, so it is gone.

---

## 2. Intended Use & Limitations

> **CRITICAL NOTICE:** the output is a **calibrated probabilistic signal and heuristic
> evidence, NOT legal or forensic proof.** It must never be the sole basis for accusing
> anyone of fabricating media, or for any punitive, legal, or employment decision.

### Prohibited Uses

- Automated legal, judicial, or law-enforcement decisions without human forensic review.
- Harassment or defamation of individuals based on an algorithmic score.
- Any claim of authenticity — a `likely_real` verdict is not proof that an image is genuine.
- Document, passport, ID, certificate, receipt, or form forgery decisions. See 3.4.

### Known Limitations

1. **Unseen generators degrade it.** Per-family recall ranges from 40% to 100% on this
   benchmark; a generator released after the CLIP encoder's training will likely do worse.
2. **Post-processing and compression reduce scores.** Recall on AI images falls from 59%
   unperturbed to 48% at JPEG quality 50.
3. **Beauty filters, HDR and AI upscaling of real photographs are false-positive risks.**
4. **Adversarial evasion is easy.** Small perturbations and re-encoding move the score.
5. **Recall at the shipped operating point is low by design.** See 3.3.

---

## 3. Evaluation

### 3.1 The benchmark

`bench-8e0bd9e97e3e` — 831 images across 366 parent images, built by
`scripts/build_benchmark.py`. Sources, with licences recorded per row in
`benchmark/SOURCES.md`:

- **AI (194 parents):** Synthbuster+ (`marco-willi/synthbuster-plus`) across **13
  generator families** — dalle2, dalle3, midjourney-v5, firefly, imagen3, glide,
  stable-diffusion-{1-3, 1-4, 2, xl}, SD3-medium, FLUX.1-dev, FLUX.1-schnell.
- **Real (172 parents):** RAISE-1k (via Synthbuster+) and COCO val2017
  (`rafaelpadilla/coco2017`).
- **Perturbations** on 30% of parents: jpeg_q95, jpeg_q75, jpeg_q50, resize_half,
  screenshot. All canonical images stored as PNG so the container format cannot be used
  as a proxy for the label — an earlier comparison in this repo was fooled by exactly
  that shortcut (PNG AI images against JPEG camera photos).

**What the benchmark does not cover:** documents and IDs (§3.4), video, and audio.

### 3.2 Detectors compared

All scored on the same 831 images. `architecture.md` P1 required this comparison and it
had never been run — the previous release's justification came from `np.random.beta()`
draws.

| checkpoint | licence | AUC | FPR @ 50% recall |
|---|---|---|---|
| **`openai/clip-vit-base-patch16` + probe (shipped)** | MIT (upstream) | **0.8213** | **6.5%** |
| `Organika/sdxl-detector` | cc-by-nc-3.0 | 0.6434 | 24.3% |
| `umm-maybe/AI-image-detector` | cc-by-4.0 | 0.5850 | 42.1% |
| `prithivMLmods/Deep-Fake-Detector-Model` | apache-2.0 | 0.4503 | 60.7% |
| `prithivMLmods/Deep-Fake-Detector-v2-Model` | apache-2.0 | 0.4411 | 57.9% |

Both Apache-2.0 checkpoints score **below 0.5 AUC**, i.e. worse than chance. Their label
sets are `{Realism, Deepfake}` and `{Fake, Real}`: they are **face-swap detectors trained
on a different task**, and invert on text-to-image generation. A below-chance AUC is the
signature of a model applied outside its domain, not of a bug in the harness — the label
index was resolved from each checkpoint's own `id2label` before scoring.

### 3.3 Measured behaviour

Headline figures are **out-of-fold** (5-fold cross-validation): every image is scored by
a head that did not train on it. Intervals are Wilson 95%.

| Metric | Target (`architecture.md`) | Measured | |
|---|---|---|---|
| ROC AUC | ≥ 0.88 | **0.8213** | miss |
| False-positive rate on real photos | ≤ 3% | **2.88%** | **met** |
| Recall on AI @ that operating point | — | **30.96%** | see below |
| Expected calibration error | < 0.08 | **0.0440** | **met** (0.0761 before Platt) |
| Abstain rate | ≤ 25% | ~26% | met |

**Recall is low and that is the honest ceiling, not a tuning failure.** The FPR/recall
trade-off is fixed by the ROC curve:

| threshold | FPR on real | recall on AI |
|---|---|---|
| 0.50 | 31.4% | 80.2% |
| 0.58 | 21.5% | 72.2% |
| 0.75 | 8.6% | 53.9% |
| 0.83 | 4.7% | 40.3% |
| **0.879 (shipped)** | **2.9%** | **31.0%** |
| 0.92 | 1.0% | 19.8% |

The shipped point was chosen by `scripts/fit_fusion.py` under the ≤3% FPR constraint
because falsely accusing a real photograph is the failure that does real damage to a
person. **Operators who would rather miss AI images than risk false accusations are
correctly configured already; anyone wanting higher recall should raise `T_hi`
knowingly**, accepting the FPR cost. Thresholds live in `config/thresholds.json`.

**Interval width:** the benchmark is 831 images. Per-family recall intervals are roughly
±10 percentage points and the FPR interval roughly ±1.5 points. These are point estimates
from a small benchmark, not precise measurements.

### 3.4 Documents are an unsupported domain

Document and ID imagery is **excluded from the benchmark and unsupported**. The previous
release measured AUC 0.375 — worse than chance — on AI documents versus real documents,
and six separate mitigation attempts all failed. Document-image detection is an unsolved
research problem as of 2026 (TextFake, AIGDoc, AIForge-Doc; best published result on
AI-forged documents is 0.751 AUC). **Do not use this service for document or ID decisions.**

### 3.5 A measurement bug worth recording

The first version of the probe head was fitted on embeddings extracted in batches of 16,
and scored AUC 0.87 offline. It scored **0.72 as actually served**.

The encoder is INT8 *dynamically* quantized, which computes each activation's clipping
range at run time from the observed tensor. An image's embedding therefore depends on
which images share its batch: the same image embedded alone versus inside a batch of 16
different images gives cosine 0.65–0.89. A benchmark scored in batches describes a model
that never serves an image that way, because every HTTP request is a batch of one.

Consequences, both now enforced:

- The service runs **exactly one image per `session.run()`**, and the head is fitted with
  `--embed-batch 1`. `tests/test_clip_probe.py` documents the constraint.
- Any future encoder rebuild must be re-checked for batch invariance before any offline
  measurement is believed. Static QDQ quantization was measured as batch-invariant but
  lost the signal entirely (AUC 0.557–0.592 across five calibration configurations), so
  it was rejected; the shipped encoder is dynamic INT8 precisely because this service
  never batches.

The fp32 encoder scores AUC 0.9807 and is batch-invariant, at 329 MB — over PyPI's 100 MB
per-file limit. It is therefore not shipped. Distributing it would mean a GitHub Release
asset and a download step, giving up the zero-download install.

---

## 4. Hardware & Resource Profile

- **Package:** 74.8 MB wheel (85.5 MB encoder).
- **RAM:** ~500 MB steady state, one ONNX Runtime session.
- **Latency (1 CPU thread, 224×224):** ~135 ms per image. Measured single-threaded;
  `ORT_INTRA_THREADS` defaults to 8.
- **Cold start:** ~10 s (ONNX weight deserialisation). Configure orchestrator readiness
  probes with `start_period` ≥ 30 s, or call `sahu65.warm()` ahead of time.
- **Runtime dependencies:** `onnxruntime`, `numpy`, `pillow`, `fastapi`, `uvicorn`,
  `pydantic`, `httpx`. torch and transformers are **build-time only**.