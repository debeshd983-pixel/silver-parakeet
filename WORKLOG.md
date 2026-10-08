# WORKLOG — what changed in this repo and why

**Date:** 2026-10-08
**Scope:** investigation of reported false positives (`report.txt`), correctness fixes,
repackaging as `sahu65`, and release automation.

Full technical detail lives in `DECISIONS.md` (chronological, P0–P8) and `MODEL_CARD.md`
(measurements and limitations). This file is the short version, including the things that
**were not** fixed.

---

## TL;DR

The reported symptom was real, but `report.txt`'s diagnosis was wrong on both counts.

* There **was** a genuine, severe bug — the AI class index was inverted — but it was not
  what produced the numbers in `report.txt`.
* The actual false positives on documents are **not fixable with this model.** Measured
  discrimination on document images is **AUC 0.375 — worse than chance.** Six independent
  approaches were tried and all failed. This is an unsolved research problem as of 2026,
  not a defect in this repository.
* Separately, the repository was **shipping with no model at all** and would silently
  return a constant `0.50`. Every number in the previous `MODEL_CARD.md` was generated
  from `np.random.beta()` draws, not from the detector.

Current state: **25 tests pass, ruff clean, `twine check` PASSED**, artifacts ready.

---

## Defects found and fixed

| # | Defect | Impact | Fix |
|---|---|---|---|
| 1 | `predict()` read `probs[:, 1]` as the AI probability, but the checkpoint's `id2label` is `{0: "artificial", 1: "human"}` | **Every score inverted.** Would flip all verdicts | AI class index resolved from the checkpoint's own config at load time; 5 regression tests in `tests/test_classifier.py` |
| 2 | Pinned revision `657a8bf…` **404s** on Hugging Face | `fetch_models.py` could never run | Corrected to real head `b37fede8…` |
| 3 | No model weights existed (`models/` gitignored, Dockerfile fetch steps commented out) | Service silently returned constant `0.50`, indistinguishable from a working model | Weights committed; missing checkpoint is now a **hard startup failure** (`/readyz` → 503 `model_load_failed`) |
| 4 | License recorded as Apache-2.0 | Wrong; implied commercial use was permitted | Actual license is **cc-by-nc-3.0 (non-commercial)**. Corrected in `LICENSE`, `README.md`, `MODEL_CARD.md`, `DECISIONS.md` |
| 5 | CLIP probe head fitted on `np.random.randn` | Contributed pure noise while doubling payload and latency | Disabled (`w2 = 0.0`) |
| 6 | `RATE_LIMIT` env var ignored (hardcoded `30/minute`) | Config silently did nothing | Applied from settings at startup |
| 7 | `Image.MAX_IMAGE_PIXELS` mutated per request | One request's limit leaked into later requests and tests (caused 3 test failures) | Enforced without mutating the Pillow global |
| 8 | `jpeg_recompressed_likely` never affected scoring | `report.txt` blamed the false positives on it | No code change — documented that it is cosmetic. Only `low_resolution` is read by the confidence policy |
| 9 | `MODEL_CARD.md` metrics (AUC 0.9042, FPR 2.6%, ECE 0.052) came from `np.random.beta()` | Numbers described random data | Retracted in place, with the real measurements substituted |
| 10 | `c2pa-python` imported but absent from all requirements | Provenance silently always off | Left as a documented optional dependency |

## Defects found *during packaging* (caught only by installing the artifact)

| # | Defect | Impact | Fix |
|---|---|---|---|
| 11 | `MANIFEST.in` `exclude` leaked into the **wheel** build | Produced a **37 KB wheel with no model** — installable, unrunnable | Model shipped in both artifacts; the trap is documented in `MANIFEST.in` and `pyproject.toml` |
| 12 | `httpx` was an optional extra | `sahu65.Client` — the headline feature — crashed on a default install | Promoted to a core dependency |
| 13 | Unbounded dependency ranges | `pip` resolved numpy 2.4.6 / onnxruntime 1.30.0, which **failed to import** on the target machine | Upper bounds anchored to the validated versions |
| 14 | `Client.health()` raised on connection-refused | Fatal inside a polling loop | Returns `False` |

---

## What was NOT fixed — read this

**Real documents are still misclassified.** This is a genuine capability limit, not a bug.

```
AI-generated documents   4/4 correct
Real documents           0/2 correct   (scored 0.995 / 0.999 as AI)
Natural photographs     10/12
Discrimination on document images:  AUC 0.375
```

Both real documents outrank two of the four AI documents. Six approaches were tested and
all failed:

| Approach | Outcome |
|---|---|
| 4 preprocessing pipelines (incl. aspect-preserving + TTA) | Real docs stay 0.996–0.999. The model card's squash is **correct**; changing it *broke* a true positive (0.9959 → 0.0226) |
| 9-tile inference, 5 aggregation statistics | Documents 0.21–0.999 vs natural 0.000–0.997 — full overlap |
| NPR (neighbouring-pixel relationships, training-free) | No separation |
| Colour / saturation / white statistics | No separation |
| Second detector (SigLIP, Apache-2.0) | AUC 0.479 vs natural photos. Its apparent 0.750 on the 6 documents is small-sample noise |
| Classical forensics | Appeared to separate 2.2×, **but only because the AI assets are PNG and the real assets are WhatsApp JPEG.** A file-format shortcut, not authenticity |

Every comparison was re-run with all images re-encoded to identical JPEG quality to
eliminate that shortcut.

Published evidence that this is open research, not a local defect:
[TextFake](https://arxiv.org/abs/2606.01050) (no evaluated method exceeds 80%),
[AIGDoc](https://arxiv.org/abs/2609.14352), AIForge-Doc (best result 0.751 AUC),
ICCV 2025 DeepID Challenge.

### The one thing that does work — and its cost

`DOCUMENT_GATE=true` (opt-in, **default off**) suppresses confident verdicts to
`inconclusive` on flat, text-heavy, out-of-distribution input:

| Group | Gate OFF | Gate ON |
|---|---|---|
| Real documents | **0/2** | **2/2** |
| AI documents | 4/4 | 3/4 |
| Natural photographs | 10/12 | **12/12** |

**Its thresholds were fitted by inspecting those same 18 images.** They are not validated
on held-out data and should not be trusted on unseen documents. It is off by default for
that reason. See `sahu65/services/domain.py`.

### Also unaddressed

The returned probability is **uncalibrated** — `T_lo`/`T_hi` are unfitted placeholders.
Every response carries `thresholds_unfitted`, and `/version` reports `calibrated: false`.
The build scripts that were supposed to fit them (`fit_fusion.py`, `evaluate.py`) operate
on synthetic data and must not be used for this.

---

## Packaging

* `app/` → **`sahu65/`**, so the import name matches the distribution
* INT8-quantized checkpoint: 337.3 MB → **91.8 MB** (verified score-identical to fp32)
* Weights and configs resolve **from inside the package** — works from any directory, no
  download step
* Three entry points: `sahu65.detect()` (local, no key), `sahu65.Client` (hosted, API
  key), `sahu65` CLI (`detect` / `serve` / `info`)
* Verified from a **clean venv install**, resolved from site-packages, reproducing the
  same results as the server

| Artifact | Size |
|---|---|
| `sahu65-1.0.0-py3-none-any.whl` | 66.60 MB |
| `sahu65-1.0.0.tar.gz` | 66.57 MB |

Both under PyPI's 100 MiB limit. `python -m twine check dist/*` → **PASSED**.

### Measured performance

| | |
|---|---|
| TCP socket listening | ~1.5 s |
| First HTTP response | **~10.3 s** |
| Steady-state inference | ~200 ms |
| Cold `sahu65.warm()` | ~10 s |

The ~10 s is ONNX Runtime session initialisation. It holds the GIL, so although the socket
binds early the event loop cannot answer until weights are resident — configure your
readiness probe with `start_period >= 30s`. An earlier claim that `/healthz` answered in
0.02 s during loading was **wrong**: it was measured with an in-process test client, which
does not reflect a real server.

---

## Verification

```bash
python -m pytest -q                      # 25 passed
python -m ruff check sahu65 tests scripts # clean
python -m build
python scripts/verify_dist.py            # 19 checks, guards the 100 MiB limit
python -m twine check dist/*
```

`scripts/verify_dist.py` exists because defect #11 shipped a model-less wheel silently. It
fails before upload if the checkpoint is missing, or if any artifact exceeds PyPI's limit.

---

## Release

Automated via GitHub Actions — **the token is never handled locally**. One pipeline,
`.github/workflows/ci.yml`: every push to `main` runs lint → tests → checkpoint guard →
build → `verify_dist.py` → `twine check` → **smoke test of the built wheel**, then a
`publish` job downloads those exact verified artifacts, uploads them to PyPI, and polls
until the version is live. PRs run every step except the upload.

The previous `publish.yml` (release / manual-dispatch triggered) was **deleted**: it
had **zero runs** because nothing ever published a GitHub Release or dispatched it —
setting the secret alone does not trigger a workflow. Pushing to `main` is now the
trigger.

1. Create a PyPI API token: <https://pypi.org/manage/account/token/> — account-wide for
   the very first upload (the project does not exist yet), then scoped to `sahu65`
2. GitHub → **Settings → Secrets and variables → Actions → New repository secret** →
   `PYPI_API_TOKEN`
3. Recommended: a GitHub **Environment** named `pypi` with required reviewers, so each
   upload can require approval

```bash
python scripts/bump_version.py patch   # PyPI releases are immutable - bump to ship
git add -A && git commit -m "sahu65 1.0.1" && git push
```

Pushing without bumping is safe: `skip-existing: true` makes the upload a no-op, so
every push rebuilds and verifies but only a new version actually reaches PyPI.

---

## Bear Token API-key auth (v1.0.12, 2026-10-08)

`sahu65 --key <name>` (or the sugar form `sahu65 --key-<name>`) mints a named token
(`bear_<name>_<random>`) and prints it once. Only its **SHA-256 digest** is written to
`~/.sahu65/keys.json` (override: `SAHU65_KEYFILE`; 0600 file / 0700 dir best effort on
POSIX). `sahu65 --list-keys` prints names and dates only. Re-minting a name rotates the
token — the previous one stops working.

- `POST /deep-guard/detect` accepts `Authorization: Bearer <token>` **and** the original
  `X-API-Key`. Auth is enforced once any key exists (file-backed or `API_KEYS`) and runs
  **before** the readiness check, so a 401 no longer leaks model state. With zero keys
  the endpoint stays open, exactly as before.
- The store is re-read on mtime change, so minting a token while the server runs takes
  effect without a restart.
- **Hardened:** when no token verifies, the rate limiter now buckets by client IP only.
  Previously an unauthenticated caller could send any `X-API-Key` value and get a fresh
  rate-limit bucket per request.
- `Client` now sends both headers (Bearer preferred, `X-API-Key` kept so older servers
  still authenticate).
- The legacy `app/` tree was deliberately **not** modified: nothing ships it (Dockerfile
  and pyproject package `sahu65/` only; `app/` is referenced solely by `scripts/_*.py`
  diagnostics), and diverging it further from `sahu65/` would add confusion, not value.
- Tests: `tests/test_keys.py` — 14 cases covering mint/rotate, digest-only storage,
  CLI sugar and listing, and the full 401/accept matrix on `/deep-guard/detect`.

---

## ⚠ Open items before this repo goes public

1. **`assets/rl/` contains an Aadhaar card and a stamped land record.** Sensitive personal
   documents. Redact or remove before pushing to a public remote.
2. **`assets/natural/` licenses are unverified.** The 12 Wikimedia Commons fixtures were
   renamed to `natXX.jpg`, stripping required attribution, and each file's license was never
   checked. Either record the licenses in `assets/natural/ATTRIBUTION.md` or delete the
   folder — the measurements in `MODEL_CARD.md` do not depend on them, and the one test
   using them now skips when they are absent.
3. **Bundled weights are CC-BY-NC-3.0 — non-commercial only.** Code is MIT. Fine for the
   stated use; blocks any commercial release.
4. **The checkpoint is 8 MiB from GitHub's 100 MiB hard limit** (96,234,407 bytes). CI
   checks this. If a re-quantization crosses it, move the file to a Release asset or enable
   Git LFS — do not delete it, since `require_real_model=true` makes a missing checkpoint
   a hard startup failure.
5. **This directory is not usable as a git repository yet.** A `.git/` directory exists but
   is incomplete — `git` reports *"fatal: not a git repository"*. Remove it and run
   `git init`, then add your GitHub remote before the workflows can run.

   Also note: **git is installed at `C:\Program Files\Git\cmd\git.exe` but is not on
   `PATH`.** Plain `git` commands in this shell fail with *"git is not recognized"*. Add
   `C:\Program Files\Git\cmd` to `PATH` first, or call the executable by full path.

   First commit will include the 91.78 MiB checkpoint, so expect it to be slow.