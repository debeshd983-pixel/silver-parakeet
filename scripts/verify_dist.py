"""Pre-publish verification of built distribution artifacts.

Run by CI before uploading to PyPI so that a bad artifact fails fast and locally
rather than as an opaque rejection from PyPI.

Checks:
  1. both the wheel and the sdist were produced
  2. the wheel actually contains the INT8 CLIP encoder (a 37 KB model-less wheel was
     produced once by a MANIFEST.in exclude leaking into the wheel build)
  3. bundled JSON configs are present in the wheel
  4. every artifact is under PyPI's 100 MiB per-file upload limit
  5. the wheel declares the runtime dependencies
"""
import sys
import tarfile
import zipfile
from pathlib import Path

PYPI_MAX_BYTES = 100 * 1024 * 1024  # PyPI rejects files larger than 100 MiB
CHECKPOINT = "sahu65/models/clip_encoder.int8.onnx"
CONFIGS = (
    "sahu65/config/thresholds.json",
    "sahu65/config/fusion.json",
    # The fitted probe head. Without it the package installs and then refuses to become
    # ready, so it is checked exactly as strictly as the encoder.
    "sahu65/models/clip_head.json",
)
MIN_CHECKPOINT_BYTES = 50 * 1024 * 1024  # guard against a truncated/placeholder file

failures = []


def check(cond: bool, msg: str) -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {msg}")
    if not cond:
        failures.append(msg)


def main() -> int:
    dist = Path("dist")
    if not dist.is_dir():
        print("FAIL  dist/ does not exist - run `python -m build` first")
        return 1

    wheels = sorted(dist.glob("*.whl"))
    sdists = sorted(dist.glob("*.tar.gz"))

    check(len(wheels) == 1, f"exactly one wheel produced (found {len(wheels)})")
    check(len(sdists) == 1, f"exactly one sdist produced (found {len(sdists)})")
    if not wheels or not sdists:
        return 1

    for path in wheels + sdists:
        size = path.stat().st_size
        check(
            size < PYPI_MAX_BYTES,
            f"{path.name} is {size/1048576:.1f} MB (limit 100 MiB)",
        )

    with zipfile.ZipFile(wheels[0]) as z:
        names = z.namelist()
        info = {i.filename: i.file_size for i in z.infolist()}

    check(CHECKPOINT in info, f"wheel contains {CHECKPOINT}")
    if CHECKPOINT in info:
        size = info[CHECKPOINT]
        check(
            size >= MIN_CHECKPOINT_BYTES,
            f"checkpoint is {size/1048576:.1f} MB (expected >= 50 MB - "
            f"a placeholder here means the model was not committed)",
        )

    for cfg in CONFIGS:
        check(cfg in info, f"wheel contains {cfg}")

    check(
        any(n.endswith("entry_points.txt") for n in names),
        "wheel declares the sahu65 console script",
    )

    meta = [n for n in names if n.endswith(".dist-info/METADATA")]
    check(bool(meta), "wheel has METADATA")
    if meta:
        text = zipfile.ZipFile(wheels[0]).read(meta[0]).decode("utf-8", "replace")
        for dep in ("onnxruntime", "pillow", "fastapi", "httpx"):
            check(f"Requires-Dist: {dep}" in text, f"declares dependency {dep}")
        check("Name: sahu65" in text, "distribution name is sahu65")

    with tarfile.open(sdists[0]) as t:
        sdist_names = t.getnames()
    check(
        any(n.endswith("LICENSE") for n in sdist_names),
        "sdist contains LICENSE",
    )
    check(
        any(n.endswith("README.md") for n in sdist_names),
        "sdist contains README.md",
    )

    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED - do not publish.")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("All distribution checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())