"""Demo: run the bundled sahu65 detector over every picture in ./assets.

    python demo.py                  # scan the assets/ folder next to this file
    python demo.py some/other/dir    # scan any directory (searched recursively)
    python demo.py --json            # one JSON object per image instead of a table

Local inference - no server and no API key: the ~92 MB checkpoint ships inside the
package, so the first call spends ~10 s loading the ONNX session and every image
after that is ~200 ms on CPU.

Output is a probability, NOT proof - see MODEL_CARD.md before acting on it,
especially for document or ID-card images (measured AUC on documents is 0.375).

Exit codes: 0 = every image scanned (regardless of verdict), 1 = no directory /
no images / at least one image failed to scan.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import sahu65

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
AI_VERDICTS = {"likely_ai", "ai_generated_verified"}


def find_images(root: Path):
    """Every image under ``root``, sorted, with non-image files (ATTRIBUTION.md...) skipped."""
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def category(verdict: str) -> str:
    if verdict in AI_VERDICTS:
        return "ai"
    if verdict == "likely_real":
        return "real"
    return "inconclusive"


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect AI-generated images under a directory")
    parser.add_argument(
        "directory",
        nargs="?",
        default=str(Path(__file__).resolve().parent / "assets"),
        help="directory to scan recursively (default: assets/ next to this script)",
    )
    parser.add_argument("--json", action="store_true", help="emit one JSON object per image")
    args = parser.parse_args()

    root = Path(args.directory).resolve()
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        return 1

    images = find_images(root)
    if not images:
        print(f"error: no images ({', '.join(sorted(IMAGE_SUFFIXES))}) found under {root}", file=sys.stderr)
        return 1

    print(f"scanning {len(images)} image(s) under {root}", file=sys.stderr)
    print("loading bundled model (first call takes ~10 s)...", file=sys.stderr)
    started = time.time()
    sahu65.warm()
    print(f"model ready in {time.time() - started:.1f}s\n", file=sys.stderr)

    counts = {"ai": 0, "real": 0, "inconclusive": 0, "error": 0}
    errors = 0

    for path in images:
        rel = path.relative_to(root)
        try:
            r = sahu65.detect(str(path))
        except Exception as e:
            errors += 1
            counts["error"] += 1
            if args.json:
                print(json.dumps({"path": str(rel), "error": f"{type(e).__name__}: {e}"}))
            else:
                print(f"  {str(rel):<42} ERROR     {type(e).__name__}: {e}")
            continue

        counts[category(r.verdict)] += 1
        if args.json:
            print(json.dumps({"path": str(rel), **r.to_dict()}))
            continue

        warnings = f"  warnings={','.join(r.warnings)}" if r.warnings else ""
        print(
            f"  {str(rel):<42} {r.verdict:<15} p={r.ai_probability:.4f}"
            f"  conf={r.confidence:<6} {r.latency_ms:7.1f}ms{warnings}"
        )

    if not args.json:
        print()
        print(f"--- summary {'-' * 47}")
        print(f"  AI-generated : {counts['ai']}")
        print(f"  Likely real  : {counts['real']}")
        print(f"  Inconclusive : {counts['inconclusive']}")
        if errors:
            print(f"  Errors       : {errors}")
        print()
        print("Reminder: output is an uncalibrated model score, not proof (MODEL_CARD.md).")

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
