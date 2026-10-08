"""Builds a real labelled benchmark for the detector.

This script downloads actual images from documented public sources, records their
provenance and licence, applies perturbations, and writes ``manifest.csv`` plus a
frozen ``benchmark_id``.

It replaces an earlier version of this file that *generated* both the "real" and the
"AI" images by drawing random coloured ellipses. Numbers fitted on that data described
the ellipses, not the detector.

Sources
-------
Synthbuster+ (``marco-willi/synthbuster-plus``)
    Labelled ``real`` / ``fake`` with a ``source`` column naming the generator for the
    fake rows (dalle2, dalle3, midjourney-v5, FLUX.1-dev, FLUX.1-schnell, SD3-medium,
    firefly, glide, stable-diffusion-{1-3,1-4,2,xl}). Fake rows come from those 12
    families; real rows come from RAISE-1k. Synthbuster was designed specifically to
    measure generalisation to generators a detector has not seen, which is the
    failure mode that matters here.
COCO val2017 (``rafaelpadilla/coco2017``)
    5,000 real photographs, used only as extra *real* data so the real class is not
    drawn from a single collection pipeline.

Open Images was evaluated and dropped: its Hub manifest returns 502 from the rows
endpoint and the official ``storage.googleapis.com/openimages`` CSVs return 403 to
anonymous callers. Only two real sources remain, which is a weaker guard against a
source-specific shortcut than intended.

Licence handling
----------------
Every row of the manifest carries ``source_dataset`` and ``license``. The Hub metadata
declares no licence for Synthbuster+ or Open Images, so those rows are recorded with the
upstream terms rather than an invented one. ``SOURCES.md`` states the same in prose.

Format normalisation
--------------------
All canonical images are stored as PNG regardless of how the source encoded them. A
file-format cue (AI datasets often ship PNG, camera photos ship JPEG) is not evidence of
authenticity, and an earlier comparison in this repository was fooled by exactly that.
Perturbed variants are written as JPEG on purpose, since re-encoding is a real-world
condition worth measuring.

Usage
-----
    python scripts/build_benchmark.py --out benchmark
    python scripts/build_benchmark.py --out benchmark --ai-per-family 25 --stride 100
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import requests
from PIL import Image

ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"

# Anonymous requests to the Hub viewer get throttled quickly. Pacing every row request
# is slower but finishes; hammering it returns 429/502 and loses data.
REQUEST_PAUSE_S = 0.4


class FatalSourceError(RuntimeError):
    """A source is unusable (no viewer, 4xx). Retrying will not help."""


def log(msg: str) -> None:
    """Progress line. Flushed so a long run shows movement instead of looking hung."""
    print(msg, flush=True)

# Every canonical image is written as PNG so the label cannot be inferred from the
# container format. See the module docstring.
CANONICAL_FORMAT = "PNG"
CANONICAL_EXT = ".png"

# Perturbations. Each returns the file format it writes, which is recorded per row.
PERTURBATIONS: Tuple[str, ...] = (
    "jpeg_q95",
    "jpeg_q75",
    "jpeg_q50",
    "resize_half",
    "screenshot",
    "center_crop",
)


@dataclass
class SourceSpec:
    """A remote row source and the licence terms recorded for it."""

    name: str
    dataset: str
    config: str
    split: str
    license: str
    license_url: str = ""
    note: str = ""


SYNTHBUSTER = SourceSpec(
    name="synthbuster_plus",
    dataset="marco-willi/synthbuster-plus",
    config="default",
    split="train",
    license="undeclared-on-hub (upstream Synthbuster terms apply)",
    license_url="https://github.com/maxsixty/synthbuster",
    note="Fake rows are attributed to a named generator in the `source` column; "
         "real rows are RAISE-1k photographs.",
)

COCO_VAL2017 = SourceSpec(
    name="coco_val2017",
    dataset="rafaelpadilla/coco2017",
    config="default",
    split="val",
    license="COCO images: Flickr terms, predominantly CC BY 2.0",
    license_url="https://cocodataset.org/#termsofuse",
    note="Used as additional REAL photographs only. No AI images come from COCO. "
         "The canonical COCO image list is served by the Hub dataset viewer, which "
         "avoids a 778 MB val2017.zip download.",
)


@dataclass
class ManifestRow:
    """One benchmark image. ``parent_id`` groups derivatives of the same source image."""

    path: str
    label: int
    source_dataset: str
    generator: str
    split: str
    parent_id: str
    perturbation: str
    sha256: str
    width: int
    height: int
    bytes: int
    fmt: str
    license: str
    image_id: str = ""


@dataclass
class SourceRecord:
    """A remote row we decided to fetch, with its resolved download URL."""

    source: SourceSpec
    label: int
    generator: str
    image_id: str
    url: str


# --------------------------------------------------------------------------- #
# Remote access
# --------------------------------------------------------------------------- #

def _get_rows(spec: SourceSpec, offset: int, length: int, session: requests.Session,
              retries: int = 5) -> List[dict]:
    """Fetches ``length`` dataset rows starting at ``offset``.

    The signed asset URLs the datasets-server returns expire, so callers must download
    promptly after this returns.

    Status handling is deliberate. The Hub viewer endpoint is flaky for anonymous
    callers: the same dataset returns 200 on one call and 502 on the next, and 429 shows
    up when requests come fast. So 5xx and 429 are treated as transient and retried with
    backoff. Only 4xx that cannot mean "try again" (401/403/404) and 501 ("not
    implemented", i.e. no viewer was ever built) are fatal, because retrying those just
    stalls the run.
    """
    params = {
        "dataset": spec.dataset,
        "config": spec.config,
        "split": spec.split,
        "offset": offset,
        "length": length,
    }
    last: Optional[Exception] = None
    for attempt in range(retries):
        try:
            resp = session.get(ROWS_ENDPOINT, params=params, timeout=90)
            code = resp.status_code
            if code in (401, 403, 404, 501):
                raise FatalSourceError(
                    f"{spec.dataset}: rows endpoint returned {code}; "
                    f"this dataset has no usable viewer"
                )
            if code in (429, 500, 502, 503, 504):
                last = RuntimeError(f"transient {code}")
                time.sleep(min(30.0, 2.0 * (2 ** attempt)))
                continue
            resp.raise_for_status()
            return [r["row"] for r in resp.json()["rows"]]
        except FatalSourceError:
            raise
        except Exception as exc:  # transient network faults
            last = exc
            time.sleep(min(30.0, 2.0 * (2 ** attempt)))
    raise RuntimeError(f"could not read rows from {spec.dataset} at offset {offset}: {last}")


def scan_synthbuster(session: requests.Session, stride: int, max_rows: int) -> List[SourceRecord]:
    """Walks Synthbuster+ and returns every row, keeping its generator attribution.

    Sampling is by stride rather than by contiguous block: the dataset is ordered by
    generator, so a contiguous read would cover one family and miss the rest.
    """
    out: List[SourceRecord] = []
    offset = 0
    consecutive_failures = 0
    while offset < max_rows:
        try:
            rows = _get_rows(SYNTHBUSTER, offset, min(stride, 100), session)
            consecutive_failures = 0
        except FatalSourceError as exc:
            log(f"  FATAL {exc}")
            raise
        except RuntimeError as exc:
            consecutive_failures += 1
            log(f"  ! {exc}")
            if consecutive_failures >= 6:
                log("  ! too many consecutive failures; stopping this source scan")
                break
            offset += stride
            continue
        for row in rows:
            img = row.get("image")
            url = img.get("src") if isinstance(img, dict) else None
            if not url:
                continue
            label = int(row.get("label", -1))
            if label not in (0, 1):
                continue
            out.append(
                SourceRecord(
                    source=SYNTHBUSTER,
                    label=label,
                    generator=str(row.get("source") or "unknown"),
                    image_id=str(row.get("image_id") or ""),
                    url=url,
                )
            )
        if offset % (stride * 4) == 0:
            log(f"  synthbuster+ offset={offset} collected={len(out)}")
        offset += stride
        time.sleep(REQUEST_PAUSE_S)
        if not rows and offset > stride * 4:
            break
    return out


def scan_coco(session: requests.Session, want: int, max_rows: int = 5000) -> List[SourceRecord]:
    """Collects real photographs from COCO val2017."""
    out: List[SourceRecord] = []
    offset = 0
    consecutive_failures = 0
    while len(out) < want and offset < max_rows:
        try:
            rows = _get_rows(COCO_VAL2017, offset, 100, session)
            consecutive_failures = 0
        except FatalSourceError as exc:
            log(f"  FATAL {exc}")
            return out
        except RuntimeError as exc:
            consecutive_failures += 1
            log(f"  ! {exc}")
            if consecutive_failures >= 6:
                log("  ! too many consecutive failures; stopping this source scan")
                break
            offset += 100
            continue
        if not rows:
            break
        for row in rows:
            img = row.get("image")
            url = img.get("src") if isinstance(img, dict) else None
            if not url:
                continue
            out.append(
                SourceRecord(
                    source=COCO_VAL2017,
                    label=0,
                    generator="real_coco",
                    image_id=str(row.get("image_id") or f"coco{offset}"),
                    url=url,
                )
            )
            if len(out) >= want:
                break
        offset += 100
        time.sleep(REQUEST_PAUSE_S)
        log(f"  coco offset={offset} collected={len(out)}/{want}")
    return out[:want]


def scan_open_images(session: requests.Session, want: int, seed: int = 0) -> List[SourceRecord]:
    """Retained but disabled.

    Open Images was dropped from the default build for two reasons, both verified: the
    Hub distribution (``bitmind/open-images-v7``) returns 502 from the rows endpoint, and
    the official URL manifests on ``storage.googleapis.com/openimages`` return 403 to
    anonymous callers. It is kept here so the reason is recorded next to the code rather
    than only in a commit message.
    """
    raise FatalSourceError(
        "Open Images is unavailable: hub manifest returns 502 and the official "
        "storage.googleapis.com/openimages CSVs return 403 anonymously"
    )


# --------------------------------------------------------------------------- #
# Download and normalisation
# --------------------------------------------------------------------------- #

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download_image(url: str, session: requests.Session, retries: int = 3) -> Optional[bytes]:
    """Downloads image bytes, or None if the URL is dead or the payload is not an image."""
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=90)
            resp.raise_for_status()
            data = resp.content
            if not data:
                return None
            # Cheap sniff so an HTML error page is never saved as an image.
            if not (data[:3] == b"\xff\xd8\xff" or data[:8] == b"\x89PNG\r\n\x1a\n"
                    or data[:4] == b"RIFF" or data[:2] == b"BM"):
                return None
            return data
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return None


def open_rgb(data: bytes) -> Optional[Image.Image]:
    """Decodes to RGB, applying EXIF orientation. Returns None if undecodable."""
    try:
        img = Image.open(io.BytesIO(data))
        try:
            from PIL import ImageOps

            img = ImageOps.exif_transpose(img)
        except Exception:
            pass
        return img.convert("RGB")
    except Exception:
        return None


def encode_png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def encode_jpeg(img: Image.Image, quality: int) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def apply_perturbation(img: Image.Image, kind: str) -> Tuple[Image.Image, str]:
    """Applies a perturbation, returning the new image and the format it will be saved as."""
    if kind == "jpeg_q95":
        return img, "JPEG-95"
    if kind == "jpeg_q75":
        return img, "JPEG-75"
    if kind == "jpeg_q50":
        return img, "JPEG-50"
    if kind == "resize_half":
        w, h = img.size
        return img.resize((max(w // 2, 1), max(h // 2, 1)), Image.Resampling.BILINEAR), "PNG"
    if kind == "screenshot":
        # A screenshot-like path: downscale, then re-encode hard.
        w, h = img.size
        return img.resize((max(w // 2, 1), max(h // 2, 1)), Image.Resampling.BILINEAR), "JPEG-60"
    if kind == "center_crop":
        w, h = img.size
        return img.crop((w // 4, h // 4, 3 * w // 4, 3 * h // 4)), "PNG"
    raise ValueError(f"unknown perturbation: {kind}")


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #

def select_synthbuster(records: Sequence[SourceRecord], ai_per_family: int,
                       real_count: int, seed: int = 0
                       ) -> List[SourceRecord]:
    """Picks a per-generator-balanced AI sample plus a real sample.

    Balancing matters: Synthbuster+ is dominated by whichever family is most common, and
    an unbalanced sample would report an accuracy that mostly measures that one family.
    """
    rng = random.Random(seed)
    by_family: Dict[str, List[SourceRecord]] = defaultdict(list)
    reals: List[SourceRecord] = []
    for rec in records:
        (reals if rec.label == 0 else by_family[rec.generator]).append(rec)

    for pool in by_family.values():
        rng.shuffle(pool)

    chosen: List[SourceRecord] = []
    for family in sorted(by_family):
        take = by_family[family][:ai_per_family]
        if take:
            chosen.extend(take)

    rng.shuffle(reals)
    chosen.extend(reals[:real_count])
    return chosen


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #

def assign_splits(parents: Sequence[Tuple[str, int, str]], seed: int = 0
                  ) -> Dict[str, str]:
    """Assigns each parent image to calibration or test, stratified by (label, generator).

    Stratifying keeps every generator represented in both halves; without it a 50/50
    split can leave a family entirely in test, and its recall becomes unreportable.

    Perturbed copies inherit their parent's split, so a re-encoded variant of one image
    can never sit on the other side of the split from the original.
    """
    rng = random.Random(seed)
    groups: Dict[Tuple[int, str], List[str]] = defaultdict(list)
    for parent_id, label, generator in parents:
        groups[(label, generator)].append(parent_id)

    out: Dict[str, str] = {}
    for key in sorted(groups):
        ids = sorted(groups[key])
        rng.shuffle(ids)
        half = len(ids) // 2
        for pid in ids[:half]:
            out[pid] = "calibration"
        for pid in ids[half:]:
            out[pid] = "test"
    return out


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #

def build(out_dir: Path, ai_per_family: int, raise_reals: int, coco_reals: int,
          stride: int, perturb_frac: float, perturb_kinds: Sequence[str],
          seed: int, session: Optional[requests.Session] = None) -> Dict[str, object]:
    session = session or requests.Session()
    session.headers.update({"User-Agent": "sahu65-benchmark-builder/2.0"})

    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    log("Scanning Synthbuster+ (labels + generator attribution) ...")
    synth = scan_synthbuster(session, stride=stride, max_rows=13_999)
    log(f"  found {len(synth)} labelled rows")
    if not synth:
        raise RuntimeError("Synthbuster+ returned no rows; refusing to build a benchmark")

    fakes = {r.generator for r in synth if r.label == 1}
    reals_avail = sum(1 for r in synth if r.label == 0)
    log(f"  generator families: {len(fakes)}   real rows available: {reals_avail}")

    log("Scanning COCO val2017 (real photographs) ...")
    coco = scan_coco(session, want=coco_reals)
    log(f"  found {len(coco)} real photographs")

    records = select_synthbuster(synth, ai_per_family, raise_reals, seed=seed) + coco

    # Drop duplicates and anything lacking a usable URL.
    seen: set = set()
    unique: List[SourceRecord] = []
    for rec in records:
        key = rec.image_id or rec.url
        if key in seen:
            continue
        seen.add(key)
        unique.append(rec)
    records = unique

    rng = random.Random(seed)
    rng.shuffle(records)

    rows: List[ManifestRow] = []
    parents: List[Tuple[str, int, str]] = []
    used_files: List[str] = []
    failures = 0

    for idx, rec in enumerate(records, start=1):
        if idx % 25 == 0 or idx == len(records):
            log(f"  downloading {idx}/{len(records)} (failures so far: {failures})")
        data = download_image(rec.url, session)
        if data is None:
            failures += 1
            continue
        img = open_rgb(data)
        if img is None:
            failures += 1
            continue
        w, h = img.size
        if min(w, h) < 128 or w * h > 50_000_000:
            failures += 1
            continue

        raw_id = rec.image_id or f"{len(used_files):06d}"
        safe_id = "".join(ch if ch.isalnum() else "_" for ch in str(raw_id))[:64]
        pid = f"{rec.source.name}_{safe_id}"
        slug = "".join(ch if ch.isalnum() else "_" for ch in rec.generator)[:32]
        canonical = f"{pid}__{slug}{CANONICAL_EXT}"
        canonical_path = images_dir / canonical
        if not canonical_path.exists():
            canonical_path.write_bytes(encode_png(img))

        parents.append((pid, rec.label, rec.generator))
        rows.append(
            ManifestRow(
                path=f"images/{canonical}",
                label=rec.label,
                source_dataset=rec.source.name,
                generator=rec.generator,
                split="",  # filled after splitting
                parent_id=pid,
                perturbation="original",
                sha256=sha256_bytes(canonical_path.read_bytes()),
                width=w,
                height=h,
                bytes=canonical_path.stat().st_size,
                fmt=CANONICAL_FORMAT,
                license=rec.source.license,
                image_id=rec.image_id,
            )
        )
        used_files.append(canonical)

        if rng.random() < perturb_frac:
            for kind in perturb_kinds:
                try:
                    pimg, pfmt = apply_perturbation(img, kind)
                except ValueError:
                    continue
                name = f"{pid}__{kind}{'.jpg' if pfmt.startswith('JPEG') else CANONICAL_EXT}"
                ppath = images_dir / name
                payload = encode_jpeg(pimg, int(pfmt.split("-")[1])) if pfmt.startswith("JPEG") \
                    else encode_png(pimg)
                if not ppath.exists():
                    ppath.write_bytes(payload)
                pw, ph = pimg.size
                rows.append(
                    ManifestRow(
                        path=f"images/{name}",
                        label=rec.label,
                        source_dataset=rec.source.name,
                        generator=rec.generator,
                        split="",
                        parent_id=pid,
                        perturbation=kind,
                        sha256=sha256_bytes(ppath.read_bytes()),
                        width=pw,
                        height=ph,
                        bytes=ppath.stat().st_size,
                        fmt=pfmt,
                        license=rec.source.license,
                        image_id=rec.image_id,
                    )
                )

    if not rows:
        raise RuntimeError("no images were retrieved; refusing to write an empty manifest")

    splits = assign_splits(parents, seed=seed)
    for row in rows:
        row.split = splits[row.parent_id]

    rows.sort(key=lambda r: (r.parent_id, r.perturbation))

    manifest_path = out_dir / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    # The benchmark id is a content hash of the manifest, so it changes if any image or
    # any label changes. It is what thresholds get pinned against.
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda r: r.path):
        digest.update(f"{row.path}:{row.sha256}:{row.label}\n".encode("utf-8"))
    benchmark_id = "bench-" + digest.hexdigest()[:12]

    sources_path = out_dir / "SOURCES.md"
    write_sources_doc(sources_path, rows, failures)

    summary = summarise(rows)
    summary.update(
        {
            "benchmark_id": benchmark_id,
            "manifest": str(manifest_path),
            "failures": failures,
            "images_on_disk": len(list(images_dir.iterdir())),
        }
    )

    meta_path = out_dir / "benchmark.json"
    meta_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    log("")
    log(f"benchmark_id = {benchmark_id}")
    log(f"manifest     = {manifest_path}")
    log(f"images       = {summary['images_on_disk']} files, {failures} downloads failed")
    for key in ("parents", "ai_parents", "real_parents", "rows",
                "ai_rows", "real_rows", "calibration_rows", "test_rows"):
        log(f"  {key:18s} {summary[key]}")
    log("  generators:")
    for gen, n in sorted(summary["by_generator"].items(), key=lambda kv: -kv[1]):
        log(f"    {gen:28s} {n}")

    return summary


def summarise(rows: Sequence[ManifestRow]) -> Dict[str, object]:
    """Summarises the manifest.

    Parent counts are reported alongside row counts because rows include perturbed
    copies. A parent-level view is the one that describes class balance honestly: rows
    can look 5:1 real-heavy purely from how many perturbations landed on each side.
    """
    parents = {r.parent_id for r in rows}
    by_gen: Dict[str, int] = defaultdict(int)
    ai_parents = real_parents = 0
    seen_parent: set = set()
    for r in rows:
        if r.perturbation == "original":
            by_gen[r.generator] += 1
        if r.parent_id not in seen_parent:
            seen_parent.add(r.parent_id)
            if r.label == 1:
                ai_parents += 1
            else:
                real_parents += 1
    return {
        "rows": len(rows),
        "parents": len(parents),
        "ai_rows": sum(1 for r in rows if r.label == 1),
        "real_rows": sum(1 for r in rows if r.label == 0),
        "ai_parents": ai_parents,
        "real_parents": real_parents,
        "calibration_rows": sum(1 for r in rows if r.split == "calibration"),
        "test_rows": sum(1 for r in rows if r.split == "test"),
        "by_generator": dict(by_gen),
    }


def write_sources_doc(path: Path, rows: Sequence[ManifestRow], failures: int) -> None:
    counts: Dict[Tuple[str, str, str], int] = defaultdict(int)
    for r in rows:
        counts[(r.source_dataset, r.license, r.generator)] += 1

    lines = [
        "# Benchmark sources and licences",
        "",
        "Generated by `scripts/build_benchmark.py`. Every image in `manifest.csv` traces",
        "to a row below.",
        "",
        "Canonical images are stored as PNG regardless of the source encoding, so the",
        "container format cannot be used as a proxy for the label. Perturbed variants are",
        "stored as JPEG on purpose: re-encoding is a real-world condition worth measuring.",
        "",
        "| source_dataset | generator | rows | recorded licence |",
        "|---|---|---|---|",
    ]
    for (src, lic, gen), n in sorted(counts.items()):
        lines.append(f"| `{src}` | `{gen}` | {n} | {lic} |")

    lines += [
        "",
        "## Notes",
        "",
        "- `marco-willi/synthbuster-plus` declares no licence in its Hub metadata. The rows",
        "  here are recorded with the upstream Synthbuster terms, not an assumed licence.",
        "- Fake rows name their generator in the `source` column; real Synthbuster rows are",
        "  RAISE-1k photographs.",
        "- COCO val2017 contributes **real** photographs only. It exists so the real class is",
        "  not drawn from a single collection pipeline.",
        "- Open Images was evaluated and dropped: the Hub manifest (`bitmind/open-images-v7`)",
        "  returns 502 from the rows endpoint and the official",
        "  `storage.googleapis.com/openimages` CSVs return 403 to anonymous callers. With",
        "  only two real sources the format-shortcut guard is weaker than intended; adding a",
        "  third real pipeline is the first thing to fix when a reachable mirror appears.",
        f"- {failures} candidate downloads failed during the build and were skipped.",
        "",
        "## What this benchmark does not cover",
        "",
        "- Document, ID, receipt, and form imagery. The detector has no measurable",
        "  discriminative power there (see MODEL_CARD.md 3.2) and this benchmark",
        "  deliberately excludes it rather than padding the numbers.",
        "- Video, audio, and animation frames.",
        "- Adversarial or deliberately re-encoded evasion beyond the listed perturbations.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="benchmark", help="output directory")
    p.add_argument("--ai-per-family", type=int, default=18,
                   help="fake images to take from each generator family")
    p.add_argument("--raise-reals", type=int, default=100,
                   help="real photographs to take from RAISE-1k via Synthbuster+")
    p.add_argument("--coco-reals", type=int, default=100,
                   help="real photographs to take from COCO val2017")
    p.add_argument("--stride", type=int, default=150,
                   help="row stride when scanning Synthbuster+; lower is a fuller sweep")
    p.add_argument("--perturb-frac", type=float, default=0.30,
                   help="fraction of parent images that get perturbed variants")
    p.add_argument("--perturb-kinds", nargs="*", default=list(PERTURBATIONS[:5]))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        build(
            out_dir=out_dir,
            ai_per_family=args.ai_per_family,
            raise_reals=args.raise_reals,
            coco_reals=args.coco_reals,
            stride=args.stride,
            perturb_frac=args.perturb_frac,
            perturb_kinds=tuple(args.perturb_kinds),
            seed=args.seed,
        )
    except Exception as exc:
        print(f"build_benchmark failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())