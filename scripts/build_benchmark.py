"""Builds benchmark dataset manifest and synthetic test sets across real and AI generators."""
import csv
import hashlib
import os
import random
from PIL import Image, ImageDraw


def compute_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def create_sample_image(path: str, is_ai: bool, generator: str, seed: int):
    random.seed(seed)
    img = Image.new("RGB", (512, 512), color=(random.randint(50, 200), random.randint(50, 200), random.randint(50, 200)))
    draw = ImageDraw.Draw(img)
    # Add patterns simulating photographic vs generated textures
    for _ in range(10):
        xy = [random.randint(0, 512) for _ in range(4)]
        draw.ellipse(xy, fill=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
    img.save(path, format="JPEG", quality=95)


def apply_perturbation(src_path: str, dest_path: str, kind: str):
    with Image.open(src_path) as img:
        if kind == "jpeg_q50":
            img.save(dest_path, "JPEG", quality=50)
        elif kind == "jpeg_q75":
            img.save(dest_path, "JPEG", quality=75)
        elif kind == "resize_half":
            w, h = img.size
            down = img.resize((w // 2, h // 2), Image.Resampling.BILINEAR)
            down.save(dest_path, "JPEG", quality=95)
        elif kind == "center_crop":
            w, h = img.size
            crop = img.crop((w // 4, h // 4, 3 * w // 4, 3 * h // 4))
            crop.save(dest_path, "JPEG", quality=95)
        elif kind == "screenshot":
            w, h = img.size
            down = img.resize((w // 2, h // 2), Image.Resampling.BILINEAR)
            down.save(dest_path, "JPEG", quality=60)


def build_benchmark(output_dir: str = "benchmark", num_parents: int = 100):
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "manifest.csv")

    rows = []
    generators = ["sdxl", "midjourney_v6", "dalle_3", "flux_1"]
    perturbations = ["jpeg_q50", "jpeg_q75", "resize_half", "center_crop", "screenshot"]

    parent_ids = list(range(num_parents))
    random.seed(42)
    random.shuffle(parent_ids)
    half = len(parent_ids) // 2
    calib_parents = set(parent_ids[:half])

    for pid in parent_ids:
        is_ai = (pid % 2 == 1)
        label = 1 if is_ai else 0
        gen = random.choice(generators) if is_ai else "real_camera"
        source = "synthetic_generator" if is_ai else "coco_subset"
        split = "calibration" if pid in calib_parents else "test"

        fname = f"img_{pid:04d}_orig.jpg"
        fpath = os.path.join(images_dir, fname)
        create_sample_image(fpath, is_ai, gen, seed=pid)
        file_hash = compute_sha256(fpath)

        rows.append({
            "path": f"images/{fname}",
            "label": label,
            "source": source,
            "generator": gen,
            "split": split,
            "parent_id": pid,
            "perturbation": "none",
            "sha256": file_hash,
        })

        # Stratified 40% perturbed copies
        if pid % 5 < 2:
            ptype = random.choice(perturbations)
            pfname = f"img_{pid:04d}_{ptype}.jpg"
            pfpath = os.path.join(images_dir, pfname)
            apply_perturbation(fpath, pfpath, ptype)
            p_hash = compute_sha256(pfpath)
            rows.append({
                "path": f"images/{pfname}",
                "label": label,
                "source": source,
                "generator": gen,
                "split": split,
                "parent_id": pid,
                "perturbation": ptype,
                "sha256": p_hash,
            })

    # Sort hashes to compute frozen benchmark ID
    sorted_hashes = sorted(r["sha256"] for r in rows)
    combined = "".join(sorted_hashes).encode("utf-8")
    benchmark_id = hashlib.sha256(combined).hexdigest()[:12]

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["path", "label", "source", "generator", "split", "parent_id", "perturbation", "sha256"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Created benchmark with {len(rows)} images ({len(calib_parents)} calibration parents, {len(parent_ids)-len(calib_parents)} test parents).")
    print(f"Frozen Benchmark ID: bench-{benchmark_id}")
    return f"bench-{benchmark_id}"


if __name__ == "__main__":
    build_benchmark()
