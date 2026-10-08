"""Image preprocessing for the shipped detector (a linear probe on frozen CLIP).

The probe is fed an aspect-preserving centre crop with CLIP mean/std normalisation. The
clip class mapping is resolved from the checkpoint's own ``id2label`` when one exists;
the shipped probe has no labels of its own, so its direction is a property of the fitted
head (``clip_head.json``) rather than something read from a config file.
"""
import io
from typing import List, Tuple
import numpy as np
from PIL import Image, ImageOps

from sahu65.core.limits import InvalidImageError


CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32).reshape(1, 3, 1, 1)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32).reshape(1, 3, 1, 1)

def decode_image(data: bytes) -> Tuple[Image.Image, List[str]]:
    """Decodes image, handles EXIF orientation, converts to RGB, and collects warnings."""
    try:
        img = Image.open(io.BytesIO(data))
        # Handle EXIF orientation
        try:
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass  # Some images may have corrupt EXIF orientation tags

        if img.mode != "RGB":
            img = img.convert("RGB")

        warnings: List[str] = []
        w, h = img.size
        if min(w, h) < 128:
            warnings.append("low_resolution")

        # Heuristic for JPEG recompression if format is JPEG
        if getattr(img, "format", None) == "JPEG" or (len(data) >= 3 and data.startswith(b"\xff\xd8\xff")):
            # Check file size to pixel ratio heuristic
            bpp = len(data) / max(w * h, 1)
            if bpp < 0.25:
                warnings.append("jpeg_recompressed_likely")

        return img, warnings
    except Exception as e:
        raise InvalidImageError(f"Failed to decode image: {str(e)}")


def preprocess_for_clip(img: Image.Image, target_size: int = 224, enable_tta: bool = False) -> np.ndarray:
    """Preprocesses image for S3 CLIP: resize shortest side to target_size, center crop, normalize with CLIP mean/std.

    If enable_tta is True: returns (5, 3, target_size, target_size) representing 5 crops (center + 4 corners).
    Otherwise returns (1, 3, target_size, target_size).
    """
    w, h = img.size
    scale = target_size / min(w, h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = img.resize((new_w, new_h), Image.Resampling.BICUBIC)

    if not enable_tta:
        # Center crop
        left = (new_w - target_size) // 2
        top = (new_h - target_size) // 2
        cropped = resized.crop((left, top, left + target_size, top + target_size))

        arr = np.asarray(cropped, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))[np.newaxis, ...]
        arr = (arr - CLIP_MEAN) / CLIP_STD
        return arr.astype(np.float32)

    # 5-crop TTA: Center, Top-Left, Top-Right, Bottom-Left, Bottom-Right
    crops = []
    # Center
    c_left = (new_w - target_size) // 2
    c_top = (new_h - target_size) // 2
    crops.append(resized.crop((c_left, c_top, c_left + target_size, c_top + target_size)))
    # Top-Left
    crops.append(resized.crop((0, 0, target_size, target_size)))
    # Top-Right
    crops.append(resized.crop((new_w - target_size, 0, new_w, target_size)))
    # Bottom-Left
    crops.append(resized.crop((0, new_h - target_size, target_size, new_h)))
    # Bottom-Right
    crops.append(resized.crop((new_w - target_size, new_h - target_size, new_w, new_h)))

    tensors = []
    for c in crops:
        arr = np.asarray(c, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))[np.newaxis, ...]
        arr = (arr - CLIP_MEAN) / CLIP_STD
        tensors.append(arr)

    return np.concatenate(tensors, axis=0).astype(np.float32)
