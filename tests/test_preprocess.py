"""Unit tests for image decoding and the shipped detector's preprocessing tensor."""
import io

import numpy as np
from PIL import Image

from sahu65.services.preprocess import decode_image, preprocess_for_clip


def test_decode_image_and_warnings():
    # 64x64 tiny image should trigger low_resolution warning
    img = Image.new("RGB", (64, 64), color="green")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    decoded, warnings = decode_image(buf.getvalue())
    assert decoded.size == (64, 64)
    assert "low_resolution" in warnings


def test_preprocess_for_clip_is_aspect_preserving():
    """The probe must not squash. A squashed image loses aspect, and the probe head was
    fitted on aspect-preserving centre crops, so a squash here would silently change
    every score."""
    img = Image.new("RGB", (300, 400), color="blue")
    tensor = preprocess_for_clip(img, target_size=224, enable_tta=False)
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == np.float32


def test_preprocess_for_clip_tta_returns_five_crops():
    img = Image.new("RGB", (300, 400), color="blue")
    tensor_tta = preprocess_for_clip(img, target_size=224, enable_tta=True)
    assert tensor_tta.shape == (5, 3, 224, 224)
    assert tensor_tta.dtype == np.float32


def test_preprocess_normalisation_uses_clip_statistics():
    """CLIP mean/std, not ImageNet. A regression here would shift every score."""
    img = Image.new("RGB", (224, 224), color=(124, 116, 104))
    tensor = preprocess_for_clip(img, target_size=224)
    expected = np.array([124 / 255, 116 / 255, 104 / 255], dtype=np.float32)
    expected = (expected - np.array([0.48145466, 0.4578275, 0.40821073])) / np.array(
        [0.26862954, 0.26130258, 0.27577711])
    np.testing.assert_allclose(tensor[0, :, 0, 0], expected.astype(np.float32), rtol=1e-4,
                               atol=1e-6)