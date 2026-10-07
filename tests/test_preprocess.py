"""Unit tests for image decoding, EXIF orientation, and model preprocessing tensors."""
import io
import numpy as np
from PIL import Image
from sahu65.services.preprocess import (
    decode_image,
    preprocess_for_classifier,
    preprocess_for_clip,
)


def test_decode_image_and_warnings():
    # 64x64 tiny image should trigger low_resolution warning
    img = Image.new("RGB", (64, 64), color="green")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    decoded, warnings = decode_image(buf.getvalue())
    assert decoded.size == (64, 64)
    assert "low_resolution" in warnings


def test_preprocess_for_classifier():
    img = Image.new("RGB", (300, 400), color="blue")
    tensor = preprocess_for_classifier(img, target_size=224)
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == np.float32


def test_preprocess_for_clip():
    img = Image.new("RGB", (300, 400), color="blue")
    # Single center crop
    tensor = preprocess_for_clip(img, target_size=224, enable_tta=False)
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == np.float32

    # 5-crop TTA
    tensor_tta = preprocess_for_clip(img, target_size=224, enable_tta=True)
    assert tensor_tta.shape == (5, 3, 224, 224)
    assert tensor_tta.dtype == np.float32
