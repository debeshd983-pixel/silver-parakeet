"""Unit tests for magic byte detection, file limits, pixel decompression guards, and rate limiter."""
import io
import pytest
from PIL import Image
from sahu65.config import Settings
from sahu65.core.limits import (
    ImageTooLargePixelsError,
    SimpleRateLimiter,
    inspect_image_pixels,
    sniff_image_type,
)


def test_sniff_image_type():
    jpeg_hdr = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01"
    png_hdr = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
    webp_hdr = b"RIFF\x14\x00\x00\x00WEBPVP8 "
    bad_hdr = b"GIF89a\x01\x00\x01\x00\x80\x00\x00"

    assert sniff_image_type(jpeg_hdr) == "jpeg"
    assert sniff_image_type(png_hdr) == "png"
    assert sniff_image_type(webp_hdr) == "webp"
    assert sniff_image_type(bad_hdr) is None


def test_inspect_image_pixels_bomb_guard():
    settings = Settings(max_pixels=1000)  # low threshold for testing

    # Valid image under limit (20x20 = 400 pixels)
    img_ok = Image.new("RGB", (20, 20), color="blue")
    buf_ok = io.BytesIO()
    img_ok.save(buf_ok, format="JPEG")
    w, h = inspect_image_pixels(buf_ok.getvalue(), settings)
    assert w == 20 and h == 20

    # Image exceeding pixel limit (50x50 = 2500 pixels)
    img_big = Image.new("RGB", (50, 50), color="red")
    buf_big = io.BytesIO()
    img_big.save(buf_big, format="JPEG")
    with pytest.raises(ImageTooLargePixelsError):
        inspect_image_pixels(buf_big.getvalue(), settings)


def test_rate_limiter():
    limiter = SimpleRateLimiter(rate_limit_str="3/minute")
    key = "127.0.0.1"
    assert limiter.check(key) is True
    assert limiter.check(key) is True
    assert limiter.check(key) is True
    # 4th request exceeds 3/minute limit
    assert limiter.check(key) is False
