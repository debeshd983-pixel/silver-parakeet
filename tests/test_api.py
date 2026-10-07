"""Integration tests for FastAPI endpoints: /v1/detect, /healthz, /readyz, /version."""
import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def create_test_image(format="JPEG", size=(256, 256), color="red") -> bytes:
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()


def test_healthz_and_readyz(client):
    r_health = client.get("/healthz")
    assert r_health.status_code == 200
    assert r_health.json() == {"status": "ok"}

    r_ready = client.get("/readyz")
    assert r_ready.status_code == 200
    assert r_ready.json() == {"status": "ready"}


def test_version_endpoint(client):
    r = client.get("/version")
    assert r.status_code == 200
    data = r.json()
    assert "model_version" in data
    assert "thresholds" in data
    assert "models" in data


def test_detect_valid_image(client):
    img_bytes = create_test_image(format="JPEG", size=(300, 300))
    files = {"file": ("sample.jpg", img_bytes, "image/jpeg")}
    r = client.post("/v1/detect", files=files)
    assert r.status_code == 200
    data = r.json()

    assert "request_id" in data
    assert data["verdict"] in ["likely_ai", "likely_real", "inconclusive", "ai_generated_verified"]
    assert 0.0 <= data["ai_probability"] <= 1.0
    assert data["confidence"] in ["high", "medium", "low"]
    assert "signals" in data
    assert "classifier" in data["signals"]
    assert "clip_probe" in data["signals"]
    assert "latency_ms" in data


def test_detect_unsupported_file_type(client):
    bad_bytes = b"This is plain text not an image"
    files = {"file": ("test.txt", bad_bytes, "text/plain")}
    r = client.post("/v1/detect", files=files)
    assert r.status_code == 415
    data = r.json()
    assert "error" in data
    assert data["error"]["code"] == "unsupported_type"


def test_detect_corrupt_jpeg(client):
    corrupt_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 20  # JPEG header followed by junk
    files = {"file": ("corrupt.jpg", corrupt_bytes, "image/jpeg")}
    r = client.post("/v1/detect", files=files)
    assert r.status_code == 400
    data = r.json()
    assert "error" in data
    assert data["error"]["code"] == "invalid_image"
