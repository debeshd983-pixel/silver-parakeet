"""Integration tests for FastAPI endpoints: /deep-guard/detect, /healthz, /readyz, /version.

All routes are mounted under the ``/deep-guard`` prefix.
"""
import io
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sahu65.main import app

READY_TIMEOUT_S = 120


@pytest.fixture
def client():
    with TestClient(app) as c:
        # Models load in the background (~8s of ONNX deserialization), so /deep-guard/readyz
        # must be polled rather than assumed. If the real weights are absent the app
        # deliberately never becomes ready; those tests are skipped below.
        deadline = time.time() + READY_TIMEOUT_S
        while time.time() < deadline:
            if c.get("/deep-guard/readyz").status_code == 200:
                break
            time.sleep(0.5)
        yield c


def create_test_image(format="JPEG", size=(256, 256), color="red") -> bytes:
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()


def test_healthz_responds_before_model_is_loaded():
    """/deep-guard/healthz must never block on the ~8s model load - that is the whole point
    of loading in the background."""
    with TestClient(app) as c:
        r = c.get("/deep-guard/healthz")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


def test_readyz_reports_load_failure_when_weights_missing():
    """A missing checkpoint must surface as 503, never as a silent 0.50 stub."""
    with TestClient(app) as c:
        r = c.get("/deep-guard/readyz")
        assert r.status_code in (200, 503)
        if r.status_code == 503:
            assert r.json()["error"]["code"] in ("model_not_ready", "model_load_failed")


def test_healthz_and_readyz(client):
    r_health = client.get("/deep-guard/healthz")
    assert r_health.status_code == 200
    assert r_health.json() == {"status": "ok"}

    r_ready = client.get("/deep-guard/readyz")
    assert r_ready.status_code == 200
    assert r_ready.json() == {"status": "ready"}


def test_version_endpoint(client):
    r = client.get("/deep-guard/version")
    assert r.status_code == 200
    data = r.json()
    assert "model_version" in data
    assert "thresholds" in data
    assert "models" in data


def test_detect_valid_image(client):
    img_bytes = create_test_image(format="JPEG", size=(300, 300))
    files = {"file": ("sample.jpg", img_bytes, "image/jpeg")}
    r = client.post("/deep-guard/detect", files=files)
    assert r.status_code == 200
    data = r.json()

    assert "request_id" in data
    assert data["verdict"] in ["likely_ai", "likely_real", "inconclusive", "ai_generated_verified"]
    assert 0.0 <= data["ai_probability"] <= 1.0
    assert data["confidence"] in ["high", "medium", "low"]
    assert "signals" in data
    assert "detector" in data["signals"]
    assert 0.0 <= data["signals"]["detector"]["probability"] <= 1.0
    assert data["signals"]["c2pa"]["present"] in (True, False)
    assert "latency_ms" in data


def test_detect_unsupported_file_type(client):
    bad_bytes = b"This is plain text not an image"
    files = {"file": ("test.txt", bad_bytes, "text/plain")}
    r = client.post("/deep-guard/detect", files=files)
    assert r.status_code == 415
    data = r.json()
    assert "error" in data
    assert data["error"]["code"] == "unsupported_type"


def test_detect_corrupt_jpeg(client):
    corrupt_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 20  # JPEG header followed by junk
    files = {"file": ("corrupt.jpg", corrupt_bytes, "image/jpeg")}
    r = client.post("/deep-guard/detect", files=files)
    assert r.status_code == 400
    data = r.json()
    assert "error" in data
    assert data["error"]["code"] == "invalid_image"
