from fastapi.testclient import TestClient

from sahu65.main import app


def test_healthz():
    with TestClient(app) as client:
        r = client.get("/deep-guard/healthz")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}
