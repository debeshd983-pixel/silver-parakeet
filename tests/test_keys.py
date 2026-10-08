"""Bear Token (API key) tests: minting, digest-only storage, CLI, and request auth.

Route tests deliberately use ``TestClient(app)`` WITHOUT the context manager:
authentication now runs before the readiness check, so a 401 proves the auth
decision without waiting ~8 s for the ONNX weights on every test.
"""
import io
import re

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from sahu65 import keys as key_store
from sahu65.cli import main as cli_main
from sahu65.config import get_settings
from sahu65.main import app


@pytest.fixture
def keyfile(tmp_path, monkeypatch):
    path = tmp_path / "keys.json"
    monkeypatch.setenv("SAHU65_KEYFILE", str(path))
    monkeypatch.delenv("API_KEYS", raising=False)
    get_settings.cache_clear()
    yield path
    get_settings.cache_clear()


def jpeg_bytes(size=(64, 64)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color="red").save(buf, format="JPEG")
    return buf.getvalue()


def post_detect(client, headers=None):
    return client.post(
        "/deep-guard/detect",
        headers=headers or {},
        files={"file": ("t.jpg", jpeg_bytes(), "image/jpeg")},
    )


# --- keys module ------------------------------------------------------------


def test_minted_token_format_and_verify(keyfile):
    token, rotated = key_store.create_key("my app!")
    assert token.startswith("bear_my-app_")
    assert rotated is False
    assert key_store.verify(token, "") is True
    assert key_store.verify(token + "x", "") is False
    assert key_store.verify(None, "") is False
    assert key_store.auth_configured("") is True


def test_only_digest_is_stored(keyfile):
    token, _ = key_store.create_key("ci")
    stored = keyfile.read_text(encoding="utf-8")
    assert token not in stored
    assert key_store._digest(token) in stored


def test_same_name_rotates_token(keyfile):
    old, _ = key_store.create_key("ci")
    new, rotated = key_store.create_key("ci")
    assert rotated is True
    assert new != old
    assert key_store.verify(old, "") is False
    assert key_store.verify(new, "") is True
    assert len(key_store.list_keys()) == 1


def test_env_api_keys_still_accepted(keyfile):
    # file is empty here - only API_KEYS matters
    assert key_store.auth_configured("") is False
    assert key_store.auth_configured("k1") is True
    assert key_store.verify("k1", "k2, k1") is True
    assert key_store.verify("nope", "k1") is False


def test_unusable_name_rejected(keyfile):
    with pytest.raises(ValueError):
        key_store.create_key("!!!")


def test_list_keys_never_exposes_tokens(keyfile):
    key_store.create_key("alpha")
    key_store.create_key("beta")
    listed = key_store.list_keys()
    assert [r["name"] for r in listed] == ["alpha", "beta"]
    assert all("sha256" not in r and "token" not in r for r in listed)


# --- CLI --------------------------------------------------------------------


def test_cli_mint_and_list(keyfile, capsys):
    assert cli_main(["--key", "ci"]) == 0
    out = capsys.readouterr().out
    token = re.search(r"bear_ci_[A-Za-z0-9_-]+", out)
    assert token, out
    assert key_store.verify(token.group(0), "")

    assert cli_main(["--list-keys"]) == 0
    listed = capsys.readouterr().out
    assert "ci" in listed
    assert token.group(0) not in listed  # listing never prints the token


def test_cli_key_sugar_form(keyfile, capsys):
    assert cli_main(["--key-prod"]) == 0
    out = capsys.readouterr().out
    assert re.search(r"bear_prod_[A-Za-z0-9_-]+", out), out


def test_cli_rotation_notice(keyfile, capsys):
    assert cli_main(["--key", "web"]) == 0
    capsys.readouterr()
    assert cli_main(["--key-web"]) == 0
    out = capsys.readouterr().out
    assert "rotated" in out


def test_cli_bare_invocation_prints_help_and_fails(capsys):
    assert cli_main([]) == 2
    assert "usage:" in capsys.readouterr().out


# --- request authentication -------------------------------------------------


def test_detect_open_when_no_keys_configured(keyfile):
    client = TestClient(app)
    r = post_detect(client)
    assert r.status_code in (200, 503)
    assert r.status_code != 401


def test_detect_requires_token_once_key_exists(keyfile):
    key_store.create_key("web")
    get_settings.cache_clear()
    client = TestClient(app)

    r = post_detect(client)
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "unauthorized"

    r = post_detect(client, headers={"X-API-Key": "bear_web_wrong"})
    assert r.status_code == 401

    r = post_detect(client, headers={"Authorization": "Bearer bear_web_wrong"})
    assert r.status_code == 401

    # Another scheme must not authenticate.
    r = post_detect(client, headers={"Authorization": "Basic dXNlcjpwdw=="})
    assert r.status_code == 401


def test_detect_accepts_bearer_and_x_api_key(keyfile):
    token, _ = key_store.create_key("web")
    get_settings.cache_clear()
    client = TestClient(app)

    # Auth passes before readiness; the model may still be loading (503) in which
    # case 503 itself proves the 401 was skipped.
    r = post_detect(client, headers={"Authorization": f"Bearer {token}"})
    assert r.status_code in (200, 503)
    assert r.status_code != 401

    r = post_detect(client, headers={"X-API-Key": token})
    assert r.status_code in (200, 503)
    assert r.status_code != 401


def test_detect_accepts_env_api_keys(keyfile, monkeypatch):
    monkeypatch.setenv("API_KEYS", "legacy-key")
    get_settings.cache_clear()
    client = TestClient(app)

    assert post_detect(client).status_code == 401
    r = post_detect(client, headers={"X-API-Key": "legacy-key"})
    assert r.status_code in (200, 503)
    assert r.status_code != 401
