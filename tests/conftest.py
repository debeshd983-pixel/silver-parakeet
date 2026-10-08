"""Shared pytest configuration.

Isolates the Bear Token keyfile. Without this the suite reads the developer's real
``~/.sahu65/keys.json``, so any machine where a token has been minted turns every
unauthenticated ``/detect`` test into a 401. Tests that assert on auth behaviour mint
their own tokens explicitly.
"""
import os
import tempfile
from pathlib import Path

_EMPTY_KEYFILE = Path(tempfile.gettempdir()) / "sahu65-test-empty-keys.json"

# Must be set before sahu65.config/keys are imported anywhere.
os.environ["SAHU65_KEYFILE"] = str(_EMPTY_KEYFILE)

if not _EMPTY_KEYFILE.exists():
    _EMPTY_KEYFILE.write_text('{"version": 1, "keys": {}}', encoding="utf-8")

# Auth is exercised explicitly by tests/test_keys.py, so the API tests run unauthenticated.
os.environ.pop("API_KEYS", None)