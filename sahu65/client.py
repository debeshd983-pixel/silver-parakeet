"""Client for a hosted sahu65 deployment (API-key authenticated).

    from sahu65 import Client
    c = Client(api_key="sk-...", base_url="https://your-host")
    result = c.detect("photo.jpg")
"""
import os
from typing import Any, Dict, Optional

import httpx

from .local import Detection

__all__ = ["Client", "Sahu65Error", "AuthError", "NotReadyError"]


class Sahu65Error(Exception):
    """Base error for API failures."""

    def __init__(self, message: str, status_code: Optional[int] = None, code: str = ""):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code


class AuthError(Sahu65Error):
    """401/403 - missing or invalid API key."""


class NotReadyError(Sahu65Error):
    """503 - the server is still loading weights, or failed to load them."""


class Client:
    """Thin synchronous client for POST /deep-guard/detect.

    Set ``SAHU65_API_KEY`` and ``SAHU65_BASE_URL`` in the environment instead of
    passing them explicitly.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.api_key = api_key or os.environ.get("SAHU65_API_KEY", "")
        self.base_url = (base_url or os.environ.get("SAHU65_BASE_URL", "")).rstrip("/")
        self.timeout = timeout
        if not self.base_url:
            raise ValueError(
                "base_url is required (pass it or set SAHU65_BASE_URL)"
            )

    def _headers(self) -> Dict[str, str]:
        h = {}
        if self.api_key:
            h["X-API-Key"] = self.api_key
        return h

    def _raise_for_status(self, resp: httpx.Response) -> None:
        if resp.status_code < 400:
            return
        code, message = "", resp.text
        try:
            err = resp.json().get("error", {})
            code = err.get("code", "")
            message = err.get("message", message)
        except Exception:
            pass
        if resp.status_code in (401, 403):
            raise AuthError(message, resp.status_code, code)
        if resp.status_code == 503:
            raise NotReadyError(message, resp.status_code, code)
        raise Sahu65Error(message, resp.status_code, code)

    def detect(self, source) -> Detection:
        """Detect on a path, bytes, or a file-like object."""
        if isinstance(source, (bytes, bytearray)):
            payload = bytes(source)
            filename = "image.jpg"
        elif hasattr(source, "read"):
            payload = source.read()
            filename = getattr(source, "name", "image.jpg")
        else:
            with open(source, "rb") as f:
                payload = f.read()
            filename = os.path.basename(source)

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                f"{self.base_url}/deep-guard/detect",
                headers=self._headers(),
                files={"file": (filename, payload, "application/octet-stream")},
            )
        self._raise_for_status(resp)
        return Detection.from_api(resp.json())

    def version(self) -> Dict[str, Any]:
        """Returns model IDs, thresholds, fusion weights and calibration state."""
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(f"{self.base_url}/deep-guard/version", headers=self._headers())
        self._raise_for_status(resp)
        return resp.json()

    def health(self) -> bool:
        """True when the service reports ready.

        Returns False rather than raising if the server is unreachable or still
        starting up - a health check should be safe to call in a polling loop.
        """
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.get(f"{self.base_url}/deep-guard/readyz", headers=self._headers())
            return resp.status_code == 200
        except Exception:
            return False

    def wait_until_ready(self, timeout_s: float = 180.0, poll_s: float = 2.0) -> bool:
        """Blocks until /deep-guard/readyz reports ready, or the timeout elapses.

        Useful right after a cold start, since the server binds its socket before the
        ~8s of ONNX weight loading completes.
        """
        import time

        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                if self.health():
                    return True
            except Exception:
                pass
            time.sleep(poll_s)
        return False