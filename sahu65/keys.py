"""Bear Tokens: named API keys that guard POST /deep-guard/detect.

    sahu65 --key myapp          # mint a token named "myapp" - printed once
    sahu65 --key-myapp          # sugar for the same command
    sahu65 --list-keys          # list names and dates, never the tokens

A minted token looks like ``bear_myapp_<random>`` and is accepted by the server
in either header:

    Authorization: Bearer bear_myapp_<random>    # preferred
    X-API-Key:      bear_myapp_<random>          # original, kept for compatibility

Storage and behaviour:

  * Only the SHA-256 digest of a token is persisted, in ``~/.sahu65/keys.json``
    (override with ``SAHU65_KEYFILE``), so a leaked key file cannot be replayed.
  * The plaintext token is shown exactly once, at mint time.
  * The file is re-read whenever its mtime changes, so a token minted while the
    server is running takes effect without a restart.
  * Re-running ``sahu65 --key <name>`` rotates the token: the digest is replaced
    and the previous token stops working.
  * ``API_KEYS`` (comma-separated plaintext - the pre-Bear-Token config) keeps
    working and is accepted alongside file-backed tokens. Authentication is only
    enforced once at least one key exists from either source; with none, the
    endpoint is open, exactly as before.
"""
import hashlib
import json
import os
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

TOKEN_PREFIX = "bear"
DEFAULT_KEYFILE = Path.home() / ".sahu65" / "keys.json"
_MAX_NAME_LEN = 32
_SLUG_ILLEGAL = re.compile(r"[^A-Za-z0-9._-]+")

# path -> (mtime_ns, frozenset of digests). A single entry; stat is cheaper than
# re-reading a JSON file on every request, and mtime gives us correctness for free.
_cache: Dict[str, Tuple[int, frozenset]] = {}


def keyfile_path() -> Path:
    """Where tokens are stored (digests only). SAHU65_KEYFILE overrides the default."""
    env = os.environ.get("SAHU65_KEYFILE", "").strip()
    return Path(env).expanduser() if env else DEFAULT_KEYFILE


def _slug(name: str) -> str:
    slug = _SLUG_ILLEGAL.sub("-", name.strip()).strip("-._")
    if not slug:
        raise ValueError(
            f"key name {name!r} has no usable characters; letters, digits, '.', '_', '-' allowed"
        )
    return slug[:_MAX_NAME_LEN]


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _read_store(path: Path) -> Dict[str, dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # Missing file or a half-written one: treat as empty rather than crash a
        # request. Writes are atomic (os.replace), so this is only ever transient.
        return {}
    keys = data.get("keys")
    return keys if isinstance(keys, dict) else {}


def _write_store(path: Path, store: Dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Best effort; a no-op on Windows. Keeps the directory listable only by owner.
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps({"version": 1, "keys": store}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    os.replace(tmp, path)
    _cache.pop(str(path), None)


def create_key(name: str) -> Tuple[str, bool]:
    """Mint a token for ``name``. Returns (token, rotated).

    ``rotated`` is True when a token with this name already existed and has been
    invalidated by this call.
    """
    slug = _slug(name)
    path = keyfile_path()
    store = _read_store(path)
    rotated = slug in store
    token = f"{TOKEN_PREFIX}_{slug}_{secrets.token_urlsafe(24)}"
    store[slug] = {
        "sha256": _digest(token),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _write_store(path, store)
    return token, rotated


def list_keys() -> List[Dict[str, str]]:
    """Names and creation dates of minted tokens. Never the tokens themselves."""
    store = _read_store(keyfile_path())
    return [
        {"name": name, "created": str(record.get("created", ""))}
        for name, record in sorted(store.items())
    ]


def file_digests() -> Set[str]:
    path = keyfile_path()
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        return set()
    cached = _cache.get(str(path))
    if cached is not None and cached[0] == mtime:
        return set(cached[1])
    digests = frozenset(
        record.get("sha256")
        for record in _read_store(path).values()
        if isinstance(record, dict) and record.get("sha256")
    )
    _cache[str(path)] = (mtime, digests)
    return set(digests)


def env_digests(api_keys: str) -> Set[str]:
    return {_digest(k.strip()) for k in (api_keys or "").split(",") if k.strip()}


def auth_configured(api_keys: str = "") -> bool:
    """True when at least one key exists (file-backed or API_KEYS); i.e. auth must run."""
    return bool(api_keys.strip()) or bool(file_digests())


def verify(token: Optional[str], api_keys: str = "") -> bool:
    """True when ``token`` matches a minted or configured key. Never raises."""
    if not token:
        return False
    digest = _digest(token)
    return digest in file_digests() or digest in env_digests(api_keys)


def describe(api_keys: str = "") -> str:
    """Human-readable auth state for the startup log. Contains no secrets."""
    n_file = len(list_keys())
    n_env = len([k for k in (api_keys or "").split(",") if k.strip()])
    if not n_file and not n_env:
        return "off - no Bear Tokens, POST /deep-guard/detect is open"
    parts = []
    if n_file:
        parts.append(f"{n_file} in {keyfile_path()}")
    if n_env:
        parts.append(f"{n_env} from API_KEYS")
    return "on - " + ", ".join(parts)
