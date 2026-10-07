"""Application configuration via environment variables (architecture.md section 8)."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Artifacts ship inside the package so an installed wheel works from any working
# directory with no download step and no cold-start stall.
PACKAGE_DIR = Path(__file__).resolve().parent
BUNDLED_MODELS = str(PACKAGE_DIR / "models")
BUNDLED_CONFIG = str(PACKAGE_DIR / "config")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Limits / validation
    max_upload_mb: int = 10
    max_pixels: int = 50_000_000
    request_timeout_s: float = 15.0

    # Inference runtime
    max_concurrent_inferences: int = 8  # default = vCPU count
    ort_intra_threads: int = 8
    enable_tta: bool = False

    # Signals
    enable_c2pa: bool = True

    # Startup safety. When true (default) the process refuses to serve unless the
    # real S2 weights loaded; a missing model is a hard failure, not a silent 0.50.
    require_real_model: bool = True

    # UNVALIDATED opt-in heuristic. Forces `inconclusive` on images the classifier scores
    # high but scores inconsistently across tiles (flat/text-heavy out-of-distribution
    # input). Thresholds were fitted by inspecting 18 images and are NOT validated.
    # Costs some true positives on AI documents. See sahu65/services/domain.py.
    document_gate: bool = False

    # Serving
    api_keys: str = ""  # comma-separated; empty = open
    rate_limit: str = "30/minute"
    allowed_origins: str = ""
    enable_docs: bool = False
    enable_metrics: bool = False
    log_level: str = "INFO"

    # Artifacts. Defaults point at the copy bundled inside the installed package;
    # override with MODEL_DIR / CONFIG_DIR to use your own checkpoint or tuning.
    model_dir: str = BUNDLED_MODELS
    config_dir: str = BUNDLED_CONFIG


@lru_cache
def get_settings() -> Settings:
    return Settings()


DEFAULT_THRESHOLDS = {"T_lo": 0.35, "T_hi": 0.65, "benchmark_id": None, "fitted_on": None}
DEFAULT_FUSION = {"w1": 1.0, "w2": 0.0, "b": 0.0, "benchmark_id": None, "fitted_on": None}


@lru_cache(maxsize=8)
def _read_json(path: str):
    import json

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


@lru_cache(maxsize=8)
def load_thresholds(config_dir: str) -> dict:
    """Reads thresholds.json, falling back to the shipped defaults.

    Shared by the server and the local SDK so both apply identical policy.
    """
    return {**DEFAULT_THRESHOLDS, **(_read_json(str(Path(config_dir) / "thresholds.json")) or {})}


@lru_cache(maxsize=8)
def load_fusion(config_dir: str) -> dict:
    """Reads fusion.json, falling back to the shipped defaults."""
    return {**DEFAULT_FUSION, **(_read_json(str(Path(config_dir) / "fusion.json")) or {})}


def is_calibrated(config_dir: str) -> bool:
    """True only once thresholds have actually been fitted on data."""
    return load_thresholds(config_dir).get("fitted_on") is not None
