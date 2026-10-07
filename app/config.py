"""Application configuration via environment variables (architecture.md section 8)."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


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

    # Serving
    api_keys: str = ""  # comma-separated; empty = open
    rate_limit: str = "30/minute"
    allowed_origins: str = ""
    enable_docs: bool = False
    enable_metrics: bool = False
    log_level: str = "INFO"

    # Artifacts
    model_dir: str = "models"
    config_dir: str = "config"


@lru_cache
def get_settings() -> Settings:
    return Settings()
