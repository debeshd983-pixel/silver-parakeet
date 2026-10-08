"""FastAPI application factory, lifespan management, model loading, and uniform error handlers."""
import asyncio
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from sahu65.config import (
    Settings,
    get_settings,
    is_calibrated,
    load_fusion,
    load_thresholds,
)
from sahu65.core.limits import DetectorError
from sahu65.core.logging import setup_logging
from sahu65.services.clip_probe import ClipProbeService
from sahu65.services.runtime import RuntimeManager

log = setup_logging()

# Global state
READY = False
LOAD_ERROR: str = None
START_TIME = time.time()
# The single shipped detector: a linear probe on a frozen CLIP ViT-B/16 encoder.
# It replaced Organika/sdxl-detector, which measured AUC 0.6434 against this probe's
# 0.8213 on benchmark bench-8e0bd9e97e3e and was cc-by-nc-3.0 (non-commercial).
# Naming stays DETECTOR rather than the old S1/S2/S3 scheme; see MODEL_CARD.md.
DETECTOR: ClipProbeService = None
RUNTIME_MGR: RuntimeManager = None
THRESHOLDS: Dict[str, Any] = {"T_lo": 0.35, "T_hi": 0.65, "benchmark_id": None}
FUSION_CFG: Dict[str, Any] = {"w1": 0.0, "w2": 1.0, "b": 0.0, "benchmark_id": None}


def load_configs(settings: Settings):
    """Loads fusion and threshold configs from config dir (shared with the local SDK)."""
    global THRESHOLDS, FUSION_CFG
    THRESHOLDS = load_thresholds(settings.config_dir)
    FUSION_CFG = load_fusion(settings.config_dir)
    log.info(
        f"Loaded thresholds: T_lo={THRESHOLDS.get('T_lo')}, T_hi={THRESHOLDS.get('T_hi')}, "
        f"calibrated={is_calibrated(settings.config_dir)}"
    )
    log.info(
        f"Loaded fusion: w1={FUSION_CFG.get('w1')}, w2={FUSION_CFG.get('w2')}, b={FUSION_CFG.get('b')}"
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager: loads models in the background, sets READY when usable.

    Model load takes ~8s (ONNX weight deserialization; NOT graph optimization, which was
    measured to make no difference). Binding the socket and serving /healthz immediately
    instead of blocking means no cold-start stall for orchestrators and load balancers.
    /readyz returns 503 until the model is genuinely usable, so traffic is never sent to
    a half-initialised process.
    """
    global READY, DETECTOR, RUNTIME_MGR, LOAD_ERROR
    settings = get_settings()

    log.info("Initializing detector application...")
    load_configs(settings)

    from sahu65.core.limits import rate_limiter
    from sahu65.keys import describe as describe_auth

    rate_limiter.configure(settings.rate_limit)
    log.info(f"Rate limit: {rate_limiter.describe()}")
    log.info(f"API auth: {describe_auth(settings.api_keys)}")

    RUNTIME_MGR = RuntimeManager(settings)
    DETECTOR = ClipProbeService(settings)
    READY = False
    LOAD_ERROR = None

    loader = asyncio.create_task(_load_models_async(settings))

    yield

    loader.cancel()
    try:
        await loader
    except (asyncio.CancelledError, Exception):
        pass

    READY = False
    log.info("Application shutting down")


async def _load_models_async(settings: Settings) -> None:
    """Loads + warms the detector off the event loop. Sets READY or LOAD_ERROR.

    Failures are fatal, not degraded. A missing or unreadable checkpoint used to return a
    constant 0.50, which looked exactly like a working model to a caller (WORKLOG defect
    3). With REQUIRE_REAL_MODEL the service refuses to become ready instead.
    """
    global READY, LOAD_ERROR

    def _blocking() -> str:
        try:
            DETECTOR.load()
        except Exception as e:
            return str(e)
        try:
            DETECTOR.warmup()
        except Exception as e:
            return f"warm-up inference failed: {e}"
        return ""

    error = await asyncio.to_thread(_blocking)

    if error:
        LOAD_ERROR = error
        if settings.require_real_model:
            log.error(f"Fatal: detector failed to load: {error}. /readyz will report 503.")
        else:
            log.warning(f"Detector unavailable, continuing in degraded mode: {error}")
        return

    READY = True
    log.info(
        f"Application ready. detector={DETECTOR.model_id} "
        f"weights={os.path.basename(DETECTOR.model_path)} "
        f"head={os.path.basename(DETECTOR.head_path)}"
    )


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="AI-Generated Image Detector API",
        version="2026.11.0",
        docs_url="/docs" if settings.enable_docs else None,
        redoc_url=None,
        openapi_url="/openapi.json" if settings.enable_docs else None,
        lifespan=lifespan,
    )

    if settings.allowed_origins:
        origins = [o.strip() for o in settings.allowed_origins.split(",") if o.strip()]
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Uniform error handlers conforming to architecture.md section 4
    @app.exception_handler(DetectorError)
    async def detector_error_handler(request: Request, exc: DetectorError):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message}},
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        code = "http_error"
        if exc.status_code == 404:
            code = "not_found"
        elif exc.status_code == 400:
            code = "bad_request"
        elif exc.status_code == 401:
            code = "unauthorized"
        elif exc.status_code == 403:
            code = "forbidden"
        elif exc.status_code == 429:
            code = "rate_limited"
        elif exc.status_code == 503:
            code = "model_not_ready"
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": code, "message": str(exc.detail)}},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "validation_error", "message": "Invalid request parameters"}},
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        log.error(f"Unhandled error: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "internal_error", "message": "Internal server error"}},
        )

    from sahu65.api.routes import router
    app.include_router(router)

    return app


app = create_app()
