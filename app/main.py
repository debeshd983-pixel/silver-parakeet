"""FastAPI application factory, lifespan management, model loading, and uniform error handlers."""
import json
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import Settings, get_settings
from app.core.limits import DetectorError
from app.core.logging import setup_logging
from app.services.classifier import ClassifierService
from app.services.clip_probe import ClipProbeService
from app.services.runtime import RuntimeManager

log = setup_logging()

# Global state
READY = False
START_TIME = time.time()
CLASSIFIER: ClassifierService = None
CLIP_PROBE: ClipProbeService = None
RUNTIME_MGR: RuntimeManager = None
THRESHOLDS: Dict[str, Any] = {"T_lo": 0.35, "T_hi": 0.65, "benchmark_id": None}
FUSION_CFG: Dict[str, Any] = {"w1": 1.0, "w2": 1.0, "b": 0.0, "benchmark_id": None}


def load_configs(settings: Settings):
    """Loads fusion and threshold configs from config dir."""
    global THRESHOLDS, FUSION_CFG
    thresh_file = os.path.join(settings.config_dir, "thresholds.json")
    if os.path.exists(thresh_file):
        try:
            with open(thresh_file, "r", encoding="utf-8") as f:
                THRESHOLDS = json.load(f)
            log.info(f"Loaded thresholds: T_lo={THRESHOLDS.get('T_lo')}, T_hi={THRESHOLDS.get('T_hi')}")
        except Exception as e:
            log.warning(f"Error reading thresholds.json: {e}")

    fusion_file = os.path.join(settings.config_dir, "fusion.json")
    if os.path.exists(fusion_file):
        try:
            with open(fusion_file, "r", encoding="utf-8") as f:
                FUSION_CFG = json.load(f)
            log.info(f"Loaded fusion: w1={FUSION_CFG.get('w1')}, w2={FUSION_CFG.get('w2')}, b={FUSION_CFG.get('b')}")
        except Exception as e:
            log.warning(f"Error reading fusion.json: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager: loads models, executes warm-up inference, sets READY flag."""
    global READY, CLASSIFIER, CLIP_PROBE, RUNTIME_MGR
    settings = get_settings()

    log.info("Initializing detector application...")
    load_configs(settings)

    RUNTIME_MGR = RuntimeManager(settings)
    CLASSIFIER = ClassifierService(settings)
    CLIP_PROBE = ClipProbeService(settings)

    CLASSIFIER.load()
    CLIP_PROBE.load()

    # Warm-up inference
    log.info("Running model warm-up inference...")
    try:
        CLASSIFIER.warmup()
        CLIP_PROBE.warmup()
    except Exception as e:
        log.warning(f"Warm-up inference warning: {e}")

    READY = True
    log.info("Application ready to receive traffic")
    yield

    READY = False
    log.info("Application shutting down")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="AI-Generated Image Detector API",
        version="2026.10.0",
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

    from app.api.routes import router
    app.include_router(router)

    return app


app = create_app()
