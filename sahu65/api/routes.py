"""API route definitions: /deep-guard/detect, /healthz, /readyz, /version (architecture.md section 4)."""
import time
import uuid
from typing import Optional
from fastapi import APIRouter, File, Header, Request, UploadFile
from fastapi.responses import JSONResponse

from sahu65.config import get_settings
from sahu65.core.limits import (
    RateLimitedError,
    UnauthorizedError,
    inspect_image_pixels,
    rate_limiter,
    read_and_validate_file,
)
from sahu65.core.logging import logger
import sahu65.main as main_module
from sahu65.schemas import (
    C2PASignal,
    DetectionResponse,
    ModelSignal,
    SignalsResponse,
    VersionResponse,
)
from sahu65.services.ensemble import (
    compute_confidence,
    compute_verdict,
    fuse_signals,
)
from sahu65.services.preprocess import (
    decode_image,
    preprocess_for_classifier,
    preprocess_for_clip,
)
from sahu65.services.domain import assess_domain
from sahu65.services.provenance import extract_c2pa_provenance

API_PREFIX = "/deep-guard"

router = APIRouter(prefix=API_PREFIX)


@router.get("/healthz")
async def healthz() -> dict:
    """Liveness probe: cheap, always returns 200 ok."""
    return {"status": "ok"}


@router.get("/readyz")
async def readyz():
    """Readiness probe: 200 only once the real model is loaded and warmed.

    This is the probe to gate traffic on — the process binds its socket immediately
    and loads ~8s of ONNX weights in the background, so /deep-guard/healthz alone will
    report "ok" long before the service can actually serve a detection.
    """
    if main_module.LOAD_ERROR:
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "code": "model_load_failed",
                    "message": main_module.LOAD_ERROR,
                }
            },
        )
    if not main_module.READY:
        return JSONResponse(
            status_code=503,
            content={"error": {"code": "model_not_ready", "message": "Models still initializing"}},
        )
    return {"status": "ready"}


@router.get("/version", response_model=VersionResponse)
async def version():
    """Returns model version metadata, benchmark ID, thresholds, and fusion parameters."""
    return VersionResponse(
        model_version="2026.10.0",
        benchmark_id=main_module.THRESHOLDS.get("benchmark_id"),
        thresholds={
            "T_lo": main_module.THRESHOLDS.get("T_lo", 0.35),
            "T_hi": main_module.THRESHOLDS.get("T_hi", 0.65),
        },
        fusion={
            "w1": main_module.FUSION_CFG.get("w1", 1.0),
            "w2": main_module.FUSION_CFG.get("w2", 1.0),
            "b": main_module.FUSION_CFG.get("b", 0.0),
        },
        models={
            "classifier": getattr(main_module.CLASSIFIER, "model_id", "unloaded"),
            "clip_probe": getattr(main_module.CLIP_PROBE, "model_id", "unloaded"),
        },
        calibrated=main_module.THRESHOLDS.get("fitted_on") is not None,
    )


@router.post("/detect", response_model=DetectionResponse)
async def detect(
    request: Request,
    file: UploadFile = File(...),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
):
    """Detects whether an uploaded image is AI-generated with calibrated probability and explicit abstain band."""
    start_time = time.time()
    req_id = uuid.uuid4().hex[:12]
    settings = get_settings()

    # 1. Check readiness
    if not main_module.READY:
        return JSONResponse(
            status_code=503,
            content={"error": {"code": "model_not_ready", "message": "Models are loading"}},
        )

    # 2. Check API Key authentication if configured
    if settings.api_keys:
        allowed_keys = {k.strip() for k in settings.api_keys.split(",") if k.strip()}
        if not x_api_key or x_api_key not in allowed_keys:
            raise UnauthorizedError()

    # 3. Rate limiting check
    client_ip = request.client.host if request.client else "unknown"
    rate_key = x_api_key or client_ip
    if not rate_limiter.check(rate_key):
        raise RateLimitedError(f"Rate limit of {rate_limiter.describe()} exceeded")

    # 4. Stream and validate file upload
    image_bytes, detected_fmt = await read_and_validate_file(file, settings)

    # 5. Pixel decompression bomb check
    inspect_image_pixels(image_bytes, settings)

    # 6. Decode image & collect warnings
    img, warnings = decode_image(image_bytes)

    # The shipped thresholds are unfitted placeholders, so the probability is not a
    # calibrated posterior. Say so on every response rather than implying it is.
    if main_module.THRESHOLDS.get("fitted_on") is None:
        warnings.append("thresholds_unfitted")

    # 7. S1: Provenance extraction (C2PA)
    c2pa_res = extract_c2pa_provenance(image_bytes, enabled=settings.enable_c2pa)

    # 8. Preprocessing
    s2_tensor = preprocess_for_classifier(img)
    s3_tensor = preprocess_for_clip(img, enable_tta=settings.enable_tta)

    # 9. Concurrency-managed model inference
    runtime = main_module.RUNTIME_MGR
    p_cls = await runtime.run_in_pool(main_module.CLASSIFIER.predict, s2_tensor)
    p_clip = await runtime.run_in_pool(main_module.CLIP_PROBE.predict, s3_tensor)

    # 10. Ensemble fusion in logit space
    w1 = float(main_module.FUSION_CFG.get("w1", 1.0))
    w2 = float(main_module.FUSION_CFG.get("w2", 1.0))
    b = float(main_module.FUSION_CFG.get("b", 0.0))
    p_fused = fuse_signals(p_cls, p_clip, w1, w2, b)

    # 11. Verdict policy
    t_lo = float(main_module.THRESHOLDS.get("T_lo", 0.35))
    t_hi = float(main_module.THRESHOLDS.get("T_hi", 0.65))
    verdict, final_prob = compute_verdict(
        p_ai=p_fused,
        t_lo=t_lo,
        t_hi=t_hi,
        c2pa_ai_declared=bool(c2pa_res.ai_declared),
    )

    # 12. Optional out-of-domain gate (UNVALIDATED, opt-in via DOCUMENT_GATE).
    # Runs after the C2PA check so a declared manifest is never suppressed.
    out_of_domain = False
    if settings.document_gate and verdict != "ai_generated_verified":
        dom = await runtime.run_in_pool(
            assess_domain, main_module.CLASSIFIER, img, final_prob
        )
        if dom.out_of_domain:
            out_of_domain = True
            warnings.append("out_of_domain_flat_text_heavy")
            verdict = "inconclusive"

    # 13. Confidence rating
    conf = compute_confidence(
        p_ai=final_prob,
        p_cls=p_cls,
        p_clip=p_clip,
        verdict=verdict,
        t_lo=t_lo,
        t_hi=t_hi,
        warnings=warnings,
    )

    latency_ms = round((time.time() - start_time) * 1000.0, 2)

    logger.info(
        "Detection executed",
        extra={
            "request_id": req_id,
            "verdict": verdict,
            "ai_probability": round(final_prob, 4),
            "confidence": conf,
            "latency_ms": latency_ms,
            "warnings": warnings,
        },
    )

    return DetectionResponse(
        request_id=req_id,
        verdict=verdict,
        ai_probability=round(final_prob, 4),
        confidence=conf,
        signals=SignalsResponse(
            c2pa=C2PASignal(present=c2pa_res.present, ai_declared=c2pa_res.ai_declared),
            classifier=ModelSignal(
                model=getattr(main_module.CLASSIFIER, "model_id", "unloaded"),
                probability=round(p_cls, 4),
            ),
            clip_probe=ModelSignal(
                model=getattr(main_module.CLIP_PROBE, "model_id", "unloaded"),
                probability=round(p_clip, 4),
            ),
        ),
        warnings=warnings,
        out_of_domain=out_of_domain,
        model_version="2026.10.0",
        benchmark_id=main_module.THRESHOLDS.get("benchmark_id"),
        latency_ms=latency_ms,
    )
