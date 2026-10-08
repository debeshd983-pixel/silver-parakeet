"""Local, in-process inference. No server, no network, no API key.

    import sahu65
    result = sahu65.detect("photo.jpg")
    print(result.verdict, result.ai_probability)
"""
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .config import get_settings, is_calibrated, load_fusion, load_thresholds
from .core.limits import InvalidImageError, UnsupportedTypeError, sniff_image_type
from .services.clip_probe import ClipProbeService
from .services.ensemble import compute_confidence, compute_verdict, fuse_signals
from .services.preprocess import decode_image, preprocess_for_clip
from .services.provenance import extract_c2pa_provenance

__all__ = ["Detection", "LocalDetector", "detect", "detect_bytes", "warm"]


@dataclass
class Detection:
    """Normalised detection result, identical whether local or via the hosted API."""

    verdict: str
    ai_probability: float
    confidence: str
    warnings: List[str] = field(default_factory=list)
    # Probability from the raw probe, before Platt calibration and thresholding. Renamed
    # from `classifier_probability`: the model that produces it is a CLIP probe, and the
    # old name described a classifier that is no longer shipped.
    detector_probability: Optional[float] = None
    c2pa_present: Optional[bool] = None
    c2pa_ai_declared: Optional[bool] = None
    calibrated: bool = False
    latency_ms: float = 0.0
    # Plain-language narration of the verdict above, when a provider is configured.
    # Purely additive: it never changes any other field on this object.
    explanation: Optional[str] = None
    explanation_provider: Optional[str] = None

    @property
    def is_ai(self) -> bool:
        return self.verdict in ("likely_ai", "ai_generated_verified")

    @property
    def is_real(self) -> bool:
        return self.verdict == "likely_real"

    @property
    def is_inconclusive(self) -> bool:
        return self.verdict == "inconclusive"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "ai_probability": self.ai_probability,
            "confidence": self.confidence,
            "warnings": list(self.warnings),
            "detector_probability": self.detector_probability,
            "c2pa_present": self.c2pa_present,
            "c2pa_ai_declared": self.c2pa_ai_declared,
            "calibrated": self.calibrated,
            "latency_ms": self.latency_ms,
            "explanation": self.explanation,
            "explanation_provider": self.explanation_provider,
        }

    @classmethod
    def from_api(cls, payload: Dict[str, Any]) -> "Detection":
        sig = payload.get("signals") or {}
        c2pa = sig.get("c2pa") or {}
        det = sig.get("detector") or sig.get("classifier") or {}
        return cls(
            verdict=payload["verdict"],
            ai_probability=float(payload["ai_probability"]),
            confidence=payload["confidence"],
            warnings=list(payload.get("warnings") or []),
            detector_probability=det.get("probability"),
            c2pa_present=c2pa.get("present"),
            c2pa_ai_declared=c2pa.get("ai_declared"),
            calibrated=bool(payload.get("calibrated", False)),
            latency_ms=float(payload.get("latency_ms", 0.0)),
            explanation=payload.get("explanation"),
            explanation_provider=payload.get("explanation_provider"),
        )


class LocalDetector:
    """Loads the bundled ONNX weights once and serves repeated inference calls.

    Model load takes roughly 8 seconds (ONNX weight deserialisation). Call :meth:`warm`
    ahead of time if you care about first-call latency.
    """

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self._detector: Optional[ClipProbeService] = None
        self._lock = threading.Lock()

    def warm(self) -> "LocalDetector":
        """Forces weight loading. Raises if the checkpoint is missing or unreadable."""
        self._ensure()
        return self

    def _ensure(self) -> ClipProbeService:
        if self._detector is None:
            with self._lock:
                if self._detector is None:
                    svc = ClipProbeService(self.settings)
                    svc.load()   # raises - never degrades to a stub
                    svc.warmup()
                    self._detector = svc
        return self._detector

    def detect_bytes(self, data: bytes, explain_result: bool = False,
                      explain_provider: Optional[str] = None) -> Detection:
        import time

        # Load first, outside the timing window, so latency_ms reports inference only.
        # Call warm() to move this ~8s cost off your first request entirely.
        svc = self._ensure()
        start = time.time()

        fmt = sniff_image_type(data)
        if fmt is None:
            raise UnsupportedTypeError("Unsupported image format")
        img, warnings = decode_image(data)

        settings = self.settings
        c2pa = extract_c2pa_provenance(data, enabled=settings.enable_c2pa)

        # Aspect-preserving preprocessing, matching what the probe head was fitted on.
        tensor = preprocess_for_clip(img)
        p_raw = svc.predict(tensor)

        fusion = load_fusion(settings.config_dir)
        th = load_thresholds(settings.config_dir)
        t_lo = float(th.get("T_lo", 0.35))
        t_hi = float(th.get("T_hi", 0.65))

        # w1 names the retired classifier and is 0; the probe weight is w2.
        p_fused = fuse_signals(
            p_raw, p_raw,
            w1=float(fusion.get("w1", 0.0)),
            w2=float(fusion.get("w2", 1.0)),
            b=float(fusion.get("b", 0.0)),
        )

        verdict, final_prob = compute_verdict(
            p_fused, t_lo, t_hi, bool(c2pa.ai_declared)
        )

        # Optional UNVALIDATED out-of-domain gate. Never suppresses a C2PA declaration.
        out_of_domain = False
        if settings.document_gate and verdict == "likely_ai":
            from .services.domain import assess_domain

            if assess_domain(svc, img, final_prob).out_of_domain:
                warnings.append("out_of_domain_flat_text_heavy")
                verdict = "inconclusive"
                out_of_domain = True

        calibrated = is_calibrated(settings.config_dir)
        if not calibrated:
            warnings.append("thresholds_unfitted")

        conf = compute_confidence(
            p_ai=final_prob, p_cls=p_raw, p_clip=p_raw, verdict=verdict,
            t_lo=t_lo, t_hi=t_hi, warnings=warnings,
        )

        # Narration happens last, strictly after the verdict is final, and cannot feed
        # back into it. Failures degrade to None rather than failing the detection.
        explanation = None
        explanation_provider = None
        if explain_result:
            from .services.explain import ExplanationRequest, explain as _explain

            result = _explain(
                ExplanationRequest(
                    verdict=verdict,
                    ai_probability=float(final_prob),
                    confidence=conf,
                    calibrated=calibrated,
                    warnings=list(warnings),
                    signals={
                        "c2pa_present": c2pa.present,
                        "c2pa_ai_declared": c2pa.ai_declared,
                        "detector_probability": round(float(p_raw), 4),
                        "detector_model": svc.model_id,
                        "probe_weight": round(float(fusion.get("w2", 1.0)), 6),
                    },
                    thresholds={"T_lo": t_lo, "T_hi": t_hi},
                    model_version=self.settings.model_version
                    if hasattr(self.settings, "model_version") else "",
                    out_of_domain=out_of_domain,
                ),
                provider=explain_provider,
            )
            if result:
                explanation = result["text"]
                explanation_provider = result["provider"]

        return Detection(
            verdict=verdict,
            ai_probability=round(float(final_prob), 4),
            confidence=conf,
            warnings=warnings,
            detector_probability=round(float(p_raw), 4),
            c2pa_present=c2pa.present,
            c2pa_ai_declared=c2pa.ai_declared,
            calibrated=calibrated,
            latency_ms=round((time.time() - start) * 1000.0, 2),
            explanation=explanation,
            explanation_provider=explanation_provider,
        )

    def detect(self, source, explain_result: bool = False,
               explain_provider: Optional[str] = None) -> Detection:
        """Accepts a path, bytes, or a file-like object."""
        if isinstance(source, (bytes, bytearray)):
            return self.detect_bytes(bytes(source), explain_result, explain_provider)
        if hasattr(source, "read"):
            return self.detect_bytes(source.read(), explain_result, explain_provider)
        path = Path(source)
        if not path.is_file():
            raise InvalidImageError(f"No such file: {path}")
        return self.detect_bytes(path.read_bytes(), explain_result, explain_provider)


_DETECTOR: Optional[LocalDetector] = None


def _singleton() -> LocalDetector:
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = LocalDetector()
    return _DETECTOR


def warm() -> LocalDetector:
    """Pre-loads weights so the first :func:`detect` call is not slow."""
    return _singleton().warm()


def detect(source, explain_result: bool = False,
           explain_provider: Optional[str] = None) -> Detection:
    """Detect on a path, bytes, or a file-like object using the bundled model."""
    return _singleton().detect(source, explain_result, explain_provider)


def detect_bytes(data: bytes, explain_result: bool = False,
                 explain_provider: Optional[str] = None) -> Detection:
    """Detect on raw image bytes using the bundled model."""
    return _singleton().detect_bytes(data, explain_result, explain_provider)


def model_info() -> Dict[str, Any]:
    """Introspection for the bundled checkpoint."""
    return _singleton()._ensure().describe()


# Silence unused-import warnings for names re-exported for convenience.
_ = (np, os)