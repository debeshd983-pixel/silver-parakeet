"""Provenance extraction via C2PA manifests (Signal S1)."""
import logging
from typing import Optional
from pydantic import BaseModel

logger = logging.getLogger("detector")


class ProvenanceResult(BaseModel):
    present: bool = False
    ai_declared: Optional[bool] = None
    generator_name: Optional[str] = None


def extract_c2pa_provenance(image_bytes: bytes, enabled: bool = True) -> ProvenanceResult:
    """Extracts C2PA provenance manifests from raw image bytes.

    If a manifest is present and explicitly declares generative AI (trained-algorithmic media),
    ai_declared is True. If present but not AI, ai_declared is False.
    If no manifest is present or C2PA is disabled/unavailable, present is False and ai_declared is None.
    """
    if not enabled:
        return ProvenanceResult(present=False, ai_declared=None)

    try:
        import c2pa

        # Use Reader from c2pa if available
        reader = c2pa.Reader.from_stream(image_bytes)
        manifest_json = reader.json()
        if not manifest_json:
            return ProvenanceResult(present=False, ai_declared=None)

        manifest_str = str(manifest_json).lower()
        is_ai = False
        generator = None

        ai_indicators = [
            "trainedalgorithmicmedia",
            "c2pa.trainedalgorithmicmedia",
            "digitalSourceType/trainedAlgorithmicMedia",
            "c2pa.action.generated",
            "dall-e",
            "midjourney",
            "stable diffusion",
            "firefly",
        ]

        for indicator in ai_indicators:
            if indicator.lower() in manifest_str:
                is_ai = True
                generator = indicator
                break

        return ProvenanceResult(
            present=True,
            ai_declared=is_ai,
            generator_name=generator,
        )

    except ImportError:
        logger.debug("c2pa-python library not installed; skipping C2PA extraction.")
        return ProvenanceResult(present=False, ai_declared=None)
    except Exception as e:
        logger.debug(f"C2PA extraction encountered no manifest or error: {e}")
        return ProvenanceResult(present=False, ai_declared=None)
