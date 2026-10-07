"""Domain-suitability assessment.

**STATUS: UNVALIDATED HEURISTIC. OPT-IN, DEFAULT DISABLED.**

The S2 classifier is a natural-image diffusion detector. Measured on this repository's
assets it has *negative* discriminative power on document images (AUC 0.375, worse than
chance), and no alternative signal tried separates them either: four preprocessing
pipelines, 9-tile inference with five aggregation statistics, NPR (neighbouring pixel
relationships, training-free), colour/saturation/white statistics, and an independent
SigLIP detector (AUC 0.479 vs natural photographs). Document images are an unresolved
research problem as of 2026.

The predicate below is a *fitted* heuristic. Its thresholds were chosen by inspecting 18
images (4 AI documents, 2 real documents, 12 natural photographs) and are therefore
almost certainly over-fitted. It MUST be validated on a real document corpus before being
trusted. It exists so that a caller who prefers a safe "unknown" to a confident false
accusation can opt into one.

Measured behaviour of the default thresholds on that same 18-image sample, for reference
only - not an accuracy claim:

    AI documents            3/4 retained as likely_ai, 1 suppressed to inconclusive
    real documents          0/2 correct today -> 2/2 would become inconclusive
    natural photographs     12/12 pass through (including the one false positive)
"""
from dataclasses import dataclass
from typing import List

import numpy as np
from PIL import Image

# Fitted thresholds - see module docstring. Not validated on held-out data.
DEFAULT_FULL_PAGE_MIN = 0.99
DEFAULT_TILE_MEAN_MIN = 0.30
TILE_GRID = 3
TILE_FRACTION = 0.60
TILE_SIZE = 224


@dataclass
class DomainAssessment:
    """Result of the out-of-domain check."""

    out_of_domain: bool
    full_page_score: float
    tile_mean: float
    tile_scores: List[float]


def _normalise(img: Image.Image) -> np.ndarray:
    """ImageNet normalisation matching sahu65.services.preprocess."""
    from sahu65.services.preprocess import IMAGENET_MEAN, IMAGENET_STD

    a = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    a = np.transpose(a, (2, 0, 1))[np.newaxis, ...]
    return ((a - IMAGENET_MEAN) / IMAGENET_STD).astype(np.float32)


def tile_tensors(img: Image.Image) -> List[np.ndarray]:
    """Aspect-preserving overlapping tiles, each scaled to TILE_SIZE.

    A full-page squash to 224x224 destroys glyph detail; tiles keep it. The spread
    between tiles is the only intra-image disagreement signal available here.
    """
    w, h = img.size
    tw, th = int(w * TILE_FRACTION / TILE_GRID), int(h * TILE_FRACTION / TILE_GRID)
    if tw < 8 or th < 8:
        return []
    out = []
    for r in range(TILE_GRID):
        for c in range(TILE_GRID):
            left = int((w - tw) * c / (TILE_GRID - 1))
            top = int((h - th) * r / (TILE_GRID - 1))
            crop = img.crop((left, top, left + tw, top + th))
            out.append(_normalise(crop.resize((TILE_SIZE, TILE_SIZE), Image.Resampling.BICUBIC)))
    return out


def assess_domain(
    classifier,
    img: Image.Image,
    full_page_score: float,
    full_page_min: float = DEFAULT_FULL_PAGE_MIN,
    tile_mean_min: float = DEFAULT_TILE_MEAN_MIN,
) -> DomainAssessment:
    """Flags images where the classifier is confidently high but internally inconsistent.

    Rationale: a saturated full-page score combined with high tile-to-tile disagreement is
    the signature of a flat, text-heavy, out-of-distribution input rather than a coherent
    natural scene. On such input the classifier's confidence is not trustworthy, so the
    caller may prefer to abstain.

    ``classifier`` must expose ``predict(tensor) -> float``.
    """
    if full_page_score < full_page_min:
        return DomainAssessment(False, full_page_score, 0.0, [])

    scores = [classifier.predict(t) for t in tile_tensors(img)]
    if not scores:
        return DomainAssessment(False, full_page_score, 0.0, [])

    mean = float(np.mean(scores))
    return DomainAssessment(mean >= tile_mean_min, full_page_score, mean, scores)
