"""Unit tests for ensemble fusion math, verdict policies, and confidence calculation."""
import pytest
from app.services.ensemble import (
    clip_prob,
    compute_confidence,
    compute_verdict,
    fuse_signals,
    logit,
    sigmoid,
)


def test_clip_prob_bounds():
    assert clip_prob(0.0) == 1e-4
    assert clip_prob(1.0) == 1.0 - 1e-4
    assert clip_prob(0.5) == 0.5


def test_logit_sigmoid_roundtrip():
    for p in [0.1, 0.25, 0.5, 0.75, 0.9]:
        z = logit(p)
        assert abs(sigmoid(z) - p) < 1e-4


def test_fuse_signals_symmetry():
    # If p_cls and p_clip are 0.5 and bias is 0, fused probability must be 0.5
    fused = fuse_signals(0.5, 0.5, w1=1.0, w2=1.0, b=0.0)
    assert abs(fused - 0.5) < 1e-4

    # Both high -> high output
    assert fuse_signals(0.9, 0.9, 1.0, 1.0, 0.0) > 0.9
    # Both low -> low output
    assert fuse_signals(0.1, 0.1, 1.0, 1.0, 0.0) < 0.1


def test_compute_verdict_c2pa_override():
    verdict, prob = compute_verdict(p_ai=0.20, t_lo=0.35, t_hi=0.65, c2pa_ai_declared=True)
    assert verdict == "ai_generated_verified"
    assert prob == 0.99


def test_compute_verdict_thresholds():
    t_lo, t_hi = 0.35, 0.65

    # Likely real
    v_real, p_r = compute_verdict(0.20, t_lo, t_hi)
    assert v_real == "likely_real"
    assert p_r == 0.20

    # Inconclusive band
    v_inc, p_i = compute_verdict(0.50, t_lo, t_hi)
    assert v_inc == "inconclusive"
    assert p_i == 0.50

    # Likely AI
    v_ai, p_a = compute_verdict(0.85, t_lo, t_hi)
    assert v_ai == "likely_ai"
    assert p_a == 0.85


def test_confidence_calculation():
    # High confidence: agreeing signals and large margin
    conf = compute_confidence(
        p_ai=0.90, p_cls=0.88, p_clip=0.85, verdict="likely_ai", t_lo=0.35, t_hi=0.65
    )
    assert conf == "high"

    # Inconclusive is always low confidence
    assert compute_confidence(0.50, 0.45, 0.55, "inconclusive", 0.35, 0.65) == "low"

    # Downgrade with low resolution warning
    conf_warn = compute_confidence(
        p_ai=0.90, p_cls=0.88, p_clip=0.85, verdict="likely_ai", t_lo=0.35, t_hi=0.65, warnings=["low_resolution"]
    )
    assert conf_warn == "medium"
