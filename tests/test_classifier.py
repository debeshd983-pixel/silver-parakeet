"""Regression tests for the S2 classifier label mapping.

The historical bug: predict() read probs[:, 1] and called it p_ai, but this
checkpoint's id2label is {0: "artificial", 1: "human"}. Every score was inverted.
These tests pin the mapping to the checkpoint's own config so the same class of
bug cannot be reintroduced.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from sahu65.config import Settings
from sahu65.services.classifier import ClassifierService

# Weights ship inside the package; tests exercise the same resolution path an
# installed wheel would use.
MODEL_DIR = str(Path(__file__).resolve().parent.parent / "sahu65" / "models")
WEIGHTS = [
    os.path.join(MODEL_DIR, "classifier.int8.onnx"),
    os.path.join(MODEL_DIR, "classifier.onnx"),
]
HAS_WEIGHTS = any(os.path.exists(p) for p in WEIGHTS)
requires_weights = pytest.mark.skipif(
    not HAS_WEIGHTS, reason="S2 weights not present; run scripts/fetch_models.py"
)


def _service():
    s = Settings(model_dir=MODEL_DIR)
    svc = ClassifierService(s)
    svc.load()
    return svc


def test_checkpoint_config_declares_artificial_at_index_zero():
    """Pins the upstream contract this service depends on."""
    cfg_path = os.path.join(MODEL_DIR, "config.json")
    if not os.path.exists(cfg_path):
        pytest.skip("classifier config.json not present")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    labels = {int(k): v for k, v in cfg["id2label"].items()}
    assert labels[0] == "artificial", (
        f"upstream id2label changed: {labels}. Re-derive AI label tokens in classifier.py."
    )
    assert labels[1] == "human"


@requires_weights
def test_ai_class_index_resolved_from_id2label_not_hardcoded():
    svc = _service()
    assert svc.is_stub is False
    assert svc.ai_class_index == 0
    assert svc.id2label[svc.ai_class_index] == "artificial"


@requires_weights
def test_missing_config_is_a_hard_failure_not_a_silent_default():
    """A missing label map must raise; guessing an index is how the bug shipped."""
    svc = ClassifierService(Settings(), model_path=WEIGHTS[0])
    svc.model_dir = "no_such_dir"
    with pytest.raises(FileNotFoundError):
        svc._load_id2label()


@requires_weights
def test_predict_raises_rather_than_returning_stub_when_unloaded():
    svc = ClassifierService(Settings())
    with pytest.raises(RuntimeError):
        svc.predict(np.zeros((1, 3, 224, 224), dtype=np.float32))


ASSET_AI = Path("assets/ai/images.jpg")
ASSET_REAL_PHOTO = Path("assets/natural/nat01.jpg")
requires_assets = pytest.mark.skipif(
    not (ASSET_AI.is_file() and ASSET_REAL_PHOTO.is_file()),
    reason="evaluation fixtures absent; see assets/natural/ATTRIBUTION.md",
)


@requires_weights
@requires_assets
def test_ai_document_scores_higher_than_natural_photograph():
    """End-to-end orientation check on real assets.

    Skips when the fixtures are absent, because assets/natural/ holds Wikimedia Commons
    images whose per-file licenses have not been verified and may be removed.
    """
    svc = _service()
    from sahu65.services.preprocess import preprocess_for_classifier
    from PIL import Image

    def p_ai(path):
        with Image.open(path) as im:
            return svc.predict(preprocess_for_classifier(im.convert("RGB")))

    ai = p_ai(ASSET_AI)
    real_photo = p_ai(ASSET_REAL_PHOTO)
    assert real_photo < 0.5, f"a genuine photograph scored {real_photo:.4f} as AI"
    assert ai > 0.5, f"a known AI document scored {ai:.4f} as real"
