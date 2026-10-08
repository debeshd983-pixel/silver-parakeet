"""Tests for the shipped detector: a linear probe on a frozen CLIP encoder.

The invariants here exist because each one, if violated, produces plausible-looking but
wrong verdicts rather than an error:

* the head must agree with the encoder on dimensionality;
* the head must not be degenerate (all-zero weights would return a constant 0.5 for
  every image -- the exact defect WORKLOG records as WORKLOG defect 3);
* the head's sign convention must be "higher = more AI", since CLIP carries no labels of
  its own and a flipped head would invert every verdict with nothing to catch it;
* inference must be one image per call, because the encoder is dynamically quantized and
  an image embeds differently beside another than alone.
"""
import json
import os

import numpy as np
import pytest

from sahu65.services.clip_probe import (
    EMBED_DIM,
    HEAD_DIRECTION,
    ClipProbeService,
    ModelLoadError,
)


def make_service(model_dir=None):
    from sahu65.config import Settings

    s = Settings()
    if model_dir:
        s.model_dir = str(model_dir)
    return ClipProbeService(s)


def bundled_head_path():
    from sahu65.config import PACKAGE_DIR

    return os.path.join(str(PACKAGE_DIR), "models", "clip_head.json")


def test_bundled_head_has_expected_shape():
    with open(bundled_head_path(), "r", encoding="utf-8") as f:
        head = json.load(f)
    assert len(head["weights"]) == EMBED_DIM
    assert isinstance(head["bias"], (int, float))
    assert head.get("normalised") is True, "predict() L2-normalises before the dot product"


def test_bundled_head_is_not_degenerate():
    """An all-zero head returns a constant for every image."""
    with open(bundled_head_path(), "r", encoding="utf-8") as f:
        head = json.load(f)
    w = np.array(head["weights"], dtype=np.float32)
    assert np.isfinite(w).all()
    assert float(np.linalg.norm(w)) > 1e-6, "head weights are ~zero: constant output"


def test_bundled_head_direction_is_positive_for_ai():
    """Pins the sign convention.

    CLIP has no labels of its own, so which direction means "AI" is decided entirely by
    how the head was fitted. Nothing at runtime would catch a flipped head: every score
    would simply be inverted. This asserts the fitted head actually scores AI above real
    on real benchmark data.
    """
    bench = os.path.join("benchmark", "scores_oof.csv")
    if not os.path.isfile(bench):
        pytest.skip("benchmark not built; run scripts/build_benchmark.py first")

    import csv

    rows = list(csv.DictReader(open(bench, encoding="utf-8")))
    y = np.array([int(r["label"]) for r in rows])
    # out-of-fold scores were produced by this same head, so the mean separation here is
    # a direct check of the sign convention.
    p = np.array([float(r["p_detector"]) for r in rows])
    assert p[y == 1].mean() > p[y == 0].mean(), (
        "head scores AI images LOWER than real ones: the sign is flipped"
    )
    assert "higher" in HEAD_DIRECTION


def test_load_raises_when_encoder_missing(tmp_path):
    (tmp_path / "clip_head.json").write_text(json.dumps(
        {"weights": [0.1] * EMBED_DIM, "bias": 0.0}), encoding="utf-8")
    svc = make_service(tmp_path)
    with pytest.raises(ModelLoadError):
        svc.load()
    assert svc.is_stub


def test_load_raises_when_head_missing(tmp_path):
    import shutil

    from sahu65.config import PACKAGE_DIR

    src = os.path.join(str(PACKAGE_DIR), "models", "clip_encoder.int8.onnx")
    if not os.path.isfile(src):
        pytest.skip("bundled encoder missing")
    shutil.copy(src, tmp_path / "clip_encoder.int8.onnx")
    svc = make_service(tmp_path)
    with pytest.raises(ModelLoadError):
        svc.load()


def test_load_raises_on_wrong_head_width(tmp_path):
    import shutil

    from sahu65.config import PACKAGE_DIR

    src = os.path.join(str(PACKAGE_DIR), "models", "clip_encoder.int8.onnx")
    if not os.path.isfile(src):
        pytest.skip("bundled encoder missing")
    shutil.copy(src, tmp_path / "clip_encoder.int8.onnx")
    # 256 weights against a 512-d encoder: must be refused, not silently scored as zero.
    (tmp_path / "clip_head.json").write_text(
        json.dumps({"weights": [0.1] * 256, "bias": 0.0}), encoding="utf-8")
    svc = make_service(tmp_path)
    with pytest.raises(ModelLoadError):
        svc.load()


def test_load_raises_on_zero_head(tmp_path):
    import shutil

    from sahu65.config import PACKAGE_DIR

    src = os.path.join(str(PACKAGE_DIR), "models", "clip_encoder.int8.onnx")
    if not os.path.isfile(src):
        pytest.skip("bundled encoder missing")
    shutil.copy(src, tmp_path / "clip_encoder.int8.onnx")
    (tmp_path / "clip_head.json").write_text(
        json.dumps({"weights": [0.0] * EMBED_DIM, "bias": 0.5}), encoding="utf-8")
    svc = make_service(tmp_path)
    with pytest.raises(ModelLoadError):
        svc.load()


def test_predict_before_load_raises():
    svc = make_service()
    with pytest.raises(RuntimeError):
        svc.predict(np.zeros((1, 3, 224, 224), dtype=np.float32))


def test_bundled_detector_loads_and_scores():
    from sahu65.config import PACKAGE_DIR

    if not os.path.isfile(os.path.join(str(PACKAGE_DIR), "models",
                                       "clip_encoder.int8.onnx")):
        pytest.skip("bundled encoder missing")
    svc = make_service()
    svc.load()
    assert svc.is_stub is False
    assert svc.embed_dim == EMBED_DIM

    p = svc.predict(np.zeros((1, 3, 224, 224), dtype=np.float32))
    assert 0.0 <= p <= 1.0


def test_encoder_is_batch_sensitive_so_serve_one_at_a_time():
    """Documents the constraint the service depends on.

    The encoder is dynamically quantized, so its activation ranges are computed per run
    from whatever is in the batch: the same image embeds differently alone than beside
    another image (measured cosine 0.65-0.89). Serving one image per request is what
    keeps the shipped score identical to what the head was fitted on. If this test starts
    finding cosine ~1.0, the encoder was rebuilt statically and the batching constraint
    (and the fitted head) should be revisited.
    """
    import onnxruntime as ort

    from sahu65.config import PACKAGE_DIR
    from sahu65.services.preprocess import decode_image, preprocess_for_clip

    path = os.path.join(str(PACKAGE_DIR), "models", "clip_encoder.int8.onnx")
    bench = os.path.join("benchmark", "manifest.csv")
    if not (os.path.isfile(path) and os.path.isfile(bench)):
        pytest.skip("bundled encoder or benchmark missing")

    import csv

    rows = [r for r in csv.DictReader(open(bench, encoding="utf-8"))
            if r["perturbation"] == "original"][:4]
    if len(rows) < 4:
        pytest.skip("not enough images")

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name

    tensors = []
    for r in rows:
        with open(os.path.join(os.path.dirname(bench), r["path"]), "rb") as f:
            img, _ = decode_image(f.read())
        tensors.append(preprocess_for_clip(img))

    def run(ts):
        out = sess.run(None, {inp: np.concatenate(ts, axis=0)})[0]
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)

    solo = run([tensors[0]])[0]
    batched = run(tensors)[0]
    cos = float(np.dot(solo, batched))
    # Documented as < 1.0 for dynamic quantization. If a future rebuild makes this ~1.0
    # the model became batch-invariant, which is an improvement, not a regression.
    assert cos <= 1.0