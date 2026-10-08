"""The primary detector: a linear probe on a frozen CLIP ViT-B/16 image encoder.

Why this replaced the fine-tuned classifier
-------------------------------------------
The service originally shipped ``Organika/sdxl-detector`` (S2) as its detector with this
CLIP probe (S3) disabled, because S3's head had been fitted on ``np.random.randn``.

Measured on benchmark ``bench-8e0bd9e97e3e`` (831 images, 13 generator families, 2 real
photo pipelines), the two were compared head to head:

=========================  =========  ==========================
signal                     AUC        false positives @50% recall
=========================  =========  ==========================
S2 sdxl-detector (shipped)  0.6434              24.3%
S3 CLIP probe              0.8704               4.5%
=========================  =========  ==========================

S2 is also cc-by-nc-3.0 (non-commercial), which is a poor licence for a package published
on PyPI. CLIP ViT-B/16 is MIT per its upstream model card. The swap is smaller (85.5 MB
vs 91.8 MB), faster (135 ms vs 179 ms per image), and far more accurate, so S2 is no
longer shipped.

The decisive detail is *where* the accuracy comes from. S2 saturates on the families it
resembles (dalle3, midjourney-v5, stable-diffusion-xl all score a constant 1.000) and
collapses elsewhere. S3 repairs exactly those failures:

    generator            S2        S3
    glide                5.9%     97.1%
    dalle2              15.6%     93.8%
    FLUX.1-dev          13.2%     78.9%
    FLUX.1-schnell      21.6%     83.8%
    firefly             33.3%     88.9%

Fusing S2 and S3 is *worse* than S3 alone at every mixing weight tried (best fused AUC
0.8086 against 0.8704 for S3 on its own), so the two are not combined. See
``scripts/train_clip_probe.py``.

Load-time failures are hard failures. Returning a constant 0.5 is exactly the defect this
repository previously shipped (WORKLOG defect 3), so it cannot happen here.
"""
import json
import logging
import os
from typing import Dict, Optional

import numpy as np
import onnxruntime as ort

from sahu65.config import Settings
from sahu65.services.runtime import create_session_options

logger = logging.getLogger("detector")

ENCODER_REPO = "openai/clip-vit-base-patch16"
ENCODER_REVISION = "57c216476eefef5ab752ec549e440a49ae4ae5f3"
# The Hub metadata declares no licence for this repo. Recorded as upstream states it
# rather than assumed. Worth confirming before a commercial release.
ENCODER_LICENSE = "undeclared-on-hub (upstream model card states MIT)"

# The probe operates on unit-normalised 512-d image embeddings.
EMBED_DIM = 512

# What the head was fitted to predict. CLIP has no labels of its own, so this direction
# is a property of the fitted weights, not something readable from the checkpoint. If
# the head were ever sign-flipped, every verdict inverts silently, exactly like the
# inverted class-index bug in WORKLOG defect 1. tests/test_clip_probe.py pins it.
HEAD_DIRECTION = "higher dot product = more likely AI-generated"


class ModelLoadError(RuntimeError):
    """Raised when the detector cannot be loaded. Never degraded to a constant."""


class ClipProbeService:
    """Frozen CLIP ViT-B/16 encoder plus a fitted linear probe head."""

    def __init__(self, settings: Settings, model_path: Optional[str] = None,
                 head_path: Optional[str] = None):
        self.settings = settings
        self.model_dir = settings.model_dir
        # Prefer the INT8 build, exactly as the retired classifier did. The head is fitted
        # on INT8 embeddings: fp32 and INT8 embeddings agree only to cosine 0.55, so a
        # head is not portable between the two builds.
        self.model_path = model_path or self._resolve_weights()
        self.head_path = head_path or os.path.join(self.model_dir, "clip_head.json")

        self.session: Optional[ort.InferenceSession] = None
        self.input_name: Optional[str] = None
        self.output_name: Optional[str] = None
        self.head_weights: Optional[np.ndarray] = None
        self.head_bias: float = 0.0

        self.load_error: Optional[str] = None
        self.is_stub: bool = True
        self.embed_dim: Optional[int] = None
        self.model_id = f"{ENCODER_REPO}@{ENCODER_REVISION[:7]}"

    def _resolve_weights(self) -> str:
        int8 = os.path.join(self.model_dir, "clip_encoder.int8.onnx")
        fp32 = os.path.join(self.model_dir, "clip_encoder.onnx")
        return int8 if os.path.exists(int8) else fp32

    def load(self) -> None:
        """Loads the encoder and head. Raises on any failure; never degrades to a stub."""
        self.load_error = None
        try:
            if not os.path.exists(self.model_path):
                raise ModelLoadError(
                    f"CLIP encoder not found at {self.model_path} "
                    f"(run scripts/export_clip_onnx.py)"
                )
            if not os.path.exists(self.head_path):
                raise ModelLoadError(
                    f"CLIP probe head not found at {self.head_path} "
                    f"(run scripts/train_clip_probe.py)"
                )

            opts = create_session_options(self.settings)
            self.session = ort.InferenceSession(
                self.model_path, sess_options=opts, providers=["CPUExecutionProvider"]
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name

            with open(self.head_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.head_weights = np.array(data["weights"], dtype=np.float32)
            self.head_bias = float(data.get("bias", 0.0))

            if self.head_weights.shape != (EMBED_DIM,):
                raise ValueError(
                    f"head has {self.head_weights.shape[0]} weights, expected {EMBED_DIM}; "
                    f"the head and encoder do not match"
                )
            if not np.isfinite(self.head_weights).all() or not np.isfinite(self.head_bias):
                raise ValueError("head contains non-finite values")
            norm = float(np.linalg.norm(self.head_weights))
            if norm < 1e-8:
                raise ValueError(
                    "head weights are all zero; the probe would return a constant "
                    f"0.5 for every image (bias={self.head_bias})"
                )

            # Probe the real output width so a mismatched encoder is caught at load, not
            # silently scored as zero by the length check in predict().
            probe = self.session.run(
                [self.output_name],
                {self.input_name: np.zeros((1, 3, 224, 224), dtype=np.float32)},
            )[0]
            self.embed_dim = int(np.asarray(probe).shape[-1])
            if self.embed_dim != EMBED_DIM:
                raise ValueError(
                    f"encoder emits {self.embed_dim}-d embeddings but the head expects "
                    f"{EMBED_DIM}"
                )

            self.is_stub = False
            logger.info(
                f"Loaded detector {self.model_id} from {self.model_path}; "
                f"embed_dim={self.embed_dim} head_norm={norm:.4f} bias={self.head_bias:.4f}"
            )
        except ModelLoadError:
            self.load_error = self.load_error or "load failed"
            self.session = None
            self.is_stub = True
            raise
        except Exception as e:
            # Every failure mode is surfaced as ModelLoadError so callers have one type to
            # catch. Letting FileNotFoundError and ValueError escape separately meant the
            # service had to reason about three different exception types to decide
            # whether the detector was usable, and it is exactly the sort of gap a
            # "handled it somewhere else" branch turns into a silent stub.
            self.load_error = str(e)
            self.session = None
            self.is_stub = True
            raise ModelLoadError(
                f"detector failed to load: {e} "
                f"(encoder={os.path.basename(self.model_path)}, "
                f"head={os.path.basename(self.head_path)})"
            ) from e

    def warmup(self) -> None:
        """Runs one inference so the first real request is not the cold one."""
        self.predict(np.zeros((1, 3, 224, 224), dtype=np.float32))

    def predict(self, tensor: np.ndarray) -> float:
        """Runs the encoder and probe.

        Accepts (1, 3, H, W) or (N, 3, H, W). When N > 1 the unit-normalised embeddings
        are averaged, which is the 5-crop TTA path.

        Returns the probability that the image is AI-generated, in [0.0, 1.0].
        """
        if self.session is None or self.head_weights is None:
            raise RuntimeError(f"detector is not loaded: {self.load_error}")

        outputs = self.session.run([self.output_name], {self.input_name: tensor})
        embeddings = np.asarray(outputs[0], dtype=np.float32)

        norms = np.maximum(np.linalg.norm(embeddings, axis=-1, keepdims=True), 1e-12)
        normalized = embeddings / norms

        if normalized.shape[0] > 1:
            feature = np.mean(normalized, axis=0)
            f_norm = float(np.linalg.norm(feature))
            if f_norm > 1e-12:
                feature = feature / f_norm
        else:
            feature = normalized[0]

        if len(self.head_weights) != len(feature):
            raise RuntimeError(
                f"head expects {len(self.head_weights)} features, encoder produced "
                f"{len(feature)}"
            )

        z = float(np.dot(feature, self.head_weights) + self.head_bias)
        return float(np.clip(1.0 / (1.0 + np.exp(-z)), 0.0, 1.0))

    def describe(self) -> Dict[str, object]:
        """Introspection payload for /version."""
        return {
            "model_id": self.model_id,
            "revision": ENCODER_REVISION,
            "license": ENCODER_LICENSE,
            "weights": os.path.basename(self.model_path),
            "head": os.path.basename(self.head_path),
            "loaded": not self.is_stub,
            "embed_dim": self.embed_dim,
            "head_direction": HEAD_DIRECTION,
            "head_fitted_on": (
                json.load(open(self.head_path, encoding="utf-8")).get("note")
                if os.path.exists(self.head_path) else None
            ),
        }