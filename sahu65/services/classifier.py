"""S2 Classifier inference service via ONNX Runtime.

The AI/real class index is resolved from the checkpoint's own ``id2label`` map at
load time. It is never hardcoded: a hardcoded index silently inverts every score
if the checkpoint's label order ever changes.
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

MODEL_REPO = "Organika/sdxl-detector"
MODEL_REVISION = "b37fede8562cb72b89ec201c0987f96ba21b518a"
MODEL_LICENSE = "cc-by-nc-3.0"

# Substrings that identify the positive ("AI-generated") class in id2label.
AI_LABEL_TOKENS = ("artificial", "ai", "fake", "generated", "synthetic", "sdxl", "deepfake")


class ClassifierService:
    """Wraps the S2 fine-tuned image classifier ONNX model."""

    def __init__(self, settings: Settings, model_path: Optional[str] = None):
        self.settings = settings
        self.model_dir = settings.model_dir
        self.model_path = model_path or self._resolve_weights()
        self.session: Optional[ort.InferenceSession] = None
        self.input_name: Optional[str] = None
        self.output_name: Optional[str] = None

        self.id2label: Dict[int, str] = {}
        self.ai_class_index: Optional[int] = None
        self.load_error: Optional[str] = None
        self.is_stub: bool = True

        self.model_id = f"{MODEL_REPO}@{MODEL_REVISION[:7]}"

    def _resolve_weights(self) -> str:
        """Prefers the INT8 build; falls back to fp32. INT8 is verified score-identical."""
        int8 = os.path.join(self.model_dir, "classifier.int8.onnx")
        fp32 = os.path.join(self.model_dir, "classifier.onnx")
        return int8 if os.path.exists(int8) else fp32

    def _load_id2label(self) -> None:
        """Reads id2label from the checkpoint config shipped alongside the weights."""
        cfg = os.path.join(self.model_dir, "config.json")
        if not os.path.exists(cfg):
            raise FileNotFoundError(f"classifier config.json not found at {cfg}")
        with open(cfg, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw = data.get("id2label") or data.get("label2id")
        if not raw:
            raise ValueError("classifier config.json has no id2label/label2id mapping")

        if "id2label" not in data:
            raw = {v: k for k, v in raw.items()}
        self.id2label = {int(k): str(v) for k, v in raw.items()}

        matches = [
            (int(k), v) for k, v in self.id2label.items()
            if any(tok in v.strip().lower() for tok in AI_LABEL_TOKENS)
        ]
        if len(matches) != 1:
            labels = ", ".join(f"{k}={v}" for k, v in sorted(self.id2label.items()))
            raise ValueError(
                f"cannot resolve a unique AI class from id2label ({labels}); "
                f"matched={len(matches)}"
            )
        self.ai_class_index = matches[0][0]

    def load(self) -> None:
        """Loads ONNX session and label map. Raises on any failure (no silent stub)."""
        self.load_error = None
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"classifier weights not found at {self.model_path}")
            self._load_id2label()

            opts = create_session_options(self.settings)
            self.session = ort.InferenceSession(
                self.model_path, sess_options=opts, providers=["CPUExecutionProvider"]
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.is_stub = False
            logger.info(
                f"Loaded S2 classifier {self.model_id} from {self.model_path}; "
                f"AI class index={self.ai_class_index} ({self.id2label[self.ai_class_index]!r}) "
                f"labels={self.id2label}"
            )
        except Exception as e:
            self.load_error = str(e)
            self.session = None
            self.is_stub = True
            raise

    def warmup(self) -> None:
        """Executes warm-up inference."""
        self.predict(np.zeros((1, 3, 224, 224), dtype=np.float32))

    def predict(self, tensor: np.ndarray) -> float:
        """Runs inference on a preprocessed tensor (1, 3, 224, 224).

        Returns: probability that the image is AI-generated, in [0.0, 1.0].
        """
        if self.session is None or self.input_name is None or self.ai_class_index is None:
            raise RuntimeError(f"S2 classifier is not loaded: {self.load_error}")

        outputs = self.session.run([self.output_name], {self.input_name: tensor})
        logits = np.asarray(outputs[0])

        if logits.ndim == 2 and logits.shape[1] == 2:
            shifted = logits - np.max(logits, axis=1, keepdims=True)
            exps = np.exp(shifted)
            probs = exps / np.sum(exps, axis=1, keepdims=True)
            p_ai = probs[:, self.ai_class_index]
        elif logits.ndim == 2 and logits.shape[1] == 1:
            p_ai = 1.0 / (1.0 + np.exp(-logits[:, 0].astype(np.float64)))
        else:
            raise ValueError(f"unexpected classifier output shape {logits.shape}")

        if p_ai.size != 1:
            raise ValueError(f"expected batch size 1, got {p_ai.size}")

        return float(np.clip(float(p_ai[0]), 0.0, 1.0))

    def describe(self) -> dict:
        """Introspection payload for /version."""
        return {
            "model_id": self.model_id,
            "revision": MODEL_REVISION,
            "license": MODEL_LICENSE,
            "weights": os.path.basename(self.model_path),
            "loaded": not self.is_stub,
            "labels": {str(k): v for k, v in sorted(self.id2label.items())},
            "ai_class_index": self.ai_class_index,
        }
