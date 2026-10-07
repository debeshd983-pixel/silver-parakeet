"""S2 Classifier inference service via ONNX Runtime."""
import json
import logging
import os
from typing import Optional
import numpy as np
import onnxruntime as ort

from app.config import Settings
from app.services.runtime import create_session_options

logger = logging.getLogger("detector")


class ClassifierService:
    """Wraps the S2 fine-tuned image classifier ONNX model."""

    def __init__(self, settings: Settings, model_path: Optional[str] = None):
        self.settings = settings
        self.model_path = model_path or os.path.join(settings.model_dir, "classifier.onnx")
        self.meta_path = os.path.join(settings.model_dir, "classifier_meta.json")
        self.session: Optional[ort.InferenceSession] = None
        self.input_name: Optional[str] = None
        self.output_name: Optional[str] = None
        # Index of the "AI-generated" class in the model's output. Defaults to 1
        # for models whose label mapping is unknown; overridden by classifier_meta.json
        # (this pinned model is {0: "artificial", 1: "human"} -> AI class is 0).
        self.ai_class_index: int = 1
        # Pinned commit ID as required by supply chain rules
        self.model_id = "Organika/sdxl-detector@657a8bf"

    def load(self) -> None:
        """Loads ONNX session if weights exist on disk."""
        if os.path.exists(self.meta_path):
            try:
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                self.ai_class_index = int(meta.get("ai_class_index", 1))
                logger.info(
                    f"Classifier label mapping: id2label={meta.get('id2label')}, "
                    f"AI class index={self.ai_class_index}"
                )
            except Exception as e:
                logger.warning(f"Could not read classifier meta from {self.meta_path}: {e}")

        if os.path.exists(self.model_path):
            try:
                opts = create_session_options(self.settings)
                self.session = ort.InferenceSession(
                    self.model_path,
                    sess_options=opts,
                    providers=["CPUExecutionProvider"],
                )
                self.input_name = self.session.get_inputs()[0].name
                self.output_name = self.session.get_outputs()[0].name
                logger.info(f"Loaded S2 classifier model from {self.model_path}")
            except Exception as e:
                logger.warning(f"Failed to load ONNX classifier from {self.model_path}: {e}")
                self.session = None
        else:
            logger.info(f"Classifier ONNX model not found at {self.model_path}; running in stub/dev mode")

    def warmup(self) -> None:
        """Executes warm-up inference."""
        dummy = np.zeros((1, 3, 224, 224), dtype=np.float32)
        _ = self.predict(dummy)

    def predict(self, tensor: np.ndarray) -> float:
        """Runs inference on preprocessed tensor (1, 3, 224, 224).

        Returns: probability that image is AI-generated [0.0, 1.0].
        """
        if self.session is not None and self.input_name:
            outputs = self.session.run([self.output_name], {self.input_name: tensor})
            logits = outputs[0]
            # Handle binary classification [batch, 2] or single logit [batch, 1]
            if logits.ndim == 2 and logits.shape[1] == 2:
                # Two-class logits: softmax, then take the AI-generated class.
                # AI class index is read from classifier_meta.json (0 for this
                # pinned model whose id2label is {0: "artificial", 1: "human"}).
                exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                probs = exps / np.sum(exps, axis=1, keepdims=True)
                idx = self.ai_class_index if self.ai_class_index in (0, 1) else 1
                p_ai = float(probs[0, idx])
            else:
                # Single logit; apply sigmoid
                z = float(logits.flatten()[0])
                p_ai = float(1.0 / (1.0 + np.exp(-z)))
            return float(np.clip(p_ai, 0.0, 1.0))

        # Fallback stub for dev / test environments
        return 0.50
