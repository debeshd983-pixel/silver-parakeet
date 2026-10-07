"""S2 Classifier inference service via ONNX Runtime."""
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
        self.session: Optional[ort.InferenceSession] = None
        self.input_name: Optional[str] = None
        self.output_name: Optional[str] = None
        # Pinned commit ID as required by supply chain rules
        self.model_id = "Organika/sdxl-detector@657a8bf"

    def load(self) -> None:
        """Loads ONNX session if weights exist on disk."""
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
                # Class 1 is AI-generated; apply softmax
                exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                probs = exps / np.sum(exps, axis=1, keepdims=True)
                p_ai = float(probs[0, 1])
            else:
                # Single logit; apply sigmoid
                z = float(logits.flatten()[0])
                p_ai = float(1.0 / (1.0 + np.exp(-z)))
            return float(np.clip(p_ai, 0.0, 1.0))

        # Fallback stub for dev / test environments
        return 0.50
