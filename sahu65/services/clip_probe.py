"""S3 Frozen CLIP ViT-B/16 image encoder + Linear Logistic Probe."""
import json
import logging
import os
from typing import Optional
import numpy as np
import onnxruntime as ort

from sahu65.config import Settings
from sahu65.services.runtime import create_session_options

logger = logging.getLogger("detector")


class ClipProbeService:
    """Wraps frozen CLIP ViT-B/16 ONNX image encoder and trained logistic head."""

    def __init__(self, settings: Settings, model_path: Optional[str] = None, head_path: Optional[str] = None):
        self.settings = settings
        self.model_path = model_path or os.path.join(settings.model_dir, "clip_encoder.onnx")
        self.head_path = head_path or os.path.join(settings.model_dir, "clip_head.json")
        self.session: Optional[ort.InferenceSession] = None
        self.input_name: Optional[str] = None
        self.output_name: Optional[str] = None
        # Pinned commit ID
        self.model_id = "openai/clip-vit-base-patch16@51a6c11"

        # Head weights
        self.head_weights: Optional[np.ndarray] = None
        self.head_bias: float = 0.0

    def load(self) -> None:
        """Loads ONNX encoder and logistic probe head weights."""
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
                logger.info(f"Loaded S3 CLIP encoder from {self.model_path}")
            except Exception as e:
                logger.warning(f"Failed to load ONNX CLIP encoder: {e}")
                self.session = None
        else:
            logger.info(f"CLIP ONNX model not found at {self.model_path}; running in stub/dev mode")

        if os.path.exists(self.head_path):
            try:
                with open(self.head_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.head_weights = np.array(data["weights"], dtype=np.float32)
                    self.head_bias = float(data.get("bias", 0.0))
                logger.info(f"Loaded CLIP probe head weights from {self.head_path}")
            except Exception as e:
                logger.warning(f"Failed to load CLIP head weights: {e}")
        else:
            # Default zero probe weights for fallback
            self.head_weights = np.zeros(512, dtype=np.float32)
            self.head_bias = 0.0

    def warmup(self) -> None:
        """Executes warm-up inference."""
        dummy = np.zeros((1, 3, 224, 224), dtype=np.float32)
        _ = self.predict(dummy)

    def predict(self, tensor: np.ndarray) -> float:
        """Runs image encoder and logistic regression probe head.

        Input: tensor of shape (1, 3, 224, 224) or (5, 3, 224, 224) if TTA is enabled.
        Returns: probability that image is AI-generated [0.0, 1.0].
        """
        if self.session is not None and self.input_name:
            outputs = self.session.run([self.output_name], {self.input_name: tensor})
            embeddings = outputs[0]  # (B, D)

            # Normalize embeddings
            norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            normalized = embeddings / norms

            # If TTA, average the normalized embeddings
            if normalized.shape[0] > 1:
                feature = np.mean(normalized, axis=0)
                feat_norm = np.linalg.norm(feature)
                if feat_norm > 1e-12:
                    feature = feature / feat_norm
            else:
                feature = normalized[0]

            if self.head_weights is not None and len(self.head_weights) == len(feature):
                z = float(np.dot(feature, self.head_weights) + self.head_bias)
            else:
                z = 0.0

            p_ai = float(1.0 / (1.0 + np.exp(-z)))
            return float(np.clip(p_ai, 0.0, 1.0))

        # Fallback stub for dev / test environments
        return 0.50
