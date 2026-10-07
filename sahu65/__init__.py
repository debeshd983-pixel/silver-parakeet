"""sahu65 - AI-generated image detection.

Two ways to use it:

Local, no server and no API key (weights are bundled, ~92 MB)::

    import sahu65
    result = sahu65.detect("photo.jpg")

Against a hosted deployment with an API key::

    from sahu65 import Client
    result = Client(api_key="sk-...").detect("photo.jpg")

Or from a shell::

    sahu65 detect photo.jpg
    sahu65 serve --port 8000

.. warning::
   The bundled model is a **natural-image** diffusion detector. Measured AUC on
   document images is **0.375** - worse than chance - so real photographs of ID cards,
   certificates and forms are frequently flagged as AI. It also returns an
   **uncalibrated** score (``T_lo``/``T_hi`` are unfitted placeholders). Do not use
   it as the sole basis for any decision about a document. See MODEL_CARD.md.
"""
from .local import Detection, LocalDetector, detect, detect_bytes, model_info, warm

__version__ = "1.0.0"

__all__ = [
    "Detection",
    "LocalDetector",
    "Client",
    "Sahu65Error",
    "AuthError",
    "NotReadyError",
    "detect",
    "detect_bytes",
    "warm",
    "model_info",
    "__version__",
]


def __getattr__(name):
    # Lazy so that `import sahu65` for local inference does not require httpx.
    if name in ("Client", "Sahu65Error", "AuthError", "NotReadyError"):
        from . import client as _client

        return getattr(_client, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")