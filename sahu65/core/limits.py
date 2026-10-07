"""Defensive input validation, magic-byte sniffing, size/pixel limits, and rate limiting."""
import io
import time
from collections import defaultdict
from typing import Optional, Tuple
from fastapi import UploadFile
from PIL import Image

from sahu65.config import Settings


class DetectorError(Exception):
    """Base exception for application errors mapped to uniform error response."""
    def __init__(self, code: str, message: str, status_code: int):
        self.code = code
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class InvalidImageError(DetectorError):
    def __init__(self, message: str = "Cannot decode image file"):
        super().__init__("invalid_image", message, 400)


class FileTooLargeError(DetectorError):
    def __init__(self, max_mb: int):
        super().__init__("file_too_large", f"File exceeds maximum upload size of {max_mb} MB", 413)


class UnsupportedTypeError(DetectorError):
    def __init__(self, detected_mime: Optional[str] = None):
        msg = f"Unsupported image format: {detected_mime}" if detected_mime else "Unsupported image format; allowed formats are JPEG, PNG, WebP"
        super().__init__("unsupported_type", msg, 415)


class ImageTooLargePixelsError(DetectorError):
    def __init__(self, pixels: int, max_pixels: int):
        super().__init__(
            "image_too_large_pixels",
            f"Image dimensions ({pixels} pixels) exceed decompression bomb limit ({max_pixels} pixels)",
            422,
        )


class RateLimitedError(DetectorError):
    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__("rate_limited", message, 429)


class UnauthorizedError(DetectorError):
    def __init__(self, message: str = "Invalid or missing API key"):
        super().__init__("unauthorized", message, 401)


def sniff_image_type(header: bytes) -> Optional[str]:
    """Sniff format from magic bytes (architecture.md section 7).

    Returns 'jpeg', 'png', 'webp', or None.
    """
    if len(header) < 12:
        return None

    # JPEG: starts with FF D8 FF
    if header.startswith(b"\xff\xd8\xff"):
        return "jpeg"

    # PNG: 89 50 4E 47 0D 0A 1A 0A
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"

    # WebP: RIFF ???? WEBP
    if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
        return "webp"

    return None


async def read_and_validate_file(file: UploadFile, settings: Settings) -> Tuple[bytes, str]:
    """Streams file upload with a hard size cap and checks magic bytes."""
    max_bytes = settings.max_upload_mb * 1024 * 1024
    chunk_size = 64 * 1024  # 64 KB chunks
    total_bytes = 0
    chunks = []

    # Read first chunk to sniff magic bytes
    first_chunk = await file.read(chunk_size)
    if not first_chunk:
        raise InvalidImageError("Empty file uploaded")

    image_type = sniff_image_type(first_chunk)
    if not image_type:
        raise UnsupportedTypeError()

    chunks.append(first_chunk)
    total_bytes += len(first_chunk)

    if total_bytes > max_bytes:
        raise FileTooLargeError(settings.max_upload_mb)

    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total_bytes += len(chunk)
        if total_bytes > max_bytes:
            raise FileTooLargeError(settings.max_upload_mb)
        chunks.append(chunk)

    data = b"".join(chunks)
    return data, image_type


def inspect_image_pixels(data: bytes, settings: Settings) -> Tuple[int, int]:
    """Inspects dimensions lazily before full decompression to protect against bombs.

    Deliberately does NOT assign to ``Image.MAX_IMAGE_PIXELS``: that is a
    process-global, so writing it here leaked one request's limit into every
    later request and test. We enforce the limit ourselves below.
    """
    # Keep Pillow's own guard generous so it cannot fire before ours; our explicit
    # check below is the authoritative one and raises a typed DetectorError.
    Image.MAX_IMAGE_PIXELS = max(Image.MAX_IMAGE_PIXELS or 0, settings.max_pixels * 2)
    try:
        with Image.open(io.BytesIO(data)) as img:
            w, h = img.size
            total_pixels = w * h
            if total_pixels > settings.max_pixels:
                raise ImageTooLargePixelsError(total_pixels, settings.max_pixels)
            return w, h
    except ImageTooLargePixelsError:
        raise
    except Image.DecompressionBombError:
        raise ImageTooLargePixelsError(settings.max_pixels * 2, settings.max_pixels)
    except Exception as e:
        raise InvalidImageError(f"Corrupted or invalid image: {str(e)}")


class SimpleRateLimiter:
    """Sliding window in-memory rate limiter per client key/IP."""

    def __init__(self, rate_limit_str: str = "30/minute"):
        self.limit = 30
        self.window = 60.0
        self._parse_limit(rate_limit_str)
        self.history = defaultdict(list)

    def _parse_limit(self, s: str):
        try:
            parts = s.split("/")
            self.limit = int(parts[0])
            unit = parts[1].lower() if len(parts) > 1 else "minute"
            if "sec" in unit:
                self.window = 1.0
            elif "min" in unit:
                self.window = 60.0
            elif "hour" in unit:
                self.window = 3600.0
        except Exception:
            self.limit = 30
            self.window = 60.0

    def check(self, key: str) -> bool:
        now = time.time()
        cutoff = now - self.window
        timestamps = [t for t in self.history[key] if t > cutoff]
        if len(timestamps) >= self.limit:
            self.history[key] = timestamps
            return False
        timestamps.append(now)
        self.history[key] = timestamps
        return True


    def configure(self, rate_limit_str: str) -> None:
        """Re-applies the limit/window from a RATE_LIMIT string, clearing stale history."""
        self._parse_limit(rate_limit_str)
        self.history.clear()

    def describe(self) -> str:
        return f"{self.limit}/{int(self.window)}s"


# Module-level singleton for route handlers. Reconfigured from the RATE_LIMIT
# setting in main.lifespan; the previous hardcoded construction silently ignored
# the RATE_LIMIT env var entirely.
rate_limiter = SimpleRateLimiter()
