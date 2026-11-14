"""Face detection using the MediaPipe Tasks Face Detector API.

The default BlazeFace short-range model is hosted by Google's official
MediaPipe model storage and is licensed under Apache License 2.0:
https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import threading
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol, TypeAlias

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, UnidentifiedImageError

LOGGER = logging.getLogger(__name__)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)
MODEL_FILENAME = "blaze_face_short_range.tflite"
SUPPORTED_FORMATS = {"JPEG", "PNG"}
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class InvalidImageError(ValueError):
    """Raised when an image is unsupported, invalid, or corrupted."""


@dataclass(frozen=True, slots=True)
class RGBImage:
    """Validated RGB uint8 pixels for reuse across preprocessing modules."""

    pixels: np.ndarray

    def __post_init__(self) -> None:
        if (
            not isinstance(self.pixels, np.ndarray)
            or self.pixels.dtype != np.uint8
            or self.pixels.ndim != 3
            or self.pixels.shape[2] != 3
            or self.pixels.shape[0] == 0
            or self.pixels.shape[1] == 0
        ):
            raise InvalidImageError("RGBImage requires a non-empty RGB uint8 array.")
        if not self.pixels.flags.c_contiguous:
            object.__setattr__(self, "pixels", np.ascontiguousarray(self.pixels))


ImageInput: TypeAlias = (
    str
    | os.PathLike[str]
    | bytes
    | bytearray
    | memoryview
    | Image.Image
    | np.ndarray
    | RGBImage
)


class FaceDetectorError(RuntimeError):
    """Raised when the detector cannot be initialized or used."""


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """A face rectangle in image pixel coordinates."""

    x: int
    y: int
    width: int
    height: int

    @property
    def x2(self) -> int:
        """Return the exclusive right edge of the box."""
        return self.x + self.width

    @property
    def y2(self) -> int:
        """Return the exclusive bottom edge of the box."""
        return self.y + self.height

    def to_dict(self) -> dict[str, int]:
        """Return a JSON-compatible bounding-box representation."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FaceDetection:
    """One detected face and its detector confidence, when available."""

    bounding_box: BoundingBox
    confidence: float | None

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible detection metadata."""
        return {
            "bounding_box": self.bounding_box.to_dict(),
            "confidence": self.confidence,
        }


def _pil_to_rgb(image: Image.Image) -> np.ndarray:
    if image.format is not None and image.format not in SUPPORTED_FORMATS:
        raise InvalidImageError(
            f"Unsupported image format {image.format!r}; use JPG or PNG."
        )
    try:
        image.load()
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except (OSError, ValueError) as exc:
        raise InvalidImageError("The image is corrupted or cannot be decoded.") from exc
    if rgb.size == 0 or rgb.ndim != 3 or min(rgb.shape[:2]) <= 0:
        raise InvalidImageError("The image must have non-zero width and height.")
    return np.ascontiguousarray(rgb)


def _decode_encoded_image(encoded: bytes) -> np.ndarray:
    try:
        with Image.open(BytesIO(encoded)) as image:
            if image.format not in SUPPORTED_FORMATS:
                raise InvalidImageError("Unsupported image format; use JPG or PNG.")
            image.verify()
        with Image.open(BytesIO(encoded)) as image:
            return _pil_to_rgb(image)
    except InvalidImageError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError("The image is corrupted or cannot be decoded.") from exc


def _load_rgb(image: ImageInput) -> np.ndarray:
    """Validate inputs and return RGB uint8 pixels.

    Ordinary inputs are decoded or converted to a new array. ``RGBImage`` is
    already validated and returns its existing buffer without conversion.
    """
    if isinstance(image, RGBImage):
        return image.pixels

    if isinstance(image, np.ndarray):
        if image.size == 0 or image.ndim not in (2, 3):
            raise InvalidImageError("Expected a non-empty grayscale or color image.")
        if image.dtype != np.uint8:
            raise InvalidImageError("OpenCV image arrays must use uint8 pixel values.")
        if image.ndim == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image.shape[2] == 4:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            raise InvalidImageError("Expected a grayscale, BGR, or BGRA image array.")
        if min(rgb.shape[:2]) <= 0:
            raise InvalidImageError("The image must have non-zero width and height.")
        return np.ascontiguousarray(rgb)

    if isinstance(image, Image.Image):
        return _pil_to_rgb(image)

    if isinstance(image, (bytes, bytearray, memoryview)):
        return _decode_encoded_image(bytes(image))

    if isinstance(image, (str, os.PathLike)):
        path = Path(image)
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise InvalidImageError("Unsupported image extension; use JPG or PNG.")
        try:
            with Image.open(path) as opened:
                if opened.format not in SUPPORTED_FORMATS:
                    raise InvalidImageError("Unsupported image format; use JPG or PNG.")
                opened.verify()
            with Image.open(path) as opened:
                return _pil_to_rgb(opened)
        except InvalidImageError:
            raise
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise InvalidImageError(
                f"The image file {str(path)!r} is corrupted or cannot be decoded."
            ) from exc

    raise InvalidImageError(
        "Expected a JPG/PNG path, encoded image bytes, PIL image, or OpenCV array."
    )


def _default_model_path() -> Path:
    cache_root = Path(
        os.environ.get("LOCALAPPDATA", str(Path.home() / ".cache"))
    )
    return cache_root / "face_preprocessing" / "models" / MODEL_FILENAME


def _ensure_model(model_path: Path, *, download: bool) -> Path:
    if model_path.is_file() and model_path.stat().st_size > 0:
        return model_path
    if not download:
        raise FaceDetectorError(f"MediaPipe model file was not found: {model_path}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=30) as response:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=model_path.parent, delete=False
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                shutil.copyfileobj(response, temporary_file)
        if temporary_path.stat().st_size == 0:
            raise FaceDetectorError("The downloaded MediaPipe model file is empty.")
        os.replace(temporary_path, model_path)
    except (OSError, urllib.error.URLError) as exc:
        raise FaceDetectorError(
            f"Could not download the MediaPipe face model from {MODEL_URL}."
        ) from exc
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    LOGGER.info("Downloaded MediaPipe face detection model to %s", model_path)
    return model_path


@dataclass(frozen=True, slots=True)
class FaceDetectorConfig:
    """Configuration for the MediaPipe Tasks face detector."""

    min_detection_confidence: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_detection_confidence <= 1.0:
            raise ValueError("min_detection_confidence must be between 0 and 1.")


class _TaskDetector(Protocol):
    def detect(self, image: mp.Image) -> Any: ...

    def close(self) -> None: ...


class FaceDetector:
    """Reusable MediaPipe Tasks face detector for images and webcam frames."""

    _model_download_lock = threading.Lock()

    def __init__(
        self,
        model_path: str | os.PathLike[str] | None = None,
        *,
        config: FaceDetectorConfig | None = None,
    ) -> None:
        self.config = config or FaceDetectorConfig()
        self._closed = False
        self._detect_lock = threading.Lock()

        model_was_provided = model_path is not None
        resolved_model_path = Path(model_path) if model_was_provided else _default_model_path()
        with self._model_download_lock:
            resolved_model_path = _ensure_model(
                resolved_model_path, download=not model_was_provided
            )
        self._detector = self._build_detector(
            resolved_model_path, self.config.min_detection_confidence
        )

    @staticmethod
    def _build_detector(
        model_path: Path, min_detection_confidence: float
    ) -> _TaskDetector:
        try:
            from mediapipe.tasks.python import BaseOptions, vision

            options = vision.FaceDetectorOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                min_detection_confidence=min_detection_confidence,
            )
            return vision.FaceDetector.create_from_options(options)
        except (ImportError, OSError, ValueError, RuntimeError) as exc:
            raise FaceDetectorError(
                "MediaPipe Tasks could not initialize the face detector."
            ) from exc

    def detect(self, image: ImageInput) -> list[FaceDetection]:
        """Detect zero or more faces in an image or OpenCV BGR frame."""
        if self._closed:
            raise FaceDetectorError("The face detector has already been closed.")
        rgb = _load_rgb(image)
        media_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        with self._detect_lock:
            if self._closed:
                raise FaceDetectorError("The face detector has already been closed.")
            result = self._detector.detect(media_image)

        image_height, image_width = rgb.shape[:2]
        detections: list[FaceDetection] = []
        for detection in result.detections or []:
            box = detection.bounding_box
            left = max(0, int(box.origin_x))
            top = max(0, int(box.origin_y))
            right = min(image_width, int(box.origin_x + box.width))
            bottom = min(image_height, int(box.origin_y + box.height))
            if right <= left or bottom <= top:
                continue

            categories = getattr(detection, "categories", None) or []
            confidence = float(categories[0].score) if categories else None
            detections.append(
                FaceDetection(
                    bounding_box=BoundingBox(
                        x=left,
                        y=top,
                        width=right - left,
                        height=bottom - top,
                    ),
                    confidence=confidence,
                )
            )
        return detections

    def close(self) -> None:
        """Release MediaPipe resources."""
        with self._detect_lock:
            if not self._closed:
                self._detector.close()
                self._closed = True

    def __enter__(self) -> FaceDetector:
        if self._closed:
            raise FaceDetectorError("The face detector has already been closed.")
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
