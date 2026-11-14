"""Facial landmark extraction using the MediaPipe Tasks Face Landmarker API."""

from __future__ import annotations

import logging
import math
import os
import shutil
import tempfile
import threading
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import mediapipe as mp

from app.face_detector import (
    BoundingBox,
    FaceDetection,
    ImageInput,
    InvalidImageError,
    _load_rgb,
)

LOGGER = logging.getLogger(__name__)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_FILENAME = "face_landmarker.task"

FACIAL_REGION_INDICES: dict[str, tuple[int, ...]] = {
    "left_eye": (
        362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,
        384, 398, 473, 474, 475, 476, 477,
    ),
    "right_eye": (
        33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161,
        246, 468, 469, 470, 471, 472,
    ),
    "nose": (1, 2, 4, 5, 6, 97, 98, 195, 197, 326, 327),
    "mouth": (
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402,
        317, 14, 87, 178, 88, 95, 78,
    ),
}


class LandmarkExtractorError(RuntimeError):
    """Raised when the MediaPipe Face Landmarker cannot be initialized or used."""


@dataclass(frozen=True, slots=True)
class NormalizedCoordinate:
    """A MediaPipe normalized face landmark, including relative depth."""

    x: float
    y: float
    z: float


@dataclass(frozen=True, slots=True)
class PixelCoordinate:
    """A landmark's x/y location in source-image pixel coordinates."""

    x: float
    y: float


@dataclass(frozen=True, slots=True)
class LandmarkPoint:
    """One indexed face landmark in normalized and pixel coordinates."""

    index: int
    normalized: NormalizedCoordinate
    pixel: PixelCoordinate

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "normalized": asdict(self.normalized),
            "pixel": asdict(self.pixel),
        }


@dataclass(frozen=True, slots=True)
class FaceLandmarks:
    """All landmarks for one face and its semantic region subsets."""

    face_index: int
    detection_index: int | None
    landmarks: tuple[LandmarkPoint, ...]
    regions: dict[str, tuple[LandmarkPoint, ...]]

    def to_dict(self) -> dict[str, object]:
        return {
            "face_index": self.face_index,
            "detection_index": self.detection_index,
            "landmarks": [point.to_dict() for point in self.landmarks],
            "regions": {
                name: [point.to_dict() for point in points]
                for name, points in self.regions.items()
            },
        }


@dataclass(frozen=True, slots=True)
class LandmarkExtractorConfig:
    """Configuration for face landmark extraction."""

    num_faces: int = 5
    min_face_detection_confidence: float = 0.5
    min_face_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    def __post_init__(self) -> None:
        if (
            isinstance(self.num_faces, bool)
            or not isinstance(self.num_faces, int)
            or self.num_faces < 1
        ):
            raise ValueError("num_faces must be a positive integer.")
        for name in (
            "min_face_detection_confidence",
            "min_face_presence_confidence",
            "min_tracking_confidence",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")


class _TaskLandmarker(Protocol):
    def detect(self, image: mp.Image) -> Any: ...

    def close(self) -> None: ...


def _default_model_path() -> Path:
    cache_root = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / ".cache")))
    return cache_root / "face_preprocessing" / "models" / MODEL_FILENAME


def _ensure_model(model_path: Path, *, download: bool) -> Path:
    if model_path.is_file() and model_path.stat().st_size > 0:
        return model_path
    if not download:
        raise LandmarkExtractorError(f"MediaPipe model file was not found: {model_path}")

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
            raise LandmarkExtractorError("The downloaded Face Landmarker model is empty.")
        os.replace(temporary_path, model_path)
    except (OSError, urllib.error.URLError) as exc:
        raise LandmarkExtractorError(
            f"Could not download the Face Landmarker model from {MODEL_URL}."
        ) from exc
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    LOGGER.info("Downloaded MediaPipe face landmark model to %s", model_path)
    return model_path


def _landmark_box(
    landmarks: Sequence[Any], image_width: int, image_height: int
) -> BoundingBox:
    xs = [float(point.x) * image_width for point in landmarks]
    ys = [float(point.y) * image_height for point in landmarks]
    left = max(0, math.floor(min(xs)))
    top = max(0, math.floor(min(ys)))
    right = min(image_width, math.ceil(max(xs)))
    bottom = min(image_height, math.ceil(max(ys)))
    return BoundingBox(
        x=left,
        y=top,
        width=max(0, right - left),
        height=max(0, bottom - top),
    )


def _intersection_over_union(first: BoundingBox, second: BoundingBox) -> float:
    left = max(first.x, second.x)
    top = max(first.y, second.y)
    right = min(first.x2, second.x2)
    bottom = min(first.y2, second.y2)
    intersection = max(0, right - left) * max(0, bottom - top)
    union = first.width * first.height + second.width * second.height - intersection
    return intersection / union if union > 0 else 0.0


def _associate_detections(
    landmark_boxes: Sequence[BoundingBox],
    detections: Sequence[FaceDetection],
) -> list[int | None]:
    candidates = [
        (_intersection_over_union(landmark_box, detection.bounding_box), face_index, detection_index)
        for face_index, landmark_box in enumerate(landmark_boxes)
        for detection_index, detection in enumerate(detections)
    ]
    candidates.sort(reverse=True)

    associations: list[int | None] = [None] * len(landmark_boxes)
    assigned_detections: set[int] = set()
    for overlap, face_index, detection_index in candidates:
        if overlap <= 0.0:
            break
        if associations[face_index] is None and detection_index not in assigned_detections:
            associations[face_index] = detection_index
            assigned_detections.add(detection_index)
    return associations


class LandmarkExtractor:
    """Reusable MediaPipe Tasks face landmarker for still images and frames."""

    _model_download_lock = threading.Lock()

    def __init__(
        self,
        model_path: str | os.PathLike[str] | None = None,
        *,
        config: LandmarkExtractorConfig | None = None,
    ) -> None:
        self.config = config or LandmarkExtractorConfig()
        self._closed = False
        self._detect_lock = threading.Lock()

        model_was_provided = model_path is not None
        resolved_model_path = (
            Path(model_path) if model_was_provided else _default_model_path()
        )
        with self._model_download_lock:
            resolved_model_path = _ensure_model(
                resolved_model_path, download=not model_was_provided
            )
        self._landmarker = self._build_landmarker(resolved_model_path)

    def _build_landmarker(self, model_path: Path) -> _TaskLandmarker:
        try:
            from mediapipe.tasks.python import BaseOptions, vision

            options = vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=self.config.num_faces,
                min_face_detection_confidence=self.config.min_face_detection_confidence,
                min_face_presence_confidence=self.config.min_face_presence_confidence,
                min_tracking_confidence=self.config.min_tracking_confidence,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            return vision.FaceLandmarker.create_from_options(options)
        except (ImportError, OSError, ValueError, RuntimeError) as exc:
            raise LandmarkExtractorError(
                "MediaPipe Tasks could not initialize the face landmarker."
            ) from exc

    def extract(
        self,
        image: ImageInput,
        *,
        detections: Sequence[FaceDetection] | None = None,
    ) -> list[FaceLandmarks]:
        """Extract per-face landmarks and optionally match detector results.

        Detection associations use one-to-one greedy IoU matches, considering
        the highest-overlap mesh and detector boxes first.
        """
        if self._closed:
            raise LandmarkExtractorError("The face landmarker has already been closed.")
        rgb = _load_rgb(image)
        media_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        with self._detect_lock:
            if self._closed:
                raise LandmarkExtractorError(
                    "The face landmarker has already been closed."
                )
            result = self._landmarker.detect(media_image)

        raw_faces = result.face_landmarks or []
        image_height, image_width = rgb.shape[:2]
        landmark_boxes = [
            _landmark_box(face, image_width, image_height) for face in raw_faces
        ]
        associations = (
            _associate_detections(landmark_boxes, detections)
            if detections is not None
            else [None] * len(raw_faces)
        )

        faces: list[FaceLandmarks] = []
        for face_index, raw_landmarks in enumerate(raw_faces):
            points = tuple(
                LandmarkPoint(
                    index=index,
                    normalized=NormalizedCoordinate(
                        x=float(point.x),
                        y=float(point.y),
                        z=float(point.z),
                    ),
                    pixel=PixelCoordinate(
                        x=float(point.x) * image_width,
                        y=float(point.y) * image_height,
                    ),
                )
                for index, point in enumerate(raw_landmarks)
            )
            point_by_index = {point.index: point for point in points}
            regions = {
                region_name: tuple(
                    point_by_index[index]
                    for index in indices
                    if index in point_by_index
                )
                for region_name, indices in FACIAL_REGION_INDICES.items()
            }
            faces.append(
                FaceLandmarks(
                    face_index=face_index,
                    detection_index=associations[face_index],
                    landmarks=points,
                    regions=regions,
                )
            )
        return faces

    def close(self) -> None:
        """Release MediaPipe resources."""
        with self._detect_lock:
            if not self._closed:
                self._landmarker.close()
                self._closed = True

    def __enter__(self) -> LandmarkExtractor:
        if self._closed:
            raise LandmarkExtractorError(
                "The face landmarker has already been closed."
            )
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = [
    "FACIAL_REGION_INDICES",
    "FaceLandmarks",
    "InvalidImageError",
    "LandmarkExtractor",
    "LandmarkExtractorConfig",
    "LandmarkExtractorError",
    "LandmarkPoint",
    "NormalizedCoordinate",
    "PixelCoordinate",
]
