"""Context-preserving face crops from MediaPipe face detections."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

import cv2
import numpy as np

from app.face_detector import BoundingBox, FaceDetection, ImageInput, _load_rgb

ResizeInterpolation = Literal["auto", "area", "linear", "cubic", "nearest"]

_INTERPOLATION_FLAGS: dict[str, int] = {
    "area": cv2.INTER_AREA,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "nearest": cv2.INTER_NEAREST,
}


class FaceCropError(ValueError):
    """Raised when a face detection cannot produce a valid image crop."""


@dataclass(frozen=True, slots=True)
class FaceCropperConfig:
    """Face crop settings.

    ``target_size`` is (width, height); ``None`` disables resizing. ``area`` is
    generally suitable for shrinking and ``linear`` or ``cubic`` for enlarging.
    Resampling can alter or remove high-frequency presentation-attack details;
    use ``original_crop`` when pixel-level evidence must be retained.
    """

    padding_ratio: float = 0.20
    target_size: tuple[int, int] | None = (224, 224)
    interpolation: ResizeInterpolation = "auto"

    def __post_init__(self) -> None:
        if not math.isfinite(self.padding_ratio) or self.padding_ratio < 0:
            raise ValueError("padding_ratio must be finite and non-negative.")
        if self.target_size is not None:
            if (
                len(self.target_size) != 2
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value <= 0
                    for value in self.target_size
                )
            ):
                raise ValueError("target_size must contain positive integer dimensions.")
        if self.interpolation not in {
            "auto",
            "area",
            "linear",
            "cubic",
            "nearest",
        }:
            raise ValueError(
                "interpolation must be auto, area, linear, cubic, or nearest."
            )


@dataclass(frozen=True, slots=True)
class FaceCropResult:
    """Original-resolution and optional resized crop with source metadata.

    Crop-dimension tuples use ``(height, width)``, matching NumPy array shapes.
    """

    face_id: int
    original_bounding_box: BoundingBox
    padded_bounding_box: BoundingBox
    crop_coordinates: BoundingBox
    original_crop: np.ndarray
    resized_crop: np.ndarray | None
    original_crop_dimensions: tuple[int, int]
    output_dimensions: tuple[int, int]
    padding_ratio: float
    clipping_flag: bool


def _validate_box(box: BoundingBox, face_id: int) -> None:
    values = (box.x, box.y, box.width, box.height)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise FaceCropError(f"Face {face_id} bounding-box coordinates must be integers.")
    if box.width <= 0 or box.height <= 0:
        raise FaceCropError(f"Face {face_id} has a zero-area or invalid bounding box.")


def _padded_box(box: BoundingBox, padding_ratio: float) -> BoundingBox:
    padding_x = math.ceil(box.width * padding_ratio)
    padding_y = math.ceil(box.height * padding_ratio)
    left = box.x - padding_x
    top = box.y - padding_y
    right = box.x2 + padding_x
    bottom = box.y2 + padding_y
    return BoundingBox(left, top, right - left, bottom - top)


def _resize_interpolation(
    interpolation: ResizeInterpolation,
    crop_width: int,
    crop_height: int,
    target_width: int,
    target_height: int,
) -> int:
    if interpolation == "auto":
        if target_width <= crop_width and target_height <= crop_height:
            return cv2.INTER_AREA
        return cv2.INTER_LINEAR
    return _INTERPOLATION_FLAGS[interpolation]


class FaceCropper:
    """Crop faces in detection order while preserving original RGB pixels.

    NumPy inputs follow the existing OpenCV BGR convention; returned crops are
    RGB uint8 arrays.
    """

    def __init__(self, config: FaceCropperConfig | None = None) -> None:
        self.config = config or FaceCropperConfig()

    def crop_faces(
        self,
        image: ImageInput,
        detections: Sequence[FaceDetection],
    ) -> list[FaceCropResult]:
        """Return crops in input detection order; ``face_id`` is its index."""
        rgb = _load_rgb(image)
        image_height, image_width = rgb.shape[:2]
        results: list[FaceCropResult] = []

        for face_id, detection in enumerate(detections):
            original_box = detection.bounding_box
            _validate_box(original_box, face_id)
            padded_box = _padded_box(original_box, self.config.padding_ratio)

            left = max(0, padded_box.x)
            top = max(0, padded_box.y)
            right = min(image_width, padded_box.x2)
            bottom = min(image_height, padded_box.y2)
            if right <= left or bottom <= top:
                raise FaceCropError(
                    f"Face {face_id} bounding box does not intersect the image."
                )

            crop_coordinates = BoundingBox(left, top, right - left, bottom - top)
            clipping_flag = (
                padded_box.x < 0
                or padded_box.y < 0
                or padded_box.x2 > image_width
                or padded_box.y2 > image_height
                or original_box.x <= 0
                or original_box.y <= 0
                or original_box.x2 >= image_width
                or original_box.y2 >= image_height
            )
            original_crop = np.array(
                rgb[top:bottom, left:right],
                dtype=np.uint8,
                copy=True,
                order="C",
            )
            original_dimensions = original_crop.shape[:2]

            resized_crop: np.ndarray | None = None
            if self.config.target_size is not None:
                target_width, target_height = self.config.target_size
                interpolation = _resize_interpolation(
                    self.config.interpolation,
                    original_crop.shape[1],
                    original_crop.shape[0],
                    target_width,
                    target_height,
                )
                resized_crop = cv2.resize(
                    original_crop,
                    (target_width, target_height),
                    interpolation=interpolation,
                )
            output_dimensions = (
                resized_crop.shape[:2]
                if resized_crop is not None
                else original_dimensions
            )
            results.append(
                FaceCropResult(
                    face_id=face_id,
                    original_bounding_box=original_box,
                    padded_bounding_box=padded_box,
                    crop_coordinates=crop_coordinates,
                    original_crop=original_crop,
                    resized_crop=resized_crop,
                    original_crop_dimensions=original_dimensions,
                    output_dimensions=output_dimensions,
                    padding_ratio=self.config.padding_ratio,
                    clipping_flag=clipping_flag,
                )
            )
        return results

    def crop_face(
        self,
        image: ImageInput,
        detection: FaceDetection,
        *,
        face_id: int = 0,
    ) -> FaceCropResult:
        """Crop one detection, optionally assigning its upstream detection index."""
        if isinstance(face_id, bool) or not isinstance(face_id, int) or face_id < 0:
            raise ValueError("face_id must be a non-negative integer.")
        result = self.crop_faces(image, [detection])[0]
        if face_id == result.face_id:
            return result
        return FaceCropResult(
            face_id=face_id,
            original_bounding_box=result.original_bounding_box,
            padded_bounding_box=result.padded_bounding_box,
            crop_coordinates=result.crop_coordinates,
            original_crop=result.original_crop,
            resized_crop=result.resized_crop,
            original_crop_dimensions=result.original_crop_dimensions,
            output_dimensions=result.output_dimensions,
            padding_ratio=result.padding_ratio,
            clipping_flag=result.clipping_flag,
        )


__all__ = [
    "FaceCropError",
    "FaceCropper",
    "FaceCropperConfig",
    "FaceCropResult",
    "ResizeInterpolation",
]
