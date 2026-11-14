"""Optional landmark-based face alignment that preserves the source crop."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from app.face_cropper import FaceCropResult
from app.landmark_extractor import FaceLandmarks


class FaceAlignmentError(ValueError):
    """Raised when a source crop cannot be used for face alignment."""


@dataclass(frozen=True, slots=True)
class FaceAlignerConfig:
    """Alignment settings; alignment is disabled unless explicitly enabled.

    ``output_size`` is (width, height). Alignment uses one affine resampling
    pass with OpenCV linear interpolation. The original crop is retained
    unchanged because any interpolation may alter fine image details.
    """

    enabled: bool = False
    output_size: tuple[int, int] = (224, 224)

    def __post_init__(self) -> None:
        if (
            len(self.output_size) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in self.output_size
            )
        ):
            raise ValueError("output_size must contain positive integer dimensions.")


@dataclass(frozen=True, slots=True)
class FaceAlignmentResult:
    """Unaligned source crop, optional aligned crop, and transform metadata."""

    face_id: int
    detection_index: int | None
    original_crop: np.ndarray
    aligned_face_crop: np.ndarray | None
    alignment_applied: bool
    rotation_angle_degrees: float | None
    transformation_matrix: tuple[tuple[float, float, float], ...] | None
    original_crop_dimensions: tuple[int, int]
    aligned_crop_dimensions: tuple[int, int] | None
    warning_flags: tuple[str, ...]


def _validate_crop(crop: FaceCropResult) -> tuple[int, int]:
    image = crop.original_crop
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or image.shape[0] == 0
        or image.shape[1] == 0
    ):
        raise FaceAlignmentError("The original crop must be a non-empty RGB uint8 array.")
    return image.shape[:2]


def _center_of_eye(
    face: FaceLandmarks,
    region_name: str,
    crop: FaceCropResult,
) -> tuple[float, float] | None:
    points = face.regions.get(region_name, ())
    if not points:
        return None

    crop_x = crop.crop_coordinates.x
    crop_y = crop.crop_coordinates.y
    try:
        local_coordinates = [
            (float(point.pixel.x) - crop_x, float(point.pixel.y) - crop_y)
            for point in points
        ]
    except (TypeError, ValueError, OverflowError):
        return None
    if any(not math.isfinite(x) or not math.isfinite(y) for x, y in local_coordinates):
        return None
    return (
        sum(x for x, _ in local_coordinates) / len(local_coordinates),
        sum(y for _, y in local_coordinates) / len(local_coordinates),
    )


def _expanded_affine(
    width: int,
    height: int,
    center: tuple[float, float],
    angle_degrees: float,
    output_size: tuple[int, int],
) -> tuple[np.ndarray, tuple[int, int]]:
    matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    corners = np.array(
        [
            [0.0, 0.0],
            [width - 1.0, 0.0],
            [width - 1.0, height - 1.0],
            [0.0, height - 1.0],
        ],
        dtype=np.float64,
    )
    rotated_corners = corners @ matrix[:, :2].T + matrix[:, 2]
    left = math.floor(float(np.min(rotated_corners[:, 0])))
    top = math.floor(float(np.min(rotated_corners[:, 1])))
    right = math.ceil(float(np.max(rotated_corners[:, 0])))
    bottom = math.ceil(float(np.max(rotated_corners[:, 1])))
    expanded_width = max(1, right - left + 1)
    expanded_height = max(1, bottom - top + 1)
    matrix[0, 2] -= left
    matrix[1, 2] -= top

    output_width, output_height = output_size
    scale_x = output_width / expanded_width
    scale_y = output_height / expanded_height
    matrix[0, :] *= scale_x
    matrix[1, :] *= scale_y
    return matrix, (expanded_height, expanded_width)


class FaceAligner:
    """Align context crops using their associated MediaPipe eye landmarks."""

    def __init__(self, config: FaceAlignerConfig | None = None) -> None:
        self.config = config or FaceAlignerConfig()

    def _result_without_alignment(
        self,
        crop: FaceCropResult,
        dimensions: tuple[int, int],
        warnings: Sequence[str],
    ) -> FaceAlignmentResult:
        warning_flags = list(warnings)
        if crop.clipping_flag:
            warning_flags.append("source_crop_clipped")
        return FaceAlignmentResult(
            face_id=crop.face_id,
            detection_index=crop.face_id,
            original_crop=crop.original_crop,
            aligned_face_crop=None,
            alignment_applied=False,
            rotation_angle_degrees=None,
            transformation_matrix=None,
            original_crop_dimensions=dimensions,
            aligned_crop_dimensions=None,
            warning_flags=tuple(dict.fromkeys(warning_flags)),
        )

    def align_face(
        self,
        crop: FaceCropResult,
        landmarks: FaceLandmarks | None,
    ) -> FaceAlignmentResult:
        """Align one crop using landmarks from the same source image.

        For multiple faces, use :meth:`align_faces` so matching is performed by
        detection index rather than assumed list order.
        """
        dimensions = _validate_crop(crop)
        if not self.config.enabled:
            return self._result_without_alignment(
                crop, dimensions, ("alignment_disabled",)
            )
        if landmarks is None:
            return self._result_without_alignment(
                crop, dimensions, ("missing_landmarks",)
            )
        if (
            landmarks.detection_index is not None
            and landmarks.detection_index != crop.face_id
        ):
            return self._result_without_alignment(
                crop, dimensions, ("detection_index_mismatch",)
            )

        left_eye = _center_of_eye(landmarks, "left_eye", crop)
        right_eye = _center_of_eye(landmarks, "right_eye", crop)
        if left_eye is None or right_eye is None:
            return self._result_without_alignment(
                crop, dimensions, ("invalid_eye_landmarks",)
            )
        height, width = dimensions
        if any(
            x < 0 or y < 0 or x >= width or y >= height
            for x, y in (left_eye, right_eye)
        ):
            return self._result_without_alignment(
                crop, dimensions, ("eye_landmarks_outside_crop",)
            )

        delta_x = right_eye[0] - left_eye[0]
        delta_y = right_eye[1] - left_eye[1]
        if math.hypot(delta_x, delta_y) <= 1e-6:
            return self._result_without_alignment(
                crop, dimensions, ("invalid_eye_landmarks",)
            )

        eye_line_angle = math.degrees(math.atan2(delta_y, delta_x))
        rotation_angle = (eye_line_angle + 90.0) % 180.0 - 90.0
        eye_center = (
            (left_eye[0] + right_eye[0]) / 2.0,
            (left_eye[1] + right_eye[1]) / 2.0,
        )
        matrix, expanded_dimensions = _expanded_affine(
            width,
            height,
            eye_center,
            rotation_angle,
            self.config.output_size,
        )
        output_width, output_height = self.config.output_size
        aligned_crop = cv2.warpAffine(
            crop.original_crop,
            matrix,
            (output_width, output_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        warnings: list[str] = []
        if crop.clipping_flag:
            warnings.append("source_crop_clipped")
        if expanded_dimensions != dimensions:
            warnings.append("rotation_canvas_expanded")

        return FaceAlignmentResult(
            face_id=crop.face_id,
            detection_index=crop.face_id,
            original_crop=crop.original_crop,
            aligned_face_crop=aligned_crop,
            alignment_applied=True,
            rotation_angle_degrees=rotation_angle,
            transformation_matrix=tuple(
                tuple(float(value) for value in row) for row in matrix
            ),
            original_crop_dimensions=dimensions,
            aligned_crop_dimensions=aligned_crop.shape[:2],
            warning_flags=tuple(warnings),
        )

    def align_faces(
        self,
        crops: Sequence[FaceCropResult],
        landmarks: Sequence[FaceLandmarks],
    ) -> list[FaceAlignmentResult]:
        """Align crops in cropper order using explicit detection-index matches."""
        landmarks_by_detection: dict[int, list[FaceLandmarks]] = {}
        for face in landmarks:
            if face.detection_index is not None:
                landmarks_by_detection.setdefault(face.detection_index, []).append(face)

        results: list[FaceAlignmentResult] = []
        for crop in crops:
            associated_landmarks = landmarks_by_detection.get(crop.face_id, [])
            if len(associated_landmarks) > 1:
                dimensions = _validate_crop(crop)
                results.append(
                    self._result_without_alignment(
                        crop, dimensions, ("ambiguous_landmark_association",)
                    )
                )
                continue
            results.append(
                self.align_face(
                    crop,
                    associated_landmarks[0] if associated_landmarks else None,
                )
            )
        return results


__all__ = [
    "FaceAligner",
    "FaceAlignerConfig",
    "FaceAlignmentError",
    "FaceAlignmentResult",
]
