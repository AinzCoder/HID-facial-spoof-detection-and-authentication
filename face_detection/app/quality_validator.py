"""Heuristic image and face quality checks; this module does not classify spoofs."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Mapping, Sequence

import cv2
import numpy as np

from app.face_detector import BoundingBox, FaceDetection, ImageInput, _load_rgb
from app.landmark_extractor import FaceLandmarks

WARNING_CODES = frozenset(
    {
        "blur",
        "underexposed",
        "overexposed",
        "low_contrast",
        "low_resolution",
        "no_face_detected",
        "small_face",
        "clipped_face",
        "non_frontal",
    }
)

DEFAULT_SCORE_PENALTIES: dict[str, float] = {
    "blur": 20.0,
    "underexposed": 15.0,
    "overexposed": 15.0,
    "low_contrast": 10.0,
    "low_resolution": 15.0,
    "no_face_detected": 10.0,
    "small_face": 10.0,
    "clipped_face": 15.0,
    "non_frontal": 10.0,
}


class QualityValidationError(ValueError):
    """Raised when supplied face metadata is not valid for quality checks."""


@dataclass(frozen=True, slots=True)
class QualityValidatorConfig:
    """Configurable thresholds and policy for heuristic quality assessment.

    The defaults are starting heuristics, not universal cutoffs. Calibrate them
    with the target camera, capture conditions, and downstream model. Landmark
    estimates cannot guarantee that facial features are visible under occlusion.
    """

    min_blur_variance: float = 100.0
    min_brightness: float = 40.0
    max_brightness: float = 215.0
    min_contrast_stddev: float = 25.0
    min_width: int = 160
    min_height: int = 160
    min_face_area_ratio: float = 0.05
    max_abs_roll_degrees: float = 20.0
    max_abs_yaw_offset_ratio: float = 0.2
    reject_on_warnings: frozenset[str] = frozenset(
        {
            "blur",
            "underexposed",
            "overexposed",
            "low_contrast",
            "low_resolution",
            "no_face_detected",
            "small_face",
            "clipped_face",
            "non_frontal",
        }
    )
    score_penalties: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_SCORE_PENALTIES)
    )

    def __post_init__(self) -> None:
        numeric_thresholds = (
            ("min_blur_variance", self.min_blur_variance),
            ("min_brightness", self.min_brightness),
            ("max_brightness", self.max_brightness),
            ("min_contrast_stddev", self.min_contrast_stddev),
            ("min_face_area_ratio", self.min_face_area_ratio),
            ("max_abs_roll_degrees", self.max_abs_roll_degrees),
            ("max_abs_yaw_offset_ratio", self.max_abs_yaw_offset_ratio),
        )
        for name, value in numeric_thresholds:
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
        if self.min_blur_variance < 0:
            raise ValueError("min_blur_variance must be non-negative.")
        if not 0 <= self.min_brightness < self.max_brightness <= 255:
            raise ValueError("Brightness thresholds must satisfy 0 <= min < max <= 255.")
        if self.min_contrast_stddev < 0:
            raise ValueError("min_contrast_stddev must be non-negative.")
        if (
            isinstance(self.min_width, bool)
            or not isinstance(self.min_width, int)
            or isinstance(self.min_height, bool)
            or not isinstance(self.min_height, int)
            or self.min_width < 1
            or self.min_height < 1
        ):
            raise ValueError("Minimum image dimensions must be positive integers.")
        if not 0 <= self.min_face_area_ratio <= 1:
            raise ValueError("min_face_area_ratio must be between 0 and 1.")
        if self.max_abs_roll_degrees < 0 or self.max_abs_yaw_offset_ratio < 0:
            raise ValueError("Pose indicator limits must be non-negative.")
        unknown_warnings = self.reject_on_warnings - WARNING_CODES
        if unknown_warnings:
            raise ValueError(f"Unknown rejection warning codes: {sorted(unknown_warnings)}")
        if set(self.score_penalties) - WARNING_CODES:
            raise ValueError("score_penalties contains an unknown warning code.")
        if any(
            not math.isfinite(penalty) or penalty < 0
            for penalty in self.score_penalties.values()
        ):
            raise ValueError("Score penalties must be finite and non-negative.")


@dataclass(frozen=True, slots=True)
class PoseIndicators:
    """Approximate 2D roll and nose-offset indicators, not calibrated 3D pose."""

    face_index: int
    detection_index: int | None
    roll_angle_degrees: float
    yaw_offset_ratio: float
    is_approximately_frontal: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FaceQualityMetrics:
    """Quality measurements associated with one detected face."""

    detection_index: int
    bounding_box: BoundingBox
    face_area_ratio: float
    is_small: bool
    is_clipped: bool
    pose: PoseIndicators | None

    def to_dict(self) -> dict[str, object]:
        return {
            "detection_index": self.detection_index,
            "bounding_box": self.bounding_box.to_dict(),
            "face_area_ratio": self.face_area_ratio,
            "is_small": self.is_small,
            "is_clipped": self.is_clipped,
            "pose": self.pose.to_dict() if self.pose is not None else None,
        }


@dataclass(frozen=True, slots=True)
class ImageQualityMetrics:
    """Raw image metrics and their threshold flags."""

    width: int
    height: int
    blur_variance: float
    is_blurry: bool
    brightness_mean: float
    is_underexposed: bool
    is_overexposed: bool
    contrast_stddev: float
    is_low_contrast: bool
    is_low_resolution: bool
    faces: tuple[FaceQualityMetrics, ...]
    pose_indicators: tuple[PoseIndicators, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "width": self.width,
            "height": self.height,
            "blur_variance": self.blur_variance,
            "is_blurry": self.is_blurry,
            "brightness_mean": self.brightness_mean,
            "is_underexposed": self.is_underexposed,
            "is_overexposed": self.is_overexposed,
            "contrast_stddev": self.contrast_stddev,
            "is_low_contrast": self.is_low_contrast,
            "is_low_resolution": self.is_low_resolution,
            "faces": [face.to_dict() for face in self.faces],
            "pose_indicators": [pose.to_dict() for pose in self.pose_indicators],
        }


@dataclass(frozen=True, slots=True)
class QualityAssessment:
    """Quality flags, configurable validity decision, and bounded heuristic score."""

    valid: bool
    quality_score: float
    warning_flags: tuple[str, ...]
    rejection_reasons: tuple[str, ...]
    metrics: ImageQualityMetrics

    def to_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "quality_score": self.quality_score,
            "warning_flags": list(self.warning_flags),
            "rejection_reasons": list(self.rejection_reasons),
            "metrics": self.metrics.to_dict(),
        }


def _pose_indicators(
    face: FaceLandmarks,
    config: QualityValidatorConfig,
) -> PoseIndicators | None:
    left_eye = face.regions.get("left_eye", ())
    right_eye = face.regions.get("right_eye", ())
    nose = next((point for point in face.landmarks if point.index == 1), None)
    if not left_eye or not right_eye or nose is None:
        return None

    left_x = sum(point.pixel.x for point in left_eye) / len(left_eye)
    left_y = sum(point.pixel.y for point in left_eye) / len(left_eye)
    right_x = sum(point.pixel.x for point in right_eye) / len(right_eye)
    right_y = sum(point.pixel.y for point in right_eye) / len(right_eye)
    eye_distance = math.hypot(right_x - left_x, right_y - left_y)
    if eye_distance == 0:
        return None

    eye_line_angle = math.degrees(math.atan2(right_y - left_y, right_x - left_x))
    roll = (eye_line_angle + 90.0) % 180.0 - 90.0
    eye_midpoint_x = (left_x + right_x) / 2
    yaw_offset_ratio = (nose.pixel.x - eye_midpoint_x) / eye_distance
    is_frontal = (
        abs(roll) <= config.max_abs_roll_degrees
        and abs(yaw_offset_ratio) <= config.max_abs_yaw_offset_ratio
    )
    return PoseIndicators(
        face_index=face.face_index,
        detection_index=face.detection_index,
        roll_angle_degrees=roll,
        yaw_offset_ratio=yaw_offset_ratio,
        is_approximately_frontal=is_frontal,
    )


def _face_metrics(
    detections: Sequence[FaceDetection],
    poses_by_detection: Mapping[int, PoseIndicators],
    image_width: int,
    image_height: int,
    config: QualityValidatorConfig,
) -> tuple[FaceQualityMetrics, ...]:
    image_area = image_width * image_height
    results: list[FaceQualityMetrics] = []

    for detection_index, detection in enumerate(detections):
        box = detection.bounding_box
        if box.width <= 0 or box.height <= 0:
            raise QualityValidationError(
                f"Face detection {detection_index} has a non-positive bounding-box size."
            )
        face_area_ratio = (box.width * box.height) / image_area
        is_clipped = (
            box.x <= 0
            or box.y <= 0
            or box.x2 >= image_width
            or box.y2 >= image_height
        )
        results.append(
            FaceQualityMetrics(
                detection_index=detection_index,
                bounding_box=box,
                face_area_ratio=face_area_ratio,
                is_small=face_area_ratio < config.min_face_area_ratio,
                is_clipped=is_clipped,
                pose=poses_by_detection.get(detection_index),
            )
        )
    return tuple(results)


class QualityValidator:
    """Compute non-destructive image and per-face quality heuristics.

    This validator assesses capture quality only. It never labels an image as
    genuine or spoofed, and possible presentation attacks are not rejection
    criteria.
    """

    def __init__(self, config: QualityValidatorConfig | None = None) -> None:
        self.config = config or QualityValidatorConfig()

    def validate(
        self,
        image: ImageInput,
        *,
        detections: Sequence[FaceDetection] | None = None,
        landmarks: Sequence[FaceLandmarks] | None = None,
    ) -> QualityAssessment:
        """Validate image quality without modifying the supplied image."""
        rgb = _load_rgb(image)
        height, width = rgb.shape[:2]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness_mean = float(np.mean(gray))
        contrast_stddev = float(np.std(gray))
        is_blurry = blur_variance < self.config.min_blur_variance
        is_underexposed = brightness_mean < self.config.min_brightness
        is_overexposed = brightness_mean > self.config.max_brightness
        is_low_contrast = contrast_stddev < self.config.min_contrast_stddev
        is_low_resolution = (
            width < self.config.min_width or height < self.config.min_height
        )

        pose_indicators = tuple(
            pose
            for face in landmarks or ()
            if (pose := _pose_indicators(face, self.config)) is not None
        )
        poses_by_detection = {
            pose.detection_index: pose
            for pose in pose_indicators
            if pose.detection_index is not None
        }
        face_metrics = _face_metrics(
            detections or (),
            poses_by_detection,
            width,
            height,
            self.config,
        )
        warning_flags: list[str] = []
        if is_blurry:
            warning_flags.append("blur")
        if is_underexposed:
            warning_flags.append("underexposed")
        if is_overexposed:
            warning_flags.append("overexposed")
        if is_low_contrast:
            warning_flags.append("low_contrast")
        if is_low_resolution:
            warning_flags.append("low_resolution")
        if detections is not None and not detections:
            warning_flags.append("no_face_detected")
        if any(face.is_small for face in face_metrics):
            warning_flags.append("small_face")
        if any(face.is_clipped for face in face_metrics):
            warning_flags.append("clipped_face")
        if any(not pose.is_approximately_frontal for pose in pose_indicators):
            warning_flags.append("non_frontal")

        rejection_reasons = tuple(
            warning
            for warning in warning_flags
            if warning in self.config.reject_on_warnings
        )
        penalty = sum(
            self.config.score_penalties.get(
                warning, DEFAULT_SCORE_PENALTIES[warning]
            )
            for warning in warning_flags
        )
        quality_score = max(0.0, min(100.0, 100.0 - penalty))
        metrics = ImageQualityMetrics(
            width=width,
            height=height,
            blur_variance=blur_variance,
            is_blurry=is_blurry,
            brightness_mean=brightness_mean,
            is_underexposed=is_underexposed,
            is_overexposed=is_overexposed,
            contrast_stddev=contrast_stddev,
            is_low_contrast=is_low_contrast,
            is_low_resolution=is_low_resolution,
            faces=face_metrics,
            pose_indicators=pose_indicators,
        )
        return QualityAssessment(
            valid=not rejection_reasons,
            quality_score=quality_score,
            warning_flags=tuple(warning_flags),
            rejection_reasons=rejection_reasons,
            metrics=metrics,
        )


__all__ = [
    "FaceQualityMetrics",
    "ImageQualityMetrics",
    "PoseIndicators",
    "QualityAssessment",
    "QualityValidationError",
    "QualityValidator",
    "QualityValidatorConfig",
]
