from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.face_detector import BoundingBox, FaceDetection, InvalidImageError
from app.landmark_extractor import (
    FaceLandmarks,
    LandmarkPoint,
    NormalizedCoordinate,
    PixelCoordinate,
)
from app.quality_validator import (
    QualityValidationError,
    QualityValidator,
    QualityValidatorConfig,
)


def _face_landmarks(
    detection_index: int,
    *,
    left_eye: tuple[float, float] = (30, 30),
    right_eye: tuple[float, float] = (70, 30),
    nose: tuple[float, float] = (50, 45),
) -> FaceLandmarks:
    def point(index: int, x: float, y: float) -> LandmarkPoint:
        return LandmarkPoint(
            index=index,
            normalized=NormalizedCoordinate(x / 100, y / 100, 0),
            pixel=PixelCoordinate(x, y),
        )

    points = (
        point(1, *nose),
        point(33, *right_eye),
        point(362, *left_eye),
    )
    return FaceLandmarks(
        face_index=detection_index,
        detection_index=detection_index,
        landmarks=points,
        regions={
            "left_eye": (points[2],),
            "right_eye": (points[1],),
            "nose": (points[0],),
            "mouth": (),
        },
    )


def _detection(
    x: int = 20, y: int = 20, width: int = 60, height: int = 60
) -> FaceDetection:
    return FaceDetection(BoundingBox(x, y, width, height), confidence=0.9)


def test_sharp_image_has_higher_blur_metric_than_blurred_image() -> None:
    sharp = np.zeros((200, 200, 3), dtype=np.uint8)
    sharp[40:160:4, :, :] = 255
    sharp[:, 40:160:4, :] = 255
    blurred = cv2.GaussianBlur(sharp, (51, 51), 12)

    sharp_result = QualityValidator().validate(sharp)
    blurred_result = QualityValidator().validate(blurred)

    assert sharp_result.metrics.blur_variance > blurred_result.metrics.blur_variance
    assert not sharp_result.metrics.is_blurry
    assert blurred_result.metrics.is_blurry
    assert "blur" in blurred_result.warning_flags
    permissive_blur_threshold = QualityValidator(
        QualityValidatorConfig(min_blur_variance=0)
    ).validate(blurred)
    assert not permissive_blur_threshold.metrics.is_blurry


def test_dark_and_bright_images_are_flagged() -> None:
    validator = QualityValidator(
        QualityValidatorConfig(min_width=1, min_height=1, min_contrast_stddev=0)
    )

    dark = validator.validate(np.zeros((40, 40, 3), dtype=np.uint8))
    bright = validator.validate(np.full((40, 40, 3), 255, dtype=np.uint8))

    assert dark.metrics.is_underexposed
    assert "underexposed" in dark.warning_flags
    assert bright.metrics.is_overexposed
    assert "overexposed" in bright.warning_flags


def test_low_resolution_and_low_contrast_are_reported() -> None:
    result = QualityValidator().validate(np.full((30, 40, 3), 100, dtype=np.uint8))

    assert result.metrics.is_low_resolution
    assert result.metrics.is_low_contrast
    assert "low_resolution" in result.warning_flags
    assert "low_contrast" in result.warning_flags


def test_small_face_area_is_reported() -> None:
    image = np.full((200, 200, 3), 120, dtype=np.uint8)
    result = QualityValidator().validate(
        image,
        detections=[_detection(10, 10, 10, 10)],
    )

    face = result.metrics.faces[0]
    assert face.face_area_ratio == pytest.approx(0.0025)
    assert face.is_small
    assert "small_face" in result.warning_flags


@pytest.mark.parametrize(
    "box",
    [
        BoundingBox(0, 20, 40, 40),
        BoundingBox(20, 0, 40, 40),
        BoundingBox(60, 20, 40, 40),
        BoundingBox(20, 60, 40, 40),
        BoundingBox(-5, 20, 40, 40),
    ],
)
def test_touching_or_outside_face_box_is_flagged(box: BoundingBox) -> None:
    image = np.full((100, 100, 3), 120, dtype=np.uint8)
    result = QualityValidator(
        QualityValidatorConfig(min_width=1, min_height=1)
    ).validate(image, detections=[FaceDetection(box, 0.9)])

    assert result.metrics.faces[0].is_clipped
    assert "clipped_face" in result.warning_flags


def test_pose_indicators_use_landmarks_matched_by_detection_index() -> None:
    image = np.full((100, 100, 3), 120, dtype=np.uint8)
    detector_box = FaceDetection(BoundingBox(10, 10, 80, 80), 0.9)
    frontal_landmarks = _face_landmarks(0)
    rotated_landmarks = _face_landmarks(
        1,
        left_eye=(30, 40),
        right_eye=(70, 20),
        nose=(50, 30),
    )

    result = QualityValidator(
        QualityValidatorConfig(
            min_width=1,
            min_height=1,
            min_contrast_stddev=0,
            max_abs_roll_degrees=10,
        )
    ).validate(
        image,
        detections=[detector_box],
        landmarks=[rotated_landmarks, frontal_landmarks],
    )

    face = result.metrics.faces[0]
    assert face.pose is not None
    assert face.pose.roll_angle_degrees == pytest.approx(0)
    assert face.pose.is_approximately_frontal


def test_pose_roll_is_stable_when_eye_landmark_groups_are_reversed() -> None:
    image = np.full((100, 100, 3), 120, dtype=np.uint8)
    reversed_eye_order = _face_landmarks(
        0,
        left_eye=(70, 30),
        right_eye=(30, 30),
        nose=(50, 45),
    )
    result = QualityValidator(
        QualityValidatorConfig(
            min_width=1,
            min_height=1,
            min_contrast_stddev=0,
        )
    ).validate(image, landmarks=[reversed_eye_order])

    pose = result.metrics.pose_indicators[0]
    assert pose.roll_angle_degrees == pytest.approx(0)
    assert pose.is_approximately_frontal


def test_non_frontal_pose_warning_is_reported() -> None:
    image = np.full((100, 100, 3), 120, dtype=np.uint8)
    landmarks = _face_landmarks(
        0,
        left_eye=(30, 40),
        right_eye=(70, 20),
        nose=(65, 30),
    )
    result = QualityValidator(
        QualityValidatorConfig(
            min_width=1,
            min_height=1,
            min_contrast_stddev=0,
        )
    ).validate(image, detections=[_detection(10, 10, 80, 80)], landmarks=[landmarks])

    assert "non_frontal" in result.warning_flags
    assert result.metrics.faces[0].pose is not None
    assert not result.metrics.faces[0].pose.is_approximately_frontal


def test_pose_is_assessed_when_landmarks_are_available_without_detections() -> None:
    image = np.full((100, 100, 3), 120, dtype=np.uint8)
    landmarks = _face_landmarks(
        0,
        left_eye=(30, 40),
        right_eye=(70, 20),
        nose=(65, 30),
    )
    result = QualityValidator(
        QualityValidatorConfig(
            min_width=1,
            min_height=1,
            min_contrast_stddev=0,
        )
    ).validate(image, landmarks=[landmarks])

    assert result.metrics.faces == ()
    assert result.metrics.pose_indicators[0].detection_index == 0
    assert "non_frontal" in result.warning_flags


def test_invalid_image_is_rejected() -> None:
    with pytest.raises(InvalidImageError):
        QualityValidator().validate(np.zeros((10, 10, 3), dtype=np.float32))


def test_non_positive_face_box_is_rejected() -> None:
    with pytest.raises(QualityValidationError, match="non-positive"):
        QualityValidator().validate(
            np.zeros((100, 100, 3), dtype=np.uint8),
            detections=[_detection(width=0)],
        )


def test_input_array_is_not_modified() -> None:
    image = np.full((80, 90, 3), (10, 80, 140), dtype=np.uint8)
    original = image.copy()

    QualityValidator().validate(image)

    np.testing.assert_array_equal(image, original)


def test_score_is_bounded_and_rejection_policy_is_configurable() -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    config = QualityValidatorConfig(
        min_width=100,
        min_height=100,
        min_blur_variance=1000,
        reject_on_warnings=frozenset(),
        score_penalties={"low_resolution": 250.0, "blur": 250.0},
    )

    result = QualityValidator(config).validate(image)

    assert result.quality_score == 0
    assert 0 <= result.quality_score <= 100
    assert result.valid
    assert result.rejection_reasons == ()
    assert "low_resolution" in result.warning_flags
    assert "blur" in result.warning_flags


def test_no_face_warning_does_not_imply_spoof_classification() -> None:
    result = QualityValidator(
        QualityValidatorConfig(min_width=1, min_height=1)
    ).validate(np.full((100, 100, 3), 120, dtype=np.uint8), detections=[])

    assert "no_face_detected" in result.warning_flags
    assert not hasattr(result, "is_spoof")
