from __future__ import annotations

import numpy as np
import pytest

from app.face_cropper import FaceCropper, FaceCropperConfig, FaceCropResult
from app.face_detector import BoundingBox, FaceDetection
from app.face_aligner import (
    FaceAligner,
    FaceAlignerConfig,
    FaceAlignmentError,
)
from app.landmark_extractor import (
    FaceLandmarks,
    LandmarkPoint,
    NormalizedCoordinate,
    PixelCoordinate,
)


def _crop(*, x: int = 20, y: int = 30) -> FaceCropResult:
    image = np.zeros((180, 180, 3), dtype=np.uint8)
    return FaceCropper(
        FaceCropperConfig(padding_ratio=0, target_size=None)
    ).crop_faces(
        image, [FaceDetection(BoundingBox(x, y, 100, 100), confidence=0.9)]
    )[0]


def _landmarks(
    detection_index: int,
    left_eye: tuple[float, float],
    right_eye: tuple[float, float],
) -> FaceLandmarks:
    def point(index: int, x: float, y: float) -> LandmarkPoint:
        return LandmarkPoint(
            index=index,
            normalized=NormalizedCoordinate(x / 180, y / 180, 0.0),
            pixel=PixelCoordinate(x, y),
        )

    left = point(362, *left_eye)
    right = point(33, *right_eye)
    nose = point(1, (left_eye[0] + right_eye[0]) / 2, 95)
    return FaceLandmarks(
        face_index=detection_index,
        detection_index=detection_index,
        landmarks=(left, right, nose),
        regions={
            "left_eye": (left,),
            "right_eye": (right,),
            "nose": (nose,),
            "mouth": (),
        },
    )


def _enabled_aligner(
    output_size: tuple[int, int] = (224, 224),
) -> FaceAligner:
    return FaceAligner(FaceAlignerConfig(enabled=True, output_size=output_size))


def test_alignment_is_disabled_by_default_and_preserves_original_crop() -> None:
    crop = _crop()

    result = FaceAligner().align_face(
        crop,
        _landmarks(0, (60, 70), (100, 70)),
    )

    assert not result.alignment_applied
    assert result.aligned_face_crop is None
    assert result.original_crop is crop.original_crop
    assert "alignment_disabled" in result.warning_flags


def test_level_eyes_produce_zero_roll_and_configured_output_dimensions() -> None:
    crop = _crop(x=20, y=30)
    landmarks = _landmarks(0, (60, 70), (100, 70))

    result = _enabled_aligner((128, 96)).align_face(crop, landmarks)

    assert result.alignment_applied
    assert result.rotation_angle_degrees == pytest.approx(0)
    assert result.aligned_face_crop is not None
    assert result.aligned_face_crop.shape == (96, 128, 3)
    assert result.aligned_crop_dimensions == (96, 128)
    assert result.aligned_face_crop.dtype == np.uint8


def test_tilted_eyes_are_rotated_to_a_level_line_without_clipping() -> None:
    crop = _crop()
    landmarks = _landmarks(0, (60, 60), (100, 80))

    result = _enabled_aligner().align_face(crop, landmarks)

    assert result.rotation_angle_degrees == pytest.approx(
        np.degrees(np.arctan2(20, 40))
    )
    assert result.aligned_face_crop is not None
    assert "rotation_canvas_expanded" in result.warning_flags
    assert result.transformation_matrix is not None
    matrix = np.asarray(result.transformation_matrix)
    eye_points = np.array([[40.0, 30.0, 1.0], [80.0, 50.0, 1.0]])
    transformed_eyes = eye_points @ matrix.T
    assert transformed_eyes[0, 1] == pytest.approx(
        transformed_eyes[1, 1], abs=1e-5
    )


def test_reversed_eye_order_does_not_create_180_degree_rotation() -> None:
    crop = _crop()
    landmarks = _landmarks(0, (100, 70), (60, 70))

    result = _enabled_aligner().align_face(crop, landmarks)

    assert result.rotation_angle_degrees == pytest.approx(0)
    assert abs(result.rotation_angle_degrees) < 90


def test_missing_landmarks_return_original_and_warning() -> None:
    crop = _crop()

    result = _enabled_aligner().align_face(crop, None)

    assert not result.alignment_applied
    assert result.aligned_face_crop is None
    assert result.original_crop is crop.original_crop
    assert result.warning_flags == ("missing_landmarks",)


def test_invalid_or_outside_eye_coordinates_are_reported() -> None:
    crop = _crop()
    invalid_landmarks = _landmarks(0, (float("nan"), 70), (100, 70))
    outside_landmarks = _landmarks(0, (5, 70), (100, 70))

    invalid = _enabled_aligner().align_face(crop, invalid_landmarks)
    outside = _enabled_aligner().align_face(crop, outside_landmarks)

    assert "invalid_eye_landmarks" in invalid.warning_flags
    assert "eye_landmarks_outside_crop" in outside.warning_flags
    assert invalid.aligned_face_crop is None
    assert outside.aligned_face_crop is None


def test_multiple_faces_keep_detection_associations_and_order() -> None:
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    detections = [
        FaceDetection(BoundingBox(10, 20, 100, 100), 0.9),
        FaceDetection(BoundingBox(160, 20, 100, 100), 0.8),
    ]
    crops = FaceCropper(
        FaceCropperConfig(padding_ratio=0, target_size=None)
    ).crop_faces(image, detections)
    landmarks = [
        _landmarks(1, (200, 60), (240, 60)),
        _landmarks(0, (50, 60), (90, 60)),
    ]

    results = _enabled_aligner().align_faces(crops, landmarks)

    assert [result.face_id for result in results] == [0, 1]
    assert [result.detection_index for result in results] == [0, 1]
    assert all(result.alignment_applied for result in results)


def test_missing_associations_and_duplicate_associations_are_not_misapplied() -> None:
    crop = _crop()
    one_landmark = _landmarks(0, (60, 70), (100, 70))
    duplicate_landmarks = [one_landmark, one_landmark]

    missing = _enabled_aligner().align_faces([crop], [])
    ambiguous = _enabled_aligner().align_faces([crop], duplicate_landmarks)

    assert missing[0].warning_flags == ("missing_landmarks",)
    assert ambiguous[0].warning_flags == ("ambiguous_landmark_association",)


def test_boundary_crop_is_reported_and_alignment_keeps_output_size() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    crop = FaceCropper(
        FaceCropperConfig(padding_ratio=0.2, target_size=None)
    ).crop_faces(
        image, [FaceDetection(BoundingBox(0, 10, 50, 50), 0.9)]
    )[0]
    landmarks = _landmarks(0, (12, 30), (40, 35))

    result = _enabled_aligner().align_face(crop, landmarks)

    assert result.alignment_applied
    assert result.aligned_face_crop is not None
    assert result.aligned_face_crop.shape == (224, 224, 3)
    assert "source_crop_clipped" in result.warning_flags


def test_original_crop_and_input_image_are_never_modified() -> None:
    image = np.full((180, 180, 3), (10, 80, 200), dtype=np.uint8)
    original_image = image.copy()
    crop = FaceCropper(
        FaceCropperConfig(padding_ratio=0, target_size=None)
    ).crop_faces(
        image, [FaceDetection(BoundingBox(20, 30, 100, 100), 0.9)]
    )[0]
    original_crop = crop.original_crop.copy()

    result = _enabled_aligner().align_face(
        crop, _landmarks(0, (60, 70), (100, 80))
    )

    np.testing.assert_array_equal(image, original_image)
    np.testing.assert_array_equal(crop.original_crop, original_crop)
    assert result.original_crop is crop.original_crop
    assert result.aligned_face_crop is not crop.original_crop


def test_crop_with_invalid_image_data_is_rejected() -> None:
    crop = _crop()
    object.__setattr__(crop, "original_crop", np.empty((0, 0, 3), dtype=np.uint8))

    with pytest.raises(FaceAlignmentError, match="non-empty RGB uint8"):
        _enabled_aligner().align_face(
            crop, _landmarks(0, (60, 70), (100, 70))
        )


def test_mismatched_detection_index_is_not_aligned() -> None:
    crop = _crop()

    result = _enabled_aligner().align_face(
        crop, _landmarks(1, (60, 70), (100, 70))
    )

    assert not result.alignment_applied
    assert result.warning_flags == ("detection_index_mismatch",)
