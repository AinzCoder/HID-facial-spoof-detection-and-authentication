from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.face_detector import BoundingBox, FaceDetection, InvalidImageError
from app.face_cropper import FaceCropError, FaceCropper, FaceCropperConfig


def _detection(
    x: int = 10, y: int = 15, width: int = 20, height: int = 25
) -> FaceDetection:
    return FaceDetection(BoundingBox(x, y, width, height), confidence=0.9)


def test_tight_face_crop_has_original_dimensions() -> None:
    image = np.zeros((80, 90, 3), dtype=np.uint8)
    cropper = FaceCropper(FaceCropperConfig(padding_ratio=0, target_size=None))

    result = cropper.crop_faces(image, [_detection()])[0]

    assert result.crop_coordinates == BoundingBox(10, 15, 20, 25)
    assert result.original_crop_dimensions == (25, 20)
    assert result.output_dimensions == (25, 20)
    assert result.resized_crop is None


def test_default_padding_preserves_surrounding_context() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    result = FaceCropper().crop_faces(
        image, [FaceDetection(BoundingBox(20, 20, 40, 50), 0.9)]
    )[0]

    assert result.padding_ratio == pytest.approx(0.20)
    assert result.padded_bounding_box == BoundingBox(12, 10, 56, 70)
    assert result.crop_coordinates == result.padded_bounding_box
    assert result.original_crop_dimensions == (70, 56)
    assert result.resized_crop is not None
    assert result.resized_crop.shape == (224, 224, 3)
    assert result.output_dimensions == (224, 224)
    assert not result.clipping_flag


def test_face_near_boundary_is_clamped_and_marked_clipped() -> None:
    image = np.zeros((60, 80, 3), dtype=np.uint8)
    result = FaceCropper(
        FaceCropperConfig(padding_ratio=0.2, target_size=None)
    ).crop_faces(image, [FaceDetection(BoundingBox(0, 10, 20, 20), 0.9)])[0]

    assert result.padded_bounding_box == BoundingBox(-4, 6, 28, 28)
    assert result.crop_coordinates == BoundingBox(0, 6, 24, 28)
    assert result.clipping_flag
    assert result.original_crop.shape == (28, 24, 3)


def test_tight_box_touching_image_edge_is_marked_clipped() -> None:
    image = np.zeros((60, 80, 3), dtype=np.uint8)
    result = FaceCropper(
        FaceCropperConfig(padding_ratio=0, target_size=None)
    ).crop_faces(image, [FaceDetection(BoundingBox(0, 10, 20, 20), 0.9)])[0]

    assert result.crop_coordinates == BoundingBox(0, 10, 20, 20)
    assert result.clipping_flag


def test_partially_outside_face_box_is_clamped() -> None:
    image = np.zeros((60, 80, 3), dtype=np.uint8)
    result = FaceCropper(
        FaceCropperConfig(padding_ratio=0, target_size=None)
    ).crop_faces(image, [FaceDetection(BoundingBox(-5, 10, 20, 20), 0.9)])[0]

    assert result.crop_coordinates == BoundingBox(0, 10, 15, 20)
    assert result.clipping_flag


def test_multiple_faces_keep_detection_order_and_indices() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = [
        FaceDetection(BoundingBox(5, 5, 10, 10), 0.8),
        FaceDetection(BoundingBox(50, 40, 20, 30), 0.9),
    ]

    results = FaceCropper(
        FaceCropperConfig(padding_ratio=0, target_size=None)
    ).crop_faces(image, detections)

    assert [result.face_id for result in results] == [0, 1]
    assert [result.original_bounding_box for result in results] == [
        detection.bounding_box for detection in detections
    ]


@pytest.mark.parametrize(
    "box",
    [
        BoundingBox(0, 0, 0, 10),
        BoundingBox(0, 0, 10, 0),
        BoundingBox(0, 0, -1, 10),
    ],
)
def test_invalid_or_zero_area_boxes_are_rejected(box: BoundingBox) -> None:
    with pytest.raises(FaceCropError, match="zero-area or invalid"):
        FaceCropper().crop_faces(
            np.zeros((40, 40, 3), dtype=np.uint8),
            [FaceDetection(box, 0.9)],
        )


def test_box_outside_image_is_rejected() -> None:
    with pytest.raises(FaceCropError, match="does not intersect"):
        FaceCropper().crop_faces(
            np.zeros((40, 40, 3), dtype=np.uint8),
            [FaceDetection(BoundingBox(50, 50, 10, 10), 0.9)],
        )


def test_crop_preserves_rgb_pixel_values() -> None:
    rgb_image = np.empty((20, 30, 3), dtype=np.uint8)
    rgb_image[:] = (12, 100, 230)
    pil_image = Image.fromarray(rgb_image)
    cropper = FaceCropper(FaceCropperConfig(padding_ratio=0, target_size=None))

    result = cropper.crop_faces(
        pil_image, [FaceDetection(BoundingBox(3, 4, 10, 8), 0.9)]
    )[0]

    assert np.all(result.original_crop == np.array([12, 100, 230], dtype=np.uint8))


def test_opencv_bgr_array_crop_is_returned_as_rgb() -> None:
    bgr_image = np.empty((20, 30, 3), dtype=np.uint8)
    bgr_image[:] = (230, 100, 12)

    result = FaceCropper(
        FaceCropperConfig(padding_ratio=0, target_size=None)
    ).crop_faces(
        bgr_image,
        [FaceDetection(BoundingBox(0, 0, 10, 10), 0.9)],
    )[0]

    assert np.all(result.original_crop == np.array([12, 100, 230], dtype=np.uint8))


def test_input_numpy_array_is_not_modified() -> None:
    image = np.full((40, 50, 3), (10, 80, 200), dtype=np.uint8)
    original = image.copy()

    FaceCropper().crop_faces(image, [_detection()])

    np.testing.assert_array_equal(image, original)


def test_configurable_padding_and_output_dimensions() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    cropper = FaceCropper(
        FaceCropperConfig(
            padding_ratio=0.5,
            target_size=(64, 48),
            interpolation="linear",
        )
    )

    result = cropper.crop_faces(
        image, [FaceDetection(BoundingBox(20, 20, 20, 20), 0.9)]
    )[0]

    assert result.padded_bounding_box == BoundingBox(10, 10, 40, 40)
    assert result.original_crop_dimensions == (40, 40)
    assert result.resized_crop is not None
    assert result.resized_crop.shape == (48, 64, 3)
    assert result.output_dimensions == (48, 64)


def test_png_path_input_is_supported(tmp_path: Path) -> None:
    path = tmp_path / "input.png"
    Image.fromarray(np.full((30, 40, 3), (20, 30, 40), dtype=np.uint8)).save(path)

    result = FaceCropper(
        FaceCropperConfig(padding_ratio=0, target_size=None)
    ).crop_faces(path, [_detection(2, 3, 10, 12)])[0]

    assert result.original_crop.shape == (12, 10, 3)


def test_empty_image_is_rejected() -> None:
    with pytest.raises(InvalidImageError):
        FaceCropper().crop_faces(np.empty((0, 0, 3), dtype=np.uint8), [_detection()])


def test_face_id_can_preserve_upstream_detection_index() -> None:
    image = np.zeros((40, 40, 3), dtype=np.uint8)

    result = FaceCropper(
        FaceCropperConfig(padding_ratio=0, target_size=None)
    ).crop_face(image, _detection(5, 6, 10, 10), face_id=7)

    assert result.face_id == 7
