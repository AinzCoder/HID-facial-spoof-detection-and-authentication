from __future__ import annotations

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from app.face_detector import (
    FaceDetector,
    FaceDetectorConfig,
    FaceDetectorError,
    InvalidImageError,
)


class FakeDetector:
    def __init__(self, detections: list[object]) -> None:
        self.detections = detections
        self.closed = False

    def detect(self, _image: object) -> SimpleNamespace:
        return SimpleNamespace(detections=self.detections)

    def close(self) -> None:
        self.closed = True


def _fake_result(
    x: int, y: int, width: int, height: int, confidence: float = 0.9
) -> SimpleNamespace:
    return SimpleNamespace(
        bounding_box=SimpleNamespace(
            origin_x=x, origin_y=y, width=width, height=height
        ),
        categories=[SimpleNamespace(score=confidence)],
    )


@pytest.fixture
def detector_factory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    detectors: list[FakeDetector] = []

    def build(detections: list[object]) -> FaceDetector:
        model_path = tmp_path / "model.tflite"
        model_path.write_bytes(b"test model")
        fake = FakeDetector(detections)
        detectors.append(fake)
        monkeypatch.setattr(
            FaceDetector,
            "_build_detector",
            staticmethod(lambda _path, _confidence: fake),
        )
        return FaceDetector(model_path)

    return build, detectors


def test_no_faces_returns_empty_list(detector_factory) -> None:
    build, _ = detector_factory
    detector = build([])

    assert detector.detect(np.zeros((40, 60, 3), dtype=np.uint8)) == []


def test_multiple_faces_return_pixel_boxes_and_confidence(detector_factory) -> None:
    build, _ = detector_factory
    detector = build([_fake_result(5, 6, 20, 21), _fake_result(30, 10, 25, 28, 0.75)])

    found = detector.detect(np.zeros((50, 70, 3), dtype=np.uint8))

    assert [face.bounding_box.to_dict() for face in found] == [
        {"x": 5, "y": 6, "width": 20, "height": 21},
        {"x": 30, "y": 10, "width": 25, "height": 28},
    ]
    assert [face.confidence for face in found] == [0.9, 0.75]


def test_boxes_are_clamped_to_image_bounds(detector_factory) -> None:
    build, _ = detector_factory
    detector = build([_fake_result(-5, -4, 20, 15)])

    found = detector.detect(np.zeros((8, 10, 3), dtype=np.uint8))

    assert found[0].bounding_box.to_dict() == {
        "x": 0,
        "y": 0,
        "width": 10,
        "height": 8,
    }


def test_invalid_array_is_rejected(detector_factory) -> None:
    build, _ = detector_factory
    detector = build([])

    with pytest.raises(InvalidImageError, match="uint8"):
        detector.detect(np.zeros((10, 10, 3), dtype=np.float32))


def test_input_array_is_not_modified(detector_factory) -> None:
    build, _ = detector_factory
    detector = build([])
    frame = np.full((10, 12, 3), (10, 20, 30), dtype=np.uint8)
    original = frame.copy()

    detector.detect(frame)

    np.testing.assert_array_equal(frame, original)


def test_corrupt_image_file_is_rejected(detector_factory, tmp_path: Path) -> None:
    build, _ = detector_factory
    detector = build([])
    corrupt_path = tmp_path / "broken.jpg"
    corrupt_path.write_bytes(b"not an image")

    with pytest.raises(InvalidImageError, match="corrupted"):
        detector.detect(corrupt_path)


def test_encoded_png_bytes_are_accepted(detector_factory) -> None:
    build, _ = detector_factory
    detector = build([])
    stream = BytesIO()
    Image.fromarray(np.zeros((12, 16, 3), dtype=np.uint8)).save(stream, format="PNG")

    assert detector.detect(stream.getvalue()) == []


def test_invalid_confidence_threshold_is_rejected() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        FaceDetectorConfig(min_detection_confidence=1.1)


def test_close_releases_detector_and_prevents_reuse(detector_factory) -> None:
    build, fakes = detector_factory
    detector = build([])

    detector.close()

    assert fakes[0].closed
    with pytest.raises(FaceDetectorError, match="closed"):
        detector.detect(np.zeros((10, 10, 3), dtype=np.uint8))
