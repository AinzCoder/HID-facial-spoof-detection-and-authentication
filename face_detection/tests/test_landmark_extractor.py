from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.face_detector import BoundingBox, FaceDetection, InvalidImageError
from app.landmark_extractor import (
    FACIAL_REGION_INDICES,
    LandmarkExtractor,
    LandmarkExtractorConfig,
    LandmarkExtractorError,
)


class FakeLandmarker:
    def __init__(self, faces: list[list[SimpleNamespace]]) -> None:
        self.faces = faces
        self.closed = False

    def detect(self, _image: object) -> SimpleNamespace:
        return SimpleNamespace(face_landmarks=self.faces)

    def close(self) -> None:
        self.closed = True


def _face(center_x: float, center_y: float) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            x=center_x + (index % 10) / 100,
            y=center_y + (index // 10) / 100,
            z=index / 1000,
        )
        for index in range(478)
    ]


@pytest.fixture
def extractor_factory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    landmarkers: list[FakeLandmarker] = []

    def build(faces: list[list[SimpleNamespace]]) -> LandmarkExtractor:
        model_path = tmp_path / "landmarker.task"
        model_path.write_bytes(b"test model")
        fake = FakeLandmarker(faces)
        landmarkers.append(fake)
        monkeypatch.setattr(
            LandmarkExtractor,
            "_build_landmarker",
            lambda _self, _path: fake,
        )
        return LandmarkExtractor(model_path)

    return build, landmarkers


def test_no_faces_returns_empty_list(extractor_factory) -> None:
    build, _ = extractor_factory
    extractor = build([])

    assert extractor.extract(np.zeros((40, 60, 3), dtype=np.uint8)) == []


def test_returns_478_landmarks_with_normalized_and_pixel_coordinates(
    extractor_factory,
) -> None:
    build, _ = extractor_factory
    extractor = build([_face(0.2, 0.3)])
    image = np.zeros((200, 300, 3), dtype=np.uint8)

    face = extractor.extract(image)[0]

    assert len(face.landmarks) == 478
    assert face.landmarks[0].normalized.x == pytest.approx(0.2)
    assert face.landmarks[0].normalized.y == pytest.approx(0.3)
    assert face.landmarks[0].pixel.x == pytest.approx(60)
    assert face.landmarks[0].pixel.y == pytest.approx(60)
    assert face.landmarks[0].normalized.z == pytest.approx(0)
    assert face.regions["left_eye"]
    assert face.regions["right_eye"]
    assert face.regions["nose"]
    assert face.regions["mouth"]


def test_multiple_faces_match_detection_boxes(extractor_factory) -> None:
    build, _ = extractor_factory
    extractor = build([_face(0.1, 0.1), _face(0.7, 0.1)])
    detections = [
        FaceDetection(BoundingBox(25, 25, 50, 50), 0.9),
        FaceDetection(BoundingBox(205, 25, 50, 50), 0.8),
    ]

    faces = extractor.extract(
        np.zeros((200, 300, 3), dtype=np.uint8), detections=detections
    )

    assert [face.detection_index for face in faces] == [0, 1]
    assert [face.face_index for face in faces] == [0, 1]


def test_landmark_indices_for_regions_are_available_in_face_mesh() -> None:
    assert set(FACIAL_REGION_INDICES) == {
        "left_eye",
        "right_eye",
        "nose",
        "mouth",
    }
    assert max(index for indices in FACIAL_REGION_INDICES.values() for index in indices) < 478


def test_invalid_input_is_rejected(extractor_factory) -> None:
    build, _ = extractor_factory
    extractor = build([])

    with pytest.raises(InvalidImageError, match="uint8"):
        extractor.extract(np.zeros((10, 10, 3), dtype=np.float32))


def test_input_array_is_not_modified(extractor_factory) -> None:
    build, _ = extractor_factory
    extractor = build([])
    image = np.full((20, 30, 3), 123, dtype=np.uint8)
    original = image.copy()

    extractor.extract(image)

    np.testing.assert_array_equal(image, original)


def test_close_releases_landmarker_and_prevents_reuse(extractor_factory) -> None:
    build, landmarkers = extractor_factory
    extractor = build([])

    extractor.close()

    assert landmarkers[0].closed
    with pytest.raises(LandmarkExtractorError, match="closed"):
        extractor.extract(np.zeros((10, 10, 3), dtype=np.uint8))


def test_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="num_faces"):
        LandmarkExtractorConfig(num_faces=0)
    with pytest.raises(ValueError, match="num_faces"):
        LandmarkExtractorConfig(num_faces=1.5)
    with pytest.raises(ValueError, match="min_tracking_confidence"):
        LandmarkExtractorConfig(min_tracking_confidence=1.5)
