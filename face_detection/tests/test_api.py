from __future__ import annotations

import base64
import io
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.face_aligner import FaceAlignerConfig
from app.face_cropper import FaceCropperConfig
from app.face_detector import BoundingBox, FaceDetection, RGBImage
from app.landmark_extractor import (
    FaceLandmarks,
    LandmarkPoint,
    NormalizedCoordinate,
    PixelCoordinate,
)
from app.main import DEFAULT_MAX_UPLOAD_BYTES, create_app
from app.preprocessing_pipeline import FacePreprocessingConfig, FacePreprocessingPipeline
from app.quality_validator import QualityValidatorConfig
from app.video_processor import VideoProcessorConfig


def _png_bytes(
    width: int = 180,
    height: int = 140,
    color: tuple[int, int, int] = (85, 130, 170),
) -> bytes:
    pixels = np.empty((height, width, 3), dtype=np.uint8)
    pixels[:] = color
    output = io.BytesIO()
    Image.fromarray(pixels).save(output, format="PNG")
    return output.getvalue()


def _landmarks(
    detection_index: int,
    box: BoundingBox,
    image_width: int = 180,
    image_height: int = 140,
) -> FaceLandmarks:
    def point(index: int, x: float, y: float) -> LandmarkPoint:
        return LandmarkPoint(
            index=index,
            normalized=NormalizedCoordinate(
                x / image_width,
                y / image_height,
                0.0,
            ),
            pixel=PixelCoordinate(x, y),
        )

    left = point(362, box.x + box.width * 0.3, box.y + box.height * 0.35)
    right = point(33, box.x + box.width * 0.7, box.y + box.height * 0.35)
    nose = point(1, box.x + box.width / 2, box.y + box.height * 0.55)
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


class StubDetector:
    def __init__(
        self,
        detections: Sequence[FaceDetection],
        *,
        delay: float = 0.0,
    ) -> None:
        self.detections = list(detections)
        self.delay = delay
        self.active = 0
        self.max_active = 0
        self._counter_lock = threading.Lock()
        self.closed = False

    def detect(self, _: RGBImage) -> list[FaceDetection]:
        with self._counter_lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                time.sleep(self.delay)
            return list(self.detections)
        finally:
            with self._counter_lock:
                self.active -= 1

    def close(self) -> None:
        self.closed = True


class StubLandmarker:
    def __init__(self, faces: Sequence[FaceLandmarks]) -> None:
        self.faces = list(faces)
        self.closed = False

    def extract(
        self,
        _: RGBImage,
        *,
        detections: Sequence[FaceDetection] | None = None,
    ) -> list[FaceLandmarks]:
        return list(self.faces) if detections else []

    def close(self) -> None:
        self.closed = True


def _make_pipeline(
    detections: Sequence[FaceDetection] = (),
    landmarks: Sequence[FaceLandmarks] = (),
    *,
    detector_delay: float = 0.0,
) -> tuple[FacePreprocessingPipeline, StubDetector, StubLandmarker]:
    detector = StubDetector(detections, delay=detector_delay)
    landmarker = StubLandmarker(landmarks)
    config = FacePreprocessingConfig(
        quality=QualityValidatorConfig(
            min_blur_variance=0,
            min_brightness=0,
            max_brightness=255,
            min_contrast_stddev=0,
            min_width=1,
            min_height=1,
            min_face_area_ratio=0,
        ),
        cropper=FaceCropperConfig(padding_ratio=0.2, target_size=(64, 64)),
        aligner=FaceAlignerConfig(enabled=False),
        video=VideoProcessorConfig(),
    )
    pipeline = FacePreprocessingPipeline(
        config,
        detector=detector,  # type: ignore[arg-type]
        landmark_extractor=landmarker,  # type: ignore[arg-type]
    )
    return pipeline, detector, landmarker


def _upload(
    content: bytes | None = None,
    *,
    filename: str = "image.png",
    content_type: str = "application/octet-stream",
) -> dict[str, tuple[str, bytes, str]]:
    return {
        "file": (
            filename,
            content if content is not None else _png_bytes(),
            content_type,
        )
    }


def test_health_reports_only_service_and_model_readiness() -> None:
    pipeline, _, _ = _make_pipeline()
    app = create_app(pipeline_factory=lambda: pipeline)

    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {
            "status": "ok",
            "models_initialized": True,
        }
        assert "model_path" not in response.json()


def test_detect_returns_only_face_boxes_and_confidence() -> None:
    detection = FaceDetection(BoundingBox(20, 15, 70, 80), 0.91)
    pipeline, _, _ = _make_pipeline([detection])
    app = create_app(pipeline_factory=lambda: pipeline)

    with TestClient(app) as client:
        response = client.post("/detect", files=_upload())

    assert response.status_code == 200
    body = response.json()
    assert body["face_count"] == 1
    assert body["image_dimensions"] == {"width": 180, "height": 140}
    assert body["faces"] == [
        {
            "bounding_box": {"x": 20, "y": 15, "width": 70, "height": 80},
            "confidence": pytest.approx(0.91),
        }
    ]
    assert "landmarks" not in body["faces"][0]


def test_upload_content_is_checked_instead_of_trusting_filename_or_mime_type() -> None:
    pipeline, _, _ = _make_pipeline()
    app = create_app(pipeline_factory=lambda: pipeline)

    with TestClient(app) as client:
        valid_png_with_jpg_name = client.post(
            "/detect",
            files=_upload(filename="misleading.jpg", content_type="image/jpeg"),
        )
        invalid_content = client.post(
            "/preprocess",
            files=_upload(b"not an image", filename="photo.png", content_type="image/png"),
        )

    assert valid_png_with_jpg_name.status_code == 200
    assert invalid_content.status_code == 400
    assert invalid_content.json() == {"detail": "Upload a valid JPG or PNG image."}


def test_oversized_upload_is_rejected_with_413() -> None:
    pipeline, _, _ = _make_pipeline()
    app = create_app(
        pipeline_factory=lambda: pipeline,
        max_upload_bytes=128,
    )

    with TestClient(app) as client:
        response = client.post("/detect", files=_upload(_png_bytes()))

    assert response.status_code == 413
    assert response.json() == {
        "detail": "Image upload exceeds the configured size limit."
    }


def test_upload_limit_defaults_to_ten_mib_and_rejects_bad_configuration() -> None:
    assert DEFAULT_MAX_UPLOAD_BYTES == 10 * 1024 * 1024
    with pytest.raises(ValueError, match="positive integer"):
        create_app(max_upload_bytes=0)


def test_no_face_image_returns_empty_face_list_and_quality_flags() -> None:
    pipeline, _, _ = _make_pipeline()
    app = create_app(pipeline_factory=lambda: pipeline)

    with TestClient(app) as client:
        response = client.post("/preprocess", files=_upload())

    assert response.status_code == 200
    assert response.json()["face_count"] == 0
    assert response.json()["faces"] == []
    assert "no_face_detected" in response.json()["quality_flags"]


def test_multiple_faces_keep_detection_order() -> None:
    detections = [
        FaceDetection(BoundingBox(10, 15, 45, 50), 0.9),
        FaceDetection(BoundingBox(100, 25, 55, 60), 0.8),
    ]
    landmarks = [
        _landmarks(0, detections[0].bounding_box),
        _landmarks(1, detections[1].bounding_box),
    ]
    pipeline, _, _ = _make_pipeline(detections, landmarks)
    app = create_app(pipeline_factory=lambda: pipeline)

    with TestClient(app) as client:
        response = client.post("/preprocess", files=_upload())

    body = response.json()
    assert response.status_code == 200
    assert body["face_count"] == 2
    assert [face["detection_index"] for face in body["faces"]] == [0, 1]
    assert [face["face_id"] for face in body["faces"]] == [0, 1]


def test_preprocess_serializes_landmarks_and_rgb_png_crop() -> None:
    detection = FaceDetection(BoundingBox(40, 25, 60, 70), 0.93)
    landmarks = _landmarks(0, detection.bounding_box)
    pipeline, _, _ = _make_pipeline([detection], [landmarks])
    app = create_app(pipeline_factory=lambda: pipeline)

    with TestClient(app) as client:
        response = client.post("/preprocess", files=_upload())

    assert response.status_code == 200
    body = response.json()
    face = body["faces"][0]
    assert face["color_format"] == "RGB"
    assert face["landmarks"]["detection_index"] == 0
    assert len(face["landmarks"]["landmarks"]) == 3
    assert face["landmarks"]["landmarks"][0]["normalized"]["x"] > 0
    assert face["crop_coordinates"] == {"x": 28, "y": 11, "width": 84, "height": 98}
    assert face["crop_dimensions"] == {"width": 84, "height": 98}
    assert face["resized_crop_dimensions"] == {"width": 64, "height": 64}
    assert face["alignment_applied"] is False
    assert "context_face_crop" not in face

    decoded_png = base64.b64decode(face["resized_crop_base64"], validate=True)
    with Image.open(io.BytesIO(decoded_png)) as decoded_image:
        assert decoded_image.format == "PNG"
        assert decoded_image.mode == "RGB"
        assert decoded_image.size == (64, 64)
        decoded_pixels = np.asarray(decoded_image)
    assert decoded_pixels.dtype == np.uint8
    np.testing.assert_array_equal(decoded_pixels[0, 0], np.array([85, 130, 170]))


def test_pipeline_is_initialized_once_and_closed_on_shutdown() -> None:
    pipeline, detector, landmarker = _make_pipeline()
    factory_calls = 0

    def factory() -> FacePreprocessingPipeline:
        nonlocal factory_calls
        factory_calls += 1
        return pipeline

    app = create_app(pipeline_factory=factory)
    with TestClient(app) as client:
        assert client.get("/health").json()["models_initialized"]
        client.post("/detect", files=_upload())
        client.post("/preprocess", files=_upload())
        assert factory_calls == 1
        assert not detector.closed
        assert not landmarker.closed

    assert detector.closed and landmarker.closed


def test_concurrent_requests_serialize_access_to_media_pipe_detector() -> None:
    pipeline, detector, _ = _make_pipeline(detector_delay=0.03)
    app = create_app(pipeline_factory=lambda: pipeline)

    with TestClient(app) as client:
        with ThreadPoolExecutor(max_workers=5) as executor:
            responses = list(
                executor.map(
                    lambda _: client.post("/detect", files=_upload()),
                    range(8),
                )
            )

    assert all(response.status_code == 200 for response in responses)
    assert detector.max_active == 1


def test_upload_limit_can_be_configured_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("FACE_API_MAX_UPLOAD_BYTES", "2048")
    pipeline, _, _ = _make_pipeline()
    app = create_app(pipeline_factory=lambda: pipeline)

    with TestClient(app) as client:
        response = client.post("/detect", files=_upload(_png_bytes()))

    assert response.status_code == 200
