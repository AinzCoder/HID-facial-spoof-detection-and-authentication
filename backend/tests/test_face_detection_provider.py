from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.services.inference import ComponentOutput, ModelState, VerificationInference
from app.services.inference_contracts import (
    DecodedFrame,
    FaceDetectionResult,
    InferenceOutput,
    ModelReasonCode,
    ModelStatus,
)
from app.services.media import TemporaryMedia
from app.services.member1_face_provider import (
    Member1FaceDetectionProvider,
    decode_image_file,
)
from app.routes.api import _safe_verify


def _frame(width: int = 8, height: int = 8, color=(15, 80, 210)) -> DecodedFrame:
    pixel = bytes(color)
    return DecodedFrame(pixels=pixel * (width * height), width=width, height=height)


def _face(
    *,
    x: int = 2,
    confidence: float | None = 0.9,
    landmarks: bool = True,
    alignment: bool = True,
    crop: np.ndarray | None = None,
) -> SimpleNamespace:
    points = (
        SimpleNamespace(
            index=0,
            normalized=SimpleNamespace(x=0.4, y=0.45),
        ),
        SimpleNamespace(
            index=1,
            normalized=SimpleNamespace(x=0.6, y=0.45),
        ),
    )
    return SimpleNamespace(
        bounding_box=SimpleNamespace(x=x, y=1, width=4, height=4),
        detection_confidence=confidence,
        landmarks=SimpleNamespace(landmarks=points) if landmarks else None,
        alignment_metadata=SimpleNamespace(alignment_applied=alignment),
        optional_aligned_crop=(
            np.full((2, 3, 3), (90, 30, 200), dtype=np.uint8)
            if crop is None
            else crop
        ),
    )


class FakePipeline:
    def __init__(self, faces) -> None:
        self.faces = faces
        self.closed = False
        self.received_rgb = None

    def process_frame(self, frame_rgb, frame_index, timestamp_ms, *, timestamp_source):
        self.received_rgb = frame_rgb.copy()
        assert frame_index == 0
        assert timestamp_ms == 0.0
        assert timestamp_source == "image"
        return SimpleNamespace(faces=self.faces)

    def close(self) -> None:
        self.closed = True


def _provider(pipeline: FakePipeline) -> Member1FaceDetectionProvider:
    return Member1FaceDetectionProvider(pipeline_factory=lambda: pipeline)


def test_rgb_frame_maps_to_existing_face_contract() -> None:
    pipeline = FakePipeline([_face()])
    provider = _provider(pipeline)

    output = provider.detect_and_align(_frame())

    assert output.model_status == ModelStatus.MOCK
    assert isinstance(output.result, FaceDetectionResult)
    assert output.result.face_count == 1
    assert output.result.bounding_box is not None
    assert output.result.bounding_box.x == 0.25
    assert output.result.bounding_box.y == 0.125
    assert output.result.aligned_face_crop is not None
    assert output.result.aligned_face_crop.pixels == bytes((90, 30, 200)) * 6
    assert [point.name for point in output.result.landmarks] == [
        "landmark_0",
        "landmark_1",
    ]
    assert pipeline.received_rgb is not None
    np.testing.assert_array_equal(
        pipeline.received_rgb[0, 0],
        np.array((15, 80, 210), dtype=np.uint8),
    )


def test_zero_faces_returns_explicit_no_face_without_confidence() -> None:
    output = _provider(FakePipeline([])).detect_and_align(_frame())

    assert output.model_status == ModelStatus.MOCK
    assert output.reason_code == ModelReasonCode.NO_FACE
    assert output.confidence is None
    assert output.result == FaceDetectionResult(face_detected=False, face_count=0)


def test_multiple_faces_selects_highest_confidence_geometry() -> None:
    output = _provider(
        FakePipeline([_face(x=0, confidence=0.7), _face(x=2, confidence=0.95)])
    ).detect_and_align(_frame())

    assert output.model_status == ModelStatus.MOCK
    assert output.reason_code == ModelReasonCode.MULTIPLE_FACES
    assert output.confidence == 0.95
    assert output.result is not None
    assert output.result.face_count == 2
    assert output.result.bounding_box is not None
    assert output.result.bounding_box.x == 0.25


@pytest.mark.parametrize(
    "face",
    [
        _face(landmarks=False),
        _face(alignment=False),
        _face(crop=np.zeros((2, 2), dtype=np.uint8)),
    ],
)
def test_incomplete_preprocessing_output_is_never_real(face) -> None:
    output = _provider(FakePipeline([face])).detect_and_align(_frame())

    assert output.model_status == ModelStatus.ERROR
    assert output.result is None
    assert output.confidence is None
    assert output.reason_code == ModelReasonCode.INFERENCE_ERROR


def test_pipeline_exception_is_reported_as_error() -> None:
    def fail() -> object:
        raise RuntimeError("initialization detail is not exposed")

    provider = Member1FaceDetectionProvider(pipeline_factory=fail)

    output = provider.detect_and_align(_frame())

    assert output.model_status == ModelStatus.ERROR
    assert output.result is None
    assert output.reason_code == ModelReasonCode.INFERENCE_ERROR


def test_missing_local_weights_are_unavailable_without_download(tmp_path: Path) -> None:
    detector_path = tmp_path / "detector.tflite"
    landmarker_path = tmp_path / "landmarker.task"

    provider = Member1FaceDetectionProvider(detector_path, landmarker_path)
    output = provider.detect_and_align(_frame())

    assert output.model_status == ModelStatus.UNAVAILABLE
    assert output.reason_code == ModelReasonCode.MODEL_UNAVAILABLE
    assert not detector_path.exists()
    assert not landmarker_path.exists()


def test_unloadable_local_weights_are_reported_as_error(tmp_path: Path) -> None:
    detector_path = tmp_path / "detector.tflite"
    landmarker_path = tmp_path / "landmarker.task"
    detector_path.write_bytes(b"invalid detector weights")
    landmarker_path.write_bytes(b"invalid landmarker weights")

    provider = Member1FaceDetectionProvider(detector_path, landmarker_path)
    output = provider.detect_and_align(_frame())

    assert output.model_status == ModelStatus.ERROR
    assert output.result is None
    assert output.confidence is None
    assert output.reason_code == ModelReasonCode.INFERENCE_ERROR


def test_close_releases_pipeline_once() -> None:
    pipeline = FakePipeline([])
    provider = _provider(pipeline)

    provider.close()
    provider.close()
    output = provider.detect_and_align(_frame())

    assert pipeline.closed
    assert output.model_status == ModelStatus.ERROR
    assert output.result is None


def test_uploaded_image_decodes_as_rgb_and_checks_content_type(tmp_path: Path) -> None:
    path = tmp_path / "portrait.png"
    Image.new("RGB", (3, 2), color=(12, 100, 230)).save(path, format="PNG")

    decoded = decode_image_file(path, "image/png")

    assert (decoded.width, decoded.height) == (3, 2)
    assert decoded.color_space == "RGB"
    assert decoded.pixels[:3] == bytes((12, 100, 230))
    with pytest.raises(ValueError, match="does not match"):
        decode_image_file(path, "image/jpeg")


def test_malformed_image_is_reported_as_decode_failure(tmp_path: Path) -> None:
    path = tmp_path / "broken.png"
    path.write_bytes(b"not an image")

    with pytest.raises(ValueError, match="cannot be decoded"):
        decode_image_file(path, "image/png")


def test_backend_and_member_packages_load_without_app_collision() -> None:
    backend_app = importlib.import_module("app")
    pipeline_module, package_name = (
        Member1FaceDetectionProvider._load_member_pipeline_module()
    )

    assert package_name != "app"
    assert sys.modules["app"] is backend_app
    assert pipeline_module.FacePreprocessingPipeline.__module__.startswith(
        package_name
    )


def test_api_uses_member_provider_for_non_real_face_output(tmp_path: Path) -> None:
    path = tmp_path / "portrait.png"
    Image.new("RGB", (8, 8), color=(20, 40, 60)).save(path, format="PNG")
    pipeline = FakePipeline([_face(x=0)])
    provider = _provider(pipeline)
    mock = ComponentOutput(state=ModelState.MOCK, model_name="test")
    base = VerificationInference(mock, mock, mock, mock, mock)
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                inference_provider=SimpleNamespace(verify=lambda *_: base),
                face_detection_provider=provider,
            )
        )
    )
    media = TemporaryMedia(str(path), "image/png", "image", path.stat().st_size)

    result = _safe_verify(request, media, None)

    assert result.face_detection.state == ModelState.MOCK
    assert result.face_detection.label == "face"
    assert pipeline.received_rgb is not None


def test_app_shutdown_closes_injected_face_pipeline(tmp_path: Path) -> None:
    settings = Settings(database_url=f"sqlite:///{(tmp_path / 'backend.db').as_posix()}")
    pipeline = FakePipeline([])
    provider = _provider(pipeline)

    with TestClient(create_app(settings, face_detection_provider=provider)):
        assert not pipeline.closed

    assert pipeline.closed


def test_enabled_provider_stays_unavailable_without_local_weights(
    tmp_path: Path,
) -> None:
    settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 'backend.db').as_posix()}",
        face_detection_enabled=True,
        face_detection_model_path=str(tmp_path / "missing.tflite"),
        face_landmarker_model_path=str(tmp_path / "missing.task"),
    )

    with TestClient(create_app(settings)) as client:
        output = client.app.state.face_detection_provider.detect_and_align(_frame())

    assert output.model_status == ModelStatus.UNAVAILABLE
    assert output.result is None


def test_real_preprocessing_runs_when_local_model_weights_exist() -> None:
    detector = Path(
        os.environ.get(
            "HID_FACE_DETECTION_MODEL_PATH",
            str(
                Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache"))
                / "face_preprocessing"
                / "models"
                / "blaze_face_short_range.tflite"
            ),
        )
    )
    landmarker = Path(
        os.environ.get(
            "HID_FACE_LANDMARKER_MODEL_PATH",
            str(
                Path(os.environ.get("LOCALAPPDATA", Path.home() / ".cache"))
                / "face_preprocessing"
                / "models"
                / "face_landmarker.task"
            ),
        )
    )
    if not detector.is_file() or not landmarker.is_file():
        pytest.skip("Local MediaPipe model weights are not provisioned.")

    provider = Member1FaceDetectionProvider(detector, landmarker)
    try:
        output = provider.detect_and_align(
            DecodedFrame(
                pixels=bytes((127, 127, 127)) * (128 * 128),
                width=128,
                height=128,
            )
        )
        assert output.model_status == ModelStatus.REAL
        assert output.result is not None
    finally:
        provider.close()
