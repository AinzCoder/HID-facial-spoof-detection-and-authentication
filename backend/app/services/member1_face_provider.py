from __future__ import annotations

import hashlib
import importlib
import importlib.util
import logging
import math
import os
import sys
import threading
import time
from pathlib import Path
from types import ModuleType
from typing import Callable, Protocol, Sequence, runtime_checkable

from PIL import Image, UnidentifiedImageError

from app.services.inference import ComponentOutput, ModelState
from app.services.inference_contracts import (
    BoundingBox,
    DecodedFrame,
    FaceDetectionResult,
    FaceLandmark,
    InferenceOutput,
    ModelReasonCode,
    ModelStatus,
)

logger = logging.getLogger(__name__)

_MEMBER_APP_PACKAGE = "_hid_member1_face_preprocessing"
_MODEL_NAME = "mediapipe_face_preprocessing"
_DEFAULT_MODEL_DIRECTORY = Path(
    os.environ.get("LOCALAPPDATA", str(Path.home() / ".cache"))
) / "face_preprocessing" / "models"
_DETECTOR_FILENAME = "blaze_face_short_range.tflite"
_LANDMARKER_FILENAME = "face_landmarker.task"
_IMAGE_FORMATS = {
    "image/jpeg": "JPEG",
    "image/png": "PNG",
    "image/webp": "WEBP",
}


class _NormalizedPoint(Protocol):
    x: float
    y: float


class _PipelineLandmark(Protocol):
    index: int
    normalized: _NormalizedPoint


class _PipelineLandmarks(Protocol):
    landmarks: Sequence[_PipelineLandmark]


class _PipelineAlignment(Protocol):
    alignment_applied: bool


class _PipelineBoundingBox(Protocol):
    x: int
    y: int
    width: int
    height: int


class _PreprocessedFace(Protocol):
    bounding_box: _PipelineBoundingBox
    detection_confidence: float | None
    landmarks: _PipelineLandmarks | None
    alignment_metadata: _PipelineAlignment | None
    optional_aligned_crop: object | None


class _PreprocessingResult(Protocol):
    faces: Sequence[_PreprocessedFace]


@runtime_checkable
class _FacePipeline(Protocol):
    def process_frame(
        self,
        frame_rgb: object,
        frame_index: int,
        timestamp_ms: float,
        *,
        timestamp_source: str,
    ) -> _PreprocessingResult: ...

    def close(self) -> None: ...


def decode_image_file(path: str | os.PathLike[str], media_type: str) -> DecodedFrame:
    """Decode a content-validated upload into the backend's contiguous RGB contract."""
    expected_format = _IMAGE_FORMATS.get(media_type)
    if expected_format is None:
        raise ValueError("Face detection requires a supported image media type.")

    try:
        with Image.open(path) as image:
            if image.format != expected_format:
                raise ValueError("Image content does not match its media type.")
            if image.width <= 0 or image.height <= 0 or image.width * image.height > 40_000_000:
                raise ValueError("Image dimensions are invalid or too large.")
            image.load()
            rgb_image = image.convert("RGB")
            return DecodedFrame(
                pixels=rgb_image.tobytes(),
                width=rgb_image.width,
                height=rgb_image.height,
            )
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Image data cannot be decoded.") from exc


def face_detection_component(
    output: InferenceOutput[FaceDetectionResult],
) -> ComponentOutput:
    if not isinstance(output, InferenceOutput):
        raise TypeError("Face provider returned an invalid inference envelope.")
    if output.result is not None and not isinstance(output.result, FaceDetectionResult):
        raise TypeError("Face provider returned an invalid face-detection result.")

    result = output.result
    label = (
        None
        if result is None
        else "no_face"
        if not result.face_detected
        else "face"
        if result.face_count == 1
        else "multiple_faces"
    )
    unavailable_or_error = output.model_status in {
        ModelStatus.UNAVAILABLE,
        ModelStatus.ERROR,
    }
    return ComponentOutput(
        state=ModelState(output.model_status.value),
        label=label,
        score=output.confidence,
        model_name=output.model_name,
        error_code=output.reason_code.value if unavailable_or_error else None,
        model_version=output.model_version,
        processing_time_ms=output.processing_time_ms,
        reason_code=output.reason_code.value,
    )


class Member1FaceDetectionProvider:
    """Backend adapter around Member 1's reusable RGB preprocessing pipeline."""

    def __init__(
        self,
        detector_model_path: str | os.PathLike[str] | None = None,
        landmarker_model_path: str | os.PathLike[str] | None = None,
        *,
        pipeline_factory: Callable[[], _FacePipeline] | None = None,
    ) -> None:
        self._pipeline: _FacePipeline | None = None
        self._lock = threading.Lock()
        self._closed = False
        self._status = ModelStatus.UNAVAILABLE
        self._reason_code = ModelReasonCode.MODEL_UNAVAILABLE
        self._model_version = "not_loaded"
        self._model_name = _MODEL_NAME

        if pipeline_factory is not None:
            try:
                pipeline = pipeline_factory()
                if not isinstance(pipeline, _FacePipeline):
                    raise TypeError("Injected face pipeline lacks required lifecycle methods.")
                self._pipeline = pipeline
            except Exception as exc:
                self._status = ModelStatus.ERROR
                self._reason_code = ModelReasonCode.INFERENCE_ERROR
                logger.error("Injected face pipeline initialization failed (%s)", type(exc).__name__)
            else:
                self._status = ModelStatus.MOCK
                self._reason_code = ModelReasonCode.INCONCLUSIVE
                self._model_name = "face_preprocessing_test_double"
                self._model_version = "injected-test-double"
            return

        detector_path = Path(detector_model_path) if detector_model_path else (
            _DEFAULT_MODEL_DIRECTORY / _DETECTOR_FILENAME
        )
        landmarker_path = Path(landmarker_model_path) if landmarker_model_path else (
            _DEFAULT_MODEL_DIRECTORY / _LANDMARKER_FILENAME
        )
        if not self._usable_model_file(detector_path) or not self._usable_model_file(landmarker_path):
            logger.info("Face preprocessing models are unavailable at the configured local paths.")
            return

        try:
            self._model_version = self._model_fingerprint(detector_path, landmarker_path)
            self._pipeline = self._build_pipeline(detector_path, landmarker_path)
        except (ImportError, ModuleNotFoundError) as exc:
            self._status = ModelStatus.UNAVAILABLE
            self._reason_code = ModelReasonCode.MODEL_UNAVAILABLE
            self._model_version = "not_loaded"
            logger.warning("Face preprocessing dependencies are unavailable (%s)", type(exc).__name__)
        except Exception as exc:
            self._status = ModelStatus.ERROR
            self._reason_code = ModelReasonCode.INFERENCE_ERROR
            self._model_version = "not_loaded"
            logger.error("Face preprocessing model initialization failed (%s)", type(exc).__name__)
        else:
            self._status = ModelStatus.REAL
            self._reason_code = ModelReasonCode.OK

    @staticmethod
    def _usable_model_file(path: Path) -> bool:
        try:
            return path.is_file() and path.stat().st_size > 0
        except OSError:
            return False

    @staticmethod
    def _model_fingerprint(detector_path: Path, landmarker_path: Path) -> str:
        digest = hashlib.sha256()
        for path in (detector_path, landmarker_path):
            with path.open("rb") as model_file:
                for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
                    digest.update(chunk)
            digest.update(b"\0")
        return f"sha256:{digest.hexdigest()}"

    @staticmethod
    def _load_member_pipeline_module() -> tuple[ModuleType, str]:
        app_directory = (
            Path(__file__).resolve().parents[3]
            / "HID-facial-spoof-detection-and-authentication"
            / "face_detection"
            / "app"
        )
        init_file = app_directory / "__init__.py"
        if not init_file.is_file():
            raise FileNotFoundError("Member 1 face preprocessing package was not found.")

        if _MEMBER_APP_PACKAGE not in sys.modules:
            spec = importlib.util.spec_from_file_location(
                _MEMBER_APP_PACKAGE,
                init_file,
                submodule_search_locations=[str(app_directory)],
            )
            if spec is None or spec.loader is None:
                raise ImportError("Member 1 face preprocessing package cannot be loaded.")
            package = importlib.util.module_from_spec(spec)
            sys.modules[_MEMBER_APP_PACKAGE] = package
            try:
                spec.loader.exec_module(package)
            except Exception:
                sys.modules.pop(_MEMBER_APP_PACKAGE, None)
                raise

        module_name = f"{_MEMBER_APP_PACKAGE}.preprocessing_pipeline"
        return importlib.import_module(module_name), _MEMBER_APP_PACKAGE

    @classmethod
    def _build_pipeline(
        cls, detector_path: Path, landmarker_path: Path
    ) -> _FacePipeline:
        pipeline_module, package_name = cls._load_member_pipeline_module()
        detector_module = importlib.import_module(f"{package_name}.face_detector")
        landmarker_module = importlib.import_module(f"{package_name}.landmark_extractor")
        aligner_module = importlib.import_module(f"{package_name}.face_aligner")

        config = pipeline_module.FacePreprocessingConfig(
            aligner=aligner_module.FaceAlignerConfig(enabled=True)
        )
        detector = None
        landmarker = None
        try:
            detector = detector_module.FaceDetector(
                model_path=detector_path,
                config=config.detector,
            )
            landmarker = landmarker_module.LandmarkExtractor(
                model_path=landmarker_path,
                config=config.landmarker,
            )
            return pipeline_module.FacePreprocessingPipeline(
                config,
                detector=detector,
                landmark_extractor=landmarker,
            )
        except Exception:
            for resource in (landmarker, detector):
                if resource is not None:
                    resource.close()
            raise

    def _output(
        self,
        status: ModelStatus,
        reason_code: ModelReasonCode,
        *,
        result: FaceDetectionResult | None = None,
        confidence: float | None = None,
        processing_time_ms: float = 0.0,
    ) -> InferenceOutput[FaceDetectionResult]:
        return InferenceOutput[FaceDetectionResult](
            model_name=self._model_name,
            model_version=self._model_version,
            model_status=status,
            processing_time_ms=max(0.0, processing_time_ms),
            confidence=confidence,
            result=result,
            reason_code=reason_code,
        )

    def detect_and_align(
        self, frame: DecodedFrame
    ) -> InferenceOutput[FaceDetectionResult]:
        if self._closed:
            return self._output(ModelStatus.ERROR, ModelReasonCode.INFERENCE_ERROR)
        if self._pipeline is None:
            return self._output(self._status, self._reason_code)
        if not isinstance(frame, DecodedFrame):
            return self._output(ModelStatus.ERROR, ModelReasonCode.INFERENCE_ERROR)

        started = time.perf_counter()
        try:
            import numpy as np

            rgb_pixels = np.frombuffer(frame.pixels, dtype=np.uint8).reshape(
                (frame.height, frame.width, 3)
            ).copy()
            with self._lock:
                if self._closed:
                    raise RuntimeError("Face preprocessing provider is closed.")
                preprocessing = self._pipeline.process_frame(
                    rgb_pixels,
                    frame_index=0,
                    timestamp_ms=0.0,
                    timestamp_source="image",
                )
            faces = tuple(preprocessing.faces)
            elapsed_ms = (time.perf_counter() - started) * 1000
            if not faces:
                result = FaceDetectionResult(face_detected=False, face_count=0)
                return self._output(
                    self._status,
                    ModelReasonCode.NO_FACE,
                    result=result,
                    processing_time_ms=elapsed_ms,
                )

            for face in faces:
                score = face.detection_confidence
                if score is not None and (
                    not isinstance(score, (int, float))
                    or isinstance(score, bool)
                    or not 0 <= score <= 1
                    or not math.isfinite(score)
                ):
                    raise ValueError("The pipeline returned an invalid detector confidence.")
            selected = max(
                faces,
                key=lambda face: (
                    face.detection_confidence
                    if face.detection_confidence is not None
                    else -1.0
                ),
            )
            if (
                selected.landmarks is None
                or selected.alignment_metadata is None
                or not selected.alignment_metadata.alignment_applied
                or selected.optional_aligned_crop is None
            ):
                raise ValueError("Face landmarks or aligned crop are unavailable.")

            box = selected.bounding_box
            normalized_box = BoundingBox(
                x=box.x / frame.width,
                y=box.y / frame.height,
                width=box.width / frame.width,
                height=box.height / frame.height,
            )
            crop = self._decoded_frame_from_array(selected.optional_aligned_crop, np)
            landmarks = tuple(
                FaceLandmark(
                    name=f"landmark_{point.index}",
                    x=point.normalized.x,
                    y=point.normalized.y,
                )
                for point in selected.landmarks.landmarks
            )
            result = FaceDetectionResult(
                face_detected=True,
                face_count=len(faces),
                bounding_box=normalized_box,
                aligned_face_crop=crop,
                landmarks=landmarks,
            )
            reason_code = (
                ModelReasonCode.MULTIPLE_FACES
                if len(faces) > 1
                else ModelReasonCode.OK
            )
            return self._output(
                self._status,
                reason_code,
                result=result,
                confidence=selected.detection_confidence,
                processing_time_ms=(time.perf_counter() - started) * 1000,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            logger.warning("Face preprocessing dependencies became unavailable (%s)", type(exc).__name__)
            return self._output(
                ModelStatus.UNAVAILABLE,
                ModelReasonCode.MODEL_UNAVAILABLE,
                processing_time_ms=(time.perf_counter() - started) * 1000,
            )
        except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
            logger.error("Face preprocessing inference failed (%s)", type(exc).__name__)
            return self._output(
                ModelStatus.ERROR,
                ModelReasonCode.INFERENCE_ERROR,
                processing_time_ms=(time.perf_counter() - started) * 1000,
            )
        except Exception as exc:
            logger.error("Face preprocessing inference failed (%s)", type(exc).__name__)
            return self._output(
                ModelStatus.ERROR,
                ModelReasonCode.INFERENCE_ERROR,
                processing_time_ms=(time.perf_counter() - started) * 1000,
            )

    @staticmethod
    def _decoded_frame_from_array(
        array: object, numpy_module: ModuleType
    ) -> DecodedFrame:
        if (
            not isinstance(array, numpy_module.ndarray)
            or array.dtype != numpy_module.uint8
            or array.ndim != 3
            or array.shape[2] != 3
            or array.shape[0] <= 0
            or array.shape[1] <= 0
        ):
            raise ValueError("The pipeline returned an invalid aligned RGB crop.")
        contiguous = numpy_module.ascontiguousarray(array)
        return DecodedFrame(
            pixels=contiguous.tobytes(),
            width=int(contiguous.shape[1]),
            height=int(contiguous.shape[0]),
        )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            pipeline = self._pipeline
            self._pipeline = None
        if pipeline is not None:
            pipeline.close()
