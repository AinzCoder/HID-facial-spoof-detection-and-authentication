"""FastAPI integration for the local face preprocessing pipeline."""

from __future__ import annotations

import base64
import io
import logging
import os
import threading
from contextlib import asynccontextmanager
from typing import Callable, Protocol

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from PIL import Image
from pydantic import BaseModel

from app.face_detector import (
    BoundingBox,
    FaceDetection,
    FaceDetectorError,
    InvalidImageError,
    RGBImage,
    _load_rgb,
)
from app.landmark_extractor import LandmarkExtractorError
from app.preprocessing_pipeline import (
    FacePreprocessingPipeline,
    FacePreprocessingPipelineError,
    PreprocessingResult,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_UPLOAD_BYTES_ENV = "FACE_API_MAX_UPLOAD_BYTES"
INVALID_IMAGE_DETAIL = "Upload a valid JPG or PNG image."
OVERSIZED_UPLOAD_DETAIL = "Image upload exceeds the configured size limit."
PROCESSING_ERROR_DETAIL = "Image processing failed."


class BoundingBoxResponse(BaseModel):
    x: int
    y: int
    width: int
    height: int


class ImageDimensionsResponse(BaseModel):
    width: int
    height: int


class HealthResponse(BaseModel):
    status: str
    models_initialized: bool


class DetectedFaceResponse(BaseModel):
    bounding_box: BoundingBoxResponse
    confidence: float | None


class DetectResponse(BaseModel):
    face_count: int
    image_dimensions: ImageDimensionsResponse
    faces: list[DetectedFaceResponse]


class PreprocessedFaceResponse(BaseModel):
    face_id: int
    detection_index: int
    bounding_box: BoundingBoxResponse
    confidence: float | None
    landmarks: dict[str, object] | None
    quality_metrics: dict[str, object] | None
    quality_flags: list[str]
    crop_coordinates: BoundingBoxResponse | None
    crop_dimensions: ImageDimensionsResponse | None
    resized_crop_dimensions: ImageDimensionsResponse | None
    resized_crop_base64: str | None
    color_format: str
    alignment_applied: bool


class PreprocessResponse(BaseModel):
    face_count: int
    image_dimensions: ImageDimensionsResponse
    quality_score: float
    quality_flags: list[str]
    quality_metrics: dict[str, object]
    faces: list[PreprocessedFaceResponse]


class DetectorProtocol(Protocol):
    def detect(self, image: RGBImage) -> list[FaceDetection]: ...


class PipelineProtocol(Protocol):
    detector: DetectorProtocol

    def process_image(self, image: RGBImage) -> PreprocessingResult: ...

    def close(self) -> None: ...


PipelineFactory = Callable[[], PipelineProtocol]


def _configured_upload_limit(value: int | None) -> int:
    if value is None:
        raw_value = os.environ.get(MAX_UPLOAD_BYTES_ENV)
        if raw_value is None:
            return DEFAULT_MAX_UPLOAD_BYTES
        try:
            value = int(raw_value)
        except ValueError as exc:
            raise ValueError(
                f"{MAX_UPLOAD_BYTES_ENV} must be a positive integer."
            ) from exc
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("max_upload_bytes must be a positive integer.")
    return value


def _read_valid_image(upload: UploadFile, max_upload_bytes: int) -> RGBImage:
    """Read a bounded upload and validate the encoded image format and contents."""
    try:
        encoded_image = upload.file.read(max_upload_bytes + 1)
    except OSError as exc:
        LOGGER.warning("Could not read uploaded image (%s).", type(exc).__name__)
        raise HTTPException(status_code=400, detail=INVALID_IMAGE_DETAIL) from None

    if len(encoded_image) > max_upload_bytes:
        raise HTTPException(status_code=413, detail=OVERSIZED_UPLOAD_DETAIL)
    if not encoded_image:
        raise HTTPException(status_code=400, detail=INVALID_IMAGE_DETAIL)

    try:
        return RGBImage(_load_rgb(encoded_image))
    except InvalidImageError:
        raise HTTPException(status_code=400, detail=INVALID_IMAGE_DETAIL) from None


def _dimensions(shape: tuple[int, ...]) -> ImageDimensionsResponse:
    return ImageDimensionsResponse(width=shape[1], height=shape[0])


def _box_response(box: BoundingBox) -> BoundingBoxResponse:
    return BoundingBoxResponse(
        x=box.x,
        y=box.y,
        width=box.width,
        height=box.height,
    )


def _encode_rgb_png(image: np.ndarray | None) -> str | None:
    if image is None:
        return None
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or image.shape[0] == 0
        or image.shape[1] == 0
    ):
        raise ValueError("Model-ready crops must be non-empty RGB uint8 arrays.")

    output = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(image)).save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode("ascii")


def _preprocess_response(result: PreprocessingResult) -> PreprocessResponse:
    faces: list[PreprocessedFaceResponse] = []
    for face in result.faces:
        context_crop = face.context_face_crop
        resized_crop = face.resized_face_crop
        faces.append(
            PreprocessedFaceResponse(
                face_id=face.face_id,
                detection_index=face.detection_index,
                bounding_box=_box_response(face.bounding_box),
                confidence=face.detection_confidence,
                landmarks=face.landmarks.to_dict() if face.landmarks else None,
                quality_metrics=(
                    face.quality_metrics.to_dict() if face.quality_metrics else None
                ),
                quality_flags=list(face.quality_flags),
                crop_coordinates=(
                    _box_response(face.crop_coordinates)
                    if face.crop_coordinates is not None
                    else None
                ),
                crop_dimensions=(
                    _dimensions(context_crop.shape)
                    if context_crop is not None
                    else None
                ),
                resized_crop_dimensions=(
                    _dimensions(resized_crop.shape)
                    if resized_crop is not None
                    else None
                ),
                resized_crop_base64=_encode_rgb_png(resized_crop),
                color_format="RGB",
                alignment_applied=(
                    face.alignment_metadata.alignment_applied
                    if face.alignment_metadata is not None
                    else False
                ),
            )
        )

    height, width = result.original_frame_dimensions
    return PreprocessResponse(
        face_count=len(faces),
        image_dimensions=ImageDimensionsResponse(width=width, height=height),
        quality_score=result.quality_assessment.quality_score,
        quality_flags=list(result.quality_assessment.warning_flags),
        quality_metrics=result.quality_assessment.metrics.to_dict(),
        faces=faces,
    )


def create_app(
    *,
    pipeline_factory: PipelineFactory | None = None,
    max_upload_bytes: int | None = None,
) -> FastAPI:
    """Create an API instance with injectable pipeline and upload-size limit."""
    upload_limit = _configured_upload_limit(max_upload_bytes)
    factory = (
        pipeline_factory
        if pipeline_factory is not None
        else FacePreprocessingPipeline
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        pipeline = factory()
        application.state.pipeline = pipeline
        application.state.inference_lock = threading.Lock()
        application.state.models_initialized = True
        try:
            yield
        finally:
            application.state.models_initialized = False
            application.state.pipeline = None
            pipeline.close()

    application = FastAPI(
        title="Face Preprocessing API",
        description=(
            "Local face detection and preprocessing. No spoof classification "
            "or identity matching is performed."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )
    application.state.pipeline = None
    application.state.models_initialized = False
    application.state.max_upload_bytes = upload_limit

    @application.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        initialized = bool(request.app.state.models_initialized)
        return HealthResponse(
            status="ok" if initialized else "unavailable",
            models_initialized=initialized,
        )

    @application.post("/detect", response_model=DetectResponse)
    def detect(
        request: Request,
        file: UploadFile = File(...),
    ) -> DetectResponse:
        image = _read_valid_image(file, request.app.state.max_upload_bytes)
        pipeline: PipelineProtocol | None = request.app.state.pipeline
        if pipeline is None:
            raise HTTPException(status_code=503, detail="Preprocessing models are unavailable.")
        try:
            with request.app.state.inference_lock:
                detections = pipeline.detector.detect(image)
        except FaceDetectorError as exc:
            LOGGER.exception("Face detection failed.")
            raise HTTPException(
                status_code=500, detail=PROCESSING_ERROR_DETAIL
            ) from exc

        faces = [
            DetectedFaceResponse(
                bounding_box=_box_response(detection.bounding_box),
                confidence=detection.confidence,
            )
            for detection in detections
        ]
        height, width = image.pixels.shape[:2]
        return DetectResponse(
            face_count=len(faces),
            image_dimensions=ImageDimensionsResponse(width=width, height=height),
            faces=faces,
        )

    @application.post("/preprocess", response_model=PreprocessResponse)
    def preprocess(
        request: Request,
        file: UploadFile = File(...),
    ) -> PreprocessResponse:
        image = _read_valid_image(file, request.app.state.max_upload_bytes)
        pipeline: PipelineProtocol | None = request.app.state.pipeline
        if pipeline is None:
            raise HTTPException(status_code=503, detail="Preprocessing models are unavailable.")
        try:
            with request.app.state.inference_lock:
                result = pipeline.process_image(image)
            return _preprocess_response(result)
        except (
            FaceDetectorError,
            LandmarkExtractorError,
            FacePreprocessingPipelineError,
        ) as exc:
            LOGGER.exception("Face preprocessing failed.")
            raise HTTPException(
                status_code=500, detail=PROCESSING_ERROR_DETAIL
            ) from exc

    return application


app = create_app()


__all__ = [
    "app",
    "create_app",
]
