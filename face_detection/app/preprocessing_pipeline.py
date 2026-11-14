"""Reusable face preprocessing pipeline for downstream model consumers.

Example:

    with FacePreprocessingPipeline() as pipeline:
        result = pipeline.process_image("portrait.jpg")
        for face in result.faces:
            model_input = face.resized_face_crop
            # Pass model_input to a downstream model when it is not None.

Alignment is disabled by default. Quality warnings describe image usability
only; the pipeline does not classify genuine versus spoofed samples.
"""

from __future__ import annotations

import logging
import math
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Iterator, Literal

import cv2
import numpy as np

from app.face_aligner import (
    FaceAligner,
    FaceAlignerConfig,
    FaceAlignmentError,
    FaceAlignmentResult,
)
from app.face_cropper import (
    FaceCropError,
    FaceCropper,
    FaceCropperConfig,
)
from app.face_detector import (
    BoundingBox,
    FaceDetector,
    FaceDetectorConfig,
    ImageInput,
    RGBImage,
    _load_rgb,
)
from app.landmark_extractor import (
    FaceLandmarks,
    LandmarkExtractor,
    LandmarkExtractorConfig,
)
from app.quality_validator import (
    FaceQualityMetrics,
    QualityAssessment,
    QualityValidator,
    QualityValidatorConfig,
)
from app.video_processor import (
    VideoFrame,
    VideoProcessor,
    VideoProcessorConfig,
    VideoSource,
)

LOGGER = logging.getLogger(__name__)

PipelineSourceType = Literal["image", "frame", "video", "webcam"]
_FACE_QUALITY_FLAGS = frozenset(
    {"no_face_detected", "small_face", "clipped_face", "non_frontal"}
)


class FacePreprocessingPipelineError(RuntimeError):
    """Raised when an operation is requested from a closed pipeline."""


@dataclass(frozen=True, slots=True)
class FacePreprocessingConfig:
    """Configuration for the existing detector, preprocessing, and video modules."""

    detector: FaceDetectorConfig = field(default_factory=FaceDetectorConfig)
    landmarker: LandmarkExtractorConfig = field(default_factory=LandmarkExtractorConfig)
    quality: QualityValidatorConfig = field(default_factory=QualityValidatorConfig)
    cropper: FaceCropperConfig = field(default_factory=FaceCropperConfig)
    aligner: FaceAlignerConfig = field(
        default_factory=lambda: FaceAlignerConfig(enabled=False)
    )
    video: VideoProcessorConfig = field(default_factory=VideoProcessorConfig)


@dataclass(frozen=True, slots=True)
class PreprocessedFace:
    """One detection's associated metadata and model-ready image variants."""

    face_id: int
    detection_index: int
    bounding_box: BoundingBox
    detection_confidence: float | None
    landmarks: FaceLandmarks | None
    quality_metrics: FaceQualityMetrics | None
    quality_flags: tuple[str, ...]
    context_face_crop: np.ndarray | None
    resized_face_crop: np.ndarray | None
    optional_aligned_crop: np.ndarray | None
    crop_coordinates: BoundingBox | None
    alignment_metadata: FaceAlignmentResult | None
    processing_warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PreprocessingResult:
    """A complete image/frame result with preserved source-time metadata."""

    faces: tuple[PreprocessedFace, ...]
    quality_assessment: QualityAssessment
    original_frame_dimensions: tuple[int, int]
    source_type: PipelineSourceType
    frame_index: int | None = None
    timestamp_ms: float | None = None
    timestamp_source: str | None = None


class FacePreprocessingPipeline:
    """Connect the existing face modules into one reusable processing interface.

    The pipeline owns the detector and landmarker instances it receives and
    closes them on :meth:`close`. Injected dependencies are useful for tests and
    transfer their lifecycle to the pipeline.
    """

    def __init__(
        self,
        config: FacePreprocessingConfig | None = None,
        *,
        detector: FaceDetector | None = None,
        landmark_extractor: LandmarkExtractor | None = None,
        quality_validator: QualityValidator | None = None,
        face_cropper: FaceCropper | None = None,
        face_aligner: FaceAligner | None = None,
        video_processor: VideoProcessor | None = None,
    ) -> None:
        self.config = config or FacePreprocessingConfig()
        self._closed = False

        resources = ExitStack()
        with resources:
            self.detector = (
                detector
                if detector is not None
                else FaceDetector(config=self.config.detector)
            )
            resources.callback(self.detector.close)
            self.landmark_extractor = (
                landmark_extractor
                if landmark_extractor is not None
                else LandmarkExtractor(config=self.config.landmarker)
            )
            resources.callback(self.landmark_extractor.close)
            self.quality_validator = (
                quality_validator
                if quality_validator is not None
                else QualityValidator(self.config.quality)
            )
            self.face_cropper = (
                face_cropper
                if face_cropper is not None
                else FaceCropper(self.config.cropper)
            )
            self.face_aligner = (
                face_aligner
                if face_aligner is not None
                else FaceAligner(self.config.aligner)
            )
            self.video_processor = (
                video_processor
                if video_processor is not None
                else VideoProcessor(self.config.video)
            )
            self._resources = resources.pop_all()

    def _ensure_open(self) -> None:
        if self._closed:
            raise FacePreprocessingPipelineError(
                "The preprocessing pipeline has already been closed."
            )

    def process_image(self, image: ImageInput) -> PreprocessingResult:
        """Validate and preprocess an image; NumPy input follows OpenCV BGR order."""
        self._ensure_open()
        rgb_image = RGBImage(_load_rgb(image))
        return self._process_rgb(
            rgb_image,
            source_type="image",
            frame_index=None,
            timestamp_ms=None,
            timestamp_source=None,
        )

    def process_frame(
        self,
        frame_rgb: np.ndarray,
        frame_index: int,
        timestamp_ms: float,
        *,
        timestamp_source: str = "provided",
    ) -> PreprocessingResult:
        """Process an already-RGB video frame without swapping or copying channels."""
        self._ensure_open()
        if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
            raise ValueError("frame_index must be a non-negative integer.")
        if (
            isinstance(timestamp_ms, bool)
            or not isinstance(timestamp_ms, (int, float))
            or not math.isfinite(timestamp_ms)
            or timestamp_ms < 0
        ):
            raise ValueError("timestamp_ms must be finite and non-negative.")
        rgb_image = RGBImage(frame_rgb)
        return self._process_rgb(
            rgb_image,
            source_type="frame",
            frame_index=frame_index,
            timestamp_ms=float(timestamp_ms),
            timestamp_source=timestamp_source,
        )

    def process_video(self, video_path: VideoSource) -> Iterator[PreprocessingResult]:
        """Yield preprocessing results for video frames in original order."""
        self._ensure_open()
        yield from self._process_video_frames(
            self.video_processor.process_video(video_path)
        )

    def process_webcam(
        self,
        camera_index: int = 0,
    ) -> Iterator[PreprocessingResult]:
        """Yield webcam preprocessing results and release capture on iterator close."""
        self._ensure_open()
        yield from self._process_video_frames(
            self.video_processor.process_webcam(camera_index)
        )

    def _process_video_frames(
        self,
        frames: Iterator[VideoFrame],
    ) -> Iterator[PreprocessingResult]:
        try:
            for frame in frames:
                yield self._process_rgb(
                    RGBImage(frame.frame_rgb),
                    source_type=frame.source_type,
                    frame_index=frame.frame_index,
                    timestamp_ms=frame.timestamp_ms,
                    timestamp_source=frame.timestamp_source,
                    original_frame_dimensions=frame.original_frame_dimensions,
                )
        finally:
            close = getattr(frames, "close", None)
            if callable(close):
                close()

    def _process_rgb(
        self,
        image: RGBImage,
        *,
        source_type: PipelineSourceType,
        frame_index: int | None,
        timestamp_ms: float | None,
        timestamp_source: str | None,
        original_frame_dimensions: tuple[int, int] | None = None,
    ) -> PreprocessingResult:
        self._ensure_open()
        height, width = image.pixels.shape[:2]

        detections = self.detector.detect(image)
        landmarks = self.landmark_extractor.extract(
            image,
            detections=detections,
        )
        quality = self.quality_validator.validate(
            image,
            detections=detections,
            landmarks=landmarks,
        )

        landmark_map: dict[int, list[FaceLandmarks]] = {}
        for face_landmarks in landmarks:
            if face_landmarks.detection_index is not None:
                landmark_map.setdefault(
                    face_landmarks.detection_index, []
                ).append(face_landmarks)
        quality_map = {
            metrics.detection_index: metrics
            for metrics in quality.metrics.faces
        }
        image_quality_flags = tuple(
            flag for flag in quality.warning_flags if flag not in _FACE_QUALITY_FLAGS
        )

        processed_faces: list[PreprocessedFace] = []
        for detection_index, detection in enumerate(detections):
            associated_landmarks = landmark_map.get(detection_index, [])
            face_landmarks = (
                associated_landmarks[0] if len(associated_landmarks) == 1 else None
            )
            face_quality = quality_map.get(detection_index)
            quality_flags = list(image_quality_flags)
            if face_quality is not None:
                if face_quality.is_small:
                    quality_flags.append("small_face")
                if face_quality.is_clipped:
                    quality_flags.append("clipped_face")
                if (
                    face_quality.pose is not None
                    and not face_quality.pose.is_approximately_frontal
                ):
                    quality_flags.append("non_frontal")

            processing_warnings: list[str] = []
            if len(associated_landmarks) > 1:
                processing_warnings.append("ambiguous_landmark_association")
            elif face_landmarks is None:
                processing_warnings.append("missing_landmarks")

            try:
                crop = self.face_cropper.crop_face(
                    image,
                    detection,
                    face_id=detection_index,
                )
            except (FaceCropError, cv2.error) as exc:
                LOGGER.warning("Face %d crop failed: %s", detection_index, exc)
                processing_warnings.append("crop_failed")
                processed_faces.append(
                    PreprocessedFace(
                        face_id=detection_index,
                        detection_index=detection_index,
                        bounding_box=detection.bounding_box,
                        detection_confidence=detection.confidence,
                        landmarks=face_landmarks,
                        quality_metrics=face_quality,
                        quality_flags=tuple(dict.fromkeys(quality_flags)),
                        context_face_crop=None,
                        resized_face_crop=None,
                        optional_aligned_crop=None,
                        crop_coordinates=None,
                        alignment_metadata=None,
                        processing_warnings=tuple(processing_warnings),
                    )
                )
                continue

            alignment: FaceAlignmentResult | None
            try:
                if len(associated_landmarks) > 1:
                    alignment = self.face_aligner.align_faces(
                        [crop], landmarks
                    )[0]
                else:
                    alignment = self.face_aligner.align_face(crop, face_landmarks)
                processing_warnings.extend(
                    warning
                    for warning in alignment.warning_flags
                    if warning
                    in {
                        "ambiguous_landmark_association",
                        "detection_index_mismatch",
                    }
                )
            except (FaceAlignmentError, cv2.error) as exc:
                LOGGER.warning("Face %d alignment failed: %s", detection_index, exc)
                alignment = None
                processing_warnings.append("alignment_failed")

            processed_faces.append(
                PreprocessedFace(
                    face_id=detection_index,
                    detection_index=detection_index,
                    bounding_box=detection.bounding_box,
                    detection_confidence=detection.confidence,
                    landmarks=face_landmarks,
                    quality_metrics=face_quality,
                    quality_flags=tuple(dict.fromkeys(quality_flags)),
                    context_face_crop=crop.original_crop,
                    resized_face_crop=crop.resized_crop,
                    optional_aligned_crop=(
                        alignment.aligned_face_crop if alignment is not None else None
                    ),
                    crop_coordinates=crop.crop_coordinates,
                    alignment_metadata=alignment,
                    processing_warnings=tuple(processing_warnings),
                )
            )

        return PreprocessingResult(
            faces=tuple(processed_faces),
            quality_assessment=quality,
            original_frame_dimensions=(
                original_frame_dimensions or (height, width)
            ),
            source_type=source_type,
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            timestamp_source=timestamp_source,
        )

    def close(self) -> None:
        """Close the owned detector and landmarker instances."""
        if not self._closed:
            self._closed = True
            self._resources.close()

    def __enter__(self) -> FacePreprocessingPipeline:
        self._ensure_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = [
    "FacePreprocessingConfig",
    "FacePreprocessingPipeline",
    "FacePreprocessingPipelineError",
    "PreprocessedFace",
    "PreprocessingResult",
]
