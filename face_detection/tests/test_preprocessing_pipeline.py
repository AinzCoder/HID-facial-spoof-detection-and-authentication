from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
import pytest

from app.face_aligner import (
    FaceAligner,
    FaceAlignerConfig,
    FaceAlignmentError,
)
from app.face_cropper import (
    FaceCropError,
    FaceCropperConfig,
)
from app.face_detector import (
    BoundingBox,
    FaceDetection,
    InvalidImageError,
    RGBImage,
)
from app.landmark_extractor import (
    FaceLandmarks,
    LandmarkPoint,
    NormalizedCoordinate,
    PixelCoordinate,
)
from app.preprocessing_pipeline import (
    FacePreprocessingConfig,
    FacePreprocessingPipeline,
    FacePreprocessingPipelineError,
)
from app.quality_validator import QualityValidatorConfig
from app.video_processor import VideoFrame


class StubDetector:
    def __init__(self, detections: Sequence[FaceDetection]) -> None:
        self.detections = list(detections)
        self.received: list[RGBImage] = []
        self.closed = False

    def detect(self, image: RGBImage) -> list[FaceDetection]:
        self.received.append(image)
        return list(self.detections)

    def close(self) -> None:
        self.closed = True


class StubLandmarker:
    def __init__(self, landmarks: Sequence[FaceLandmarks]) -> None:
        self.landmarks = list(landmarks)
        self.received_detections: list[list[FaceDetection]] = []
        self.closed = False

    def extract(
        self,
        image: RGBImage,
        *,
        detections: Sequence[FaceDetection] | None = None,
    ) -> list[FaceLandmarks]:
        self.received_detections.append(list(detections or ()))
        return list(self.landmarks)

    def close(self) -> None:
        self.closed = True


class StubVideoProcessor:
    def __init__(self, frames: Sequence[VideoFrame]) -> None:
        self.frames = list(frames)
        self.video_stream_closed = False
        self.webcam_stream_closed = False

    def process_video(self, _path: str) -> Iterator[VideoFrame]:
        try:
            yield from self.frames
        finally:
            self.video_stream_closed = True

    def process_webcam(self, _camera_index: int = 0) -> Iterator[VideoFrame]:
        try:
            yield from self.frames
        finally:
            self.webcam_stream_closed = True


class FailingCropper:
    def crop_face(
        self,
        _image: RGBImage,
        _detection: FaceDetection,
        *,
        face_id: int = 0,
    ) -> None:
        raise FaceCropError(f"simulated crop failure for face {face_id}")


class FailingAligner:
    def align_face(
        self,
        _crop: object,
        _landmarks: FaceLandmarks | None,
    ) -> None:
        raise FaceAlignmentError("simulated alignment failure")


def _image(height: int = 140, width: int = 180) -> np.ndarray:
    y, x = np.indices((height, width))
    return np.stack(
        ((x * 7 + y * 3) % 256, (x * 11 + y * 5) % 256, (x * 2 + y * 13) % 256),
        axis=2,
    ).astype(np.uint8)


def _detection(x: int = 40, y: int = 25, width: int = 60, height: int = 70) -> FaceDetection:
    return FaceDetection(BoundingBox(x, y, width, height), confidence=0.93)


def _landmarks(
    detection_index: int,
    box: BoundingBox,
    image_shape: tuple[int, int] = (140, 180),
) -> FaceLandmarks:
    image_height, image_width = image_shape

    def point(index: int, x: float, y: float) -> LandmarkPoint:
        return LandmarkPoint(
            index=index,
            normalized=NormalizedCoordinate(x / image_width, y / image_height, 0.0),
            pixel=PixelCoordinate(x, y),
        )

    left_eye = point(362, box.x + box.width * 0.3, box.y + box.height * 0.35)
    right_eye = point(33, box.x + box.width * 0.7, box.y + box.height * 0.35)
    nose = point(1, box.x + box.width * 0.5, box.y + box.height * 0.55)
    return FaceLandmarks(
        face_index=detection_index,
        detection_index=detection_index,
        landmarks=(left_eye, right_eye, nose),
        regions={
            "left_eye": (left_eye,),
            "right_eye": (right_eye,),
            "nose": (nose,),
            "mouth": (),
        },
    )


def _config(
    *,
    width: int = 1,
    height: int = 1,
    align: bool = False,
) -> FacePreprocessingConfig:
    return FacePreprocessingConfig(
        quality=QualityValidatorConfig(
            min_blur_variance=0,
            min_brightness=0,
            max_brightness=255,
            min_contrast_stddev=0,
            min_width=width,
            min_height=height,
            min_face_area_ratio=0,
        ),
        cropper=FaceCropperConfig(padding_ratio=0.2, target_size=(64, 64)),
        aligner=FaceAlignerConfig(enabled=align, output_size=(80, 80)),
    )


def _pipeline(
    detections: Sequence[FaceDetection] | None = None,
    landmarks: Sequence[FaceLandmarks] = (),
    *,
    config: FacePreprocessingConfig | None = None,
    cropper: object | None = None,
    aligner: object | None = None,
    video_processor: object | None = None,
) -> tuple[FacePreprocessingPipeline, StubDetector, StubLandmarker]:
    detector = StubDetector(detections or ())
    landmarker = StubLandmarker(landmarks)
    pipeline = FacePreprocessingPipeline(
        config or _config(),
        detector=detector,  # type: ignore[arg-type]
        landmark_extractor=landmarker,  # type: ignore[arg-type]
        face_cropper=cropper,  # type: ignore[arg-type]
        face_aligner=aligner,  # type: ignore[arg-type]
        video_processor=video_processor,  # type: ignore[arg-type]
    )
    return pipeline, detector, landmarker


def test_process_image_runs_modules_and_returns_model_ready_rgb_crops() -> None:
    detection = _detection()
    landmarks = _landmarks(0, detection.bounding_box)
    pipeline, detector, landmarker = _pipeline([detection], [landmarks])
    image_bgr = _image()
    original = image_bgr.copy()

    result = pipeline.process_image(image_bgr)

    assert len(result.faces) == 1
    face = result.faces[0]
    assert face.face_id == face.detection_index == 0
    assert face.bounding_box == detection.bounding_box
    assert face.detection_confidence == pytest.approx(0.93)
    assert face.landmarks is landmarks
    assert face.quality_metrics is not None
    assert face.context_face_crop is not None
    assert face.context_face_crop.shape == (98, 84, 3)
    assert face.resized_face_crop is not None
    assert face.resized_face_crop.shape == (64, 64, 3)
    assert face.resized_face_crop.dtype == np.uint8
    assert face.optional_aligned_crop is None
    assert face.alignment_metadata is not None
    assert "alignment_disabled" in face.alignment_metadata.warning_flags
    assert result.original_frame_dimensions == image_bgr.shape[:2]
    assert result.source_type == "image"
    assert result.frame_index is None and result.timestamp_ms is None
    assert not result.quality_assessment.warning_flags
    # process_image keeps the established OpenCV BGR input convention.
    np.testing.assert_array_equal(detector.received[0].pixels, image_bgr[:, :, ::-1])
    np.testing.assert_array_equal(image_bgr, original)
    assert landmarker.received_detections == [[detection]]


def test_process_image_without_faces_returns_quality_warning_and_empty_faces() -> None:
    pipeline, _, _ = _pipeline()

    result = pipeline.process_image(_image())

    assert result.faces == ()
    assert "no_face_detected" in result.quality_assessment.warning_flags


def test_multiple_faces_keep_detection_order_and_landmark_associations() -> None:
    first = _detection(15, 20, 45, 50)
    second = _detection(105, 25, 50, 55)
    first_landmarks = _landmarks(0, first.bounding_box)
    second_landmarks = _landmarks(1, second.bounding_box)
    pipeline, _, _ = _pipeline(
        [first, second],
        [second_landmarks, first_landmarks],
    )

    result = pipeline.process_image(_image())

    assert [face.detection_index for face in result.faces] == [0, 1]
    assert [face.landmarks for face in result.faces] == [
        first_landmarks,
        second_landmarks,
    ]
    assert [face.face_id for face in result.faces] == [0, 1]


def test_quality_warnings_propagate_without_spoof_decisions() -> None:
    detection = _detection()
    pipeline, _, _ = _pipeline(
        [detection],
        config=_config(width=500),
    )

    result = pipeline.process_image(_image())

    assert "low_resolution" in result.quality_assessment.warning_flags
    assert "low_resolution" in result.faces[0].quality_flags
    assert not result.quality_assessment.valid
    assert not hasattr(result.faces[0], "is_spoof")


def test_face_specific_quality_flags_do_not_leak_between_faces() -> None:
    first = _detection(0, 20, 20, 20)
    second = _detection(90, 20, 80, 80)
    config = _config()
    config = FacePreprocessingConfig(
        quality=QualityValidatorConfig(
            min_blur_variance=0,
            min_brightness=0,
            max_brightness=255,
            min_contrast_stddev=0,
            min_width=1,
            min_height=1,
            min_face_area_ratio=0.05,
        ),
        cropper=config.cropper,
        aligner=config.aligner,
        detector=config.detector,
        landmarker=config.landmarker,
        video=config.video,
    )
    pipeline, _, _ = _pipeline([first, second], config=config)

    result = pipeline.process_image(_image())

    assert "small_face" in result.faces[0].quality_flags
    assert "small_face" not in result.faces[1].quality_flags


def test_alignment_is_optional_and_when_enabled_returns_aligned_output() -> None:
    detection = _detection()
    landmarks = _landmarks(0, detection.bounding_box)
    pipeline, _, _ = _pipeline(
        [detection],
        [landmarks],
        config=_config(align=True),
        aligner=FaceAligner(FaceAlignerConfig(enabled=True, output_size=(80, 80))),
    )

    result = pipeline.process_image(_image())

    face = result.faces[0]
    assert face.alignment_metadata is not None
    assert face.alignment_metadata.alignment_applied
    assert face.optional_aligned_crop is not None
    assert face.optional_aligned_crop.shape == (80, 80, 3)
    assert face.optional_aligned_crop.dtype == np.uint8


def test_missing_landmarks_are_reported_and_do_not_prevent_cropping() -> None:
    pipeline, _, _ = _pipeline(
        [_detection()],
        config=_config(align=True),
        aligner=FaceAligner(FaceAlignerConfig(enabled=True, output_size=(80, 80))),
    )

    result = pipeline.process_image(_image())

    face = result.faces[0]
    assert face.landmarks is None
    assert "missing_landmarks" in face.processing_warnings
    assert face.context_face_crop is not None
    assert face.alignment_metadata is not None
    assert "missing_landmarks" in face.alignment_metadata.warning_flags


def test_failed_crop_and_alignment_are_returned_as_processing_warnings() -> None:
    detection = _detection()
    pipeline, _, _ = _pipeline([detection], cropper=FailingCropper())

    failed_crop = pipeline.process_image(_image()).faces[0]

    assert failed_crop.context_face_crop is None
    assert "crop_failed" in failed_crop.processing_warnings

    pipeline.close()
    pipeline, _, _ = _pipeline([detection], aligner=FailingAligner())
    failed_alignment = pipeline.process_image(_image()).faces[0]

    assert failed_alignment.context_face_crop is not None
    assert failed_alignment.optional_aligned_crop is None
    assert "alignment_failed" in failed_alignment.processing_warnings


def test_process_frame_preserves_rgb_buffer_and_frame_metadata() -> None:
    frame_rgb = _image()
    original = frame_rgb.copy()
    pipeline, detector, _ = _pipeline()

    result = pipeline.process_frame(
        frame_rgb,
        frame_index=7,
        timestamp_ms=245.5,
        timestamp_source="container",
    )

    assert result.frame_index == 7
    assert result.timestamp_ms == pytest.approx(245.5)
    assert result.timestamp_source == "container"
    assert result.source_type == "frame"
    assert detector.received[0].pixels is frame_rgb
    np.testing.assert_array_equal(frame_rgb, original)


def test_invalid_image_and_frame_metadata_are_rejected() -> None:
    pipeline, _, _ = _pipeline()

    with pytest.raises(InvalidImageError):
        pipeline.process_image(np.empty((0, 0, 3), dtype=np.uint8))
    with pytest.raises(InvalidImageError):
        pipeline.process_frame(np.zeros((10, 10), dtype=np.uint8), 0, 0.0)
    with pytest.raises(ValueError, match="frame_index"):
        pipeline.process_frame(_image(), -1, 0.0)
    with pytest.raises(ValueError, match="timestamp_ms"):
        pipeline.process_frame(_image(), 0, float("nan"))


def _video_frame(index: int, source_type: str = "video") -> VideoFrame:
    return VideoFrame(
        frame_index=index,
        timestamp_ms=float(index * 40),
        original_frame_dimensions=(140, 180),
        frame_rgb=_image(),
        source_type=source_type,  # type: ignore[arg-type]
        timestamp_source="container" if source_type == "video" else "monotonic",
    )


def test_video_results_preserve_frame_order_and_timestamp_provenance() -> None:
    video = StubVideoProcessor([_video_frame(0), _video_frame(2)])
    pipeline, _, _ = _pipeline(video_processor=video)

    results = list(pipeline.process_video("sample.mp4"))

    assert [result.frame_index for result in results] == [0, 2]
    assert [result.timestamp_ms for result in results] == [0.0, 80.0]
    assert [result.timestamp_source for result in results] == [
        "container",
        "container",
    ]
    assert all(result.source_type == "video" for result in results)
    assert video.video_stream_closed


def test_webcam_results_preserve_source_and_close_stream() -> None:
    video = StubVideoProcessor([_video_frame(5, "webcam")])
    pipeline, _, _ = _pipeline(video_processor=video)

    results = list(pipeline.process_webcam(camera_index=1))

    result = results[0]
    assert result.frame_index == 5
    assert result.timestamp_ms == 200.0
    assert result.timestamp_source == "monotonic"
    assert result.source_type == "webcam"
    assert video.webcam_stream_closed


def test_closing_video_result_iterator_closes_underlying_capture_iterator() -> None:
    video = StubVideoProcessor([_video_frame(0), _video_frame(1)])
    pipeline, _, _ = _pipeline(video_processor=video)
    results = pipeline.process_video("sample.mp4")

    next(results)
    results.close()  # type: ignore[attr-defined]

    assert video.video_stream_closed


def test_pipeline_closes_owned_models_once_and_rejects_reuse() -> None:
    pipeline, detector, landmarker = _pipeline()

    pipeline.close()
    pipeline.close()

    assert detector.closed and landmarker.closed
    with pytest.raises(FacePreprocessingPipelineError, match="closed"):
        pipeline.process_image(_image())


def test_context_manager_closes_injected_models() -> None:
    pipeline, detector, landmarker = _pipeline()

    with pipeline as active:
        assert active is pipeline
        assert not detector.closed

    assert detector.closed and landmarker.closed


def test_rgb_image_marker_reuses_contiguous_pixels_and_copies_noncontiguous() -> None:
    source = _image()
    wrapped = RGBImage(source)
    assert wrapped.pixels is source

    noncontiguous = source[:, ::2]
    copied = RGBImage(noncontiguous)
    assert copied.pixels.flags.c_contiguous
    assert copied.pixels is not noncontiguous
    np.testing.assert_array_equal(copied.pixels, noncontiguous)
