from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np

from app.face_detector import BoundingBox
from app.live_preprocessing_demo import (
    CROP_WINDOW,
    PREVIEW_WINDOW,
    build_crop_preview,
    draw_preview,
    run_live_demo,
)
from app.preprocessing_pipeline import FacePreprocessingConfig
from app.video_processor import VideoFrame


def _result(
    *,
    faces: list[SimpleNamespace] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        faces=faces or [],
        frame_index=12,
        quality_assessment=SimpleNamespace(
            quality_score=87.0,
            warning_flags=("blur", "low_resolution"),
            metrics=SimpleNamespace(blur_variance=123.456),
        ),
    )


def test_preview_draws_boxes_and_keeps_source_rgb_frame_unchanged() -> None:
    face = SimpleNamespace(
        face_id=0,
        bounding_box=BoundingBox(5, 5, 10, 10),
        detection_confidence=0.91,
        landmarks=None,
        resized_face_crop=None,
        quality_flags=(),
    )
    result = _result(faces=[face])
    frame_rgb = np.zeros((240, 320, 3), dtype=np.uint8)
    frame_rgb[:, :] = (255, 0, 0)
    original = frame_rgb.copy()

    preview = draw_preview(frame_rgb, result, 15.0)

    assert preview.shape == frame_rgb.shape
    assert preview.dtype == np.uint8
    np.testing.assert_array_equal(preview[239, 319], np.array([0, 0, 255]))
    np.testing.assert_array_equal(preview[5, 5], np.array([0, 255, 0]))
    np.testing.assert_array_equal(frame_rgb, original)


def test_face_labels_are_one_based_and_quality_text_includes_blur_metric(
    monkeypatch,
) -> None:
    face = SimpleNamespace(
        face_id=0,
        detection_index=0,
        bounding_box=BoundingBox(5, 5, 10, 10),
        detection_confidence=0.91,
        landmarks=None,
        resized_face_crop=None,
        quality_flags=(),
    )
    texts: list[str] = []
    original_put_text = cv2.putText

    def capture_text(
        image: np.ndarray,
        text: str,
        origin: tuple[int, int],
        font: int,
        scale: float,
        color: tuple[int, int, int],
        thickness: int,
        line_type: int,
    ) -> np.ndarray:
        texts.append(text)
        return original_put_text(
            image, text, origin, font, scale, color, thickness, line_type
        )

    monkeypatch.setattr("app.live_preprocessing_demo.cv2.putText", capture_text)
    draw_preview(np.zeros((240, 320, 3), dtype=np.uint8), _result(faces=[face]), 15.0)

    assert "Face 1  confidence 0.91" in texts
    assert not any(text.startswith("Face 0") for text in texts)
    assert "Blur variance: 123.5" in texts
    assert any("blur" in text for text in texts)
    assert any("low_resolution" in text for text in texts)
    assert face.detection_index == 0


def test_face_labels_and_backgrounds_stay_inside_boundary_faces(monkeypatch) -> None:
    faces = [
        SimpleNamespace(
            face_id=index,
            detection_index=index,
            bounding_box=box,
            detection_confidence=0.9,
            landmarks=None,
            resized_face_crop=None,
            quality_flags=(),
        )
        for index, box in enumerate(
            (BoundingBox(0, 0, 20, 20), BoundingBox(145, 75, 15, 15))
        )
    ]
    frame = np.zeros((90, 160, 3), dtype=np.uint8)
    text_origins: list[tuple[str, tuple[int, int], tuple[int, int], int]] = []
    label_rectangles: list[tuple[tuple[int, int], tuple[int, int]]] = []
    original_put_text = cv2.putText
    original_rectangle = cv2.rectangle

    def capture_text(
        image: np.ndarray,
        text: str,
        origin: tuple[int, int],
        font: int,
        scale: float,
        color: tuple[int, int, int],
        thickness: int,
        line_type: int,
    ) -> np.ndarray:
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, scale, thickness
        )
        text_origins.append(
            (text, origin, (text_width, text_height), baseline)
        )
        return original_put_text(
            image, text, origin, font, scale, color, thickness, line_type
        )

    def capture_rectangle(
        image: np.ndarray,
        first: tuple[int, int],
        second: tuple[int, int],
        color: tuple[int, int, int],
        thickness: int,
    ) -> np.ndarray:
        if thickness == -1:
            label_rectangles.append((first, second))
        return original_rectangle(image, first, second, color, thickness)

    monkeypatch.setattr("app.live_preprocessing_demo.cv2.putText", capture_text)
    monkeypatch.setattr(
        "app.live_preprocessing_demo.cv2.rectangle",
        capture_rectangle,
    )

    draw_preview(frame, _result(faces=faces), 10.0)

    height, width = frame.shape[:2]
    assert any(text.startswith("Face 1") for text, *_ in text_origins)
    assert any(text.startswith("Face 2") for text, *_ in text_origins)
    assert len(label_rectangles) >= 2
    for (left, top), (right, bottom) in label_rectangles:
        assert 0 <= left <= right < width
        assert 0 <= top <= bottom < height
    for _, (x, y), (text_width, text_height), baseline in text_origins:
        assert x >= 0
        assert x + text_width <= width
        assert y - text_height >= 0
        assert y + baseline <= height


def test_crop_preview_converts_rgb_to_bgr_and_handles_no_face() -> None:
    crop_rgb = np.zeros((224, 224, 3), dtype=np.uint8)
    crop_rgb[:, :] = (200, 30, 10)
    face = SimpleNamespace(resized_face_crop=crop_rgb)

    crop_preview = build_crop_preview(_result(faces=[face]))
    empty_preview = build_crop_preview(_result())

    np.testing.assert_array_equal(crop_preview[100, 100], np.array([10, 30, 200]))
    assert crop_preview.shape == empty_preview.shape == (224, 224, 3)
    assert empty_preview.dtype == np.uint8


def test_live_demo_closes_frame_stream_pipeline_and_windows(monkeypatch) -> None:
    rgb_frame = np.zeros((40, 40, 3), dtype=np.uint8)
    original_frame = rgb_frame.copy()
    stream_closed = False
    displayed_windows: list[str] = []
    displayed_images: list[np.ndarray] = []
    destroyed_windows = False
    configs: list[FacePreprocessingConfig] = []

    def frame_stream():
        nonlocal stream_closed
        try:
            yield VideoFrame(
                frame_index=0,
                timestamp_ms=0.0,
                original_frame_dimensions=(40, 40),
                frame_rgb=rgb_frame,
                source_type="webcam",
                timestamp_source="monotonic",
            )
        finally:
            stream_closed = True

    class FakeVideoProcessor:
        def process_webcam(self, camera_index: int):
            assert camera_index == 2
            return frame_stream()

    class FakePipeline:
        def __init__(self, config: FacePreprocessingConfig) -> None:
            configs.append(config)
            self.video_processor = FakeVideoProcessor()
            self.closed = False

        def process_frame(
            self,
            frame_rgb: np.ndarray,
            frame_index: int,
            timestamp_ms: float,
            *,
            timestamp_source: str,
        ) -> SimpleNamespace:
            assert frame_rgb is rgb_frame
            assert frame_index == 0
            assert timestamp_ms == 0.0
            assert timestamp_source == "monotonic"
            return _result()

        def close(self) -> None:
            self.closed = True

    pipeline: FakePipeline | None = None

    def fake_pipeline_factory(config: FacePreprocessingConfig) -> FakePipeline:
        nonlocal pipeline
        pipeline = FakePipeline(config)
        return pipeline

    def show(window_name: str, image: np.ndarray) -> None:
        displayed_windows.append(window_name)
        displayed_images.append(image)

    def quit_immediately(delay: int) -> int:
        assert delay == 1
        return ord("q")

    monkeypatch.setattr(
        "app.live_preprocessing_demo.FacePreprocessingPipeline",
        fake_pipeline_factory,
    )
    monkeypatch.setattr("app.live_preprocessing_demo.cv2.imshow", show)
    monkeypatch.setattr(
        "app.live_preprocessing_demo.cv2.waitKey",
        quit_immediately,
    )

    def destroy_all_windows() -> None:
        nonlocal destroyed_windows
        destroyed_windows = True

    monkeypatch.setattr(
        "app.live_preprocessing_demo.cv2.destroyAllWindows",
        destroy_all_windows,
    )

    run_live_demo(camera_index=2, sample_every_n_frames=3)

    assert displayed_windows == [PREVIEW_WINDOW, CROP_WINDOW]
    assert len(displayed_images) == 2
    assert stream_closed
    assert pipeline is not None and pipeline.closed
    assert destroyed_windows
    assert configs[0].video.sample_every_n_frames == 3
    assert configs[0].aligner.enabled is False
    np.testing.assert_array_equal(rgb_frame, original_frame)
