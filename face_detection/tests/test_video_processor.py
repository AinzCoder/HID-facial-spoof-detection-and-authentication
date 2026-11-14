from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from app.video_processor import (
    VideoFrame,
    VideoProcessor,
    VideoProcessorConfig,
    VideoProcessorError,
)


class FakeCapture:
    def __init__(
        self,
        frames: list[tuple[bool, np.ndarray | None]],
        *,
        opened: bool = True,
        fps: float = 30.0,
        timestamps: list[float] | None = None,
    ) -> None:
        self.frames = frames
        self.opened = opened
        self.fps = fps
        self.timestamps = timestamps or []
        self.read_count = 0
        self.released = False

    def isOpened(self) -> bool:
        return self.opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.read_count >= len(self.frames):
            return False, None
        result = self.frames[self.read_count]
        self.read_count += 1
        return result

    def get(self, property_id: int) -> float:
        if property_id == cv2.CAP_PROP_FPS:
            return self.fps
        if property_id == cv2.CAP_PROP_POS_MSEC:
            index = max(0, self.read_count - 1)
            return self.timestamps[index] if index < len(self.timestamps) else 0.0
        return 0.0

    def release(self) -> None:
        self.released = True


def _bgr_frame(value: int, shape: tuple[int, int] = (3, 4)) -> np.ndarray:
    frame = np.empty((*shape, 3), dtype=np.uint8)
    frame[:] = (value, value + 1, value + 2)
    return frame


def _video_processor(capture: FakeCapture, config: VideoProcessorConfig | None = None):
    return VideoProcessor(
        config,
        capture_factory=lambda _source: capture,
    )


def test_video_capture_opens_and_releases() -> None:
    capture = FakeCapture([(True, _bgr_frame(1)), (False, None)])

    frames = list(_video_processor(capture).process_video("sample.mp4"))

    assert len(frames) == 1
    assert capture.released
    assert frames[0].source_type == "video"


def test_frame_order_dimensions_and_rgb_format_are_preserved() -> None:
    source_frames = [_bgr_frame(value) for value in (1, 10, 20)]
    originals = [frame.copy() for frame in source_frames]
    capture = FakeCapture([(True, frame) for frame in source_frames] + [(False, None)])

    frames = list(_video_processor(capture).process_video("sample.mp4"))

    assert [frame.frame_index for frame in frames] == [0, 1, 2]
    assert [int(frame.frame_rgb[0, 0, 0]) for frame in frames] == [3, 12, 22]
    assert all(frame.frame_rgb.dtype == np.uint8 for frame in frames)
    assert all(frame.frame_rgb.shape == (3, 4, 3) for frame in frames)
    assert all(frame.original_frame_dimensions == (3, 4) for frame in frames)
    for frame, original in zip(source_frames, originals):
        np.testing.assert_array_equal(frame, original)


def test_sampling_keeps_original_frame_indices() -> None:
    capture = FakeCapture(
        [(True, _bgr_frame(value)) for value in range(5)] + [(False, None)]
    )
    config = VideoProcessorConfig(sample_every_n_frames=2)

    frames = list(_video_processor(capture, config).process_video("sample.mp4"))

    assert [frame.frame_index for frame in frames] == [0, 2, 4]


def test_container_timestamps_are_used_when_strictly_increasing() -> None:
    capture = FakeCapture(
        [(True, _bgr_frame(value)) for value in range(3)] + [(False, None)],
        timestamps=[0.0, 40.0, 85.0],
    )

    frames = list(_video_processor(capture).process_video("sample.mp4"))

    assert [frame.timestamp_ms for frame in frames] == [0.0, 40.0, 85.0]
    assert [frame.timestamp_source for frame in frames] == [
        "container",
        "container",
        "container",
    ]


def test_invalid_container_timestamps_fall_back_to_frame_index_and_fps() -> None:
    capture = FakeCapture(
        [(True, _bgr_frame(value)) for value in range(4)] + [(False, None)],
        fps=25.0,
        timestamps=[0.0, 0.0, float("nan"), -1.0],
    )

    frames = list(_video_processor(capture).process_video("sample.mp4"))

    assert [frame.timestamp_ms for frame in frames] == [0.0, 40.0, 80.0, 120.0]
    assert [frame.timestamp_source for frame in frames] == [
        "container",
        "fps_fallback",
        "fps_fallback",
        "fps_fallback",
    ]


def test_invalid_reported_fps_uses_configured_fallback_rate() -> None:
    capture = FakeCapture(
        [(True, _bgr_frame(1)), (True, _bgr_frame(2)), (False, None)],
        fps=float("nan"),
        timestamps=[float("nan"), float("nan")],
    )
    config = VideoProcessorConfig(fallback_fps=20.0)

    frames = list(_video_processor(capture, config).process_video("sample.mp4"))

    assert [frame.timestamp_ms for frame in frames] == [0.0, 50.0]
    assert all(frame.timestamp_source == "fps_fallback" for frame in frames)


def test_webcam_timestamps_use_monotonic_elapsed_time(monkeypatch) -> None:
    capture = FakeCapture(
        [(True, _bgr_frame(1)), (True, _bgr_frame(2)), (False, None)]
    )
    clock_values = iter((10.0, 10.025, 10.060))
    monkeypatch.setattr("app.video_processor.time.monotonic", lambda: next(clock_values))

    frames = list(_video_processor(capture).process_webcam())

    assert [frame.timestamp_ms for frame in frames] == pytest.approx([25.0, 60.0])
    assert all(frame.timestamp_source == "monotonic" for frame in frames)
    assert all(frame.source_type == "webcam" for frame in frames)
    assert capture.released


def test_invalid_video_source_raises_and_releases_capture() -> None:
    capture = FakeCapture([], opened=False)

    with pytest.raises(VideoProcessorError, match="Could not open video"):
        list(_video_processor(capture).process_video("missing.mp4"))

    assert capture.released


def test_webcam_initialization_error_is_reported() -> None:
    capture = FakeCapture([], opened=False)

    with pytest.raises(VideoProcessorError, match="Could not open webcam"):
        list(_video_processor(capture).process_webcam(0))

    assert capture.released


def test_negative_webcam_index_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        list(VideoProcessor().process_webcam(-1))


def test_empty_frames_are_skipped_without_losing_source_indices() -> None:
    capture = FakeCapture(
        [
            (True, np.empty((0, 0, 3), dtype=np.uint8)),
            (True, None),
            (True, _bgr_frame(7)),
            (False, None),
        ],
        timestamps=[0.0, 33.0, 66.0],
    )

    frames = list(_video_processor(capture).process_video("sample.mp4"))

    assert [frame.frame_index for frame in frames] == [2]
    assert frames[0].timestamp_ms == 66.0
    assert capture.released


def test_callback_is_invoked_and_can_interrupt_processing() -> None:
    capture = FakeCapture(
        [(True, _bgr_frame(value)) for value in range(5)] + [(False, None)]
    )
    callback_indices: list[int] = []

    def stop_after_two(frame: VideoFrame) -> bool:
        callback_indices.append(frame.frame_index)
        return len(callback_indices) < 2

    frames = list(
        _video_processor(capture).process_video(
            "sample.mp4", callback=stop_after_two
        )
    )

    assert [frame.frame_index for frame in frames] == [0, 1]
    assert callback_indices == [0, 1]
    assert capture.released


def test_maximum_processed_frames_is_respected() -> None:
    capture = FakeCapture(
        [(True, _bgr_frame(value)) for value in range(6)] + [(False, None)]
    )
    processor = _video_processor(
        capture,
        VideoProcessorConfig(sample_every_n_frames=2, max_processed_frames=2),
    )

    frames = list(processor.process_video("sample.mp4"))

    assert [frame.frame_index for frame in frames] == [0, 2]
    assert capture.read_count == 3
    assert capture.released


def test_callback_exception_still_releases_capture() -> None:
    capture = FakeCapture([(True, _bgr_frame(1))])

    def fail(_frame: VideoFrame) -> None:
        raise RuntimeError("callback failed")

    with pytest.raises(RuntimeError, match="callback failed"):
        list(_video_processor(capture).process_video("sample.mp4", callback=fail))

    assert capture.released


def test_closing_generator_releases_capture() -> None:
    capture = FakeCapture([(True, _bgr_frame(1)), (True, _bgr_frame(2))])
    frames = _video_processor(capture).process_video("sample.mp4")

    next(frames)
    frames.close()

    assert capture.released


def test_invalid_sampling_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="sample_every_n_frames"):
        VideoProcessorConfig(sample_every_n_frames=0)
    with pytest.raises(ValueError, match="max_processed_frames"):
        VideoProcessorConfig(max_processed_frames=0)
    with pytest.raises(ValueError, match="fallback_fps"):
        VideoProcessorConfig(fallback_fps=math.nan)
