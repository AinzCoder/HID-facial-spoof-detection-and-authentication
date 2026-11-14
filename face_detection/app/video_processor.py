"""Sequential video and webcam frame capture with RGB callback integration."""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Callable, Iterator, Literal, Protocol

import cv2
import numpy as np

from app.face_detector import InvalidImageError, _load_rgb

LOGGER = logging.getLogger(__name__)

SourceType = Literal["video", "webcam"]
TimestampSource = Literal["container", "fps_fallback", "monotonic"]
CaptureSource = str | int
VideoSource = str | os.PathLike[str]
FrameCallback = Callable[["VideoFrame"], bool | None]


class VideoProcessorError(RuntimeError):
    """Raised when a video or webcam source cannot be opened or read."""


class _Capture(Protocol):
    def isOpened(self) -> bool: ...

    def read(self) -> tuple[bool, np.ndarray | None]: ...

    def get(self, property_id: int) -> float: ...

    def release(self) -> None: ...


@dataclass(frozen=True, slots=True)
class VideoProcessorConfig:
    """Frame sampling and timestamp fallback settings."""

    sample_every_n_frames: int = 1
    max_processed_frames: int | None = None
    fallback_fps: float = 30.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.sample_every_n_frames, bool)
            or not isinstance(self.sample_every_n_frames, int)
            or self.sample_every_n_frames < 1
        ):
            raise ValueError("sample_every_n_frames must be a positive integer.")
        if self.max_processed_frames is not None and (
            isinstance(self.max_processed_frames, bool)
            or not isinstance(self.max_processed_frames, int)
            or self.max_processed_frames < 1
        ):
            raise ValueError("max_processed_frames must be a positive integer or None.")
        if not math.isfinite(self.fallback_fps) or self.fallback_fps <= 0:
            raise ValueError("fallback_fps must be finite and positive.")


@dataclass(frozen=True, slots=True)
class VideoFrame:
    """One sampled source frame and its original temporal metadata.

    Dimension tuples use ``(height, width)``, matching ``frame_rgb.shape``.
    ``timestamp_source`` identifies the timestamp basis actually used.
    """

    frame_index: int
    timestamp_ms: float
    original_frame_dimensions: tuple[int, int]
    frame_rgb: np.ndarray
    source_type: SourceType
    timestamp_source: TimestampSource


class VideoProcessor:
    """Yield frames sequentially without saving or enhancing biometric data."""

    def __init__(
        self,
        config: VideoProcessorConfig | None = None,
        *,
        capture_factory: Callable[[CaptureSource], _Capture] | None = None,
    ) -> None:
        self.config = config or VideoProcessorConfig()
        self._capture_factory = capture_factory or cv2.VideoCapture

    def process_video(
        self,
        source: VideoSource,
        callback: FrameCallback | None = None,
    ) -> Iterator[VideoFrame]:
        """Yield sampled video frames in source order.

        OpenCV container timestamps are used when they are finite, non-negative,
        and strictly increasing. Invalid or repeated timestamps fall back to
        frame-index/FPS time, using ``fallback_fps`` if the reported FPS is bad.
        Container timestamp quality depends on the codec and OpenCV backend.
        """
        yield from self._process_capture(str(source), "video", callback)

    def process_webcam(
        self,
        camera_index: int = 0,
        callback: FrameCallback | None = None,
    ) -> Iterator[VideoFrame]:
        """Yield webcam frames with elapsed timestamps from a monotonic clock.

        Webcam timestamps start when capture begins and measure elapsed host
        time after each sampled frame is read; they are not camera exposure
        timestamps and can include driver and scheduling latency.
        """
        if (
            isinstance(camera_index, bool)
            or not isinstance(camera_index, int)
            or camera_index < 0
        ):
            raise ValueError("camera_index must be a non-negative integer.")
        yield from self._process_capture(camera_index, "webcam", callback)

    def _process_capture(
        self,
        source: CaptureSource,
        source_type: SourceType,
        callback: FrameCallback | None,
    ) -> Iterator[VideoFrame]:
        try:
            capture = self._capture_factory(source)
        except (cv2.error, OSError) as exc:
            raise VideoProcessorError(
                f"Could not initialize {source_type} capture for {source!r}."
            ) from exc
        if capture is None:
            raise VideoProcessorError(
                f"Could not initialize {source_type} capture for {source!r}."
            )

        try:
            try:
                is_open = capture.isOpened()
            except cv2.error as exc:
                raise VideoProcessorError(
                    f"Could not initialize {source_type} source {source!r}."
                ) from exc
            if not is_open:
                raise VideoProcessorError(
                    f"Could not open {source_type} source {source!r}."
                )

            fps = self.config.fallback_fps
            if source_type == "video":
                try:
                    reported_fps = float(capture.get(cv2.CAP_PROP_FPS))
                except (cv2.error, TypeError, ValueError):
                    reported_fps = float("nan")
                if math.isfinite(reported_fps) and reported_fps > 0:
                    fps = reported_fps

            webcam_start_ms = (
                time.monotonic() * 1000.0 if source_type == "webcam" else 0.0
            )
            previous_timestamp_ms: float | None = None
            frame_index = 0
            processed_count = 0

            while (
                self.config.max_processed_frames is None
                or processed_count < self.config.max_processed_frames
            ):
                try:
                    has_frame, frame = capture.read()
                except cv2.error as exc:
                    raise VideoProcessorError(
                        f"Could not read from {source_type} source {source!r}."
                    ) from exc
                if not has_frame:
                    break

                current_frame_index = frame_index
                frame_index += 1
                if current_frame_index % self.config.sample_every_n_frames != 0:
                    continue
                if not isinstance(frame, np.ndarray) or frame.size == 0:
                    LOGGER.warning(
                        "Skipping empty %s frame %d from %r.",
                        source_type,
                        current_frame_index,
                        source,
                    )
                    continue

                original_dimensions = frame.shape[:2]
                try:
                    frame_rgb = _load_rgb(frame)
                except InvalidImageError:
                    LOGGER.warning(
                        "Skipping invalid %s frame %d from %r.",
                        source_type,
                        current_frame_index,
                        source,
                    )
                    continue

                if source_type == "video":
                    timestamp_ms, timestamp_source = self._video_timestamp(
                        capture,
                        current_frame_index,
                        fps,
                        previous_timestamp_ms,
                    )
                else:
                    timestamp_ms = max(
                        0.0, time.monotonic() * 1000.0 - webcam_start_ms
                    )
                    if previous_timestamp_ms is not None:
                        timestamp_ms = max(timestamp_ms, previous_timestamp_ms)
                    timestamp_source = "monotonic"
                previous_timestamp_ms = timestamp_ms

                result = VideoFrame(
                    frame_index=current_frame_index,
                    timestamp_ms=timestamp_ms,
                    original_frame_dimensions=original_dimensions,
                    frame_rgb=frame_rgb,
                    source_type=source_type,
                    timestamp_source=timestamp_source,
                )
                should_continue = callback(result) if callback is not None else True
                processed_count += 1
                yield result
                if should_continue is False:
                    break
        finally:
            capture.release()

    def _video_timestamp(
        self,
        capture: _Capture,
        frame_index: int,
        fps: float,
        previous_timestamp_ms: float | None,
    ) -> tuple[float, TimestampSource]:
        try:
            container_timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC))
        except (cv2.error, TypeError, ValueError, OverflowError):
            container_timestamp = float("nan")
        if (
            math.isfinite(container_timestamp)
            and container_timestamp >= 0
            and (
                previous_timestamp_ms is None
                or container_timestamp > previous_timestamp_ms
            )
        ):
            return container_timestamp, "container"

        frame_period_ms = 1000.0 / fps
        fallback_timestamp = frame_index * frame_period_ms
        if previous_timestamp_ms is not None:
            fallback_timestamp = max(
                fallback_timestamp, previous_timestamp_ms + frame_period_ms
            )
        return fallback_timestamp, "fps_fallback"


__all__ = [
    "FrameCallback",
    "SourceType",
    "TimestampSource",
    "VideoFrame",
    "VideoProcessor",
    "VideoProcessorConfig",
    "VideoProcessorError",
]
