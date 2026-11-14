"""Live webcam preview using the reusable face preprocessing pipeline."""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Iterator, Sequence

import cv2
import numpy as np

from app.face_aligner import FaceAlignerConfig
from app.face_detector import FaceDetectorError
from app.landmark_extractor import LandmarkExtractorError
from app.preprocessing_pipeline import (
    FacePreprocessingConfig,
    FacePreprocessingPipeline,
    PreprocessingResult,
)
from app.video_processor import VideoFrame, VideoProcessorConfig, VideoProcessorError

LOGGER = logging.getLogger(__name__)
PREVIEW_WINDOW = "Face preprocessing - webcam"
CROP_WINDOW = "Processed face crop - first detected face"
DEFAULT_CROP_SIZE = (224, 224)


def draw_preview(
    frame_rgb: np.ndarray,
    result: PreprocessingResult,
    fps: float,
    *,
    show_landmarks: bool = False,
) -> np.ndarray:
    """Draw detections and quality information onto a new OpenCV BGR preview."""
    preview = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    height, _ = preview.shape[:2]
    quality = result.quality_assessment
    lines = [
        f"Faces: {len(result.faces)}",
        f"Frame: {result.frame_index if result.frame_index is not None else 'n/a'}",
        f"Approx FPS: {fps:.1f}",
        f"Quality: {quality.quality_score:.0f}/100",
        f"Blur variance: {quality.metrics.blur_variance:.1f}",
    ]
    lines.extend(
        _warning_lines(quality.warning_flags, max_text_width=max(1, preview.shape[1] - 24))
    )
    line_height = 26
    first_y = max(0, height - line_height * len(lines) - 6)
    for index, text in enumerate(lines):
        _draw_label(
            preview,
            text,
            8,
            first_y + index * line_height,
        )

    for face in result.faces:
        box = face.bounding_box
        top_left = (box.x, box.y)
        bottom_right = (box.x2 - 1, box.y2 - 1)
        cv2.rectangle(preview, top_left, bottom_right, (0, 255, 0), 2)

        if show_landmarks and face.landmarks is not None:
            for point in face.landmarks.landmarks:
                cv2.circle(
                    preview,
                    (round(point.pixel.x), round(point.pixel.y)),
                    1,
                    (0, 255, 255),
                    -1,
                )

        confidence = (
            f"{face.detection_confidence:.2f}"
            if face.detection_confidence is not None
            else "n/a"
        )
        label = f"Face {face.face_id + 1}  confidence {confidence}"
        _draw_label(
            preview,
            label,
            box.x,
            box.y,
            place_above=True,
            below_y=box.y2,
        )
    return preview


def _draw_label(
    preview: np.ndarray,
    text: str,
    x: int,
    y: int,
    *,
    place_above: bool = False,
    below_y: int | None = None,
) -> None:
    """Draw high-contrast text fully inside the preview bounds."""
    height, width = preview.shape[:2]
    if height < 1 or width < 1:
        return

    padding = min(3, (width - 1) // 2, (height - 1) // 2)
    maximum_text_width = width - 2 * padding
    scale = 0.55
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, scale, 2
        )
        if (
            text_width <= maximum_text_width
            and text_height + baseline + 2 * padding <= height
        ):
            break
        if scale > 0.25:
            scale = max(0.25, scale - 0.05)
            continue
        text = _truncate_text(text, maximum_text_width, font, scale)
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, scale, 2
        )
        break

    if (
        not text
        or text_width > maximum_text_width
        or text_height + baseline + 2 * padding > height
    ):
        return
    label_width = text_width + 2 * padding
    label_height = text_height + baseline + 2 * padding
    left = min(max(x, 0), width - label_width)
    if place_above and y >= label_height:
        top = y - label_height
    elif place_above and below_y is not None:
        top = below_y
    else:
        top = y
    top = min(max(top, 0), height - label_height)
    right = left + label_width - 1
    bottom = top + label_height - 1

    cv2.rectangle(preview, (left, top), (right, bottom), (0, 0, 0), -1)
    cv2.putText(
        preview,
        text,
        (left + padding, top + padding + text_height),
        font,
        scale,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def _truncate_text(
    text: str,
    maximum_width: int,
    font: int,
    scale: float,
) -> str:
    suffix = "..."
    shortened = text
    while shortened:
        measured_width = cv2.getTextSize(
            shortened + suffix,
            font,
            scale,
            2,
        )[0][0]
        if measured_width <= maximum_width:
            return shortened + suffix
        shortened = shortened[:-1]
    return ""


def _warning_lines(
    warning_flags: tuple[str, ...],
    *,
    max_text_width: int,
) -> list[str]:
    if not warning_flags:
        return ["Warnings: none"]

    lines: list[str] = []
    current = "Warnings: "
    for warning in warning_flags:
        separator = ", " if current != "Warnings: " else ""
        candidate = f"{current}{separator}{warning}"
        candidate_width = cv2.getTextSize(
            candidate,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            2,
        )[0][0]
        if candidate_width <= max_text_width:
            current = candidate
        else:
            if current.strip() != "Warnings:":
                lines.append(current)
            current = f"  {warning}"
    if current.strip():
        lines.append(current)
    return lines


def build_crop_preview(result: PreprocessingResult) -> np.ndarray:
    """Return the first resized RGB face crop as BGR, or a no-face placeholder."""
    if result.faces:
        crop = result.faces[0].resized_face_crop
        if crop is not None:
            return cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)

    width, height = DEFAULT_CROP_SIZE
    placeholder = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        placeholder,
        "No face crop",
        (18, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )
    return placeholder


def run_live_demo(
    camera_index: int = 0,
    sample_every_n_frames: int = 1,
    *,
    show_landmarks: bool = False,
) -> None:
    """Run the live preview; Q quits and all resources are released on exit."""
    config = FacePreprocessingConfig(
        aligner=FaceAlignerConfig(enabled=False),
        video=VideoProcessorConfig(
            sample_every_n_frames=sample_every_n_frames,
        ),
    )
    pipeline = FacePreprocessingPipeline(config)
    frame_stream: Iterator[VideoFrame] | None = None
    previous_frame_time: float | None = None
    smoothed_fps = 0.0
    user_quit = False

    try:
        # Use the pipeline's one VideoProcessor capture and feed its RGB frames
        # through process_frame so the original image is also available to draw.
        frame_stream = pipeline.video_processor.process_webcam(camera_index)
        for captured_frame in frame_stream:
            result = pipeline.process_frame(
                captured_frame.frame_rgb,
                captured_frame.frame_index,
                captured_frame.timestamp_ms,
                timestamp_source=captured_frame.timestamp_source,
            )

            now = time.perf_counter()
            if previous_frame_time is not None:
                elapsed = now - previous_frame_time
                if elapsed > 0:
                    instant_fps = 1.0 / elapsed
                    smoothed_fps = (
                        instant_fps
                        if smoothed_fps == 0
                        else 0.8 * smoothed_fps + 0.2 * instant_fps
                    )
            previous_frame_time = now

            cv2.imshow(
                PREVIEW_WINDOW,
                draw_preview(
                    captured_frame.frame_rgb,
                    result,
                    smoothed_fps,
                    show_landmarks=show_landmarks,
                ),
            )
            cv2.imshow(CROP_WINDOW, build_crop_preview(result))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                user_quit = True
                break

        if not user_quit:
            LOGGER.warning("Webcam stream ended or disconnected.")
    finally:
        try:
            if frame_stream is not None:
                close_stream = getattr(frame_stream, "close", None)
                if callable(close_stream):
                    close_stream()
        finally:
            try:
                pipeline.close()
            finally:
                cv2.destroyAllWindows()


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a non-negative integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Preview face preprocessing from a local webcam."
    )
    parser.add_argument(
        "--camera-index",
        type=_nonnegative_int,
        default=0,
        help="OpenCV camera index (default: 0).",
    )
    parser.add_argument(
        "--sample-every",
        type=_positive_int,
        default=1,
        help="Process every Nth captured frame (default: 1).",
    )
    parser.add_argument(
        "--show-landmarks",
        action="store_true",
        help="Draw all extracted face landmarks on the preview.",
    )
    args = parser.parse_args(argv)

    try:
        run_live_demo(
            camera_index=args.camera_index,
            sample_every_n_frames=args.sample_every,
            show_landmarks=args.show_landmarks,
        )
    except (
        FaceDetectorError,
        LandmarkExtractorError,
        VideoProcessorError,
        cv2.error,
    ) as exc:
        LOGGER.error("Live preprocessing demo failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
