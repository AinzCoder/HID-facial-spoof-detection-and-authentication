"""Display webcam frames with index and elapsed capture time; saves nothing."""

from __future__ import annotations

import logging
import sys
from collections import deque

import cv2

from app.video_processor import VideoProcessor, VideoProcessorError

LOGGER = logging.getLogger(__name__)
WINDOW_NAME = "Face preprocessing webcam demo"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    processor = VideoProcessor()

    def display_frame(frame) -> bool:
        display = cv2.cvtColor(frame.frame_rgb, cv2.COLOR_RGB2BGR)
        cv2.putText(
            display,
            f"Frame {frame.frame_index}  {frame.timestamp_ms:.1f} ms",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(WINDOW_NAME, display)
        return cv2.waitKey(1) & 0xFF != ord("q")

    try:
        deque(processor.process_webcam(callback=display_frame), maxlen=0)
    except VideoProcessorError as exc:
        LOGGER.error("Webcam demo could not start: %s", exc)
        return 1
    except KeyboardInterrupt:
        LOGGER.info("Webcam demo interrupted.")
    finally:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
