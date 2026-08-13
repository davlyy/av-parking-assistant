from __future__ import annotations

import cv2

from input.config import VideoConfig
from input.source_iface import Frame


class VideoSource:
    def __init__(self, config: VideoConfig) -> None:
        self._config = config
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self._config.path)

        if not self._cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {self._config.path}")

    def read(self) -> Frame | None:
        if self._cap is None:
            return None

        ret, frame = self._cap.read()

        if ret:
            return frame

        if self._config.loop:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self._cap.read()
            return frame if ret else None

        return None

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __iter__(self):
        return self

    def __next__(self) -> Frame:
        frame = self.read()

        if frame is None:
            raise StopIteration

        return frame