from __future__ import annotations
from input.config import CameraSetup
from input.source_iface import Frame
import cv2
import numpy as np


class CameraSource:
    def __init__(self, config: CameraSetup) -> None:
        self._config = config
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self._config.device_id)
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open camera device {self._config.device_id}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.height)
        self._cap.set(cv2.CAP_PROP_FPS, self._config.fps)

    def read(self) -> Frame | None:
        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        return frame if ret else None

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
