from __future__ import annotations

import cv2

from input.config import ImageConfig
from input.source_iface import Frame


class ImageSource:
    def __init__(self, config: ImageConfig) -> None:
        self._config = config
        self._frame = None
        self._is_open = False

    def open(self) -> None:
        self._frame = cv2.imread(self._config.path, cv2.IMREAD_COLOR)

        if self._frame is None:
            raise FileNotFoundError(f"Could not load image: {self._config.path}")

        self._is_open = True

    def read(self) -> Frame | None:
        if not self._is_open or self._frame is None:
            return None

        return self._frame.copy()

    def release(self) -> None:
        self._is_open = False
        self._frame = None

    def __iter__(self):
        return self

    def __next__(self) -> Frame:
        frame = self.read()

        if frame is None:
            raise StopIteration

        return frame