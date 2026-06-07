from __future__ import annotations
from typing import Protocol, Iterator
import numpy as np

Frame = np.ndarray


class InputSource(Protocol):
    def open(self) -> None:
        ...

    def read(self) -> Frame | None:
        ...

    def release(self) -> None:
        ...

    def __iter__(self) -> Iterator[Frame]:
        ...

    def __next__(self) -> Frame:
        ...
