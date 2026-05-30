from typing import Protocol

class Preprocessing(Protocol):
    """
    Prepares image for model prediction.

    @img: the image to be preprocessed in bytes
    returns image in bytes
    """
    def __call__(self, img: bytes) -> bytes:
        ...

class Display(Protocol):
    """
    Displays the guidance hints

    @img: the finalized image with parking guidance to be displayed in bytes TODO: define hints
    returns nothing
    """
    def __call__(self, img: bytes) -> None:
        ...