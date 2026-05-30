from ifaces.data_iface import *
from typing import Protocol

class GetParkingSpace(Protocol):
    """
    Gets the predicted bounding boxes from the input image.

    @img: Input image as bytes
    returns bounding boxe of parking space
    """
    def __call__(self, img: bytes) -> list[Box]:
        ...

class GetObstacles(Protocol):
    """
    Gets the predicted obstacles from the input image.

    @img: Input image as bytes
    returns bounding boxes of obstacles
    """
    def __call__(self, img: bytes) -> list[Box]:
        ...

class GetCurrentPose(Protocol):
    """
    Gets the predicted current pose from the input image.

    @img: Input image as bytes
    returns current pose TODO: determine yaw angle
    """
    def __call__(self, img: bytes) -> Pose:
        ...