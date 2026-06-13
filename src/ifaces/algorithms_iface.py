from ifaces.data_iface import *
from typing import Protocol

#class CalculateFinalPose(Protocol):
"""
    Calculates the exact target pose inside the target parking space.

    @current_pose: coordinates and yaw angle of vehicle
    @boundaries: bounding box of the owned vehicle
    returns final pose of vehicle
    """
#    def __call__(self, current_pose: Pose, boundaries: Box) -> Pose:
#        ...

class PathPlanning(Protocol):
    """
    Plans the shortest path using a suitable algorithm around obstacles to the target pose.

    @current_pose: coordinates and yaw angle of vehicle
    @final_pose: final pose of vehicle
    returns rough path as list of poses
    """
    def __call__(self, payload: dict) -> list[Node]:
        ...

#class PathToRoute(Protocol):
"""
    Smooths out the returned rough pathing of PathPlanning.

    @path: List of poses
    returns smooth drivable route as list of poses
    """
#    def __call__(self, path: list[Pose]) -> list[Pose]:
#        ...