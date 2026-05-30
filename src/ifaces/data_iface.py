from dataclasses import dataclass

Box = tuple[int, int, int, int]
Coordinate2D = tuple[int, int]

@dataclass(frozen=True)
class YawAngle:
    """
    Angle of orientation of vehicle, in degrees(float) from 0...360

    Class is immutable. Please use randomPose.orientation = YawAngle(180.0) when updating the value.
    """
    def __init__(self, value: float):
        if not 0.0 <= value < 360.0:
            raise ValueError("Orientation value must be between 0 and 360.")

@dataclass()
class Pose:
    """
    @coordinates: current coordinates of vehicle
    @yaw: angle of orientation of vehicle, in degrees(float) from 0...360
    """
    coordinates: Coordinate2D
    orientation: YawAngle