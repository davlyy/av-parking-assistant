from dataclasses import dataclass

import numpy as np

fac = 10
Box = tuple[int, int, int, int]
Coordinate2D = tuple[int, int]
WHEELBASE = 3.309 #TODO: replace with real value
MAX_STEERING_ANGLE = 0.44157 #TODO: replace with real value
GRID_RESOLUTION = 0.25
THETA_RESOLUTION = 0.05
D_SIZE = 0.6

OMEGA_1 = 10.0/fac  # distance cost
OMEGA_2 = 5.0/fac  # steer change cost
OMEGA_3 = 15.0/fac  # obstacle distance cost

D0 = 1.5
EPSILON = 0.1
STEERING_ANGLES = [-MAX_STEERING_ANGLE, #- left, + right
                   -2/3* MAX_STEERING_ANGLE,
                   -1/3* MAX_STEERING_ANGLE,
                   0.0,
                   1/3* MAX_STEERING_ANGLE,
                   2/3* MAX_STEERING_ANGLE,
                   MAX_STEERING_ANGLE]
DIRECTIONS = [-1, #reverse
              1] #forward

#@dataclass(frozen=True)
#class YawAngle:
#    value: float
"""
    Angle of orientation of vehicle, in degrees(float) from 0...360

    Class is immutable. Please use randomPose.orientation = YawAngle(180.0) when updating the value.
    """
#    def __init__(self, value: float):
#        if not 0.0 <= value < 360.0:
#            raise ValueError("Orientation value must be between 0 and 360.")

#@dataclass()
#class Pose:
"""
    @coordinates: current coordinates of vehicle
    @yaw: angle of orientation of vehicle, in degrees(float) from 0...360
    """
#    coordinates: Coordinate2D
#    orientation: YawAngle

@dataclass
class Node:
    x: float
    y: float
    theta: float
    idx: tuple[int, int, int]
    parent: object = None
    g_cost: float = 0.0
    h_cost: float = 0.0
    steer: float = 0.0
    direction: int = 1

    @property
    def f_cost(self):
        return self.g_cost + self.h_cost

@dataclass()
class GridMap:
    occupancy: np.ndarray
    distance: np.ndarray
    resolution: float
    origin_x: float = 0.0
    origin_y: float = 0.0