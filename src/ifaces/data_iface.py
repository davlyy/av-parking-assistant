from dataclasses import dataclass, field
import math
import numpy as np

Box = tuple[int, int, int, int]
Coordinate2D = tuple[int, int]

@dataclass()
class Config:
    dsize: float = 1.0
    omega_1 = 0.5
    omega_2 = 0.5
    omega_3 = 5.0
    d0 = 1.5
    epsilon = 0.1
    heuristic_weight = 1.5
    search_margin: float = 20.0

    goal_tolerance: float = 1.2
    planner_goal_tolerance: float = 0.45
    goal_yaw_tolerance: float = math.radians(20)

    near_goal_radius: float = 3.0
    near_goal_dsize: float = 0.4

    max_iterations: int = 200000
    omega_steer_angle: float = 2.0
    omega_reverse: float = 0.0
    omega_direction_change: float = 1.5

@dataclass
class CarlaConfig:
    coordinate_transform_type: str = "carla_ego"
    transform_scale_x: float = 1.0
    transform_scale_y: float = 1.0
    transform_offset_x: float = 9.8
    transform_offset_y: float = 0.0
    transform_swap_xy: bool = True
    transform_invert_x: bool = True
    transform_invert_y: bool = False

@dataclass
class Vehicle:
    wheelbase: float= 2.85
    max_steering: float= 0.44157
    length: float= 4.98
    width: float= 1.9

    @property
    def steering_angles(self) -> list[float]:
        return [
            -self.max_steering,
            -2 / 3 * self.max_steering,
            -1 / 3 * self.max_steering,
            0.0,
            1 / 3 * self.max_steering,
            2 / 3 * self.max_steering,
            self.max_steering,
        ]
    @property
    def directions(self) -> list[int]:
        return [-1,
        1]


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
    motion_length: float = 1.0
    segment_points: list = field(default_factory=list)

    @property
    def f_cost(self):
        return self.g_cost + self.h_cost

@dataclass()
class GridMap3D:
    occupancy: np.ndarray | None = None
    distance: np.ndarray | None = None
    resolution: float= 0.25
    theta_resolution: float = math.radians(10)  # vehicle can be positioned in x degree slices per position on the grid map
    origin_x: float = 0.0
    origin_y: float = 0.0