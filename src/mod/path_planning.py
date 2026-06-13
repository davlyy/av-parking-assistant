from ifaces.data_iface import *
from ifaces.algorithms_iface import PathPlanning
import numpy as np
import rsplan as rp

def plan_path(payload: dict, ego_world_pose: dict | None = None) -> list[Node]:
    if ego_world_pose is not None:
        payload = convert_payload_to_carla_world(payload, ego_world_pose)

        print("After conversion:")
        print("start:", payload["start_pose"])
        print("goal:", payload["goal_pose"])
        for i, obs in enumerate(payload.get("obstacles", [])):
            print(f"obs {i}:", obs)

    gridmap = build_gridMap(payload)

    start = payload_pose_to_node(payload["start_pose"], gridmap)
    end = payload_pose_to_node(payload["goal_pose"], gridmap)

    print("Start inside:", is_inside_grid(gridmap, start))
    print("End inside:", is_inside_grid(gridmap, end))
    print("Start collision free:", is_collision_free(gridmap, start))
    print("End collision free:", is_collision_free(gridmap, end))

    sx, sy = world_to_grid(start.x, start.y, gridmap)
    gx, gy = world_to_grid(end.x, end.y, gridmap)

    print("Start grid:", sx, sy, "occ:", gridmap.occupancy[sy, sx], "dist:", gridmap.distance[sy, sx])
    print("Goal grid:", gx, gy, "occ:", gridmap.occupancy[gy, gx], "dist:", gridmap.distance[gy, gx])

    path = hybrid_A_star(gridmap, start, end)

    if path is None:
        print("Hybrid A* found no path")
        return []

    return flatten_path(path)

type_checking_variable: PathPlanning = plan_path

import heapq
import itertools

def hybrid_A_star(gridmap: GridMap, start: Node, end: Node) -> list | None:
    GOAL_TOLERANCE = 3.0
    MAX_ITERATIONS = 200000

    closed = set()
    open_nodes = {}

    counter = itertools.count()
    heap = []

    start.g_cost = 0.0
    start.h_cost = heuristic_cost(start, end)

    open_nodes[start.idx] = start
    heapq.heappush(heap, (start.f_cost, next(counter), start.idx))

    iterations = 0
    best_dist_seen = float("inf")

    while heap:
        iterations += 1

        if iterations > MAX_ITERATIONS:
            print("Stopped after iteration limit")
            print("iterations:", iterations)
            print("closed:", len(closed))
            print("open:", len(open_nodes))
            print("best_dist_seen:", best_dist_seen)
            return None

        _, _, current_idx = heapq.heappop(heap)

        # Wurde dieser Eintrag inzwischen durch einen besseren ersetzt?
        if current_idx not in open_nodes:
            continue

        current = open_nodes.pop(current_idx)

        if current_idx in closed:
            continue

        closed.add(current_idx)

        dist_to_goal = heuristic_cost(current, end)
        best_dist_seen = min(best_dist_seen, dist_to_goal)

        if iterations % 10000 == 0:
            print(
                "iter:", iterations,
                "open:", len(open_nodes),
                "closed:", len(closed),
                "best_dist:", round(best_dist_seen, 2),
                "current_dist:", round(dist_to_goal, 2),
            )

        if dist_to_goal < GOAL_TOLERANCE:
            print("iterations:", iterations)
            print("closed:", len(closed))
            print("best_dist_seen:", best_dist_seen)
            return reconstruct_path(current)

        for node in node_expansion(current, gridmap, end):
            if node.idx in closed:
                continue

            if not is_collision_free(gridmap, node):
                continue

            if not is_inside_search_corridor(node, start, end, margin=20.0):
                continue

            old = open_nodes.get(node.idx)

            if old is None or node.g_cost < old.g_cost:
                open_nodes[node.idx] = node
                heapq.heappush(heap, (node.f_cost, next(counter), node.idx))

    print("iterations:", iterations)
    print("closed:", len(closed))
    print("open:", len(open_nodes))
    print("best_dist_seen:", best_dist_seen)

    return None

def reconstruct_path(node):
    path = []

    while node is not None:
        path.append(node)
        node = node.parent

    path.reverse()
    return path

def is_inside_search_corridor(node: Node, start: Node, end: Node, margin: float = 8.0) -> bool:
    min_x = min(start.x, end.x) - margin
    max_x = max(start.x, end.x) + margin
    min_y = min(start.y, end.y) - margin
    max_y = max(start.y, end.y) + margin

    return min_x <= node.x <= max_x and min_y <= node.y <= max_y

def heuristic_cost(node: Node, end: Node) -> float:
    return np.hypot(end.x - node.x, end.y - node.y)

def world_to_grid(x: float, y: float, gridmap: GridMap) -> tuple[int, int]:
    x_idx = int(round((x - gridmap.origin_x) / gridmap.resolution))
    y_idx = int(round((y - gridmap.origin_y) / gridmap.resolution))
    return x_idx, y_idx

def payload_pose_to_node(pose: dict, gridmap: GridMap) -> Node:
    x = float(pose["x"])
    y = float(pose["y"])
    theta = float(pose["yaw"])

    return Node(
        x=x,
        y=y,
        theta=theta,
        idx=make_idx(x, y, theta, gridmap),
    )
"""
def node_to_pose(node: Node) -> Pose:
    return Pose(
        coordinates=(node.x, node.y),
        orientation=YawAngle(float(np.rad2deg(node.theta) % 360.0))
    )
"""
def make_idx(x: float, y: float, theta: float, gridmap: GridMap) -> tuple[int, int, int]:
    x_idx, y_idx = world_to_grid(x, y, gridmap)

    theta_norm = normalize_angle(theta)
    theta_idx = int(np.floor((theta_norm + np.pi) / THETA_RESOLUTION))

    return x_idx, y_idx, theta_idx

def normalize_angle(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi

def build_gridMap(payload: dict) -> GridMap:
    margin = 5.0
    obstacle_inflation = 0.3  # extra safety margin around obstacles in meters

    xs = [
        float(payload["start_pose"]["x"]),
        float(payload["goal_pose"]["x"]),
    ]
    ys = [
        float(payload["start_pose"]["y"]),
        float(payload["goal_pose"]["y"]),
    ]

    for obs in payload.get("obstacles", []):
        r = 0.5 * np.hypot(float(obs["length"]), float(obs["width"]))
        xs.extend([float(obs["x"]) - r, float(obs["x"]) + r])
        ys.extend([float(obs["y"]) - r, float(obs["y"]) + r])

    if "parking_slot" in payload:
        slot = payload["parking_slot"]
        r = 0.5 * np.hypot(float(slot["length"]), float(slot["width"]))
        xs.extend([float(slot["x"]) - r, float(slot["x"]) + r])
        ys.extend([float(slot["y"]) - r, float(slot["y"]) + r])

    origin_x = min(xs) - margin
    origin_y = min(ys) - margin

    max_x = max(xs) + margin
    max_y = max(ys) + margin

    width_m = max_x - origin_x
    height_m = max_y - origin_y

    width_cells = int(np.ceil(width_m / GRID_RESOLUTION)) + 1
    height_cells = int(np.ceil(height_m / GRID_RESOLUTION)) + 1

    occupancy = np.zeros((height_cells, width_cells), dtype=np.uint8)

    # Helper: world coordinates -> grid indices
    def _world_to_grid_local(x: float, y: float) -> tuple[int, int]:
        x_idx = int(round((x - origin_x) / GRID_RESOLUTION))
        y_idx = int(round((y - origin_y) / GRID_RESOLUTION))
        return x_idx, y_idx

    # Rasterize every obstacle rectangle into occupancy
    for obs in payload.get("obstacles", []):
        cx = float(obs["x"])
        cy = float(obs["y"])
        yaw = float(obs.get("yaw", 0.0))

        half_l = float(obs["length"]) / 2.0 + obstacle_inflation
        half_w = float(obs["width"]) / 2.0 + obstacle_inflation

        radius = np.hypot(half_l, half_w)

        x_min, y_min = _world_to_grid_local(cx - radius, cy - radius)
        x_max, y_max = _world_to_grid_local(cx + radius, cy + radius)

        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(width_cells - 1, x_max)
        y_max = min(height_cells - 1, y_max)

        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)

        for y_idx in range(y_min, y_max + 1):
            for x_idx in range(x_min, x_max + 1):
                wx = origin_x + x_idx * GRID_RESOLUTION
                wy = origin_y + y_idx * GRID_RESOLUTION

                dx = wx - cx
                dy = wy - cy

                # Transform world point into obstacle-local frame
                local_x = cos_yaw * dx + sin_yaw * dy
                local_y = -sin_yaw * dx + cos_yaw * dy

                if abs(local_x) <= half_l and abs(local_y) <= half_w:
                    occupancy[y_idx, x_idx] = 1

    # Distance map: distance to nearest occupied cell in meters
    try:
        from scipy.ndimage import distance_transform_edt

        distance = distance_transform_edt(1 - occupancy) * GRID_RESOLUTION
    except ImportError:
        # Fallback: no distance penalty, only hard occupancy collision
        distance = np.full_like(occupancy, fill_value=999.0, dtype=float)

    return GridMap(
        occupancy=occupancy,
        distance=distance,
        resolution=GRID_RESOLUTION,
        origin_x=origin_x,
        origin_y=origin_y,
    )

def is_collision_free(gridmap: GridMap, node) -> bool:
    if not is_inside_grid(gridmap, node):
        return False

    x_idx, y_idx = world_to_grid(node.x, node.y, gridmap)
    return gridmap.occupancy[y_idx, x_idx] == 0

def is_collision_free_rs_curve(gridmap: GridMap, rs_curve) -> bool:
    height, width = gridmap.occupancy.shape

    for wp in rs_curve.waypoints():
        x_idx, y_idx = world_to_grid(wp.x, wp.y, gridmap)

        if not (0 <= x_idx < width and 0 <= y_idx < height):
            return False

        if gridmap.occupancy[y_idx, x_idx] != 0:
            return False

    return True

def is_inside_grid(gridmap: GridMap, node) -> bool:
    height, width = gridmap.occupancy.shape
    x_idx, y_idx = world_to_grid(node.x, node.y, gridmap)

    return 0 <= x_idx < width and 0 <= y_idx < height

def node_expansion(current: Node, gridmap: GridMap, end:Node) -> list[Node]:
    children = []

    for direction in DIRECTIONS:
        for steer in STEERING_ANGLES:
            x_new = current.x + direction * D_SIZE * np.cos(current.theta)
            y_new = current.y + direction * D_SIZE * np.sin(current.theta)

            theta_new = current.theta + (
                direction * D_SIZE / WHEELBASE * np.tan(steer)
            )
            theta_new = normalize_angle(theta_new)

            idx_new = make_idx(x_new, y_new, theta_new, gridmap)

            child = Node(
                x=x_new,
                y=y_new,
                theta=theta_new,
                idx=idx_new,
                parent=current,
                steer=steer,
                direction=direction
            )

            child.g_cost = current.g_cost + transition_cost(current, child, gridmap)
            child.h_cost = heuristic_cost(child, end)

            children.append(child)

    return children

def transition_cost(current: Node, node: Node, gridmap: GridMap) -> float:
    cost1 = OMEGA_1 * D_SIZE

    current_steer = getattr(current, "steer", 0.0)
    node_steer = getattr(node, "steer", 0.0)
    cost2 = OMEGA_2 * abs(node_steer - current_steer)

    cost3 = OMEGA_3 * obstacle_distance_cost(gridmap, node)

    return cost1 + cost2 + cost3

def obstacle_distance_cost(gridmap: GridMap, node: Node) -> float:
    if not is_inside_grid(gridmap, node):
        return float("inf")

    x_idx, y_idx = world_to_grid(node.x, node.y, gridmap)
    d = gridmap.distance[y_idx, x_idx]

    if d >= D0:
        return 0.0

    return EPSILON / (EPSILON + d)

def flatten_path(path) -> list[Node]:
    nodes = []

    for item in path:
        if isinstance(item, Node):
            nodes.append(item)

        elif hasattr(item, "waypoints"):
            for wp in item.waypoints():
                x = float(wp.x)
                y = float(wp.y)
                theta = float(wp.yaw)

                nodes.append(
                    Node(
                        x=x,
                        y=y,
                        theta=theta,
                        idx=(-1, -1, -1),
                    )
                )
        else:
            raise TypeError(f"Unsupported path item type: {type(item)}")

    return nodes

def convert_payload_to_carla_world(payload: dict, ego_world_pose: dict) -> dict:
    payload = payload.copy()

    local_start = payload["start_pose"]

    local_start_x = float(local_start["x"])
    local_start_y = float(local_start["y"])
    local_start_yaw = float(local_start.get("yaw", 0.0))

    world_start_x = float(ego_world_pose["x"])
    world_start_y = float(ego_world_pose["y"])
    world_start_yaw = float(ego_world_pose.get("yaw", 0.0))

    def transform_point(local_x: float, local_y: float) -> tuple[float, float]:
        local_dx = local_x - local_start_x
        local_dy = local_y - local_start_y

        SCALE_X = 1.0
        SCALE_Y = 1.0

        OFFSET_X = 9.8
        OFFSET_Y = 0.0

        dx = -SCALE_X * local_dy + OFFSET_X
        dy = SCALE_Y * local_dx + OFFSET_Y

        c = np.cos(world_start_yaw)
        s = np.sin(world_start_yaw)

        world_x = world_start_x + c * dx - s * dy
        world_y = world_start_y + s * dx + c * dy

        return float(round(world_x, 3)), float(round(world_y, 3))

    def transform_yaw(local_yaw: float) -> float:
        return round(world_start_yaw + (local_yaw - local_start_yaw), 4)

    # start wird exakt auf echte CARLA-Ego-Pose gesetzt
    payload["start_pose"] = {
        "x": round(world_start_x, 3),
        "y": round(world_start_y, 3),
        "yaw": round(world_start_yaw, 4),
    }

    # goal
    gx, gy = transform_point(
        float(payload["goal_pose"]["x"]),
        float(payload["goal_pose"]["y"]),
    )
    payload["goal_pose"] = {
        "x": gx,
        "y": gy,
        "yaw": transform_yaw(float(payload["goal_pose"].get("yaw", 0.0))),
    }

    # obstacles
    new_obstacles = []
    for obs in payload.get("obstacles", []):
        ox, oy = transform_point(float(obs["x"]), float(obs["y"]))
        new_obstacles.append({
            **obs,
            "x": ox,
            "y": oy,
            "yaw": transform_yaw(float(obs.get("yaw", 0.0))),
        })

    payload["obstacles"] = new_obstacles

    # parking slot
    if "parking_slot" in payload:
        slot = payload["parking_slot"]
        sx, sy = transform_point(float(slot["x"]), float(slot["y"]))
        payload["parking_slot"] = {
            **slot,
            "x": sx,
            "y": sy,
            "yaw": transform_yaw(float(slot.get("yaw", 0.0))),
        }

    return payload

def draw_obstacles_in_carla(source, payload: dict, z: float = 1.2, life_time: float = 0.1) -> None:
    import carla
    import math

    world = source._client.get_world()
    debug = world.debug

    for i, obs in enumerate(payload.get("obstacles", [])):
        cx = float(obs["x"])
        cy = float(obs["y"])
        yaw = float(obs.get("yaw", 0.0))

        half_l = float(obs["length"]) / 2.0
        half_w = float(obs["width"]) / 2.0

        c = math.cos(yaw)
        s = math.sin(yaw)

        corners = []
        for lx, ly in [
            (+half_l, +half_w),
            (+half_l, -half_w),
            (-half_l, -half_w),
            (-half_l, +half_w),
        ]:
            x = cx + c * lx - s * ly
            y = cy + s * lx + c * ly
            corners.append(carla.Location(x=x, y=y, z=z))

        for a, b in zip(corners, corners[1:] + corners[:1]):
            debug.draw_line(
                a,
                b,
                thickness=0.08,
                color=carla.Color(255, 255, 0),
                life_time=life_time,
            )

        center = carla.Location(x=cx, y=cy, z=z)

        debug.draw_point(
            center,
            size=0.2,
            color=carla.Color(255, 255, 0),
            life_time=life_time,
        )

        debug.draw_string(
            center,
            f"obs {i}",
            draw_shadow=False,
            color=carla.Color(255, 255, 0),
            life_time=life_time,
        )