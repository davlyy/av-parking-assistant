from ifaces.data_iface import *
from ifaces.algorithms_iface import PathPlanning
import numpy as np
import heapq
import itertools

vehicle: Vehicle = Vehicle()
gridmap: GridMap3D = GridMap3D()
config: Config = Config()
carla_config: CarlaConfig = CarlaConfig()

def plan_path(
    payload: dict,
    ego_world_pose: dict | None = None,
    already_world: bool = False,
) -> list[Node]:
    if not already_world:
        transform_cfg = get_coordinate_transform(payload)
        transform_type = transform_cfg["type"]

        if transform_type == "carla_ego":
            if ego_world_pose is None:
                raise ValueError(
                    "ego_world_pose is required for coordinate_transform type='carla_ego'"
                )

            payload = convert_payload_to_carla_world(
                payload,
                ego_world_pose,
            )

        elif transform_type == "world":
            payload = payload.copy()

        else:
            raise ValueError(
                f"Unsupported coordinate_transform type: {transform_type}"
            )

    vehicle_init(payload)
    gridmap_init(payload)

    start = payload_pose_to_node(payload["start_pose"], gridmap)
    end = payload_pose_to_node(payload["goal_pose"], gridmap)

    path = hybrid_A_star(gridmap, start, end)

    if path is None:
        print("Hybrid A* found no path")
        return []

    return flatten_path(path)

type_checking_variable: PathPlanning = plan_path

def hybrid_A_star(gridmap: GridMap3D, start: Node, end: Node) -> list | None:

    closed = set()
    open_nodes = {}

    counter = itertools.count()
    heap = []

    start.g_cost = 0.0
    start.h_cost = heuristic_cost(start, end)

    open_nodes[start.idx] = start
    heapq.heappush(heap, (priority(start), next(counter), start.idx))

    iterations = 0
    best_dist_seen = float("inf")

    while heap:
        iterations += 1

        if iterations > config.max_iterations:
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

        if dist_to_goal < config.goal_tolerance:
            print("iterations:", iterations)
            print("closed:", len(closed))
            print("best_dist_seen:", best_dist_seen)
            return reconstruct_path(current)

        for node in node_expansion(current, gridmap, end):
            if node.idx in closed:
                continue

            if not is_collision_free(gridmap, node):
                continue

            if not is_inside_search_corridor(node, start, end, margin=config.search_margin):
                continue
            node.g_cost = current.g_cost + transition_cost(current, node, gridmap)
            node.h_cost = heuristic_cost(node, end)

            old = open_nodes.get(node.idx)

            if old is None or node.g_cost < old.g_cost:
                open_nodes[node.idx] = node
                heapq.heappush(heap, (priority(node), next(counter), node.idx))

    print("iterations:", iterations)
    print("closed:", len(closed))
    print("open:", len(open_nodes))
    print("best_dist_seen:", best_dist_seen)

    return None

def distance_to_goal(pose: dict, goal_pose: dict) -> float:
    dx = float(pose["x"]) - float(goal_pose["x"])
    dy = float(pose["y"]) - float(goal_pose["y"])

    return (dx ** 2 + dy ** 2) ** 0.5


def is_within_goal_tolerance(
    pose: dict | None,
    goal_pose: dict | None,
    goal_tolerance: float,
) -> bool:
    if pose is None or goal_pose is None:
        return False

    return distance_to_goal(pose, goal_pose) <= goal_tolerance

def carla_transform_from_config() -> dict:
    cfg = carla_config.__dict__.copy()

    transform_cfg = {
        key.removeprefix("transform_"): value
        for key, value in cfg.items()
        if key.startswith("transform_")
    }

    transform_cfg["type"] = cfg["coordinate_transform_type"]

    return transform_cfg


def get_coordinate_transform(payload: dict) -> dict:
    payload_cfg = payload.get("coordinate_transform")

    # Kein Eintrag im Payload -> CARLA-Fallback
    if payload_cfg is None:
        return carla_transform_from_config()

    if not isinstance(payload_cfg, dict):
        raise TypeError(
            f"'coordinate_transform' must be a dict, got {type(payload_cfg)}"
        )

    if "type" not in payload_cfg:
        raise ValueError(
            "'coordinate_transform' in payload must contain a 'type' field"
        )

    # Wichtig: nicht mit carla_config mergen.
    # Payload ist entweder vollständig world oder vollständig carla_ego.
    return payload_cfg


def normalize_parking_slots(payload: dict) -> list[dict]:
    raw_slots = payload.get("parking_slots", None)

    if raw_slots is None:
        raw_slots = payload.get("parking_slot", None)

    if raw_slots is None:
        return []

    if isinstance(raw_slots, dict):
        raw_slots = [raw_slots]

    if not isinstance(raw_slots, list):
        raise TypeError(
            "'parking_slot' or 'parking_slots' must be either a dict or a list of dicts"
        )

    slots = []

    for i, slot in enumerate(raw_slots):
        if not isinstance(slot, dict):
            raise TypeError(
                f"Parking slot at index {i} must be a dict, got {type(slot)}"
            )

        if not slot.get("free", True):
            continue

        slots.append({
            **slot,
            "id": slot.get("id", i),
            "free": slot.get("free", True),
        })

    return slots


def local_to_carla_relative(
    local_dx: float,
    local_dy: float,
    transform_cfg: dict,
) -> tuple[float, float]:
    transform_type = transform_cfg.get("type")

    if transform_type != "carla_ego":
        raise ValueError(
            f"local_to_carla_relative requires type='carla_ego', got {transform_type}"
        )

    required_keys = [
        "scale_x",
        "scale_y",
        "offset_x",
        "offset_y",
        "swap_xy",
        "invert_x",
        "invert_y",
    ]

    missing = [key for key in required_keys if key not in transform_cfg]

    if missing:
        raise ValueError(
            f"Missing keys for carla_ego transform: {missing}"
        )

    scale_x = float(transform_cfg["scale_x"])
    scale_y = float(transform_cfg["scale_y"])

    offset_x = float(transform_cfg["offset_x"])
    offset_y = float(transform_cfg["offset_y"])

    swap_xy = bool(transform_cfg["swap_xy"])
    invert_x = bool(transform_cfg["invert_x"])
    invert_y = bool(transform_cfg["invert_y"])

    if swap_xy:
        dx = local_dy
        dy = local_dx
    else:
        dx = local_dx
        dy = local_dy

    if invert_x:
        dx = -dx

    if invert_y:
        dy = -dy

    dx = scale_x * dx + offset_x
    dy = scale_y * dy + offset_y

    return dx, dy

def vehicle_init(payload: dict):
    global vehicle
    vehicle_data = payload.get("vehicle", {})
    vehicle.wheelbase = vehicle_data["wheelbase"]
    vehicle.max_steering = vehicle_data["max_steer"]
    vehicle.width = vehicle_data["width"]
    vehicle.length = vehicle_data["length"]

def priority(node: Node) -> float:
    return node.g_cost + config.heuristic_weight * node.h_cost

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

def world_to_grid(x: float, y: float, gridmap: GridMap3D) -> tuple[int, int]:
    x_idx = int(round((x - gridmap.origin_x) / gridmap.resolution))
    y_idx = int(round((y - gridmap.origin_y) / gridmap.resolution))
    return x_idx, y_idx

def payload_pose_to_node(pose: dict, gridmap: GridMap3D) -> Node:
    x = float(pose["x"])
    y = float(pose["y"])
    theta = float(pose["yaw"])

    return Node(
        x=x,
        y=y,
        theta=theta,
        idx=make_idx(x, y, theta, gridmap),
    )

def make_idx(x: float, y: float, theta: float, gridmap: GridMap3D) -> tuple[int, int, int]:
    x_idx, y_idx = world_to_grid(x, y, gridmap)

    theta_norm = normalize_angle(theta)
    theta_idx = int(np.floor((theta_norm + np.pi) / gridmap.theta_resolution))

    return x_idx, y_idx, theta_idx

def normalize_angle(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi

def gridmap_init(payload: dict):
    global gridmap
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

    for slot in normalize_parking_slots(payload):
        r = 0.5 * np.hypot(float(slot["length"]), float(slot["width"]))
        xs.extend([float(slot["x"]) - r, float(slot["x"]) + r])
        ys.extend([float(slot["y"]) - r, float(slot["y"]) + r])

    gridmap.origin_x = min(xs) - margin
    gridmap.origin_y = min(ys) - margin

    max_x = max(xs) + margin
    max_y = max(ys) + margin

    width_m = max_x - gridmap.origin_x
    height_m = max_y - gridmap.origin_y

    width_cells = int(np.ceil(width_m / gridmap.resolution)) + 1
    height_cells = int(np.ceil(height_m / gridmap.resolution)) + 1

    gridmap.occupancy = np.zeros((height_cells, width_cells), dtype=np.uint8)

    # Helper: world coordinates -> grid indices
    def _world_to_grid_local(x: float, y: float) -> tuple[int, int]:
        x_idx = int(round((x - gridmap.origin_x) / gridmap.resolution))
        y_idx = int(round((y - gridmap.origin_y) / gridmap.resolution))
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
                wx = gridmap.origin_x + x_idx * gridmap.resolution
                wy = gridmap.origin_y + y_idx * gridmap.resolution

                dx = wx - cx
                dy = wy - cy

                # Transform world point into obstacle-local frame
                local_x = cos_yaw * dx + sin_yaw * dy
                local_y = -sin_yaw * dx + cos_yaw * dy

                if abs(local_x) <= half_l and abs(local_y) <= half_w:
                    gridmap.occupancy[y_idx, x_idx] = 1

    # Distance map: distance to nearest occupied cell in meters
    try:
        from scipy.ndimage import distance_transform_edt

        gridmap.distance = distance_transform_edt(1 - gridmap.occupancy) * gridmap.resolution
    except ImportError:
        # Fallback: no distance penalty, only hard occupancy collision
        gridmap.distance = np.full_like(gridmap.occupancy, fill_value=999.0, dtype=float)


def is_collision_free(gridmap: GridMap3D, node) -> bool:
    if not is_inside_grid(gridmap, node):
        return False

    x_idx, y_idx = world_to_grid(node.x, node.y, gridmap)
    return gridmap.occupancy[y_idx, x_idx] == 0

def is_collision_free_rs_curve(gridmap: GridMap3D, rs_curve) -> bool:
    height, width = gridmap.occupancy.shape

    for wp in rs_curve.waypoints():
        x_idx, y_idx = world_to_grid(wp.x, wp.y, gridmap)

        if not (0 <= x_idx < width and 0 <= y_idx < height):
            return False

        if gridmap.occupancy[y_idx, x_idx] != 0:
            return False

    return True

def is_inside_grid(gridmap: GridMap3D, node) -> bool:
    height, width = gridmap.occupancy.shape
    x_idx, y_idx = world_to_grid(node.x, node.y, gridmap)

    return 0 <= x_idx < width and 0 <= y_idx < height

def node_expansion(current: Node, gridmap: GridMap3D, end:Node) -> list[Node]:
    global vehicle
    children = []

    for direction in vehicle.directions:
        for steer in vehicle.steering_angles:
            x_new = current.x + direction * config.dsize * np.cos(current.theta)
            y_new = current.y + direction * config.dsize * np.sin(current.theta)

            theta_new = current.theta + (
                direction * config.dsize / vehicle.wheelbase * np.tan(steer)
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

            children.append(child)

    return children

def transition_cost(current: Node, node: Node, gridmap: GridMap3D) -> float:
    current_steer = float(getattr(current, "steer", 0.0))
    node_steer = float(getattr(node, "steer", 0.0))

    cost_length = config.omega_1 * config.dsize
    cost_steer_change = config.omega_2 * abs(node_steer - current_steer)
    cost_obstacle = config.omega_3 * obstacle_distance_cost(gridmap, node)

    max_steer = max(float(vehicle.max_steering), 1e-6)
    normalized_steer = abs(node_steer) / max_steer

    cost_steer_angle = 0.15 * normalized_steer * config.dsize

    return (
        cost_length
        + cost_steer_change
        + cost_steer_angle
        + cost_obstacle
    )

def obstacle_distance_cost(gridmap: GridMap3D, node: Node) -> float:
    if not is_inside_grid(gridmap, node):
        return float("inf")

    x_idx, y_idx = world_to_grid(node.x, node.y, gridmap)
    d = gridmap.distance[y_idx, x_idx]

    if d >= config.d0:
        return 0.0

    return config.epsilon / (config.epsilon + d)

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

    transform_cfg = get_coordinate_transform(payload)
    transform_type = transform_cfg["type"]

    if transform_type != "carla_ego":
        raise ValueError(
            f"convert_payload_to_carla_world requires type='carla_ego', got {transform_type}"
        )

    if ego_world_pose is None:
        raise ValueError(
            "ego_world_pose is required for convert_payload_to_carla_world"
        )

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

        dx, dy = local_to_carla_relative(
            local_dx,
            local_dy,
            transform_cfg,
        )

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
    if "goal_pose" in payload:
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
        ox, oy = transform_point(
            float(obs["x"]),
            float(obs["y"]),
        )

        new_obstacles.append({
            **obs,
            "x": ox,
            "y": oy,
            "yaw": transform_yaw(float(obs.get("yaw", 0.0))),
        })

    payload["obstacles"] = new_obstacles

    # parking slot(s)
    new_slots = []

    for slot in normalize_parking_slots(payload):
        sx, sy = transform_point(
            float(slot["x"]),
            float(slot["y"]),
        )

        new_slots.append({
            **slot,
            "x": sx,
            "y": sy,
            "yaw": transform_yaw(float(slot.get("yaw", 0.0))),
        })

    payload["parking_slots"] = new_slots
    payload.pop("parking_slot", None)

    # Wichtig: Ab jetzt ist das Payload bereits in Welt-/CARLA-Koordinaten.
    # Dadurch vermeidest du versehentliche Doppeltransformationen.
    payload["coordinate_transform"] = {
        "type": "world"
    }

    return payload