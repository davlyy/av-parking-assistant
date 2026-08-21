import cv2
import math
from input import SourceConfig
import numpy as np
from typing import Callable

Projector = Callable[[float, float, np.ndarray], tuple[int, int]]
ARUCO_WORLD = {
    0: {"x": 0.00, "y": 0.00},
    1: {"x": 1.80, "y": 0.00},
    2: {"x": 1.80, "y": 2.40},
    3: {"x": 0.00, "y": 2.40},
}
DEBUG_WORLD_X_MIN = 0.0
DEBUG_WORLD_X_MAX = 1.8
DEBUG_WORLD_Y_MIN = 0.0
DEBUG_WORLD_Y_MAX = 1.2
DEBUG_GRID_STEP = 0.1

def _w2i(payload, pts):
    H = np.asarray(payload["projection"]["H_world_to_image"], dtype=np.float32)
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
    img = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    return np.round(img).astype(int)

def _rot_box(x, y, length, width, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    hl, hw = length / 2.0, width / 2.0
    local = np.array([
        [-hl, -hw],
        [ hl, -hw],
        [ hl,  hw],
        [-hl,  hw],
    ], dtype=np.float32)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    world = local @ R.T
    world[:, 0] += x
    world[:, 1] += y
    return world

def _draw_text_block(frame, lines, x, y, color=(255, 255, 255), scale=0.45, thickness=1):
    dy = int(18 * scale / 0.45)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x, y + i * dy), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(frame, line, (x, y + i * dy), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

def draw_debug_measurements_on_frame(frame, payload, selected_slot_id=None):
    #for x in np.arange(DEBUG_WORLD_X_MIN, DEBUG_WORLD_X_MAX + 1e-6, DEBUG_GRID_STEP):
    #    p0, p1 = _w2i(payload, [(x, DEBUG_WORLD_Y_MIN), (x, DEBUG_WORLD_Y_MAX)])
    #    cv2.line(frame, tuple(p0), tuple(p1), (60, 60, 60), 1, cv2.LINE_AA)
    #for y in np.arange(DEBUG_WORLD_Y_MIN, DEBUG_WORLD_Y_MAX + 1e-6, DEBUG_GRID_STEP):
    #    p0, p1 = _w2i(payload, [(DEBUG_WORLD_X_MIN, y), (DEBUG_WORLD_X_MAX, y)])
    #    cv2.line(frame, tuple(p0), tuple(p1), (60, 60, 60), 1, cv2.LINE_AA)

    if payload.get("drivable_area") and payload["drivable_area"].get("type") == "image_polygon":
        poly = np.asarray(payload["drivable_area"]["points"], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [poly], True, (0, 255, 0), 2, cv2.LINE_AA)

    v = payload.get("vehicle", {})
    ego = payload.get("start_pose")
    if ego and v:
        x, y, yaw = float(ego["x"]), float(ego["y"]), float(ego["yaw"])
        length = float(v["length"])
        width = float(v["width"])
        wheelbase = float(v["wheelbase"])

        box = _rot_box(x, y, length, width, yaw)
        box_i = _w2i(payload, box).reshape(-1, 1, 2)
        cv2.polylines(frame, [box_i], True, (0, 140, 255), 2, cv2.LINE_AA)

        center = _w2i(payload, [(x, y)])[0]
        front = _w2i(payload, [(x + 0.20 * math.cos(yaw), y + 0.20 * math.sin(yaw))])[0]
        rear_axle = _w2i(payload, [(x - 0.5 * wheelbase * math.cos(yaw), y - 0.5 * wheelbase * math.sin(yaw))])[0]
        front_axle = _w2i(payload, [(x + 0.5 * wheelbase * math.cos(yaw), y + 0.5 * wheelbase * math.sin(yaw))])[0]

        cv2.circle(frame, tuple(center), 4, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.arrowedLine(frame, tuple(center), tuple(front), (0, 255, 255), 2, cv2.LINE_AA, tipLength=0.25)
        cv2.line(frame, tuple(rear_axle), tuple(front_axle), (255, 255, 0), 2, cv2.LINE_AA)

        tl = box_i.reshape(-1, 2).min(axis=0)
        _draw_text_block(
            frame,
            [
                f"EGO x={x:.2f} y={y:.2f}",
                f"yaw={math.degrees(yaw):.1f} deg",
                f"L={length:.3f} W={width:.3f} WB={wheelbase:.3f}",
            ],
            int(tl[0]),
            int(tl[1]) - 38,
            color=(0, 220, 255),
            scale=0.45,
        )

    slots = sorted(
        payload.get("parking_slots", []),
        key=lambda slot: int(slot["id"]),
    )

    for slot in slots:
        slot_id = int(slot["id"])

        x, y, yaw = float(slot["x"]), float(slot["y"]), float(slot["yaw"])
        length = float(slot["length"])
        width = float(slot["width"])

        box = _rot_box(x, y, length, width, yaw)
        box_i = _w2i(payload, box).reshape(-1, 1, 2)

        selected = slot_id == selected_slot_id
        color = (255, 0, 0) if selected else (0, 255, 255)
        thick = 2 if selected else 1

        cv2.polylines(frame, [box_i], True, color, thick, cv2.LINE_AA)

        center = _w2i(payload, [(x, y)])[0]
        px = _w2i(payload, [(x + 0.12 * math.cos(yaw), y + 0.12 * math.sin(yaw))])[0]
        py = _w2i(payload, [(x - 0.12 * math.sin(yaw), y + 0.12 * math.cos(yaw))])[0]

        cv2.circle(frame, tuple(center), 3, color, -1, cv2.LINE_AA)
        cv2.arrowedLine(frame, tuple(center), tuple(px), (0, 0, 255), 1, cv2.LINE_AA, tipLength=0.3)
        cv2.arrowedLine(frame, tuple(center), tuple(py), (0, 255, 0), 1, cv2.LINE_AA, tipLength=0.3)

        label_pos = box_i.reshape(-1, 2).mean(axis=0).astype(int)

        _draw_text_block(
            frame,
            [
                f"S{slot_id}: L={length:.3f} W={width:.3f}",
                f"yaw={math.degrees(yaw):.1f}",
            ],
            int(label_pos[0]) + 6,
            int(label_pos[1]) - 8,
            color=color,
            scale=0.38,
        )

def draw_planner_nodes(frame, nodes, project):
    if not nodes:
        return

    count = len(nodes)

    for i, (x, y) in enumerate(nodes):
        t = i / max(1, count - 1)

        color = (
            int(255 * (1.0 - t)),
            int(255 * t),
            0,
        )

        p = project(float(x), float(y), frame)
        cv2.circle(frame, p, 1, color, -1)

def draw_drivable_area_on_frame(frame, payload: dict) -> None:
    area = payload.get("drivable_area")
    #print("drivable_area:", payload.get("drivable_area"))
    if not area:
        return

    if area.get("type") != "image_polygon":
        return

    pts = area.get("points", [])
    if len(pts) < 3:
        return

    frame_h, frame_w = frame.shape[:2]

    payload_w = float(payload.get("image_width", frame_w))
    payload_h = float(payload.get("image_height", frame_h))

    sx = frame_w / payload_w
    sy = frame_h / payload_h

    poly = np.array(
        [
            [int(round(u * sx)), int(round(v * sy))]
            for u, v in pts
        ],
        dtype=np.int32,
    )

    cv2.polylines(
        frame,
        [poly],
        isClosed=True,
        color=(0, 255, 0),
        thickness=3,
    )

def draw_box_on_frame(
    frame,
    box: dict,
    project,
    label: str,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> np.ndarray:
    pts = box_to_pixel_polygon(box, project, frame)

    cv2.polylines(
        frame,
        [pts],
        isClosed=True,
        color=color,
        thickness=thickness,
    )

    cx = float(box["x"])
    cy = float(box["y"])
    center_u, center_v = project(cx, cy, frame)

    cv2.circle(
        frame,
        (center_u, center_v),
        4,
        color,
        thickness=-1,
    )

    cv2.putText(
        frame,
        label,
        (center_u + 6, center_v - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        2,
    )

    return pts

def get_parking_slots(payload):
    slots = payload.get("parking_slots")

    if slots is None:
        slots = payload.get("available_parking_slots", [])

    return sorted(
        [
            {**slot, "id": int(slot.get("id", i))}
            for i, slot in enumerate(slots)
        ],
        key=lambda slot: slot["id"],
    )

def set_goal_from_slot(payload: dict, slot: dict) -> dict:
    payload = payload.copy()

    payload["goal_pose"] = {
        "x": float(slot["x"]),
        "y": float(slot["y"]),
        "yaw": float(slot.get("yaw", 0.0)),
    }

    return payload

def select_parking_slot_graphical(source, config: SourceConfig, payload: dict) -> dict:
    slots = get_parking_slots(payload)

    if not slots:
        raise ValueError("No free parking slots available")

    window_name = "Select parking slot"

    state = {
        "selected_index": None,
        "buttons": [],
    }

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # Klick auf Button-Liste
        for idx, (x1, y1, x2, y2) in state["buttons"]:
            if x1 <= x <= x2 and y1 <= y <= y2:
                state["selected_index"] = idx
                return

        # Optional: Klick nahe Slot-Zentrum im Bild
        frame = param.get("frame")
        if frame is None:
            return

        for idx, slot in enumerate(slots):
            u, v = carla_world_to_pixel(
                float(slot["x"]),
                float(slot["y"]),
                frame,
                config,
            )

            if (x - u) ** 2 + (y - v) ** 2 < 25 ** 2:
                state["selected_index"] = idx
                return

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse, {"frame": None})

    print("Select parking slot in OpenCV window.")
    print("Click a slot button, click near a slot marker, or press number key 0-9.")

    for frame in source:
        if frame is None:
            break


        overlay = frame.copy()
        state["buttons"] = []

        # Maus-Callback braucht aktuelles Frame für world->pixel
        cv2.setMouseCallback(window_name, on_mouse, {"frame": overlay})

        draw_slot_selection_overlay(
            overlay,
            slots,
            config,
            state,
        )

        cv2.imshow(window_name, overlay)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            raise RuntimeError("Parking slot selection cancelled")

        # Zahlentasten 0-9
        if ord("0") <= key <= ord("9"):
            idx = key - ord("0")
            if idx < len(slots):
                state["selected_index"] = idx

        if state["selected_index"] is not None:
            selected = slots[state["selected_index"]]
            cv2.destroyWindow(window_name)
            print("Selected parking slot:", selected)
            return selected

    raise RuntimeError("No frame available for parking slot selection")

def carla_world_to_pixel(x: float, y: float, frame, config: SourceConfig) -> tuple[int, int]:
    boundaries = config.carla.scenario.boundaries

    cx = boundaries.center.x
    cy = boundaries.center.y
    ex = boundaries.extent.x
    ey = boundaries.extent.y

    height, width = frame.shape[:2]

    min_x = cx - ex
    max_x = cx + ex
    min_y = cy - ey
    max_y = cy + ey

    u = int((x - min_x) / (max_x - min_x) * width)
    v = int((max_y - y) / (max_y - min_y) * height)

    return u, v

def draw_slot_selection_overlay(
    frame,
    slots: list[dict],
    config: SourceConfig,
    state: dict,
) -> None:
    height, width = frame.shape[:2]

    panel_x1 = width - 330
    panel_x2 = width - 10
    panel_y1 = 20
    panel_y2 = min(height - 20, 80 + len(slots) * 45)

    cv2.rectangle(
        frame,
        (panel_x1, panel_y1),
        (panel_x2, panel_y2),
        (30, 30, 30),
        thickness=-1,
    )

    cv2.putText(
        frame,
        "Select parking slot",
        (panel_x1 + 15, panel_y1 + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    state["buttons"] = []

    for idx, slot in enumerate(slots):
        slot_id = slot.get("id", idx)

        x1 = panel_x1 + 15
        y1 = panel_y1 + 50 + idx * 42
        x2 = panel_x2 - 15
        y2 = y1 + 32

        state["buttons"].append((idx, (x1, y1, x2, y2)))

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (70, 70, 70),
            thickness=-1,
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            thickness=1,
        )

        text = f"[{idx}] slot {slot_id}  x={slot['x']:.1f} y={slot['y']:.1f}"

        cv2.putText(
            frame,
            text,
            (x1 + 10, y1 + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        u, v = carla_world_to_pixel(
            float(slot["x"]),
            float(slot["y"]),
            frame,
            config,
        )

        if 0 <= u < width and 0 <= v < height:
            cv2.circle(frame, (u, v), 18, (0, 255, 0), thickness=2)

            cv2.putText(
                frame,
                str(idx),
                (u - 6, v + 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

def build_slot_panel(slots, selected_index: int) -> np.ndarray:
    panel_h = 260
    panel_w = 420
    panel = np.full((panel_h, panel_w, 3), 25, dtype=np.uint8)

    cv2.putText(panel, "Select parking slot", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(panel, "UP/DOWN: select   ENTER: replan   Q: quit", (15, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    y = 90
    for i, slot in enumerate(slots):
        color = (0, 255, 0) if i == selected_index else (180, 180, 180)
        text = f"[{i}] x={slot['x']:.1f}  y={slot['y']:.1f}"
        cv2.putText(panel, text, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2 if i == selected_index else 1)
        y += 32

    return panel

def make_slot_overlay_mouse_callback(state: dict):
    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        scale = state.get("display_scale", 1.0)
        if scale <= 0:
            scale = 1.0

        x = int(x / scale)
        y = int(y / scale)

        for slot_id, rect in state.get("buttons", []):
            x1, y1, x2, y2 = rect

            if x1 <= x <= x2 and y1 <= y <= y2:
                state["clicked_slot_id"] = slot_id
                return

        for slot_id, poly in state.get("slot_polygons", []):
            if cv2.pointPolygonTest(poly, (x, y), False) >= 0:
                state["clicked_slot_id"] = slot_id
                return

    return on_mouse
def compose_frame_with_side_panel(frame, panel_width=420, gap=10):
    h, w = frame.shape[:2]
    canvas = np.full((h, w + gap + panel_width, 3), 35, dtype=np.uint8)
    canvas[:, :w] = frame
    return canvas, (0, 0, w, h), (w + gap, 0, panel_width, h)

def draw_slot_overlay(frame, slots, selected_slot_id, overlay_state, panel_rect, planner_state="idle"):
    px, py, pw, ph = panel_rect
    overlay_state["buttons"] = []

    cv2.rectangle(frame, (px, py), (px + pw - 1, py + ph - 1), (35, 35, 35), -1)

    cv2.putText(frame, "Parking slot selection", (px + 18, py + 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "click / 0-9 / W-S / Q", (px + 18, py + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Planner: {planner_state}", (px + 18, py + 87),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    if not slots:
        return

    top = py + 110
    bottom_margin = 15
    available_height = max(1, ph - 110 - bottom_margin)
    row_step = min(44, max(27, available_height // len(slots)))
    row_height = max(23, row_step - 5)

    for i, slot in enumerate(slots):
        slot_id = int(slot["id"])

        x1 = px + 14
        x2 = px + pw - 14
        y1 = top + i * row_step
        y2 = min(y1 + row_height, py + ph - bottom_margin)

        if y1 >= py + ph - bottom_margin:
            break

        selected = slot_id == selected_slot_id
        fill = (45, 75, 45) if selected else (48, 48, 48)
        border = (0, 255, 0) if selected else (110, 110, 110)
        text_color = (220, 255, 220) if selected else (220, 220, 220)

        cv2.rectangle(frame, (x1, y1), (x2, y2), fill, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), border, 2 if selected else 1)

        text = f"[{slot_id}] slot {slot_id}   x={slot['x']:.2f}   y={slot['y']:.2f}"

        cv2.putText(frame, text, (x1 + 9, y1 + min(21, row_height - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, text_color, 1, cv2.LINE_AA)

        overlay_state["buttons"].append((slot_id, (x1, y1, x2, y2)))

def draw_scene_on_frame(frame, payload, slots, selected_slot_id, project, state=None, parked_slot_id=None):
    if state is not None:
        state["slot_polygons"] = []

    for i, obs in enumerate(payload.get("obstacles", [])):
        draw_box_on_frame(
            frame,
            obs,
            project,
            label=f"obs {i}",
            color=(0, 255, 255),
            thickness=2,
        )

    for slot in slots:
        slot_id = int(slot["id"])
        is_selected = slot_id == selected_slot_id
        is_parked = slot_id == parked_slot_id

        if is_parked:
            color = (0, 255, 0)
            thickness = 4
            label = f"PARKED {slot_id}"
        elif is_selected:
            color = (255, 0, 0)
            thickness = 3
            label = f"SELECTED {slot_id}"
        else:
            color = (0, 255, 255)
            thickness = 2
            label = f"slot {slot_id}"

        pts = draw_box_on_frame(frame, slot, project, label, color, thickness)

        if state is not None:
            state["slot_polygons"].append((slot_id, pts))

def resize_for_display(frame, max_w: int = 1400, max_h: int = 950):
    h, w = frame.shape[:2]
    scale = min(max_w / w, max_h / h)

    if scale <= 0:
        scale = 1.0

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return resized, scale

def draw_path_to_goal_on_frame(frame, path, project, goal_pose: dict, color=(0, 0, 255), thickness: int = 4, stride: int = 1) -> None:
    if not path:
        return

    sampled_path = list(path[::stride])

    if sampled_path[-1] is not path[-1]:
        sampled_path.append(path[-1])

    points = [
        project(float(node.x), float(node.y), frame)
        for node in sampled_path
    ]

    if len(points) >= 2:
        for i in range(len(points) - 1):
            direction = int(getattr(sampled_path[i + 1], "direction", 1))
            segment_color = (0, 0, 255) if direction > 0 else (0, 165, 255)

            cv2.line(
                frame,
                points[i],
                points[i + 1],
                segment_color,
                thickness,
            )

    goal_point = project(float(goal_pose["x"]), float(goal_pose["y"]), frame)

    if points:
        cv2.circle(frame, points[0], 6, (0, 255, 0), -1)
        cv2.circle(frame, points[-1], 5, (255, 255, 255), -1)

    cv2.circle(frame, goal_point, 7, (255, 0, 255), -1)
    cv2.circle(frame, goal_point, 11, (255, 255, 255), 2)

    cv2.putText(
        frame,
        "GOAL",
        (goal_point[0] + 8, goal_point[1] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 255),
        2,
    )

def make_bounds_projector(projection: dict):
    required = ["center_x", "center_y", "extent_x", "extent_y"]
    missing = [key for key in required if key not in projection]

    if missing:
        raise ValueError(f"Missing bounds projection keys: {missing}")

    center_x = float(projection["center_x"])
    center_y = float(projection["center_y"])
    extent_x = float(projection["extent_x"])
    extent_y = float(projection["extent_y"])

    invert_y = bool(projection.get("invert_y", True))

    def project_bounds(x: float, y: float, frame):
        height, width = frame.shape[:2]

        min_x = center_x - extent_x
        max_x = center_x + extent_x
        min_y = center_y - extent_y
        max_y = center_y + extent_y

        u = (float(x) - min_x) / (max_x - min_x) * width

        if invert_y:
            v = (max_y - float(y)) / (max_y - min_y) * height
        else:
            v = (float(y) - min_y) / (max_y - min_y) * height

        return int(round(u)), int(round(v))

    return project_bounds

def make_carla_camera_projector(camera, ground_z: float = 0.1):
    def build_intrinsic_matrix(width: int, height: int, fov_deg: float) -> np.ndarray:
        focal = width / (2.0 * np.tan(np.deg2rad(fov_deg) / 2.0))

        return np.array(
            [
                [focal, 0.0, width / 2.0],
                [0.0, focal, height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def project_carla_camera(x: float, y: float, frame):
        height, width = frame.shape[:2]

        fov = float(camera.attributes.get("fov", 90.0))
        K = build_intrinsic_matrix(width, height, fov)

        world_to_camera = np.array(
            camera.get_transform().get_inverse_matrix(),
            dtype=np.float64,
        )

        # CARLA world point on ground plane
        point_world = np.array(
            [float(x), float(y), float(ground_z), 1.0],
            dtype=np.float64,
        )

        point_camera = world_to_camera @ point_world

        # CARLA/Unreal camera coordinates -> standard image coordinates
        # CARLA camera looks along local +X.
        point_image_frame = np.array(
            [
                point_camera[1],
                -point_camera[2],
                point_camera[0],
            ],
            dtype=np.float64,
        )

        # Punkt liegt hinter der Kamera
        if point_image_frame[2] <= 1e-6:
            return -999999, -999999

        projected = K @ point_image_frame

        u = projected[0] / projected[2]
        v = projected[1] / projected[2]

        return int(round(u)), int(round(v))

    return project_carla_camera

def identity(x: float, y: float, frame):
    return int(round(x)), int(round(y))

def make_projector(config, payload: dict | None = None, source=None):
    payload = payload or {}
    projection = payload.get("projection")

    if projection is not None:
        projection_type = projection.get("type") or projection.get("mode")

        if projection_type == "homography":
            H = np.array(projection["H_world_to_image"], dtype=np.float64)

            payload_image_width = float(payload.get("image_width", 0.0))
            payload_image_height = float(payload.get("image_height", 0.0))

            def project_homography(x: float, y: float, frame):
                p = np.array([float(x), float(y), 1.0], dtype=np.float64)
                q = H @ p

                if abs(q[2]) < 1e-9:
                    return -999999, -999999

                u = q[0] / q[2]
                v = q[1] / q[2]

                frame_h, frame_w = frame.shape[:2]

                if payload_image_width > 0 and payload_image_height > 0:
                    u *= frame_w / payload_image_width
                    v *= frame_h / payload_image_height

                return int(round(u)), int(round(v))

            return project_homography

        if projection_type == "bounds":
            return make_bounds_projector(projection)

    if config.source_type == "carla" and source is not None:
        camera = getattr(source, "_camera", None)

        if camera is not None:
            ground_z = float(payload.get("projection", {}).get("ground_z", 0.1))
            return make_carla_camera_projector(camera=camera, ground_z=ground_z)

    return identity

def box_to_pixel_polygon(
    box: dict,
    project,
    frame,
) -> np.ndarray:
    cx = float(box["x"])
    cy = float(box["y"])
    yaw = float(box.get("yaw", 0.0))

    half_l = float(box["length"]) / 2.0
    half_w = float(box["width"]) / 2.0

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
        u, v = project(x, y, frame)
        corners.append((u, v))

    return np.array(corners, dtype=np.int32)

def draw_aruco_debug(frame, detected, H_world_to_image=None):
    for marker_id, corners in detected.items():
        pts = np.asarray(corners, dtype=np.float32).reshape(4, 2)
        pts_i = np.round(pts).astype(np.int32)

        cv2.polylines(frame, [pts_i], True, (255, 0, 255), 2)

        center = pts.mean(axis=0)
        cx, cy = int(center[0]), int(center[1])

        cv2.circle(frame, (cx, cy), 4, (255, 0, 255), -1)
        cv2.putText(frame, f"ArUco {marker_id}", (cx + 6, cy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 2)

        left = (pts[0] + pts[3]) / 2
        right = (pts[1] + pts[2]) / 2
        top = (pts[0] + pts[1]) / 2
        bottom = (pts[2] + pts[3]) / 2

        x_dir = right - left
        y_dir = bottom - top

        x_norm = np.linalg.norm(x_dir)
        y_norm = np.linalg.norm(y_dir)

        if x_norm > 1e-6:
            x_dir /= x_norm
        if y_norm > 1e-6:
            y_dir /= y_norm

        axis_len = max(20.0, min(x_norm, y_norm) * 0.8)

        px = tuple(np.round(center + x_dir * axis_len).astype(int))
        py = tuple(np.round(center + y_dir * axis_len).astype(int))

        cv2.arrowedLine(frame, (cx, cy), px, (0, 0, 255), 2, tipLength=0.25)
        cv2.arrowedLine(frame, (cx, cy), py, (0, 255, 0), 2, tipLength=0.25)

        cv2.putText(frame, "X", (px[0] + 3, px[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(frame, "Y", (py[0] + 3, py[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    if H_world_to_image is not None:
        draw_world_axes(frame, H_world_to_image, (0.0, 0.0), axis_len=0.25, label="MAT")

def project_point(H_world_to_image, x, y):
    p = np.array([x, y, 1.0], dtype=np.float32)
    q = H_world_to_image @ p
    if abs(q[2]) < 1e-9:
        return None
    return int(q[0] / q[2]), int(q[1] / q[2])

def draw_world_axes(frame, H_world_to_image, origin_xy, axis_len=0.12, label=None):
    ox, oy = origin_xy

    p0 = project_point(H_world_to_image, ox, oy)
    px = project_point(H_world_to_image, ox + axis_len, oy)
    py = project_point(H_world_to_image, ox, oy + axis_len)

    if p0 is None or px is None or py is None:
        return

    cv2.arrowedLine(frame, p0, px, (0, 0, 255), 2, tipLength=0.2)   # X = rot
    cv2.arrowedLine(frame, p0, py, (0, 255, 0), 2, tipLength=0.2)   # Y = grün
    cv2.circle(frame, p0, 3, (255, 255, 0), -1)

    if label is not None:
        cv2.putText(frame, str(label), (p0[0] + 5, p0[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)