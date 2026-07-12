import cv2
import math
from input import SourceConfig
import numpy as np
from typing import Callable

Projector = Callable[[float, float, np.ndarray], tuple[int, int]]

def draw_drivable_area_on_frame(frame, payload: dict) -> None:
    area = payload.get("drivable_area")
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

def get_parking_slots(payload: dict) -> list[dict]:
    if "available_parking_slots" in payload:
        return [
            {**slot, "id": i}
            for i, slot in enumerate(payload["available_parking_slots"])
        ]

    if "parking_slots" in payload:
        return [
            {**slot, "id": slot.get("id", i)}
            for i, slot in enumerate(payload["parking_slots"])
            if slot.get("free", True)
        ]

    if "parking_slot" in payload:
        slot = payload["parking_slot"]

        if isinstance(slot, list):
            return [{**s, "id": i} for i, s in enumerate(slot)]

        return [{**slot, "id": 0}]

    return []

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

        for idx, rect in state.get("buttons", []):
            x1, y1, x2, y2 = rect

            if x1 <= x <= x2 and y1 <= y <= y2:
                state["clicked_index"] = idx
                return

        for idx, poly in state.get("slot_polygons", []):
            if cv2.pointPolygonTest(poly, (x, y), False) >= 0:
                state["clicked_index"] = idx
                return

    return on_mouse

def draw_slot_overlay(frame, slots: list[dict], selected_index: int, overlay_state: dict) -> None:
    h, w = frame.shape[:2]

    display_scale = float(overlay_state.get("display_scale", 1.0))
    if display_scale <= 0:
        display_scale = 1.0

    inv = 1.0 / display_scale

    panel_w = int(min(w * 0.48, 560 * inv))
    row_h = int(44 * inv)
    header_h = int(78 * inv)
    pad = int(16 * inv)

    margin_x = int(18 * inv)
    margin_y = int(18 * inv)

    panel_h = header_h + len(slots) * row_h + pad
    x1 = w - panel_w - margin_x
    y1 = margin_y
    x2 = w - margin_x
    y2 = min(h - margin_y, y1 + panel_h)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), max(2, int(2 * inv)))

    cv2.putText(
        frame,
        "Parking slot selection",
        (x1 + int(16 * inv), y1 + int(34 * inv)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8 * inv,
        (255, 255, 255),
        max(1, int(2 * inv)),
    )

    cv2.putText(
        frame,
        "click / 0-9 / W-S / Q",
        (x1 + int(16 * inv), y1 + int(60 * inv)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48 * inv,
        (210, 210, 210),
        max(1, int(1 * inv)),
    )

    overlay_state["buttons"] = []

    y = y1 + header_h

    for i, slot in enumerate(slots):
        bx1 = x1 + int(14 * inv)
        by1 = y
        bx2 = x2 - int(14 * inv)
        by2 = y + int(32 * inv)

        overlay_state["buttons"].append((i, (bx1, by1, bx2, by2)))

        selected = i == selected_index
        border = (0, 255, 0) if selected else (120, 120, 120)
        text_color = (220, 255, 220) if selected else (220, 220, 220)

        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (40, 40, 40), -1)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), border, max(1, int((2 if selected else 1) * inv)))

        text = f"[{i}] slot {slot.get('id', i)}   x={slot['x']:.1f}   y={slot['y']:.1f}"

        cv2.putText(
            frame,
            text,
            (bx1 + int(10 * inv), by1 + int(22 * inv)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5 * inv,
            text_color,
            max(1, int(1 * inv)),
        )

        y += row_h

def draw_scene_on_frame(
    frame,
    payload: dict,
    slots: list[dict],
    selected_slot_index: int,
    project,
    state: dict | None = None,
    parked_slot_index: int | None = None,
) -> None:
    if state is not None:
        state["slot_polygons"] = []

    # Obstacles: gelb
    for i, obs in enumerate(payload.get("obstacles", [])):
        draw_box_on_frame(
            frame,
            obs,
            project,
            label=f"obs {i}",
            color=(0, 255, 255),
            thickness=2,
        )

    # Parking slots
    for i, slot in enumerate(slots):
        is_selected = i == selected_slot_index
        is_parked = parked_slot_index == i

        if is_parked:
            color = (0, 255, 0)      # grün
            thickness = 4
            label = f"PARKED {i}"
        elif is_selected:
            color = (255, 0, 0)      # blau
            thickness = 3
            label = f"SELECTED {i}"
        else:
            color = (0, 255, 255)    # gelb
            thickness = 2
            label = f"slot {i}"

        pts = draw_box_on_frame(
            frame,
            slot,
            project,
            label=label,
            color=color,
            thickness=thickness,
        )

        if state is not None:
            state["slot_polygons"].append((i, pts))

def resize_for_display(frame, max_w: int = 1400, max_h: int = 950):
    h, w = frame.shape[:2]
    scale = min(max_w / w, max_h / h)

    if scale <= 0:
        scale = 1.0

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return resized, scale

def draw_path_to_goal_on_frame(frame, path, project, goal_pose: dict, color=(0, 0, 255), thickness: int = 2, stride: int = 1) -> None:
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