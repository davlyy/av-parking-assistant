import cv2
import math
from input import SourceConfig
import numpy as np
from typing import Callable

Projector = Callable[[float, float, np.ndarray], tuple[int, int]]


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
    raw_slots = payload.get("parking_slots", None)

    if raw_slots is None:
        raw_slots = payload.get("parking_slot", None)

    if raw_slots is None:
        return []

    # altes Format: "parking_slot": {...}
    if isinstance(raw_slots, dict):
        raw_slots = [raw_slots]

    # neues Format: "parking_slot": [{...}, {...}]
    if not isinstance(raw_slots, list):
        raise TypeError(
            "'parking_slot' or 'parking_slots' must be either a dict or a list of dicts"
        )

    slots = []

    for i, slot in enumerate(raw_slots):
        if not isinstance(slot, dict):
            raise TypeError(f"Parking slot at index {i} is not a dict: {type(slot)}")

        if not slot.get("free", True):
            continue

        slots.append({
            **slot,
            "id": slot.get("id", i),
            "free": slot.get("free", True),
        })

    return slots

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

        # 1. Klick auf Overlay-Buttons
        for idx, (x1, y1, x2, y2) in state.get("buttons", []):
            if x1 <= x <= x2 and y1 <= y <= y2:
                state["clicked_index"] = idx
                print("Clicked overlay slot:", idx)
                return

        # 2. Klick direkt auf gezeichnete Parking-Slot-Box
        for idx, polygon in state.get("slot_polygons", []):
            inside = cv2.pointPolygonTest(
                polygon,
                (float(x), float(y)),
                False,
            )

            if inside >= 0:
                state["clicked_index"] = idx
                print("Clicked parking slot polygon:", idx)
                return

    return on_mouse

def draw_slot_overlay(
    frame,
    slots: list[dict],
    selected_index: int,
    state: dict,
) -> None:
    height, width = frame.shape[:2]

    panel_w = 390
    panel_x1 = width - panel_w - 20
    panel_y1 = 20
    panel_x2 = width - 20
    panel_y2 = min(height - 20, panel_y1 + 90 + len(slots) * 42)

    # dunkles transparentes Panel
    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (panel_x1, panel_y1),
        (panel_x2, panel_y2),
        (20, 20, 20),
        thickness=-1,
    )

    cv2.addWeighted(
        overlay,
        0.72,
        frame,
        0.28,
        0,
        frame,
    )

    # Rahmen
    cv2.rectangle(
        frame,
        (panel_x1, panel_y1),
        (panel_x2, panel_y2),
        (0, 255, 255),
        thickness=2,
    )

    cv2.putText(
        frame,
        "Parking slot selection",
        (panel_x1 + 15, panel_y1 + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        "click / 0-9 / W-S / Q",
        (panel_x1 + 15, panel_y1 + 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (190, 190, 190),
        1,
    )

    state["buttons"] = []

    for idx, slot in enumerate(slots):
        slot_id = slot.get("id", idx)

        x1 = panel_x1 + 15
        y1 = panel_y1 + 75 + idx * 42
        x2 = panel_x2 - 15
        y2 = y1 + 32

        state["buttons"].append((idx, (x1, y1, x2, y2)))

        is_selected = idx == selected_index

        if is_selected:
            fill_color = (40, 90, 40)
            border_color = (0, 255, 0)
            text_color = (255, 255, 255)
            thickness = 2
        else:
            fill_color = (55, 55, 55)
            border_color = (140, 140, 140)
            text_color = (220, 220, 220)
            thickness = 1

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            fill_color,
            thickness=-1,
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            border_color,
            thickness=thickness,
        )

        text = f"[{idx}] slot {slot_id}   x={slot['x']:.1f}  y={slot['y']:.1f}"

        cv2.putText(
            frame,
            text,
            (x1 + 10, y1 + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            text_color,
            1,
        )

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


def draw_path_on_frame(
    frame,
    path,
    project: Projector,
    color: tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
    stride: int = 1,
    goal_pose: dict | None = None,
) -> None:
    if not path:
        return

    points = []

    for node in path[::stride]:
        u, v = project(float(node.x), float(node.y), frame)
        points.append((u, v))

    # Visuell bis zur echten Parkplatzmitte weiterzeichnen
    if goal_pose is not None:
        goal_point = project(
            float(goal_pose["x"]),
            float(goal_pose["y"]),
            frame,
        )

        if not points or points[-1] != goal_point:
            points.append(goal_point)

    if len(points) < 2:
        return

    pts = np.array(points, dtype=np.int32)

    cv2.polylines(
        frame,
        [pts],
        isClosed=False,
        color=color,
        thickness=thickness,
    )

    # Startpunkt
    cv2.circle(frame, points[0], 6, (0, 255, 0), thickness=-1)

    # Zielpunkt = Parkplatzmitte
    cv2.circle(frame, points[-1], 7, (255, 0, 255), thickness=-1)
    cv2.circle(frame, points[-1], 11, (255, 255, 255), thickness=2)

    cv2.putText(
        frame,
        "GOAL",
        (points[-1][0] + 8, points[-1][1] - 8),
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

def make_projector(
    config: SourceConfig,
    payload: dict | None = None,
    source=None,
):
    payload = payload or {}
    projection = payload.get("projection")

    # ------------------------------------------------------------
    # 1. Payload-Projektion: echte Welt / Homography
    # ------------------------------------------------------------
    if projection is not None:
        projection_type = projection.get("type") or projection.get("mode")

        if projection_type == "homography":
            if "H_world_to_image" in projection:
                H = np.array(
                    projection["H_world_to_image"],
                    dtype=np.float64,
                )

                if H.shape != (3, 3):
                    raise ValueError(
                        f"H_world_to_image must be a 3x3 matrix, got shape {H.shape}"
                    )

                def project_homography(x: float, y: float, frame):
                    point = np.array(
                        [float(x), float(y), 1.0],
                        dtype=np.float64,
                    )

                    projected = H @ point

                    if abs(projected[2]) < 1e-9:
                        return -999999, -999999

                    u = projected[0] / projected[2]
                    v = projected[1] / projected[2]

                    return int(round(u)), int(round(v))

                return project_homography

            if "world_points" in projection and "image_points" in projection:
                world_points = np.array(
                    projection["world_points"],
                    dtype=np.float32,
                )

                image_points = np.array(
                    projection["image_points"],
                    dtype=np.float32,
                )

                if len(world_points) < 4 or len(image_points) < 4:
                    raise ValueError(
                        "Homography needs at least 4 world_points and 4 image_points"
                    )

                H, _ = cv2.findHomography(world_points, image_points)

                if H is None:
                    raise RuntimeError("Could not compute homography")

                def project_homography_points(x: float, y: float, frame):
                    point = np.array(
                        [[[float(x), float(y)]]],
                        dtype=np.float32,
                    )

                    projected = cv2.perspectiveTransform(point, H)

                    u = projected[0, 0, 0]
                    v = projected[0, 0, 1]

                    return int(round(u)), int(round(v))

                return project_homography_points

            raise ValueError(
                "Homography projection requires either 'H_world_to_image' "
                "or both 'world_points' and 'image_points'"
            )

        # Optionaler Debug-/Fallback-Modus für perfekte Top-Down-Ansichten
        if projection_type == "bounds":
            return make_bounds_projector(projection)

    # ------------------------------------------------------------
    # 2. CARLA: echte Kamera-Projektion verwenden
    # ------------------------------------------------------------
    if config.source_type == "carla" and source is not None:
        camera = getattr(source, "_camera", None)

        if camera is not None:
            ground_z = float(
                payload.get("projection", {}).get("ground_z", 0.1)
            )

            return make_carla_camera_projector(
                camera=camera,
                ground_z=ground_z,
            )

    # ------------------------------------------------------------
    # 3. Letzter CARLA-Fallback: alte boundary-Projektion
    #    Nur benutzen, wenn du wirklich eine perfekte Top-Down-Karte hast.
    # ------------------------------------------------------------
    if config.source_type == "carla" and config.carla is not None:
        def project_carla_bounds(x: float, y: float, frame):
            return carla_world_to_pixel(x, y, frame, config)

        return project_carla_bounds

    # ------------------------------------------------------------
    # 4. Fallback für Pixel-Koordinaten
    # ------------------------------------------------------------
    def project_identity(x: float, y: float, frame):
        return int(round(x)), int(round(y))

    return project_identity

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