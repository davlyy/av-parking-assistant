from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from input import load_config, CameraSource, CarlaSource, ImageSource, SourceConfig
from mod.path_planning import convert_payload_to_carla_world, plan_path
from mod.draw import (
    draw_box_on_frame,
    draw_scene_on_frame,
    draw_slot_overlay,
    draw_path_to_goal_on_frame,
    get_parking_slots,
    make_projector,
    make_slot_overlay_mouse_callback,
    set_goal_from_slot,
    resize_for_display,
    draw_drivable_area_on_frame
)


def create_source(config: SourceConfig):
    if config.source_type == "carla":
        if config.carla is None:
            raise ValueError("CARLA config missing")
        return CarlaSource(config.carla)

    if config.source_type == "camera":
        if config.camera is None:
            raise ValueError("Camera config missing")
        return CameraSource(config.camera)

    if config.source_type == "image":
        if config.image is None:
            raise ValueError("Image config missing")
        return ImageSource(config.image)

    raise ValueError(f"Unknown source_type: {config.source_type}")


def ensure_bgr(frame):
    if frame is None:
        return None
    if len(frame.shape) == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


def _draw_coords(frame, source) -> None:
    if not isinstance(source, CarlaSource):
        return

    x, y, z, yaw = source.vehicle_pose()
    for i, line in enumerate([f"X: {x:.1f}", f"Y: {y:.1f}", f"Z: {z:.1f}", f"Yaw: {yaw:.1f}"]):
        cv2.putText(frame, line, (10, 30 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

def carla_yaw_to_rad(yaw_deg: float) -> float:
    return math.radians(float(yaw_deg))

def current_source_pose(source, fallback_payload: dict | None = None) -> dict | None:
    if isinstance(source, CarlaSource):
        x, y, z, yaw_deg = source.vehicle_pose()
        return {
            "x": float(x),
            "y": float(y),
            "yaw": carla_yaw_to_rad(yaw_deg),
        }

    if fallback_payload is not None and "start_pose" in fallback_payload:
        pose = fallback_payload["start_pose"]
        return {
            "x": float(pose["x"]),
            "y": float(pose["y"]),
            "yaw": float(pose.get("yaw", 0.0)),
        }

    return None


def get_planner_goal_tolerance(default: float = 3.0) -> float:
    planner_cfg = getattr(plan_path, "__globals__", {}).get("config", None)
    return float(getattr(planner_cfg, "goal_tolerance", default)) if planner_cfg else default


def distance_xy(a: dict | None, b: dict | None) -> float:
    if a is None or b is None:
        return float("inf")
    return math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))


def is_within_goal_tolerance(pose: dict | None, goal_pose: dict | None, goal_tolerance: float) -> bool:
    return distance_xy(pose, goal_pose) <= goal_tolerance


def closest_path_index(path, pose: dict | None, start_index: int = 0, search_back: int = 5, search_forward: int = 120) -> tuple[int, float]:
    if not path or pose is None:
        return 0, float("inf")

    px, py = float(pose["x"]), float(pose["y"])
    start = max(0, start_index - search_back)
    end = min(len(path), start_index + search_forward)

    best_index = start
    best_dist = float("inf")

    for i in range(start, end):
        node = path[i]
        dist = math.hypot(float(node.x) - px, float(node.y) - py)
        if dist < best_dist:
            best_dist = dist
            best_index = i

    return best_index, best_dist


def trim_path_for_display(path, current_pose: dict | None, progress_index: int):
    if not path:
        return []

    progress_index = max(0, min(progress_index, len(path) - 1))
    remaining = list(path[progress_index:])

    if current_pose is None:
        return remaining

    current_node = SimpleNamespace(x=float(current_pose["x"]), y=float(current_pose["y"]))
    return [current_node] + remaining

def draw_top_status_banner(frame, text: str, color=(0, 255, 0)) -> None:
    _, width = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.4
    thickness = 3
    padding_x = 28
    padding_y = 18

    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    box_w = text_w + 2 * padding_x
    box_h = text_h + 2 * padding_y + baseline

    x1 = (width - box_w) // 2
    y1 = 20
    x2 = x1 + box_w
    y2 = y1 + box_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    cv2.putText(frame, text, (x1 + padding_x, y1 + padding_y + text_h), font, font_scale, color, thickness)


def run(source, config: SourceConfig, payload: dict) -> None:
    source.open()
    path = []

    try:
        if isinstance(source, CarlaSource):
            x, y, z, yaw_deg = source.vehicle_pose()

            ego_world_pose = {
                "x": float(x),
                "y": float(y),
                "yaw": carla_yaw_to_rad(yaw_deg),
            }

            print("CARLA ego pose:", {
                "x": ego_world_pose["x"],
                "y": ego_world_pose["y"],
                "yaw_deg": float(yaw_deg),
                "yaw_rad": ego_world_pose["yaw"],
            })
            base_payload = convert_payload_to_carla_world(payload, ego_world_pose)
        else:
            base_payload = payload.copy()

        slots = get_parking_slots(base_payload)
        if not slots:
            raise ValueError("No free parking slots available")

        project = make_projector(config, base_payload, source=source)

        selected_slot_index = 0
        planned_slot_index = None
        planning_payload = None

        path_progress_index = 0
        path_replan_deviation = 0.5
        goal_tolerance = get_planner_goal_tolerance(default=3.0)

        overlay_state = {
            "buttons": [],
            "slot_polygons": [],
            "clicked_index": None,
            "display_scale": 1.0,
        }

        window_name = "AV Parking Assistant"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, make_slot_overlay_mouse_callback(overlay_state))

        for frame in source:
            if frame is None:
                break

            frame = ensure_bgr(frame)

            if overlay_state["clicked_index"] is not None:
                clicked = overlay_state["clicked_index"]
                overlay_state["clicked_index"] = None

                if 0 <= clicked < len(slots) and clicked != selected_slot_index:
                    selected_slot_index = clicked
                    print("Selected slot by mouse:", selected_slot_index)

            raw_key = cv2.waitKey(20)
            if raw_key != -1:
                key = raw_key & 0xFF

                if key == ord("q") or key == 27:
                    break

                char = chr(key).lower()

                if char == "w":
                    selected_slot_index = max(0, selected_slot_index - 1)
                    print("Selected slot by key:", selected_slot_index)

                elif char == "s":
                    selected_slot_index = min(len(slots) - 1, selected_slot_index + 1)
                    print("Selected slot by key:", selected_slot_index)

                elif char.isdigit() and int(char) < len(slots):
                    selected_slot_index = int(char)
                    print("Selected slot by key:", selected_slot_index)

            current_pose = current_source_pose(source, fallback_payload=base_payload)
            selected_slot = slots[selected_slot_index]
            current_goal_pose = {
                "x": float(selected_slot["x"]),
                "y": float(selected_slot["y"]),
                "yaw": float(selected_slot.get("yaw", 0.0)),
            }

            slot_changed = planned_slot_index != selected_slot_index
            path_deviation = float("inf")

            if path and not slot_changed:
                closest_idx, path_deviation = closest_path_index(
                    path, current_pose, start_index=path_progress_index, search_back=5, search_forward=120
                )
                path_progress_index = max(path_progress_index, closest_idx)

            off_path = bool(path) and path_deviation > path_replan_deviation
            needs_replan = planning_payload is None or slot_changed or off_path

            if needs_replan:
                planning_payload = base_payload.copy()
                planning_payload["start_pose"] = {
                    "x": float(current_pose["x"]),
                    "y": float(current_pose["y"]),
                    "yaw": float(current_pose.get("yaw", 0.0)),
                }
                planning_payload = set_goal_from_slot(planning_payload, selected_slot)

                print(
                    "Replanning:",
                    "slot_changed=", slot_changed,
                    "off_path=", off_path,
                    "path_deviation=", round(path_deviation, 2) if path_deviation != float("inf") else "inf",
                    "threshold=", path_replan_deviation,
                    "slot=", selected_slot.get("id", selected_slot_index),
                    "start=", planning_payload["start_pose"],
                    "goal=", planning_payload["goal_pose"],
                )

                path = plan_path(planning_payload, already_world=True)
                print("Path length:", len(path) if path else 0)

                if path:
                    reverse_count = sum(1 for node in path if getattr(node, "direction", 1) < 0)
                    forward_count = sum(1 for node in path if getattr(node, "direction", 1) > 0)

                    print("Forward nodes:", forward_count)
                    print("Reverse nodes:", reverse_count)

                planned_slot_index = selected_slot_index
                path_progress_index = 0

                if path:
                    closest_idx, _ = closest_path_index(
                        path, current_pose, start_index=0, search_back=0, search_forward=min(len(path), 200)
                    )
                    path_progress_index = closest_idx

            parked = is_within_goal_tolerance(current_pose, current_goal_pose, goal_tolerance)
            parked_slot_index = selected_slot_index if parked else None

            display_frame = frame.copy()
            draw_drivable_area_on_frame(display_frame, planning_payload if planning_payload else base_payload)

            _, display_scale = resize_for_display(display_frame, max_w=1400, max_h=950)
            overlay_state["display_scale"] = display_scale

            draw_scene_on_frame(display_frame, base_payload, slots, selected_slot_index, project, overlay_state)

            if parked_slot_index is not None:
                draw_box_on_frame(display_frame, slots[parked_slot_index], project, label="", color=(0, 255, 0), thickness=4)

            display_path = trim_path_for_display(path, current_pose, path_progress_index)

            if display_path and planning_payload is not None:
                draw_path_to_goal_on_frame(
                    display_frame,
                    display_path,
                    project,
                    goal_pose=planning_payload["goal_pose"],
                    color=(0, 0, 255),
                    thickness=2,
                    stride=1,
                )

            if isinstance(source, CarlaSource):
                _draw_coords(display_frame, source)

            draw_slot_overlay(display_frame, slots, selected_slot_index, overlay_state)

            if parked_slot_index is not None:
                draw_top_status_banner(display_frame, "PARKED", color=(0, 255, 0))

            display_frame, display_scale = resize_for_display(display_frame, max_w=1400, max_h=950)
            cv2.imshow(window_name, display_frame)

    finally:
        source.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="AV Parking Assistant")
    parser.add_argument("--source", choices=["camera", "carla", "image"], default="camera")
    parser.add_argument("--config", type=str, default=str(Path(__file__).resolve().parent.parent / "config" / "config.json"))
    parser.add_argument("--scenario", type=str, default="scenario_1")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.scenario and config.carla and args.scenario in config.carla.scenarios:
        preset = config.carla.scenarios[args.scenario]
        config = dataclasses.replace(
            config,
            carla=dataclasses.replace(config.carla, scenario=preset, scenario_name=args.scenario),
        )

    payload_path = Path(__file__).resolve().parent.parent / "config" / "payload.json"
    with open(payload_path, encoding="utf-8") as f:
        payload = json.load(f)

    source = create_source(config)
    run(source, config, payload)


if __name__ == "__main__":
    main()