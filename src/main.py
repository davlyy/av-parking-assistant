from __future__ import annotations
import argparse
import math
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import dataclasses
import os

import cv2

from input import load_config, CameraSource, CarlaSource, ImageSource, VideoSource, SourceConfig
from mod.path_planning import plan_path
from mod.draw import (
    draw_scene_on_frame, draw_slot_overlay, draw_path_to_goal_on_frame,
    get_parking_slots, make_projector, make_slot_overlay_mouse_callback,
    set_goal_from_slot, resize_for_display, draw_drivable_area_on_frame,
    compose_frame_with_side_panel,
    draw_debug_measurements_on_frame, draw_planner_nodes,
    draw_parking_mat_on_frame, draw_pose_predictions,
)

PATH_PREPLAN_DEVIATION = 0.03
PATH_REPLAN_DEVIATION = 0.05

GOAL_PREPLAN_DISTANCE = 0.02
GOAL_REPLAN_DISTANCE = 0.04

GOAL_PREPLAN_YAW = math.radians(5)
GOAL_REPLAN_YAW = math.radians(10)

GOAL_REACHED_DISTANCE = 0.04

OBSTACLE_PATH_MARGIN = 0.015
PREPLAN_OBSTACLE_PATH_MARGIN = 0.03

PREPLAN_END_NODES = 6
REPLAN_END_NODES = 2
REPLAN_COOLDOWN_FRAMES = 10

UI_PANEL_WIDTH = 420
UI_PANEL_GAP = 10
DISPLAY_MAX_W = 1600
DISPLAY_MAX_H = 950

try:
    from frame_analyze import process_frame, YOLO_MODEL_PATH
except Exception:
    process_frame = None
    YOLO_MODEL_PATH = "besttoy.pt"

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_model_path(model_name: str = YOLO_MODEL_PATH) -> str:
    candidates = [
        Path(os.environ.get("BESTTOY_MODEL_PATH", "")) if os.environ.get("BESTTOY_MODEL_PATH") else None,
        PROJECT_ROOT / model_name,
        PROJECT_ROOT / "src" / model_name,
        Path.cwd() / model_name,
        Path(model_name),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        if candidate.is_file():
            return str(candidate)
    return str(PROJECT_ROOT / model_name)


def resolve_source_config(config: SourceConfig, source_override: str | None) -> SourceConfig:
    if source_override is None:
        return config
    return dataclasses.replace(config, source_type=source_override)


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
    if config.source_type == "video":
        if config.video is None:
            raise ValueError("Video config missing")
        return VideoSource(config.video)
    raise ValueError(f"Unknown source_type: {config.source_type}")


def _draw_coords(frame, source) -> None:
    if isinstance(source, CarlaSource):
        x, y, z, yaw = source.vehicle_pose()
        for i, line in enumerate((f"X: {x:.1f}", f"Y: {y:.1f}", f"Z: {z:.1f}", f"Yaw: {yaw:.1f}")):
            cv2.putText(frame, line, (10, 30 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def run_carla(source, config: SourceConfig) -> None:
    source.open()
    try:
        if config.carla and config.carla.manual:
            for frame in source:
                if frame is None:
                    break
            return

        for frame in source:
            if frame is None:
                break
            display = ensure_bgr(frame)
            _draw_coords(display, source)
            cv2.imshow("Input - carla", display)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        source.release()
        cv2.destroyAllWindows()


def ensure_bgr(frame):
    if frame is not None and len(frame.shape) == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


def distance_xy(a: dict, b: dict) -> float:
    return math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))


def angle_distance(a: float, b: float) -> float:
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)

def parking_angle_distance(a, b):
    d = angle_distance(a, b)
    return min(d, abs(math.pi - d))


def find_parked_slot_id(pose, slots, position_tolerance=0.09, yaw_tolerance=math.radians(20)):
    x = float(pose["x"])
    y = float(pose["y"])
    yaw = float(pose.get("yaw", 0.0))

    best_slot_id = None
    best_distance = float("inf")

    for slot in slots:
        dx = x - float(slot["x"])
        dy = y - float(slot["y"])
        distance = math.hypot(dx, dy)

        if distance > position_tolerance:
            continue

        slot_yaw = float(slot.get("yaw", 0.0))

        if parking_angle_distance(yaw, slot_yaw) > yaw_tolerance:
            continue

        if distance < best_distance:
            best_distance = distance
            best_slot_id = int(slot["id"])

    return best_slot_id

def slot_distance(a: dict, b: dict) -> float:
    return math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))


def goal_difference(slot: dict, goal: dict | None):
    if goal is None: return float("inf"), float("inf")
    return slot_distance(slot, goal), angle_distance(float(slot.get("yaw", 0.0)), float(goal.get("yaw", 0.0)))

def is_slot_free(slot):
    return bool(slot.get("free", True))

def next_free_slot_index(slots, current_index, direction):
    free_indices = [i for i, slot in enumerate(slots) if is_slot_free(slot)]
    if not free_indices:
        return None
    if current_index not in free_indices:
        return free_indices[0]
    pos = free_indices.index(current_index)
    return free_indices[(pos + direction) % len(free_indices)]

def closest_path_index(path, pose: dict, start_index=0):
    if not path: return 0, float("inf")
    start = max(0, start_index - 5)
    px, py = float(pose["x"]), float(pose["y"])
    best_index, best_distance = start, float("inf")

    for i in range(start, len(path)):
        d = math.hypot(float(path[i].x) - px, float(path[i].y) - py)
        if d < best_distance:
            best_distance, best_index = d, i

    return best_index, best_distance


def trim_path_for_display(path, current_pose: dict, path_index: int):
    if not path: return []
    path_index = max(0, min(path_index, len(path) - 1))
    current = SimpleNamespace(x=float(current_pose["x"]), y=float(current_pose["y"]))
    return [current] + list(path[path_index:])


def path_blocked(path, obstacles, start_index: int, margin: float):
    if not path: return False

    for node in path[start_index:]:
        for obs in obstacles:
            cx, cy = float(obs["x"]), float(obs["y"])
            yaw = float(obs.get("yaw", 0.0))
            dx, dy = float(node.x) - cx, float(node.y) - cy
            c, s = math.cos(yaw), math.sin(yaw)
            local_x = c * dx + s * dy
            local_y = -s * dx + c * dy
            half_l = float(obs["length"]) / 2 + margin
            half_w = float(obs["width"]) / 2 + margin

            if abs(local_x) <= half_l and abs(local_y) <= half_w:
                return True

    return False

def slot_index_by_id(slots, slot_id):
    if slot_id is None:
        return None
    return next((i for i, slot in enumerate(slots) if slot.get("id") == slot_id), None)


def run_planner_async(payload, frame_id, reason, slot_id):
    t = time.perf_counter()

    path, debug_nodes = plan_path(
        payload,
        already_world=True,
        collect_debug=True,
    )

    return (
        path,
        debug_nodes,
        payload,
        frame_id,
        reason,
        slot_id,
        time.perf_counter() - t,
    )


def candidate_is_valid(candidate_path, candidate_payload, selected_slot, current_pose, obstacles):
    if not candidate_path: return False, 0

    goal_distance, goal_yaw = goal_difference(selected_slot, candidate_payload["goal_pose"])
    if goal_distance > GOAL_REPLAN_DISTANCE or goal_yaw > GOAL_REPLAN_YAW:
        return False, 0

    candidate_index, deviation = closest_path_index(candidate_path, current_pose)
    if deviation > PATH_REPLAN_DEVIATION:
        return False, candidate_index

    if path_blocked(candidate_path, obstacles, candidate_index, OBSTACLE_PATH_MARGIN):
        return False, candidate_index

    return True, candidate_index


def build_display_canvas(display_frame, slots, selected_slot_id, overlay_state, planner_future):
    planner_state = "running" if planner_future is not None else "idle"

    canvas, _, panel_rect = compose_frame_with_side_panel(
        display_frame,
        panel_width=UI_PANEL_WIDTH,
        gap=UI_PANEL_GAP,
    )

    draw_slot_overlay(
        canvas,
        slots,
        selected_slot_id,
        overlay_state,
        panel_rect,
        planner_state,
    )

    canvas, scale = resize_for_display(canvas, max_w=DISPLAY_MAX_W, max_h=DISPLAY_MAX_H)
    overlay_state["display_scale"] = scale
    return canvas


def run(source, config: SourceConfig) -> None:
    if process_frame is None:
        raise RuntimeError("frame_analyze could not be imported; the dev perception/planning pipeline is unavailable")

    source.open()

    selected_slot_index = 0
    selected_slot_id = None
    active_path = []
    active_path_index = 0
    active_planning_payload = None
    planned_goal = None
    planner_debug_nodes = []
    force_replan = True
    parked_slot_id = None

    planner_executor = ProcessPoolExecutor(max_workers=1)
    planner_future = None
    last_plan_submit_frame = -REPLAN_COOLDOWN_FRAMES
    planner_total = 0.0
    planner_calls = 0
    planner_submitted = 0
    planner_discarded = 0
    replan_reasons = Counter()

    overlay_state = {
        "buttons": [],
        "debug_buttons": [],
        "slot_polygons": [],
        "clicked_slot_id": None,
        "display_scale": 1.0,
        "debug": {
            "planner_nodes": True,
            "measurements": False,
            "aruco_axes": True,
            "drivable_area": True,
            "pose_prediction": True,
        },
    }

    timing_sum = {
        "payload": 0.0,
        "prepare": 0.0,
        "planner": 0.0,
        "draw": 0.0,
        "total": 0.0,
    }

    timing_frames = 0
    window_name = "AV Parking Assistant"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, make_slot_overlay_mouse_callback(overlay_state))

    frame_id = 0

    try:

        for frame in source:
            t_frame = time.perf_counter()
            frame = ensure_bgr(frame)
            if frame is None: break

            t = time.perf_counter()
            prediction_frame, payload = process_frame(
                frame,
                frame_id=frame_id,
                detector="yolo",
                yolo_model_path=resolve_model_path(),
                confidence=0.2,
                aruco_marker_size_mm=0.98 * 24,
                visualize=True,
            )
            t_payload = time.perf_counter() - t

            if payload is None:
                display_frame = prediction_frame if prediction_frame is not None else frame
                canvas = build_display_canvas(display_frame, [], 0, overlay_state, planner_future)
                cv2.imshow(window_name, canvas)

                if cv2.waitKey(1) & 0xFF in (ord("q"), 27): break

                frame_id += 1
                continue

            t = time.perf_counter()
            slots = get_parking_slots(payload)

            if not slots:
                display_frame = prediction_frame if prediction_frame is not None else frame
                canvas = build_display_canvas(display_frame, [], 0, overlay_state, planner_future)
                cv2.imshow(window_name, canvas)

                if cv2.waitKey(1) & 0xFF in (ord("q"), 27): break

                frame_id += 1
                continue

            stable_index = slot_index_by_id(slots, selected_slot_id)

            if stable_index is None:
                stable_index = next_free_slot_index(slots, 0, 1)
                if stable_index is None:
                    display_frame = prediction_frame if prediction_frame is not None else frame
                    canvas = build_display_canvas(display_frame, slots, 0, overlay_state, planner_future)
                    cv2.imshow(window_name, canvas)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27): break
                    frame_id += 1
                    continue
                selected_slot_id = slots[stable_index]["id"]

            selected_slot_index = stable_index

            force_replan = False
            if not is_slot_free(slots[selected_slot_index]):
                new_index = next_free_slot_index(slots, selected_slot_index, 1)

                if new_index is not None:
                    selected_slot_index = new_index
                    selected_slot_id = int(slots[new_index]["id"])

                    active_path = []
                    active_path_index = 0
                    active_planning_payload = None
                    planned_goal = None
                    planner_debug_nodes = []
                    force_replan = True

            if overlay_state["clicked_slot_id"] is not None:
                clicked_id = overlay_state["clicked_slot_id"]
                overlay_state["clicked_slot_id"] = None

                clicked_index = slot_index_by_id(slots, clicked_id)

                if clicked_index is not None and is_slot_free(slots[clicked_index]):
                    selected_slot_id = clicked_id
                    active_path = []
                    active_path_index = 0
                    active_planning_payload = None
                    planned_goal = None
                    planner_debug_nodes = []
                    force_replan = True

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27): break

            old_index = selected_slot_index

            current_index = slot_index_by_id(slots, selected_slot_id)

            if key == ord("w"):
                new_index = next_free_slot_index(slots, current_index, -1)
                if new_index is not None:
                    selected_slot_id = int(slots[new_index]["id"])

            elif key == ord("s"):
                new_index = next_free_slot_index(slots, current_index, 1)
                if new_index is not None:
                    selected_slot_id = int(slots[new_index]["id"])

            elif ord("0") <= key <= ord("9"):
                requested_id = int(chr(key))
                requested_index = slot_index_by_id(slots, requested_id)

                if requested_index is not None and is_slot_free(slots[requested_index]):
                    selected_slot_id = requested_id

            if selected_slot_index != old_index:
                selected_slot_id = slots[selected_slot_index]["id"]
                active_path = []
                active_path_index = 0
                active_planning_payload = None
                planned_goal = None
                force_replan = True

            selected_slot = slots[selected_slot_index]

            if selected_slot_id is None:
                selected_slot_id = selected_slot["id"]
            current_pose = payload["start_pose"]
            obstacles = payload.get("obstacles", [])
            parked_slot_id = find_parked_slot_id(current_pose, slots)
            planning_payload_current = set_goal_from_slot(payload.copy(), selected_slot)
            project = make_projector(config, payload, source=source)

            if planner_future is not None and planner_future.done():
                try:
                    candidate_path, candidate_debug_nodes, candidate_payload, planned_frame, reason, candidate_slot_id, planner_time = planner_future.result()

                    planner_debug_nodes = candidate_debug_nodes
                    planner_total += planner_time
                    planner_calls += 1

                    if candidate_slot_id != selected_slot_id:
                        planner_discarded += 1
                        print(
                            f"[Candidate stale] frame={planned_frame} "
                            f"planned_slot={candidate_slot_id} current_slot={selected_slot_id}"
                        )
                    else:
                        planner_debug_nodes = candidate_debug_nodes

                        valid, candidate_index = candidate_is_valid(
                            candidate_path,
                            candidate_payload,
                            selected_slot,
                            current_pose,
                            obstacles,
                        )

                        if valid:
                            active_path = candidate_path
                            active_path_index = candidate_index
                            active_planning_payload = candidate_payload
                            planned_goal = candidate_payload["goal_pose"].copy()
                        else:
                            planner_discarded += 1
                            print(
                                f"[Candidate discarded] frame={planned_frame} "
                                f"reason={reason} time={planner_time * 1000:.1f} ms"
                            )

                except Exception as error:
                    planner_calls += 1
                    planner_discarded += 1
                    print(f"[Planner error] {error}")

                planner_future = None

            path_deviation = float("inf")

            if active_path:
                active_path_index, path_deviation = closest_path_index(active_path, current_pose, active_path_index)

            goal_distance_delta, goal_yaw_delta = goal_difference(selected_slot, planned_goal)
            goal_pose = {
                "x": float(selected_slot["x"]),
                "y": float(selected_slot["y"]),
                "yaw": float(selected_slot.get("yaw", 0.0)),
            }

            distance_to_goal = distance_xy(current_pose, goal_pose)
            remaining_nodes = len(active_path) - active_path_index if active_path else 0

            hard_off_path = bool(active_path) and path_deviation > PATH_REPLAN_DEVIATION
            pre_off_path = bool(active_path) and path_deviation > PATH_PREPLAN_DEVIATION

            hard_goal_changed = planned_goal is not None and (
                goal_distance_delta > GOAL_REPLAN_DISTANCE or goal_yaw_delta > GOAL_REPLAN_YAW
            )

            pre_goal_changed = planned_goal is not None and (
                goal_distance_delta > GOAL_PREPLAN_DISTANCE or goal_yaw_delta > GOAL_PREPLAN_YAW
            )

            hard_blocked = path_blocked(active_path, obstacles, active_path_index, OBSTACLE_PATH_MARGIN)
            pre_blocked = path_blocked(active_path, obstacles, active_path_index, PREPLAN_OBSTACLE_PATH_MARGIN)

            hard_path_end = bool(active_path) and remaining_nodes <= REPLAN_END_NODES and distance_to_goal > GOAL_REACHED_DISTANCE
            pre_path_end = bool(active_path) and remaining_nodes <= PREPLAN_END_NODES and distance_to_goal > GOAL_REACHED_DISTANCE

            cooldown_done = frame_id - last_plan_submit_frame >= REPLAN_COOLDOWN_FRAMES

            plan_reason = None

            if parked_slot_id == selected_slot_id:
                active_path = []
                active_path_index = 0
            elif force_replan:
                plan_reason = "slot"
            elif not active_path and cooldown_done:
                plan_reason = "initial"
            elif hard_off_path:
                plan_reason = "off_path"
            elif hard_goal_changed:
                plan_reason = "goal_changed"
            elif hard_blocked:
                plan_reason = "blocked"
            elif hard_path_end:
                plan_reason = "path_finished"
            elif cooldown_done:
                if pre_off_path:
                    plan_reason = "pre_off_path"
                elif pre_goal_changed:
                    plan_reason = "pre_goal_changed"
                elif pre_blocked:
                    plan_reason = "pre_blocked"
                elif pre_path_end:
                    plan_reason = "pre_path_finished"

            if plan_reason is not None and planner_future is None:
                planner_future = planner_executor.submit(
                    run_planner_async,
                    planning_payload_current,
                    frame_id,
                    plan_reason,
                    selected_slot_id,
                )

                planner_submitted += 1
                replan_reasons[plan_reason] += 1
                last_plan_submit_frame = frame_id

                print(f"[Planner start] frame={frame_id} reason={plan_reason}")

            t_prepare = time.perf_counter() - t
            t_planner = 0.0

            t = time.perf_counter()

            display_frame = prediction_frame.copy() if prediction_frame is not None else frame.copy()
            #draw_parking_mat_on_frame(display_frame, payload)
            if overlay_state["debug"]["drivable_area"]:
                draw_drivable_area_on_frame(display_frame, payload)

            if overlay_state["debug"]["pose_prediction"]:
                draw_pose_predictions(display_frame, payload)

            draw_scene_on_frame(
                display_frame,
                payload,
                slots,
                selected_slot_id,
                project,
                overlay_state,
                parked_slot_id,
            )
            if overlay_state["debug"]["measurements"]:
                draw_debug_measurements_on_frame(
                    display_frame,
                    planning_payload_current,
                    selected_slot_id,
                )
            display_path = trim_path_for_display(active_path, current_pose, active_path_index)

            if display_path and active_planning_payload is not None:
                draw_path_to_goal_on_frame(
                    display_frame,
                    display_path,
                    project,
                    active_planning_payload["goal_pose"],
                    thickness=6,
                )
                if overlay_state["debug"]["planner_nodes"]:
                    draw_planner_nodes(
                        display_frame,
                        planner_debug_nodes,
                        project,
                    )

            canvas = build_display_canvas(
                display_frame,
                slots,
                selected_slot_id,
                overlay_state,
                planner_future,
            )

            t_draw = time.perf_counter() - t

            cv2.imshow(window_name, canvas)

            t_total = time.perf_counter() - t_frame

            timings = {
                "payload": t_payload,
                "prepare": t_prepare,
                "planner": t_planner,
                "draw": t_draw,
                "total": t_total,
            }

            for name, value in timings.items():
                timing_sum[name] += value

            timing_frames += 1
            frame_id += 1

            if timing_frames % 30 == 0:
                values = " | ".join(
                    f"{name}: {timing_sum[name] / timing_frames * 1000:.1f} ms"
                    for name in timings
                )

                planner_avg = planner_total / planner_calls * 1000 if planner_calls else 0.0
                planner_state = "running" if planner_future is not None else "idle"

                print(
                    f"[Timing] {values} | planner={planner_state} | "
                    f"plans: {planner_calls}/{planner_submitted} | "
                    f"discarded: {planner_discarded} | "
                    f"planner/call: {planner_avg:.1f} ms"
                )


    finally:
        if planner_future is not None:
            planner_future.cancel()

        planner_executor.shutdown(wait=False, cancel_futures=True)

        if timing_frames:
            values = " | ".join(
                f"{name}: {timing_sum[name] / timing_frames * 1000:.1f} ms"
                for name in timing_sum
            )

            planner_avg = planner_total / planner_calls * 1000 if planner_calls else 0.0

            print(
                f"[Timing final] {values} | "
                f"plans: {planner_calls}/{planner_submitted} | "
                f"discarded: {planner_discarded} | "
                f"planner/call: {planner_avg:.1f} ms"
            )

            if replan_reasons:
                print(
                    "[Planner reasons] "
                    + " | ".join(
                        f"{reason}: {count}"
                        for reason, count in replan_reasons.most_common()
                    )
                )

        source.release()
        cv2.destroyAllWindows()


def main() -> None:

    parser = argparse.ArgumentParser(description="AV Parking Assistant")
    parser.add_argument(
        "--source",
        choices=["camera", "carla"],
        default=None,
        help="Override the input source configured in the JSON file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "config" / "config.json"),
        help="Path to JSON config file",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="scenario_1",
        help="Named scenario preset from config",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    config = resolve_source_config(config, args.source)

    if args.scenario and config.carla and args.scenario in config.carla.scenarios:
        preset = config.carla.scenarios[args.scenario]
        config = dataclasses.replace(
            config,
            carla=dataclasses.replace(config.carla, scenario=preset, scenario_name=args.scenario),
        )

    source = create_source(config)
    if config.source_type == "carla":
        run_carla(source, config)
    else:
        run(source, config)



if __name__ == "__main__":
    main()