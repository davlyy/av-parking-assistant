from __future__ import annotations
import argparse
import math
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import cv2

from frame_analyze import process_frame, YOLO_MODEL_PATH
from input import load_config, CameraSource, ImageSource, VideoSource, SourceConfig
from mod.path_planning import plan_path
from mod.draw import draw_scene_on_frame, draw_slot_overlay, draw_path_to_goal_on_frame, get_parking_slots, make_projector, make_slot_overlay_mouse_callback, set_goal_from_slot, resize_for_display, draw_drivable_area_on_frame

PATH_PREPLAN_DEVIATION = 0.4
PATH_REPLAN_DEVIATION = 0.6
GOAL_PREPLAN_DISTANCE = 0.3
GOAL_REPLAN_DISTANCE = 0.5
GOAL_PREPLAN_YAW = math.radians(10)
GOAL_REPLAN_YAW = math.radians(15)
GOAL_REACHED_DISTANCE = 0.6
OBSTACLE_PATH_MARGIN = 0.3
PREPLAN_OBSTACLE_PATH_MARGIN = 0.6
PREPLAN_END_NODES = 6
REPLAN_END_NODES = 2
REPLAN_COOLDOWN_FRAMES = 10

def create_source(config: SourceConfig):
    if config.source_type == "camera": return CameraSource(config.camera)
    if config.source_type == "image": return ImageSource(config.image)
    if config.source_type == "video": return VideoSource(config.video)
    raise ValueError(f"Unknown source_type: {config.source_type}")

def ensure_bgr(frame):
    if frame is not None and len(frame.shape) == 3 and frame.shape[2] == 4: return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame

def distance_xy(a: dict, b: dict) -> float:
    return math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))

def angle_distance(a: float, b: float) -> float:
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)

def slot_distance(a: dict, b: dict) -> float:
    return math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))

def goal_difference(slot: dict, goal: dict | None):
    if goal is None: return float("inf"), float("inf")
    return slot_distance(slot, goal), angle_distance(float(slot.get("yaw", 0.0)), float(goal.get("yaw", 0.0)))

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
            local_x, local_y = c * dx + s * dy, -s * dx + c * dy
            half_l = float(obs["length"]) / 2 + margin
            half_w = float(obs["width"]) / 2 + margin
            if abs(local_x) <= half_l and abs(local_y) <= half_w: return True
    return False

def run_planner_async(payload, frame_id, reason):
    t = time.perf_counter()
    path = plan_path(payload, already_world=True)
    return path, payload, frame_id, reason, time.perf_counter() - t

def candidate_is_valid(candidate_path, candidate_payload, selected_slot, current_pose, obstacles):
    if not candidate_path: return False, 0
    goal_distance, goal_yaw = goal_difference(selected_slot, candidate_payload["goal_pose"])
    if goal_distance > GOAL_REPLAN_DISTANCE or goal_yaw > GOAL_REPLAN_YAW: return False, 0
    candidate_index, deviation = closest_path_index(candidate_path, current_pose)
    if deviation > PATH_REPLAN_DEVIATION: return False, candidate_index
    if path_blocked(candidate_path, obstacles, candidate_index, OBSTACLE_PATH_MARGIN): return False, candidate_index
    return True, candidate_index

def run(source, config: SourceConfig) -> None:
    source.open()
    selected_slot_index = 0
    selected_slot_anchor = None
    active_path = []
    active_path_index = 0
    active_planning_payload = None
    planned_goal = None

    planner_executor = ProcessPoolExecutor(max_workers=1)
    planner_future = None
    last_plan_submit_frame = -REPLAN_COOLDOWN_FRAMES
    planner_total = 0.0
    planner_calls = 0
    planner_submitted = 0
    planner_discarded = 0
    replan_reasons = Counter()

    overlay_state = {"buttons": [], "slot_polygons": [], "clicked_index": None, "display_scale": 1.0}
    timing_sum = {"payload": 0.0, "prepare": 0.0, "planner": 0.0, "draw": 0.0, "total": 0.0}
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
            _, payload = process_frame(frame, frame_id=frame_id, detector="yolo", yolo_model_path=YOLO_MODEL_PATH, confidence=0.2, aruco_marker_size_mm=0.98 * 24, visualize=False)
            t_payload = time.perf_counter() - t

            if payload is None:
                cv2.imshow(window_name, resize_for_display(frame)[0])
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27): break
                frame_id += 1
                continue

            t = time.perf_counter()
            slots = get_parking_slots(payload)
            if not slots:
                cv2.imshow(window_name, resize_for_display(frame)[0])
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27): break
                frame_id += 1
                continue

            if selected_slot_anchor is not None:
                selected_slot_index = min(range(len(slots)), key=lambda i: slot_distance(slots[i], selected_slot_anchor))
            else:
                selected_slot_index = min(selected_slot_index, len(slots) - 1)

            force_replan = False
            if overlay_state["clicked_index"] is not None:
                clicked = overlay_state["clicked_index"]
                overlay_state["clicked_index"] = None
                if 0 <= clicked < len(slots):
                    selected_slot_index = clicked
                    selected_slot_anchor = slots[selected_slot_index].copy()
                    force_replan = True

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27): break

            old_index = selected_slot_index
            if key == ord("w"): selected_slot_index = max(0, selected_slot_index - 1)
            elif key == ord("s"): selected_slot_index = min(len(slots) - 1, selected_slot_index + 1)
            elif ord("0") <= key <= ord("9") and int(chr(key)) < len(slots): selected_slot_index = int(chr(key))

            if selected_slot_index != old_index:
                selected_slot_anchor = slots[selected_slot_index].copy()
                force_replan = True

            selected_slot = slots[selected_slot_index]
            selected_slot_anchor = selected_slot.copy()
            current_pose = payload["start_pose"]
            obstacles = payload.get("obstacles", [])
            planning_payload_current = set_goal_from_slot(payload.copy(), selected_slot)
            project = make_projector(config, payload, source=source)

            if planner_future is not None and planner_future.done():
                try:
                    candidate_path, candidate_payload, planned_frame, reason, planner_time = planner_future.result()
                    planner_total += planner_time
                    planner_calls += 1
                    valid, candidate_index = candidate_is_valid(candidate_path, candidate_payload, selected_slot, current_pose, obstacles)

                    if valid:
                        active_path = candidate_path
                        active_path_index = candidate_index
                        active_planning_payload = candidate_payload
                        planned_goal = candidate_payload["goal_pose"].copy()
                        print(f"[Candidate accepted] frame={planned_frame} reason={reason} time={planner_time * 1000:.1f} ms")
                    else:
                        planner_discarded += 1
                        print(f"[Candidate discarded] frame={planned_frame} reason={reason} time={planner_time * 1000:.1f} ms")
                except Exception as error:
                    planner_calls += 1
                    planner_discarded += 1
                    print(f"[Planner error] {error}")
                planner_future = None

            path_deviation = float("inf")
            if active_path:
                active_path_index, path_deviation = closest_path_index(active_path, current_pose, active_path_index)

            goal_distance_delta, goal_yaw_delta = goal_difference(selected_slot, planned_goal)
            goal_pose = {"x": float(selected_slot["x"]), "y": float(selected_slot["y"]), "yaw": float(selected_slot.get("yaw", 0.0))}
            distance_to_goal = distance_xy(current_pose, goal_pose)
            remaining_nodes = len(active_path) - active_path_index if active_path else 0

            hard_off_path = bool(active_path) and path_deviation > PATH_REPLAN_DEVIATION
            pre_off_path = bool(active_path) and path_deviation > PATH_PREPLAN_DEVIATION
            hard_goal_changed = planned_goal is not None and (goal_distance_delta > GOAL_REPLAN_DISTANCE or goal_yaw_delta > GOAL_REPLAN_YAW)
            pre_goal_changed = planned_goal is not None and (goal_distance_delta > GOAL_PREPLAN_DISTANCE or goal_yaw_delta > GOAL_PREPLAN_YAW)
            hard_blocked = path_blocked(active_path, obstacles, active_path_index, OBSTACLE_PATH_MARGIN)
            pre_blocked = path_blocked(active_path, obstacles, active_path_index, PREPLAN_OBSTACLE_PATH_MARGIN)
            hard_path_end = bool(active_path) and remaining_nodes <= REPLAN_END_NODES and distance_to_goal > GOAL_REACHED_DISTANCE
            pre_path_end = bool(active_path) and remaining_nodes <= PREPLAN_END_NODES and distance_to_goal > GOAL_REACHED_DISTANCE
            cooldown_done = frame_id - last_plan_submit_frame >= REPLAN_COOLDOWN_FRAMES

            plan_reason = None
            if force_replan: plan_reason = "slot"
            elif not active_path: plan_reason = "initial"
            elif hard_off_path: plan_reason = "off_path"
            elif hard_goal_changed: plan_reason = "goal_changed"
            elif hard_blocked: plan_reason = "blocked"
            elif hard_path_end: plan_reason = "path_finished"
            elif cooldown_done:
                if pre_off_path: plan_reason = "pre_off_path"
                elif pre_goal_changed: plan_reason = "pre_goal_changed"
                elif pre_blocked: plan_reason = "pre_blocked"
                elif pre_path_end: plan_reason = "pre_path_finished"

            if plan_reason is not None and planner_future is None:
                planner_future = planner_executor.submit(run_planner_async, planning_payload_current, frame_id, plan_reason)
                planner_submitted += 1
                replan_reasons[plan_reason] += 1
                last_plan_submit_frame = frame_id
                print(f"[Planner start] frame={frame_id} reason={plan_reason}")

            t_prepare = time.perf_counter() - t
            t_planner = 0.0

            t = time.perf_counter()
            display_frame = frame.copy()
            draw_drivable_area_on_frame(display_frame, payload)
            _, scale = resize_for_display(display_frame, max_w=1400, max_h=950)
            overlay_state["display_scale"] = scale
            draw_scene_on_frame(display_frame, payload, slots, selected_slot_index, project, overlay_state)

            display_path = trim_path_for_display(active_path, current_pose, active_path_index)
            if display_path and active_planning_payload is not None:
                draw_path_to_goal_on_frame(display_frame, display_path, project, active_planning_payload["goal_pose"], thickness=2)

            draw_slot_overlay(display_frame, slots, selected_slot_index, overlay_state)
            display_frame, scale = resize_for_display(display_frame, max_w=1400, max_h=950)
            overlay_state["display_scale"] = scale
            t_draw = time.perf_counter() - t

            cv2.imshow(window_name, display_frame)
            t_total = time.perf_counter() - t_frame
            timings = {"payload": t_payload, "prepare": t_prepare, "planner": t_planner, "draw": t_draw, "total": t_total}

            for name, value in timings.items(): timing_sum[name] += value
            timing_frames += 1
            frame_id += 1

            if timing_frames % 30 == 0:
                values = " | ".join(f"{name}: {timing_sum[name] / timing_frames * 1000:.1f} ms" for name in timings)
                planner_avg = planner_total / planner_calls * 1000 if planner_calls else 0.0
                planner_state = "running" if planner_future is not None else "idle"
                print(f"[Timing] {values} | planner={planner_state} | plans: {planner_calls}/{planner_submitted} | discarded: {planner_discarded} | planner/call: {planner_avg:.1f} ms")

    finally:
        if planner_future is not None: planner_future.cancel()
        planner_executor.shutdown(wait=False, cancel_futures=True)

        if timing_frames:
            values = " | ".join(f"{name}: {timing_sum[name] / timing_frames * 1000:.1f} ms" for name in timing_sum)
            planner_avg = planner_total / planner_calls * 1000 if planner_calls else 0.0
            print(f"[Timing final] {values} | plans: {planner_calls}/{planner_submitted} | discarded: {planner_discarded} | planner/call: {planner_avg:.1f} ms")
            if replan_reasons:
                print("[Planner reasons] " + " | ".join(f"{reason}: {count}" for reason, count in replan_reasons.most_common()))

        source.release()
        cv2.destroyAllWindows()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(Path(__file__).resolve().parent.parent / "config" / "config.json"))
    args = parser.parse_args()
    config = load_config(args.config)
    run(create_source(config), config)

if __name__ == "__main__":
    main()