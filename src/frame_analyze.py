import argparse
import cv2
import numpy as np
from pathlib import Path
import os
import json

from inference_sdk import InferenceHTTPClient

# Constants
ROBOFLOW_API_KEY  = os.environ.get("ROBOFLOW_API_KEY")
ROBOFLOW_MODEL_ID = "parking-lot-npjkj/2"

# Vehicle spec (Lincoln MKZ 2017) — ground truth for calibration
VEHICLE_SPEC = {
    "wheelbase":  2.850,
    "length":     4.980,   # meters
    "width":      1.900,   # meters
    "max_steer":  0.44157
}

# Runtime calibration
# Set once from ego vehicle detection
# Two independent ratios (length and width axis) are averaged for robustness
PIXELS_PER_METER: float = None  # set by calibrate_from_ego()

#Roboflow Client
roboflow_client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=ROBOFLOW_API_KEY
)

#Calibration
def calibrate_from_ego(ego: dict) -> float:
    """
    Derive PIXELS_PER_METER from the ego vehicle detection bbox.

    The Roboflow bbox gives us (width_px, height_px) of the detected car.
    We know the real dimensions from VEHICLE_SPEC.

    In a BEV image the car's longer bbox axis = vehicle length,
    shorter axis = vehicle width. We compute both ratios and average
    them to reduce bbox detection noise.

        px_per_m_from_length = longer_px  / real_length_m
        px_per_m_from_width  = shorter_px / real_width_m
        PIXELS_PER_METER     = mean(above two)

    Returns:
        float: calibrated pixels-per-meter value
    """
    w_px = ego['width']
    h_px = ego['height']

    longer_px  = max(w_px, h_px)
    shorter_px = min(w_px, h_px)

    real_length = VEHICLE_SPEC['length']  # 4.980 m
    real_width  = VEHICLE_SPEC['width']   # 1.900 m

    px_per_m_length = longer_px  / real_length
    px_per_m_width  = shorter_px / real_width

    px_per_m = (px_per_m_length + px_per_m_width) / 2.0

    print(f"Ego bbox:       {w_px:.1f} x {h_px:.1f} px")
    print(f"Real vehicle:   {real_length} x {real_width} m")
    print(f"px/m (length):  {px_per_m_length:.4f}")
    print(f"px/m (width):   {px_per_m_width:.4f}")
    print(f"px/m (average): {px_per_m:.4f}  ← using this")

    # Sanity check: the two axes should roughly agree
    discrepancy = abs(px_per_m_length - px_per_m_width) / px_per_m * 100
    if discrepancy > 20.0:
        print(f"WARNING: {discrepancy:.1f}% discrepancy between axes "
              f"— image may not be perfectly top-down, or bbox is noisy")

    return px_per_m

#Helpers
def px_to_metric(x_px, y_px):
    assert PIXELS_PER_METER is not None, "Call calibrate_from_ego() first"
    return round(x_px / PIXELS_PER_METER, 3), \
           round(y_px / PIXELS_PER_METER, 3)

def size_to_metric(w_px, h_px):
    assert PIXELS_PER_METER is not None, "Call calibrate_from_ego() first"
    return round(w_px / PIXELS_PER_METER, 3), \
           round(h_px / PIXELS_PER_METER, 3)

def estimate_yaw(pred):
    """Estimate yaw from slot/car orientation. Vertical slot = pi/2."""
    return round(np.pi / 2, 4) if pred['height'] > pred['width'] else 0.0

def dist(a, b):
    return np.hypot(a['x'] - b['x'], a['y'] - b['y'])

#[Stage 0] Camera Calibration / IPM Transform
#TODO: Implement Camera Calibration

#[Stage 1] Preprocessing
def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    denoised = cv2.bilateralFilter(gray_eq, d=24, sigmaColor=40, sigmaSpace=100)
    return denoised, cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#Stage 2: Lane Marking Detection
def detect_slot_lines(gray, bev_img):
    _, white_mask = cv2.threshold(gray, 185, 255, cv2.THRESH_BINARY)

    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    horiz_lines = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel_h)
    vert_lines  = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel_v)
    line_mask   = cv2.bitwise_or(horiz_lines, vert_lines)

    edges = cv2.Canny(line_mask, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1, theta=np.pi / 180,
        threshold=30,
        minLineLength=40,
        maxLineGap=80
    )
    return lines

#[Stage 2] helpers
def merge_lines(lines, axis='y', dist_thresh=15, gap_thresh=40):
    if len(lines) < 2:
        return [list(l) for l in lines]
    pool = [list(l) for l in lines]
    merged = True
    while merged:
        merged = False
        i = 0
        while i < len(pool):
            j = i + 1
            while j < len(pool):
                l1, l2 = pool[i], pool[j]
                if axis == 'y':
                    y1 = (l1[1] + l1[3]) / 2
                    y2 = (l2[1] + l2[3]) / 2
                    if abs(y1 - y2) > dist_thresh:
                        j += 1; continue
                    x1_min, x1_max = min(l1[0], l1[2]), max(l1[0], l1[2])
                    x2_min, x2_max = min(l2[0], l2[2]), max(l2[0], l2[2])
                    gap = max(0, x2_min - x1_max, x1_min - x2_max)
                    if gap < gap_thresh:
                        l1[0] = min(x1_min, x2_min)
                        l1[2] = max(x1_max, x2_max)
                        l1[1] = l1[3] = (y1 + y2) / 2
                        pool.pop(j); merged = True; continue
                else:
                    x1 = (l1[0] + l1[2]) / 2
                    x2 = (l2[0] + l2[2]) / 2
                    if abs(x1 - x2) > dist_thresh:
                        j += 1; continue
                    y1_min, y1_max = min(l1[1], l1[3]), max(l1[1], l1[3])
                    y2_min, y2_max = min(l2[1], l2[3]), max(l2[1], l2[3])
                    gap = max(0, y2_min - y1_max, y1_min - y2_max)
                    if gap < gap_thresh:
                        l1[1] = min(y1_min, y2_min)
                        l1[3] = max(y1_max, y2_max)
                        l1[0] = l1[2] = (x1 + x2) / 2
                        pool.pop(j); merged = True; continue
                j += 1
            i += 1
    return pool

def segment_intersection(l1, l2):
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-6: return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den
    tol = 0.05
    if -tol <= t <= 1 + tol and -tol <= u <= 1 + tol:
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
    return None

def cluster_lines_to_slots(lines, merge_dist=30, size_threshold=2.0):
    from sklearn.cluster import DBSCAN
    if lines is None:
        return [], [], [], [], []

    horizontal, vertical = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        if angle < 20 or angle > 160:
            horizontal.append(line[0])
        elif 70 < angle < 110:
            vertical.append(line[0])

    merged_h = merge_lines(horizontal, axis='y')
    merged_v = merge_lines(vertical,   axis='x')
    print(f"[Debug] Lines: {len(merged_h)} horiz, {len(merged_v)} vert")

    raw_points = []
    for l in merged_h + merged_v:
        raw_points.append((l[0], l[1]))
        raw_points.append((l[2], l[3]))
    for h in merged_h:
        for v in merged_v:
            pt = segment_intersection(h, v)
            if pt:
                raw_points.append(pt)

    if not raw_points:
        return merged_h, merged_v, [], [], []

    pts = np.array(raw_points)
    db  = DBSCAN(eps=merge_dist, min_samples=1).fit(pts)
    corners = []
    for label in set(db.labels_):
        cluster_pts = pts[db.labels_ == label]
        corners.append((np.mean(cluster_pts[:, 0]), np.mean(cluster_pts[:, 1])))

    print(f"[Debug] Merged into {len(corners)} corners")

    slot_rects = []
    corners.sort(key=lambda p: (p[1], p[0]))
    composite_count = 0

    for i, tl in enumerate(corners):
        for j in range(i + 1, len(corners)):
            tr = corners[j]
            if abs(tl[1] - tr[1]) > merge_dist * 1.5: continue
            if tl[0] >= tr[0]: continue
            width = tr[0] - tl[0]
            if width < 30: continue
            for k in range(j + 1, len(corners)):
                bl = corners[k]
                if abs(bl[0] - tl[0]) > merge_dist * 1.5: continue
                if bl[1] <= tl[1]: continue
                height = bl[1] - tl[1]
                if height < 30: continue
                for m in range(k + 1, len(corners)):
                    br = corners[m]
                    if abs(br[0] - tr[0]) <= merge_dist * 1.5 and \
                       abs(br[1] - bl[1]) <= merge_dist * 1.5:
                        x_min, x_max = min(tl[0], br[0]), max(tl[0], br[0])
                        y_min, y_max = min(tl[1], br[1]), max(tl[1], br[1])
                        is_composite = False
                        for c in corners:
                            if (abs(c[0]-tl[0])<1 and abs(c[1]-tl[1])<1) or \
                               (abs(c[0]-tr[0])<1 and abs(c[1]-tr[1])<1) or \
                               (abs(c[0]-bl[0])<1 and abs(c[1]-bl[1])<1) or \
                               (abs(c[0]-br[0])<1 and abs(c[1]-br[1])<1):
                                continue
                            if (x_min-5 <= c[0] <= x_max+5) and \
                               (y_min-5 <= c[1] <= y_max+5):
                                is_composite = True; break
                        if not is_composite:
                            slot_rects.append(((tl[0], tl[1]), (br[0], br[1])))
                        else:
                            composite_count += 1

    print(f"[Debug] Found {len(slot_rects)} raw rectangles")

    if not slot_rects:
        return merged_h, merged_v, [], [], corners

    areas = [abs(s[1][0]-s[0][0]) * abs(s[1][1]-s[0][1]) for s in slot_rects]
    clean_areas = [a for a in areas if a > 2500]
    if not clean_areas:
        return merged_h, merged_v, [], slot_rects, corners

    median_area = np.median(clean_areas)
    slot_props  = []
    for s in slot_rects:
        w = abs(s[1][0] - s[0][0])
        h = abs(s[1][1] - s[0][1])
        area   = w * h
        aspect = max(w, h) / (min(w, h) + 1e-5)
        slot_props.append({'slot': s, 'area': area, 'aspect': aspect})

    area_candidates = [p for p in slot_props
                       if abs(p['area'] - median_area) / median_area <= size_threshold]
    if not area_candidates:
        return merged_h, merged_v, [], slot_rects, corners

    median_aspect   = np.median([p['aspect'] for p in area_candidates])
    aspect_threshold = 0.15

    valid_slots, rejected_slots = [], []
    for p in slot_props:
        area_ok   = abs(p['area']   - median_area)   / median_area   <= size_threshold
        aspect_ok = abs(p['aspect'] - median_aspect) / median_aspect <= aspect_threshold
        if area_ok and aspect_ok:
            valid_slots.append(p['slot'])
        else:
            rejected_slots.append(p['slot'])

    print(f"[Debug] Kept {len(valid_slots)} slots, Rejected {len(rejected_slots)}")
    return merged_h, merged_v, valid_slots, rejected_slots, corners

#[Stage 3]: Vehicle + Slot Detection via Roboflow
def detect_vehicles(frame):
    """
    Returns:
        cars       – list of occupied car predictions
        avail      – list of available slot predictions
        raw_preds  – all predictions (for visualization)
    """
    api_result = roboflow_client.infer(frame, model_id=ROBOFLOW_MODEL_ID)

    # api_result is a dict with key 'predictions'
    predictions = api_result.get('predictions', [])
    print(f"[Debug] Roboflow raw predictions: {len(predictions)}")

    cars  = []
    avail = []

    for pred in predictions:
        entry = {
            'x':          pred['x'],
            'y':          pred['y'],
            'width':      pred['width'],
            'height':     pred['height'],
            'conf':       pred['confidence'],
            'class':      pred['class'],
            # Derived fields for downstream use
            'center_px':  (pred['x'], pred['y']),
            'size_px':    (pred['width'], pred['height']),
            'bbox_px': (
                int(pred['x'] - pred['width']  / 2),
                int(pred['y'] - pred['height'] / 2),
                int(pred['x'] + pred['width']  / 2),
                int(pred['y'] + pred['height'] / 2),
            )
        }

        cls = pred['class'].lower()
        if cls == 'cars':
            cars.append(entry)
        elif cls == 'avail':
            avail.append(entry)

    print(f"[Debug] Cars: {len(cars)}, Available slots: {len(avail)}")
    return cars, avail, predictions

#[Stage 4]: Identify Ego Vehicle
def identify_ego(cars, avail):
    """
    Ego vehicle = the car driving in the aisle.
    Heuristic: car furthest from any available slot center.
    """
    if not cars:
        return None, []

    if not avail:
        # No slots detected — pick car closest to image center as ego
        return cars[0], cars[1:]

    def dist_to_nearest_slot(car):
        return min(dist(car, s) for s in avail)

    ego       = max(cars, key=dist_to_nearest_slot)
    obstacles = [c for c in cars if c is not ego]
    return ego, obstacles

#[Stage 5]: Build A* JSON Payload
def build_astar_payload(ego, obstacles, avail):
    """
    Converts pixel-space detections to metric A* payload.
    Target slot = available slot closest to ego vehicle.
    """
    if ego is None or not avail:
        print("[Warn] Cannot build payload: missing ego or available slots")
        return None

    target_slot = min(avail, key=lambda s: dist(s, ego))

    ego_x,  ego_y  = px_to_metric(ego['x'], ego['y'])
    goal_x, goal_y = px_to_metric(target_slot['x'], target_slot['y'])
    slot_w, slot_h = size_to_metric(target_slot['width'], target_slot['height'])

    payload = {
        "start_pose": {
            "x":   ego_x,
            "y":   ego_y,
            "yaw": estimate_yaw(ego)
        },
        "goal_pose": {
            "x":   goal_x,
            "y":   goal_y,
            "yaw": estimate_yaw(target_slot)
        },
        "obstacles": [
            {
                "x":      round(o['x'] / PIXELS_PER_METER, 3),
                "y":      round(o['y'] / PIXELS_PER_METER, 3),
                "length": round(max(o['width'], o['height']) / PIXELS_PER_METER, 3),
                "width":  round(min(o['width'], o['height']) / PIXELS_PER_METER, 3),
                "yaw":    estimate_yaw(o)
            }
            for o in obstacles
        ],
        "parking_slot": {
            "x":      goal_x,
            "y":      goal_y,
            "yaw":    estimate_yaw(target_slot),
            "length": round(max(slot_w, slot_h), 3),
            "width":  round(min(slot_w, slot_h), 3)
        },
        "vehicle": VEHICLE_SPEC
    }

    print("\nA* Payload")
    print(json.dumps(payload, indent=2))
    return payload

#Visualization
def draw_detections(img, lines, slots, rejected_slots, corners,
                    cars, avail, ego, obstacles, merged_h=None, merged_v=None):
    vis = img.copy()

    # Corners (magenta)
    for i, (cx, cy) in enumerate(corners):
        cv2.circle(vis, (int(cx), int(cy)), 5, (255, 0, 255), -1)
        cv2.putText(vis, f"P{i}", (int(cx)+5, int(cy)+15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)

    # Valid slots (unique colors)
    if slots:
        rng    = np.random.default_rng(42)
        colors = rng.integers(50, 255, size=(len(slots), 3)).tolist()
        for i, (tl, br) in enumerate(slots):
            color = tuple(colors[i])
            cv2.rectangle(vis, (int(tl[0]), int(tl[1])),
                               (int(br[0]), int(br[1])), color, 2)
            cv2.putText(vis, f"S{i+1}", (int(tl[0]), max(15, int(tl[1])-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # Available slots from Roboflow (green)
    for s in avail:
        x1, y1, x2, y2 = s['bbox_px']
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, f"FREE {s['conf']:.2f}", (x1, y1-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # Obstacle cars (red)
    for o in obstacles:
        x1, y1, x2, y2 = o['bbox_px']
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(vis, f"OBS {o['conf']:.2f}", (x1, y1-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # Ego vehicle (orange)
    if ego:
        x1, y1, x2, y2 = ego['bbox_px']
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 165, 255), 3)
        cv2.putText(vis, f"EGO {ego['conf']:.2f}", (x1, y1-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

        # Arrow from ego to nearest available slot
        if avail:
            from operator import itemgetter
            target = min(avail, key=lambda s: dist(s, ego))
            cx_ego = int(ego['x'])
            cy_ego = int(ego['y'])
            cx_tgt = int(target['x'])
            cy_tgt = int(target['y'])
            cv2.arrowedLine(vis, (cx_ego, cy_ego), (cx_tgt, cy_tgt),
                            (0, 255, 255), 2, tipLength=0.15)
            cv2.circle(vis, (cx_tgt, cy_tgt), 6, (0, 255, 255), -1)

    return vis

def process_frame(frame):
    global PIXELS_PER_METER

    # Stage 1: Preprocess
    gray, hsv = preprocess(frame)

    # Stage 2: Lane line detection → slot geometry
    lines = detect_slot_lines(gray, frame)
    merged_h, merged_v, slots, rejected_slots, corners = \
        cluster_lines_to_slots(lines)

    # Stage 3: Roboflow detection → cars + available slots
    cars, avail, raw_preds = detect_vehicles(frame)

    # Stage 4: Identify ego vs obstacles
    ego, obstacles = identify_ego(cars, avail)
    print(f"[Debug] Ego: {ego['class'] if ego else 'None'} | "
          f"Obstacles: {len(obstacles)} | Available: {len(avail)}")

    # Stage 4b: Calibrate px/m from ego vehicle bbox
    if ego is not None:
        PIXELS_PER_METER = calibrate_from_ego(ego)
    else:
        print("ERROR: No ego vehicle detected — cannot calibrate.")
        print("Falling back to default 17.6 px/m (unreliable)")
        PIXELS_PER_METER = 17.6

    # Stage 5: Build A* payload
    payload = build_astar_payload(ego, obstacles, avail)

    # Visualization
    result = draw_detections(
        frame, lines, slots, rejected_slots, corners,
        cars, avail, ego, obstacles, merged_h, merged_v
    )

    return result, payload

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input",  type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument("--payload", type=str, default="payload.json",
                        help="Path to save A* JSON payload")
    args = parser.parse_args()

    frame = cv2.imread(args.input)
    if frame is None:
        raise FileNotFoundError(f"Cannot read image: {args.input}")

    result, payload = process_frame(frame)

    # Save visualization
    cv2.imwrite(args.output, result)
    print(f"\nDetection result saved to: {args.output}")

    # Save payload
    if payload:
        with open(args.payload, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f"A* payload saved to: {args.payload}")