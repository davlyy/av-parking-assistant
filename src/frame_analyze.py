import argparse
import cv2
import numpy as np
from pathlib import Path
import os
import json

from inference_sdk import InferenceConfiguration, InferenceHTTPClient
from dotenv import load_dotenv

load_dotenv()

# Constants
ROBOFLOW_API_KEY  = os.environ.get("ROBOFLOW_API_KEY")
ROBOFLOW_MODEL_ID = "parking-lot-npjkj/2"
YOLO_MODEL_PATH = "model.pt"
_yolo_model = None
_yolo_model_path = None

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

#[Stage 0] Parking-Lot Detection and Crop
def _order_quad_points(points):
    """Order quadrilateral vertices as top-left, top-right, bottom-right, bottom-left."""
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    ordered = np.empty((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).ravel()
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(diffs)]
    ordered[3] = points[np.argmax(diffs)]
    return ordered


def find_parking_lot_corners(frame):
    """Find the four corners of the largest dark parking-lot surface."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    parking_lot_mask = cv2.inRange(hsv, (0, 0, 0), (180, 105, 145))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    # Join the dark surface across white bay markings and parked cars.
    parking_lot_mask = cv2.morphologyEx(parking_lot_mask, cv2.MORPH_CLOSE, kernel)
    parking_lot_mask = cv2.morphologyEx(parking_lot_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(parking_lot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Could not locate the dark parking lot in the frame")

    parking_lot_contour = max(contours, key=cv2.contourArea)
    min_area = frame.shape[0] * frame.shape[1] * 0.12
    if cv2.contourArea(parking_lot_contour) < min_area:
        raise ValueError("Detected parking lot is too small; ensure the complete lot is visible")
    return _order_quad_points(cv2.boxPoints(cv2.minAreaRect(parking_lot_contour)))


def detect_and_crop_parking_mat(frame):
    """Crop to the detected parking-lot rectangle without perspective warping.

    The returned image is an axis-aligned crop of the four parking-lot-corner
    bounds. The quadrilateral is detected, but no
    ``getPerspectiveTransform`` or ``warpPerspective`` is applied here.
    """
    corners = find_parking_lot_corners(frame)
    x1, y1 = np.floor(corners.min(axis=0)).astype(int)
    x2, y2 = np.ceil(corners.max(axis=0)).astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    return frame[y1:y2, x1:x2].copy()

#[Stage 1] Preprocessing
def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    denoised = cv2.bilateralFilter(gray_eq, d=24, sigmaColor=40, sigmaSpace=100)
    return denoised, cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def reduce_specular_highlights(frame, value_cap=220, saturation_limit=100):
    """Cap bright, low-saturation glare while preserving normal car colours.

    Glossy reflections are usually close to white: high V and low S in HSV.
    Only those pixels are capped, avoiding a global brightness reduction that
    would weaken the white parking lines and red car body.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    glare = (value > value_cap) & (saturation < saturation_limit)
    value[glare] = value_cap
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def prepare_model_input(frame):
    """Suppress glare and apply 9x9 colour-preserving Gaussian smoothing."""
    glare_reduced = reduce_specular_highlights(frame)
    return cv2.GaussianBlur(glare_reduced, (9, 9), 0)

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

    print("**"*60)
    print(predictions)
    print("**" * 60)

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


def detect_vehicles_yolo(frame, model_path=YOLO_MODEL_PATH, confidence=0.1):
    """Detect ``cars`` and ``free``/``avail`` slots using a local YOLO model.

    Returns the same ``cars, avail, raw_predictions`` structure as
    ``detect_vehicles`` so downstream ego, occupancy, and payload logic is
    unchanged. YOLO class labels ``car``/``cars`` map to cars and
    ``free``/``avail`` map to available slots.
    """
    global _yolo_model, _yolo_model_path

    model_path = str(Path(model_path))
    if not Path(model_path).is_file():
        raise FileNotFoundError(f"YOLO model not found: {model_path}")

    if _yolo_model is None or _yolo_model_path != model_path:
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                "YOLO detection requires ultralytics. Run: pip install ultralytics"
            ) from error
        _yolo_model = YOLO(model_path)
        _yolo_model_path = model_path

    result = _yolo_model.predict(frame, conf=confidence, verbose=False)[0]
    names = result.names
    cars, avail, predictions = [], [], []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
        class_id = int(box.cls[0].item())
        class_name = str(names[class_id])
        conf = float(box.conf[0].item())
        width, height = x2 - x1, y2 - y1
        prediction = {
            "x": (x1 + x2) / 2,
            "y": (y1 + y2) / 2,
            "width": width,
            "height": height,
            "confidence": conf,
            "class": class_name,
            "class_id": class_id,
        }
        predictions.append(prediction)
        entry = {
            "x": prediction["x"], "y": prediction["y"],
            "width": width, "height": height,
            "conf": conf, "class": class_name,
            "center_px": (prediction["x"], prediction["y"]),
            "size_px": (width, height),
            "bbox_px": (int(x1), int(y1), int(x2), int(y2)),
        }
        label = class_name.lower()
        if label in {"car", "cars"}:
            cars.append(entry)
        elif label in {"free", "avail"}:
            avail.append(entry)

    print(f"[Debug] YOLO raw predictions: {len(predictions)}")
    print(f"[Debug] YOLO Cars: {len(cars)}, Available slots: {len(avail)}")
    return cars, avail, predictions


def run_vehicle_detector(frame, detector, yolo_model_path, confidence):
    """Run the selected perception backend using a common output format."""
    if detector == "yolo":
        return detect_vehicles_yolo(frame, yolo_model_path, confidence)
    return detect_vehicles(frame)

#[Stage 4]: Identify Ego Vehicle
def identify_ego(cars, frame):
    """
    Ego vehicle = the detected car with the strongest average red colour.

    For each Roboflow car bounding box, calculate the mean BGR pixel colour
    and score red dominance as R / (B + G + 1). The red toy car therefore has
    the highest score even when it is not the furthest car from a free slot.
    """
    if not cars:
        return None, []

    def redness_score(car):
        x1, y1, x2, y2 = car['bbox_px']
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        b, g, r, _ = cv2.mean(frame[y1:y2, x1:x2])
        return r / (b + g + 1.0)

    ego       = max(cars, key=redness_score)
    obstacles = [c for c in cars if c is not ego]
    return ego, obstacles


def remove_slots_containing_ego(avail, ego):
    """Remove free-slot detections that fully contain the red ego car box."""
    if ego is None:
        return avail

    car_x1, car_y1, car_x2, car_y2 = ego['bbox_px']
    free_slots = []
    occupied_slots = 0
    for slot in avail:
        slot_x1, slot_y1, slot_x2, slot_y2 = slot['bbox_px']
        contains_entire_car = (
            slot_x1 <= car_x1 and slot_y1 <= car_y1 and
            slot_x2 >= car_x2 and slot_y2 >= car_y2
        )
        if contains_entire_car:
            occupied_slots += 1
        else:
            free_slots.append(slot)

    if occupied_slots:
        print(f"[Debug] Removed {occupied_slots} free-slot prediction(s) containing the ego car")
    return free_slots


def filter_slots_by_median_area(avail, tolerance=0.30):
    """Reject anomalously sized free-slot boxes when enough slots are present."""
    if len(avail) <= 3:
        return avail

    areas = np.array([slot['width'] * slot['height'] for slot in avail])
    median_area = float(np.median(areas))
    min_area = median_area * (1.0 - tolerance)
    max_area = median_area * (1.0 + tolerance)
    filtered = [
        slot for slot, area in zip(avail, areas)
        if min_area <= area <= max_area
    ]
    removed = len(avail) - len(filtered)
    if removed:
        print(f"[Debug] Removed {removed} free-slot prediction(s) outside ±{tolerance:.0%} of median area")
    return filtered

#[Stage 5]: Build A* JSON Payload
def build_astar_payload(ego, obstacles, avail, frame_id=0, image_width=None, image_height=None):
    """
    Converts pixel-space detections to metric A* payload.
    Target slot = available slot closest to ego vehicle.
    """
    if ego is None or not avail:
        print("[Warn] Cannot build payload: missing ego or available slots")
        return None

    # The first entry remains the A* goal; retain every other free-space
    # detection for callers that want to choose an alternative goal.
    ordered_slots = sorted(avail, key=lambda s: dist(s, ego))
    target_slot = ordered_slots[0]

    def slot_to_payload(slot):
        slot_x, slot_y = px_to_metric(slot['x'], slot['y'])
        slot_w, slot_h = size_to_metric(slot['width'], slot['height'])
        return {
            "x":      slot_x,
            "y":      slot_y,
            "yaw":    estimate_yaw(slot),
            "length": round(max(slot_w, slot_h), 3),
            "width":  round(min(slot_w, slot_h), 3)
        }

    ego_x,  ego_y  = px_to_metric(ego['x'], ego['y'])
    goal_x, goal_y = px_to_metric(target_slot['x'], target_slot['y'])

    image_width = image_width if image_width is not None else 0
    image_height = image_height if image_height is not None else 0
    # World coordinates in this payload are metres derived from x_px / ppm and
    # y_px / ppm. This is the corresponding metric-to-image homography for the
    # current cropped image coordinate system.
    world_to_image = [
        [round(PIXELS_PER_METER, 6), 0.0, 0.0],
        [0.0, round(PIXELS_PER_METER, 6), 0.0],
        [0.0, 0.0, 1.0],
    ]

    payload = {
        "frame_id": frame_id,
        "image_width": image_width,
        "image_height": image_height,
        "projection": {
            "type": "homography",
            "H_world_to_image": world_to_image,
        },
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
            "length": slot_to_payload(target_slot)["length"],
            "width":  slot_to_payload(target_slot)["width"]
        },
        "available_parking_slots": [
            slot_to_payload(slot) for slot in ordered_slots
        ],
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

def _save_debug_image(debug_dir, name, image):
    """Save one pipeline stage and print its path."""
    output_path = Path(debug_dir) / name
    cv2.imwrite(str(output_path), image)
    print(f"[Debug] Saved {output_path}")


def _draw_line_detection(frame, lines, merged_h, merged_v, slots, rejected_slots, corners):
    """Visualize raw/merged line geometry before model inference."""
    vis = frame.copy()
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            cv2.line(vis, (x1, y1), (x2, y2), (255, 0, 0), 1)
    for x1, y1, x2, y2 in merged_h + merged_v:
        cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
    for tl, br in slots:
        cv2.rectangle(vis, tuple(map(int, tl)), tuple(map(int, br)), (0, 255, 0), 2)
    for tl, br in rejected_slots:
        cv2.rectangle(vis, tuple(map(int, tl)), tuple(map(int, br)), (0, 0, 255), 1)
    for x, y in corners:
        cv2.circle(vis, (int(x), int(y)), 4, (255, 0, 255), -1)
    return vis


def _draw_raw_predictions(frame, raw_predictions):
    """Visualize every Roboflow prediction before ego/obstacle assignment."""
    vis = frame.copy()
    for prediction in raw_predictions:
        x, y = prediction['x'], prediction['y']
        width, height = prediction['width'], prediction['height']
        x1, y1 = int(x - width / 2), int(y - height / 2)
        x2, y2 = int(x + width / 2), int(y + height / 2)
        label = prediction['class']
        confidence = prediction['confidence']
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(vis, f"{label} {confidence:.2f}", (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
    return vis


def _draw_ego_selection(frame, cars, ego, obstacles):
    """Show the red-colour ego decision before payload generation."""
    vis = frame.copy()
    for car in cars:
        x1, y1, x2, y2 = car['bbox_px']
        color = (0, 165, 255) if car is ego else (0, 0, 255)
        label = "EGO" if car is ego else "OBSTACLE"
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)
        cv2.putText(vis, label, (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return vis


def process_frame(frame, debug_dir=None, frame_id=0, detector="roboflow",
                  yolo_model_path=YOLO_MODEL_PATH, confidence=0.1):
    global PIXELS_PER_METER
    # Stage 0: Detect the four parking-lot corners and crop their bounds only.
    frame = detect_and_crop_parking_mat(frame)

    # Stage 1 is not used after removing line detection. Skipping its bilateral
    # filter avoids processing the full-resolution image unnecessarily.

    # Stage 2 removed: parking availability comes directly from the detector.
    lines = None
    merged_h, merged_v, slots, rejected_slots, corners = [], [], [], [], []

    # Stage 3: Roboflow detection → cars + available slots
    model_frame = prepare_model_input(frame)
    cars, avail, raw_preds = run_vehicle_detector(
        model_frame, detector, yolo_model_path, confidence
    )

    # Stage 4: Identify ego vs obstacles
    ego, obstacles = identify_ego(cars, model_frame)
    avail = filter_slots_by_median_area(avail)
    avail = remove_slots_containing_ego(avail, ego)
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
    payload = build_astar_payload(
        ego, obstacles, avail,
        frame_id=frame_id,
        image_width=frame.shape[1],
        image_height=frame.shape[0],
    )

    # Visualization.  ``process_video`` writes this image to the annotated
    # output video, so it must always be a valid frame rather than ``None``.
    result = draw_detections(
        frame, lines, slots, rejected_slots, corners,
        cars, avail, ego, obstacles, merged_h, merged_v
    )
    if debug_dir:
        _save_debug_image(debug_dir, "stage_5_final.png", result)
    return result, payload


def test_model(frame, detector="roboflow", yolo_model_path=YOLO_MODEL_PATH,
               confidence=0.2):
    """Run only Stages 3, 4, and 4b on an input image.

    This is intended for validating the Roboflow model labels and ego-car
    selection without crop, perspective, line, slot, or payload processing.
    """
    global PIXELS_PER_METER

    # Stage 3: suppress glare, then detect cars + available slots
    model_frame = prepare_model_input(frame)
    cars, avail, raw_preds = run_vehicle_detector(
        model_frame, detector, yolo_model_path, confidence
    )

    # Stage 4: Identify red ego vehicle
    ego, obstacles = identify_ego(cars, model_frame)
    print(f"[Test model] Ego: {ego['class'] if ego else 'None'} | "
          f"Obstacles: {len(obstacles)} | Available: {len(avail)}")

    # Stage 4b: Calibrate from the selected ego detection
    if ego is not None:
        PIXELS_PER_METER = calibrate_from_ego(ego)
    else:
        print("[Test model] No ego vehicle detected — calibration skipped.")
        PIXELS_PER_METER = None

    # No Stage 5 payload is created in model-test mode.
    result = draw_detections(
        frame, None, [], [], [], cars, avail, ego, obstacles
    )
    return result


def process_video(input_path, output_path, sample_fps, payload_path,
                  detector="roboflow", yolo_model_path=YOLO_MODEL_PATH,
                  confidence=0.2):
    """Process a video at ``sample_fps`` and write annotated video + payloads."""
    capture = cv2.VideoCapture(input_path)
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_path}")

    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        capture.release()
        raise ValueError("Video does not report a valid FPS")
    if sample_fps <= 0:
        capture.release()
        raise ValueError("--fps must be greater than zero")

    frame_id = 0
    next_sample_time = 0.0
    writer = None
    output_size = None
    payloads = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = frame_id / source_fps
            if timestamp + 1e-9 < next_sample_time:
                frame_id += 1
                continue

            result, payload = process_frame(
                frame, debug_dir=None, frame_id=frame_id,
                detector=detector, yolo_model_path=yolo_model_path,
                confidence=confidence,
            )
            if writer is None:
                height, width = result.shape[:2]
                output_size = (width, height)
                writer = cv2.VideoWriter(
                    output_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    sample_fps,
                    output_size,
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Cannot create output video: {output_path}")
            if result.shape[1] != output_size[0] or result.shape[0] != output_size[1]:
                result = cv2.resize(result, output_size)
            writer.write(result)
            if payload:
                payloads.append(payload)

            next_sample_time += 1.0 / sample_fps
            frame_id += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    with open(payload_path, "w") as payload_file:
        json.dump({"source_fps": source_fps, "sample_fps": sample_fps, "frames": payloads}, payload_file, indent=2)
    print(f"\nProcessed {len(payloads)} sampled video frames")
    print(f"Annotated video saved to: {output_path}")
    print(f"Video payloads saved to: {payload_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input",  type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument("--payload", type=str, default="payload.json",
                        help="Path to save A* JSON payload")
    parser.add_argument("--debug-dir", type=str, default="debug_frames",
                        help="Directory for images saved after every pipeline stage")
    parser.add_argument("--test-model", action="store_true",
                        help="Run only Roboflow detection, ego selection, and calibration")
    parser.add_argument("--confidence", type=float, default=0.2,
                        help="Detection confidence threshold from 0.0 to 1.0; lower returns more candidate boxes")
    parser.add_argument("--detector", choices=["roboflow", "yolo"], default="yolo",
                        help="Perception backend")
    parser.add_argument("--yolo-model", type=str, default=YOLO_MODEL_PATH,
                        help="Path to local YOLO weights used with --detector yolo")
    parser.add_argument("--fps", type=float, default=3.0,
                        help="Video processing rate in sampled frames per second")
    args = parser.parse_args()

    if not 0.0 <= args.confidence <= 1.0:
        parser.error("--confidence must be between 0.0 and 1.0")
    if args.detector == "roboflow":
        roboflow_client.configure(
            InferenceConfiguration(confidence_threshold=args.confidence)
        )
        print(f"[Debug] Roboflow confidence threshold: {args.confidence:.3f}")

    video_extensions = {".avi", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}
    is_video = Path(args.input).suffix.lower() in video_extensions
    if is_video:
        if args.test_model:
            parser.error("--test-model currently accepts images only")
        process_video(
            args.input, args.output, args.fps, args.payload,
            detector=args.detector, yolo_model_path=args.yolo_model,
            confidence=args.confidence,
        )
        raise SystemExit(0)

    frame = cv2.imread(args.input)
    if frame is None:
        raise FileNotFoundError(f"Cannot read image: {args.input}")

    if args.test_model:
        result = test_model(
            frame, detector=args.detector, yolo_model_path=args.yolo_model,
            confidence=args.confidence,
        )
        payload = None
    else:
        result, payload = process_frame(
            frame, debug_dir=args.debug_dir, detector=args.detector,
            yolo_model_path=args.yolo_model, confidence=args.confidence,
        )

    # Save visualization
    if result:
        cv2.imwrite(args.output, result)
        print(f"\nDetection result saved to: {args.output}")

    # Save payload
    if payload:
        with open(args.payload, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f"A* payload saved to: {args.payload}")
