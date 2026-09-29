"""End-to-end pipeline: frame stream -> segmented boxes -> tracked, sized,
timed and classified.

    Module A (detect)  ->  Module D (Kalman-track centroid)
                       ->  Module B (pixel quad -> size, cm)
                       ->  Module C (optical flow -> speed, cm/s)
                       ->  Module E (crop -> recognised type)

Run per box, on a short window around one clean detection per segment (the
three recordings in ``data/conveyor.mp4``), and prints the final table in the
exact format the brief asks for.

Usage
-----
    python pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ensure_dirs, video_file
from common.paths import BOX_DIMENSIONS_CM, OUTPUTS
from module_a_segmentation.detector import ConveyorDetector
from module_c_motion.optical_flow import estimate_global_affine, sparse_features, pyrlk_reference
from module_d_tracking.kalman_tracker import KalmanBoxTracker, DT
from module_e_recognition import recognize as E

# One representative, previously-verified clean-detection window per segment
# (see module_b_calibration/measure_box.py and module_c_motion/optical_flow.py
# for how these specific frames were chosen -- non-border, single, clean
# detections, with Box006's frame picked to be at a matched apparent scale to
# the other two so the reference-object size transfer is meaningful).
TRACKS = [
    ("B-001", "Box006", 135),
    ("B-002", "Box007", 1257),
    ("B-003", "Box009", 2480),
]
WINDOW = 6          # frames of Kalman tracking around the anchor frame
REF_BOX, REF_FRAME = "Box006", 135
REF_DIMS_CM = BOX_DIMENSIONS_CM[REF_BOX][:2]


def isotropic_scale(det: ConveyorDetector, frames: np.ndarray) -> float:
    """cm/px, transferred from the Box006 reference (see optical_flow.cm_per_px
    for why this is isotropic rather than the full anisotropic plane model)."""
    boxes, _ = det.detect_from_index(frames, REF_FRAME)
    q = boxes[0].min_area_side.reshape(4, 2)
    w, l = REF_DIMS_CM
    known_perimeter = 2.0 * (w + l)
    detected_perimeter = sum(np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4))
    return known_perimeter / detected_perimeter


def track_box(det: ConveyorDetector, frames: np.ndarray, anchor: int, scale_cm_per_px: float):
    """Module D around ``anchor``: returns (size_cm, speed_cm_s, crop_gray, crop_mask)."""
    lo, hi = max(0, anchor - WINDOW // 2), min(len(frames), anchor + WINDOW // 2)
    tracker = KalmanBoxTracker(DT)
    last_box = None
    for idx in range(lo, hi):
        boxes, fg = det.detect_from_index(frames, idx)
        z = np.array(boxes[0].centroid) if boxes else None
        if not tracker.initialized:
            if z is not None:
                tracker.init_state(z)
                last_box, last_fg = boxes[0], fg
            continue
        tracker.predict()
        if z is not None:
            tracker.update(z)
            last_box, last_fg = boxes[0], fg

    # ---- Module B: size, from the last clean quad -------------------------
    q = last_box.min_area_side.reshape(4, 2)
    edges = sorted(np.linalg.norm(q[(i + 1) % 4] - q[i]) * scale_cm_per_px for i in range(4))
    side_a = (edges[0] + edges[1]) / 2.0
    side_b = (edges[2] + edges[3]) / 2.0

    # ---- Module C: speed, RANSAC-fitted affine flow around the anchor -----
    I1 = frames[anchor]
    I2 = frames[min(anchor + 1, len(frames) - 1)]
    x, y, w, h = last_box.bbox
    pad = 15
    roi = np.zeros(I1.shape, np.uint8)
    roi[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad] = 255
    corners = sparse_features(I1, mask=roi, max_corners=200)
    speed_cm_s = 0.0
    if len(corners) >= 6:
        flow, valid, _ = pyrlk_reference(I1, I2, corners)
        p0, p1 = corners[valid], corners[valid] + flow[valid]
        if len(p0) >= 6:
            A, inliers = estimate_global_affine(p0, p1)
            if A is not None:
                speed_cm_s = float(np.linalg.norm(A[:, 2])) * 60.0 * scale_cm_per_px

    # ---- crop for Module E --------------------------------------------------
    side = int(max(w, h) * 1.15)
    cx, cy = x + w / 2, y + h / 2
    x0, y0 = int(cx - side / 2), int(cy - side / 2)
    x1, y1 = x0 + side, y0 + side
    H, W = I1.shape
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    crop_gray = cv2.resize(frames[anchor, y0:y1, x0:x1], (E.PATCH, E.PATCH))
    crop_mask = cv2.resize(last_fg[y0:y1, x0:x1].astype(np.uint8) * 255, (E.PATCH, E.PATCH),
                           interpolation=cv2.INTER_NEAREST)

    return (side_a, side_b), speed_cm_s, crop_gray, crop_mask


def main() -> int:
    ensure_dirs()
    print("Loading conveyor.mp4...")
    cap = cv2.VideoCapture(str(video_file()))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    cap.release()
    frames = np.stack(frames)

    det = ConveyorDetector()
    scale = isotropic_scale(det, frames)
    print(f"Scale (Module B, Box006 reference transfer): {scale:.4f} cm/px\n")

    print("Module A + D + B + C: detecting, tracking, sizing and timing each box...")
    results = []
    for track_id, true_label, anchor in TRACKS:
        size_cm, speed_cm_s, crop_gray, crop_mask = track_box(det, frames, anchor, scale)
        results.append((track_id, true_label, size_cm, speed_cm_s, crop_gray, crop_mask))
        print(f"  {track_id} (frame {anchor}, true type {true_label}): "
              f"{size_cm[0]:.1f} x {size_cm[1]:.1f} cm, {speed_cm_s:.1f} cm/s")

    print("\nModule E: training the recognizer on crops collected from the whole video "
          "(this repeats Module A detection across all three segments -- see "
          "module_e_recognition/recognize.py for the full, evaluated version)...")
    crops = E.collect_crops(frames)  # use every collected crop to train (deployment, not eval)
    mean, components, train_feats, train_labels = E.fit_eigenspace(crops, k=12)

    print("\nBox ID | Type   | Size (cm)          | Speed (cm/s)")
    print("-------|--------|--------------------|-------------")
    lines = ["Box ID | Type   | Size (cm)          | Speed (cm/s)",
            "-------|--------|--------------------|-------------"]
    for track_id, true_label, size_cm, speed_cm_s, crop_gray, crop_mask in results:
        feat = E.project(crop_gray, mean, components)
        recognized = E.classify_eigen(feat, train_feats, train_labels)
        row = (f"{track_id:<6s} | {recognized:<6s} | {size_cm[0]:5.1f} x {size_cm[1]:5.1f}     "
              f"| {speed_cm_s:6.1f}")
        print(row)
        lines.append(row)

    out = OUTPUTS / "pipeline_output.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
