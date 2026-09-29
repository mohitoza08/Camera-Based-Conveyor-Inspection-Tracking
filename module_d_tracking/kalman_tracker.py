"""Module D: a from-scratch Kalman filter tracking one box's centroid.

State ``[x, y, vx, vy]``, constant-velocity motion model, no
``cv2.KalmanFilter`` -- ``KalmanBoxTracker`` below implements the predict and
update steps directly from the textbook equations.

Tracked recording: Box006, viewpoint 1, frames 0-1191 of the archive (this is
also the first third of ``data/conveyor.mp4``). The brief asks to track "one
box... across the full video"; the concatenated ``conveyor.mp4`` actually
contains three different physical boxes in sequence (one per recording), so
"the full video" here means this one complete recording end to end, rather
than a track that would have to silently jump onto a different physical
object at a segment boundary.

Usage
-----
    python module_d_tracking/kalman_tracker.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import ensure_dirs, video_file
from common.paths import OUTPUTS
from module_a_segmentation.detector import ConveyorDetector

SEGMENT_START, SEGMENT_LEN = 0, 1192  # Box006, viewpoint 1
DT = 1.0 / 60.0

# Item 3 of the brief: an artificial occlusion, in addition to whatever
# natural detection gaps the detector already produces (the box is briefly
# undetectable near the very start/end of its arc, when it barely differs
# from the rolling-median background).
DROPOUT_START, DROPOUT_LEN = 500, 4


class KalmanBoxTracker:
    """From-scratch constant-velocity Kalman filter, state = [x, y, vx, vy].

        F = [[1, 0, dt, 0],      state transition (constant velocity)
             [0, 1, 0, dt],
             [0, 0, 1, 0],
             [0, 0, 0, 1]]
        H = [[1, 0, 0, 0],       measurement model (position only)
             [0, 1, 0, 0]]
        Q = diag(q_pos, q_pos, q_vel, q_vel)     process noise
        R = diag(r_pos, r_pos)                   measurement noise

    ``Q``'s velocity terms are deliberately large relative to a "smooth
    tracking" default: this is a thrown, bouncing box, not a box gliding at
    constant velocity, so the constant-velocity model is only ever
    approximately right, and a small Q would make the filter trust its own
    (wrong) extrapolation too much through the parabola.
    """

    def __init__(self, dt: float, q_pos: float = 4.0, q_vel: float = 800.0,
                 r_pos: float = 16.0) -> None:
        self.dt = dt
        self.F = np.array([[1, 0, dt, 0],
                            [0, 1, 0, dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]], dtype=np.float64)
        self.H = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0]], dtype=np.float64)
        self.Q = np.diag([q_pos, q_pos, q_vel, q_vel]).astype(np.float64)
        self.R = np.diag([r_pos, r_pos]).astype(np.float64)
        self.x = np.zeros(4)
        self.P = np.eye(4) * 1e3
        self.initialized = False

    def init_state(self, z: np.ndarray) -> None:
        self.x = np.array([z[0], z[1], 0.0, 0.0])
        self.P = np.diag([25.0, 25.0, 1.0e3, 1.0e3])
        self.initialized = True

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64)
        y = z - self.H @ self.x                       # innovation
        S = self.H @ self.P @ self.H.T + self.R        # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)       # Kalman gain
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.x.copy()

    @property
    def state(self) -> np.ndarray:
        return self.x.copy()


# ---------------------------------------------------------------------------


def collect_measurements(frames: np.ndarray, start: int, n: int, det: ConveyorDetector):
    """Per-frame noisy centroid measurement, or None where nothing was detected."""
    boxes_by_frame = []
    z = []
    for i, idx in enumerate(range(start, start + n)):
        boxes, _ = det.detect_from_index(frames, idx)
        if boxes:
            boxes_by_frame.append(boxes[0])
            z.append(np.array(boxes[0].centroid))
        else:
            boxes_by_frame.append(None)
            z.append(None)
        if (i + 1) % 200 == 0:
            print(f"  detected {i + 1}/{n} frames", flush=True)
    return boxes_by_frame, z


def run_tracker(z_list: list[np.ndarray | None]):
    """Predict/update the whole sequence. Returns (raw, predicted, filtered, is_measured)."""
    tracker = KalmanBoxTracker(DT)
    raw = np.full((len(z_list), 2), np.nan)
    predicted = np.full((len(z_list), 2), np.nan)
    filtered = np.full((len(z_list), 2), np.nan)
    is_measured = np.zeros(len(z_list), bool)

    for t, z in enumerate(z_list):
        used = z is not None and not (DROPOUT_START <= t < DROPOUT_START + DROPOUT_LEN)
        if not tracker.initialized:
            if used:
                tracker.init_state(z)
                raw[t] = z
                predicted[t] = tracker.state[:2]
                filtered[t] = tracker.state[:2]
                is_measured[t] = True
            continue

        pred = tracker.predict()
        predicted[t] = pred[:2]
        if used:
            corrected = tracker.update(z)
            filtered[t] = corrected[:2]
            raw[t] = z
            is_measured[t] = True
        else:
            filtered[t] = pred[:2]     # pure prediction, no correction
    return raw, predicted, filtered, is_measured


def print_predict_update_cycle(raw, predicted, filtered, is_measured, n: int = 10):
    """Item 2 of the brief: show predict -> measure -> update explicitly."""
    start = None
    for t in range(len(is_measured) - n):
        if is_measured[t:t + n].all():
            start = t
            break
    if start is None:
        print("(no run of measured frames long enough to print)")
        return
    print(f"\nPredict -> measure -> update, frames {start}-{start + n - 1}:")
    header = f"{'frame':>6s} {'predicted (x,y)':>20s} {'measured (x,y)':>20s} {'corrected (x,y)':>20s}"
    print(header)
    for t in range(start, start + n):
        p, m, c = predicted[t], raw[t], filtered[t]
        print(f"{t:6d} ({p[0]:7.2f},{p[1]:7.2f})   ({m[0]:7.2f},{m[1]:7.2f})   "
              f"({c[0]:7.2f},{c[1]:7.2f})")


def make_dropout_plot(raw, filtered, is_measured, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(is_measured)
    t = np.arange(n)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=False)

    ax = axes[0]
    ax.plot(raw[is_measured, 0], raw[is_measured, 1], "o", ms=3, color="tab:red",
           alpha=0.5, label="raw measurement")
    ax.plot(filtered[:, 0], filtered[:, 1], "-", lw=1.5, color="tab:blue",
           label="Kalman-filtered trajectory")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_title("Module D - raw detections vs Kalman trajectory (full track)")
    ax.legend(fontsize=9)
    ax.invert_yaxis()

    ax = axes[1]
    valid = np.isfinite(raw[:, 1])
    ax.plot(t[valid], raw[valid, 1], "o", ms=3, color="tab:red", alpha=0.5,
           label="raw measurement (y)")
    ax.plot(t, filtered[:, 1], "-", lw=1.3, color="tab:blue", label="Kalman-filtered (y)")
    ax.axvspan(DROPOUT_START, DROPOUT_START + DROPOUT_LEN - 1, color="red", alpha=0.15)
    for edge in (DROPOUT_START, DROPOUT_START + DROPOUT_LEN - 1):
        ax.axvline(edge, color="red", linestyle="--", linewidth=1)
    ax.set_xlim(max(0, DROPOUT_START - 60), DROPOUT_START + DROPOUT_LEN + 60)
    ax.set_xlabel("frame")
    ax.set_ylabel("y (px)")
    ax.set_title(f"Module D - simulated dropout, frames {DROPOUT_START}-"
                f"{DROPOUT_START + DROPOUT_LEN - 1} (red band): pure prediction, no measurement")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    print(f"wrote {out_path}")


def make_tracking_video(frames_bgr_iter, boxes_by_frame, predicted, filtered, is_measured,
                        out_path: Path, size: tuple[int, int], fps: float = 60.0):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, size)
    trail: list[tuple[int, int]] = []
    for t, frame in enumerate(frames_bgr_iter):
        vis = frame.copy()
        box = boxes_by_frame[t]
        if box is not None and is_measured[t]:
            quad = box.min_area_side.reshape(4, 2).astype(np.int32)
            cv2.polylines(vis, [quad], True, (0, 200, 0), 2)

        if np.isfinite(predicted[t]).all():
            p = tuple(int(v) for v in predicted[t])
            cv2.drawMarker(vis, p, (0, 215, 255), cv2.MARKER_DIAMOND, 14, 2)  # amber: prediction
        if np.isfinite(filtered[t]).all():
            c = tuple(int(v) for v in filtered[t])
            cv2.drawMarker(vis, c, (255, 200, 0), cv2.MARKER_CROSS, 14, 2)     # cyan: corrected
            trail.append(c)

        if len(trail) >= 2:
            pts = np.array(trail[-91:], dtype=np.int32)
            cv2.polylines(vis, [pts], False, (255, 200, 0), 1)

        label = "OCCLUDED (prediction only)" if not is_measured[t] else "track B-001"
        colour = (0, 0, 255) if not is_measured[t] else (255, 255, 255)
        cv2.putText(vis, f"frame {t}  {label}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, colour, 2, cv2.LINE_AA)
        writer.write(vis)
    writer.release()
    print(f"wrote {out_path}")


def main() -> int:
    ensure_dirs()
    print(f"Loading frames {SEGMENT_START}-{SEGMENT_START + SEGMENT_LEN - 1} "
          f"({SEGMENT_LEN} frames, ~{SEGMENT_LEN / 60:.1f} s at 60 fps)...")

    cap = cv2.VideoCapture(str(video_file()))
    all_gray = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        all_gray.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    cap.release()
    all_gray = np.stack(all_gray)

    det = ConveyorDetector()
    print("Detecting boxes (Module A front-end) frame by frame...")
    boxes_by_frame, z_list = collect_measurements(all_gray, SEGMENT_START, SEGMENT_LEN, det)
    n_detected = sum(z is not None for z in z_list)
    print(f"{n_detected}/{SEGMENT_LEN} frames had a detection "
          f"({SEGMENT_LEN - n_detected} natural gaps, before the artificial dropout)")

    raw, predicted, filtered, is_measured = run_tracker(z_list)
    print_predict_update_cycle(raw, predicted, filtered, is_measured)

    make_dropout_plot(raw, filtered, is_measured, OUTPUTS / "module_d_dropout.png")

    print("\nRendering tracking video (this reads the segment's frames again, in colour)...")
    cap = cv2.VideoCapture(str(video_file()))
    idx = 0
    bgr_frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if SEGMENT_START <= idx < SEGMENT_START + SEGMENT_LEN:
            bgr_frames.append(f)
        idx += 1
        if idx >= SEGMENT_START + SEGMENT_LEN:
            break
    cap.release()
    H, W = bgr_frames[0].shape[:2]
    make_tracking_video(bgr_frames, boxes_by_frame, predicted, filtered, is_measured,
                        OUTPUTS / "module_d_tracking.mp4", size=(W, H))

    print(f"\nDropout summary: {DROPOUT_LEN} consecutive frames "
          f"({DROPOUT_START}-{DROPOUT_START + DROPOUT_LEN - 1}) had their measurement "
          f"discarded; the filter tracked through them on prediction alone, then resumed "
          f"correcting once real measurements returned.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
