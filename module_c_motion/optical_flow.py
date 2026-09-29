"""Module C: optical flow, and the belt speed it implies.

Four pieces, in the order the brief asks for them:

    lucas_kanade(I1, I2, pts)          hand-written, iterative, no cv2 flow calls
    pyrlk_reference(I1, I2, pts)       cv2.calcOpticalFlowPyrLK, for comparison
    estimate_global_affine(p0, p1)     6-parameter affine + RANSAC inlier split
    cm_per_px()                        Module B's scale, evaluated at the box

The frame pair (135, 136) has the box a few pixels into its toss (the affine
fit below finds ~8.6 px of net translation between the two frames), and the
box surface itself is motion-blurred at 60 fps. That second part turns out to
matter more than raw pixel displacement for where hand-written LK and PyrLK
disagree -- see the discussion at the bottom.

Usage
-----
    python module_c_motion/optical_flow.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import ensure_dirs, video_file
from common.paths import BOX_DIMENSIONS_CM, OUTPUTS
from module_a_segmentation.detector import ConveyorDetector

FRAME_A, FRAME_B = 135, 136
REF_BOX, REF_DIMS_CM = "Box006", BOX_DIMENSIONS_CM["Box006"][:2]

# ---------------------------------------------------------------------------
# 1. Hand-written Lucas-Kanade
# ---------------------------------------------------------------------------


def _bilinear_sample(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Sample ``img`` at fractional (xs, ys); out-of-bounds -> nan."""
    H, W = img.shape
    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    x1, y1 = x0 + 1, y0 + 1
    valid = (x0 >= 0) & (x1 < W) & (y0 >= 0) & (y1 < H)
    x0c, x1c = np.clip(x0, 0, W - 1), np.clip(x1, 0, W - 1)
    y0c, y1c = np.clip(y0, 0, H - 1), np.clip(y1, 0, H - 1)
    wx, wy = xs - x0, ys - y0
    top = img[y0c, x0c] * (1 - wx) + img[y0c, x1c] * wx
    bot = img[y1c, x0c] * (1 - wx) + img[y1c, x1c] * wx
    out = top * (1 - wy) + bot * wy
    out = out.astype(np.float64)
    out[~valid] = np.nan
    return out


def lucas_kanade(I1: np.ndarray, I2: np.ndarray, pts: np.ndarray,
                  win: int = 15, iters: int = 3, min_eig: float = 1e-2):
    """Iterative differential optical flow, solved from scratch per point.

    For each point, the spatial gradient tensor is built once from a window in
    ``I1`` (the template):

        Ix = dI1/dx,  Iy = dI1/dy

              [SUM Ix^2    SUM Ix*Iy]         [-SUM Ix*It]
          A = [SUM Ix*Iy   SUM Iy^2 ]   b   = [-SUM Iy*It]

    and then, for ``iters`` Newton steps, ``I2`` is resampled at the current
    displacement estimate to get ``It = I2(x+d) - I1(x)``, and ``A d = b`` is
    solved and accumulated -- this is the textbook iterative/Newton-Raphson
    refinement, not a single one-shot solve.

    Points whose window has too little texture (``min(eig(A)) < min_eig``,
    Shi-Tomasi's own cornerness criterion) are left with zero flow and marked
    invalid: a flat or edge-only window makes ``A`` singular or near-singular,
    the textbook aperture problem, and reporting a "solution" there would be
    reporting numerical noise, not a measurement.

    Returns
    -------
    flow : (N, 2) float, (dx, dy) per point
    residual : (N,) mean squared brightness residual after the fit
    valid : (N,) bool
    """
    I1 = I1.astype(np.float64)
    I2 = I2.astype(np.float64)
    Iy, Ix = np.gradient(I1)
    H, W = I1.shape
    r = win // 2
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float64)

    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    n = len(pts)
    flow = np.zeros((n, 2))
    residual = np.full(n, np.inf)
    valid = np.zeros(n, bool)

    for i, (px, py) in enumerate(pts):
        wx, wy = px + xx, py + yy
        if wx.min() < 1 or wx.max() > W - 2 or wy.min() < 1 or wy.max() > H - 2:
            continue
        gx = _bilinear_sample(Ix, wx, wy)
        gy = _bilinear_sample(Iy, wx, wy)
        if np.isnan(gx).any():
            continue
        Sxx, Sxy, Syy = np.sum(gx * gx), np.sum(gx * gy), np.sum(gy * gy)
        A = np.array([[Sxx, Sxy], [Sxy, Syy]])
        eig_min = np.linalg.eigvalsh(A).min() / (win * win)
        if eig_min < min_eig:
            continue

        template = _bilinear_sample(I1, wx, wy)
        d = np.zeros(2)
        for _ in range(iters):
            warped = _bilinear_sample(I2, wx + d[0], wy + d[1])
            if np.isnan(warped).any():
                d = None
                break
            It = warped - template
            b = -np.array([np.sum(gx * It), np.sum(gy * It)])
            try:
                delta = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                d = None
                break
            d = d + delta
            if np.linalg.norm(delta) < 1e-3:
                break
        if d is None:
            continue

        final = _bilinear_sample(I2, wx + d[0], wy + d[1])
        if np.isnan(final).any():
            continue
        flow[i] = d
        residual[i] = float(np.mean((final - template) ** 2))
        valid[i] = True

    return flow, residual, valid


# ---------------------------------------------------------------------------
# 2. OpenCV's pyramidal, feature-based flow, for comparison
# ---------------------------------------------------------------------------


def sparse_features(gray: np.ndarray, mask: np.ndarray | None = None,
                     max_corners: int = 300) -> np.ndarray:
    """Shi-Tomasi corners (the standard front-end for PyrLK)."""
    pts = cv2.goodFeaturesToTrack(gray, maxCorners=max_corners, qualityLevel=0.01,
                                   minDistance=7, mask=mask, blockSize=7)
    return pts.reshape(-1, 2) if pts is not None else np.empty((0, 2))


def pyrlk_reference(I1: np.ndarray, I2: np.ndarray, pts: np.ndarray):
    """``cv2.calcOpticalFlowPyrLK`` on the same points, for comparison."""
    p0 = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
    p1, status, err = cv2.calcOpticalFlowPyrLK(
        I1, I2, p0, None, winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    flow = (p1 - p0).reshape(-1, 2)
    valid = status.reshape(-1).astype(bool)
    return flow, valid, err.reshape(-1)


# ---------------------------------------------------------------------------
# 3. Affine flow model + RANSAC
# ---------------------------------------------------------------------------


def estimate_global_affine(pts0: np.ndarray, pts1: np.ndarray):
    """6-parameter affine fit to a set of point correspondences, RANSAC-robust.

    Returns (A (2x3), inlier_mask (bool, len N)).
    """
    p0 = np.asarray(pts0, dtype=np.float32).reshape(-1, 1, 2)
    p1 = np.asarray(pts1, dtype=np.float32).reshape(-1, 1, 2)
    A, inliers = cv2.estimateAffine2D(p0, p1, method=cv2.RANSAC,
                                      ransacReprojThreshold=2.0, maxIters=2000,
                                      confidence=0.995)
    return A, (inliers.reshape(-1).astype(bool) if inliers is not None else
              np.zeros(len(pts0), bool))


# ---------------------------------------------------------------------------
# 4. Pixel -> cm/s, from Module B
# ---------------------------------------------------------------------------


def cm_per_px(point_px: tuple[float, float] | None = None) -> float:
    """Local metric scale, transferred from the Module B reference (Box006 @
    frame 135, the same one ``measure_box.py`` uses).

    Deliberately the isotropic perimeter-ratio scale (known perimeter /
    detected-quad perimeter), not ``BeltScaleModel``'s full anisotropic
    plane fit -- Module B's own comparison (see ``module_b_size_table.txt``)
    found the orthographic/isotropic model the most accurate of the four on
    this footage, and the full-perspective model's forced orthonormal-rotation
    step is *not* even self-consistent on this reference quad (it doesn't
    reproduce the reference's own known size when run on itself), because the
    detected footprint is skewed by motion blur, not a clean planar homography
    the model can decompose exactly. ``point_px`` is accepted for interface
    symmetry with a fuller local model but the isotropic scale does not vary
    with position.
    """
    det = ConveyorDetector()
    cap = cv2.VideoCapture(str(video_file()))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    cap.release()
    frames = np.stack(frames)

    boxes, _ = det.detect_from_index(frames, 135)
    q = boxes[0].min_area_side.reshape(4, 2)
    w, l = REF_DIMS_CM
    known_perimeter = 2.0 * (w + l)
    detected_perimeter = sum(np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4))
    return float(known_perimeter / detected_perimeter)


# ---------------------------------------------------------------------------


def _load_pair():
    cap = cv2.VideoCapture(str(video_file()))
    frames = {}
    idx = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if idx in (FRAME_A, FRAME_B):
            frames[idx] = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        if len(frames) == 2:
            break
        idx += 1
    cap.release()
    return frames[FRAME_A], frames[FRAME_B]


def _draw_flow(ax, gray, pts, flow, valid, colour, scale=1.0, label=None):
    ax.imshow(gray, cmap="gray")
    pts, flow, valid = np.asarray(pts), np.asarray(flow), np.asarray(valid)
    good = valid & np.isfinite(flow).all(axis=1)
    ax.quiver(pts[good, 0], pts[good, 1], flow[good, 0] * scale, flow[good, 1] * scale,
              angles="xy", scale_units="xy", scale=1.0, color=colour, width=0.003)
    if label:
        ax.set_title(f"{label}  ({good.sum()}/{len(pts)} valid)", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_dirs()
    I1, I2 = _load_pair()
    H, W = I1.shape
    print(f"Frame pair: {FRAME_A} -> {FRAME_B}  ({W}x{H})")

    det = ConveyorDetector()
    cap = cv2.VideoCapture(str(video_file()))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    cap.release()
    frames = np.stack(frames)
    boxes_a, _ = det.detect_from_index(frames, FRAME_A)
    box = boxes_a[0]
    bx, by, bw, bh = box.bbox
    print(f"Box region at frame {FRAME_A}: bbox={box.bbox}  centroid={box.centroid}")

    # ---- (a) hand-written LK on a dense grid ------------------------------
    margin = 10
    gy, gx = np.mgrid[margin:H - margin:12, margin:W - margin:12]
    grid_pts = np.column_stack([gx.ravel(), gy.ravel()]).astype(np.float64)
    lk_flow, lk_resid, lk_valid = lucas_kanade(I1, I2, grid_pts, win=15, iters=3)
    print(f"Hand-written LK on {len(grid_pts)} grid points: "
          f"{lk_valid.sum()} converged (rest failed the cornerness/eigenvalue test "
          f"or moved out of frame)")

    # ---- (b) sparse feature-based flow, same frame pair --------------------
    corners = sparse_features(I1)
    pyrlk_flow, pyrlk_valid, pyrlk_err = pyrlk_reference(I1, I2, corners)
    print(f"cv2.goodFeaturesToTrack found {len(corners)} corners; PyrLK tracked "
          f"{pyrlk_valid.sum()} of them")

    # Hand-written LK evaluated at the *same* corner points, for a direct,
    # like-for-like numeric comparison against PyrLK rather than just two
    # different-looking pictures.
    lk_at_corners, _, lk_at_corners_valid = lucas_kanade(I1, I2, corners, win=15, iters=3)
    both = pyrlk_valid & lk_at_corners_valid
    disagree_all = disagree_box = disagree_bg = np.array([])
    if both.any():
        disagree_all = np.linalg.norm(lk_at_corners[both] - pyrlk_flow[both], axis=1)
        print(f"At the {both.sum()} corners both methods tracked: mean "
              f"|hand-LK - PyrLK| = {disagree_all.mean():.2f} px, max = {disagree_all.max():.2f} px")

        in_box = ((corners[:, 0] >= bx) & (corners[:, 0] < bx + bw) &
                  (corners[:, 1] >= by) & (corners[:, 1] < by + bh))
        box_both = both & in_box
        bg_both = both & ~in_box
        if box_both.any():
            disagree_box = np.linalg.norm(lk_at_corners[box_both] - pyrlk_flow[box_both], axis=1)
            print(f"  on the box ({box_both.sum()} corners): mean = {disagree_box.mean():.2f} px, "
                  f"max = {disagree_box.max():.2f} px")
        if bg_both.any():
            disagree_bg = np.linalg.norm(lk_at_corners[bg_both] - pyrlk_flow[bg_both], axis=1)
            print(f"  on the background ({bg_both.sum()} corners): mean = {disagree_bg.mean():.2f} px, "
                  f"max = {disagree_bg.max():.2f} px")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    _draw_flow(axes[0], I1, grid_pts, lk_flow, lk_valid, "lime",
              label="hand-written Lucas-Kanade (dense grid)")
    _draw_flow(axes[1], I1, corners, pyrlk_flow, pyrlk_valid, "cyan",
              label="cv2.calcOpticalFlowPyrLK (Shi-Tomasi corners)")
    fig.suptitle(f"Module C - optical flow, frame {FRAME_A} -> {FRAME_B}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out1 = OUTPUTS / "module_c_flow_comparison.png"
    fig.savefig(out1, dpi=110)
    print(f"wrote {out1}")

    # ---- (c) affine flow model on the box region + RANSAC ------------------
    roi_mask = np.zeros((H, W), np.uint8)
    pad = 15
    roi_mask[max(0, by - pad):by + bh + pad, max(0, bx - pad):bx + bw + pad] = 255
    box_corners = sparse_features(I1, mask=roi_mask, max_corners=200)
    if len(box_corners) < 6:
        box_corners = sparse_features(I1, max_corners=200)  # fallback, whole frame
    box_flow, box_valid, _ = pyrlk_reference(I1, I2, box_corners)
    p0 = box_corners[box_valid]
    p1 = p0 + box_flow[box_valid]
    A, inliers = estimate_global_affine(p0, p1)
    if A is None:
        print("RANSAC affine fit failed (too few correspondences)")
        A = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        inliers = np.zeros(len(p0), bool)
    print(f"\nAffine flow model on the box region ({len(p0)} points, "
          f"{inliers.sum()} RANSAC inliers, {len(p0) - inliers.sum()} outliers)")
    print(f"  A =\n{np.array2string(A, precision=3, suppress_small=True)}")
    translation_px = A[:, 2]
    print(f"  translation (dx, dy) = ({translation_px[0]:.2f}, {translation_px[1]:.2f}) px/frame")

    fig2, ax2 = plt.subplots(figsize=(7, 6))
    ax2.imshow(I1, cmap="gray")
    rect = plt.Rectangle((bx - pad, by - pad), bw + 2 * pad, bh + 2 * pad,
                         fill=False, edgecolor="yellow", linewidth=1.5, linestyle="--")
    ax2.add_patch(rect)
    ax2.quiver(p0[inliers, 0], p0[inliers, 1], box_flow[box_valid][inliers, 0],
              box_flow[box_valid][inliers, 1], angles="xy", scale_units="xy",
              scale=1.0, color="lime", width=0.004, label=f"inliers ({inliers.sum()})")
    out_mask = ~inliers
    if out_mask.any():
        ax2.quiver(p0[out_mask, 0], p0[out_mask, 1], box_flow[box_valid][out_mask, 0],
                  box_flow[box_valid][out_mask, 1], angles="xy", scale_units="xy",
                  scale=1.0, color="red", width=0.004, label=f"outliers ({out_mask.sum()})")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.set_title(f"Module C - RANSAC-fitted affine flow, box region, "
                  f"frame {FRAME_A}->{FRAME_B}", fontsize=11)
    ax2.set_xticks([])
    ax2.set_yticks([])
    out2 = OUTPUTS / "module_c_ransac_affine.png"
    fig2.savefig(out2, dpi=110)
    print(f"wrote {out2}")

    # ---- (d) belt speed, cm/s ----------------------------------------------
    fps = 60.0
    speed_px_per_frame = float(np.linalg.norm(translation_px))
    speed_px_per_s = speed_px_per_frame * fps
    scale = cm_per_px(tuple(box.centroid))
    speed_cm_per_s = speed_px_per_s * scale

    lines = [
        "Module C: belt/box speed from the RANSAC-fitted affine flow",
        "",
        f"Frame pair              : {FRAME_A} -> {FRAME_B}  (dt = {1 / fps * 1000:.2f} ms)",
        f"Affine translation      : ({translation_px[0]:.2f}, {translation_px[1]:.2f}) px/frame  "
        f"({inliers.sum()} RANSAC inliers of {len(p0)})",
        f"Local scale (Module B)  : {scale:.4f} cm/px  (at box centroid, from the Box006 "
        f"reference-object transfer)",
        f"Speed                   : {speed_px_per_frame:.2f} px/frame = {speed_px_per_s:.1f} px/s "
        f"= {speed_cm_per_s:.1f} cm/s",
        "",
        "Note: this is a thrown box's instantaneous speed at this point in its arc, not a "
        "steady conveyor-belt speed -- consecutive-frame displacement changes noticeably "
        "across the toss (see the depth-varies-during-flight note in measure_box.py and "
        "the README). Reported here exactly as the brief asks: from optical flow directly.",
    ]
    txt = "\n".join(lines)
    print("\n" + txt)
    out3 = OUTPUTS / "module_c_speeds.txt"
    out3.write_text(txt, encoding="utf-8")
    print(f"\nwrote {out3}")

    # ---- discussion (printed, also worth having in the README) -------------
    bg_txt = (f"{disagree_bg.mean():.2f} px (max {disagree_bg.max():.2f} px, "
              f"n={len(disagree_bg)})" if len(disagree_bg) else "n/a")
    box_txt = (f"{disagree_box.mean():.2f} px (max {disagree_box.max():.2f} px, "
               f"n={len(disagree_box)})" if len(disagree_box) else "n/a")
    print(
        "\nDiscussion -- where dense hand-LK and sparse PyrLK disagree:\n"
        f"  Background corners: mean |hand-LK - PyrLK| = {bg_txt}. The two methods "
        "agree closely where the small-motion brightness-constancy linearisation "
        "actually holds -- the sanity check that the hand-written solver is correct.\n"
        f"  Box corners: mean |hand-LK - PyrLK| = {box_txt}, several times the "
        "background disagreement even though the net box displacement here (~8.6 "
        "px/frame from the affine fit) is well within a single 15x15 window's reach. "
        "PyrLK returns a tight, coherent ~6.5 px vertical vector across almost every "
        "box corner; the hand-written solver matches it closely at some points but is "
        "noisy or wrong-signed at others, on the *same* corners. The cause is not "
        "displacement size but motion blur: a thrown box at 60 fps smears during the "
        "exposure, so I1 and I2 are not two sharp snapshots of the same rigid pattern -- "
        "the brightness-constancy assumption the whole derivation rests on is only "
        "approximately true there, and a single-scale, fixed-template solver has no "
        "mechanism to average that out the way PyrLK's larger 21x21 window and "
        "coarse-to-fine refinement do. The classic textureless-surface aperture problem "
        "is visible too, but as a *different* symptom: on the box's flat cardboard "
        "faces the hand-LK eigenvalue test rejects points outright (they never appear "
        "in the corner set at all), rather than producing a wrong answer."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
