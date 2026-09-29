"""Side-by-side comparison grid for the seven Module A methods.

The assignment asks for the *same* set of five representative frames put
through at least four of the listed methods, so this module is the single place
that defines that set and the colour mapping, and the notebook just calls it.

``make_grid`` returns the figure and a tidy summary table; ``main`` writes the
PNG under ``outputs/`` and caches the raw label maps as a ``.npz`` so the
notebook can redraw without paying for mean shift again (it is ~30 s/frame).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from common import paths
from module_a_segmentation import segmentation as S
from module_a_segmentation.detector import ConveyorDetector

# Columns of the grid, in order.  "original" is the untouched input frame.
COLUMNS = ["original", *S.PRETTY]

# Only the snake needs a seed; the rest are unsupervised.  The seed is the
# motion-detector's box for the same frame, so every method in a row is looking
# at the same object.
SEEDED = {"active_contour"}


def load_meta() -> dict:
    return json.loads(paths.meta_file().read_text(encoding="utf-8"))


def representative_frames(meta: dict | None = None) -> list[dict]:
    meta = meta or load_meta()
    return meta["representative_frames"]


def frame_rgb(rep: dict) -> np.ndarray:
    """Load a representative frame by name, not by the absolute path in
    meta.json, so the repo still works if it is moved."""
    name = Path(rep["path"]).name
    img = cv2.imread(str(paths.FRAMES_DIR / name))
    if img is None:
        raise FileNotFoundError(f"missing representative frame {name} in {paths.FRAMES_DIR}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _load_video_gray() -> np.ndarray:
    """Whole clip as a grey stack; the rolling median needs neighbours of each frame."""
    cap = cv2.VideoCapture(str(paths.video_file()))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {paths.video_file()}")
    return np.stack(frames)


def seed_boxes(frames_gray: np.ndarray, indices: list[int]) -> list[tuple[int, int, int, int]]:
    """Largest motion blob per frame, used only to seed the snake."""
    det = ConveyorDetector()
    out = []
    for i in indices:
        boxes, fg = det.detect_from_index(frames_gray, i)
        if boxes:
            b = boxes[0]
            x, y, w, h = b.bbox
            out.append((x, y, w, h))
        else:
            # No motion: fall back to a box around the frame centre so the
            # comparison still renders and the column is visibly empty.
            h, w = frames_gray.shape[1:]
            out.append((int(w * 0.38), int(h * 0.38), int(w * 0.24), int(h * 0.24)))
    return out


def run_frame(rgb: np.ndarray, seed: tuple[int, int, int, int]) -> dict[str, np.ndarray]:
    """Every method on one frame, keyed by the column names above."""
    labels: dict[str, np.ndarray] = {"original": np.zeros(rgb.shape[:2], np.int32)}
    for name in S.PRETTY:
        fn = S.get_method(name)
        kwargs = {"seed_box": seed} if name in SEEDED else {}
        labels[name] = fn(rgb, **kwargs)
    return labels


def summarise(labels: np.ndarray,
              box: tuple[int, int, int, int] | None = None) -> dict:
    """Region count, the size of the biggest region, and -- given the box -- how
    well the best-matching region lines up with it.  Those are the three things
    the comments in the notebook actually have to talk about."""
    n = int(labels.max())
    if n == 0:
        d = {"n_regions": 0, "largest_px": 0, "largest_frac": 0.0, "box_iou": 0.0}
        return d
    counts = np.bincount(labels.ravel(), minlength=n + 1)[1:]
    biggest = int(counts.max())
    d = {
        "n_regions": n,
        "largest_px": biggest,
        "largest_frac": round(biggest / labels.size, 4),
        "box_iou": 0.0,
    }
    if box is not None:
        ref = np.zeros(labels.shape, bool)
        x, y, w, h = (int(v) for v in box)
        ref[max(0, y):y + h, max(0, x):x + w] = True
        ious = []
        for r in np.unique(labels):
            if r == 0:
                continue
            m = labels == r
            # NB: the parentheses matter -- ``labels == r & ref`` would compare
            # labels against the elementwise ``r & ref`` and count every
            # labelled pixel in the frame.
            ious.append((m & ref).sum() / max(int((m | ref).sum()), 1))
        d["box_iou"] = round(float(max(ious)), 4) if ious else 0.0
    return d


# Vivid, well-separated region colours.  Index 0 is unused so that label 0 can
# stay background.  Deliberately short: when a method returns thousands of
# regions, cycling a small palette keeps neighbouring regions distinguishable,
# whereas a unique colour per region just produces noise.
PALETTE = np.array([
    [255, 92, 92], [92, 255, 120], [96, 152, 255], [255, 206, 84],
    [236, 120, 220], [96, 224, 224], [176, 128, 255], [168, 255, 132],
], np.float32) / 255.0


def _palette_layer(labels: np.ndarray) -> np.ndarray:
    """Float RGB in [0, 1] with label k painted PALETTE[(k - 1) % len(PALETTE)]."""
    n = int(labels.max())
    out = np.zeros(labels.shape + (3,), np.float32)
    if n == 0:
        return out
    colours = PALETTE[np.arange(n) % len(PALETTE)]
    nz = labels > 0
    out[nz] = colours[labels[nz] - 1]
    return out


def best_match_region(labels: np.ndarray,
                      box: tuple[int, int, int, int]) -> int:
    """Label of the region with the highest IoU against ``box`` (x, y, w, h).

    Several of these methods label every pixel, so "which region is the box?" is
    not answerable from the raw overlay.  Picking the best-IoU region turns each
    column into a direct answer to that question.
    """
    ref = np.zeros(labels.shape, bool)
    x, y, w, h = (int(v) for v in box)
    ref[max(0, y):y + h, max(0, x):x + w] = True
    best, best_iou = 0, 0.0
    for lab in np.unique(labels):
        if lab == 0:
            continue
        m = labels == lab
        iou = (m & ref).sum() / max(int((m | ref).sum()), 1)
        if iou > best_iou:
            best, best_iou = int(lab), float(iou)
    return best


def _overlay(rgb: np.ndarray, labels: np.ndarray, alpha: float = 0.55,
             min_outline_area: int = 400,
             highlight_box: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """Boundaries on top of a dimmed copy of the frame, so the box the method
    found is readable without a 200-colour legend.

    Two details matter here.  The region layer is built from an explicit palette
    rather than ``skimage.color.label2rgb``: the colour wheel hands out near
    black for a one-region label image, which made the snake column render as a
    black panel.  And only regions above ``min_outline_area`` get an outline --
    the split-and-merge methods return a few thousand regions, and outlining all
    of them fills the panel with lines and hides the result.
    """
    dim = (rgb.astype(np.float32) * 0.40) / 255.0
    regions = _palette_layer(labels)
    blend = np.clip((1.0 - alpha) * dim + alpha * regions, 0.0, 1.0)
    out = (blend * 255.0).round().astype(np.uint8)

    if labels.max() > 0:
        counts = np.bincount(labels.ravel())
        big = np.zeros(counts.size, bool)
        big[counts >= min_outline_area] = True
        solid = big[labels]
        edges = np.zeros(labels.shape, bool)
        edges[:, :-1] |= labels[:, :-1] != labels[:, 1:]
        edges[:-1, :] |= labels[:-1, :] != labels[1:, :]
        out[edges & solid] = (255, 255, 255)

    if highlight_box is not None:
        lab = best_match_region(labels, highlight_box)
        if lab:
            sel = labels == lab
            # Tint the winning region and trace it, so "this method put the box
            # in its own region" is visible at a glance.
            out[sel] = (0.62 * out[sel] + 0.38 * np.array([255, 255, 0])).round().astype(np.uint8)
            cnts, _ = cv2.findContours(sel.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cnts, -1, (0, 255, 255), 2)
    return out


def make_grid(cache: Path | None = None, force: bool = False):
    """Build the 5x9 comparison figure.  Returns ``(figure, table, labels)``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meta = load_meta()
    reps = representative_frames(meta)
    indices = [r["index"] for r in reps]
    cache = cache or (paths.OUTPUTS / "module_a_labels.npz")

    if cache.exists() and not force:
        z = np.load(cache)
        labels_by_frame = [
            {name: z[f"f{i}_{name}"] for name in COLUMNS if f"f{i}_{name}" in z}
            for i in range(len(reps))
        ]
        # Seeds are cheap (one rolling median per frame) and the cache does not
        # hold them, so recompute rather than store a second copy.
        seeds = seed_boxes(_load_video_gray(), indices)
    else:
        gray = _load_video_gray()
        seeds = seed_boxes(gray, indices)
        del gray
        labels_by_frame = []
        for n, (rep, seed) in enumerate(zip(reps, seeds), start=1):
            print(f"  frame {rep['index']:5d} ({Path(rep['path']).name}) seed={seed}", flush=True)
            labels_by_frame.append(run_frame(frame_rgb(rep), seed))
        paths.OUTPUTS.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache,
            **{f"f{i}_{k}": v for i, d in enumerate(labels_by_frame) for k, v in d.items()},
        )

    n_rows, n_cols = len(reps), len(COLUMNS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.1 * n_cols, 2.5 * n_rows))
    table = {}
    for r, rep in enumerate(reps):
        rgb = frame_rgb(rep)
        seed = seeds[r]
        for c, name in enumerate(COLUMNS):
            ax = axes[r, c]
            lab = labels_by_frame[r][name]
            if name == "original":
                ax.imshow(rgb, interpolation="nearest")
                x, y, w, h = seed
                ax.add_patch(plt.Rectangle((x, y), w, h, fill=False,
                                           edgecolor="lime", linewidth=2))
            else:
                ax.imshow(_overlay(rgb, lab, highlight_box=seed), interpolation="nearest")
            if r == 0:
                ax.set_title(name.replace("_", " "), fontsize=11)
            if c == 0:
                ax.set_ylabel(f"frame {rep['index']}\n(motion box in green)", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            if name != "original":
                table[(rep["index"], name)] = summarise(lab, seed)
    fig.suptitle("Module A - the same five representative frames through all seven methods",
                 fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig, table, labels_by_frame


def main(force: bool = False) -> None:
    paths.ensure_dirs()
    print("Module A comparison grid"
          + ("  (recomputing all methods)" if force else "  (reusing cached labels)"))
    fig, table, _ = make_grid(force=force)
    out = paths.OUTPUTS / "module_a_comparison_grid.png"
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")

    names = COLUMNS[1:]
    lines = ["Module A: per frame and method,  n_regions / IoU of the best-matching region "
             "against the motion-detected box", ""]
    head = f"{'frame':>6s}  " + "  ".join(f"{c[:11]:>12s}" for c in names)
    lines += [head, "-" * len(head)]
    for idx in [r["index"] for r in representative_frames()]:
        cells = [f"{table[(idx, c)]['n_regions']:5d}/{table[(idx, c)]['box_iou']*100:5.1f}%"
                 for c in names]
        lines.append(f"{idx:6d}  " + "  ".join(f"{c:>12s}" for c in cells))
    lines += ["", "largest single region as a share of the frame:"]
    for idx in [r["index"] for r in representative_frames()]:
        cells = [f"{table[(idx, c)]['largest_frac']*100:10.1f}%" for c in names]
        lines.append(f"{idx:6d}  " + "  ".join(f"{c:>12s}" for c in cells))
    txt = "\n".join(lines)
    (paths.OUTPUTS / "module_a_summary.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    import argparse

    # Default is to reuse the cached label maps: mean shift costs ~30 s/frame,
    # so redrawing the figure should not re-run every method.
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true",
                    help="recompute every method instead of reusing the cached labels")
    main(force=ap.parse_args().force)
