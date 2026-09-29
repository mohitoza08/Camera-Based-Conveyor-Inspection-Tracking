"""Module A: seven classical segmentation methods, one common interface.

Every function here takes an RGB image and returns a label image of dtype
int32, where 0 is background and 1..N are the segmented regions.  That common
signature is what lets ``compare_methods.ipynb`` put them side by side and lets
``pipeline.py`` reuse whichever one wins.

The seven methods, in the order the syllabus lists them:

    1. active_contour     -- snakes, energy minimisation from a seed curve
    2. quadtree_split_merge -- written from scratch: recursive split on
       intensity variance, merge adjacent regions on mean-intensity distance
    3. watershed          -- distance-transform markers, separates touching blobs
    4. region_split / region_merge -- the two halves of #2 as separate passes
    5. felzenszwalb       -- graph-based, plus an inspection of the region graph
    6. mean_shift         -- mode finding in colour+position space
    7. normalized_cut     -- spectral N-Cut on a superpixel region adjacency graph
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage import color, filters, graph, measure, segmentation
from sklearn.cluster import MeanShift

__all__ = [
    "active_contour",
    "felzenszwalb",
    "mean_shift",
    "normalized_cut",
    "quadtree_split_merge",
    "region_merge",
    "region_split",
    "watershed",
    "METHODS",
    "run_all",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _to_float_gray(rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 -> float64 gray in [0, 1], with edges white so snakes sit on them."""
    return color.rgb2gray(rgb).astype(np.float64)


def _edge_image(rgb: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Inverted, min-max normalised gradient magnitude.

    ``skimage.filters.sobel`` has no smoothing parameter, so the Gaussian
    pre-smoothing that the classic snake formulation calls for is applied
    first.  The normalisation matters: the raw Sobel magnitude of a [0, 1]
    image spans only about 0.02, which is far too flat an energy landscape --
    the curve then contracts to a point under the smoothness term alone
    instead of following an edge.
    """
    g = filters.gaussian(_to_float_gray(rgb), sigma=sigma, preserve_range=True)
    mag = filters.sobel(g)
    lo, hi = float(mag.min()), float(mag.max())
    if hi - lo < 1e-12:
        return np.ones_like(mag)
    return 1.0 - (mag - lo) / (hi - lo)


def _drop_small(labels: np.ndarray, min_area: int) -> np.ndarray:
    """Zero out regions smaller than ``min_area`` pixels."""
    if min_area <= 1:
        return labels
    counts = np.bincount(labels.ravel())
    keep = np.where(counts >= min_area)[0]
    bad = np.where(counts < min_area)[0]
    bad = bad[bad != 0]
    if bad.size:
        lut = np.zeros(counts.size, dtype=labels.dtype)
        lut[keep] = np.arange(1, keep.size + 1)
        lut[0] = 0
        return lut[labels]
    return labels


def _prepare(rgb: np.ndarray, work_size: int | None = 320):
    """Optionally shrink an image before the expensive methods run.

    Returns ``(small_rgb, scale)`` where ``scale`` is 1.0 when no resize
    happened.  N-Cut and mean shift are the two methods that need this.
    """
    h, w = rgb.shape[:2]
    if work_size is None or max(h, w) <= work_size:
        return rgb, 1.0
    s = work_size / max(h, w)
    small = cv2_resize(rgb, s)
    return small, s


def cv2_resize(img: np.ndarray, scale: float) -> np.ndarray:
    import cv2

    h, w = img.shape[:2]
    return cv2.resize(img, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                      interpolation=cv2.INTER_AREA)


def _upscale(labels: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize of a label image back to the original size."""
    if labels.shape[:2] == shape:
        return labels
    import cv2

    return cv2.resize(labels.astype(np.int32), (shape[1], shape[0]),
                      interpolation=cv2.INTER_NEAREST)


# ---------------------------------------------------------------------------
# 1. active contours (snakes)
# ---------------------------------------------------------------------------

def active_contour(rgb: np.ndarray, seed_box: tuple[int, int, int, int] | None = None,
                   max_num_iter: int = 300, search_radius: int = 3,
                   smooth: float = 0.25, n_points: int = 160) -> np.ndarray:
    """Snake with a hard region constraint, implemented here rather than called.

    Energy, as in Kass/Witkin/Prince/Baney, is the sum of

    * an **image term** -- every contour point steps to the darkest pixel of a
      small search window in a blurred edge map, i.e. onto the nearest strong
      gradient;
    * a **membrane term** -- a 1-2-1 curvature smoother along the contour;
    * a **re-parameterisation** -- the contour is resampled to equal arc length
      every iteration.

    The re-parameterisation plus the hard clamp to ``seed_box`` are what make
    this usable.  ``skimage.segmentation.active_contour`` has neither, and on a
    cluttered low-contrast frame the membrane term wins outright: the curve
    slides onto whichever single pixel has the lowest energy and collapses to a
    point, which is what it does here for every alpha/beta/gamma combination.
    The syllabus asks for "hard constraints", and those are precisely the two
    things that library version lacks.

    ``seed_box`` is ``(x, y, w, h)``; when omitted the centre of the frame is
    used.  Returns a 2-label image: 1 = inside the converged contour.
    """
    import cv2

    h, w = rgb.shape[:2]
    if seed_box is None:
        seed_box = (int(w * 0.38), int(h * 0.38), int(w * 0.24), int(h * 0.24))
    sx, sy, sw, sh = (int(v) for v in seed_box)
    sw, sh = max(sw, 4), max(sh, 4)
    mx, my = max(2, int(sw * 0.30)), max(2, int(sh * 0.30))   # hard-constraint slack
    x0, y0 = max(0, sx - mx), max(0, sy - my)
    x1, y1 = min(w - 1, sx + sw + mx), min(h - 1, sy + sh + my)

    # Image term: dark where a strong edge is, bright across flat areas.
    field = filters.gaussian(_edge_image(rgb), sigma=2.0, preserve_range=True)

    # Seed: an ellipse inscribed in the seed box (no corners for the membrane
    # term to fight against).
    cx0, cy0 = sx + sw / 2.0, sy + sh / 2.0
    theta = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    px = cx0 + (sw / 2.0) * np.cos(theta)
    py = cy0 + (sh / 2.0) * np.sin(theta)

    r = int(max(1, search_radius))
    offs = np.arange(-r, r + 1)
    oy, ox = np.meshgrid(offs, offs, indexing="ij")
    oy, ox = oy.ravel().astype(np.int32), ox.ravel().astype(np.int32)

    for _ in range(int(max_num_iter)):
        # --- image term: step onto the best pixel in each search window
        yy = np.clip(py.astype(np.int32)[:, None] + oy, 0, h - 1)
        xx = np.clip(px.astype(np.int32)[:, None] + ox, 0, w - 1)
        best = field[yy, xx].argmin(axis=1)
        px = np.clip(px + ox[best], x0, x1).astype(np.float64)
        py = np.clip(py + oy[best], y0, y1).astype(np.float64)

        # --- membrane term: periodic 1-2-1 curvature smoothing
        if smooth > 0:
            px = (1.0 - 2.0 * smooth) * px + smooth * (np.roll(px, 1) + np.roll(px, -1))
            py = (1.0 - 2.0 * smooth) * py + smooth * (np.roll(py, 1) + np.roll(py, -1))

        # --- hard constraint + re-parameterisation to equal arc length
        px = np.clip(px, x0, x1)
        py = np.clip(py, y0, y1)
        # s[i] = arc length from point 0 to point i (one value per point)
        seg = np.hypot(np.diff(px), np.diff(py))
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        total = cum[-1]
        if total < 1e-6:
            break
        t = np.linspace(0.0, total, n_points, endpoint=False)
        px, py = np.interp(t, cum, px), np.interp(t, cum, py)

    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [np.round(np.stack([px, py], 1)).astype(np.int32)], 1)
    if not mask.any():
        return np.zeros((h, w), np.int32)
    return mask.astype(np.int32)


# ---------------------------------------------------------------------------
# 2 & 4. split & merge, written from scratch
# ---------------------------------------------------------------------------

def quadtree_split_merge(rgb: np.ndarray, variance_threshold: float = 12.0,
                         merge_threshold: float = 12.0, min_size: int = 8,
                         do_merge: bool = True) -> np.ndarray:
    """Quadtree split-and-merge, implemented here rather than called.

    Split: recurse into the four quadrants of a region whenever its intensity
    variance exceeds ``variance_threshold`` and it is larger than ``min_size``.

    Merge: sweep the region adjacency graph and fuse neighbouring regions whose
    mean intensities differ by less than ``merge_threshold``.  Set ``do_merge``
    to False to see the split phase on its own.
    """
    gray = _to_float_gray(rgb) * 255.0
    h, w = gray.shape
    labels = np.zeros((h, w), np.int32)
    next_label = 1

    def split(x0, y0, x1, y1):
        nonlocal next_label
        patch = gray[y0:y1, x0:x1]
        if patch.size == 0:
            return
        if patch.var() > variance_threshold and (x1 - x0) > min_size and (y1 - y0) > min_size:
            xm, ym = (x0 + x1) // 2, (y0 + y1) // 2
            if xm > x0 and ym > y0:
                split(x0, y0, xm, ym)
                split(xm, y0, x1, ym)
                split(x0, ym, xm, y1)
                split(xm, ym, x1, y1)
                return
        labels[y0:y1, x0:x1] = next_label
        next_label += 1

    split(0, 0, w, h)
    if not do_merge:
        return labels
    return merge_adjacent(labels, gray, merge_threshold)


def merge_adjacent(labels: np.ndarray, gray: np.ndarray, threshold: float) -> np.ndarray:
    """Fuse 4-connected neighbouring regions with similar mean intensity."""
    # Collect the pairs of region ids that touch across a 4-connected border.
    left, right = labels[:, :-1], labels[:, 1:]
    touching = (left != right) & (left > 0) & (right > 0)
    vx, vy = left[touching], right[touching]
    pairs = np.unique(np.stack([vx, vy], 1), axis=0)
    if pairs.size == 0:
        return labels

    n = int(labels.max())
    means = ndi.mean(gray, labels, index=np.arange(1, n + 1))
    parent = {i: i for i in range(1, n + 1)}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for x, y in pairs:
        rx, ry = find(int(x)), find(int(y))
        if rx == ry:
            continue
        if abs(means[rx - 1] - means[ry - 1]) < threshold:
            parent[max(rx, ry)] = min(rx, ry)

    roots = np.array([find(i) for i in range(1, n + 1)])
    _, inv = np.unique(roots, return_inverse=True)
    lut = np.zeros(n + 1, np.int32)
    lut[1:] = inv + 1
    return lut[labels]


def region_split(rgb: np.ndarray, variance_threshold: float = 12.0,
                 min_size: int = 8) -> np.ndarray:
    """The split half on its own: quadtree recursion with no merge pass."""
    return quadtree_split_merge(rgb, variance_threshold, do_merge=False,
                                min_size=min_size)


def region_merge(rgb: np.ndarray, merge_threshold: float = 12.0) -> np.ndarray:
    """The merge half on its own, seeded by a SLIC oversegmentation.

    A pure split needs a partner to be meaningful, so this pass starts from
    400 SLIC superpixels and merges them down, which is the classical
    "region splitting and region merging" arrangement from the syllabus.
    """
    labels = segmentation.slic(rgb, n_segments=400, compactness=12.0, start_label=1,
                               channel_axis=-1)
    gray = _to_float_gray(rgb) * 255.0
    return _drop_small(merge_adjacent(labels.astype(np.int32), gray, merge_threshold), 200)


# ---------------------------------------------------------------------------
# 3. watershed on distance-transform markers
# ---------------------------------------------------------------------------

def watershed(rgb: np.ndarray, markers: np.ndarray | None = None,
              min_distance: int = 25, mask: np.ndarray | None = None) -> np.ndarray:
    """Separate touching blobs with watershed driven by distance-transform peaks.

    The distance transform of the foreground peaks in the middle of each blob;
    the local maxima of that distance map are the watershed markers, which is
    what lets two boxes that overlap in the image separate at the waist
    between them.
    """
    gray = _to_float_gray(rgb)
    if mask is None:
        # Foreground cue: the boxes are the brighter, moving objects on a dark
        # industrial background.  A heavy blur plus an Otsu threshold gives the
        # silhouette; the caller can pass a better mask (see detect.py).
        sm = ndi.gaussian_filter(gray, 3.0)
        thr = filters.threshold_otsu(sm)
        mask = sm > thr

    mask = ndi.binary_fill_holes(mask)
    dist = ndi.distance_transform_edt(mask)
    dist = ndi.gaussian_filter(dist, 1.0)

    if markers is None:
        peaks = feature_peaks(dist, labels=mask, min_distance=min_distance,
                              exclude_border=False)
        if peaks.size == 0:
            peaks = np.argwhere(dist > dist.max() * 0.6)[:4]
        markers = np.zeros(dist.shape, np.int32)
        for i, (r, c) in enumerate(peaks, start=1):
            markers[r, c] = i

    labels = segmentation.watershed(-dist, markers, mask=mask)
    return _drop_small(labels.astype(np.int32), 300)


def feature_peaks(image: np.ndarray, labels: np.ndarray, min_distance: int = 5,
                  exclude_border: bool = True) -> np.ndarray:
    """Local maxima of ``image`` that are at least ``min_distance`` apart.

    Thin wrapper over ``skimage.feature.peak_local_max`` that returns an empty
    (0, 2) array instead of raising when nothing is found.
    """
    from skimage.feature import peak_local_max

    if not np.any(labels):
        return np.empty((0, 2), int)
    try:
        return peak_local_max(image, labels=labels.astype(int),
                              min_distance=min_distance, exclude_border=exclude_border)
    except ValueError:
        return np.empty((0, 2), int)


# ---------------------------------------------------------------------------
# 5. Felzenszwalb graph-based segmentation
# ---------------------------------------------------------------------------

def felzenszwalb(rgb: np.ndarray, scale: float = 120.0, sigma: float = 0.8,
                 min_size: int = 400) -> np.ndarray:
    """Graph-based segmentation by Felzenszwalb & Huttenlocher.

    Treats the image as a graph over pixels with edge weights from intensity
    differences, then does region merging with an internal-dissimilarity
    threshold that scales with region size.  Call :func:`show_region_graph` to
    actually look at the resulting region adjacency graph.
    """
    labels = segmentation.felzenszwalb(rgb, scale=scale, sigma=sigma,
                                       min_size=min_size, channel_axis=-1)
    return _drop_small(labels.astype(np.int32), 300)


def show_region_graph(rgb: np.ndarray, labels: np.ndarray, ax=None, path=None):
    """Draw the region adjacency graph over the image and save it."""
    rag = graph.rag_mean_color(rgb, labels, mode="similarity", sigma=255.0)
    lc = graph.show_rag(labels, rag, rgb, border_color="white", edge_width=1.2,
                        edge_cmap="magma", img_cmap="bone", ax=ax)
    if path is not None:
        import matplotlib.pyplot as plt

        plt.savefig(path, dpi=130, bbox_inches="tight")
    return rag, lc


# ---------------------------------------------------------------------------
# 6. mean shift / mode finding
# ---------------------------------------------------------------------------

def mean_shift(rgb: np.ndarray, quantile: float = 0.05,
               work_size: int = 260, min_size: int = 400) -> np.ndarray:
    """Mean-shift clustering in colour+position space.

    Each pixel becomes a 5-D point ``(R, G, B, y/2, x/2)``; mean shift slides a
    ball of radius ``bandwidth`` uphill to the nearest mode, and the converged
    mode index is the label.  Position is included with a spatial weight so the
    clustering respects locality -- without it this degenerates into a global
    colour histogram.  The bandwidth is estimated from the data with
    ``estimate_bandwidth`` at the given quantile rather than hard-coded.
    """
    from sklearn.cluster import estimate_bandwidth

    small, scale = _prepare(rgb, work_size)
    h, w = small.shape[:2]
    flat = small.reshape(-1, 3).astype(np.float64)
    ys, xs = np.mgrid[0:h, 0:w]
    pos = np.stack([ys.ravel(), xs.ravel()], 1).astype(np.float64) * 0.5
    feats = np.hstack([flat, pos])

    bandwidth = estimate_bandwidth(feats, quantile=quantile, random_state=0)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, max_iter=300,
                   min_bin_freq=10)
    labels = ms.fit_predict(feats).reshape(h, w).astype(np.int32)
    labels = _relabel_sequential(labels)
    labels = _upscale(_drop_small(labels, int(min_size / max(scale * scale, 1e-6))),
                      rgb.shape[:2])
    return labels


def _relabel_sequential(labels: np.ndarray) -> np.ndarray:
    """Map arbitrary cluster ids to 1..K, 0 for the smallest cluster."""
    uniq, inv = np.unique(labels, return_inverse=True)
    out = inv.reshape(labels.shape).astype(np.int32) + 1
    return out


# ---------------------------------------------------------------------------
# 7. normalized cut
# ---------------------------------------------------------------------------

def normalized_cut(rgb: np.ndarray, n_segments: int = 220, work_size: int = 220,
                   thresh: float = 0.05, num_cuts: int = 6,
                   min_size: int = 60) -> np.ndarray:
    """Spectral N-Cut on a superpixel region adjacency graph.

    Shi & Malik's normalized cut: repeatedly split the region graph so as to
    minimise ``cut(A,B) * (1/vol(A) + 1/vol(B))``.  Each split is an eigenvector
    problem on the Laplacian of the superpixel graph, which is why this runs on
    a downsampled image -- the assignment explicitly allows that.

    The RAG edges are weighted by *colour distance* rather than similarity.  A
    similarity weighting makes every edge tiny, the eigenvector is then constant
    over the whole graph, and N-Cut degenerates to "one region"; distance is
    what the method actually needs.  ``min_size`` is counted in pixels of the
    *downsampled* image, which is why it is much smaller than the 300 px used
    by the full-resolution methods.
    """
    small, scale = _prepare(rgb, work_size)
    labels = segmentation.slic(small, n_segments=n_segments, compactness=10.0,
                               start_label=1, channel_axis=-1)
    labels = labels.astype(np.int32)
    rag = graph.rag_mean_color(small.astype(np.float64) / 255.0, labels,
                               mode="distance", sigma=255.0)
    cut = graph.cut_normalized(labels, rag, thresh=thresh, num_cuts=num_cuts)
    # cut_normalized leaves label 0 unused -- every pixel lands in some region.
    cut = _drop_small(cut.astype(np.int32), min_size)
    return _upscale(cut, rgb.shape[:2])


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

METHODS: dict[str, callable] = {
    "active_contour": active_contour,
    "quadtree_split_merge": quadtree_split_merge,
    "watershed": watershed,
    "felzenszwalb": felzenszwalb,
    "mean_shift": mean_shift,
    "normalized_cut": normalized_cut,
}

PRETTY = {
    "active_contour": "1. Active contour (snake)",
    "quadtree_split_merge": "2. Split & merge (quadtree)",
    "watershed": "3. Watershed (distance markers)",
    "region_split": "4a. Region splitting",
    "region_merge": "4b. Region merging",
    "felzenszwalb": "5. Felzenszwalb (graph-based)",
    "mean_shift": "6. Mean shift (mode finding)",
    "normalized_cut": "7. Normalized cut (spectral)",
}

# The two halves of method 2 are also exposed as their own passes so the
# comparison can show the split and merge stages separately.
STANDALONE = {
    "region_split": region_split,
    "region_merge": region_merge,
}

ALL_METHODS: dict[str, callable] = {**METHODS, **STANDALONE}


def get_method(name: str) -> callable:
    """Look a method up by the key used in ``METHODS`` / ``PRETTY``.

    Raises ``KeyError`` naming the valid keys rather than failing later with an
    obscure ``NoneType is not callable``.
    """
    try:
        return ALL_METHODS[name]
    except KeyError:
        raise KeyError(f"unknown method {name!r}; expected one of "
                       f"{sorted(ALL_METHODS)}") from None


def run_all(rgb: np.ndarray, names: list[str] | None = None,
            **kwargs) -> dict[str, np.ndarray]:
    """Run the requested methods, skipping (with a warning) any that fail.

    Extra keyword arguments are forwarded to every method, so a caller can pass
    ``seed_box=...`` for the snake without it breaking the other seven.
    """
    names = names or list(PRETTY)
    out: dict[str, np.ndarray] = {}
    for name in names:
        try:
            fn = get_method(name)
        except KeyError as exc:
            print(f"  !! {exc}")
            continue
        try:
            out[name] = fn(rgb, **kwargs)
        except TypeError:
            # Method does not accept the extra kwargs; call it plainly.
            out[name] = fn(rgb)
        except Exception as exc:  # a comparison grid is more useful than a crash
            print(f"  !! {name} failed: {type(exc).__name__}: {exc}")
    return out
