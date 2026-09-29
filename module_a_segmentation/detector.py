"""Per-frame box detection built on top of Module A's watershed.

The seven methods in ``segmentation.py`` are what the assignment asks us to
compare.  Feeding a raw conveyor frame to any of them cold produces a hundred
regions, because the scene is a dark industrial hall with shelving, the robot
arm and the belt itself all competing for the same thresholds.  So for the
pipeline we add one classical front-end ahead of the winning method:

    temporal median background  ->  absolute difference  ->  box silhouette
                                                              |
                                                      watershed on the
                                                      distance transform

The background is estimated from a sliding window of frames either side of the
current one, which keeps it valid even though the three concatenated source
recordings have visibly different overall brightness.  The silhouette is then
refined by watershed, which is what separates the box from its own motion
blur and shadow instead of returning one lumped blob.

Nothing here is learned; it is all median filtering, differencing, thresholding
and the Module A watershed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import filters, measure, segmentation

__all__ = ["Box", "ConveyorDetector", "mask_to_boxes", "polish_mask"]


@dataclass
class Box:
    """One detected box in a single frame, in pixels."""

    contour: np.ndarray          # (N, 2) float, x/y pixel coordinates
    bbox: tuple[int, int, int, int]  # x, y, w, h
    centroid: tuple[float, float]
    area_px: float
    orientation_rad: float
    corners: np.ndarray = field(default_factory=lambda: np.empty((0, 4, 2)))

    @property
    def min_area_side(self) -> np.ndarray:
        """The four polygon corners of the minimum-area rectangle, ordered tl,tr,br,bl."""
        if self.corners.size:
            return self.corners
        (x, y), (w, h), angle = cv2.minAreaRect(self.contour)
        rect = ((x, y), (w, h), angle)
        box = cv2.boxPoints(rect)
        return _order_quad(box).reshape(1, 4, 2)


def _as_gray(img: np.ndarray) -> np.ndarray:
    """Coerce a frame or frame stack to single-channel uint8 grey.

    Handles a single grey image ``(H, W)``, a single colour image ``(H, W, C)``
    with ``C`` in 3/4, a grey stack ``(N, H, W)`` and a colour stack
    ``(N, H, W, C)``.  The 3-D case is ambiguous, so it is disambiguated on the
    channel axis rather than guessed from ``ndim`` alone -- a grey stack must
    not be mistaken for a colour image and sent to ``cvtColor``.
    """
    a = np.asarray(img)
    if a.ndim == 2:
        return a.astype(np.uint8, copy=False)
    if a.ndim == 3:
        if a.shape[-1] in (3, 4):
            return cv2.cvtColor(a, cv2.COLOR_BGR2GRAY if a.shape[-1] == 3 else cv2.COLOR_BGRA2GRAY)
        return a.astype(np.uint8, copy=False)          # (N, H, W) grey stack
    if a.ndim == 4:
        code = cv2.COLOR_BGR2GRAY if a.shape[-1] == 3 else (
            cv2.COLOR_BGRA2GRAY if a.shape[-1] == 4 else None)
        if code is None:
            return a[..., 0].astype(np.uint8, copy=False)   # (N, H, W, 1)
        return np.stack([cv2.cvtColor(f, code) for f in a])
    raise ValueError(f"cannot interpret array of shape {a.shape} as a grey frame")


def _order_quad(quad: np.ndarray) -> np.ndarray:
    """Order a 4-point quad as [tl, tr, br, bl] by angle about its centroid."""
    pts = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    c = pts.mean(axis=0)
    order = np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))
    return pts[order]


def polish_mask(mask: np.ndarray, open_px: int = 3, close_px: int = 9) -> np.ndarray:
    """Clean a raw silhouette without moving its boundary much."""
    m = mask.astype(np.uint8)
    if open_px > 0:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px, open_px)))
    if close_px > 0:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px)))
    return ndi.binary_fill_holes(m.astype(bool))


def mask_to_boxes(mask: np.ndarray, min_area: int = 1500, max_area_frac: float = 0.35,
                  rectangularity: float = 0.0) -> list[Box]:
    """Extract box candidates from a binary silhouette.

    ``rectangularity`` optionally requires the candidate to fill
    ``area / minAreaRect area`` above a threshold, which rejects the long thin
    bands the moving belt produces.
    """
    m = mask.astype(np.uint8)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(m, 8)
    total = m.size
    out: list[Box] = []
    for i in range(1, n):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area_frac * total:
            continue
        comp = (labels == i).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        contour = max(cnts, key=cv2.contourArea)
        rect = cv2.minAreaRect(contour)
        rect_area = cv2.contourArea(cv2.boxPoints(rect))
        if rectangularity > 0 and rect_area > 0 and area / rect_area < rectangularity:
            continue
        (cx, cy), (rw, rh), angle = rect
        x, y, w, h = (int(v) for v in stats[i, :4])
        quad = _order_quad(cv2.boxPoints(rect))
        out.append(
            Box(
                contour=contour.reshape(-1, 2).astype(np.float64),
                bbox=(x, y, w, h),
                centroid=(float(centroids[i, 0]), float(centroids[i, 1])),
                area_px=area,
                orientation_rad=float(np.radians(angle)),
                corners=quad.reshape(1, 4, 2),
            )
        )
    out.sort(key=lambda b: -b.area_px)
    return out


class ConveyorDetector:
    """Sliding-window background model plus a watershed box extractor.

    Parameters
    ----------
    window
        Number of frames either side of the current one used to build the
        median background.  Larger is more robust but lags faster motion.
    thresh
        Absolute-difference threshold, in grey levels, that marks foreground.
    """

    def __init__(self, window: int = 22, thresh: float = 16.0,
                 min_distance: int = 20, min_area: int = 1500,
                 use_watershed: bool = True) -> None:
        self.window = int(window)
        self.thresh = float(thresh)
        self.min_distance = int(min_distance)
        self.min_area = int(min_area)
        self.use_watershed = use_watershed
        self._bg: np.ndarray | None = None
        self._bg_span: tuple[int, int] = (0, 0)

    # ------------------------------------------------------------ background

    def set_background(self, frames: np.ndarray, index: int | None = None) -> None:
        """Precompute a median background for one frame index, or for the whole clip."""
        frames = _as_gray(frames)
        n = len(frames)
        if index is None:
            self._bg = np.median(frames, axis=0).astype(np.uint8)
            self._bg_span = (0, n)
        else:
            self._update_background(frames, index)

    def _update_background(self, frames: np.ndarray, index: int) -> None:
        frames = _as_gray(frames)
        a = max(0, index - self.window)
        b = min(len(frames), index + self.window + 1)
        if b - a < 5:
            a, b = 0, len(frames)
        self._bg = np.median(frames[a:b], axis=0).astype(np.uint8)
        self._bg_span = (a, b)

    def background_for(self, frames: np.ndarray, index: int) -> np.ndarray:
        a = max(0, index - self.window)
        b = min(len(frames), index + self.window + 1)
        if self._bg is None or self._bg_span != (a, b):
            self._update_background(frames, index)
        return self._bg

    # ------------------------------------------------------------- detection

    def foreground(self, frame_gray: np.ndarray, background: np.ndarray) -> np.ndarray:
        diff = cv2.absdiff(_as_gray(frame_gray), background)
        diff = cv2.GaussianBlur(diff, (5, 5), 0)
        return polish_mask(diff > self.thresh)

    def detect(self, frame_bgr: np.ndarray, background: np.ndarray) -> tuple[list[Box], np.ndarray]:
        """Detect boxes in one frame.  Returns ``(boxes, silhouette)``."""
        gray = _as_gray(frame_bgr)
        fg = self.foreground(gray, background)
        if not fg.any():
            return [], fg

        labels = self._watershed_labels(fg) if self.use_watershed else fg.astype(np.int32)
        boxes: list[Box] = []
        for lab in np.unique(labels):
            if lab == 0:
                continue
            comp = labels == lab
            if comp.sum() < self.min_area:
                continue
            for b in mask_to_boxes(comp, min_area=self.min_area):
                boxes.append(b)
        boxes.sort(key=lambda b: -b.area_px)
        return boxes, fg

    def detect_from_index(self, frames: np.ndarray, index: int,
                          color: bool = False) -> tuple[list[Box], np.ndarray]:
        """Detect in ``frames[index]`` using the background built around it.

        ``frames`` is a grey or colour stack; pass ``color=True`` when it is
        colour and you want the detection run on the grey version anyway.
        """
        frame = frames[index]
        if color:
            frame = _as_gray(frame)
        return self.detect(frame, self.background_for(frames, index))

    def _watershed_labels(self, fg: np.ndarray) -> np.ndarray:
        """Module A's watershed: distance-transform peaks as markers."""
        dist = ndi.gaussian_filter(ndi.distance_transform_edt(fg), 1.0)
        from skimage.feature import peak_local_max

        try:
            peaks = peak_local_max(dist, labels=fg.astype(int),
                                   min_distance=self.min_distance, exclude_border=False)
        except ValueError:
            peaks = np.empty((0, 2), int)
        markers = np.zeros(dist.shape, np.int32)
        for i, (r, c) in enumerate(peaks, start=1):
            markers[r, c] = i
        if markers.max() == 0:
            return fg.astype(np.int32)
        ws = segmentation.watershed(-dist, markers, mask=fg)
        # Watershed expands a marker to the whole mask even for markers on
        # weak peaks; keep only the regions that are actually compact.
        keep = np.zeros_like(ws, dtype=bool)
        for lab in np.unique(ws):
            if lab == 0:
                continue
            comp = ws == lab
            area = comp.sum()
            if area < self.min_area:
                continue
            rect = cv2.minAreaRect(
                np.column_stack(np.nonzero(comp.T)[::-1]).astype(np.int32))
            rect_area = cv2.contourArea(cv2.boxPoints(rect))
            fill = area / rect_area if rect_area > 0 else 0.0
            if fill >= 0.45:
                keep |= comp
        return (ws * keep).astype(np.int32)
