"""Pixel -> centimetre conversion on the belt plane.

The scale model is deliberately explicit about *where* its numbers come from,
because the calibration photographs in Module B were taken with a different
camera than the conveyor footage. What carries over between the two is not the
intrinsic matrix K -- it is the observation that a known real-world length
projects to a measurable pixel length, and that a planar scene can be lifted
back to metric coordinates once the plane is known.

Three things live here:

* :class:`BeltScaleModel` -- fits the belt plane from one known quadrilateral
  (a box footprint whose real size is published) and converts any pixel on that
  plane to centimetres. This is the model the pipeline uses.
* the four camera models required by Module B item 4, used by ``measure_box.py``
* :func:`cm_per_px`, the first-order scalar scale at a point, which is what
  Module C multiplies an optical-flow displacement by.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "BeltScaleModel",
    "MODEL_NAMES",
    "estimate_size_all_models",
    "orthographic_size",
    "weak_perspective_size",
    "affine_size",
    "perspective_size",
]

MODEL_NAMES = (
    "orthographic",
    "weak_perspective",
    "affine",
    "full_perspective",
)


def _order_quad(quad: np.ndarray) -> np.ndarray:
    """Return a 4x2 quad ordered [tl, tr, br, bl]."""
    pts = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    centre = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0])
    return pts[np.argsort(angles)]


def _undistort_normalized(K: np.ndarray, dist: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Pixel coordinates -> normalised (undistorted, K^-1 applied) image coords."""
    import cv2

    cam = np.asarray(K, dtype=np.float64)
    d = np.zeros(8) if dist is None else np.asarray(dist, dtype=np.float64).ravel()
    src = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    norm = cv2.undistortPoints(src, cam, d).reshape(-1, 2)
    return norm


def _undistort_pixels(K: np.ndarray, dist: np.ndarray | None, pts: np.ndarray) -> np.ndarray:
    """Pixel coordinates -> ideal (distortion-removed) *pixel* coordinates.

    Unlike :func:`_undistort_normalized` this stays in pixel units (passes
    ``P=K`` to ``cv2.undistortPoints``), because :meth:`BeltScaleModel.
    fit_from_reference` needs a genuine projective homography between pixel
    coordinates and metric coordinates -- see that method's docstring for why.
    """
    import cv2

    if dist is None:
        return np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    cam = np.asarray(K, dtype=np.float64)
    d = np.asarray(dist, dtype=np.float64).ravel()
    src = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.undistortPoints(src, cam, d, P=cam)
    return out.reshape(-1, 2)


class BeltScaleModel:
    """Maps pixels on a known plane to centimetres on that plane.

    Parameters
    ----------
    K, dist
        Intrinsics and distortion of the camera the *scale reference* was
        measured in.  When no calibration is available, pass ``K=None`` and the
        model degenerates gracefully to an isotropic scale factor.
    """

    def __init__(self, K: np.ndarray | None = None, dist: np.ndarray | None = None,
                 image_size: tuple[int, int] | None = None) -> None:
        self.K = None if K is None else np.asarray(K, dtype=np.float64)
        self.dist = None if dist is None else np.asarray(dist, dtype=np.float64)
        self.image_size = image_size
        self.r1: np.ndarray | None = None
        self.r2: np.ndarray | None = None
        self.r3: np.ndarray | None = None
        self.t: np.ndarray | None = None
        self._iso_scale: float | None = None

    # ---------------------------------------------------------------- fitting

    def fit_from_reference(self, quad_px: np.ndarray, quad_cm: np.ndarray) -> "BeltScaleModel":
        """Fit the plane from a correspondence set of 4 image/metric points.

        ``quad_px`` are image pixels of a rectangle lying flat on the belt,
        ``quad_cm`` the same rectangle in centimetres in a belt frame whose
        z-axis is the belt normal.

        The homography ``H`` between (undistorted) pixel coordinates and
        metric plane coordinates is a genuine 2D projective homography, so it
        is related to the camera parameters by ``H^-1 = mu * K [r1 r2 t]``
        for a *single* scalar ``mu`` shared by every point (standard planar
        pose recovery, e.g. Zhang's calibration method) -- so
        ``K^-1 @ H^-1`` gives ``[r1 r2 t]`` up to that one scalar, which is
        then fixed by ``r1``/``r2`` both needing unit norm. An earlier version
        of this method instead ran every corner through
        :func:`_undistort_normalized` (which already divides out ``K``) and
        then divided by ``K`` a second time, and fed the result into
        ``np.linalg.solve`` with a shape ``np.linalg.solve`` doesn't accept
        for this use -- both are fixed here by working consistently in pixel
        units up to the homography step.
        """
        import cv2

        src = _order_quad(quad_px)
        dst = _order_quad(quad_cm)
        H, _ = cv2.findHomography(src, dst, 0)  # image px -> metric belt plane
        if H is None:
            raise ValueError("reference quadrilateral is degenerate")

        K = self.K
        if K is None:
            # No intrinsics: fall back to the mean isotropic scale.  Documented
            # in the README as the degraded path.
            side_cm = 0.5 * (np.linalg.norm(dst[1] - dst[0]) + np.linalg.norm(dst[2] - dst[1]))
            side_px = 0.5 * (np.linalg.norm(src[1] - src[0]) + np.linalg.norm(src[2] - src[1]))
            self._iso_scale = side_cm / max(side_px, 1e-9)
            return self

        src_u = _undistort_pixels(K, self.dist, src)
        Hp, _ = cv2.findHomography(src_u, dst, 0)  # undistorted px -> metric belt plane
        if Hp is None:
            raise ValueError("reference quadrilateral is degenerate")
        M = np.linalg.solve(K, np.linalg.inv(Hp))  # K^-1 H^-1 = mu * [r1 r2 t]
        lam1 = 1.0 / max(np.linalg.norm(M[:, 0]), 1e-12)
        lam2 = 1.0 / max(np.linalg.norm(M[:, 1]), 1e-12)
        lam = 0.5 * (lam1 + lam2)
        r1, r2, t = lam * M[:, 0], lam * M[:, 1], lam * M[:, 2]

        # Project onto the nearest rotation matrix (r1, r2, r1 x r2).
        R = np.column_stack([r1, r2, np.cross(r1, r2)])
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        self.r1, self.r2, self.r3 = R[:, 0].copy(), R[:, 1].copy(), R[:, 2].copy()
        self.t = t
        return self

    def fit_from_scalar(self, cm_per_px: float) -> "BeltScaleModel":
        """Degenerate fit: a single isotropic scale with no plane geometry."""
        self._iso_scale = float(cm_per_px)
        return self

    @property
    def has_plane(self) -> bool:
        return self.r1 is not None

    # ------------------------------------------------------------- projection

    def _ray(self, pts_px: np.ndarray) -> np.ndarray:
        """Undistorted pixel -> unit-direction ray in camera coordinates."""
        pts = np.asarray(pts_px, dtype=np.float64).reshape(-1, 2)
        if self.K is None:
            cam = pts - (pts.mean(axis=0) if self.image_size is None else
                         np.array([self.image_size[0] / 2, self.image_size[1] / 2]))
            return np.hstack([cam, np.ones((len(pts), 1))])
        return np.hstack([_undistort_normalized(self.K, self.dist, pts), np.ones((len(pts), 1))])

    def pixels_to_cm(self, pts_px: np.ndarray) -> np.ndarray:
        """Project pixels on the belt plane to metric (x, y) in centimetres."""
        d = self._ray(pts_px)
        if not self.has_plane:
            return d[:, :2] * (self._iso_scale or 1.0)
        num = float(self.r3 @ self.t)
        den = d @ self.r3
        den[np.abs(den) < 1e-9] = 1e-9
        X = d * (num / den)[:, None]
        return np.column_stack([X @ self.r1, X @ self.r2])

    def cm_per_px(self, pts_px: np.ndarray | None = None) -> float | np.ndarray:
        """First-order metric scale at the given pixel(s).

        Computed from the Jacobian of :meth:`pixels_to_cm`, so it correctly
        shrinks towards the image periphery the way a real perspective camera
        does.  Returns a scalar when ``pts_px`` is a single point.
        """
        if pts_px is None or np.ndim(pts_px) == 1:
            p = np.asarray(pts_px if pts_px is not None else [0.0, 0.0], dtype=np.float64)
            base = self.pixels_to_cm(p[None, :])[0]
            eps = 0.5
            dx = (self.pixels_to_cm((p + [eps, 0])[None, :])[0] - base) / eps
            dy = (self.pixels_to_cm((p + [0, eps])[None, :])[0] - base) / eps
            J = np.column_stack([dx, dy])
            s = float(np.sqrt(abs(np.linalg.det(J.T @ J))))
            return s if np.ndim(pts_px) == 1 else s
        pts = np.asarray(pts_px, dtype=np.float64).reshape(-1, 2)
        return np.array([self.cm_per_px(p) for p in pts])

    def quad_length_cm(self, quad_px: np.ndarray) -> tuple[float, float]:
        """Return (mean of opposite edge lengths, mean of the other pair), in cm."""
        q = _order_quad(quad_px)
        m = self.pixels_to_cm(q)
        e = [np.linalg.norm(m[(i + 1) % 4] - m[i]) for i in range(4)]
        return (0.5 * (e[0] + e[2]), 0.5 * (e[1] + e[3]))


# ---------------------------------------------------------------------------
# The four camera models required by Module B item 4.
#
# All four take the same inputs: a reference quadrilateral whose real size in
# centimetres is known, and a *target* quadrilateral to measure.  They differ
# only in what they assume about how the camera projects.
# ---------------------------------------------------------------------------


def orthographic_size(ref_px, ref_cm, tgt_px) -> tuple[float, float]:
    """Parallel projection. One scalar scale, applied everywhere."""
    scale = _isotropic_scale(ref_px, ref_cm)
    q = _order_quad(tgt_px)
    side_a = 0.5 * (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3]))
    side_b = 0.5 * (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1]))
    return (side_a * scale, side_b * scale)


def weak_perspective_size(ref_px, ref_cm, tgt_px, K=None) -> tuple[float, float]:
    """Orthographic plus a single global depth-ratio correction.

    The textbook weak-perspective model: project the scene onto a plane parallel
    to the image plane at a representative depth, i.e. all depths get the same
    correction, no perspective divide.  With a calibration we use the mean depth
    ratio between reference and target; without one the correction is 1.
    """
    base = orthographic_size(ref_px, ref_cm, tgt_px)
    if K is None:
        return base
    K = np.asarray(K, dtype=np.float64)
    d_ref = np.hstack([_undistort_normalized(K, None, _order_quad(ref_px)),
                       np.ones((4, 1))])
    d_tgt = np.hstack([_undistort_normalized(K, None, _order_quad(tgt_px)),
                       np.ones((4, 1))])
    # A proxy for depth: the mean of the ray direction z, which is 1 for an
    # ideal pinhole, so use the lateral spread instead -- larger spread means
    # the object subtends more of the frustum, i.e. it is nearer.
    spread_ref = float(np.mean(np.linalg.norm(d_ref[:, :2], axis=1)))
    spread_tgt = float(np.mean(np.linalg.norm(d_tgt[:, :2], axis=1)))
    if spread_ref <= 1e-9:
        return base
    # Farther -> smaller on the sensor -> a larger cm/px is needed.
    ratio = spread_ref / max(spread_tgt, 1e-9)
    return (base[0] * ratio, base[1] * ratio)


def affine_size(ref_px, ref_cm, tgt_px) -> tuple[float, float]:
    """Full 2x3 affine (6 DOF: rotation, two scales, shear, translation)
    fitted on the reference quad, applied to the target.

    Deliberately the general 6-parameter affine (``cv2.estimateAffine2D``),
    not the 4-parameter similarity transform (``cv2.estimateAffinePartial2D``,
    rotation+uniform-scale+translation only) -- the module brief asks for
    "general affine", and a similarity transform cannot represent the
    different x/y foreshortening a tilted camera actually produces.
    """
    import cv2

    A, _ = cv2.estimateAffine2D(
        _order_quad(ref_px).reshape(-1, 1, 2),
        _order_quad(ref_cm).reshape(-1, 1, 2),
        method=cv2.LMEDS,
    )
    if A is None:
        return orthographic_size(ref_px, ref_cm, tgt_px)
    m = _order_quad(tgt_px)
    mh = np.hstack([m, np.ones((4, 1))])
    metric = mh @ A.T
    e = [np.linalg.norm(metric[(i + 1) % 4] - metric[i]) for i in range(4)]
    return (0.5 * (e[0] + e[2]), 0.5 * (e[1] + e[3]))


def perspective_size(ref_px, ref_cm, tgt_px, K=None, dist=None) -> tuple[float, float]:
    """Full pinhole model: lift both quads onto the belt plane, measure there."""
    model = BeltScaleModel(K, dist)
    try:
        model.fit_from_reference(ref_px, ref_cm)
    except (ValueError, np.linalg.LinAlgError):
        return orthographic_size(ref_px, ref_cm, tgt_px)
    a, b = model.quad_length_cm(tgt_px)
    return (a, b)


def estimate_size_all_models(ref_px, ref_cm, tgt_px, K=None, dist=None) -> dict[str, tuple[float, float]]:
    """Run all four models on the same data and return their estimates."""
    return {
        "orthographic": orthographic_size(ref_px, ref_cm, tgt_px),
        "weak_perspective": weak_perspective_size(ref_px, ref_cm, tgt_px, K),
        "affine": affine_size(ref_px, ref_cm, tgt_px),
        "full_perspective": perspective_size(ref_px, ref_cm, tgt_px, K, dist),
    }


def _isotropic_scale(ref_px, ref_cm) -> float:
    sp = _order_quad(ref_px)
    sc = _order_quad(ref_cm)
    px = [np.linalg.norm(sp[(i + 1) % 4] - sp[i]) for i in range(4)]
    cm = [np.linalg.norm(sc[(i + 1) % 4] - sc[i]) for i in range(4)]
    px_mean = float(np.mean(px))
    return float(np.mean(cm)) / max(px_mean, 1e-9)
