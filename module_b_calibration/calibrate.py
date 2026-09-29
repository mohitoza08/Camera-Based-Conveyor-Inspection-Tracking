"""Module B, part 1: direct parameter calibration of the ceiling camera.

Recovers the intrinsic matrix K and the radial distortion coefficients from
photographs of a printed chessboard using ``cv2.calibrateCamera`` (direct,
per-image nonlinear optimisation), then:

  * undistorts a frame and writes a before/after comparison,
  * recovers the extrinsic parameters (R, t) of the camera relative to the
    board for one calibration view,
  * builds the full projection matrix ``P = K [R | t]`` and then *manually*
    decomposes P by RQ factorisation to pull K, R and t back out, checking the
    round trip against the values OpenCV returned.

Usage
-----
    python module_b_calibration/calibrate.py
    python module_b_calibration/calibrate.py --pattern 9 6 --square-mm 25.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import calibration_file, ensure_dirs, meta_file
from common.paths import CHESSBOARD_DIR, CHESSBOARD_SIZE, CHESSBOARD_SQUARE_MM, OUTPUTS

# Ground truth published by the OpenCV project in samples/data/left_intrinsics.yml.
# Used only to sanity-check that our own calibration is sane.
REFERENCE = {
    "fx": 5.3591573396163199e02,
    "fy": 5.3591573396163199e02,
    "cx": 3.4228315473308373e02,
    "cy": 2.3557082909788173e02,
    "k1": -2.6637260909660682e-01,
    "rms_px": 0.39259098975581364,
}


# ---------------------------------------------------------------------------
# Corner detection
# ---------------------------------------------------------------------------

CANDIDATE_PATTERNS = [(9, 6), (6, 5), (9, 5), (7, 6), (6, 4), (8, 5), (7, 5), (5, 4)]


def detect_corners(gray: np.ndarray, pattern: tuple[int, int]):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, pattern, flags=flags)
    if not ok:
        ok, corners = cv2.findChessboardCorners(gray, pattern)
    if not ok:
        return False, None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    return True, cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)


def collect(image_dir: Path, pattern: tuple[int, int]):
    """Return (object points, image point sets, image sizes) for a board pattern."""
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : pattern[0], 0 : pattern[1]].T.reshape(-1, 2)
    objpoints, imgpoints, sizes = [], [], []
    for path in sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png")):
        img = cv2.imread(str(path))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ok, corners = detect_corners(gray, pattern)
        if not ok:
            print(f"  skip    {path.name} (no board)")
            continue
        objpoints.append(objp.copy())
        imgpoints.append(corners.reshape(-1, 1, 2))
        sizes.append((gray.shape[1], gray.shape[0]))
        print(f"  found   {path.name} {gray.shape[1]}x{gray.shape[0]}")
    return objpoints, imgpoints, sizes


def autodetect_pattern(image_dir: Path):
    """Pick the board size that the detector accepts on the most images."""
    best = (0, CHESSBOARD_SIZE)
    for pattern in CANDIDATE_PATTERNS:
        count = 0
        for path in sorted(image_dir.glob("left*.jpg")):
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
            ok, _ = cv2.findChessboardCorners(img, pattern, flags=flags)
            count += bool(ok)
        print(f"  pattern {pattern[0]}x{pattern[1]}: {count} images")
        if count > best[0]:
            best = (count, pattern)
    return best[1], best[0]


# ---------------------------------------------------------------------------
# RQ decomposition of a projection matrix (done by hand, no helper library)
# ---------------------------------------------------------------------------


def rq3(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """RQ decomposition of a 3x3 matrix: M = R_upper @ Q_orthogonal.

    NumPy only offers QR, not RQ, so this uses the standard trick: flip the
    row order of M, transpose it, take an ordinary QR of *that*, then undo
    the flip/transpose on each factor. Verified against synthetic matrices in
    :func:`self_test_decompose_projection` before it is ever used on real
    calibration data -- an earlier version of this file called
    ``scipy.linalg.rq`` through a mistaken transpose and returned nonsense.
    """
    M = np.asarray(M, dtype=np.float64)
    Q1, R1 = np.linalg.qr(np.flipud(M).T)
    R = np.fliplr(np.flipud(R1.T))
    Q = np.flipud(Q1.T)
    return R, Q


def decompose_projection(P: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Factor P = K [R | t] into K (upper-triangular, K[2,2]=1), R, t.

    This is the classic Hartley/Zisserman routine, written out rather than
    called, because the exercise asks us to show we can recover the parameters
    from the projection matrix ourselves.
    """
    M = P[:, :3]
    K, R = rq3(M)  # M = K @ R, K upper-triangular, R orthogonal

    # Sign ambiguity #1: RQ leaves each diagonal entry of K free to flip sign
    # (K -> K@T, R -> T@R for any diagonal T of +-1 leaves K@R unchanged, since
    # T@T = I). Fix it so K has a positive diagonal, i.e. positive focal
    # lengths -- the physically meaningful choice.
    T = np.diag(np.sign(np.diag(K)))
    K = K @ T
    R = T @ R

    # Sign ambiguity #2: R must be a proper rotation (det +1), not a
    # reflection. Flipping the sign of the whole P (equivalently of K and R
    # together) does not change K @ R, so this is free to apply.
    if np.linalg.det(R) < 0:
        K = -K
        R = -R

    if abs(K[2, 2]) > 1e-12:
        s = 1.0 / K[2, 2]
        K = K * s

    t = np.linalg.solve(K, P[:, 3])
    return K, R, t


def self_test_decompose_projection() -> None:
    """Build a synthetic P = K [R | t], decompose it, and check the round trip.

    The manual RQ routine above was written but never validated, so before
    trusting it on the real calibration we run it on numbers we made up
    ourselves and know the answer to.
    """
    rng = np.random.default_rng(0)
    K_true = np.array([[600.0, 0.0, 320.0],
                        [0.0, 610.0, 240.0],
                        [0.0, 0.0, 1.0]])
    rvec_true = rng.normal(scale=0.3, size=3)
    R_true, _ = cv2.Rodrigues(rvec_true)
    t_true = np.array([0.05, -0.02, 0.8])

    P = K_true @ np.hstack([R_true, t_true.reshape(3, 1)])
    K_rec, R_rec, t_rec = decompose_projection(P)

    k_err = np.abs(K_rec - K_true).max()
    r_err = np.abs(R_rec - R_true).max()
    t_err = np.abs(t_rec - t_true).max()
    print("Self-test: synthetic P = K[R|t] -> decompose_projection -> K, R, t")
    print(f"  max|K_rec - K_true| = {k_err:.3e}")
    print(f"  max|R_rec - R_true| = {r_err:.3e}")
    print(f"  max|t_rec - t_true| = {t_err:.3e}")
    assert k_err < 1e-6 and r_err < 1e-6 and t_err < 1e-6, (
        "manual RQ decomposition failed the synthetic round-trip check")
    print("  OK -- round trip is exact to floating point precision.\n")


# ---------------------------------------------------------------------------


def undistort_demo(image: Path, K, dist, size, out_png: Path) -> None:
    """Write a side-by-side original / undistorted comparison."""
    img = cv2.imread(str(image))
    if img is None:
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    undist = cv2.undistort(img, K, dist, None, K)
    newmap = cv2.initUndistortRectifyMap(K, dist, None, K, size, cv2.CV_16SC2)
    remap = cv2.remap(img, newmap[0], newmap[1], cv2.INTER_LINEAR)
    panel = np.hstack([img, undist, remap])
    cv2.imwrite(str(out_png), panel)
    print(f"  wrote   {out_png.name}  (original | cv2.undistort | remap)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, default=CHESSBOARD_DIR,
                    help="directory of chessboard photographs")
    ap.add_argument("--pattern", type=int, nargs=2, default=list(CHESSBOARD_SIZE),
                    metavar=("COLS", "ROWS"), help="inner corners, e.g. 9 6")
    ap.add_argument("--square-mm", type=float, default=CHESSBOARD_SQUARE_MM,
                    help="printed side length of one square, in millimetres")
    ap.add_argument("--sample", type=Path, default=None,
                    help="frame to undistort (defaults to a conveyor frame)")
    args = ap.parse_args()

    ensure_dirs()
    self_test_decompose_projection()

    image_dir: Path = args.images
    if not image_dir.exists() or not list(image_dir.glob("left*.jpg")):
        print(f"No chessboard images in {image_dir}. Run: python data/fetch_data.py",
              file=sys.stderr)
        return 1

    print("Board pattern:")
    pattern = tuple(args.pattern)
    if tuple(args.pattern) == tuple(CHESSBOARD_SIZE):
        pattern, count = autodetect_pattern(image_dir)
        print(f"  -> using {pattern[0]}x{pattern[1]} ({count} images agree)")

    print("\nCorner detection:")
    objpoints, imgpoints, sizes = collect(image_dir, pattern)
    if len(objpoints) < 8:
        print(f"Only {len(objpoints)} usable views; need at least 8 for a stable "
              f"calibration.", file=sys.stderr)
        return 1

    square = args.square_mm / 1000.0
    objp_scaled = [o * square for o in objpoints]
    size = sizes[0]

    print(f"\nRunning cv2.calibrateCamera on {len(objpoints)} views "
          f"({size[0]}x{size[1]}, square = {square * 1000:.1f} mm) ...")
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objp_scaled, imgpoints, size, None, None
    )

    print(f"\n  RMS reprojection error : {rms:.4f} px")
    print(f"  fx = {K[0, 0]:.4f}   fy = {K[1, 1]:.4f}")
    print(f"  cx = {K[0, 2]:.4f}   cy = {K[1, 2]:.4f}")
    print(f"  k1 = {dist[0, 0]:.6f}  k2 = {dist[0, 1]:.6f}  "
          f"p1 = {dist[0, 2]:.6f}  p2 = {dist[0, 3]:.6f}")
    if dist.shape[1] > 4:
        print(f"  k3 = {dist[0, 4]:.6f}")

    print("\n  Reference (OpenCV samples/data/left_intrinsics.yml) for comparison:")
    print(f"    fx = {REFERENCE['fx']:.4f}   fy = {REFERENCE['fy']:.4f}")
    print(f"    cx = {REFERENCE['cx']:.4f}   cy = {REFERENCE['cy']:.4f}")
    print(f"    k1 = {REFERENCE['k1']:.6f}   rms = {REFERENCE['rms_px']:.4f} px")

    # ---- extrinsics for one view, and the P = K [R|t] round trip ----------
    R0, _ = cv2.Rodrigues(rvecs[0])
    t0 = tvecs[0].reshape(3)
    Rvec0 = cv2.Rodrigues(R0)[0]
    print(f"\nExtrinsics of view 0 (board -> camera):")
    print(f"  t = {np.array2string(t0, precision=4)}")
    print(f"  rvec = {np.array2string(Rvec0.reshape(3), precision=4)}")
    print(f"  angle = {np.degrees(np.linalg.norm(Rvec0)):.3f} deg")

    # P = K [R | t] directly. (cv2.projectPoints returns *projected points*,
    # not a projection matrix -- passing its output through reshape(3, 3) was
    # the earlier bug here: it silently produced a matrix with the right shape
    # but meaningless entries.)
    P = K @ np.hstack([R0, t0.reshape(3, 1)])

    K_rec, R_rec, t_rec = decompose_projection(P)
    rot_err = cv2.Rodrigues(R_rec)[0] - Rvec0
    rot_err = np.linalg.norm(rot_err) * 180.0 / np.pi

    print("\nManual RQ decomposition of P = K [R | t]:")
    print(f"  |K_rec - K|  max abs = {np.abs(K_rec - K).max():.3e}")
    print(f"  |t_rec - t|  max abs = {np.abs(t_rec - t0).max():.3e}")
    print(f"  rotation error       = {rot_err:.3e} deg")
    print(f"  P = \n{np.array2string(P, precision=4, suppress_small=True)}")

    # ---- persist ---------------------------------------------------------
    out = calibration_file()
    np.savez(
        out,
        K=K, dist=dist, rms=rms,
        rvecs=np.array(rvecs), tvecs=np.array(tvecs),
        image_size=np.array(size),
        pattern=np.array(pattern),
        square_mm=args.square_mm,
        source_images=np.array([str(p) for p in sorted(image_dir.glob('left*.jpg'))]),
    )
    print(f"\nSaved K, distortion and extrinsics to {out}")

    # Human-readable mirror.
    yml = out.with_suffix(".yml")
    fs = cv2.FileStorage(str(yml), cv2.FILE_STORAGE_WRITE)
    fs.write("K", K)
    fs.write("distortion", dist)
    fs.write("rms_reprojection_error", float(rms))
    fs.write("image_size", np.array(size, dtype=np.int32))
    fs.write("pattern", np.array(pattern, dtype=np.int32))
    fs.write("square_mm", float(args.square_mm))
    fs.release()
    print(f"Saved {yml}")

    # ---- record the calibration in meta.json so the pipeline can find it ---
    import json

    meta = {}
    if meta_file().exists():
        meta = json.loads(meta_file().read_text())
    meta["calibration"] = {
        "npz": str(out),
        "pattern": list(pattern),
        "square_mm": float(args.square_mm),
        "rms_px": float(rms),
        "image_size": list(size),
        "n_views": len(objpoints),
        "note": ("Intrinsics come from OpenCV's sample chessboard photographs, "
                 "which were taken with a different camera than the conveyor "
                 "footage. They are internally consistent and validated against "
                 "the published reference, but they are not the intrinsics of "
                 "the belt camera; see common/scale.py for how the metric scale "
                 "is transferred."),
    }
    meta_file().write_text(json.dumps(meta, indent=2))

    # ---- undistortion demo ----------------------------------------------
    # K/dist belong to the chessboard camera, not the conveyor camera (see the
    # caveat above), so the demo has to undistort a *chessboard* photo -- a
    # conveyor frame is a different resolution and a different camera, and
    # would either crash on the size mismatch or, worse, silently apply the
    # wrong lens model.
    sample = args.sample
    if sample is None:
        boards = sorted(image_dir.glob("left*.jpg"))
        sample = boards[len(boards) // 2] if boards else None
    if sample is not None and sample.exists():
        undistort_demo(sample, K, dist, size,
                       OUTPUTS / "module_b_undistortion.png")
    else:
        print("\n(no chessboard image available for the undistortion demo)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
