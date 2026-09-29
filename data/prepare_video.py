"""Turn the raw I.AM. HDF5 archive into a plain video plus ground-truth metadata.

The archive stores each camera recording as a single opaque ``datalog`` byte
blob under ``SENSOR_MEASUREMENT/Reference_camera``.  Those bytes are a complete
H.264 MP4 file (ISO base media, ``ftypisom``/``avc1``), so extraction is just
"write the blob to disk".  Alongside them the archive gives us a per-frame
time vector, the camera's own intrinsic matrix, and a motion-capture ground
truth pose for each box -- the last of which is only used for validation, never
as an input to the vision pipeline.

The output video is the three viewpoint-1 toss recordings (Box006, Box007,
Box009) concatenated, so that the pipeline sees one instance of each box type
in a single run.

Usage
-----
    python data/prepare_video.py
    python data/prepare_video.py --viewpoint 2
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import ensure_dirs, meta_file, video_file
from common.paths import FRAMES_DIR, IAM_DIR

# Recording -> box, parsed from each recording's `note` attribute:
#   "Tossing Box006 on a running conveyor with industrial background with
#    viewpoint 1 for Object Tracking."
VIEWPOINT_RE = re.compile(r"Tossing\s+(\w+).*viewpoint\s+(\d)", re.S)


def find_archive() -> Path | None:
    for p in sorted(IAM_DIR.rglob("*.h5")):
        return p
    return None


def _as_text(value) -> str:
    """HDF5 attributes come back as str, bytes, or a 1-element array of either."""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, np.ndarray):
        return _as_text(value.ravel()[0]) if value.size else ""
    if isinstance(value, (list, tuple)):
        return _as_text(value[0]) if len(value) else ""
    return str(value)


def describe(f: h5py.File) -> list[dict]:
    """List every recording that carries a camera stream."""
    out = []
    for rec in sorted(k for k in f if k.startswith("Rec_")):
        sm = f[rec].get("SENSOR_MEASUREMENT")
        if sm is None:
            continue
        cam = sm.get("Reference_camera")
        if cam is None or "datalog" not in cam:
            continue
        note = f[rec].attrs.get("note", "")
        note = _as_text(note)
        m = VIEWPOINT_RE.search(note)
        out.append(
            {
                "recording": rec,
                "box": m.group(1) if m else "unknown",
                "viewpoint": int(m.group(2)) if m else 0,
                "note": note,
            }
        )
    return out


def extract_clip(f: h5py.File, rec: str, dest: Path) -> dict:
    """Write one recording's embedded MP4 to ``dest`` and describe it."""
    cam = f[f"{rec}/SENSOR_MEASUREMENT/Reference_camera"]
    blob = np.array(cam["datalog"], dtype=np.uint8).ravel()
    dest.write_bytes(blob.tobytes())

    res = cam["datalog"].attrs.get("resolution", ["", ""])
    res = res[0] if hasattr(res, "__len__") else str(res)
    width, _, height = res.partition(";")

    tvec = np.array(cam["TimeVector"], dtype=np.float64).ravel()
    K = np.array(cam["IntrinsicMatrix"], dtype=np.float64)

    cap = cv2.VideoCapture(str(dest))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return {
        "path": str(dest),
        "fps": float(fps),
        "n_frames": n,
        "width": w or int(width),
        "height": h or int(height),
        "time_vector": tvec.tolist(),
        "camera_K_archive": K.tolist(),
    }


def mocap_pose_path(f: h5py.File, rec: str, box: str) -> str | None:
    p = f.get(f"{rec}/SENSOR_MEASUREMENT/Mocap/POSTPROCESSING/{box}/transforms")
    return p.name if p is not None else None


def box_dimensions(f: h5py.File, box: str) -> list[float] | None:
    """Published real-world size of a box, in metres, straight from the archive."""
    d = f.get(f"Rec_20230309T131723Z/OBJECT/{box}/dimensions")
    if d is None:
        for rec in f:
            d = f.get(f"{rec}/OBJECT/{box}/dimensions")
            if d is not None:
                break
    if d is None:
        return None
    return (np.array(d, dtype=np.float64).ravel() * 100.0).tolist()  # cm


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--viewpoint", type=int, default=1, choices=(1, 2))
    ap.add_argument("--clips", type=int, default=3,
                    help="how many recordings to concatenate")
    args = ap.parse_args()

    ensure_dirs()
    archive = find_archive()
    if archive is None:
        print(f"No .h5 archive under {IAM_DIR}. Run: python data/fetch_data.py",
              file=sys.stderr)
        return 1
    print(f"Archive: {archive.name}")

    clips_dir = IAM_DIR / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(archive, "r") as f:
        recs = describe(f)
        print(f"\n{len(recs)} recordings with a camera stream:")
        for r in recs:
            print(f"  {r['recording']}  {r['box']:8s} viewpoint {r['viewpoint']}")

        chosen = [r for r in recs if r["viewpoint"] == args.viewpoint]
        # One recording per box type, in a stable order.
        seen: dict[str, dict] = {}
        for r in chosen:
            seen.setdefault(r["box"], r)
        chosen = sorted(seen.values(), key=lambda r: r["box"])[: args.clips]
        if not chosen:
            print(f"No recordings for viewpoint {args.viewpoint}", file=sys.stderr)
            return 1

        print(f"\nUsing viewpoint {args.viewpoint}: "
              f"{', '.join(c['box'] for c in chosen)}")

        out_clips = []
        for r in chosen:
            dest = clips_dir / f"{r['box']}_vp{args.viewpoint}.mp4"
            info = extract_clip(f, r["recording"], dest)
            info.update({"box": r["box"], "viewpoint": r["viewpoint"],
                         "recording": r["recording"]})
            info["dimensions_cm"] = box_dimensions(f, r["box"])
            info["ground_truth_mocap"] = mocap_pose_path(f, r["recording"], r["box"])
            out_clips.append(info)
            print(f"  {dest.name}: {info['n_frames']} frames @ {info['fps']:.2f} fps, "
                  f"{info['width']}x{info['height']}, box {info['dimensions_cm']} cm")

    # ---- concatenate ------------------------------------------------------
    dst = video_file()
    writer = None
    segments = []
    try:
        for info in out_clips:
            cap = cv2.VideoCapture(info["path"])
            w, h = info["width"], info["height"]
            if writer is None:
                writer = cv2.VideoWriter(
                    str(dst), cv2.VideoWriter_fourcc(*"mp4v"),
                    out_clips[0]["fps"], (w, h),
                )
            read = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame.shape[1] != w or frame.shape[0] != h:
                    frame = cv2.resize(frame, (w, h))
                writer.write(frame)
                read += 1
            cap.release()
            info["frames_written"] = read
            info["start_frame"] = sum(s["frames_written"] for s in segments)
            segments.append(info)
            print(f"  wrote   {read} frames of {info['box']}")
    finally:
        if writer is not None:
            writer.release()

    total = sum(s["frames_written"] for s in segments)
    print(f"\nWrote {dst.name}: {total} frames @ {out_clips[0]['fps']:.2f} fps")

    # ---- representative frames for Module A ------------------------------
    cap = cv2.VideoCapture(str(dst))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    picks = np.linspace(int(n_total * 0.06), int(n_total * 0.94), 5).astype(int)
    frames_written = []
    for i, idx in enumerate(picks):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        p = FRAMES_DIR / f"frame_{i:02d}_f{idx:05d}.png"
        cv2.imwrite(str(p), frame)
        frames_written.append({"index": int(idx), "path": str(p)})
    cap.release()
    print(f"Wrote {len(frames_written)} representative frames to {FRAMES_DIR}")

    meta = {
        "video": str(dst),
        "fps": out_clips[0]["fps"],
        "n_frames": total,
        "width": out_clips[0]["width"],
        "height": out_clips[0]["height"],
        "viewpoint": args.viewpoint,
        "source_archive": str(archive),
        "source_license": "CC BY-NC-SA 4.0 (TU/e Impact-Aware Robotics Archive)",
        "segments": [{k: v for k, v in s.items() if k != "time_vector"} for s in segments],
        "representative_frames": frames_written,
        "note": (
            "Each source recording contains exactly one tossed box "
            "(object_environment = 'point; single; bounce'), so the pipeline "
            "sees one box at a time, and three box types in sequence across the "
            "concatenated video. The mocap paths are recorded for validation "
            "only; the vision pipeline never reads them."
        ),
    }
    mf = meta_file()
    existing = json.loads(mf.read_text()) if mf.exists() else {}
    existing.update(meta)
    mf.write_text(json.dumps(existing, indent=2))
    print(f"Wrote {mf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
