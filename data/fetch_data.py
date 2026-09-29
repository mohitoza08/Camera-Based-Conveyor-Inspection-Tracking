"""Download the two external data sources this project needs.

1. The I.AM. dataset (Impact-Aware Robotics Archive, TU/e) -- real footage of
   carton boxes on a running conveyor, two viewpoints, three box types.
   DOI 10.4121/51de87d3-1c8e-4849-96cd-8045c5e6ab7f.v3, CC BY-NC-SA 4.0.
2. The OpenCV sample chessboard photographs (13 real photos of a printed
   9x6 board, 25.0 mm squares, 640x480) used for Module B calibration.

Everything lands in data/raw/ and is skipped if already present.
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

RAW = Path(__file__).resolve().parent / "raw"
CHESSBOARD_DIR = RAW / "chessboard"
IAM_DIR = RAW / "iam"

IAM_URL = "https://data.4tu.nl/ndownloader/items/51de87d3-1c8e-4849-96cd-8045c5e6ab7f/versions/3"
CHESSBOARD_BASE = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/"
# The board has 9x6 *inner* corners => 10x7 squares. 25.0 mm per square, per
# samples/data/left_intrinsics.yml in the OpenCV repository.
CHESSBOARD_FILES = [f"left{i:02d}.jpg" for i in (*range(1, 10), *range(11, 15))]
CHESSBOARD_FILES.append("left_intrinsics.yml")


def _download(url: str, dest: Path, retries: int = 3) -> bool:
    if dest.exists() and dest.stat().st_size > 1024:
        print(f"  cached  {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            print(f"  get     {dest.name} (attempt {attempt}) ...", flush=True)
            urllib.request.urlretrieve(url, dest)
            print(f"  ok      {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
            return True
        except (urllib.error.URLError, OSError) as exc:  # pragma: no cover - network
            print(f"  retry   {exc}")
    return False


def fetch_chessboard() -> bool:
    print("OpenCV sample chessboard photographs (Module B):")
    ok = True
    for name in CHESSBOARD_FILES:
        ok &= _download(CHESSBOARD_BASE + name, CHESSBOARD_DIR / name)
    return ok


def fetch_iam() -> bool:
    print("I.AM. conveyor dataset (Modules A, C, D, E, pipeline):")
    # Already extracted?
    if any(IAM_DIR.rglob("*.h5")):
        print(f"  cached  extracted archive in {IAM_DIR}")
        return True
    archive = IAM_DIR / "iam_dataset.zip"
    if not _download(IAM_URL, archive):
        return False
    print("  unzip   ...", flush=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(IAM_DIR)
    print(f"  ok      extracted to {IAM_DIR}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-iam",
        action="store_true",
        help="only fetch the chessboard photos (much faster)",
    )
    args = parser.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    ok = fetch_chessboard()
    if not args.skip_iam:
        ok &= fetch_iam()

    if not ok:
        print("\nOne or more downloads failed. Re-run to retry.", file=sys.stderr)
        return 1
    print("\nAll data present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
