"""Canonical locations for data, modules and generated outputs."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

DATA = REPO / "data"
RAW = DATA / "raw"
OUTPUTS = REPO / "outputs"

MODULES = REPO

CHESSBOARD_DIR = RAW / "chessboard"
IAM_DIR = RAW / "iam"
FRAMES_DIR = DATA / "frames"

# Published dimensions of the three carton boxes in the I.AM. archive, scraped
# from https://impact-aware-robotics-database.tue.nl/objects/<name>.  The page
# embeds an NXDL attribute block with `dimensions: ds: [...]` in metres.
# Box006 and Box007 are within 1 cm of each other in every axis, which is
# deliberate on our part: it shows that box *type* cannot be recovered from
# size alone, so Module E has to rely on appearance.
BOX_DIMENSIONS_CM: dict[str, tuple[float, float, float]] = {
    "Box006": (20.52, 15.54, 10.06),
    "Box007": (20.74, 15.76, 9.89),
    "Box009": (16.44, 12.53, 12.02),
}
BOX_MASS_KG: dict[str, float] = {
    "Box006": 0.365,
    "Box007": 1.431,
    "Box009": 0.285,
}

# OpenCV's sample board: 9x6 inner corners, 25.0 mm squares, 640x480.
CHESSBOARD_SIZE = (9, 6)
CHESSBOARD_SQUARE_MM = 25.0


def video_file() -> Path:
    return DATA / "conveyor.mp4"


def meta_file() -> Path:
    return DATA / "meta.json"


def calibration_file() -> Path:
    return DATA / "calibration.npz"


def ensure_dirs() -> None:
    for d in (DATA, RAW, OUTPUTS, FRAMES_DIR, CHESSBOARD_DIR):
        d.mkdir(parents=True, exist_ok=True)
