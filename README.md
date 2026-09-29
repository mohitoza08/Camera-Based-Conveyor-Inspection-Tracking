# Camera-Based Conveyor Inspection & Tracking

## What it does

A single ceiling camera watches boxes tossed past it; classical CV (no pretrained
models) segments each box, tracks it with a hand-written Kalman filter, measures its
real size in cm and speed in cm/s, and recognises which of three box types it is.
`pipeline.py` chains all five modules and prints one line per tracked box.

## Dataset

TU/e Impact-Aware Robotics archive, `240816_Archive_025_ImpactAwareObjectTracking.h5`
(CC BY-NC-SA 4.0), viewpoint 1, three recordings (Box006, Box007, Box009 -- one
tossed box per recording) concatenated into `data/conveyor.mp4`: 3412 frames,
720x540, 60 fps. Box footprint dimensions are published on the archive's object
pages and hard-coded in `common/paths.py::BOX_DIMENSIONS_CM`. Chessboard
calibration photos are OpenCV's own `samples/data/left*.jpg` (13 images, 9x6
inner corners, 25 mm squares) -- see the Module B caveat below for why these are
a *different camera* from the conveyor footage. `data/fetch_data.py` and
`data/prepare_video.py` reproduce both from scratch.

## How to run

```
python -m pip install -r requirements.txt

python -m module_a_segmentation.compare                       # Module A grid + summary
python -m jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=900 module_a_segmentation/compare_methods.ipynb

python module_b_calibration/calibrate.py                      # Module B: K, distortion
python module_b_calibration/measure_box.py                    # Module B: 4 camera models

python module_c_motion/optical_flow.py                        # Module C
python module_d_tracking/kalman_tracker.py                    # Module D
python module_e_recognition/recognize.py                      # Module E

python pipeline.py                                             # full integration
```

All outputs land in `outputs/`.

## Module A -- Segmentation

Seven methods (brief asked for four) on the same five representative frames, scored
by IoU of the best-matching region against the motion-detected box:
**active contour** (hand-written snake, seeded -- 0.66-0.72 IoU, an upper bound, not
like-for-like), **watershed** (best *unsupervised* method, 0.11-0.49), **quadtree
split** (over-fragments, <0.10) and **merge** (recovers to 0.12-0.34 -- the
split/merge gap is the real result), **Felzenszwalb** (stable ~165 regions,
0.15-0.34), **mean shift** (fewest regions but groups by colour not object, ~30
s/frame), **normalized cut** (needs distance-mode edge weights or it collapses to
one region; still the most unstable, 0.02-0.27). Full commentary is in
`compare_methods.ipynb`. These recordings never have two boxes touching, so the
comparison is single-object isolation against clutter, not the touching-blob case.

## Module B -- Calibration & size

`calibrate.py` runs `cv2.calibrateCamera` on the 13 chessboard photos: RMS 0.41 px
(published reference: 0.39 px), fx/fy/cx/cy all within 0.2 px of OpenCV's own
`left_intrinsics.yml`. `P = K[R|t]` is built directly and decomposed back into
K, R, t with a hand-written RQ decomposition (via NumPy QR + row/column flips,
verified against a synthetic P before being trusted on real data -- see the
self-test printed at the top of `calibrate.py`'s output, exact to float precision).
**Caveat:** the chessboard photos are a different camera than the conveyor
footage, so K does not transfer. `measure_box.py` instead transfers *metric scale*
from one box of known size (Box006) to the other two under all four camera
models (orthographic/weak-perspective/affine/full-perspective); on this
near-top-down footage orthographic and full-perspective come out identical and
best (mean error 2.15 cm), because the small residual perspective leaves the
correction almost nothing to do, while affine's free shear amplifies the
detector's own motion-blur distortion of the footprint outline.

## Module C -- Optical flow

Hand-written iterative Lucas-Kanade (spatial gradient tensor + 2-3 Newton steps,
solved from scratch, no `cv2` flow calls) is compared against
`cv2.calcOpticalFlowPyrLK` on Shi-Tomasi corners, same frame pair. On the static
background the two agree to 0.16 px on average -- the correctness check. On the
box the hand-written solver disagrees with PyrLK by ~1.85 px on average (max 9.82
px) even though the true motion (~8.6 px/frame) is well inside a single window's
reach; the cause is motion blur (the thrown box smears during the 60 fps
exposure), which breaks brightness constancy locally in a way a single-scale,
fixed-template solver can't average out but PyrLK's larger window and
coarse-to-fine pyramid can. RANSAC-fitted affine flow on the box region gives
48.6 cm/s for this frame pair -- an instantaneous toss speed, not a steady belt
speed, since these are thrown boxes.

## Module D -- Kalman tracking

From-scratch constant-velocity Kalman filter, state `[x, y, vx, vy]`,
`F`/`H`/`Q`/`R` as in `kalman_tracker.py`'s docstring (`Q`'s velocity terms are
kept large because a thrown, bouncing box only approximately obeys
constant-velocity). Tracked across the full 1192-frame Box006 recording; the
detector only fires on 632 of those frames, so 560 are natural predict-only
gaps, plus an additional artificial 4-frame dropout (frames 500-503) is forced
to demonstrate pure-prediction tracking through occlusion, per the brief.
`outputs/module_d_tracking.mp4` overlays detected corners (green), predicted
position (amber diamond) and corrected position (cyan cross); the console
prints predict/measure/update for ten consecutive frames.

## Module E -- Recognition

Each box type is one recording, but a tumbling toss gives many different
orientations per frame, so crops from ~15-30 clean detections per segment give a
genuine held-out (not leave-one-out) train/test split. **Eigenboxes** (PCA on
150x150 flattened crops, k=12, nearest neighbour): 91.7% held-out accuracy --
robust to the box006/box007 shape ambiguity (they differ by <1 cm on every axis)
because it sees pixel intensity, not just outline. **Hu moments** (rotation/scale
invariant, silhouette-only): 66.7% -- confuses Box006/Box007 exactly because it
can't see anything but the (near-identical) outline. **Alignment** (ORB +
RANSAC homography, scored by reprojection residual): 37.5% -- needs real texture
to find keypoints at all (72 px crops gave *zero* ORB keypoints; 150 px was
needed), and cardboard is nearly featureless wherever motion blur has smeared
its few printed marks.

## Limitations

- Three box types, one recording each -- accuracy numbers above are real
  held-out splits across *orientations* of one instance per class, not
  independent physical samples.
- Chessboard calibration is from a different camera than the conveyor footage;
  metric scale is transferred via a known reference box instead, and the
  archive's own published `camera_K_archive` (not our chessboard K) is used
  wherever a perspective correction needs *a* camera matrix.
- These are thrown, bouncing boxes, not objects on a steady conveyor -- apparent
  scale and speed both vary within a single toss, so all size/speed numbers are
  instantaneous, snapshotted at hand-picked, scale-matched frames, not steady
  averages.
- No two boxes ever overlap in this footage, so Module A's touching-blob case
  is untested; only single-object isolation against clutter is demonstrated.
- Reference-camera viewpoint (vp1) only; vp2 was not prepared.
