# Hybrid ChArUco and NDI tracking

Compare camera-detected ChArUco corners with independently NDI-reprojected corners,
matched by OpenCV corner ID. Outputs retain subpixel coordinates, signed x/y
residuals and Euclidean pixel error. This evaluates the combined calibration,
mount geometry, tracking and synchronization chain.

From the repository root, install the locked dependencies and run the tests:

```bash
uv sync --locked
uv run --locked python -m unittest discover -s tests -v
```

With the recording available, run detection. For full reprojection, first set the
measured board rotation as described below.

```bash
uv run hybrid-tracking --config configs/nov12.json --detect-only
uv run hybrid-tracking --config configs/nov12.json
```

Alternatively, with NumPy and OpenCV contrib installed:

```bash
python3 scripts/evaluate_reprojection.py --detect-only --stride 30
python3 scripts/evaluate_reprojection.py --config configs/nov12.json
python3 -m unittest discover -s tests -v
```

Use `--max-frames N` to limit processed frames and `--stride N` to sample every
Nth frame. Outputs go to `output.directory`: `corners.csv` contains one row per
compared corner, `frames.csv` records detections and skipped matches,
`summary.json` records the configuration, counts and pooled corner-error
mean/RMSE/median/95th percentile/maximum. `detected.jpg` shows green detected
corners, red projected crosses and yellow residual lines. Detection-only mode
produces frame counts and a detection preview, without NDI error estimates.
Files with the same names are replaced on subsequent runs.

## November 12 recording

Both November 12 configs read the recording from `data/20261112_hybrid/`.
Place the recording directory there or symlink it from the external drive.
The hand-eye and camera intrinsics files come from the adjacent
`../NDI_robot_calibration/` repository. For a different layout, copy a config
to `configs/nov12.local.json`, edit its paths and pass it with `--config`.
Git ignores local configs, recordings and generated results.

The November 12 board has **8×6 squares** and uses the legacy ChArUco pattern.
The first frame contains 35 detectable intersections. The configurations below
distinguish CAD dimensions from measured settings. `configs/hybrid.json`
retains the original reference-point settings.

Before NDI evaluation:

1. Set measured `marker_R_board` (currently null). This rotates board axes into
   the tracked NDI rigid body's axes. `NDI_position` is the NDI rigid body origin
   in board coordinates. Board origin is bottom left, x right, y up, z out of
   the printed face. Do not assume identity rotation without measuring it.
2. Confirm the intrinsics and ECM-to-camera calibration apply to stream 1.
   The config uses the existing July 27 hand-eye and camera intrinsics; their
   applicability to this recording has not been established.
3. The user confirmed that the board and tracker remained fixed; the November 12
   configuration now uses `ndi_mode="static"` and the first valid NDI pose.
   The CSV contains 149 valid poses with less than 0.09 mm maximum translation
   deviation from their mean and 0.13 degrees rotation from the first sample.
   NDI is recorded around 18:07;
   video/robot data are around 18:10. The `ndi_mode="synchronized"` option rejects
   unmatched times. Only if the board assembly **and NDI tracker** remained
   fixed between recordings, use `ndi_mode="static"` and
   `static_pose_confirmed=true`. This explicitly selects `static_pose_index`
   (default 0) from the valid NDI rows; it does not silently extrapolate or
   estimate a mount from the evaluation images.

The supplied position calibration reports approximately 13.69 mm fitting RMSE
on its training poses. This is not independent accuracy, and the recording's
final pixel errors will depend on that calibration as well as the mount.

## Geometry and timestamps

`A_T_B` maps B coordinates into A. All internal translations are metres.
The evaluation uses:

```text
camera_T_board = inverse(ndi_T_base @ base_T_ecm @ ecm_T_camera)
                 @ ndi_T_marker @ marker_T_board
```

Robot poses use the 12 columns after `mip Index (ECM)` and microsecond epoch
timestamps. NDI CSV translations are converted from millimetres to metres,
orientation comes from R00…R22, and `ndi_tool_id` selects the rigid body.
Small rotation rounding errors in CSVs are orthonormalized; invalid poses are
excluded. Timestamp matching uses the nearest pose with an explicit tolerance
(default 50 ms per stream), without interpolation. Naive ISO video timestamps
are interpreted in `input.timezone` (America/Vancouver by default).
`video_offset_s` is added to the video timestamps only when an independently
established clock correction is needed. Positive camera depth is required.

Projection uses OpenCV intrinsics and distortion on original video coordinates.
No resizing, crop correction, clock fitting or camera-pose fitting is performed.
The residual is `projected - observed`; errors are in pixels, not millimetres.
NPZ/JSON calibration inputs specify an explicit `path` and array `key`.
Relative data/calibration/output paths resolve against the project root.

The original reference-point exporter remains available as
`hybrid_tracking.pipeline.run(config_path)`. It saves ID-aligned board, marker
and NDI coordinates and requires its original JSON calibration schema.

## CAD geometry and provisional November 12 evaluation

`configs/cad_geometry.json` records dimensions and SHA-256 hashes extracted by
`toolbox.cad.read_apple_geometry` from both discovered SCAD versions. The file
explicitly supplied by the user specifies **19 mm squares, 15 mm markers,
DICT_4X4_250, 8×6**, with a 5 mm border and 4 mm base thickness. Its cross origin
is at `(157,119,0)` mm in SCAD coordinates. Our board origin is on the printed
surface, so the same point is `(157,119,-4)` mm in board coordinates. The +x
post base is at `(197.7941,119,5.9999)` mm in board coordinates; it is distinct
from the cross intersection. Do not add the 40.7941 mm arm to the cross origin
unless that post is actually the ROM's reference origin.

The newer file in `3d_printed_charuco` instead specifies 17 mm squares, 13 mm
markers and 5 mm extra x/y offsets, yielding a cross origin `(146,112,-4)` mm.
Both versions use the legacy pattern. IDs 0–23 have identical patterns in the
4×4 dictionaries with capacities 50 and 250; dictionary detection alone cannot
resolve the physical square size.

A provisional interpretation of the explicitly supplied file is runnable with:

```bash
python3 scripts/evaluate_reprojection.py --config configs/nov12_cad_nominal.json
```

This configuration records its assumptions: the tracked axes match the CAD
cross axes (identity rotation) and the tracked origin is the cross intersection
at SCAD z=0. The SCAD labels cross axes and reflector coordinates, but does not
establish the ROM origin's vertical height or axis convention. These assumptions
and the actual printed version need confirmation before interpreting the output
as measured tracking accuracy. The default configuration retains the missing
rotation rather than silently marking this nominal interpretation as measured.

Each evaluated corner now also includes NDI-predicted and vision-estimated
camera-frame coordinates, signed XYZ differences in metres, and Euclidean error
in millimetres. Vision coordinates come from a planar IPPE PnP fit of the known
board geometry to detected corners. The lower-reprojection-error positive-depth
solution is selected; planar pose ambiguity remains a limitation. These spatial
errors compare two rigid-board pose estimates, not independent measurements of
each corner's depth. `vision_fit_rmse_px` records the image fit. `summary.json`
includes pooled spatial statistics and pixel/spatial statistics by corner ID.

The nominal configuration uses `Etc/GMT+8` (fixed UTC−08:00) for the video's
naive local timestamps. In this environment the `America/Vancouver` zone maps
November 12, 2026 to UTC−07:00, creating a one-hour mismatch with the robot's
epoch timestamps and no matches. UTC−08:00 aligns the robot recording start
with the 18:10 filename/video timeline. This inference is recorded in the
nominal configuration; verify the capture machines' clock settings before
using the residuals as accuracy measurements. Early frames precede robot
collection and are still rejected. `comparison.jpg` shows the first frame with
matched poses and compared corners; this may differ from the detection preview.

The completed full-video nominal run (OpenCV 4.13.0) compared **97,038 corners**:
pixel RMSE **118.63 px**, mean **114.91 px**; spatial RMSE **16.03 mm**,
mean **15.52 mm**. There were 470 detected frames without a robot match. These
are conditional residuals for the recorded nominal assumptions, not a validated
accuracy estimate. Full results and the first comparison overlay are under
`results/nov12_cad_nominal/`. Software versions are recorded with each new run.

### Confirmed CAD version correction

The user identified `../3d_printed_charuco/charuco_with_apple1_post.scad` as the
correct design: **17 mm squares, 13 mm markers, 8×6**, 5 mm border, 4 mm base,
with 5 mm extra Apple offsets in both x and y. Both November 12 configurations
now use these dimensions and cross center `(146,112,-4)` mm relative to the
printed board surface. The nominal run output directory is now
`results/nov12_cad_17mm_nominal` to avoid confusing it with the earlier 19 mm run.
The previously reported 118.63 px / 16.03 mm RMSE used the superseded 19 mm
geometry and must not be treated as results for the confirmed board.
The NDI ROM origin-height/axis assumptions remain unresolved.

### Annotated video

Set `output.annotated_video` to `true` to write `reprojection.mp4` alongside the
CSV files. `output.keypoint_radius_px` controls both detected green and projected
red filled dots (default 6 pixels). Yellow lines connect corresponding points.
Each frame displays its index and matching status. All known board points are
projected independently of image detection. Set `output.hold_nearest_robot_pose`
to `true` to keep the overlay visible outside the ECM recording range; the
nearest recorded pose is then held and clearly labeled. Held-pose frames are
excluded from pixel/spatial error statistics. With that option disabled, frames
without valid robot matches have no red projected points. Output FPS follows the input video,
divided by `--stride` when sampling. The November 12 nominal configuration now
enables video output and uses a 16.8 mm square pitch and an NDI offset of
`(146,112,+13.93)` mm. The saved correction study used `(146,112,-13.93)` mm.
Use the settings in that run’s `summary.json` to reproduce its figures.

`projections.csv` records all finite, positive-depth NDI corner projections,
including undetected corners and held-pose frames. `frames.csv` records
`projected_corners`, signed `robot_dt_s`, and `robot_pose_source` (`matched` or
`held_nearest`). `summary.json` reports projected-frame and held-pose counts.
The November 12 nominal configuration enables held poses so red dots remain
visible over the whole video; this estimates camera pose at recording boundaries,
not newly measured ECM tracking. Points outside the image naturally remain
outside the visible frame.

### Learn a reprojection correction from the static-board recording

From the repository root, run the study on a completed full-rate recording with all-corner projections. The input video and pose files referenced by its saved summary must be accessible.

```bash
uv run python scripts/study_correction.py results/nov12_cad_17mm_nominal
uv run python scripts/render_correction.py results/nov12_cad_17mm_nominal/correction_study
```

`correction_study/dataset.npz` joins frame and ChArUco ID by array index. `observed_px`, `raw_px` and `corrected_px` have shape `(frame, id, 2)` in original distorted image coordinates. Missing green detections are NaN. `board_m` gives each corner's metric coordinates. `vision_camera_m` contains the camera PnP model estimate where the original evaluation computed it. Frame-level arrays contain timestamps, the actual ECM sample time, held-pose flags, `base_T_ecm` and `camera_T_board`. Static arrays store the NDI pose and timestamp, base and hand-eye calibration, board attachment, intrinsics and distortion. Translations use metres. To recover raw camera-space XYZ, apply `camera_T_board[frame]` to `board_m[id]`. For a frame's ECM translation in base coordinates use `base_T_ecm[frame,:3,3]`.

The study fits a pixel shift, affine pixel correction, camera-frame rigid correction, board-frame rigid correction and combined rigid correction. Fit uses every fifth frame in the first 55% of the ECM-matched interval. The middle interval selects the model by pixel RMSE, and the final quarter evaluates the selected model against the original projection. Gaps of 30 frames precede validation and test. Only detected IDs with a time-matched ECM pose are scored. Rigid fits minimize robust pixel residuals across all sampled corner correspondences, with a 5 px Huber threshold. No image observations enter inference.

For the selected joint model, compute `corrected_camera_T_board = C @ camera_T_board @ B`, then project the board points using the original intrinsics and distortion. In `study.json`, `parameters.joint` stores B followed by C. Each six-vector contains a Rodrigues rotation vector in radians followed by XYZ translation in metres. Apply exactly the same board origin and camera convention as the training run. The fitted transforms describe this recording's combined calibration error. A single static target does not identify the physical source of the offsets. Independent recordings are needed to evaluate transfer to a changed setup.

`study.json` contains the fitted parameters for all candidates, chronological split boundaries and pixel statistics. The spatial comparison uses visible corners with a valid camera PnP estimate. `corrected.mp4` shows the raw projections in red, detections in green and corrected projections as filled yellow dots. Held ECM poses are labeled in the video.
