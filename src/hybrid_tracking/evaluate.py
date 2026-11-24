"""ID-matched NDI reprojection evaluation. All internal distances are metres."""
import argparse
import csv
import json
import platform
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np

from toolbox.board import create_board, generate_board_points, marker_T_board
from toolbox.config import load_config, project_path
from toolbox.transforms import transform_points, validate_transform
from hybrid_tracking.pipeline import PROJECT_ROOT


def rigid_pose(rotation, translation):
    """Repair only CSV rounding error, rejecting invalid tracking rotations."""
    r = np.asarray(rotation, float).reshape(3, 3)
    if not np.isfinite(r).all() or np.linalg.norm(r.T @ r - np.eye(3)) > 1e-3 or np.linalg.det(r) <= 0:
        raise ValueError('Invalid tracking rotation')
    u, _, vt = np.linalg.svd(r)
    t = np.eye(4)
    t[:3, :3], t[:3, 3] = u @ vt, translation
    return validate_transform(t)


def load_poses(path, kind, tool_id=1):
    times, poses = [], []
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = [s.strip() for s in next(reader)]
        start = header.index('mip Index (ECM)') + 1 if kind == 'robot' else None
        for row in reader:
            if not row:
                continue
            if kind == 'ndi' and int(row[0]) != tool_id:
                continue
            try:
                if kind == 'robot':
                    v = np.asarray(row[start:start+12], float)
                    timestamp, pose = float(row[0]) / 1e6, rigid_pose(v[3:], v[:3])
                else:
                    timestamp, pose = float(row[1]), rigid_pose(row[6:15], np.asarray(row[3:6], float) / 1000)
            except (ValueError, IndexError):
                continue
            times.append(timestamp)
            poses.append(pose)
    if not times:
        raise ValueError(f'No valid {kind} poses in {path}')
    order = np.argsort(times, kind='stable')
    return np.asarray(times)[order], np.asarray(poses)[order]


def nearest(times, timestamp, tolerance):
    i = int(np.searchsorted(times, timestamp))
    candidates = [j for j in (i-1, i) if 0 <= j < len(times)]
    j = min(candidates, key=lambda j: abs(times[j] - timestamp))
    return j if abs(times[j] - timestamp) <= tolerance else None


def load_array(root, spec, default_key):
    path = project_path(spec['path'], root)
    key = spec.get('key', default_key)
    if path.suffix == '.npz':
        with np.load(path, allow_pickle=False) as data:
            return np.array(data[key], dtype=float)
    data = load_config(path)
    return np.asarray(data[key] if isinstance(data, dict) else data, float)


def compare(points_camera, ids, observed, K, distortion):
    """Retain detected IDs and subpixel coordinates; reject nonpositive depth."""
    ids = np.asarray(ids).reshape(-1).astype(int)
    observed = np.asarray(observed).reshape(-1, 2)
    if len(ids) != len(observed) or np.any(ids < 0) or np.any(ids >= len(points_camera)):
        raise ValueError('Invalid detected corner IDs')
    points = np.asarray(points_camera)[ids]
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
    ids, observed, points = ids[valid], observed[valid], points[valid]
    if not len(ids):
        return ids, observed, np.empty((0, 2)), np.empty((0, 2))
    projected = cv2.projectPoints(points, np.zeros(3), np.zeros(3), K, distortion)[0].reshape(-1, 2)
    return ids, observed, projected, projected - observed


def camera_board_pose(points, ids, observed, K, distortion):
    """Estimate the metric board pose from image corners, independent of NDI.

    Spatial errors use the rigid planar board model, not per-corner depth sensing.
    Select the positive-depth IPPE solution with the lowest reprojection error.
    """
    ids = np.asarray(ids).reshape(-1).astype(int)
    obj = np.ascontiguousarray(points[ids], dtype=np.float64)
    img = np.ascontiguousarray(observed, dtype=np.float64).reshape(-1, 1, 2)
    if len(ids) < 4 or np.linalg.matrix_rank(obj - obj.mean(axis=0)) < 2:
        return None
    result = cv2.solvePnPGeneric(obj, img, K, distortion, flags=cv2.SOLVEPNP_IPPE)
    candidates = []
    for r, t in zip(result[1], result[2]):
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = cv2.Rodrigues(r)[0], t.reshape(3)
        if np.any(transform_points(pose, obj)[:, 2] <= 0):
            continue
        projected = cv2.projectPoints(obj, r, t, K, distortion)[0].reshape(-1, 2)
        rmse = float(np.sqrt(np.mean(np.sum((projected-img.reshape(-1, 2))**2, axis=1))))
        candidates.append((rmse, pose))
    return min(candidates, key=lambda item: item[0]) if candidates else None


def statistics(values):
    e = np.asarray(values, dtype=float)
    return dict(mean=float(e.mean()), rmse=float(np.sqrt(np.mean(e**2))),
                median=float(np.median(e)), p95=float(np.percentile(e, 95)), max=float(e.max()))


def evaluate(config_path, root=PROJECT_ROOT, max_frames=None, stride=1, detect_only=False):
    if stride < 1 or (max_frames is not None and max_frames < 1):
        raise ValueError('stride and max_frames must be positive')
    cfg = load_config(config_path)
    board = create_board(cfg['charuco'])
    points = generate_board_points(cfg['charuco'])['charuco_points_board_m']
    detector = cv2.aruco.CharucoDetector(board)
    inp, cal = cfg['input'], cfg['calibration']
    output = project_path(cfg['output']['directory'], root)
    # Validate all required calibration before creating result files.
    if not detect_only:
        if cfg.get('marker_R_board') is None:
            raise ValueError('Set measured marker_R_board before NDI evaluation')
        attachment = marker_T_board([cfg['NDI_position'][a] for a in 'xyz'], cfg['marker_R_board'])
        ndi_T_base = validate_transform(load_array(root, cal['ndi_T_base'], 'ndi_T_base'))
        ecm_T_camera = validate_transform(load_array(root, cal['ecm_T_camera'], 'X'))
        K = load_array(root, cal['intrinsics'], 'K')
        distortion = load_array(root, cal['distortion'], 'distCoeffs')
        if K.shape != (3, 3) or not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0 or not np.isfinite(distortion).all():
            raise ValueError('Invalid camera intrinsics')
        rt, rp = load_poses(project_path(inp['robot'], root), 'robot')
        nt, np_ = load_poses(project_path(inp['ndi'], root), 'ndi', inp.get('ndi_tool_id', 1))
        mode = inp.get('ndi_mode', 'synchronized')
        if mode not in ('synchronized', 'static'):
            raise ValueError('ndi_mode must be synchronized or static')
        if mode == 'static' and not inp.get('static_pose_confirmed', False):
            raise ValueError('Static mode requires static_pose_confirmed=true after verifying board and tracker stayed fixed')
        static_index = inp.get('static_pose_index', 0)
        if mode == 'static' and not 0 <= static_index < len(nt):
            raise ValueError('static_pose_index out of range')
    zone = ZoneInfo(inp.get('timezone', 'America/Vancouver'))
    timestamps = []
    for line in project_path(inp['timestamps'], root).read_text().splitlines():
        dt = datetime.fromisoformat(line.strip())
        timestamps.append((dt.replace(tzinfo=zone) if dt.tzinfo is None else dt).timestamp())
    if not timestamps or np.any(np.diff(timestamps) < 0):
        raise ValueError('Video timestamps must be nonempty and sorted')
    cap = cv2.VideoCapture(str(project_path(inp['video'], root)))
    if not cap.isOpened():
        raise ValueError('Cannot open video')
    output.mkdir(parents=True, exist_ok=True)
    counts = dict(frames=0, detected_frames=0, unmatched_robot=0, unmatched_ndi=0, invalid_depth_corners=0, projected_frames=0, held_robot_frames=0)
    errors, spatial_errors = [], []
    saved_comparison_overlay = False
    per_corner = {}
    video_writer = None
    radius = int(cfg['output'].get('keypoint_radius_px', 6))
    if radius < 1:
        cap.release()
        raise ValueError('keypoint_radius_px must be positive')
    try:
        if cfg['output'].get('annotated_video', False):
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not np.isfinite(fps) or fps <= 0:
                raise ValueError('Video has no valid frame rate')
            size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            video_writer = cv2.VideoWriter(str(output/'reprojection.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps/stride, size)
            if not video_writer.isOpened():
                raise ValueError('Cannot create annotated video')
        with (output/'corners.csv').open('w', newline='') as f, (output/'frames.csv').open('w', newline='') as ff, (output/'projections.csv').open('w', newline='') as pf:
            pw = csv.writer(pf)
            pw.writerow(['frame', 'timestamp_s', 'charuco_id', 'projected_x_px', 'projected_y_px', 'robot_dt_s', 'robot_pose_source'])
            writer, fw = csv.writer(f), csv.writer(ff)
            writer.writerow(['frame', 'timestamp_s', 'charuco_id', 'observed_x_px', 'observed_y_px', 'projected_x_px', 'projected_y_px', 'dx_px', 'dy_px', 'error_px', 'robot_dt_s', 'ndi_dt_s', 'ndi_x_camera_m', 'ndi_y_camera_m', 'ndi_z_camera_m', 'vision_x_camera_m', 'vision_y_camera_m', 'vision_z_camera_m', 'dx_m', 'dy_m', 'dz_m', 'error_mm', 'vision_fit_rmse_px'])
            fw.writerow(['frame', 'timestamp_s', 'detected_corners', 'compared_corners', 'status', 'projected_corners', 'robot_dt_s', 'robot_pose_source'])
            index = 0
            while max_frames is None or counts['frames'] < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_index = index
                index += 1
                if frame_index >= len(timestamps):
                    raise ValueError('Video contains more frames than timestamps')
                if frame_index % stride:
                    continue
                counts['frames'] += 1
                timestamp = timestamps[frame_index] + inp.get('video_offset_s', 0)
                corners, ids, _, _ = detector.detectBoard(frame)
                n = 0 if ids is None else len(ids)
                counts['detected_frames'] += int(n > 0)
                if n:
                    cv2.aruco.drawDetectedCornersCharuco(frame, corners, ids, (0, 255, 0))
                    for corner in corners.reshape(-1, 2):
                        cv2.circle(frame, tuple(np.rint(corner).astype(int)), radius, (0, 255, 0), -1, cv2.LINE_AA)
                status, compared = 'no_detection', 0
                projected_count, robot_dt, pose_source = 0, '', ''
                overlay_pixels = []
                if detect_only:
                    status = 'detection_only'
                else:
                    ri = nearest(rt, timestamp, inp.get('max_robot_dt_s', .05))
                    ni = static_index if mode == 'static' else nearest(nt, timestamp, inp.get('max_ndi_dt_s', .05))
                    synchronized_robot = ri is not None
                    if ri is None:
                        counts['unmatched_robot'] += 1
                        status = 'unmatched_robot'
                        if cfg['output'].get('hold_nearest_robot_pose', False):
                            ri = nearest(rt, timestamp, float('inf'))
                            counts['held_robot_frames'] += 1
                    if ri is not None:
                        robot_dt = float(rt[ri] - timestamp)
                        pose_source = 'matched' if synchronized_robot else 'held_nearest'
                    if ni is None:
                        counts['unmatched_ndi'] += 1
                        status = 'unmatched_ndi'
                    elif ri is not None:
                        camera_T_board = np.linalg.inv(ndi_T_base @ rp[ri] @ ecm_T_camera) @ np_[ni] @ attachment
                        predicted_xyz = transform_points(camera_T_board, points)
                        # Project the entire known board, independent of image detections.
                        valid_ids = np.flatnonzero(np.isfinite(predicted_xyz).all(axis=1) & (predicted_xyz[:, 2] > 0))
                        counts['invalid_depth_corners'] += len(points) - len(valid_ids)
                        if len(valid_ids):
                            all_pixels = cv2.projectPoints(predicted_xyz[valid_ids], np.zeros(3), np.zeros(3), K, distortion)[0].reshape(-1, 2)
                            for corner_id, pixel in zip(valid_ids, all_pixels):
                                if np.isfinite(pixel).all():
                                    pw.writerow([frame_index, timestamp, corner_id, *pixel, robot_dt, pose_source])
                                    projected_count += 1
                                    overlay_pixels.append(pixel)
                            counts['projected_frames'] += int(projected_count > 0)
                        status = 'projected_no_detection' if synchronized_robot else 'held_robot_pose'
                        # Only valid timestamp matches contribute accuracy statistics.
                        if n and synchronized_robot:
                            vision = camera_board_pose(points, ids, corners, K, distortion)
                            vision_xyz = transform_points(vision[1], points) if vision is not None else None
                            cid, obs, projected, delta = compare(predicted_xyz, ids, corners, K, distortion)
                            compared = len(cid)
                            status = 'compared' if compared else 'invalid_depth'
                            for id_, o, p, d in zip(cid, obs, projected, delta):
                                error = float(np.linalg.norm(d))
                                errors.append(error)
                                corner_stats = per_corner.setdefault(int(id_), dict(pixel=[], spatial=[]))
                                corner_stats['pixel'].append(error)
                                spatial = [''] * 8
                                if vision_xyz is not None:
                                    diff = predicted_xyz[id_] - vision_xyz[id_]
                                    distance_mm = float(np.linalg.norm(diff)*1000)
                                    spatial_errors.append(distance_mm)
                                    corner_stats['spatial'].append(distance_mm)
                                    spatial = [*vision_xyz[id_], *diff, distance_mm, vision[0]]
                                writer.writerow([frame_index, timestamp, id_, *o, *p, *d, error, rt[ri]-timestamp, nt[ni]-timestamp, *predicted_xyz[id_], *spatial])
                                if np.isfinite(p).all() and np.max(np.abs(p)) < 1e6:
                                    cv2.line(frame, tuple(np.rint(o).astype(int)), tuple(np.rint(p).astype(int)), (0, 255, 255), 1)
                # Draw points after residual lines so every visible dot stays clear.
                if n:
                    for corner in corners.reshape(-1, 2):
                        cv2.circle(frame, tuple(np.rint(corner).astype(int)), radius, (0, 255, 0), -1, cv2.LINE_AA)
                for pixel in overlay_pixels:
                    if np.max(np.abs(pixel)) < 1e6:
                        cv2.circle(frame, tuple(np.rint(pixel).astype(int)), radius, (0, 0, 255), -1, cv2.LINE_AA)
                if pose_source == 'held_nearest':
                    cv2.putText(frame, f'HELD ECM POSE: timestamp difference {robot_dt:+.3f}s (excluded from errors)', (20, 65), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 200, 255), 2)
                cv2.putText(frame, f'Observed: green | NDI: red | Frame {frame_index} | {status}', (20, 35), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
                if video_writer is not None:
                    video_writer.write(frame)
                if compared and not saved_comparison_overlay:
                    cv2.imwrite(str(output/'comparison.jpg'), frame)
                    saved_comparison_overlay = True
                fw.writerow([frame_index, timestamp, n, compared, status, projected_count, robot_dt, pose_source])
                if counts['frames'] == 1 or (n and not (output/'detected.jpg').exists()):
                    cv2.imwrite(str(output/('detected.jpg' if n else 'first_frame.jpg')), frame)
    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()
    summary = dict(counts, software=dict(python=platform.python_version(), opencv=cv2.__version__, numpy=np.__version__), compared_corners=len(errors), detection_only=detect_only,
                   config=cfg, error_definition='projected minus observed, distorted image pixels')
    if errors:
        summary['error_px'] = statistics(errors)
    if spatial_errors:
        summary['error_mm'] = statistics(spatial_errors)
    summary['spatial_definition'] = 'NDI minus vision PnP board-model coordinates in camera frame; not independently measured corner depths'
    summary['per_corner'] = {str(i): dict(count=len(v['pixel']), error_px=statistics(v['pixel']), **({'error_mm': statistics(v['spatial'])} if v['spatial'] else {})) for i, v in sorted(per_corner.items())}
    (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('config', 'per_corner')}, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT/'configs/nov12.json')
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--detect-only', action='store_true')
    args = parser.parse_args()
    evaluate(args.config, max_frames=args.max_frames, stride=args.stride, detect_only=args.detect_only)
