"""Collect the static-board recording and compare corrections on contiguous time splits."""
import argparse
import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from hybrid_tracking.evaluate import load_array, load_poses, nearest
from toolbox.board import create_board, generate_board_points, marker_T_board


def rows(path):
    with path.open(newline='') as stream:
        yield from csv.DictReader(stream)


def collect(run, cfg):
    inp, cal = cfg['input'], cfg['calibration']
    if inp['ndi_mode'] != 'static' or not inp['static_pose_confirmed']:
        raise ValueError('This study requires a confirmed static NDI recording')
    frames = list(rows(run/'frames.csv'))
    points = generate_board_points(cfg['charuco'])['charuco_points_board_m']
    shape = (len(frames), len(points))
    raw = np.full((*shape, 2), np.nan)
    observed = np.full_like(raw, np.nan)
    vision = np.full((*shape, 3), np.nan)
    for row in rows(run/'projections.csv'):
        raw[int(row['frame']), int(row['charuco_id'])] = row['projected_x_px'], row['projected_y_px']
    for row in rows(run/'corners.csv'):
        key = int(row['frame']), int(row['charuco_id'])
        observed[key] = row['observed_x_px'], row['observed_y_px']
        if row['vision_x_camera_m']:
            vision[key] = [row[f'vision_{a}_camera_m'] for a in 'xyz']
    timestamps = np.array([float(r['timestamp_s']) for r in frames])
    held = np.array([r['robot_pose_source']=='held_nearest' for r in frames])
    # Held-pose green detections were shown in the video but omitted from corners.csv.
    detector = cv2.aruco.CharucoDetector(create_board(cfg['charuco']))
    cap = cv2.VideoCapture(inp['video'])
    last = -1
    for i in np.flatnonzero(held):
        if i != last + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f'Cannot read frame {i}')
        corners, ids, _, _ = detector.detectBoard(frame)
        if ids is not None:
            observed[i, ids.ravel()] = corners.reshape(-1, 2)
        last = i
    cap.release()
    counts = np.isfinite(observed[:,:,0]).sum(axis=1)
    np.testing.assert_array_equal(counts, [int(r['detected_corners']) for r in frames])
    rt, poses = load_poses(inp['robot'], 'robot')
    indices = [nearest(rt, t, np.inf) for t in timestamps]
    ecm = poses[indices]
    np.testing.assert_allclose(rt[indices]-timestamps, [float(r['robot_dt_s']) for r in frames], atol=1e-6)
    nt, ndi = load_poses(inp['ndi'], 'ndi', inp['ndi_tool_id'])
    i = inp.get('static_pose_index', 0)
    base = load_array(ROOT, cal['ndi_T_base'], 'ndi_T_base')
    hand = load_array(ROOT, cal['ecm_T_camera'], 'X')
    mount = marker_T_board([cfg['NDI_position'][a] for a in 'xyz'], cfg['marker_R_board'])
    chain = np.linalg.inv(base @ ecm @ hand) @ ndi[i] @ mount
    return dict(timestamp_s=timestamps, ecm_timestamp_s=rt[indices], base_T_ecm=ecm,
                frame=np.arange(len(frames)), charuco_id=np.arange(len(points)), held_ecm=held,
                observed_px=observed, raw_px=raw, vision_camera_m=vision, board_m=points,
                camera_T_board=chain, ndi_T_base=base, ecm_T_camera=hand, marker_T_board=mount,
                ndi_T_marker=ndi[i], ndi_timestamp_s=nt[i], K=load_array(ROOT, cal['intrinsics'], 'K'),
                distortion=load_array(ROOT, cal['distortion'], 'distCoeffs'))


def rigid(x):
    pose = np.eye(4)
    pose[:3,:3] = cv2.Rodrigues(x[:3])[0]
    pose[:3,3] = x[3:]
    return pose


def predict(kind, x, d, f, ids, return_xyz=False):
    raw = d['raw_px'][f, ids]
    if kind == 'raw':
        return raw
    if kind == 'shift':
        return raw + x
    if kind == 'affine':
        return raw + np.column_stack((raw/1000, np.ones(len(raw)))) @ x.reshape(3,2)
    chain = d['camera_T_board'][f]
    p = d['board_m'][ids]
    if kind in ('board', 'joint'):
        b = rigid(x[:6]); p = p @ b[:3,:3].T + b[:3,3]
    xyz = np.einsum('nij,nj->ni', chain[:,:3,:3], p) + chain[:,:3,3]
    if kind in ('camera', 'joint'):
        c = rigid(x[-6:]); xyz = xyz @ c[:3,:3].T + c[:3,3]
    if return_xyz:
        return xyz
    if np.any(xyz[:,2] <= 0):
        raise ValueError('Correction would move a point behind camera')
    return cv2.projectPoints(xyz, np.zeros(3), np.zeros(3), d['K'], d['distortion'])[0].reshape(-1,2)


def fit(kind, d, f, ids):
    y = d['observed_px'][f,ids]
    delta = y - d['raw_px'][f,ids]
    if kind == 'raw': return np.array([])
    if kind == 'shift': return delta.mean(0)
    if kind == 'affine':
        design = np.column_stack((d['raw_px'][f,ids]/1000, np.ones(len(f))))
        return np.linalg.lstsq(design, delta, rcond=None)[0].ravel()
    # Robust damped Gauss-Newton fit to image residuals. All parameters are shared
    # across corners and frames. The two adjustments in the joint model are empirical.
    x = np.zeros(12 if kind == 'joint' else 6)
    for _ in range(35):
        pred = predict(kind, x, d, f, ids)
        error = y-pred
        weight = np.sqrt(np.minimum(1, 5 / np.maximum(1e-8, np.linalg.norm(error,axis=1))))
        jac = np.column_stack([((predict(kind,x+step,d,f,ids)-pred)*weight[:,None]/1e-6).ravel()
                               for step in np.eye(len(x))*1e-6])
        b = (error*weight[:,None]).ravel()
        dx = np.linalg.solve(jac.T @ jac + np.eye(len(x))*100, jac.T @ b)
        for fraction in (1, .5, .25, .125, .0625):
            trial = x + fraction*dx
            residual = (y-predict(kind,trial,d,f,ids))*weight[:,None]
            if np.sum(residual**2) < b@b:
                x = trial
                break
        else: break
        if np.linalg.norm(fraction*dx) < 1e-7: break
    return x


def metrics(errors):
    return dict(count=len(errors), rmse=float(np.sqrt(np.mean(errors**2))),
                median=float(np.median(errors)), p95=float(np.percentile(errors,95)))


def study(run):
    cfg = json.loads((run/'summary.json').read_text())['config']
    out = run/'correction_study'; out.mkdir(exist_ok=True)
    d = collect(run, cfg)
    f, ids = np.where(np.isfinite(d['observed_px'][:,:,0]) & ~d['held_ecm'][:,None])
    lo, hi = int(f.min()), int(f.max())+1
    cut1, cut2 = [lo+int((hi-lo)*q) for q in (.55,.75)]
    parts = dict(train=(lo,cut1), validation=(cut1+30,cut2), test=(cut2+30,hi))
    report = dict(frame_ranges_exclusive=parts, models={}, config=cfg)
    pop = {k:(f>=a)&(f<b) for k,(a,b) in parts.items()}
    sampled = pop['train'] & ((f-lo)%5==0)
    models = {}
    for kind in ['raw','shift','affine','camera','board','joint']:
        x = fit(kind,d,f[sampled],ids[sampled]); models[kind]=x
        report['models'][kind] = {}
        for name in ('train','validation'):
            ff, ii = f[pop[name]],ids[pop[name]]
            err = predict(kind,x,d,ff,ii)-d['observed_px'][ff,ii]
            report['models'][kind][name] = metrics(np.linalg.norm(err,axis=1))
    winner = min(models,key=lambda k:report['models'][k]['validation']['rmse'])
    report['selected']=winner
    for kind in {winner,'raw'}:
        ff, ii = f[pop['test']],ids[pop['test']]
        error = predict(kind,models[kind],d,ff,ii)-d['observed_px'][ff,ii]
        report['models'][kind]['test'] = metrics(np.linalg.norm(error,axis=1))
        if kind in ('camera','board','joint','raw'):
            xyz = predict('camera' if kind=='raw' else kind,np.zeros(6) if kind=='raw' else models[kind],d,ff,ii,True)
            dist = np.linalg.norm(xyz-d['vision_camera_m'][ff,ii],axis=1)*1000
            report['models'][kind]['test_spatial_mm'] = metrics(dist[np.isfinite(dist)])
    ff, ii = np.indices(d['raw_px'].shape[:2]).reshape(2,-1)
    np.testing.assert_allclose(predict('camera',np.zeros(6),d,ff,ii),d['raw_px'][ff,ii],atol=1e-6)
    d['corrected_px'] = predict(winner,models[winner],d,ff,ii).reshape(d['raw_px'].shape)
    np.savez_compressed(out/'dataset.npz',**d)
    report['parameters']={k:v.tolist() for k,v in models.items()}
    (out/'study.json').write_text(json.dumps(report,indent=2)+'\n')
    print(winner, json.dumps(report['models'],indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    study(parser.parse_args().run)
