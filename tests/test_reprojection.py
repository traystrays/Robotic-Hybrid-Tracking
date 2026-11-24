"""Independent pixel expectations and an end-to-end synthetic recording."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from hybrid_tracking.evaluate import compare, evaluate, nearest, load_poses, camera_board_pose
from toolbox.board import create_board


class ReprojectionTests(unittest.TestCase):
    def test_pixel_error_id_order_and_depth(self):
        K = np.array([[100., 0, 50], [0, 100, 60], [0, 0, 1]])
        ids, observed, projected, delta = compare(
            np.array([[.1, .2, 1], [0, 0, -1], [.4, -.2, 2]]),
            [2, 1, 0], [[68, 53], [1, 1], [61, 78]], K, np.zeros(5))
        np.testing.assert_array_equal(ids, [2, 0])
        np.testing.assert_allclose(projected, [[70, 50], [60, 80]])
        np.testing.assert_allclose(delta, [[2, -3], [-1, 2]])
        self.assertIsNone(nearest(np.array([1., 2.]), 3, .1))
        self.assertEqual(nearest(np.array([1., 2.]), 1.9, .2), 1)

    def test_vision_pose_recovers_metric_geometry(self):
        points = np.array([[x, y, 0.] for y in [0., .03, .06] for x in [0., .03, .06]])
        K = np.array([[1000., 0., 960.], [0., 1000., 540.], [0., 0., 1.]])
        r, t = np.array([.2, -.1, .03]), np.array([.01, -.02, .5])
        corners = cv2.projectPoints(points, r, t, K, np.zeros(5))[0]
        rmse, pose = camera_board_pose(points, np.arange(len(points)), corners, K, np.zeros(5))
        self.assertLess(rmse, 1e-6)
        np.testing.assert_allclose(pose[:3, 3], t, atol=1e-8)
        np.testing.assert_allclose(pose[:3, :3], cv2.Rodrigues(r)[0], atol=1e-8)
        self.assertIsNone(camera_board_pose(points[:3], [0, 1, 2], corners[:3], K, np.zeros(5)))

    def test_recording_transform_chain_and_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            board_cfg = dict(num_cols=8, num_rows=6, square_length=.017,
                             marker_length=.012, dictionary='DICT_4X4_50', legacy_pattern=True)
            board = create_board(board_cfg)
            image = board.generateImage((800, 600), marginSize=0)
            image = cv2.copyMakeBorder(image, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=255)
            video = cv2.VideoWriter(str(root/'video.avi'), cv2.VideoWriter_fourcc(*'MJPG'), 10, (900, 700))
            self.assertTrue(video.isOpened())
            video.write(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
            video.write(np.full((700, 900, 3), 255, dtype=np.uint8))
            video.release()
            (root/'times.txt').write_text('2026-01-01T00:00:00+00:00\n2026-01-01T00:00:00.010+00:00\n')
            import datetime
            t = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc).timestamp()
            # Nonidentity base and ECM offsets cancel; camera-board pose is front-facing.
            base = np.eye(4); base[0, 3] = .2
            ecm = np.eye(4); ecm[1, 3] = .1
            ndi_board = np.diag([1., -1., -1., 1.]); ndi_board[:3, 3] = [.2, .202, 1.]
            with (root/'robot.csv').open('w', newline='') as f:
                w = csv.writer(f); w.writerow(['Time_stamp','mip Index (ECM)']+['v']*12)
                w.writerow([t*1e6, 2, 0, .1, 0, *np.eye(3).ravel()])
            with (root/'ndi.csv').open('w', newline='') as f:
                w=csv.writer(f);w.writerow(['Tool ID','Timestamp','Frame #','Tx','Ty','Tz']+['R']*9)
                w.writerow([1,t,0,*(ndi_board[:3,3]*1000),*ndi_board[:3,:3].ravel()])
            K=np.array([[800/.136,0,50],[0,600/.102,50],[0,0,1.]])
            np.savez(root/'cal.npz', ndi_T_base=base, X=np.eye(4), K=K, distCoeffs=np.zeros(5))
            cfg=dict(charuco=board_cfg, NDI_position=dict(x=0,y=0,z=0), marker_R_board=np.eye(3).tolist(),
                calibration={k:dict(path='cal.npz',key=key) for k,key in [('ndi_T_base','ndi_T_base'),('ecm_T_camera','X'),('intrinsics','K'),('distortion','distCoeffs')]},
                input=dict(video='video.avi',timestamps='times.txt',robot='robot.csv',ndi='ndi.csv'),output=dict(directory='out', annotated_video=True))
            (root/'cfg.json').write_text(json.dumps(cfg))
            summary=evaluate(root/'cfg.json', root=root)
            self.assertEqual(summary['compared_corners'],35)
            self.assertEqual(summary['projected_frames'],2)
            with (root/'out/projections.csv').open() as stream:
                projected_rows = list(csv.DictReader(stream))
            self.assertEqual(len(projected_rows),70)
            with (root/'out/frames.csv').open() as stream:
                frame_rows = list(csv.DictReader(stream))
            self.assertEqual(frame_rows[1]['status'],'projected_no_detection')
            self.assertEqual(frame_rows[1]['projected_corners'],'35')
            self.assertLess(summary['error_px']['rmse'],2)
            # Rasterized image corners can differ by a subpixel across OpenCV versions.
            # At 1 m and ~5900 px focal length, 1 px is about 0.17 mm.
            self.assertLess(summary['error_mm']['rmse'], .25)
            self.assertEqual(len(summary['per_corner']),35)
            with (root/'out/corners.csv').open() as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),35)
            annotated = cv2.VideoCapture(str(root/'out/reprojection.mp4'))
            self.assertTrue(annotated.isOpened())
            self.assertTrue(annotated.read()[0])
            self.assertEqual(int(annotated.get(cv2.CAP_PROP_FRAME_COUNT)), 2)
            annotated.release()
            # Synchronization must reject the same data when its clock shifts.
            cfg['input']['video_offset_s']=10
            (root/'cfg.json').write_text(json.dumps(cfg))
            summary=evaluate(root/'cfg.json',root=root)
            self.assertEqual(summary['compared_corners'],0)
            self.assertEqual(summary['unmatched_robot'],2)
            # A held camera pose keeps the entire known board visible without
            # allowing extrapolated observations into accuracy statistics.
            cfg['input'].update(ndi_mode='static', static_pose_confirmed=True)
            cfg['output'].update(hold_nearest_robot_pose=True, annotated_video=True)
            (root/'cfg.json').write_text(json.dumps(cfg))
            summary=evaluate(root/'cfg.json',root=root)
            self.assertEqual(summary['projected_frames'],2)
            self.assertEqual(summary['held_robot_frames'],2)
            self.assertEqual(summary['compared_corners'],0)
            self.assertNotIn('error_px',summary)
            with (root/'out/frames.csv').open() as stream:
                self.assertTrue(all(row['robot_pose_source']=='held_nearest' for row in csv.DictReader(stream)))
            cap=cv2.VideoCapture(str(root/'out/reprojection.mp4'))
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),2)
            cap.set(cv2.CAP_PROP_POS_FRAMES,1)
            ok, frame = cap.read()
            cap.release()
            self.assertTrue(ok)
            red = (frame[:,:,2] > 180) & (frame[:,:,1] < 100) & (frame[:,:,0] < 100)
            self.assertGreater(int(red.sum()),100)

if __name__ == '__main__':
    unittest.main()
