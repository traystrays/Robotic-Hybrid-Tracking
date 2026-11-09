"""Geometry and reference artifact checks using synthetic calibration."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from toolbox.board import generate_board_points, marker_T_board
from toolbox.transforms import transform_points
from hybrid_tracking.pipeline import run


class BoardTests(unittest.TestCase):
    def setUp(self):
        self.board = dict(num_cols=7, num_rows=5, square_length=.017,
                          marker_length=.012, dictionary='DICT_4X4_50')

    def test_board_geometry_and_ids(self):
        points = generate_board_points(self.board)
        p = points['charuco_points_board_m']
        self.assertEqual(p.shape, (24, 3))
        np.testing.assert_allclose(p[0], [.017, .068, 0], atol=1e-8)
        np.testing.assert_allclose(p[-1], [.102, .017, 0], atol=1e-8)
        np.testing.assert_array_equal(points['charuco_ids'], np.arange(24))
        self.assertEqual(len(set(points['marker_ids'])), len(points['marker_ids']))
        lengths = np.linalg.norm(np.diff(points['marker_corners_board_m'], axis=1), axis=2)
        np.testing.assert_allclose(lengths, .012, atol=1e-8)

    def test_marker_origin_and_rotated_axes(self):
        center = np.array([.19, .16, .02])
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        transform = marker_T_board(center, rotation)
        np.testing.assert_allclose(transform_points(transform, center), [0, 0, 0])
        np.testing.assert_allclose(transform_points(transform, center + [.01, 0, 0]), [0, .01, 0], atol=1e-12)

    def test_rejects_invalid_geometry_and_rotation(self):
        with self.assertRaises(ValueError):
            generate_board_points(dict(self.board, marker_length=.020))
        with self.assertRaises(ValueError):
            marker_T_board([0, 0, 0], np.diag([1, -1, 1]))

    def test_pipeline_composition_and_saved_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ndi = np.eye(4)
            ndi[:3, 3] = [1, 2, 3]
            (root/'ndi.json').write_text(json.dumps({'ndi_T_marker': ndi.tolist()}))
            cfg = dict(charuco=self.board, NDI_position=dict(x=.19, y=.16, z=.02),
                       marker_R_board=np.eye(3).tolist(), calib={'ndi_to_marker':'ndi.json'})
            path = root/'config.json'
            path.write_text(json.dumps(cfg))
            points = run(path, root)
            np.testing.assert_allclose(points['charuco_points_ndi_m'][0], [.827, 1.908, 2.98], atol=1e-8)
            saved = np.load(root/'results/board_points.npz')
            np.testing.assert_array_equal(saved['charuco_ids'], points['charuco_ids'])
            np.testing.assert_allclose(saved['marker_corners_ndi_m'], points['marker_corners_ndi_m'])
            cfg['marker_R_board'] = None
            path.write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError, 'marker_R_board'):
                run(path, root)


if __name__ == '__main__':
    unittest.main()
