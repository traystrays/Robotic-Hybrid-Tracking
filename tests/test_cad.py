"""CAD frame offsets must include the pattern surface and the post arm."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from toolbox.cad import read_apple_geometry


class CadTests(unittest.TestCase):
    def test_pattern_origin_and_post_are_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'board.scad'
            path.write_text('''// sq_size = 999;
                sq_size=19; marker_size=15; board_cols=8; board_rows=6;
                thickness=4; border_size=5; apple_offset_x=0;
                apple_offset_y=0; dx1=40.7941;''')
            data = read_apple_geometry(path)
            np.testing.assert_allclose(data['cross_center_board_m'], [.157,.119,-.004])
            np.testing.assert_allclose(data['positive_x_post_base_board_m'], [.1977941,.119,.0059999])
            self.assertEqual(data['charuco']['square_length'], .019)
            # Unevaluated expressions must fail rather than yield guessed dimensions.
            path.write_text(path.read_text().replace('sq_size=19;', 'sq_size=17+2;'))
            with self.assertRaisesRegex(ValueError, 'sq_size'):
                read_apple_geometry(path)
