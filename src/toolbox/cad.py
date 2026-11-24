"""Extract literal dimensions from the supplied Apple-1/ChArUco SCAD design.

This reads parameters only; it never evaluates OpenSCAD or guesses ROM axes.
The board reference is the bottom-left of the printed pattern at its top face.
"""
import hashlib
from pathlib import Path
import re

import numpy as np


def read_apple_geometry(path):
    path = Path(path)
    source = path.read_text()
    # Remove comments so documentation cannot override real parameters.
    text = re.sub(r'/\*.*?\*/|//[^\n]*', '', source, flags=re.S)

    def number(name):
        matches = re.findall(r'\b' + re.escape(name) + r'\s*=\s*([-+]?\d+(?:\.\d+)?)\s*;', text)
        if len(matches) != 1:
            raise ValueError(f'Expected one literal assignment for {name} in {path}')
        return float(matches[0])

    square, marker = number('sq_size'), number('marker_size')
    cols, rows = number('board_cols'), number('board_rows')
    if cols != int(cols) or rows != int(rows) or min(cols, rows) < 2 or not 0 < marker < square:
        raise ValueError('Invalid CAD board dimensions')
    thickness, border = number('thickness'), number('border_size')
    anchor = np.array([cols*square + border + number('apple_offset_x'),
                       rows*square + border + number('apple_offset_y'), -thickness])
    # The assembly has a pure translation; no rotation is applied to the cross.
    # A reflector post location is NOT the tracked tool origin unless the ROM
    # or a separate measurement explicitly defines it that way.
    return dict(source=str(path.resolve()), sha256=hashlib.sha256(source.encode()).hexdigest(),
                charuco=dict(num_cols=int(cols), num_rows=int(rows), square_length=square/1000,
                             marker_length=marker/1000, dictionary='DICT_4X4_250', legacy_pattern=True),
                cross_center_board_m=(anchor/1000).tolist(),
                positive_x_post_base_board_m=((anchor + [number('dx1'), 0, 9.9999])/1000).tolist(),
                positive_x_arm_length_m=number('dx1')/1000,
                board_surface_height_cad_m=thickness/1000,
                cad_R_board=np.eye(3).tolist(),
                note='Cross center z is the CAD underside plane, not an inferred reflector-center plane. ROM-to-CAD registration is separate.')
