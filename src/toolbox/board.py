"""ChArUco points with OpenCV IDs and a bottom-left board reference frame.

Board axes: x right, y up, z out of the printed face. num_cols/num_rows
count squares, not internal intersections. Distances are in metres.
"""

import cv2
import numpy as np

from toolbox.transforms import transform_points, validate_rotation


def create_board(config):
    cols, rows = config["num_cols"], config["num_rows"]
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 2 for n in (cols, rows)):
        raise ValueError("num_cols and num_rows must be integer square counts >= 2.")
    square, marker = float(config["square_length"]), float(config["marker_length"])
    if not (np.isfinite(square) and np.isfinite(marker) and 0 < marker < square):
        raise ValueError("Require 0 < marker_length < square_length (metres).")
    name = config["dictionary"]
    if not name.startswith("DICT_") or not hasattr(cv2.aruco, name):
        raise ValueError(f"Unknown ArUco dictionary: {name}")
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))
    board = cv2.aruco.CharucoBoard((cols, rows), square, marker, dictionary)
    if "legacy_pattern" in config:
        if not isinstance(config["legacy_pattern"], bool):
            raise ValueError("legacy_pattern must be a boolean.")
        board.setLegacyPattern(config["legacy_pattern"])
    return board


def generate_board_points(config):
    """Return ID-aligned intersections, marker corners, and marker centers."""
    board = create_board(config)
    # OpenCV's printed-board coordinates use x right, y down. Flip y and z
    # together for a proper rotation, and move the origin to bottom left.
    board_T_opencv = np.diag([1., -1., -1., 1.])
    board_T_opencv[1, 3] = config["num_rows"] * config["square_length"]
    corners = transform_points(board_T_opencv, board.getChessboardCorners())
    marker_corners = transform_points(board_T_opencv, np.asarray(board.getObjPoints()))
    return {
        "charuco_ids": np.arange(len(corners), dtype=np.int32),
        "charuco_points_board_m": corners,
        "marker_ids": np.asarray(board.getIds(), dtype=np.int32).reshape(-1),
        "marker_corners_board_m": marker_corners,
        "marker_centers_board_m": marker_corners.mean(axis=1),
        "board_T_opencv": board_T_opencv,
    }


def marker_T_board(marker_center_board_m, marker_R_board):
    """Use the measured NDI marker center in board coordinates and orientation.

    marker_R_board rotates board-coordinate vectors into NDI marker axes.
    This is the tracked NDI rigid body, not a printed ArUco marker.
    """
    center = np.asarray(marker_center_board_m, dtype=float)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("NDI_position must contain finite x, y, z in metres.")
    transform = np.eye(4)
    transform[:3, :3] = validate_rotation(marker_R_board)
    transform[:3, 3] = -transform[:3, :3] @ center
    return transform
