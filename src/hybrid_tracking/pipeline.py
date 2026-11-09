"""Generate reference board points for the fixed NDI marker."""

from pathlib import Path
import numpy as np

from toolbox.board import generate_board_points, marker_T_board
from toolbox.config import load_config, load_transform, project_path
from toolbox.transforms import transform_points

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs/hybrid.json"


def run(config_path=CONFIG_PATH, project_root=PROJECT_ROOT):
    """Save ID-aligned reference points; no video processing is performed yet."""
    config = load_config(config_path)
    points = generate_board_points(config["charuco"])
    rotation = config.get("marker_R_board")
    if rotation is None:
        raise ValueError(
            "Set marker_R_board in hybrid.json to the measured 3x3 rotation "
            "from board axes into NDI marker axes. Use identity only if aligned."
        )
    position = config["NDI_position"]
    attachment = marker_T_board([position[axis] for axis in "xyz"], rotation)
    ndi_T_marker = load_transform(
        project_path(config["calib"]["ndi_to_marker"], project_root), "ndi_T_marker"
    )
    ndi_T_board = ndi_T_marker @ attachment
    points.update(marker_T_board=attachment, ndi_T_marker=ndi_T_marker,
                  ndi_T_board=ndi_T_board)
    for kind in ("charuco_points", "marker_corners", "marker_centers"):
        board_points = points[f"{kind}_board_m"]
        points[f"{kind}_marker_m"] = transform_points(attachment, board_points)
        points[f"{kind}_ndi_m"] = transform_points(ndi_T_board, board_points)
    output = project_path(config.get("output", {}).get(
        "board_points", "results/board_points.npz"), project_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **points)
    print(f"Saved {len(points['charuco_ids'])} ChArUco intersections and "
          f"{len(points['marker_ids'])} ArUco markers to {output}")
    return points
