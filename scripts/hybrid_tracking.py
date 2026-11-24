"""
Calculate the distance between the reprojected ChArUco keypoints and
the detected keypoints from endoscope images.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hybrid_tracking.evaluate import main


if __name__ == "__main__":
    main()
