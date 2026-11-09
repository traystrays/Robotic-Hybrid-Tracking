# Hybrid Charuco and NDI tracking
This repository explores a hybrid tracking system that combines a Charuco board with a NDI markers. The Charuco board allows robot cameras to locate keypoints in the robot frane, and the NDI marker allows each each keypoint to be known in the NDi coordinate system.

# Why a Charuco Board? 
ChArUco board is a checkerboard and ArUco markers merged together. The numbered ArUco markers establish known reference points and help OpenCV predict where the black-and-white checkerboard intersections will be at. 

1. Higher precision using a ChArUco board
2. Robust against partial visibility: Knows exactly which part of the board we are looking at, despite parts of the board being out of frame.
3. Robust correspondence: Reducing ambiguity when matching image points, especially needed for mtching "ground-truth NDI" keypoints against observed keypoints.

