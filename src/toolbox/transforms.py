"""Rigid transforms: A_T_B maps points in frame B into frame A; metres."""

import numpy as np


def validate_rotation(value):
    rotation = np.asarray(value, dtype=float)
    if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6, rtol=0)):
        raise ValueError("Expected a finite proper 3x3 rotation matrix.")
    return rotation


def validate_transform(value):
    transform = np.asarray(value, dtype=float)
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8, rtol=0)):
        raise ValueError("Expected a finite homogeneous 4x4 transform.")
    validate_rotation(transform[:3, :3])
    return transform


def transform_points(transform, points):
    """Apply a rigid transform to an (..., 3) array without changing its shape."""
    transform = validate_transform(transform)
    points = np.asarray(points, dtype=float)
    if points.ndim < 1 or points.shape[-1] != 3 or not np.isfinite(points).all():
        raise ValueError("Points must be finite with shape (..., 3).")
    return points @ transform[:3, :3].T + transform[:3, 3]
