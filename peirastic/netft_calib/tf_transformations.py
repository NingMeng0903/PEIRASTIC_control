"""Small tf.transformations-compatible helpers backed by peirastic.utils.transform_utils."""

from __future__ import annotations

import numpy as np

from peirastic.utils import transform_utils


def euler_matrix(ai, aj, ak, axes: str = "sxyz") -> np.ndarray:
    """Homogeneous transform for Euler angles (same convention as ROS tf)."""
    rmat = transform_utils.euler2mat(np.array([ai, aj, ak], dtype=np.float64))
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = rmat
    return m


def euler_from_matrix(matrix: np.ndarray, axes: str = "sxyz"):
    rmat = np.asarray(matrix, dtype=np.float64)
    if rmat.shape == (4, 4):
        rmat = rmat[:3, :3]
    return transform_utils.mat2euler(rmat, axes=axes)


def rotation_matrix(angle, direction, point=None) -> np.ndarray:
    return transform_utils.rotation_matrix(angle, direction, point=point)
