from __future__ import annotations

import numpy as np


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Multiply two quaternions using the Hamilton product.

    Args:
        q1: Quaternion in (qx, qy, qz, qw) format.
        q2: Quaternion in (qx, qy, qz, qw) format.

    Returns:
        Quaternion in (qx, qy, qz, qw) format as float32.
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float32,
    )


def quat_to_rotation_matrix(q: np.ndarray, return_6d: bool = False) -> np.ndarray:
    """
    Convert quaternion (qx, qy, qz, qw) to either:
      - flattened 3x3 rotation matrix, shape (9,), row-major
      - 6D rotation representation, shape (6,), first two columns

    Args:
        q: Quaternion in (qx, qy, qz, qw) format.
        return_6d: If True, return first two columns of rotation matrix.

    Returns:
        np.ndarray of dtype float32.
    """
    q = np.asarray(q, dtype=np.float32)
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        raise ValueError(f"Cannot convert near-zero quaternion: norm={norm}")
    qx, qy, qz, qw = q / norm

    r00 = 1 - 2 * (qy * qy + qz * qz)
    r01 = 2 * (qx * qy - qz * qw)
    r02 = 2 * (qx * qz + qy * qw)

    r10 = 2 * (qx * qy + qz * qw)
    r11 = 1 - 2 * (qx * qx + qz * qz)
    r12 = 2 * (qy * qz - qx * qw)

    r20 = 2 * (qx * qz - qy * qw)
    r21 = 2 * (qy * qz + qx * qw)
    r22 = 1 - 2 * (qx * qx + qy * qy)

    if return_6d:
        return np.array([r00, r10, r20, r01, r11, r21], dtype=np.float32)

    return np.array(
        [
            r00, r01, r02,
            r10, r11, r12,
            r20, r21, r22,
        ],
        dtype=np.float32,
    )


def gram_schmidt_6d_to_rotation_matrix(a1: np.ndarray, a2: np.ndarray) -> np.ndarray:
    """
    Convert 6D rotation representation to 3x3 rotation matrix using Gram-Schmidt.

    Args:
        a1: First column vector (3,)
        a2: Second column vector (3,)

    Returns:
        3x3 rotation matrix, or identity matrix if inputs are invalid.

    The 6D representation encodes the first two columns of a rotation matrix.
    This function orthonormalizes them and computes the third column via cross product.
    """
    # Validate vectors are not zero or near-zero.
    norm_a1 = np.linalg.norm(a1)
    norm_a2 = np.linalg.norm(a2)

    if norm_a1 < 1e-6 or norm_a2 < 1e-6:
        print("Warning: Invalid 6D representation with near-zero vectors!")
        print(f"  ||a1||: {norm_a1:.6f}, ||a2||: {norm_a2:.6f}")
        return np.eye(3)

    # Gram-Schmidt orthogonalization.
    b1 = a1 / norm_a1
    b2 = a2 - np.dot(a2, b1) * b1
    norm_b2 = np.linalg.norm(b2)

    if norm_b2 < 1e-6:
        print("Warning: Vectors a1 and a2 are nearly collinear!")
        print(f"  ||b2|| after orthogonalization: {norm_b2:.6f}")
        return np.eye(3)

    b2 = b2 / norm_b2
    b3 = np.cross(b1, b2)
    rot = np.stack((b1, b2, b3), axis=1)

    # Validate the resulting rotation matrix.
    det = np.linalg.det(rot)
    if np.abs(det - 1.0) > 1e-3:
        print(f"Warning: 6D->rotation matrix has bad determinant: {det:.6f}")
        return np.eye(3)

    return rot
    