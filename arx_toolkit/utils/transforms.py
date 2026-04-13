"""Quaternion / RPY transform utilities for ARX LIFT2.

Ported from the reference codebase (arx_ros2_env_utils.py), keeping only
the functions actually needed by the delta_eef action mode.
"""

from __future__ import annotations

import numpy as np


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Normalize a quaternion [x, y, z, w]."""
    n = float(np.linalg.norm(q))
    if n <= 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return (q / n).astype(np.float32)


def quat_multiply(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    """Hamilton product for quaternions [x, y, z, w]."""
    x0, y0, z0, w0 = [float(v) for v in quat_normalize(q0)]
    x1, y1, z1, w1 = [float(v) for v in quat_normalize(q1)]
    return quat_normalize(np.array([
        w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
        w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
        w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
    ], dtype=np.float32))


def quat_from_rpy(rpy: np.ndarray) -> np.ndarray:
    """Roll/pitch/yaw -> quaternion [x, y, z, w]."""
    roll, pitch, yaw = [float(x) for x in rpy]
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    return quat_normalize(np.array([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ], dtype=np.float32))


def rpy_from_quat(q: np.ndarray) -> np.ndarray:
    """Quaternion [x, y, z, w] -> [roll, pitch, yaw]."""
    q = quat_normalize(q)
    x, y, z, w = [float(v) for v in q]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = np.sign(sinp) * (np.pi / 2.0) if abs(sinp) >= 1.0 else np.arcsin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw], dtype=np.float32)
