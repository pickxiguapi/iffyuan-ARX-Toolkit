"""Smooth EEF planning helpers."""

from __future__ import annotations

import numpy as np

from arx_toolkit.utils.transforms import quat_from_rpy, quat_normalize, rpy_from_quat


def quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical interpolation for quaternions [x, y, z, w]."""
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return quat_normalize(q0 + float(t) * (q1 - q0))
    theta0 = float(np.arccos(dot))
    sin_theta0 = float(np.sin(theta0))
    theta = theta0 * float(t)
    s0 = np.sin(theta0 - theta) / sin_theta0
    s1 = np.sin(theta) / sin_theta0
    return quat_normalize(s0 * q0 + s1 * q1)


def quat_angle(q0: np.ndarray, q1: np.ndarray) -> float:
    """Shortest angular distance between two quaternions, in radians."""
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)
    dot = float(np.dot(q0, q1))
    dot = float(np.clip(abs(dot), -1.0, 1.0))
    return 2.0 * float(np.arccos(dot))


def trapezoid_params(
    d: float,
    v_max: float,
    a_max: float,
) -> tuple[float, float, float, float]:
    """Return total/accel/flat time and peak velocity for a short motion."""
    if d <= 0.0:
        return 0.0, 0.0, 0.0, 0.0
    v_max = max(float(v_max), 1e-6)
    a_max = max(float(a_max), 1e-6)
    t_accel = v_max / a_max
    d_accel = 0.5 * a_max * t_accel * t_accel
    if d <= 2.0 * d_accel:
        t_accel = float(np.sqrt(d / a_max))
        v_peak = a_max * t_accel
        return 2.0 * t_accel, t_accel, 0.0, v_peak
    t_flat = (d - 2.0 * d_accel) / v_max
    return 2.0 * t_accel + t_flat, t_accel, t_flat, v_max


def trapezoid_position(t: float, d: float, v_max: float, a_max: float) -> float:
    """Distance travelled at time ``t`` under a trapezoid velocity profile."""
    if d <= 0.0:
        return 0.0
    total, t_accel, t_flat, v_peak = trapezoid_params(d, v_max, a_max)
    if t <= 0.0:
        return 0.0
    if t >= total:
        return d
    if t_flat <= 1e-9:
        if t <= t_accel:
            return 0.5 * a_max * t * t
        t_remain = total - t
        return d - 0.5 * a_max * t_remain * t_remain
    d_accel = 0.5 * a_max * t_accel * t_accel
    if t <= t_accel:
        return 0.5 * a_max * t * t
    if t <= t_accel + t_flat:
        return d_accel + v_peak * (t - t_accel)
    t_dec = t - t_accel - t_flat
    return d_accel + v_peak * t_flat + v_peak * t_dec - 0.5 * a_max * t_dec * t_dec


def trapezoid_fraction(t: float, d: float, v_max: float, a_max: float) -> float:
    """Map a trapezoid-profile sample time to an interpolation fraction."""
    if d <= 0.0:
        return 1.0
    return float(np.clip(trapezoid_position(t, d, v_max, a_max) / d, 0.0, 1.0))


def plan_smooth_eef_sequences(
    action: dict[str, np.ndarray],
    start_eef: dict[str, np.ndarray],
    normalize_gripper,
) -> dict[str, list[np.ndarray]]:
    """Plan smooth EEF command sequences from current raw EEF states.

    The numeric parameters are empirical for ARX LIFT2. They are intentionally
    kept local to this planner and are not part of the public ARXEnv API.
    """
    duration_per_step = 1.0 / 20.0
    min_steps = 20
    max_v_xyz = 0.15
    max_a_xyz = 0.1
    max_v_rpy = 0.65
    max_a_rpy = 0.6

    eps_xyz = 5e-4
    eps_rpy = 5e-4
    eps_grip = 5e-4
    results: dict[str, list[np.ndarray]] = {}

    for side, target in action.items():
        start = np.asarray(start_eef[side], dtype=np.float32).reshape(-1)
        target = np.asarray(target, dtype=np.float32).reshape(-1)
        start_gripper = float(normalize_gripper(float(start[6])))
        target_gripper = float(np.clip(target[6], 0.0, 1.0))

        delta_xyz = target[:3] - start[:3]
        d_xyz = float(np.max(np.abs(delta_xyz)))
        q0 = quat_from_rpy(start[3:6])
        q1 = quat_from_rpy(target[3:6])
        d_rpy = quat_angle(q0, q1)
        d_grip = abs(target_gripper - start_gripper)

        t_xyz = (
            trapezoid_params(d_xyz, max_v_xyz, max_a_xyz)[0]
            if d_xyz > eps_xyz
            else 0.0
        )
        t_rpy = (
            trapezoid_params(d_rpy, max_v_rpy, max_a_rpy)[0]
            if d_rpy > eps_rpy
            else 0.0
        )
        steps_xyz = (
            int(np.ceil(t_xyz / duration_per_step)) if t_xyz > 0.0 else 0
        )
        steps_rpy = (
            int(np.ceil(t_rpy / duration_per_step)) if t_rpy > 0.0 else 0
        )
        if steps_xyz > 0:
            steps_xyz = max(steps_xyz, min_steps)
        if steps_rpy > 0:
            steps_rpy = max(steps_rpy, min_steps)
        pose_steps = max(steps_xyz, steps_rpy)
        grip_steps = (
            max(1, int(np.ceil(pose_steps * 0.5)))
            if d_grip > eps_grip and pose_steps > 0
            else 0
        )
        if d_grip > eps_grip and grip_steps == 0:
            grip_steps = max(1, min_steps // 2)
        max_steps = max(pose_steps, grip_steps)
        if max_steps <= 0:
            continue

        seq: list[np.ndarray] = []
        for idx in range(max_steps):
            if d_xyz > eps_xyz:
                progress = min((idx + 1) / float(max(steps_xyz, 1)), 1.0)
                s_xyz = trapezoid_fraction(
                    progress * t_xyz,
                    d_xyz,
                    max_v_xyz,
                    max_a_xyz,
                )
            else:
                s_xyz = 1.0
            if d_rpy > eps_rpy:
                progress = min((idx + 1) / float(max(steps_rpy, 1)), 1.0)
                s_rpy = trapezoid_fraction(
                    progress * t_rpy,
                    d_rpy,
                    max_v_rpy,
                    max_a_rpy,
                )
                rpy = rpy_from_quat(quat_slerp(q0, q1, s_rpy))
            else:
                rpy = start[3:6]

            grip_s = (
                min((idx + 1) / float(grip_steps), 1.0)
                if grip_steps > 0
                else 1.0
            )
            xyz = start[:3] + delta_xyz * s_xyz
            gripper = start_gripper + (target_gripper - start_gripper) * grip_s
            seq.append(np.concatenate([xyz, rpy, [gripper]]).astype(np.float32))
        results[side] = seq
    return results
