"""
trajectory.py
=============
Joint-space trajectory planning with trapezoidal velocity profiling.
"""

import numpy as np
from typing import List, Tuple
from .kinematics import DOF, JOINT_LIMITS


def trapezoidal_profile(
    q_start: np.ndarray,
    q_end: np.ndarray,
    duration: float = 2.0,
    freq: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a trapezoidal velocity profile trajectory.

    Returns time, position, and velocity arrays.
    """
    t = np.linspace(0, duration, int(duration * freq))
    n = len(t)
    accel_frac = 0.3  # fraction of duration for accel/decel phase

    q_traj = np.zeros((n, DOF))
    qd_traj = np.zeros((n, DOF))
    dq = q_end - q_start

    t1 = duration * accel_frac
    t2 = duration * (1 - accel_frac)

    for i, ti in enumerate(t):
        if ti <= t1:
            s = (ti ** 2) / (2 * t1 * (t2 - t1 + duration * accel_frac))
            sd = ti / (t1 * (t2 - t1 + duration * accel_frac))
        elif ti <= t2:
            s_t1 = t1 / (2 * (t2 - t1 + duration * accel_frac))
            s = s_t1 + (ti - t1) / (t2 - t1 + duration * accel_frac)
            sd = 1.0 / (t2 - t1 + duration * accel_frac)
        else:
            tr = duration - ti
            s = 1 - (tr ** 2) / (2 * t1 * (t2 - t1 + duration * accel_frac))
            sd = tr / (t1 * (t2 - t1 + duration * accel_frac))

        q_traj[i] = q_start + s * dq
        qd_traj[i] = sd * dq

    return t, np.clip(q_traj, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1]), qd_traj


def linear_interpolation(
    q_start: np.ndarray,
    q_end: np.ndarray,
    steps: int = 50,
) -> np.ndarray:
    """Simple linear joint-space interpolation."""
    return np.array([
        q_start + (q_end - q_start) * t for t in np.linspace(0, 1, steps)
    ])


def multi_point_trajectory(
    waypoints: List[np.ndarray],
    duration_per_segment: float = 1.5,
    freq: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Chain multiple trapezoidal segments through a list of waypoints.

    Returns (time, q_trajectory) arrays.
    """
    all_t, all_q = [], []
    t_offset = 0.0

    for i in range(len(waypoints) - 1):
        t, q, _ = trapezoidal_profile(
            waypoints[i], waypoints[i + 1], duration_per_segment, freq
        )
        all_t.append(t + t_offset)
        all_q.append(q)
        t_offset += duration_per_segment

    return np.concatenate(all_t), np.vstack(all_q)
