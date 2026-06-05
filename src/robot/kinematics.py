"""
kinematics.py
=============
6-DOF Robot Arm Kinematics using Denavit-Hartenberg parameters.

Implements:
  - Forward Kinematics (FK) via DH transformation matrices
  - Inverse Kinematics (IK) via Jacobian pseudo-inverse iteration
  - Geometric Jacobian computation
"""

import numpy as np
from typing import List, Tuple, Optional


# ── DH Parameters (Panda-inspired 6-DOF) ──────────────────────────────────────
# Each row: [a, d, alpha, theta_offset]
DH_PARAMS = np.array([
    [0.0,     0.333,  0.0,        0.0],   # Joint 1
    [0.0,     0.0,   -np.pi / 2,  0.0],   # Joint 2
    [0.0,     0.316,  np.pi / 2,  0.0],   # Joint 3
    [0.0825,  0.0,    np.pi / 2,  0.0],   # Joint 4
    [-0.0825, 0.384, -np.pi / 2,  0.0],   # Joint 5
    [0.0,     0.0,    np.pi / 2,  0.0],   # Joint 6
])

# Joint limits [min, max] in radians
JOINT_LIMITS = np.array([
    [-2.8973,  2.8973],
    [-1.7628,  1.7628],
    [-2.8973,  2.8973],
    [-3.0718, -0.0698],
    [-2.8973,  2.8973],
    [-0.0175,  3.7525],
])

DOF = 6


def dh_transform(a: float, d: float, alpha: float, theta: float) -> np.ndarray:
    """Compute a single DH transformation matrix T(i-1 -> i)."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct,  -st * ca,  st * sa,  a * ct],
        [st,   ct * ca, -ct * sa,  a * st],
        [0.0,  sa,       ca,       d     ],
        [0.0,  0.0,      0.0,      1.0   ],
    ])


def forward_kinematics(q: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Compute forward kinematics for joint angles q (radians).

    Returns
    -------
    T_ee : (4,4) end-effector transform in base frame
    T_list : list of (4,4) transforms for each joint (useful for visualisation)
    """
    assert len(q) == DOF, f"Expected {DOF} joint angles, got {len(q)}"
    T = np.eye(4)
    transforms = []
    for i in range(DOF):
        a, d, alpha, theta_off = DH_PARAMS[i]
        Ti = dh_transform(a, d, alpha, q[i] + theta_off)
        T = T @ Ti
        transforms.append(T.copy())
    return T, transforms


def get_ee_position(q: np.ndarray) -> np.ndarray:
    """Return (x, y, z) end-effector position for joint angles q."""
    T, _ = forward_kinematics(q)
    return T[:3, 3]


def geometric_jacobian(q: np.ndarray) -> np.ndarray:
    """
    Compute the 6×n geometric Jacobian at joint angles q.

    Returns (6, DOF) matrix where:
      - rows 0:3  → linear velocity contribution (z_i × (p_ee - p_i))
      - rows 3:6  → angular velocity contribution (z_i)
    """
    _, transforms = forward_kinematics(q)
    T_ee = transforms[-1]
    p_ee = T_ee[:3, 3]

    J = np.zeros((6, DOF))
    p_prev = np.zeros(3)
    z_prev = np.array([0.0, 0.0, 1.0])

    for i in range(DOF):
        J[:3, i] = np.cross(z_prev, p_ee - p_prev)
        J[3:, i] = z_prev
        p_prev = transforms[i][:3, 3]
        z_prev = transforms[i][:3, 2]

    return J


def inverse_kinematics(
    target_pos: np.ndarray,
    target_rot: Optional[np.ndarray] = None,
    q_init: Optional[np.ndarray] = None,
    max_iter: int = 500,
    pos_tol: float = 1e-4,
    damping: float = 0.05,
) -> Tuple[np.ndarray, bool, float]:
    """
    Iterative IK via damped least-squares (Levenberg-Marquardt) Jacobian.

    Parameters
    ----------
    target_pos   : (3,) desired end-effector position [x, y, z]
    target_rot   : (3,3) optional desired rotation matrix (ignored if None)
    q_init       : (DOF,) initial joint guess; random if None
    max_iter     : maximum iterations
    pos_tol      : convergence tolerance (metres)
    damping      : damping factor λ for DLS

    Returns
    -------
    q_sol  : (DOF,) joint angles solution
    success: True if converged within tolerance
    error  : final position error magnitude
    """
    if q_init is None:
        q = np.zeros(DOF)
    else:
        q = q_init.copy()

    for _ in range(max_iter):
        T_curr, _ = forward_kinematics(q)
        p_curr = T_curr[:3, 3]
        dp = target_pos - p_curr
        err = np.linalg.norm(dp)

        if err < pos_tol:
            return q, True, float(err)

        J = geometric_jacobian(q)
        Jp = J[:3, :]  # position-only rows

        # Damped Least Squares: dq = Jp^T (Jp Jp^T + λ²I)^{-1} dp
        A = Jp @ Jp.T + (damping ** 2) * np.eye(3)
        dq = Jp.T @ np.linalg.solve(A, dp)

        q = q + dq

        # Enforce joint limits
        q = np.clip(q, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])

    _, _ = forward_kinematics(q)
    final_err = float(np.linalg.norm(target_pos - get_ee_position(q)))
    return q, False, final_err


def joint_angles_to_dict(q: np.ndarray) -> dict:
    """Convert joint angle array to labelled dictionary."""
    return {f"joint_{i + 1}": float(round(np.degrees(qi), 4)) for i, qi in enumerate(q)}
