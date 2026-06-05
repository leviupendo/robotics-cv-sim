"""
robot_arm.py
============
High-level RobotArm class wrapping kinematics and trajectory modules.
"""

import numpy as np
from .kinematics import (
    forward_kinematics, inverse_kinematics,
    get_ee_position, joint_angles_to_dict, DOF
)
from .trajectory import trapezoidal_profile, linear_interpolation


class RobotArm:
    """6-DOF robot arm with FK, IK, and trajectory support."""

    def __init__(self):
        self.q = np.zeros(DOF)          # current joint angles (rad)
        self.q_home = np.zeros(DOF)     # home configuration
        self.trajectory: list = []       # queued trajectory points
        self._traj_index = 0

    # ── State ──────────────────────────────────────────────────────────────────

    @property
    def ee_position(self) -> np.ndarray:
        return get_ee_position(self.q)

    @property
    def ee_transform(self) -> np.ndarray:
        T, _ = forward_kinematics(self.q)
        return T

    @property
    def joint_transforms(self):
        _, transforms = forward_kinematics(self.q)
        return transforms

    def get_state(self) -> dict:
        T, transforms = forward_kinematics(self.q)
        return {
            "joints_deg": joint_angles_to_dict(self.q),
            "joints_rad": {f"joint_{i+1}": float(v) for i, v in enumerate(self.q)},
            "ee_position": T[:3, 3].tolist(),
            "ee_rotation": T[:3, :3].tolist(),
            "link_positions": [t[:3, 3].tolist() for t in transforms],
        }

    # ── Control ────────────────────────────────────────────────────────────────

    def set_joints(self, q: np.ndarray):
        """Directly set joint angles."""
        self.q = np.array(q, dtype=float)

    def move_to_position(
        self,
        target: np.ndarray,
        duration: float = 2.0,
        freq: int = 100,
    ) -> bool:
        """
        Plan and load a trajectory to reach a Cartesian target position.
        Returns True if IK converged.
        """
        q_goal, success, err = inverse_kinematics(target, q_init=self.q.copy())
        if not success:
            print(f"[RobotArm] IK did not converge. Final error: {err:.4f} m")
        t, q_traj, _ = trapezoidal_profile(self.q, q_goal, duration, freq)
        self.trajectory = q_traj.tolist()
        self._traj_index = 0
        return success

    def step_trajectory(self) -> bool:
        """
        Advance one step along the loaded trajectory.
        Returns False when trajectory is exhausted.
        """
        if self._traj_index >= len(self.trajectory):
            return False
        self.q = np.array(self.trajectory[self._traj_index])
        self._traj_index += 1
        return True

    def go_home(self):
        """Load trajectory back to the home configuration."""
        t, q_traj, _ = trapezoidal_profile(self.q, self.q_home)
        self.trajectory = q_traj.tolist()
        self._traj_index = 0

    # ── Utility ────────────────────────────────────────────────────────────────

    def __repr__(self):
        pos = self.ee_position
        return (
            f"RobotArm(q={np.round(np.degrees(self.q), 1)} deg, "
            f"ee=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] m)"
        )
