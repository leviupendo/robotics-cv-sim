"""tests/test_kinematics.py — Unit tests for FK/IK/Jacobian."""

import numpy as np
import pytest
from src.robot.kinematics import (
    forward_kinematics, inverse_kinematics,
    geometric_jacobian, get_ee_position, DOF
)


class TestForwardKinematics:
    def test_zero_config_shape(self):
        T, transforms = forward_kinematics(np.zeros(DOF))
        assert T.shape == (4, 4)
        assert len(transforms) == DOF

    def test_zero_config_homogeneous(self):
        T, _ = forward_kinematics(np.zeros(DOF))
        assert np.allclose(T[3], [0, 0, 0, 1])

    def test_zero_config_rotation_is_rotation(self):
        T, _ = forward_kinematics(np.zeros(DOF))
        R = T[:3, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-6)

    def test_different_configs_differ(self):
        T1, _ = forward_kinematics(np.zeros(DOF))
        T2, _ = forward_kinematics(np.radians([10, 20, 30, -10, 5, 0]))
        assert not np.allclose(T1, T2)

    def test_wrong_length_raises(self):
        with pytest.raises(AssertionError):
            forward_kinematics(np.zeros(3))


class TestJacobian:
    def test_shape(self):
        J = geometric_jacobian(np.zeros(DOF))
        assert J.shape == (6, DOF)

    def test_numerical_consistency(self):
        """FK position gradient ≈ Jacobian position rows."""
        q = np.radians([10, -20, 15, -45, 5, 30])
        J = geometric_jacobian(q)
        Jp_num = np.zeros((3, DOF))
        eps = 1e-6
        p0 = get_ee_position(q)
        for i in range(DOF):
            dq = np.zeros(DOF); dq[i] = eps
            Jp_num[:, i] = (get_ee_position(q + dq) - p0) / eps
        assert np.allclose(J[:3], Jp_num, atol=1e-4)


class TestInverseKinematics:
    def test_reachable_target(self):
        target = np.array([0.4, 0.0, 0.4])
        q, success, err = inverse_kinematics(target)
        assert success
        assert err < 1e-3

    def test_solution_reaches_target(self):
        target = np.array([0.3, 0.2, 0.3])
        q, success, err = inverse_kinematics(target)
        ee = get_ee_position(q)
        assert np.linalg.norm(ee - target) < 5e-3

    def test_multiple_targets(self):
        targets = [
            np.array([0.4, 0.0, 0.4]),
            np.array([0.2, 0.3, 0.3]),
            np.array([-0.2, 0.3, 0.4]),
        ]
        for t in targets:
            q, ok, err = inverse_kinematics(t)
            assert ok, f"IK failed for target {t}, err={err}"
