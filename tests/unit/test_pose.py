# ==========================================================
# tests/unit/test_pose.py
# ==========================================================
# Unit tests of 2D and 3D poses.

import math

import numpy as np
import pytest

from lidar_sim.core.pose import Pose, rot2


# ------------------------------------------
# test_from_xi_round_trip
# ------------------------------------------
# A pose survives the round trip through (x, y, yaw).
def test_from_xi_round_trip():
    xi = np.array([1.5, -2.0, 2.5])
    np.testing.assert_allclose(Pose.from_xi(xi).xi(), xi, atol=1e-15)


# ------------------------------------------
# test_xi_wraps_yaw
# ------------------------------------------
# Yaw is wrapped to (-π, π].
def test_xi_wraps_yaw():
    p = Pose.from_xi([0.0, 0.0, 3 * math.pi / 2])
    assert p.xi()[2] == pytest.approx(-math.pi / 2)


# ------------------------------------------
# test_apply_and_rotate
# ------------------------------------------
# Applying and rotating points.
def test_apply_and_rotate():
    p = Pose.from_xi([1.0, 2.0, math.pi / 2])
    np.testing.assert_allclose(p.apply(np.array([1.0, 0.0])), [1.0, 3.0], atol=1e-15)
    np.testing.assert_allclose(p.rotate(np.array([[1.0, 0.0], [0.0, 1.0]])),
                               [[0.0, 1.0], [-1.0, 0.0]], atol=1e-15)


# ------------------------------------------
# test_compose_inverse_identity
# ------------------------------------------
# A pose composed with its inverse is the identity.
def test_compose_inverse_identity():
    a = Pose.from_xi([0.3, -1.2, 0.7])
    b = Pose.from_xi([2.0, 0.5, -2.1])
    np.testing.assert_allclose(a.compose(a.inverse()).R, np.eye(2), atol=1e-15)
    np.testing.assert_allclose(a.compose(a.inverse()).t, np.zeros(2), atol=1e-15)
    pts = np.array([[0.1, 0.2], [3.0, -1.0]])
    np.testing.assert_allclose(a.compose(b).apply(pts), a.apply(b.apply(pts)), atol=1e-14)


# ------------------------------------------
# test_dimension_agnostic_3d
# ------------------------------------------
# Poses work in 3D too.
def test_dimension_agnostic_3d():
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    p = Pose(R, np.array([0.0, 0.0, 1.0]))
    assert p.dim == 3
    np.testing.assert_allclose(p.apply(np.array([1.0, 0.0, 0.0])), [0.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        p.xi()                                     # 2D-only parameterization


# ------------------------------------------
# test_validation
# ------------------------------------------
# Non-rotation matrices are rejected.
def test_validation():
    with pytest.raises(ValueError):
        Pose(np.array([[1.0, 0.1], [0.0, 1.0]]), np.zeros(2))     # not orthonormal
    with pytest.raises(ValueError):
        Pose(np.array([[1.0, 0.0], [0.0, -1.0]]), np.zeros(2))    # det = -1
    with pytest.raises(ValueError):
        Pose(np.eye(2), np.zeros(3))                               # shape mismatch


# ------------------------------------------
# test_rot2
# ------------------------------------------
# The 2D rotation matrix.
def test_rot2():
    np.testing.assert_allclose(rot2(math.pi / 2), [[0.0, -1.0], [1.0, 0.0]], atol=1e-15)
