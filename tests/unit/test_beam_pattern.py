# ==========================================================
# tests/unit/test_beam_pattern.py
# ==========================================================
# Unit tests: sub-ray quadrature (E4) and the planar scan pattern (E1, E2, E3).

import math

import numpy as np
import pytest

from lidar_sim.core.config import SensorConfig
from lidar_sim.core.pose import Pose
from lidar_sim.sensor.beam import gauss_hermite_offsets
from lidar_sim.sensor.scan_pattern import Planar2DPattern, ScanPattern, in_geometric_fov


# ------------------------------------------
# test_gauss_hermite_k5_values
# ------------------------------------------
# Five-point Gauss-Hermite offsets and weights match the tabulated values.
def test_gauss_hermite_k5_values():
    delta, w = gauss_hermite_offsets(5, 5e-3)
    np.testing.assert_allclose(delta * 1e3, [-7.142425, -3.389065, 0.0, 3.389065, 7.142425], rtol=5e-7, atol=1e-12)
    np.testing.assert_allclose(w, [0.011257, 0.222076, 0.533333, 0.222076, 0.011257], rtol=5e-5)
    assert abs(w.sum() - 1.0) < 1e-12
    assert np.all(np.diff(delta) > 0)
    np.testing.assert_array_equal(delta, -delta[::-1])


# ------------------------------------------
# test_gauss_hermite_k1_is_ideal_ray
# ------------------------------------------
# One sub-ray is the ideal ray.
def test_gauss_hermite_k1_is_ideal_ray():
    delta, w = gauss_hermite_offsets(1, 5e-3)
    np.testing.assert_array_equal(delta, [0.0])
    np.testing.assert_array_equal(w, [1.0])


# ------------------------------------------
# test_gauss_hermite_moments
# ------------------------------------------
# Quadrature reproduces E[δ²] = θ₀²/4 of the Gaussian profile exp(−2δ²/θ₀²) (E4).
@pytest.mark.parametrize("K", [1, 2, 3, 4, 7, 10])
def test_gauss_hermite_moments(K):
    theta0 = 5e-3
    delta, w = gauss_hermite_offsets(K, theta0)
    assert w.sum() == pytest.approx(1.0, abs=1e-12)
    if K >= 2:
        assert np.sum(w * delta**2) == pytest.approx(theta0**2 / 4, rel=1e-12)


# ------------------------------------------
# test_gauss_hermite_rejects_bad_input
# ------------------------------------------
# Invalid sub-ray counts and divergences are rejected.
@pytest.mark.parametrize("K, theta0", [(0, 5e-3), (3, -1e-3), (3, float("nan"))])
def test_gauss_hermite_rejects_bad_input(K, theta0):
    with pytest.raises(ValueError):
        gauss_hermite_offsets(K, theta0)


# ------------------------------------------
# test_pattern_angles_and_dirs
# ------------------------------------------
# Beam angles and directions of the planar pattern.
def test_pattern_angles_and_dirs():
    pat = Planar2DPattern()
    a = pat.beam_angle()
    assert a.shape == (682,)
    assert a[0] == pytest.approx(-2 * math.pi / 3, abs=1e-15)
    assert a[-1] == pytest.approx(2 * math.pi / 3, rel=1e-12)
    np.testing.assert_allclose(np.diff(a), (4 * math.pi / 3) / 681, rtol=1e-9)
    u = pat.beam_dirs_sensor()
    np.testing.assert_allclose(np.linalg.norm(u, axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(u, np.stack([np.cos(a), np.sin(a)], axis=1), atol=0)
    assert isinstance(pat, ScanPattern) and pat.dim == 2


# ------------------------------------------
# test_pattern_fire_times
# ------------------------------------------
# Beam fire times across the sweep.
def test_pattern_fire_times():
    t = Planar2DPattern().fire_times()
    assert t[0] == 0.0
    assert t[-1] == pytest.approx(0.1 * (240 / 360), rel=1e-12)
    assert np.all(np.diff(t) > 0)


# ------------------------------------------
# test_subrays_shapes_and_offsets
# ------------------------------------------
# Sub-ray shapes and angular offsets.
def test_subrays_shapes_and_offsets():
    cfg = SensorConfig(n_beams=5)
    pat = Planar2DPattern(cfg)
    dirs, w = pat.subrays(3)
    assert dirs.shape == (5, 3, 2) and w.shape == (5, 3)
    delta, wk = gauss_hermite_offsets(3, cfg.theta0)
    ang = pat.beam_angle()[:, None] + delta[None, :]
    np.testing.assert_allclose(dirs, np.stack([np.cos(ang), np.sin(ang)], axis=-1), atol=1e-15)
    np.testing.assert_array_equal(w, np.broadcast_to(wk, (5, 3)))
    d1, w1 = pat.subrays(1)
    np.testing.assert_allclose(d1[:, 0], pat.beam_dirs_sensor(), atol=0)
    np.testing.assert_array_equal(w1, 1.0)


# ------------------------------------------
# test_pattern_arrays_read_only
# ------------------------------------------
# Pattern arrays are read-only.
def test_pattern_arrays_read_only():
    pat = Planar2DPattern()
    with pytest.raises(ValueError):
        pat.beam_angle()[0] = 1.0


# ------------------------------------------
# test_in_geometric_fov_boundaries
# ------------------------------------------
# Field-of-view test at its angular and range bounds.
def test_in_geometric_fov_boundaries():
    cfg = SensorConfig()
    pose = Pose.identity(2)
    a120, a12001 = math.radians(120.0), math.radians(120.01)
    pts = np.array([
        [math.cos(a120), math.sin(a120)],          # boundary angle -> inside (inclusive)
        [math.cos(-a120), math.sin(-a120)],
        [math.cos(a12001), math.sin(a12001)],      # just outside angle
        [5.61, 0.0],                               # beyond r_max
        [5.6, 0.0],                                # r_max inclusive
        [0.02, 0.0],                               # r_min inclusive
        [0.019, 0.0],                              # inside r_min
        [0.0, 0.0],                                # sensor origin
        [-1.0, 0.0],                               # behind
    ])
    np.testing.assert_array_equal(in_geometric_fov(pose, pts, cfg),
                                  [True, True, False, False, True, True, False, False, False])


# ------------------------------------------
# test_in_geometric_fov_uses_pose
# ------------------------------------------
# The field-of-view test follows the sensor pose.
def test_in_geometric_fov_uses_pose():
    cfg = SensorConfig()
    pose = Pose.from_xi([1.0, 1.0, math.pi])
    pts = np.array([[0.0, 1.0], [2.0, 1.0]])           # ahead of / behind the rotated sensor
    np.testing.assert_array_equal(in_geometric_fov(pose, pts, cfg), [True, False])
