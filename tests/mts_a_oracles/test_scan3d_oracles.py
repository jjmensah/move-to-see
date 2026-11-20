# ==========================================================
# tests/mts_a_oracles/test_scan3d_oracles.py
# ==========================================================
# T-S1/T-S2: independent F3–F6 raster, cubature, and FOV oracles.

import numpy as np
import pytest

from lidar_sim.core.pose import Pose
from lidar_sim.sensor.scan3d import Raster3DConfig, Raster3DPattern, in_geometric_fov_3d


# ------------------------------------------
# axis
# ------------------------------------------
# Unit beam axis for an azimuth and elevation.
def axis(az, el):
    return np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])


# ------------------------------------------
# test_ts1_axes_order_and_frame_times
# ------------------------------------------
# Beam axes come in raster order with the right fire times.
def test_ts1_axes_order_and_frame_times():
    cfg = Raster3DConfig(fov=np.pi, n_az=3, el_min=-np.pi/6, el_max=np.pi/6, n_el=3, t_scan=0.18)
    pattern = Raster3DPattern(cfg)
    expected = np.array([axis(a, e) for e in [-np.pi/6, 0, np.pi/6] for a in [-np.pi/2, 0, np.pi/2]])
    np.testing.assert_allclose(pattern.beam_dirs_sensor(), expected, rtol=0, atol=1e-15)
    np.testing.assert_allclose(pattern.fire_times(), np.arange(9)*0.02, rtol=1e-12, atol=0)
    np.testing.assert_allclose(pattern.beam_angle(), np.tile([-np.pi/2, 0, np.pi/2], 3), atol=1e-15)
    np.testing.assert_allclose(pattern.beam_elevation(), np.repeat([-np.pi/6, 0, np.pi/6], 3), atol=1e-15)
    np.testing.assert_allclose(np.linalg.norm(expected, axis=1), 1, atol=1e-15)


# ------------------------------------------
# test_ts1_projected_offsets_and_product_weights
# ------------------------------------------
# Sub-ray offsets and weights follow the Gauss-Hermite product rule.
@pytest.mark.parametrize('K', [1, 9])
def test_ts1_projected_offsets_and_product_weights(K):
    cfg = Raster3DConfig(fov=2.4, n_az=5, el_min=-0.4, el_max=0.7, n_el=4, theta0=0.05)
    pattern = Raster3DPattern(cfg)
    dirs, weights = pattern.subrays(K)
    # H3(x)=8x^3-12x: roots 0, ±sqrt(3/2); normalized weights 2/3, 1/6, 1/6.
    delta = np.array([0.]) if K == 1 else cfg.theta0*np.array([-np.sqrt(3)/2, 0, np.sqrt(3)/2])
    w = np.array([1.]) if K == 1 else np.array([1/6, 2/3, 1/6])
    m = len(w)
    assert dirs.shape == (20, K, 3) and weights.shape == (20, K)
    np.testing.assert_allclose(weights, np.broadcast_to(np.outer(w, w).ravel(), (20, K)), atol=1e-15)
    np.testing.assert_allclose(weights.sum(axis=1), 1, atol=1e-15)
    np.testing.assert_allclose(np.linalg.norm(dirs, axis=2), 1, atol=1e-15)
    for e in range(4):
        el = -0.4 + e*1.1/3
        for c in range(5):
            az = -1.2+c*0.6
            u = axis(az, el)
            ea = np.array([-np.sin(az), np.cos(az), 0])
            ee = np.array([-np.sin(el)*np.cos(az), -np.sin(el)*np.sin(az), np.cos(el)])
            row = dirs[e*5+c]
            np.testing.assert_allclose(np.arctan2(row@ea, row@u), np.repeat(delta, m), atol=1e-12, rtol=0)
            np.testing.assert_allclose(np.arctan2(row@ee, row@u), np.tile(delta, m), atol=1e-12, rtol=0)
            if K == 1:
                np.testing.assert_allclose(row[0], u, atol=1e-15)


# ------------------------------------------
# test_ts1_nonsquare_rejected
# ------------------------------------------
# Sub-ray counts that are not perfect squares are rejected.
@pytest.mark.parametrize('K', [0, 2, 3, 5, 8, 10])
def test_ts1_nonsquare_rejected(K):
    with pytest.raises(ValueError):
        Raster3DPattern().subrays(K)


# ------------------------------------------
# test_ts2_inclusive_bounds_and_just_outside
# ------------------------------------------
# The geometric field of view includes its bounds and excludes points just outside.
@pytest.mark.parametrize('yaw', [0., np.pi/2, -0.71])
def test_ts2_inclusive_bounds_and_just_outside(yaw):
    cfg = Raster3DConfig(fov=2., n_az=5, el_min=-0.3, el_max=0.6, n_el=4)
    # Build world points independently of Pose methods.
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    translation = np.array([2., -1., 0.4])
    polar = [(a, e, r) for a in [-1., 1.] for e in [-0.3, 0.6] for r in [0.02, 5.6]]
    polar += [(-1.00001, 0, 2), (1.00001, 0, 2), (0, -0.30001, 2),
              (0, 0.60001, 2), (0, 0, 0.01999), (0, 0, 5.60001), (np.pi, 0, 2)]
    points = np.array([r*axis(a, e) for a, e, r in polar])@rotation.T + translation
    pose = Pose.from_xyz_yaw(*translation, yaw)
    got = in_geometric_fov_3d(pose, points, cfg, 0.02, 5.6)
    assert got.tolist() == [True]*8 + [False]*7
    # F6 excludes the sensor origin, including when caller supplies a zero lower bound.
    assert not in_geometric_fov_3d(pose, translation[None], cfg, 0., 5.6)[0]


# ------------------------------------------
# test_ts2_every_axis_midrange
# ------------------------------------------
# Every beam axis at mid-range lies inside the field of view.
def test_ts2_every_axis_midrange():
    cfg = Raster3DConfig(n_az=121, n_el=11, el_min=np.deg2rad(-10), el_max=np.deg2rad(40))
    pattern = Raster3DPattern(cfg)
    points = 2.81 * pattern.beam_dirs_sensor()
    assert in_geometric_fov_3d(Pose.identity(3), points, cfg, 0.02, 5.6).all()
