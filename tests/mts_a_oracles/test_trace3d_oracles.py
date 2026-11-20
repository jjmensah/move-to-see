# ==========================================================
# tests/mts_a_oracles/test_trace3d_oracles.py
# ==========================================================
# T-S3/T-G3: F1, E7–E8 hand radiometry, skew, and planar embedding.

import numpy as np
import pytest

from lidar_sim.core.config import EnvConfig, NoiseConfig, SensorConfig
from lidar_sim.core.pose import Pose
from lidar_sim.core.types import HitBatch
from lidar_sim.geometry.scene2d import Scene2D
from lidar_sim.geometry.scene3d import Scene3D
from lidar_sim.sensor.lidar import LidarSensor, radiometric_from_hits, scan_rays
from lidar_sim.sensor.scan3d import Raster3DConfig, Raster3DPattern
from lidar_sim.sensor.scan_pattern import Planar2DPattern
from lidar_sim.world.materials import GREY, MATERIALS, WHITE


# ------------------------------------------
# test_ts3_pose_f1
# ------------------------------------------
# The sensor pose from position and yaw (F1).
@pytest.mark.parametrize('yaw', [0, np.pi/2, -0.73, np.pi])
def test_ts3_pose_f1(yaw):
    pose = Pose.from_xyz_yaw(1.2, -3.4, 0.4, yaw)
    c, s = np.cos(yaw), np.sin(yaw)
    np.testing.assert_allclose(pose.R, [[c, -s, 0], [s, c, 0], [0, 0, 1]], atol=1e-15)
    np.testing.assert_array_equal(pose.t, [1.2, -3.4, 0.4])


# ------------------------------------------
# test_ts3_box_power_by_hand
# ------------------------------------------
# Return power from a box face matches a hand calculation, with and without fog.
@pytest.mark.parametrize('visibility,q', [(20000., 1.3), (20., 0.)])
def test_ts3_box_power_by_hand(visibility, q):
    # Central ray (0.8,0,0.6) hits x=4 at (4,0,3.4): r=5, cos=0.8.
    el = np.arctan2(3, 4)
    cfg = Raster3DConfig(fov=0.2, n_az=3, n_el=1, el_min=el, el_max=el)
    noise = NoiseConfig(P0=2., A_rx=0.01, eta=0.7)
    sensor = LidarSensor(cfg.sensor_config(n_subrays=1), noise, Raster3DPattern(cfg))
    scene = Scene3D([[4, -2, 0]], [[5, 2, 5]], [WHITE], [42])
    trace = sensor.trace(Pose.from_xyz_yaw(0, 0, 0.4, 0), scene,
                         EnvConfig(visibility_m=visibility, P_amb=1e-6))
    sigma = 3.912/visibility * (785/550)**(-q)
    expected = 2 * (0.9/np.pi) * 0.8 * (0.01/25) * np.exp(-2*sigma*5) * 0.7
    assert trace.hit[1, 0]
    np.testing.assert_allclose(trace.r_path[1, 0], 5, rtol=1e-12)
    np.testing.assert_allclose(trace.cos_inc[1, 0], 0.8, rtol=1e-12)
    np.testing.assert_allclose(trace.power[1, 0], expected, rtol=1e-12, atol=0)
    assert trace.P_amb == 1e-6


# ------------------------------------------
# test_ts3_radiometric_from_synthetic_hits
# ------------------------------------------
# Radiometry from hand-made hits matches the formulas.
@pytest.mark.parametrize('sigma', [0., 3.912/20])
def test_ts3_radiometric_from_synthetic_hits(sigma):
    # Supply hand-built hits, so this tests radiometry independently of ray casting.
    d = np.array([[[0.8, 0., 0.6]], [[1., 0., 0.]]])
    hits = HitBatch(hit=[True, False], r=[5., np.nan], point=[[4, 0, 3], [0, 0, 0]],
                    normal=[[-1, 0, 0], [0, 0, 0]], material_id=[WHITE, -1],
                    object_id=[42, -1], surface_id=[0, -1])
    trace = radiometric_from_hits(Pose.identity(3), np.array([2., 2.1]), d, np.ones((2, 1)),
                                  hits, MATERIALS, sigma, NoiseConfig(P0=2, A_rx=0.01, eta=0.7), 0.)
    expected = 2*0.9/np.pi*0.8*0.01/25*0.7*np.exp(-10*sigma)
    np.testing.assert_allclose(trace.power[:, 0], [expected, 0], rtol=1e-12, atol=0)
    np.testing.assert_allclose(trace.cos_inc[0, 0], 0.8, rtol=1e-12)
    assert np.isnan(trace.r_path[1, 0])


# ------------------------------------------
# test_ts3_scan_rays_absolute_time_and_moving_pose
# ------------------------------------------
# Scan rays use absolute fire times and the moving sensor pose.
def test_ts3_scan_rays_absolute_time_and_moving_pose():
    cfg = Raster3DConfig(fov=np.pi/2, n_az=3, n_el=1, el_min=0, el_max=0, t_scan=0.3)
    pose = lambda t: Pose.from_xyz_yaw(2*t, -t, 0.4, np.pi/2)
    tau, d, w, origins = scan_rays(Raster3DPattern(cfg), 1, pose, t0=4)
    np.testing.assert_allclose(tau, [4, 4.1, 4.2], atol=1e-15)
    np.testing.assert_allclose(origins, [[8, -4, 0.4], [8.2, -4.1, 0.4], [8.4, -4.2, 0.4]], atol=1e-15)
    h = np.sqrt(0.5)
    np.testing.assert_allclose(d[:, 0], [[h, h, 0], [0, 1, 0], [-h, h, 0]], atol=1e-15)
    np.testing.assert_array_equal(w, np.ones((3, 1)))


# ------------------------------------------
# test_tg3_extruded_walls_match_2d_and_hand_ranges
# ------------------------------------------
# Walls extruded into 3D give the 2D ranges and the hand-computed ones.
@pytest.mark.parametrize('yaw', [0., 0.37])
def test_tg3_extruded_walls_match_2d_and_hand_ranges(yaw):
    # Three inner faces of a rectangular room; rear opening also exercises misses.
    segments = np.array([[[4, -3], [4, 3]], [[-2, 3], [4, 3]], [[-2, -3], [4, -3]]], float)
    scene2 = Scene2D(segments, [WHITE, GREY, WHITE], [1, 2, 3])
    scene3 = Scene3D([[4, -3, -2], [-2, 3, -2], [-2, -3.1, -2]],
                    [[4.1, 3, 2], [4, 3.1, 2], [4, -3, 2]], [WHITE, GREY, WHITE], [1, 2, 3])
    cfg2 = SensorConfig(n_beams=31, fov=4*np.pi/3, n_subrays=1)
    cfg3 = Raster3DConfig(n_az=31, fov=4*np.pi/3, n_el=1, el_min=0, el_max=0)
    t2 = LidarSensor(cfg2, pattern=Planar2DPattern(cfg2)).trace(Pose.from_xi([0, 0, yaw]), scene2, EnvConfig())
    t3 = LidarSensor(cfg3.sensor_config(n_subrays=1), pattern=Raster3DPattern(cfg3)).trace(
        Pose.from_xyz_yaw(0, 0, 0.4, yaw), scene3, EnvConfig())
    np.testing.assert_array_equal(t3.hit, t2.hit)
    for field in ('r_path', 'cos_inc', 'power'):
        np.testing.assert_allclose(getattr(t3, field), getattr(t2, field), rtol=1e-12, atol=0, equal_nan=True)
    # Also use bounded 2D plane intersections, so agreement cannot hide shared bugs.
    sigma = 3.912/20000*(785/550)**-1.3
    for i in range(31):
        a = -2*np.pi/3+i*4*np.pi/(3*30)+yaw
        dx, dy = np.cos(a), np.sin(a)
        candidates = []
        if dx > 0 and -3 <= 4*dy/dx <= 3:
            candidates.append((4/dx, abs(dx), 0.9))
        for y, rho in [(3, 0.5), (-3, 0.9)]:
            if dy != 0 and y/dy > 0 and -2 <= y*dx/dy <= 4:
                candidates.append((y/dy, abs(dy), rho))
        assert t3.hit[i, 0] == bool(candidates)
        if candidates:
            r, cosine, rho = min(candidates)
            p = 0.030*rho/np.pi*cosine*(np.pi*0.01**2)/r**2*0.8*np.exp(-2*sigma*r)
            np.testing.assert_allclose([t3.r_path[i, 0], t3.cos_inc[i, 0], t3.power[i, 0]],
                                       [r, cosine, p], rtol=1e-12, atol=0)


# ------------------------------------------
# test_ts3_k9_radiometric_invariants
# ------------------------------------------
# Nine-sub-ray traces keep the radiometric invariants.
def test_ts3_k9_radiometric_invariants(rng):
    cfg = Raster3DConfig(n_az=17, n_el=5, el_min=-0.4, el_max=0.6)
    lo = rng.uniform([1, -3, 0], [4, 2, 1], (6, 3))
    scene = Scene3D(lo, lo+rng.uniform(0.3, 1, (6, 3)), [WHITE]*6, np.arange(6)+1,
                    ground_material=GREY)
    trace = LidarSensor(cfg.sensor_config(n_subrays=9), pattern=Raster3DPattern(cfg)).trace(
        Pose.from_xyz_yaw(0, 0, 0.4, 0), scene, EnvConfig())
    assert trace.dir_world.shape == (85, 9, 3) and trace.hit.shape == (85, 9)
    assert trace.hit.any() and (~trace.hit).any()
    np.testing.assert_allclose(np.linalg.norm(trace.dir_world, axis=2), 1, atol=1e-12)
    np.testing.assert_allclose(trace.w.sum(axis=1), 1, atol=1e-12)
    assert np.all(trace.w >= 0)
    np.testing.assert_array_equal(trace.sub_idx, np.tile(np.arange(9), (85, 1)))
    assert np.isfinite(trace.r_path[trace.hit]).all() and (trace.r_path[trace.hit] > 0).all()
    assert np.isnan(trace.r_path[~trace.hit]).all()
    assert np.isfinite(trace.power).all() and (trace.power >= 0).all()
    assert (trace.power[~trace.hit] == 0).all() and not trace.bounce.any()
    np.testing.assert_allclose(np.linalg.norm(trace.normal[trace.hit], axis=1), 1, atol=1e-12)
    cos = -np.sum(trace.normal * trace.dir_world, axis=2)
    assert (cos[trace.hit] >= -1e-12).all()
    np.testing.assert_allclose(trace.cos_inc[trace.hit], cos[trace.hit], atol=1e-12)
    assert ((trace.cos_inc >= 0) & (trace.cos_inc <= 1)).all()
    for field in ('object_id', 'material_id', 'surface_id'):
        ids = getattr(trace, field)
        assert (ids[~trace.hit] == -1).all() and (ids[trace.hit] >= 0).all()
