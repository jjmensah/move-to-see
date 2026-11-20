# ==========================================================
# tests/oracles/test_oracles_sensor.py
# ==========================================================
# Hand-computed sensor, radiometry, and atmosphere oracles.

import numpy as np

from lidar_sim.core.config import EnvConfig, SensorConfig
from lidar_sim.core.pose import Pose
from lidar_sim.physics.atmosphere import extinction, transmission
from lidar_sim.physics.radiometry import lambertian_power
from lidar_sim.physics.specular import bounce_power, reflect
from lidar_sim.sensor.beam import gauss_hermite_offsets
from lidar_sim.sensor.lidar import LidarSensor
from lidar_sim.sensor.scan_pattern import Planar2DPattern, in_geometric_fov
from lidar_sim.world.scenes import mirror_room, single_wall


# ------------------------------------------
# test_o01_default_beam_angles
# ------------------------------------------
# O-01, E1: default scan spans the inclusive ±120 degree endpoints.
def test_o01_default_beam_angles():
    pattern = Planar2DPattern()
    angles = pattern.beam_angle()
    assert angles.shape == (682,)
    np.testing.assert_allclose(angles[[0, -1]], [-2 * np.pi / 3, 2 * np.pi / 3], rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(np.diff(angles), 4 * np.pi / (3 * 681), rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(np.linalg.norm(pattern.beam_dirs_sensor(), axis=1), 1.0, rtol=0.0, atol=1e-12)


# ------------------------------------------
# test_o02_default_firing_times
# ------------------------------------------
# O-02, E2: firing time is proportional to beam index over the scan sector.
def test_o02_default_firing_times():
    times = Planar2DPattern().fire_times()
    assert times.shape == (682,)
    assert times[0] == 0.0
    np.testing.assert_allclose(times[-1], (2 / 3) * 0.1, rtol=1e-12, atol=0.0)
    assert np.all(np.diff(times) > 0.0)


# ------------------------------------------
# test_o03_geometric_fov_boundary_inclusive
# ------------------------------------------
# O-03, E3: sector and radial boundaries are inclusive.
def test_o03_geometric_fov_boundary_inclusive():
    cfg = SensorConfig()
    pose = Pose.identity(2)
    points = np.array([
        [np.cos(2 * np.pi / 3), np.sin(2 * np.pi / 3)],
        [np.cos(np.deg2rad(120.01)), np.sin(np.deg2rad(120.01))],
        [5.61, 0.0],
        [cfg.r_min, 0.0],
    ])
    assert in_geometric_fov(pose, points, cfg).tolist() == [True, False, False, True]


# ------------------------------------------
# test_o04_gauss_hermite_five_offsets
# ------------------------------------------
# O-04, E4: five point Hermite quadrature is symmetric and normalized.
def test_o04_gauss_hermite_five_offsets():
    delta, weights = gauss_hermite_offsets(5, 5e-3)
    # Hermite roots ±2.0201828704560856, ±0.9585724646138185, 0 and
    # normalized weights 0.0112574113277207, 0.2220759220056126, 0.5333333333333333.
    np.testing.assert_allclose(delta * 1e3,
                               [-7.142425, -3.389065, 0.0, 3.389065, 7.142425],
                               rtol=5e-7, atol=0.0)
    np.testing.assert_allclose(weights,
                               [0.0112574113277207, 0.2220759220056126, 0.5333333333333333,
                                0.2220759220056126, 0.0112574113277207],
                               rtol=5e-7, atol=0.0)
    assert np.all(np.diff(delta) > 0.0)
    np.testing.assert_allclose(weights.sum(), 1.0, rtol=0.0, atol=1e-12)


# ------------------------------------------
# test_o07_lambertian_scalings_and_single_wall_trace
# ------------------------------------------
# O-07, E6-E7: power scales as cos(theta)/r² in clear air.
def test_o07_lambertian_scalings_and_single_wall_trace():
    args = {"P0": 1.0, "w": 1.0, "rho": 0.9, "cos_inc": 1.0,
            "A_rx": 1.0, "eta": 1.0, "sigma_ext": 0.0}
    p2 = lambertian_power(r=np.array([2.0]), **args)[0]
    p4 = lambertian_power(r=np.array([4.0]), **args)[0]
    p60 = lambertian_power(r=np.array([2.0]), cos_inc=0.5, **{k: v for k, v in args.items() if k != "cos_inc"})[0]
    np.testing.assert_allclose(p4 / p2, 0.25, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(p60 / p2, 0.5, rtol=1e-12, atol=0.0)

    sensor_cfg = SensorConfig(n_beams=3, fov=0.01, n_subrays=1, theta0=0.0)
    trace = LidarSensor(sensor=sensor_cfg).trace(Pose.identity(2), single_wall(), EnvConfig())
    centre = 1                                        # odd B = 3: beam 1 points along +x; return slot 0 (K = 1)
    np.testing.assert_allclose(trace.r_path[centre, 0], 2.0, rtol=1e-12)
    assert trace.hit[centre, 0]


# ------------------------------------------
# test_o08_kim_extinction_and_transmission
# ------------------------------------------
# O-08, E8: extinction uses the piecewise Kim visibility exponent.
def test_o08_kim_extinction_and_transmission():
    expected = [1.231714e-4, 1.546658e-3, 4.394979e-3, 3.912e-2, 7.824e-2]
    expected_t2 = [0.999015, 0.987703, 0.965451, 0.731279, 0.534769]
    for visibility, sigma, transmission_squared in zip(
        [20000.0, 2000.0, 800.0, 100.0, 50.0], expected, expected_t2, strict=True
    ):
        got = extinction(visibility, 785e-9)
        np.testing.assert_allclose(got, sigma, rtol=5e-7)
        np.testing.assert_allclose(transmission(np.array([4.0]), got)[0] ** 2,
                                   transmission_squared, rtol=5e-7)


# ------------------------------------------
# test_o09_mirror_reflection_and_bounce_power
# ------------------------------------------
# O-09, E9: the mirror reflects +x upward and yields a 4 m image path.
def test_o09_mirror_reflection_and_bounce_power():
    d_reflected = reflect(np.array([[1.0, 0.0]]), np.array([[1 / np.sqrt(2), -1 / np.sqrt(2)]]))
    np.testing.assert_allclose(d_reflected, [[0.0, 1.0]], atol=1e-12)
    # From (1,1.3), mirror at x=3.3 then ceiling y=3: 2.3 + 1.7 = 4 m.
    trace = LidarSensor(sensor=SensorConfig(specular=True, n_beams=3, fov=0.01,
                                           n_subrays=1, theta0=0.0)).trace(
        Pose(np.eye(2), np.array([1.0, 1.3])), mirror_room(), EnvConfig()
    )
    center = 1
    bounce = trace.bounce[center]
    assert bounce.sum() == 1
    np.testing.assert_allclose(trace.r_path[center, bounce], [4.0], rtol=1e-12)
    rho_m, rho2 = 0.95, 0.90
    expected = bounce_power(0.030, 1.0, rho_m, rho2, 1.0, 4.0,
                            np.pi * 0.01**2, 0.8, 1.231714e-4)
    np.testing.assert_allclose(trace.power[center, bounce], [expected], rtol=5e-7)
