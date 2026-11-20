# ==========================================================
# tests/unit/test_world_targets_env.py
# ==========================================================
# Unit tests: target trajectories (world/targets.py) and environment presets (world/environment.py).

import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from lidar_sim.core.config import EnvConfig, NoiseConfig
from lidar_sim.world import environment as env
from lidar_sim.world.targets import constant_velocity, waypoints


# ------------------------------------------
# test_constant_velocity
# ------------------------------------------
# Constant-velocity trajectory.
def test_constant_velocity():
    f = constant_velocity((1.0, 2.0), (0.5, -1.0), yaw0=0.3, yaw_rate=2.0)
    p = f(0.0)
    np.testing.assert_allclose(p.t, [1.0, 2.0])
    np.testing.assert_allclose(p.xi()[2], 0.3)
    p = f(0.25)
    np.testing.assert_allclose(p.t, [1.125, 1.75])
    np.testing.assert_allclose(p.xi()[2], 0.8, atol=1e-15)
    np.testing.assert_allclose(constant_velocity((0, 0), (1, 0))(2.0).R, np.eye(2))


# ------------------------------------------
# test_constant_velocity_validates_shapes
# ------------------------------------------
# Constant-velocity inputs of the wrong shape are rejected.
def test_constant_velocity_validates_shapes():
    with pytest.raises(ValueError):
        constant_velocity((0, 0, 0), (1, 0))


# ------------------------------------------
# test_waypoints_linear_and_clamped
# ------------------------------------------
# Waypoint trajectories interpolate linearly and clamp at the ends.
def test_waypoints_linear_and_clamped():
    f = waypoints([0.0, 1.0, 3.0], [[0, 0, 0], [1, 2, 0.5], [3, 2, 0.5]])
    np.testing.assert_allclose(f(0.5).xi(), [0.5, 1.0, 0.25], atol=1e-15)
    np.testing.assert_allclose(f(2.0).xi(), [2.0, 2.0, 0.5], atol=1e-15)
    np.testing.assert_allclose(f(1.0).xi(), [1.0, 2.0, 0.5], atol=1e-15)
    np.testing.assert_allclose(f(-5.0).xi(), [0, 0, 0], atol=1e-15)
    np.testing.assert_allclose(f(10.0).xi(), [3, 2, 0.5], atol=1e-15)


# ------------------------------------------
# test_waypoints_shortest_arc_yaw
# ------------------------------------------
# Yaw interpolates along the shorter arc.
def test_waypoints_shortest_arc_yaw():
    a, b = math.radians(170), math.radians(-170)
    f = waypoints([0.0, 1.0], [[0, 0, a], [0, 0, b]])
    assert f(0.5).xi()[2] == pytest.approx(-math.pi, abs=1e-12) or f(0.5).xi()[2] == pytest.approx(math.pi)
    assert f(0.25).xi()[2] == pytest.approx(math.radians(175), abs=1e-12)


# ------------------------------------------
# test_waypoints_single_and_invalid
# ------------------------------------------
# A single waypoint, and invalid waypoint times.
def test_waypoints_single_and_invalid():
    f = waypoints([1.0], [[1, 2, 0.1]])
    np.testing.assert_allclose(f(0.0).xi(), [1, 2, 0.1])
    with pytest.raises(ValueError):
        waypoints([0.0, 0.0], [[0, 0, 0], [1, 0, 0]])
    with pytest.raises(ValueError):
        waypoints([0.0, 1.0], [[0, 0, 0]])
    with pytest.raises(ValueError):
        waypoints([], np.zeros((0, 3)))


# ------------------------------------------
# test_environment_presets
# ------------------------------------------
# Clear and foggy environment presets.
def test_environment_presets():
    assert env.CLEAR == EnvConfig()
    assert env.FOG_LIGHT.visibility_m == 200.0 and env.FOG_DENSE.visibility_m == 20.0
    assert env.FOG_LIGHT.P_amb == 0.0
    with pytest.raises(FrozenInstanceError):
        env.CLEAR.P_amb = 1.0  # type: ignore[misc]


# ------------------------------------------
# test_sunlight
# ------------------------------------------
# Sunlight presets scale the calibrated ambient level.
def test_sunlight():
    n = NoiseConfig()
    assert env.sunlight().P_amb == n.P_amb_sun
    s = env.sunlight(0.25, base=env.FOG_DENSE)
    assert s.P_amb == 0.25 * n.P_amb_sun and s.visibility_m == 20.0
    assert env.sunlight(1.0, noise=NoiseConfig(P_amb_sun=1e-6)).P_amb == 1e-6
    assert env.sunlight(0.0).P_amb == 0.0
    with pytest.raises(ValueError):
        env.sunlight(-0.1)
