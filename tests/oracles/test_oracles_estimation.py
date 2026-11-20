# ==========================================================
# tests/oracles/test_oracles_estimation.py
# ==========================================================
# Independent oracle tests for estimation, written from the spec alone (E23, O-17).

import math

import numpy as np
import pytest

from lidar_sim.core.config import SensorConfig
from lidar_sim.core.pose import Pose
from lidar_sim.estimation.jacobian2d import Planar2DParam
from lidar_sim.geometry.scene2d import Scene2D
from lidar_sim.models.beam_model import BeamModel
from lidar_sim.sensor.lidar import LidarSensor
from lidar_sim.world.environment import CLEAR
from lidar_sim.world.scenes import single_wall


# ------------------------------------------
# test_O17_wall_range_jacobian
# ------------------------------------------
# O-17 / E23: wall x = 2, sensor (0, 0, ψ), beam world angle β = ψ + α:
# r = 2/cos β, ∂r/∂x = −1/cos β, ∂r/∂y = 0, ∂r/∂ψ = 2 sin β / cos² β. K = 1 (ideal ray).
@pytest.mark.parametrize("psi", [0.0, 0.1, -0.3])
def test_O17_wall_range_jacobian(psi):
    cfg = SensorConfig(n_subrays=1)
    sensor, param = LidarSensor(cfg), Planar2DParam()
    pose = param.to_pose(np.array([0.0, 0.0, psi]))
    trace = sensor.trace(pose, Scene2D.from_spec(single_wall()), CLEAR)
    pred = BeamModel(cfg).predict(trace, CLEAR)
    J = param.range_jacobian(trace, pred)
    beta = np.arctan2(trace.dir_world[:, 0, 1], trace.dir_world[:, 0, 0])
    on_wall = pred.smooth & (np.abs(2.0 * np.tan(beta)) < 0.95)      # stay clear of the wall ends
    assert on_wall.sum() > 50
    cb, sb = np.cos(beta[on_wall]), np.sin(beta[on_wall])
    np.testing.assert_allclose(trace.r_path[on_wall, 0], 2.0 / cb, rtol=1e-12)
    np.testing.assert_allclose(J[on_wall, 0], -1.0 / cb, rtol=1e-9)
    np.testing.assert_allclose(J[on_wall, 1], 0.0, atol=1e-12)
    np.testing.assert_allclose(J[on_wall, 2], 2.0 * sb / cb**2, rtol=1e-9, atol=1e-12)
    assert np.isnan(J[~pred.smooth]).all()


# ------------------------------------------
# test_O17_param_round_trip
# ------------------------------------------
# Planar2DParam: to_pose/from_pose are inverse; dof = 3; yaw is the only rotational dof.
def test_O17_param_round_trip():
    p = Planar2DParam()
    xi = np.array([1.2, -0.4, 2.0])
    assert isinstance(p.to_pose(xi), Pose)
    np.testing.assert_allclose(p.from_pose(p.to_pose(xi)), xi, atol=1e-15)
    assert p.dof == 3 and tuple(p.is_rotation) == (False, False, True)
    assert math.isclose(p.from_pose(p.to_pose([0, 0, 3 * math.pi]))[2], -math.pi)
