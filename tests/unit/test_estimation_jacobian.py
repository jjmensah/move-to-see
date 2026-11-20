# ==========================================================
# tests/unit/test_estimation_jacobian.py
# ==========================================================
# Unit tests of the range Jacobian.

import numpy as np
import pytest

from lidar_sim.core.config import EnvConfig, SensorConfig
from lidar_sim.estimation.jacobian2d import Planar2DParam
from lidar_sim.models.beam_model import expected_range
from lidar_sim.sensor.lidar import LidarSensor
from lidar_sim.world.scenes import room_asym


# ------------------------------------------
# test_range_jacobian_matches_expected_range_finite_difference
# ------------------------------------------
# The range Jacobian matches finite differences of the expected range.
@pytest.mark.parametrize("xi", ([1.5, 1.5, .3], [2.0, 1.0, -.2], [1.0, 2.0, .8]))
def test_range_jacobian_matches_expected_range_finite_difference(xi):
    cfg = SensorConfig(n_beams=72, n_subrays=1)
    sensor, param, env = LidarSensor(cfg), Planar2DParam(), EnvConfig()
    pose = param.to_pose(np.array(xi))
    trace = sensor.trace(pose, room_asym(), env)
    from lidar_sim.models.beam_model import BeamModel
    model = BeamModel(sensor=cfg)
    pred = model.predict(trace, env)
    jac = param.range_jacobian(trace, pred)
    smooth = np.flatnonzero(pred.smooth)
    assert smooth.size > 0
    for k, h in enumerate((1e-6, 1e-6, 1e-7)):
        dx = np.zeros(3)
        dx[k] = h
        rp = expected_range(sensor.trace(param.to_pose(np.array(xi) + dx), room_asym(), env), cfg, model.noise)
        rm = expected_range(sensor.trace(param.to_pose(np.array(xi) - dx), room_asym(), env), cfg, model.noise)
        fd = (rp - rm) / (2 * h)
        valid = smooth[np.isfinite(fd[smooth])]
        np.testing.assert_allclose(jac[valid, k], fd[valid], rtol=1e-5, atol=1e-6)
