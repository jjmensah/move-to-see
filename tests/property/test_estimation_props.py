# ==========================================================
# tests/property/test_estimation_props.py
# ==========================================================
# Property tests of pose estimation.

import numpy as np

from lidar_sim.core.config import EnvConfig, SensorConfig
from lidar_sim.estimation.fisher import fisher_information
from lidar_sim.estimation.jacobian2d import Planar2DParam
from lidar_sim.models.beam_model import BeamModel
from lidar_sim.sensor.lidar import LidarSensor
from lidar_sim.world.scenes import room_asym


# ------------------------------------------
# test_fisher_remains_symmetric_psd_across_room_poses
# ------------------------------------------
# Fisher information stays symmetric and positive semidefinite across poses.
def test_fisher_remains_symmetric_psd_across_room_poses():
    cfg = SensorConfig(n_beams=48)
    sensor, model, env = LidarSensor(cfg), BeamModel(sensor=cfg), EnvConfig()
    for xi in (np.array([1.2, 1.2, .1]), np.array([2., 1.5, .6]), np.array([3., 1.2, -1.])):
        info, _ = fisher_information(model, sensor, xi, room_asym(), env, Planar2DParam(), "range_only")
        np.testing.assert_allclose(info, info.T, atol=1e-10)
        assert np.linalg.eigvalsh(info).min() >= -1e-9 * np.linalg.eigvalsh(info).max()
