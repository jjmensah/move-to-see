# ==========================================================
# tests/unit/test_estimation_fisher.py
# ==========================================================
# Unit tests of the Fisher information.

import numpy as np

from lidar_sim.core.config import EnvConfig, SensorConfig
from lidar_sim.estimation.fisher import crb, fisher_information
from lidar_sim.estimation.jacobian2d import Planar2DParam
from lidar_sim.models.beam_model import BeamModel
from lidar_sim.sensor.lidar import LidarSensor
from lidar_sim.world.scenes import corridor, room_asym


# ------------------------------------------
# setup
# ------------------------------------------
# Sensor, model, environment and pose parametrization for these tests.
def setup():
    cfg = SensorConfig(n_beams=96)
    sensor, model, env = LidarSensor(cfg), BeamModel(sensor=cfg), EnvConfig()
    return model, sensor, env, Planar2DParam()


# ------------------------------------------
# test_fisher_is_symmetric_psd_and_full_dominates_range_only
# ------------------------------------------
# Fisher information is symmetric, positive semidefinite and at least the range-only one.
def test_fisher_is_symmetric_psd_and_full_dominates_range_only():
    model, sensor, env, param = setup()
    xi = np.array([1.5, 1.5, .3])
    a, ra = fisher_information(model, sensor, xi, room_asym(), env, param, "range_only")
    b, rb = fisher_information(model, sensor, xi, room_asym(), env, param, "full")
    for mat in (a, b):
        np.testing.assert_allclose(mat, mat.T, atol=1e-10)
        assert np.linalg.eigvalsh(mat).min() >= -1e-9 * np.linalg.eigvalsh(mat).max()
    assert ra == rb == 3
    assert np.linalg.eigvalsh(b - a).min() >= -1e-9 * np.linalg.eigvalsh(b).max()


# ------------------------------------------
# test_room_rank_and_corridor_null_translation
# ------------------------------------------
# Full rank in the room; no information along a corridor.
def test_room_rank_and_corridor_null_translation():
    model, sensor, env, param = setup()
    room, rr = fisher_information(model, sensor, np.array([1.5, 1.5, .3]), room_asym(), env, param)
    hall, rh = fisher_information(model, sensor, np.array([0., 0., 0.]), corridor(), env, param)
    assert rr == 3 and rh == 2
    assert crb(room) is not None and crb(hall) is None
    v = np.linalg.eigh(hall)[1][:, 0]
    assert abs(np.dot(v, [1., 0., 0.])) > .999
