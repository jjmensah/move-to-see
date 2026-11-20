# ==========================================================
# tests/integration/test_estimation_scenes.py
# ==========================================================
# Pose estimation on the canonical scenes.

import numpy as np

from lidar_sim.core.config import EnvConfig, SensorConfig
from lidar_sim.estimation.fisher import crb, fisher_information
from lidar_sim.estimation.jacobian2d import Planar2DParam
from lidar_sim.estimation.pose_mle import pose_mle
from lidar_sim.models.beam_model import BeamModel
from lidar_sim.sensor.lidar import LidarSensor
from lidar_sim.world.scenes import corridor, room_asym


# ------------------------------------------
# test_corridor_mle_reports_rank_deficiency
# ------------------------------------------
# In a corridor the pose estimate reports the unobservable direction.
def test_corridor_mle_reports_rank_deficiency():
    cfg = SensorConfig(n_beams=48)
    sensor, model, env = LidarSensor(cfg), BeamModel(sensor=cfg), EnvConfig()
    pose = Planar2DParam().to_pose(np.array([0., 0., 0.]))
    scan = sensor.scan(pose, corridor(), env, np.random.default_rng(4))
    result = pose_mle(model, sensor, scan, corridor(), env, np.array([.01, .01, .001]), max_iter=30)
    assert not result.success
    assert result.message.startswith("rank-deficient")


# ------------------------------------------
# test_room_pose_mle_recovers_seeded_scan
# ------------------------------------------
# In the asymmetric room the pose estimate recovers the true pose.
def test_room_pose_mle_recovers_seeded_scan():
    cfg = SensorConfig(n_beams=72)
    sensor, model, env = LidarSensor(cfg), BeamModel(sensor=cfg), EnvConfig()
    param = Planar2DParam()
    truth = np.array([1.5, 1.5, .3])
    scan = sensor.scan(param.to_pose(truth), room_asym(), env, np.random.default_rng(42))
    start = truth + np.array([.03, -.02, np.deg2rad(.5)])
    result = pose_mle(model, sensor, scan, room_asym(), env, start, param, max_iter=100)
    assert result.success
    assert model.log_likelihood(scan, model.predict(sensor.trace(param.to_pose(result.xi), room_asym(), env), env)) >= \
           model.log_likelihood(scan, model.predict(sensor.trace(param.to_pose(truth), room_asym(), env), env))
    info, rank = fisher_information(model, sensor, truth, room_asym(), env, param)
    bound = crb(info)
    error = result.xi - truth
    error[2] = (error[2] + np.pi) % (2 * np.pi) - np.pi
    assert rank == 3 and bound is not None
    assert np.all(np.abs(error) <= 3 * np.sqrt(np.diag(bound)))
