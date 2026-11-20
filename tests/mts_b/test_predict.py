# ==========================================================
# tests/mts_b/test_predict.py
# ==========================================================
# Brief predictor checks by the implementation track; the independent oracles are in tests/mts_a_oracles.

import numpy as np

from lidar_sim.core.config import EnvConfig
from lidar_sim.core.pose import Pose
from mts.config import make_model, make_sensor
from mts.predict import Observation, Predictor
from mts.world import yard


# ------------------------------------------
# test_trace_and_likelihood
# ------------------------------------------
# Fast prediction matches the full trace and likelihood.
def test_trace_and_likelihood():
    world, env = yard(), EnvConfig()
    predictor = Predictor(world, env, chunk=37)
    pose = Pose.from_xyz_yaw(6., 2., .4, .7)
    q = np.array([7.8, 4., .8])
    geom = predictor.geometry(pose)
    actual = predictor.trace(geom, q)
    expected = make_sensor().trace(pose, world.truth_scene(q), env)
    for name, value in expected.fields().items():
        if name != 'pose':
            np.testing.assert_allclose(getattr(actual, name), value, rtol=1e-12, atol=1e-14)
    obs = Observation(np.zeros(expected.B, bool), np.full(expected.B, np.nan), expected.fire_time)
    model = make_model()
    reference = model.log_likelihood_arrays(obs.detected, obs.range, model.predict(expected, env))
    np.testing.assert_allclose(predictor.loglik(geom, obs, q[None]), reference[geom.beams].sum())
    batch = predictor.predict(geom, np.stack([q, q]))
    assert batch.P_D.shape == (2, len(geom.beams))
    np.testing.assert_array_equal(batch.P_D[0], batch.P_D[1])


# ------------------------------------------
# test_design_and_empty_candidates
# ------------------------------------------
# Design channels for both planners, and scans with no candidate beams.
def test_design_and_empty_candidates():
    p = Predictor(yard(), EnvConfig())
    q = np.array([[7.8, 4., .8]])
    geom = p.geometry(Pose.from_xyz_yaw(6., 2., .4, .7), K=1)
    for kind in ('aware', 'blind'):
        design = p.design(geom, q, kind)
        assert design.pi.shape == (1, len(geom.beams))
        assert np.all((design.pi >= 0) & (design.pi <= 1))
    empty = p.geometry(Pose.from_xyz_yaw(-20., 4., .4, np.pi))
    assert len(empty.beams) == 0
    assert p.predict(empty, q).P_D.shape == (1, 0)
