# ==========================================================
# tests/mts_b_oracles/test_predict_oracles.py
# ==========================================================
# T-P1..T-P3: fast predictor against the reference path (final-spec §5.2, F9, F14).
#
# Reference path: ``LidarSensor.trace`` on the truth scene, then ``BeamModel.predict`` / ``log_likelihood_beams``.

import numpy as np
import pytest
from scipy.stats import norm

from lidar_sim.core.config import EnvConfig
from lidar_sim.world.materials import BLACK, WHITE
from mts.config import CONDITIONS, SENSOR, RobotConfig, make_model, make_sensor
from mts.predict import Observation, Predictor
from mts.robot import sensor_pose, step
from mts.world import yard

FIELDS = ("hit", "r_path", "power", "normal", "cos_inc", "object_id", "material_id", "surface_id", "bounce", "w",
          "dir_world", "fire_time")
VIEWS = [np.array([6.1, 4.0, 0.0, 0.0]), np.array([7.8, 6.3, -1.5, 0.0]), np.array([9.8, 3.4, 3.0, 0.0]),
         np.array([1.0, 4.0, 0.0, 0.0])]


# ------------------------------------------
# _close
# ------------------------------------------
# Exact for integers and booleans, to round-off for floats.
def _close(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype == bool or np.issubdtype(a.dtype, np.integer):
        return np.array_equal(a, b)
    return np.allclose(a, b, rtol=1e-12, atol=1e-15, equal_nan=True)


# ------------------------------------------
# _targets
# ------------------------------------------
# Random targets in the prior region, corners included.
def _targets(world, rng, n):
    q = rng.uniform(world.prior_lo, world.prior_hi, size=(n, 3))
    q[0] = world.prior_lo                                       # corners of R_T included
    q[1] = world.prior_hi
    return q


# ------------------------------------------
# test_predictor_trace_equals_truth_trace
# ------------------------------------------
# The fast predictor's trace equals the full trace of the truth scene.
@pytest.mark.parametrize("cond", ["C1", "C2", "C3", "C4"])
@pytest.mark.parametrize("view", range(len(VIEWS)))
def test_predictor_trace_equals_truth_trace(cond, view, rng):
    c = CONDITIONS[cond]
    world = yard(c.target_material)
    pose = sensor_pose(VIEWS[view], RobotConfig())
    sensor, pred = make_sensor(9), Predictor(world, c.env)
    geom = pred.geometry(pose, t0=0.3, K=9)
    background = sensor.trace(pose, world.background_scene(), c.env, t0=0.3)
    for q in _targets(world, rng, 5):
        truth = sensor.trace(pose, world.truth_scene(q), c.env, t0=0.3)
        fast = pred.trace(geom, q)
        for f in FIELDS:
            assert _close(getattr(fast, f), getattr(truth, f)), (f, q)
        assert fast.P_amb == truth.P_amb
        # beams outside the candidate set do not depend on the target (ids aside: the ground id shifts by 6)
        other = np.setdiff1d(np.arange(truth.B), geom.beams)
        for f in ("hit", "r_path", "power", "cos_inc", "object_id", "material_id"):
            assert _close(getattr(truth, f)[other], getattr(background, f)[other]), f


# ------------------------------------------
# test_predictor_trace_with_moving_pose
# ------------------------------------------
# Scan skew (E2): the predictor must use the same per-beam pose trajectory as the truth trace.
def test_predictor_trace_with_moving_pose(rng):
    world, cfg = yard(), RobotConfig()
    traj = step(world, cfg, np.array([5.9, 6.4, -0.6, 0.8]), np.array([0.3, -0.7]), t0=2.0)
    assert traj.feasible
    t0 = traj.t1 - 0.1
    sensor, pred = make_sensor(9), Predictor(world, EnvConfig())
    geom = pred.geometry(traj.pose, t0=t0, K=9)
    q = np.array([7.3, 4.9, 0.9])
    truth = sensor.trace(traj.pose, world.truth_scene(q), EnvConfig(), t0=t0)
    fast = pred.trace(geom, q)
    assert (truth.object_id == 99).any()                        # the target is actually seen
    for f in FIELDS:
        assert _close(getattr(fast, f), getattr(truth, f)), f


# ------------------------------------------
# test_candidate_beams_cover_every_target_dependent_beam
# ------------------------------------------
# Candidate beams include every beam that the target can change.
def test_candidate_beams_cover_every_target_dependent_beam(rng):
    world = yard()
    sensor, pred = make_sensor(9), Predictor(world, EnvConfig())
    for x in VIEWS:
        pose = sensor_pose(x, RobotConfig())
        geom = pred.geometry(pose, K=9)
        background = sensor.trace(pose, world.background_scene(), EnvConfig())
        changed = np.zeros(background.B, bool)
        for q in _targets(world, rng, 12):
            truth = sensor.trace(pose, world.truth_scene(q), EnvConfig())
            changed |= (truth.object_id == 99).any(axis=1)
            changed |= ~np.isclose(truth.power, background.power, rtol=1e-12, atol=0).all(axis=1)
        assert set(np.flatnonzero(changed)) <= set(geom.beams.tolist())
        assert len(geom.beams) < background.B                  # the cull removes something


# ------------------------------------------
# test_loglik_equals_reference_e20
# ------------------------------------------
# T-P2: F9 = Σ over candidate beams of the E20 per-beam log-likelihood on the reference path.
@pytest.mark.parametrize("cond", ["C1", "C2", "C3"])
def test_loglik_equals_reference_e20(cond, rng):
    c = CONDITIONS[cond]
    world = yard(c.target_material)
    pose = sensor_pose(VIEWS[0], RobotConfig())
    sensor, model, pred = make_sensor(9), make_model(9), Predictor(world, c.env)
    q_true = np.array([7.4, 4.1, 0.8])
    scan = sensor.measure(sensor.trace(pose, world.truth_scene(q_true), c.env), c.env, rng)
    obs = Observation.from_scan(scan)
    geom = pred.geometry(pose, K=9)
    qs = np.vstack([q_true, _targets(world, rng, 6)])
    got = pred.loglik(geom, obs, qs)
    for p, q in enumerate(qs):
        ref_pred = model.predict(sensor.trace(pose, world.truth_scene(q), c.env), c.env)
        ref = model.log_likelihood_beams(scan, ref_pred)[geom.beams].sum()
        assert got[p] == pytest.approx(ref, rel=1e-10, abs=1e-9)
    assert np.argmax(got) == 0                                  # the truth explains its own scan best here


# ------------------------------------------
# test_observation_carries_no_truth
# ------------------------------------------
# An observation carries nothing about the true target.
def test_observation_carries_no_truth():
    world = yard()
    sensor = make_sensor(9)
    pose = sensor_pose(VIEWS[0], RobotConfig())
    scan = sensor.measure(sensor.trace(pose, world.truth_scene([7.4, 4.1, 0.8]), EnvConfig()), EnvConfig(),
                          np.random.default_rng(0))
    obs = Observation.from_scan(scan)
    assert set(vars(obs)) == {"detected", "range", "fire_time"}
    assert np.array_equal(obs.detected, scan.detected) and np.array_equal(obs.range, scan.range, equal_nan=True)


# ------------------------------------------
# test_aware_design_model
# ------------------------------------------
# T-P3 (F14): π = P_D·Z with the window mass Z of N(r̄, σ_R²); no-return beams are uniform with P_FA|W|/L_c.
@pytest.mark.parametrize("cond", ["C1", "C2", "C3", "C4"])
def test_aware_design_model(cond, rng):
    c = CONDITIONS[cond]
    world = yard(c.target_material)
    pose = sensor_pose(VIEWS[0], RobotConfig())
    sensor, model, pred = make_sensor(1), make_model(1), Predictor(world, c.env)
    geom = pred.geometry(pose, K=1)
    qs = _targets(world, rng, 4)
    des = pred.design(geom, qs, "aware")
    W = SENSOR.r_max - SENSOR.r_min
    for p, q in enumerate(qs):
        ref = model.predict(sensor.trace(pose, world.truth_scene(q), c.env), c.env)
        b = geom.beams
        hit = ref.hit_any[b]
        Z = norm.cdf((SENSOR.r_max - ref.r_bar[b]) / ref.sigma_R[b]) - norm.cdf((SENSOR.r_min - ref.r_bar[b]) / ref.sigma_R[b])
        assert np.array_equal(des.uniform[p], ~hit)
        assert np.allclose(des.pi[p][hit], (ref.P_D[b] * Z)[hit], rtol=1e-9, atol=1e-14)
        assert np.allclose(des.pi[p][~hit], (ref.P_FA[b] * W / SENSOR.Lc)[~hit], rtol=1e-12)
        assert np.allclose(des.mu[p][hit], ref.r_bar[b][hit], rtol=1e-12)
        assert np.allclose(des.sigma[p][hit], ref.sigma_R[b][hit], rtol=1e-12)
        assert np.allclose(des.Z[p][hit], Z[hit], rtol=1e-9, atol=1e-14)
    assert ((des.pi >= 0) & (des.pi <= 1)).all()


# ------------------------------------------
# test_blind_design_ignores_physics
# ------------------------------------------
# T-P3: the physics-blind model depends on geometry only (not on reflectivity, fog or sunlight).
def test_blind_design_ignores_physics(rng):
    pose = sensor_pose(VIEWS[0], RobotConfig())
    qs = None
    out = []
    for c in CONDITIONS.values():
        world = yard(c.target_material)
        qs = _targets(world, np.random.default_rng(3), 4) if qs is None else qs
        pred = Predictor(world, c.env)
        out.append(pred.design(pred.geometry(pose, K=1), qs, "blind"))
    for d in out[1:]:
        for f in ("pi", "mu", "sigma", "Z", "uniform", "beams"):
            assert np.array_equal(getattr(d, f), getattr(out[0], f)), f
    # π = 1 exactly where the ideal ray hits a surface inside the window, with μ the geometric range
    world = yard(BLACK)
    sensor = make_sensor(1)
    d0 = out[0]
    for p, q in enumerate(qs):
        tr = sensor.trace(pose, world.truth_scene(q), EnvConfig())
        r = tr.r_path[d0.beams, 0]
        inside = tr.hit[d0.beams, 0] & (r >= SENSOR.r_min) & (r <= SENSOR.r_max)
        assert np.array_equal(d0.pi[p] == 1.0, inside) and set(np.unique(d0.pi[p])) <= {0.0, 1.0}
        assert np.allclose(d0.mu[p][inside], r[inside], rtol=1e-12)
    assert np.all(d0.sigma == 0.012) and np.all(d0.Z == 1.0) and not d0.uniform.any()
    # and the aware model does change with the target reflectivity
    pw, pb = Predictor(yard(WHITE), EnvConfig()), Predictor(yard(BLACK), EnvConfig())
    aw = pw.design(pw.geometry(pose, K=1), qs, "aware").pi
    ab = pb.design(pb.geometry(pose, K=1), qs, "aware").pi
    assert not np.allclose(aw, ab)
