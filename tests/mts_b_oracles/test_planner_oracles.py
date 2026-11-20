# ==========================================================
# tests/mts_b_oracles/test_planner_oracles.py
# ==========================================================
# T-L1: viewpoint planner (final-spec §7.3–7.4) against a brute-force evaluation of F17–F18 through public APIs.

import math

import numpy as np
import pytest

from mts.belief import ParticleBelief
from mts.config import CONDITIONS, SENSOR, BeliefConfig, PlannerConfig, RobotConfig
from mts.info import mi_lower_bound, pair_distance
from mts.planner import ViewpointPlanner
from mts.predict import Predictor
from mts.robot import NavGrid, sensor_pose, step
from mts.world import yard

RCFG, PCFG = RobotConfig(), PlannerConfig()


# ------------------------------------------
# _Belief
# ------------------------------------------
# Minimal belief with a fixed support and mean (the planner may use only ``support`` and ``mean``).
class _Belief:
    def __init__(self, support):
        self._s = np.asarray(support, dtype=float)
        self.calls = 0

    def support(self, M, rng):
        self.calls += 1
        assert M == len(self._s)
        return self._s.copy()

    def mean(self):
        return self._s.mean(axis=0)


# ------------------------------------------
# _brute_force
# ------------------------------------------
# All candidates of §7.4 with their information (F17), travel time and score (F18).
def _brute_force(world, env, kind, x, support):
    grid = NavGrid(world, RCFG, cell=PCFG.cell)
    times, _ = grid.travel_time(x[:2])
    pred = Predictor(world, env)
    kx0, kx1, ky0, ky1 = world.keepouts(RCFG)[-1]
    stride = round(PCFG.lattice / PCFG.cell)
    mean = support.mean(axis=0)
    here = grid.cell_of(x[:2])
    out = []
    for iy in range(grid.shape[0]):
        for ix in range(grid.shape[1]):
            on_lattice = iy % stride == 0 and ix % stride == 0
            if (iy, ix) != here and not (on_lattice and grid.free[iy, ix]):
                continue
            c = grid.centre((iy, ix))
            dist = math.hypot(max(kx0 - c[0], c[0] - kx1, 0.0), max(ky0 - c[1], c[1] - ky1, 0.0))
            if ((iy, ix) != here and dist > PCFG.radius) or not np.isfinite(times[iy, ix]):
                continue
            heading = math.atan2(mean[1] - c[1], mean[0] - c[0])
            geom = pred.geometry(sensor_pose(np.array([c[0], c[1], heading, 0.0]), RCFG), K=1)
            info = mi_lower_bound(pair_distance(pred.design(geom, support, kind), SENSOR.r_min, SENSOR.r_max))
            out.append({"cell": (iy, ix), "xy": c, "info": info, "t": float(times[iy, ix]),
                        "J": info - PCFG.lambda_t * float(times[iy, ix])})
    return out


# ------------------------------------------
# _support
# ------------------------------------------
# Support points spread around a centre inside the prior region.
def _support(world, rng, spread, centre=None):
    centre = 0.5 * (world.prior_lo + world.prior_hi) if centre is None else np.asarray(centre)
    half = 0.5 * (world.prior_hi - world.prior_lo) * spread
    return np.clip(centre + rng.uniform(-half, half, size=(PCFG.M, 3)), world.prior_lo, world.prior_hi)


# ------------------------------------------
# test_goal_maximises_f18_by_brute_force
# ------------------------------------------
# The chosen goal maximises F18 over every viewpoint.
@pytest.mark.parametrize("cond,kind,x,spread", [
    ("C1", "aware", [1.0, 4.0, 0.0, 0.0], 1.0),         # prior-sized support, from the start
    ("C2", "aware", [1.0, 4.0, 0.0, 0.0], 1.0),         # fog: nothing informative nearby
    ("C3", "aware", [5.9, 6.5, -0.5, 0.4], 0.3),        # sunlight, belief partly concentrated, robot moving
    ("C2", "blind", [1.0, 4.0, 0.0, 0.0], 1.0),
    ("C1", "aware", [9.85, 4.0, 3.1, 0.0], 0.08),       # late: tight belief
])
def test_goal_maximises_f18_by_brute_force(cond, kind, x, spread, rng):
    c = CONDITIONS[cond]
    world = yard(c.target_material)
    x = np.array(x)
    support = _support(world, rng, spread)
    cands = _brute_force(world, c.env, kind, x, support)
    best = max(cands, key=lambda d: (d["J"], -d["t"]))
    planner = ViewpointPlanner(world, RCFG, PCFG, Predictor(world, c.env), kind=kind)
    belief = _Belief(support)
    act = planner.act(0, x, belief, np.random.default_rng(0))
    assert belief.calls == 1                                         # one support draw per planning step
    assert act.info["score"] == pytest.approx(best["J"], abs=1e-9)
    assert act.info["info_gain"] == pytest.approx(best["info"], abs=1e-9)
    assert act.info["travel_time"] == pytest.approx(best["t"], abs=1e-9)
    ties = [d for d in cands if abs(d["J"] - best["J"]) < 1e-12 and abs(d["t"] - best["t"]) < 1e-12]
    assert any(np.allclose(act.goal, d["xy"]) for d in ties)
    assert act.info["n_candidates"] == len(cands)
    assert 0.0 <= act.info["info_gain"] <= math.log(PCFG.M) + 1e-12
    assert step(world, RCFG, x, act.u).feasible and not act.fallback


# ------------------------------------------
# test_planner_leaves_an_uninformative_start
# ------------------------------------------
# No 'stop forever': from the start in fog or sunlight nothing can be learned locally (Î(stay) ≈ 0), yet the
# planner must choose a distant viewpoint with positive information and start moving.
@pytest.mark.parametrize("cond", ["C2", "C3"])
def test_planner_leaves_an_uninformative_start(cond, rng):
    c = CONDITIONS[cond]
    world = yard(c.target_material)
    x = world.start.copy()
    support = _support(world, rng, 1.0)
    planner = ViewpointPlanner(world, RCFG, PCFG, Predictor(world, c.env), kind="aware")
    act = planner.act(0, x, _Belief(support), np.random.default_rng(0))
    assert act.info["best_stay"] < 0.02
    assert act.info["info_gain"] > 0.5 and act.info["travel_time"] > 3.0
    assert np.hypot(*(act.goal - x[:2])) > 3.0
    moved = step(world, RCFG, x, act.u).end
    assert moved[3] > 0.05 or abs(moved[2] - x[2]) > 0.05                # it accelerates or turns towards the path


# ------------------------------------------
# test_aware_and_blind_disagree_where_physics_matters
# ------------------------------------------
# In sunlight the blind model believes the start already separates the hypotheses; the aware model does not.
def test_aware_and_blind_disagree_where_physics_matters(rng):
    c = CONDITIONS["C3"]
    world = yard(c.target_material)
    support = _support(world, rng, 1.0)
    x = np.array([5.9, 6.5, -0.5, 0.0])
    aware = ViewpointPlanner(world, RCFG, PCFG, Predictor(world, c.env), "aware").act(0, x, _Belief(support), rng)
    blind = ViewpointPlanner(world, RCFG, PCFG, Predictor(world, c.env), "blind").act(0, x, _Belief(support), rng)
    assert blind.info["best_stay"] > aware.info["best_stay"] + 0.3


# ------------------------------------------
# test_goal_persistence_rule
# ------------------------------------------
# §7.4: keep the previous goal unless the best candidate beats it by at least the switch margin.
def test_goal_persistence_rule(rng):
    c = CONDITIONS["C1"]
    world = yard(c.target_material)
    x = world.start.copy()
    support = _support(world, rng, 0.5)
    cands = _brute_force(world, c.env, "aware", x, support)
    ranked = sorted(cands, key=lambda d: (-d["J"], d["t"]))
    best = ranked[0]
    close = next(d for d in ranked[1:] if best["J"] - d["J"] < PCFG.switch_margin)      # within the margin
    far = next(d for d in ranked if best["J"] - d["J"] > PCFG.switch_margin)            # clearly worse
    for prev, expect_kept in ((close, True), (far, False)):
        planner = ViewpointPlanner(world, RCFG, PCFG, Predictor(world, c.env), "aware")
        planner._goal = int(np.ravel_multi_index(prev["cell"], planner.grid.free.shape))
        act = planner.act(1, x, _Belief(support), np.random.default_rng(0))
        assert act.info["goal_kept"] is expect_kept
        assert np.allclose(act.goal, prev["xy"] if expect_kept else best["xy"])
        assert act.info["switched"] is (not expect_kept)


# ------------------------------------------
# test_planner_is_deterministic_and_uses_only_support_and_mean
# ------------------------------------------
# The planner is deterministic and depends only on the support and the mean.
def test_planner_is_deterministic_and_uses_only_support_and_mean(rng):
    c = CONDITIONS["C1"]
    world = yard(c.target_material)
    b = ParticleBelief(world, BeliefConfig(n_particles=300), np.random.default_rng(3))
    acts = []
    for _ in range(2):
        planner = ViewpointPlanner(world, RCFG, PCFG, Predictor(world, c.env), "aware")
        acts.append(planner.act(0, world.start.copy(), b, np.random.default_rng(11)))
    assert np.array_equal(acts[0].u, acts[1].u) and np.array_equal(acts[0].goal, acts[1].goal)
    assert acts[0].info["score"] == acts[1].info["score"]
