# ==========================================================
# tests/mts_b/test_planner.py
# ==========================================================
# Brief planner integration checks using the actual design channel and follower.

import numpy as np
import pytest

from mts.belief import ParticleBelief
from mts.config import CONDITIONS, BeliefConfig, PlannerConfig, RobotConfig
from mts.planner import ViewpointPlanner
from mts.predict import Predictor
from mts.robot import step
from mts.world import yard


# ------------------------------------------
# test_moves_feasibly_and_repeats
# ------------------------------------------
# Both planners move feasibly and repeat themselves with the same seed.
@pytest.mark.parametrize("kind,name", [("aware", "info"), ("blind", "blind")])
def test_moves_feasibly_and_repeats(kind, name):
    world, cfg = yard(), RobotConfig()
    belief = ParticleBelief(world, BeliefConfig(n_particles=128), np.random.default_rng(17))
    belief.particles[:] = np.array([7.8, 4., .8]) + .2 * (belief.particles - belief.mean())
    planner = ViewpointPlanner(world, cfg, PlannerConfig(), Predictor(world, CONDITIONS["C1"].env), kind)
    assert planner.name == name
    action = planner.act(0, world.start, belief, np.random.default_rng(4))
    again = planner.act(0, world.start, belief, np.random.default_rng(4))
    np.testing.assert_array_equal(action.u, again.u)
    np.testing.assert_array_equal(action.goal, again.goal)
    assert not action.info["switched"] and not action.info["goal_kept"]
    assert not again.info["switched"] and again.info["goal_kept"]
    for key in ("score", "info_gain", "travel_time", "best_stay"):
        assert action.info[key] == again.info[key]
    assert step(world, cfg, world.start, action.u).feasible
    assert np.linalg.norm(action.goal - planner.grid.centre(planner.grid.cell_of(world.start[:2]))) > .1
    assert action.info["score"] > action.info["best_stay"]
    assert action.info["score"] == pytest.approx(action.info["info_gain"] - .05 * action.info["travel_time"])
    assert action.info["n_candidates"] >= action.info["n_evaluated"] > 1


# ------------------------------------------
# test_shared_support_and_pruning_match_exhaustive
# ------------------------------------------
# Pruning picks the same goal as scoring every viewpoint.
def test_shared_support_and_pruning_match_exhaustive(monkeypatch):
    world, cfg, pc = yard(), RobotConfig(), PlannerConfig(M=6, lattice=1.)
    belief = ParticleBelief(world, BeliefConfig(n_particles=128), np.random.default_rng(12))
    planner = ViewpointPlanner(world, cfg, pc, Predictor(world, CONDITIONS["C1"].env), "blind")
    support = belief.support(pc.M, np.random.default_rng(9))
    calls = []

    def sample(count, rng):
        calls.append(count)
        return support

    monkeypatch.setattr(belief, "support", sample)
    action = planner.act(0, world.start, belief, np.random.default_rng(9))
    assert calls == [pc.M]
    times, _ = planner.grid.travel_time(world.start[:2])
    stay = np.ravel_multi_index(planner.grid.cell_of(world.start[:2]), times.shape)
    candidates = np.union1d(planner._lattice, [stay])
    scored = []
    for index in candidates:
        t = times.ravel()[index]
        if np.isfinite(t):
            score = planner._gain(int(index), belief.mean(), support) - pc.lambda_t * t
            scored.append((score, -t, -int(index)))
    score, _, negative_index = max(scored)
    np.testing.assert_array_equal(action.goal, planner.grid.centre(np.unravel_index(-negative_index, times.shape)))
    assert action.info["score"] == score
    # Every view ties at zero information for one support point: stay wins on time.
    singleton = ViewpointPlanner(world, cfg, PlannerConfig(M=1, lambda_t=0.), planner.predictor)
    monkeypatch.setattr(belief, "support", lambda count, rng: support[:count])
    action = singleton.act(0, world.start, belief, np.random.default_rng(9))
    np.testing.assert_array_equal(action.goal, singleton.grid.centre(singleton.grid.cell_of(world.start[:2])))
    assert action.info["score"] == 0.


# ------------------------------------------
# test_goal_persistence_current_scores
# ------------------------------------------
# The current goal is kept unless another beats it by the margin.
@pytest.mark.parametrize("advantage,kept", [(0.049, True), (0.05, False), (0.051, False)])
def test_goal_persistence_current_scores(monkeypatch, advantage, kept):
    world, cfg = yard(), RobotConfig()
    pc = PlannerConfig(M=2, lambda_t=0., switch_margin=.05)
    belief = ParticleBelief(world, BeliefConfig(n_particles=32), np.random.default_rng(2))
    planner = ViewpointPlanner(world, cfg, pc, Predictor(world, CONDITIONS['C1'].env))
    times, _ = planner.grid.travel_time(world.start[:2])
    candidates = planner._lattice[np.isfinite(times.ravel()[planner._lattice])]
    previous, challenger = map(int, candidates[:2])
    planner._lattice = np.array([previous, challenger])
    gains = {previous: 0.5, challenger: 0.}
    monkeypatch.setattr(planner, '_gains',
                        lambda indices, mean, support: {int(i): gains.get(i, 0.) for i in indices})
    first = planner.act(0, world.start, belief, np.random.default_rng(1))
    assert not first.info['switched'] and not first.info['goal_kept']
    # Change the old goal's gain: retaining its stale score would select incorrectly.
    gains.update({previous: 0., challenger: advantage})
    second = planner.act(1, world.start, belief, np.random.default_rng(1))
    assert second.info['goal_kept'] == kept
    assert second.info['switched'] != kept
    expected = previous if kept else challenger
    np.testing.assert_array_equal(second.goal, planner.grid.centre(np.unravel_index(expected, times.shape)))
    assert second.info['info_gain'] == gains[expected]


# ------------------------------------------
# test_previous_goal_evaluated_even_beyond_pruning
# ------------------------------------------
# The previous goal is always rescored, even when pruning would skip it.
def test_previous_goal_evaluated_even_beyond_pruning(monkeypatch):
    world, cfg = yard(), RobotConfig()
    belief = ParticleBelief(world, BeliefConfig(n_particles=32), np.random.default_rng(2))
    planner = ViewpointPlanner(world, cfg, PlannerConfig(M=1), Predictor(world, CONDITIONS['C1'].env))
    times, _ = planner.grid.travel_time(world.start[:2])
    valid = planner._lattice[np.isfinite(times.ravel()[planner._lattice])]
    planner._goal = int(valid[np.argmax(times.ravel()[valid])])
    calls = []

    def gain(index, mean, support):
        calls.append(index)
        return 0.

    monkeypatch.setattr(planner, '_gain', gain)
    action = planner.act(0, world.start, belief, np.random.default_rng(1))
    assert len(calls) == 2
    assert action.info['switched'] and not action.info['goal_kept']
