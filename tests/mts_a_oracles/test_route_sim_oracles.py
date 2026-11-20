# ==========================================================
# tests/mts_a_oracles/test_route_sim_oracles.py
# ==========================================================
# Spec v0.3 §8 fixed-loop and §9 paired random-stream plumbing oracles.

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mts.config import RobotConfig
from mts.control import Action
from mts.robot import step
from mts.route import FixedRoute
from mts.sim import draw_target, stream
from mts.world import yard

LOOP = np.array([[1., 4.], [2.6, 6.7], [5.8, 6.7], [9.9, 6.7],
                 [9.9, 1.3], [5.8, 0.75], [2.6, 1.1], [1., 4.]])


# ------------------------------------------
# ForbiddenBelief
# ------------------------------------------
# A belief that fails the test if anything reads it.
class ForbiddenBelief:
    def __getattribute__(self, name):
        raise AssertionError(f'FixedRoute read belief.{name}; §8 forbids measurement dependence')


# ------------------------------------------
# test_fixed_route_spec_waypoints_and_length
# ------------------------------------------
# The fixed route has the specified waypoints and length.
def test_fixed_route_spec_waypoints_and_length():
    route = FixedRoute(yard(), RobotConfig())
    assert_array_equal(route.loop, LOOP)
    length = np.linalg.norm(np.diff(LOOP, axis=0), axis=1).sum()
    assert route.length == pytest.approx(length, abs=1e-12)
    assert route.length == pytest.approx(26.5, abs=0.05)


# ------------------------------------------
# test_fixed_route_60_steps_progress_feasibility_and_no_belief_access
# ------------------------------------------
# The fixed route keeps moving, stays feasible and never reads the belief.
def test_fixed_route_60_steps_progress_feasibility_and_no_belief_access():
    world, cfg = yard(), RobotConfig()
    route = FixedRoute(world, cfg)
    x, progress = world.start.copy(), 0.
    vectors = np.diff(LOOP, axis=0)
    for k in range(60):
        action = route.act(k, x, ForbiddenBelief(), np.random.default_rng(94240+k))
        assert isinstance(action, Action)
        assert np.asarray(action.u).shape == (2,)
        assert route.s >= progress-1e-12, f'k={k}: progress regressed'
        progress = route.s
        tr = step(world, cfg, x, action.u)
        assert tr.feasible, f'k={k}, state={x}, control={action.u}'
        if k >= 3:
            # Point-to-each-segment projection, independently from the follower.
            delta = tr.states[:, None, :2] - LOOP[None, :-1, :]
            fraction = np.clip(np.sum(delta*vectors, axis=-1)/np.sum(vectors**2, axis=-1), 0, 1)
            distances = np.linalg.norm(delta-fraction[..., None]*vectors, axis=-1).min(axis=1)
            assert np.all(distances <= 0.5), f'k={k}, max loop distance={distances.max()}'
        x = tr.end
    assert progress > 1, 'route controller never made progress'


# ------------------------------------------
# test_sim_stream_reproducible
# ------------------------------------------
# Random streams are reproducible from their keys.
@pytest.mark.parametrize(
    ('purpose', 'k'), [('target', None), ('filter', None), ('planner', None), ('noise', 0), ('noise', 17)],
)
def test_sim_stream_reproducible(purpose, k):
    assert_array_equal(stream(94240, 3, purpose, k).random(32), stream(94240, 3, purpose, k).random(32))


# ------------------------------------------
# test_sim_stream_separates_seed_trial_purpose_and_scan
# ------------------------------------------
# Different seeds, trials, purposes and scans give different streams.
def test_sim_stream_separates_seed_trial_purpose_and_scan():
    keys = [(94240, 3, p, None) for p in ['target', 'filter', 'planner']]
    keys += [(94240, 3, 'noise', k) for k in [0, 1, 17]]
    keys += [(94240, 4, 'noise', 0), (94241, 3, 'noise', 0)]
    draws = [stream(*key).random(32) for key in keys]
    for i, draw in enumerate(draws):
        for other in draws[:i]:
            assert not np.array_equal(draw, other)


# ------------------------------------------
# test_sim_draw_target_prior_support_and_pairing
# ------------------------------------------
# Targets are drawn inside the prior region and are the same for every controller.
def test_sim_draw_target_prior_support_and_pairing():
    world = yard()
    # Use literal R_T from §4.3 rather than deriving the expected support from draw_target.
    lo, hi = np.array([7., 2.8, 0.5]), np.array([8.6, 5.2, 1.1])
    targets = []
    for trial in range(100):
        q = draw_target(world, 94240, trial)
        assert q.shape == (3,)
        assert np.all(q >= lo) and np.all(q <= hi)
        assert_array_equal(q, draw_target(world, 94240, trial))
        targets.append(q)
    assert len(np.unique(targets, axis=0)) == 100
    # Loose distribution check catches a constant or collapsed coordinate, without flaky quantiles.
    scaled = (np.array(targets)-lo)/(hi-lo)
    assert_allclose(scaled.mean(axis=0), 0.5, atol=0.12)
    assert np.all(np.ptp(scaled, axis=0) > 0.8)
