# ==========================================================
# tests/mts_b/test_belief.py
# ==========================================================
# Brief regularized-filter sanity checks (F10–F13).

import numpy as np

from mts.belief import ParticleBelief
from mts.config import BeliefConfig
from mts.world import yard


# ------------------------------------------
# test_gaussian_concentration_and_determinism
# ------------------------------------------
# A Gaussian likelihood concentrates the belief, the same way every time.
def test_gaussian_concentration_and_determinism():
    results = []
    for _ in range(2):
        rng = np.random.default_rng(42)
        belief = ParticleBelief(yard(), BeliefConfig(n_particles=1000), rng)
        info = belief.update(lambda q: -.5*((q[:, 0]-7.6)/.08)**2, rng)
        assert not info.skipped
        assert info.stages >= 1
        np.testing.assert_allclose(belief.weights.sum(), 1)
        assert abs(belief.mean()[0]-7.6) < .025
        assert np.sqrt(belief.cov()[0, 0]) < .11
        assert np.all(belief.particles >= belief.world.prior_lo)
        assert np.all(belief.particles <= belief.world.prior_hi)
        assert belief.support(24, rng).shape == (24, 3)
        results.append(belief.particles)
    np.testing.assert_array_equal(*results)


# ------------------------------------------
# test_invalid_skipped
# ------------------------------------------
# An update with no finite likelihood is skipped.
def test_invalid_skipped():
    rng = np.random.default_rng(1)
    belief = ParticleBelief(yard(), BeliefConfig(n_particles=20), rng)
    before = belief.particles.copy()
    assert belief.update(lambda q: np.full(len(q), -np.inf), rng).skipped
    np.testing.assert_array_equal(before, belief.particles)
