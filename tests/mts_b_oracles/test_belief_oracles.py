# ==========================================================
# tests/mts_b_oracles/test_belief_oracles.py
# ==========================================================
# T-B1, T-B2: regularized particle filter (final-spec §6) against invariants and fine-grid Bayes oracles.

import numpy as np
import pytest

from mts.belief import ParticleBelief
from mts.config import BeliefConfig
from mts.world import yard

WORLD = yard()
LO, HI = WORLD.prior_lo, WORLD.prior_hi


# ------------------------------------------
# _grid_posterior
# ------------------------------------------
# Posterior mean and std of x and y on a fine grid, uniform prior on R_T's (x, y) rectangle.
def _grid_posterior(logliks, n=600):
    xs = LO[0] + (np.arange(n) + 0.5) * (HI[0] - LO[0]) / n
    ys = LO[1] + (np.arange(n) + 0.5) * (HI[1] - LO[1]) / n
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    q = np.stack([X.ravel(), Y.ravel(), np.full(X.size, 0.8)], axis=1)
    ll = sum(f(q) for f in logliks)
    w = np.exp(ll - ll.max())
    w /= w.sum()
    m = np.array([w @ q[:, 0], w @ q[:, 1]])
    s = np.sqrt(np.array([w @ (q[:, 0] - m[0]) ** 2, w @ (q[:, 1] - m[1]) ** 2]))
    return m, s


# ------------------------------------------
# _run
# ------------------------------------------
# Runs a sequence of likelihood updates on a fresh belief.
def _run(logliks, seed=0, cfg=BeliefConfig()):  # noqa: B008
    rng = np.random.default_rng(seed)
    b = ParticleBelief(WORLD, cfg, rng)
    infos = [b.update(f, rng) for f in logliks]
    return b, infos


# ------------------------------------------
# test_prior_is_uniform_on_region
# ------------------------------------------
# The prior is uniform over the target region.
def test_prior_is_uniform_on_region():
    b = ParticleBelief(WORLD, BeliefConfig(), np.random.default_rng(0))
    assert b.particles.shape == (2000, 3) and np.allclose(b.weights, 1 / 2000)
    assert (b.particles >= LO).all() and (b.particles <= HI).all()
    assert np.allclose(b.mean(), 0.5 * (LO + HI), atol=0.01)                    # low-discrepancy draw
    assert np.allclose(np.sqrt(np.diag(b.cov())), (HI - LO) / np.sqrt(12), rtol=0.02)
    assert b.ess() == pytest.approx(2000)
    # each octant of the region holds ~1/8 of the particles
    octant = ((b.particles > 0.5 * (LO + HI)) * np.array([1, 2, 4])).sum(axis=1)
    assert np.abs(np.bincount(octant, minlength=8) / 2000 - 0.125).max() < 0.01


# ------------------------------------------
# test_flat_likelihood_changes_nothing
# ------------------------------------------
# A flat likelihood leaves the belief unchanged.
def test_flat_likelihood_changes_nothing():
    b, (info,) = _run([lambda q: np.full(len(q), -3.7)])
    assert info.stages == 1 and info.resamples == 0 and not info.skipped
    assert np.allclose(b.weights, 1 / 2000) and info.ess == pytest.approx(2000)


# ------------------------------------------
# test_nonfinite_likelihood_is_skipped_atomically
# ------------------------------------------
# An update with no finite likelihood changes nothing at all.
def test_nonfinite_likelihood_is_skipped_atomically():
    rng = np.random.default_rng(1)
    b = ParticleBelief(WORLD, BeliefConfig(), rng)
    p0, w0 = b.particles.copy(), b.weights.copy()
    info = b.update(lambda q: np.full(len(q), -np.inf), rng)
    assert info.skipped and np.array_equal(b.particles, p0) and np.array_equal(b.weights, w0)
    info = b.update(lambda q: np.full(len(q), np.nan), rng)
    assert info.skipped and np.array_equal(b.particles, p0)


# ------------------------------------------
# test_weights_normalised_and_particles_stay_in_region
# ------------------------------------------
# Boundary rejection: a likelihood peaked on a corner of R_T must not push particles outside.
def test_weights_normalised_and_particles_stay_in_region():
    corner = HI.copy()
    f = lambda q: -0.5 * (((q - corner) / 0.03) ** 2).sum(axis=1)
    b, infos = _run([f, f, f])
    assert np.isclose(b.weights.sum(), 1.0) and (b.weights >= 0).all()
    assert (b.particles >= LO - 1e-12).all() and (b.particles <= HI + 1e-12).all()
    assert infos[0].stages > 1 and infos[0].resamples >= 1                        # tempering was needed
    assert np.linalg.norm(b.mean() - corner) < 0.06


# ------------------------------------------
# test_tempering_applies_the_full_likelihood_once
# ------------------------------------------
# With a mild likelihood (no resampling), the update is exactly w ∝ w·exp(ℓ) however it is staged.
def test_tempering_applies_the_full_likelihood_once():
    f = lambda q: -0.5 * ((q[:, 0] - 7.8) / 0.6) ** 2
    rng = np.random.default_rng(2)
    b = ParticleBelief(WORLD, BeliefConfig(), rng)
    p0 = b.particles.copy()
    info = b.update(f, rng)
    want = np.exp(f(p0))
    want /= want.sum()
    assert info.resamples == 0 and np.array_equal(b.particles, p0)
    assert np.allclose(b.weights, want, rtol=1e-10)


# ------------------------------------------
# test_support_is_a_systematic_resample
# ------------------------------------------
# Support points are a systematic resample of the particles.
def test_support_is_a_systematic_resample():
    rng = np.random.default_rng(3)
    b = ParticleBelief(WORLD, BeliefConfig(), rng)
    b.update(lambda q: -0.5 * ((q[:, 0] - 7.3) / 0.05) ** 2, rng)
    s = b.support(24, np.random.default_rng(0))
    assert s.shape == (24, 3)
    assert abs(s[:, 0].mean() - b.mean()[0]) < 0.03
    assert np.array_equal(s, b.support(24, np.random.default_rng(0)))             # deterministic given the rng
    assert all((b.particles == row).all(axis=1).any() for row in s)               # drawn from the particle set


GAUSS = [lambda q: -0.5 * (((q[:, 0] - 7.9) / 0.06) ** 2 + ((q[:, 1] - 3.6) / 0.10) ** 2)]
CORR = [lambda q: -0.5 * (((q[:, 0] - 7.6) + (q[:, 1] - 4.4)) / 0.04) ** 2 - 0.5 * (((q[:, 0] - 7.6) - (q[:, 1] - 4.4)) / 0.5) ** 2]
STEP = [lambda q: np.where(q[:, 0] > 7.8, 0.0, -60.0) + np.where(q[:, 1] < 4.0, 0.0, -60.0)]
BIMODAL = [lambda q: np.logaddexp(-0.5 * ((q[:, 0] - 7.3) / 0.05) ** 2, -0.5 * ((q[:, 0] - 8.3) / 0.05) ** 2)
           - 0.5 * ((q[:, 1] - 4.0) / 0.2) ** 2]
REPEATED = [lambda q, c=c: -0.5 * (((q[:, 0] - c[0]) / 0.08) ** 2 + ((q[:, 1] - c[1]) / 0.08) ** 2)
            for c in [(7.50, 4.00), (7.56, 4.05), (7.47, 3.96), (7.53, 4.02), (7.49, 3.99), (7.52, 4.03)]]


# ------------------------------------------
# test_posterior_matches_grid_bayes
# ------------------------------------------
# T-B2: posterior mean (absolute, m) and standard deviation (relative) against the grid oracle, 5 seeds.
#
# The filter is an approximation (final-spec §6.2): the Liu–West move diffuses slightly at every resampling,
# so the standard-deviation tolerance is loose and one-sided inflation is expected for repeated scans.
@pytest.mark.parametrize("name,logliks,tol_mean,tol_std", [
    ("gaussian", GAUSS, 0.012, 0.20),
    ("correlated ridge", CORR, 0.03, 0.25),
    ("sharp step (silhouette-like)", STEP, 0.03, 0.15),
    ("bimodal", BIMODAL, 0.06, 0.15),
    ("six repeated scans", REPEATED, 0.012, 0.35),
])
def test_posterior_matches_grid_bayes(name, logliks, tol_mean, tol_std):
    m_ref, s_ref = _grid_posterior(logliks)
    means, stds = [], []
    for seed in range(5):
        b, _ = _run(logliks, seed)
        means.append(b.mean()[:2])
        stds.append(np.sqrt(np.diag(b.cov()))[:2])
    means, stds = np.array(means), np.array(stds)
    assert np.abs(means - m_ref).max() < tol_mean, (name, means, m_ref)
    assert np.abs(stds.mean(axis=0) / s_ref - 1).max() < tol_std, (name, stds.mean(axis=0), s_ref)
    assert (stds.mean(axis=0) / s_ref > 0.7).all()                                # never badly over-confident


# ------------------------------------------
# test_unobserved_coordinate_keeps_its_prior_spread
# ------------------------------------------
# A likelihood that says nothing about z must leave the z marginal close to uniform (no spurious collapse).
def test_unobserved_coordinate_keeps_its_prior_spread():
    b, _ = _run(REPEATED, seed=4)
    z_std = np.sqrt(b.cov()[2, 2])
    assert z_std == pytest.approx((HI[2] - LO[2]) / np.sqrt(12), rel=0.25)
