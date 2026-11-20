# ==========================================================
# tests/mts_b/test_experiment_info_mc.py
# ==========================================================
# Small independent oracles for F04's sampling and exact mixture evaluation.

import numpy as np
from scipy.integrate import quad
from scipy.stats import truncnorm

from mts.predict import DesignBatch
from sim.experiments.F04_info_bound import exact_mi_mc


# ------------------------------------------
# test_identical_channels_cancel
# ------------------------------------------
# Identical channels carry no information.
def test_identical_channels_cancel():
    channel = DesignBatch(np.array([0]), np.full((2, 1), .4), np.zeros((2, 1)),
                          np.ones((2, 1)), np.ones((2, 1)), np.ones((2, 1), bool))
    assert exact_mi_mc(channel, 32, np.random.default_rng(3), 0., 1.) == (0., 0., 0)


# ------------------------------------------
# test_report_flags_against_enumeration
# ------------------------------------------
# Mutual information of report flags matches exact enumeration.
def test_report_flags_against_enumeration():
    pi = np.array([[.1, .8], [.7, .3]])
    channel = DesignBatch(np.arange(2), pi, np.zeros_like(pi), np.ones_like(pi),
                          np.ones_like(pi), np.ones_like(pi, bool))
    exact = 0.
    for a in (0, 1):
        for b in (0, 1):
            prob = np.prod(np.where(np.array([a, b]), pi, 1 - pi), axis=1)
            exact += np.mean(prob * np.log(prob / prob.mean()))
    estimate, se, beams = exact_mi_mc(channel, 6000, np.random.default_rng(17), 0., 1.)
    assert beams == 2
    assert abs(estimate - exact) < 4 * se


# ------------------------------------------
# test_truncated_gaussian_uniform_against_quadrature
# ------------------------------------------
# Mutual information of a truncated Gaussian and a uniform matches quadrature.
def test_truncated_gaussian_uniform_against_quadrature():
    pi = np.array([.8, .4])
    mu, sigma = -.1, .3
    channel = DesignBatch(np.array([0]), pi[:, None], np.array([[mu], [0.]]),
                          np.array([[sigma], [1.]]), np.array([[.36], [1.]]), np.array([[False], [True]]))

    def integrand(z):
        density = pi * np.array([truncnorm.pdf(z, -mu / sigma, (1 - mu) / sigma, loc=mu, scale=sigma), 1.])
        return np.mean(density * np.log(density / density.mean()))

    missing = 1 - pi
    exact = np.mean(missing * np.log(missing / missing.mean())) + quad(integrand, 0, 1)[0]
    estimate, se, _ = exact_mi_mc(channel, 6000, np.random.default_rng(24), 0., 1.)
    assert abs(estimate - exact) < 4 * se
