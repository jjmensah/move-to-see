# ==========================================================
# tests/mts_b_oracles/test_metrics_oracles.py
# ==========================================================
# T-M1: metrics and statistics (F20–F22, V&V §1) against hand calculations.

import itertools
import math

import numpy as np
import pytest

from mts.metrics import (
    bootstrap_ci,
    dwell_time,
    entropy_proxy,
    holm,
    nees,
    paired_wilcoxon,
    position_error,
    time_score,
    wilson,
)


# ------------------------------------------
# test_position_error
# ------------------------------------------
# Position error on hand-picked points.
def test_position_error():
    q_hat = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 12.0], [1.0, 2.0, 2.0]])
    assert np.allclose(position_error(q_hat, np.zeros(3)), [0.0, 13.0, 3.0])


# ------------------------------------------
# test_dwell_time_and_score
# ------------------------------------------
# F21: first k with e_j ≤ tol for j = k..k+4; index k is time k·Δt (index 0 = prior).
def test_dwell_time_and_score():
    dt = 0.5
    e = np.array([1.0, 0.5, 0.09, 0.2, 0.08, 0.07, 0.10, 0.05, 0.04, 0.3, 0.01])
    assert dwell_time(e, dt) == pytest.approx(4 * dt)          # 0.08, 0.07, 0.10 (inclusive), 0.05, 0.04
    assert time_score(e, dt) == pytest.approx(4 * dt)
    never = np.array([1.0, 0.09, 0.09, 0.09, 0.09, 0.2, 0.09, 0.09, 0.09, 0.09, 0.5])
    assert dwell_time(never, dt) == math.inf
    assert time_score(never, dt) == pytest.approx(10 * dt)     # restricted to the trial length T·Δt
    late = np.concatenate([np.ones(56), np.full(5, 0.05)])     # qualifies only with the last five scans
    assert dwell_time(late, dt) == pytest.approx(56 * dt)
    almost = np.concatenate([np.ones(57), np.full(4, 0.05)])   # only four accurate scans at the end
    assert dwell_time(almost, dt) == math.inf
    assert dwell_time(e, dt, tol=0.10, dwell=1) == pytest.approx(2 * dt)
    assert dwell_time(np.array([0.05, 0.05]), dt) == math.inf  # shorter than the dwell


# ------------------------------------------
# test_entropy_proxy_and_nees
# ------------------------------------------
# F22 with the (1 mm)² regularisation.
def test_entropy_proxy_and_nees():
    cov = np.diag([0.04, 0.01, 0.0025])
    reg = cov + 1e-6 * np.eye(3)
    want_h = 0.5 * math.log((2 * math.pi * math.e) ** 3 * np.prod(np.diag(reg)))
    assert entropy_proxy(cov) == pytest.approx(want_h, rel=1e-12)
    q_hat, q = np.array([1.0, 2.0, 3.0]), np.array([1.2, 1.9, 3.05])
    d = q_hat - q
    assert nees(q_hat, cov, q) == pytest.approx(float(np.sum(d * d / np.diag(reg))), rel=1e-12)
    assert np.isfinite(entropy_proxy(np.zeros((3, 3)))) and np.isfinite(nees(q_hat, np.zeros((3, 3)), q))
    # batched over time
    covs = np.stack([cov, 4 * cov])
    assert entropy_proxy(covs).shape == (2,) and nees(np.stack([q_hat, q_hat]), covs, q).shape == (2,)


# ------------------------------------------
# test_wilson_interval
# ------------------------------------------
# Wilson interval against the closed form.
def test_wilson_interval():
    lo, hi = wilson(7, 20, conf=0.95)
    z, p, n = 1.959963984540054, 0.35, 20
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    assert (lo, hi) == pytest.approx((centre - half, centre + half), rel=1e-9)
    assert wilson(0, 10)[0] == 0.0 and wilson(10, 10)[1] == 1.0
    assert wilson(5, 10, 0.99)[1] - wilson(5, 10, 0.99)[0] > wilson(5, 10, 0.95)[1] - wilson(5, 10, 0.95)[0]


# ------------------------------------------
# _exact_signed_rank_p_less
# ------------------------------------------
# Exact one-sided p (a < b, i.e. negative differences) by enumerating sign patterns (no ties, no zeros).
def _exact_signed_rank_p_less(delta):
    r = np.argsort(np.argsort(np.abs(delta))) + 1
    w_plus = r[delta > 0].sum()
    count = sum(1 for signs in itertools.product([0, 1], repeat=len(delta)) if np.dot(signs, r) <= w_plus)
    return count / 2 ** len(delta)


# ------------------------------------------
# test_paired_wilcoxon_matches_exact_enumeration
# ------------------------------------------
# Wilcoxon p-value against exact enumeration of signs.
def test_paired_wilcoxon_matches_exact_enumeration():
    a = np.array([1.2, 3.4, 2.2, 5.0, 4.1, 2.9, 6.3, 3.3, 4.4, 2.0])
    b = np.array([2.0, 3.5, 3.7, 5.3, 6.3, 2.5, 8.0, 4.4, 5.8, 3.3])
    assert paired_wilcoxon(a, b, "less") == pytest.approx(_exact_signed_rank_p_less(a - b), rel=1e-9)
    assert paired_wilcoxon(b, a, "less") > 0.9
    assert paired_wilcoxon(a, a, "less") == 1.0                # all-zero differences


# ------------------------------------------
# test_holm_adjustment
# ------------------------------------------
# Holm adjustment on hand-worked examples.
def test_holm_adjustment():
    assert np.allclose(holm([0.01, 0.04, 0.03, 0.005]), [0.03, 0.06, 0.06, 0.02])
    assert np.allclose(holm([0.5, 0.9]), [1.0, 1.0])
    assert np.allclose(holm([0.002]), [0.002])


# ------------------------------------------
# test_bootstrap_ci_is_deterministic_and_sensible
# ------------------------------------------
# The bootstrap interval is reproducible and brackets the median.
def test_bootstrap_ci_is_deterministic_and_sensible():
    x = np.arange(1.0, 41.0)
    a = bootstrap_ci(x, np.median, n=2000, conf=0.95, rng=np.random.default_rng(5))
    b = bootstrap_ci(x, np.median, n=2000, conf=0.95, rng=np.random.default_rng(5))
    assert a == b and a[0] < 20.5 < a[1] and 10 < a[0] and a[1] < 31
    with pytest.raises(ValueError):
        bootstrap_ci(x, rng=None)
