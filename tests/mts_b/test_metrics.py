# ==========================================================
# tests/mts_b/test_metrics.py
# ==========================================================
# Quick metric and registered-statistics checks (F20–F22).

import numpy as np

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
# test_metrics
# ------------------------------------------
# Position error, dwell time, time score, NEES and entropy on small cases.
def test_metrics():
    assert position_error([3, 4, 0], [0, 0, 0]) == 5
    assert dwell_time([1, 0, 0, 0, 0, 0], .5) == .5
    assert np.isinf(dwell_time([1, 0], .5))
    assert time_score([1]*41, .5) == 20
    np.testing.assert_allclose(nees([1, 0, 0], np.eye(3), [0, 0, 0]), 1/1.000001)
    assert np.isfinite(entropy_proxy(np.zeros((3, 3))))


# ------------------------------------------
# test_statistics
# ------------------------------------------
# Holm, Wilson, Wilcoxon and bootstrap on small cases.
def test_statistics():
    np.testing.assert_allclose(holm([.03, .01, .04]), [.06, .03, .06])
    lo, hi = wilson(5, 10)
    assert 0 < lo < .5 < hi < 1
    assert paired_wilcoxon([1, 1], [1, 1]) == 1
    a = bootstrap_ci(np.arange(10), n=100, rng=np.random.default_rng(2))
    b = bootstrap_ci(np.arange(10), n=100, rng=np.random.default_rng(2))
    assert a == b
