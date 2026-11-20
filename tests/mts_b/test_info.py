# ==========================================================
# tests/mts_b/test_info.py
# ==========================================================
# Quick information-channel sanity checks (F15–F17).

import numpy as np
from scipy.special import ndtr

from mts.info import log_bc, mi_lower_bound, pair_distance
from mts.predict import DesignBatch


# ------------------------------------------
# test_overlap_and_bound
# ------------------------------------------
# Bhattacharyya overlaps and the information bound for three hypotheses.
def test_overlap_and_bound():
    mu = np.array([[.2], [.8], [.5]])
    sigma = np.full_like(mu, .1)
    Z = ndtr((1-mu)/sigma) - ndtr(-mu/sigma)
    design = DesignBatch(np.array([0]), np.ones_like(mu), mu, sigma, Z, np.array([[False], [False], [True]]))
    bc = log_bc(design, 0, 1)
    np.testing.assert_allclose(bc, bc.swapaxes(0, 1))
    np.testing.assert_allclose(np.diag(bc[..., 0]), 0)
    assert np.isfinite(bc).all()
    assert 0 <= mi_lower_bound(pair_distance(design, 0, 1)) <= np.log(3)


# ------------------------------------------
# test_zero_one_and_separable
# ------------------------------------------
# Bound limits: no information, and fully separable hypotheses.
def test_zero_one_and_separable():
    pi = np.array([[0.], [1.]])
    design = DesignBatch(np.array([0]), pi, np.zeros_like(pi), np.ones_like(pi),
                         np.ones_like(pi), np.ones_like(pi, bool))
    D = pair_distance(design, 0, 1)
    assert np.isinf(D[0, 1])
    np.testing.assert_allclose(mi_lower_bound(D), np.log(2))
    assert mi_lower_bound(np.zeros((2, 2))) == 0
    assert mi_lower_bound(D, np.array([1., 0.])) == 0
