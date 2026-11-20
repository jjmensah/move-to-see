# ==========================================================
# tests/mts_b_oracles/test_info_oracles.py
# ==========================================================
# T-I1, T-I2: Bhattacharyya coefficients and the mutual-information bound (F15–F17) against quadrature.

import math

import numpy as np
import pytest
from scipy import integrate
from scipy.stats import norm

from mts.info import log_bc, mi_lower_bound, pair_distance
from mts.predict import DesignBatch

R_MIN, R_MAX = 0.02, 5.6
W = R_MAX - R_MIN


# ------------------------------------------
# _density
# ------------------------------------------
# Conditional report density on the window: truncated Gaussian, untruncated Gaussian ('blind') or uniform.
def _density(kind, mu, sigma):
    if kind == "u":
        return (lambda z: 1.0 / W), 1.0
    if kind == "blind":
        return (lambda z: norm.pdf(z, mu, sigma)), 1.0
    Z = norm.cdf((R_MAX - mu) / sigma) - norm.cdf((R_MIN - mu) / sigma)
    return (lambda z: norm.pdf(z, mu, sigma) / Z), Z


# ------------------------------------------
# _bc_quad
# ------------------------------------------
# BC of two beam outcome distributions by numerical integration (atom for 'no report' + density on W).
def _bc_quad(a, b):
    (ka, pa, ma, sa), (kb, pb, mb, sb) = a, b
    fa, _ = _density(ka, ma, sa)
    fb, _ = _density(kb, mb, sb)
    lo, hi = (-math.inf, math.inf) if ka == kb == "blind" else (R_MIN, R_MAX)
    pts = None if lo == -math.inf else sorted({min(max(m, R_MIN), R_MAX) for m in (ma, mb)})
    if lo == -math.inf:
        centre = 0.5 * (ma + mb)
        lo, hi = centre - 60 * max(sa, sb), centre + 60 * max(sa, sb)
        pts = [ma, mb]
    val, _ = integrate.quad(lambda z: math.sqrt(fa(z) * fb(z)), lo, hi, points=pts, limit=400, epsabs=1e-13, epsrel=1e-11)
    return math.sqrt((1 - pa) * (1 - pb)) + math.sqrt(pa * pb) * val


# ------------------------------------------
# _batch
# ------------------------------------------
# DesignBatch with one hypothesis per case and a single beam.
def _batch(cases):
    kinds = [c[0] for c in cases]
    pi = np.array([[c[1]] for c in cases])
    mu = np.array([[c[2]] for c in cases])
    sigma = np.array([[c[3]] for c in cases])
    Z = np.array([[_density(*(c[0], c[2], c[3]))[1]] for c in cases])
    return DesignBatch(np.array([0]), pi, mu, sigma, Z, np.array([[k == "u"] for k in kinds]))


CASES = [
    ("g", 0.9, 2.0, 0.05), ("g", 0.4, 2.03, 0.02), ("g", 0.7, 2.5, 0.3),      # Gaussians inside the window
    ("g", 0.6, 5.55, 0.08), ("g", 0.3, 5.7, 0.2), ("g", 0.8, 0.05, 0.05),     # truncated at the window edges
    ("u", 2.5e-5, 0.0, 1.0), ("u", 1e-3, 0.0, 1.0),                           # no optical return (uniform)
    ("g", 1.0, 3.0, 0.012), ("g", 0.0, 3.0, 0.012),                           # π = 1 and π = 0
]


# ------------------------------------------
# test_bc_closed_forms_match_quadrature
# ------------------------------------------
# T-I1: every pair among Gaussian (truncated or not), uniform, π = 0 and π = 1 cases.
def test_bc_closed_forms_match_quadrature():
    got = np.exp(log_bc(_batch(CASES), R_MIN, R_MAX)[:, :, 0])
    for i, a in enumerate(CASES):
        for j, b in enumerate(CASES):
            want = 1.0 if i == j else _bc_quad(a, b)
            assert got[i, j] == pytest.approx(want, rel=2e-8, abs=1e-10), (a, b)
    assert np.allclose(got, got.T) and (got >= 0).all() and (got <= 1 + 1e-12).all()
    assert np.allclose(np.diag(got), 1.0)


# ------------------------------------------
# test_bc_blind_model_is_untruncated
# ------------------------------------------
# Z = 1 for both hypotheses denotes the physics-blind channel: Gaussian overlap on the whole real line.
def test_bc_blind_model_is_untruncated():
    cases = [("blind", 1.0, 5.59, 0.012), ("blind", 1.0, 5.6, 0.012), ("blind", 1.0, 3.0, 0.012), ("blind", 0.0, 0.0, 0.012)]
    got = np.exp(log_bc(_batch(cases), R_MIN, R_MAX)[:, :, 0])
    for i, a in enumerate(cases):
        for j, b in enumerate(cases):
            if i != j:
                assert got[i, j] == pytest.approx(_bc_quad(a, b), rel=1e-8, abs=1e-12)
    d = (5.6 - 5.59) ** 2 / (8 * 0.012**2)
    assert got[0, 1] == pytest.approx(math.exp(-d), rel=1e-10)          # equal-variance closed form


# ------------------------------------------
# test_pair_distance_adds_over_beams
# ------------------------------------------
# F16: independent beams multiply their coefficients.
def test_pair_distance_adds_over_beams(rng):
    M, B = 4, 6
    pi = rng.uniform(0.05, 0.95, (M, B))
    mu = rng.uniform(1.0, 4.5, (M, B))
    sigma = rng.uniform(0.01, 0.2, (M, B))
    Z = norm.cdf((R_MAX - mu) / sigma) - norm.cdf((R_MIN - mu) / sigma)
    full = DesignBatch(np.arange(B), pi, mu, sigma, Z, np.zeros((M, B), bool))
    D = pair_distance(full, R_MIN, R_MAX)
    per = sum(-log_bc(DesignBatch(np.array([b]), pi[:, [b]], mu[:, [b]], sigma[:, [b]], Z[:, [b]], np.zeros((M, 1), bool)),
                      R_MIN, R_MAX)[:, :, 0] for b in range(B))
    assert np.allclose(D, per, rtol=1e-12, atol=1e-12)
    assert np.allclose(np.diag(D), 0) and (D >= -1e-12).all() and np.allclose(D, D.T)


# ------------------------------------------
# test_mi_bound_limits_and_invariances
# ------------------------------------------
# T-I2: range, extremes, duplicates and permutation invariance of F17.
def test_mi_bound_limits_and_invariances(rng):
    M = 6
    assert mi_lower_bound(np.zeros((M, M))) == pytest.approx(0.0, abs=1e-12)            # indistinguishable
    far = np.full((M, M), 1e6)
    np.fill_diagonal(far, 0.0)
    assert mi_lower_bound(far) == pytest.approx(math.log(M), rel=1e-12)                 # perfectly separable
    dup = far.copy()
    dup[0, 1] = dup[1, 0] = 0.0                                                         # two identical hypotheses
    assert mi_lower_bound(dup) == pytest.approx(math.log(M) - (2 / M) * math.log(2), rel=1e-12)
    A = rng.uniform(0, 3, (M, M))
    D = A + A.T
    np.fill_diagonal(D, 0.0)
    v = mi_lower_bound(D)
    assert 0.0 <= v <= math.log(M)
    want = -np.mean(np.log(np.mean(np.exp(-D), axis=1)))                                # F17 written out
    assert v == pytest.approx(want, rel=1e-12)
    perm = rng.permutation(M)
    assert mi_lower_bound(D[np.ix_(perm, perm)]) == pytest.approx(v, rel=1e-12)


# ------------------------------------------
# _exact_mi_one_beam
# ------------------------------------------
# Exact mutual information (nats) between a uniform hypothesis index and one beam outcome, by quadrature.
def _exact_mi_one_beam(cases):
    M = len(cases)
    dens = [_density(k, m, s)[0] for k, _, m, s in cases]
    pis = [c[1] for c in cases]
    p_none = [1 - p for p in pis]
    mix_none = sum(p_none) / M
    total = sum((pn / M) * math.log(pn / mix_none) for pn in p_none if pn > 0)

    def integrand(z, m):
        pm = pis[m] * dens[m](z)
        mix = sum(pis[j] * dens[j](z) for j in range(M)) / M
        return 0.0 if pm <= 0 else (pm / M) * math.log(pm / mix)

    pts = sorted({min(max(c[2], R_MIN), R_MAX) for c in cases if c[0] == "g"})
    for m in range(M):
        val, _ = integrate.quad(integrand, R_MIN, R_MAX, args=(m,), points=pts, limit=400)
        total += val
    return total


# ------------------------------------------
# test_mi_bound_is_below_exact_mutual_information
# ------------------------------------------
# T-I2: Î (F17) ≤ exact MI of the design channel (one beam; random hypotheses incl. a no-return one).
@pytest.mark.parametrize("seed", range(8))
def test_mi_bound_is_below_exact_mutual_information(seed):
    rng = np.random.default_rng(seed)
    M = int(rng.integers(2, 6))
    cases = [("g", float(rng.uniform(0.1, 1.0)), float(rng.uniform(1.0, 4.5)), float(rng.uniform(0.02, 0.3))) for _ in range(M)]
    if seed % 2:
        cases[0] = ("u", 1e-3, 0.0, 1.0)
    exact = _exact_mi_one_beam(cases)
    bound = mi_lower_bound(pair_distance(_batch(cases), R_MIN, R_MAX))
    assert bound <= exact + 1e-9
    assert 0.0 <= bound <= math.log(M) and exact <= math.log(M) + 1e-9


# ------------------------------------------
# test_mi_bound_below_exact_for_two_beams
# ------------------------------------------
# Two independent beams: exact MI by a 2-D integral over the (report, report) branch plus the mixed branches.
def test_mi_bound_below_exact_for_two_beams():
    cases = [[("g", 0.8, 2.0, 0.05), ("g", 0.6, 3.0, 0.10)],      # hypothesis 0: beam 0, beam 1
             [("g", 0.7, 2.08, 0.05), ("g", 0.9, 3.05, 0.10)],
             [("g", 0.2, 2.6, 0.20), ("g", 0.5, 3.4, 0.15)]]
    M = len(cases)
    f = [[_density(k, m, s)[0] for k, _, m, s in h] for h in cases]
    pi = [[c[1] for c in h] for h in cases]

    def term(pm, pmix):
        return 0.0 if pm <= 0 else (pm / M) * math.log(pm / pmix)

    total = 0.0
    for m in range(M):                                            # neither beam reports
        pm = [(1 - pi[j][0]) * (1 - pi[j][1]) for j in range(M)]
        total += term(pm[m], sum(pm) / M)
    for b, other in ((0, 1), (1, 0)):                             # exactly one beam reports
        for m in range(M):
            def g(z, m=m, b=b, other=other):
                pj = [pi[j][b] * f[j][b](z) * (1 - pi[j][other]) for j in range(M)]
                return term(pj[m], sum(pj) / M)
            total += integrate.quad(g, R_MIN, R_MAX, points=[2.0, 2.08, 2.6, 3.0, 3.05, 3.4], limit=300)[0]
    for m in range(M):                                            # both report
        def g2(z1, z0, m=m):
            pj = [pi[j][0] * f[j][0](z0) * pi[j][1] * f[j][1](z1) for j in range(M)]
            return term(pj[m], sum(pj) / M)
        total += integrate.dblquad(g2, 1.2, 3.8, 2.0, 4.6, epsabs=1e-7, epsrel=1e-6)[0]
    pi_a = np.array(pi)
    mu = np.array([[c[2] for c in h] for h in cases])
    sg = np.array([[c[3] for c in h] for h in cases])
    Z = norm.cdf((R_MAX - mu) / sg) - norm.cdf((R_MIN - mu) / sg)
    bound = mi_lower_bound(pair_distance(DesignBatch(np.arange(2), pi_a, mu, sg, Z, np.zeros((M, 2), bool)), R_MIN, R_MAX))
    assert 0.0 < bound <= total + 1e-5 <= math.log(M) + 1e-5
