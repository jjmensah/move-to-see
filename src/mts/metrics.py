# ==========================================================
# src/mts/metrics.py
# ==========================================================
# Trial metrics (F20–F22) and registered paired statistics (V&V §1).

import numpy as np
from scipy.stats import norm, wilcoxon


# ------------------------------------------
# position_error
# ------------------------------------------
# Euclidean position error (F20).
def position_error(q_hat, q_true) -> np.ndarray:
    return np.linalg.norm(np.asarray(q_hat) - q_true, axis=-1)


# ------------------------------------------
# dwell_time
# ------------------------------------------
# First start time of a qualifying dwell (F21); err[0] is the prior at t=0.
def dwell_time(err, dt, tol=0.10, dwell=5) -> float:
    err = np.asarray(err)
    if err.ndim != 1 or dwell < 1 or dt <= 0:
        raise ValueError('expected a time series, positive dt and positive dwell')
    if len(err) < dwell:
        return float('inf')
    good = np.convolve((err <= tol).astype(int), np.ones(dwell, int), mode='valid') == dwell
    starts = np.flatnonzero(good)
    return float(starts[0]*dt) if starts.size else float('inf')


# ------------------------------------------
# time_score
# ------------------------------------------
# Restricted dwell time (F21), for T+1 errors including the prior.
def time_score(err, dt, tol=0.10, dwell=5) -> float:
    return min(dwell_time(err, dt, tol, dwell), max(0, len(err)-1)*dt)


# ------------------------------------------
# _regularized
# ------------------------------------------
# Adds the 1 mm covariance floor (F22).
def _regularized(cov):
    cov = np.asarray(cov, float)
    return cov + 1e-6*np.eye(cov.shape[-1])


# ------------------------------------------
# entropy_proxy
# ------------------------------------------
# Gaussian entropy with the 1 mm covariance floor (F22).
def entropy_proxy(cov) -> np.ndarray:
    c = _regularized(cov)
    return .5*(np.linalg.slogdet(c)[1] + c.shape[-1]*np.log(2*np.pi*np.e))


# ------------------------------------------
# nees
# ------------------------------------------
# Normalized squared estimation error with the 1 mm floor (F22).
def nees(q_hat, cov, q_true) -> np.ndarray:
    delta = np.asarray(q_hat) - q_true
    return np.sum(delta * np.linalg.solve(_regularized(cov), delta[..., None])[..., 0], axis=-1)


# ------------------------------------------
# wilson
# ------------------------------------------
# Wilson score interval for a binomial success rate (V&V §1).
def wilson(k, n, conf=0.99) -> tuple[float, float]:
    if n <= 0 or not 0 <= k <= n or not 0 < conf < 1:
        raise ValueError('invalid binomial count or confidence')
    z = norm.ppf((1+conf)/2)
    p = k/n
    centre = (p + z*z/(2*n))/(1 + z*z/n)
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/(1 + z*z/n)
    return float(max(0, centre-half)), float(min(1, centre+half))


# ------------------------------------------
# paired_wilcoxon
# ------------------------------------------
# Paired signed-rank p-value with Pratt zeros (V&V §1).
def paired_wilcoxon(a, b, alternative='less') -> float:
    delta = np.asarray(a, float) - np.asarray(b, float)
    if np.all(delta == 0):
        return 1.
    return float(wilcoxon(delta, alternative=alternative, zero_method='pratt').pvalue)


# ------------------------------------------
# holm
# ------------------------------------------
# Holm adjusted p-values in the original order (V&V §1).
def holm(pvalues) -> np.ndarray:
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(1, np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]))
    return adjusted


# ------------------------------------------
# bootstrap_ci
# ------------------------------------------
# Percentile bootstrap interval (V&V §1); requires an explicit Generator.
def bootstrap_ci(x, stat=np.median, n=10_000, conf=0.95, rng=None) -> tuple[float, float]:
    if rng is None:
        raise ValueError('an explicit rng is required')
    x = np.asarray(x)
    if x.ndim != 1 or not len(x) or n < 1 or not 0 < conf < 1:
        raise ValueError('invalid bootstrap sample or configuration')
    values = np.empty(n)
    for i in range(n):
        values[i] = stat(x[rng.integers(len(x), size=len(x))])
    lo, hi = np.quantile(values, [(1-conf)/2, (1+conf)/2])
    return float(lo), float(hi)
