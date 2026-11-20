# ==========================================================
# src/mts/info.py
# ==========================================================
# Closed-form Bhattacharyya overlaps and information bound (F15–F17).

import numpy as np
from scipy.special import log_ndtr, logsumexp

from mts.predict import DesignBatch


# ------------------------------------------
# _log_window
# ------------------------------------------
# Log Gaussian window mass, using survival probabilities in the upper tail.
def _log_window(mu, sigma, lo, hi):
    a, b = (lo - mu) / sigma, (hi - mu) / sigma
    upper = np.where(a > 0, log_ndtr(-a), log_ndtr(b))
    lower = np.where(a > 0, log_ndtr(-b), log_ndtr(a))
    with np.errstate(divide='ignore', invalid='ignore'):
        return upper + np.log(-np.expm1(lower - upper))


# ------------------------------------------
# log_bc
# ------------------------------------------
# Per-beam log coefficients with all three closed-form overlaps (F15).
#
# Z=1 for both Gaussians denotes the untruncated blind channel. For aware
# Gaussians whose masses round to one, this has only roundoff-level effect.
# Densities of zero-probability report outcomes are never evaluated.
def log_bc(design: DesignBatch, r_min: float, r_max: float) -> np.ndarray:
    indices = np.arange(len(design.pi))
    result = _log_bc_pairs(design, r_min, r_max, indices[:, None], indices[None, :])
    result[indices, indices] = 0.
    return result


# ------------------------------------------
# _log_bc_pairs
# ------------------------------------------
# Closed forms for selected pairs; shared by the full and triangular paths.
def _log_bc_pairs(design, r_min, r_max, left, right):
    pi = np.asarray(design.pi, float)
    active = (pi > 0) & ~design.uniform
    mu = np.where(active, design.mu, 0.)
    sigma = np.where(active, design.sigma, 1.)
    with np.errstate(divide='ignore'):
        logz = np.log(np.where(active, design.Z, 1.))
    missing = active & (design.Z == 0)
    if missing.any():
        logz[missing] = _log_window(mu[missing], sigma[missing], r_min, r_max)
    m, mp = mu[left], mu[right]
    s, sp = sigma[left], sigma[right]
    v = s*s + sp*sp
    star_z = 0.
    if not np.all(design.Z == 1):
        star_mu = (m*sp*sp + mp*s*s) / v
        star_sigma = np.sqrt(2 * s*s * sp*sp / v)
        star_z = _log_window(star_mu, star_sigma, r_min, r_max)
        star_z = np.where((design.Z[left] == 1) & (design.Z[right] == 1), 0., star_z)
    overlap = .5 * np.log(2*s*sp/v) - (m-mp)**2/(4*v) + star_z - .5*(logz[left] + logz[right])
    if design.uniform.any():
        ug = (.25 * np.log(8*np.pi*sigma**2) + _log_window(mu, np.sqrt(2)*sigma, r_min, r_max)
              - .5*(logz + np.log(r_max-r_min)))
        u, up = design.uniform[left], design.uniform[right]
        overlap = np.where(u, ug[right], np.where(up, ug[left], overlap))
        overlap = np.where(u & up, 0., overlap)
    report = (pi[left] > 0) & (pi[right] > 0)
    overlap = np.where(report, np.minimum(overlap, 0.), 0.)
    with np.errstate(divide='ignore'):
        no_report = .5 * (np.log1p(-pi[left]) + np.log1p(-pi[right]))
        yes_report = .5 * (np.log(pi[left]) + np.log(pi[right])) + overlap
    return np.minimum(np.logaddexp(no_report, yes_report), 0.)


# ------------------------------------------
# pair_distance
# ------------------------------------------
# Add independent beam distances (F16), evaluating each unordered pair once.
def pair_distance(design, r_min, r_max) -> np.ndarray:
    M = len(design.pi)
    left, right = np.triu_indices(M, 1)
    distance = -_log_bc_pairs(design, r_min, r_max, left, right).sum(axis=-1)
    result = np.zeros((M, M))
    result[left, right] = result[right, left] = distance
    return result


# ------------------------------------------
# mi_lower_bound
# ------------------------------------------
# Mixture entropy lower bound in nats (F17), optionally weighted.
def mi_lower_bound(D: np.ndarray, weights: np.ndarray | None = None) -> float:
    D = np.asarray(D, float)
    M = len(D)
    if D.shape != (M, M) or M == 0:
        raise ValueError('D must be a nonempty square matrix')
    w = np.full(M, 1/M) if weights is None else np.asarray(weights, float)
    if w.shape != (M,) or np.any(w < 0) or not np.isfinite(w).all() or w.sum() <= 0:
        raise ValueError('invalid weights')
    w = w / w.sum()
    keep = w > 0
    logw = np.log(w[keep])
    value = -np.dot(w[keep], logsumexp(logw[None, :] - D[np.ix_(keep, keep)], axis=1))
    return float(np.clip(value, 0., -np.dot(w[keep], logw)))
