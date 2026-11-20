# ==========================================================
# src/mts/belief.py
# ==========================================================
# Regularized particle inference with adaptive likelihood tempering (F10–F13).

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp
from scipy.stats import qmc

from mts.config import BeliefConfig
from mts.world import World


# ------------------------------------------
# UpdateInfo
# ------------------------------------------
# Tempering diagnostics for one scan (§6.2).
@dataclass(frozen=True)
class UpdateInfo:
    stages: int
    resamples: int
    ess: float
    skipped: bool


# ------------------------------------------
# _systematic
# ------------------------------------------
# Systematic resampling: indices drawn with one uniform offset.
def _systematic(weights, count, rng):
    positions = (rng.random() + np.arange(count)) / count
    cdf = np.cumsum(weights)
    cdf[-1] = 1.
    return np.searchsorted(cdf, positions, side='right')


# ------------------------------------------
# ParticleBelief
# ------------------------------------------
# Scrambled Halton prior and approximate Bayes update (§6).
class ParticleBelief:
    def __init__(self, world: World, cfg: BeliefConfig, rng: np.random.Generator):
        if cfg.n_particles < 1 or not 0 < cfg.ess_frac < 1 or not 0 < cfg.min_increment <= 1:
            raise ValueError('invalid particle count or tempering configuration')
        if not 0 <= cfg.a <= 1 or cfg.sigma_j < 0:
            raise ValueError('invalid Liu–West configuration')
        self.world, self.cfg = world, cfg
        # An explicit seed draw consumes the supplied stream, also on older SciPy versions.
        unit = qmc.Halton(3, scramble=True, seed=int(rng.integers(0, 2**32))).random(cfg.n_particles)
        self.particles = world.prior_lo + unit * (world.prior_hi - world.prior_lo)
        self.weights = np.full(cfg.n_particles, 1/cfg.n_particles)

    # Weighted mean (F13).
    def mean(self) -> np.ndarray:
        return self.weights @ self.particles

    # Population weighted covariance, without a sample correction (F13).
    def cov(self) -> np.ndarray:
        delta = self.particles - self.mean()
        return (delta * self.weights[:, None]).T @ delta

    # Effective sample size (F11).
    def ess(self) -> float:
        return float(1 / np.dot(self.weights, self.weights))

    # Equal-weight systematic support for planning (§7.3).
    def support(self, M: int, rng) -> np.ndarray:
        if M < 1:
            raise ValueError('M must be positive')
        return self.particles[_systematic(self.weights, M, rng)].copy()

    # Temper (F10–F11), systematically resample and move (F12).
    #
    # Non-finite likelihoods get zero mass; an entirely invalid update is
    # skipped atomically, including if recomputation fails after a move.
    def update(self, loglik: Callable[[np.ndarray], np.ndarray], rng) -> UpdateInfo:
        original_particles, original_weights = self.particles.copy(), self.weights.copy()
        N = len(self.weights)
        threshold = self.cfg.ess_frac * N

        def evaluate():
            ll = np.asarray(loglik(self.particles), float)
            if ll.shape != (N,):
                raise ValueError('loglik must return one value per particle')
            finite = np.isfinite(ll)
            if not finite.any():
                return None
            return np.where(finite, ll - np.max(ll[finite]), -np.inf)

        ll = evaluate()
        phi, stages, resamples = 0., 0, 0
        while phi < 1 and ll is not None:
            with np.errstate(divide='ignore'):
                lw = np.log(self.weights)

            def log_cess(delta, lw=lw, ll=ll):
                if delta == 0:
                    return np.log(N)
                return np.log(N) + 2*logsumexp(lw + delta*ll) - logsumexp(lw + 2*delta*ll)

            remaining = 1 - phi
            delta = remaining
            if log_cess(delta) < np.log(threshold):
                low, high = 0., remaining
                for _ in range(50):
                    mid = (low + high) / 2
                    if log_cess(mid) >= np.log(threshold):
                        low = mid
                    else:
                        high = mid
                delta = min(remaining, max(low, self.cfg.min_increment))
            lw += delta * ll
            norm = logsumexp(lw)
            if not np.isfinite(norm):
                ll = None
                break
            self.weights = np.exp(lw - norm)
            phi = min(1., phi + delta)
            stages += 1
            if self.ess() < threshold:
                mean, cov = self.mean(), self.cov()
                self.particles = self.support(N, rng)
                self.weights = np.full(N, 1/N)
                kernel = (1 - self.cfg.a**2)*cov + self.cfg.sigma_j**2*np.eye(3)
                proposal = (self.cfg.a*self.particles + (1-self.cfg.a)*mean
                            + rng.multivariate_normal(np.zeros(3), kernel, size=N))
                valid = ((proposal >= self.world.prior_lo) & (proposal <= self.world.prior_hi)).all(axis=1)
                self.particles[valid] = proposal[valid]
                resamples += 1
                ll = evaluate()
        if ll is None:
            self.particles, self.weights = original_particles, original_weights
        return UpdateInfo(stages, resamples, self.ess(), ll is None)
