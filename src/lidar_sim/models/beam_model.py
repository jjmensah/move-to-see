# ==========================================================
# src/lidar_sim/models/beam_model.py
# ==========================================================
# Independent analytic per-beam prediction and likelihood (E16–E21).

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from scipy.special import ndtr

from lidar_sim.core.config import ModelConfig, NoiseConfig, SensorConfig
from lidar_sim.core.constants import C_LIGHT
from lidar_sim.core.types import BeamPrediction, RadiometricScan, Scan
from lidar_sim.models.detection import cfar_gamma, pfa, rice_pd


# ------------------------------------------
# _phasor
# ------------------------------------------
# Expected envelope phasor of a beam at one modulation frequency (E10).
def _phasor(trace: RadiometricScan, frequency: float, noise: NoiseConfig) -> np.ndarray:
    active = trace.hit & (trace.power > 0)
    phase = 4.0 * np.pi * frequency * np.where(active, trace.r_path, 0.0) / C_LIGHT
    weighted = np.where(active, trace.power, 0.0) * np.exp(1j * phase)
    return noise.responsivity * noise.mod_depth * np.sum(weighted, axis=1)


# ------------------------------------------
# _fine
# ------------------------------------------
# Fine range from a phase at one frequency (E13).
def _fine(phi: np.ndarray, frequency: float) -> np.ndarray:
    return np.mod(C_LIGHT * np.mod(phi, 2 * np.pi) / (4 * np.pi * frequency), C_LIGHT / (2 * frequency))


# ------------------------------------------
# _fused_range
# ------------------------------------------
# Unwrapped, variance-weighted range from both frequencies (E14).
def _fused_range(z1: np.ndarray, z2: np.ndarray, sig1: np.ndarray, sig2: np.ndarray,
                 sensor: SensorConfig) -> np.ndarray:
    phi1, phi2 = np.angle(z1), np.angle(z2)
    r1, r2 = _fine(phi1, sensor.f1), _fine(phi2, sensor.f2)
    rc = C_LIGHT * np.mod(phi2 - phi1, 2 * np.pi) / (4 * np.pi * (sensor.f2 - sensor.f1))
    k1 = np.rint((rc - r1) / sensor.L1)
    k2 = np.rint((rc - r2) / sensor.L2)
    with np.errstate(over="ignore", invalid="ignore"):
        sig1_sq, sig2_sq = np.square(sig1), np.square(sig2)
    g1 = np.divide(1.0, sig1_sq, out=np.zeros_like(sig1), where=np.isfinite(sig1) & (sig1 > 0))
    g2 = np.divide(1.0, sig2_sq, out=np.zeros_like(sig2), where=np.isfinite(sig2) & (sig2 > 0))
    denom = g1 + g2
    value = np.divide(g1 * (r1 + k1 * sensor.L1) + g2 * (r2 + k2 * sensor.L2), denom,
                      out=np.zeros_like(denom), where=denom > 0)
    return np.mod(value, sensor.Lc)


# ------------------------------------------
# expected_range
# ------------------------------------------
# Return noise-free dual-frequency fused range r̄ [m], shape (B,).
#
# Parameters
# trace : RadiometricScan
#     Deterministic optical returns, with ambient in ``trace.P_amb``.
# sensor : SensorConfig
#     Modulation frequencies and unambiguous range intervals.
# noise : NoiseConfig
#     Responsivity, modulation depth and calibrated noise parameters.
#
# Returns
# ndarray, shape (B,)
#     E13–E14 fused noise-free range [m].
#
# Implements (E19).
def expected_range(trace: RadiometricScan, sensor: SensorConfig, noise: NoiseConfig) -> np.ndarray:
    z1, z2 = _phasor(trace, sensor.f1, noise), _phasor(trace, sensor.f2, noise)
    powers = np.sum(np.where(trace.hit, trace.power, 0.0), axis=1)
    sigma_n = np.sqrt(noise.sigma_th**2 + noise.kappa_sh * noise.responsivity * (powers + trace.P_amb))
    a1, a2 = np.abs(z1), np.abs(z2)
    sig1 = np.divide(C_LIGHT * np.sqrt(2.0) * sigma_n, 4 * np.pi * sensor.f1 * np.sqrt(sensor.n_samples) * a1,
                     out=np.full_like(a1, np.inf), where=a1 > 0)
    sig2 = np.divide(C_LIGHT * np.sqrt(2.0) * sigma_n, 4 * np.pi * sensor.f2 * np.sqrt(sensor.n_samples) * a2,
                     out=np.full_like(a2, np.inf), where=a2 > 0)
    return _fused_range(z1, z2, sig1, sig2, sensor)


# ------------------------------------------
# BeamModel
# ------------------------------------------
# Vectorized analytic detection, range and likelihood model (E16–E21).
class BeamModel:
    def __init__(self, sensor: SensorConfig = SensorConfig(), noise: NoiseConfig = NoiseConfig(),  # noqa: B008
                 model: ModelConfig = ModelConfig()):  # noqa: B008
        self.sensor, self.noise, self.model = sensor, noise, model

    # Predict per-beam amplitudes, detection and range statistics from a trace.
    #
    # Parameters
    # trace : RadiometricScan
    #     Optical returns; ambient is read only from ``trace.P_amb`` [W].
    # env : EnvConfig
    #     Accepted for API symmetry; fog is already represented in trace powers.
    #
    # Returns
    # BeamPrediction
    #     Per-beam predictions with A/sigma [A], ranges [m], and probabilities.
    #
    # Implements (E16–E19).
    def predict(self, trace: RadiometricScan, env) -> BeamPrediction:
        del env
        z1, z2 = _phasor(trace, self.sensor.f1, self.noise), _phasor(trace, self.sensor.f2, self.noise)
        z = np.stack((z1, z2), axis=1)
        A = np.abs(z)
        pbar = np.sum(np.where(trace.hit, trace.power, 0.0), axis=1)
        sigma0 = np.sqrt(self.noise.sigma_th**2 + self.noise.kappa_sh * self.noise.responsivity *
                         (pbar + trace.P_amb))
        sigma_n = np.broadcast_to(sigma0[:, None], A.shape).copy()
        s = sigma_n * np.sqrt(2.0 / self.sensor.n_samples)
        gamma = cfar_gamma(self.noise, trace.P_amb, self.sensor.n_samples)
        pd = rice_pd(A[:, 0], s[:, 0], gamma)
        # P_FA describes the background-only noise distribution, independent of
        # signal shot noise in a beam that contains a return.
        s_background = np.sqrt(2.0 / self.sensor.n_samples) * np.sqrt(
            self.noise.sigma_th**2 + self.noise.kappa_sh * self.noise.responsivity * trace.P_amb
        )
        pfa_vals = pfa(np.full(trace.B, s_background), gamma)
        snr = A**2 / (2 * sigma_n**2)
        sig_phase = np.divide(np.sqrt(2.0) * sigma_n, np.sqrt(self.sensor.n_samples) * A,
                              out=np.full_like(A, np.inf), where=A > 0)
        sig_r = C_LIGHT * sig_phase / (4 * np.pi * np.array([self.sensor.f1, self.sensor.f2]))
        # Weak returns can make sigma infinite; squaring that value is part of
        # the model's zero-weight limit, so suppress NumPy's overflow warning.
        with np.errstate(over="ignore", invalid="ignore"):
            sig_r_sq = np.square(sig_r)
        g = np.divide(1.0, sig_r_sq, out=np.zeros_like(sig_r), where=np.isfinite(sig_r) & (sig_r > 0))
        sigma_R = np.sqrt(np.divide(1.0, g.sum(axis=1), out=np.full(trace.B, np.inf), where=g.sum(axis=1) > 0))
        rbar = _fused_range(z1, z2, sig_r[:, 0], sig_r[:, 1], self.sensor)

        denom = (2 * np.pi * (self.sensor.f2 - self.sensor.f1)) ** 2
        with np.errstate(over="ignore", invalid="ignore"):
            phase_variance = np.square(sig_phase)
        var_zeta = (self.sensor.f1**2 * phase_variance[:, 1] + self.sensor.f2**2 * phase_variance[:, 0]) / denom
        sz = np.sqrt(var_zeta)
        ks = np.array([-3, -2, -1, 1, 2, 3], float)
        probs = ndtr((ks[None, :] + .5) / sz[:, None]) - ndtr((ks[None, :] - .5) / sz[:, None])
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        delta = (g[:, 0] * self.sensor.L1 + g[:, 1] * self.sensor.L2) / np.maximum(g.sum(axis=1), np.finfo(float).tiny)
        shifts = ks[None, :] * delta[:, None]

        active = trace.hit & (trace.power > 0)
        hit_any = np.any(active, axis=1)
        weights = trace.w[np.arange(trace.B)[:, None], trace.sub_idx]
        eligible = active & (weights > self.model.w_min)
        rmin = np.min(np.where(eligible, trace.r_path, np.inf), axis=1)
        rmax = np.max(np.where(eligible, trace.r_path, -np.inf), axis=1)
        mixed = (rmax - rmin) > self.model.mixed_tol
        direct = trace.hit[:, :trace.K] & ~trace.bounce[:, :trace.K]
        sid = np.where(direct, trace.surface_id[:, :trace.K], -2)
        smooth = np.all(direct, axis=1) & np.all(sid == sid[:, :1], axis=1) & (pd > .5)
        return BeamPrediction(A, sigma_n, snr, pd, pfa_vals, rbar, sigma_R, probs, shifts,
                              hit_any, mixed, smooth)

    # E16–E19 for trusted direct-return arrays of shape (B, K).
    #
    # Internal callers with already validated geometry can avoid constructing a
    # full immutable optical record. This delegates all equations to predict;
    # every slot is direct and slot j has quadrature index j.
    def predict_direct_arrays(self, hit, r_path, power, w, surface_id, P_amb) -> BeamPrediction:
        B, K = hit.shape
        trace = SimpleNamespace(hit=hit, r_path=r_path, power=power, w=w,
                                surface_id=surface_id, P_amb=P_amb, B=B, K=K,
                                sub_idx=np.broadcast_to(np.arange(K), (B, K)),
                                bounce=np.zeros((B, K), bool))
        return self.predict(trace, None)

    # Return per-beam E20 log density/mass for observations.
    #
    # Parameters
    # scan : Scan
    #     Reported ranges [m]; detected observations must be finite and within W.
    # pred : BeamPrediction
    #     Matching per-beam predicted statistics.
    #
    # Returns
    # ndarray, shape (B,)
    #     Per-beam log likelihoods.
    #
    # Implements (E20).
    def log_likelihood_beams(self, scan: Scan, pred: BeamPrediction) -> np.ndarray:
        if scan.B != pred.B:
            raise ValueError("scan and prediction beam counts differ")
        return self.log_likelihood_arrays(scan.detected, scan.range, pred)

    # Array-only entry point for the identical per-beam likelihood (E20).
    def log_likelihood_arrays(self, detected, ranges, pred: BeamPrediction) -> np.ndarray:
        detected = np.asarray(detected, dtype=bool)
        ranges = np.asarray(ranges, dtype=float)
        if detected.shape != (pred.B,) or ranges.shape != (pred.B,):
            raise ValueError("observation and prediction beam counts differ")
        lo, hi = self.sensor.r_min, self.sensor.r_max
        if np.any(detected & (~np.isfinite(ranges) | (ranges < lo) | (ranges > hi))):
            raise ValueError("detected range must be finite and inside the reporting window")
        eps, width = self.model.eps_floor, hi - lo
        out = np.empty(pred.B)
        miss = ~detected
        nohit = ~pred.hit_any
        out[miss & nohit] = np.log((1 - eps) * (1 - pred.P_FA[miss & nohit] * width / self.sensor.Lc) + eps / 2)
        if np.any(miss & ~nohit):
            mask = miss & ~nohit
            miss_mass = (1 - pred.P_D[mask]) + pred.P_D[mask] * (1 - self._mi(pred, mask))
            out[mask] = np.log((1 - eps) * miss_mass + eps / 2)
        if np.any(detected & nohit):
            val = (1 - eps) * pred.P_FA[detected & nohit] / self.sensor.Lc + eps / (2 * width)
            out[detected & nohit] = np.log(val)
        if np.any(detected & ~nohit):
            mask = detected & ~nohit
            z = ranges[mask]
            g = self._density(z, pred.r_bar[mask], pred.sigma_R[mask], pred.out_prob[mask], pred.out_shift[mask])
            out[mask] = np.log((1 - eps) * pred.P_D[mask] * g + eps / (2 * width))
        return out

    def _density(self, z, mean, sigma, probs, shifts):
        v = np.zeros_like(z)
        pwrap = probs.sum(axis=1)
        core = np.maximum(0.0, 1.0 - pwrap)
        v += core * np.exp(-.5 * ((z - mean) / sigma)**2) / (np.sqrt(2 * np.pi) * sigma)
        v += np.sum(probs * np.exp(-.5 * ((z[:, None] - mean[:, None] - shifts) / sigma[:, None])**2) /
                    (np.sqrt(2 * np.pi) * sigma[:, None]), axis=1)
        return v

    def _mi(self, pred, mask):
        sigma = pred.sigma_R[mask]
        mean = pred.r_bar[mask]
        probs, shifts = pred.out_prob[mask], pred.out_shift[mask]
        if sigma.size == 0:
            return np.empty(0)
        def cdf(mu, sd):
            return ndtr((self.sensor.r_max - mu) / sd) - ndtr((self.sensor.r_min - mu) / sd)

        pwrap = probs.sum(axis=1)
        core = np.maximum(0, 1 - pwrap) * cdf(mean, sigma)
        return core + np.sum(probs * cdf(mean[:, None] + shifts, sigma[:, None]), axis=1)

    # Return total scan log likelihood, dimensionless [log probability].
    #
    # Parameters
    # scan : Scan
    #     Scan observations.
    # pred : BeamPrediction
    #     Predictions for the same scan.
    #
    # Returns
    # float
    #     Sum of per-beam log likelihoods.
    #
    # Implements (E21).
    def log_likelihood(self, scan: Scan, pred: BeamPrediction) -> float:
        return float(np.sum(self.log_likelihood_beams(scan, pred)))
