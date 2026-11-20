# ==========================================================
# src/lidar_sim/physics/waveform.py
# ==========================================================
# Waveform-level AMCW measurement simulator — the "truth" path (spec §4.5, E10–E15).
#
# Implements (E10), (E11), (E12), (E13), (E14), (E15) on a :class:`RadiometricScan`. All signals are in
# detector-current units [A] (ADR-0004); multi-return phasors add with amplitude ∝ optical power
# (ADR-0002). Dimension-agnostic: only the ``(B, J)`` return arrays are used, never directions or angles.
#
# Independent of the analytic model (ADR-0005): nothing here is shared with ``lidar_sim.models``.

from __future__ import annotations

import numpy as np

from lidar_sim.core.config import NoiseConfig, SensorConfig
from lidar_sim.core.constants import C_LIGHT
from lidar_sim.core.types import RadiometricScan, Scan


# ------------------------------------------
# _total_power
# ------------------------------------------
# P̄ = Σ_j P_j over the return axis [W], shape (B,). Misses carry power 0.
def _total_power(trace: RadiometricScan) -> np.ndarray:
    return trace.power.sum(axis=1)


# ------------------------------------------
# phasors
# ------------------------------------------
# Noise-free envelope phasor per beam.
#
# Implements (E10): ``Z = 𝓡 m Σ_j P_j exp(i 4π f r_path,j / c)``.
#
# Parameters
# trace : RadiometricScan
#     Returns on the ``(B, J)`` axis; misses (``hit=False``, ``r_path=NaN``, ``power=0``) contribute 0.
# f : float
#     Modulation frequency [Hz].
# noise : NoiseConfig
#     Supplies responsivity 𝓡 [A/W] and modulation depth m.
#
# Returns
# (B,) complex [A]
def phasors(trace: RadiometricScan, f: float, noise: NoiseConfig) -> np.ndarray:
    r = np.where(trace.hit, trace.r_path, 0.0)          # NaN of misses must not propagate
    p = np.where(trace.hit, trace.power, 0.0)
    phase = (4.0 * np.pi * f / C_LIGHT) * r
    return noise.responsivity * noise.mod_depth * (p * np.exp(1j * phase)).sum(axis=1)


# ------------------------------------------
# dc_level
# ------------------------------------------
# Sample DC level ``𝓡 (P̄ + P_amb)`` [A], shape (B,) (E11). Ambient comes only from ``trace.P_amb``.
def dc_level(trace: RadiometricScan, noise: NoiseConfig) -> np.ndarray:
    return noise.responsivity * (_total_power(trace) + trace.P_amb)


# ------------------------------------------
# noise_std
# ------------------------------------------
# Per-sample noise std ``σ_n = sqrt(σ_th² + κ_sh 𝓡 (P̄ + P_amb))`` [A], shape (B,) (E11).
def noise_std(trace: RadiometricScan, noise: NoiseConfig) -> np.ndarray:
    return np.sqrt(noise.sigma_th**2 + noise.kappa_sh * noise.responsivity * (_total_power(trace) + trace.P_amb))


# ------------------------------------------
# _wrap0
# ------------------------------------------
# ``wrap₀(φ) ∈ [0, 2π)`` (spec §2). Float ``mod`` can round tiny negatives to exactly 2π; map those to 0.
def _wrap0(phi: np.ndarray) -> np.ndarray:
    two_pi = 2.0 * np.pi
    out = np.mod(phi, two_pi)
    return np.where(out >= two_pi, 0.0, out)


# ------------------------------------------
# _sample_phases
# ------------------------------------------
# Equivalent-time sample phases ``θ_n = 2πn/N``, shape (N,) (E11).
def _sample_phases(n: int) -> np.ndarray:
    return 2.0 * np.pi * np.arange(n) / n


# ------------------------------------------
# sample
# ------------------------------------------
# Draw one equivalent-time sample set per beam.
#
# Implements (E11): ``s_n = dc + Re{Z e^{−iθ_n}} + ε_n``, ``θ_n = 2πn/N``, ``ε_n ~ N(0, σ_n²)`` i.i.d.
#
# Parameters
# Z : (B,) complex [A]
#     Envelope phasors (E10).
# dc : (B,) [A]
#     DC level ``𝓡 (P̄ + P_amb)``.
# sigma_n : (B,) [A]
#     Per-sample noise std; 0 gives noise-free samples (the normal draw is still consumed so the
#     stream position does not depend on the noise level).
# N : int
#     Number of samples, ``N >= 3`` (orthogonal regressors, E12).
# rng : numpy.random.Generator
#     Exactly one ``standard_normal((B, N))`` draw is consumed.
#
# Returns
# (B, N) [A]
def sample(Z: np.ndarray, dc: np.ndarray, sigma_n: np.ndarray, N: int, rng: np.random.Generator) -> np.ndarray:
    if N < 3:
        raise ValueError("N must be >= 3 (E12 needs orthogonal regressors)")
    Z = np.asarray(Z, complex)
    theta = _sample_phases(N)
    eps = rng.standard_normal((Z.shape[0], N))
    clean = np.asarray(dc, float)[:, None] + np.real(Z[:, None] * np.exp(-1j * theta))
    return clean + np.asarray(sigma_n, float)[:, None] * eps


# ------------------------------------------
# estimate
# ------------------------------------------
# DFT phase/amplitude estimator at the modulation bin.
#
# Implements (E12): ``X = Σ_n s_n e^{+iθ_n}``, ``φ̂ = wrap₀(arg X) ∈ [0, 2π)``, ``Â = 2|X|/N``.
# ``E[X] = (N/2) Z`` (ADR-0007), so ``φ̂`` estimates ``arg Z`` (not its conjugate).
#
# Parameters
# samples : (B, N) [A]
#
# Returns
# phi_hat : (B,) [rad] in [0, 2π)
# A_hat : (B,) [A]
def estimate(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    samples = np.asarray(samples, float)
    n = samples.shape[-1]
    X = samples @ np.exp(1j * _sample_phases(n))
    return _wrap0(np.angle(X)), 2.0 * np.abs(X) / n


# ------------------------------------------
# fine_range
# ------------------------------------------
# Wrapped fine range ``R = c φ̂ / (4π f) ∈ [0, L)``, ``L = c/(2f)`` [m] (E13).
def fine_range(phi_hat: np.ndarray, f: float) -> np.ndarray:
    L = C_LIGHT / (2.0 * f)
    R = C_LIGHT * np.asarray(phi_hat, float) / (4.0 * np.pi * f)
    return np.where(R >= L, R - L, R)                     # φ̂ just below 2π may round to L


# ------------------------------------------
# _range_std
# ------------------------------------------
# Per-frequency range std from the phase CRLB, evaluated with the *estimated* amplitude.
#
# (E17) ``σ_φ² = 2σ_n² / (N Â²)``, ``σ_R = c σ_φ / (4π f)`` [m], shape (B,). ``Â = 0`` ⇒ ``σ_R = inf``
# (no information). Used for the E14 fusion weights (the sensor knows its noise floor σ_n).
def _range_std(A_hat: np.ndarray, sigma_n: np.ndarray, N: int, f: float) -> np.ndarray:
    A = np.asarray(A_hat, float)
    pos = A > 0
    s_phi = np.where(pos, np.sqrt(2.0 / N) * np.asarray(sigma_n, float) / np.where(pos, A, 1.0), np.inf)
    return C_LIGHT * s_phi / (4.0 * np.pi * f)


# ------------------------------------------
# _weight1
# ------------------------------------------
# Inverse-variance weight of frequency 1, ``g₁/(g₁+g₂)`` with ``g_j = σ_{R,j}^{−2}`` (E14).
#
# Degenerate cases: both σ infinite (no information, e.g. Â = 0) or both zero ⇒ equal weights;
# one infinite ⇒ all weight on the other; one zero ⇒ all weight on it. NaN σ is treated as infinite.
def _weight1(sig_R1: np.ndarray, sig_R2: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        v1, v2 = np.square(np.asarray(sig_R1, float)), np.square(np.asarray(sig_R2, float))
    v1, v2 = np.where(np.isnan(v1), np.inf, v1), np.where(np.isnan(v2), np.inf, v2)
    inf1, inf2 = np.isinf(v1), np.isinf(v2)
    den = v1 + v2
    ok = ~inf1 & ~inf2 & (den > 0)
    w1 = np.where(ok, v2 / np.where(ok, den, 1.0), 0.5)
    w1 = np.where(inf1 & ~inf2, 0.0, w1)
    return np.where(inf2 & ~inf1, 1.0, w1)


# ------------------------------------------
# unwrap_fuse
# ------------------------------------------
# Dual-frequency unwrap, inverse-variance fusion and fold.
#
# Implements (E14)::
#
#     R_c = c · wrap₀(φ̂₂ − φ̂₁) / (4π Δf) ∈ [0, L_c)
#     k_j = round((R_c − R_j) / L_j),   R'_j = R_j + k_j L_j
#     R̂  = mod((R'₁/σ²_{R,1} + R'₂/σ²_{R,2}) / (1/σ²_{R,1} + 1/σ²_{R,2}), L_c)
#
# The final fold is required because ``L_c = 7 L₁ = 8 L₂`` (ADR-0006): a target near 0 m whose coarse
# estimate wraps to ≈ L_c is folded back to ≈ 0 instead of being reported at ≈ 22 m.
#
# Parameters
# R1, R2 : (B,) [m]
#     Wrapped fine ranges (E13) at ``sensor.f1`` and ``sensor.f2``.
# phi1, phi2 : (B,) [rad]
#     Estimated phases φ̂₁, φ̂₂ (E12).
# sig_R1, sig_R2 : (B,) [m]
#     Per-frequency range std (E17) for the fusion weights; see :func:`_weight1` for 0/inf/NaN.
# sensor : SensorConfig
#     Supplies f1, f2 (hence L₁, L₂, L_c).
#
# Returns
# (B,) [m] in [0, L_c)
def unwrap_fuse(R1: np.ndarray, R2: np.ndarray, phi1: np.ndarray, phi2: np.ndarray, sig_R1: np.ndarray,
                sig_R2: np.ndarray, sensor: SensorConfig) -> np.ndarray:
    f1, f2 = sensor.f1, sensor.f2
    L1, L2, Lc = sensor.L1, sensor.L2, sensor.Lc
    R1, R2 = np.asarray(R1, float), np.asarray(R2, float)
    Rc = C_LIGHT * _wrap0(np.asarray(phi2, float) - np.asarray(phi1, float)) / (4.0 * np.pi * (f2 - f1))
    R1u = R1 + np.round((Rc - R1) / L1) * L1
    R2u = R2 + np.round((Rc - R2) / L2) * L2
    w1 = _weight1(sig_R1, sig_R2)
    out = np.mod(w1 * R1u + (1.0 - w1) * R2u, Lc)
    return np.where(out >= Lc, 0.0, out)


# ------------------------------------------
# cfar_threshold
# ------------------------------------------
# CFAR amplitude threshold (ADR-0008).
#
# Implements (E15): ``γ = s_b √(−2 ln P_FA,0)``, ``s_b = √(2/N) · √(σ_th² + κ_sh 𝓡 P_amb)``. For a
# no-return beam ``Â₁`` is Rayleigh with scale ``s_b``, so ``P(Â₁ ≥ γ) = P_FA,0`` for every ambient level.
#
# Returns
# float [A]
def cfar_threshold(noise: NoiseConfig, P_amb: float, N: int) -> float:
    s_b = np.sqrt(2.0 / N) * np.sqrt(noise.sigma_th**2 + noise.kappa_sh * noise.responsivity * P_amb)
    return float(s_b * np.sqrt(-2.0 * np.log(noise.P_FA0)))


# ------------------------------------------
# measure
# ------------------------------------------
# Simulate what the sensor reports for a traced scan (truth path).
#
# Implements (E10)–(E15): phasors per frequency (E10); independent noisy sample sets (E11), drawn from
# ``rng`` in the order f1 samples ``(B, N)`` then f2 samples ``(B, N)``; DFT estimates (E12); fine ranges
# (E13); unwrap + inverse-variance fusion with weights from E17 at the estimated ``Â_j`` and the known
# σ_n, folded mod L_c (E14); CFAR detection ``Â₁ ≥ γ`` and window gate ``R̂ ∈ [r_min, r_max]`` (E15).
# Optional ``sensor.range_walk``: ``R̂ += b₀ ln(P_ref / P̄)`` where ``P̄ > 0``, applied before the gate.
#
# Ambient light enters only via ``trace.P_amb``.
#
# Parameters
# trace : RadiometricScan
# sensor : SensorConfig
# noise : NoiseConfig
# rng : numpy.random.Generator
#
# Returns
# Scan
#     ``range_raw`` is R̂ before the gate (including range walk if enabled); ``range`` is R̂ where
#     detected, NaN elsewhere; ``amplitude[:, j] = Â_j``; ``truth`` is ``trace``.
def measure(trace: RadiometricScan, sensor: SensorConfig, noise: NoiseConfig, rng: np.random.Generator) -> Scan:
    N = sensor.n_samples
    dc = dc_level(trace, noise)
    sig = noise_std(trace, noise)
    est = []
    for f in (sensor.f1, sensor.f2):                      # draw order: f1 then f2, fixed for reproducibility
        phi, A = estimate(sample(phasors(trace, f, noise), dc, sig, N, rng))
        est.append((phi, A, fine_range(phi, f), _range_std(A, sig, N, f)))
    (phi1, A1, R1, s1), (phi2, A2, R2, s2) = est
    R_hat = unwrap_fuse(R1, R2, phi1, phi2, s1, s2, sensor)
    if sensor.range_walk:
        p_bar = _total_power(trace)
        pos = p_bar > 0
        R_hat = R_hat + np.where(pos, sensor.b0 * np.log(sensor.P_ref / np.where(pos, p_bar, 1.0)), 0.0)
    gamma = cfar_threshold(noise, trace.P_amb, N)
    detected = (A1 >= gamma) & (R_hat >= sensor.r_min) & (R_hat <= sensor.r_max)
    return Scan(detected=detected, range=np.where(detected, R_hat, np.nan), amplitude=np.stack([A1, A2], axis=1),
                range_raw=R_hat, fire_time=trace.fire_time, truth=trace)
