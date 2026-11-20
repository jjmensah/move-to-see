# ==========================================================
# sim/calibrate.py
# ==========================================================
# ADR-0004 noise calibration.
#
# Analytic solve: given nominal gains, find ``sigma_th``, ``kappa_sh`` and ``P_amb_sun`` so that
#   1. ρ=0.90 at r=4 m, normal incidence, clear air (V=20 km), no ambient → fused σ_R = 12 mm (E17);
#   2. at that point shot-noise variance equals thermal variance (E11);
#   3. ``P_amb_sun`` halves the P_D = 0.5 range of a ρ=0.90 target (E16).
#
# This module re-implements E7, E8, E11, E16, E17 in closed form on purpose (single-surface,
# normal-incidence case) so the frozen constants do not depend on Phase-1 code. Phase 2 adds a Monte
# Carlo ``--verify`` through the full simulator.
#
# Usage: ``python -m sim.calibrate`` prints the constants to freeze in ``core/config.py``.

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from scipy import optimize, stats

from lidar_sim.core import constants as K
from lidar_sim.core.config import NoiseConfig, SensorConfig

REF_RHO = 0.90
REF_RANGE = 4.0
REF_VIS = 20000.0
TARGET_SIGMA_R = 0.012


# ------------------------------------------
# _kim_q
# ------------------------------------------
# Kim's size-distribution exponent q for a visibility in km (E8).
def _kim_q(v_km: float) -> float:
    if v_km > 50:
        return 1.6
    if v_km > 6:
        return 1.3
    if v_km > 1:
        return 0.16 * v_km + 0.34
    if v_km > 0.5:
        return v_km - 0.5
    return 0.0


# ------------------------------------------
# received_power
# ------------------------------------------
# E7 + E8 for an extended Lambertian target at normal incidence [W].
def received_power(noise: NoiseConfig, rho: float, r: float, vis_m: float = REF_VIS,
                   wavelength: float = K.WAVELENGTH) -> float:
    sigma = K.KOSCHMIEDER / vis_m * (wavelength / 550e-9) ** (-_kim_q(vis_m / 1000.0))
    return noise.P0 * (rho / math.pi) * (noise.A_rx / r**2) * math.exp(-2 * sigma * r) * noise.eta


# ------------------------------------------
# sigma_n
# ------------------------------------------
# E11 per-sample noise std [A].
def sigma_n(noise: NoiseConfig, p_bar: float, p_amb: float) -> float:
    return math.sqrt(noise.sigma_th**2 + noise.kappa_sh * noise.responsivity * (p_bar + p_amb))


# ------------------------------------------
# fused_sigma_r
# ------------------------------------------
# E17 fused range std for a single return of power ``p`` [m].
def fused_sigma_r(noise: NoiseConfig, sensor: SensorConfig, p: float, p_amb: float = 0.0) -> float:
    a = noise.responsivity * noise.mod_depth * p
    s_phi2 = 2 * sigma_n(noise, p, p_amb) ** 2 / (sensor.n_samples * a**2)
    return K.C_LIGHT * math.sqrt(s_phi2) / (4 * math.pi * math.hypot(sensor.f1, sensor.f2))


# ------------------------------------------
# detection_probability
# ------------------------------------------
# E15 CFAR threshold (ADR-0008) + E16 Rice detection probability.
def detection_probability(noise: NoiseConfig, sensor: SensorConfig, p: float, p_amb: float) -> float:
    q = math.sqrt(2.0 / sensor.n_samples)
    gamma = sigma_n(noise, 0.0, p_amb) * q * math.sqrt(-2 * math.log(noise.P_FA0))
    s = sigma_n(noise, p, p_amb) * q
    a = noise.responsivity * noise.mod_depth * p
    return float(stats.rice.sf(gamma, b=a / s, scale=s))


# ------------------------------------------
# r50
# ------------------------------------------
# Range at which P_D = 0.5 for a normal-incidence target (bisection on log r).
def r50(noise: NoiseConfig, sensor: SensorConfig, rho: float, p_amb: float) -> float:
    lo, hi = math.log(0.01), math.log(1000.0)
    f = lambda lr: detection_probability(noise, sensor, received_power(noise, rho, math.exp(lr)), p_amb) - 0.5
    if f(lo) <= 0.0:
        return 0.0                         # not detectable even at 1 cm
    return math.exp(optimize.brentq(f, lo, hi, xtol=1e-12))


# ------------------------------------------
# Calibration
# ------------------------------------------
# The solved noise constants and the ranges they imply.
@dataclass(frozen=True)
class Calibration:
    sigma_th: float
    kappa_sh: float
    P_amb_sun: float
    r50_white_clear: float
    r50_white_sun: float
    r50_black_clear: float


# ------------------------------------------
# solve
# ------------------------------------------
# Solves the three calibration targets for the noise constants and the sunlight level.
def solve(noise: NoiseConfig | None = None, sensor: SensorConfig | None = None) -> Calibration:
    noise = noise or NoiseConfig()
    sensor = sensor or SensorConfig()
    p = received_power(noise, REF_RHO, REF_RANGE)
    a = noise.responsivity * noise.mod_depth * p
    s_phi = TARGET_SIGMA_R * 4 * math.pi * math.hypot(sensor.f1, sensor.f2) / K.C_LIGHT
    sn2 = sensor.n_samples * a**2 * s_phi**2 / 2.0
    cal = replace(noise, sigma_th=math.sqrt(sn2 / 2.0), kappa_sh=sn2 / (2.0 * noise.responsivity * p))
    r_clear = r50(cal, sensor, REF_RHO, 0.0)
    g = lambda lp: r50(cal, sensor, REF_RHO, math.exp(lp)) - r_clear / 2.0
    p_sun = math.exp(optimize.brentq(g, math.log(p * 1e-4), math.log(p * 1e4), xtol=1e-12))
    return Calibration(cal.sigma_th, cal.kappa_sh, p_sun, r_clear,
                       r50(cal, sensor, REF_RHO, p_sun), r50(cal, sensor, 0.05, 0.0))


# ------------------------------------------
# verify
# ------------------------------------------
# Monte Carlo confirmation of the frozen constants through the full waveform simulator (Phase 2, Task 2.2).
#
# 1. ρ = 0.9 at 4 m, clear air: std of the fused range (k = 0 core) vs the 12 mm target (exact χ² 99 % CI).
# 2. At the analytic r50 (P_D = 0.5) of ρ = 0.9 in clear air and in bright sun: simulated rate of
#    Â₁ ≥ γ vs 0.5 (99 % Wilson CI). Uses ``sim.experiments._common.probe_trace`` (E7/E8 facing probe).
def verify(n: int = 20000, seed: int = 4240) -> dict:
    import numpy as np
    from scipy import stats as st

    from lidar_sim.core.config import EnvConfig
    from lidar_sim.physics import waveform
    from sim.experiments._common import probe_trace, wilson

    noise, sensor = NoiseConfig(), SensorConfig()
    rng = np.random.default_rng(seed)
    out = {}
    sc = waveform.measure(probe_trace(np.full(n, REF_RANGE), REF_RHO, EnvConfig(), noise), sensor, noise, rng)
    e = sc.range_raw - REF_RANGE
    core = e[np.abs(e) < 1.5]
    var = float(core.var(ddof=1))
    k = core.size - 1
    ci = (k * var / st.chi2.ppf(0.995, k), k * var / st.chi2.ppf(0.005, k))
    out["sigma_R_ref"] = {"n": int(core.size), "std_m": math.sqrt(var), "ci99_m": [math.sqrt(ci[0]), math.sqrt(ci[1])],
                          "target_m": TARGET_SIGMA_R, "pass": bool(ci[0] <= TARGET_SIGMA_R**2 <= ci[1])}
    for name, amb in (("clear", 0.0), ("sun", noise.P_amb_sun)):
        r = r50(noise, sensor, REF_RHO, amb)
        tr = probe_trace(np.full(n, r), REF_RHO, EnvConfig(P_amb=amb), noise)
        sc = waveform.measure(tr, sensor, noise, rng)
        x = int(np.count_nonzero(sc.amplitude[:, 0] >= waveform.cfar_threshold(noise, amb, sensor.n_samples)))
        lo, hi = wilson(x, n, 0.99)
        out[f"P_D_at_r50_{name}"] = {"r50_m": r, "rate": x / n, "wilson99": [float(lo), float(hi)],
                                     "pass": bool(lo <= 0.5 <= hi)}
    out["pass"] = bool(all(v["pass"] for v in out.values() if isinstance(v, dict)))
    return out


if __name__ == "__main__":
    import json
    import sys

    if "--verify" in sys.argv:
        print(json.dumps(verify(), indent=2))
        raise SystemExit(0)
    c = solve()
    for k, v in c.__dict__.items():
        print(f"{k:16s} = {v!r}")
    print("check sigma_R(ref) =", fused_sigma_r(replace(NoiseConfig(), sigma_th=c.sigma_th,
          kappa_sh=c.kappa_sh), SensorConfig(), received_power(NoiseConfig(), REF_RHO, REF_RANGE)))
