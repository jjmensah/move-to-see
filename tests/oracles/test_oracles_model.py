# ==========================================================
# tests/oracles/test_oracles_model.py
# ==========================================================
# Independent oracle tests for the analytic model, written from the spec alone.
#
# Values re-derived independently (docs/verification/vv-plan.md §2). Each test cites
# its oracle ID and equation.

import math

import numpy as np
import pytest
from scipy import integrate, stats

from lidar_sim.core.config import EnvConfig, ModelConfig, NoiseConfig, SensorConfig
from lidar_sim.core.constants import C_LIGHT
from lidar_sim.core.types import RadiometricScan, Scan
from lidar_sim.models.beam_model import BeamModel, expected_range
from lidar_sim.models.detection import cfar_gamma, pfa, rice_pd

SENSOR, NOISE = SensorConfig(), NoiseConfig()
GAMMA_OVER_S = math.sqrt(2.0 * math.log(1e4))          # O-14: √(2 ln 10⁴)


# ------------------------------------------
# single_return_trace
# ------------------------------------------
# B identical beams, K = 1, one Lambertian return at range r with given received power.
def single_return_trace(r: float, power: float, P_amb: float = 0.0, B: int = 1) -> RadiometricScan:
    base = RadiometricScan.empty(B=B, K=1, d=2, P_amb=P_amb)
    return RadiometricScan(**{**base.fields(),
        "hit": np.ones((B, 1), bool), "r_path": np.full((B, 1), r), "power": np.full((B, 1), power),
        "normal": np.tile([-1.0, 0.0], (B, 1, 1)), "cos_inc": np.ones((B, 1)),
        "object_id": np.zeros((B, 1), int), "material_id": np.zeros((B, 1), int),
        "surface_id": np.zeros((B, 1), int)})


# ------------------------------------------
# power_for_amplitude_ratio
# ------------------------------------------
# Received power P with 𝓡 m P / σ_n(P) = ratio (E11, no ambient): solve x² − ratio²κx − ratio²σ_th² = 0.
def power_for_amplitude_ratio(ratio: float) -> float:
    k, s2, a = NOISE.kappa_sh, NOISE.sigma_th**2, ratio**2
    x = (a * k + math.sqrt((a * k) ** 2 + 4 * a * s2)) / 2.0          # x = 𝓡 m P (m = 1, κ term uses 𝓡P)
    return x / (NOISE.responsivity * NOISE.mod_depth)


# ------------------------------------------
# test_O14_cfar_gamma_ratio_no_ambient
# ------------------------------------------
# O-14 / E15: γ = s_b √(−2 ln P_FA0), s_b = √(2/N) σ_th at P_amb = 0.
def test_O14_cfar_gamma_ratio_no_ambient():
    s_b = math.sqrt(2.0 / SENSOR.n_samples) * NOISE.sigma_th
    assert cfar_gamma(NOISE, 0.0, SENSOR.n_samples) / s_b == pytest.approx(GAMMA_OVER_S, rel=5e-7)
    assert GAMMA_OVER_S == pytest.approx(4.291932, rel=5e-7)


# ------------------------------------------
# test_O14_rice_values
# ------------------------------------------
# O-14 / E16: P_D(ν = γ) = 0.546801, P_D(ν/s = 6) = 0.964604 (s = 1); mpmath-checked.
def test_O14_rice_values():
    g = GAMMA_OVER_S
    assert float(rice_pd(np.array(g), np.array(1.0), g)) == pytest.approx(0.5468014426, rel=5e-7)
    assert float(rice_pd(np.array(6.0), np.array(1.0), g)) == pytest.approx(0.9646041880, rel=5e-7)
    assert float(pfa(np.array(1.0), g)) == pytest.approx(1e-4, rel=1e-9)


# ------------------------------------------
# test_O14_cfar_keeps_false_alarm_constant_under_ambient
# ------------------------------------------
# E15/E16 (ADR-0008): a no-signal beam has P_D = P_FA0 for any ambient level.
def test_O14_cfar_keeps_false_alarm_constant_under_ambient():
    for amb in (0.0, NOISE.P_amb_sun, 10 * NOISE.P_amb_sun):
        pred = BeamModel(SENSOR, NOISE).predict(RadiometricScan.empty(B=1, K=1, P_amb=amb), EnvConfig(P_amb=amb))
        assert pred.P_FA[0] == pytest.approx(NOISE.P_FA0, rel=1e-9)
        assert not pred.hit_any[0]


# ------------------------------------------
# test_O15_reference_sigma_R_is_12mm
# ------------------------------------------
# O-15 / E17 + ADR-0004: ρ = 0.9 at 4 m, clear air → fused σ_R = 12 mm; c/(4π√(f₁²+f₂²)) = 0.3374815.
def test_O15_reference_sigma_R_is_12mm():
    from sim.calibrate import received_power
    p = received_power(NOISE, 0.90, 4.0)
    pred = BeamModel(SENSOR, NOISE).predict(single_return_trace(4.0, p), EnvConfig())
    assert pred.sigma_R[0] == pytest.approx(0.012, rel=1e-6)
    assert C_LIGHT / (4 * math.pi * math.hypot(SENSOR.f1, SENSOR.f2)) == pytest.approx(0.3374815, rel=5e-7)
    assert pred.snr[0, 0] == pytest.approx(26.3643, rel=5e-6)


# ------------------------------------------
# test_O13_mixed_pixel_expected_range
# ------------------------------------------
# O-13 / E19: equal-power returns at 1 m and 3 m (two sub-rays, w = ½). Wrapped fine ranges
# 0.389944 m (f₁), 0.591201 m (f₂); coarse 2.000 m; both unwrap with k = 1; fused (weights ∝ f_j² A_j²)
# and folded r̄ = 3.452998 m — outside both surfaces (phasor cancellation).
def test_O13_mixed_pixel_expected_range():
    base = RadiometricScan.empty(B=1, K=2, d=2)
    p = 1e-7
    tr = RadiometricScan(**{**base.fields(), "w": np.array([[0.5, 0.5]]),
        "hit": np.ones((1, 2), bool), "r_path": np.array([[1.0, 3.0]]), "power": np.full((1, 2), p),
        "normal": np.tile([-1.0, 0.0], (1, 2, 1)), "cos_inc": np.ones((1, 2)),
        "object_id": np.array([[1, 0]]), "material_id": np.zeros((1, 2), int),
        "surface_id": np.array([[1, 0]])})
    assert expected_range(tr, SENSOR, NOISE)[0] == pytest.approx(3.452998183, rel=5e-9)
    pred = BeamModel(SENSOR, NOISE).predict(tr, EnvConfig())
    assert pred.mixed[0] and not pred.smooth[0]
    assert pred.r_bar[0] == pytest.approx(3.452998183, rel=5e-9)


# ------------------------------------------
# test_O13_single_surface_expected_range_exact
# ------------------------------------------
# E19: single surface ⇒ r̄ = r_path for r across the window and near L₁, L₂ multiples.
def test_O13_single_surface_expected_range_exact():
    for r in (0.02, 0.5, 2.8175, 2.8177, 3.2200, 3.2202, 5.6):
        assert expected_range(single_return_trace(r, 1e-7), SENSOR, NOISE)[0] == pytest.approx(r, abs=1e-9)


# ------------------------------------------
# test_O18_wrap_outlier_components
# ------------------------------------------
# O-18 / E18: A/σ_n = 2 at both f, N = 30 ⇒ σ²_φ = 1/60, σ_ζ = 0.2184156, p±1 = 0.01103348,
# Δ = c(f₁+f₂)/(2(f₁²+f₂²)) = 2.992140 m; ordering k = −3..−1, +1..+3.
def test_O18_wrap_outlier_components():
    tr = single_return_trace(3.0, power_for_amplitude_ratio(2.0))
    pred = BeamModel(SENSOR, NOISE).predict(tr, EnvConfig())
    f1, f2 = SENSOR.f1, SENSOR.f2
    sz = math.sqrt((f1**2 + f2**2) / 60.0) / (2 * math.pi * (f2 - f1))
    assert sz == pytest.approx(0.2184156, rel=5e-7)
    k = np.array([-3, -2, -1, 1, 2, 3])
    expect = stats.norm.cdf((k + 0.5) / sz) - stats.norm.cdf((k - 0.5) / sz)
    delta = C_LIGHT * (f1 + f2) / (2 * (f1**2 + f2**2))
    assert delta == pytest.approx(2.992140, rel=5e-7)
    np.testing.assert_allclose(pred.out_prob[0], expect, rtol=1e-6, atol=1e-15)
    assert pred.out_prob[0, 3] == pytest.approx(0.01103348, rel=5e-7)
    np.testing.assert_allclose(pred.out_shift[0], k * delta, rtol=1e-9)


# ------------------------------------------
# _total_mass
# ------------------------------------------
# Total probability of the observation model over both outcomes.
def _total_mass(model: BeamModel, tr: RadiometricScan) -> float:
    pred = model.predict(tr, EnvConfig(P_amb=tr.P_amb))

    def obs(det: bool, z: float) -> Scan:
        return Scan(detected=np.array([det]), range=np.array([z if det else np.nan]),
                    amplitude=np.zeros((1, 2)), range_raw=np.array([z]), fire_time=tr.fire_time, truth=tr)

    p_miss = math.exp(model.log_likelihood_beams(obs(False, 0.0), pred)[0])
    dens = lambda z: math.exp(model.log_likelihood_beams(obs(True, z), pred)[0])
    pts = [pred.r_bar[0] + s for s in [*pred.out_shift[0].tolist(), 0.0]
           if SENSOR.r_min < pred.r_bar[0] + s < SENSOR.r_max]
    mass, _ = integrate.quad(dens, SENSOR.r_min, SENSOR.r_max, points=sorted(pts) or None,
                             limit=500, epsabs=1e-12, epsrel=1e-11)
    return p_miss + mass


# ------------------------------------------
# test_O16_likelihood_normalized_hit_branch
# ------------------------------------------
# O-16 / E20: P(not detected) + ∫_W p(detected, z) dz = 1 for hit beams (incl. window edges).
@pytest.mark.parametrize("r, ratio", [(3.0, 7.25), (3.0, 2.0), (0.05, 1.4), (5.5, 4.0), (5.59, 1.1)])
def test_O16_likelihood_normalized_hit_branch(r, ratio):
    tr = single_return_trace(r, power_for_amplitude_ratio(ratio))
    assert _total_mass(BeamModel(SENSOR, NOISE, ModelConfig()), tr) == pytest.approx(1.0, abs=1e-9)


# ------------------------------------------
# test_O16_likelihood_normalized_no_hit_branch
# ------------------------------------------
# O-16 / E20 no-hit branch: uniform false-alarm density P_FA/L_c on W plus the ε floor.
@pytest.mark.parametrize("amb", [0.0, NOISE.P_amb_sun])
def test_O16_likelihood_normalized_no_hit_branch(amb):
    tr = RadiometricScan.empty(B=1, K=1, P_amb=amb)
    assert _total_mass(BeamModel(SENSOR, NOISE), tr) == pytest.approx(1.0, abs=1e-9)


# ------------------------------------------
# test_E20_detected_outside_window_is_invalid
# ------------------------------------------
# E20: p(detected, z) is defined only on W; out-of-window or non-finite detected z raises.
def test_E20_detected_outside_window_is_invalid():
    tr = single_return_trace(3.0, 1e-7)
    model = BeamModel(SENSOR, NOISE)
    pred = model.predict(tr, EnvConfig())
    for z in (SENSOR.r_max + 0.1, -0.1):
        bad = Scan(detected=np.array([True]), range=np.array([z]), amplitude=np.zeros((1, 2)),
                   range_raw=np.array([z]), fire_time=tr.fire_time, truth=tr)
        with pytest.raises(ValueError):
            model.log_likelihood_beams(bad, pred)
