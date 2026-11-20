# ==========================================================
# tests/unit/test_waveform.py
# ==========================================================
# Unit tests for physics/waveform.py primitives (E10–E15), noise-free and closed-form checks.

import numpy as np
import pytest

from lidar_sim.core.config import NoiseConfig, SensorConfig
from lidar_sim.core.constants import C_LIGHT
from lidar_sim.core.types import RadiometricScan
from lidar_sim.physics import waveform as wf

pytestmark = pytest.mark.filterwarnings("error")

NOISE = NoiseConfig()
SENSOR = SensorConfig()
F1, F2, N = SENSOR.f1, SENSOR.f2, SENSOR.n_samples


# ------------------------------------------
# make_trace
# ------------------------------------------
# Hand-built (B, J) RadiometricScan; NaN entries of ``r`` are misses (J = K, no specular).
def make_trace(r, P, P_amb=0.0, d=2):
    r = np.atleast_2d(np.asarray(r, float))
    B, J = r.shape
    hit = np.isfinite(r)
    P = np.where(hit, np.broadcast_to(np.asarray(P, float), r.shape), 0.0)
    base = RadiometricScan.empty(B, J, d=d, P_amb=P_amb)
    normal = np.zeros((B, J, d))
    normal[..., 0] = -1.0
    ids = np.where(hit, 0, -1)
    return RadiometricScan(**{**base.fields(), "hit": hit, "r_path": r, "power": P, "normal": normal,
                              "cos_inc": np.where(hit, 1.0, 0.0), "object_id": ids, "material_id": ids,
                              "surface_id": ids})


# ------------------------------------------
# noise_free_samples
# ------------------------------------------
# Noise-free samples of each beam for a phasor and a DC level.
def noise_free_samples(Z, dc, n=N):
    theta = 2 * np.pi * np.arange(n) / n
    return dc[:, None] + np.real(Z[:, None] * np.exp(-1j * theta))


# ------------------------------------------
# test_phasor_single_return_r1m
# ------------------------------------------
# Phasor of a single return at 1 m.
def test_phasor_single_return_r1m():
    P = 1e-6
    Z = wf.phasors(make_trace([[1.0]], P), F1, NOISE)
    assert Z.shape == (1,) and Z.dtype == complex
    assert abs(Z[0]) == pytest.approx(NOISE.responsivity * NOISE.mod_depth * P, rel=1e-14)
    assert np.angle(Z[0]) == pytest.approx(4 * np.pi * F1 / C_LIGHT, abs=1e-12)
    assert np.angle(Z[0]) == pytest.approx(1.951232, abs=1e-6)


# ------------------------------------------
# test_phasor_sums_returns_and_ignores_misses
# ------------------------------------------
# Phasors of several returns add, and misses add nothing.
def test_phasor_sums_returns_and_ignores_misses():
    r = np.array([[1.0, 3.0, np.nan]])
    P = np.array([[1e-6, 2e-6, 0.0]])
    Z = wf.phasors(make_trace(r, P), F2, NOISE)
    k = 4 * np.pi * F2 / C_LIGHT
    expect = NOISE.responsivity * NOISE.mod_depth * (1e-6 * np.exp(1j * k) + 2e-6 * np.exp(3j * k))
    assert np.isfinite(Z).all()
    assert Z[0] == pytest.approx(expect, rel=1e-13)


# ------------------------------------------
# test_phasor_all_miss_is_zero
# ------------------------------------------
# A beam that hits nothing has a zero phasor.
def test_phasor_all_miss_is_zero():
    Z = wf.phasors(RadiometricScan.empty(4, 3), F1, NOISE)
    assert np.array_equal(Z, np.zeros(4, complex))


# ------------------------------------------
# test_dc_level_and_noise_std
# ------------------------------------------
# DC level and noise standard deviation.
def test_dc_level_and_noise_std():
    tr = make_trace([[1.0, 2.0], [np.nan, np.nan]], [[1e-6, 3e-6], [0, 0]], P_amb=5e-7)
    Pbar = np.array([4e-6, 0.0])
    dc = wf.dc_level(tr, NOISE)
    np.testing.assert_allclose(dc, NOISE.responsivity * (Pbar + 5e-7), rtol=1e-15)
    sn = wf.noise_std(tr, NOISE)
    np.testing.assert_allclose(sn, np.sqrt(NOISE.sigma_th**2 + NOISE.kappa_sh * NOISE.responsivity
                                           * (Pbar + 5e-7)), rtol=1e-15)
    assert dc.shape == sn.shape == (2,)


# ------------------------------------------
# test_noise_std_no_signal_no_ambient_is_thermal
# ------------------------------------------
# With no signal and no ambient light, only thermal noise remains.
def test_noise_std_no_signal_no_ambient_is_thermal():
    sn = wf.noise_std(RadiometricScan.empty(3, 1), NOISE)
    np.testing.assert_array_equal(sn, np.full(3, NOISE.sigma_th))


# ------------------------------------------
# test_sample_noise_free_matches_model
# ------------------------------------------
# Noise-free samples match the sample model.
def test_sample_noise_free_matches_model(rng):
    Z = np.array([1e-5 * np.exp(0.7j), 0j, -3e-6 + 2e-6j])
    dc = np.array([4e-5, 1e-6, 0.0])
    s = wf.sample(Z, dc, np.zeros(3), N, rng)
    assert s.shape == (3, N)
    np.testing.assert_allclose(s, noise_free_samples(Z, dc), rtol=0, atol=1e-20)


# ------------------------------------------
# test_sample_noise_scale_per_beam
# ------------------------------------------
# Each beam gets its own noise level.
def test_sample_noise_scale_per_beam(rng):
    B = 4000
    sig = np.where(np.arange(B) % 2 == 0, 1e-7, 3e-7)
    s = wf.sample(np.zeros(B, complex), np.full(B, 2e-6), sig, 16, rng)
    eps = s - 2e-6
    assert eps[0::2].std() == pytest.approx(1e-7, rel=0.02)
    assert eps[1::2].std() == pytest.approx(3e-7, rel=0.02)
    assert abs(eps.mean()) < 1e-8


# ------------------------------------------
# test_sample_rejects_too_few_samples
# ------------------------------------------
# Fewer than three samples are rejected.
def test_sample_rejects_too_few_samples(rng):
    with pytest.raises(ValueError):
        wf.sample(np.zeros(1, complex), np.zeros(1), np.zeros(1), 2, rng)


# ------------------------------------------
# test_estimate_r1m_noise_free_O10
# ------------------------------------------
# O-10: r = 1 m ⇒ φ̂₁ = 1.951232 rad, Â = 𝓡 m P, X = (N/2) Z (ADR-0007 sign convention).
def test_estimate_r1m_noise_free_O10():
    P = 2e-6
    tr = make_trace([[1.0]], P, P_amb=1e-6)
    Z = wf.phasors(tr, F1, NOISE)
    s = noise_free_samples(Z, wf.dc_level(tr, NOISE))
    phi, A = wf.estimate(s)
    assert phi[0] == pytest.approx(1.951232, abs=1e-6)
    assert phi[0] == pytest.approx(np.angle(Z[0]), abs=1e-12)
    assert A[0] == pytest.approx(NOISE.responsivity * NOISE.mod_depth * P, rel=1e-10)
    theta = 2 * np.pi * np.arange(N) / N
    X = (s * np.exp(1j * theta)).sum(axis=1)
    assert X[0] == pytest.approx(N / 2 * Z[0], rel=1e-10)


# ------------------------------------------
# test_estimate_phase_range_is_wrap0
# ------------------------------------------
# Estimated phases lie in [0, 2π).
@pytest.mark.parametrize("phi_true", [0.0, 1e-3, np.pi - 1e-9, np.pi, 4.0, 2 * np.pi - 1e-6])
def test_estimate_phase_range_is_wrap0(phi_true):
    Z = np.array([3e-6 * np.exp(1j * phi_true)])
    phi, A = wf.estimate(noise_free_samples(Z, np.array([1e-5])))
    assert 0.0 <= phi[0] < 2 * np.pi
    d = (phi[0] - phi_true + np.pi) % (2 * np.pi) - np.pi
    assert abs(d) < 1e-9
    assert A[0] == pytest.approx(3e-6, rel=1e-9)


# ------------------------------------------
# test_estimate_tiny_negative_angle_wraps_below_2pi
# ------------------------------------------
# arg X = −1e−17 must map into [0, 2π) (float mod would give exactly 2π).
def test_estimate_tiny_negative_angle_wraps_below_2pi():
    s = np.zeros((1, 4))
    s[0, 0] = 1.0
    s[0, 1] = -1e-17                # X = 1 + (−1e−17)(i) ⇒ arg X ≈ −1e−17
    phi, _ = wf.estimate(s)
    assert 0.0 <= phi[0] < 2 * np.pi


# ------------------------------------------
# test_estimate_zero_samples_is_finite
# ------------------------------------------
# All-zero samples give a finite estimate.
def test_estimate_zero_samples_is_finite():
    phi, A = wf.estimate(np.zeros((2, N)))
    assert np.isfinite(phi).all() and np.array_equal(A, np.zeros(2))
    assert ((phi >= 0) & (phi < 2 * np.pi)).all()


# ------------------------------------------
# noise_free_range
# ------------------------------------------
# Run E10–E14 with σ = 0 samples; fusion weights from E17 with the true σ_n (as in ``measure``).
def noise_free_range(r, P, noise=NOISE, sensor=SENSOR, return_parts=False):
    tr = make_trace(r, P)
    dc, sn = wf.dc_level(tr, noise), wf.noise_std(tr, noise)
    out = []
    for f in (sensor.f1, sensor.f2):
        phi, A = wf.estimate(noise_free_samples(wf.phasors(tr, f, noise), dc, sensor.n_samples))
        out.append((phi, A, wf.fine_range(phi, f), wf._range_std(A, sn, sensor.n_samples, f)))
    (p1, _, R1, s1), (p2, _, R2, s2) = out
    Rhat = wf.unwrap_fuse(R1, R2, p1, p2, s1, s2, sensor)
    return (Rhat, R1, R2) if return_parts else Rhat


# ------------------------------------------
# test_unambiguous_intervals
# ------------------------------------------
# Unambiguous intervals of both frequencies and the coarse one.
def test_unambiguous_intervals():
    assert SENSOR.L1 == pytest.approx(3.220112, abs=1e-6)
    assert SENSOR.L2 == pytest.approx(2.817598, abs=1e-6)
    assert SENSOR.Lc == pytest.approx(22.540786, abs=1e-6)
    assert SENSOR.Lc / SENSOR.L1 == pytest.approx(7.0, abs=1e-12)
    assert SENSOR.Lc / SENSOR.L2 == pytest.approx(8.0, abs=1e-12)


# ------------------------------------------
# test_fine_range_O11
# ------------------------------------------
# O-11: r = 1 m and r = 1 + L₁ give identical R₁; R ∈ [0, L).
def test_fine_range_O11():
    phis = [wf.estimate(noise_free_samples(wf.phasors(make_trace([[r]], 1e-6), F1, NOISE), np.zeros(1)))[0]
            for r in (1.0, 1.0 + SENSOR.L1)]
    R = [wf.fine_range(p, F1) for p in phis]
    assert R[0][0] == pytest.approx(1.0, abs=1e-12)
    assert R[1][0] == pytest.approx(R[0][0], abs=1e-12)


# ------------------------------------------
# test_fine_range_formula_and_interval
# ------------------------------------------
# Fine range formula and its interval.
def test_fine_range_formula_and_interval():
    phi = np.array([0.0, 1.0, np.nextafter(2 * np.pi, 0)])
    R = wf.fine_range(phi, F2)
    np.testing.assert_allclose(R[:2], C_LIGHT * phi[:2] / (4 * np.pi * F2), rtol=1e-15)
    assert ((R >= 0) & (R < SENSOR.L2)).all()


# ------------------------------------------
# test_unwrap_fuse_noise_free_O12
# ------------------------------------------
# Without noise, unwrapping and fusion recover the range (O-12).
@pytest.mark.parametrize("r", [0.5, 2.9, 3.3, 5.5, 22.0])
def test_unwrap_fuse_noise_free_O12(r):
    assert noise_free_range([[r]], 1e-6)[0] == pytest.approx(r, abs=1e-9)


# ------------------------------------------
# test_fold_mod_Lc_O19
# ------------------------------------------
# O-19: phases of r ≡ −0.05 m fold to L_c − 0.05 = 22.490786 m; r = 0.05 m stays 0.05 m.
def test_fold_mod_Lc_O19():
    k1, k2 = 4 * np.pi * F1 / C_LIGHT, 4 * np.pi * F2 / C_LIGHT
    phi1 = wf._wrap0(np.array([-0.05 * k1, 0.05 * k1]))
    phi2 = wf._wrap0(np.array([-0.05 * k2, 0.05 * k2]))
    R1, R2 = wf.fine_range(phi1, F1), wf.fine_range(phi2, F2)
    s = np.full(2, 1e-3)
    Rhat = wf.unwrap_fuse(R1, R2, phi1, phi2, s, s, SENSOR)
    assert Rhat[0] == pytest.approx(22.490786, abs=1e-6)
    assert Rhat[1] == pytest.approx(0.05, abs=1e-12)
    assert not (SENSOR.r_min <= Rhat[0] <= SENSOR.r_max)


# ------------------------------------------
# test_mixed_pixel_two_equal_returns_E19_example
# ------------------------------------------
# Equal returns at 1 m and 3 m: wrapped fine ranges 0.389944 m (f1), 0.591201 m (f2).
def test_mixed_pixel_two_equal_returns_E19_example():
    _, R1, R2 = noise_free_range([[1.0, 3.0]], 1e-6, return_parts=True)
    assert R1[0] == pytest.approx(0.389944, abs=1e-6)
    assert R2[0] == pytest.approx(0.591201, abs=1e-6)


# ------------------------------------------
# test_fusion_weights_inverse_variance
# ------------------------------------------
# φ̂₁ = φ̂₂ = 0 ⇒ R_c = 0 ⇒ k_j = 0, R'_j = R_j: isolates the inverse-variance weights of E14.
def test_fusion_weights_inverse_variance():
    p = np.zeros(1)
    out = wf.unwrap_fuse(np.array([1.00]), np.array([1.03]), p, p, np.array([0.01]), np.array([0.02]), SENSOR)
    assert out[0] == pytest.approx((1.00 / 1e-4 + 1.03 / 4e-4) / (1 / 1e-4 + 1 / 4e-4), rel=1e-12)


# ------------------------------------------
# test_fusion_weight_edge_cases
# ------------------------------------------
# Fusion weights when one or both noise levels are zero.
@pytest.mark.parametrize("s1, s2, w1", [
    (np.inf, np.inf, 0.5), (0.0, 0.0, 0.5), (np.inf, 0.01, 0.0), (0.01, np.inf, 1.0),
    (0.0, 0.01, 1.0), (0.0, np.inf, 1.0), (np.nan, 0.01, 0.0), (np.nan, np.nan, 0.5),
])
def test_fusion_weight_edge_cases(s1, s2, w1):
    p = np.zeros(1)
    with np.errstate(all="raise"):
        out = wf.unwrap_fuse(np.array([1.00]), np.array([1.04]), p, p, np.array([s1]), np.array([s2]), SENSOR)
    assert out[0] == pytest.approx(w1 * 1.00 + (1 - w1) * 1.04, abs=1e-12)


# ------------------------------------------
# test_range_std_E17
# ------------------------------------------
# Range noise matches E17.
def test_range_std_E17():
    A, sn = np.array([2e-6, 0.0]), np.array([1e-7, 1e-7])
    with np.errstate(all="raise"):
        s = wf._range_std(A, sn, N, F1)
    sphi = np.sqrt(2 * 1e-14 / (N * 4e-12))
    assert s[0] == pytest.approx(C_LIGHT * sphi / (4 * np.pi * F1), rel=1e-12)
    assert s[1] == np.inf
