# ==========================================================
# tests/oracles/test_oracles_waveform.py
# ==========================================================
# Hand-computed waveform and ranging oracles.

import numpy as np

from lidar_sim.core.config import NoiseConfig, SensorConfig
from lidar_sim.core.constants import C_LIGHT
from lidar_sim.core.types import RadiometricScan
from lidar_sim.physics import waveform


# ------------------------------------------
# _noise_free_samples
# ------------------------------------------
# Noise-free samples of one beam for a phasor and a DC level.
def _noise_free_samples(z: complex, dc: float, n: int) -> np.ndarray:
    theta = 2 * np.pi * np.arange(n) / n
    return (dc + np.real(z * np.exp(-1j * theta)))[None, :]


# ------------------------------------------
# test_o10_single_return_phasor_sampling_and_estimate
# ------------------------------------------
# O-10, E10 and E12: synthetic sinusoid recovers its phasor and amplitude.
def test_o10_single_return_phasor_sampling_and_estimate():
    noise = NoiseConfig()
    trace = RadiometricScan.empty(1, 1)
    fields = trace.fields()
    fields.update(hit=np.array([[True]]), r_path=np.array([[1.0]]), power=np.array([[2e-8]]),
                  normal=np.array([[[-1.0, 0.0]]]), cos_inc=np.array([[1.0]]),
                  object_id=np.array([[0]]), material_id=np.array([[0]]), surface_id=np.array([[0]]))
    trace = RadiometricScan(**fields)
    z = waveform.phasors(trace, 46.55e6, noise)
    expected_z = noise.responsivity * noise.mod_depth * 2e-8 * np.exp(1j * 4 * np.pi * 46.55e6 / C_LIGHT)
    np.testing.assert_allclose(z, [expected_z], rtol=1e-12, atol=1e-20)

    n = 30
    samples = _noise_free_samples(expected_z, 7e-7, n)
    sampled = waveform.sample(np.array([expected_z]), np.array([7e-7]), np.array([0.0]), n,
                              np.random.default_rng(10))
    np.testing.assert_allclose(sampled, samples, rtol=0.0, atol=1e-20)
    phi, amplitude = waveform.estimate(samples)
    expected_phi = (4 * np.pi * 46.55e6 / C_LIGHT) % (2 * np.pi)
    np.testing.assert_allclose(phi, [expected_phi], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(amplitude, [abs(expected_z)], rtol=1e-12)


# ------------------------------------------
# test_o11_fine_range_alias_period
# ------------------------------------------
# O-11, E13: ranges one fine ambiguity interval apart have identical phase range.
def test_o11_fine_range_alias_period():
    sensor = SensorConfig()
    ranges = np.array([1.0, 1.0 + sensor.L1])
    phases = 4 * np.pi * sensor.f1 * ranges / C_LIGHT
    fine = waveform.fine_range(phases, sensor.f1)
    np.testing.assert_allclose(fine, [1.0, 1.0], rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(sensor.L1, C_LIGHT / (2 * sensor.f1), rtol=1e-12)
    np.testing.assert_allclose(sensor.L2, C_LIGHT / (2 * sensor.f2), rtol=1e-12)


# ------------------------------------------
# test_o12_dual_frequency_unwrap_and_fusion
# ------------------------------------------
# O-12, E14: paired phases recover ranges across both fine ambiguity lengths.
def test_o12_dual_frequency_unwrap_and_fusion():
    sensor = SensorConfig()
    ranges = np.array([0.5, 2.9, 3.3, 5.5, 22.0])
    phi1 = np.mod(4 * np.pi * sensor.f1 * ranges / C_LIGHT, 2 * np.pi)
    phi2 = np.mod(4 * np.pi * sensor.f2 * ranges / C_LIGHT, 2 * np.pi)
    r1 = waveform.fine_range(phi1, sensor.f1)
    r2 = waveform.fine_range(phi2, sensor.f2)
    sig1 = np.ones_like(r1)
    sig2 = np.ones_like(r2)
    fused = waveform.unwrap_fuse(r1, r2, phi1, phi2, sig1, sig2, sensor)
    np.testing.assert_allclose(fused, ranges, rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(sensor.Lc, C_LIGHT / (2 * (sensor.f2 - sensor.f1)), rtol=1e-12)


# ------------------------------------------
# test_o14_sim_cfar_threshold
# ------------------------------------------
# O-14, E15: CFAR threshold scales with background sample noise.
def test_o14_sim_cfar_threshold():
    noise = NoiseConfig()
    n = 30
    threshold = waveform.cfar_threshold(noise, 0.0, n)
    s0 = np.sqrt(2 / n) * noise.sigma_th
    np.testing.assert_allclose(threshold / s0, np.sqrt(2 * np.log(1e4)), rtol=1e-12)


# ------------------------------------------
# test_o19_coarse_wrap_folds_negative_near_zero_range
# ------------------------------------------
# O-19, E14: final modulo folds negative range to just below Lc.
def test_o19_coarse_wrap_folds_negative_near_zero_range():
    sensor = SensorConfig()
    ranges = np.array([-0.05, 0.05])
    phi1 = np.mod(4 * np.pi * sensor.f1 * ranges / C_LIGHT, 2 * np.pi)
    phi2 = np.mod(4 * np.pi * sensor.f2 * ranges / C_LIGHT, 2 * np.pi)
    r1 = waveform.fine_range(phi1, sensor.f1)
    r2 = waveform.fine_range(phi2, sensor.f2)
    fused = waveform.unwrap_fuse(r1, r2, phi1, phi2, np.ones(2), np.ones(2), sensor)
    np.testing.assert_allclose(fused, [sensor.Lc - 0.05, 0.05], rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(sensor.Lc / sensor.L1, 7.0, rtol=1e-12)
    np.testing.assert_allclose(sensor.Lc / sensor.L2, 8.0, rtol=1e-12)
