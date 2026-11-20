# ==========================================================
# tests/unit/test_calibration.py
# ==========================================================
# ADR-0004: frozen noise constants reproduce the calibration targets.

from itertools import pairwise

import pytest

from lidar_sim.core.config import NoiseConfig, SensorConfig
from sim import calibrate as cal


# ------------------------------------------
# test_reference_sigma_r_is_12mm
# ------------------------------------------
# The reference operating point has a 12 mm range noise.
def test_reference_sigma_r_is_12mm():
    n, s = NoiseConfig(), SensorConfig()
    p = cal.received_power(n, cal.REF_RHO, cal.REF_RANGE)
    assert cal.fused_sigma_r(n, s, p) == pytest.approx(0.012, rel=1e-6)


# ------------------------------------------
# test_shot_equals_thermal_at_reference
# ------------------------------------------
# Shot noise equals thermal noise at the reference point.
def test_shot_equals_thermal_at_reference():
    n = NoiseConfig()
    p = cal.received_power(n, cal.REF_RHO, cal.REF_RANGE)
    assert n.kappa_sh * n.responsivity * p == pytest.approx(n.sigma_th**2, rel=1e-6)


# ------------------------------------------
# test_sun_halves_white_r50
# ------------------------------------------
# One sun halves the 50 % detection range of a white target.
def test_sun_halves_white_r50():
    n, s = NoiseConfig(), SensorConfig()
    clear = cal.r50(n, s, cal.REF_RHO, 0.0)
    sun = cal.r50(n, s, cal.REF_RHO, n.P_amb_sun)
    assert sun == pytest.approx(clear / 2.0, rel=1e-6)


# ------------------------------------------
# test_frozen_constants_match_solver
# ------------------------------------------
# The frozen constants match the calibration solver.
def test_frozen_constants_match_solver():
    c = cal.solve()
    n = NoiseConfig()
    assert (n.sigma_th, n.kappa_sh, n.P_amb_sun) == pytest.approx(
        (c.sigma_th, c.kappa_sh, c.P_amb_sun), rel=1e-9)


# ------------------------------------------
# test_cfar_false_alarm_independent_of_ambient
# ------------------------------------------
# The false-alarm rate does not depend on ambient light.
def test_cfar_false_alarm_independent_of_ambient():
    n, s = NoiseConfig(), SensorConfig()
    for p_amb in (0.0, n.P_amb_sun, 10 * n.P_amb_sun):
        assert cal.detection_probability(n, s, 0.0, p_amb) == pytest.approx(n.P_FA0, rel=1e-9)


# ------------------------------------------
# test_ambient_monotonically_shrinks_range
# ------------------------------------------
# More ambient light always shortens the detection range.
def test_ambient_monotonically_shrinks_range():
    n, s = NoiseConfig(), SensorConfig()
    r = [cal.r50(n, s, 0.9, k * n.P_amb_sun) for k in (0, 0.5, 1, 2, 4)]
    assert all(a > b for a, b in pairwise(r))


# ------------------------------------------
# test_calibration_verified_by_waveform_simulator
# ------------------------------------------
# Task 2.2: frozen constants confirmed through the full simulator (σ_R 12 mm; P_D = 0.5 at r50, clear & sun).
@pytest.mark.stat
def test_calibration_verified_by_waveform_simulator():
    res = cal.verify(n=20000, seed=4240)
    assert res["pass"], res
