# ==========================================================
# tests/property/test_radiometry_props.py
# ==========================================================
# Property tests: P-04 (power non-increasing in r), P-05 (power increasing in ρ, V, cos θ).

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from lidar_sim.core.config import NoiseConfig
from lidar_sim.physics.atmosphere import extinction
from lidar_sim.physics.radiometry import lambertian_power

SETTINGS = settings(max_examples=200, derandomize=True, deadline=None)
NZ = NoiseConfig()
LAM = 785e-9
rng_m = st.floats(0.02, 50.0)
rho = st.floats(1e-3, 1.0)
cosv = st.floats(1e-3, 1.0)
vis = st.floats(10.0, 1e5)


# ------------------------------------------
# _P
# ------------------------------------------
# Return power for a range, reflectivity, incidence and visibility.
def _P(r, rho_=0.9, cos=1.0, V=20000.0):
    return float(lambertian_power(NZ.P0, 1.0, rho_, cos, r, NZ.A_rx, NZ.eta, extinction(V, LAM)))


# ------------------------------------------
# test_p04_nonincreasing_in_range
# ------------------------------------------
# Return power never grows with range.
@SETTINGS
@given(r1=rng_m, r2=rng_m, rho_=rho, cos=cosv, V=vis)
def test_p04_nonincreasing_in_range(r1, r2, rho_, cos, V):
    lo, hi = sorted((r1, r2))
    assert _P(lo, rho_, cos, V) >= _P(hi, rho_, cos, V)
    if hi > lo * (1 + 1e-9):
        assert _P(lo, rho_, cos, V) > _P(hi, rho_, cos, V)


# ------------------------------------------
# test_p05_increasing_in_rho
# ------------------------------------------
# Return power grows with reflectivity.
@SETTINGS
@given(r=rng_m, a=rho, b=rho, cos=cosv, V=vis)
def test_p05_increasing_in_rho(r, a, b, cos, V):
    lo, hi = sorted((a, b))
    assume(hi > lo * (1 + 1e-9))
    assert _P(r, hi, cos, V) > _P(r, lo, cos, V)


# ------------------------------------------
# test_p05_increasing_in_cos
# ------------------------------------------
# Return power grows as incidence becomes normal.
@SETTINGS
@given(r=rng_m, rho_=rho, a=cosv, b=cosv, V=vis)
def test_p05_increasing_in_cos(r, rho_, a, b, V):
    lo, hi = sorted((a, b))
    assume(hi > lo * (1 + 1e-9))
    assert _P(r, rho_, hi, V) > _P(r, rho_, lo, V)


# ------------------------------------------
# test_p05_increasing_in_visibility
# ------------------------------------------
# Return power grows with visibility.
@SETTINGS
@given(r=rng_m, rho_=rho, cos=cosv, a=vis, b=vis)
def test_p05_increasing_in_visibility(r, rho_, cos, a, b):
    lo, hi = sorted((a, b))
    assert extinction(hi, LAM) <= extinction(lo, LAM)
    assert _P(r, rho_, cos, hi) >= _P(r, rho_, cos, lo)
    if hi > lo * 1.01:
        assert extinction(hi, LAM) < extinction(lo, LAM)
