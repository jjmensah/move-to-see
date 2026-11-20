# ==========================================================
# tests/property/test_waveform_props.py
# ==========================================================
# Property tests for physics/waveform.py (P-06 and estimator invariants).

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from lidar_sim.core.config import NoiseConfig, SensorConfig
from lidar_sim.core.types import RadiometricScan
from lidar_sim.physics import waveform as wf

pytestmark = pytest.mark.filterwarnings("error")

NOISE = NoiseConfig()
SENSOR = SensorConfig()
N = SENSOR.n_samples


# ------------------------------------------
# make_trace
# ------------------------------------------
# A radiometric trace with the given ranges and powers.
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
# noise_free_Rhat
# ------------------------------------------
# E10–E14 with σ = 0 samples (``sample`` with zero σ_n); fusion weights from E17 with the true σ_n.
def noise_free_Rhat(r, P):
    tr = make_trace(np.asarray(r, float)[:, None], P)
    dc, sn = wf.dc_level(tr, NOISE), wf.noise_std(tr, NOISE)
    rng = np.random.default_rng(0)
    parts = []
    for f in (SENSOR.f1, SENSOR.f2):
        phi, A = wf.estimate(wf.sample(wf.phasors(tr, f, NOISE), dc, np.zeros_like(sn), N, rng))
        parts.append((phi, wf.fine_range(phi, f), wf._range_std(A, sn, N, f)))
    (p1, R1, s1), (p2, R2, s2) = parts
    return wf.unwrap_fuse(R1, R2, p1, p2, s1, s2, SENSOR)


# ------------------------------------------
# test_P06_noise_free_single_return_exact
# ------------------------------------------
# Without noise a single return is ranged exactly.
@settings(max_examples=300, deadline=None)
@given(r=st.floats(SENSOR.r_min, SENSOR.Lc - 1e-6), logP=st.floats(-12.0, -3.0))
def test_P06_noise_free_single_return_exact(r, logP):
    Rhat = noise_free_Rhat([r], 10.0**logP)
    assert abs(Rhat[0] - r) < 1e-9, (r, Rhat[0])


# ------------------------------------------
# _near_multiples
# ------------------------------------------
# Ranges just around multiples of the fine ambiguity intervals.
def _near_multiples():
    L1, L2, Lc = SENSOR.L1, SENSOR.L2, SENSOR.Lc
    base = [k * L1 for k in range(1, 7)] + [k * L2 for k in range(1, 8)]
    rs = [b + e for b in base for e in (-1e-6, -1e-9, 0.0, 1e-9, 1e-6)]
    return np.array([x for x in rs if SENSOR.r_min <= x < Lc - 1e-6])


# ------------------------------------------
# test_P06_near_fine_wrap_boundaries
# ------------------------------------------
# Without noise, ranges near a phase wrap are still exact.
def test_P06_near_fine_wrap_boundaries():
    r = _near_multiples()
    err = noise_free_Rhat(r, 1e-6) - r
    assert np.abs(err).max() < 1e-9, r[np.argmax(np.abs(err))]


# ------------------------------------------
# test_P06_last_nanometres_before_Lc_circular
# ------------------------------------------
# Within 1 nm of L_c the estimate may legitimately fold to ≈ 0: check circular distance.
def test_P06_last_nanometres_before_Lc_circular():
    Lc = SENSOR.Lc
    r = np.array([Lc - 1e-6, Lc - 1e-9, np.nextafter(Lc, 0)])
    Rhat = noise_free_Rhat(r, 1e-6)
    d = np.abs((Rhat - r + Lc / 2) % Lc - Lc / 2)
    assert d.max() < 1e-9
    assert ((Rhat >= 0) & (Rhat < Lc)).all()


# ------------------------------------------
# test_unwrap_fuse_always_in_coarse_interval
# ------------------------------------------
# The fused range always lies in the coarse interval.
@settings(max_examples=200, deadline=None)
@given(phi1=st.floats(0, 2 * np.pi, exclude_max=True), phi2=st.floats(0, 2 * np.pi, exclude_max=True),
       s1=st.floats(1e-6, 10.0), s2=st.floats(1e-6, 10.0))
def test_unwrap_fuse_always_in_coarse_interval(phi1, phi2, s1, s2):
    p1, p2 = np.array([phi1]), np.array([phi2])
    R = wf.unwrap_fuse(wf.fine_range(p1, SENSOR.f1), wf.fine_range(p2, SENSOR.f2), p1, p2,
                       np.array([s1]), np.array([s2]), SENSOR)
    assert 0.0 <= R[0] < SENSOR.Lc


# ------------------------------------------
# test_estimate_recovers_noise_free_phasor
# ------------------------------------------
# The estimator recovers a noise-free phasor.
@settings(max_examples=200, deadline=None)
@given(amp=st.floats(1e-9, 1e-3), ph=st.floats(-10.0, 10.0), dc=st.floats(0.0, 1e-2))
def test_estimate_recovers_noise_free_phasor(amp, ph, dc):
    Z = np.array([amp * np.exp(1j * ph)])
    s = wf.sample(Z, np.array([dc]), np.zeros(1), N, np.random.default_rng(1))
    phi, A = wf.estimate(s)
    assert 0.0 <= phi[0] < 2 * np.pi
    tol = 1e-12 * (1 + dc / amp) * N
    assert abs(np.angle(np.exp(1j * (phi[0] - ph)))) < tol
    assert A[0] == pytest.approx(amp, rel=tol)
