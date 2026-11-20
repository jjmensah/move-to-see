# ==========================================================
# tests/contract/test_waveform_contract.py
# ==========================================================
# C-01 for physics/waveform.py: dimension-agnostic over d ∈ {2, 3}, preserves the (B,) leading axis.

import numpy as np
import pytest

from lidar_sim.core.config import NoiseConfig, SensorConfig
from lidar_sim.core.types import RadiometricScan, Scan
from lidar_sim.physics import waveform as wf

pytestmark = pytest.mark.filterwarnings("error")

NOISE, SENSOR = NoiseConfig(), SensorConfig()


# ------------------------------------------
# trace_d
# ------------------------------------------
# Hand-built record: B = 4 beams, K = 3 sub-rays, J = K or 2K; mixed hits/misses/bounces.
def trace_d(d: int, specular: bool = False) -> RadiometricScan:
    B, K = 4, 3
    J = 2 * K if specular else K
    base = RadiometricScan.empty(B, K, J=J, d=d, P_amb=1e-7)
    hit = np.zeros((B, J), bool)
    hit[0, :K] = True
    hit[1, 0] = True
    hit[3, :2] = True
    if specular:
        hit[3, K] = True                                  # one bounced return
    r = np.where(hit, 1.0 + np.arange(J)[None, :] * 0.4 + np.arange(B)[:, None], np.nan)
    P = np.where(hit, 1e-7, 0.0)
    normal = np.zeros((B, J, d))
    normal[..., -1] = -1.0
    ids = np.where(hit, 1, -1)
    bounce = np.zeros((B, J), bool)
    if specular:
        bounce[:, K:] = hit[:, K:]
    return RadiometricScan(**{**base.fields(), "hit": hit, "r_path": r, "power": P, "normal": normal,
                              "cos_inc": np.where(hit, 0.8, 0.0), "object_id": ids, "material_id": ids,
                              "surface_id": ids, "bounce": bounce})


# ------------------------------------------
# test_waveform_dimension_agnostic
# ------------------------------------------
# The waveform simulator gives the same results for 2D and 3D traces.
@pytest.mark.parametrize("d", [2, 3])
@pytest.mark.parametrize("specular", [False, True])
def test_waveform_dimension_agnostic(d, specular):
    tr = trace_d(d, specular)
    B = tr.B
    Z = wf.phasors(tr, SENSOR.f1, NOISE)
    dc, sn = wf.dc_level(tr, NOISE), wf.noise_std(tr, NOISE)
    for x in (Z, dc, sn):
        assert x.shape == (B,) and np.isfinite(x).all()
    s = wf.sample(Z, dc, sn, SENSOR.n_samples, np.random.default_rng(0))
    assert s.shape == (B, SENSOR.n_samples)
    phi, A = wf.estimate(s)
    R = wf.fine_range(phi, SENSOR.f1)
    Rh = wf.unwrap_fuse(R, R, phi, phi, np.ones(B), np.ones(B), SENSOR)
    for x in (phi, A, R, Rh):
        assert x.shape == (B,) and np.isfinite(x).all()
    scan = wf.measure(tr, SENSOR, NOISE, np.random.default_rng(1))
    assert isinstance(scan, Scan) and scan.B == B and scan.truth.dim == d
    assert np.isfinite(scan.amplitude).all() and np.isfinite(scan.range_raw).all()


# ------------------------------------------
# test_waveform_same_result_for_d2_and_d3
# ------------------------------------------
# Only the (B, J) return arrays matter: the same returns in d = 2 and d = 3 give identical Scans.
def test_waveform_same_result_for_d2_and_d3():
    a = wf.measure(trace_d(2), SENSOR, NOISE, np.random.default_rng(7))
    b = wf.measure(trace_d(3), SENSOR, NOISE, np.random.default_rng(7))
    for name in ("detected", "range", "amplitude", "range_raw"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
