# ==========================================================
# tests/unit/test_waveform_measure.py
# ==========================================================
# Tests for the CFAR threshold (E15) and the end-to-end ``measure`` (E10–E15), incl. quick Monte Carlo
# sanity checks (not the `stat` suite), C-08 determinism and two edge cases.

import dataclasses
import math

import numpy as np
import pytest

from lidar_sim.core.config import NoiseConfig, SensorConfig
from lidar_sim.core.types import RadiometricScan, Scan
from lidar_sim.physics import waveform as wf
from sim.calibrate import REF_RANGE, REF_RHO, received_power

pytestmark = pytest.mark.filterwarnings("error")

NOISE = NoiseConfig()
SENSOR = SensorConfig()
N = SENSOR.n_samples
W = SENSOR.r_max - SENSOR.r_min
Z99 = 2.5758293035489


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
                              "surface_id": ids, "fire_time": np.linspace(0.0, 0.05, B)})


# ------------------------------------------
# single
# ------------------------------------------
# A trace of identical beams with one return each.
def single(r, P, B, P_amb=0.0):
    return make_trace(np.full((B, 1), r), P, P_amb)


# ------------------------------------------
# wilson
# ------------------------------------------
# Wilson interval used to judge simulated rates.
def wilson(k, n, z=Z99):
    p = k / n
    den = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return c - h, c + h


P_REF4 = received_power(NOISE, REF_RHO, REF_RANGE)      # ADR-0004 reference point (ρ=0.9 at 4 m)


# ------------------------------------------
# test_cfar_threshold_ratio_O14
# ------------------------------------------
# The CFAR threshold ratio matches O-14.
def test_cfar_threshold_ratio_O14():
    sb = math.sqrt(2 / N) * NOISE.sigma_th
    g = wf.cfar_threshold(NOISE, 0.0, N)
    assert isinstance(g, float)
    assert g / sb == pytest.approx(4.291932, abs=1e-6)
    assert g / sb == pytest.approx(math.sqrt(2 * math.log(1e4)), rel=1e-14)


# ------------------------------------------
# test_cfar_threshold_tracks_ambient
# ------------------------------------------
# The CFAR threshold rises with ambient light.
def test_cfar_threshold_tracks_ambient():
    Pa = NOISE.P_amb_sun
    sb = math.sqrt(2 / N) * math.sqrt(NOISE.sigma_th**2 + NOISE.kappa_sh * NOISE.responsivity * Pa)
    g = wf.cfar_threshold(NOISE, Pa, N)
    assert g == pytest.approx(sb * math.sqrt(-2 * math.log(NOISE.P_FA0)), rel=1e-14)
    assert g > wf.cfar_threshold(NOISE, 0.0, N)


# ------------------------------------------
# test_cfar_false_alarm_rate_no_hit
# ------------------------------------------
# No-hit beam: detection rate = P_FA0 · |W| / L_c for any ambient (CFAR, E15 + E20 no-hit branch).
@pytest.mark.parametrize("P_amb", [0.0, NoiseConfig().P_amb_sun, 10 * NoiseConfig().P_amb_sun])
def test_cfar_false_alarm_rate_no_hit(P_amb, rng):
    noise = dataclasses.replace(NOISE, P_FA0=0.05)
    n = 20000
    scan = wf.measure(RadiometricScan.empty(n, 1, P_amb=P_amb), SENSOR, noise, rng)
    k = int(scan.detected.sum())
    lo, hi = wilson(k, n)
    p = 0.05 * W / SENSOR.Lc
    assert lo <= p <= hi, f"n={n} k={k} rate={k / n:.5f} expected {p:.5f} Wilson99=({lo:.5f},{hi:.5f})"
    # Above-threshold amplitude rate (before the window gate) is P_FA0 itself.
    k_amp = int((scan.amplitude[:, 0] >= wf.cfar_threshold(noise, P_amb, N)).sum())
    lo, hi = wilson(k_amp, n)
    assert lo <= 0.05 <= hi, f"n={n} k_amp={k_amp}"


# ------------------------------------------
# test_measure_returns_valid_scan
# ------------------------------------------
# Measurement returns a valid scan.
def test_measure_returns_valid_scan(rng):
    tr = make_trace([[1.0, 1.0], [np.nan, np.nan], [5.9, np.nan], [3.0, 3.2]],
                    [[1e-6, 1e-6], [0, 0], [1e-5, 0], [2e-6, 1e-6]], P_amb=1e-7)
    scan = wf.measure(tr, SENSOR, NOISE, rng)
    assert isinstance(scan, Scan) and scan.B == 4
    assert scan.truth is tr
    np.testing.assert_array_equal(scan.fire_time, tr.fire_time)
    assert scan.amplitude.shape == (4, 2) and np.isfinite(scan.amplitude).all()
    assert np.isfinite(scan.range_raw).all()
    assert ((scan.range_raw >= 0) & (scan.range_raw < SENSOR.Lc)).all()
    assert scan.detected[0] and scan.range[0] == pytest.approx(1.0, abs=0.01)
    assert not scan.detected[2] and np.isnan(scan.range[2])            # strong but beyond r_max
    assert scan.range_raw[2] == pytest.approx(5.9, abs=0.01)


# ------------------------------------------
# test_measure_rng_draw_order_f1_then_f2
# ------------------------------------------
# Interface sheet: noise streams drawn from rng as f1 samples (B, N), then f2 samples (B, N).
def test_measure_rng_draw_order_f1_then_f2():
    tr = make_trace([[2.0], [4.5], [np.nan]], [[3e-7], [5e-8], [0.0]], P_amb=2e-7)
    scan = wf.measure(tr, SENSOR, NOISE, np.random.default_rng(11))
    rng = np.random.default_rng(11)
    dc, sn = wf.dc_level(tr, NOISE), wf.noise_std(tr, NOISE)
    est = []
    for f in (SENSOR.f1, SENSOR.f2):
        phi, A = wf.estimate(wf.sample(wf.phasors(tr, f, NOISE), dc, sn, N, rng))
        est.append((phi, A, wf.fine_range(phi, f), wf._range_std(A, sn, N, f)))
    (p1, A1, R1, s1), (p2, A2, R2, s2) = est
    Rhat = wf.unwrap_fuse(R1, R2, p1, p2, s1, s2, SENSOR)
    np.testing.assert_array_equal(scan.range_raw, Rhat)
    np.testing.assert_array_equal(scan.amplitude, np.stack([A1, A2], axis=1))
    gamma = wf.cfar_threshold(NOISE, tr.P_amb, N)
    det = (A1 >= gamma) & (Rhat >= SENSOR.r_min) & (Rhat <= SENSOR.r_max)
    np.testing.assert_array_equal(scan.detected, det)
    np.testing.assert_array_equal(scan.range, np.where(det, Rhat, np.nan))


# ------------------------------------------
# test_measure_uses_trace_P_amb_only
# ------------------------------------------
# Ambient enters only via trace.P_amb: it raises σ_n and γ together.
def test_measure_uses_trace_P_amb_only(rng):
    lo = wf.measure(single(3.0, 1e-7, 2000, P_amb=0.0), SENSOR, NOISE, np.random.default_rng(3))
    hi = wf.measure(single(3.0, 1e-7, 2000, P_amb=10 * NOISE.P_amb_sun), SENSOR, NOISE, np.random.default_rng(3))
    assert np.nanstd(hi.range) > np.nanstd(lo.range)


# ------------------------------------------
# test_mc_phase_variance_matches_crlb
# ------------------------------------------
# A/σ_n = 2.5: var(wrapπ(φ̂ − φ)) within 5 % of the CRLB 1/(N·SNR) = 2σ_n²/(N A²) (E17, D-1).
def test_mc_phase_variance_matches_crlb(rng):
    n, A, phi0 = 40000, 1e-6, 1.3
    sig = A / 2.5
    s = wf.sample(np.full(n, A * np.exp(1j * phi0)), np.full(n, 5e-6), np.full(n, sig), N, rng)
    phi, _ = wf.estimate(s)
    err = (phi - phi0 + np.pi) % (2 * np.pi) - np.pi
    crlb = 2 * sig**2 / (N * A**2)
    v = err.var()
    assert v == pytest.approx(crlb, rel=0.05), f"n={n} var={v:.6f} crlb={crlb:.6f} ratio={v / crlb:.4f}"
    assert abs(err.mean()) < 5 * math.sqrt(crlb / n), f"bias {err.mean():.2e}"


# ------------------------------------------
# test_mc_fused_range_std_at_calibration_point
# ------------------------------------------
# ADR-0004 reference point (ρ = 0.9, 4 m, clear air, no ambient): fused σ_R ≈ 12 mm within 5 %.
def test_mc_fused_range_std_at_calibration_point(rng):
    n = 20000
    scan = wf.measure(single(REF_RANGE, P_REF4, n), SENSOR, NOISE, rng)
    assert scan.detected.all()
    sd, mu = scan.range.std(), scan.range.mean()
    assert sd == pytest.approx(0.012, rel=0.05), f"n={n} std={sd * 1e3:.3f} mm mean={mu:.5f} m"
    assert abs(mu - REF_RANGE) < 5 * sd / math.sqrt(n)


# ------------------------------------------
# _scan_arrays
# ------------------------------------------
# The arrays of a scan, for comparisons.
def _scan_arrays(s):
    return [s.detected, s.range, s.amplitude, s.range_raw, s.fire_time]


# ------------------------------------------
# test_C08_same_seed_bit_identical
# ------------------------------------------
# The same seed gives bit-identical scans (C-08).
def test_C08_same_seed_bit_identical():
    tr = make_trace([[1.0], [3.0], [np.nan], [4.9]], [[1e-7], [5e-8], [0], [2e-8]], P_amb=1e-7)
    a = wf.measure(tr, SENSOR, NOISE, np.random.default_rng(2024))
    b = wf.measure(tr, SENSOR, NOISE, np.random.default_rng(2024))
    for x, y in zip(_scan_arrays(a), _scan_arrays(b), strict=True):
        assert x.tobytes() == y.tobytes()


# ------------------------------------------
# test_C08_spawned_streams_differ
# ------------------------------------------
# Spawned streams give different scans (C-08).
def test_C08_spawned_streams_differ():
    tr = single(3.0, 5e-8, 50)
    s1, s2 = np.random.SeedSequence(2024).spawn(2)
    a = wf.measure(tr, SENSOR, NOISE, np.random.default_rng(s1))
    b = wf.measure(tr, SENSOR, NOISE, np.random.default_rng(s2))
    assert not np.array_equal(a.range_raw, b.range_raw)
    assert not np.array_equal(a.amplitude, b.amplitude)


# ------------------------------------------
# test_open_space_scan_all_missing
# ------------------------------------------
# All-miss trace → no detections except false alarms at the P_FA0 rate; ranges NaN
# where not detected; amplitudes finite; no warnings (module-level filterwarnings=error).
def test_open_space_scan_all_missing(rng):
    B = 682 * 30
    for P_amb in (0.0, NOISE.P_amb_sun):
        scan = wf.measure(RadiometricScan.empty(B, 5, P_amb=P_amb), SENSOR, NOISE, rng)
        k = int(scan.detected.sum())
        lam = NOISE.P_FA0 * W / SENSOR.Lc * B                 # ≈ 0.51 expected false alarms
        assert k <= 8, f"B={B} false alarms={k} expected≈{lam:.2f} (P(Poisson>8) < 1e-8)"
        assert np.isnan(scan.range[~scan.detected]).all()
        assert np.isfinite(scan.range[scan.detected]).all()
        assert np.isfinite(scan.amplitude).all() and (scan.amplitude >= 0).all()
        assert np.isfinite(scan.range_raw).all()
        assert ((scan.range_raw >= 0) & (scan.range_raw < SENSOR.Lc)).all()


# ------------------------------------------
# test_zero_power_hits_behave_as_open_space
# ------------------------------------------
# ρ = 0 surfaces: hit=True but power 0 ⇒ Z = 0; same statistics as no return.
def test_zero_power_hits_behave_as_open_space(rng):
    B = 5000
    scan = wf.measure(single(2.0, 0.0, B), SENSOR, NOISE, rng)
    assert int(scan.detected.sum()) <= 6
    assert np.isfinite(scan.amplitude).all()


# ------------------------------------------
# test_near_zero_range_folds_out
# ------------------------------------------
# Targets at ≈ 0 m (inside r_min) and at the coarse-wrap boundary are never reported
# — neither as negative ranges nor as ≈ L_c ≈ 22 m — and the raw estimate stays in [0, L_c).
def test_near_zero_range_folds_out(rng):
    Lc, n = SENSOR.Lc, 4000
    P = 100 * P_REF4                                           # σ_R ≈ 0.9 mm
    for r in (1e-6, 0.005, 0.01, Lc - 0.003, Lc - 1e-6):
        scan = wf.measure(single(r, P, n), SENSOR, NOISE, rng)
        assert not scan.detected.any(), f"r={r}: {scan.range[scan.detected][:5]}"
        assert ((scan.range_raw >= 0) & (scan.range_raw < Lc)).all()
        circ = np.abs((scan.range_raw - r + Lc / 2) % Lc - Lc / 2)
        assert circ.max() < 0.01, f"r={r}: max circular error {circ.max():.4f}"
    ok = wf.measure(single(0.05, P, n), SENSOR, NOISE, rng)         # just inside the window
    assert ok.detected.mean() > 0.999 and np.nanmax(np.abs(ok.range - 0.05)) < 0.01


# ------------------------------------------
# test_range_walk_bias
# ------------------------------------------
# Optional range walk shifts each range by the log of its power ratio; empty beams are unchanged.
def test_range_walk_bias():
    sensor = dataclasses.replace(SENSOR, range_walk=True, b0=0.02, P_ref=1e-7)
    tr = make_trace([[3.0], [3.0], [np.nan]], [[1e-6], [1e-8], [0.0]])
    a = wf.measure(tr, SENSOR, NOISE, np.random.default_rng(5))
    b = wf.measure(tr, sensor, NOISE, np.random.default_rng(5))
    shift = 0.02 * np.log(1e-7 / np.array([1e-6, 1e-8]))
    np.testing.assert_allclose(b.range_raw[:2] - a.range_raw[:2], shift, atol=1e-12)
    assert b.range_raw[2] == a.range_raw[2]                        # P̄ = 0: no bias, no log(0)
