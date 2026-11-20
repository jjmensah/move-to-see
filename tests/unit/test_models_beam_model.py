# ==========================================================
# tests/unit/test_models_beam_model.py
# ==========================================================
# Unit tests of the analytic beam model.

from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import quad

from lidar_sim.core.config import EnvConfig, NoiseConfig, SensorConfig
from lidar_sim.core.types import RadiometricScan, Scan
from lidar_sim.models.beam_model import BeamModel, expected_range
from lidar_sim.models.detection import cfar_gamma


# ------------------------------------------
# trace_with_hit
# ------------------------------------------
# A trace whose beams hit at the given ranges and powers.
def trace_with_hit(ranges=(1.0,), powers=(1e-7,), *, ambient=0.0, B=1):
    tr = RadiometricScan.empty(B, max(1, len(ranges)), P_amb=ambient)
    hit = np.ones((B, len(ranges)), bool)
    fields = tr.fields()
    fields.update(hit=hit, r_path=np.tile(ranges, (B, 1)), power=np.tile(powers, (B, 1)),
                  normal=np.tile(np.array([[[-1.0, 0.0]]]), (B, len(ranges), 1)),
                  cos_inc=np.ones((B, len(ranges))), object_id=np.zeros((B, len(ranges)), int),
                  material_id=np.zeros((B, len(ranges)), int), surface_id=np.tile(np.arange(len(ranges)), (B, 1)))
    return RadiometricScan(**fields)


# ------------------------------------------
# scan_for
# ------------------------------------------
# A scan with the given detections and ranges for a trace.
def scan_for(trace, detected, ranges):
    B = trace.B
    return Scan(np.array(detected), np.array(ranges, float), np.zeros((B, 2)), np.zeros(B),
                trace.fire_time, trace)


# ------------------------------------------
# test_no_hit_finite_likelihood_and_nan_range_branch
# ------------------------------------------
# A beam that hits nothing has a finite likelihood.
def test_no_hit_finite_likelihood_and_nan_range_branch():
    trace = RadiometricScan.empty(1, 1)
    model = BeamModel()
    pred = model.predict(trace, EnvConfig())
    scan = scan_for(trace, [False], [np.nan])
    assert np.isfinite(model.log_likelihood_beams(scan, pred)).all()


# ------------------------------------------
# test_single_surface_expected_range_exact_and_two_surface_phasors
# ------------------------------------------
# Expected range of one surface, and of two surfaces by phasor sum.
def test_single_surface_expected_range_exact_and_two_surface_phasors():
    sensor, noise = SensorConfig(), NoiseConfig()
    one = trace_with_hit((1.0,), (1e-7,))
    assert expected_range(one, sensor, noise)[0] == 1.0
    two = trace_with_hit((1.0, 3.0), (1e-7, 1e-7))
    result = expected_range(two, sensor, noise)[0]
    assert np.isclose(result, 3.452998183141784, atol=1e-9)


# ------------------------------------------
# test_extreme_ambient_and_tiny_power_stays_finite
# ------------------------------------------
# Extreme ambient light and tiny power keep the likelihood finite.
def test_extreme_ambient_and_tiny_power_stays_finite():
    tr = trace_with_hit((2.0,), (1e-20,), ambient=1e20)
    model = BeamModel()
    pred = model.predict(tr, EnvConfig())
    scan = scan_for(tr, [True], [2.0])
    assert pred.P_D[0] <= NoiseConfig().P_FA0 * 1.01
    assert np.isfinite(model.log_likelihood(scan, pred))


# ------------------------------------------
# test_weak_return_no_overflow
# ------------------------------------------
# Very weak distant echoes have zero precision weight without warning.
def test_weak_return_no_overflow():
    noise = NoiseConfig()
    # The geometric ρ=.9, r=1000 m return approaches this underflow-scale
    # power. Keep it explicit so the phase-variance square reaches infinity.
    trace = trace_with_hit((1000.0,), (1e-200,))
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        pred = BeamModel(noise=noise).predict(trace, EnvConfig())
    assert pred.P_D[0] >= 0
    assert np.isinf(pred.sigma_R[0])


# ------------------------------------------
# test_likelihood_normalization_hit_and_no_hit
# ------------------------------------------
# The likelihood is normalized for beams that hit and miss.
def test_likelihood_normalization_hit_and_no_hit():
    sensor = SensorConfig(r_min=0.0 + .02, r_max=5.6)
    model = BeamModel(sensor=sensor)
    for tr in (trace_with_hit((2.0,), (1e-7,)), RadiometricScan.empty(1, 1)):
        pred = model.predict(tr, EnvConfig())
        scan = scan_for(tr, [False], [np.nan])
        mass_detected = quad(lambda z, tr=tr, pred=pred: np.exp(model.log_likelihood_beams(
            scan_for(tr, [True], [z]), pred)[0]), sensor.r_min, sensor.r_max, epsabs=1e-12)[0]
        assert abs(mass_detected + np.exp(model.log_likelihood_beams(scan, pred)[0]) - 1) < 1e-9


# ------------------------------------------
# test_detected_outside_window_rejected_and_nan_when_detected_rejected
# ------------------------------------------
# A detection without a range, or with one outside the window, is rejected.
def test_detected_outside_window_rejected_and_nan_when_detected_rejected():
    tr = trace_with_hit()
    model = BeamModel()
    pred = model.predict(tr, EnvConfig())
    for value in (np.nan, 0.0, 5.7):
        with pytest.raises(ValueError):
            model.log_likelihood_beams(scan_for(tr, [True], [value]), pred)


# ------------------------------------------
# test_missed_hit_has_two_sided_floor_even_at_pd_one
# ------------------------------------------
# A missed detection keeps a finite likelihood even at P_D = 1 (E20).
def test_missed_hit_has_two_sided_floor_even_at_pd_one():
    tr = trace_with_hit()
    model = BeamModel()
    pred = model.predict(tr, EnvConfig())
    pred = replace(pred, P_D=np.ones(1))
    ll = model.log_likelihood_beams(scan_for(tr, [False], [np.nan]), pred)[0]
    assert np.isfinite(ll)
    assert ll >= np.log(model.model.eps_floor / 2)


# ------------------------------------------
# test_pfa_is_background_value_for_hit_and_nohit
# ------------------------------------------
# The false-alarm rate is the background value for every beam.
def test_pfa_is_background_value_for_hit_and_nohit():
    model = BeamModel()
    noise = model.noise
    gamma = cfar_gamma(noise, 0.0, model.sensor.n_samples)
    background_s = np.sqrt(2 / model.sensor.n_samples) * noise.sigma_th
    expected = np.exp(-gamma**2 / (2 * background_s**2))
    for tr in (trace_with_hit(), RadiometricScan.empty(1, 1)):
        pred = model.predict(tr, EnvConfig())
        assert pred.P_FA[0] == pytest.approx(expected, rel=1e-14)


# ------------------------------------------
# test_e19_mixed_truth_table
# ------------------------------------------
# Mixed-pixel flags for combinations of returns (E19).
@pytest.mark.parametrize("ranges,weights,expected", [
    ((1.0, 1.0), (0.5, 0.5), False),       # partial hit, single range
    ((1.0, 1.2), (0.5, 0.5), True),        # partial hit, separated surfaces
    ((1.0, 1.04), (0.5, 0.5), False),      # below mixed tolerance
    ((1.0, 1.06), (0.5, 0.5), True),       # above mixed tolerance
    ((1.0, 1.2), (0.995, 0.005), False),   # second surface below w_min
])
def test_e19_mixed_truth_table(ranges, weights, expected):
    tr = trace_with_hit(ranges, (1e-7,) * len(ranges))
    fields = tr.fields()
    fields["w"] = np.array([weights])
    fields["surface_id"] = np.arange(len(ranges))[None, :]
    pred = BeamModel().predict(RadiometricScan(**fields), EnvConfig())
    assert bool(pred.mixed[0]) is expected


# ------------------------------------------
# test_e19_partial_direct_and_bounce_return_truth_table
# ------------------------------------------
# Mixed-pixel flags with a direct and a bounced return (E19).
def test_e19_partial_direct_and_bounce_return_truth_table():
    # A partially hit K=2 beam with one direct surface is smooth, not mixed.
    tr = RadiometricScan.empty(1, 2)
    fields = tr.fields()
    fields.update(hit=np.array([[True, False]]), r_path=np.array([[1.0, np.nan]]),
                  power=np.array([[1e-7, 0.0]]), normal=np.array([[[-1.0, 0.0], [0.0, 0.0]]]),
                  cos_inc=np.array([[1.0, 0.0]]), object_id=np.array([[0, -1]]),
                  material_id=np.array([[0, -1]]), surface_id=np.array([[4, -1]]))
    pred = BeamModel().predict(RadiometricScan(**fields), EnvConfig())
    assert not pred.mixed[0]
    assert not pred.smooth[0]  # smooth requires every direct sub-ray to hit

    # A separated bounce return participates in mixed classification.
    tr = RadiometricScan.empty(1, 1, J=2)
    fields = tr.fields()
    fields.update(hit=np.array([[True, True]]), r_path=np.array([[1.0, 1.2]]),
                  power=np.array([[1e-7, 1e-7]]), normal=np.array([[[-1.0, 0.0], [-1.0, 0.0]]]),
                  cos_inc=np.ones((1, 2)), object_id=np.zeros((1, 2), int),
                  material_id=np.zeros((1, 2), int), surface_id=np.array([[4, 5]]),
                  bounce=np.array([[False, True]]))
    pred = BeamModel().predict(RadiometricScan(**fields), EnvConfig())
    assert pred.mixed[0]


# ------------------------------------------
# test_e18_wrap_probability_symmetric_and_vanishes_at_high_snr
# ------------------------------------------
# Wrap probabilities are symmetric and vanish at high SNR (E18).
def test_e18_wrap_probability_symmetric_and_vanishes_at_high_snr():
    model = BeamModel()
    high_power = trace_with_hit((1.0,), (1e-3,))
    pred = model.predict(high_power, EnvConfig())
    assert np.allclose(pred.out_prob, pred.out_prob[:, ::-1], atol=1e-15)
    assert pred.out_prob.sum() < 1e-12


# ------------------------------------------
# test_predict_accepts_three_dimensional_trace
# ------------------------------------------
# The model accepts 3D traces.
def test_predict_accepts_three_dimensional_trace():
    tr = RadiometricScan.empty(1, 1, d=3)
    pred = BeamModel().predict(tr, EnvConfig())
    assert pred.B == 1
    assert np.isfinite(pred.P_D).all()


# ------------------------------------------
# test_multibeam_misses_broadcast_regression
# ------------------------------------------
# Integration bug (end-to-end room_asym check): `_mi` divided (n, 6) shifts by (n,) sigmas, which only
# broadcast for n in {1, 6}. Several missed hit-beams with different predictions must give finite,
# per-beam-correct log-likelihoods (equal to evaluating each beam alone).
def test_multibeam_misses_broadcast_regression():
    import numpy as np

    from lidar_sim.core.config import SensorConfig
    from lidar_sim.core.types import RadiometricScan, Scan
    from lidar_sim.models.beam_model import BeamModel
    B = 3
    base = RadiometricScan.empty(B=B, K=1)
    tr = RadiometricScan(**{**base.fields(), "hit": np.ones((B, 1), bool),
        "r_path": np.array([[1.0], [3.0], [5.5]]), "power": np.array([[1e-7], [3e-8], [1e-9]]),
        "normal": np.tile([-1.0, 0.0], (B, 1, 1)), "cos_inc": np.ones((B, 1)),
        "object_id": np.zeros((B, 1), int), "material_id": np.zeros((B, 1), int),
        "surface_id": np.zeros((B, 1), int)})
    m = BeamModel(SensorConfig())
    pred = m.predict(tr, None)
    miss = Scan(detected=np.zeros(B, bool), range=np.full(B, np.nan), amplitude=np.zeros((B, 2)),
                range_raw=np.zeros(B), fire_time=tr.fire_time, truth=tr)
    ll = m.log_likelihood_beams(miss, pred)
    assert np.isfinite(ll).all()
    for i in range(B):
        sub = {k: (v[i:i + 1] if isinstance(v, np.ndarray) else v) for k, v in pred.fields().items()}
        tri = RadiometricScan(**{k: (v[i:i + 1] if isinstance(v, np.ndarray) else v)
                                 for k, v in tr.fields().items()})
        one = Scan(detected=np.zeros(1, bool), range=np.full(1, np.nan), amplitude=np.zeros((1, 2)),
                   range_raw=np.zeros(1), fire_time=tri.fire_time, truth=tri)
        assert ll[i] == m.log_likelihood_beams(one, type(pred)(**sub))[0]
