# ==========================================================
# tests/property/test_models_props.py
# ==========================================================
# Property tests of the detection and likelihood model.

import numpy as np
from scipy.integrate import quad

from lidar_sim.core.config import EnvConfig, NoiseConfig
from lidar_sim.core.types import RadiometricScan, Scan
from lidar_sim.models.beam_model import BeamModel
from lidar_sim.models.detection import cfar_gamma, rice_pd


# ------------------------------------------
# test_pd_monotone_amplitude_and_ambient
# ------------------------------------------
# Detection probability rises with amplitude and falls with ambient light.
def test_pd_monotone_amplitude_and_ambient():
    noise = NoiseConfig()
    amps = np.linspace(0, 5e-6, 20)
    s0 = np.sqrt(2 / 30) * noise.sigma_th
    p_amp = rice_pd(amps, np.full_like(amps, s0), cfar_gamma(noise, 0, 30))
    assert np.all(np.diff(p_amp) >= -1e-15)
    ambients = np.array([0.0, 1e-9, 1e-7, 1e-5])
    s = np.sqrt(2 / 30) * np.sqrt(noise.sigma_th**2 + noise.kappa_sh * noise.responsivity * ambients)
    gammas = np.array([cfar_gamma(noise, x, 30) for x in ambients])
    p_amb = rice_pd(np.full_like(s, 2e-6), s, gammas)
    assert np.all(np.diff(p_amb) <= 1e-15)


# ------------------------------------------
# test_likelihood_normalization_random_nohit_predictions
# ------------------------------------------
# The likelihood sums to one for random beams that hit nothing.
def test_likelihood_normalization_random_nohit_predictions():
    rng = np.random.default_rng(4240)
    model = BeamModel()
    for ambient in 10.0 ** rng.uniform(-12, -4, 4):
        tr = RadiometricScan.empty(1, 1, P_amb=float(ambient))
        pred = model.predict(tr, EnvConfig())
        def make(detected, value=np.nan, tr=tr):
            return Scan(np.array([detected]), np.array([value]), np.zeros((1, 2)), np.zeros(1),
                        tr.fire_time, tr)
        mass = quad(lambda z, pred=pred: np.exp(model.log_likelihood_beams(make(True, z), pred)[0]),
                    model.sensor.r_min, model.sensor.r_max, epsabs=1e-12)[0]
        not_detected = np.exp(model.log_likelihood_beams(make(False), pred)[0])
        assert abs(mass + not_detected - 1.0) < 1e-9
