# ==========================================================
# tests/unit/test_models_detection.py
# ==========================================================
# Unit tests of the detection model.

import numpy as np
from scipy.stats import rice

from lidar_sim.core.config import NoiseConfig
from lidar_sim.models.detection import cfar_gamma, pfa, rice_pd


# ------------------------------------------
# test_cfar_and_rice_zero_signal_independent_of_ambient
# ------------------------------------------
# With no signal, detection equals the false-alarm rate at any ambient level.
def test_cfar_and_rice_zero_signal_independent_of_ambient():
    noise = NoiseConfig()
    for ambient in (0.0, 1e-8, 1e-3):
        gamma = cfar_gamma(noise, ambient, 30)
        s = np.sqrt(2 / 30) * np.sqrt(noise.sigma_th**2 + noise.kappa_sh * noise.responsivity * ambient)
        assert np.isclose(rice_pd(np.array([0.0]), np.array([s]), gamma)[0], noise.P_FA0, rtol=1e-14)
        assert np.isclose(pfa(np.array([s]), gamma)[0], noise.P_FA0)
        assert np.isclose(rice_pd(np.array([2*s]), np.array([s]), gamma)[0], rice.sf(gamma, 2, scale=s))
