# ==========================================================
# src/lidar_sim/models/detection.py
# ==========================================================
# Rice detection and CFAR threshold helpers (E15–E16).

from __future__ import annotations

import numpy as np
from scipy.stats import rice


# ------------------------------------------
# cfar_gamma
# ------------------------------------------
# Return CFAR amplitude threshold γ [A] for ambient power P_amb [W].
#
# Parameters
# noise : NoiseConfig
#     Receiver noise and responsivity parameters.
# P_amb : float
#     Ambient optical power [W].
# N : int
#     Number of equivalent-time samples.
#
# Returns
# float
#     Amplitude threshold [A].
#
# Implements (E15).
def cfar_gamma(noise, P_amb: float, N: int) -> float:
    s_b = np.sqrt(2.0 / N) * np.sqrt(noise.sigma_th**2 + noise.kappa_sh * noise.responsivity * P_amb)
    return float(s_b * np.sqrt(-2.0 * np.log(noise.P_FA0)))


# ------------------------------------------
# pfa
# ------------------------------------------
# Return noise-only false alarm probability for quadrature scale s [A].
#
# Parameters
# s : ndarray, shape (...,)
#     Quadrature standard deviation [A].
# gamma : float or ndarray
#     Detection threshold [A].
#
# Returns
# ndarray
#     False alarm probabilities, shape broadcast from inputs.
#
# Implements (E16).
def pfa(s: np.ndarray, gamma: float | np.ndarray) -> np.ndarray:
    return np.exp(-np.asarray(gamma) ** 2 / (2.0 * np.asarray(s) ** 2))


# ------------------------------------------
# rice_pd
# ------------------------------------------
# Return Rice detection probability for resultant amplitude A [A].
#
# Parameters
# A : ndarray, shape (...,)
#     Noise-free phasor magnitude [A].
# s : ndarray, shape (...,)
#     Quadrature standard deviation [A].
# gamma : float or ndarray
#     Detection threshold [A].
#
# Returns
# ndarray
#     Detection probabilities, broadcast shape.
#
# Implements (E16).
def rice_pd(A: np.ndarray, s: np.ndarray, gamma: float | np.ndarray) -> np.ndarray:
    a, scale = np.broadcast_arrays(np.asarray(A, float), np.asarray(s, float))
    out = rice.sf(gamma, b=np.divide(a, scale, out=np.zeros_like(a), where=scale > 0), scale=scale)
    return np.where(a == 0, pfa(scale, gamma), out)
