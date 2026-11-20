# ==========================================================
# src/lidar_sim/physics/radiometry.py
# ==========================================================
# Direct-return radiometry of extended Lambertian targets (spec §4.3). Dimension-agnostic.
#
# All functions operate on ``(..., d)`` vectors / ``(...)`` arrays and broadcast like NumPy.

from __future__ import annotations

import math

import numpy as np

from lidar_sim.physics.atmosphere import transmission


# ------------------------------------------
# incidence_cos
# ------------------------------------------
# Incidence cosine ``cos θ = clip(−d·n, 0, 1)``.
#
# Implements (E6).
#
# Parameters
# dirs : (..., d) float
#     Unit ray directions (any frame, same as ``normals``).
# normals : (..., d) float
#     Unit surface normals oriented towards the ray (``n·d ≤ 0``).
#
# Returns
# (...) float in [0, 1].
def incidence_cos(dirs: np.ndarray, normals: np.ndarray) -> np.ndarray:
    c = -np.sum(np.asarray(dirs, float) * np.asarray(normals, float), axis=-1)
    return np.clip(c, 0.0, 1.0)


# ------------------------------------------
# lambertian_power
# ------------------------------------------
# Received optical power of one sub-ray return from a Lambertian surface [W].
#
# Implements (E7): ``P = P₀ · w · (ρ/π) · cos θ · (A_rx / r²) · T(r)² · η`` with ``T`` from (E8).
# All arguments broadcast.
#
# Parameters
# P0 : float
#     Emitted optical power [W].
# w : array_like (...)
#     Sub-ray quadrature weight (E4), dimensionless.
# rho : array_like (...)
#     Lambertian reflectivity in [0, 1].
# cos_inc : array_like (...)
#     Incidence cosine (E6).
# r : array_like (...)
#     One-way range [m] (> 0).
# A_rx : float
#     Receiver aperture area [m²].
# eta : float
#     Optical efficiency.
# sigma_ext : float
#     Extinction coefficient [1/m] (E8).
#
# Returns
# ndarray (...)
#     Power [W].
def lambertian_power(P0, w, rho, cos_inc, r, A_rx, eta, sigma_ext) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    T = transmission(r, sigma_ext)
    return (P0 * np.asarray(w, float) * (np.asarray(rho, float) / math.pi) * np.asarray(cos_inc, float)
            * (A_rx / r**2) * T**2 * eta)
