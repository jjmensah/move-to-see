# ==========================================================
# src/lidar_sim/physics/specular.py
# ==========================================================
# Optional single specular bounce (spec §4.4, feature flag ``specular``). Dimension-agnostic.

from __future__ import annotations

import numpy as np

from lidar_sim.physics.radiometry import lambertian_power


# ------------------------------------------
# reflect
# ------------------------------------------
# Mirror reflection of a direction: ``d' = d − 2(d·n)n`` (E9).
#
# Parameters
# d : (..., d) float
#     Unit incident directions.
# n : (..., d) float
#     Unit mirror normals (either orientation).
#
# Returns
# (..., d) float
#     Unit reflected directions.
def reflect(d: np.ndarray, n: np.ndarray) -> np.ndarray:
    d = np.asarray(d, dtype=float)
    n = np.asarray(n, dtype=float)
    return d - 2.0 * np.sum(d * n, axis=-1, keepdims=True) * n


# ------------------------------------------
# bounce_power
# ------------------------------------------
# Power of a return via one mirror bounce onto a diffuse surface [W].
#
# Implements (E9): ``P = P₀ w ρ_m² (ρ₂/π) cos θ₂ (A_rx / r_path²) T(r_path)² η`` with
# ``r_path = r₁ + r₂`` (image-source argument: the return retraces the path, so ρ_m enters twice).
#
# Parameters
# P0 : float
#     Emitted optical power [W].
# w : array_like (...)
#     Sub-ray quadrature weight.
# rho_m : array_like (...)
#     Specular reflectivity of the mirror.
# rho2 : array_like (...)
#     Lambertian reflectivity of the second (diffuse) surface.
# cos2 : array_like (...)
#     Incidence cosine at the second surface (E6).
# r_path : array_like (...)
#     One-way optical path ``r₁ + r₂`` [m].
# A_rx, eta, sigma_ext : float
#     Aperture [m²], optical efficiency, extinction [1/m] (E8).
def bounce_power(P0, w, rho_m, rho2, cos2, r_path, A_rx, eta, sigma_ext) -> np.ndarray:
    rho_m = np.asarray(rho_m, dtype=float)
    return rho_m**2 * lambertian_power(P0, w, rho2, cos2, r_path, A_rx, eta, sigma_ext)
