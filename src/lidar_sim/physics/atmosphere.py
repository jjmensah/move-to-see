# ==========================================================
# src/lidar_sim/physics/atmosphere.py
# ==========================================================
# Homogeneous atmospheric attenuation (spec §4.3, Kim et al. visibility model). Dimension-agnostic.
#
# Implements (E8): ``σ_ext = (3.912 / V) · (λ / 550 nm)^(−q(V_km))``, ``T(r) = exp(−σ_ext r)``.
# No fog backscatter returns (limitation, spec §1 non-goals).

from __future__ import annotations

import numpy as np

from lidar_sim.core.constants import KOSCHMIEDER

_LAMBDA_REF = 550e-9      # reference wavelength of the visibility definition [m]


# ------------------------------------------
# kim_q
# ------------------------------------------
# Kim size-distribution exponent ``q`` as a function of visibility (E8).
#
# ``q = 1.6`` (V_km > 50); ``1.3`` (6 < V_km ≤ 50); ``0.16 V_km + 0.34`` (1 < V_km ≤ 6);
# ``V_km − 0.5`` (0.5 < V_km ≤ 1); ``0`` (V_km ≤ 0.5).
#
# Parameters
# visibility_km : float or ndarray
#     Meteorological visibility [km] (note: km, the model's native unit).
#
# Returns
# float or ndarray
#     Dimensionless exponent, same shape as the input (float for scalar input).
def kim_q(visibility_km: float | np.ndarray) -> float | np.ndarray:
    v = np.asarray(visibility_km, dtype=float)
    q = np.select([v > 50.0, v > 6.0, v > 1.0, v > 0.5],
                  [1.6, 1.3, 0.16 * v + 0.34, v - 0.5], default=0.0)
    return float(q) if q.ndim == 0 else q


# ------------------------------------------
# extinction
# ------------------------------------------
# Extinction coefficient ``σ_ext`` [1/m] (E8).
#
# Parameters
# visibility_m : float
#     Meteorological visibility V [m] (> 0); ``V_km = V/1000`` selects ``q``.
# wavelength : float
#     Laser wavelength λ [m] (> 0).
def extinction(visibility_m: float, wavelength: float) -> float:
    V = float(visibility_m)
    lam = float(wavelength)
    if not (V > 0.0 and np.isfinite(V)) or not (lam > 0.0 and np.isfinite(lam)):
        raise ValueError("visibility_m and wavelength must be finite and > 0")
    q = kim_q(V / 1000.0)
    return KOSCHMIEDER / V * (lam / _LAMBDA_REF) ** (-q)


# ------------------------------------------
# transmission
# ------------------------------------------
# One-way transmission ``T(r) = exp(−σ_ext r)`` (E8).
#
# Parameters
# r : ndarray, any shape (...)
#     Path length [m].
# sigma_ext : float
#     Extinction coefficient [1/m].
#
# Returns
# ndarray (...)
#     Dimensionless transmission in (0, 1].
def transmission(r: np.ndarray, sigma_ext: float) -> np.ndarray:
    return np.exp(-float(sigma_ext) * np.asarray(r, dtype=float))
