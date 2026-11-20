# ==========================================================
# src/lidar_sim/sensor/beam.py
# ==========================================================
# Beam divergence as sub-ray quadrature (spec §4.2, ADR-0003).
#
# Implements (E4): Gauss–Hermite nodes/weights for the Gaussian angular profile
# ``I(δ) ∝ exp(−2δ²/θ₀²)``. In 2D the offsets are angles added to the beam axis; the 3D extension uses a
# 2D cubature over the same 1D rule.

from __future__ import annotations

import math

import numpy as np


# ------------------------------------------
# gauss_hermite_offsets
# ------------------------------------------
# Angular sub-ray offsets and quadrature weights of a Gaussian beam.
#
# Implements (E4): with Gauss–Hermite nodes/weights ``(x_k, h_k)`` for ``∫ e^{−x²} g(x) dx``,
# ``δ_k = θ₀ x_k / √2`` and ``w_k = h_k / √π`` so that ``Σ_k w_k = 1``.
#
# Parameters
# K : int
#     Number of sub-rays (>= 1). ``K = 1`` gives ``δ = [0]``, ``w = [1]`` (ideal ray caster).
# theta0 : float
#     1/e² half-angle divergence θ₀ [rad] (>= 0).
#
# Returns
# delta : (K,) float
#     Angular offsets [rad], ascending and exactly antisymmetric (``δ = −δ[::-1]``).
# w : (K,) float
#     Quadrature weights, symmetric, non-negative, summing to 1 (renormalized to absorb rounding).
def gauss_hermite_offsets(K: int, theta0: float) -> tuple[np.ndarray, np.ndarray]:
    if int(K) != K or K < 1:
        raise ValueError(f"K must be an integer >= 1, got {K}")
    if not (math.isfinite(theta0) and theta0 >= 0.0):
        raise ValueError(f"theta0 must be finite and >= 0, got {theta0}")
    x, h = np.polynomial.hermite.hermgauss(int(K))
    delta = theta0 * x / math.sqrt(2.0)
    delta = 0.5 * (delta - delta[::-1])        # exact antisymmetry (removes rounding asymmetry)
    w = h / math.sqrt(math.pi)
    w = 0.5 * (w + w[::-1])
    w = w / w.sum()
    return delta, w
