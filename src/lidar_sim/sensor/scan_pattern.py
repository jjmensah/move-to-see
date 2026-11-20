# ==========================================================
# src/lidar_sim/sensor/scan_pattern.py
# ==========================================================
# Scan patterns (spec §3.1, §4.1–4.2): the ``ScanPattern`` protocol and the 2D ``Planar2DPattern``.
#
# ``Planar2DPattern`` and :func:`in_geometric_fov` are 2D-only (spec §3.2 allowlist).

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

import numpy as np

from lidar_sim.core.config import SensorConfig
from lidar_sim.core.pose import Pose
from lidar_sim.sensor.beam import gauss_hermite_offsets

_DEFAULT_SENSOR = SensorConfig()
_FOV_ANGLE_TOL = 1e-12     # [rad] absorbs atan2 rounding at the inclusive angular boundary (E3)
_FOV_RANGE_TOL = 1e-12     # relative tolerance at the inclusive r_min / r_max boundaries (E3)


# ------------------------------------------
# ScanPattern
# ------------------------------------------
# Beam layout in the sensor frame.
#
# - ``beam_dirs_sensor() -> (B, d)`` unit beam axes.
# - ``fire_times() -> (B,)`` [s] from scan start, non-decreasing.
# - ``subrays(K) -> ((B, K, d) unit dirs, (B, K) weights summing to 1)`` (E4 quadrature).
# - ``beam_angle() -> (B,)`` [rad] 2D-only plotting metadata; never used by physics/models.
@runtime_checkable
class ScanPattern(Protocol):
    dim: int

    def beam_dirs_sensor(self) -> np.ndarray: ...

    def fire_times(self) -> np.ndarray: ...

    def subrays(self, K: int) -> tuple[np.ndarray, np.ndarray]: ...

    def beam_angle(self) -> np.ndarray: ...


# ------------------------------------------
# _ro
# ------------------------------------------
# A read-only float copy of an array.
def _ro(a: np.ndarray) -> np.ndarray:
    a = np.array(a, dtype=float, copy=True)
    a.setflags(write=False)
    return a


# ------------------------------------------
# _u
# ------------------------------------------
# Unit vectors ``[cos a, sin a]`` with shape ``angle.shape + (2,)``.
def _u(angle: np.ndarray) -> np.ndarray:
    return np.stack([np.cos(angle), np.sin(angle)], axis=-1)


# ------------------------------------------
# Planar2DPattern
# ------------------------------------------
# Single-plane rotating-mirror scan (URG-04LX-like), firing CCW from −Θ/2.
#
# Implements (E1) beam angles ``α_i = −Θ/2 + iΔα``, ``Δα = Θ/(B−1)``, sensor-frame axes
# ``u_i = [cos α_i, sin α_i]``; (E2) fire times ``τ_i = iΔα/ω``, ``ω = 2π/T_scan``; (E4) sub-ray
# directions ``u(α_i + δ_k)`` with Gauss–Hermite weights.
#
# Parameters
# cfg : SensorConfig
#     Uses ``fov`` Θ [rad], ``n_beams`` B, ``t_scan`` T_scan [s], ``theta0`` θ₀ [rad].
class Planar2DPattern:
    dim = 2

    def __init__(self, cfg: SensorConfig = _DEFAULT_SENSOR) -> None:
        self.cfg = cfg
        i = np.arange(cfg.n_beams, dtype=float)
        dalpha = cfg.delta_alpha
        self._alpha = _ro(-cfg.fov / 2.0 + i * dalpha)
        omega = 2.0 * math.pi / cfg.t_scan
        self._tau = _ro(i * dalpha / omega)
        self._dirs = _ro(_u(self._alpha))

    # (B,) beam angles α_i [rad] in the sensor frame (E1). 2D plotting metadata only.
    def beam_angle(self) -> np.ndarray:
        return self._alpha

    # (B, 2) unit beam axes ``u_i`` in the sensor frame S (E1).
    def beam_dirs_sensor(self) -> np.ndarray:
        return self._dirs

    # (B,) fire times τ_i [s] from scan start (E2), strictly increasing.
    def fire_times(self) -> np.ndarray:
        return self._tau

    # Sub-ray directions and weights (E4).
    #
    # Returns
    # dirs : (B, K, 2) float
    #     Unit directions ``u(α_i + δ_k)`` in the sensor frame S.
    # w : (B, K) float
    #     Quadrature weights ``w_k`` (identical for every beam), rows sum to 1.
    def subrays(self, K: int) -> tuple[np.ndarray, np.ndarray]:
        delta, w = gauss_hermite_offsets(K, self.cfg.theta0)
        dirs = _u(self._alpha[:, None] + delta[None, :])
        return dirs, np.broadcast_to(w, (self.cfg.n_beams, len(w))).copy()


# ------------------------------------------
# in_geometric_fov
# ------------------------------------------
# Geometric field-of-view membership (inclusive bounds).
#
# Implements (E3): ``p ∈ S_geo`` iff ``|atan2(q_y, q_x)| ≤ Θ/2`` and ``r_min ≤ ‖q‖ ≤ r_max`` with
# ``q = Rᵀ(p − t)`` the point in the sensor frame. Bounds are inclusive up to a rounding tolerance
# of 1e-12 rad in angle and 1e-12 relative in range (so a point constructed exactly on the boundary
# is inside). No line-of-sight test (that is E5 / E3a).
#
# Parameters
# pose : Pose
#     2D sensor pose (sensor → world).
# points : (N, 2) float
#     World-frame points [m].
# cfg : SensorConfig
#     Uses ``fov``, ``r_min``, ``r_max``.
#
# Returns
# (N,) bool
def in_geometric_fov(pose: Pose, points: np.ndarray, cfg: SensorConfig = _DEFAULT_SENSOR) -> np.ndarray:
    p = np.asarray(points, dtype=float).reshape(-1, 2)
    q = (p - pose.t) @ pose.R            # row-vector form of Rᵀ(p − t)
    rng = np.hypot(q[:, 0], q[:, 1])
    ang = np.arctan2(q[:, 1], q[:, 0])
    in_ang = np.abs(ang) <= cfg.fov / 2.0 + _FOV_ANGLE_TOL
    in_rng = (rng >= cfg.r_min * (1.0 - _FOV_RANGE_TOL)) & (rng <= cfg.r_max * (1.0 + _FOV_RANGE_TOL))
    return in_ang & in_rng
