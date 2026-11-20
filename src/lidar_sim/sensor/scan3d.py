# ==========================================================
# src/lidar_sim/sensor/scan3d.py
# ==========================================================
# 3D raster scan pattern and geometric field of view (final-spec §3.2, F3–F6).

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from lidar_sim.core.config import SensorConfig
from lidar_sim.core.pose import Pose
from lidar_sim.sensor.beam import gauss_hermite_offsets

_FOV_ANGLE_TOL = 1e-12     # [rad], inclusive angular bounds (as E3)
_FOV_RANGE_TOL = 1e-12     # relative, inclusive range bounds (as E3)


# ------------------------------------------
# Raster3DConfig
# ------------------------------------------
# Raster geometry: ``n_az`` azimuths over ``fov`` (centred) × ``n_el`` elevations in [el_min, el_max].
@dataclass(frozen=True)
class Raster3DConfig:
    fov: float = 4.0 * math.pi / 3.0          # Θ, azimuth field of view [rad]
    n_az: int = 121
    el_min: float = math.radians(-10.0)       # ε_min [rad]
    el_max: float = math.radians(30.0)        # ε_max [rad]
    n_el: int = 9
    t_scan: float = 0.1                       # frame time [s]
    theta0: float = 5e-3                      # 1/e² half-angle divergence [rad]

    def __post_init__(self) -> None:
        if self.n_az < 2 or self.n_el < 1:
            raise ValueError("need n_az >= 2 and n_el >= 1")
        if not 0.0 < self.fov <= 2.0 * math.pi:
            raise ValueError("fov must lie in (0, 2π]")
        if not -math.pi / 2 < self.el_min <= self.el_max < math.pi / 2:
            raise ValueError("require −π/2 < el_min <= el_max < π/2")
        if self.n_el == 1 and self.el_min != self.el_max:
            raise ValueError("n_el = 1 requires el_min == el_max")
        if self.t_scan <= 0.0 or self.theta0 < 0.0:
            raise ValueError("t_scan > 0 and theta0 >= 0 required")

    @property
    def n_beams(self) -> int:
        return self.n_az * self.n_el

    # Matching :class:`SensorConfig` (``n_beams``, ``fov``, ``t_scan``, ``theta0`` taken from here).
    def sensor_config(self, **overrides) -> SensorConfig:
        return SensorConfig(**{"fov": self.fov, "n_beams": self.n_beams, "t_scan": self.t_scan,
                               "theta0": self.theta0, **overrides})


# ------------------------------------------
# _ro
# ------------------------------------------
# A read-only float copy of an array.
def _ro(a: np.ndarray) -> np.ndarray:
    a = np.array(a, dtype=float, copy=True)
    a.setflags(write=False)
    return a


# ------------------------------------------
# Raster3DPattern
# ------------------------------------------
# Rows of constant elevation fired in order; beam index ``i = e·n_az + c``.
#
# Implements (F3) beam axes ``u_i = (cos ε cos α, cos ε sin α, sin ε)``, (F4) fire times
# ``τ_i = i·T_scan/B`` and (F5) sub-rays ``normalize(u + tan δ_p e_α + tan δ_q e_ε)`` with weights
# ``w_p w_q`` from the Gauss–Hermite rule of (E4).
class Raster3DPattern:
    dim = 3

    def __init__(self, cfg: Raster3DConfig = Raster3DConfig()) -> None:
        self.cfg = cfg
        az = -cfg.fov / 2.0 + np.arange(cfg.n_az) * cfg.fov / (cfg.n_az - 1)
        el = (np.full(1, cfg.el_min) if cfg.n_el == 1
              else cfg.el_min + np.arange(cfg.n_el) * (cfg.el_max - cfg.el_min) / (cfg.n_el - 1))
        A, E = np.meshgrid(az, el)                                 # (n_el, n_az): rows of constant elevation
        self._az, self._el = _ro(A.reshape(-1)), _ro(E.reshape(-1))
        ca, sa, ce, se = np.cos(self._az), np.sin(self._az), np.cos(self._el), np.sin(self._el)
        self._u = _ro(np.stack([ce * ca, ce * sa, se], axis=-1))
        self._e_az = _ro(np.stack([-sa, ca, np.zeros_like(sa)], axis=-1))
        self._e_el = _ro(np.stack([-se * ca, -se * sa, ce], axis=-1))
        B = cfg.n_beams
        self._tau = _ro(np.arange(B) * cfg.t_scan / B)

    # (B,) azimuths α [rad] (plot metadata only).
    def beam_angle(self) -> np.ndarray:
        return self._az

    # (B,) elevations ε [rad] (plot metadata only).
    def beam_elevation(self) -> np.ndarray:
        return self._el

    # (B, 3) unit beam axes in the sensor frame (F3).
    def beam_dirs_sensor(self) -> np.ndarray:
        return self._u

    # (B,) fire times [s] from scan start (F4), strictly increasing.
    def fire_times(self) -> np.ndarray:
        return self._tau

    # Sub-ray directions ``(B, K, 3)`` and weights ``(B, K)`` (F5); ``K = m²``, order ``j = p·m + q``.
    def subrays(self, K: int) -> tuple[np.ndarray, np.ndarray]:
        m = math.isqrt(int(K))
        if int(K) != K or m * m != K or K < 1:
            raise ValueError(f"K must be a perfect square >= 1 for a 3D pattern, got {K}")
        delta, w = gauss_hermite_offsets(m, self.cfg.theta0)
        tp, tq = np.meshgrid(np.tan(delta), np.tan(delta), indexing="ij")     # (m, m): p rows, q columns
        wk = (w[:, None] * w[None, :]).reshape(-1)
        dirs = (self._u[:, None, :] + tp.reshape(-1)[None, :, None] * self._e_az[:, None, :]
                + tq.reshape(-1)[None, :, None] * self._e_el[:, None, :])
        dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
        return dirs, np.broadcast_to(wk / wk.sum(), (self.cfg.n_beams, K)).copy()


# ------------------------------------------
# in_geometric_fov_3d
# ------------------------------------------
# Geometric field-of-view membership in 3D (F6), inclusive bounds; no line-of-sight test.
#
# Parameters
# pose : Pose
#     3D sensor pose (sensor → world).
# points : (N, 3) float
#     World-frame points [m].
#
# Returns
# (N,) bool
def in_geometric_fov_3d(pose: Pose, points: np.ndarray, cfg: Raster3DConfig, r_min: float, r_max: float) -> np.ndarray:
    p = np.asarray(points, dtype=float).reshape(-1, 3)
    s = (p - pose.t) @ pose.R                                   # Rᵀ(p − t) in row-vector form
    rng = np.linalg.norm(s, axis=1)
    az = np.arctan2(s[:, 1], s[:, 0])
    with np.errstate(invalid="ignore", divide="ignore"):
        el = np.arcsin(np.clip(np.where(rng > 0, s[:, 2] / np.where(rng > 0, rng, 1.0), 0.0), -1.0, 1.0))
    in_az = np.abs(az) <= cfg.fov / 2.0 + _FOV_ANGLE_TOL
    in_el = (el >= cfg.el_min - _FOV_ANGLE_TOL) & (el <= cfg.el_max + _FOV_ANGLE_TOL)
    in_rng = (rng >= r_min * (1.0 - _FOV_RANGE_TOL)) & (rng <= r_max * (1.0 + _FOV_RANGE_TOL))
    return in_az & in_el & in_rng & (rng > 0.0)                # the sensor origin itself is outside (F6)
