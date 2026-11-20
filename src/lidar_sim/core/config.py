# ==========================================================
# src/lidar_sim/core/config.py
# ==========================================================
# Frozen configuration dataclasses with spec defaults (docs/design/sensor-spec.md §4, §7).
#
# All values are SI. The only non-SI entry points are :func:`deg2rad` and :func:`km_to_m`.

from __future__ import annotations

import math
from dataclasses import dataclass

from lidar_sim.core import constants as K


# ------------------------------------------
# deg2rad
# ------------------------------------------
# Degrees → radians (config convenience only).
def deg2rad(deg: float) -> float:
    return deg * math.pi / 180.0


# ------------------------------------------
# km_to_m
# ------------------------------------------
# Kilometres → metres (config convenience only).
def km_to_m(km: float) -> float:
    return km * 1000.0


# ------------------------------------------
# SensorConfig
# ------------------------------------------
# Scan geometry, timing, modulation and beam parameters (E1–E5, E10–E15).
@dataclass(frozen=True)
class SensorConfig:
    fov: float = 4.0 * math.pi / 3.0      # Θ [rad]
    n_beams: int = 682                    # B
    r_min: float = 0.02                   # [m]
    r_max: float = 5.6                    # [m]
    t_scan: float = 0.1                   # one mirror revolution [s]
    f1: float = K.F1                      # [Hz]
    f2: float = K.F2                      # [Hz]
    n_samples: int = K.N_SAMPLES          # N
    theta0: float = 5e-3                  # 1/e² half-angle divergence [rad] (ADR-0003)
    n_subrays: int = 5                    # K
    r_eps: float = 1e-9                   # self-intersection guard [m]
    specular: bool = False                # E9 flag
    range_walk: bool = False              # optional intensity-dependent bias
    b0: float = 0.0                       # range-walk gain [m]
    P_ref: float = 1e-7                   # range-walk reference power [W]

    def __post_init__(self) -> None:
        if self.n_samples < 3:
            raise ValueError("n_samples must be >= 3 (orthogonal regressors, E12)")
        if not 0.0 < self.r_min < self.r_max:
            raise ValueError("require 0 < r_min < r_max")
        if self.n_beams < 2 or self.n_subrays < 1:
            raise ValueError("need n_beams >= 2 and n_subrays >= 1")
        if not 0.0 < self.f1 < self.f2:
            raise ValueError("require 0 < f1 < f2")
        if not 0.0 < self.fov <= 2.0 * math.pi:
            raise ValueError("fov must lie in (0, 2π]")
        if self.t_scan <= 0.0 or self.theta0 < 0.0 or self.r_eps <= 0.0 or self.P_ref <= 0.0:
            raise ValueError("t_scan > 0, theta0 >= 0, r_eps > 0, P_ref > 0 required")
        if not all(math.isfinite(v) for v in (self.fov, self.r_min, self.r_max, self.t_scan, self.f1,
                                               self.f2, self.theta0, self.b0, self.r_eps, self.P_ref)):
            raise ValueError("sensor parameters must be finite")

    # Angular beam spacing Δα = Θ/(B−1) [rad] (E1).
    @property
    def delta_alpha(self) -> float:
        return self.fov / (self.n_beams - 1)

    # Unambiguous interval at f1, c/(2 f1) [m] (E13).
    @property
    def L1(self) -> float:
        return K.C_LIGHT / (2.0 * self.f1)

    # Unambiguous interval at f2, c/(2 f2) [m] (E13).
    @property
    def L2(self) -> float:
        return K.C_LIGHT / (2.0 * self.f2)

    # Coarse (beat) unambiguous interval c/(2 Δf) [m] (E14).
    @property
    def Lc(self) -> float:
        return K.C_LIGHT / (2.0 * (self.f2 - self.f1))


# ------------------------------------------
# NoiseConfig
# ------------------------------------------
# Radiometric gains and noise constants (E7, E11, E15). Calibrated per ADR-0004.
@dataclass(frozen=True)
class NoiseConfig:
    P0: float = 0.030                     # emitted optical power [W]
    A_rx: float = math.pi * 0.01**2       # receiver aperture [m²]
    eta: float = 0.8                      # optical efficiency
    responsivity: float = 25.0            # 𝓡 incl. APD gain [A/W]
    mod_depth: float = 1.0                # m
    sigma_th: float = 3.283278644238764e-07   # thermal/electronics noise [A] (ADR-0004)
    kappa_sh: float = 3.197198847453723e-08   # lumped shot-noise factor [A] (ADR-0004)
    P_FA0: float = 1e-4                   # design false-alarm rate, no signal/ambient
    P_amb_sun: float = 2.019618716475848e-06  # "bright sunlight" ambient [W] (ADR-0004)

    def __post_init__(self) -> None:
        if min(self.P0, self.A_rx, self.eta, self.responsivity, self.sigma_th) <= 0.0:
            raise ValueError("gains and sigma_th must be positive")
        if not 0.0 < self.mod_depth <= 1.0:
            raise ValueError("mod_depth must be in (0, 1]")
        if not 0.0 < self.P_FA0 < 1.0 or self.kappa_sh < 0.0:
            raise ValueError("invalid P_FA0 or kappa_sh")


# ------------------------------------------
# EnvConfig
# ------------------------------------------
# Environment: meteorological visibility, ambient light, wavelength (E8, E11).
@dataclass(frozen=True)
class EnvConfig:
    visibility_m: float = 20000.0         # V [m] (clear air)
    P_amb: float = 0.0                    # ambient optical power at detector [W]
    wavelength: float = K.WAVELENGTH      # λ [m]

    def __post_init__(self) -> None:
        if self.visibility_m <= 0.0 or self.P_amb < 0.0 or self.wavelength <= 0.0:
            raise ValueError("visibility_m > 0, P_amb >= 0, wavelength > 0 required")


# ------------------------------------------
# ModelConfig
# ------------------------------------------
# Analytic-model and estimation settings (E18–E25).
@dataclass(frozen=True)
class ModelConfig:
    eps_floor: float = 1e-3               # robustness floor ε (E20)
    mixed_tol: float = 0.05               # mixed-pixel range spread threshold [m] (E19)
    w_min: float = 0.01                   # sub-ray weight threshold for `mixed` (E19)
    fd_step_t: float = 1e-5               # finite-difference step, translation [m]
    fd_step_psi: float = 1e-6             # finite-difference step, yaw [rad]
