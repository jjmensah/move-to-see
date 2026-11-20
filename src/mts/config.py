# ==========================================================
# src/mts/config.py
# ==========================================================
# Frozen configuration and defaults (final-spec §3.3, §4, §6, §7, §9). All values are SI.

from __future__ import annotations

import math
from dataclasses import dataclass

from lidar_sim.core.config import EnvConfig, ModelConfig, NoiseConfig, SensorConfig
from lidar_sim.models.beam_model import BeamModel
from lidar_sim.sensor.lidar import LidarSensor
from lidar_sim.sensor.scan3d import Raster3DConfig, Raster3DPattern
from lidar_sim.world.environment import sunlight
from lidar_sim.world.materials import BLACK, WHITE


# ------------------------------------------
# RobotConfig
# ------------------------------------------
# Robot limits, step sizes and path-following gains (F7, F8, F19).
@dataclass(frozen=True)
class RobotConfig:
    dt: float = 0.5                # decision step Δt [s]
    substep: float = 0.02          # RK4 substep δt [s]
    a_max: float = 1.0             # [m/s²]
    w_max: float = 1.0             # [rad/s]
    v_max: float = 1.0             # [m/s]
    v_rough: float = 0.3           # speed limit on rough terrain [m/s] (F8)
    radius: float = 0.30           # robot disc radius ρ_r [m]
    margin: float = 0.05           # clearance margin m_c [m]
    h_s: float = 0.40              # sensor height [m] (F1)
    lookahead: float = 0.6         # path-following look-ahead [m] (F19)
    k_heading: float = 2.0         # heading gain [1/s] (F19)
    brake_margin: float = 0.05     # m_s [m] (F19)
    goal_tol: float = 0.15         # arrival tolerance [m] (F19)

    def __post_init__(self) -> None:
        n = self.dt / self.substep
        if abs(n - round(n)) > 1e-9 or round(n) < 1:
            raise ValueError("dt must be an integer multiple of substep")

    @property
    def n_sub(self) -> int:
        return round(self.dt / self.substep)


# ------------------------------------------
# BeliefConfig
# ------------------------------------------
# Particle-filter settings (F10 to F12).
@dataclass(frozen=True)
class BeliefConfig:
    n_particles: int = 2000
    ess_frac: float = 0.5          # CESS / ESS threshold as a fraction of N (F11)
    a: float = 0.98                # Liu–West shrinkage (F12)
    sigma_j: float = 0.003         # kernel floor σ_j [m] (F12)
    min_increment: float = 1e-6    # minimum tempering increment
    chunk: int = 200_000           # max (hypothesis, beam) pairs per predictor chunk


# ------------------------------------------
# PlannerConfig
# ------------------------------------------
# Planner settings: support size, travel penalty, lattice and switching margin (F17, F18).
@dataclass(frozen=True)
class PlannerConfig:
    M: int = 24                    # support hypotheses (F17)
    lambda_t: float = 0.05         # travel-time penalty [nat/s] (F18)
    lattice: float = 0.5           # viewpoint lattice pitch [m]
    radius: float = 4.0            # viewpoints within this distance of the target keep-out [m]
    cell: float = 0.1              # navigation grid cell [m]
    sigma_ref: float = 0.012       # constant range noise of the physics-blind model [m]
    switch_margin: float = 0.05   # goal persistence threshold [nat] (§7.4)


# ------------------------------------------
# Condition
# ------------------------------------------
# One test condition: environment and target material.
@dataclass(frozen=True)
class Condition:
    name: str
    env: EnvConfig
    target_material: int


RASTER = Raster3DConfig(fov=4.0 * math.pi / 3.0, n_az=121, el_min=math.radians(-10.0),
                        el_max=math.radians(40.0), n_el=11, t_scan=0.1, theta0=5e-3)
SENSOR: SensorConfig = RASTER.sensor_config(n_subrays=9)
NOISE = NoiseConfig()
MODEL = ModelConfig()
_CLEAR = EnvConfig()
CONDITIONS = {
    "C1": Condition("C1", _CLEAR, BLACK),                                       # clear, dark, black target
    "C2": Condition("C2", EnvConfig(visibility_m=20.0), BLACK),                 # fog V = 20 m
    "C3": Condition("C3", sunlight(1.0, _CLEAR, NOISE), BLACK),                 # 1 sun
    "C4": Condition("C4", _CLEAR, WHITE),                                       # control: white target
}
T_STEPS = 60
ACC_TOL = 0.10
DWELL = 5


# ------------------------------------------
# make_sensor
# ------------------------------------------
# 3D LiDAR with the default raster and ``n_subrays`` sub-rays (a perfect square).
def make_sensor(n_subrays: int = 9) -> LidarSensor:
    return LidarSensor(RASTER.sensor_config(n_subrays=n_subrays), NOISE, pattern=Raster3DPattern(RASTER))


# ------------------------------------------
# make_model
# ------------------------------------------
# The analytic beam model with the default raster and the given number of sub-rays.
def make_model(n_subrays: int = 9) -> BeamModel:
    return BeamModel(RASTER.sensor_config(n_subrays=n_subrays), NOISE, MODEL)
