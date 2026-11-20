# ==========================================================
# src/lidar_sim/world/environment.py
# ==========================================================
# Environment presets (spec §4.8): visibility (fog, E8) and ambient light (E11, E15).
#
# Dimension-agnostic (spec §3.2). Ambient light enters the pipeline only via ``EnvConfig.P_amb``, which
# ``LidarSensor.trace`` copies into ``RadiometricScan.P_amb``.

from __future__ import annotations

from dataclasses import replace

from lidar_sim.core.config import EnvConfig, NoiseConfig

CLEAR = EnvConfig()
"""Clear air: V = 20 km, no ambient light."""

FOG_LIGHT = EnvConfig(visibility_m=200.0)
"""Light fog: V = 200 m."""

FOG_DENSE = EnvConfig(visibility_m=20.0)
"""Dense fog: V = 20 m."""

_DEFAULT_NOISE = NoiseConfig()


# ------------------------------------------
# sunlight
# ------------------------------------------
# Environment with ambient light ``P_amb = fraction · noise.P_amb_sun`` [W] (ADR-0004).
#
# Parameters
# fraction : float
#     Fraction (>= 0) of the calibrated "bright sunlight" ambient power.
# base : EnvConfig
#     Environment whose visibility and wavelength are kept.
# noise : NoiseConfig
#     Supplies ``P_amb_sun`` [W].
def sunlight(fraction: float = 1.0, base: EnvConfig = CLEAR, noise: NoiseConfig = _DEFAULT_NOISE) -> EnvConfig:
    if not fraction >= 0.0:
        raise ValueError(f"fraction must be >= 0, got {fraction}")
    return replace(base, P_amb=fraction * noise.P_amb_sun)
