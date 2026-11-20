# ==========================================================
# src/lidar_sim/core/constants.py
# ==========================================================
# Physical and sensor constants (SI). See docs/design/sensor-spec.md §4.

C_LIGHT: float = 299_792_458.0
"""Speed of light in vacuum [m/s]."""

F1: float = 46.55e6
"""First modulation frequency [Hz] (URG-04LX, Heiden et al.)."""

F2: float = 53.2e6
"""Second modulation frequency [Hz]; F1 = 7·ΔF, F2 = 8·ΔF exactly (ADR-0006)."""

N_SAMPLES: int = 30
"""Equivalent-time samples per frequency (E11)."""

WAVELENGTH: float = 785e-9
"""Laser wavelength [m] (used by the visibility model, E8)."""

KOSCHMIEDER: float = 3.912
"""−ln(0.02): 2 % contrast-threshold constant (E8)."""

WRAP_K_MAX: int = 3
"""Wrap-outlier truncation |k| <= 3 (E18); BeamPrediction.out_prob has 2·WRAP_K_MAX columns."""
