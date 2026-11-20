# ==========================================================
# src/lidar_sim/geometry/base.py
# ==========================================================
# Scene protocol (spec §3.1). Implementations: ``scene2d.Scene2D`` (midterm), ``scene3d`` (final).

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from lidar_sim.core.types import HitBatch


# ------------------------------------------
# Scene
# ------------------------------------------
# Ray-castable world geometry.
#
# ``intersect`` returns, per ray, the **nearest** valid hit with ``r > r_eps`` and no upper range gate
# (E5). ``t_scene`` (N,) [s] is the per-ray time at which moving objects are posed (E2 scan skew).
# Normals are unit and oriented so ``normal · dir <= 0``; ids are −1 where there is no hit.
@runtime_checkable
class Scene(Protocol):
    dim: int

    def intersect(self, origins: np.ndarray, dirs: np.ndarray,
                  t_scene: np.ndarray) -> HitBatch: ...
