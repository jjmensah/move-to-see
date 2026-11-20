# ==========================================================
# src/lidar_sim/estimation/jacobian2d.py
# ==========================================================
# Planar pose parameterization and analytic range Jacobian (E23).

from __future__ import annotations

import numpy as np

from lidar_sim.core.pose import Pose
from lidar_sim.core.types import BeamPrediction, RadiometricScan


# ------------------------------------------
# Planar2DParam
# ------------------------------------------
# Minimal planar pose coordinates ``(x, y, yaw)`` and E23 range derivative.
class Planar2DParam:
    dof = 3
    is_rotation = (False, False, True)

    def to_pose(self, xi: np.ndarray) -> Pose:
        return Pose.from_xi(xi)

    def from_pose(self, pose: Pose) -> np.ndarray:
        return pose.xi()

    # Return ``(B,3)`` derivatives of expected range; non-smooth rows are NaN.
    def range_jacobian(self, trace: RadiometricScan, pred: BeamPrediction) -> np.ndarray:
        if not isinstance(trace.pose, Pose):
            raise ValueError("range_jacobian requires a static Pose")
        if trace.dim != 2:
            raise ValueError("Planar2DParam requires a 2D trace")
        out = np.full((trace.B, 3), np.nan)
        for i in np.flatnonzero(pred.smooth):
            active = trace.hit[i, :trace.K] & (trace.power[i, :trace.K] > 0)
            if not active.any():
                continue
            weights = np.where(active, trace.w[i], -np.inf)
            k = int(np.argmax(weights))
            d = trace.dir_world[i, k]
            n = trace.normal[i, k]
            nd = float(n @ d)
            if abs(nd) < 1e-12:
                continue
            r = trace.r_path[i, k]
            d_perp = np.array([-d[1], d[0]])
            out[i, :2] = -n / nd
            out[i, 2] = -r * float(n @ d_perp) / nd
        return out
