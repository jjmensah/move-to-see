# ==========================================================
# src/lidar_sim/estimation/fisher.py
# ==========================================================
# Pose Fisher information and Cramér–Rao bound (E22, E24).

from __future__ import annotations

from typing import Literal, Protocol

import numpy as np

from lidar_sim.core.types import BeamPrediction
from lidar_sim.estimation.jacobian2d import Planar2DParam


# ------------------------------------------
# PoseParam
# ------------------------------------------
# A pose parametrization: degrees of freedom, which are rotations, and the pose map.
class PoseParam(Protocol):
    dof: int
    is_rotation: tuple[bool, ...]

    def to_pose(self, xi: np.ndarray): ...
    def from_pose(self, pose): ...
    def range_jacobian(self, trace, pred: BeamPrediction) -> np.ndarray: ...


# ------------------------------------------
# fisher_information
# ------------------------------------------
# Return pose FIM and numerical rank using smooth beams only (E22, E24).
def fisher_information(model, sensor, xi: np.ndarray, scene, env,
                       param: PoseParam = Planar2DParam(),  # noqa: B008
                       kind: Literal["full", "range_only"] = "full") -> tuple[np.ndarray, int]:
    if kind not in ("full", "range_only"):
        raise ValueError("kind must be 'full' or 'range_only'")
    xi = np.asarray(xi, dtype=float)
    if xi.shape != (param.dof,):
        raise ValueError(f"xi must have shape ({param.dof},)")
    trace = sensor.trace(param.to_pose(xi), scene, env)
    pred = model.predict(trace, env)
    mask = pred.smooth & np.isfinite(pred.sigma_R) & (pred.sigma_R > 0)
    indices = np.flatnonzero(mask)
    p = param.dof
    info = np.zeros((p, p))
    if not indices.size:
        return info, 0
    jac = param.range_jacobian(trace, pred)[indices]
    sigma = pred.sigma_R[indices]
    pd = pred.P_D[indices]
    info += (jac.T * (pd / sigma**2)) @ jac
    if kind == "full":
        grad_sigma = np.zeros((indices.size, p))
        grad_pd = np.zeros((indices.size, p))
        for k in range(p):
            h = model.model.fd_step_psi if param.is_rotation[k] else model.model.fd_step_t
            step = np.zeros(p)
            step[k] = h
            pp = model.predict(sensor.trace(param.to_pose(xi + step), scene, env), env)
            pm = model.predict(sensor.trace(param.to_pose(xi - step), scene, env), env)
            grad_sigma[:, k] = (pp.sigma_R[indices] - pm.sigma_R[indices]) / (2 * h)
            grad_pd[:, k] = (pp.P_D[indices] - pm.P_D[indices]) / (2 * h)
        info += (grad_sigma.T * (2 * pd / sigma**2)) @ grad_sigma
        safe = np.clip(pd, 1e-12, 1 - 1e-12)
        info += (grad_pd.T / (safe * (1 - safe))) @ grad_pd
    info = (info + info.T) * 0.5
    eig = np.linalg.eigvalsh(info)
    rank = int(np.count_nonzero(eig > 1e-9 * max(float(eig[-1]), 0.0))) if eig.size and eig[-1] > 0 else 0
    return info, rank


# ------------------------------------------
# crb
# ------------------------------------------
# Return inverse FIM, or ``None`` when the matrix is rank deficient.
def crb(info: np.ndarray) -> np.ndarray | None:
    info = np.asarray(info, dtype=float)
    eig = np.linalg.eigvalsh((info + info.T) * 0.5)
    if eig.size == 0 or eig[-1] <= 0 or np.any(eig <= 1e-9 * eig[-1]):
        return None
    return np.linalg.inv(info)
