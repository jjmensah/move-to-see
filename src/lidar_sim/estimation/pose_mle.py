# ==========================================================
# src/lidar_sim/estimation/pose_mle.py
# ==========================================================
# Maximum-likelihood pose estimation (E25).

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from lidar_sim.core.pose import Pose
from lidar_sim.core.types import MLEResult
from lidar_sim.estimation.fisher import fisher_information
from lidar_sim.estimation.jacobian2d import Planar2DParam


# ------------------------------------------
# pose_mle
# ------------------------------------------
# Maximize scan log likelihood; returns invalid status for unobservable poses (E25).
def pose_mle(model, sensor, scan, scene, env, xi0: np.ndarray,
             param: Planar2DParam = Planar2DParam(),  # noqa: B008
             method: str = "L-BFGS-B",
             max_iter: int = 200) -> MLEResult:
    if not isinstance(scan.truth.pose, Pose):
        raise ValueError("pose_mle requires a scan from a static Pose")
    start = np.asarray(xi0, dtype=float)
    if start.shape != (param.dof,):
        raise ValueError(f"xi0 must have shape ({param.dof},)")
    scale = np.where(param.is_rotation, 5e-4, 1e-2)
    evaluations = 0

    def likelihood(x):
        nonlocal evaluations
        evaluations += 1
        estimate = start + scale * x
        pred = model.predict(sensor.trace(param.to_pose(estimate), scene, env), env)
        return model.log_likelihood(scan, pred)

    def objective(x):
        return -likelihood(x)

    def gradient(x):
        grad = np.zeros(param.dof)
        step = 1e-3
        for k in range(param.dof):
            dx = np.zeros(param.dof)
            dx[k] = step
            grad[k] = (objective(x + dx) - objective(x - dx)) / (2 * step)
        return grad

    method = method or "L-BFGS-B"
    result = minimize(objective, np.zeros(param.dof), method=method,
                      jac=gradient if method == "L-BFGS-B" else None,
                      options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-5})
    if (not result.success or not np.isfinite(result.fun)) and method != "Nelder-Mead":
        result = minimize(objective, result.x if np.all(np.isfinite(result.x)) else np.zeros(param.dof),
                          method="Nelder-Mead", options={"maxiter": max_iter * 2, "xatol": 1e-5, "fatol": 1e-7})
    estimate = start + scale * result.x
    _, rank = fisher_information(model, sensor, estimate, scene, env, param)
    if rank < param.dof:
        return MLEResult(estimate, False, int(getattr(result, "nit", 0)), -float(result.fun),
                         f"rank-deficient: rank {rank} of {param.dof}")
    return MLEResult(estimate, bool(result.success), int(getattr(result, "nit", 0)),
                     -float(result.fun), str(result.message))
