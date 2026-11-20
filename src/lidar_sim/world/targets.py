# ==========================================================
# src/lidar_sim/world/targets.py
# ==========================================================
# Rigid target trajectories ``t ↦ Pose`` (spec §4.8): constant velocity and piecewise-linear waypoints.
#
# Used as ``Mover.pose_at`` (scene objects, posed at ``t_scene = τ_i``, E2) or as a time-varying sensor
# pose passed to ``LidarSensor.trace``. 2D (planar) poses.

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from lidar_sim.core.pose import Pose, rot2, wrap_pi


# ------------------------------------------
# constant_velocity
# ------------------------------------------
# Constant linear and angular velocity trajectory.
#
# ``pose(t) = (R(yaw0 + yaw_rate·t), p0 + v·t)``.
#
# Parameters
# p0 : (2,) float
#     Position at t = 0 [m], world frame.
# v : (2,) float
#     Velocity [m/s], world frame.
# yaw0 : float
#     Yaw at t = 0 [rad].
# yaw_rate : float
#     Yaw rate [rad/s].
#
# Returns
# Callable[[float], Pose]
#     ``t`` [s] → pose (body → world).
def constant_velocity(p0, v, yaw0: float = 0.0, yaw_rate: float = 0.0) -> Callable[[float], Pose]:
    p0a = np.asarray(p0, dtype=float)
    va = np.asarray(v, dtype=float)
    if p0a.shape != (2,) or va.shape != (2,):
        raise ValueError(f"p0 and v must have shape (2,), got {p0a.shape}, {va.shape}")
    yaw0, yaw_rate = float(yaw0), float(yaw_rate)

    def pose_at(t: float) -> Pose:
        return Pose(rot2(yaw0 + yaw_rate * t), p0a + va * t)

    return pose_at


# ------------------------------------------
# waypoints
# ------------------------------------------
# Piecewise-linear trajectory through planar poses.
#
# Linear interpolation of (x, y); yaw interpolated along the shortest arc
# (``ψ_a + s·wrapπ(ψ_b − ψ_a)``); clamped to the first/last waypoint outside ``[times[0], times[-1]]``.
#
# Parameters
# times : (T,) float
#     Strictly increasing waypoint times [s], T >= 1.
# xis : (T, 3) float
#     Waypoint poses ξ = (x [m], y [m], ψ [rad]).
#
# Returns
# Callable[[float], Pose]
def waypoints(times, xis) -> Callable[[float], Pose]:
    ts = np.asarray(times, dtype=float).reshape(-1)
    X = np.asarray(xis, dtype=float)
    if ts.size < 1 or X.shape != (ts.size, 3):
        raise ValueError(f"need T >= 1 times and (T, 3) xis, got {ts.shape}, {X.shape}")
    if np.any(np.diff(ts) <= 0) or not np.all(np.isfinite(ts)) or not np.all(np.isfinite(X)):
        raise ValueError("times must be finite and strictly increasing; xis finite")
    dpsi = np.array([wrap_pi(float(b - a)) for a, b in zip(X[:-1, 2], X[1:, 2], strict=True)])

    def pose_at(t: float) -> Pose:
        t = float(t)
        if ts.size == 1 or t <= ts[0]:
            return Pose.from_xi(X[0])
        if t >= ts[-1]:
            return Pose.from_xi(X[-1])
        k = int(np.searchsorted(ts, t, side="right")) - 1
        s = (t - ts[k]) / (ts[k + 1] - ts[k])
        xy = (1.0 - s) * X[k, :2] + s * X[k + 1, :2]
        return Pose(rot2(X[k, 2] + s * dpsi[k]), xy)

    return pose_at
