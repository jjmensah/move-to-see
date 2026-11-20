# ==========================================================
# src/mts/fov.py
# ==========================================================
# Field-of-view sets in 3D: geometric (F6), line of sight (E5) and effective (E3a) (final-spec §3.2).

from __future__ import annotations

import numpy as np

from lidar_sim.core.config import EnvConfig
from lidar_sim.core.pose import Pose
from lidar_sim.core.types import RadiometricScan
from lidar_sim.physics.atmosphere import extinction
from lidar_sim.physics.radiometry import lambertian_power
from lidar_sim.sensor.scan3d import in_geometric_fov_3d
from mts.config import NOISE, RASTER, SENSOR, make_model


# ------------------------------------------
# probe_trace
# ------------------------------------------
# Synthetic record of a facing Lambertian probe of reflectivity ``rho`` at ranges ``r (n,)``: one ideal
# return per beam at normal incidence (E7–E8).
def probe_trace(r, rho: float, env: EnvConfig) -> RadiometricScan:
    r = np.asarray(r, dtype=float).reshape(-1)
    n = r.size
    power = lambertian_power(NOISE.P0, np.ones(n), np.full(n, rho), np.ones(n), r, NOISE.A_rx, NOISE.eta,
                             extinction(env.visibility_m, env.wavelength))
    dirs = np.zeros((n, 1, 3))
    dirs[..., 0] = 1.0
    normal = np.zeros((n, 1, 3))
    normal[..., 0] = -1.0
    ids = np.zeros((n, 1), dtype=np.int64)
    return RadiometricScan(pose=Pose.identity(3), fire_time=np.zeros(n), dir_world=dirs, w=np.ones((n, 1)),
                           sub_idx=np.zeros((n, 1), dtype=np.int64), hit=np.ones((n, 1), bool), r_path=r[:, None],
                           power=power[:, None], normal=normal, cos_inc=np.ones((n, 1)), object_id=ids,
                           material_id=ids, surface_id=ids, bounce=np.zeros((n, 1), bool), P_amb=env.P_amb)


# ------------------------------------------
# probe_pd
# ------------------------------------------
# Detection probability (E16) of a facing probe at ranges ``r``.
def probe_pd(r, rho: float, env: EnvConfig) -> np.ndarray:
    return make_model(1).predict(probe_trace(r, rho, env), env).P_D


# ------------------------------------------
# r50
# ------------------------------------------
# Range at which a facing probe has ``P_D = 0.5`` (bisection; ``P_D`` decreases with range).
def r50(rho: float, env: EnvConfig, lo: float = 0.02, hi: float = 200.0) -> float:
    if probe_pd([hi], rho, env)[0] >= 0.5:
        return hi
    if probe_pd([lo], rho, env)[0] < 0.5:
        return lo
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if probe_pd([mid], rho, env)[0] >= 0.5 else (lo, mid)
    return 0.5 * (lo + hi)


# ------------------------------------------
# line_of_sight
# ------------------------------------------
# True where the segment from ``origin (3,)`` to each of ``points (n, 3)`` is unobstructed (E5).
def line_of_sight(scene, origin, points, tol: float = 1e-9) -> np.ndarray:
    p = np.asarray(points, dtype=float).reshape(-1, 3)
    o = np.asarray(origin, dtype=float).reshape(3)
    v = p - o
    dist = np.linalg.norm(v, axis=1)
    ok = dist > 0
    d = np.where(ok[:, None], v / np.where(ok, dist, 1.0)[:, None], np.array([1.0, 0.0, 0.0]))
    h = scene.intersect(np.broadcast_to(o, p.shape), d, np.zeros(len(p)))
    return ok & (~h.hit | (np.where(h.hit, h.r, np.inf) >= dist - tol))


# ------------------------------------------
# fov_masks
# ------------------------------------------
# Membership of world points in the geometric FOV (F6), the line-of-sight region and the effective FOV
# (E3a: geometric ∩ line of sight ∩ ``P_D ≥ 0.5`` for a facing probe of reflectivity ``rho``).
def fov_masks(pose: Pose, scene, points, rho: float, env: EnvConfig) -> dict[str, np.ndarray]:
    p = np.asarray(points, dtype=float).reshape(-1, 3)
    geo = in_geometric_fov_3d(pose, p, RASTER, SENSOR.r_min, SENSOR.r_max)
    los = geo & line_of_sight(scene, pose.t, p)
    eff = los.copy()
    idx = np.flatnonzero(los)
    if idx.size:
        eff[idx] = probe_pd(np.linalg.norm(p[idx] - pose.t, axis=1), rho, env) >= 0.5
    return {"geometric": geo, "los": los, "effective": eff}
