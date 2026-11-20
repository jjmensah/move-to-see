# ==========================================================
# src/lidar_sim/geometry/scene2d.py
# ==========================================================
# 2D segment scene with vectorized nearest-hit ray casting (spec §4.2, E5). 2D-only (spec §3.2).
#
# Surface ids: static segment row index ``0..M−1``, then mover segments ``M..`` in mover order (each
# mover's local segment order preserved). Surfaces are two-sided.

from __future__ import annotations

import numpy as np

from lidar_sim.core.types import HitBatch
from lidar_sim.world.materials import MATERIALS, Material
from lidar_sim.world.scenes import Mover, SceneSpec

_U_TOL = 1e-12        # inclusive-endpoint slack on the segment parameter u (rounding at shared vertices)
_TIE_RTOL = 1e-12     # hits within this relative range of the nearest one count as ties → lowest id


# ------------------------------------------
# _cross
# ------------------------------------------
# 2D cross product ``×(p, q) = p_x q_y − p_y q_x`` over the last axis (E5).
def _cross(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return p[..., 0] * q[..., 1] - p[..., 1] * q[..., 0]


# ------------------------------------------
# _pose_segments
# ------------------------------------------
# A moving object's segments in the world frame at time t.
def _pose_segments(mover: Mover, t: float) -> np.ndarray:
    pose = mover.pose_at(float(t))
    return mover.segments_local @ pose.R.T + pose.t        # (m, 2, 2), world frame


# ------------------------------------------
# Scene2D
# ------------------------------------------
# Static and moving line segments; satisfies :class:`lidar_sim.geometry.base.Scene`.
#
# Attributes
# dim : int
#     Always 2.
# materials : dict[int, Material]
#     ``{**MATERIALS, **spec.materials}``; used by ``LidarSensor.trace`` for reflectivity lookup.
# material_ids, object_ids : (M_tot,) int
#     Per-surface ids indexed by ``surface_id``.
# r_eps : float
#     Self-intersection guard [m]: only hits with ``r > r_eps`` count.
# eps_par : float
#     Parallel threshold on ``|D| = |×(d, e)|`` [m] (unit ``d``): ``|D| <= eps_par`` ⇒ no hit.
class Scene2D:
    dim = 2

    def __init__(self, segments: np.ndarray, material_id: np.ndarray, object_id: np.ndarray,
                 movers: tuple[Mover, ...] = (), materials: dict[int, Material] | None = None,
                 r_eps: float = 1e-9, eps_par: float = 1e-12) -> None:
        if not (r_eps > 0.0 and eps_par >= 0.0):
            raise ValueError("require r_eps > 0 and eps_par >= 0")
        self._static = np.array(segments, dtype=float).reshape(-1, 2, 2)
        self._static.setflags(write=False)
        self.movers = tuple(movers)
        mats = [np.asarray(material_id, np.int64).reshape(-1)] + [m.material_id for m in self.movers]
        objs = [np.asarray(object_id, np.int64).reshape(-1)]
        objs += [np.full(len(m.material_id), m.object_id, np.int64) for m in self.movers]
        self.material_ids = np.concatenate(mats)
        self.object_ids = np.concatenate(objs)
        for a in (self.material_ids, self.object_ids):
            a.setflags(write=False)
        self._n_mover_segs = sum(len(m.material_id) for m in self.movers)
        self.materials: dict[int, Material] = {**MATERIALS, **(materials or {})}
        self.r_eps = float(r_eps)
        self.eps_par = float(eps_par)

    # Build from canonical scene data (:mod:`lidar_sim.world.scenes`).
    @classmethod
    def from_spec(cls, spec: SceneSpec, r_eps: float = 1e-9, eps_par: float = 1e-12) -> Scene2D:
        return cls(spec.segments, spec.material_id, spec.object_id, spec.movers, spec.materials,
                   r_eps=r_eps, eps_par=eps_par)

    # Total number of segments (static + movers).
    @property
    def n_surfaces(self) -> int:
        return len(self.material_ids)

    # All segments ``(M_tot, 2, 2)`` [m] in the world frame at scene time ``t`` [s], id order.
    def segments_at(self, t: float) -> np.ndarray:
        parts = [self._static] + [_pose_segments(m, t) for m in self.movers]
        return np.concatenate(parts, axis=0)

    # Segments broadcastable to ``(N, M_tot, 2, 2)``: static shared, movers posed per ray time.
    def _segments_per_ray(self, t_scene: np.ndarray) -> np.ndarray:
        if not self.movers:
            return self._static[None]
        times, inv = np.unique(t_scene, return_inverse=True)
        mov = np.empty((len(times), self._n_mover_segs, 2, 2))
        for i, t in enumerate(times):
            mov[i] = np.concatenate([_pose_segments(m, t) for m in self.movers], axis=0)
        static = np.broadcast_to(self._static, (len(t_scene), *self._static.shape))
        return np.concatenate([static, mov[inv]], axis=1)

    # Nearest ray–segment hit per ray.
    #
    # Implements (E5): ray ``o + r d``, segment ``a + u e`` (``e = b − a``, ``w = a − o``,
    # ``D = ×(d, e)``): ``r = ×(w, e)/D``, ``u = ×(w, d)/D``; valid iff ``|D| > eps_par``,
    # ``r > r_eps`` and ``0 ≤ u ≤ 1`` (endpoints inclusive, with a 1e-12 rounding slack on ``u``).
    # Nearest valid hit wins (occlusion); hits within 1e-12 relative of the nearest are ties,
    # resolved to the lowest ``surface_id``. No upper range gate. Normal ``n = ±perp(e)/‖e‖`` with
    # the sign chosen so ``n·d ≤ 0`` (two-sided surfaces).
    #
    # Parameters
    # origins : (N, 2) float
    #     Ray origins [m], world frame.
    # dirs : (N, 2) float
    #     Unit ray directions, world frame.
    # t_scene : (N,) float
    #     Scene time [s] at which movers are posed, per ray (E2 scan skew).
    #
    # Returns
    # HitBatch
    #     Hits: ``r`` [m], ``point = o + r d`` [m], unit ``normal``, ids. Misses: ``hit=False``,
    #     ``r = NaN``, ``point = NaN``, ``normal = 0``, ids ``−1``.
    def intersect(self, origins: np.ndarray, dirs: np.ndarray, t_scene: np.ndarray) -> HitBatch:
        o = np.asarray(origins, dtype=float)
        d = np.asarray(dirs, dtype=float)
        if o.ndim != 2 or o.shape[1] != 2 or d.shape != o.shape:
            raise ValueError(f"origins/dirs: expected matching (N, 2), got {o.shape}, {d.shape}")
        N = o.shape[0]
        t = np.asarray(t_scene, dtype=float)
        if t.shape != (N,):
            raise ValueError(f"t_scene: expected ({N},), got {t.shape}")
        M = self.n_surfaces
        if N == 0 or M == 0:
            return self._pack(np.zeros(N, bool), np.full(N, np.nan), o, d, np.zeros((N, 2)),
                              np.full(N, -1, np.int64))

        seg = self._segments_per_ray(t)                       # (N|1, M, 2, 2)
        a = seg[..., 0, :]
        e = seg[..., 1, :] - a                                # (N|1, M, 2)
        w = a - o[:, None, :]                                 # (N, M, 2)
        dd = d[:, None, :]
        D = _cross(dd, e)                                     # (N|1→N, M)
        nonpar = np.abs(D) > self.eps_par
        Ds = np.where(nonpar, D, 1.0)
        r = _cross(w, e) / Ds
        u = _cross(w, dd) / Ds
        valid = nonpar & (r > self.r_eps) & (u >= -_U_TOL) & (u <= 1.0 + _U_TOL)
        rv = np.where(valid, r, np.inf)
        rmin = rv.min(axis=1)                                 # (N,)
        hit = np.isfinite(rmin)
        tie = valid & (rv <= (rmin * (1.0 + _TIE_RTOL))[:, None])
        sid = np.where(hit, np.argmax(tie, axis=1), -1).astype(np.int64)

        rows = np.arange(N)
        idx = np.where(hit, sid, 0)
        e_hit = np.broadcast_to(e, (N, M, 2))[rows, idx]      # (N, 2)
        n = np.stack([-e_hit[:, 1], e_hit[:, 0]], axis=1)
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        flip = np.einsum("ni,ni->n", n, d) > 0.0
        n[flip] *= -1.0
        r_hit = np.where(hit, rmin, np.nan)                   # always the nearest range; ties pick ids only
        n[~hit] = 0.0
        return self._pack(hit, r_hit, o, d, n, sid)

    def _pack(self, hit, r, o, d, n, sid) -> HitBatch:
        point = o + r[:, None] * d
        mat = np.where(hit, self.material_ids[np.maximum(sid, 0)] if len(self.material_ids) else -1, -1)
        obj = np.where(hit, self.object_ids[np.maximum(sid, 0)] if len(self.object_ids) else -1, -1)
        return HitBatch(hit=hit, r=r, point=point, normal=n, material_id=mat, object_id=obj, surface_id=sid)
