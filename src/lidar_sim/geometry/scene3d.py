# ==========================================================
# src/lidar_sim/geometry/scene3d.py
# ==========================================================
# 3D scenes of axis-aligned boxes and a ground plane (final-spec §3.1).
#
# Implements the ``Scene`` protocol with ``dim = 3``: nearest-hit ray casting (E5) by the slab method (F2).

from __future__ import annotations

import numpy as np

from lidar_sim.core.types import HitBatch

_FACE_AXIS = np.array([0, 0, 1, 1, 2, 2])
_FACE_SIGN = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0])


# ------------------------------------------
# ray_box_batch
# ------------------------------------------
# Slab-method intersection (F2) of N rays with P boxes, every pair independently.
#
# Parameters
# origins, dirs : (N, 3) float
#     Ray origins [m] and unit directions.
# lo, hi : (P, 3) float
#     Box corners [m], ``lo < hi`` componentwise.
# r_eps : float
#     Self-intersection guard [m].
#
# Returns
# r : (P, N) float
#     Hit distance [m]: ``r_near`` if ``r_near > r_eps``, else ``r_far`` (origin inside the box: exit
#     face); ``inf`` where the box is not hit (``r_near > r_far`` or ``r_far <= r_eps``).
# face : (P, N) int
#     Hit face 0..5 = (−x, +x, −y, +y, −z, +z), ties broken by the lowest axis index; −1 where no hit.
def ray_box_batch(origins: np.ndarray, dirs: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                  r_eps: float = 1e-9) -> tuple[np.ndarray, np.ndarray]:
    o = np.asarray(origins, dtype=float)[None, :, :]            # (1, N, 3)
    d = np.asarray(dirs, dtype=float)[None, :, :]
    lo = np.asarray(lo, dtype=float).reshape(-1, 1, 3)          # (P, 1, 3)
    hi = np.asarray(hi, dtype=float).reshape(-1, 1, 3)
    par = d == 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.where(par, 1.0, 1.0 / np.where(par, 1.0, d))
        t_lo = (lo - o) * inv
        t_hi = (hi - o) * inv
    t_minus = np.minimum(t_lo, t_hi)
    t_plus = np.maximum(t_lo, t_hi)
    inside = (o >= lo) & (o <= hi)
    t_minus = np.where(par, np.where(inside, -np.inf, np.inf), t_minus)
    t_plus = np.where(par, np.where(inside, np.inf, -np.inf), t_plus)
    ax_near = np.argmax(t_minus, axis=-1)                       # first max -> lowest axis on ties
    ax_far = np.argmin(t_plus, axis=-1)
    r_near = np.take_along_axis(t_minus, ax_near[..., None], axis=-1)[..., 0]
    r_far = np.take_along_axis(t_plus, ax_far[..., None], axis=-1)[..., 0]
    hit = (r_near <= r_far) & (r_far > r_eps)
    enter = r_near > r_eps
    r = np.where(hit, np.where(enter, r_near, r_far), np.inf)
    axis = np.where(enter, ax_near, ax_far)
    d_ax = np.take_along_axis(np.broadcast_to(d, t_minus.shape), axis[..., None], axis=-1)[..., 0]
    # entering through the low face when moving in +axis; leaving through the high face when moving in +axis
    high = np.where(enter, d_ax < 0.0, d_ax > 0.0)
    face = np.where(hit, 2 * axis + high.astype(np.int64), -1)
    return r, face


# ------------------------------------------
# Scene3D
# ------------------------------------------
# Static axis-aligned boxes plus an optional ground plane z = 0.
#
# Parameters
# lo, hi : (n, 3) float
#     Box corners [m] with ``lo < hi``.
# material_id, object_id : (n,) int
#     Per-box ids (>= 0).
# materials : dict[int, Material] or None
#     Material table used by ``LidarSensor.trace`` (falls back to the sensor's table if ``None``).
# ground_material : int or None
#     Material id of the ground plane z = 0 (``None``: no ground).
# ground_object : int
#     Object id reported for ground hits.
# r_eps : float
#     Self-intersection guard [m].
#
# Surface ids: ``6 m + f`` for box ``m`` and face ``f`` (0..5 = −x, +x, −y, +y, −z, +z); the ground is
# ``6 n``.
class Scene3D:
    dim = 3

    def __init__(self, lo, hi, material_id, object_id, materials: dict | None = None,
                 ground_material: int | None = None, ground_object: int = 0, r_eps: float = 1e-9) -> None:
        lo = np.array(lo, dtype=float).reshape(-1, 3)
        hi = np.array(hi, dtype=float).reshape(-1, 3)
        n = lo.shape[0]
        mid = np.array(material_id, dtype=np.int64).reshape(-1)
        oid = np.array(object_id, dtype=np.int64).reshape(-1)
        if hi.shape != (n, 3) or mid.shape != (n,) or oid.shape != (n,):
            raise ValueError("Scene3D: lo, hi must be (n, 3) and material_id, object_id (n,)")
        if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)) and np.all(hi > lo)):
            raise ValueError("Scene3D: boxes need finite corners with lo < hi")
        if (mid < 0).any() or (oid < 0).any() or (ground_material is not None and ground_material < 0):
            raise ValueError("Scene3D: ids must be >= 0")
        if r_eps <= 0.0:
            raise ValueError("Scene3D: r_eps must be > 0")
        for a in (lo, hi, mid, oid):
            a.setflags(write=False)
        self.lo, self.hi, self.material_id, self.object_id = lo, hi, mid, oid
        self.materials = None if materials is None else dict(materials)
        self.ground_material = None if ground_material is None else int(ground_material)
        self.ground_object = int(ground_object)
        self.r_eps = float(r_eps)

    @property
    def n_boxes(self) -> int:
        return self.lo.shape[0]

    @property
    def n_surfaces(self) -> int:
        return 6 * self.n_boxes + (0 if self.ground_material is None else 1)

    # Copy of the scene with box ``m`` replaced by ``[lo, hi]`` (same ids).
    def with_box(self, m: int, lo, hi) -> Scene3D:
        new_lo, new_hi = self.lo.copy(), self.hi.copy()
        new_lo[m], new_hi[m] = lo, hi
        return Scene3D(new_lo, new_hi, self.material_id, self.object_id, self.materials, self.ground_material,
                       self.ground_object, self.r_eps)

    # Copy of the scene with box ``m`` removed (surface ids of later boxes shift down by 6).
    def without_box(self, m: int) -> Scene3D:
        keep = np.arange(self.n_boxes) != m
        return Scene3D(self.lo[keep], self.hi[keep], self.material_id[keep], self.object_id[keep], self.materials,
                       self.ground_material, self.ground_object, self.r_eps)

    # Nearest hit of each ray with the boxes and the ground (E5, F2). ``t_scene`` is unused (static).
    def intersect(self, origins: np.ndarray, dirs: np.ndarray, t_scene: np.ndarray | None = None) -> HitBatch:
        o = np.asarray(origins, dtype=float).reshape(-1, 3)
        d = np.asarray(dirs, dtype=float).reshape(-1, 3)
        N = o.shape[0]
        r = np.full(N, np.inf)
        sid = np.full(N, -1, np.int64)
        if self.n_boxes:
            rb, fb = ray_box_batch(o, d, self.lo, self.hi, self.r_eps)        # (n, N)
            m = np.argmin(rb, axis=0)                                          # first min -> lowest box index
            cols = np.arange(N)
            r = rb[m, cols]
            sid = np.where(np.isfinite(r), 6 * m + fb[m, cols], -1)
        if self.ground_material is not None:
            down = (d[:, 2] < 0.0) & (o[:, 2] > 0.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                rg = np.where(down, -o[:, 2] / np.where(down, d[:, 2], -1.0), np.inf)
            rg = np.where(rg > self.r_eps, rg, np.inf)
            g = rg < r
            r = np.where(g, rg, r)
            sid = np.where(g, 6 * self.n_boxes, sid)
        return self._pack(o, d, r, sid)

    def _pack(self, o: np.ndarray, d: np.ndarray, r: np.ndarray, sid: np.ndarray) -> HitBatch:
        hit = sid >= 0
        N = o.shape[0]
        is_ground = hit & (sid == 6 * self.n_boxes)
        box = np.where(hit & ~is_ground, sid // 6, 0)
        face = np.where(hit & ~is_ground, sid % 6, 5)                         # ground normal = +z (face 5)
        normal = np.zeros((N, 3))
        normal[np.arange(N), _FACE_AXIS[face]] = _FACE_SIGN[face]
        flip = np.einsum("ij,ij->i", normal, d) > 0.0                         # exit hits: orient against the ray
        normal[flip] *= -1.0
        normal[~hit] = 0.0
        rr = np.where(hit, r, 0.0)
        point = np.where(hit[:, None], o + rr[:, None] * d, 0.0)
        gm = -1 if self.ground_material is None else self.ground_material
        mat = np.where(is_ground, gm, np.where(hit, self.material_id[box] if self.n_boxes else -1, -1))
        obj = np.where(is_ground, self.ground_object, np.where(hit, self.object_id[box] if self.n_boxes else -1, -1))
        return HitBatch(hit=hit, r=np.where(hit, r, np.nan), point=point, normal=normal,
                        material_id=mat, object_id=obj, surface_id=np.where(hit, sid, -1))
