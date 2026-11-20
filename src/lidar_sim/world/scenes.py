# ==========================================================
# src/lidar_sim/world/scenes.py
# ==========================================================
# Canonical scene **data** shared by tests and experiments (spec §4.8).
#
# Each builder documents hand-computable facts that oracle/integration tests rely on. Geometry is a set of
# 2D segments ``(M, 2, 2)`` = M × (endpoint a, endpoint b); ``surface_id`` of a static segment is its row
# index; mover segments follow, in order. Ray casting lives in ``geometry/scene2d.py`` (Phase 1).

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from lidar_sim.core.pose import Pose
from lidar_sim.world.materials import BLACK, GREY, MIRROR, WHITE


# ------------------------------------------
# _ro
# ------------------------------------------
# A read-only copy of an array.
def _ro(a, dtype) -> np.ndarray:
    arr = np.array(a, dtype=dtype, copy=True)
    arr.setflags(write=False)
    return arr


# ------------------------------------------
# Mover
# ------------------------------------------
# Rigid moving object: segments in its body frame, posed at time t by ``pose_at``.
@dataclass(frozen=True, eq=False)
class Mover:
    segments_local: np.ndarray            # (m, 2, 2)
    material_id: np.ndarray               # (m,)
    object_id: int
    pose_at: Callable[[float], Pose]

    def __post_init__(self) -> None:
        seg = _ro(self.segments_local, float)
        if seg.ndim != 3 or seg.shape[1:] != (2, 2):
            raise ValueError(f"segments_local: expected (m, 2, 2), got {seg.shape}")
        mat = _ro(self.material_id, np.int64)
        if mat.shape != (seg.shape[0],):
            raise ValueError("material_id: expected (m,)")
        object.__setattr__(self, "segments_local", seg)
        object.__setattr__(self, "material_id", mat)


# ------------------------------------------
# SceneSpec
# ------------------------------------------
# A 2D scene: static segments, materials, object ids, moving objects and its known facts.
@dataclass(frozen=True, eq=False)
class SceneSpec:
    name: str
    segments: np.ndarray                  # (M, 2, 2) static segments [m]
    material_id: np.ndarray               # (M,)
    object_id: np.ndarray                 # (M,)
    movers: tuple[Mover, ...] = ()
    facts: str = ""
    materials: dict = field(default_factory=dict)  # optional id -> Material overrides

    def __post_init__(self) -> None:
        seg = _ro(self.segments, float)
        if seg.ndim != 3 or seg.shape[1:] != (2, 2):
            raise ValueError(f"segments: expected (M, 2, 2), got {seg.shape}")
        M = seg.shape[0]
        for name in ("material_id", "object_id"):
            if np.asarray(getattr(self, name)).shape != (M,):
                raise ValueError(f"{name}: expected ({M},)")
        if (np.linalg.norm(seg[:, 1] - seg[:, 0], axis=1) <= 0).any():
            raise ValueError("segments: zero-length segment")
        object.__setattr__(self, "segments", seg)
        object.__setattr__(self, "material_id", _ro(self.material_id, np.int64))
        object.__setattr__(self, "object_id", _ro(self.object_id, np.int64))
        object.__setattr__(self, "movers", tuple(self.movers))


# ------------------------------------------
# _polyline
# ------------------------------------------
# Segments joining consecutive points, optionally closed.
def _polyline(points, closed: bool = False) -> np.ndarray:
    p = np.asarray(points, float)
    if closed:
        p = np.vstack([p, p[:1]])
    return np.stack([p[:-1], p[1:]], axis=1)


# ------------------------------------------
# _square
# ------------------------------------------
# The four segments of an axis-aligned square.
def _square(center, side) -> np.ndarray:
    cx, cy = center
    h = side / 2
    return _polyline([(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)], closed=True)


# ------------------------------------------
# single_wall
# ------------------------------------------
def single_wall(material: int = WHITE) -> SceneSpec:
    """Facts: one segment x = 2, y ∈ [−1, 1]. From the origin along +x: r = 2, u = 0.5, n = (−1, 0)."""
    return SceneSpec("single_wall", np.array([[[2.0, -1.0], [2.0, 1.0]]]), [material], [0],
                     facts=single_wall.__doc__)


# ------------------------------------------
# two_walls_occluded
# ------------------------------------------
def two_walls_occluded() -> SceneSpec:
    """Facts: grey occluder x = 2, y ∈ [−0.5, 0.5] (object 1) in front of white wall x = 4, y ∈ [−2, 2]
    (object 0). From the origin, beams with |y/x| < 0.25 (|α| < 14.04°) hit the occluder at r = 2/cos α;
    beams with 14.04° < |α| < 26.57° hit the far wall at r = 4/cos α."""
    seg = np.array([[[2.0, -0.5], [2.0, 0.5]], [[4.0, -2.0], [4.0, 2.0]]])
    return SceneSpec("two_walls_occluded", seg, [GREY, WHITE], [1, 0], facts=two_walls_occluded.__doc__)


# ------------------------------------------
# corner
# ------------------------------------------
def corner() -> SceneSpec:
    """Facts: walls x = 3 (y ∈ [−3, 2]) and y = 2 (x ∈ [−3, 3]) meeting at (3, 2), white. From the origin
    the corner is at α = atan2(2, 3) = 33.69°, r = √13."""
    seg = _polyline([(3.0, -3.0), (3.0, 2.0), (-3.0, 2.0)])
    return SceneSpec("corner", seg, [WHITE, WHITE], [0, 0], facts=corner.__doc__)


# ------------------------------------------
# room_asym
# ------------------------------------------
def room_asym() -> SceneSpec:
    """Facts: closed L-shaped room, vertices (0,0) (6,0) (6,2.5) (4.5,2.5) (4.5,4) (0,4), walls white
    except a black patch on the floor wall x ∈ [2, 3] (y = 0) and a grey patch on the left wall
    y ∈ [1, 2] (x = 0). Asymmetric, so a 2D pose is fully observable from inside (FIM rank 3).
    Nominal sensor pose ξ = (1.5, 1.5, 0)."""
    pts = [(0, 0), (2, 0), (3, 0), (6, 0), (6, 2.5), (4.5, 2.5), (4.5, 4), (0, 4), (0, 2), (0, 1)]
    seg = _polyline(pts, closed=True)
    mat = [WHITE, BLACK, WHITE, WHITE, WHITE, WHITE, WHITE, WHITE, GREY, WHITE]
    return SceneSpec("room_asym", seg, mat, [0] * len(seg), facts=room_asym.__doc__)


# ------------------------------------------
# post_before_wall
# ------------------------------------------
def post_before_wall(post_center=(2.0, 0.0), post_side: float = 0.10,
                     post_material: int = WHITE, wall_material: int = WHITE) -> SceneSpec:
    """Facts: white wall x = 4, y ∈ [−2, 2] (object 0) and a square post of side 0.10 m centred at
    (2, 0) (object 1). Sweeping ``post_center`` in y moves the post edge across a beam (mixed pixels, X07)."""
    seg = np.concatenate([np.array([[[4.0, -2.0], [4.0, 2.0]]]), _square(post_center, post_side)])
    return SceneSpec("post_before_wall", seg, [wall_material] + [post_material] * 4, [0] + [1] * 4,
                     facts=post_before_wall.__doc__)


# ------------------------------------------
# mirror_room
# ------------------------------------------
def mirror_room() -> SceneSpec:
    """Facts: white room (0,0)-(4,3); 45° mirror segment (3.0, 1.0)-(3.6, 1.6) (object 2, on the line
    y = x − 2). From sensor (1, 1.3) along +x: mirror hit at (3.3, 1.3), r₁ = 2.3, normal ±(1, −1)/√2,
    reflected direction (0, 1); diffuse hit on wall y = 3 after r₂ = 1.7 with cos θ₂ = 1: apparent
    range 4.0 m, i.e. a phantom point at (5.0, 1.3) outside the room (X11). Specular flag required."""
    walls = _polyline([(0, 0), (4, 0), (4, 3), (0, 3)], closed=True)
    seg = np.concatenate([walls, np.array([[[3.0, 1.0], [3.6, 1.6]]])])
    return SceneSpec("mirror_room", seg, [WHITE] * 4 + [MIRROR], [0] * 4 + [2], facts=mirror_room.__doc__)


# ------------------------------------------
# moving_box
# ------------------------------------------
def moving_box(p0=(2.0, -1.0), v=(1.0, 0.0), side: float = 0.5) -> SceneSpec:
    """Facts: 10 × 10 m white enclosure centred on the origin; a white 0.5 m box (object 1) centred at
    ``p0 + v t`` (constant velocity, no rotation). Beam i sees the box at time τ_i (E2), so the box
    appears displaced by v·τ_i relative to its t = 0 position (scan skew, X03)."""
    walls = _polyline([(-5, -5), (5, -5), (5, 5), (-5, 5)], closed=True)
    p0a, va = np.asarray(p0, float), np.asarray(v, float)
    mover = Mover(_square((0.0, 0.0), side), np.full(4, WHITE), 1,
                  lambda t: Pose(np.eye(2), p0a + va * t))
    return SceneSpec("moving_box", walls, [WHITE] * 4, [0] * 4, movers=(mover,), facts=moving_box.__doc__)


# ------------------------------------------
# corridor
# ------------------------------------------
def corridor(half_width: float = 1.0, length: float = 100.0) -> SceneSpec:
    """Facts: two parallel white walls y = ±half_width, x ∈ [−length/2, length/2]. From the origin with
    r_max = 5.6 m the ends are invisible, so x-translation is unobservable: FIM rank 2, null direction
    (1, 0, 0)."""
    L = length / 2
    seg = np.array([[[-L, -half_width], [L, -half_width]], [[-L, half_width], [L, half_width]]])
    return SceneSpec("corridor", seg, [WHITE, WHITE], [0, 0], facts=corridor.__doc__)


CANONICAL = {f.__name__: f for f in (single_wall, two_walls_occluded, corner, room_asym,
                                      post_before_wall, mirror_room, moving_box, corridor)}
