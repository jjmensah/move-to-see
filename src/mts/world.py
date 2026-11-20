# ==========================================================
# src/mts/world.py
# ==========================================================
# World model: obstacles, terrain, target prior region and keep-out geometry (final-spec §4.2–4.3).

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from lidar_sim.geometry.scene3d import Scene3D
from lidar_sim.world.materials import BLACK, GREY, MATERIALS, WHITE
from mts.config import RobotConfig

TARGET_OBJECT = 99          # object id of the target box in truth scenes
GROUND_OBJECT = 0
_BOUNDARY_TOL = 1e-9        # [m] a position this close to a keep-out boundary counts as touching it


# ------------------------------------------
# _ro
# ------------------------------------------
# A read-only copy of an array.
def _ro(a, dtype=float) -> np.ndarray:
    a = np.array(a, dtype=dtype, copy=True)
    a.setflags(write=False)
    return a


# ------------------------------------------
# World
# ------------------------------------------
# Static world known to the robot, plus the description of the unknown target.
#
# Parameters
# arena : (x0, x1, y0, y1) [m]
# box_lo, box_hi : (n, 3) obstacle boxes, walls included [m]
# box_material : (n,) int material ids
# rough : (R, 4) rough-terrain rectangles (x0, x1, y0, y1) [m] (F8)
# prior_lo, prior_hi : (3,) region R_T of target centres [m]
# target_half : (3,) target half-sizes s_T [m]
# target_material : int
# start : (4,) initial robot state (p_x, p_y, θ, v)
# ground_material : int
@dataclass(frozen=True, eq=False)
class World:
    arena: tuple[float, float, float, float]
    box_lo: np.ndarray
    box_hi: np.ndarray
    box_material: np.ndarray
    rough: np.ndarray
    prior_lo: np.ndarray
    prior_hi: np.ndarray
    target_half: np.ndarray
    target_material: int
    start: np.ndarray
    ground_material: int = GREY

    def __post_init__(self) -> None:
        object.__setattr__(self, "arena", tuple(float(v) for v in self.arena))
        object.__setattr__(self, "box_lo", _ro(np.reshape(self.box_lo, (-1, 3))))
        object.__setattr__(self, "box_hi", _ro(np.reshape(self.box_hi, (-1, 3))))
        object.__setattr__(self, "box_material", _ro(self.box_material, np.int64))
        object.__setattr__(self, "rough", _ro(np.reshape(self.rough, (-1, 4))))
        for name in ("prior_lo", "prior_hi", "target_half"):
            object.__setattr__(self, name, _ro(np.reshape(getattr(self, name), 3)))
        object.__setattr__(self, "start", _ro(np.reshape(self.start, 4)))
        if self.box_lo.shape != self.box_hi.shape or self.box_material.shape != (self.box_lo.shape[0],):
            raise ValueError("World: inconsistent obstacle arrays")
        if not (np.all(self.prior_hi >= self.prior_lo) and np.all(self.target_half > 0)):
            raise ValueError("World: invalid prior region or target size")

    # Scenes
    @property
    def n_boxes(self) -> int:
        return self.box_lo.shape[0]

    # Obstacles (object ids 1..n) and ground (object 0); no target.
    def background_scene(self) -> Scene3D:
        return Scene3D(self.box_lo, self.box_hi, self.box_material, np.arange(1, self.n_boxes + 1),
                       materials=MATERIALS, ground_material=self.ground_material, ground_object=GROUND_OBJECT)

    # Background plus the target at ``q`` as the last box (object id ``TARGET_OBJECT``).
    def truth_scene(self, q) -> Scene3D:
        lo, hi = self.target_box(np.asarray(q, dtype=float).reshape(3))
        return Scene3D(np.vstack([self.box_lo, lo]), np.vstack([self.box_hi, hi]),
                       np.append(self.box_material, self.target_material),
                       np.append(np.arange(1, self.n_boxes + 1), TARGET_OBJECT),
                       materials=MATERIALS, ground_material=self.ground_material, ground_object=GROUND_OBJECT)

    # Corners ``(lo, hi)`` of the target box for centre(s) ``q (..., 3)``.
    def target_box(self, q) -> tuple[np.ndarray, np.ndarray]:
        q = np.asarray(q, dtype=float)
        return q - self.target_half, q + self.target_half

    # Box containing every possible target: ``[prior_lo − s_T, prior_hi + s_T]``.
    def envelope(self) -> tuple[np.ndarray, np.ndarray]:
        return self.prior_lo - self.target_half, self.prior_hi + self.target_half

    def with_target_material(self, material: int) -> World:
        return replace(self, target_material=int(material))

    # Feasibility and terrain
    # (Q, 4) keep-out rectangles (x0, x1, y0, y1): obstacle footprints and the union of possible target
    # footprints, each grown by ``radius + margin`` on every side (final-spec §4.2).
    def keepouts(self, cfg: RobotConfig) -> np.ndarray:
        g = cfg.radius + cfg.margin
        elo, ehi = self.envelope()
        lo = np.vstack([self.box_lo[:, :2], elo[:2]])
        hi = np.vstack([self.box_hi[:, :2], ehi[:2]])
        return np.column_stack([lo[:, 0] - g, hi[:, 0] + g, lo[:, 1] - g, hi[:, 1] + g])

    # Distance [m] from ``xy (..., 2)`` to the nearest keep-out rectangle (0 inside or on a boundary).
    def clearance(self, xy, cfg: RobotConfig) -> np.ndarray:
        p = np.asarray(xy, dtype=float)
        k = self.keepouts(cfg)
        dx = np.maximum(np.maximum(k[:, 0] - p[..., None, 0], p[..., None, 0] - k[:, 1]), 0.0)
        dy = np.maximum(np.maximum(k[:, 2] - p[..., None, 1], p[..., None, 1] - k[:, 3]), 0.0)
        return np.hypot(dx, dy).min(axis=-1)

    # True where the robot centre ``xy (..., 2)`` is strictly outside every keep-out and inside the arena
    # (touching a boundary, within 1e-9 m, is infeasible).
    def feasible(self, xy, cfg: RobotConfig) -> np.ndarray:
        p = np.asarray(xy, dtype=float)
        x0, x1, y0, y1 = self.arena
        inside = (p[..., 0] > x0) & (p[..., 0] < x1) & (p[..., 1] > y0) & (p[..., 1] < y1)
        return inside & (self.clearance(p, cfg) > _BOUNDARY_TOL)

    def in_rough(self, xy) -> np.ndarray:
        p = np.asarray(xy, dtype=float)
        if self.rough.shape[0] == 0:
            return np.zeros(p.shape[:-1], dtype=bool)
        r = self.rough
        return ((p[..., None, 0] >= r[:, 0]) & (p[..., None, 0] <= r[:, 1])
                & (p[..., None, 1] >= r[:, 2]) & (p[..., None, 1] <= r[:, 3])).any(axis=-1)

    # Terrain speed limit (F8).
    def v_lim(self, xy, cfg: RobotConfig) -> np.ndarray:
        return np.where(self.in_rough(xy), cfg.v_rough, cfg.v_max)


# ------------------------------------------
# _walls
# ------------------------------------------
# The four boundary walls of the arena as boxes.
def _walls(arena, thickness: float = 0.1, height: float = 1.5):
    x0, x1, y0, y1 = arena
    t = thickness
    lo = [(x0, y0, 0), (x0, y1 - t, 0), (x0, y0, 0), (x1 - t, y0, 0)]
    hi = [(x1, y0 + t, height), (x1, y1, height), (x0 + t, y1, height), (x1, y1, height)]
    return lo, hi


# ------------------------------------------
# yard
# ------------------------------------------
# Default world (final-spec §4.3). ``rough=False`` removes the rough patch and ``dark_background=True``
# makes walls, crates and ground black (ablations of V&V F06).
def yard(target_material: int = BLACK, rough: bool = True, dark_background: bool = False) -> World:
    arena = (0.0, 11.0, 0.0, 8.0)
    wlo, whi = _walls(arena)
    crates_lo = [(3.2, 3.3, 0.0), (3.4, 3.5, 1.2), (4.4, 3.6, 0.0)]
    crates_hi = [(4.4, 4.7, 1.2), (4.2, 4.4, 1.9), (5.3, 4.6, 0.85)]
    wall_mat, crate_mat, ground = (BLACK, BLACK, BLACK) if dark_background else (GREY, WHITE, GREY)
    return World(arena=arena, box_lo=np.array(wlo + crates_lo), box_hi=np.array(whi + crates_hi),
                 box_material=np.array([wall_mat] * 4 + [crate_mat] * 3),
                 rough=np.array([[3.6, 6.2, 1.0, 2.6]]) if rough else np.zeros((0, 4)),
                 prior_lo=np.array([7.0, 2.8, 0.5]), prior_hi=np.array([8.6, 5.2, 1.1]),
                 target_half=np.array([0.25, 0.25, 0.25]), target_material=target_material,
                 start=np.array([1.0, 4.0, 0.0, 0.0]), ground_material=ground)
