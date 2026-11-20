# ==========================================================
# src/mts/route.py
# ==========================================================
# Fixed-route baseline: an inspection loop that ignores the measurements (final-spec §8).

from __future__ import annotations

import numpy as np

from mts.config import RobotConfig
from mts.control import Action
from mts.robot import _point_at, _polyline_progress, follow
from mts.world import World

DEFAULT_WAYPOINTS = np.array([(1.0, 4.0), (2.6, 6.7), (5.8, 6.7), (9.9, 6.7), (9.9, 1.3), (5.8, 0.75), (2.6, 1.1),
                              (1.0, 4.0)])


# ------------------------------------------
# FixedRoute
# ------------------------------------------
# Follow a closed polyline with the shared path follower (F19), repeating until the trial ends.
#
# The controller keeps its arc-length progress along the loop (monotone, searched in a window ahead of the
# previous value) and hands the follower the next ``horizon`` metres of the route.
class FixedRoute:
    name = "fixed"

    def __init__(self, world: World, robot_cfg: RobotConfig, waypoints: np.ndarray | None = None,
                 horizon: float = 3.0) -> None:
        self.world, self.cfg = world, robot_cfg
        wp = np.asarray(DEFAULT_WAYPOINTS if waypoints is None else waypoints, dtype=float).reshape(-1, 2)
        if not np.allclose(wp[0], wp[-1]):
            wp = np.vstack([wp, wp[0]])
        self.loop = wp
        seg = np.diff(wp, axis=0)
        self.cum = np.concatenate([[0.0], np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))])
        self.length = float(self.cum[-1])
        self.horizon = float(horizon)
        self.s = 0.0

    # Route points at arc lengths ``s0 … s0 + length`` (wrapping), including the vertices in between.
    def _window(self, s0: float, length: float, n: int = 60) -> np.ndarray:
        s = np.linspace(s0, s0 + length, n)
        verts = self.cum[(self.cum > s0 % self.length)] + self.length * np.floor(s0 / self.length)
        verts = np.concatenate([verts, self.cum[1:] + self.length * (np.floor(s0 / self.length) + 1)])
        s = np.unique(np.concatenate([s, verts[(verts > s0) & (verts < s0 + length)]]))
        return np.array([_point_at(self.loop, self.cum, v % self.length) for v in s])

    def act(self, k: int, x: np.ndarray, belief=None, rng=None) -> Action:
        p = np.asarray(x, dtype=float)[:2]
        ahead = self._window(self.s, 1.5)                       # search the closest point just ahead
        ds, _ = _polyline_progress(ahead, p)
        self.s += ds
        path = np.vstack([p, self._window(self.s, self.horizon)])
        u, fb = follow(self.world, self.cfg, x, path)
        return Action(u=u, goal=path[-1], fallback=fb, info={"progress": self.s})
