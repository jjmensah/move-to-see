# ==========================================================
# src/mts/robot.py
# ==========================================================
# Unicycle robot on terrain: dynamics, feasibility, navigation grid and path following (final-spec §4, §7.4–7.5).

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from lidar_sim.core.pose import Pose
from mts.config import RobotConfig
from mts.world import World


# ------------------------------------------
# wrap_pi
# ------------------------------------------
# Wrap angle(s) to [−π, π).
def wrap_pi(a):
    return (np.asarray(a, dtype=float) + math.pi) % (2.0 * math.pi) - math.pi


# ------------------------------------------
# _f
# ------------------------------------------
# Unicycle with acceleration input (F7): ẋ = (v cos θ, v sin θ, ω, a).
def _f(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    return np.array([x[3] * math.cos(x[2]), x[3] * math.sin(x[2]), u[1], u[0]])


# ------------------------------------------
# _rk4
# ------------------------------------------
# One fourth-order Runge-Kutta step of the unicycle (F7).
def _rk4(x: np.ndarray, u: np.ndarray, h: float) -> np.ndarray:
    k1 = _f(x, u)
    k2 = _f(x + 0.5 * h * k1, u)
    k3 = _f(x + 0.5 * h * k2, u)
    k4 = _f(x + h * k3, u)
    return x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# ------------------------------------------
# _advance
# ------------------------------------------
# One RK4 step of length ``h`` followed by the terrain speed projection (F8).
def _advance(world: World, cfg: RobotConfig, x: np.ndarray, u: np.ndarray, h: float) -> np.ndarray:
    y = _rk4(x, u, h)
    y[3] = min(max(y[3], 0.0), float(world.v_lim(y[:2], cfg)))
    return y


# ------------------------------------------
# Trajectory
# ------------------------------------------
# Motion over one decision step: substep states and a continuous-time state/pose lookup.
#
# ``states[s]`` is the state at ``t0 + s·substep``. ``state(t)`` advances the last substep state by one RK4
# step of the remaining fraction (final-spec §4.1). ``feasible`` is True iff every substep state after the
# first is feasible (§4.2).
@dataclass(frozen=True, eq=False)
class Trajectory:
    t0: float
    states: np.ndarray
    u: np.ndarray
    substep: float
    feasible: bool
    world: World
    cfg: RobotConfig

    @property
    def end(self) -> np.ndarray:
        return self.states[-1]

    @property
    def t1(self) -> float:
        return self.t0 + (self.states.shape[0] - 1) * self.substep

    def state(self, t: float) -> np.ndarray:
        s = (float(t) - self.t0) / self.substep
        n = self.states.shape[0] - 1
        if s < -1e-9 or s > n + 1e-9:
            raise ValueError(f"t = {t} outside the step [{self.t0}, {self.t1}]")
        i = min(max(math.floor(s + 1e-12), 0), n)
        frac = min(max(s - i, 0.0), 1.0)
        if frac < 1e-12 or i == n:
            return self.states[i].copy()
        return _advance(self.world, self.cfg, self.states[i], self.u, frac * self.substep)

    # Sensor pose (F1) at absolute time ``t``.
    def pose(self, t: float) -> Pose:
        x = self.state(t)
        return Pose.from_xyz_yaw(x[0], x[1], self.cfg.h_s, x[2])


# ------------------------------------------
# sensor_pose
# ------------------------------------------
# Sensor pose (F1) for a robot state.
def sensor_pose(x: np.ndarray, cfg: RobotConfig) -> Pose:
    return Pose.from_xyz_yaw(x[0], x[1], cfg.h_s, x[2])


# ------------------------------------------
# clip_control
# ------------------------------------------
# Clips the acceleration and turn rate to the robot's limits.
def clip_control(cfg: RobotConfig, u) -> np.ndarray:
    u = np.asarray(u, dtype=float).reshape(2)
    return np.array([min(max(u[0], -cfg.a_max), cfg.a_max), min(max(u[1], -cfg.w_max), cfg.w_max)])


# ------------------------------------------
# step
# ------------------------------------------
# Integrate one decision step with the control held constant (F7–F8).
#
# The control is clipped to the input limits. Returns the trajectory with its feasibility flag.
def step(world: World, cfg: RobotConfig, x, u, t0: float = 0.0) -> Trajectory:
    u = clip_control(cfg, u)
    x = np.asarray(x, dtype=float).reshape(4).copy()
    x[3] = min(max(x[3], 0.0), float(world.v_lim(x[:2], cfg)))
    states = np.empty((cfg.n_sub + 1, 4))
    states[0] = x
    for s in range(cfg.n_sub):
        states[s + 1] = _advance(world, cfg, states[s], u, cfg.substep)
    ok = bool(np.all(world.feasible(states[1:, :2], cfg)))
    states.setflags(write=False)
    return Trajectory(float(t0), states, u, cfg.substep, ok, world, cfg)


# ------------------------------------------
# safe_speed
# ------------------------------------------
# Cruise speed that can be held for one decision step and then braked within the clearance.
#
# Solves ``v Δt + v²/(2 a_max) = max(clearance − m_s, 0)``; used for the navigation-grid speed (§7.4).
def safe_speed(cfg: RobotConfig, clearance) -> np.ndarray:
    d = np.maximum(np.asarray(clearance, dtype=float) - cfg.brake_margin, 0.0)
    return cfg.a_max * (-cfg.dt + np.sqrt(cfg.dt**2 + 2.0 * d / cfg.a_max))


# ------------------------------------------
# end_speed_bound
# ------------------------------------------
# Largest end-of-step speed ``v′`` that keeps a braking distance within the room ``d`` (F19).
#
# With constant acceleration over the step the robot travels ``(v + v′)Δt/2`` and then needs ``v′²/(2a)``
# to stop, so ``v′`` is the positive root of ``v′²/(2a) + v′Δt/2 + vΔt/2 − d = 0`` (0 if there is none).
def end_speed_bound(cfg: RobotConfig, v: float, d: float) -> float:
    a, dt = cfg.a_max, cfg.dt
    disc = dt * dt / 4.0 + 2.0 * (d - v * dt / 2.0) / a
    return max(a * (-dt / 2.0 + math.sqrt(disc)), 0.0) if disc > 0.0 else 0.0


# ------------------------------------------
# _polyline_progress
# ------------------------------------------
# Arc length of the point of ``path`` closest to ``p`` and the cumulative arc lengths of its vertices.
def _polyline_progress(path: np.ndarray, p: np.ndarray) -> tuple[float, np.ndarray]:
    seg = np.diff(path, axis=0)
    L = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(L)])
    best_s, best_d = 0.0, math.inf
    for i in range(len(seg)):
        if L[i] <= 0.0:
            continue
        t = min(max(float(np.dot(p - path[i], seg[i])) / L[i] ** 2, 0.0), 1.0)
        d = float(np.hypot(*(path[i] + t * seg[i] - p)))
        if d < best_d - 1e-12:
            best_d, best_s = d, cum[i] + t * L[i]
    return best_s, cum


# ------------------------------------------
# _point_at
# ------------------------------------------
# The point at arc length s along a polyline.
def _point_at(path: np.ndarray, cum: np.ndarray, s: float) -> np.ndarray:
    s = min(max(s, 0.0), cum[-1])
    i = min(int(np.searchsorted(cum, s, side="right")) - 1, len(path) - 2)
    L = cum[i + 1] - cum[i]
    t = 0.0 if L <= 0.0 else (s - cum[i]) / L
    return path[i] + t * (path[i + 1] - path[i])


# ------------------------------------------
# follow
# ------------------------------------------
# Path-following control for one decision step (F19) with fallbacks.
#
# Parameters
# x : (4,) robot state.
# path : (n, 2) polyline to follow; its last point is the goal.
# face : (2,) or None, point to turn towards once the goal is reached.
#
# Returns
# u : (2,) control (a, ω), feasible over the step whenever a braking fallback is feasible.
# fallback : bool, True if a fallback control replaced the nominal one.
def follow(world: World, cfg: RobotConfig, x, path, face=None) -> tuple[np.ndarray, bool]:
    x = np.asarray(x, dtype=float).reshape(4)
    path = np.asarray(path, dtype=float).reshape(-1, 2)
    p, th, v = x[:2], x[2], x[3]
    brake = -min(cfg.a_max, v / cfg.dt)
    if len(path) < 2:
        path = np.vstack([p, path[-1]])
    s0, cum = _polyline_progress(path, p)
    d_goal = cum[-1] - s0
    if float(np.hypot(*(path[-1] - p))) < cfg.goal_tol:
        w = 0.0
        if face is not None:
            e = float(wrap_pi(math.atan2(face[1] - p[1], face[0] - p[0]) - th))
            w = min(max(cfg.k_heading * e, -cfg.w_max), cfg.w_max)
        u = np.array([brake, w])
    else:
        look = _point_at(path, cum, s0 + cfg.lookahead)
        e = float(wrap_pi(math.atan2(look[1] - p[1], look[0] - p[0]) - th))
        w = min(max(cfg.k_heading * e, -cfg.w_max), cfg.w_max)
        room = max(float(world.clearance(p, cfg)) - cfg.brake_margin, 0.0)
        v_cmd = min(float(world.v_lim(p, cfg)), end_speed_bound(cfg, v, room),
                    end_speed_bound(cfg, v, max(d_goal, 0.0))) * max(math.cos(e), 0.0)
        u = np.array([min(max((v_cmd - v) / cfg.dt, -cfg.a_max), cfg.a_max), w])
    if step(world, cfg, x, u).feasible:
        return u, False
    for cand in (np.array([brake, u[1]]), np.array([brake, 0.0])):
        if step(world, cfg, x, cand).feasible:
            return cand, True
    return np.array([brake, 0.0]), True


# ------------------------------------------
# NavGrid
# ------------------------------------------
# Navigation grid over the arena with travel-time shortest paths (final-spec §7.4).
#
# A cell is free iff its centre is feasible and its safe speed (clearance-limited, F19) is at least
# ``v_min``. Edges join 8-neighbours (diagonals only if both orthogonal neighbours are free) with cost
# ``length / min(speed at the two centres)``, ``speed = min(v_lim, safe_speed(clearance))``.
class NavGrid:
    def __init__(self, world: World, cfg: RobotConfig, cell: float = 0.1, v_min: float = 0.15) -> None:
        self.world, self.cfg, self.cell = world, cfg, float(cell)
        x0, x1, y0, y1 = world.arena
        nx, ny = round((x1 - x0) / cell), round((y1 - y0) / cell)
        self.xs = x0 + (np.arange(nx) + 0.5) * cell
        self.ys = y0 + (np.arange(ny) + 0.5) * cell
        X, Y = np.meshgrid(self.xs, self.ys)
        xy = np.stack([X, Y], axis=-1)
        self.speed = np.minimum(world.v_lim(xy, cfg), safe_speed(cfg, world.clearance(xy, cfg)))
        self.free = world.feasible(xy, cfg) & (self.speed >= v_min)
        self.shape = (ny, nx)
        self._graph = self._build()

    def _build(self):
        ny, nx = self.shape
        idx = np.arange(ny * nx).reshape(ny, nx)
        rows, cols, cost = [], [], []
        for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
            a = (slice(0, ny - dy), slice(max(0, -dx), nx - max(0, dx)))
            b = (slice(dy, ny), slice(max(0, dx), nx - max(0, -dx)))
            ok = self.free[a] & self.free[b]
            if dy and dx:                                    # diagonal: both orthogonal neighbours must be free
                ok &= self.free[(a[0], b[1])] & self.free[(b[0], a[1])]
            c = self.cell * math.hypot(dy, dx) / np.maximum(np.minimum(self.speed[a], self.speed[b]), 1e-9)
            rows.append(idx[a][ok])
            cols.append(idx[b][ok])
            cost.append(c[ok])
        r, c, w = np.concatenate(rows), np.concatenate(cols), np.concatenate(cost)
        n = ny * nx
        both = (np.concatenate([r, c]), np.concatenate([c, r]))
        return coo_matrix((np.concatenate([w, w]), both), shape=(n, n)).tocsr()

    def centre(self, cell: tuple[int, int]) -> np.ndarray:
        return np.array([self.xs[cell[1]], self.ys[cell[0]]])

    # Nearest free cell ``(iy, ix)`` to ``xy``.
    def cell_of(self, xy) -> tuple[int, int]:
        xy = np.asarray(xy, dtype=float).reshape(2)
        ix = min(max(int((xy[0] - self.world.arena[0]) / self.cell), 0), self.shape[1] - 1)
        iy = min(max(int((xy[1] - self.world.arena[2]) / self.cell), 0), self.shape[0] - 1)
        if self.free[iy, ix]:
            return iy, ix
        fy, fx = np.nonzero(self.free)
        k = int(np.argmin((self.xs[fx] - xy[0]) ** 2 + (self.ys[fy] - xy[1]) ** 2))
        return int(fy[k]), int(fx[k])

    # Travel time [s] from the cell of ``xy`` to every cell (``inf`` if unreachable) and predecessors.
    def travel_time(self, xy) -> tuple[np.ndarray, np.ndarray]:
        iy, ix = self.cell_of(xy)
        dist, pred = dijkstra(self._graph, directed=False, indices=iy * self.shape[1] + ix, return_predecessors=True)
        return dist.reshape(self.shape), pred

    # Cell centres ``(n, 2)`` along the shortest path from the start cell to ``goal`` (inclusive).
    def path(self, pred: np.ndarray, goal: tuple[int, int]) -> np.ndarray:
        nx = self.shape[1]
        k = goal[0] * nx + goal[1]
        out = [k]
        while pred[k] >= 0:
            k = int(pred[k])
            out.append(k)
        out = np.array(out[::-1])
        return np.column_stack([self.xs[out % nx], self.ys[out // nx]])
