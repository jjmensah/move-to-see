# ==========================================================
# src/mts/sim.py
# ==========================================================
# Closed-loop trial: robot, waveform-level sensor truth, belief update and controller (final-spec §9).

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from mts.belief import ParticleBelief
from mts.config import RASTER, T_STEPS, BeliefConfig, Condition, PlannerConfig, RobotConfig, make_sensor
from mts.predict import Observation, Predictor
from mts.robot import step
from mts.world import TARGET_OBJECT, World

_PURPOSE = {"target": 0, "filter": 1, "planner": 2, "noise": 3}


# ------------------------------------------
# stream
# ------------------------------------------
# Independent random stream for ``(experiment seed, trial, purpose[, scan index])`` (final-spec §9).
#
# Paired controllers share the target draw and the measurement-noise stream of every scan index.
def stream(seed: int, trial: int, purpose: str, k: int | None = None) -> np.random.Generator:
    key = [int(seed), int(trial), _PURPOSE[purpose]] + ([] if k is None else [int(k)])
    return np.random.default_rng(np.random.SeedSequence(key))


# ------------------------------------------
# draw_target
# ------------------------------------------
# Target centre ``q ~ Uniform(R_T)`` from the trial's target stream.
def draw_target(world: World, seed: int, trial: int) -> np.ndarray:
    return stream(seed, trial, "target").uniform(world.prior_lo, world.prior_hi)


# ------------------------------------------
# make_controller
# ------------------------------------------
# Controller by name: ``"info"`` (physics-aware planner), ``"blind"`` (physics-blind), ``"fixed"``.
def make_controller(name: str, world: World, predictor: Predictor, robot_cfg: RobotConfig,
                    planner_cfg: PlannerConfig):
    if name == "fixed":
        from mts.route import FixedRoute
        return FixedRoute(world, robot_cfg)
    if name in ("info", "blind"):
        from mts.planner import ViewpointPlanner
        return ViewpointPlanner(world, robot_cfg, planner_cfg, predictor, kind="aware" if name == "info" else "blind")
    raise ValueError(f"unknown controller {name!r}")


# ------------------------------------------
# Trial
# ------------------------------------------
# Record of one trial. Arrays over steps have length T; states and estimates have T + 1 (initial first).
@dataclass(eq=False)
class Trial:
    controller: str
    condition: str
    seed: int
    trial: int
    q_true: np.ndarray
    dt: float
    states: np.ndarray          # (T+1, 4)
    controls: np.ndarray        # (T, 2)
    goals: np.ndarray           # (T, 2), NaN if none
    fallback: np.ndarray        # (T,) bool
    infeasible: np.ndarray      # (T,) bool: commanded step was infeasible and the emergency brake was used
    n_detected: np.ndarray      # (T,) reported beams
    n_target_detected: np.ndarray   # (T,) reported beams whose strongest return is the target (evaluation only)
    q_hat: np.ndarray           # (T+1, 3)
    cov: np.ndarray             # (T+1, 3, 3)
    ess: np.ndarray             # (T,)
    stages: np.ndarray          # (T,) tempering stages
    score: np.ndarray           # (T,) planner score of the chosen goal (NaN if none)
    wall_time: np.ndarray       # (T, 2) seconds spent in planning and in the belief update
    extra: dict = field(default_factory=dict)

    def to_npz(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if isinstance(v, np.ndarray)}
        d.update(controller=self.controller, condition=self.condition, seed=self.seed, trial=self.trial, dt=self.dt)
        return d


# ------------------------------------------
# run_trial
# ------------------------------------------
# Run one closed-loop trial (final-spec §5.1, §9).
#
# The truth scene, the scan's ``truth`` field and ``q_true`` never reach the controller or the belief: they
# receive the background world, the :class:`Observation` and the robot state.
#
# Parameters
# keep : bool
#     If True, per-step particle snapshots and observations are stored in ``Trial.extra`` (for media).
def run_trial(world: World, condition: Condition, controller: str, seed: int, trial: int,
              robot_cfg: RobotConfig = RobotConfig(), belief_cfg: BeliefConfig = BeliefConfig(),  # noqa: B008
              planner_cfg: PlannerConfig = PlannerConfig(), steps: int = T_STEPS, q_true=None,  # noqa: B008
              keep: bool = False) -> Trial:
    world = world.with_target_material(condition.target_material)
    env = condition.env
    q_true = draw_target(world, seed, trial) if q_true is None else np.asarray(q_true, dtype=float).reshape(3)
    sensor = make_sensor(9)
    truth_scene = world.truth_scene(q_true)
    predictor = Predictor(world, env, chunk=belief_cfg.chunk)
    rng_filter, rng_plan = stream(seed, trial, "filter"), stream(seed, trial, "planner")
    belief = ParticleBelief(world, belief_cfg, rng_filter)
    ctrl = make_controller(controller, world, predictor, robot_cfg, planner_cfg)

    T = int(steps)
    rec = Trial(controller=controller, condition=condition.name, seed=int(seed), trial=int(trial), q_true=q_true,
                dt=robot_cfg.dt, states=np.zeros((T + 1, 4)), controls=np.zeros((T, 2)), goals=np.full((T, 2), np.nan),
                fallback=np.zeros(T, bool), infeasible=np.zeros(T, bool), n_detected=np.zeros(T, int),
                n_target_detected=np.zeros(T, int), q_hat=np.zeros((T + 1, 3)), cov=np.zeros((T + 1, 3, 3)),
                ess=np.zeros(T), stages=np.zeros(T, int), score=np.full(T, np.nan), wall_time=np.zeros((T, 2)))
    x = world.start.copy()
    rec.states[0], rec.q_hat[0], rec.cov[0] = x, belief.mean(), belief.cov()
    if keep:
        rec.extra["particles"] = [belief.particles.copy()]
        rec.extra["weights"] = [belief.weights.copy()]
    for k in range(T):
        t_plan = time.perf_counter()
        act = ctrl.act(k, x.copy(), belief, rng_plan)
        t_plan = time.perf_counter() - t_plan
        traj = step(world, robot_cfg, x, act.u, t0=k * robot_cfg.dt)
        if not traj.feasible:                                   # emergency brake (recorded; should not happen)
            rec.infeasible[k] = True
            traj = step(world, robot_cfg, x, np.array([-min(robot_cfg.a_max, x[3] / robot_cfg.dt), 0.0]),
                        t0=k * robot_cfg.dt)
        t_scan = traj.t1 - RASTER.t_scan
        trace = sensor.trace(traj.pose, truth_scene, env, t0=t_scan)
        scan = sensor.measure(trace, env, stream(seed, trial, "noise", k))
        obs = Observation.from_scan(scan)
        t_upd = time.perf_counter()
        geom = predictor.geometry(traj.pose, t0=t_scan, K=9)
        info = belief.update(lambda q, g=geom, o=obs: predictor.loglik(g, o, q), rng_filter)
        t_upd = time.perf_counter() - t_upd

        x = traj.end.copy()
        rec.states[k + 1], rec.controls[k], rec.fallback[k] = x, traj.u, act.fallback
        if act.goal is not None:
            rec.goals[k] = np.asarray(act.goal, dtype=float)[:2]
        rec.score[k] = float(act.info.get("score", np.nan))
        rec.n_detected[k] = int(scan.detected.sum())
        tr = scan.truth
        strongest = np.take_along_axis(tr.object_id, np.argmax(tr.power, axis=1)[:, None], axis=1)[:, 0]
        rec.n_target_detected[k] = int((scan.detected & (strongest == TARGET_OBJECT)).sum())
        rec.q_hat[k + 1], rec.cov[k + 1] = belief.mean(), belief.cov()
        rec.ess[k], rec.stages[k] = info.ess, info.stages
        rec.wall_time[k] = (t_plan, t_upd)
        if keep:
            rec.extra["particles"].append(belief.particles.copy())
            rec.extra["weights"].append(belief.weights.copy())
            rec.extra.setdefault("obs", []).append(obs)
            rec.extra.setdefault("info", []).append(act.info)
    return rec
