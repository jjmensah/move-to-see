# ==========================================================
# sim/experiments/F06_ablations.py
# ==========================================================
# F06: paired, one-factor-at-a-time C2 ablations; descriptive, without pass gates.

from __future__ import annotations

from pathlib import Path

import numpy as np

from lidar_sim.viz import plots
from mts.config import RobotConfig
from mts.metrics import bootstrap_ci, dwell_time, position_error, time_score, wilson
from mts.robot import step
from mts.viz import CTRL_LABEL, ctrl_style
from mts.world import yard
from sim.experiments._common import load_trial, run_jobs, trial_job


# ------------------------------------------
# variants
# ------------------------------------------
# Reuse the M=24, lambda=0.05, rough-on, ordinary-background baseline.
def variants(params):
    return [("baseline", {}),
            *[(f"M={m}", {"planner": {"M": m}}) for m in params["support_sizes"] if m != 24],
            *[(f"lambda={v:g}", {"planner": {"lambda_t": v}}) for v in params["travel_penalties"] if v != 0.05],
            ("rough off", {"world": {"rough": False}}),
            ("dark background", {"world": {"dark_background": True}})]


# ------------------------------------------
# patch_exposure
# ------------------------------------------
# Replay logged controls; integrate centre occupancy on 0.02 s substep midpoints.
#
# Always use the baseline patch footprint, including when its speed limit is off.
# Distance is the sum of substep chords inside that footprint.
def patch_exposure(row, world):
    cfg, footprint = RobotConfig(), yard()
    duration, length = 0., 0.
    for k, u in enumerate(row["controls"]):
        traj = step(world, cfg, row["states"][k], u, t0=k * cfg.dt)
        times = traj.t0 + np.arange(cfg.n_sub + 1) * cfg.substep
        states = np.array([traj.state(t) for t in times])
        inside = footprint.in_rough(np.array([traj.state(t)[:2] for t in times[:-1] + cfg.substep / 2]))
        duration += inside.sum() * cfg.substep
        length += np.linalg.norm(np.diff(states[:, :2], axis=0), axis=1)[inside].sum()
    return float(duration), float(length)


# ------------------------------------------
# run
# ------------------------------------------
# Runs F06: paired fog ablations of the planner and the world.
def run(entry: dict, out: Path, quick: bool, jobs: int) -> dict:
    p = entry["params"]
    n, steps = (2, 3) if quick else (entry["n"], p["steps"])
    sweep = variants(p)
    todo = []
    for tag, cfg in sweep:
        for ctrl in ("info", "fixed"):
            for t in range(n):
                overrides = {**cfg, **({"belief": {"n_particles": 64}} if quick else {})}
                todo.append(trial_job(out, tag, "C2", ctrl, entry["seed"], t, steps, **overrides))
    run_jobs(todo, jobs, "F06")
    summary = {}
    for vi, (tag, cfg) in enumerate(sweep):
        summary[tag] = {}
        for ci, ctrl in enumerate(("info", "fixed")):
            rows = [load_trial(j) for j in todo if j["key"].split("/")[0] == tag and j["controller"] == ctrl]
            errors = [position_error(r["q_hat"], r["q_true"]) for r in rows]
            scores = np.array([time_score(e, float(r["dt"])) for e, r in zip(errors, rows, strict=True)])
            ok = np.array([np.isfinite(dwell_time(e, float(r["dt"]))) for e, r in zip(errors, rows, strict=True)])
            lengths = [float(np.linalg.norm(np.diff(r["states"][:, :2], axis=0), axis=1).sum()) for r in rows]
            result = {"n": n, "success_rate": float(ok.mean()), "success_wilson95": list(wilson(int(ok.sum()), n, 0.95)),
                      "median_time_score_s": float(np.median(scores)),
                      "time_score_ci95": list(bootstrap_ci(scores, rng=np.random.default_rng([entry["seed"], 777, vi, ci]))),
                      "mean_path_length_m": float(np.mean(lengths)), "path_lengths_m": lengths,
                      "time_scores_s": scores.tolist(), "success": ok.tolist()}
            if tag in ("baseline", "rough off"):
                exposure = [patch_exposure(r, yard(**cfg.get("world", {}))) for r in rows]
                result.update(patch_times_s=[v[0] for v in exposure], patch_path_lengths_m=[v[1] for v in exposure],
                              mean_patch_time_s=float(np.mean([v[0] for v in exposure])),
                              mean_patch_path_length_m=float(np.mean([v[1] for v in exposure])))
            summary[tag][ctrl] = result
    fig, axs = plots.figure(1, 3, height_in=3.1)
    xs = np.arange(len(sweep))
    for ci, ctrl in enumerate(("info", "fixed")):
        st = ctrl_style(ctrl)
        rows = [summary[tag][ctrl] for tag, _ in sweep]
        med = np.array([r["median_time_score_s"] for r in rows])
        ci95 = np.array([r["time_score_ci95"] for r in rows])
        x = xs + (ci - 0.5) * 0.22
        axs[0].errorbar(x, med, yerr=np.maximum([med - ci95[:, 0], ci95[:, 1] - med], 0),
                        fmt=st["marker"], color=st["color"], label=CTRL_LABEL[ctrl], capsize=2)
        for ax, key in zip(axs[1:], ("success_rate", "mean_path_length_m"), strict=True):
            ax.plot(x, [r[key] for r in rows], st["marker"], color=st["color"])
    for ax, label in zip(axs, ("median time score [s], 95% CI", "success rate", "mean path length [m]"), strict=True):
        ax.set_xticks(xs, [tag for tag, _ in sweep], rotation=55, ha="right", fontsize=7)
        ax.set_ylabel(label)
    axs[1].set_ylim(-0.03, 1.03)
    plots.legend_below(fig, axs[0], ncols=2)
    plots.save(fig, out / "ablations.pdf")
    return {"exploratory": True, "condition": "C2", "n_paired_trials": n, "steps": steps,
            "summary": summary, "patch_integration_step_s": RobotConfig().substep,
            "omitted_variant": "report-flag-only utility", "verdicts": {}}
