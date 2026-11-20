# ==========================================================
# sim/experiments/M01_media.py
# ==========================================================
# M01 — Media: one fog trial animated for each controller, plus a keyframe figure (vv-plan §3).

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.patches import Polygon

from lidar_sim.viz import plots
from mts.config import CONDITIONS, SENSOR
from mts.fov import r50
from mts.metrics import position_error, time_score
from mts.sim import run_trial
from mts.viz import (
    COND_LABEL,
    CTRL_LABEL,
    animate_trial,
    cov_ellipse,
    ctrl_style,
    draw_robot,
    draw_target,
    draw_world,
    fov_footprint,
)
from mts.world import yard

CTRLS = ("info", "blind", "fixed")


# ------------------------------------------
# run
# ------------------------------------------
# Runs M01: one fog trial per controller, rendered as animations and keyframes.
def run(entry: dict, out: Path, quick: bool, jobs: int) -> dict:
    p = entry["params"]
    cond = CONDITIONS[p["condition"]]
    steps = 16 if quick else p["steps"]
    world = yard(cond.target_material)
    r_eff = r50(0.05, cond.env)
    recs, files, checks, drawn_ok = {}, [], {}, True
    for ctrl in CTRLS:
        rec = run_trial(world, cond, ctrl, entry["seed"], p["trial"], steps=steps, keep=True)
        recs[ctrl] = rec
        title = f"{CTRL_LABEL[ctrl]} — {COND_LABEL[cond.name]}"
        mp4, drawn = animate_trial(world, rec, out / f"{ctrl}.mp4", r_eff=r_eff, title=title, fps=8, intro_s=8.0)
        files.append(mp4)
        err = position_error(rec.q_hat, rec.q_true)
        drawn_ok &= bool(drawn.shape == err.shape and np.allclose(drawn, np.maximum(err, 1e-3), rtol=0, atol=1e-12))
        checks[ctrl] = {"final_error_m": float(err[-1]), "time_score_s": float(time_score(err, rec.dt)),
                        "path_length_m": float(np.hypot(*np.diff(rec.states[:, :2], axis=0).T).sum())}
    _keyframes(world, recs, r_eff, out / "keyframes.pdf", cond.name)
    exist = all(f.exists() and f.stat().st_size > 10_000 for f in [*files, out / "keyframes.pdf"])
    with (out / "keyframes.pdf").open("rb") as fh:
        exist &= fh.read(5) == b"%PDF-"
    return {"condition": cond.name, "trial": p["trial"], "q_true": recs["info"].q_true.tolist(), "controllers": checks,
            "files": [f.name for f in files] + ["keyframes.pdf"],
            "verdicts": {"media_files_written": bool(exist), "drawn_error_matches_record": bool(drawn_ok)}}


# ------------------------------------------
# _keyframes
# ------------------------------------------
# Rows: controllers; columns: three instants. Shows path so far, particles, belief ellipse, sensing footprint.
def _keyframes(world, recs, r_eff, path: Path, cond: str) -> None:
    T = recs["info"].controls.shape[0]
    ks = [round(T * f) for f in (0.2, 0.5, 1.0)]
    fig, axs = plots.figure(3, 3, height_in=1.95, sharex=True, sharey=True)
    for i, ctrl in enumerate(CTRLS):
        rec, st = recs[ctrl], ctrl_style(ctrl)
        err = position_error(rec.q_hat, rec.q_true)
        for j, k in enumerate(ks):
            ax = axs[i, j]
            draw_world(ax, world)
            x = rec.states[k]
            ax.add_patch(Polygon(fov_footprint(x, SENSOR.r_max), fc="none", ec=plots.INK_2, lw=0.5, ls=(0, (4, 3)), zorder=2))
            ax.add_patch(Polygon(fov_footprint(x, min(r_eff, SENSOR.r_max)), fc=plots.FOV_FILL, ec="none", alpha=0.35, zorder=2))
            pts, w = rec.extra["particles"][k], rec.extra["weights"][k]
            keep = w > 0.05 * w.max()
            ax.scatter(pts[keep, 0], pts[keep, 1], s=1.5, color="#d40000", alpha=0.3, linewidths=0, zorder=3.5)
            draw_target(ax, world, rec.q_true, fc="none", ec="#111111", lw=0.9)
            ax.plot(rec.states[: k + 1, 0], rec.states[: k + 1, 1], color=st["color"], lw=1.4, zorder=4)
            ax.add_patch(cov_ellipse(rec.q_hat[k], rec.cov[k], 2.0, fc="none", ec="#d40000", lw=0.9, zorder=5))
            draw_robot(ax, x)
            ax.set_title(f"t = {k * rec.dt:g} s, error {err[k]:.2f} m" if err[k] >= 0.01 else
                         f"t = {k * rec.dt:g} s, error {1e3 * err[k]:.0f} mm", fontsize=8)
            ax.set_xlabel("x [m]" if i == 2 else "")
            ax.set_ylabel(f"{CTRL_LABEL[ctrl].split(' (')[0]}\ny [m]" if j == 0 else "", fontsize=8)
    fig.suptitle(f"{COND_LABEL[cond]}: black target (outlined), particles and 2σ belief ellipse (red), "
                 f"nominal detection range of the target, not clipped by obstacles (blue)", fontsize=8.5)
    plots.save(fig, path)
