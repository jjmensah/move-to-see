# ==========================================================
# src/mts/viz.py
# ==========================================================
# Figures and animation for trials: top view of the yard with paths, belief and fields of view.

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse, Patch, Polygon, Rectangle

from lidar_sim.viz import plots
from mts.config import RASTER, SENSOR, RobotConfig
from mts.world import World

CTRL_LABEL = {"info": "information-driven (physics-aware)", "blind": "physics-blind planner", "fixed": "fixed route"}
CTRL_STYLE = {"info": 0, "blind": 1, "fixed": 2}                 # series slot (colour, marker, line style)
COND_LABEL = {"C1": "clear", "C2": "fog (V = 20 m)", "C3": "sunlight (1 sun)", "C4": "white target, clear"}
ROUGH = "#e9dcc6"
KEEPOUT = "#f0efec"


# ------------------------------------------
# ctrl_style
# ------------------------------------------
# Plot colour, line and marker of a controller.
def ctrl_style(name: str) -> dict:
    return plots.series(CTRL_STYLE[name])


# ------------------------------------------
# draw_world
# ------------------------------------------
# Top view: arena, rough terrain, obstacles, target keep-out and the prior region of the target centre.
def draw_world(ax, world: World, cfg: RobotConfig = RobotConfig(), keepout: bool = True,  # noqa: B008
               prior: bool = True, labels: bool = False) -> None:
    x0, x1, y0, y1 = world.arena
    for r in world.rough:
        ax.add_patch(Rectangle((r[0], r[2]), r[1] - r[0], r[3] - r[2], fc=ROUGH, ec="#a08a65", lw=0.6, zorder=0.5))
        if labels:
            ax.text(0.5 * (r[0] + r[1]), 0.5 * (r[2] + r[3]), "rough terrain", ha="center", va="center", fontsize=7,
                    color="#6b5a3c", zorder=0.6)
    if keepout:
        k = world.keepouts(cfg)[-1]
        ax.add_patch(Rectangle((k[0], k[2]), k[1] - k[0], k[3] - k[2], fc=KEEPOUT, ec=plots.INK_2, lw=0.6,
                               ls=(0, (3, 2)), zorder=0.7))
    if prior:
        lo, hi = world.prior_lo, world.prior_hi
        ax.add_patch(Rectangle((lo[0], lo[1]), hi[0] - lo[0], hi[1] - lo[1], fc="none", ec=plots.INK_2, lw=0.8,
                               ls=":", zorder=0.8))
        if labels:
            ax.text(hi[0] - 0.05, hi[1] - 0.08, "prior region", ha="right", va="top", fontsize=7, color=plots.INK_2)
    for lo, hi in zip(world.box_lo, world.box_hi, strict=True):
        ax.add_patch(Rectangle((lo[0], lo[1]), hi[0] - lo[0], hi[1] - lo[1], fc=plots.WALL, ec=plots.WALL, lw=0.4,
                               zorder=1))
    ax.set_xlim(x0 - 0.1, x1 + 0.1)
    ax.set_ylim(y0 - 0.1, y1 + 0.1)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set(xlabel="x [m]", ylabel="y [m]")


# ------------------------------------------
# draw_target
# ------------------------------------------
# Draws the target's footprint.
def draw_target(ax, world: World, q, **kw) -> None:
    lo, hi = world.target_box(np.asarray(q, dtype=float))
    kw = {"fc": "#111111", "ec": "#111111", "lw": 0.5, "zorder": 3, **kw}
    ax.add_patch(Rectangle((lo[0], lo[1]), hi[0] - lo[0], hi[1] - lo[1], **kw))


# ------------------------------------------
# draw_robot
# ------------------------------------------
# Draws the robot as a disc with a heading tick.
def draw_robot(ax, x, cfg: RobotConfig = RobotConfig(), color=plots.INK, zorder=6) -> list:  # noqa: B008
    body = Circle((x[0], x[1]), cfg.radius, fc="white", ec=color, lw=1.0, zorder=zorder)
    tip = (x[0] + cfg.radius * math.cos(x[2]), x[1] + cfg.radius * math.sin(x[2]))
    (line,) = ax.plot([x[0], tip[0]], [x[1], tip[1]], color=color, lw=1.2, zorder=zorder + 0.1)
    ax.add_patch(body)
    return [body, line]


# ------------------------------------------
# fov_footprint
# ------------------------------------------
# Polygon of the azimuth sector of the given radius around the robot (plot aid, not a model).
def fov_footprint(x, radius: float, n: int = 40) -> np.ndarray:
    a = x[2] + np.linspace(-RASTER.fov / 2, RASTER.fov / 2, n)
    return np.vstack([x[:2], np.column_stack([x[0] + radius * np.cos(a), x[1] + radius * np.sin(a)])])


# ------------------------------------------
# cov_ellipse
# ------------------------------------------
# Ellipse of the (x, y) marginal covariance at ``n_std`` standard deviations.
def cov_ellipse(mean, cov, n_std: float = 2.0, **kw) -> Ellipse:
    vals, vecs = np.linalg.eigh(np.asarray(cov)[:2, :2])
    vals = np.maximum(vals, 0.0)
    ang = math.degrees(math.atan2(vecs[1, 1], vecs[0, 1]))
    return Ellipse((mean[0], mean[1]), 2 * n_std * math.sqrt(vals[1]), 2 * n_std * math.sqrt(vals[0]), angle=ang, **kw)


# ------------------------------------------
# plot_paths
# ------------------------------------------
# Paths of several controllers on one trial (same target), with the true target and final estimates.
def plot_paths(ax, world: World, trials: dict, cfg: RobotConfig = RobotConfig(), title: str | None = None) -> None:  # noqa: B008
    draw_world(ax, world, cfg)
    q = next(iter(trials.values()))["q_true"]
    draw_target(ax, world, q)
    for name, tr in trials.items():
        st = ctrl_style(name)
        s = tr["states"]
        ax.plot(s[:, 0], s[:, 1], color=st["color"], ls=st["linestyle"], lw=1.6, zorder=4, label=CTRL_LABEL[name])
        ax.plot(s[::10, 0], s[::10, 1], st["marker"], color=st["color"], ms=3.2, zorder=4.1)
        ax.plot(*s[-1, :2], st["marker"], color=st["color"], ms=6, mec=plots.INK, mew=0.6, zorder=5)
    ax.plot(*world.start[:2], "o", color=plots.INK, ms=4, zorder=5)
    if title:
        ax.set_title(title)


INTRO = {
    "common": "Move to See: a unicycle rover with a 3D AMCW LiDAR must localize a dark target that is partly hidden "
              "behind crates. Every 0.5 s it scans, updates a 2000-particle belief over the target centre with the "
              "physics-based beam likelihood (which knows that fog and sunlight shorten the range at which a dark "
              "surface still returns light), and chooses where to drive next.",
    "info": "This rover: INFORMATION-DRIVEN PLANNER. It scores candidate viewpoints by J = (lower bound on the "
            "expected information gain of the next scan) - 0.05 nat/s x (travel time) and drives towards the best "
            "one (star). Because its sensor model knows the reduced detection range, it goes close to the target.",
    "blind": "This rover: PHYSICS-BLIND PLANNER. The same planner, but its sensor model assumes every surface inside "
             "the 5.6 m range window returns a measurement, so it overrates distant viewpoints and stays too far away "
             "to see the dark target in fog.",
    "fixed": "This rover: FIXED INSPECTION ROUTE. It drives a pre-set loop around the yard and does not use the "
             "belief to decide where to go; the belief is still updated from every scan.",
    "legend": "Left: top view. Rover (circle with heading), its path, next goal (star), particles and 2-sigma belief "
              "ellipse (red), true target (black outline, shown to the viewer only, never to the rover). Blue: nominal "
              "detection range of the black target in this weather, not clipped by obstacles; dashed: 5.6 m range "
              "window. Right: position error of the belief mean and belief standard deviations.",
}


# ------------------------------------------
# _legend_handles
# ------------------------------------------
# Legend entries for the animation frames.
def _legend_handles(color) -> list:
    return [Line2D([], [], color=color, lw=1.8, label="rover path"),
            Line2D([], [], ls="none", marker="*", color=color, mec=plots.INK, mew=0.5, ms=9, label="next goal"),
            Line2D([], [], ls="none", marker="o", color="#d40000", alpha=0.5, ms=3, label="particles"),
            Patch(fc="none", ec="#d40000", lw=1.2, label="2-sigma belief"),
            Patch(fc="none", ec="#111111", lw=1.2, label="true target"),
            Patch(fc=plots.FOV_FILL, alpha=0.35, label="nominal detection range (not clipped)"),
            Patch(fc="none", ec=plots.INK_2, lw=0.6, ls=(0, (4, 3)), label="5.6 m range window")]


# ------------------------------------------
# _intro_frame
# ------------------------------------------
# A title card that explains the method, the controller and the legend (video only).
def _intro_frame(controller: str, title: str):
    import textwrap

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(9.6, 5.4), dpi=110)
    fig.text(0.06, 0.90, title, fontsize=15, weight="bold", va="top")
    y = 0.80
    for key, size in (("common", 10.5), (controller, 10.5), ("legend", 9.5)):
        block = textwrap.fill(INTRO[key], 105 if size > 10 else 115)
        fig.text(0.06, y, block, fontsize=size, va="top", linespacing=1.35)
        y -= 0.042 * (block.count("\n") + 1) + 0.06
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return frame


# ------------------------------------------
# animate_trial
# ------------------------------------------
# Animation of one trial recorded with ``keep=True``: top view with particles and belief, and error curve.
#
# ``rec`` is an ``mts.sim.Trial``. Truth (target, error) is drawn for the viewer only. With ``intro_s > 0`` the
# video opens with a title card explaining the method, held for that many seconds. Returns the file path
# and the error series plotted in the final frame (floored at 1 mm for the log axis), so that callers can
# check the drawing against the record.
def animate_trial(world: World, rec, path: Path, cfg: RobotConfig = RobotConfig(), r_eff: float | None = None,  # noqa: B008
                  title: str = "", fps: int = 8, tol: float = 0.10, intro_s: float = 0.0) -> tuple[Path, np.ndarray]:
    import imageio.v2 as imageio
    import matplotlib.pyplot as plt

    plots.apply_style()
    T = rec.controls.shape[0]
    err = np.linalg.norm(rec.q_hat - rec.q_true, axis=1)
    t = np.arange(T + 1) * rec.dt
    st = ctrl_style(rec.controller)
    frames = []
    for k in range(T + 1):
        fig = plt.figure(figsize=(9.6, 5.4), dpi=110)
        ax = fig.add_axes([0.05, 0.2, 0.58, 0.7])
        draw_world(ax, world, cfg, labels=True)
        x = rec.states[k]
        if r_eff is not None:
            ax.add_patch(Polygon(fov_footprint(x, SENSOR.r_max), fc="none", ec=plots.INK_2, lw=0.6, ls=(0, (4, 3)), zorder=2))
            ax.add_patch(Polygon(fov_footprint(x, min(r_eff, SENSOR.r_max)), fc=plots.FOV_FILL, ec="none", alpha=0.35, zorder=2))
        p, w = rec.extra["particles"][k], rec.extra["weights"][k]
        keep = w > 0.05 * w.max()
        ax.scatter(p[keep, 0], p[keep, 1], s=3, color="#d40000", alpha=0.35, linewidths=0, zorder=3.5)
        draw_target(ax, world, rec.q_true, fc="none", ec="#111111", lw=1.2)
        ax.plot(rec.states[: k + 1, 0], rec.states[: k + 1, 1], color=st["color"], lw=1.8, zorder=4)
        if k < T and np.isfinite(rec.goals[k]).all():
            ax.plot(*rec.goals[k], "*", color=st["color"], ms=11, mec=plots.INK, mew=0.5, zorder=5)
        ax.add_patch(cov_ellipse(rec.q_hat[k], rec.cov[k], 2.0, fc="none", ec="#d40000", lw=1.2, zorder=5))
        draw_robot(ax, x, cfg)
        ax.set_title(f"{title}   t = {t[k]:4.1f} s", loc="left", fontsize=10)
        fig.legend(handles=_legend_handles(st["color"]), loc="lower left", bbox_to_anchor=(0.03, 0.0), ncols=4,
                   fontsize=7, frameon=False, handlelength=1.6, columnspacing=1.2)
        ax2 = fig.add_axes([0.70, 0.56, 0.27, 0.32])
        (curve,) = ax2.semilogy(t[: k + 1], np.maximum(err[: k + 1], 1e-3), color=st["color"], lw=1.6)
        drawn = np.asarray(curve.get_ydata(), dtype=float).copy()         # what the last frame actually shows
        ax2.axhline(tol, color=plots.INK_2, ls="--", lw=0.8)
        ax2.set(xlim=(0, t[-1]), ylim=(1e-3, 3), xlabel="time [s]", ylabel="position error [m]")
        ax2.text(t[-1], tol * 1.15, "0.10 m", ha="right", va="bottom", fontsize=7, color=plots.INK_2)
        ax3 = fig.add_axes([0.70, 0.1, 0.27, 0.32])
        sd = np.sqrt(np.einsum("kii->ki", rec.cov[: k + 1]))
        for j, lab in enumerate("xyz"):
            s3 = plots.series(j)
            ax3.semilogy(t[: k + 1], np.maximum(sd[:, j], 1e-3), color=s3["color"], ls=s3["linestyle"], lw=1.3, label=lab)
        ax3.set(xlim=(0, t[-1]), ylim=(1e-3, 3), xlabel="time [s]", ylabel="belief std [m]")
        ax3.legend(ncols=3, fontsize=7, loc="upper right")
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    intro = [_intro_frame(rec.controller, title)] * round(intro_s * fps) if intro_s > 0 else []
    imageio.mimsave(path, intro + frames + [frames[-1]] * fps, fps=fps, macro_block_size=1)
    return path, drawn
