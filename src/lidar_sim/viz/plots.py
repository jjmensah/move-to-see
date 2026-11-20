# ==========================================================
# src/lidar_sim/viz/plots.py
# ==========================================================
# Shared figure style and drawing helpers for experiment figures (static PDF for the LaTeX report).
#
# Palette: validated reference categorical slots (dataviz skill; all-pairs CVD ΔE ≥ 9.2 for the first three,
# adjacent-pairs pass for four). Slot 3 (aqua) and 4 (yellow) are < 3:1 against the surface, so every
# multi-series figure carries a legend AND a secondary encoding (marker shape / line style). Sequential
# magnitude uses one blue ramp (light → dark); signed quantities use blue ↔ red with a grey midpoint.
# Rules: one y-axis per panel, colour follows the entity (never its rank), text in ink colours only.

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
LINESTYLES = ["-", "--", "-.", ":", "-", "--", "-.", ":"]
SEQ_STEPS = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIV_STEPS = ["#0d366b", "#2a78d6", "#86b6ef", "#f0efec", "#f19c9b", "#e34948", "#8f1f1f"]
SEQ = LinearSegmentedColormap.from_list("lidar_seq", SEQ_STEPS)
DIV = LinearSegmentedColormap.from_list("lidar_div", DIV_STEPS)
WALL = "#3d3c39"                         # scene geometry (neutral ink, not a series colour)
FOV_FILL = "#86b6ef"                     # sequential blue step 250, translucent

TEXT_WIDTH_IN = 6.3                      # LaTeX 1-inch margins, letter paper


# ------------------------------------------
# apply_style
# ------------------------------------------
# Install the report rcParams (idempotent).
def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
        "text.color": INK, "axes.titlecolor": INK, "legend.labelcolor": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
        "lines.linewidth": 1.6, "lines.markersize": 5.0, "patch.linewidth": 0.0,
        "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.frameon": False,
        "axes.prop_cycle": matplotlib.cycler(color=SERIES),
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
        "mathtext.fontset": "dejavusans",
    })


# ------------------------------------------
# figure
# ------------------------------------------
# New styled figure sized for the report text width.
def figure(n_rows: int = 1, n_cols: int = 1, height_in: float = 2.6, width_in: float = TEXT_WIDTH_IN, **kw):
    apply_style()
    return plt.subplots(n_rows, n_cols, figsize=(width_in, height_in * n_rows), layout="constrained", **kw)


# ------------------------------------------
# series
# ------------------------------------------
# Matplotlib kwargs for categorical series slot ``i`` (colour + marker + line style, fixed order).
def series(i: int) -> dict:
    if i >= len(SERIES):
        raise ValueError("more than 8 series: fold into 'Other' or use small multiples")
    return {"color": SERIES[i], "marker": MARKERS[i], "linestyle": LINESTYLES[i]}


# ------------------------------------------
# save
# ------------------------------------------
# Save ``fig`` as PDF (report) and optionally PNG (preview); returns written paths.
def save(fig, path: str | Path, png: bool = True) -> list[Path]:
    path = Path(path).with_suffix(".pdf")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    out = [path]
    if png:
        fig.savefig(path.with_suffix(".png"))
        out.append(path.with_suffix(".png"))
    plt.close(fig)
    return out


# ------------------------------------------
# draw_scene
# ------------------------------------------
# Draw a 2D ``SceneSpec`` (static segments + movers posed at time ``t``) in neutral ink.
def draw_scene(ax, spec, t: float = 0.0, lw: float = 1.8) -> None:
    for a, b in spec.segments:
        ax.plot([a[0], b[0]], [a[1], b[1]], color=WALL, lw=lw, solid_capstyle="round", zorder=2)
    for m in spec.movers:
        seg = m.pose_at(t).apply(m.segments_local.reshape(-1, 2)).reshape(-1, 2, 2)
        for a, b in seg:
            ax.plot([a[0], b[0]], [a[1], b[1]], color=WALL, lw=lw, zorder=2)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


# ------------------------------------------
# draw_fov
# ------------------------------------------
# Shade the geometric FOV sector (E3) of a 2D pose.
def draw_fov(ax, pose, fov: float, r_max: float, n: int = 120, alpha: float = 0.18, label: str | None = None):
    xi = pose.xi()
    ang = xi[2] + np.linspace(-fov / 2, fov / 2, n)
    pts = np.vstack([xi[:2], np.c_[xi[0] + r_max * np.cos(ang), xi[1] + r_max * np.sin(ang)], xi[:2]])
    ax.fill(pts[:, 0], pts[:, 1], color=FOV_FILL, alpha=alpha, lw=0, zorder=1, label=label)


# ------------------------------------------
# draw_sensor
# ------------------------------------------
# Sensor glyph: dot + heading tick.
def draw_sensor(ax, pose, size: float = 0.12) -> None:
    xi = pose.xi()
    ax.plot(*xi[:2], "o", color=INK, ms=4, zorder=5)
    ax.plot([xi[0], xi[0] + size * 2 * np.cos(xi[2])], [xi[1], xi[1] + size * 2 * np.sin(xi[2])],
            color=INK, lw=1.4, zorder=5)


# ------------------------------------------
# scan_points
# ------------------------------------------
# World-frame points (n_detected, 2) of a 2D Scan: beam origin + range × central sub-ray direction.
def scan_points(scan, pose_fn=None) -> np.ndarray:
    tr = scan.truth
    k = np.argmax(tr.w, axis=1)
    d = tr.dir_world[np.arange(tr.B), k]
    if callable(tr.pose):
        o = np.array([tr.pose(t).t for t in tr.fire_time])
    else:
        o = np.broadcast_to(tr.pose.t, (tr.B, 2))
    m = scan.detected
    return o[m] + scan.range[m, None] * d[m]


# ------------------------------------------
# legend_below
# ------------------------------------------
# One shared legend outside the plotting area (never on top of data).
def legend_below(fig, ax=None, ncols: int = 3) -> None:
    src = ax if ax is not None else fig.axes[0]
    handles, labels = src.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=ncols)
