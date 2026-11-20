# ==========================================================
# scripts/concept_sketch.py
# ==========================================================
# Hero concept illustration for the proposal slide (a sketch of the planned scenario, not a result).
#
# Hand-projected oblique view: a rover with a 3D LiDAR, an occluder that hides a dark, elevated target, the
# geometric and effective fields of view, fog, sunlight, rough ground, and two candidate routes.
# Run:  python hero.py   ->  hero.png (7.75 x 3.9 in at 300 dpi)

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, Polygon
from matplotlib.path import Path as MPath

plt.rcParams.update(
    {
        "font.family": ["Helvetica", "Arial"],
        "font.size": 11,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Helvetica",
        "mathtext.it": "Helvetica:italic",
        "hatch.linewidth": 0.5,
    }
)
INK, GREY, BLUE, RED = "#000000", "#7A7A7A", "#1F5FBF", "#D40000"
YEL, ORA, GRN, PBL = "#FFF2CC", "#FFE6CC", "#D5E8D4", "#DAE8FC"
R_GEO, R_EFF, HALF = 7.2, 2.9, np.radians(30)  # illustrative geometric and effective ranges, half-angle


# ------------------------------------------
# P
# ------------------------------------------
# Oblique projection of world (x, y, z) to the page.
def P(x, y, z=0.0):
    return np.array([x + 0.45 * y, 0.5 * y + z])


# ------------------------------------------
# poly
# ------------------------------------------
# Draws a filled polygon given in world coordinates.
def poly(ax, pts, **kw):
    kw.setdefault("lw", 0.8)
    kw.setdefault("ec", INK)
    kw.setdefault("joinstyle", "round")
    ax.add_patch(Polygon([P(*p) for p in pts], closed=True, **kw))


# ------------------------------------------
# box
# ------------------------------------------
# Axis-aligned box showing its top, front (y = y0) and right (x = x1) faces.
def box(ax, x, y, z, dx, dy, dz, top, front, side, z0=5, lw=0.8, ec=INK):
    poly(
        ax,
        [(x + dx, y, z), (x + dx, y + dy, z), (x + dx, y + dy, z + dz), (x + dx, y, z + dz)],
        fc=side,
        zorder=z0,
        lw=lw,
        ec=ec,
    )
    poly(ax, [(x, y, z), (x + dx, y, z), (x + dx, y, z + dz), (x, y, z + dz)], fc=front, zorder=z0, lw=lw, ec=ec)
    poly(
        ax,
        [(x, y, z + dz), (x + dx, y, z + dz), (x + dx, y + dy, z + dz), (x, y + dy, z + dz)],
        fc=top,
        zorder=z0,
        lw=lw,
        ec=ec,
    )


# ------------------------------------------
# sector
# ------------------------------------------
# Points of a flat field-of-view sector on the ground.
def sector(c, heading, radius, half=HALF, n=40):
    a = heading + np.linspace(-half, half, n)
    return [(c[0], c[1], 0)] + [(c[0] + radius * np.cos(t), c[1] + radius * np.sin(t), 0) for t in a]


# ------------------------------------------
# rover
# ------------------------------------------
# Small rover drawn as an axis-aligned body with wheels and a LiDAR puck; returns the puck centre.
def rover(ax, c, z0, ghost=False):
    x, y = c[0] - 0.5, c[1] - 0.32
    body = ("#FFFFFF", "#F4F7FB", "#E6ECF4") if ghost else ("#EAF1FB", PBL, "#B9CDE9")
    ec = BLUE if ghost else INK
    box(ax, x, y, 0.16, 1.0, 0.64, 0.26, *body, z0=z0, ec=ec)
    for wx in (x + 0.22, x + 0.78):  # near-side wheels
        ax.add_patch(
            Ellipse(P(wx, y, 0.15), 0.32, 0.32, fc="#FFFFFF" if ghost else "#333333", ec=ec, lw=0.8, zorder=z0 + 0.001)
        )
    px, py = c[0] + 0.05, c[1]
    base, top = P(px, py, 0.42), P(px, py, 0.62)
    col = "#FFFFFF" if ghost else GRN
    ax.add_patch(
        Polygon(
            [base + (-0.17, 0), base + (0.17, 0), top + (0.17, 0), top + (-0.17, 0)], fc=col, ec=ec, lw=0.8, zorder=z0
        )
    )
    ax.add_patch(Ellipse(base, 0.34, 0.14, fc=col, ec=ec, lw=0.8, zorder=z0 - 0.01))
    ax.add_patch(Ellipse(top, 0.34, 0.14, fc=col, ec=ec, lw=0.8, zorder=z0))
    return np.array([px, py, 0.55])


# ------------------------------------------
# fog
# ------------------------------------------
# A low, soft fog band: a few overlapping flat ellipses in page coordinates.
def fog(ax, cx, cy, w, h, z0, alpha=0.5):
    for dx, dy, sw, sh in [(0, 0, 1.0, 1.0), (-0.28, 0.35, 0.62, 0.8), (0.3, -0.3, 0.7, 0.75)]:
        ax.add_patch(
            Ellipse((cx + dx * w, cy + dy * h), sw * w, sh * h, fc="#C3D3E0", ec="none", alpha=alpha, zorder=z0)
        )


# ------------------------------------------
# label
# ------------------------------------------
# Text at page position ``xy`` with an optional thin leader line to ``anchor``.
def label(ax, text, xy, anchor=None, ha="center", va="center", color=INK, **kw):
    if anchor is not None:
        ax.plot([xy[0], anchor[0]], [xy[1], anchor[1]], color="#555555", lw=0.6, zorder=40, solid_capstyle="round")
    ax.text(
        xy[0],
        xy[1],
        text,
        ha=ha,
        va=va,
        color=color,
        zorder=41,
        bbox={"fc": "white", "ec": "none", "pad": 1.2, "alpha": 0.92},
        **kw,
    )


fig = plt.figure(figsize=(7.75, 3.9), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_aspect("equal")
ax.axis("off")
ax.set_xlim(-0.45, 13.0)
ax.set_ylim(-0.5, 6.27)

GX, GY = 9.7, 7.2
poly(ax, [(0, 0), (GX, 0), (GX, GY), (0, GY)], fc="#F7F7F4", ec="#9A9A9A", lw=0.8, zorder=0)
for gx in np.arange(1.7, GX, 2.0):
    ax.plot(*zip(P(gx, 0), P(gx, GY)), color="#E3E3DE", lw=0.5, zorder=0.1)
for gy in np.arange(1.2, GY, 2.0):
    ax.plot(*zip(P(0, gy), P(GX, gy)), color="#E3E3DE", lw=0.5, zorder=0.1)

R0, V = np.array([1.2, 3.7]), np.array([6.75, 1.5])  # start pose, chosen vantage point
OCC = (3.75, 3.05, 1.25, 1.35)  # occluder footprint x, y, dx, dy
TGT = np.array([7.95, 3.95])  # target centre (plan view)

geo = sector(R0, 0.0, R_GEO)
poly(ax, geo, fc="#8FA3B8", ec="none", alpha=0.10, zorder=0.5)
poly(ax, geo, fc="none", ec="#555555", lw=1.0, ls=(0, (4, 3)), zorder=1.5)
# shadow of the occluder inside the geometric sector
corners = [
    (OCC[0], OCC[1]),
    (OCC[0], OCC[1] + OCC[3]),
    (OCC[0] + OCC[2] + 0.95, OCC[1] + 0.25),
    (OCC[0] + OCC[2], OCC[1] + OCC[3]),
]
ang = [np.arctan2(cy - R0[1], cx - R0[0]) for cx, cy in corners]
a0, a1 = min(ang), max(ang)
ts = np.linspace(a0, a1, 24)
shadow = (
    [(OCC[0], R0[1] + (OCC[0] - R0[0]) * np.tan(a0), 0)]
    + [(R0[0] + R_GEO * np.cos(t), R0[1] + R_GEO * np.sin(t), 0) for t in ts]
    + [(OCC[0], R0[1] + (OCC[0] - R0[0]) * np.tan(a1), 0)]
)
poly(ax, shadow, fc="#9A9A9A", ec="none", zorder=1.1, alpha=0.22)
poly(ax, sector(R0, 0.0, R_EFF), fc=BLUE, ec="none", alpha=0.2, zorder=1.3)
poly(ax, sector(R0, 0.0, R_EFF), fc="none", ec=BLUE, lw=1.0, zorder=1.4)

rough = [(3.15, 2.1), (4.1, 1.72), (5.5, 1.8), (6.0, 2.4), (5.2, 2.95), (3.7, 2.92)]
poly(ax, rough, fc="#E9DCC6", ec="#A08A65", lw=0.6, zorder=1.6)
rng = np.random.default_rng(7)
pts = rng.uniform((3.15, 1.7), (6.0, 2.95), size=(150, 2))
keep = MPath(rough).contains_points(pts)
ax.scatter(*np.array([P(*p) for p in pts[keep]]).T, s=1.1, color="#8C7650", zorder=1.7, linewidths=0)

heading_v = np.arctan2(TGT[1] - V[1], TGT[0] - V[0])
poly(ax, sector(V, heading_v, R_EFF), fc=BLUE, ec="none", alpha=0.2, zorder=1.3)
poly(ax, sector(V, heading_v, R_EFF), fc="none", ec=BLUE, lw=1.0, zorder=1.4)

b = [np.array(p) for p in [(1.8, 3.7), (2.7, 3.5), (3.2, 0.25), (6.2, 1.25)]]  # cubic Bezier, skirts the rough patch
t = np.linspace(0, 1, 80)[:, None]
curve = (1 - t) ** 3 * b[0] + 3 * (1 - t) ** 2 * t * b[1] + 3 * (1 - t) * t**2 * b[2] + t**3 * b[3]
cp = np.array([P(*p) for p in curve])
ax.plot(cp[:-4, 0], cp[:-4, 1], color=BLUE, lw=2.2, zorder=3, solid_capstyle="round")
ax.add_patch(
    FancyArrowPatch(
        cp[-8], cp[-1], arrowstyle="-|>", mutation_scale=13, color=BLUE, lw=2.2, zorder=3, shrinkA=0, shrinkB=0
    )
)
f0, f1 = P(1.85, 3.5), P(3.25, 2.98)
ax.add_patch(
    FancyArrowPatch(
        f0,
        f1,
        arrowstyle="-|>",
        mutation_scale=10,
        color="#666666",
        lw=1.4,
        ls=(0, (3, 2.2)),
        zorder=3,
        shrinkA=0,
        shrinkB=0,
    )
)

box(ax, TGT[0] - 0.6, TGT[1] - 0.55, 0, 1.2, 1.1, 0.75, "#EEF4EC", GRN, "#BFD3BD", z0=6)  # shelf
box(ax, TGT[0] - 0.33, TGT[1] - 0.3, 0.75, 0.66, 0.6, 0.58, "#5A5A5A", "#3A3A3A", "#262626", z0=7)  # dark target
box(ax, OCC[0] + OCC[2], OCC[1] + 0.25, 0, 0.95, 1.0, 0.85, YEL, ORA, "#E9CDA6", z0=8)  # occluder: side crate
box(ax, OCC[0], OCC[1], 0, OCC[2], OCC[3], 1.2, YEL, ORA, "#E9CDA6", z0=9)  # occluder: main crate
box(ax, OCC[0] + 0.2, OCC[1] + 0.25, 1.2, 0.85, 0.9, 0.7, YEL, ORA, "#E9CDA6", z0=10)  # occluder: top crate
fog(ax, *(P(7.2, 6.0) + (0.0, 0.3)), 5.0, 0.95, 4.5, alpha=0.55)  # fog bank behind the target

# belief ellipsoids around the target
tc = P(TGT[0], TGT[1], 1.04)
ax.add_patch(Ellipse(tc, 3.3, 1.75, angle=12, fc=RED, ec="none", alpha=0.07, zorder=12))
ax.add_patch(Ellipse(tc, 3.3, 1.75, angle=12, fc="none", ec=RED, lw=1.5, zorder=13))
ax.add_patch(Ellipse(tc, 1.02, 0.86, angle=0, fc="none", ec=RED, lw=1.5, ls=(0, (3, 2)), zorder=14))

# rovers and LiDAR rays
lid0 = rover(ax, R0, z0=15)
lidv = rover(ax, V, z0=16, ghost=True)
for dz in (0.9, 1.2):  # sightlines onto the elevated target
    ax.plot(*zip(P(*lidv), P(TGT[0] - 0.12, TGT[1] - 0.3, dz)), color=BLUE, lw=1.0, ls=(0, (4, 2)), zorder=17)
stop = P(OCC[0], R0[1] + 0.06, 0.62)  # start sightline ends on the crate
ax.plot(*zip(P(*lid0), stop), color="#555555", lw=1.0, ls=(0, (4, 2)), zorder=17)
ax.plot(*stop, marker="x", ms=6, mew=1.6, color="#333333", zorder=18)

# sun
sun = np.array([0.55, 5.45])
ax.add_patch(Circle(sun, 0.27, fc=YEL, ec=INK, lw=0.8, zorder=30))
for a in np.linspace(0, 2 * np.pi, 9)[:-1]:
    d = np.array([np.cos(a), np.sin(a)])
    ax.plot(*zip(sun + 0.36 * d, sun + 0.52 * d), color=INK, lw=0.8, zorder=30, solid_capstyle="round")

label(ax, "sunlight", (0.55, 4.86))
label(
    ax,
    "Start: obstructed +\nbeyond effective range",
    (1.55, 0.72),
    anchor=P(R0[0] - 0.25, R0[1] - 0.32, 0.2),
    fontweight="bold",
    linespacing=1.05,
)
label(
    ax,
    "Next view: visible +\nwithin effective range",
    (11.15, 0.42),
    anchor=P(V[0] + 0.5, V[1] - 0.1, 0.28),
    color=BLUE,
    fontweight="bold",
    linespacing=1.05,
)
label(ax, r"effective FOV ($P_D \geq 0.5$)", (2.35, 3.72), anchor=P(R0[0] + 1.75, R0[1] + 0.85), color=BLUE)
label(
    ax,
    "geometric FOV",
    (8.35, -0.2),
    anchor=P(R0[0] + 5.6 * np.cos(HALF), R0[1] - 5.6 * np.sin(HALF)) + (0.0, -0.02),
    color="#444444",
)
label(ax, "obstacle", (5.75, 5.35), anchor=P(OCC[0] + 0.65, OCC[1] + 0.7, 1.9))
ax.text(
    *((f0 + f1) / 2 + (0.2, 0.27)),
    "fixed route",
    color="#333333",
    fontsize=10,
    ha="center",
    va="center",
    zorder=41,
    rotation=-10,
    bbox={"fc": "white", "ec": "none", "pad": 0.8, "alpha": 0.85},
)
ax.text(
    *(stop + (0.2, -0.02)), "blocked", color="#333333", fontsize=10, style="italic", ha="left", va="center", zorder=41
)
label(ax, "rough terrain", P(4.6, 2.36), fontsize=10)
label(ax, "information-driven route", (4.75, -0.2), anchor=cp[40] + (0.0, -0.05), color=BLUE)
label(ax, "dark target", (8.55, 5.55), anchor=P(TGT[0] - 0.1, TGT[1], 1.33))
label(ax, "current uncertainty", (11.6, 4.72), anchor=tc + (1.2, 0.72), color=RED)
label(ax, "expected\nafter scan", (12.25, 2.7), anchor=tc + (0.5, -0.1), color=RED, linespacing=1.05)
ax.text(*(P(7.2, 6.0) + (-2.0, 0.52)), "fog", color="#2F4558", ha="center", va="center", zorder=41)
ax.text(12.9, 6.08, "concept sketch", color="#777777", style="italic", fontsize=9.5, ha="right", va="center", zorder=41)

fig.savefig(Path(__file__).resolve().parents[1] / "docs/figures/readme/concept.png", facecolor="white")
