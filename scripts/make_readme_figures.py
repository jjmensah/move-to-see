# ==========================================================
# scripts/make_readme_figures.py
# ==========================================================
# Rebuild README plots from the checked-in evidence; run with .venv/bin/python.
#
# Only world, sensor, results, decisions and replay JSON are read. No simulation,
# network access, random jitter, or synthetic trial data. The concept is generated
# separately by concept_sketch.py, preserved from the proposal artwork.

from pathlib import Path
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "web/public/data"
OUT = ROOT / "docs/figures/readme"
BLUE, GREY, RED, ORANGE = "#1F5FBF", "#7A7A7A", "#D40000", "#E08A00"
CONTROLLERS = ("info", "fixed", "blind")
NAMES = dict(info="information-driven", fixed="fixed route", blind="physics-blind")
COLORS = dict(info=BLUE, fixed="#222222", blind=ORANGE)
STYLES = dict(info="-", fixed="--", blind="-.")
CONDITIONS = ["clear", "fog (20 m)", "sunlight (1 sun)", "white target"]
plt.rcParams.update(
    {
        "font.family": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 16,
        "axes.titlesize": 21,
        "axes.labelsize": 17,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 15,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "mathtext.fontset": "dejavusans",
        "hatch.linewidth": 0.5,
    }
)


# ------------------------------------------
# read
# ------------------------------------------
# Loads one of the site's data files.
def read(name):
    return json.loads((DATA / (name + ".json")).read_text())


# ------------------------------------------
# log
# ------------------------------------------
# Prints a number that appears in a figure, so it can be checked.
def log(label, value):
    print(f"{label}: {value}")


# ------------------------------------------
# save
# ------------------------------------------
# Saves a figure as PNG after checking every visible number on it.
def save(fig, name):
    fig.savefig(OUT / (name + ".png"), dpi=200, bbox_inches="tight", pad_inches=0.18)
    # Audit all visible numerical text, including automatically generated ticks.
    fig.canvas.draw()
    from matplotlib.text import Text

    labels = sorted(
        {t.get_text() for t in fig.findobj(Text) if t.get_visible() and any(c.isdigit() for c in t.get_text())}
    )
    log(name + " numerical labels", labels)
    plt.close(fig)


# ------------------------------------------
# base
# ------------------------------------------
# A figure with the shared title, optional subtitle and axis style.
def base(title, size=(10, 6), subtitle=None):
    fig, ax = plt.subplots(figsize=size)
    ax.set_title(title, loc="left", pad=42 if subtitle else 22)
    if subtitle:
        ax.text(0, 1.025, subtitle, transform=ax.transAxes, fontsize=15, color=GREY, va="bottom")
    return fig, ax


# ------------------------------------------
# handles
# ------------------------------------------
# Legend entries for the three controllers.
def handles():
    return [Line2D([], [], color=COLORS[c], ls=STYLES[c], lw=2.5, label=NAMES[c]) for c in CONTROLLERS]


# ------------------------------------------
# yard
# ------------------------------------------
# Draws the yard: arena, crates, rough ground, prior region and, optionally, the fixed route.
def yard(ax, world, route=False):
    x0, x1, y0, y1 = world["arena"]
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="none", ec=GREY, lw=0.7))
    for b in world["boxes"][4:]:
        lo, hi = np.array(b["lo"]), np.array(b["hi"])
        ax.add_patch(Rectangle(lo[:2], *(hi - lo)[:2], fc="#FFE6CC", ec="black", lw=0.8, zorder=2))
    for a, b, c, d in world["rough"]:
        ax.add_patch(Rectangle((a, c), b - a, d - c, fc="#FFF2CC", ec="#B7A784", lw=0.7, hatch="///"))
    lo, hi = np.array(world["prior"]["lo"]), np.array(world["prior"]["hi"])
    ax.add_patch(Rectangle(lo[:2], *(hi - lo)[:2], fc="none", ec=RED, lw=1.2, ls=":"))
    if route:
        r = np.array(world["fixedRoute"])
        ax.plot(*r.T, color="#222222", ls="--", alpha=0.22, lw=1.3)
    ax.set(xlim=(x0 - 0.2, x1 + 0.2), ylim=(y0 - 0.2, y1 + 0.2), xlabel="x [m]", ylabel="y [m]", aspect="equal")
    ax.set_xticks(np.arange(0, 12, 2))
    ax.set_yticks(np.arange(0, 9, 2))


# ------------------------------------------
# loop
# ------------------------------------------
# The perception loop: robot, LiDAR, belief and planner.
def loop():
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.set(xlim=(-0.08, 10.04), ylim=(0, 3.2))
    ax.axis("off")
    ax.text(0, 2.95, "Move, sense, update, and plan again", fontsize=20, weight="bold")
    titles = ["Robot +\nenvironment", "3D AMCW LiDAR", "Target belief", "Viewpoint\nplanner"]
    bodies = [
        "unicycle; rough\nground limits speed",
        "$\\sigma_R$, $P_D$, 3D FOV\n+ line of sight",
        "particles\n$p(q \\mid z_{1:k})$",
        "information bound\n− λ · travel time",
    ]
    xs = [0, 2.58, 5.16, 7.74]
    for x, title, body, col in zip(xs, titles, bodies, ["#D5E8D4", "#DAE8FC", "#FFF2CC", "#FFE6CC"]):
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.8), 2.22, 1.45, boxstyle="round,pad=.015,rounding_size=.06", fc=col, ec="black", lw=0.8
            )
        )
        ax.text(x + 1.11, 1.91, title, ha="center", va="center", fontsize=13.5, weight="bold", linespacing=1.05)
        ax.text(x + 1.11, 1.22, body, ha="center", va="center", fontsize=13, linespacing=1.25)
    for i, label in enumerate(["state", "scan", "belief"]):
        a, b = xs[i] + 2.24, xs[i + 1] - 0.02
        ax.annotate("", (b, 1.5), (a, 1.5), arrowprops=dict(arrowstyle="-|>", lw=0.9, color="black"))
        ax.text((a + b) / 2, 2.35, label, fontsize=15, ha="center", style="italic")
    ax.plot([8.85, 8.85, 1.11], [0.78, 0.35, 0.35], color="black", lw=0.9)
    ax.annotate("", (1.11, 0.8), (1.11, 0.35), arrowprops=dict(arrowstyle="-|>", lw=0.9, color="black"))
    ax.text(
        5,
        0.35,
        "execute first control, then replan",
        ha="center",
        va="center",
        fontsize=16,
        style="italic",
        bbox=dict(fc="white", ec="none", pad=4),
    )
    save(fig, "loop")


# ------------------------------------------
# detection
# ------------------------------------------
# Detection probability against range for the black target in each condition.
def detection(sensor):
    fig, ax = base("Fog and sunlight shorten detection range")
    r = np.array(sensor["rangeM"])

    def curve(rho, vis, sun):
        i, j, k = sensor["rho"].index(rho), sensor["visibilityM"].index(vis), sensor["sun"].index(sun)
        return sensor["curves"][(i * len(sensor["visibilityM"]) + j) * len(sensor["sun"]) + k]

    ref = curve(0.9, 20000, 0)
    ax.plot(r, ref["pd"], color=GREY, lw=1.3, label="white target (clear)")
    cases = [
        ("clear", "#222222", "-", 20000, 0, (3.6, 0.66)),
        ("fog (20 m)", BLUE, "--", 20, 0, (2.14, 0.36)),
        ("sunlight (1 sun)", ORANGE, "-.", 20000, 1, (0.18, 0.22)),
    ]
    for name, col, ls, vis, sun, xy in cases:
        c = curve(0.05, vis, sun)
        ax.plot(r, c["pd"], color=col, ls=ls, lw=2.6, label=name)
        ax.plot(c["r50M"], 0.5, "o", color=col, ms=7)
        label = f"{c['r50M']:.2f} m"
        ax.annotate(
            label,
            (c["r50M"], 0.5),
            xytext=xy,
            color=col,
            fontsize=17,
            arrowprops=dict(arrowstyle="-", color=col, lw=0.8),
        )
        log(name + " r50 [m]", c["r50M"])
    ax.axhline(0.5, color=GREY, ls=":", lw=0.9, zorder=0)
    ax.set(xlim=(0, 5.6), ylim=(-0.02, 1.04), xlabel="range to target [m]", ylabel="detection probability $P_D$")
    ax.set_yticks([0, 0.5, 1])
    ax.legend(frameon=False, loc="upper right")
    fig.text(
        0.125,
        -0.075,
        "Black target (ρ = 0.05), facing the sensor and unobstructed · dots mark 50% detection",
        fontsize=15,
    )
    save(fig, "detection_range")


# ------------------------------------------
# paths
# ------------------------------------------
# The three controllers' paths in one fog trial.
def paths(world, trial):
    fig, ax = base(
        "Paths through the yard in fog (20 m)", (10, 8.8), "Trial 3 · full 30 s paths · × marks each endpoint"
    )
    yard(ax, world, route=True)
    for c in CONTROLLERS:
        d = trial["controllers"][c]
        xy = np.array(d["states"])[:, :2]
        outcome = f"localized in {d['timeScoreS']:g} s" if d["dwellStart"] is not None else "not localized in 30 s"
        ax.plot(*xy.T, color=COLORS[c], ls=STYLES[c], lw=2.4, label=NAMES[c] + " — " + outcome, zorder=4)
        ax.scatter(*xy[-1], marker="X", s=95, color=COLORS[c], ec="white", lw=0.6, zorder=6)
        log("fog trial 3 " + c, outcome)
    ax.scatter(*world["start"][:2], s=90, fc="white", ec="black", zorder=7)
    ax.annotate("start", world["start"][:2], xytext=(0.3, 2.9), arrowprops=dict(arrowstyle="-", lw=0.7))
    ax.scatter(*trial["qTrue"][:2], s=90, marker="s", color="black", zorder=7)
    ax.annotate(
        "true target", trial["qTrue"][:2], xytext=(5.75, 5.3), fontsize=15, arrowprops=dict(arrowstyle="-", lw=0.7)
    )
    ax.text(7, 2.45, "prior region", color=RED, fontsize=15)
    ax.text(4.9, 1.8, "rough\nground", fontsize=15, ha="center", va="center", bbox=dict(fc="#FFF2CC", ec="none", pad=1))
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.15), handlelength=3)
    save(fig, "paths_fog")


# ------------------------------------------
# timing
# ------------------------------------------
# Median time to localize with 95 % intervals and success counts, per condition.
def timing(results):
    fig, ax = base("Time to localize across conditions", (10, 6.5))
    for j, c in enumerate(CONTROLLERS):
        for i, condition in enumerate(["C1", "C2", "C3", "C4"]):
            s = results["summary"][condition][c]
            m = s["medianTimeScoreS"]
            lo, hi = s["timeScoreCi95"]
            x = i + (j - 1) * 0.25
            ax.plot([x, x], [lo, hi], color=COLORS[c], ls=STYLES[c], lw=2.5)
            ax.plot([x - 0.035, x + 0.035], [lo, lo], color=COLORS[c], lw=1.2)
            ax.plot([x - 0.035, x + 0.035], [hi, hi], color=COLORS[c], lw=1.2)
            ax.plot(x, m, "o", ms=8, color=COLORS[c], zorder=3)
            ax.annotate(
                f"{s['successCount']}/40",
                (x, hi),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                color=COLORS[c],
                fontsize=15,
            )
            log(condition + " " + c, dict(median=m, ci95=[lo, hi], successes=s["successCount"], n=40))
    ax.axhline(30, color=GREY, lw=0.8, ls=":", zorder=0)
    ax.set(xlim=(-0.5, 3.5), ylim=(0, 34), ylabel="time to localize [s]")
    ax.set_yticks([0, 10, 20, 30])
    ax.set_xticks(range(4), CONDITIONS)
    ax.text(1, 1.015, "30 s = not localized", transform=ax.transAxes, ha="right", fontsize=15, color=GREY)
    ax.legend(handles=handles(), frameon=False, loc="upper left", bbox_to_anchor=(0, -0.16), ncol=3, columnspacing=1.2)
    fig.text(0.125, -0.15, "Dots: medians · intervals: 95% CI · labels: successful trials / 40", fontsize=15)
    save(fig, "time_to_localize")


# ------------------------------------------
# errors
# ------------------------------------------
# Median localization error over time in fog, with interquartile bands.
def errors(world, trials):
    fig, ax = base("Localization error over time in fog (20 m)", (10, 6.5))
    for c in CONTROLLERS:
        e = np.array([d["controllers"][c]["error"] for d in trials])
        q1, med, q3 = np.quantile(e, [0.25, 0.5, 0.75], axis=0)
        t = np.arange(e.shape[1]) * world["dt"]
        ax.fill_between(t, q1, q3, color=COLORS[c], alpha=0.13, lw=0)
        ax.plot(t, med, color=COLORS[c], ls=STYLES[c], lw=2.5, label=NAMES[c])
    ax.axhline(world["toleranceM"], color=RED, ls=":", lw=1.2)
    ax.text(29, 0.12, "0.10 m tolerance", ha="right", color=RED, fontsize=15)
    ax.set(yscale="log", ylim=(0.001, 2), xlim=(0, 30), xlabel="time [s]", ylabel="position error [m]")
    ax.set_yticks([0.001, 0.01, 0.1, 1, 2], ["0.001", "0.01", "0.1", "1", "2"])
    ax.minorticks_off()
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.17), ncol=3, columnspacing=1.2)
    fig.text(0.125, -0.15, "Median of 40 trials · shaded bands show the middle 50%", fontsize=15)
    save(fig, "error_vs_time")


# ------------------------------------------
# paired
# ------------------------------------------
# Paired fog times of the information-driven planner against the fixed route.
def paired(results, trials):
    fig, ax = base("Paired localization times in fog (20 m)", (10, 7.6))
    xy = np.array([[d["controllers"][c]["timeScoreS"] for c in ("fixed", "info")] for d in trials])
    count = int(np.sum(xy[:, 1] < xy[:, 0]))
    comparison = next(c for c in results["comparisons"] if c["condition"] == "C2" and c["against"] == "fixed")
    assert count == comparison["infoFaster"] == 30
    ax.plot([0, 30], [0, 30], color=GREY, ls=":", lw=1.2)
    # Concentric rings preserve exact paired coordinates when trials coincide.
    unique, counts = np.unique(xy, axis=0, return_counts=True)
    for p, n in zip(unique, counts):
        failed = any(
            d["controllers"][c]["dwellStart"] is None
            for d in trials
            if all(d["controllers"][c]["timeScoreS"] == v for c, v in zip(("fixed", "info"), p))
            for c in ("fixed", "info")
        )
        for j in range(n - 1, -1, -1):
            ax.scatter(*p, s=(9 + j * 4) ** 2, fc="white" if failed or j else BLUE, ec=BLUE, lw=1.2, zorder=3)
    ax.text(0.04, 0.94, f"{count} of 40 faster", transform=ax.transAxes, color=BLUE, fontsize=21, weight="bold")
    ax.text(0.04, 0.86, "Below the diagonal: information-driven is faster", transform=ax.transAxes, fontsize=15)
    ax.set(
        xlim=(0, 31.5), ylim=(0, 31.5), xlabel="fixed route time score [s]", ylabel="information-driven time score [s]"
    )
    ax.set_xticks([0, 10, 20, 30])
    ax.set_yticks([0, 10, 20, 30])
    fig.text(0.125, -0.015, "Hollow: not localized at 30 s · concentric rings: coincident trials", fontsize=15)
    log(
        "fog paired comparison",
        dict(faster=count, total=len(trials), slower=comparison["infoSlower"], ties=comparison["ties"]),
    )
    save(fig, "paired_fog")


# ------------------------------------------
# planner
# ------------------------------------------
# One planning step: candidate viewpoints coloured by score, particles and the chosen goal.
def planner(world, decisions):
    step = next(s for s in decisions["steps"] if s["k"] == 10)
    fig, ax = base(
        "Choosing the next viewpoint in fog (20 m)", (10, 8.4), "Step k = 10 · t = 5 s · candidates coloured by score"
    )
    yard(ax, world)
    particles, support = np.array(step["particlesXY"]), np.array(step["support"])
    ax.scatter(*particles.T, s=9, color=GREY, alpha=0.5, zorder=3, label="particle sample")
    ax.scatter(*support.T, s=55, facecolors="none", edgecolors=RED, lw=1, zorder=4, label="24 support points")
    candidates = step["candidates"]
    xy = np.array([c["xy"] for c in candidates])
    scores = [c["scoreNat"] for c in candidates]
    cmap = LinearSegmentedColormap.from_list("proposal_blues", ["#DAE8FC", BLUE, "#103365"])
    sc = ax.scatter(*xy.T, c=scores, s=52, cmap=cmap, ec="white", lw=0.35, zorder=3)
    cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.04, shrink=0.8)
    cb.set_label("score J [nat]")
    goal = next(c for c in candidates if c["id"] == step["selectedId"])
    ax.scatter(*goal["xy"], s=340, fc="none", ec=BLUE, lw=2.5, zorder=6)
    ax.annotate(
        "chosen goal",
        goal["xy"],
        xytext=(5.9, 7.45),
        fontsize=15,
        color=BLUE,
        bbox=dict(fc="white", ec="none", pad=2),
        arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.9),
    )
    ax.scatter(*step["state"][:2], s=110, marker="^", color="black", zorder=7)
    ax.annotate("robot", step["state"][:2], xytext=(1.4, 7.1), fontsize=15, arrowprops=dict(arrowstyle="-", lw=0.9))
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.16), ncol=2)
    log(
        "planner",
        dict(
            k=step["k"],
            time_s=step["k"] * world["dt"],
            support_count=len(support),
            selected_goal=goal["xy"],
            selected_score_nat=goal["scoreNat"],
        ),
    )
    assert len(support) == 24
    save(fig, "planner_step")


# ------------------------------------------
# main
# ------------------------------------------
# Builds every README figure from the site data.
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    world, results, sensor, decisions = [read(n) for n in ["world", "results", "sensor", "decisions"]]
    trials = [read("replay/C2/" + p.stem) for p in sorted((DATA / "replay/C2").glob("*.json"))]
    assert len(trials) == results["n"] == 40
    for trial, paired_row in zip(trials, results["paired"]["C2"]):
        assert trial["trial"] == paired_row["trial"]
        for index, controller in enumerate(("info", "blind", "fixed")):
            assert trial["controllers"][controller]["timeScoreS"] == paired_row["timeScoreS"][index]
    log("preserved concept sketch detection threshold", 0.5)
    loop()
    detection(sensor)
    paths(world, next(d for d in trials if d["trial"] == 3))
    timing(results)
    errors(world, trials)
    paired(results, trials)
    planner(world, decisions)


if __name__ == "__main__":
    main()
