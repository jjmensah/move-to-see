# ==========================================================
# sim/experiments/F05_main.py
# ==========================================================
# F05 — Main comparison: information-driven planner against the fixed route and the physics-blind planner
# (vv-plan §3). Paired trials: the three controllers share the target draw and the per-scan noise streams.

from __future__ import annotations

from pathlib import Path

import numpy as np

from lidar_sim.viz import plots
from mts.config import ACC_TOL, DWELL
from mts.metrics import bootstrap_ci, dwell_time, holm, paired_wilcoxon, position_error, time_score, wilson
from mts.viz import COND_LABEL, CTRL_LABEL, ctrl_style, plot_paths
from mts.world import yard
from sim.experiments._common import load_trial, run_jobs, trial_job

CTRLS = ("info", "blind", "fixed")


# ------------------------------------------
# _load
# ------------------------------------------
# Trial records of one condition and controller.
def _load(todo, cond, ctrl):
    return [load_trial(j) for j in todo if j["condition"] == cond and j["controller"] == ctrl]


# ------------------------------------------
# run
# ------------------------------------------
# Runs F05: 40 paired trials per condition and the registered tests H1 to H3.
def run(entry: dict, out: Path, quick: bool, jobs: int) -> dict:
    p = entry["params"]
    n = 6 if quick else entry["n"]
    steps = p["steps"]
    todo = [trial_job(out, "main", c, ctrl, entry["seed"], t, steps)
            for c in p["conditions"] for ctrl in CTRLS for t in range(n)]
    run_jobs(todo, jobs, "F05")
    rng = np.random.default_rng([entry["seed"], 777])
    T_end = None
    data, summary = {}, {}
    for c in p["conditions"]:
        data[c], summary[c] = {}, {}
        for ctrl in CTRLS:
            rows = _load(todo, c, ctrl)
            dt = float(rows[0]["dt"])
            T_end = steps * dt
            err = np.array([position_error(r["q_hat"], r["q_true"]) for r in rows])
            ts = np.array([time_score(e, dt, ACC_TOL, DWELL) for e in err])
            ok = np.array([np.isfinite(dwell_time(e, dt, ACC_TOL, DWELL)) for e in err])
            path = np.array([np.hypot(*np.diff(r["states"][:, :2], axis=0).T).sum() for r in rows])
            data[c][ctrl] = {"err": err, "ts": ts, "ok": ok, "rows": rows}
            summary[c][ctrl] = {
                "n": len(rows), "success_rate": float(ok.mean()), "success_wilson95": list(wilson(int(ok.sum()), len(rows), 0.95)),
                "median_time_score_s": float(np.median(ts)), "time_score_ci95": list(bootstrap_ci(ts, np.median, rng=rng)),
                "mean_time_score_s": float(ts.mean()),
                "median_final_error_m": float(np.median(err[:, -1])),
                "final_error_ci95": list(bootstrap_ci(err[:, -1], np.median, rng=rng)),
                "max_final_error_m": float(err[:, -1].max()), "mean_path_length_m": float(path.mean()),
                "no_target_return_fraction": float(np.mean([r["n_target_detected"].sum() == 0 for r in rows])),
                "fallback_steps": int(sum(r["fallback"].sum() for r in rows)),
                "infeasible_steps": int(sum(r["infeasible"].sum() for r in rows)),
                "mean_plan_time_s": float(np.mean([r["wall_time"][:, 0].mean() for r in rows])),
                "mean_update_time_s": float(np.mean([r["wall_time"][:, 1].mean() for r in rows]))}

    tests = [("H1", c, "fixed") for c in ("C1", "C2", "C3")] + [("H2", c, "blind") for c in ("C2", "C3")]
    raw = [paired_wilcoxon(data[c]["info"]["ts"], data[c][other]["ts"], "less") for _, c, other in tests]
    adj = holm(raw)
    comparisons = []
    for (h, c, other), p_raw, p_adj in zip(tests, raw, adj, strict=True):
        d = data[c]["info"]["ts"] - data[c][other]["ts"]
        a, b = data[c]["info"]["ok"], data[c][other]["ok"]
        comparisons.append({"hypothesis": h, "condition": c, "against": other, "p_raw": float(p_raw), "p_holm": float(p_adj),
                            "median_paired_difference_s": float(np.median(d)),
                            "difference_ci95": list(bootstrap_ci(d, np.median, rng=rng)),
                            "info_faster": int((d < 0).sum()), "ties": int((d == 0).sum()), "info_slower": int((d > 0).sum()),
                            "mcnemar_info_only_success": int((a & ~b).sum()), "mcnemar_other_only_success": int((~a & b).sum())})
    d4 = data["C4"]["info"]["ts"] - data["C4"]["blind"]["ts"]
    ci4 = bootstrap_ci(d4, np.median, rng=rng)
    control = {"median_paired_difference_s": float(np.median(d4)), "difference_ci95": list(ci4), "margin_s": 2.0}
    _figures(data, summary, out, T_end, p["conditions"])
    alpha = 0.01
    return {"n_trials": n, "steps": steps, "trial_length_s": T_end, "summary": summary, "comparisons": comparisons,
            "control_C4": control,
            "verdicts": {"H1_info_faster_than_fixed_C1_C2_C3": bool(all(c["p_holm"] < alpha for c in comparisons if c["hypothesis"] == "H1")),
                         "H2_info_faster_than_blind_C2_C3": bool(all(c["p_holm"] < alpha for c in comparisons if c["hypothesis"] == "H2")),
                         "H3_control_equivalent_within_2s_C4": bool(-2.0 <= ci4[0] and ci4[1] <= 2.0)}}


# ------------------------------------------
# _figures
# ------------------------------------------
# Plots time and success, example paths and error over time.
def _figures(data, summary, out: Path, T_end: float, conds) -> None:
    # (1) time score and success per condition
    fig, axs = plots.figure(1, 2, height_in=2.7)
    xs = np.arange(len(conds))
    for j, ctrl in enumerate(CTRLS):
        st = ctrl_style(ctrl)
        med = [summary[c][ctrl]["median_time_score_s"] for c in conds]
        ci = np.array([summary[c][ctrl]["time_score_ci95"] for c in conds])
        axs[0].errorbar(xs + (j - 1) * 0.2, med, yerr=np.maximum([med - ci[:, 0], ci[:, 1] - med], 0), fmt=st["marker"],
                        color=st["color"], ms=5.5, elinewidth=1.0, capsize=2.5, label=CTRL_LABEL[ctrl])
        axs[1].bar(xs + (j - 1) * 0.26, [summary[c][ctrl]["success_rate"] for c in conds], 0.24, color=st["color"],
                   label=CTRL_LABEL[ctrl])
    axs[0].axhline(T_end, color=plots.INK_2, ls=":", lw=0.9)
    axs[0].text(len(conds) - 0.55, T_end - 0.6, "trial length (not localized)", ha="right", va="top", fontsize=7, color=plots.INK_2)
    for ax in axs:
        ax.set_xticks(xs, [COND_LABEL[c].replace(" (", "\n(").replace(", clear", ",\nclear") for c in conds], fontsize=7.5)
    axs[0].set(ylabel="time to localize within 0.10 m [s]", ylim=(0, T_end + 2), title="(a) median time score, 95 % CI")
    axs[1].set(ylabel="trials localized", ylim=(0, 1.05), title="(b) success rate")
    plots.legend_below(fig, axs[0], ncols=3)
    plots.save(fig, out / "time_and_success.pdf")

    # (2) error versus time
    fig, axs = plots.figure(1, len(conds), height_in=2.5, sharey=True)
    for ax, c in zip(axs, conds, strict=True):
        for ctrl in CTRLS:
            st = ctrl_style(ctrl)
            e = np.maximum(data[c][ctrl]["err"], 1e-3)
            t = np.arange(e.shape[1]) * (T_end / (e.shape[1] - 1))
            ax.fill_between(t, np.quantile(e, 0.25, axis=0), np.quantile(e, 0.75, axis=0), color=st["color"], alpha=0.18, lw=0)
            ax.semilogy(t, np.median(e, axis=0), color=st["color"], ls=st["linestyle"], label=CTRL_LABEL[ctrl])
        ax.axhline(ACC_TOL, color=plots.INK_2, ls="--", lw=0.8)
        ax.set(title=COND_LABEL[c], xlabel="time [s]", ylim=(1e-3, 3))
    axs[0].set_ylabel("position error [m]\n(median, IQR)")
    plots.legend_below(fig, axs[1], ncols=3)
    plots.save(fig, out / "error_vs_time.pdf")

    # (3) example paths: first trial of each condition
    fig, axs = plots.figure(1, 2, height_in=2.9)
    for ax, c in zip(axs, ("C2", "C3"), strict=True):
        plot_paths(ax, yard(), {ctrl: data[c][ctrl]["rows"][0] for ctrl in CTRLS}, title=f"{COND_LABEL[c]}: trial 0")
    plots.legend_below(fig, axs[0], ncols=3)
    plots.save(fig, out / "paths.pdf")
