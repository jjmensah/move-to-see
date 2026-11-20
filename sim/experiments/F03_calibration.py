# ==========================================================
# sim/experiments/F03_calibration.py
# ==========================================================
# F03 — Calibration of the regularized particle filter (vv-plan §3).
#
# Fixed-route trials (no planner in the loop). After scans 20, 40 and 60: position error, NEES against the
# χ²₃ 90 % quantile, and per-coordinate standardised errors. The filter is an approximation with a kernel floor
# (final-spec §6.2), so the registered criterion is one-sided: it must not be over-confident.

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import chi2

from lidar_sim.viz import plots
from mts.metrics import nees, position_error, wilson
from mts.viz import COND_LABEL
from sim.experiments._common import load_trial, run_jobs, trial_job

CHECK = (20, 40, 60)


# ------------------------------------------
# run
# ------------------------------------------
# Runs F03: NEES coverage of the particle filter along the fixed route.
def run(entry: dict, out: Path, quick: bool, jobs: int) -> dict:
    p = entry["params"]
    n = 8 if quick else entry["n"]
    steps = p["steps"]
    q90 = float(chi2.ppf(0.90, 3))
    todo = [trial_job(out, "cal", c, "fixed", entry["seed"], t, steps) for c in p["conditions"] for t in range(n)]
    run_jobs(todo, jobs, "F03")
    res = {}
    for c in p["conditions"]:
        rows = [load_trial(j) for j in todo if j["condition"] == c]
        cond = {}
        for k in CHECK:
            ne = np.array([nees(r["q_hat"][k], r["cov"][k], r["q_true"]) for r in rows])
            err = np.array([position_error(r["q_hat"][k], r["q_true"]) for r in rows])
            z = np.array([(r["q_hat"][k] - r["q_true"]) / np.sqrt(np.diag(r["cov"][k]) + 1e-6) for r in rows])
            cover = int((ne <= q90).sum())
            cond[str(k)] = {"coverage_90": cover / len(rows), "coverage_wilson95": list(wilson(cover, len(rows), 0.95)),
                            "median_nees": float(np.median(ne)), "nees_over_chi2_median": float(np.median(ne) / chi2.ppf(0.5, 3)),
                            "median_error_m": float(np.median(err)), "p90_error_m": float(np.quantile(err, 0.9)),
                            "z_std_xyz": z.std(axis=0, ddof=1).tolist(),
                            "median_posterior_std_m": np.median([np.sqrt(np.diag(r["cov"][k])) for r in rows], axis=0).tolist()}
        res[c] = cond
    _figure(res, out)
    ok = all(res["C1"][str(k)]["coverage_90"] >= 0.80 for k in CHECK)
    return {"n_trials": n, "chi2_3_q90": q90, "conditions": res,
            "verdicts": {"not_overconfident_C1": bool(ok)}}


# ------------------------------------------
# _figure
# ------------------------------------------
# Plots coverage and median NEES at the checked scans.
def _figure(res: dict, out: Path) -> None:
    fig, axs = plots.figure(1, 2, height_in=2.5)
    xs = np.arange(len(CHECK))
    for j, (c, cond) in enumerate(res.items()):
        st = plots.series(j)
        cov = [cond[str(k)]["coverage_90"] for k in CHECK]
        lo = [cond[str(k)]["coverage_90"] - cond[str(k)]["coverage_wilson95"][0] for k in CHECK]
        hi = [cond[str(k)]["coverage_wilson95"][1] - cond[str(k)]["coverage_90"] for k in CHECK]
        axs[0].errorbar(xs + (j - 1) * 0.12, cov, yerr=np.maximum([lo, hi], 0), fmt=st["marker"], color=st["color"],
                        ms=4.5, elinewidth=0.9, label=COND_LABEL[c])
        axs[1].semilogy(CHECK, [cond[str(k)]["median_error_m"] for k in CHECK], st["marker"] + st["linestyle"],
                        color=st["color"], label=COND_LABEL[c])
    axs[0].axhline(0.9, color=plots.INK_2, ls="--", lw=0.9)
    axs[0].set_xticks(xs, [str(k) for k in CHECK])
    axs[0].set(xlabel="scans", ylabel="truth inside 90 % region", ylim=(0.5, 1.03), title="(a) coverage (nominal 0.90)")
    axs[1].set(xlabel="scans", ylabel="median position error [m]", title="(b) error along the fixed route")
    axs[1].set_xticks(list(CHECK))
    axs[1].legend(fontsize=7)
    plots.save(fig, out / "calibration.pdf")
