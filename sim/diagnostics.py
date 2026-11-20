# ==========================================================
# sim/diagnostics.py
# ==========================================================
# Trial-level diagnostics of the main comparison (F05), quoted in the discussion of the results.
#
#     python -m sim.diagnostics          (after ``python -m sim.run F05``)
#
# Reads the F05 trial records in ``sim/out/F05/trials/main/`` and writes ``sim/out/F05/diagnostics.json``:
#
# * ``route_distance_m``: in fog and sunlight (pooled), the median distance in the horizontal plane from the target
#   centre to the fixed route (the closed waypoint loop), for the pairs in which the fixed route had the shorter time
#   score and for all other pairs;
# * ``stop_distance_to_prior_m``: per condition and controller, the median distance from the robot's final position
#   to the prior region of the target centre (0 inside it);
# * ``last10_objective_nat``: per condition and controller, the median of the planner's own objective
#   J = (information lower bound) - lambda_t * (travel time) over the last ten decision steps, pooled over trials;
# * ``no_target_return_trials``: per condition and controller, the number of trials in which no reported beam had
#   the target as its strongest return.
#
# This is an analysis of stored records only; it draws no random numbers and does not rerun the simulator.

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mts.metrics import time_score
from mts.route import DEFAULT_WAYPOINTS
from mts.world import yard

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "sim" / "out" / "F05"
CONDITIONS = ("C1", "C2", "C3", "C4")
CONTROLLERS = ("info", "blind", "fixed")


# ------------------------------------------
# load
# ------------------------------------------
# {condition: {controller: {trial: record}}} with the fields used below.
def load() -> dict:
    recs: dict = {}
    for cond in CONDITIONS:
        for ctrl in CONTROLLERS:
            for f in sorted((BASE / "trials" / "main" / cond / ctrl).glob("*.npz")):
                with np.load(f) as z:
                    err = np.linalg.norm(z["q_hat"] - z["q_true"], axis=1)
                    recs.setdefault(cond, {}).setdefault(ctrl, {})[int(z["trial"])] = {
                        "q_true": z["q_true"].copy(), "final_xy": z["states"][-1, :2].copy(),
                        "score": z["score"].copy(), "n_target_detected": z["n_target_detected"].copy(),
                        "time_score": time_score(err, float(z["dt"]))}
    if not recs:
        raise SystemExit("no F05 trial records found; run `python -m sim.run F05` first")
    return recs


# ------------------------------------------
# distance_to_loop
# ------------------------------------------
# Distance from point ``p`` to the closed polyline through ``waypoints``.
def distance_to_loop(p, waypoints) -> float:
    pts = np.asarray(waypoints, dtype=float)
    a, b = pts, np.roll(pts, -1, axis=0)
    keep = np.linalg.norm(b - a, axis=1) > 0                      # a repeated closing point gives no segment
    a, b = a[keep], b[keep]
    ab = b - a
    t = np.clip(np.einsum("ij,ij->i", np.asarray(p) - a, ab) / np.einsum("ij,ij->i", ab, ab), 0.0, 1.0)
    return float(np.min(np.linalg.norm(a + t[:, None] * ab - p, axis=1)))


# ------------------------------------------
# distance_to_box
# ------------------------------------------
# Distance from point ``p`` to the axis-aligned rectangle [lo, hi] (0 inside).
def distance_to_box(p, lo, hi) -> float:
    d = np.maximum(np.maximum(np.asarray(lo) - p, np.asarray(p) - np.asarray(hi)), 0.0)
    return float(np.linalg.norm(d))


# ------------------------------------------
# diagnostics
# ------------------------------------------
# Computes the trial-level diagnostics quoted in the discussion of F05.
def diagnostics(recs: dict) -> dict:
    world = yard()
    lo, hi = np.asarray(world.prior_lo[:2]), np.asarray(world.prior_hi[:2])
    faster, other = [], []
    for cond in ("C2", "C3"):
        for trial, fx in recs[cond]["fixed"].items():
            d = distance_to_loop(fx["q_true"][:2], DEFAULT_WAYPOINTS)
            (faster if fx["time_score"] < recs[cond]["info"][trial]["time_score"] else other).append(d)
    out = {
        "route_distance_m": {"conditions": ["C2", "C3"],
                             "fixed_faster": {"n": len(faster), "median": float(np.median(faster))},
                             "other": {"n": len(other), "median": float(np.median(other))}},
        "stop_distance_to_prior_m": {}, "last10_objective_nat": {}, "no_target_return_trials": {},
    }
    for cond, by_ctrl in recs.items():
        for key in ("stop_distance_to_prior_m", "last10_objective_nat", "no_target_return_trials"):
            out[key][cond] = {}
        for ctrl, trials in by_ctrl.items():
            r = list(trials.values())
            stops = [distance_to_box(x["final_xy"], lo, hi) for x in r]
            out["stop_distance_to_prior_m"][cond][ctrl] = float(np.median(stops))
            last = np.concatenate([x["score"][-10:] for x in r])
            last = last[np.isfinite(last)]
            out["last10_objective_nat"][cond][ctrl] = float(np.median(last)) if last.size else None
            out["no_target_return_trials"][cond][ctrl] = int(sum(int(x["n_target_detected"].sum()) == 0 for x in r))
    return out


# ------------------------------------------
# main
# ------------------------------------------
# Writes sim/out/F05/diagnostics.json and prints it.
def main() -> int:
    out = diagnostics(load())
    path = BASE / "diagnostics.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    print("wrote", path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
