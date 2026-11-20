# ==========================================================
# sim/pilot.py
# ==========================================================
# Pilot runs on reserved seeds (>= 9000): never used by registered experiments (vv-plan §1.2).
#
# python -m sim.pilot [n_trials] [tag] [key=value ...]     e.g.  python -m sim.pilot 12 base planner.lambda_t=0.1

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from mts.config import CONDITIONS, T_STEPS
from mts.metrics import dwell_time, nees, position_error, time_score
from sim.experiments._common import load_trial, run_jobs, trial_job

OUT = Path(__file__).resolve().parent / "out" / "pilot"
SEED = 9001


# ------------------------------------------
# main
# ------------------------------------------
# Runs pilot trials on reserved seeds with optional configuration overrides.
def main(argv: list[str]) -> None:
    n = int(argv[0]) if argv else 12
    tag = argv[1] if len(argv) > 1 else "base"
    cfg: dict = {}
    for kv in argv[2:]:
        key, val = kv.split("=")
        group, name = key.split(".")
        cfg.setdefault(group, {})[name] = type(val)(val) if not val.replace(".", "").lstrip("-").isdigit() else (
            float(val) if "." in val else int(val))
    steps = cfg.pop("steps", {}).get("n", T_STEPS)
    jobs = [trial_job(OUT, tag, c, ctrl, SEED, t, steps, **cfg)
            for c in CONDITIONS for ctrl in ("info", "blind", "fixed") for t in range(n)]
    run_jobs(jobs, 10, f"pilot {tag}")
    print(f"{'cond':4s} {'ctrl':5s} {'succ':>5s} {'med τ̃':>7s} {'mean τ̃':>7s} {'med e_T':>8s} {'max e_T':>8s} "
          f"{'NEES<6.25':>9s} {'path':>5s} {'no-tgt':>6s} {'t/trial':>7s}")
    for c in CONDITIONS:
        for ctrl in ("info", "blind", "fixed"):
            rows = [load_trial(j) for j in jobs if j["condition"] == c and j["controller"] == ctrl]
            err = [position_error(r["q_hat"], r["q_true"]) for r in rows]
            dt = float(rows[0]["dt"])
            ts = np.array([time_score(e, dt) for e in err])
            succ = np.mean([np.isfinite(dwell_time(e, dt)) for e in err])
            eT = np.array([e[-1] for e in err])
            ne = np.array([nees(r["q_hat"][-1], r["cov"][-1], r["q_true"]) for r in rows])
            path = np.mean([np.hypot(*np.diff(r["states"][:, :2], axis=0).T).sum() for r in rows])
            notgt = np.mean([r["n_target_detected"].sum() == 0 for r in rows])
            wall = np.mean([r["wall_time"].sum() for r in rows])
            print(f"{c:4s} {ctrl:5s} {succ:5.2f} {np.median(ts):7.1f} {ts.mean():7.1f} {1e3 * np.median(eT):7.1f}mm "
                  f"{1e3 * eT.max():7.1f}mm {np.mean(ne <= 6.251):9.2f} {path:5.1f} {notgt:6.2f} {wall:7.1f}")


if __name__ == "__main__":
    main(sys.argv[1:])
