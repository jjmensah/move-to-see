# ==========================================================
# sim/experiments/_common.py
# ==========================================================
# Shared helpers for experiments: parallel trial batches with on-disk caching, and loading.

from __future__ import annotations

import os

# One numerical thread per worker process; must be set before NumPy loads its BLAS (workers are spawned).
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import hashlib  # noqa: E402
import json  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402


# ------------------------------------------
# _one
# ------------------------------------------
# Worker: run one trial and save it (imports inside so that thread limits apply before NumPy loads).
def _one(job: dict) -> tuple[str, str]:
    from mts.config import CONDITIONS, BeliefConfig, PlannerConfig, RobotConfig
    from mts.sim import run_trial
    from mts.world import yard

    path = Path(job["path"])
    if path.exists():
        return job["key"], "cached"
    world = yard(**job.get("world", {}))
    rec = run_trial(world, CONDITIONS[job["condition"]], job["controller"], job["seed"], job["trial"],
                    robot_cfg=RobotConfig(**job.get("robot", {})), belief_cfg=BeliefConfig(**job.get("belief", {})),
                    planner_cfg=PlannerConfig(**job.get("planner", {})), steps=job["steps"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **rec.to_npz())
    tmp.replace(path)
    return job["key"], "ran"


# ------------------------------------------
# trial_job
# ------------------------------------------
# Job description; ``cfg`` may hold ``world``, ``robot``, ``belief`` and ``planner`` keyword overrides.
def trial_job(out: Path, tag: str, condition: str, controller: str, seed: int, trial: int, steps: int, **cfg) -> dict:
    key = f"{tag}/{condition}/{controller}/{trial:03d}"
    job = {"condition": condition, "controller": controller, "seed": int(seed), "trial": int(trial),
           "steps": int(steps), **cfg}
    # The cache file name carries a fingerprint of everything that defines the trial, so a record computed with
    # another seed, length or configuration can never be mistaken for this one.
    digest = hashlib.sha256(json.dumps(job, sort_keys=True, default=str).encode()).hexdigest()[:10]
    return {"key": key, "path": str(out / "trials" / f"{key}_{digest}.npz"), **job}


# ------------------------------------------
# run_jobs
# ------------------------------------------
# Run trial jobs (cached trials are skipped), printing coarse progress.
def run_jobs(jobs: list[dict], n_workers: int, label: str = "") -> None:
    todo = [j for j in jobs if not Path(j["path"]).exists()]
    print(f"{label}: {len(jobs)} trials, {len(jobs) - len(todo)} cached, running {len(todo)} on {n_workers} workers",
          flush=True)
    if not todo:
        return
    if n_workers <= 1:
        for i, j in enumerate(todo):
            _one(j)
            print(f"  {i + 1}/{len(todo)}", flush=True)
        return
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(_one, j) for j in todo]
        for i, f in enumerate(as_completed(futs)):
            f.result()
            if (i + 1) % max(1, len(todo) // 20) == 0 or i + 1 == len(todo):
                print(f"  {i + 1}/{len(todo)}", flush=True)


# ------------------------------------------
# load_trial
# ------------------------------------------
# Load a cached trial and check that it is the trial the job describes.
def load_trial(job: dict) -> dict:
    with np.load(job["path"], allow_pickle=False) as z:
        rec = {k: z[k] for k in z.files}
    same = (str(rec["controller"]) == job["controller"] and str(rec["condition"]) == job["condition"]
            and int(rec["seed"]) == job["seed"] and int(rec["trial"]) == job["trial"]
            and rec["controls"].shape[0] == job["steps"])
    if not same:
        raise ValueError(f"cached trial {job['path']} does not match its job description")
    return rec
