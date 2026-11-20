#!/usr/bin/env python3
# ==========================================================
# web/scripts/export_decisions.py
# ==========================================================
# Replay C2/0 and export exact pre-action diagnostics without changing policy RNG.

import sys

sys.dont_write_bytecode = True
import copy
import hashlib
import time
from unittest.mock import patch

from build_data import DATA, FIXTURES, arguments, encode, manifest, np, read, require, source_hash, write
from mts.config import CONDITIONS, PlannerConfig
from mts.planner import ViewpointPlanner
from mts.sim import run_trial
from mts.world import yard
from sim.experiments._common import load_trial, trial_job


# ------------------------------------------
# main
# ------------------------------------------
# Reruns fog trial 0 and exports three planning steps if the rerun matches the record.
def main():
    out = arguments()
    # A failed rebuild must never leave a stale decision visible.
    (DATA / "decisions.json").unlink(missing_ok=True)
    manifest()
    original = load_trial(trial_job(out / "F05", "main", "C2", "info", 5105, 0, 60))
    run = read(out / "F05/run.json")
    print(f"Recorded source hash matches current: {run['source_sha256'] == source_hash()}", flush=True)
    snapshots = []
    private = []
    original_act = ViewpointPlanner.act

    def act(self, k, x, belief, rng):
        if k not in (0, 10, 20):
            return original_act(self, k, x, belief, rng)
        previous = self._goal
        mean = belief.mean().copy()
        rng_before = copy.deepcopy(rng.bit_generator.state)
        particles = belief.particles.copy()
        weights = belief.weights.copy()
        captured = {}
        evaluated = set()
        support_fn = belief.support
        gains_fn = self._gains

        def support(m, generator):
            value = support_fn(m, generator)
            captured["support"] = value.copy()
            return value

        def gains(indices, m, s):
            evaluated.update(map(int, indices))
            return gains_fn(indices, m, s)

        with patch.object(belief, "support", support), patch.object(self, "_gains", gains):
            action = original_act(self, k, x, belief, rng)
        rng_after = copy.deepcopy(rng.bit_generator.state)
        support = captured["support"]
        selected = int(self._goal)
        times = self._travel[0].ravel()
        stay = int(np.ravel_multi_index(self.grid.cell_of(x[:2]), self.grid.free.shape))
        candidates = np.union1d(self._lattice, [stay])
        candidates = candidates[np.isfinite(times[candidates])]
        # A separate planner owns diagnostic geometry caches; supplemental scoring cannot
        # affect the running controller, filter, or either random stream.
        diagnostic = ViewpointPlanner(self.world, self.cfg, self.pc, self.predictor, self.kind)
        all_gains = {}
        for offset in range(0, len(candidates), 16):
            all_gains.update(diagnostic._gains(candidates[offset : offset + 16], mean, support))
        rows = []
        for index in candidates:
            index = int(index)
            xy = self.grid.centre(np.unravel_index(index, self.grid.free.shape))
            gain = float(all_gains[index])
            travel = float(times[index])
            rows.append(
                dict(
                    id=index,
                    xy=xy.tolist(),
                    heading=float(np.arctan2(mean[1] - xy[1], mean[0] - xy[0])),
                    informationNat=gain,
                    travelS=travel,
                    scoreNat=gain - self.pc.lambda_t * travel,
                    evaluatedByPolicy=index in evaluated,
                )
            )
        by_id = {r["id"]: r for r in rows}
        best = max(rows, key=lambda r: (r["scoreNat"], -r["travelS"], -r["id"]))["id"]
        kept = previous in by_id and by_id[best]["scoreNat"] - by_id[previous]["scoreNat"] < self.pc.switch_margin
        require(selected == (previous if kept else best), f"k={k}: selected ID mismatch")
        require(kept == action.info["goal_kept"], f"k={k}: persistence mismatch")
        require(abs(by_id[selected]["scoreNat"] - action.info["score"]) <= 1e-9, f"k={k}: supplemental score mismatch")
        require(encode(rng_after) == encode(rng.bit_generator.state), "Diagnostic consumed RNG")
        require(
            np.array_equal(particles, belief.particles) and np.array_equal(weights, belief.weights),
            "Diagnostic mutated belief",
        )
        stride = int(np.ceil(len(particles) / 300))
        sample = particles[::stride, :2]
        sample_weights = weights[::stride]
        sample_weights = sample_weights / sample_weights.sum()
        # Quantize normalized weights to units of 1e-4 with deterministic largest
        # remainder allocation, preserving sum=1 after rounding.
        units = np.floor(sample_weights * 10000).astype(int)
        order = np.argsort(-(sample_weights * 10000 - units), kind="stable")
        units[order[: 10000 - int(units.sum())]] += 1
        digest = hashlib.sha256(support.tobytes()).hexdigest()
        snapshots.append(
            dict(
                k=k,
                state=x.tolist(),
                mean=mean.tolist(),
                supportSha256=digest,
                previousId=int(previous) if previous is not None else None,
                bestId=best,
                selectedId=selected,
                goalKept=bool(kept),
                candidates=rows,
                particlesXY=np.round(sample, 4).tolist(),
                weights=(units / 10000).tolist(),
                support=support[:, :2].tolist(),
            )
        )
        private.append(
            dict(
                k=k,
                support=support.tolist(),
                supportSha256=digest,
                rngBefore=rng_before,
                rngAfter=rng_after,
                evaluatedIds=sorted(evaluated),
                selectedId=selected,
                goalKept=bool(kept),
            )
        )
        print(
            f"Captured k={k}: {len(rows)} candidates, {len(evaluated)} policy evaluations, selected={selected}",
            flush=True,
        )
        return action

    t0 = time.perf_counter()
    with patch.object(ViewpointPlanner, "act", act):
        rec = run_trial(yard(), CONDITIONS["C2"], "info", 5105, 0, steps=21)
    discrepancies = {}
    for field, tol in [("states", 1e-8), ("goals", 1e-8), ("score", 1e-9), ("q_hat", 1e-8), ("cov", 1e-8)]:
        a = getattr(rec, field)
        b = original[field][: len(a)]
        require(np.array_equal(np.isnan(a), np.isnan(b)), f"{field}: NaN mask differs")
        finite = np.isfinite(a)
        delta = float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else 0.0
        discrepancies[field] = delta
        require(delta <= tol, f"Exact replay failed: {field} max error {delta} exceeds {tol}; no decisions published")
    for field in ("n_detected", "n_target_detected", "fallback", "infeasible", "controls"):
        require(np.array_equal(getattr(rec, field), original[field][:21]), f"Exact replay failed: {field}")
    config = PlannerConfig()
    payload = dict(
        version=1,
        condition="C2",
        trial=0,
        seed=5105,
        controller="info",
        **{"lambda": config.lambda_t},
        switchMargin=config.switch_margin,
        supportSize=config.M,
        steps=snapshots,
    )
    write(
        FIXTURES / "decision_reproduction.json",
        dict(
            elapsedS=time.perf_counter() - t0,
            maxAbsoluteError=discrepancies,
            sourceSha256=source_hash(),
            recordedSourceSha256=run["source_sha256"],
            steps=private,
        ),
    )
    write(DATA / "decisions.json", payload)
    manifest()
    print("Decision replay reproduced:", discrepancies, flush=True)
    print(f"Elapsed: {time.perf_counter() - t0:.2f} s", flush=True)


if __name__ == "__main__":
    main()
