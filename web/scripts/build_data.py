#!/usr/bin/env python3
# ==========================================================
# web/scripts/build_data.py
# ==========================================================
# Export validated, compact website data from read-only registered results.

from __future__ import annotations

import sys

sys.dont_write_bytecode = True
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from datetime import UTC, datetime
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "web"
DATA = WEB / "public/data"
FIXTURES = WEB / "tests/fixtures"
# Imports, plotting initialization and test caches must not modify the source tree.
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mts-mpl-"))
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import numpy as np
import scipy
from mts.metrics import dwell_time, position_error, time_score
from sim.run import load_registry, source_hash

CONDS = ("C1", "C2", "C3", "C4")
CTRLS = ("info", "blind", "fixed")


# ------------------------------------------
# require
# ------------------------------------------
# Raises with a message unless a check holds.
def require(ok, message):
    if not ok:
        raise ValueError(message)


# ------------------------------------------
# read
# ------------------------------------------
# Reads a JSON file.
def read(path):
    return json.loads(Path(path).read_text())


# ------------------------------------------
# encode
# ------------------------------------------
# Compact, sorted JSON that refuses NaN.
def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=False).encode()


# ------------------------------------------
# write
# ------------------------------------------
# Writes a JSON file atomically.
def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(encode(value))
    tmp.replace(path)


# ------------------------------------------
# sha
# ------------------------------------------
# SHA-256 of a file.
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ------------------------------------------
# arguments
# ------------------------------------------
# Reads the results directory from the command line or MTS_RESULTS_DIR.
def arguments():
    parser = argparse.ArgumentParser(description="Export the website data from the registered results.")
    parser.add_argument("--results-dir", default=os.environ.get("MTS_RESULTS_DIR"))
    args = parser.parse_args()
    if not args.results_dir:
        parser.error("--results-dir or MTS_RESULTS_DIR is required")
    path = Path(args.results_dir).resolve()
    require(path.is_dir(), f"Missing results directory: {path}")
    return path


# ------------------------------------------
# compare
# ------------------------------------------
# Compares recomputed values with the recorded ones, within tight tolerances.
def compare(actual, expected, path="F05"):
    if isinstance(expected, dict):
        require(actual.keys() == expected.keys(), f"{path}: keys differ")
        for k in expected:
            compare(actual[k], expected[k], f"{path}.{k}")
    elif isinstance(expected, list):
        require(len(actual) == len(expected), f"{path}: length differs")
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
            compare(a, e, f"{path}[{i}]")
    elif isinstance(expected, float):
        tol = 1e-12 if path.endswith(("p_raw", "p_holm")) else 1e-10
        require(
            np.isfinite(actual) and abs(actual - expected) <= tol, f"{path}: {actual!r} != {expected!r} (atol={tol})"
        )
    else:
        require(actual == expected, f"{path}: {actual!r} != {expected!r}")


# ------------------------------------------
# camel
# ------------------------------------------
# snake_case to camelCase for the browser.
def camel(key):
    overrides = {"mcnemar_info_only_success": "infoOnlySuccess", "mcnemar_other_only_success": "otherOnlySuccess"}
    return overrides.get(key, re.sub(r"_([a-z])", lambda m: m[1].upper(), key))


# ------------------------------------------
# start
# ------------------------------------------
# Index of the first qualifying dwell window, or None.
def start(error):
    value = dwell_time(error, 0.5)
    return int(value / 0.5) if np.isfinite(value) else None


# ------------------------------------------
# load_records
# ------------------------------------------
# Loads and validates all 480 F05 trial records.
def load_records(out):
    from sim.experiments._common import trial_job, load_trial

    records = {}
    shapes = {
        "states": (61, 4),
        "q_hat": (61, 3),
        "cov": (61, 3, 3),
        "q_true": (3,),
        "controls": (60, 2),
        "goals": (60, 2),
        "wall_time": (60, 2),
    }
    for key in ("fallback", "infeasible", "n_detected", "n_target_detected", "ess", "stages", "score"):
        shapes[key] = (60,)
    expected = set()
    for c in CONDS:
        for ctrl in CTRLS:
            for t in range(40):
                job = trial_job(out / "F05", "main", c, ctrl, 5105, t, 60)
                expected.add(Path(job["path"]))
                r = load_trial(job)
                require(float(r["dt"]) == 0.5, f"{c}/{ctrl}/{t}: dt")
                for field, shape in shapes.items():
                    require(r[field].shape == shape, f"{c}/{ctrl}/{t}: {field} shape")
                    if field not in ("goals", "score"):
                        require(np.isfinite(r[field]).all(), f"{c}/{ctrl}/{t}: nonfinite {field}")
                require(np.all(np.isfinite(r["goals"]).all(axis=1) | np.isnan(r["goals"]).all(axis=1)), "Invalid goal")
                require(not np.isinf(r["score"]).any(), "Infinite score")
                require(np.allclose(r["cov"], r["cov"].transpose(0, 2, 1), rtol=0, atol=1e-12), "Asymmetric covariance")
                require(np.linalg.eigvalsh(r["cov"]).min() >= -1e-10, "Negative covariance")
                for field in ("n_detected", "n_target_detected"):
                    require(
                        np.all(r[field] >= 0) and np.all(r[field] == r[field].astype(int)), "Invalid detection count"
                    )
                require(np.all(r["n_target_detected"] <= r["n_detected"]), "Target counts exceed detections")
                records[c, ctrl, t] = r
    require(set((out / "F05/trials/main").rglob("*.npz")) == expected, "Missing, extra or duplicate F05 trials")
    for c in CONDS:
        for t in range(40):
            for ctrl in CTRLS:
                require(
                    np.array_equal(records[c, ctrl, t]["q_true"], records[c, "info", t]["q_true"]), "Unpaired truth"
                )
    return records


# ------------------------------------------
# aggregate
# ------------------------------------------
# Recomputes F05 with the registered code and checks it against metrics.json.
def aggregate(out, records):
    # Reuse the registered aggregation verbatim, but never schedule jobs or generate plots.
    from sim.experiments import F05_main as f05
    from unittest.mock import patch

    entry = next(e for e in load_registry() if e["id"] == "F05")

    def cached(job):
        return records[job["condition"], job["controller"], job["trial"]]

    with (
        patch.object(f05, "run_jobs", lambda *args: None),
        patch.object(f05, "_figures", lambda *args: None),
        patch.object(f05, "load_trial", cached),
    ):
        computed = f05.run(entry, out / "F05", False, 1)
    compare(computed, read(out / "F05/metrics.json"))
    result = dict(
        version=1,
        n=40,
        steps=60,
        dt=0.5,
        paired={},
        summary={},
        comparisons=[{camel(k): v for k, v in row.items()} for row in computed["comparisons"]],
        control={camel(k): v for k, v in computed["control_C4"].items()},
        verdicts={k[:2]: v for k, v in computed["verdicts"].items()},
    )
    for c in CONDS:
        result["summary"][c] = {}
        result["paired"][c] = []
        for ctrl in CTRLS:
            row = computed["summary"][c][ctrl]
            result["summary"][c][ctrl] = {camel(k): v for k, v in row.items()}
            result["summary"][c][ctrl]["successCount"] = round(row["success_rate"] * 40)
        for t in range(40):
            errors = [position_error(records[c, ctrl, t]["q_hat"], records[c, ctrl, t]["q_true"]) for ctrl in CTRLS]
            result["paired"][c].append(
                dict(
                    trial=t,
                    timeScoreS=[time_score(e, 0.5) for e in errors],
                    dwellStart=[start(e) for e in errors],
                    finalErrorM=[float(e[-1]) for e in errors],
                )
            )
    return result


# ------------------------------------------
# world
# ------------------------------------------
# The yard, route and conditions for the browser.
def world():
    from mts.world import yard
    from mts.route import DEFAULT_WAYPOINTS

    w = yard()
    return dict(
        version=1,
        arena=w.arena,
        boxes=[dict(lo=a.tolist(), hi=b.tolist()) for a, b in zip(w.box_lo, w.box_hi)],
        rough=w.rough.tolist(),
        prior=dict(lo=w.prior_lo.tolist(), hi=w.prior_hi.tolist()),
        targetHalf=w.target_half.tolist(),
        start=w.start.tolist(),
        fixedRoute=DEFAULT_WAYPOINTS.tolist(),
        routeClosed=True,
        dt=0.5,
        steps=60,
        toleranceM=0.1,
        dwellSamples=5,
        conditionsOrder=list(CONDS),
        controllerLabels=dict(info="Information-driven", blind="Physics-blind", fixed="Fixed route"),
        conditions={
            c: dict(label=label, rho=rho, visibilityM=vis, sun=sun)
            for c, label, rho, vis, sun in [
                ("C1", "Clear (black target)", 0.05, 20000, 0),
                ("C2", "Fog (20 m visibility)", 0.05, 20, 0),
                ("C3", "Sunlight (1 sun)", 0.05, 20000, 1),
                ("C4", "White target (clear)", 0.9, 20000, 0),
            ]
        },
    )


# ------------------------------------------
# replay
# ------------------------------------------
# One replay file per condition and trial, with rounding checked.
def replay(records):
    payloads = {}
    max_error = dict(states=0.0, mean=0.0, covXY=0.0, error=0.0, goals=0.0)
    ellipse_cases = []
    for c in CONDS:
        for t in range(40):
            controllers = {}
            for ctrl in CTRLS:
                r = records[c, ctrl, t]
                error = position_error(r["q_hat"], r["q_true"])
                rounded = np.round(error, 7)
                require(np.array_equal(error <= 0.1, rounded <= 0.1), "Error rounding changes threshold classification")
                require(start(error) == start(rounded), "Dwell rounding changes start")
                xy = r["cov"][:, [0, 0, 1], [0, 1, 1]]
                item = dict(
                    dwellStart=start(error),
                    timeScoreS=time_score(error, 0.5),
                    nDetected=r["n_detected"].tolist(),
                    nTargetDetected=r["n_target_detected"].tolist(),
                )
                for field, array, dp in [
                    ("states", r["states"][:, :3], 5),
                    ("mean", r["q_hat"], 6),
                    ("covXY", xy, 10),
                    ("error", error, 7),
                ]:
                    rounded_array = np.round(array, dp)
                    max_error[field] = max(max_error[field], float(np.max(np.abs(array - rounded_array))))
                    item[field] = rounded_array.tolist()
                item["goals"] = [np.round(g, 5).tolist() if np.isfinite(g).all() else None for g in r["goals"]]
                valid = r["goals"][np.isfinite(r["goals"]).all(axis=1)]
                if valid.size:
                    max_error["goals"] = max(max_error["goals"], float(np.max(np.abs(valid - np.round(valid, 5)))))
                ellipse_cases.extend(item["covXY"])
                controllers[ctrl] = item
            payloads[f"replay/{c}/{t:02d}.json"] = dict(
                version=1,
                condition=c,
                trial=t,
                seed=5105,
                dt=0.5,
                qTrue=records[c, "info", t]["q_true"].tolist(),
                controllers=controllers,
            )
    return payloads, max_error, ellipse_cases


# ------------------------------------------
# sensor
# ------------------------------------------
# Detection and range-noise curves over reflectivity, fog and sunlight.
def sensor():
    from lidar_sim.core.config import EnvConfig
    from lidar_sim.world.environment import sunlight
    from lidar_sim.physics.atmosphere import extinction
    from mts.config import make_model, NOISE
    from mts.fov import probe_trace, probe_pd, r50

    ranges = np.arange(1, 281) * 0.02
    rhos, visibility, suns = [0.05, 0.1, 0.2, 0.4, 0.6, 0.9], [20, 50, 100, 200, 20000], [0, 0.25, 0.5, 0.75, 1]
    curves = []
    for rho in rhos:
        for vis in visibility:
            for sun in suns:
                env = sunlight(sun, EnvConfig(visibility_m=vis), NOISE)
                pred = make_model(1).predict(probe_trace(ranges, rho, env), env)
                require(np.allclose(pred.P_D, probe_pd(ranges, rho, env), rtol=0, atol=5e-8), "Probe PD mismatch")
                require(np.all(np.diff(pred.P_D) <= 1e-10), "PD is not monotone")
                require(np.all(np.diff(pred.sigma_R) >= -1e-10), "Sigma is not monotone")
                curve = dict(
                    pd=np.round(pred.P_D, 8).tolist(),
                    sigmaM=np.round(pred.sigma_R, 8).tolist(),
                    r50M=round(r50(rho, env), 8),
                )
                require(np.max(np.abs(np.array(curve["sigmaM"]) - pred.sigma_R)) <= 5e-8, "Sigma rounding")
                require(np.all((np.array(curve["pd"]) >= 0) & (np.array(curve["pd"]) <= 1)), "Invalid PD")
                curves.append(curve)
    return dict(
        version=1,
        rangeM=np.round(ranges, 2).tolist(),
        rho=rhos,
        visibilityM=visibility,
        sun=suns,
        extinctionPerM=[float(extinction(v, EnvConfig().wavelength)) for v in visibility],
        reportingWindow=[0.02, 5.6],
        curves=curves,
    )


# ------------------------------------------
# fixtures
# ------------------------------------------
# Python oracle values for the browser helpers.
def fixtures(covs):
    cases = []
    for cov in covs + [[0, 0, 0], [1, 0, 1], [4, 0, 0], [0, 0, 4], [1, -1, 1], [-5e-11, 0, 1]]:
        xx, xy, yy = cov
        vals, vecs = np.linalg.eigh([[xx, xy], [xy, yy]])
        require(vals.min() >= -1e-10, "Invalid XY covariance")
        angle = float(np.arctan2(vecs[1, 1], vecs[0, 1])) if vals[1] != vals[0] else 0.0
        cases.append(
            dict(covXY=cov, a=float(2 * np.sqrt(max(0, vals[1]))), b=float(2 * np.sqrt(max(0, vals[0]))), angle=angle)
        )
    write(FIXTURES / "ellipses.json", cases)
    errors = [
        [],
        [0.1] * 5,
        [0.1] * 4,
        [1] * 56 + [0.1] * 5,
        [1] * 57 + [0.1] * 4,
        [1] * 61,
        [0.10000000001] * 5,
        [0.1, 0.1, 0.1, 0.1, 1, 0.1, 0.1, 0.1, 0.1, 0.1],
        [0] * 61,
    ]
    cases = []
    for error in errors:
        for dt, tol, dwell in [(0.5, 0.1, 5), (0.25, 0.1, 1), (2, 0.2, 3)]:
            d = dwell_time(error, dt, tol, dwell)
            cases.append(
                dict(
                    error=error,
                    dt=dt,
                    tol=tol,
                    dwell=dwell,
                    dwellStart=int(d / dt) if np.isfinite(d) else None,
                    timeScoreS=time_score(error, dt, tol, dwell),
                )
            )
    write(FIXTURES / "metrics.json", cases)
    write(
        FIXTURES / "sensor.json",
        [
            dict(rhoIdx=r, visIdx=v, sunIdx=s, index=(r * 5 + v) * 5 + s)
            for r in range(6)
            for v in range(5)
            for s in range(5)
        ],
    )


# ------------------------------------------
# tests
# ------------------------------------------
# Runs the test suite and records its counts.
def tests():
    with tempfile.TemporaryDirectory(prefix="mts-tests-") as tmp:
        junit = str(Path(tmp) / "pytest.xml")
        env = dict(os.environ, HYPOTHESIS_STORAGE_DIRECTORY=str(Path(tmp) / "hypothesis"))
        cmd = [".venv/bin/python", "-m", "pytest", "-p", "no:cacheprovider", f"--junitxml={junit}"]
        run = subprocess.run(cmd, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(run.stdout, flush=True)
        require(Path(junit).exists(), "Pytest did not produce JUnit")
        cases = ET.parse(junit).findall(".//testcase")
        failed = sum(c.find("failure") is not None or c.find("error") is not None for c in cases)
        skipped = sum(c.find("skipped") is not None for c in cases)
        deselected_match = re.search(r"(\d+) deselected", run.stdout)
        require(deselected_match is not None, "Missing pytest deselection count")
        deselected = int(deselected_match[1])
        stat_cmd = [".venv/bin/python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--collect-only", "-m", "stat"]
        stat = subprocess.run(stat_cmd, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(stat.stdout, flush=True)
        # Quiet collection prints one path: count per module.
        stat_count = sum(int(n) for n in re.findall(r"^tests/[^\n]+: (\d+)\s*$", stat.stdout, re.M))
        require(
            stat.returncode == 0 and stat_count == deselected, "Stat collection count differs from deselected tests"
        )
        result = dict(
            command=" ".join(cmd[:-1] + ["--junitxml=<temporary file>"]),
            sourceSha256=source_hash(),
            collected=len(cases) + deselected,
            passed=len(cases) - failed - skipped,
            failed=failed,
            skipped=skipped,
            deselected=deselected,
            executedAt=datetime.now(UTC).isoformat(),
            statCollected=stat_count,
            statCommand=" ".join(stat_cmd),
        )
        write(FIXTURES / "test_run.json", result)
        require(run.returncode == 0, f"Pytest failed: {failed} failures/errors; see fixtures/test_run.json")
        return result


# ------------------------------------------
# evidence
# ------------------------------------------
# The verification ledger: criteria, verdicts, observations and deviations.
def evidence(out, test_result):
    from scipy.stats import spearmanr

    experiments = []
    for entry in sorted(load_registry(), key=lambda x: x["id"]):
        eid = entry["id"]
        m = read(out / eid / "metrics.json")
        run = read(out / eid / "run.json")
        require(set(m["verdicts"]) == set(entry["criteria"]), f"{eid}: registry criteria differ")
        require(run["entry"]["criteria"] == entry["criteria"], f"{eid}: recorded registry criteria differ")
        source = f"OUT/{eid}/metrics.json"
        obs = []

        def add(label, value, unit=""):
            obs.append(dict(label=label, value=value, unit=unit, source=source))

        if eid == "F01":
            require(len(m["cells"]) == m["n_cells"], "F01 cell count")
            compare(max(abs(c["crossing_rate"] - c["P_D"]) for c in m["cells"]), m["max_abs_crossing_error"])
            add("Cells", m["n_cells"])
            add("High-SNR cells", m["n_high_snr"])
            add("Largest crossing-rate gap", m["max_abs_crossing_error"])
        elif eid == "F02":
            worst = max(c["rel_diff"] for row in m["volumes"] for c in row["cases"].values())
            compare(worst, m["max_rel_diff"])
            add("Maximum relative volume discrepancy", worst)
        elif eid == "F03":
            for c, rows in m["conditions"].items():
                for k, row in rows.items():
                    add(f"{c}, scan {k}: coverage", row["coverage_90"])
        elif eid == "F04":
            compare(len(m["cases"]), m["n_cases"])
            compare(sum(not c["pass"] for c in m["cases"]), m["n_violations"])
            groups = {}
            for c in m["cases"]:
                groups.setdefault(c["belief"], []).append(c)
            median = float(
                np.median(
                    [
                        spearmanr([c["bound_nats"] for c in rows], [c["mc_nats"] for c in rows]).statistic
                        for rows in groups.values()
                    ]
                )
            )
            compare(median, m["median_spearman"])
            for key in ("n_cases", "n_violations", "median_spearman"):
                add(key.replace("_", " "), m[key])
        elif eid == "F05":
            for c, rows in m["summary"].items():
                for ctrl, row in rows.items():
                    add(f"{c} {ctrl}: median time score", row["median_time_score_s"], "s")
        elif eid == "F06":
            require(m["exploratory"] and not m["verdicts"], "F06 must remain exploratory")
            for variant, rows in m["summary"].items():
                for ctrl, row in rows.items():
                    for key in ("success_rate", "median_time_score_s", "mean_path_length_m", "mean_patch_time_s"):
                        if key in row:
                            add(
                                f"{variant} {ctrl}: {key}",
                                row[key],
                                "s" if key.endswith("_s") else "m" if key.endswith("_m") else "",
                            )
        else:
            for ctrl, row in m["controllers"].items():
                add(f"{ctrl}: final error", row["final_error_m"], "m")
        experiments.append(
            dict(
                id=eid,
                title=entry["title"],
                exploratory=entry["exploratory"],
                source=source,
                criteria=[dict(key=k, text=v, supported=m["verdicts"][k]) for k, v in entry["criteria"].items()],
                observations=obs,
                provenance=dict(
                    seed=run["seed"],
                    sourceSha256=run["source_sha256"],
                    timestamp=run["timestamp"],
                ),
            )
        )
    text = (
        (ROOT / "docs/verification/vv-plan.md")
        .read_text()
        .split("## 5. Deviations from the registration")[1]
        .split("\n## ")[0]
    )
    deviations = [
        dict(text=re.sub(r"\s+", " ", s).strip(), source="docs/verification/vv-plan.md")
        for s in re.split(r"\n\d\. ", text)[1:]
    ]
    require(len(deviations) == 4, "Missing deviations")
    return dict(
        version=1, experiments=experiments, deviations=deviations, tests=test_result, source="sim/registry.yaml"
    )


# ------------------------------------------
# manifest
# ------------------------------------------
# Lists every data file with its hash and size, and checks the size budget.
def manifest():
    files = []
    for path in sorted(DATA.rglob("*.json")):
        if path.name == "manifest.json":
            continue
        raw = path.read_bytes()
        files.append(
            dict(
                path=path.relative_to(DATA).as_posix(),
                sha256=hashlib.sha256(raw).hexdigest(),
                bytes=len(raw),
                gzipBytes=len(gzip.compress(raw, mtime=0)),
            )
        )
    m = dict(
        version=1,
        sourceSha256=source_hash(),
        registrySha256=sha(ROOT / "sim/registry.yaml"),
        files=files,
        totalBytes=sum(f["bytes"] for f in files),
        totalGzipBytes=sum(f["gzipBytes"] for f in files),
        decisionAvailable=(DATA / "decisions.json").exists(),
    )
    write(DATA / "manifest.json", m)
    total = m["totalGzipBytes"] + len(gzip.compress(encode(m), mtime=0))
    require(total <= 1_600_000, f"Data gzip budget exceeded: {total}")
    families = {}
    for f in files + [
        dict(path="manifest.json", bytes=len(encode(m)), gzipBytes=len(gzip.compress(encode(m), mtime=0)))
    ]:
        family = f["path"].split("/")[0].removesuffix(".json")
        a = families.setdefault(family, [0, 0])
        a[0] += f["bytes"]
        a[1] += f["gzipBytes"]
    print("Family bytes (raw / gzip):", json.dumps(families, sort_keys=True), flush=True)
    print(f"Total gzip including manifest: {total:,} bytes", flush=True)
    return families


# ------------------------------------------
# main
# ------------------------------------------
# Builds every data file of the site.
def main():
    out = arguments()
    runs = {e["id"]: read(out / e["id"] / "run.json") for e in load_registry()}
    provenance = dict(
        currentSourceSha256=source_hash(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        experiments={
            eid: dict(
                sourceMatches=r["source_sha256"] == source_hash(),
                versions=r["versions"],
                metricsSha256=sha(out / eid / "metrics.json"),
                runSha256=sha(out / eid / "run.json"),
            )
            for eid, r in runs.items()
        },
    )
    write(FIXTURES / "provenance.json", provenance)
    print("Source hash matches:", {k: v["sourceMatches"] for k, v in provenance["experiments"].items()}, flush=True)
    records = load_records(out)
    results = aggregate(out, records)
    print("Validated all F05 fields, five comparisons, control interval and verdicts.", flush=True)
    payloads, rounding, covs = replay(records)
    payloads.update({"world.json": world(), "results.json": results, "sensor.json": sensor()})
    fixtures(covs[::50])  # a deterministic sample; every covariance is still checked by check_models.mjs
    print("Maximum encoding errors:", rounding, flush=True)
    write(FIXTURES / "encoding.json", rounding)
    payloads["evidence.json"] = evidence(out, tests())
    for name, value in payloads.items():
        write(DATA / name, value)
    manifest()
    print("Success counts:", {c: [results["summary"][c][k]["successCount"] for k in CTRLS] for c in CONDS}, flush=True)


if __name__ == "__main__":
    main()
