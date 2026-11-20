# ==========================================================
# sim/run.py
# ==========================================================
# Experiment runner: ``python -m sim.run --list | <ID>… | all [--quick] [--jobs N]``.
#
# Each experiment module ``sim/experiments/<ID>_*.py`` exposes
# ``run(entry: dict, out: Path, quick: bool, jobs: int) -> dict`` and returns a JSON-serialisable metrics dict
# that contains a ``verdicts`` mapping (criterion name → bool) for registered pass criteria. The runner writes
# ``metrics.json`` and ``run.json`` (resolved entry, seed, git revision, versions, timing) to ``sim/out/<ID>/``.

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "sim" / "registry.yaml"
REQUIRED = ["id", "title", "hypothesis", "criteria", "equations", "params", "n", "seed", "script", "outputs",
            "tests", "exploratory"]
ID_RE = re.compile(r"^(F|M)\d{2}$")
QUICK_SEED_BASE = 9500                       # pilot range: --quick replaces the registered seed


# ------------------------------------------
# load_registry
# ------------------------------------------
# Reads and validates sim/registry.yaml.
def load_registry(path: Path = REGISTRY) -> list[dict]:
    entries = yaml.safe_load(path.read_text())
    validate(entries)
    return entries


# ------------------------------------------
# validate
# ------------------------------------------
# Validate the registry (raises ``ValueError`` naming the entry and field).
def validate(entries: list[dict]) -> None:
    if not isinstance(entries, list) or not entries:
        raise ValueError("registry: must be a non-empty list")
    ids = [e.get("id") if isinstance(e, dict) else None for e in entries]
    if len(ids) != len(set(ids)):
        raise ValueError("registry: duplicate ids")
    for e in entries:
        eid = e.get("id") if isinstance(e, dict) else None
        if not isinstance(e, dict) or not isinstance(eid, str) or not ID_RE.match(eid):
            raise ValueError(f"{eid}: id must match F## or M##")
        missing = [k for k in REQUIRED if k not in e]
        unknown = set(e) - set(REQUIRED)
        if missing or unknown:
            raise ValueError(f"{eid}: missing keys {missing}, unknown keys {sorted(unknown)}")
        for k in ("title", "hypothesis", "script"):
            if not isinstance(e[k], str) or not e[k].strip():
                raise ValueError(f"{eid}: {k} must be a non-empty string")
        if not isinstance(e["criteria"], dict) or not all(isinstance(v, str) for v in e["criteria"].values()):
            raise ValueError(f"{eid}: criteria must map a criterion name to its statement")
        if not isinstance(e["exploratory"], bool) or (not e["exploratory"] and not e["criteria"]):
            raise ValueError(f"{eid}: a confirmatory experiment needs at least one criterion")
        if not (isinstance(e["params"], dict) and isinstance(e["equations"], list) and isinstance(e["tests"], list)):
            raise ValueError(f"{eid}: params must be a mapping; equations and tests lists")
        for k in ("n", "seed"):
            if not isinstance(e[k], int) or isinstance(e[k], bool) or e[k] < 0:
                raise ValueError(f"{eid}: {k} must be a non-negative integer")
        if e["seed"] >= 9000:
            raise ValueError(f"{eid}: seeds >= 9000 are reserved for pilots")
        if Path(e["script"]).parent.as_posix() != "sim/experiments" or Path(e["script"]).stem.split("_")[0] != eid:
            raise ValueError(f"{eid}: script must be sim/experiments/{eid}_*.py")
        if not isinstance(e["outputs"], list) or not all(str(f).startswith(f"sim/out/{eid}/") for f in e["outputs"]):
            raise ValueError(f"{eid}: outputs must be paths under sim/out/{eid}/")


# ------------------------------------------
# git_revision
# ------------------------------------------
# Current revision and the tracked files under ``src/`` and ``sim/`` with uncommitted modifications.
#
# Untracked files are ignored (they cannot change the behaviour of committed code that does not import them);
# modified tracked files are listed so that a reader can judge whether they matter for the experiment.
def git_revision() -> tuple[str, list[str]]:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        status = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no", "--", "src", "sim"],
                                         cwd=ROOT, text=True)
        return sha, sorted(line[3:] for line in status.splitlines() if line.strip())
    except (OSError, subprocess.CalledProcessError):
        return "unknown", ["<git unavailable>"]


# ------------------------------------------
# source_hash
# ------------------------------------------
# SHA-256 over every Python source under ``src/`` and ``sim/`` and the registry (tracked or not).
def source_hash() -> str:
    h = hashlib.sha256()
    files = sorted([*(ROOT / "src").rglob("*.py"), *(ROOT / "sim").glob("*.py"),
                    *(ROOT / "sim" / "experiments").glob("*.py"), REGISTRY])
    for f in files:
        h.update(f.relative_to(ROOT).as_posix().encode())
        h.update(f.read_bytes())
    return h.hexdigest()


# ------------------------------------------
# run_entry
# ------------------------------------------
# Runs one registered experiment and records its seed, revision and source fingerprint.
def run_entry(entry: dict, quick: bool = False, jobs: int = 1) -> dict:
    sha0, modified0, hash0 = *git_revision(), source_hash()        # state before any experiment code is loaded
    script = ROOT / entry["script"]
    spec = importlib.util.spec_from_file_location(f"sim.experiments.{script.stem}", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = ROOT / "sim" / "out" / (entry["id"] + ("_quick" if quick else ""))
    out.mkdir(parents=True, exist_ok=True)
    if quick:                                   # smoke runs never touch the registered random streams
        entry = {**entry, "seed": QUICK_SEED_BASE + int(entry["id"][1:])}
    t0 = time.perf_counter()
    metrics = mod.run(entry, out, quick, jobs)
    wall = time.perf_counter() - t0
    sha, modified = git_revision()
    import matplotlib
    import scipy
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    (out / "run.json").write_text(json.dumps({
        "entry": entry, "seed": entry["seed"], "quick": quick, "jobs": jobs, "git_sha": sha0,
        "git_dirty": bool(modified0), "git_modified": modified0, "source_sha256": hash0,
        "git_sha_end": sha, "git_modified_end": modified, "source_unchanged_during_run": source_hash() == hash0,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                     "matplotlib": matplotlib.__version__},
        "timestamp": datetime.now(UTC).isoformat(), "wall_time_s": wall}, indent=2))
    return metrics


# ------------------------------------------
# main
# ------------------------------------------
# Command line: list, run or smoke-run registered experiments.
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sim.run")
    ap.add_argument("ids", nargs="*", help="experiment ids (F01 …), or 'all'")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="smoke run: reduced sample sizes, pilot seed, output in <ID>_quick")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2), help="worker processes")
    a = ap.parse_args(argv)
    entries = load_registry()
    if a.list or not a.ids:
        for e in entries:
            print(f"{e['id']:4s} {e['title']}{' (exploratory)' if e['exploratory'] else ''}")
        return 0
    by_id = {e["id"]: e for e in entries}
    for e in (entries if a.ids == ["all"] else [by_id[i] for i in a.ids]):
        print(f"== {e['id']}: {e['title']}", flush=True)
        m = run_entry(e, a.quick, a.jobs)
        print(json.dumps(m.get("verdicts", {}), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
