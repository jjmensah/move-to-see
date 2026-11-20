# ==========================================================
# tests/contract/test_mts_contracts.py
# ==========================================================
# T-C1: contracts of the final-project code (final-spec §10, vv-plan §1.4).
#
# Import boundaries, no access to truth in inference and planning (structural and metamorphic), and equation
# ids cited in the code exist in the specifications.

import ast
import inspect
import re
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
INFERENCE = ["belief", "planner", "info", "predict", "control", "route"]
FORBIDDEN_PARAMS = {"q_true", "truth", "truth_scene", "scan", "trial"}


# ------------------------------------------
# _imports
# ------------------------------------------
# Module names imported by a file.
def _imports(path: Path) -> set[str]:
    out = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            out |= {node.module} | {f"{node.module}.{a.name}" for a in node.names}
    return out


# ------------------------------------------
# test_lidar_sim_never_imports_mts
# ------------------------------------------
# The sensor package never depends on the robot package.
def test_lidar_sim_never_imports_mts():
    for path in (SRC / "lidar_sim").rglob("*.py"):
        assert not any(m == "mts" or m.startswith("mts.") for m in _imports(path)), path


# ------------------------------------------
# test_inference_never_imports_waveform_or_sim
# ------------------------------------------
# Inference uses the analytic model only (ADR-0005) and never the trial runner.
@pytest.mark.parametrize("module", INFERENCE)
def test_inference_never_imports_waveform_or_sim(module):
    mods = _imports(SRC / "mts" / f"{module}.py")
    assert not any("waveform" in m for m in mods), mods
    assert not any(m in ("mts.sim", "sim") or m.startswith("mts.sim.") for m in mods), mods


# ------------------------------------------
# test_inference_signatures_do_not_accept_truth
# ------------------------------------------
# No inference or planning function can be handed the true target.
@pytest.mark.parametrize("module", INFERENCE)
def test_inference_signatures_do_not_accept_truth(module):
    mod = __import__(f"mts.{module}", fromlist=["*"])
    for _, obj in inspect.getmembers(mod, lambda o: inspect.isfunction(o) or inspect.isclass(o)):
        if getattr(obj, "__module__", None) != mod.__name__:
            continue
        funcs = [obj] if inspect.isfunction(obj) else [f for _, f in inspect.getmembers(obj, inspect.isfunction)]
        for f in funcs:
            params = set(inspect.signature(f).parameters)
            if f.__qualname__ == "Observation.from_scan":          # the one sanctioned entry point for a Scan
                continue
            assert not (params & FORBIDDEN_PARAMS), (f.__qualname__, params & FORBIDDEN_PARAMS)


# ------------------------------------------
# test_inference_source_never_touches_truth_attributes
# ------------------------------------------
# No ``.truth`` attribute access and no use of the truth-scene builder inside inference modules.
def test_inference_source_never_touches_truth_attributes():
    for module in INFERENCE:
        tree = ast.parse((SRC / "mts" / f"{module}.py").read_text())
        attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        assert not (attrs & {"truth", "truth_scene", "q_true"}), (module, attrs & {"truth", "truth_scene", "q_true"})


# ------------------------------------------
# test_run_trial_passes_only_observations_to_inference
# ------------------------------------------
# In ``run_trial`` the controller, the belief and the predictor are never called with truth-bearing names.
def test_run_trial_passes_only_observations_to_inference():
    tree = ast.parse((SRC / "mts" / "sim.py").read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "run_trial")
    bad = {"truth_scene", "scan", "q_true", "trace", "tr", "strongest"}
    checked = 0
    for call in (n for n in ast.walk(fn) if isinstance(n, ast.Call)):
        f = call.func
        owner = f.value.id if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) else None
        name = f.id if isinstance(f, ast.Name) else None
        if owner in ("ctrl", "belief", "predictor") or name in ("Predictor", "ParticleBelief", "make_controller"):
            checked += 1
            names = {n.id for a in [*call.args, *[k.value for k in call.keywords]] for n in ast.walk(a)
                     if isinstance(n, ast.Name)}
            assert not (names & bad), (ast.unparse(call), names & bad)
    assert checked >= 6


# ------------------------------------------
# test_metamorphic_hidden_truth_does_not_influence_inference
# ------------------------------------------
# Change the hidden target while holding the observations fixed: estimates and actions must not change.
#
# ``World.truth_scene`` is patched to ignore its argument, so every scan is generated from the same
# reference target whatever ``q_true`` the trial is given. Any path from ``q_true`` to the belief or the
# controller other than through the observation would make the two trials differ.
def test_metamorphic_hidden_truth_does_not_influence_inference(monkeypatch):
    from mts import sim
    from mts.config import CONDITIONS, BeliefConfig
    from mts.world import World, yard

    q_ref = np.array([7.6, 4.3, 0.8])
    original = World.truth_scene
    monkeypatch.setattr(World, "truth_scene", lambda self, q: original(self, q_ref))
    runs = [sim.run_trial(yard(), CONDITIONS["C1"], "info", seed=9100, trial=0, steps=12, q_true=q,
                          belief_cfg=BeliefConfig(n_particles=400))
            for q in (np.array([7.1, 3.0, 0.6]), np.array([8.5, 5.1, 1.0]))]
    a, b = runs
    assert not np.allclose(a.q_true, b.q_true)
    for f in ("states", "controls", "goals", "q_hat", "cov", "ess", "stages", "n_detected"):
        assert np.array_equal(getattr(a, f), getattr(b, f), equal_nan=True), f


# ------------------------------------------
# test_equation_ids_cited_in_mts_exist_in_the_specs
# ------------------------------------------
# Equation ids cited in the robot package exist in the specifications.
def test_equation_ids_cited_in_mts_exist_in_the_specs():
    final = (ROOT / "docs" / "design" / "final-spec.md").read_text()
    sensor = (ROOT / "docs" / "design" / "sensor-spec.md").read_text()
    defined_f = set(re.findall(r"\((F\d+)\)", final))
    defined_e = set(re.findall(r"\b(E\d+a?)\b", sensor))
    assert {f"F{i}" for i in range(1, 23)} <= defined_f                    # F1–F22 are all defined
    cited_f, cited_e = set(), set()
    files = [*sorted((SRC / "mts").glob("*.py")), SRC / "lidar_sim" / "geometry" / "scene3d.py",
             SRC / "lidar_sim" / "sensor" / "scan3d.py", *sorted((ROOT / "sim" / "experiments").glob("F*.py"))]
    for path in files:
        text = path.read_text()
        for lo, hi in re.findall(r"\bF(\d+)[–-]F?(\d+)\b", text):           # ranges such as F10–F12
            cited_f |= {f"F{i}" for i in range(int(lo), int(hi) + 1)}
        cited_f |= set(re.findall(r"\b(F\d+)\b(?!_)", text)) - {"F01", "F02", "F03", "F04", "F05", "F06"}
        cited_e |= set(re.findall(r"\b(E\d+a?)\b", text))
    assert cited_f <= defined_f, cited_f - defined_f
    assert cited_e <= defined_e, cited_e - defined_e
    assert len(cited_f) >= 15                                              # the code actually cites the spec
