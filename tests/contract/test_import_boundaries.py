# ==========================================================
# tests/contract/test_import_boundaries.py
# ==========================================================
# C-06 (ADR-0005): the analytic model and estimation layers never import the waveform simulator.

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "lidar_sim"
GUARDED = ["models", "estimation"]
FORBIDDEN = "lidar_sim.physics.waveform"


# ------------------------------------------
# _imports
# ------------------------------------------
# Absolute module names imported by a file, with relative imports resolved.
def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    pkg = ".".join(path.relative_to(SRC.parent).with_suffix("").parts[:-1])
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parts = pkg.split(".")[: len(pkg.split(".")) - node.level + 1]
                base = ".".join(parts + ([base] if base else []))
            out.add(base)
            out |= {f"{base}.{a.name}" for a in node.names}
    return out


# ------------------------------------------
# _files
# ------------------------------------------
# Files of the guarded packages.
def _files():
    return [p for g in GUARDED for p in sorted((SRC / g).rglob("*.py"))]


# ------------------------------------------
# test_no_waveform_import
# ------------------------------------------
# The analytic model and estimation never import the waveform simulator.
@pytest.mark.parametrize("path", _files(), ids=lambda p: str(p.relative_to(SRC)))
def test_no_waveform_import(path):
    bad = {m for m in _imports(path) if m == FORBIDDEN or m.startswith(FORBIDDEN + ".")}
    assert not bad, f"{path.name} imports {bad} (violates ADR-0005)"


# ------------------------------------------
# test_detector_catches_violation
# ------------------------------------------
# The import check catches every way of importing the simulator.
@pytest.mark.parametrize("stmt", [
    "from lidar_sim.physics import waveform",
    "import lidar_sim.physics.waveform",
    "import lidar_sim.physics.waveform as wf",
    "from lidar_sim.physics.waveform import measure",
    "from ..physics import waveform",
    "from ..physics.waveform import measure",
])
def test_detector_catches_violation(stmt):
    f = SRC / "models" / "_probe_tmp.py"
    f.write_text(stmt + "\n")
    try:
        found = _imports(f)
        assert any(m == FORBIDDEN or m.startswith(FORBIDDEN + ".") for m in found), found
    finally:
        f.unlink()


# ------------------------------------------
# test_detector_allows_other_physics
# ------------------------------------------
# The import check allows the other physics modules.
def test_detector_allows_other_physics():
    f = SRC / "models" / "_probe_tmp.py"
    f.write_text("from ..physics import radiometry\nfrom lidar_sim.physics.atmosphere import extinction\n")
    try:
        assert not any(m.startswith(FORBIDDEN) for m in _imports(f))
    finally:
        f.unlink()
