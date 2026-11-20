# ==========================================================
# tests/contract/test_dimension_allowlist.py
# ==========================================================
# C-07 (spec §3.2): dimension-agnostic modules use no angles or 2D-only helpers (AST-based).

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "lidar_sim"
AGNOSTIC = ["physics", "models", "estimation/fisher.py", "world/materials.py", "world/environment.py"]
FORBIDDEN = {"atan2", "arctan2", "cross", "beam_angle", "rot2", "from_xi", "Planar2DPattern", "Scene2D"}
FORBIDDEN_ATTRS = FORBIDDEN | {"xi"}      # Pose.xi() is 2D-only; a *variable* named xi is fine (PoseParam)


# ------------------------------------------
# violations
# ------------------------------------------
# Names/attributes/imports of 2D-only constructs. Docstrings and comments are ignored by AST.
def violations(source: str) -> set[str]:
    found = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
            found.add(node.attr)
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN:
            found.add(node.id)
        elif isinstance(node, ast.ImportFrom):
            found |= {a.name for a in node.names if a.name in FORBIDDEN}
            found |= {a.asname for a in node.names if a.name in FORBIDDEN and a.asname}
        elif isinstance(node, ast.Import):
            found |= {a.name for a in node.names if a.name.split(".")[-1] in FORBIDDEN}
    return found


# ------------------------------------------
# _files
# ------------------------------------------
# Source files that must stay dimension-agnostic.
def _files():
    out = []
    for entry in AGNOSTIC:
        p = SRC / entry
        out += sorted(p.rglob("*.py")) if p.is_dir() else ([p] if p.exists() else [])
    return out


# ------------------------------------------
# test_no_2d_specific_constructs
# ------------------------------------------
# Dimension-agnostic files use no 2D-only constructs.
@pytest.mark.parametrize("path", _files(), ids=lambda p: str(p.relative_to(SRC)))
def test_no_2d_specific_constructs(path):
    hits = violations(path.read_text())
    assert not hits, f"{path.relative_to(SRC)} uses 2D-specific {sorted(hits)} (spec §3.2)"


# ------------------------------------------
# test_detector
# ------------------------------------------
# The 2D-construct detector flags what it should and nothing else.
@pytest.mark.parametrize("src, bad", [
    ("import numpy as np\nnp.cross(a, b)", True),
    ("import numpy\nnumpy.cross(a, b)", True),
    ("from numpy import cross as c\nc(a, b)", True),
    ("import math\nmath.atan2(1, 2)", True),
    ("np.arctan2(y, x)", True),
    ("p.xi()", True),
    ("xi = param.from_pose(pose)", False),
    ("def f(xi):\n    return xi + 1", False),
    ('"""uses atan2 in prose only"""\nx = 1  # cross in a comment', False),
    ("np.einsum('...i,...i->...', a, b)", False),
])
def test_detector(src, bad):
    assert bool(violations(src)) is bad
