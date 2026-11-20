# ==========================================================
# tests/contract/test_equation_refs.py
# ==========================================================
# Equation IDs cited in src exist in the spec; every core equation is cited.

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = (ROOT / "docs" / "design" / "sensor-spec.md").read_text()
DEFINED = set(re.findall(r"\((E\d+a?)\)", SPEC))
CORE = {f"E{i}" for i in range(1, 26)} - {"E9"} | {"E3a"}     # E9 specular is a stretch flag


# ------------------------------------------
# _cited
# ------------------------------------------
# Equation ids cited in each source file.
def _cited() -> dict[str, set[str]]:
    out = {}
    for p in sorted((ROOT / "src").rglob("*.py")):
        ids = set(re.findall(r"\b(E\d+a?)\b", p.read_text()))
        if ids:
            out[str(p.relative_to(ROOT))] = ids
    return out


# ------------------------------------------
# test_spec_defines_all_core_equations
# ------------------------------------------
# The sensor specification defines every core equation.
def test_spec_defines_all_core_equations():
    assert CORE <= DEFINED, f"spec missing {sorted(CORE - DEFINED)}"


# ------------------------------------------
# test_cited_ids_exist_in_spec
# ------------------------------------------
# Every equation id cited in the code exists in the specification.
def test_cited_ids_exist_in_spec():
    bad = {f: ids - DEFINED for f, ids in _cited().items() if ids - DEFINED}
    assert not bad, f"undefined equation ids cited: {bad}"


# ------------------------------------------
# test_every_core_equation_cited
# ------------------------------------------
# Every core equation is cited somewhere in the code.
def test_every_core_equation_cited():
    cited = set().union(*_cited().values()) if _cited() else set()
    assert CORE <= cited, f"uncited: {sorted(CORE - cited, key=lambda s: int(re.sub(r'\D', '', s)))}"
