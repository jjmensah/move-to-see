# ==========================================================
# src/mts/control.py
# ==========================================================
# Controller protocol shared by the planners and the fixed-route baseline (interfaces.md).

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


# ------------------------------------------
# Action
# ------------------------------------------
# One decision: control ``u = (a, ω)``, the goal it pursues (or ``None``), and diagnostics.
@dataclass(frozen=True, eq=False)
class Action:
    u: np.ndarray
    goal: np.ndarray | None = None
    fallback: bool = False
    info: dict = field(default_factory=dict)


# ------------------------------------------
# Controller
# ------------------------------------------
# What every controller provides: a name and an action per step.
class Controller(Protocol):
    name: str

    def act(self, k: int, x: np.ndarray, belief, rng: np.random.Generator) -> Action: ...
