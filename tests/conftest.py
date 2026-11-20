# ==========================================================
# tests/conftest.py
# ==========================================================
# Shared pytest fixtures. Every stochastic test draws from an explicitly seeded Generator.

import hashlib

import numpy as np
import pytest

BASE_SEED = 4240


# ------------------------------------------
# seeded
# ------------------------------------------
# Deterministic Generator derived from ``name`` (e.g. a test node id).
def seeded(name: str) -> np.random.Generator:
    digest = hashlib.sha256(name.encode()).digest()
    return np.random.default_rng([BASE_SEED, int.from_bytes(digest[:8], "little")])


# ------------------------------------------
# rng
# ------------------------------------------
# A generator seeded from the test's id, so every test is reproducible.
@pytest.fixture
def rng(request) -> np.random.Generator:
    return seeded(request.node.nodeid)
