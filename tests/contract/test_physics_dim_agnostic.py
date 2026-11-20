# ==========================================================
# tests/contract/test_physics_dim_agnostic.py
# ==========================================================
# C-01: physics/atmosphere, radiometry, specular accept (B,J), (B,J,d) arrays for d = 2 and 3,
# preserve leading dims and return finite outputs.

import math

import numpy as np
import pytest

from lidar_sim.core.config import NoiseConfig
from lidar_sim.core.types import RadiometricScan
from lidar_sim.physics.atmosphere import extinction, transmission
from lidar_sim.physics.radiometry import incidence_cos, lambertian_power
from lidar_sim.physics.specular import bounce_power, reflect

B, J = 4, 6


# ------------------------------------------
# _unit
# ------------------------------------------
# Random unit vectors of the given shape.
def _unit(rng, shape):
    v = rng.normal(size=shape)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


# ------------------------------------------
# test_vector_functions_preserve_leading_dims
# ------------------------------------------
# Physics functions accept batches of 2D and 3D vectors and keep their shape.
@pytest.mark.parametrize("d", [2, 3])
def test_vector_functions_preserve_leading_dims(d):
    rng = np.random.default_rng(d)
    dirs = _unit(rng, (B, J, d))
    n = _unit(rng, (B, J, d))
    n = np.where((np.sum(n * dirs, -1) > 0)[..., None], -n, n)      # orient n·d <= 0
    c = incidence_cos(dirs, n)
    assert c.shape == (B, J) and np.all(np.isfinite(c)) and np.all((c >= 0) & (c <= 1))
    np.testing.assert_allclose(c, -np.sum(dirs * n, -1), atol=1e-15)
    rr = reflect(dirs, n)
    assert rr.shape == (B, J, d)
    np.testing.assert_allclose(np.linalg.norm(rr, axis=-1), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.sum(rr * n, -1), c, atol=1e-12)          # angle of reflection


# ------------------------------------------
# test_power_functions_on_return_axis
# ------------------------------------------
# Hand-built d-dimensional RadiometricScan-like arrays: power on (B, J) without any 2D assumption.
@pytest.mark.parametrize("d", [2, 3])
def test_power_functions_on_return_axis(d):
    rng = np.random.default_rng(10 + d)
    nz = NoiseConfig()
    sigma = extinction(800.0, 785e-9)
    trace = RadiometricScan.empty(B, J // 2, J, d=d)
    r = rng.uniform(0.5, 5.0, (B, J))
    cos = incidence_cos(trace.dir_world[:, trace.sub_idx[0]], -trace.dir_world[:, trace.sub_idx[0]])
    w = trace.w[:, trace.sub_idx[0]]
    P = lambertian_power(nz.P0, w, 0.5, cos, r, nz.A_rx, nz.eta, sigma)
    Pb = bounce_power(nz.P0, w, 0.95, 0.5, cos, r, nz.A_rx, nz.eta, sigma)
    for arr in (P, Pb):
        assert arr.shape == (B, J) and np.all(np.isfinite(arr)) and np.all(arr > 0)
    np.testing.assert_allclose(Pb, 0.95**2 * P, rtol=1e-14)
    T = transmission(r, sigma)
    assert T.shape == (B, J)
    np.testing.assert_allclose(P, nz.P0 * w * 0.5 / math.pi * nz.A_rx / r**2 * T**2 * nz.eta, rtol=1e-14)
