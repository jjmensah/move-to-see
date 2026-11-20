# ==========================================================
# tests/unit/test_physics_radiometry.py
# ==========================================================
# Unit tests: atmosphere (E8), Lambertian radiometry (E6, E7), single specular bounce (E9).

import math

import numpy as np
import pytest

from lidar_sim.core.config import NoiseConfig
from lidar_sim.physics.atmosphere import extinction, kim_q, transmission
from lidar_sim.physics.radiometry import incidence_cos, lambertian_power
from lidar_sim.physics.specular import bounce_power, reflect

LAM = 785e-9


# ------------------------------------------
# test_kim_q_piecewise
# ------------------------------------------
# Kim's exponent on each piece of its definition.
@pytest.mark.parametrize("V_km, q", [(60, 1.6), (50.0001, 1.6), (50, 1.3), (20, 1.3), (6.0001, 1.3),
                                     (6, 0.16 * 6 + 0.34), (2, 0.66), (1.0001, 0.16 * 1.0001 + 0.34),
                                     (1, 0.5), (0.8, 0.3), (0.5001, 0.5001 - 0.5), (0.5, 0.0), (0.02, 0.0)])
def test_kim_q_piecewise(V_km, q):
    assert kim_q(V_km) == pytest.approx(q, abs=1e-12)


# ------------------------------------------
# test_kim_q_vectorized
# ------------------------------------------
# Kim's exponent for arrays and scalars.
def test_kim_q_vectorized():
    np.testing.assert_allclose(kim_q(np.array([100.0, 20.0, 0.1])), [1.6, 1.3, 0.0])
    assert isinstance(kim_q(20.0), float)


# ------------------------------------------
# test_extinction_o08
# ------------------------------------------
# Extinction and two-way transmission match O-08.
@pytest.mark.parametrize("V, sigma, T2", [(20000, 1.231714e-4, 0.999015), (2000, 1.546658e-3, 0.987703),
                                          (800, 4.394979e-3, 0.965451), (100, 3.912e-2, 0.731279),
                                          (50, 7.824e-2, 0.534769)])
def test_extinction_o08(V, sigma, T2):
    s = extinction(V, LAM)
    assert s == pytest.approx(sigma, rel=5e-7)
    assert transmission(np.array(4.0), s) ** 2 == pytest.approx(T2, rel=5e-6)


# ------------------------------------------
# test_extinction_validation_and_transmission_shape
# ------------------------------------------
# Invalid inputs are rejected and transmission keeps its shape.
def test_extinction_validation_and_transmission_shape():
    with pytest.raises(ValueError):
        extinction(0.0, LAM)
    with pytest.raises(ValueError):
        extinction(100.0, -1.0)
    r = np.array([[0.0, 1.0], [2.0, 3.0]])
    T = transmission(r, 0.1)
    assert T.shape == (2, 2) and T[0, 0] == 1.0
    np.testing.assert_allclose(T, np.exp(-0.1 * r))


# ------------------------------------------
# test_incidence_cos
# ------------------------------------------
# Cosine of incidence for chosen directions and normals.
def test_incidence_cos():
    d = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    n = np.array([[-1.0, 0.0], [-0.5, math.sqrt(3) / 2], [1.0, 0.0]])
    np.testing.assert_allclose(incidence_cos(d, n), [1.0, 0.5, 0.0])


# ------------------------------------------
# _P
# ------------------------------------------
# Lambertian return power with default settings.
def _P(rho=0.9, cos=1.0, r=2.0, sigma=0.0, w=1.0):
    n = NoiseConfig()
    return lambertian_power(n.P0, w, rho, cos, r, n.A_rx, n.eta, sigma)


# ------------------------------------------
# test_lambertian_power_formula_and_o07
# ------------------------------------------
# Lambertian power matches the formula and O-07.
def test_lambertian_power_formula_and_o07():
    n = NoiseConfig()
    sigma = extinction(20000.0, LAM)
    expect = n.P0 * 0.3 * (0.9 / math.pi) * 0.8 * (n.A_rx / 2.5**2) * math.exp(-2 * sigma * 2.5) * n.eta
    assert _P(0.9, 0.8, 2.5, sigma, 0.3) == pytest.approx(expect, rel=1e-14)
    # O-07: inverse-square in r (no fog), cosine law in incidence
    assert _P(r=4.0) / _P(r=2.0) == pytest.approx(0.25, rel=1e-12)
    assert _P(cos=math.cos(math.radians(60))) / _P(cos=1.0) == pytest.approx(0.5, rel=1e-12)


# ------------------------------------------
# test_lambertian_power_broadcasts
# ------------------------------------------
# Lambertian power broadcasts over reflectivity and range.
def test_lambertian_power_broadcasts():
    P = _P(rho=np.array([[0.1], [0.9]]), r=np.array([1.0, 2.0, 3.0]))
    assert P.shape == (2, 3)
    assert np.all(np.diff(P, axis=1) < 0) and np.all(P[1] > P[0])


# ------------------------------------------
# test_reflect
# ------------------------------------------
# Mirror reflection of a direction.
def test_reflect():
    d = np.array([1.0, 0.0])
    n = np.array([-1.0, 1.0]) / math.sqrt(2)
    np.testing.assert_allclose(reflect(d, n), [0.0, 1.0], atol=1e-15)
    rng = np.random.default_rng(1)
    dd = rng.normal(size=(10, 2))
    dd /= np.linalg.norm(dd, axis=1, keepdims=True)
    nn = rng.normal(size=(10, 2))
    nn /= np.linalg.norm(nn, axis=1, keepdims=True)
    rr = reflect(dd, nn)
    np.testing.assert_allclose(np.linalg.norm(rr, axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.sum(rr * nn, axis=1), -np.sum(dd * nn, axis=1), atol=1e-12)
    np.testing.assert_allclose(reflect(rr, nn), dd, atol=1e-12)


# ------------------------------------------
# test_bounce_power_is_rho_m_squared_times_diffuse
# ------------------------------------------
# A mirror bounce scales the diffuse power by the mirror reflectivity squared.
def test_bounce_power_is_rho_m_squared_times_diffuse():
    n = NoiseConfig()
    sigma = extinction(2000.0, LAM)
    Pb = bounce_power(n.P0, 0.5, 0.95, 0.9, 1.0, 4.0, n.A_rx, n.eta, sigma)
    Pd = lambertian_power(n.P0, 0.5, 0.9, 1.0, 4.0, n.A_rx, n.eta, sigma)
    assert Pb == pytest.approx(0.95**2 * Pd, rel=1e-14)
