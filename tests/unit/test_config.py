# ==========================================================
# tests/unit/test_config.py
# ==========================================================
# Unit tests of the sensor, noise and environment configuration.

import dataclasses
import math

import pytest

from lidar_sim.core import constants as K
from lidar_sim.core.config import (
    EnvConfig,
    ModelConfig,
    NoiseConfig,
    SensorConfig,
    deg2rad,
    km_to_m,
)


# ------------------------------------------
# test_unit_helpers
# ------------------------------------------
# Degree and kilometre conversions.
def test_unit_helpers():
    assert deg2rad(240.0) == pytest.approx(4 * math.pi / 3, rel=1e-15)
    assert km_to_m(20.0) == 20000.0


# ------------------------------------------
# test_sensor_defaults_match_spec
# ------------------------------------------
# Sensor defaults match the specification.
def test_sensor_defaults_match_spec():
    s = SensorConfig()
    assert s.fov == pytest.approx(4 * math.pi / 3, rel=1e-15)
    assert (s.n_beams, s.n_samples, s.n_subrays) == (682, 30, 5)
    assert (s.r_min, s.r_max, s.t_scan) == (0.02, 5.6, 0.1)
    assert (s.f1, s.f2) == (K.F1, K.F2) == (46.55e6, 53.2e6)
    assert s.theta0 == 5e-3


# ------------------------------------------
# test_ambiguity_intervals_O11_O12
# ------------------------------------------
# Ambiguity intervals of both frequencies and the coarse one (O-11, O-12).
def test_ambiguity_intervals_O11_O12():
    s = SensorConfig()
    assert s.L1 == pytest.approx(3.220112330827068, rel=1e-12)
    assert s.L2 == pytest.approx(2.8175982894736844, rel=1e-12)
    assert s.Lc == pytest.approx(22.540786315789475, rel=1e-12)
    assert s.Lc / s.L1 == pytest.approx(7.0, rel=1e-12)
    assert s.Lc / s.L2 == pytest.approx(8.0, rel=1e-12)


# ------------------------------------------
# test_configs_are_frozen
# ------------------------------------------
# Configurations cannot be changed after creation.
@pytest.mark.parametrize("cls", [SensorConfig, NoiseConfig, EnvConfig, ModelConfig])
def test_configs_are_frozen(cls):
    obj = cls()
    field = dataclasses.fields(obj)[0].name
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(obj, field, 1.0)


# ------------------------------------------
# test_env_defaults_and_replace
# ------------------------------------------
# Environment defaults and replacing a field.
def test_env_defaults_and_replace():
    e = EnvConfig()
    assert (e.visibility_m, e.P_amb, e.wavelength) == (20000.0, 0.0, 785e-9)
    fog = dataclasses.replace(e, visibility_m=km_to_m(0.05))
    assert fog.visibility_m == 50.0


# ------------------------------------------
# test_config_validation
# ------------------------------------------
# Invalid configurations are rejected.
def test_config_validation():
    with pytest.raises(ValueError):
        SensorConfig(n_samples=2)          # N >= 3 required for orthogonal regressors (E12)
    with pytest.raises(ValueError):
        EnvConfig(visibility_m=0.0)
    with pytest.raises(ValueError):
        SensorConfig(r_min=6.0)
    for bad in ({"fov": 0.0}, {"fov": 7.0}, {"t_scan": 0.0}, {"theta0": -1e-3},
                {"r_eps": 0.0}, {"P_ref": 0.0}, {"f2": 40e6}, {"b0": float("nan")},
                {"t_scan": float("inf")}, {"r_eps": float("inf")}, {"P_ref": float("inf")}):
        with pytest.raises(ValueError):
            SensorConfig(**bad)
