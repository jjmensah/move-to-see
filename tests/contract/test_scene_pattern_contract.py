# ==========================================================
# tests/contract/test_scene_pattern_contract.py
# ==========================================================
# C-02 (Scene.intersect contract) and C-03 (ScanPattern contract) for the 2D implementations.

import math
import zlib

import numpy as np
import pytest

from lidar_sim.core.config import SensorConfig
from lidar_sim.geometry.base import Scene
from lidar_sim.geometry.scene2d import Scene2D
from lidar_sim.sensor.scan_pattern import Planar2DPattern, ScanPattern
from lidar_sim.world import scenes


# ------------------------------------------
# test_c02_intersect_contract
# ------------------------------------------
# Every canonical scene meets the intersection contract (C-02).
@pytest.mark.parametrize("name", sorted(scenes.CANONICAL))
def test_c02_intersect_contract(name):
    spec = scenes.CANONICAL[name]()
    scene = Scene2D.from_spec(spec)
    assert isinstance(scene, Scene) and scene.dim == 2
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    N = 2000
    o = rng.uniform(-6, 6, (N, 2))
    a = rng.uniform(-math.pi, math.pi, N)
    d = np.stack([np.cos(a), np.sin(a)], axis=1)
    t = rng.uniform(0, 0.1, N)
    h = scene.intersect(o, d, t)
    assert h.n == N and h.dim == 2
    assert h.hit.any()
    hit = h.hit
    np.testing.assert_allclose(np.linalg.norm(h.normal[hit], axis=1), 1.0, atol=1e-12)
    assert (np.sum(h.normal[hit] * d[hit], axis=1) <= 0).all()
    assert (h.r[hit] > scene.r_eps).all()
    scale = 1.0 + np.abs(o[hit]).max() + h.r[hit].max()
    np.testing.assert_allclose(h.point[hit], o[hit] + h.r[hit, None] * d[hit], atol=1e-9 * scale)
    assert (h.surface_id[~hit] == -1).all()
    assert ((h.surface_id[hit] >= 0) & (h.surface_id[hit] < scene.n_surfaces)).all()


# ------------------------------------------
# test_c03_pattern_contract
# ------------------------------------------
# The planar scan pattern meets its contract for each sub-ray count (C-03).
@pytest.mark.parametrize("cfg", [SensorConfig(), SensorConfig(n_beams=2, fov=2 * math.pi, t_scan=0.025),
                                 SensorConfig(n_beams=100, theta0=0.02), SensorConfig(theta0=0.0)])
@pytest.mark.parametrize("K", [1, 3, 5, 8])
def test_c03_pattern_contract(cfg, K):
    pat = Planar2DPattern(cfg)
    assert isinstance(pat, ScanPattern) and pat.dim == 2
    u = pat.beam_dirs_sensor()
    assert u.shape == (cfg.n_beams, 2)
    np.testing.assert_allclose(np.linalg.norm(u, axis=1), 1.0, atol=1e-12)
    dirs, w = pat.subrays(K)
    assert dirs.shape == (cfg.n_beams, K, 2) and w.shape == (cfg.n_beams, K)
    np.testing.assert_allclose(np.linalg.norm(dirs, axis=-1), 1.0, atol=1e-12)
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-12)
    assert (w >= 0).all()
    t = pat.fire_times()
    assert t.shape == (cfg.n_beams,) and (np.diff(t) >= 0).all() and t[0] == 0.0
    assert pat.beam_angle().shape == (cfg.n_beams,)
