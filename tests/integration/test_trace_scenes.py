# ==========================================================
# tests/integration/test_trace_scenes.py
# ==========================================================
# Integration (vv-plan §6): pose + canonical scene → RadiometricScan, asserting each builder's documented
# hand-computable facts. (Checks at the ``measure`` / ``Scan`` level are in the waveform tests.)

import math
import time

import numpy as np
import pytest

from lidar_sim.core.config import NoiseConfig, SensorConfig
from lidar_sim.core.pose import Pose
from lidar_sim.geometry.scene2d import Scene2D
from lidar_sim.physics.atmosphere import extinction
from lidar_sim.physics.radiometry import lambertian_power
from lidar_sim.sensor.lidar import LidarSensor
from lidar_sim.sensor.scan_pattern import Planar2DPattern
from lidar_sim.world import environment, scenes
from lidar_sim.world.materials import BLACK, GREY, MATERIALS, MIRROR, WHITE

NZ = NoiseConfig()
ORIGIN = Pose.identity(2)


# ------------------------------------------
# _trace
# ------------------------------------------
# Traces a canonical scene from a pose in clear air.
def _trace(spec, pose=ORIGIN, **cfg):
    sensor = LidarSensor(SensorConfig(**cfg))
    return sensor.trace(pose, Scene2D.from_spec(spec), environment.CLEAR)


# ------------------------------------------
# _angles
# ------------------------------------------
# World angle of each direct-return sub-ray (B, K).
def _angles(tr, j=slice(None)):
    d = tr.dir_world
    return np.arctan2(d[..., 1], d[..., 0])[:, j]


# ------------------------------------------
# _assert_finite_hits
# ------------------------------------------
# Every hit carries finite range, power, incidence and normal.
def _assert_finite_hits(tr):
    h = tr.hit
    for f in (tr.r_path, tr.power, tr.cos_inc):
        assert np.all(np.isfinite(f[h]))
    assert np.all(np.isfinite(tr.normal[h]))


# ------------------------------------------
# _assert_points_on_surfaces
# ------------------------------------------
# Every direct hit point o + r d lies on the reported surface segment with matching ids.
def _assert_points_on_surfaces(tr, scene, origins=None, tol=1e-9):
    K = tr.K
    o = np.zeros((tr.B, K, 2)) if origins is None else origins
    h = tr.hit[:, :K]
    p = o + tr.r_path[:, :K, None] * tr.dir_world
    for b, k in zip(*np.nonzero(h), strict=True):
        sid = tr.surface_id[b, k]
        a, e = (seg := scene.segments_at(tr.fire_time[b])[sid])[0], seg[1] - seg[0]
        q = p[b, k] - a
        assert abs(q[0] * e[1] - q[1] * e[0]) / np.linalg.norm(e) < tol
        s = q @ e / (e @ e)
        assert -1e-9 <= s <= 1 + 1e-9
        assert tr.material_id[b, k] == scene.material_ids[sid]
        assert tr.object_id[b, k] == scene.object_ids[sid]


# ------------------------------------------
# test_single_wall
# ------------------------------------------
# Single wall: hits, ranges and incidence by hand.
def test_single_wall():
    spec = scenes.single_wall()
    tr = _trace(spec)
    a = _angles(tr)
    expect_hit = (np.abs(a) < math.pi / 2) & (np.abs(2 * np.tan(a)) <= 1.0)
    margin = np.abs(np.abs(2 * np.tan(a)) - 1.0) > 1e-9
    np.testing.assert_array_equal(tr.hit[margin], expect_hit[margin])
    np.testing.assert_allclose(tr.r_path[tr.hit], (2 / np.cos(a))[tr.hit], rtol=1e-12)
    np.testing.assert_allclose(tr.normal[tr.hit], np.tile([-1.0, 0.0], (tr.hit.sum(), 1)), atol=1e-15)
    assert (tr.surface_id[tr.hit] == 0).all() and (tr.material_id[tr.hit] == WHITE).all()
    # O-07 via trace: beam power ∝ w cos α / r² with r = 2/cos α (clear air, single surface)
    sigma = extinction(environment.CLEAR.visibility_m, environment.CLEAR.wavelength)
    expect = lambertian_power(NZ.P0, tr.w, 0.9, np.cos(a), 2 / np.cos(a), NZ.A_rx, NZ.eta, sigma)
    np.testing.assert_allclose(tr.power[tr.hit], expect[tr.hit], rtol=1e-12)
    _assert_finite_hits(tr)
    tr2 = _trace(spec)                                    # deterministic / reproducible
    for f in ("hit", "power", "surface_id"):
        np.testing.assert_array_equal(getattr(tr, f), getattr(tr2, f))
    np.testing.assert_array_equal(np.isnan(tr.r_path), np.isnan(tr2.r_path))


# ------------------------------------------
# test_two_walls_occluded
# ------------------------------------------
# Two walls: the near wall hides the far one where it should.
def test_two_walls_occluded():
    spec = scenes.two_walls_occluded()
    tr = _trace(spec)
    a = _angles(tr)
    occ = math.atan(0.25)                                  # 14.04°
    far = math.atan(0.5)                                   # 26.57°
    assert math.degrees(occ) == pytest.approx(14.04, abs=5e-3)
    assert math.degrees(far) == pytest.approx(26.57, abs=5e-3)
    tol = 1e-9
    m_occ = np.abs(a) < occ - tol
    m_far = (np.abs(a) > occ + tol) & (np.abs(a) < far - tol)
    m_none = np.abs(a) > far + tol
    assert m_occ.any() and m_far.any()
    assert (tr.surface_id[m_occ] == 0).all() and (tr.material_id[m_occ] == GREY).all()
    np.testing.assert_allclose(tr.r_path[m_occ], 2 / np.cos(a[m_occ]), rtol=1e-12)
    assert (tr.surface_id[m_far] == 1).all() and (tr.material_id[m_far] == WHITE).all()
    np.testing.assert_allclose(tr.r_path[m_far], 4 / np.cos(a[m_far]), rtol=1e-12)
    assert not tr.hit[m_none].any()
    # P-07 at scene level: the far wall is never hit inside the occluder's shadow
    assert not ((tr.surface_id == 1) & (np.abs(a) < occ - tol)).any()


# ------------------------------------------
# test_corner
# ------------------------------------------
# Corner: each beam hits the wall it points at, switching exactly at the corner.
def test_corner():
    spec = scenes.corner()
    tr = _trace(spec)
    a = _angles(tr)
    ac = math.atan2(2, 3)
    assert math.degrees(ac) == pytest.approx(33.69, abs=5e-3)
    tol = 1e-9
    m0 = (a > -math.pi / 4 + tol) & (a < ac - tol)
    m1 = (a > ac + tol) & (a < math.pi - ac - tol)
    assert (tr.surface_id[m0] == 0).all() and (tr.surface_id[m1] == 1).all()
    np.testing.assert_allclose(tr.r_path[m0], 3 / np.cos(a[m0]), rtol=1e-12)
    np.testing.assert_allclose(tr.r_path[m1], 2 / np.sin(a[m1]), rtol=1e-12)
    assert not tr.hit[a < -math.pi / 4 - tol].any()
    # the corner is the farthest point of both walls near α_c: r → √13
    near = np.abs(a - ac) < 0.02
    assert tr.r_path[near].max() <= math.sqrt(13) * (1 + 1e-12)
    assert tr.r_path[near].max() == pytest.approx(math.sqrt(13), abs=0.01)
    # surfaces switch exactly at α_c
    assert a[tr.surface_id == 0].max() <= ac + 1e-12 and a[tr.surface_id == 1].min() >= ac - 1e-12


# ------------------------------------------
# test_room_asym_closed_and_materials
# ------------------------------------------
# Asymmetric room: every beam hits and materials are reported.
def test_room_asym_closed_and_materials():
    spec = scenes.room_asym()
    scene = Scene2D.from_spec(spec)
    pose = Pose.from_xi([1.5, 1.5, 0.0])
    tr = LidarSensor().trace(pose, scene, environment.CLEAR)
    assert tr.hit.all() and (tr.r_path < 10.0).all()
    _assert_finite_hits(tr)
    _assert_points_on_surfaces(tr, scene, origins=np.broadcast_to(pose.t, (tr.B, tr.K, 2)))
    # black floor patch x ∈ [2, 3] (surface 1) is seen at ψ = 0; the grey left-wall patch y ∈ [1, 2]
    # (surface 8) lies behind the sensor (outside ±120°) and is seen after turning to ψ = π
    assert (tr.surface_id == 1).any() and (tr.material_id[tr.surface_id == 1] == BLACK).all()
    assert not (tr.surface_id == 8).any()
    tr_back = LidarSensor().trace(Pose.from_xi([1.5, 1.5, math.pi]), scene, environment.CLEAR)
    assert (tr_back.surface_id == 8).any() and (tr_back.material_id[tr_back.surface_id == 8] == GREY).all()
    # power follows E7 with each surface's reflectivity
    rho = np.vectorize(lambda m: MATERIALS[m].rho_diffuse)(tr.material_id)
    sigma = extinction(environment.CLEAR.visibility_m, environment.CLEAR.wavelength)
    expect = lambertian_power(NZ.P0, tr.w, rho, tr.cos_inc, tr.r_path, NZ.A_rx, NZ.eta, sigma)
    np.testing.assert_allclose(tr.power, expect, rtol=1e-12)
    # every ray from many interior points of the L-shaped room hits within 10 m
    for xi in ([0.3, 0.3, 1.0], [5.7, 2.2, 3.0], [1.0, 3.7, -2.0], [4.4, 1.0, 0.5]):
        assert LidarSensor().trace(Pose.from_xi(xi), scene, environment.CLEAR).hit.all()


# ------------------------------------------
# test_post_before_wall
# ------------------------------------------
# A thin post hides the wall behind it, and the beams at its edges are mixed.
def test_post_before_wall():
    spec = scenes.post_before_wall()
    tr = _trace(spec)
    a = _angles(tr)
    t_post = 0.05 / 1.95
    tol = 1e-9
    m_post = np.abs(np.tan(a)) < t_post - tol
    m_wall = (np.abs(np.tan(a)) > t_post + tol) & (np.abs(np.tan(a)) < 0.5 - tol) & (np.abs(a) < math.pi / 2)
    assert m_post.any() and m_wall.any()
    assert (tr.surface_id[m_post] == 4).all() and (tr.object_id[m_post] == 1).all()   # front face x = 1.95
    np.testing.assert_allclose(tr.r_path[m_post], 1.95 / np.cos(a[m_post]), rtol=1e-12)
    assert (tr.surface_id[m_wall] == 0).all()
    np.testing.assert_allclose(tr.r_path[m_wall], 4 / np.cos(a[m_wall]), rtol=1e-12)
    # side faces of the post are self-occluded from the origin
    assert not np.isin(tr.surface_id, [1, 2, 3]).any()
    # the edge beams are mixed: some sub-rays on the post, others on the wall
    mixed = (tr.surface_id == 4).any(axis=1) & (tr.surface_id == 0).any(axis=1)
    assert mixed.sum() >= 2


# ------------------------------------------
# test_moving_box_scan_skew
# ------------------------------------------
# A moving box is seen where it is at each beam's fire time.
@pytest.mark.parametrize("v", [(1.0, 0.0), (5.0, 0.0)])
def test_moving_box_scan_skew(v):
    spec = scenes.moving_box(v=v)
    scene = Scene2D.from_spec(spec)
    tr = _trace(spec)
    tau = np.broadcast_to(tr.fire_time[:, None], tr.hit.shape)
    on_box = tr.object_id == 1
    assert on_box.any()
    p = tr.r_path[..., None] * tr.dir_world                 # hit points (origin sensor)
    shifted = p - np.asarray(v)[None, None, :] * tau[..., None]   # undo the skew displacement v·τ_i
    left = on_box & (tr.surface_id == 4 + 3)                # left face x = 1.75 + v_x τ
    top = on_box & (tr.surface_id == 4 + 2)                 # top face y = −0.75
    assert left.any() and top.any()
    np.testing.assert_allclose(p[left][:, 0], 1.75 + v[0] * tau[left], atol=1e-3)
    np.testing.assert_allclose(shifted[left][:, 0], 1.75, atol=1e-9)
    np.testing.assert_allclose(shifted[top][:, 1], -0.75, atol=1e-9)
    assert ((shifted[top][:, 0] >= 1.75 - 1e-9) & (shifted[top][:, 0] <= 2.25 + 1e-9)).all()
    # against a static box at its t = 0 pose the apparent displacement is exactly v·τ_i (1 mm tol)
    static = scenes.SceneSpec("static_box", np.concatenate([spec.segments, spec.movers[0].segments_local
                                                            + np.array([2.0, -1.0])]),
                              [WHITE] * 8, [0] * 4 + [1] * 4)
    tr0 = _trace(static)
    both = left & (tr0.surface_id == 7)
    dx = p[both][:, 0] - (tr0.r_path[..., None] * tr0.dir_world)[both][:, 0]
    np.testing.assert_allclose(dx, v[0] * tau[both], atol=1e-3)
    _assert_points_on_surfaces(tr, scene)


# ------------------------------------------
# test_mirror_room_specular_phantom
# ------------------------------------------
# With specular reflection on, a mirror adds a phantom return.
def test_mirror_room_specular_phantom():
    pat = Planar2DPattern()
    i = 341
    psi = -pat.beam_angle()[i]                              # beam i points along world +x
    pose = Pose.from_xi([1.0, 1.3, psi])
    spec = scenes.mirror_room()
    sensor = LidarSensor(SensorConfig(specular=True))
    tr = sensor.trace(pose, Scene2D.from_spec(spec), environment.CLEAR)
    K = tr.K
    kc = K // 2                                             # δ = 0 sub-ray
    np.testing.assert_allclose(tr.dir_world[i, kc], [1.0, 0.0], atol=1e-15)
    # direct: mirror hit at (3.3, 1.3), r₁ = 2.3, weak diffuse ρ_d
    assert tr.hit[i, kc] and tr.material_id[i, kc] == MIRROR and tr.surface_id[i, kc] == 4
    assert tr.r_path[i, kc] == pytest.approx(2.3, abs=1e-9)
    np.testing.assert_allclose(np.abs(tr.normal[i, kc]), [1 / math.sqrt(2)] * 2, atol=1e-12)
    sigma = extinction(20000.0, 785e-9)
    w = tr.w[i, kc]
    Pd = lambertian_power(NZ.P0, w, 0.02, 1 / math.sqrt(2), 2.3, NZ.A_rx, NZ.eta, sigma)
    assert tr.power[i, kc] == pytest.approx(Pd, rel=1e-9)
    # bounce: wall y = 3 (surface 2) after r₂ = 1.7, cos θ₂ = 1 → r_path = 4.0
    jb = K + kc
    assert tr.hit[i, jb] and tr.bounce[i, jb] and tr.sub_idx[i, jb] == kc
    assert tr.r_path[i, jb] == pytest.approx(4.0, abs=1e-9)
    assert tr.surface_id[i, jb] == 2 and tr.material_id[i, jb] == WHITE
    assert tr.cos_inc[i, jb] == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(tr.normal[i, jb], [0.0, -1.0], atol=1e-12)
    Pb = 0.95**2 * lambertian_power(NZ.P0, w, 0.9, 1.0, 4.0, NZ.A_rx, NZ.eta, sigma)
    assert tr.power[i, jb] == pytest.approx(Pb, rel=1e-9)
    # NOTE: the mirror_room docstring says (5.3, 1.3); sensor (1, 1.3) + 4.0 m along +x is (5.0, 1.3)
    phantom = pose.t + tr.r_path[i, jb] * tr.dir_world[i, kc]
    np.testing.assert_allclose(phantom, [5.0, 1.3], atol=1e-9)
    assert phantom[0] > 4.0                                  # outside the room
    _assert_finite_hits(tr)


# ------------------------------------------
# test_corridor
# ------------------------------------------
# Corridor: both walls at their hand-computed ranges.
def test_corridor():
    spec = scenes.corridor()
    tr = _trace(spec)
    a = _angles(tr)
    tol = 1e-9
    m_hit = (np.abs(np.tan(a)) > 1 / 50 + tol) | (np.abs(a) > math.pi / 2)
    m_miss = np.abs(np.tan(a)) < 1 / 50 - tol
    assert m_miss.any()
    assert not tr.hit[m_miss & (np.abs(a) < math.pi / 2)].any()
    assert tr.hit[m_hit].all()
    np.testing.assert_allclose(tr.r_path[m_hit], 1 / np.abs(np.sin(a[m_hit])), rtol=1e-12)
    assert (tr.surface_id[m_hit & (a > 0)] == 1).all() and (tr.surface_id[m_hit & (a < 0)] == 0).all()
    np.testing.assert_allclose(tr.normal[m_hit & (a > 0)], np.tile([0, -1.0], ((m_hit & (a > 0)).sum(), 1)),
                               atol=1e-15)


# ------------------------------------------
# test_trace_performance_room_asym
# ------------------------------------------
# Review budget: one 682 × 5 trace of room_asym in < 50 ms (spec §7 target 10 ms).
def test_trace_performance_room_asym():
    sensor = LidarSensor()
    scene = Scene2D.from_spec(scenes.room_asym())
    pose = Pose.from_xi([1.5, 1.5, 0.0])
    sensor.trace(pose, scene, environment.CLEAR)            # warm-up
    best = math.inf
    for _ in range(5):
        t0 = time.perf_counter()
        sensor.trace(pose, scene, environment.CLEAR)
        best = min(best, time.perf_counter() - t0)
    print(f"room_asym trace (K=5): {best * 1e3:.2f} ms")
    assert best < 0.050
