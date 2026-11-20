# ==========================================================
# tests/unit/test_scene2d.py
# ==========================================================
# Unit tests: 2D ray–segment scene (E5), including grazing and endpoint rays.

import math

import numpy as np
import pytest

from lidar_sim.core.pose import Pose
from lidar_sim.geometry.base import Scene
from lidar_sim.geometry.scene2d import Scene2D
from lidar_sim.world import scenes
from lidar_sim.world.materials import MATERIALS, MIRROR, WHITE, Material
from lidar_sim.world.scenes import Mover, SceneSpec


# ------------------------------------------
# _spec
# ------------------------------------------
# A scene specification from raw segments.
def _spec(segs, mats=None, objs=None, **kw):
    segs = np.asarray(segs, float)
    M = len(segs)
    return SceneSpec("t", segs, mats if mats is not None else [WHITE] * M,
                     objs if objs is not None else list(range(M)), **kw)


# ------------------------------------------
# _cast
# ------------------------------------------
# Casts rays into a scene.
def _cast(scene, o, d, t=0.0):
    o = np.atleast_2d(np.asarray(o, float))
    d = np.atleast_2d(np.asarray(d, float))
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    return scene.intersect(o, d, np.full(len(o), t))


# ------------------------------------------
# test_protocol_and_dim
# ------------------------------------------
# The 2D scene meets the scene protocol.
def test_protocol_and_dim():
    s = Scene2D.from_spec(scenes.single_wall())
    assert isinstance(s, Scene) and s.dim == 2


# ------------------------------------------
# test_single_wall_o05
# ------------------------------------------
# A ray hits the single wall at the hand-computed point (O-05).
def test_single_wall_o05():
    h = _cast(Scene2D.from_spec(scenes.single_wall()), [0, 0], [1, 0])
    assert h.hit[0] and h.r[0] == pytest.approx(2.0, abs=1e-12)
    np.testing.assert_allclose(h.point[0], [2, 0], atol=1e-12)
    np.testing.assert_allclose(h.normal[0], [-1, 0], atol=1e-15)
    assert (h.surface_id[0], h.material_id[0], h.object_id[0]) == (0, WHITE, 0)


# ------------------------------------------
# test_nearest_parallel_endpoint_o06
# ------------------------------------------
# Nearest hits, parallel segments and endpoints (O-06).
def test_nearest_parallel_endpoint_o06():
    segs = [[[3, -1], [3, 1]],          # 0: behind
            [[2, -1], [2, 1]],          # 1: nearest
            [[0, 0.5], [5, 0.5]]]       # 2: parallel to the ray (never hit by +x ray at y=0)
    s = Scene2D.from_spec(_spec(segs))
    h = _cast(s, [0, 0], [1, 0])
    assert h.hit[0] and h.surface_id[0] == 1 and h.r[0] == pytest.approx(2.0)
    # parallel ray along y=0.5 below/at the segment line: no hit on segment 2 (collinear => D = 0)
    h = _cast(Scene2D.from_spec(_spec([[[1, 0.5], [5, 0.5]]])), [0, 0.5], [1, 0])
    assert not h.hit[0]
    # endpoint hit: segment x=2, y in [0, 1]; ray along +x hits a = (2, 0) exactly (u = 0)
    h = _cast(Scene2D.from_spec(_spec([[[2, 0], [2, 1]]])), [0, 0], [1, 0])
    assert h.hit[0] and h.r[0] == pytest.approx(2.0)


# ------------------------------------------
# test_grazing_and_endpoint_no_nan
# ------------------------------------------
# Parallel = no hit, endpoint = hit, and hit entries contain no NaN/inf.
def test_grazing_and_endpoint_no_nan():
    segs = [[[1, 0], [3, 0]],           # 0: collinear with the +x ray from the origin
            [[1, 1], [3, 1]],           # 1: parallel, offset
            [[4, -1], [4, 0]],          # 2: endpoint b = (4, 0) on the +x ray
            [[2, 2], [2, 3]],           # 3: endpoint a on the diagonal, occluded by endpoint (1, 1) of 1
            [[5, 5], [6, 6]]]           # 4: on the 45° line (collinear with the diagonal ray)
    s = Scene2D.from_spec(_spec(segs))
    o = np.zeros((5, 2))
    d = np.array([[1, 0], [-1, 0], [1, 1], [0, 1], [1, 1.0000001]])
    o[1] = [5, 1]                       # parallel ray running along segment 1 (collinear)
    h = _cast(s, o, d)
    assert h.hit[0] and h.surface_id[0] == 2 and h.r[0] == pytest.approx(4.0, abs=1e-12)
    assert not h.hit[1]                 # collinear along segment 1, nothing else on its line
    assert h.hit[2] and h.surface_id[2] == 1 and h.r[2] == pytest.approx(math.sqrt(2), abs=1e-12)
    assert not h.hit[3]
    for f in (h.r, h.point, h.normal):
        assert np.all(np.isfinite(f[h.hit]))
    np.testing.assert_allclose(np.linalg.norm(h.normal[h.hit], axis=1), 1.0, atol=1e-12)
    assert (h.surface_id[~h.hit] == -1).all() and (h.material_id[~h.hit] == -1).all()
    assert (h.object_id[~h.hit] == -1).all()


# ------------------------------------------
# test_shared_vertex_tie_lowest_id
# ------------------------------------------
# A ray through a shared polyline vertex hits both segments at equal r: lowest id wins.
def test_shared_vertex_tie_lowest_id():
    s = Scene2D.from_spec(scenes.corner())                  # vertex (3, 2) shared by ids 0 and 1
    h = _cast(s, [0, 0], [3, 2])
    assert h.hit[0] and h.surface_id[0] == 0 and h.r[0] == pytest.approx(math.sqrt(13), rel=1e-12)
    s2 = Scene2D.from_spec(scenes.room_asym())               # every vertex: ray from inside hits
    pts = s2.segments_at(0.0)[:, 0]
    o = np.tile([1.5, 1.5], (len(pts), 1))
    h = _cast(s2, o, pts - o)
    assert h.hit.all()
    np.testing.assert_allclose(h.point, pts, atol=1e-12)


# ------------------------------------------
# test_normals_face_ray_two_sided
# ------------------------------------------
# Normals face the incoming ray from either side.
def test_normals_face_ray_two_sided():
    s = Scene2D.from_spec(scenes.single_wall())
    h = _cast(s, [[0, 0], [4, 0]], [[1, 0], [-1, 0]])
    np.testing.assert_allclose(h.normal, [[-1, 0], [1, 0]], atol=1e-15)


# ------------------------------------------
# test_r_eps_and_behind
# ------------------------------------------
# Hits behind the origin or within the guard distance are ignored.
def test_r_eps_and_behind():
    s = Scene2D.from_spec(scenes.single_wall())
    h = _cast(s, [[2, 0], [3, 0], [2 - 1e-10, 0]], [[1, 0], [1, 0], [1, 0]])
    assert not h.hit.any()             # on the wall (r = 0), behind it, and within r_eps


# ------------------------------------------
# test_miss_and_empty_batch
# ------------------------------------------
# Misses and empty batches.
def test_miss_and_empty_batch():
    s = Scene2D.from_spec(scenes.single_wall())
    h = _cast(s, [0, 0], [-1, 0])
    assert not h.hit[0] and np.isnan(h.r[0]) and h.surface_id[0] == -1
    h0 = s.intersect(np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0))
    assert h0.n == 0 and h0.dim == 2


# ------------------------------------------
# test_input_validation
# ------------------------------------------
# Invalid ray inputs are rejected.
def test_input_validation():
    s = Scene2D.from_spec(scenes.single_wall())
    with pytest.raises(ValueError):
        s.intersect(np.zeros((2, 3)), np.ones((2, 3)), np.zeros(2))
    with pytest.raises(ValueError):
        s.intersect(np.zeros((2, 2)), np.ones((3, 2)), np.zeros(2))
    with pytest.raises(ValueError):
        s.intersect(np.zeros((2, 2)), np.ones((2, 2)), np.zeros(3))


# ------------------------------------------
# test_surface_ids_movers_after_static
# ------------------------------------------
# Moving objects get surface ids after the static ones.
def test_surface_ids_movers_after_static():
    spec = scenes.moving_box()
    s = Scene2D.from_spec(spec)
    assert s.n_surfaces == 8
    np.testing.assert_array_equal(s.material_ids, [WHITE] * 8)
    np.testing.assert_array_equal(s.object_ids, [0, 0, 0, 0, 1, 1, 1, 1])
    # box at t=0 centred (2, -1); ray from the origin to its left face centre (1.75, -1)
    h = _cast(s, [0, 0], [1.75, -1.0], t=0.0)
    assert h.surface_id[0] == 4 + 3 and h.object_id[0] == 1     # left face = 4th square edge
    assert h.r[0] == pytest.approx(math.hypot(1.75, 1.0), rel=1e-12)


# ------------------------------------------
# test_mover_posed_per_ray_time
# ------------------------------------------
# A moving object is placed at each ray's own time.
def test_mover_posed_per_ray_time():
    v = 2.0
    spec = scenes.moving_box(p0=(2.0, 0.0), v=(v, 0.0))
    s = Scene2D.from_spec(spec)
    t = np.array([0.0, 0.1, 0.25, 0.1])
    h = s.intersect(np.zeros((4, 2)), np.tile([1.0, 0.0], (4, 1)), t)
    np.testing.assert_allclose(h.r, 1.75 + v * t, atol=1e-12)


# ------------------------------------------
# test_rotating_mover
# ------------------------------------------
# A rotating object is hit where it is at each time.
def test_rotating_mover():
    seg = np.array([[[0.0, -1.0], [0.0, 1.0]]])               # body-frame segment along y
    mover = Mover(seg, [WHITE], 7, lambda t: Pose(np.array([[math.cos(t), -math.sin(t)],
                                                            [math.sin(t), math.cos(t)]]),
                                                   np.array([3.0, 0.0])))
    spec = SceneSpec("rot", np.zeros((0, 2, 2)), np.zeros(0, int), np.zeros(0, int), movers=(mover,))
    s = Scene2D.from_spec(spec)
    t = np.array([0.0, math.pi / 4])
    h = s.intersect(np.zeros((2, 2)), np.tile([1.0, 0.0], (2, 1)), t)
    np.testing.assert_allclose(h.r, [3.0, 3.0], atol=1e-12)
    np.testing.assert_allclose(h.normal[1], [-math.cos(math.pi / 4), -math.sin(math.pi / 4)], atol=1e-12)
    assert h.object_id.tolist() == [7, 7] and h.surface_id.tolist() == [0, 0]


# ------------------------------------------
# test_materials_merge
# ------------------------------------------
# Scene materials merge with the defaults.
def test_materials_merge():
    custom = Material("dull", "lambertian", 0.3)
    spec = _spec([[[2, -1], [2, 1]]], mats=[17], materials={17: custom})
    s = Scene2D.from_spec(spec)
    assert s.materials[17] is custom and s.materials[MIRROR] is MATERIALS[MIRROR]
    assert Scene2D.from_spec(scenes.single_wall()).materials == MATERIALS


# ------------------------------------------
# test_vectorized_matches_loop
# ------------------------------------------
# The vectorized caster matches a ray-by-ray loop.
def test_vectorized_matches_loop():
    rng = np.random.default_rng(0)
    s = Scene2D.from_spec(scenes.room_asym())
    o = rng.uniform([0.2, 0.2], [4.3, 3.8], (200, 2))
    a = rng.uniform(-math.pi, math.pi, 200)
    d = np.stack([np.cos(a), np.sin(a)], axis=1)
    h = s.intersect(o, d, np.zeros(200))
    for i in range(200):
        hi = s.intersect(o[i:i + 1], d[i:i + 1], np.zeros(1))
        assert hi.surface_id[0] == h.surface_id[i] and hi.r[0] == h.r[i]


# ------------------------------------------
# test_tie_never_reports_a_farther_range
# ------------------------------------------
# A farther segment inside the tie tolerance with a lower surface id may
# supply the ids/normal, but the reported range is always the nearest computed one (E5).
def test_tie_never_reports_a_farther_range():
    far, near = 2.0 + 2e-12, 2.0                      # within the 1e-12 relative tie window
    seg = np.array([[[far, -1.0], [far, 1.0]], [[near, -1.0], [near, 1.0]]])   # farther one has id 0
    sc = Scene2D(seg, np.array([0, 1]), np.array([0, 1]))
    h = sc.intersect(np.zeros((1, 2)), np.array([[1.0, 0.0]]), np.zeros(1))
    assert h.r[0] == near
