# ==========================================================
# tests/mts_a_oracles/test_scene3d_oracles.py
# ==========================================================
# T-G1/T-G2: F2/E5 hand oracles and independent face-plane enumeration.

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from lidar_sim.geometry.scene3d import Scene3D, ray_box_batch


# ------------------------------------------
# face_oracle
# ------------------------------------------
# Enumerate six bounded planes, not slabs; choose first crossing beyond guard.
#
# For continuous random inputs ties have probability zero. Explicit exact ties
# are covered separately, including the F2 exit-face tie convention.
def face_oracle(origins, dirs, lo, hi, eps):
    distances = np.full((len(lo), len(origins)), np.inf)
    faces = np.full(distances.shape, -1, dtype=int)
    for m, (low, high) in enumerate(zip(lo, hi, strict=True)):
        for face in range(6):
            axis = face // 2
            moving = dirs[:, axis] != 0
            r = np.full(len(origins), np.inf)
            r[moving] = ((high if face % 2 else low)[axis]
                         - origins[moving, axis]) / dirs[moving, axis]
            valid = moving & (r > eps)
            for other in range(3):
                if other != axis:
                    coordinate = origins[:, other] + np.where(moving, r, 0) * dirs[:, other]
                    valid &= (coordinate >= low[other]) & (coordinate <= high[other])
            take = valid & (r < distances[m])
            distances[m, take], faces[m, take] = r[take], face
    return distances, faces


# ------------------------------------------
# assert_record
# ------------------------------------------
# HitBatch promises; miss coordinates/range are intentionally unspecified.
def assert_record(h, o, d):
    assert h.hit.dtype == bool
    assert h.r.shape == (len(o),)
    assert h.point.shape == h.normal.shape == (len(o), 3)
    assert np.all(np.isfinite(h.r[h.hit])) and np.all(h.r[h.hit] > 0)
    np.testing.assert_allclose(np.linalg.norm(h.normal[h.hit], axis=1), 1, atol=1e-12)
    assert np.all(np.sum(h.normal[h.hit] * d[h.hit], axis=1) <= 1e-12)
    np.testing.assert_allclose(h.point[h.hit], o[h.hit] + h.r[h.hit, None] * d[h.hit], atol=1e-12)
    for name in ('surface_id', 'material_id', 'object_id'):
        ids = getattr(h, name)
        assert ids.shape == (len(o),) and np.issubdtype(ids.dtype, np.integer)
        assert np.all(ids[~h.hit] == -1) and np.all(ids[h.hit] >= 0)


# ------------------------------------------
# test_tg1_six_axis_faces
# ------------------------------------------
# A ray along each axis hits the expected box face.
@pytest.mark.parametrize('axis', range(3))
@pytest.mark.parametrize('sign', [-1, 1])
def test_tg1_six_axis_faces(axis, sign):
    o = np.zeros((1, 3))
    o[0, axis] = -3 * sign
    d = np.zeros((1, 3))
    d[0, axis] = sign
    scene = Scene3D([[-1]*3], [[1]*3], [7], [19])
    r, f = ray_box_batch(o, d, scene.lo, scene.hi)
    face = 2 * axis + (sign < 0)
    np.testing.assert_array_equal(r, [[2]])
    np.testing.assert_array_equal(f, [[face]])
    h = scene.intersect(o, d)
    assert h.surface_id.tolist() == [face]
    assert h.material_id.tolist() == [7] and h.object_id.tolist() == [19]
    np.testing.assert_array_equal(h.normal, -d)
    assert_record(h, o, d)


# ------------------------------------------
# test_tg1_guard_parallel_oblique_and_ties
# ------------------------------------------
# Guard distance, parallel and oblique rays and face ties.
@pytest.mark.parametrize('origin,direction,distance,face', [
    ((-2, -2, -2), (1, 1, 1), np.sqrt(3), 0),  # entry axis tie
    ((0, 0, 0), (1, 1, 1), np.sqrt(3), 1),     # exit axis tie
    ((0, 0, 0), (0, -1, 0), 1, 2),            # inside -> exit, normal flipped
    ((-1.0005, 0, 0), (1, 0, 0), 2.0005, 1), # entry inside guard
    ((-1, 0, 0), (1, 0, 0), 2, 1),            # origin on entry
    ((-3, 1, 0), (1, 0, 0), 2, 0),            # grazing parallel slab boundary
    ((-3, 1.01, 0), (1, 0, 0), np.inf, -1),    # parallel outside slab
    ((-3, 0, 0), (1, 0, 0), 2, 0),            # parallel inside two slabs
    ((-3, 0, 0), (-1, 0, 0), np.inf, -1),      # behind ray
    ((-2, -0.5, 0), (1, 0.5, 0), np.sqrt(1.25), 0), # oblique
    ((1, 0, 0), (1, 0, 0), np.inf, -1),        # exit at zero fails guard
    ((-100, 0, 0), (1, 0, 0), 99, 0),          # no upper gate
])
def test_tg1_guard_parallel_oblique_and_ties(origin, direction, distance, face):
    o = np.array([origin], float)
    d = np.array([direction], float)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    scene = Scene3D([[-1]*3], [[1]*3], [4], [8], r_eps=0.001)
    r, f = ray_box_batch(o, d, scene.lo, scene.hi, 0.001)
    np.testing.assert_allclose(r, [[distance]], rtol=1e-12, atol=0)
    assert f.tolist() == [[face]]
    h = scene.intersect(o, d)
    assert h.hit.tolist() == [face >= 0]
    assert h.surface_id.tolist() == [face]
    if face >= 0:
        np.testing.assert_allclose(h.r, [distance], rtol=1e-12)
    assert_record(h, o, d)


# ------------------------------------------
# test_tg1_box_ground_ties_and_nearest
# ------------------------------------------
# Ties between boxes and the ground resolve to the nearest hit.
def test_tg1_box_ground_ties_and_nearest():
    # Both boxes have top z=0: at r=2 both boxes and ground tie.
    scene = Scene3D([[-1, -1, -1]]*2, [[1, 1, 0]]*2, [4, 5], [8, 9],
                    ground_material=6, ground_object=10)
    o, d = np.array([[0., 0, 2]]), np.array([[0., 0, -1]])
    h = scene.intersect(o, d)
    assert h.surface_id.tolist() == [5]
    assert h.object_id.tolist() == [8]
    np.testing.assert_array_equal(h.r, [2])
    # Index 1 now nearer: distance, not box order, determines winner.
    moved = scene.with_box(1, [-1, -1, 0.5], [1, 1, 1])
    h = moved.intersect(o, d)
    assert h.surface_id.tolist() == [11] and h.object_id.tolist() == [9]
    np.testing.assert_array_equal(h.r, [1])
    assert scene.intersect(o, d).object_id.tolist() == [8]
    removed = moved.without_box(0).intersect(o, d)
    assert removed.surface_id.tolist() == [5] and removed.object_id.tolist() == [9]


# ------------------------------------------
# test_tg1_ground_only_or_empty
# ------------------------------------------
# Rays in a scene with only a ground plane, or nothing at all.
@pytest.mark.parametrize('ground', [None, 6])
def test_tg1_ground_only_or_empty(ground):
    o = np.array([[0, 0, 2], [0, 0, 2], [0, 0, 2], [0, 0, 0], [0, 0, -1], [0, 0, 0.0005]])
    d = np.array([[0, 0, -1], [0, 0, 1], [1, 0, 0], [0, 0, -1], [0, 0, 1], [0, 0, -1]])
    h = Scene3D([], [], [], [], ground_material=ground, ground_object=10, r_eps=0.001).intersect(o, d)
    assert h.hit.tolist() == [ground is not None, False, False, False, False, False]
    if ground is not None:
        assert (h.r[0], h.surface_id[0], h.material_id[0], h.object_id[0]) == (2, 0, 6, 10)
    assert_record(h, o, d)


# ------------------------------------------
# test_tg1_random_faces_against_brute_force
# ------------------------------------------
# 10,000 rays × 17 random boxes = 170,000 independent ray/box pairs.
def test_tg1_random_faces_against_brute_force(rng):
    o = rng.uniform(-6, 6, (10_000, 3))
    d = rng.normal(size=o.shape)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    lo = rng.uniform(-4, 3, (17, 3))
    hi = lo + rng.uniform(0.2, 3, (17, 3))
    expected_r, expected_f = face_oracle(o, d, lo, hi, 1e-9)
    r, f = ray_box_batch(o, d, lo, hi)
    np.testing.assert_allclose(r, expected_r, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(f, expected_f)
    h = Scene3D(lo, hi, np.arange(17)+10, np.arange(17)+40).intersect(o, d)
    best = np.argmin(expected_r, axis=0)
    col = np.arange(len(o))
    valid = np.isfinite(expected_r[best, col])
    np.testing.assert_array_equal(h.hit, valid)
    np.testing.assert_allclose(h.r[valid], expected_r[best[valid], col[valid]], rtol=1e-12)
    np.testing.assert_array_equal(h.surface_id[valid], 6*best[valid] + expected_f[best[valid], col[valid]])
    np.testing.assert_array_equal(h.material_id[valid], best[valid]+10)
    np.testing.assert_array_equal(h.object_id[valid], best[valid]+40)
    assert_record(h, o, d)


vector = st.tuples(*(st.integers(-20, 20) for _ in range(3)))


# ------------------------------------------
# test_tg2_hitbatch_properties
# ------------------------------------------
# Random rays hit the nearest box face or the ground, as an independent face oracle predicts.
@given(origin=vector, direction=vector.filter(lambda v: any(v)),
       corners=st.lists(vector, min_size=1, max_size=5), ground=st.booleans())
@settings(max_examples=150, deadline=None, derandomize=True)
def test_tg2_hitbatch_properties(origin, direction, corners, ground):
    # Integer origins and half-integer faces avoid accidental face-edge roundoff.
    o, d = np.array([origin], float), np.array([direction], float)
    d /= np.linalg.norm(d)
    lo = np.array(corners, float) - 0.5
    hi = lo + 3
    n = len(lo)
    scene = Scene3D(lo, hi, np.arange(n)+2, np.arange(n)+12,
                    ground_material=99 if ground else None, ground_object=100)
    h = scene.intersect(o, d)
    assert_record(h, o, d)
    rr, ff = face_oracle(o, d, lo, hi, 1e-9)
    candidates = [(rr[m, 0], 6*m+ff[m, 0], m+2, m+12) for m in range(n) if np.isfinite(rr[m, 0])]
    if ground and o[0, 2] > 0 and d[0, 2] < 0:
        candidates.append((-o[0, 2]/d[0, 2], 6*n, 99, 100))
    assert h.hit[0] == bool(candidates)
    if candidates:
        r, sid, mat, obj = min(candidates)
        np.testing.assert_allclose(h.r[0], r, rtol=1e-12)
        assert (h.surface_id[0], h.material_id[0], h.object_id[0]) == (sid, mat, obj)


# ------------------------------------------
# test_tg1_entry_exactly_guard_and_ground_nearer
# ------------------------------------------
# An entry exactly at the guard distance, and a ground hit nearer than a box.
def test_tg1_entry_exactly_guard_and_ground_nearer():
    # Binary-exact r_eps removes rounding ambiguity at the strict guard.
    scene = Scene3D([[1, -1, -1]], [[2, 1, 1]], [4], [8], r_eps=0.125)
    o, d = np.array([[0.875, 0, 0]]), np.array([[1., 0, 0]])
    r, f = ray_box_batch(o, d, scene.lo, scene.hi, 0.125)
    np.testing.assert_array_equal(r, [[1.125]])
    np.testing.assert_array_equal(f, [[1]])
    h = scene.intersect(o, d)
    np.testing.assert_array_equal(h.r, [1.125])
    assert h.surface_id.tolist() == [1]
    # Ground beats a farther box; its id includes all box faces.
    scene = Scene3D([[-1, -1, -3]], [[1, 1, -2]], [4], [8], ground_material=6, ground_object=10)
    h = scene.intersect([[0, 0, 2]], [[0, 0, -1]])
    assert (h.r[0], h.surface_id[0], h.object_id[0], h.material_id[0]) == (2, 6, 10, 6)
