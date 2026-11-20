# ==========================================================
# tests/property/test_scene2d_props.py
# ==========================================================
# Property tests for E5: P-03 (hit lies on ray and segment), P-07 (occlusion / nearest hit).

import math

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from lidar_sim.geometry.scene2d import Scene2D
from lidar_sim.world.materials import WHITE
from lidar_sim.world.scenes import SceneSpec

SETTINGS = settings(max_examples=200, derandomize=True, deadline=None)
coord = st.floats(-10.0, 10.0, allow_nan=False)
angle = st.floats(-math.pi, math.pi, allow_nan=False)
point = st.tuples(coord, coord)


# ------------------------------------------
# _segment_ok
# ------------------------------------------
# A segment long enough to test.
def _segment_ok(a, b):
    return math.dist(a, b) > 1e-3


# ------------------------------------------
# _solve
# ------------------------------------------
# Independent reference: solve o + r d = a + u (b − a) with numpy.linalg.solve; None if degenerate.
def _solve(o, d, a, b):
    e = np.subtract(b, a)
    M = np.column_stack([d, -e])
    if abs(np.linalg.det(M)) < 1e-9:
        return None
    r, u = np.linalg.solve(M, np.subtract(a, o))
    return r, u


# ------------------------------------------
# test_p03_hit_on_ray_and_segment
# ------------------------------------------
# Every hit lies on both the ray and the segment.
@SETTINGS
@given(o=point, ang=angle, a=point, b=point)
def test_p03_hit_on_ray_and_segment(o, ang, a, b):
    if not _segment_ok(a, b):
        return
    scene = Scene2D.from_spec(SceneSpec("p03", np.array([[a, b]]), [WHITE], [0]))
    d = np.array([[math.cos(ang), math.sin(ang)]])
    h = scene.intersect(np.array([o]), d, np.zeros(1))
    ref = _solve(np.array(o), d[0], np.array(a), np.array(b))
    if h.hit[0]:
        p = h.point[0]
        scale = 1.0 + h.r[0]
        np.testing.assert_allclose(p, np.array(o) + h.r[0] * d[0], atol=1e-9 * scale)
        e = np.subtract(b, a)
        L = np.linalg.norm(e)
        dist_line = abs((p - a) @ np.array([-e[1], e[0]])) / L
        s = (p - a) @ e / L**2
        assert dist_line < 1e-9 * (scale + L + np.abs(a).max())
        assert -1e-9 <= s <= 1 + 1e-9
        assert h.r[0] > scene.r_eps
        assert abs(np.linalg.norm(h.normal[0]) - 1.0) < 1e-12
        assert h.normal[0] @ d[0] <= 1e-15
    elif ref is not None:
        r, u = ref
        # a clear (non-marginal) intersection must not be missed
        assert not (r > 1e-6 and 1e-6 < u < 1 - 1e-6), (r, u)


# ------------------------------------------
# test_p07_nearest_hit_occludes
# ------------------------------------------
# The reported hit is the nearest one.
@SETTINGS
@given(o=point, ang=angle, segs=st.lists(st.tuples(point, point), min_size=1, max_size=6))
def test_p07_nearest_hit_occludes(o, ang, segs):
    segs = [s for s in segs if _segment_ok(*s)]
    if not segs:
        return
    scene = Scene2D.from_spec(SceneSpec("p07", np.array(segs), [WHITE] * len(segs), list(range(len(segs)))))
    d = np.array([math.cos(ang), math.sin(ang)])
    h = scene.intersect(np.array([o]), d[None], np.zeros(1))
    refs = [_solve(np.array(o), d, np.array(a), np.array(b)) for a, b in segs]
    clear = [ref[0] for ref in refs if ref is not None and ref[0] > 1e-6 and 1e-6 < ref[1] < 1 - 1e-6]
    if clear:
        assert h.hit[0]
    if h.hit[0]:
        # no other surface is intersected strictly closer than the reported (direct) hit
        for k, ref in enumerate(refs):
            if ref is None or k == h.surface_id[0]:
                continue
            r, u = ref
            if 1e-6 < u < 1 - 1e-6 and r > 1e-6:
                assert r >= h.r[0] - 1e-6 * (1 + r), (k, r, h.r[0])


# ------------------------------------------
# test_p07_occluded_wall_never_direct
# ------------------------------------------
# Occluder spanning the whole ray cone in front of a wall: the wall never appears as a hit.
@SETTINGS
@given(y_occ=st.floats(-0.4, 0.4), x_occ=st.floats(0.5, 3.5), ang=st.floats(-0.2, 0.2))
def test_p07_occluded_wall_never_direct(y_occ, x_occ, ang):
    segs = np.array([[[4.0, -5.0], [4.0, 5.0]], [[x_occ, y_occ - 1.0], [x_occ, y_occ + 1.0]]])
    scene = Scene2D.from_spec(SceneSpec("occ", segs, [WHITE, WHITE], [0, 1]))
    d = np.array([[math.cos(ang), math.sin(ang)]])
    h = scene.intersect(np.zeros((1, 2)), d, np.zeros(1))
    if abs(x_occ * math.tan(ang) - y_occ) < 1.0 - 1e-9:
        assert h.hit[0] and h.surface_id[0] == 1
