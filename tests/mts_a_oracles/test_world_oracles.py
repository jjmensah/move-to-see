# ==========================================================
# tests/mts_a_oracles/test_world_oracles.py
# ==========================================================
# T-R3 and world sanity: final-spec §3.3, §4.2–4.3, F8.

from itertools import product

import numpy as np
import pytest

from lidar_sim.core.config import EnvConfig
from lidar_sim.core.pose import Pose
from lidar_sim.world.materials import BLACK, GREY, WHITE
from mts.config import RASTER, SENSOR, RobotConfig, make_sensor
from mts.world import GROUND_OBJECT, TARGET_OBJECT, yard

# Hand inflation by 0.30 + 0.05 = 0.35; wall ordering is not prescribed.
KEEPOUTS = np.array([
    [-0.35, 0.45, -0.35, 8.35], [10.55, 11.35, -0.35, 8.35],
    [-0.35, 11.35, -0.35, 0.45], [-0.35, 11.35, 7.55, 8.35],
    [2.85, 4.75, 2.95, 5.05], [3.05, 4.55, 3.15, 4.75],
    [4.05, 5.65, 3.25, 4.95], [6.40, 9.20, 2.20, 5.80],
])


# ------------------------------------------
# sorted_rows
# ------------------------------------------
# Rows sorted for an order-free comparison.
def sorted_rows(a):
    return np.array(sorted(map(tuple, a)))


# ------------------------------------------
# test_tr3_keepouts_by_hand
# ------------------------------------------
# The keep-out rectangles match the hand-computed ones.
def test_tr3_keepouts_by_hand():
    np.testing.assert_allclose(sorted_rows(yard().keepouts(RobotConfig())), sorted_rows(KEEPOUTS),
                               rtol=0, atol=2e-15)


# ------------------------------------------
# test_tr3_clearance_and_closed_boundaries
# ------------------------------------------
# Clearance and feasibility at chosen points, boundaries included.
@pytest.mark.parametrize('point,clearance,feasible', [
    ((1., 4.), 0.55, True), ((0.45, 4.), 0., False), ((10.55, 4.), 0., False),
    ((5., 0.45), 0., False), ((5., 7.55), 0., False),
    ((6.4, 4.), 0., False), ((9.2, 4.), 0., False),
    ((8., 2.2), 0., False), ((8., 5.8), 0., False),
    ((6.4, 2.2), 0., False), ((9.2, 5.8), 0., False),
    ((3., 4.), 0., False), ((8., 4.), 0., False),
    ((6., 2.), np.sqrt(0.4**2+0.2**2), True),
    ((6.4-1e-6, 4.), 1e-6, True), ((6.4+1e-6, 4.), 0., False),
])
def test_tr3_clearance_and_closed_boundaries(point, clearance, feasible):
    world, cfg = yard(), RobotConfig()
    np.testing.assert_allclose(world.clearance(point, cfg), clearance, rtol=1e-9, atol=2e-15)
    assert bool(world.feasible(point, cfg)) is feasible


# ------------------------------------------
# test_tr3_clearance_batch_independent_rectangles
# ------------------------------------------
# Batched clearance matches an independent rectangle distance.
def test_tr3_clearance_batch_independent_rectangles(rng):
    xy = rng.uniform([0, 0], [11, 8], (7, 11, 2))
    expected = np.full((7, 11), np.inf)
    for x0, x1, y0, y1 in KEEPOUTS:
        dx = np.maximum(np.maximum(x0-xy[..., 0], xy[..., 0]-x1), 0)
        dy = np.maximum(np.maximum(y0-xy[..., 1], xy[..., 1]-y1), 0)
        expected = np.minimum(expected, np.hypot(dx, dy))
    world, cfg = yard(), RobotConfig()
    np.testing.assert_allclose(world.clearance(xy, cfg), expected, atol=2e-15, rtol=1e-12)
    np.testing.assert_array_equal(world.feasible(xy, cfg), expected > 0)


# ------------------------------------------
# test_world_f8_closed_rough_patch
# ------------------------------------------
# The rough patch caps the speed, boundary included (F8).
@pytest.mark.parametrize('xy,expected', [
    ((4., 2.), 0.3), ((3.6, 1.), 0.3), ((6.2, 2.6), 0.3),
    ((3.6, 2.), 0.3), ((6.2, 2.), 0.3), ((5., 1.), 0.3), ((5., 2.6), 0.3),
    ((3.6-1e-9, 2.), 1.), ((6.2+1e-9, 2.), 1.),
    ((5., 1.-1e-9), 1.), ((5., 2.6+1e-9), 1.),
])
def test_world_f8_closed_rough_patch(xy, expected):
    world, cfg = yard(), RobotConfig()
    assert float(world.v_lim(xy, cfg)) == expected
    assert bool(world.in_rough(xy)) == (expected == 0.3)
    assert float(yard(rough=False).v_lim(xy, cfg)) == 1.


# ------------------------------------------
# test_world_dimensions_envelope_and_target_boxes
# ------------------------------------------
# The yard's size, start, prior region and target box.
def test_world_dimensions_envelope_and_target_boxes():
    world = yard()
    assert world.arena == (0, 11, 0, 8) and world.n_boxes == 7
    np.testing.assert_array_equal(world.start, [1, 4, 0, 0])
    np.testing.assert_array_equal(world.prior_lo, [7, 2.8, 0.5])
    np.testing.assert_array_equal(world.prior_hi, [8.6, 5.2, 1.1])
    np.testing.assert_array_equal(world.target_half, [0.25]*3)
    np.testing.assert_allclose(world.envelope(), [[6.75, 2.55, 0.25], [8.85, 5.45, 1.35]], atol=1e-15)
    q = np.array([[7., 2.8, 0.5], [8.6, 5.2, 1.1]])
    lo, hi = world.target_box(q)
    np.testing.assert_allclose(lo, [[6.75, 2.55, 0.25], [8.35, 4.95, 0.85]], atol=1e-15)
    np.testing.assert_allclose(hi, [[7.25, 3.05, 0.75], [8.85, 5.45, 1.35]], atol=1e-15)
    # Check the physical boxes as an unordered collection, independent of wall index order.
    expected = [
        [0, 0, 0, 0.1, 8, 1.5], [10.9, 0, 0, 11, 8, 1.5],
        [0, 0, 0, 11, 0.1, 1.5], [0, 7.9, 0, 11, 8, 1.5],
        [3.2, 3.3, 0, 4.4, 4.7, 1.2], [3.4, 3.5, 1.2, 4.2, 4.4, 1.9],
        [4.4, 3.6, 0, 5.3, 4.6, 0.85],
    ]
    np.testing.assert_allclose(sorted_rows(np.column_stack([world.box_lo, world.box_hi])),
                               sorted_rows(expected), atol=1e-15)
    assert world.ground_material == GREY and world.target_material == BLACK
    assert sorted(world.box_material.tolist()) == sorted([GREY]*4+[WHITE]*3)


# ------------------------------------------
# test_world_scene_ids_and_hits
# ------------------------------------------
# Object ids of the background and truth scenes, and what rays hit.
def test_world_scene_ids_and_hits():
    world = yard()
    background, truth = world.background_scene(), world.truth_scene([8., 4., 0.8])
    assert GROUND_OBJECT == 0 and TARGET_OBJECT == 99
    np.testing.assert_array_equal(background.object_id, np.arange(1, 8))
    np.testing.assert_array_equal(truth.object_id, [1, 2, 3, 4, 5, 6, 7, 99])
    np.testing.assert_array_equal(truth.lo[:-1], background.lo)
    np.testing.assert_array_equal(truth.hi[:-1], background.hi)
    np.testing.assert_allclose(truth.lo[-1], [7.75, 3.75, 0.55], atol=1e-15)
    np.testing.assert_allclose(truth.hi[-1], [8.25, 4.25, 1.05], atol=1e-15)
    assert truth.material_id[-1] == BLACK
    for scene, sid in [(background, 42), (truth, 48)]:
        h = scene.intersect([[1, 4, 0.4]], [[0, 0, -1]])
        assert (h.object_id[0], h.surface_id[0], h.material_id[0]) == (0, sid, GREY)
        np.testing.assert_allclose(h.r, [0.4], atol=1e-15)
    h = truth.intersect([[7, 4, 0.8]], [[1, 0, 0]])
    assert (h.object_id[0], h.surface_id[0]) == (99, 42)
    np.testing.assert_allclose(h.r, [0.75], atol=1e-15)


# Corners, edge midpoints, and face centres of the 3D prior boundary.
BOUNDARY_CENTRES = [q for q in product([7., 7.8, 8.6], [2.8, 4., 5.2], [0.5, 0.8, 1.1])
                    if q != (7.8, 4., 0.8)]


# ------------------------------------------
# test_tr3_target_boundary_disc_overlap_never_feasible
# ------------------------------------------
# No feasible robot position overlaps the target.
@pytest.mark.parametrize('q', BOUNDARY_CENTRES)
def test_tr3_target_boundary_disc_overlap_never_feasible(q):
    world, cfg = yard(), RobotConfig()
    # Dense grid in a circumscribed square plus exact radius offsets from all
    # four faces. Distance-to-footprint oracle is independent of world helpers.
    offsets = np.linspace(-0.55, 0.55, 45)
    xy = np.array(list(product(offsets, offsets))) + np.array(q[:2])
    face_points = [[sx, sy] for sx in [-0.55, 0.55] for sy in [-0.25, 0, 0.25]]
    face_points += [[sy, sx] for sx in [-0.55, 0.55] for sy in [-0.25, 0, 0.25]]
    xy = np.vstack([xy, np.array(face_points)+q[:2]])
    delta = np.maximum(np.abs(xy-np.array(q[:2]))-0.25, 0)
    overlapping = np.linalg.norm(delta, axis=1) <= 0.30+1e-14
    assert overlapping.sum() > 100
    assert not world.feasible(xy[overlapping], cfg).any()


# ------------------------------------------
# test_world_default_robot_and_sensor_spec
# ------------------------------------------
# Default robot and sensor settings match the specification.
def test_world_default_robot_and_sensor_spec():
    cfg = RobotConfig()
    assert (cfg.radius, cfg.margin, cfg.h_s, cfg.v_max, cfg.v_rough) == (0.30, 0.05, 0.40, 1., 0.3)
    assert (RASTER.n_az, RASTER.n_el, RASTER.n_beams) == (121, 11, 1331)
    np.testing.assert_allclose([RASTER.fov, RASTER.el_min, RASTER.el_max],
                               np.deg2rad([240, -10, 40]), atol=1e-15)
    assert (RASTER.t_scan, RASTER.theta0) == (0.1, 0.005)
    assert (SENSOR.n_beams, SENSOR.n_subrays, SENSOR.r_min, SENSOR.r_max) == (1331, 9, 0.02, 5.6)
    trace = make_sensor().trace(
        Pose.from_xyz_yaw(1, 4, 0.4, 0), yard().background_scene(), EnvConfig())
    assert trace.dir_world.shape == (1331, 9, 3)
    np.testing.assert_allclose(trace.fire_time, np.arange(1331)*0.1/1331, atol=1e-15)
