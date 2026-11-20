# ==========================================================
# tests/unit/test_scenes_data.py
# ==========================================================
# Unit tests of the canonical scene data.

import numpy as np
import pytest

from lidar_sim.geometry.base import Scene
from lidar_sim.sensor.scan_pattern import ScanPattern
from lidar_sim.world import scenes
from lidar_sim.world.materials import MATERIALS, Material


# ------------------------------------------
# test_canonical_scenes_valid
# ------------------------------------------
# Every canonical scene is well formed and documents its facts.
@pytest.mark.parametrize("name", sorted(scenes.CANONICAL))
def test_canonical_scenes_valid(name):
    s = scenes.CANONICAL[name]()
    assert s.segments.shape[1:] == (2, 2)
    assert s.facts.strip(), "every canonical scene documents its facts"
    assert set(s.material_id.tolist()) <= set(MATERIALS)


# ------------------------------------------
# test_room_asym_is_closed_polygon
# ------------------------------------------
# The asymmetric room is a closed polygon.
def test_room_asym_is_closed_polygon():
    s = scenes.room_asym()
    seg = s.segments
    np.testing.assert_array_equal(seg[1:, 0], seg[:-1, 1])
    np.testing.assert_array_equal(seg[0, 0], seg[-1, 1])


# ------------------------------------------
# test_moving_box_pose_at
# ------------------------------------------
# The moving box is where its velocity puts it.
def test_moving_box_pose_at():
    s = scenes.moving_box(p0=(2.0, -1.0), v=(0.0, 1.0))
    np.testing.assert_allclose(s.movers[0].pose_at(0.5).t, [2.0, -0.5])
    np.testing.assert_allclose(scenes.moving_box().movers[0].pose_at(1.0).t, [3.0, -1.0])  # default v=(1,0)


# ------------------------------------------
# test_scene_arrays_read_only
# ------------------------------------------
# Scene arrays are read-only.
@pytest.mark.parametrize("name", sorted(scenes.CANONICAL))
def test_scene_arrays_read_only(name):
    s = scenes.CANONICAL[name]()
    for arr in (s.segments, s.material_id, s.object_id):
        with pytest.raises(ValueError):
            arr.flat[0] = 0
    for m in s.movers:
        with pytest.raises(ValueError):
            m.segments_local.flat[0] = 0.0


# ------------------------------------------
# test_bad_segments_rejected
# ------------------------------------------
# Zero-length segments are rejected.
def test_bad_segments_rejected():
    with pytest.raises(ValueError):
        scenes.SceneSpec("bad", np.zeros((1, 2, 2)), [0], [0])


# ------------------------------------------
# test_material_validation_and_mirror_diffuse
# ------------------------------------------
# Material validation, and the mirror's small diffuse part.
def test_material_validation_and_mirror_diffuse():
    with pytest.raises(ValueError):
        Material("x", "lambertian", 1.2)
    assert MATERIALS[3].rho_diffuse == 0.02 and MATERIALS[0].rho_diffuse == 0.90


# ------------------------------------------
# test_protocols_are_runtime_checkable
# ------------------------------------------
# The scene and pattern protocols can be checked at run time.
def test_protocols_are_runtime_checkable():
    assert not isinstance(object(), Scene)
    assert not isinstance(object(), ScanPattern)
