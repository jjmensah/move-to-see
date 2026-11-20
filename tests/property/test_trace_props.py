# ==========================================================
# tests/property/test_trace_props.py
# ==========================================================
# Property tests on LidarSensor.trace: P-01 (rigid-motion invariance), P-02 (rotation of dir_world),
# P-10 (K = 1 equals an ideal ray caster).

import math

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from lidar_sim.core.config import SensorConfig
from lidar_sim.core.pose import Pose, rot2
from lidar_sim.geometry.scene2d import Scene2D
from lidar_sim.sensor.lidar import LidarSensor
from lidar_sim.sensor.scan_pattern import Planar2DPattern
from lidar_sim.world import environment, scenes
from lidar_sim.world.scenes import SceneSpec

SETTINGS = settings(max_examples=200, derandomize=True, deadline=None)
CFG = SensorConfig(n_beams=61, n_subrays=3)
CFG1 = SensorConfig(n_beams=121, n_subrays=1)
ROOM = scenes.room_asym()
# interior of the L-shaped room (excludes the cut-out x > 4.5, y > 2.5), kept 0.2 m from the walls
xin = st.floats(0.2, 4.3)
yin = st.floats(0.2, 3.8)
yaw = st.floats(-math.pi, math.pi)


# ------------------------------------------
# _transform_spec
# ------------------------------------------
# A scene moved by a rigid transform.
def _transform_spec(spec: SceneSpec, g: Pose) -> SceneSpec:
    return SceneSpec(spec.name, g.apply(spec.segments), spec.material_id, spec.object_id)


# ------------------------------------------
# test_p01_rigid_motion_invariance
# ------------------------------------------
# Moving the scene and the sensor together changes nothing.
@SETTINGS
@given(x=xin, y=yin, psi=yaw, gx=st.floats(-20, 20), gy=st.floats(-20, 20), gpsi=yaw)
def test_p01_rigid_motion_invariance(x, y, psi, gx, gy, gpsi):
    sensor = LidarSensor(CFG)
    pose = Pose.from_xi([x, y, psi])
    g = Pose.from_xi([gx, gy, gpsi])
    tr = sensor.trace(pose, Scene2D.from_spec(ROOM), environment.CLEAR)
    tr_g = sensor.trace(g.compose(pose), Scene2D.from_spec(_transform_spec(ROOM, g)), environment.CLEAR)
    assert tr.hit.all() and tr_g.hit.all()
    np.testing.assert_allclose(tr_g.r_path, tr.r_path, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(tr_g.cos_inc, tr.cos_inc, atol=1e-7)
    np.testing.assert_allclose(tr_g.power, tr.power, rtol=1e-6)
    np.testing.assert_allclose(tr_g.dir_world, g.rotate(tr.dir_world), atol=1e-12)


# ------------------------------------------
# test_p02_rotation_rotates_dir_world
# ------------------------------------------
# Rotating the sensor rotates its beam directions.
@SETTINGS
@given(x=xin, y=yin, psi0=yaw, dpsi=yaw)
def test_p02_rotation_rotates_dir_world(x, y, psi0, dpsi):
    sensor = LidarSensor(CFG)
    scene = Scene2D.from_spec(ROOM)
    a = sensor.trace(Pose.from_xi([x, y, psi0]), scene, environment.CLEAR)
    b = sensor.trace(Pose.from_xi([x, y, psi0 + dpsi]), scene, environment.CLEAR)
    np.testing.assert_allclose(b.dir_world, a.dir_world @ rot2(dpsi).T, atol=1e-12)
    # spacing unchanged: consecutive beam-axis dot products identical
    da = np.sum(a.dir_world[1:, 1] * a.dir_world[:-1, 1], axis=-1)
    db = np.sum(b.dir_world[1:, 1] * b.dir_world[:-1, 1], axis=-1)
    np.testing.assert_allclose(db, da, atol=1e-12)


# ------------------------------------------
# _ideal_ranges
# ------------------------------------------
# Independent brute-force caster: per ray, per segment, np.linalg.solve of o + r d = a + u e.
def _ideal_ranges(o, dirs, segs):
    out = np.full(len(dirs), np.inf)
    for i, d in enumerate(dirs):
        for a, b in segs:
            e = b - a
            M = np.column_stack([d, -e])
            if abs(np.linalg.det(M)) < 1e-12:
                continue
            r, u = np.linalg.solve(M, a - o)
            if r > 1e-9 and -1e-12 <= u <= 1 + 1e-12:
                out[i] = min(out[i], r)
    return out


# ------------------------------------------
# test_p10_k1_equals_ideal_ray_caster
# ------------------------------------------
# One sub-ray per beam reproduces an ideal ray caster.
@settings(max_examples=40, derandomize=True, deadline=None)
@given(x=xin, y=yin, psi=yaw)
def test_p10_k1_equals_ideal_ray_caster(x, y, psi):
    pose = Pose.from_xi([x, y, psi])
    tr = LidarSensor(CFG1).trace(pose, Scene2D.from_spec(ROOM), environment.CLEAR)
    np.testing.assert_array_equal(tr.w, 1.0)
    dirs = pose.rotate(Planar2DPattern(CFG1).beam_dirs_sensor())
    np.testing.assert_allclose(tr.dir_world[:, 0], dirs, atol=1e-15)
    ref = _ideal_ranges(pose.t, dirs, ROOM.segments)
    np.testing.assert_allclose(tr.r_path[:, 0], ref, rtol=1e-12, atol=1e-12)
