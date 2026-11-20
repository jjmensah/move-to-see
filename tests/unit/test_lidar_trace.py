# ==========================================================
# tests/unit/test_lidar_trace.py
# ==========================================================
# Unit tests: LidarSensor.trace (E1–E9) record assembly, poses, materials, environment, specular slots.

import math

import numpy as np
import pytest

from lidar_sim.core.config import EnvConfig, NoiseConfig, SensorConfig
from lidar_sim.core.pose import Pose
from lidar_sim.core.types import HitBatch, RadiometricScan
from lidar_sim.geometry.scene2d import Scene2D
from lidar_sim.physics.atmosphere import extinction
from lidar_sim.physics.radiometry import lambertian_power
from lidar_sim.sensor.lidar import LidarSensor
from lidar_sim.sensor.scan_pattern import Planar2DPattern
from lidar_sim.world import environment, scenes
from lidar_sim.world.materials import MATERIALS, WHITE, Material
from lidar_sim.world.targets import constant_velocity

NZ = NoiseConfig()


# ------------------------------------------
# test_trace_shapes_and_record
# ------------------------------------------
# Trace shapes and the record type.
def test_trace_shapes_and_record():
    sensor = LidarSensor()
    tr = sensor.trace(Pose.identity(2), Scene2D.from_spec(scenes.room_asym()), environment.CLEAR)
    assert isinstance(tr, RadiometricScan)
    assert (tr.B, tr.K, tr.J, tr.dim) == (682, 5, 5, 2)
    np.testing.assert_array_equal(tr.fire_time, Planar2DPattern().fire_times())
    assert not tr.bounce.any() and tr.P_amb == 0.0


# ------------------------------------------
# test_single_wall_power_matches_e7
# ------------------------------------------
# Return power from a single wall matches E7.
def test_single_wall_power_matches_e7():
    sensor = LidarSensor(SensorConfig(n_subrays=1))
    tr = sensor.trace(Pose.identity(2), Scene2D.from_spec(scenes.single_wall()), environment.FOG_LIGHT)
    a = Planar2DPattern().beam_angle()
    hitmask = np.abs(np.tan(a)) <= 0.5 - 1e-9
    np.testing.assert_array_equal(tr.hit[:, 0] & (np.abs(a) < math.pi / 2), hitmask & (np.abs(a) < math.pi / 2))
    sel = hitmask & (np.abs(a) < math.pi / 2)
    r = 2.0 / np.cos(a[sel])
    np.testing.assert_allclose(tr.r_path[sel, 0], r, rtol=1e-12)
    np.testing.assert_allclose(tr.cos_inc[sel, 0], np.cos(a[sel]), rtol=1e-12)
    sigma = extinction(200.0, 785e-9)
    expect = lambertian_power(NZ.P0, 1.0, 0.9, np.cos(a[sel]), r, NZ.A_rx, NZ.eta, sigma)
    np.testing.assert_allclose(tr.power[sel, 0], expect, rtol=1e-12)
    assert (tr.power[~tr.hit] == 0).all() and np.isnan(tr.r_path[~tr.hit]).all()


# ------------------------------------------
# test_ambient_and_fog_from_env
# ------------------------------------------
# Ambient light and fog come from the environment.
def test_ambient_and_fog_from_env():
    sensor = LidarSensor(SensorConfig(n_subrays=1))
    scene = Scene2D.from_spec(scenes.single_wall())
    clear = sensor.trace(Pose.identity(2), scene, environment.CLEAR)
    sunny = sensor.trace(Pose.identity(2), scene, environment.sunlight(0.5))
    fog = sensor.trace(Pose.identity(2), scene, environment.FOG_DENSE)
    assert sunny.P_amb == 0.5 * NZ.P_amb_sun and clear.P_amb == 0.0
    np.testing.assert_array_equal(sunny.power, clear.power)          # ambient never enters power
    h = clear.hit
    assert (fog.power[h] < clear.power[h]).all()
    np.testing.assert_array_equal(fog.r_path[h], clear.r_path[h])


# ------------------------------------------
# test_callable_pose_per_beam
# ------------------------------------------
# Sensor moving towards the wall: beam i is fired from pose(τ_i).
def test_callable_pose_per_beam():
    v = 3.0
    traj = constant_velocity((0.0, 0.0), (v, 0.0))
    sensor = LidarSensor(SensorConfig(n_subrays=1))
    tr = sensor.trace(traj, Scene2D.from_spec(scenes.single_wall()), environment.CLEAR)
    a = Planar2DPattern().beam_angle()
    tau = tr.fire_time
    x = v * tau
    y_hit = (2.0 - x) * np.tan(a)
    sel = (np.abs(a) < math.pi / 2) & (np.abs(y_hit) < 1.0 - 1e-6)
    np.testing.assert_allclose(tr.r_path[sel, 0], (2.0 - x[sel]) / np.cos(a[sel]), rtol=1e-12)
    assert tr.pose is traj


# ------------------------------------------
# test_rotating_pose_rotates_dirs
# ------------------------------------------
# A rotating sensor rotates each beam by its fire time.
def test_rotating_pose_rotates_dirs():
    traj = constant_velocity((0.0, 0.0), (0.0, 0.0), yaw0=0.0, yaw_rate=1.0)
    sensor = LidarSensor(SensorConfig(n_subrays=1))
    tr = sensor.trace(traj, Scene2D.from_spec(scenes.room_asym()), environment.CLEAR)
    a = Planar2DPattern().beam_angle() + tr.fire_time
    np.testing.assert_allclose(tr.dir_world[:, 0], np.stack([np.cos(a), np.sin(a)], 1), atol=1e-12)


# ------------------------------------------
# test_scene_materials_override_and_fallback
# ------------------------------------------
# Scene materials override the defaults, which remain as a fallback.
def test_scene_materials_override_and_fallback():
    dull = Material("dull", "lambertian", 0.3)
    spec = scenes.SceneSpec("w", scenes.single_wall().segments, [42], [0], materials={42: dull})
    sensor = LidarSensor(SensorConfig(n_subrays=1))
    tr42 = sensor.trace(Pose.identity(2), Scene2D.from_spec(spec), environment.CLEAR)
    tr0 = sensor.trace(Pose.identity(2), Scene2D.from_spec(scenes.single_wall()), environment.CLEAR)
    h = tr0.hit
    np.testing.assert_allclose(tr42.power[h], tr0.power[h] * 0.3 / 0.9, rtol=1e-12)

    class Bare:                                   # a Scene without a materials attribute
        dim = 2

        def __init__(self, inner):
            self.inner = inner

        def intersect(self, o, d, t) -> HitBatch:
            return self.inner.intersect(o, d, t)

    s2 = LidarSensor(SensorConfig(n_subrays=1), materials={**MATERIALS, WHITE: Material("w", "lambertian", 0.45)})
    tr_b = s2.trace(Pose.identity(2), Bare(Scene2D.from_spec(scenes.single_wall())), environment.CLEAR)
    np.testing.assert_allclose(tr_b.power[h], tr0.power[h] * 0.5, rtol=1e-12)
    with pytest.raises(KeyError):
        LidarSensor(materials={}).trace(Pose.identity(2), Bare(Scene2D.from_spec(scenes.single_wall())),
                                        environment.CLEAR)


# ------------------------------------------
# test_weights_sum_and_subray_power_split
# ------------------------------------------
# Sub-ray weights sum to one and split the beam power.
def test_weights_sum_and_subray_power_split():
    sensor = LidarSensor(SensorConfig(n_subrays=5, theta0=0.0))     # zero divergence: sub-rays coincide
    tr = sensor.trace(Pose.identity(2), Scene2D.from_spec(scenes.single_wall()), environment.CLEAR)
    tr1 = LidarSensor(SensorConfig(n_subrays=1)).trace(Pose.identity(2), Scene2D.from_spec(scenes.single_wall()),
                                                        environment.CLEAR)
    np.testing.assert_allclose(tr.power.sum(axis=1), tr1.power[:, 0], rtol=1e-12)


# ------------------------------------------
# test_specular_flag_slots
# ------------------------------------------
# Specular returns fill their own slots only when enabled.
def test_specular_flag_slots():
    pose = Pose.from_xi([1.0, 1.3, 0.0])
    scene = Scene2D.from_spec(scenes.mirror_room())
    off = LidarSensor(SensorConfig()).trace(pose, scene, environment.CLEAR)
    on = LidarSensor(SensorConfig(specular=True)).trace(pose, scene, environment.CLEAR)
    assert off.J == 5 and on.J == 10
    for f in ("hit", "r_path", "power", "surface_id", "cos_inc"):
        np.testing.assert_array_equal(getattr(on, f)[:, :5], getattr(off, f))
    mirror_direct = on.material_id[:, :5] == 3
    assert mirror_direct.any()
    np.testing.assert_array_equal(on.bounce[:, 5:], on.hit[:, 5:])
    assert not on.hit[:, 5:][~mirror_direct].any()                   # bounce only behind mirror hits
    assert on.hit[:, 5:][mirror_direct].all()                       # closed room: all reflections land
    assert (on.material_id[:, 5:][on.hit[:, 5:]] != 3).all()


# ------------------------------------------
# test_specular_double_mirror_slot_unused
# ------------------------------------------
# Reflected ray hitting another mirror: single-bounce model leaves the slot as an unused miss.
def test_specular_double_mirror_slot_unused():
    segs = np.array([[[2.0, -1.0], [2.0, 1.0]], [[-1.0, -1.0], [-1.0, 1.0]]])
    spec = scenes.SceneSpec("mm", segs, [3, 3], [0, 1])
    tr = LidarSensor(SensorConfig(specular=True, n_subrays=1)).trace(Pose.identity(2), Scene2D.from_spec(spec),
                                                                     EnvConfig())
    assert tr.hit[:, 0].any() and not tr.hit[:, 1].any() and not tr.bounce.any()


# ------------------------------------------
# test_trace_accepts_scene_spec_directly
# ------------------------------------------
# A trace accepts a scene specification directly.
def test_trace_accepts_scene_spec_directly():
    from lidar_sim.geometry.scene2d import Scene2D
    from lidar_sim.world.environment import CLEAR
    from lidar_sim.world.scenes import single_wall
    s = LidarSensor(SensorConfig(n_subrays=1))
    a = s.trace(Pose.identity(2), single_wall(), CLEAR)
    b = s.trace(Pose.identity(2), Scene2D.from_spec(single_wall()), CLEAR)
    np.testing.assert_array_equal(a.r_path, b.r_path)
    np.testing.assert_array_equal(a.power, b.power)


# ------------------------------------------
# test_trace_t0_shifts_fire_times_and_movers
# ------------------------------------------
# The scan start time shifts fire times and moving objects.
def test_trace_t0_shifts_fire_times_and_movers():
    from lidar_sim.world.environment import CLEAR
    from lidar_sim.world.scenes import moving_box
    s = LidarSensor(SensorConfig(n_subrays=1))
    spec = moving_box(p0=(2.0, -1.0), v=(1.0, 0.0))
    t0 = 0.5
    late = s.trace(Pose.identity(2), spec, CLEAR, t0=t0)
    np.testing.assert_allclose(late.fire_time, t0 + Planar2DPattern(s.sensor).fire_times())
    # box moved by v·t0 = 0.5 m along +x: beams hitting it see it farther along x
    early = s.trace(Pose.identity(2), spec, CLEAR, t0=0.0)
    box_e = early.object_id[:, 0] == 1
    box_l = late.object_id[:, 0] == 1
    assert box_e.any() and box_l.any()
    xe = (early.r_path[box_e, 0] * early.dir_world[box_e, 0, 0]).min()
    xl = (late.r_path[box_l, 0] * late.dir_world[box_l, 0, 0]).min()
    assert xl - xe == pytest.approx(0.5, abs=0.02)
