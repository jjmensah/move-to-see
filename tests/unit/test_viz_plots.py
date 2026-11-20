# ==========================================================
# tests/unit/test_viz_plots.py
# ==========================================================
# Unit tests of the scan plots.

import numpy as np

from lidar_sim.core.pose import Pose
from lidar_sim.sensor.lidar import LidarSensor
from lidar_sim.viz import plots
from lidar_sim.world.environment import CLEAR
from lidar_sim.world.scenes import room_asym


# ------------------------------------------
# test_scene_figure_round_trip
# ------------------------------------------
# A scan figure is drawn and saved.
def test_scene_figure_round_trip(tmp_path):
    spec, pose = room_asym(), Pose.from_xi([1.5, 1.5, 0.3])
    s = LidarSensor()
    scan = s.scan(pose, spec, CLEAR, np.random.default_rng(0))
    pts = plots.scan_points(scan)
    assert pts.shape == (scan.detected.sum(), 2)
    assert (pts[:, 0] > -0.1).all() and (pts[:, 0] < 6.1).all()        # inside the room (± noise)
    fig, ax = plots.figure()
    plots.draw_fov(ax, pose, s.sensor.fov, s.sensor.r_max)
    plots.draw_scene(ax, spec)
    plots.draw_sensor(ax, pose)
    ax.plot(*pts.T, ".", ms=2, **{k: v for k, v in plots.series(0).items() if k == "color"})
    out = plots.save(fig, tmp_path / "scan")
    assert all(p.exists() and p.stat().st_size > 1000 for p in out)


# ------------------------------------------
# test_series_fixed_order_and_cap
# ------------------------------------------
# Plot series styles come in a fixed order, with a cap.
def test_series_fixed_order_and_cap():
    assert plots.series(0)["color"] == "#2a78d6" and plots.series(2)["marker"] == "^"
    import pytest
    with pytest.raises(ValueError):
        plots.series(8)
