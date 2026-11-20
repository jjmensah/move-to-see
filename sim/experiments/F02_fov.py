# ==========================================================
# sim/experiments/F02_fov.py
# ==========================================================
# F02 — Field of view in 3D and dependence on the robot state (vv-plan §3).
#
# (a) Volumes of the geometric, line-of-sight and effective fields of view by Monte Carlo integration, against
#     an independent direction-quadrature calculation that uses the midterm calibration formulas for r_50.
# (b) Orientation: fraction of the prior region inside the geometric field of view versus yaw.
# (c) Velocity: scan skew of a robot driving towards a wall against the analytic moving-origin range.

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from lidar_sim.geometry.scene3d import Scene3D
from lidar_sim.viz import plots
from lidar_sim.world.materials import GREY, MATERIALS
from mts.config import CONDITIONS, NOISE, RASTER, SENSOR, RobotConfig, make_sensor
from mts.fov import fov_masks
from mts.robot import sensor_pose, step
from mts.world import World, yard
from sim.calibrate import detection_probability, received_power

RHO = {"black": 0.05, "white": 0.9}


# ------------------------------------------
# _sample_sector
# ------------------------------------------
# Points uniform in volume in the geometric FOV (F6) of ``pose``.
def _sample_sector(rng, n: int, pose) -> np.ndarray:
    az = rng.uniform(-RASTER.fov / 2, RASTER.fov / 2, n)
    sin_el = rng.uniform(math.sin(RASTER.el_min), math.sin(RASTER.el_max), n)
    r = np.cbrt(rng.uniform(SENSOR.r_min**3, SENSOR.r_max**3, n))
    cos_el = np.sqrt(1.0 - sin_el**2)
    return pose.apply(np.stack([r * cos_el * np.cos(az), r * cos_el * np.sin(az), r * sin_el], axis=1))


# ------------------------------------------
# _sector_volume
# ------------------------------------------
# Volume of the geometric field of view between r_min and r_max.
def _sector_volume() -> float:
    return RASTER.fov * (math.sin(RASTER.el_max) - math.sin(RASTER.el_min)) * (SENSOR.r_max**3 - SENSOR.r_min**3) / 3.0


# ------------------------------------------
# _r50_reference
# ------------------------------------------
# r_50 from the midterm calibration formulas (independent of ``BeamModel``).
def _r50_reference(rho: float, env) -> float:
    f = lambda r: detection_probability(NOISE, SENSOR, received_power(NOISE, rho, r, env.visibility_m), env.P_amb) - 0.5
    return brentq(f, 0.02, 500.0, xtol=1e-10)


# ------------------------------------------
# _reach_by_face_planes
# ------------------------------------------
# Distance to the first obstacle along each direction, by enumerating box-face planes and the ground.
#
# Independent of ``Scene3D.intersect`` (no slab method): for every box and each of its six faces, intersect the
# ray with the face's plane and keep the hit if it lies within the face rectangle; the ground is the plane
# z = 0. Uses the raw world bounds.
def _reach_by_face_planes(world, origin, d) -> np.ndarray:
    reach = np.full(len(d), np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        for lo, hi in zip(world.box_lo, world.box_hi, strict=True):
            for axis in range(3):
                others = [a for a in range(3) if a != axis]
                for plane in (lo[axis], hi[axis]):
                    t = (plane - origin[axis]) / d[:, axis]
                    ok = np.isfinite(t) & (t > 1e-9)
                    for a in others:
                        c = origin[a] + t * d[:, a]
                        ok &= (c >= lo[a]) & (c <= hi[a])
                    reach = np.where(ok & (t < reach), t, reach)
        t = -origin[2] / d[:, 2]
        reach = np.where((d[:, 2] < 0) & (t > 1e-9) & (t < reach), t, reach)
    return reach


# ------------------------------------------
# _quadrature_volumes
# ------------------------------------------
# Line-of-sight volume and its part within ``r_cap``, by midpoint quadrature over directions.
#
# ``reference="planes"`` uses the independent face-plane enumeration; ``"scene"`` uses ``Scene3D.intersect``
# (the reference of the first registered run, kept for comparison).
def _quadrature_volumes(world, scene, pose, r_cap: float, n_az: int = 1201, n_el: int = 301,
                        reference: str = "planes") -> tuple[float, float]:
    az = -RASTER.fov / 2 + (np.arange(n_az) + 0.5) * RASTER.fov / n_az
    s_lo, s_hi = math.sin(RASTER.el_min), math.sin(RASTER.el_max)
    sin_el = s_lo + (np.arange(n_el) + 0.5) * (s_hi - s_lo) / n_el          # uniform in sin ε ⇒ equal solid angle
    A, S = np.meshgrid(az, sin_el)
    C = np.sqrt(1 - S**2)
    d = pose.rotate(np.stack([C * np.cos(A), C * np.sin(A), S], axis=-1).reshape(-1, 3))
    if reference == "planes":
        raw = _reach_by_face_planes(world, pose.t, d)
    else:
        h = scene.intersect(np.broadcast_to(pose.t, d.shape), d, np.zeros(len(d)))
        raw = np.where(h.hit, h.r, np.inf)
    reach = np.clip(raw, SENSOR.r_min, SENSOR.r_max)
    dω = RASTER.fov * (s_hi - s_lo) / (n_az * n_el)
    vol = lambda R: float(np.sum(np.maximum(R**3 - SENSOR.r_min**3, 0.0) / 3.0) * dω)
    return vol(reach), vol(np.minimum(reach, max(r_cap, SENSOR.r_min)))


# ------------------------------------------
# _part_a
# ------------------------------------------
# Part A: nested field-of-view volumes per pose, probe and condition.
def _part_a(entry, rng, quick):
    world, cfg = yard(), RobotConfig()
    scene = world.background_scene()
    n = 100_000 if quick else entry["n"]
    v_geo = _sector_volume()
    out, ok_nested, ok_order, ok_match = [], True, True, True
    for x in entry["params"]["poses"]:
        pose = sensor_pose(np.array([*x, 0.0]), cfg)
        pts = _sample_sector(rng, n, pose)
        row = {"pose": x, "geometric_m3": v_geo, "cases": {}}
        vols = {}
        for probe, rho in RHO.items():
            for cname in ("C1", "C2", "C3"):
                env = CONDITIONS[cname].env
                m = fov_masks(pose, scene, pts, rho, env)
                assert m["geometric"].all()
                v_los, v_eff = v_geo * m["los"].mean(), v_geo * m["effective"].mean()
                r_ref = _r50_reference(rho, env)
                q_los, q_eff = _quadrature_volumes(world, scene, pose, r_ref)
                _, q_eff_coarse = _quadrature_volumes(world, scene, pose, r_ref, n_az=601, n_el=151)
                _, q_eff_scene = _quadrature_volumes(world, scene, pose, r_ref, reference="scene")
                v_half = v_geo * m["effective"][: n // 2].mean()
                nested = bool((m["effective"] <= m["los"]).all() and (m["los"] <= m["geometric"]).all())
                rel = abs(v_eff - q_eff) / q_eff
                se = v_geo * math.sqrt(max(m["effective"].mean() * (1 - m["effective"].mean()), 1e-12) / n)
                row["cases"][f"{probe}/{cname}"] = {
                    "los_m3": v_los, "effective_m3": v_eff, "effective_se_m3": se, "quadrature_los_m3": q_los,
                    "quadrature_effective_m3": q_eff, "rel_diff": rel, "nested": nested,
                    "r50_reference_m": r_ref,
                    # convergence and reference diagnostics (reported, not gated)
                    "half_sample_effective_m3": v_half, "half_sample_rel_change": abs(v_half - v_eff) / v_eff,
                    "quadrature_coarse_effective_m3": q_eff_coarse,
                    "quadrature_refinement_rel_change": abs(q_eff_coarse - q_eff) / q_eff,
                    "quadrature_scene_reference_m3": q_eff_scene,
                    "rel_diff_scene_reference": abs(v_eff - q_eff_scene) / q_eff_scene}
                vols[(probe, cname)] = v_eff
                ok_nested &= nested
                ok_match &= rel <= 0.03
        ok_order &= vols[("black", "C1")] > vols[("black", "C2")] > vols[("black", "C3")]
        out.append(row)
    return out, {"nested_geometric_los_effective": bool(ok_nested), "black_volume_clear_gt_fog_gt_sun": bool(ok_order),
                 "effective_volume_matches_independent_3pct": bool(ok_match)}


# ------------------------------------------
# _part_b
# ------------------------------------------
# Fraction of the prior region (grid of points) inside the geometric FOV versus yaw, at the start position.
def _part_b(entry):
    from lidar_sim.sensor.scan3d import in_geometric_fov_3d
    world, cfg = yard(), RobotConfig()
    g = [np.linspace(lo, hi, 9) for lo, hi in zip(world.prior_lo, world.prior_hi, strict=True)]
    pts = np.stack(np.meshgrid(*g, indexing="ij"), axis=-1).reshape(-1, 3)
    rows = []
    for name, xy in (("start", world.start[:2]), ("west viewpoint", np.array([6.1, 4.0]))):
        yaws = np.radians(np.arange(-180, 181, 15))
        frac = [float(in_geometric_fov_3d(sensor_pose(np.array([*xy, y, 0.0]), cfg), pts, RASTER, SENSOR.r_min,
                                          SENSOR.r_max).mean()) for y in yaws]
        rows.append({"position": name, "xy": xy.tolist(), "yaw_deg": np.degrees(yaws).tolist(), "fraction": frac})
    return rows


# ------------------------------------------
# _part_c
# ------------------------------------------
# Robot driving at constant speed towards a wall at x = 6: per-beam range against the moving-origin oracle.
def _part_c(entry):
    cfg = RobotConfig()
    wall_x = 6.0
    world = World(arena=(0.0, 12.0, -40.0, 40.0), box_lo=[[wall_x, -40.0, 0.0]], box_hi=[[wall_x + 0.2, 40.0, 30.0]],
                  box_material=[GREY], rough=np.zeros((0, 4)), prior_lo=[9.0, 0.0, 0.5], prior_hi=[9.5, 0.5, 1.0],
                  target_half=[0.25, 0.25, 0.25], target_material=GREY, start=[2.0, 0.0, 0.0, 0.0],
                  ground_material=GREY)
    scene = Scene3D(world.box_lo, world.box_hi, world.box_material, [1], materials=MATERIALS)   # wall only, no ground
    sensor = make_sensor(1)
    pat = sensor.pattern
    az, el, tau = pat.beam_angle(), pat.beam_elevation(), pat.fire_times()
    sel = (np.abs(az) < math.radians(60)) & (el >= 0)
    rows, ok = [], True
    static = None
    for v in entry["params"]["speeds"]:
        traj = step(world, cfg, np.array([2.0, 0.0, 0.0, v]), np.array([0.0, 0.0]), t0=0.0)
        t0 = traj.t1 - RASTER.t_scan
        tr = sensor.trace(traj.pose, scene, CONDITIONS["C1"].env, t0=t0)
        x_i = 2.0 + v * (t0 + tau)                                             # closed form: constant speed along +x
        oracle = (wall_x - x_i) / (np.cos(el) * np.cos(az))
        err = float(np.abs(tr.r_path[sel, 0] - oracle[sel]).max())
        assert tr.hit[sel, 0].all()
        if static is None:
            static = tr.r_path[:, 0].copy()
        x0 = 2.0 + v * t0                                                      # same scan with a frozen pose
        frozen = (wall_x - x0) / (np.cos(el) * np.cos(az))
        skew = frozen[sel] - tr.r_path[sel, 0]                                  # = v τ_i / (cos ε cos α)
        slope = float(np.polyfit(tau[sel], skew * np.cos(el[sel]) * np.cos(az[sel]), 1)[0]) if v > 0 else 0.0
        rows.append({"speed": v, "max_abs_range_error_m": err, "max_skew_m": float(np.abs(skew).max()),
                     "fitted_speed": slope})
        ok &= err < 1e-9
    skews = [r["max_skew_m"] for r in rows]
    return rows, {"moving_origin_range_error_below_1e-9": bool(ok),
                  "skew_nonzero_and_increasing_with_speed": bool(skews[0] < 1e-12 and all(np.diff(skews) > 1e-6))}


# ------------------------------------------
# run
# ------------------------------------------
# Runs F02: field-of-view volumes, the independent reference and scan skew with speed.
def run(entry: dict, out: Path, quick: bool, jobs: int) -> dict:
    rng = np.random.default_rng(entry["seed"])
    a, va = _part_a(entry, rng, quick)
    b = _part_b(entry)
    c, vc = _part_c(entry)
    _figure(a, b, c, out)
    cases = [x for row in a for x in row["cases"].values()]
    return {"volumes": a, "orientation": b, "velocity": c,
            "reference": "independent face-plane enumeration (see deviations in docs/verification/vv-plan.md)",
            "max_rel_diff": max(x["rel_diff"] for x in cases),
            "max_rel_diff_scene_reference": max(x["rel_diff_scene_reference"] for x in cases),
            "max_half_sample_rel_change": max(x["half_sample_rel_change"] for x in cases),
            "max_quadrature_refinement_rel_change": max(x["quadrature_refinement_rel_change"] for x in cases),
            "verdicts": {**va, **vc}}


# ------------------------------------------
# _figure
# ------------------------------------------
# Plots the effective volumes and the scan skew against speed.
def _figure(a, b, c, out: Path) -> None:
    fig, axs = plots.figure(1, 3, height_in=2.6)
    ax = axs[0]
    labels = ["clear", "fog", "sun"]
    xs = np.arange(3)
    row = a[0]
    for j, probe in enumerate(RHO):
        st = plots.series(j)
        v = [row["cases"][f"{probe}/{c}"]["effective_m3"] for c in ("C1", "C2", "C3")]
        q = [row["cases"][f"{probe}/{c}"]["quadrature_effective_m3"] for c in ("C1", "C2", "C3")]
        ax.bar(xs + (j - 0.5) * 0.36, v, 0.34, color=st["color"], label=f"{probe} probe (Monte Carlo)")
        ax.plot(xs + (j - 0.5) * 0.36, q, "_", color=plots.INK, ms=12, mew=1.4,
                label="independent quadrature" if j == 0 else None)
    los = row["cases"]["white/C1"]["los_m3"]
    ax.axhline(los, color=plots.INK_2, ls=":", lw=1.0, label="line-of-sight volume")
    ax.set_xticks(xs, labels)
    ax.set(ylabel="effective FOV volume [m³]", title="(a) start pose")
    ax = axs[1]
    for j, r in enumerate(b):
        st = plots.series(j)
        ax.plot(r["yaw_deg"], r["fraction"], st["marker"] + st["linestyle"], color=st["color"], ms=3,
                label=r["position"])
    ax.set(xlabel="robot yaw [deg]", ylabel="prior region in geometric FOV", title="(b) orientation",
           ylim=(-0.03, 1.05))
    ax.legend(fontsize=6.5, loc="center", handlelength=1.3, borderaxespad=0.2)
    ax = axs[2]
    ax.plot([r["speed"] for r in c], [1e3 * r["max_skew_m"] for r in c], "o-", color=plots.SERIES[0])
    ax.set(xlabel="robot speed [m/s]", ylabel="max scan skew [mm]", title="(c) velocity")
    plots.legend_below(fig, axs[0], ncols=4)
    plots.save(fig, out / "fov_state.pdf")
