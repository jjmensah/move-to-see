# ==========================================================
# sim/experiments/F01_measurement.py
# ==========================================================
# F01 — 3D measurement model against the waveform truth (vv-plan §3).
#
# A 1 m square panel faces one beam of the 3D raster (the sensor is pitched so that the chosen beam is normal to
# the panel) at a set of ranges, for two elevation rows, two reflectivities and three environments. The traced
# record of that beam is replicated n times and measured once by the waveform simulator.

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from lidar_sim.core.pose import Pose
from lidar_sim.core.types import RadiometricScan
from lidar_sim.geometry.scene3d import Scene3D
from lidar_sim.physics.waveform import cfar_threshold
from lidar_sim.viz import plots
from lidar_sim.world.materials import BLACK, MATERIALS, WHITE
from mts.config import CONDITIONS, NOISE, SENSOR, make_model, make_sensor
from mts.metrics import wilson


# ------------------------------------------
# _beam_index
# ------------------------------------------
# Index of the raster beam at azimuth 0 and the given elevation.
def _beam_index(el_deg: float) -> int:
    pat = make_sensor(9).pattern
    az, el = pat.beam_angle(), pat.beam_elevation()
    i = int(np.argmin(np.abs(az) + np.abs(el - math.radians(el_deg))))
    assert abs(az[i]) < 1e-12 and abs(el[i] - math.radians(el_deg)) < 1e-12
    return i


# ------------------------------------------
# _pose_facing
# ------------------------------------------
# Sensor at (0, 0, 2) pitched so that the (azimuth 0, elevation el) beam points along world +x.
def _pose_facing(el_deg: float) -> Pose:
    e = math.radians(el_deg)
    R = np.array([[math.cos(e), 0.0, math.sin(e)], [0.0, 1.0, 0.0], [-math.sin(e), 0.0, math.cos(e)]])
    return Pose(R, np.array([0.0, 0.0, 2.0]))


# ------------------------------------------
# _hand_power
# ------------------------------------------
# E7–E8 by hand for a facing surface at range ``r``: P0 (ρ/π)(A_rx/r²) η exp(−2σr), σ from Koschmieder–Kim.
#
# Written out here (not imported from the package) as an independent check of the traced power.
def _hand_power(rho: float, r: float, env) -> float:
    v_km = env.visibility_m / 1000.0
    q = 1.6 if v_km > 50 else 1.3 if v_km > 6 else 0.16 * v_km + 0.34 if v_km > 1 else v_km - 0.5 if v_km > 0.5 else 0.0
    sigma = 3.912 / env.visibility_m * (env.wavelength / 550e-9) ** (-q)
    return NOISE.P0 * (rho / math.pi) * (NOISE.A_rx / r**2) * NOISE.eta * math.exp(-2.0 * sigma * r)


# ------------------------------------------
# _replicate
# ------------------------------------------
# Record with ``n`` copies of beam ``i`` of ``trace``.
def _replicate(trace: RadiometricScan, i: int, n: int) -> RadiometricScan:
    f = trace.fields()
    rep = {k: np.repeat(np.asarray(v)[i:i + 1], n, axis=0) for k, v in f.items() if k not in ("pose", "P_amb")}
    return RadiometricScan(pose=trace.pose, P_amb=trace.P_amb, **rep)


# ------------------------------------------
# run
# ------------------------------------------
# Runs F01: simulated crossing rates and range noise against the analytic model.
def run(entry: dict, out: Path, quick: bool, jobs: int) -> dict:
    p = entry["params"]
    n = 400 if quick else entry["n"]
    n_empty = 100_000 if quick else p["n_no_return"]
    sensor, model = make_sensor(9), make_model(9)
    cells = []
    for ci, (cname, mat, el, r) in enumerate((c, m, e, r) for c in p["conditions"] for m in (BLACK, WHITE)
                                             for e in p["elevations_deg"] for r in p["ranges_m"]):
        env = CONDITIONS[cname].env
        scene = Scene3D([[r, -0.5, 1.5]], [[r + 0.05, 0.5, 2.5]], [mat], [1], materials=MATERIALS)
        i = _beam_index(el)
        trace = sensor.trace(_pose_facing(el), scene, env)
        assert trace.hit[i].all() and np.allclose(trace.r_path[i], r, atol=2e-3) and trace.cos_inc[i].min() > 0.9998
        pred = model.predict(trace, env)
        p_hand = _hand_power(MATERIALS[mat].rho_diffuse, r, env)
        power_err = abs(float(trace.power[i].sum()) - p_hand) / p_hand      # sub-rays: cos³δ ≈ 1 − 1e-4
        rep = _replicate(trace, i, n)
        scan = sensor.measure(rep, env, np.random.default_rng([entry["seed"], ci]))
        gamma = cfar_threshold(NOISE, env.P_amb, SENSOR.n_samples)
        k_cross = int((scan.amplitude[:, 0] >= gamma).sum())
        pd, sig = float(pred.P_D[i]), float(pred.sigma_R[i])
        m_win = float(model._mi(pred, np.arange(pred.B) == i)[0])
        p_wrap = float(pred.out_prob[i].sum())
        raw = scan.range_raw
        core = np.abs(raw - pred.r_bar[i]) < 0.5 * float(np.abs(pred.out_shift[i]).min())
        cells.append({"condition": cname, "material": "black" if mat == BLACK else "white", "elevation_deg": el,
                      "range_m": r, "n": n, "P_D": pd, "crossing_rate": k_cross / n, "k_cross": k_cross,
                      "report_prob": pd * m_win, "report_rate": float(scan.detected.mean()), "sigma_R": sig,
                      "range_std_all": float(raw.std(ddof=1)), "range_std_core": float(raw[core].std(ddof=1)),
                      "n_core": int(core.sum()), "wrap_prob": p_wrap, "power_rel_error_vs_hand": power_err,
                      "high_snr": bool(pd > 0.999999 and p_wrap < 1e-6)})
    conf = 1.0 - 0.01 / len(cells)                                # Bonferroni: familywise 99 %
    for c in cells:
        lo, hi = wilson(c["k_cross"], c["n"], conf)
        c["wilson"] = [lo, hi]
        c["pd_inside"] = bool(lo <= c["P_D"] <= hi)
        c["std_ratio"] = c["range_std_core"] / c["sigma_R"] if c["high_snr"] else None
    hs = [c for c in cells if c["high_snr"]]

    # false reports from beams with no optical return (clear air and sunlight)
    empty = []
    for cname in ("C1", "C3"):
        env = CONDITIONS[cname].env
        tr = RadiometricScan.empty(n_empty, 9, d=3, pose=Pose.identity(3), P_amb=env.P_amb)
        sc = sensor.measure(tr, env, np.random.default_rng([entry["seed"], 1000, len(empty)]))
        k = int(sc.detected.sum())
        want = NOISE.P_FA0 * (SENSOR.r_max - SENSOR.r_min) / SENSOR.Lc
        lo, hi = wilson(k, n_empty, 0.99)
        empty.append({"condition": cname, "n": n_empty, "reports": k, "rate": k / n_empty, "predicted": want,
                      "wilson99": [lo, hi], "inside": bool(lo <= want <= hi)})

    _figure(cells, out)
    near = [c for c in cells if 0.02 < c["P_D"] < 0.98]
    return {
        "cells": cells, "no_return": empty, "bonferroni_conf": conf,
        "n_cells": len(cells), "n_high_snr": len(hs),
        "max_power_rel_error_vs_hand": max(c["power_rel_error_vs_hand"] for c in cells),
        "std_ratio_range": [min(c["std_ratio"] for c in hs), max(c["std_ratio"] for c in hs)] if hs else None,
        "max_abs_crossing_error": max(abs(c["crossing_rate"] - c["P_D"]) for c in cells),
        "near_threshold_report_gap": [
            {"condition": c["condition"], "material": c["material"], "range_m": c["range_m"],
             "elevation_deg": c["elevation_deg"], "predicted": c["report_prob"], "simulated": c["report_rate"]}
            for c in near],
        "verdicts": {"crossing_rate_in_simultaneous_wilson": all(c["pd_inside"] for c in cells),
                     "high_snr_std_ratio_within_5pct": bool(hs) and all(abs(c["std_ratio"] - 1) <= 0.05 for c in hs),
                     "false_report_rate_in_wilson99": all(e["inside"] for e in empty)},
    }


# ------------------------------------------
# _figure
# ------------------------------------------
# Plots detection probability against range, model and simulation, per condition.
def _figure(cells: list[dict], out: Path) -> None:
    fig, axs = plots.figure(1, 3, height_in=2.5)
    names = {"C1": "clear", "C2": "fog (V = 20 m)", "C3": "sunlight (1 sun)"}
    rr = np.linspace(0.8, 5.4, 200)
    from mts.fov import probe_pd
    for ax, cname in zip(axs, names, strict=True):
        for j, mat in enumerate(("black", "white")):
            st = plots.series(j)
            ax.plot(rr, probe_pd(rr, 0.05 if mat == "black" else 0.9, CONDITIONS[cname].env), color=st["color"],
                    ls=st["linestyle"], label=f"{mat}: model $P_D$")
            for el, mk in ((0.0, "o"), (20.0, "^")):
                sel = [c for c in cells
                       if c["condition"] == cname and c["material"] == mat and c["elevation_deg"] == el]
                x = [c["range_m"] for c in sel]
                y = [c["crossing_rate"] for c in sel]
                err = np.maximum([[c["crossing_rate"] - c["wilson"][0] for c in sel],
                                  [c["wilson"][1] - c["crossing_rate"] for c in sel]], 0.0)
                ax.errorbar(x, y, yerr=err, fmt=mk, color=st["color"], ms=4, mfc="none" if el else st["color"],
                            elinewidth=0.8, label=f"{mat}: simulated, ε = {el:g}°")
        ax.set(title=names[cname], xlabel="range [m]", ylim=(-0.04, 1.06))
    axs[0].set_ylabel("threshold-crossing rate")
    plots.legend_below(fig, axs[1], ncols=3)
    plots.save(fig, out / "pd_vs_range.pdf")
