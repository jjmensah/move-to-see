# ==========================================================
# sim/experiments/F04_info_bound.py
# ==========================================================
# F04: F17 against independently sampled, exact finite-mixture F14 densities.

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import spearmanr, truncnorm

from lidar_sim.viz import plots
from mts.belief import ParticleBelief
from mts.config import CONDITIONS, SENSOR, BeliefConfig, PlannerConfig, RobotConfig
from mts.info import mi_lower_bound, pair_distance
from mts.planner import ViewpointPlanner
from mts.predict import Predictor
from mts.robot import sensor_pose
from mts.sim import run_trial
from mts.viz import ctrl_style
from mts.world import yard

FIELDS = ("pi", "mu", "sigma", "Z", "uniform")


ABS_TOL = 0.01      # [nat] absolute tolerance: a Monte Carlo estimate cannot resolve information carried by
                    # events rarer than about 1/n (its standard error is then 0), see vv-plan F04


# ------------------------------------------
# varying_channel
# ------------------------------------------
# Drop exactly identical beam distributions, including inactive density parameters.
def varying_channel(channel):
    pi = channel.pi
    gaussian = (pi > 0) & ~channel.uniform
    values = [pi, (pi > 0) & channel.uniform,
              *[np.where(gaussian, getattr(channel, f), 0.) for f in ("mu", "sigma", "Z")]]
    keep = np.logical_or.reduce([np.any(v != v[:1], axis=0) for v in values])
    return replace(channel, beams=channel.beams[keep],
                   **{f: getattr(channel, f)[:, keep] for f in FIELDS})


# ------------------------------------------
# exact_mi_mc
# ------------------------------------------
# Sample hypothesis then independent beams; evaluate every mixture component in log space.
#
# Uses scipy's truncated-normal sampler/log density, independently of mts.info's
# Bhattacharyya formulas. Only an n by M log-density array is retained.
def exact_mi_mc(channel, n, rng, lo, hi):
    channel = varying_channel(channel)
    M, B = channel.pi.shape
    chosen = rng.integers(M, size=n)
    logs = np.zeros((n, M))
    for b in range(B):
        pi = channel.pi[:, b]
        reported = rng.random(n) < pi[chosen]
        with np.errstate(divide="ignore"):
            logs[~reported] += np.log1p(-pi)
        ids = np.flatnonzero(reported)
        if not len(ids):
            continue
        source = chosen[ids]
        z = np.empty(len(ids))
        uni = channel.uniform[source, b]
        z[uni] = rng.uniform(lo, hi, uni.sum())
        g = source[~uni]
        mu, sigma = channel.mu[g, b], channel.sigma[g, b]
        z[~uni] = truncnorm.rvs((lo - mu) / sigma, (hi - mu) / sigma,
                               loc=mu, scale=sigma, random_state=rng)
        density = np.full((len(ids), M), -np.inf)
        for m in np.flatnonzero(pi > 0):
            if channel.uniform[m, b]:
                density[:, m] = np.log(pi[m]) - np.log(hi - lo)
            else:
                mu, sigma = channel.mu[m, b], channel.sigma[m, b]
                density[:, m] = np.log(pi[m]) + truncnorm.logpdf(
                    z, (lo - mu) / sigma, (hi - mu) / sigma, loc=mu, scale=sigma)
        logs[ids] += density
    information = logs[np.arange(n), chosen] - logsumexp(logs, axis=1) + np.log(M)
    return float(information.mean()), float(information.std(ddof=1) / np.sqrt(n)), B


# ------------------------------------------
# _snapshots
# ------------------------------------------
# Index zero is the prior; indices 15 and 40 follow that many updates.
def _snapshots(entry, out, quick):
    scans = [0, 1, 2] if quick else entry["params"]["scans"]
    cfg = BeliefConfig(n_particles=64) if quick else BeliefConfig()
    count = 1 if quick else 4
    for trial in range(count):
        ks = scans if trial < 3 else [0]
        path = out / "trials" / f"snapshots_{trial:03d}.npz"
        if not path.exists():
            rec = run_trial(yard(), CONDITIONS[entry["params"]["condition"]], "fixed", entry["seed"], trial,
                            steps=max(ks), belief_cfg=cfg, keep=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **rec.to_npz(), snapshot_scans=ks,
                                particles=np.array(rec.extra["particles"])[ks],
                                weights=np.array(rec.extra["weights"])[ks])
        with np.load(path, allow_pickle=False) as data:
            for i, k in enumerate(ks):
                yield trial, k, data["particles"][i], data["weights"][i]


# ------------------------------------------
# _viewpoints
# ------------------------------------------
# Spread-out viewpoints chosen from the planner's lattice by farthest-point selection.
def _viewpoints(world, predictor, count):
    planner = ViewpointPlanner(world, RobotConfig(), PlannerConfig(), predictor)
    cells = np.array([planner.grid.centre(np.unravel_index(i, planner.grid.free.shape))
                      for i in planner._lattice])
    # Farthest-point selection gives a deterministic spread over free lattice cells.
    selected = [0]
    distance = np.full(len(cells), np.inf)
    for _ in range(count - 1):
        distance = np.minimum(distance, np.linalg.norm(cells - cells[selected[-1]], axis=1))
        selected.append(int(np.argmax(distance)))
    return cells[selected]


# ------------------------------------------
# run
# ------------------------------------------
# Runs F04: the information bound against a Monte Carlo estimate of the mutual information.
def run(entry: dict, out: Path, quick: bool, jobs: int) -> dict:
    p = entry["params"]
    world = yard()
    predictor = Predictor(world, CONDITIONS[p["condition"]].env)
    views = _viewpoints(world, predictor, 4 if quick else p["viewpoints"])
    n = 128 if quick else entry["n"]
    rows, rankings = [], []
    for belief_id, (trial, scan, particles, weights) in enumerate(_snapshots(entry, out, quick)):
        rng = np.random.default_rng([entry["seed"], 100, belief_id])
        belief = ParticleBelief(world, BeliefConfig(n_particles=len(particles)), rng)
        belief.particles, belief.weights = particles, weights
        support = belief.support(p["M"], rng)
        mean = belief.mean()
        local = []
        for j, xy in enumerate(views):
            heading = np.arctan2(mean[1] - xy[1], mean[0] - xy[0])
            geom = predictor.geometry(sensor_pose(np.r_[xy, heading, 0.], RobotConfig()), K=1)
            channel = varying_channel(predictor.design(geom, support))
            bound = mi_lower_bound(pair_distance(channel, SENSOR.r_min, SENSOR.r_max))
            mc, se, beams = exact_mi_mc(channel, n, np.random.default_rng([entry["seed"], 200, belief_id, j]),
                                      SENSOR.r_min, SENSOR.r_max)
            local.append({"belief": belief_id, "trial": trial, "scan": scan, "viewpoint": j,
                          "xy_m": xy.tolist(), "heading_rad": float(heading), "varying_beams": beams,
                          "bound_nats": bound, "mc_nats": mc, "se_nats": se,
                          "pass": bool(bound <= mc + 3 * se + ABS_TOL)})
        lower = np.array([r["bound_nats"] for r in local])
        exact = np.array([r["mc_nats"] for r in local])
        rho = float(spearmanr(lower, exact).statistic) if np.ptp(lower) > 0 and np.ptp(exact) > 0 else None
        rankings.append({"belief": belief_id, "trial": trial, "scan": scan, "spearman": rho,
                         "bound_best": int(lower.argmax()), "mc_best": int(exact.argmax()),
                         "regret_nats": float(exact.max() - exact[lower.argmax()])})
        rows.extend(local)
        print(f"F04: belief {belief_id + 1}, scan {scan}, {len(local)} viewpoints", flush=True)
    fig, axs = plots.figure(1, 2, height_in=2.7)
    st = ctrl_style("info")
    axs[0].errorbar([r["bound_nats"] for r in rows], [r["mc_nats"] for r in rows],
                    yerr=[3 * r["se_nats"] for r in rows], fmt=st["marker"], color=st["color"], ms=3)
    axs[0].plot([0, np.log(p["M"])], [0, np.log(p["M"])], ":", color=plots.INK_2)
    axs[0].set(xlabel=r"pairwise lower bound $\hat I$ [nat]", ylabel="exact information, MC ± 3 SE [nat]", title="(a) bound vs exact information")
    axs[1].plot(np.arange(len(rankings)), [r["regret_nats"] for r in rankings], st["marker"], color=st["color"], ms=6)
    axs[1].set_ylim(-0.01, max(0.1, 1.2 * max(r["regret_nats"] for r in rankings)))
    axs[1].set(xlabel="belief index", ylabel="selection regret [nat]", title="(b) regret of the bound-best viewpoint")
    plots.save(fig, out / "information_bound.pdf")
    rhos = [r["spearman"] for r in rankings if r["spearman"] is not None]
    regrets = [r["regret_nats"] for r in rankings]
    excess = [(r["bound_nats"] - r["mc_nats"] - ABS_TOL) / r["se_nats"] for r in rows if r["se_nats"] > 0]
    return {"samples_per_case": n, "n_cases": len(rows), "condition": p["condition"], "M": p["M"],
            "abs_tolerance_nats": ABS_TOL, "n_violations": sum(not r["pass"] for r in rows),
            "n_cases_below_tolerance": sum(r["bound_nats"] < ABS_TOL for r in rows),
            "max_excess_in_se": float(max(excess)) if excess else 0.0,
            "median_spearman": float(np.median(rhos)) if rhos else None,
            "median_regret_nat": float(np.median(regrets)), "max_regret_nat": float(max(regrets)),
            "cases": rows, "rankings": rankings,
            "verdicts": {"bound_below_mc_plus_3se": all(r["pass"] for r in rows)}}
