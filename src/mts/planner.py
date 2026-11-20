# ==========================================================
# src/mts/planner.py
# ==========================================================
# Receding-horizon viewpoint selection with travel-time cost (F17–F19).

from dataclasses import replace

import numpy as np

from lidar_sim.core.types import HitBatch
from mts.config import SENSOR
from mts.control import Action
from mts.info import mi_lower_bound, pair_distance
from mts.predict import ScanGeometry
from mts.robot import NavGrid, follow, sensor_pose


# ------------------------------------------
# ViewpointPlanner
# ------------------------------------------
# Score static K=1 views of one shared belief support, without truth access.
#
# The lattice starts at the grid's first centre, with an integer cell stride.
# Candidate indices are flattened row-major grid indices (including stay).
# Headings are exact; only the last geometry per cell is cached, bounding memory.
class ViewpointPlanner:
    def __init__(self, world, robot_cfg, planner_cfg, predictor, kind="aware"):
        if kind not in ("aware", "blind"):
            raise ValueError("kind must be aware or blind")
        pc = planner_cfg
        if (pc.M < 1 or int(pc.M) != pc.M or not np.isfinite(pc.lambda_t) or pc.lambda_t < 0
                or not np.isfinite(pc.radius) or pc.radius < 0
                or not np.isfinite(pc.cell) or pc.cell <= 0
                or not np.isfinite(pc.lattice) or pc.lattice < pc.cell
                or not np.isfinite(pc.switch_margin) or pc.switch_margin < 0
                or not np.isfinite(pc.sigma_ref) or pc.sigma_ref <= 0):
            raise ValueError("invalid planner configuration")
        stride = round(pc.lattice / pc.cell)
        if not np.isclose(stride * pc.cell, pc.lattice, rtol=0, atol=1e-10):
            raise ValueError("lattice must be an integer multiple of cell")
        self.world, self.cfg, self.pc = world, robot_cfg, pc
        self.predictor, self.kind = predictor, kind
        self.name = "info" if kind == "aware" else "blind"
        self.grid = NavGrid(world, robot_cfg, cell=pc.cell)
        iy, ix = np.indices(self.grid.free.shape)
        x, y = self.grid.xs[ix], self.grid.ys[iy]
        xmin, xmax, ymin, ymax = world.keepouts(robot_cfg)[-1]
        dx = np.maximum(np.maximum(xmin - x, x - xmax), 0.)
        dy = np.maximum(np.maximum(ymin - y, y - ymax), 0.)
        mask = (self.grid.free & (iy % stride == 0) & (ix % stride == 0)
                & (np.hypot(dx, dy) <= pc.radius))
        self._lattice = np.flatnonzero(mask)
        self._source = None
        self._travel = None
        self._geometries = {}
        self._goal = None

    def _geometry(self, index, mean):
        cell = np.unravel_index(index, self.grid.free.shape)
        xy = self.grid.centre(cell)
        heading = float(np.arctan2(mean[1] - xy[1], mean[0] - xy[0]))
        cached = self._geometries.get(index)
        if cached is None or cached[0] != heading:
            geom = self.predictor._design_geometry(sensor_pose(np.r_[xy, heading, 0.], self.cfg))
            self._geometries[index] = (heading, geom)
        else:
            geom = cached[1]
        return geom

    # Amortize model setup across a small batch of static viewpoints.
    def _gains(self, indices, mean, support):
        geometries, sizes = [], []
        for index in indices:
            geom = self._geometry(int(index), mean)
            columns = self.predictor._support_columns(geom, support)
            beams = geom.beams[columns]
            sizes.append(len(beams))
            if len(beams):
                hits = HitBatch(**{name: getattr(geom.background, name)[beams]
                                   for name in HitBatch.__dataclass_fields__})
                geometries.append(ScanGeometry(geom.pose, geom.fire_time[beams], geom.dir_world[beams],
                                               geom.w[beams], geom.origins[beams], hits,
                                               np.arange(len(beams))))
        if not geometries:
            return dict.fromkeys(map(int, indices), 0.)
        hits = HitBatch(**{name: np.concatenate([getattr(g.background, name) for g in geometries])
                           for name in HitBatch.__dataclass_fields__})
        arrays = {name: np.concatenate([getattr(g, name) for g in geometries])
                  for name in ('fire_time', 'dir_world', 'w', 'origins')}
        geom = ScanGeometry(geometries[0].pose, **arrays, background=hits, beams=np.arange(sum(sizes)))
        design = self.predictor.design(geom, support, self.kind)
        gains, offset = {}, 0
        fields = ('pi', 'mu', 'sigma', 'Z', 'uniform')
        for index, size in zip(indices, sizes, strict=True):
            part = slice(offset, offset + size)
            channel = replace(design, beams=design.beams[part],
                              **{field: getattr(design, field)[:, part] for field in fields})
            gains[int(index)] = self._information(channel)
            offset += size
        return gains

    def _gain(self, index, mean, support):
        return self._gains([index], mean, support)[index]

    def _information(self, design):
        # Predictor.design uses the default sigma_ref; honour this planner's config.
        if self.kind == "blind":
            design = replace(design, sigma=np.full_like(design.sigma, self.pc.sigma_ref))
        # Identical channels have BC=1 for every pair and contribute exactly zero.
        # Cull them before allocating the M by M by beam overlap arrays.
        varying = np.zeros(len(design.beams), dtype=bool)
        fields = ("pi", "mu", "sigma", "Z", "uniform")
        for field in fields:
            values = getattr(design, field)
            varying |= np.any(values != values[:1], axis=0)
        design = replace(design, beams=design.beams[varying],
                         **{field: getattr(design, field)[:, varying] for field in fields})
        return mi_lower_bound(pair_distance(design, SENSOR.r_min, SENSOR.r_max))

    # Select the F18 maximizer and follow its shortest path for one step.
    def act(self, k, x, belief, rng):
        support = belief.support(self.pc.M, rng)
        mean = belief.mean()
        source = self.grid.cell_of(x[:2])
        if source != self._source:
            self._travel = self.grid.travel_time(x[:2])
            self._source = source
        times, pred = self._travel
        stay = int(np.ravel_multi_index(source, self.grid.free.shape))
        candidates = np.union1d(self._lattice, [stay])
        candidates = candidates[np.isfinite(times.ravel()[candidates])]
        # Drop old off-lattice stay geometries; memory is O(number of lattice cells).
        for index in self._geometries.keys() - set(candidates):
            del self._geometries[index]
        best_stay = self._gain(stay, mean, support)
        best, gain, travel, score = stay, best_stay, 0., best_stay
        evaluated = 1
        previous = self._goal
        previous_result = None
        if previous is not None and previous in candidates:
            t = float(times.ravel()[previous])
            information = best_stay if previous == stay else self._gain(previous, mean, support)
            evaluated += int(previous != stay)
            previous_result = (previous, information, t, information - self.pc.lambda_t * t)
        order = candidates[np.lexsort((candidates, times.ravel()[candidates]))]
        gains = {}
        for position, index in enumerate(order):
            if index == stay:
                continue
            t = float(times.ravel()[index])
            # Strict inequality preserves score ties and their travel/index tie-breaks.
            if np.log(self.pc.M) - self.pc.lambda_t * t < score:
                break
            if previous_result is not None and index == previous:
                information = previous_result[1]
            else:
                if index not in gains:
                    batch = [int(i) for i in order[position:position + 16]
                             if i != stay and (previous_result is None or i != previous)
                             and np.log(self.pc.M) - self.pc.lambda_t * times.ravel()[i] >= score]
                    gains.update(self._gains(batch, mean, support))
                    evaluated += len(batch)
                information = gains[int(index)]
            value = information - self.pc.lambda_t * t
            if (value, -t, -int(index)) > (score, -travel, -best):
                best, gain, travel, score = int(index), information, t, value
        kept = previous_result is not None and score - previous_result[3] < self.pc.switch_margin
        if kept:
            best, gain, travel, score = previous_result
        switched = previous is not None and best != previous
        self._goal = best
        goal_cell = np.unravel_index(best, self.grid.free.shape)
        path = np.vstack([x[:2], self.grid.path(pred, goal_cell)])
        u, fallback = follow(self.world, self.cfg, x, path, face=mean[:2])
        return Action(u, goal=self.grid.centre(goal_cell), fallback=fallback,
                      info={"score": score, "info_gain": gain, "travel_time": travel,
                            "n_candidates": len(candidates), "n_evaluated": evaluated,
                            "best_stay": best_stay, "kind": self.kind,
                            "switched": bool(switched), "goal_kept": bool(kept)})
