# ==========================================================
# src/mts/predict.py
# ==========================================================
# Cached background and chunked hypothesis prediction (F9, F14; spec §5.2).

from dataclasses import dataclass, replace

import numpy as np
from scipy.special import ndtr

from lidar_sim.core.config import EnvConfig
from lidar_sim.core.types import HitBatch, PoseLike, RadiometricScan, Scan
from lidar_sim.geometry.scene3d import ray_box_batch
from lidar_sim.physics.atmosphere import extinction
from lidar_sim.physics.radiometry import incidence_cos, lambertian_power
from lidar_sim.sensor.lidar import radiometric_from_hits, scan_rays
from mts.config import NOISE, SENSOR, PlannerConfig, make_model, make_sensor
from mts.world import TARGET_OBJECT, World


# ------------------------------------------
# Observation
# ------------------------------------------
# Observation-only public scan fields (F9).
@dataclass(frozen=True)
class Observation:
    detected: np.ndarray
    range: np.ndarray
    fire_time: np.ndarray

    # Copy only the three public measurement arrays.
    @classmethod
    def from_scan(cls, scan: Scan):
        return cls(scan.detected.copy(), scan.range.copy(), scan.fire_time.copy())


# ------------------------------------------
# ScanGeometry
# ------------------------------------------
# Reusable rays and background hits (F2, spec §5.2).
@dataclass(frozen=True)
class ScanGeometry:
    pose: PoseLike
    fire_time: np.ndarray
    dir_world: np.ndarray
    w: np.ndarray
    origins: np.ndarray
    background: HitBatch
    beams: np.ndarray


# ------------------------------------------
# BeamBatch
# ------------------------------------------
# E16–E19 arrays on the hypothesis and candidate-beam axes.
@dataclass(frozen=True)
class BeamBatch:
    beams: np.ndarray
    P_D: np.ndarray
    P_FA: np.ndarray
    r_bar: np.ndarray
    sigma_R: np.ndarray
    hit_any: np.ndarray
    out_prob: np.ndarray
    out_shift: np.ndarray
    on_target: np.ndarray


# ------------------------------------------
# DesignBatch
# ------------------------------------------
# Report probabilities and conditional densities (F14).
@dataclass(frozen=True)
class DesignBatch:
    beams: np.ndarray
    pi: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    Z: np.ndarray
    uniform: np.ndarray


# ------------------------------------------
# Predictor
# ------------------------------------------
# Predict from background geometry and hypothesised target centres only.
class Predictor:
    def __init__(self, world: World, env: EnvConfig, chunk: int = 200_000):
        if chunk < 1:
            raise ValueError("chunk must be positive")
        self.world, self.env, self.chunk = world, env, int(chunk)
        self.background = world.background_scene()
        self.sigma_ext = extinction(env.visibility_m, env.wavelength)
        self._background_cache = None
        self._trace_cache = None

    # Cache rays, remap ground ids and cull by the target envelope (§5.2).
    def geometry(self, pose: PoseLike, t0: float = 0.0, K: int = 9) -> ScanGeometry:
        sensor = make_sensor(K)
        tau, dw, w, origins = scan_rays(sensor.pattern, K, pose, t0)
        hits = self.background.intersect(origins, dw.reshape(-1, 3), np.repeat(tau, K))
        sid = np.where(hits.surface_id == 6 * self.world.n_boxes, 6 * (self.world.n_boxes + 1), hits.surface_id)
        hits = replace(hits, surface_id=sid)
        lo, hi = self.world.envelope()
        r, _ = ray_box_batch(origins, dw.reshape(-1, 3), lo, hi, self.background.r_eps)
        beams = np.flatnonzero(np.isfinite(r).reshape(-1, K).any(axis=1))
        return ScanGeometry(pose, tau, dw, w, origins, hits, beams)

    # Static K=1 design geometry, tracing background only inside the envelope.
    #
    # Compact beam indices are private to planning; discarded beams have the
    # same channel for every hypothesis and contribute zero information.
    def _design_geometry(self, pose):
        tau, dw, w, origins = scan_rays(make_sensor(1).pattern, 1, pose, 0.)
        lo, hi = self.world.envelope()
        r, _ = ray_box_batch(origins, dw.reshape(-1, 3), lo, hi, self.background.r_eps)
        keep = np.flatnonzero(np.isfinite(r).ravel())
        tau, dw, w, origins = tau[keep], dw[keep], w[keep], origins[keep]
        hits = self.background.intersect(origins, dw.reshape(-1, 3), tau)
        hits = replace(hits, surface_id=np.where(hits.surface_id == 6 * self.world.n_boxes,
                                                6 * (self.world.n_boxes + 1), hits.surface_id))
        return ScanGeometry(pose, tau, dw, w, origins, hits, np.arange(len(keep)))

    # Intersect a rectangular batch, then flatten pairs into synthetic beams (F2).
    def _trace_batch(self, geom, q, beams):
        K = geom.w.shape[1]
        idx = (beams[:, None] * K + np.arange(K)).ravel()
        o, d = geom.origins[idx], geom.dir_world[beams].reshape(-1, 3)
        lo, hi = self.world.target_box(q)
        r, face = ray_box_batch(o, d, lo, hi, self.background.r_eps)
        bg = geom.background
        bg_r = np.where(bg.hit[idx], bg.r[idx], np.inf)
        target = np.isfinite(r) & ((r < bg_r) | ((r == bg_r) &
                                  (bg.surface_id[idx] == 6 * (self.world.n_boxes + 1))))
        shape = r.shape
        directions = np.broadcast_to(d, (*shape, 3)).reshape(-1, 3)
        normal = np.zeros((*shape, 3))
        axis = np.maximum(face, 0) // 2
        np.put_along_axis(normal, axis[..., None], np.where(face % 2 == 0, -1., 1.)[..., None], axis=-1)
        normal = normal.reshape(-1, 3)
        normal[np.einsum('ij,ij->i', normal, directions) > 0] *= -1
        normal = np.where(target.ravel()[:, None], normal,
                          np.broadcast_to(bg.normal[idx], (*shape, 3)).reshape(-1, 3))
        rr = np.where(target, r, bg.r[idx]).ravel()
        hit = (target | bg.hit[idx]).ravel()
        point = np.where(hit[:, None], np.tile(o, (len(q), 1)) + np.where(hit, rr, 0)[:, None] * directions, 0)
        hits = HitBatch(hit, rr, point, normal,
                        np.where(target, self.world.target_material, bg.material_id[idx]).ravel(),
                        np.where(target, TARGET_OBJECT, bg.object_id[idx]).ravel(),
                        np.where(target, 6 * self.world.n_boxes + face, bg.surface_id[idx]).ravel())
        return radiometric_from_hits(
            geom.pose, np.tile(geom.fire_time[beams], len(q)), directions.reshape(-1, K, 3),
            np.tile(geom.w[beams], (len(q), 1)), hits, self.background.materials,
            self.sigma_ext, NOISE, self.env.P_amb)

    # Full scan for one hypothesis, sharing E6–E8 with the sensor.
    def trace(self, geom: ScanGeometry, q) -> RadiometricScan:
        return self._trace_batch(geom, np.asarray(q, float).reshape(1, 3), np.arange(len(geom.fire_time)))

    # Envelope candidates intersecting the bounding box of this support.
    def _support_columns(self, geom, q):
        if not len(q) or not len(geom.beams):
            return np.empty(0, int)
        lo, hi = self.world.target_box(q)
        K = geom.w.shape[1]
        idx = (geom.beams[:, None] * K + np.arange(K)).ravel()
        r, _ = ray_box_batch(geom.origins[idx], geom.dir_world.reshape(-1, 3)[idx],
                             lo.min(axis=0), hi.max(axis=0), self.background.r_eps)
        return np.flatnonzero(np.isfinite(r).reshape(-1, K).any(axis=1))

    # F2 with separate coordinate arrays, avoiding a P × rays × 3 temporary.
    #
    # Strict updates retain the first axis on ties, as in ray_box_batch.
    # Arithmetic within each slab is identical to the reference implementation.
    def _box_hits(self, origins, directions, lo, hi):
        shape = (len(lo), len(origins))
        near, far = np.full(shape, -np.inf), np.full(shape, np.inf)
        near_face, far_face = np.zeros(shape, int), np.zeros(shape, int)
        for axis in range(3):
            o, d = origins[:, axis], directions[:, axis]
            parallel = d == 0
            inv = 1. / np.where(parallel, 1., d)
            low = (lo[:, axis, None] - o) * inv
            high = (hi[:, axis, None] - o) * inv
            minus, plus = np.minimum(low, high), np.maximum(low, high)
            inside = (o >= lo[:, axis, None]) & (o <= hi[:, axis, None])
            minus = np.where(parallel, np.where(inside, -np.inf, np.inf), minus)
            plus = np.where(parallel, np.where(inside, np.inf, -np.inf), plus)
            near_face = np.where(minus > near, 2 * axis + (d < 0), near_face)
            far_face = np.where(plus < far, 2 * axis + (d > 0), far_face)
            near, far = np.maximum(near, minus), np.minimum(far, plus)
        hit = (near <= far) & (far > self.background.r_eps)
        enter = near > self.background.r_eps
        return (np.where(hit, np.where(enter, near, far), np.inf),
                np.where(hit, np.where(enter, near_face, far_face), -1))

    def _background_prediction(self, geom, model):
        # Belief tempering evaluates many supports against the same scan. Keep one
        # geometry alive, never an unbounded cache of moving scans.
        cached = self._background_cache
        if cached is None or cached[0] is not geom:
            cached = (geom, model.predict(self._pair_trace(geom, geom.beams), self.env))
            self._background_cache = cached
        return cached[1]

    # Yield only nearest-target pairs; cull rays against this support first.
    def _pairs(self, geom, q):
        K = geom.w.shape[1]
        if not len(q) or not len(geom.beams):
            return
        lo, hi = self.world.target_box(q)
        candidates = self._support_columns(geom, q)
        # Bound both slab working memory and the number of synthetic beams.
        limit = min(self.chunk, 8192)
        for start in range(0, len(candidates), limit):
            columns = candidates[start:start + limit]
            beams = geom.beams[columns]
            idx = (beams[:, None] * K + np.arange(K)).ravel()
            bg = geom.background
            bg_r = np.where(bg.hit[idx], bg.r[idx], np.inf)
            for p in range(0, len(q), max(1, limit // len(beams))):
                stop = min(len(q), p + max(1, limit // len(beams)))
                r, face = self._box_hits(geom.origins[idx], geom.dir_world[beams].reshape(-1, 3),
                                         lo[p:stop], hi[p:stop])
                target = np.isfinite(r) & ((r < bg_r) | ((r == bg_r) &
                                          (bg.surface_id[idx] == 6 * (self.world.n_boxes + 1))))
                shape = (stop - p, len(beams), K)
                rows, cols = np.nonzero(target.reshape(shape).any(axis=-1))
                if len(rows):
                    yield (p + rows, columns[cols], r.reshape(shape)[rows, cols],
                           face.reshape(shape)[rows, cols], target.reshape(shape)[rows, cols])

    # Background radiometry, computed once per geometry.
    def _pair_trace(self, geom, beams):
        K = geom.w.shape[1]
        idx = (beams[:, None] * K + np.arange(K)).ravel()
        bg = geom.background
        fields = ('hit', 'r', 'point', 'normal', 'material_id', 'object_id', 'surface_id')
        arrays = {name: getattr(bg, name)[idx] for name in fields}
        return radiometric_from_hits(geom.pose, geom.fire_time[beams], geom.dir_world[beams],
                                     geom.w[beams], HitBatch(**arrays), self.background.materials,
                                     self.sigma_ext, NOISE, self.env.P_amb)

    # Overwrite background direct-return arrays for the sparse target pairs.
    def _pair_prediction(self, geom, cols, r, face, target, model):
        cached = self._trace_cache
        if cached is None or cached[0] is not geom:
            cached = (geom, self._pair_trace(geom, geom.beams))
            self._trace_cache = cached
        bg = cached[1]
        hit, ranges, power, sid = (getattr(bg, name)[cols].copy()
                                    for name in ('hit', 'r_path', 'power', 'surface_id'))
        beams = geom.beams[cols]
        selected_face = face[target]
        normal = np.zeros((len(selected_face), 3))
        np.put_along_axis(normal, (selected_face // 2)[:, None],
                          np.where(selected_face % 2 == 0, -1., 1.)[:, None], axis=1)
        d = geom.dir_world[beams][target]
        normal[np.einsum('ij,ij->i', normal, d) > 0] *= -1
        rho = self.background.materials[self.world.target_material].rho_diffuse
        power[target] = lambertian_power(NOISE.P0, geom.w[beams][target], rho, incidence_cos(d, normal),
                                          r[target], NOISE.A_rx, NOISE.eta, self.sigma_ext)
        hit[target], ranges[target] = True, r[target]
        sid[target] = 6 * self.world.n_boxes + selected_face
        return model.predict_direct_arrays(hit, ranges, power, geom.w[beams], sid, self.env.P_amb)

    # Fill from the background, evaluating E16–E19 only for target pairs.
    def predict(self, geom: ScanGeometry, q) -> BeamBatch:
        q = np.asarray(q, float).reshape(-1, 3)
        shape = (len(q), len(geom.beams))
        model = make_model(geom.w.shape[1])
        background = self._background_prediction(geom, model)
        arrays = {}
        for name in BeamBatch.__dataclass_fields__:
            if name == 'beams':
                continue
            if name == 'on_target':
                arrays[name] = np.zeros(shape, bool)
            else:
                value = getattr(background, name)
                arrays[name] = np.broadcast_to(value, (len(q), *value.shape)).copy()
        for rows, cols, r, face, target in self._pairs(geom, q):
            pred = self._pair_prediction(geom, cols, r, face, target, model)
            for name, arr in arrays.items():
                arr[rows, cols] = True if name == 'on_target' else getattr(pred, name)
        return BeamBatch(geom.beams, **arrays)

    # E20 including the hypothesis-independent background contribution.
    def loglik(self, geom: ScanGeometry, obs: Observation, q) -> np.ndarray:
        q = np.asarray(q, float).reshape(-1, 3)
        model = make_model(geom.w.shape[1])
        background = self._background_prediction(geom, model)
        bg_ll = model.log_likelihood_arrays(obs.detected[geom.beams], obs.range[geom.beams], background)
        # Retain beam order for the same floating-point reduction as the dense path.
        ll = np.broadcast_to(bg_ll, (len(q), len(geom.beams))).copy()
        for rows, cols, r, face, target in self._pairs(geom, q):
            beams = geom.beams[cols]
            pred = self._pair_prediction(geom, cols, r, face, target, model)
            ll[rows, cols] = model.log_likelihood_arrays(obs.detected[beams], obs.range[beams], pred)
        return ll.sum(axis=1)

    # Design channel (F14); the blind baseline uses untruncated Gaussians (§7.1).
    def design(self, geom: ScanGeometry, q, kind: str = 'aware') -> DesignBatch:
        if geom.w.shape[1] != 1:
            raise ValueError("design requires K=1 geometry")
        if kind not in ('aware', 'blind'):
            raise ValueError("kind must be aware or blind")
        if kind == 'blind':
            shape = (np.asarray(q).reshape(-1, 3).shape[0], len(geom.beams))
            mu, pi = np.zeros(shape), np.zeros(shape)
            r = np.where(geom.background.hit[geom.beams], geom.background.r[geom.beams], np.nan)
            mu[:] = np.nan_to_num(r)
            pi[:] = (r >= SENSOR.r_min) & (r <= SENSOR.r_max)
            for rows, cols, r, _, _ in self._pairs(geom, np.asarray(q, float).reshape(-1, 3)):
                r = r[:, 0]
                mu[rows, cols] = np.nan_to_num(r)
                pi[rows, cols] = (r >= SENSOR.r_min) & (r <= SENSOR.r_max)
            return DesignBatch(geom.beams, pi, mu, np.full(shape, PlannerConfig().sigma_ref),
                               np.ones(shape), np.zeros(shape, bool))
        pred = self.predict(geom, q)
        a = (SENSOR.r_min - pred.r_bar) / pred.sigma_R
        b = (SENSOR.r_max - pred.r_bar) / pred.sigma_R
        Z = np.where(a > 0, ndtr(-a) - ndtr(-b), ndtr(b) - ndtr(a))
        pi = np.where(pred.hit_any, pred.P_D * Z, pred.P_FA * (SENSOR.r_max - SENSOR.r_min) / SENSOR.Lc)
        return DesignBatch(geom.beams, np.clip(pi, 0, 1), pred.r_bar, pred.sigma_R, Z, ~pred.hit_any)
