# ==========================================================
# src/lidar_sim/sensor/lidar.py
# ==========================================================
# Virtual AMCW LiDAR: deterministic optical trace (E1–E9) and waveform measurement (E10–E15).
#
# ``trace`` produces the :class:`RadiometricScan` shared by the waveform simulator (``measure``) and the
# analytic model. The trace itself is dimension-agnostic: geometry enters only through the
# ``ScanPattern`` and ``Scene`` protocols.

from __future__ import annotations

import numpy as np

from lidar_sim.core.config import EnvConfig, NoiseConfig, SensorConfig
from lidar_sim.core.pose import Pose
from lidar_sim.core.types import PoseLike, RadiometricScan, Scan
from lidar_sim.geometry.base import Scene
from lidar_sim.geometry.scene2d import Scene2D
from lidar_sim.physics.atmosphere import extinction
from lidar_sim.physics.radiometry import incidence_cos, lambertian_power
from lidar_sim.physics.specular import bounce_power, reflect
from lidar_sim.sensor.scan_pattern import Planar2DPattern, ScanPattern
from lidar_sim.world.materials import MATERIALS, Material
from lidar_sim.world.scenes import SceneSpec

_DEFAULT_SENSOR = SensorConfig()
_DEFAULT_NOISE = NoiseConfig()


# ------------------------------------------
# _beam_poses
# ------------------------------------------
# Per-beam rotation ``(B, d, d)`` and translation ``(B, d)`` (E2: ``pose(τ_i)`` if callable).
def _beam_poses(pose: PoseLike, tau: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(pose, Pose):
        B = len(tau)
        return np.broadcast_to(pose.R, (B, *pose.R.shape)), np.broadcast_to(pose.t, (B, *pose.t.shape))
    if not callable(pose):
        raise TypeError("pose must be a Pose or a callable t -> Pose")
    poses = [pose(float(t)) for t in tau]
    return np.stack([p.R for p in poses]), np.stack([p.t for p in poses])


# ------------------------------------------
# _lookup
# ------------------------------------------
# Vectorized per-entry material property; entries with id −1 (misses) get 0.
def _lookup(materials: dict[int, Material], ids: np.ndarray, attr) -> np.ndarray:
    out = np.zeros(ids.shape, dtype=float)
    for mid in np.unique(ids[ids >= 0]):
        if int(mid) not in materials:
            raise KeyError(f"material id {int(mid)} not defined (scene.materials / sensor.materials)")
        out[ids == mid] = attr(materials[int(mid)])
    return out


# ------------------------------------------
# scan_rays
# ------------------------------------------
# Ray set of one scan (E1, E2, E4).
#
# Returns ``(tau (B,), dir_world (B, K, d), w (B, K), origins (B·K, d))``: absolute fire times, unit
# world-frame sub-ray directions, quadrature weights and per-sub-ray origins (beam i uses ``pose(τ_i)``).
def scan_rays(pattern: ScanPattern, K: int, pose: PoseLike, t0: float = 0.0):
    dirs_s, w = pattern.subrays(K)                             # (B, K, d), (B, K)
    tau = float(t0) + np.asarray(pattern.fire_times(), dtype=float)
    B, _, d = dirs_s.shape
    Rb, tb = _beam_poses(pose, tau)
    dw = np.einsum("bij,bkj->bki", Rb, dirs_s)
    dw /= np.linalg.norm(dw, axis=-1, keepdims=True)
    o = np.broadcast_to(tb[:, None, :], (B, K, d)).reshape(-1, d)
    return tau, dw, np.asarray(w, dtype=float), o


# ------------------------------------------
# _direct_slot
# ------------------------------------------
# Direct-return slot arrays (E6–E8) for flattened sub-rays and their nearest hits ``h``.
def _direct_slot(dflat: np.ndarray, wflat: np.ndarray, h, mats: dict, sigma: float, nz: NoiseConfig) -> dict:
    N = dflat.shape[0]
    hit = np.array(h.hit, dtype=bool, copy=True)
    n1 = np.where(hit[:, None], h.normal, 0.0)
    cos1 = np.where(hit, incidence_cos(dflat, n1), 0.0)
    rho1 = _lookup(mats, np.where(hit, h.material_id, -1), lambda m: m.rho_diffuse)
    r1 = np.where(hit, h.r, np.nan)
    P1 = np.zeros(N)
    P1[hit] = lambertian_power(nz.P0, wflat[hit], rho1[hit], cos1[hit], r1[hit], nz.A_rx, nz.eta, sigma)
    return {"hit": hit, "r": r1, "P": P1, "n": n1, "cos": cos1, "obj": h.object_id,
            "mat": h.material_id, "sid": h.surface_id, "bounce": np.zeros(N, bool)}


# ------------------------------------------
# _assemble
# ------------------------------------------
# Stacks the per-slot trace arrays into one radiometric scan.
def _assemble(pose: PoseLike, tau: np.ndarray, dw: np.ndarray, w: np.ndarray, slots: list[dict],
              P_amb: float) -> RadiometricScan:
    B, K, d = dw.shape

    def stack(key, extra=()):
        return np.concatenate([s[key].reshape(B, K, *extra) for s in slots], axis=1)

    J = K * len(slots)
    return RadiometricScan(
        pose=pose, fire_time=tau, dir_world=dw, w=np.asarray(w, dtype=float),
        sub_idx=np.tile(np.arange(J) % K, (B, 1)), hit=stack("hit"), r_path=stack("r"),
        power=stack("P"), normal=stack("n", (d,)), cos_inc=stack("cos"), object_id=stack("obj"),
        material_id=stack("mat"), surface_id=stack("sid"), bounce=stack("bounce"), P_amb=P_amb)


# ------------------------------------------
# radiometric_from_hits
# ------------------------------------------
# Radiometric record from precomputed nearest hits (E6–E8; direct returns only, no E9).
#
# ``hits`` holds one entry per sub-ray in C order of ``dir_world`` (B·K rays). This is the function
# ``LidarSensor.trace`` uses for its direct returns, exposed so that predictors that obtain the hits
# differently (final-spec §5.2) share the same radiometry.
def radiometric_from_hits(pose: PoseLike, fire_time: np.ndarray, dir_world: np.ndarray, w: np.ndarray, hits,
                          materials: dict, sigma_ext: float, noise: NoiseConfig, P_amb: float) -> RadiometricScan:
    d = dir_world.shape[2]
    slot = _direct_slot(dir_world.reshape(-1, d), np.asarray(w, dtype=float).reshape(-1), hits, materials,
                        sigma_ext, noise)
    return _assemble(pose, np.asarray(fire_time, dtype=float), dir_world, w, [slot], P_amb)


# ------------------------------------------
# LidarSensor
# ------------------------------------------
# AMCW LiDAR (URG-04LX-inspired virtual sensor); 2D by default, 3D with a 3D pattern and scene.
#
# Parameters
# sensor : SensorConfig
#     Geometry, timing, divergence, modulation; ``specular`` enables E9 returns (J = 2K).
# noise : NoiseConfig
#     Radiometric gains (``P0``, ``A_rx``, ``eta``) and noise constants.
# pattern : ScanPattern or None
#     Defaults to ``Planar2DPattern(sensor)``.
# materials : dict[int, Material]
#     Fallback material table, used if the scene has no ``materials`` attribute.
class LidarSensor:
    def __init__(self, sensor: SensorConfig = _DEFAULT_SENSOR, noise: NoiseConfig = _DEFAULT_NOISE,
                 pattern: ScanPattern | None = None, materials: dict = MATERIALS) -> None:
        self.sensor = sensor
        self.noise = noise
        self.pattern = Planar2DPattern(sensor) if pattern is None else pattern
        self.materials = dict(materials)

    # Deterministic optical trace of one scan.
    #
    # Implements (E1, E2) beam layout and per-beam pose ``pose(τ_i)`` with scene time ``τ_i``;
    # (E4) K Gauss–Hermite sub-rays per beam; (E5) nearest-hit ray casting; (E6) incidence cosine;
    # (E7, E8) Lambertian power with two-way atmospheric transmission (direct diffuse returns use
    # ``Material.rho_diffuse``, i.e. ``ρ_d`` for mirrors); (E9, flag ``specular``) one specular
    # bounce per mirror hit.
    #
    # Return axis (J): ``J = K`` without the flag, ``J = 2K`` with it. Slot ``j < K`` is the direct
    # return of sub-ray ``j``; slot ``j ≥ K`` is the bounced return of sub-ray ``j − K``, filled only
    # when the direct hit is a mirror and the reflected ray (origin offset by ``r_eps`` along
    # ``d'``) hits a non-mirror surface; otherwise it is an unused miss. Bounced slots carry
    # ``r_path = r₁ + r₂`` and the normal / cosine / ids of the second (diffuse) surface.
    #
    # Parameters
    # pose : Pose or Callable[[float], Pose]
    #     Sensor pose (sensor → world), or a trajectory evaluated at each fire time τ_i [s].
    # scene : Scene or SceneSpec
    #     World geometry; a ``SceneSpec`` is wrapped with ``Scene2D.from_spec``. Its ``materials``
    #     dict (if present) supplies reflectivities.
    # env : EnvConfig
    #     Visibility (E8) and ambient power (copied to ``P_amb``).
    # t0 : float
    #     Scan start time [s]. Beam i fires at ``t0 + τ_i``; trajectories and movers are evaluated at
    #     that absolute time and ``fire_time`` reports it (consecutive scans: ``t0 = k·T_scan``).
    #
    # Returns
    # RadiometricScan
    def trace(self, pose: PoseLike, scene: Scene | SceneSpec, env: EnvConfig, t0: float = 0.0) -> RadiometricScan:
        if isinstance(scene, SceneSpec):
            scene = Scene2D.from_spec(scene)
        K = self.sensor.n_subrays
        tau, dw, w, o = scan_rays(self.pattern, K, pose, t0)
        d = dw.shape[2]
        dflat = dw.reshape(-1, d)
        ts = np.repeat(tau, K)
        wflat = w.reshape(-1)
        mats = getattr(scene, "materials", None) or self.materials
        sigma = extinction(env.visibility_m, env.wavelength)

        h = scene.intersect(o, dflat, ts)
        slots = [_direct_slot(dflat, wflat, h, mats, sigma, self.noise)]
        if self.sensor.specular:
            slots.append(self._bounce(scene, mats, h, dflat, ts, wflat, sigma, self.sensor.r_eps))
        return _assemble(pose, tau, dw, w, slots, env.P_amb)

    # Bounced-return slot arrays (E9) for flattened sub-rays (see ``trace``).
    def _bounce(self, scene, mats, h, dflat, ts, wflat, sigma, r_eps) -> dict:
        N, d = dflat.shape
        nz = self.noise
        is_mirror = h.hit & (_lookup(mats, np.where(h.hit, h.material_id, -1),
                                     lambda m: 1.0 if m.kind == "mirror" else 0.0) > 0)
        out = {"hit": np.zeros(N, bool), "r": np.full(N, np.nan), "P": np.zeros(N), "n": np.zeros((N, d)),
               "cos": np.zeros(N), "obj": np.full(N, -1, np.int64), "mat": np.full(N, -1, np.int64),
               "sid": np.full(N, -1, np.int64), "bounce": np.zeros(N, bool)}
        idx = np.flatnonzero(is_mirror)
        if idx.size == 0:
            return out
        d2 = reflect(dflat[idx], h.normal[idx])
        d2 /= np.linalg.norm(d2, axis=-1, keepdims=True)
        o2 = h.point[idx] + r_eps * d2
        h2 = scene.intersect(o2, d2, ts[idx])
        mat2 = np.where(h2.hit, h2.material_id, -1)
        diffuse2 = h2.hit & (_lookup(mats, mat2, lambda m: 0.0 if m.kind == "mirror" else 1.0) > 0)
        sel = idx[diffuse2]
        if sel.size == 0:
            return out
        n2 = h2.normal[diffuse2]
        cos2 = incidence_cos(d2[diffuse2], n2)
        r_path = h.r[sel] + r_eps + h2.r[diffuse2]
        rho_m = _lookup(mats, h.material_id[sel], lambda m: m.rho_m)
        rho2 = _lookup(mats, mat2[diffuse2], lambda m: m.rho_diffuse)
        out["hit"][sel] = True
        out["bounce"][sel] = True
        out["r"][sel] = r_path
        out["P"][sel] = bounce_power(nz.P0, wflat[sel], rho_m, rho2, cos2, r_path, nz.A_rx, nz.eta, sigma)
        out["n"][sel] = n2
        out["cos"][sel] = cos2
        out["obj"][sel] = h2.object_id[diffuse2]
        out["mat"][sel] = h2.material_id[diffuse2]
        out["sid"][sel] = h2.surface_id[diffuse2]
        return out

    # Waveform-level measurement (E10–E15), delegated to ``physics.waveform.measure``.
    #
    # ``env`` is unused (ambient light is already in ``trace.P_amb``).
    def measure(self, trace: RadiometricScan, env: EnvConfig, rng: np.random.Generator) -> Scan:
        from lidar_sim.physics import waveform  # imported here: only measurement needs the simulator

        return waveform.measure(trace, self.sensor, self.noise, rng)

    # ``measure(trace(pose, scene, env, t0), env, rng)`` (E1–E15).
    def scan(self, pose: PoseLike, scene: Scene | SceneSpec, env: EnvConfig, rng: np.random.Generator,
             t0: float = 0.0) -> Scan:
        return self.measure(self.trace(pose, scene, env, t0), env, rng)
