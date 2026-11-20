# ==========================================================
# src/lidar_sim/core/types.py
# ==========================================================
# Frozen measurement records exchanged between layers (spec §3.1).
#
# Axis names: ``N`` generic batch, ``B`` beams, ``K`` sub-rays per beam, ``J`` optical returns per beam
# (J = K, or 2K with the specular flag), ``d`` spatial dimension. All arrays are stored read-only.

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields

import numpy as np

from lidar_sim.core.constants import WRAP_K_MAX
from lidar_sim.core.pose import Pose

PoseLike = Pose | Callable[[float], Pose]
_UNIT_TOL = 1e-9


# ------------------------------------------
# _freeze
# ------------------------------------------
# Copies an array field and makes it read-only.
def _freeze(obj, name: str, dtype=None) -> np.ndarray:
    arr = np.array(getattr(obj, name), dtype=dtype, copy=True)
    arr.setflags(write=False)
    object.__setattr__(obj, name, arr)
    return arr


# ------------------------------------------
# _expect
# ------------------------------------------
# Raises if an array does not have the expected shape.
def _expect(name: str, arr: np.ndarray, shape: tuple) -> None:
    if arr.shape != shape:
        raise ValueError(f"{name}: expected shape {shape}, got {arr.shape}")


# ------------------------------------------
# _check_dim
# ------------------------------------------
# Raises unless the spatial dimension is 2 or 3.
def _check_dim(name: str, d: int) -> None:
    if d not in (2, 3):
        raise ValueError(f"{name}: spatial dimension d must be 2 or 3, got {d}")


# ------------------------------------------
# _check_unit
# ------------------------------------------
# Raises unless the given vectors have unit length.
def _check_unit(name: str, v: np.ndarray, mask: np.ndarray | None = None) -> None:
    nrm = np.linalg.norm(v, axis=-1)
    if mask is not None:
        nrm = nrm[mask]
    if nrm.size and (not np.all(np.isfinite(nrm)) or np.abs(nrm - 1.0).max() > _UNIT_TOL):
        raise ValueError(f"{name}: vectors must be finite and unit-norm")


# ------------------------------------------
# _check_miss_ids
# ------------------------------------------
# Raises unless missed entries carry the id -1.
def _check_miss_ids(names: tuple[str, ...], obj, miss: np.ndarray) -> None:
    for name in names:
        if (getattr(obj, name)[miss] != -1).any():
            raise ValueError(f"{name}: must be -1 where hit is False")


# ------------------------------------------
# _Record
# ------------------------------------------
# Base of the frozen records, with a field dictionary for rebuilding them.
class _Record:
    # Field dict (for ``dataclasses.replace``-style reconstruction).
    def fields(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


# ------------------------------------------
# HitBatch
# ------------------------------------------
# Nearest-hit results for N rays (E5). Entries with ``hit=False`` are undefined except ids = −1.
@dataclass(frozen=True, eq=False)
class HitBatch(_Record):
    hit: np.ndarray          # (N,) bool
    r: np.ndarray            # (N,) [m] distance along the ray
    point: np.ndarray        # (N, d) [m] world frame
    normal: np.ndarray       # (N, d) unit, oriented so normal·dir <= 0
    material_id: np.ndarray  # (N,) int
    object_id: np.ndarray    # (N,) int
    surface_id: np.ndarray   # (N,) int, unique per segment/facet, −1 if no hit

    def __post_init__(self) -> None:
        hit = _freeze(self, "hit", bool)
        n = hit.shape[0]
        _expect("hit", hit, (n,))
        _expect("r", _freeze(self, "r", float), (n,))
        point = _freeze(self, "point", float)
        if point.ndim != 2 or point.shape[0] != n:
            raise ValueError(f"point: expected (N, d), got {point.shape}")
        d = point.shape[1]
        _check_dim("point", d)
        normal = _freeze(self, "normal", float)
        _expect("normal", normal, (n, d))
        _check_unit("normal", normal, hit)
        for name in ("material_id", "object_id", "surface_id"):
            _expect(name, _freeze(self, name, np.int64), (n,))
        _check_miss_ids(("surface_id", "object_id", "material_id"), self, ~hit)
        if not np.all(np.isfinite(self.r[hit])) or (self.r[hit] <= 0).any():
            raise ValueError("r: hit distances must be finite and positive")

    @property
    def n(self) -> int:
        return self.hit.shape[0]

    @property
    def dim(self) -> int:
        return self.point.shape[1]


# ------------------------------------------
# RadiometricScan
# ------------------------------------------
# Deterministic output of ``LidarSensor.trace`` (E1–E9), shared by simulator and analytic model.
@dataclass(frozen=True, eq=False)
class RadiometricScan(_Record):
    pose: PoseLike
    fire_time: np.ndarray    # (B,) [s]
    dir_world: np.ndarray    # (B, K, d) unit sub-ray directions
    w: np.ndarray            # (B, K) quadrature weights, rows sum to 1
    sub_idx: np.ndarray      # (B, J) int, sub-ray producing each return
    hit: np.ndarray          # (B, J) bool
    r_path: np.ndarray       # (B, J) [m] one-way optical path (NaN where no hit)
    power: np.ndarray        # (B, J) [W] received optical power (0 where no hit)
    normal: np.ndarray       # (B, J, d) at the final diffuse reflection
    cos_inc: np.ndarray      # (B, J) incidence cosine at the final diffuse reflection
    object_id: np.ndarray    # (B, J) int (−1 where no hit)
    material_id: np.ndarray  # (B, J) int
    surface_id: np.ndarray   # (B, J) int
    bounce: np.ndarray       # (B, J) bool, arrived via specular bounce
    P_amb: float = 0.0       # ambient power at detector [W]

    def __post_init__(self) -> None:
        if not (isinstance(self.pose, Pose) or callable(self.pose)):
            raise TypeError("pose: must be a Pose or a callable t -> Pose")
        dw = _freeze(self, "dir_world", float)
        if dw.ndim != 3:
            raise ValueError(f"dir_world: expected (B, K, d), got {dw.shape}")
        B, K, d = dw.shape
        _check_dim("dir_world", d)
        _check_unit("dir_world", dw)
        _expect("fire_time", _freeze(self, "fire_time", float), (B,))
        w = _freeze(self, "w", float)
        _expect("w", w, (B, K))
        if not np.allclose(w.sum(axis=1), 1.0, atol=1e-12) or (w < 0).any():
            raise ValueError("w: weights must be non-negative and sum to 1 per beam")
        sub = _freeze(self, "sub_idx", np.int64)
        if sub.ndim != 2 or sub.shape[0] != B:
            raise ValueError(f"sub_idx: expected (B, J), got {sub.shape}")
        J = sub.shape[1]
        if J not in (K, 2 * K):
            raise ValueError(f"sub_idx: return axis J must be K ({K}) or 2K (specular), got {J}")
        if not np.array_equal(sub, np.broadcast_to(np.arange(J) % K, (B, J))):
            raise ValueError("sub_idx: must equal j mod K (j < K direct, j >= K bounced)")
        hit = _freeze(self, "hit", bool)
        _expect("hit", hit, (B, J))
        r_path = _freeze(self, "r_path", float)
        _expect("r_path", r_path, (B, J))
        if not np.isnan(r_path[~hit]).all():
            raise ValueError("r_path: must be NaN where hit is False")
        if not np.all(np.isfinite(r_path[hit])) or (r_path[hit] <= 0).any():
            raise ValueError("r_path: must be finite and positive where hit is True")
        power = _freeze(self, "power", float)
        _expect("power", power, (B, J))
        if not np.all(np.isfinite(power)) or (power < 0).any() or (power[~hit] != 0).any():
            raise ValueError("power: must be finite, >= 0, and exactly 0 where hit is False")
        normal = _freeze(self, "normal", float)
        _expect("normal", normal, (B, J, d))
        _check_unit("normal", normal, hit)
        cos_inc = _freeze(self, "cos_inc", float)
        _expect("cos_inc", cos_inc, (B, J))
        if ((cos_inc < 0) | (cos_inc > 1)).any():
            raise ValueError("cos_inc: must lie in [0, 1]")
        for name in ("object_id", "material_id", "surface_id"):
            _expect(name, _freeze(self, name, np.int64), (B, J))
        _check_miss_ids(("object_id", "material_id", "surface_id"), self, ~hit)
        bounce = _freeze(self, "bounce", bool)
        _expect("bounce", bounce, (B, J))
        if bounce[:, :K].any():
            raise ValueError("bounce: direct slots j < K must have bounce=False")
        if J == 2 * K and not np.array_equal(bounce[:, K:], hit[:, K:]):
            raise ValueError("bounce: bounced slots j >= K must have bounce == hit (unused slot = miss)")
        if self.P_amb < 0:
            raise ValueError("P_amb: must be >= 0")
        object.__setattr__(self, "P_amb", float(self.P_amb))

    @property
    def B(self) -> int:
        return self.dir_world.shape[0]

    @property
    def K(self) -> int:
        return self.dir_world.shape[1]

    @property
    def J(self) -> int:
        return self.sub_idx.shape[1]

    @property
    def dim(self) -> int:
        return self.dir_world.shape[2]

    # All-miss record (tests / builder base). ``J`` defaults to K, ``pose`` to identity.
    @classmethod
    def empty(cls, B: int, K: int, J: int | None = None, d: int = 2, pose: PoseLike | None = None,
              P_amb: float = 0.0) -> RadiometricScan:
        J = K if J is None else J
        _check_dim("d", d)
        pose = Pose.identity(d) if pose is None else pose
        dw = np.zeros((B, K, d))
        dw[..., 0] = 1.0
        return cls(pose=pose, fire_time=np.zeros(B), dir_world=dw, w=np.full((B, K), 1.0 / K),
                   sub_idx=np.tile(np.arange(J) % K, (B, 1)), hit=np.zeros((B, J), bool),
                   r_path=np.full((B, J), np.nan), power=np.zeros((B, J)),
                   normal=np.zeros((B, J, d)), cos_inc=np.zeros((B, J)),
                   object_id=np.full((B, J), -1), material_id=np.full((B, J), -1),
                   surface_id=np.full((B, J), -1), bounce=np.zeros((B, J), bool), P_amb=P_amb)


# ------------------------------------------
# Scan
# ------------------------------------------
# What the sensor reports (E15). ``range`` is NaN iff ``detected`` is False.
@dataclass(frozen=True, eq=False)
class Scan(_Record):
    detected: np.ndarray     # (B,) bool
    range: np.ndarray        # (B,) [m]
    amplitude: np.ndarray    # (B, 2) [A] Â at f1, f2 (diagnostics; not used by the likelihood)
    range_raw: np.ndarray    # (B,) [m] fused R̂ before the detection/window gate (V&V only)
    fire_time: np.ndarray    # (B,) [s]
    truth: RadiometricScan   # evaluation only

    def __post_init__(self) -> None:
        det = _freeze(self, "detected", bool)
        B = det.shape[0]
        rng_ = _freeze(self, "range", float)
        _expect("range", rng_, (B,))
        if not np.array_equal(np.isnan(rng_), ~det):
            raise ValueError("range: must be NaN exactly where detected is False")
        if not np.all(np.isfinite(rng_[det])):
            raise ValueError("range: detected ranges must be finite")
        _expect("amplitude", _freeze(self, "amplitude", float), (B, 2))
        _expect("range_raw", _freeze(self, "range_raw", float), (B,))
        _expect("fire_time", _freeze(self, "fire_time", float), (B,))
        if self.truth.B != B:
            raise ValueError("truth: beam count mismatch")
        if not np.array_equal(self.fire_time, self.truth.fire_time):
            raise ValueError("fire_time: must equal truth.fire_time")

    @property
    def B(self) -> int:
        return self.detected.shape[0]


# ------------------------------------------
# BeamPrediction
# ------------------------------------------
# Analytic per-beam prediction (E16–E19) from a RadiometricScan only (ADR-0005).
@dataclass(frozen=True, eq=False)
class BeamPrediction(_Record):
    A: np.ndarray            # (B, 2) [A] noise-free resultant amplitude at f1, f2
    sigma_n: np.ndarray      # (B, 2) [A] per-sample noise std
    snr: np.ndarray          # (B, 2) per-sample SNR A²/(2σ_n²)
    P_D: np.ndarray          # (B,) detection probability (E16)
    P_FA: np.ndarray         # (B,) no-signal false-alarm probability (E16; = P_FA0 under CFAR)
    r_bar: np.ndarray        # (B,) [m] expected (mixed-pixel-aware) range (E19)
    sigma_R: np.ndarray      # (B,) [m] fused range std (E17)
    out_prob: np.ndarray     # (B, 6) wrap-outlier probabilities, k = −3..−1, +1..+3 (E18)
    out_shift: np.ndarray    # (B, 6) [m] corresponding range shifts kΔ
    hit_any: np.ndarray      # (B,) bool, selects the E20 branch
    mixed: np.ndarray        # (B,) bool (E19)
    smooth: np.ndarray       # (B,) bool, eligible for E22–E24

    def __post_init__(self) -> None:
        A = _freeze(self, "A", float)
        if A.ndim != 2 or A.shape[1] != 2:
            raise ValueError(f"A: expected (B, 2), got {A.shape}")
        B = A.shape[0]
        for name in ("sigma_n", "snr"):
            _expect(name, _freeze(self, name, float), (B, 2))
        for name in ("P_D", "P_FA"):
            p = _freeze(self, name, float)
            _expect(name, p, (B,))
            if ((p < 0) | (p > 1)).any():
                raise ValueError(f"{name}: probabilities must lie in [0, 1]")
        for name in ("r_bar", "sigma_R"):
            _expect(name, _freeze(self, name, float), (B,))
        op = _freeze(self, "out_prob", float)
        _expect("out_prob", op, (B, 2 * WRAP_K_MAX))
        if (op < 0).any() or (op.sum(axis=1) > 1 + 1e-12).any():
            raise ValueError("out_prob: must be >= 0 with row sums <= 1")
        _expect("out_shift", _freeze(self, "out_shift", float), op.shape)
        for name in ("hit_any", "mixed", "smooth"):
            _expect(name, _freeze(self, name, bool), (B,))

    @property
    def B(self) -> int:
        return self.A.shape[0]

    @property
    def n_out(self) -> int:
        return self.out_prob.shape[1]


# ------------------------------------------
# MLEResult
# ------------------------------------------
# Pose maximum-likelihood result (E25).
@dataclass(frozen=True)
class MLEResult:
    xi: np.ndarray
    success: bool
    n_iter: int
    loglik: float
    message: str
