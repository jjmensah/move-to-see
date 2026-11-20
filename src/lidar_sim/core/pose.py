# ==========================================================
# src/lidar_sim/core/pose.py
# ==========================================================
# Rigid poses in SE(d) with a 2D minimal parameterization ξ = (x, y, ψ) (spec §2).
#
# ``p_W = R p_S + t``. Only :meth:`Pose.from_xi`, :meth:`Pose.xi` and :func:`rot2` are 2D-specific.

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ------------------------------------------
# rot2
# ------------------------------------------
# 2D rotation matrix for yaw ``psi`` [rad], CCW positive.
def rot2(psi: float) -> np.ndarray:
    c, s = math.cos(psi), math.sin(psi)
    return np.array([[c, -s], [s, c]])


# ------------------------------------------
# wrap_pi
# ------------------------------------------
# Wrap an angle to [−π, π).
def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


# ------------------------------------------
# Pose
# ------------------------------------------
# Rigid transform sensor → world.
#
# Parameters
# R : (d, d) float, rotation in SO(d) (orthonormal, det = +1 within 1e-9).
# t : (d,) float, translation [m].
@dataclass(frozen=True, eq=False)
class Pose:
    R: np.ndarray
    t: np.ndarray

    def __post_init__(self) -> None:
        R = np.asarray(self.R, dtype=float)
        t = np.asarray(self.t, dtype=float)
        if R.ndim != 2 or R.shape[0] != R.shape[1] or t.shape != (R.shape[0],):
            raise ValueError(f"Pose: R must be (d,d) and t (d,), got {R.shape}, {t.shape}")
        if not np.allclose(R.T @ R, np.eye(R.shape[0]), atol=1e-9) or np.linalg.det(R) < 0:
            raise ValueError("Pose: R must be a proper rotation (orthonormal, det=+1)")
        object.__setattr__(self, "R", R)
        object.__setattr__(self, "t", t)

    @property
    def dim(self) -> int:
        return self.R.shape[0]

    @classmethod
    def identity(cls, d: int = 2) -> Pose:
        return cls(np.eye(d), np.zeros(d))

    # 2D pose from ξ = (x [m], y [m], ψ [rad]).
    @classmethod
    def from_xi(cls, xi) -> Pose:
        x, y, psi = (float(v) for v in xi)
        return cls(rot2(psi), np.array([x, y]))

    # 3D pose of a level sensor: rotation ``R_z(yaw)`` and translation ``(x, y, z)`` (final-spec F1).
    @classmethod
    def from_xyz_yaw(cls, x: float, y: float, z: float, yaw: float) -> Pose:
        R = np.eye(3)
        R[:2, :2] = rot2(float(yaw))
        return cls(R, np.array([float(x), float(y), float(z)]))

    # 2D minimal parameters (x, y, ψ) with ψ ∈ [−π, π).
    def xi(self) -> np.ndarray:
        if self.dim != 2:
            raise ValueError("xi() is defined for 2D poses only")
        return np.array([self.t[0], self.t[1], wrap_pi(math.atan2(self.R[1, 0], self.R[0, 0]))])

    # Transform points ``(..., d)`` from sensor to world frame.
    def apply(self, p: np.ndarray) -> np.ndarray:
        return np.asarray(p) @ self.R.T + self.t

    # Rotate direction vectors ``(..., d)`` from sensor to world frame.
    def rotate(self, v: np.ndarray) -> np.ndarray:
        return np.asarray(v) @ self.R.T

    # ``self ∘ other`` (apply ``other`` first).
    def compose(self, other: Pose) -> Pose:
        return Pose(self.R @ other.R, self.R @ other.t + self.t)

    def inverse(self) -> Pose:
        return Pose(self.R.T, -self.R.T @ self.t)
