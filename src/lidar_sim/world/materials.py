# ==========================================================
# src/lidar_sim/world/materials.py
# ==========================================================
# Surface materials (spec §4.8). Ids are stable and used in scene data and records.

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

WHITE, GREY, BLACK, MIRROR = 0, 1, 2, 3


# ------------------------------------------
# Material
# ------------------------------------------
# ``rho``: Lambertian reflectivity (E7). Mirrors: ``rho_m`` specular, ``rho_d`` weak diffuse (E9).
@dataclass(frozen=True)
class Material:
    name: str
    kind: Literal["lambertian", "mirror"]
    rho: float
    rho_m: float = 0.0
    rho_d: float = 0.0

    def __post_init__(self) -> None:
        for v in (self.rho, self.rho_m, self.rho_d):
            if not 0.0 <= v <= 1.0:
                raise ValueError("reflectivities must lie in [0, 1]")

    # Reflectivity used for the direct diffuse return (E7).
    @property
    def rho_diffuse(self) -> float:
        return self.rho_d if self.kind == "mirror" else self.rho


MATERIALS: dict[int, Material] = {
    WHITE: Material("white", "lambertian", 0.90),
    GREY: Material("grey", "lambertian", 0.50),
    BLACK: Material("black", "lambertian", 0.05),
    MIRROR: Material("mirror", "mirror", 0.0, rho_m=0.95, rho_d=0.02),
}


# ------------------------------------------
# lambertian
# ------------------------------------------
# Ad-hoc Lambertian material (for reflectivity sweeps, X04).
def lambertian(name: str, rho: float) -> Material:
    return Material(name, "lambertian", rho)
