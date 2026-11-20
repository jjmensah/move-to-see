# ADR-0004 — Noise in detector-current units with calibrated constants

- **Status:** Accepted · **Spec:** E11, E15

## Context
Design review: mixing optical power and current in the sample model was dimensionally inconsistent,
and a first-principles shot-noise formula needs bandwidth/gain data we do not have.

## Decision
All sampled signals are in detector-current units [A]: `s = 𝓡(P̄+P_amb) + Re{Z e^{−iθ}} + ε`,
`σ_n² = σ_th² + κ_sh 𝓡 (P̄ + P_amb)`. `σ_th` and `κ_sh` are calibrated proxies (κ_sh ≙ 2qMF·B_n),
not claimed as physical values. Detection threshold: CFAR at `P_FA,0 = 1e-4` (ADR-0008 supersedes the original fixed threshold).

## Calibration targets (solved by `sim/calibrate.py`, values frozen in `core/config.py`)
1. ρ = 0.90, r = 4 m, normal incidence, clear air, no ambient → fused σ_R = 12 mm.
2. At that operating point, shot-noise variance equals thermal variance.
3. "Bright sunlight" level `P_amb,sun` defined as the ambient that halves the P_D = 0.5 range of a
   ρ = 0.90 target.

## Frozen values (`python -m sim.calibrate`, nominal P₀ = 30 mW, A_rx = π·(1 cm)², η = 0.8, 𝓡 = 25 A/W, m = 1; CFAR per ADR-0008)
| Constant | Value |
|---|---|
| `sigma_th` | 3.283278644238764e-07 A |
| `kappa_sh` | 3.197198847453723e-08 A |
| `P_amb_sun` | 2.019618716475848e-06 W |

Resulting predictions (not tuned): P_D = 0.5 range of ρ = 0.90 is 12.36 m in clear air (beyond
r_max = 5.6 m, so white walls are always seen indoors) and 6.18 m in bright sun; of ρ = 0.05 (black)
2.92 m in clear air (the earlier ≈ 2.4 m hand estimate ignored shot noise).

## Consequences
Black-target range is a *prediction*, reported as such, not tuned.
