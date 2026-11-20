# ADR-0003 — Beam divergence: Gaussian angular profile, Gauss–Hermite sub-rays

- **Status:** Accepted · **Spec:** E4

## Context
Heiden et al. quote a 40 mm beam diameter at 4 m and use 3 equally weighted rays. The quoted figure
does not state a width definition; design review noted that different interpretations change
footprints and mixed returns.

## Decision
Interpret 40 mm @ 4 m as the 1/e² full width → θ₀ = 5 mrad (1/e² half-angle). Intensity
`I(δ) ∝ exp(−2δ²/θ₀²)`. K-point Gauss–Hermite quadrature: `δ_k = θ₀ x_k/√2`, `w_k = h_k/√π`.
Default K = 5; K = 1 recovers the ideal ray caster.

## Alternatives
Hard cone with uniform rays (Heiden) — discontinuous footprint; arbitrary weights.

## Consequences
Deterministic, reproducible sub-ray layout; X07 compares K = 1 vs K = 5. 3D will use a 2D
Gauss–Hermite cubature.
