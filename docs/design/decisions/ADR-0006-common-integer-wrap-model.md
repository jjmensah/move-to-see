# ADR-0006 — Wrap outliers follow a common-integer model; fused range folded mod L_c

- **Status:** Accepted (found by Monte Carlo and confirmed in review) · **Spec:** E14, E18, E20 · **Supersedes:** spec v0.2 E18

## Context
Spec v0.2 E18 thresholded the coarse error alone at L₂/2 and L₁/2, predicting separate "both wrap" and
"only f₂ wraps" components. A review flagged the correlated-error issue, and a Monte Carlo run found zero
"only f₂" events.

## Decision
Both unwrap integers err by the same `round(ζ)`, `ζ = (f₁e₂ − f₂e₁)/(2πΔf)` (see the derivation below). E18
becomes a symmetric integer mixture k ∈ {±1, ±2, ±3} with shift kΔ. E14 folds the fused range mod L_c
because L_c = 7L₁ = 8L₂ exactly.

## Alternatives
Keep the v0.2 model (refuted by simulation); exact non-Gaussian phase pdf (deferred, stretch).

## Consequences
`BeamPrediction.out_prob/out_shift` shape (B,6). S-05 uses a stated ≤ 15 % relative tolerance because
the Gaussian linearization under-predicts near threshold by up to ≈ 10 % — reported as a model limit.

## Evidence
The unwrap errors of both frequencies are driven by the same combination of phase errors, so they wrap by the same
integer; the Monte Carlo run of the waveform simulator confirms it (no "only f₂" events).
