# ADR-0009 — Robustness floor ε covers both the miss and the detection outcome

- **Status:** Accepted (found while running oracle O-16 against the model code) · **Spec:** E20 · **Supersedes:** spec v1.1 E20 floor

## Context
v1.1 E20 put the whole floor ε on detections (`ε/|W|`) and none on misses. At high SNR,
`p(not detected) = (1−ε)[(1−P_D) + P_D(1−M)]` underflows to exactly 0, so one unmodeled dropout
(occluding person, absorbing patch, glare) gives log-likelihood −∞ and breaks pose MLE. Observed as a
`log(0)` RuntimeWarning in oracle O-16 at A/σ_n = 7.25.

## Decision
The floor is a mixture with a "junk" observation model split evenly between the two outcomes:
`p(o) = (1−ε) p_model(o) + ε p_junk(o)`, `p_junk(not detected) = ½`, `p_junk(detected, z) = ½·1/|W|` on W.
Both branches of E20 use it; normalization is unchanged (each part integrates to 1).

## Consequences
Every observation has log-likelihood ≥ log(ε/2) − log|W| (finite). Spec bumped to v1.2. The model code updates
`log_likelihood_beams`; O-16 tests still apply (normalization) and a new test asserts finiteness for a
missed beam at P_D = 1.
