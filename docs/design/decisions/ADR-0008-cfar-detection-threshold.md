# ADR-0008 — Detection threshold tracks the background noise (CFAR)

- **Status:** Accepted (found during calibration) · **Spec:** E15, E16 · **Supersedes:** spec v1.0 fixed threshold

## Context
Spec v1.0 fixed `γ` from thermal noise only. Calibration (ADR-0004 step 3) showed that ambient light then
*raises* detections: at `P_amb = 10·P_ref` the no-signal false-alarm rate is 0.43 and the P_D = 0.5
range *grows* (12.4 m → 17.6 m), the opposite of the intended "sunlight shrinks range" effect, and a
no-hit beam would report random ranges ~40 % of the time.

## Decision
`γ = s_b √(−2 ln P_FA,0)` with `s_b = √(2/N)·√(σ_th² + κ_sh 𝓡 P_amb)`. The sensor can measure its
background current (sample DC with no return), so this is implementable; it is the standard
constant-false-alarm-rate (CFAR) design. Analytic `P_FA = P_FA,0` for every ambient level.

## Alternatives
Fixed threshold (rejected: unphysical behaviour above); adaptive threshold per beam on the beam's own DC
(rejected: signal DC would bias it; background is measured between pulses/beams in practice).

## Consequences
X06 shows ambient shrinking range/FOV via SNR at constant false-alarm rate; S-06 sets `P_FA0 = 0.01`
directly to obtain enough false alarms. Spec bumped to v1.1.
