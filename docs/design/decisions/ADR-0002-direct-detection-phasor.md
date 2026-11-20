# ADR-0002 — Multi-return phasor amplitude is proportional to optical power

- **Status:** Accepted (after design review) · **Spec:** E10, E19

## Context
A design review proposed amplitude ∝ √P (field amplitude). This was rejected for the reason below.

## Decision
AMCW LiDAR uses direct (intensity) detection: APD photocurrent ∝ incident optical power; ranging phase
is carried by the RF modulation envelope (tens of MHz). Returns from different surfaces are mutually
incoherent at the optical carrier (path differences ≫ coherence length), so their modulated
photocurrents add as envelope phasors: `Z_j = 𝓡 m Σ_k P_k exp(i 4π f_j r_k / c)`.

## Alternatives
`√P` field summation — correct only for coherent/heterodyne detection (e.g. coherent FMCW); would
add spurious optical-interference terms here.

## Consequences
Mixed pixels arise physically from envelope-phasor addition. Speckle is neglected (limitation).
