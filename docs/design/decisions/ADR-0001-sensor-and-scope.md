# ADR-0001 — Sensor choice and scope: 2D AMCW LiDAR now, 3D for the final project

- **Status:** Accepted (2026-10-07) · **Supersedes:** — · **Spec:** §1, §8

## Context
The brief rewards technical challenge, rigor and effectiveness, and lists five modeling tools
(physics-based, estimation-theoretic, probabilistic, FOV, LOS). Candidates assessed: 2D/3D LiDAR, PT camera, acoustic, tactile, gas, radar.

## Decision
Model a 2D amplitude-modulated continuous-wave LiDAR inspired by the Hokuyo URG-04LX (Heiden et al.,
ICRA 2020) for the midterm. Extend to 3D (multi-layer, SE(3)) and use the model for inference in the
final project.

## Alternatives considered
- 3D LiDAR now — too much geometry/performance work for the midterm; little extra perception theory.
- PT camera — strong FOV story, weak physics. Tactile — no meaningful FOV/LOS. Gas — transport
  dominates. Radar — waveform scope too large.

## Consequences
- Only LiDAR naturally exercises all five tools. Core layers must be dimension-agnostic (spec §3.2).
- 3D alone would be a moderate upgrade; the final's step up comes from inference on top of the model.

## Evidence
A scoring of the candidates against the five tools (2D LiDAR 5/5); Heiden et al. provide divergence, CW phase ranging, mixed-pixel
and surface-interaction effects.
