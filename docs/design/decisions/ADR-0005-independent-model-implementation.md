# ADR-0005 — Analytic model and waveform simulator implemented independently

- **Status:** Accepted (2026-10-07) · **Spec:** §3, §4.6, experiment X10

## Context
Experiment X10 validates the analytic likelihood (E16–E20) against the waveform simulator (E10–E15). If both are written
as one piece of code, shared misconceptions pass silently.

## Decision
- Shared, frozen first (Phase 0): `core/`, records in §3.1, `RadiometricScan` producer (E1–E9),
  canonical scenes, config.
- Track 1 implements `physics/waveform.py` (E10–E15) and the sensor pipeline.
- Track 2 implements `models/` (E16–E20) and `estimation/` (E22–E25), from the specification alone.
- `models/` must not import `physics.waveform` (enforced by `tests/contract/test_import_boundaries.py`).
- Each track gets hand-computed oracle tests for its equations, derived from the spec alone, not from the code.
- Disagreements → ADR with both derivations + a discriminating fixture or Monte Carlo.

## Consequences
Some duplicated effort (phase/unwrap logic appears twice, noise-free in `models`); this duplication is
the point.
