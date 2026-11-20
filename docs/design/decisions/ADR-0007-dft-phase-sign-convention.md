# ADR-0007 — DFT phase sign convention (E11–E12)

- **Status:** Accepted (review objection rejected with evidence) · **Spec:** E11, E12

## Context
A review claimed `X = Σ s_n e^{+iθ_n}` with `s_n = Re{Z e^{−iθ_n}}` returns `Z*`, reversing range.

## Decision
Keep the convention. `X = (N/2)Z` (direct expansion of the sum), so `arg X = arg Z`.

## Evidence
Direct expansion of the sum; Monte Carlo variance of `φ̂ − φ` equals the CRLB (an estimator of −φ would show a
gross bias). Oracle O-10 pins it: r = 1 m ⇒ φ̂₁ = 1.951232 rad.
