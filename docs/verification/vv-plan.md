# Move to See — verification and validation plan

Version 0.3 (design review applied; criteria fixed after pilot runs on reserved seeds, before registration). Companion to `docs/design/final-spec.md`. Sensor-level V&V of
E1–E25 was done in the midterm (its fast tests are vendored under `tests/` and must stay green); this plan
covers what is new.

## 1. Principles

1. **Independent checks.** Every module's oracle tests belong to the track that did not implement it and are written
   from the specification, with reference values computed independently of the production code (hand
   calculations, brute force, numerical integration, grid Bayes).
2. **Pre-registration.** Every experiment fixes its hypothesis, sweep, sample size, seed and pass criteria
   in `sim/registry.yaml` before its first full run. Scenario and algorithm parameters are tuned only on
   **pilot seeds** (seed base 9000+) that are never used by a registered experiment. The registered verdict
   is reported whatever it is; analyses added afterwards are labelled exploratory, and every change made after
   registration is listed in §5.
3. **Scientific hypotheses may fail; execution may not.** The exit criteria (§4) require correct execution
   and honest reporting, not that every hypothesis be supported.
4. **No information leak.** Inference and planning receive only the background world, the `Observation`
   record, the robot states and the environment. Checked structurally (imports, signatures) and
   metamorphically (changing the hidden truth while keeping the observation fixed leaves inference and
   actions unchanged).
5. **Determinism.** Every stochastic function takes an explicit `Generator`; paired controllers share the
   target draw and the per-scan noise stream.
6. Statistics: rates → Wilson intervals; paired comparisons → Wilcoxon signed-rank tests (Pratt zeros) at
   familywise α = 0.01 with Holm correction over the tests of one experiment; medians → bootstrap 95 %
   intervals (10 000 resamples, registered seed).

## 2. Test suite (fast, `pytest`)

| ID | What | Method |
|----|------|--------|
| T-G1 | `Scene3D.intersect`, `ray_box_batch` (F2) | hand oracles (axis rays, oblique, grazing, axis/box/ground ties, origin inside, origin within `r_eps` of the entry face, parallel rays inside and outside slabs, ground only); brute force against per-face plane intersection on 10⁴ random rays × random boxes |
| T-G2 | HitBatch invariants | unit normals, `n·d ≤ 0`, ids, nearest hit (property test) |
| T-G3 | 2D ⊂ 3D | walls extruded in z scanned by an ε = 0 row with K = 1 reproduce the midterm `Scene2D` + `Planar2DPattern` ranges, incidence cosines and powers to 1e-12 |
| T-S1 | `Raster3DPattern` (F3–F5) | unit vectors, ordering, fire times, weights sum to 1; projected sub-ray offsets `atan2(d·e_α, d·u) = δ_p`, `atan2(d·e_ε, d·u) = δ_q` to 1e-12; K = 1 ideal |
| T-S2 | Geometric FOV (F6) | inclusive bounds; every beam axis point at mid-range is inside |
| T-S3 | Trace refactor | vendored midterm tests green (incl. the specular path); independent power oracle (E7–E8 by hand) on a 3D box face |
| T-R1 | Unicycle (F7) | closed forms on smooth segments: straight line with constant a; constant-ω arc at constant v; RK4 step-halving error ratio ≈ 16 |
| T-R2 | Hybrid terrain and limits (F8) | speed projected on entering a rough patch; input limits; `Trajectory.state` continuity |
| T-R3 | Feasibility (§4.2) | keep-out rectangles by hand; for target positions on the boundary of `R_T`, no feasible robot state overlaps the true target footprint with the robot disc |
| T-R4 | Navigation and following (§7.4–7.5) | Dijkstra times against hand-computed corridors (incl. rough patch); the follower reaches goals without infeasible steps on 200 random start/goal pairs; braking rule: from every visited state full braking stops before any keep-out (speed ≤ `√(2 a_max (c − m_s))`) |
| T-P1 | Predictor consistency (§5.2) | predictor trace at the true q equals the truth trace (candidate beams) and the background trace (others) field by field; candidate set ⊇ beams whose trace changes over random q |
| T-P2 | Likelihood (F9) | equals `BeamModel.log_likelihood_beams` on the reference path, per hypothesis |
| T-P3 | Design models (F14) | π, μ, σ, Z against direct evaluation; blind model independent of reflectivity and environment |
| T-B1 | Belief (§6) | weights normalised; CESS bisection (monotone cases and termination); exponents sum to 1; ideal Liu–West kernel preserves mean and covariance; boundary rejection keeps particles in `R_T` |
| T-B2 | Belief vs grid Bayes | 1-D/2-D toy likelihoods (Gaussian, bimodal, sharp step/"silhouette", repeated scans): posterior mean and standard deviation against a fine-grid oracle within stated tolerances |
| T-I1 | BC closed forms (F15) | Gaussian–Gaussian (truncated), uniform–uniform, uniform–Gaussian against numerical quadrature; π = 0 and π = 1 edge cases; symmetry; `BC ∈ [0, 1]` |
| T-I2 | MI bound (F17) | `0 ≤ Î ≤ log M`; 0 for identical hypotheses; `log M` for separable ones; duplicates cap the value; permutation invariance; **Î ≤ exact MI** of the design channel on small cases (enumeration + quadrature) |
| T-L1 | Planner | goal maximises (F18) by brute force on a toy world; with an entirely obstructed prior and zero information nearby, the planner still selects a distant informative viewpoint (no "stop forever"); tie-breaking deterministic |
| T-C1 | Contracts | import boundaries; `mts.belief/planner/info/predict` signatures never accept truth; metamorphic leak test; equation ids cited in code exist in the specs |
| T-M1 | Metrics | F20–F22 by hand; dwell rule; Holm; Wilson; bootstrap determinism |

## 3. Registered experiments (`python -m sim.run <ID>`)

Each writes `sim/out/<ID>/metrics.json` (results and verdicts), `run.json` (registry entry, seed, revision,
source hash, versions, timing) and figures.

**F01 — 3D measurement model against the waveform truth (target state and environment).**
A 1 m square panel facing the sensor, black or white, at ranges {1, 2, 3, 4, 5} m, at elevations 0° and
20°, in clear air, fog (V = 20 m) and sunlight (1 sun): 60 cells, 4000 scans of the central beam per cell
(K = 9). Separately, 10⁶ no-return beams for the false-report rate.
*H:* the analytic model predicts the simulated amplitude-threshold crossing rate (`P_D`), the reporting
rate (`P_D M`) and, at high SNR, the range standard deviation.
*Pass:* (a) crossing rate: predicted `P_D` inside the simultaneous (Bonferroni, familywise 99 %) Wilson
interval in every cell; (b) high-SNR cells (`P_D > 0.999999` and wrap probability < 10⁻⁶): range standard
deviation ratio within 1 ± 0.05; (c) false-report rate: predicted `P_FA |W|/L_c` inside the 99 % Wilson
interval. *Reported, not gated:* reporting-rate agreement near threshold (model adequacy; the midterm
showed detection–range coupling there).

**F02 — Field of view and robot-state dependence.**
(a) Effective FOV volume (E3a with F6) of a facing probe, by Monte Carlo integration (10⁶ points) at the
start pose and two other poses, for black and white probes in C1–C3, against an independent calculation:
the volume of `{geometric FOV} ∩ {line of sight} ∩ {range ≤ r_50(ρ, env)}` with `r_50` from the midterm
calibration. (b) Orientation: fraction of the prior region inside the geometric FOV versus yaw.
(c) Velocity: with the robot moving at v = 0.5 and 1.0 m/s past a wall, the sensor origin used by beam i
equals `p(t_i)`; the range to the wall differs from the static scan by the analytic amount implied by the
displaced origin.
*Pass:* (a) nested volumes geometric ⊇ line-of-sight ⊇ effective for every case, black-probe effective
volume ordered clear > fog > sun, and agreement with the independent calculation within 3 % (after a
sample convergence check); (b) descriptive; (c) maximum range discrepancy against the analytic moving-origin
oracle below 1e-9 m, and the skew (difference from the static scan) non-zero and increasing with v.

**F03 — Belief calibration.**
For each of C1, C2, C3: 100 trials with `q ~ Uniform(R_T)`; the robot follows the fixed route (no planner in
the loop); after scans 20, 40 and 60: error, NEES, per-coordinate standardised errors and posterior spread.
*H:* the regularized particle filter is not over-confident.
*Pass (C1; C2 and C3 reported):* after each of scans 20, 40 and 60 the fraction of trials with
NEES ≤ χ²₃(0.90) is at least 0.80. The criterion is one-sided on purpose: pilot runs (seeds ≥ 9000) showed
that the kernel floor `σ_j = 3 mm` of (F12) keeps the posterior standard deviation near 3.5 mm when the actual
error is about 2 mm, i.e. the filter is conservative below a ~4 mm resolution floor, and close to nominal
above it. *Reported, not gated:* median NEES relative to the χ²₃ median, standardised-error spread per
coordinate, median error.

**F04 — Information bound against the exact design-channel mutual information.**
10 beliefs taken from fixed-route trials on the experiment's own seed, in fog (snapshots at scans 0, 15 and
40); for each, 12 candidate viewpoints spread over the lattice around the target keep-out; Î (F17) versus a
Monte Carlo estimate of the exact mutual information of the finite-support design channel
(`E[log p_m(z) − log (1/M) Σ_j p_j(z)]`, 4000 samples, exact mixture density).
*H:* Î is a lower bound.
*Pass:* Î ≤ MC estimate + 3 standard errors + 0.01 nat in all 120 cases. The absolute tolerance was added
after a smoke run on a pilot seed showed that, when the information is carried by events rarer than about
1/n, the Monte Carlo estimate and its standard error are both exactly zero (no such event is drawn), so a
purely relative criterion is meaningless there; 0.01 nat is 0.3 % of `log M`. *Reported, not gated:* number
of cases below the tolerance, within-belief Spearman rank correlation, and the regret (in MC information) of
choosing the Î-best candidate.

**F05 — Main comparison (effectiveness of active perception).**
Conditions C1–C4, 40 paired trials each, three controllers, T = 60 steps.
*H1:* the information-driven planner localizes faster than the fixed route in C1–C3 (time score τ̃).
*H2:* it localizes faster than the physics-blind planner in C2 and C3.
*H3 (control):* in C4 (white target, clear air) the two planners perform about the same.
*Pass:* H1 and H2: one-sided Wilcoxon signed-rank tests on paired τ̃ (information-driven smaller), Holm over
the five tests, familywise α = 0.01. H3: the 95 % bootstrap interval of the median paired difference of τ̃
lies within ±2.0 s.
*Reported regardless:* success rates (dwell criterion) with Wilson intervals and paired McNemar counts;
median final error with bootstrap intervals; error-versus-time curves; path lengths; fraction of trials
with no detected target return (the "negative information only" trials); planner run time.

**F06 — Ablations and diagnostics (exploratory, registered sweep, 20 paired trials each, C2).**
Support size M ∈ {8, 24, 48}; travel penalty λ_t ∈ {0, 0.05, 0.2}; rough patch on/off (time and path through
the patch, integrated at the substep resolution); dark background variant (walls, crates and ground black)
to quantify how much performance relies on negative information. The M = 24, λ_t = 0.05 runs are shared
between sweeps. Reported descriptively; no gate. (The report-flag-only utility variant of the earlier draft
was dropped.)

**M01 — Media.**
One C2 trial animated for each controller: top view (obstacles, rough terrain, target keep-out, prior region,
path, chosen viewpoint, geometric range and the unobstructed effective range for the target as sectors,
particles, 2σ belief ellipse, true target outline) with side panels (position error and belief standard
deviations versus time), plus a keyframe figure (three instants per controller, top view).
*Pass:* the files exist and are non-trivial, and the error curve drawn in the last frame of each animation
equals the trial record.

## 4. Exit criteria

All fast tests green; `ruff` clean; F01–F05 and M01 run from a clean checkout with their registered seeds
and write verdicts; `docs/verification/results.md` generated from `sim/out/*/metrics.json`; every hypothesis
reported as supported or not supported under its registered criterion, with exploratory analyses labelled.

## 5. Deviations from the registration

Recorded here so that no change after registration is silent. None of them alters a registered hypothesis or
threshold.

1. **F02 reference (after the first registered run, on independent review).** The independent
   volume calculation originally obtained the line-of-sight distance per direction from `Scene3D.intersect`,
   the same routine the field-of-view code uses, so a geometry error common to both could not have been
   detected. The reference now enumerates box-face planes and the ground plane directly. The registered
   criterion (`effective_volume_matches_independent_3pct`) and its 3 % threshold are unchanged; the result with
   the original reference is kept in `metrics.json` (`rel_diff_scene_reference`) next to the new one, and
   convergence diagnostics (half-sample Monte Carlo, quadrature refinement) are reported for every case.
2. **M01 gate (on the same review).** The registry criterion checked only that the media files exist, while
   this plan also promised that the error drawn equals the record. The criterion
   `drawn_error_matches_record` was added to the registry and the experiment (the drawn curve is the recorded
   error floored at 1 mm for the logarithmic axis); the PDF is now checked like the videos. The media content is the top-view version described above (the earlier draft of this plan
   mentioned an entropy panel and a 3D keyframe figure, which were not built).
3. **F01 (on the same review).** The waveform simulator and the analytic model are compared on the same traced
   record, so F01 tests the model given the radiometry, not the radiometry itself. A hand-written E7–E8 power
   for each cell is now reported next to the traced power (`power_rel_error_vs_hand`); no gate was added.
4. **Provenance.** `run.json` records the revision, modified tracked files and a hash of all sources before
   the experiment code is loaded and again afterwards; trial caches carry a fingerprint of the job that
   produced them. The first runs of F01, F02 and M01 predate this and were repeated from the final revision
   (same seeds, identical numerical results where the experiment code did not change).

## 6. Scope of the claims (from the final independent audit)

The audit recomputed every F05 statistic from the raw trial records with separate code (all values agree) and
asked that the following be stated wherever the results are reported:

1. H3 shows that the *median* paired difference of the restricted time score is within the margin (30 of 40
   differences are exactly zero; individual differences range from −3.0 to +2.5 s). It is not a claim that the
   two planners behave identically.
2. The Wilcoxon tests concern signed ranks of a time score capped at the trial length (failures count as
   30 s); "median speed-up" should be read with that cap in mind. In sunlight the information-driven planner
   was slower than the fixed route in 12 of 40 pairs.
3. The physics-blind baseline is deliberately mis-specified (perfect detection inside the window, constant
   12 mm noise) and the fixed route is one predetermined loop. The results do not establish superiority over
   other active planners.
4. "No target return" counts reported beams whose strongest return is the target; the dark-background ablation
   reduces, but does not eliminate, information from the background.
5. Simulated time excludes computation: planning plus belief update take about 1.1 s of wall time per 0.5 s
   decision step in fog (unoptimised Python).
6. Known robot pose, obstacle map, terrain, environment parameters and target size/reflectivity; an approximate
   particle filter whose calibration gate is one-sided (C1 only); F04 concerns the finite-support design channel,
   not the waveform simulator. All results are simulation results of one physics-inspired sensor model.
7. Cache fingerprints cover the job arguments, not the source code: after a code change the trial cache of an
   experiment must be deleted before re-running it.
