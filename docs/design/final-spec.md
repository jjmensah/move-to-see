# Move to See — design specification (final project)

Version 0.3 (design review applied; implementation decisions recorded). Normative for everything under `src/mts/`, the 3D additions to
`src/lidar_sim/`, and `sim/`. The sensor physics and likelihood (equations E1–E25) remain governed by
`docs/design/sensor-spec.md` (the midterm specification, vendored unchanged); this document adds equations
F1–F22 and never redefines an E-equation.

## 1. Purpose and scope

A ground robot carrying a 3D AMCW LiDAR must **localize** a stationary, dark, partly obstructed target. The
robot plans its own motion (active perception): at every step it chooses the viewpoint with the best
trade-off between expected information about the target position and travel time (F18), using the physics-based
sensor model (detection probability, range noise, line of sight) to predict what each candidate viewpoint
would measure.

Brief objective: **Localization** (of the target; the robot pose is known). Brief items covered: robot model
from class (unicycle ODE), sensor measurement + FOV models (midterm model in 3D), environmental conditions
acting on the sensor (fog, sunlight, target reflectivity) and on the robot (rough terrain limits speed), a
target model (fixed target with box geometry and unknown position), and an active perception method with
baselines.

**In scope:** 3D ray casting against axis-aligned boxes and a ground plane; a raster 3D scan pattern;
unicycle dynamics with terrain-dependent speed limit and obstacle avoidance; a particle belief over the
target position; a next-best-viewpoint planner that maximizes a lower bound on expected information gain
minus travel time; two baselines; paired Monte Carlo evaluation; animations.

**Out of scope (stated limitations):** robot localization / SLAM (robot pose, obstacle map, terrain map,
environment parameters, target size, orientation and reflectivity are known); moving targets; 6-DoF robot
motion; non-box geometry; fog backscatter and multipath beyond the midterm model; specular surfaces
(`specular=False` throughout). The planner exploits the known background: a dark target that hides a
brighter surface is informative even when the target itself returns nothing ("negative information").

## 2. Frames, notation, units

World frame W: right-handed, z up, ground plane z = 0, SI units, angles in radians. Robot state
`x = (p_x, p_y, θ, v)`; control `u = (a, ω)`. Sensor frame S: x forward, y left, z up, origin at height
`h_s` above the robot reference point. Sensor pose (sensor → world):

    R_WS = R_z(θ),   t_WS = (p_x, p_y, h_s)                                                    (F1)

Target position `q = (q_x, q_y, q_z)` = centre of the target box (half-size `s_T`). Belief
`b_k(q) = p(q | z_{1:k}, x_{1:k})`. Scan index k, beam index i (B beams), sub-ray index j (K sub-rays).

## 3. 3D sensor layer (additions to `lidar_sim`)

### 3.1 Scene3D (`lidar_sim/geometry/scene3d.py`)

A static scene of `n` axis-aligned boxes `[lo_m, hi_m] ⊂ R³` (each with a material id and an object id) and
an optional ground plane z = 0. Implements the `Scene` protocol with `dim = 3`.

Ray `o + r d` (unit d). Slab method per box and axis a ∈ {x, y, z}:

    r_a^- = min((lo_a − o_a)/d_a, (hi_a − o_a)/d_a),  r_a^+ = max(…)   (if d_a = 0 the slab is (−∞, +∞)
    when lo_a ≤ o_a ≤ hi_a and empty otherwise)
    r_near = max_a r_a^-,   r_far = min_a r_a^+                                                 (F2)

The box is hit iff `r_near ≤ r_far` and `r_far > r_eps`. The reported distance is the first intersection
beyond the guard: `r_near` if `r_near > r_eps`, otherwise `r_far` (exit face; this covers origins inside
the box and origins within `r_eps` of the entry face). The hit face is the axis attaining the max (min for
an exit hit); ties between axes → lowest axis index. Ground plane: hit iff `d_z < 0`, `o_z > 0` and
`r = −o_z/d_z > r_eps`. The nearest hit over all boxes and the ground is returned (E5); ties between boxes
→ lowest box index; a box wins a tie with the ground. No upper range gate. The normal is the outward face
normal of the geometric face, flipped if necessary so that `n·d ≤ 0`.

Ids: `surface_id = 6 m + f` for box m, face f ∈ {0: −x, 1: +x, 2: −y, 3: +y, 4: −z, 5: +z};
ground `surface_id = 6 n`; `object_id` is the box's object id, the ground uses `ground_object`. Misses have
all ids −1.

Additional API: `with_box(m, lo, hi)`, `without_box(m)`, and
`ray_box_batch(o (N,3), d (N,3), lo (P,3), hi (P,3), r_eps) -> (r (P,N), face (P,N))` (distance `inf` and
face −1 where no hit), by (F2).

### 3.2 Raster3DPattern (`lidar_sim/sensor/scan3d.py`)

Azimuths `α_c = −Θ/2 + c Θ/(n_az − 1)`; elevations `ε_e = ε_min + e (ε_max − ε_min)/(n_el − 1)`. Beam index
`i = e n_az + c` (rows of constant elevation fired in order; with `n_el = 1` the single row is at
`ε_min = ε_max`). Sensor-frame axis and fire time:

    u_i = (cos ε_e cos α_c, cos ε_e sin α_c, sin ε_e)                                           (F3)
    τ_i = i T_scan / B,   B = n_az n_el                                                         (F4)

Here `T_scan` is the duration of one complete raster frame (in the 2D pattern it is one mirror revolution).
Sub-rays: with the m-point Gauss–Hermite offsets `(δ_p, w_p)` of (E4) and the orthonormal tangent basis
`e_α = (−sin α_c, cos α_c, 0)`, `e_ε = (−sin ε_e cos α_c, −sin ε_e sin α_c, cos ε_e)`,

    u_{i,(p,q)} = normalize(u_i + tan(δ_p) e_α + tan(δ_q) e_ε),   w_{(p,q)} = w_p w_q,   K = m²   (F5)

so the projected angular offsets are exactly `δ_p` and `δ_q` (the radial offset is
`atan √(tan²δ_p + tan²δ_q)`). The product weights integrate a Gaussian profile that is separable in the two
projected angles — a paraxial convention; for θ₀ = 5 mrad it differs negligibly from a rotationally
symmetric profile per unit solid angle. `subrays(K)` requires a perfect square; sub-ray order j = p m + q.

Geometric FOV (3D counterpart of E3): a world point p ≠ t_WS with sensor-frame coordinates
`s = R_WSᵀ(p − t_WS)`, `r = ‖s‖`, `α = atan2(s_y, s_x)`, `ε = asin(s_z / r)` is in `S_geo` iff

    |α| ≤ Θ/2,   ε_min ≤ ε ≤ ε_max,   r_min ≤ r ≤ r_max                                        (F6)

(inclusive, same tolerances as E3). The effective FOV (E3a) is unchanged in form.

### 3.3 Pose and sensor plumbing

`Pose.from_xyz_yaw` builds (F1). `LidarSensor.trace` is refactored, without changing results, into
`scan_rays(pattern, K, pose, t0)` and `radiometric_from_hits(pose, fire_time, dir_world, w, hits, materials,
sigma_ext, noise, P_amb)` so the truth trace and the fast predictor share one implementation of E6–E8 (the
optional specular path E9 is preserved). All vendored midterm tests must pass unchanged.

Default 3D sensor (`mts.config`): Θ = 240°, n_az = 121 (2°), ε ∈ [−10°, +40°], n_el = 11 (5°), B = 1331,
T_scan = 0.1 s, θ₀ = 5 mrad, K = 9 (m = 3), r_min = 0.02 m, r_max = 5.6 m, h_s = 0.40 m; tones, sampling
and noise constants are the midterm calibration, unchanged.

## 4. Robot, terrain and world (`mts/robot.py`, `mts/world.py`)

### 4.1 Dynamics

Unicycle with acceleration input (class model), for `x = (p_x, p_y, θ, v)`, `u = (a, ω)`:

    ṗ_x = v cos θ,   ṗ_y = v sin θ,   θ̇ = ω,   v̇ = a                                          (F7)

integrated with RK4 at substep `δt = 0.02 s`, u held constant over a decision step `Δt = 0.5 s`
(25 substeps). Input limits `|a| ≤ a_max = 1.0 m/s²`, `|ω| ≤ ω_max = 1.0 rad/s`. Terrain makes this a
hybrid system: the world has rectangular rough patches with speed limit

    v_lim(p) = v_rough = 0.3 m/s if p lies in a rough patch (closed rectangle), else v_max = 1.0 m/s   (F8)

and after every substep the speed is projected, `v ← clip(v, 0, v_lim(p))` (entering a rough patch can
reduce the speed instantaneously; the environment acts on the robot state). Motion is deterministic and
the state is known to the estimator and the planner.

Pose during a step: for `t ∈ [t_k, t_k + Δt]` the state is the RK4 state at the last substep boundary
advanced by one RK4 step of the remaining fraction. Scan k is acquired over `[t_{k+1} − T_scan, t_{k+1}]`
(the last 0.1 s of the step), with beam i fired at `t_{k+1} − T_scan + τ_i`.

### 4.2 Feasibility (collision model)

The robot is a disc of radius `ρ_r = 0.30 m`; clearance margin `m_c = 0.05 m`. Keep-out set: every
obstacle footprint (box projected on the ground, including walls) inflated to the axis-aligned rectangle
grown by `ρ_r + m_c` on every side (conservative rectangular inflation; touching the boundary is
infeasible), plus the **target keep-out**: because the target is somewhere in the prior region, the union of
all possible target footprints `U_T = R_T^{xy} ⊕ [−s_T, s_T]²` is inflated the same way. A state is feasible
iff its position lies outside every keep-out rectangle (the walls keep it inside the arena). A step is
feasible iff every substep state is feasible (substep travel ≤ 2 cm ≪ m_c).

### 4.3 World `yard()` (all dimensions in metres)

Arena `[0, 11] × [0, 8]`; perimeter walls 0.1 thick (inside the arena), 1.5 high, grey (ρ = 0.5);
obstacle crates (white, ρ = 0.9) `A = [3.2, 4.4] × [3.3, 4.7] × [0, 1.2]`,
`B = [3.4, 4.2] × [3.5, 4.4] × [1.2, 1.9]`, `C = [4.4, 5.3] × [3.6, 4.6] × [0, 0.85]`; ground grey
(ρ = 0.5); rough patch `[3.6, 6.2] × [1.0, 2.6]`; robot start `x_0 = (1.0, 4.0, 0, 0)`.
Target: black (ρ = 0.05) box 0.5 × 0.5 × 0.5 (`s_T = 0.25`) on a thin mast (not modelled), centre in the
prior region `R_T = [7.0, 8.6] × [2.8, 5.2] × [0.5, 1.1]`; hence `U_T = [6.75, 8.85] × [2.55, 5.45]` and
target keep-out `[6.40, 9.20] × [2.20, 5.80]`. In the truth scene the target is the last box (stable ids).

## 5. Measurement prediction

### 5.1 Truth (simulator)

`LidarSensor.trace` on the scene with the target at the true `q`, with the pose trajectory of §4.1 (E2 scan
skew: the measurement depends on the robot velocity), then `measure` (waveform level, E10–E15). The
environment `env = (V, P_amb)` is constant within a trial. Inference receives an **observation-only** record
`Observation(detected (B,), range (B,), fire_time (B,))` — never `Scan.truth`.

### 5.2 Fast predictor (`mts/predict.py`)

For a known pose trajectory: rays `(o, d)` (B·K) from `scan_rays`; background hits from the scene without
the target, computed once per scan. **Envelope cull:** only beams with at least one sub-ray intersecting the
envelope box `U_T × [z_min − s_T, z_max + s_T]` (of all possible targets) can depend on q; all other beams
are *unaffected*. For hypotheses `q^{(1..P)}`: `ray_box_batch` on the candidate rays; per ray the nearer of
background and target hit (F2/E5; the target wins exact ties with the ground, a background box wins exact
ties with the target only if its index is lower — in the truth scene the target is last, so background
boxes win); ids follow the truth-scene convention (target `surface_id = 6(n−1) + f`, ground `6n`).
`(hypothesis, candidate beam)` pairs are flattened into a synthetic beam axis, in chunks, to build one
`RadiometricScan` per chunk with `radiometric_from_hits`, and `BeamModel.predict` gives E16–E19 for all
pairs. Output: arrays of shape `(P, B_c)` for the candidate beams plus their beam indices.

Only (hypothesis, beam) pairs in which some sub-ray's nearest hit is the hypothesised target differ from the
background prediction, so the background prediction is computed once per candidate beam and E16–E20 are
evaluated only for those pairs; per call, candidate beams are further restricted to those intersecting the
bounding box of the hypotheses passed in. Both are exact.

Consistency requirement: for `q` equal to the true target position, the predictor's per-beam record equals
the truth trace of §5.1 on the candidate beams, field by field (≤ 1e-12 relative), and the truth trace on
non-candidate beams equals the background trace.

## 6. Belief (`mts/belief.py`) — a regularized particle filter (approximate Bayes)

### 6.1 Prior

`b_0 = Uniform(R_T)`, N = 2000 particles, equal weights, drawn by a scrambled Halton sequence.

### 6.2 Update

With the observation `z_k` and the E20 per-beam likelihood on the prediction for particle n,

    ℓ_k^{(n)} = Σ_{i ∈ candidate beams} log p(z_{k,i} | q^{(n)}, x_k)                           (F9)
    w_k^{(n)} ∝ w_{k−1}^{(n)} exp(ℓ_k^{(n)})                                                    (F10)

(unaffected beams cancel). (F10) is applied by adaptive tempering with exponents
`0 = φ_0 < … < φ_S = 1`: the increment Δ is the largest value in `(0, 1 − φ]` with conditional effective
sample size

    CESS(Δ) = N (Σ_n w_n e^{Δ ℓ_n})² / Σ_n w_n e^{2Δ ℓ_n}  ≥  0.5 N   (w normalised)            (F11)

found by bisection on Δ (CESS is continuous, equals N at Δ = 0; if CESS(1 − φ) ≥ 0.5 N take Δ = 1 − φ);
a minimum increment of 10⁻⁶ guarantees termination. After each increment, if
`ESS = 1/Σ w_n² < 0.5 N`, the particles are resampled (systematic, weights reset to 1/N) and moved with a
Liu–West kernel

    q′ = a q + (1 − a) q̄ + ε,   ε ~ N(0, (1 − a²) Σ̂ + σ_j² I),   a = 0.98,   σ_j = 3 mm          (F12)

(q̄, Σ̂: weighted mean and covariance before resampling). A proposal outside `R_T` is rejected and the
particle keeps its resampled value. The log-likelihood of moved particles is recomputed and the remaining
exponent `1 − φ` is applied to it. **This is an approximation:** the move does not leave the tempered
posterior invariant (past scans are not revisited), so the filter slightly diffuses the posterior at every
resampling; the kernel preserves the first two moments only up to the `σ_j` term and boundary rejections.
Its calibration is measured, not assumed (V&V F03); in pilot runs the posterior standard deviation does not
fall below about 3.5 mm (the kernel floor), so the filter is conservative at the millimetre scale. Particles with a non-finite log-likelihood get zero
weight; if all log-likelihoods are non-finite (also after a move) the whole update is undone and flagged.

### 6.3 Estimate

    q̂_k = Σ_n w^{(n)} q^{(n)},   Σ̂_k = Σ_n w^{(n)} (q^{(n)} − q̂_k)(q^{(n)} − q̂_k)ᵀ               (F13)

## 7. Information-driven planner (`mts/info.py`, `mts/planner.py`)

### 7.1 Design measurement model

For planning, the outcome of beam i under hypothesis q at a candidate viewpoint (static pose, K = 1 ideal
rays, so no mixed pixels and no skew) is: no report with probability `1 − π_i`, or a report with range
density `f_i` on the window `W = [r_min, r_max]` with probability `π_i`, independently across beams:

- beam with an optical return (`hit_any`): with `μ_i = r̄_i` (E19), `σ_i = σ_{R,i}` (E17),
  `Z_i = Φ((r_max − μ_i)/σ_i) − Φ((r_min − μ_i)/σ_i)`,

      π_i = P_{D,i} Z_i,   f_i(z) = N(z; μ_i, σ_i²) 1_W(z) / Z_i                               (F14)

- beam with no optical return: `π_i = P_FA |W| / L_c`, `f_i = 1_W / |W|` (uniform).

Wrap outliers and the robustness floor ε are ignored in the design model (they are kept in the belief
update). If `π_i = 0` the density is irrelevant. The design model is defined for K = 1 geometry only.

**Physics-blind design model** (baseline): `π_i = 1` if the ideal ray hits a surface with
`r_min ≤ r ≤ r_max`, else `π_i = 0`; `μ_i` = geometric range, `σ_i = σ_ref = 12 mm`, no truncation
(`Z_i = 1`). Same geometry and line of sight; no dependence on reflectivity, incidence, fog or sunlight.

### 7.2 Pairwise Bhattacharyya coefficients

For hypotheses m, m′ and beam i (dominating measure: an atom for "no report" plus Lebesgue measure on W):

    BC_i = √((1−π)(1−π′)) + √(π π′) ∫_W √(f f′) dz                                              (F15)

with the overlap integral in closed form:
- Gaussian–Gaussian: `C · Z_* / √(Z Z′)`, `C = √(2σσ′/(σ²+σ′²)) exp(−(μ−μ′)²/(4(σ²+σ′²)))`,
  `Z_*` the window mass of `N(μ_*, σ_*²)`, `σ_*² = 2σ²σ′²/(σ²+σ′²)`, `μ_* = (μσ′² + μ′σ²)/(σ²+σ′²)`;
- uniform–uniform: 1;
- uniform–Gaussian: `(8πσ²)^{1/4} [Φ((r_max−μ)/(√2σ)) − Φ((r_min−μ)/(√2σ))] / √(Z |W|)`.

By conditional independence of beams given q (independent sample noise in the simulator),

    D_{mm′}(g) = −Σ_i log BC_i(m, m′; g)                                                        (F16)

for one scan from viewpoint g (only candidate beams contribute; others have BC = 1).

### 7.3 Utility: a lower bound on expected information gain

Support hypotheses `{q^{(m)}}_{m=1..M}`, M = 24, are drawn once per planning step from the belief by
systematic resampling (weights 1/M) and shared by all candidates. For the finite-support design channel,

    I_M(q; Z_g) ≥ Î(g) = −(1/M) Σ_m log[(1/M) Σ_{m′} exp(−D_{mm′}(g))]                          (F17)

(pairwise Chernoff-½ bound on mixture entropy, Kolchinsky & Tracey 2017), with `0 ≤ Î ≤ log M`. (F17)
bounds the mutual information of the *finite-support design channel*; it is not a bound for the continuous
posterior or for the waveform simulator, and it saturates at `log M` (less if support points coincide).

### 7.4 Next-best-viewpoint selection with travel time

Navigation grid: 0.1 m cells over the arena, free cells and edge costs as defined in §7.5. Travel time from
the robot's cell (the nearest free cell) to every free cell: Dijkstra on the 8-connected free grid (diagonal
moves only between cells whose two orthogonal neighbours are free).

Candidate viewpoints `G`: free, reachable cells of the 0.5 m lattice (grid cells whose row and column
indices are both multiples of 5) within 4.0 m (horizontal) of the target keep-out rectangle, plus the
robot's current cell; each with heading towards the belief mean `q̂` (the azimuth FOV is 240°, so the heading
matters little). Score and choice:

    J(g) = Î(g) − λ_t t(g),   g* = argmax_{g ∈ G} J(g),   λ_t = 0.05 nat/s                      (F18)

(`t(g)` = travel time to g; ties → smaller t, then lowest row-major cell index). **Goal persistence:** if the
previously chosen goal `g_prev` is still a candidate and `J(g*) − J(g_prev) < δ = 0.05 nat` (both evaluated
with the current support and travel times), the robot keeps `g_prev`; this avoids switching between
near-equivalent viewpoints. The robot then follows the shortest path to the goal for one decision step (§7.5)
and replans (receding horizon). Candidates whose upper bound `log M − λ_t t(g)` is below the best score found
are not evaluated (this cannot change the choice), and beams whose design distribution is identical for all
support hypotheses are dropped before the pairwise computation (they have BC = 1). Scans taken on the way update the belief
but are not counted in (F18) (a myopic, one-scan-at-the-goal criterion).

### 7.5 Path following (shared by all controllers)

Given a path (polyline from the robot to a goal), the look-ahead point is the path point 0.6 m ahead of the
robot's closest path point (or the goal). With heading error `e_θ` to the look-ahead point, remaining path
length `d_goal`, clearance `c(p)` (distance from p to the nearest keep-out rectangle) and `m_s = 0.05 m`:

    ω = clip(2.0 e_θ, ±ω_max)
    v̄(v, d) = max(a_max(−Δt/2 + √(Δt²/4 + 2(d − vΔt/2)/a_max)), 0)   (0 if the root is not real)
    v_cmd = min(v_lim(p), v̄(v, c(p) − m_s), v̄(v, d_goal)) · max(cos e_θ, 0)                    (F19)
    a = clip((v_cmd − v)/Δt, ±a_max)

`v̄(v, d)` is the largest end-of-step speed v′ for which the distance travelled during the step with
constant acceleration, `(v + v′)Δt/2`, plus the braking distance `v′²/(2a_max)` does not exceed d. Because
the control is held for a whole decision step, this discrete-time bound (not the continuous `√(2ad)`)
guarantees by induction that the robot can always brake before a keep-out and stops at the goal without
overshoot. Within `0.15 m` of the goal the robot brakes (`a = −min(a_max, v/Δt)`) and turns towards a given
point (the belief mean). If the resulting step is infeasible, the fallbacks are tried in order: same ω with
full braking; `ω = 0` with full braking; the controller reports a `fallback` flag.

The navigation grid uses the consistent cruise speed `s(p) = min(v_lim(p), v_c(p))`, where `v_c` solves
`v Δt + v²/(2a_max) = c(p) − m_s`: a cell is free iff its centre is feasible and `s ≥ 0.15 m/s`, and the edge
cost is `length / min(s at the two centres)`, so shortest paths keep away from obstacles where the robot
would have to crawl.

## 8. Baselines

- **Fixed route:** an inspection loop that does not use measurements: waypoints
  `(1.0,4.0) → (2.6,6.7) → (5.8,6.7) → (9.9,6.7) → (9.9,1.3) → (5.8,0.75) → (2.6,1.1) → (1.0,4.0)`
  (26.5 m: north side, east corridor, south side). The controller keeps its arc-length progress along the
  loop and follows the next 3 m of it with §7.5, repeating until the trial ends.
- **Physics-blind planner:** §7 with the physics-blind design model of §7.1.

All three controllers use the same simulator, the same belief update (§6, full physics likelihood), the
same number of scans, the same path follower and the same motion constraints. The only difference is how
the next goal is chosen.

## 9. Trials and metrics (`mts/sim.py`, `mts/metrics.py`)

A trial: target `q ~ Uniform(R_T)`; T = 60 decision steps (30 s, 60 scans; the viewpoints around the
target region are 9–16 s of travel from the start, so shorter trials would mostly measure driving). Random streams are derived from
`SeedSequence([experiment_seed, trial, purpose])` with separate purposes for the target draw, the particle
filter, the planner support, and the measurement noise of each scan index, so paired controllers see the
same target and the same per-scan noise stream. Per step the trial records state, control, goal, fallback
flag, observation summary, `q̂_k`, `Σ̂_k`, ESS, tempering stages, and planner scores.

    e_k = ‖q̂_k − q‖                                                    (position error)        (F20)
    τ = min{k Δt : e_j ≤ 0.10 m for j = k … k+4}  (dwell of 5 scans; ∞ if none)                 (F21)
    time score  τ̃ = min(τ, T Δt)   (restricted; failures are τ̃ = T Δt and counted separately)
    (index k = 0 is the prior at t = 0; k counts scans, so e_k is the error after k scans at time k Δt)
    H_k = ½ log det(2πe (Σ̂_k + (1 mm)² I)),   NEES_k = (q̂_k − q)ᵀ (Σ̂_k + (1 mm)² I)⁻¹ (q̂_k − q)   (F22)

Conditions (environment × target): **C1** clear, dark (V = 20 km, 0 sun), black target; **C2** fog
(V = 20 m), black; **C3** sunlight (1 sun), black; **C4** control: clear, dark, white target (ρ = 0.9).

## 10. Software layout and frozen interfaces

    src/lidar_sim/…        vendored midterm package + scene3d, scan3d, pose/trace additions (§3)
    src/mts/config.py      frozen dataclasses and defaults (sensor, robot, belief, planner, conditions)
    src/mts/world.py       World (scene builder, keep-outs, terrain, prior region, nav grid), yard()
    src/mts/robot.py       unicycle step (F7–F8), pose trajectory, feasibility, path follower (F19)
    src/mts/predict.py     Observation, ScanGeometry, fast predictor (§5.2), design models (F14)
    src/mts/belief.py      ParticleBelief (§6)
    src/mts/info.py        Bhattacharyya matrix and MI bound (F15–F17)
    src/mts/control.py     Action, Controller protocol
    src/mts/planner.py     ViewpointPlanner (aware / blind) (§7)
    src/mts/route.py       FixedRoute baseline (§8)
    src/mts/sim.py         run_trial, Trial record, RNG streams
    src/mts/metrics.py     F20–F22, paired statistics
    src/mts/viz.py         figures and animation
    sim/run.py, sim/registry.yaml, sim/experiments/F*.py    registered experiments (vv-plan.md)

Import boundaries: `lidar_sim` never imports `mts`; `mts.belief`, `mts.planner`, `mts.info`, `mts.predict`
never import `lidar_sim.physics.waveform` (inference uses the analytic model only, ADR-0005) and never
receive the truth scene, `Scan.truth`, or the true `q`: they are given the *background* world, the
`Observation`, the robot state history and the environment. All randomness flows from explicit
`numpy.random.Generator` arguments.

Frozen interface signatures are in `docs/design/interfaces.md`.
