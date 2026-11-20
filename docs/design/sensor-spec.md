# Design Spec — Surface-Aware Probabilistic 2D AMCW LiDAR Simulator

| Field | Value |
|---|---|
| Status | **v1.2 — APPROVED v1.0 + ADR-0008 (CFAR) + ADR-0009 (two-sided ε floor)** |
| Date | 2026-10-07 |
| Course | ECE 4240 Robot Perception — midterm project |
| Companion | [`final-spec.md`](final-spec.md): the 3D final project built on this sensor model |
| Decisions | [`decisions/`](decisions/) ADR-0001 … ADR-0009 |

This spec is **normative**: code, tests and the report cite its equation IDs `(E#)`. Any change to
an equation requires an ADR and a spec version bump.

---

## 1. Purpose and success criteria

**Goal.** Model and simulate a virtual 2D amplitude-modulated continuous-wave (AMCW) LiDAR,
inspired by the Hokuyo URG-04LX as modeled by Heiden et al. (ICRA 2020). The model must make the
sensor's measurements **and** its field of view depend, through explicit equations, on
(i) sensor pose, (ii) target state and surface properties, and (iii) environment (fog, ambient light).

| Brief tool | Realized by | Equations |
|---|---|---|
| Physics-based measurement model | radiometry, atmosphere, photocurrent waveform | E6–E11 |
| Estimation-theoretic model | DFT phase/amplitude estimator, dual-frequency unwrap, phase CRLB, pose MLE + FIM/CRB | E12–E14, E17, E22–E24 |
| Probabilistic model | analytic per-beam likelihood: Rice detection, CRLB noise, wrap outliers, mixed returns, false alarms | E16–E21 |
| Field of view | geometric sector + SNR-dependent *effective* FOV | E3, E3a |
| Line-of-sight visibility | nearest-hit ray casting, occlusion | E5 |

**Success criteria (midterm "done").**
1. Every edge of the brief's Fig. 1 (§6.1) is demonstrated by a registered experiment with a figure
   and a quantitative metric.
2. The analytic model (§4.6, implemented independently, ADR-0005) is statistically consistent with
   the waveform simulator (§4.5) within the V&V tolerances in the regimes where it claims validity,
   and its breakdown regimes are quantified.
3. Pose MLE empirical covariance agrees with the CRB in the smooth, high-SNR regime; departures
   elsewhere are measured and explained.
4. Every figure regenerates from one command (`python -m sim.run X##`) with recorded seed and config.
5. The physics/probability layers contain no 2D-specific code (§3.2), so the 3D extension (§8)
   adds geometry, scan pattern and pose parameterization only.

**Non-goals (midterm):** 3D; refraction/glass; rain-drop or fog backscatter returns; multi-bounce
(> 1 specular bounce); speckle; hardware calibration; GUI.

**Honesty constraint.** This is a *physics-inspired virtual sensor*, not a validated replica of the
URG-04LX. Datasheet values set magnitudes only; noise constants are calibrated to plausible
accuracy (ADR-0004). The report states this explicitly.

---

## 2. Conventions

- **Units:** SI in all code (m, s, rad, Hz, W, A). Config may accept `*_deg` / `*_km` helpers,
  converted once at the config boundary. Visibility `V` is stored in **metres**.
- **Arrays:** `float64`. Vectors have shape `(..., d)`, `d ∈ {2,3}`. Per-scan arrays use axes
  `B` (beams), `K` (sub-rays per beam), `J` (optical returns per beam, §3.1): e.g. `(B,K,d)`, `(B,J)`.
- **Poses:** `Pose(R: (d,d) ∈ SO(d), t: (d,))`, `p_W = R p_S + t`. 2D parameter vector
  `ξ = (x, y, ψ)`; conversions only in `core/pose.py`.
- **Frames:** world `W`; sensor `S` with `+x` = scan centre, angles counter-clockwise (CCW).
- **Wrapping:** `wrap₀(φ) ∈ [0, 2π)`, `wrapπ(φ) ∈ [−π, π)`.
- **Randomness:** every stochastic function takes `rng: np.random.Generator`; no global RNG.
  Independent streams via `np.random.SeedSequence(seed).spawn(n)`.
- **Equation IDs:** a docstring cites `Implements (E#)`; the report uses `\label{eq:E#}`.

---

## 3. Architecture

```
src/lidar_sim/
  core/        pose.py, types.py, config.py, constants.py
  geometry/    base.py (Scene protocol), scene2d.py (segments)            [3D: scene3d.py]
  sensor/      scan_pattern.py (ScanPattern, Planar2DPattern), beam.py (sub-ray quadrature),
               lidar.py (LidarSensor: trace → measure)
  physics/     radiometry.py, atmosphere.py, specular.py, waveform.py
  world/       materials.py, targets.py, environment.py, scenes.py (canonical scenes)
  models/      detection.py, beam_model.py (analytic likelihood)        ← independent of physics/waveform.py
  estimation/  fisher.py, pose_mle.py
  viz/         plots.py, animate.py
sim/           run.py, registry.yaml, experiments/X##_*.py, calibrate.py, out/ (git-ignored)
tests/         unit/ property/ integration/ statistical/ contract/ oracles/
```

**Dependency rule:** `core ← geometry ← world ← sensor ← physics`; `models` depends on
`core, sensor.types, physics.radiometry/atmosphere` but **must not import `physics.waveform`**
(enforced by a test, ADR-0005). `estimation ← models`. `viz`, `sim` may import anything; `src`
never imports `viz` or `sim`.

### 3.1 Interfaces and records

```python
class Scene(Protocol):
    dim: int
    def intersect(self, origins: (N,d), dirs: (N,d), t_scene: (N,)) -> HitBatch
    # nearest hit with r > r_eps (no upper gate). HitBatch: hit (N,) bool, r (N,),
    # point (N,d), normal (N,d) (oriented so normal·dir ≤ 0), material_id (N,), object_id (N,),
    # surface_id (N,) (unique per segment/facet; -1 where no hit)

class ScanPattern(Protocol):
    dim: int
    def beam_dirs_sensor(self) -> (B,d)            # unit beam axes in S
    def fire_times(self) -> (B,)                   # s from scan start
    def subrays(self, K: int) -> tuple[(B,K,d), (B,K)]   # sub-ray dirs in S, quadrature weights Σ_k w=1
    def beam_angle(self) -> (B,)                   # 2D-only metadata (plots); never used by physics

@dataclass(frozen=True)
class RadiometricScan:      # output of LidarSensor.trace — deterministic, shared by BOTH paths
    pose: Pose | Callable[[float], Pose]; fire_time: (B,)
    dir_world: (B,K,d); w: (B,K)        # sub-ray directions and quadrature weights
    # Return axis J: every optical return that enters the phasor sum (E10). J = K when the
    # `specular` flag is off (return j ↔ sub-ray j); J = 2K when on (direct + bounced per sub-ray).
    # Unused entries: hit=False, power=0, r_path=NaN, ids=-1.
    sub_idx: (B,J) int                  # sub-ray that produced the return
    hit: (B,J) bool; r_path: (B,J)      # one-way optical path length [m] (r₁ + r₂ for bounces)
    power: (B,J)                        # received optical power [W] (E7 / E9)
    normal: (B,J,d); cos_inc: (B,J)     # at the final diffuse reflection
    object_id, material_id, surface_id: (B,J) int
    bounce: (B,J) bool                  # return arrived via a specular bounce
    P_amb: float                        # ambient power at detector [W]

@dataclass(frozen=True)
class Scan:                 # output of LidarSensor.measure — what a real sensor reports
    detected: (B,) bool; range: (B,) [m], NaN iff not detected
    amplitude: (B,2) [A]  (Â at f1,f2; diagnostics only, NOT used by the likelihood)
    range_raw: (B,) [m]   (R̂ from E14 before the detection/window gate of E15; diagnostics/V&V only)
    fire_time: (B,); truth: RadiometricScan  (for evaluation only)

@dataclass(frozen=True)
class BeamPrediction:       # output of BeamModel.predict — analytic, from RadiometricScan only
    A: (B,2); sigma_n: (B,2); snr: (B,2); P_D: (B,); P_FA: (B,); r_bar: (B,); sigma_R: (B,)
    out_prob: (B,6); out_shift: (B,6)   # wrap-outlier components k=-3..-1,+1..+3 (E18)
    hit_any: (B,) bool                  # any return with power > 0 (selects the E20 branch)
    mixed: (B,) bool                    # E19 definition
    smooth: (B,) bool                   # all K sub-rays hit the same surface_id, no bounce, P_D > 0.5
                                        # (only smooth beams enter E22–E24 and E23)

class LidarSensor:
    def trace(self, pose, scene, env) -> RadiometricScan          # E1–E9
    def measure(self, trace, env, rng) -> Scan                    # E10–E15
    def scan(self, pose, scene, env, rng) -> Scan                 # measure(trace(...))

class BeamModel:
    def predict(self, trace, env) -> BeamPrediction               # E16–E19
    def log_likelihood_beams(self, scan, pred) -> (B,)            # E20
    def log_likelihood(self, scan, pred) -> float                 # E21
def fisher_information(model, sensor, xi, scene, env, param, kind="full"|"range_only") -> ((p,p), rank)  # E22–E24
def pose_mle(model, sensor, scan, scene, env, xi0) -> MLEResult                                    # E25
```

### 3.2 Dimension-agnosticism (scoped, per review)
- **2D-only (midterm):** `geometry/scene2d.py`, `Planar2DPattern`, 2D pose parameterization and the
  analytic pose Jacobian (E23), `beam_angle`, `viz`.
- **Dimension-agnostic (must pass `tests/contract/` for d = 2 and a toy d = 3 fixture):** everything in
  `physics/`, `models/`, `estimation/fisher.py` (except E23, which is supplied by the pose
  parameterization), `world/materials.py`, `world/environment.py`. These consume
  `(B,J)` / `(B,K,d)` / `(B,J,d)` arrays only — no angles, no cross products.

---

## 4. Mathematical model

### 4.1 Scan geometry, timing, FOV
Defaults (URG-04LX-like): `Θ = 240°`, `B = 682`, `Δα = Θ/(B−1)`, `r_min = 0.02 m`,
`r_max = 5.6 m`, `T_scan = 0.1 s` per mirror revolution, firing CCW from `−Θ/2`.

`α_i = −Θ/2 + iΔα`,  `u_i = [cos α_i, sin α_i]ᵀ`,  `d_i = R u_i`. **(E1)**

`τ_i = i Δα / ω`,  `ω = 2π / T_scan` (240° sector spans 66.7 ms). The sensor pose may be a function
`pose(τ)`; each beam uses `pose(τ_i)` and the scene at `t_scene = τ_i` (scan-time skew). **(E2)**

Geometric FOV: `S_geo(ξ) = { p : |atan2((Rᵀ(p−t))_y, (Rᵀ(p−t))_x)| ≤ Θ/2,  r_min ≤ ‖p−t‖ ≤ r_max }` (inclusive). **(E3)**

Effective FOV for a probe surface of reflectivity `ρ` facing the sensor:
`S_eff(ξ; ρ, env) = { p ∈ S_geo(ξ) : p visible (E5),  P_D(p; ρ, env) ≥ 0.5 }` (P_D from E16). **(E3a)**

### 4.2 Beam divergence (sub-ray quadrature) and LOS
Beam intensity profile is Gaussian in angle, `I(δ) ∝ exp(−2δ²/θ₀²)`. The quoted 40 mm diameter at
4 m is interpreted as the **1/e² full width** (ADR-0003), giving `θ₀ = 0.020/4 = 5 mrad` (1/e² half-angle).
Sub-rays use K-point Gauss–Hermite quadrature: with nodes/weights `(x_k, h_k)` for `∫e^{−x²}g(x)dx`,
`δ_k = θ₀ x_k / √2`,  `w_k = h_k / √π` (so `Σ w_k = 1`). Default `K = 5`; `K = 1` ⇒ `δ = 0, w = 1`
(ideal ray caster). In 2D, sub-ray `k` of beam `i` has direction `u(α_i + δ_k)`. **(E4)**

**LOS / ray–segment intersection (2D).** Ray `o + r d`, segment `a + u e`, `e = b − a`, `w = a − o`,
`×(p,q) = p_x q_y − p_y q_x`, `D = ×(d, e)`:
`r = ×(w, e)/D`,  `u = ×(w, d)/D`. Valid iff `|D| > ε_par`, `r > r_eps`, `0 ≤ u ≤ 1`.
Surfaces are two-sided; normal `n = ±perp(e)/‖e‖` with sign chosen so `n·d ≤ 0`.
The **nearest** valid hit over all segments is returned (occlusion). No `r_max` gate here. **(E5)**

### 4.3 Radiometry (direct return)
`cos θ = −n·d ∈ [0,1]`. **(E6)**

Extended Lambertian target, `P₀` = emitted optical power [W], `A_rx` = receiver aperture [m²],
`η` = optical efficiency:
```
P_k = P₀ · w_k · (ρ/π) · cos θ_k · (A_rx / r_k²) · T(r_k)² · η            [W]     (E7)
```
(`ρ/π` [sr⁻¹] × `A_rx/r²` [sr] → dimensionless; `cos θ` once — the reflected radiant intensity
toward the sensor of a Lambertian surface.) Near-field overlap (`r < 0.1 m`) not modeled.

**Atmosphere** (Kim et al., λ = 785 nm), `V` in m, `V_km = V/1000`:
```
σ_ext = (3.912 / V) · (λ / 550 nm)^(−q)      [m⁻¹],     T(r) = exp(−σ_ext r)          (E8)
q = 1.6 (V_km > 50);  1.3 (6 < V_km ≤ 50);  0.16 V_km + 0.34 (1 < V_km ≤ 6);
    V_km − 0.5 (0.5 < V_km ≤ 1);  0 (V_km ≤ 0.5)
```
3.912 = −ln 0.02 (Koschmieder 2 % contrast). Homogeneous medium; no fog backscatter (limitation).

**Ambient:** `P_amb` [W] at the detector, unmodulated; affects only noise (E11) and hence detection,
false alarms and effective FOV.

### 4.4 Optional single specular bounce (feature flag `specular`)
Mirror material `{ρ_m (specular), ρ_d (weak diffuse)}`. At a mirror hit at path `r₁` with normal `n₁`:
(a) a direct diffuse contribution from E7 with `ρ = ρ_d`;
(b) a reflected ray `d' = d − 2(d·n₁)n₁` from the hit point; if it hits a diffuse surface after
additional path `r₂` with incidence `cos θ₂` and reflectivity `ρ₂`, its return retraces the same path
(image-source argument), so
```
r_path = r₁ + r₂,    P = P₀ w_k ρ_m² (ρ₂/π) cos θ₂ (A_rx / r_path²) T(r_path)² η           (E9)
```
Both contributions are separate entries on the return axis J (spec §3.1; `bounce=True` for (b)) and
enter the phasor sum (E10), which produces phantom points behind the mirror.

### 4.5 Waveform-level measurement (truth simulator; ADR-0002, ADR-0004)
Direct (incoherent) detection: the APD photocurrent is proportional to optical **power**; ranging
phase is on the RF envelope, and returns from different surfaces add as envelope phasors with
amplitude ∝ power (ADR-0002). For each modulation frequency `f_j ∈ {f₁ = 46.55 MHz, f₂ = 53.2 MHz}`:
```
φ_{j,k} = 4π f_j r_{path,k} / c,     Z_j = 𝓡 m Σ_k P_k e^{iφ_{j,k}}      [A]               (E10)
```
`𝓡` = responsivity incl. APD gain [A/W], `m` = modulation depth.

Equivalent-time sampling, `N = 30`, sample phases `θ_n = 2πn/N` (N ≥ 3 ⇒ orthogonal regressors).
Independent sample sets for `f₁` and `f₂`:
```
s_{j,n} = 𝓡(P̄ + P_amb) + Re{ Z_j e^{−iθ_n} } + ε_{j,n},   ε ~ N(0, σ_n²) i.i.d.
σ_n² = σ_th² + κ_sh 𝓡 (P̄ + P_amb),    P̄ = Σ_k P_k                                       (E11)
```
`σ_th` [A] thermal/electronics noise; `κ_sh` [A] lumped shot-noise factor (≙ 2qMF·B_n). Both are
**calibrated constants** (ADR-0004), not first-principles values. Only the products
`G_dc = 𝓡 P₀ A_rx η` and `G_ac = m G_dc` are identifiable; config stores `P₀, A_rx, η, 𝓡, m` with
documented nominal values and calibration adjusts `P₀` only.

**Phase/amplitude estimation** (ML for known-frequency sinusoid in white Gaussian noise):
```
X_j = Σ_n s_{j,n} e^{+iθ_n},    φ̂_j = wrap₀(arg X_j),    Â_j = 2|X_j| / N                (E12)
```
Check: `Re{Z e^{−iθ}} e^{iθ} = ½(Z + Z* e^{2iθ})`, so `Σ_n → (N/2) Z`; DC terms vanish. Hence
`E[X_j] = (N/2) Z_j` and `arg X_j` estimates `arg Z_j` (verified by Monte Carlo, V&V O-10, S-02).

**Wrapped fine range:** `R_j = c φ̂_j / (4π f_j) ∈ [0, L_j)`,  `L_j = c/(2f_j)`
(`L₁ = 3.2201 m`, `L₂ = 2.8176 m`). **(E13)**

**Dual-frequency unwrap and fusion.** `Δf = f₂ − f₁`, `L_c = c/(2Δf) = 22.541 m`.
```
R_c = c · wrap₀(φ̂₂ − φ̂₁) / (4π Δf) ∈ [0, L_c)
k_j = round((R_c − R_j)/L_j),   R'_j = R_j + k_j L_j
R̂  = mod( (R'₁/σ²_{R,1} + R'₂/σ²_{R,2}) / (1/σ²_{R,1} + 1/σ²_{R,2}),  L_c )                    (E14)
```
The final fold `mod L_c` is required: the frequencies satisfy `L_c = 7 L₁ = 8 L₂` **exactly**, so
the whole estimator is consistent modulo `L_c`, and without the fold a target near 0 m whose coarse
estimate wraps to ≈ `L_c` yields spurious ≈ 22 m ranges (38 % of beams at r = 0.3 m, A/σ_n = 1.4,
Monte Carlo check, review round 2).
`σ_{R,j}` from E17 evaluated with `Â_j` and the known `σ_n` (the sensor knows its noise floor). Degenerate weights: `Â_j = 0` ⇒ `σ_{R,j} = ∞` (weight 0); if both weights are 0 (or both σ = 0) use equal weights.
Scenes must keep strong reflectors within `L_c` (aliasing beyond is out of scope).
Wrong `k_j` ⇒ gross errors of ≈ ±3 m (E18): a **physically derived outlier mechanism**.

**Detection and reporting (CFAR, ADR-0008).** The background level `𝓡 P_amb` is observable (sample DC
with no return), so the threshold tracks the background noise:
`γ = s_b √(−2 ln P_FA,0)`,  `s_b = √(2/N) · √(σ_th² + κ_sh 𝓡 P_amb)`,  `P_FA,0 = 10⁻⁴`.
Beam reported iff `Â₁ ≥ γ` **and** `R̂ ∈ [r_min, r_max]`; else `detected = False`, `range = NaN`.
Ambient raises `σ_n` and `γ` together: the false-alarm rate stays `P_FA,0` while SNR, P_D, range
accuracy and effective FOV degrade — the environment → measurement/FOV effect. **(E15)**

*Optional (flag `range_walk`, off by default):* `R̂ += b₀ ln(P_ref/P̄)` (applied where P̄ > 0, after the fold, before the window gate; included in `range_raw`). With **b₀ < 0** a dark target (P̄ < P_ref) appears closer — the URG "dark appears closer" bias.

### 4.6 Analytic probabilistic model (ADR-0005: implemented independently of E11–E15 code)
From the `RadiometricScan`, with `Z_j` per E10 (noise-free), `A_j = |Z_j|`, `σ_n` per E11,
quadrature std `s = σ_n √(2/N)`, per-sample SNR `SNR_j = A_j² / (2σ_n²)`:

**Detection (Rice).** `Â₁ ~ Rice(ν = A₁, σ = s)`:
`P_D = Q₁(A₁/s, γ/s) = scipy.stats.rice.sf(γ, b = A₁/s, scale = s)`;
no-signal (background) false alarm `P_FA = exp(−γ² / (2s_b²)) = P_FA,0` by construction of E15 (CFAR); `BeamPrediction.P_FA` always reports this background value. **(E16)**

**Range noise (phase CRLB).** For equally spaced `θ_n`, FIM is diagonal in (DC, amplitude, phase), so
`σ²_{φ,j} = 1/(N · SNR_j) = 2σ_n²/(N A_j²)` holds with unknown amplitude and DC.
`σ_{R,j} = c σ_{φ,j} / (4π f_j)`;   fused `σ_R² = (σ_{R,1}^{−2} + σ_{R,2}^{−2})^{−1}`
(= `(c/4π)² σ_φ² / (f₁² + f₂²)` when `σ_{φ,1} = σ_{φ,2}`). **(E17)**

**Wrap outliers (common-integer model; corrected in review round 2).** Let `e_j` be the phase errors
of `φ̂_j`. The rounding argument of `k_j` in E14 is, for **both** j,
`(R_c − R_j)/L_j − m_j = ζ`,  `ζ = (f₁ e₂ − f₂ e₁) / (2π Δf)`
(the same random variable for j = 1, 2). Hence both
unwrap integers err by the **same** integer `k = round(ζ)`; an "only one frequency wraps" event does not
occur. Linearized Gaussian phase errors give `ζ ~ N(0, σ_ζ²)`, `σ_ζ² = (f₁² σ²_{φ,2} + f₂² σ²_{φ,1})/(2πΔf)²`:
```
p_k = Φ((k + ½)/σ_ζ) − Φ((k − ½)/σ_ζ),   k ∈ {±1, ±2, ±3}  (|k| > 3 truncated; p_k < 1e-12 for P_D > 0.5)
shift_k = k Δ,   Δ = (g₁L₁ + g₂L₂)/(g₁ + g₂),  g_j = σ_{R,j}^{−2}    (Δ ≈ 2.992 m for equal σ_φ)     (E18)
P_wrap = Σ_{k≠0} p_k
```
`out_prob`, `out_shift` have shape `(B, 6)` ordered k = −3…−1, +1…+3. **Known approximation error:** near
threshold the Gaussian phase approximation **under-predicts** `P_wrap` by up to ≈ 10 % relative
(A/σ_n = 1.1: 0.208 predicted vs 0.218 simulated); exact at high SNR. Optional refinement: exact
phase-error pdf of a noisy phasor convolved for ζ (stretch).

**Expected (mixed-pixel-aware) range.** `r̄` = E13–E14 applied to the **noise-free** phases
`arg Z_j` with weights from `A_j`. For a single surface `r̄ = r_path`; for multiple surfaces it is the
phasor-resultant range, which can lie **outside** the interval spanned by the surfaces when they are
separated by more than ≈ `L_j/4` (phasor cancellation). Verified example: equal returns at 1 m and 3 m
give wrapped fine ranges R₁ = 0.389944 m, R₂ = 0.591201 m (E13), coarse 2.000 m, and fused/folded
**r̄ = 3.452998 m** — beyond both surfaces (oracle O-13). `mixed = (max − min of r_path over hit returns whose sub-ray has w > 0.01) > 0.05 m`. **(E19)**

**Per-beam likelihood** of observation `o_i = (detected, z)`; reporting window
`W = [r_min, r_max]`, robustness floor `ε = 10⁻³` split equally between the two outcomes (ADR-0009), `N(·)` Gaussian pdf:
```
g_i(z) = (1 − P_wrap) N(z; r̄, σ_R²) + Σ_{k≠0} p_k N(z; r̄ + kΔ, σ_R²)
M_i    = ∫_W g_i(z) dz                                                    (Gaussian CDFs)
beam hits a surface:   p(detected, z) = (1−ε) P_D g_i(z) + ε / (2|W|)
                       p(not detected) = (1−ε) [ (1 − P_D) + P_D (1 − M_i) ] + ε/2
beam hits nothing:     p(detected, z) = (1−ε) P_FA / L_c + ε/(2|W|),  p(not det.) = (1−ε)(1 − P_FA |W|/L_c) + ε/2   (E20)
```
`p(detected, z)` is a density defined **only for z ∈ W**; an observation with `detected = True` and
`z ∉ W` or non-finite is an invalid input (raise). Mass of `g_i` outside `W` (incl. negative z, which the
simulator folds to ≈ L_c) is not-reported mass. Normalized by construction (V&V O-16, P-09). The no-hit false-alarm range is
uniform on `[0, L_c)` because noise-only phases are uniform.

**Modeling assumption (tested by S-08):** the factorization `P_D · g_i(z)` treats detection (on Â₁) as
independent of the range error. For a noisy phasor, amplitude and phase errors are independent only
asymptotically (high SNR); near threshold the detected population is biased toward favorable noise.
The assumption is claimed in R1 and quantified (not claimed) in R2.

`log p(Z | ξ) = Σ_i log p(o_i | ξ)` (beams conditionally independent given ξ, scene, env). **(E21)**

### 4.7 Estimation: Fisher information, CRB, pose MLE
For pose parameters `ξ ∈ ℝ^p` (p = 3 in 2D), over beams with `smooth = True`:
```
range_only:  I = Σ_i P_D,i · J_iᵀ J_i / σ²_{R,i}                                         (E22)
full:        I = Σ_i [ P_D,i ( J_iᵀJ_i/σ²_{R,i} + 2 (∇σ_{R,i})ᵀ(∇σ_{R,i})/σ²_{R,i} )
                       + (∇P_D,i)ᵀ(∇P_D,i) / (P_D,i (1 − P_D,i)) ]                         (E24)
```
(E24 = Bernoulli detection information + Gaussian with ξ-dependent mean and variance; outlier
components and ε are neglected, so E24 is the Gaussian-core FIM — stated as such.)
CRB: `Cov(ξ̂) ⪰ I⁻¹` only if `I` is nonsingular (rank reported; null directions reported otherwise).

**Analytic range Jacobian (2D, planar hit, fixed world surface)**, `r = n·(a − t)/(n·d)`:
`∂r/∂t = −nᵀ/(n·d)`,  `∂r/∂ψ = −r (n·d⊥)/(n·d)`,  `d⊥ = ∂d/∂ψ = R(ψ)[−sin α, cos α]ᵀ`. **(E23)**
`∇σ_R`, `∇P_D` by central finite differences of `predict(trace(ξ))` (step 1e−5 m / 1e−6 rad);
E23 is also checked against finite differences (V&V O-17). E23 is evaluated at the max-weight sub-ray's return; for `smooth` beams (one surface) this equals ∂r̄/∂ξ of E19 up to second order in the sub-ray spread (exact for K = 1).

**Pose MLE.** `ξ̂ = argmax_ξ log p(Z | ξ)` (E21) via `scipy.optimize.minimize` (L-BFGS-B with
numerical gradient; Nelder–Mead fallback); initialized at `ξ_true + δ`, δ from a stated distribution;
convergence basin reported. **(E25)**

### 4.8 World model
- **Materials** (`materials.py`): `white 0.90`, `grey 0.50`, `black 0.05`, `mirror (ρ_m 0.95, ρ_d 0.02)`.
- **Targets:** rigid polygons with pose `T(t)` (constant velocity or piecewise-linear waypoints).
- **Environment:** `V` [m] (clear air default 20 000), `P_amb` [W] (default 0), `λ = 785 nm`.
- **Canonical scenes** (`scenes.py`, shared by tests & experiments; each documents its hand-computable
  facts): `single_wall`, `two_walls_occluded`, `corner`, `room_asym` (asymmetric 6 × 4 m room with
  dark/bright wall patches; the grey patch at x = 0 is visible only for headings near ψ = π),
  `post_before_wall`, `mirror_room`, `moving_box`, `corridor` (rank-deficient FIM fixture).
- **Default pattern note:** B = 682 is even, so no beam lies exactly on the sensor +x axis
  (α₃₄₀,₃₄₁ = ∓Δα/2); fixtures needing an on-axis beam use odd B.

---

## 5. Data flow
```
ξ / pose(τ), scene, env
 └─ trace: ScanPattern (E1,E2,E4) → world sub-rays → Scene.intersect (E5) → [bounce E9]
           → cos θ, P_k (E6–E8)                                      ⇒ RadiometricScan  (shared)
      ├─ measure (truth):  Z_j (E10) → samples+noise (E11) → φ̂, Â (E12) → R_j (E13)
      │                    → unwrap+fuse (E14) → detect/report (E15)  ⇒ Scan
      └─ predict (analytic): A, σ_n, SNR, P_D (E16), σ_R (E17), outliers (E18), r̄ (E19)
                                                                       ⇒ BeamPrediction
 log_likelihood(Scan, BeamPrediction) (E20,E21) → pose_mle (E25);  fisher_information (E22–E24)
```

---

## 6. Experiments

### 6.1 Fig.-1 edge coverage (brief: measurements and FOV depend on sensor state, target, environment)

| Edge | Experiments | Metric |
|---|---|---|
| Sensor pose → measurements | X01, X09 | range residual maps; r, P_r, SNR vs pose; pose CRB |
| Sensor pose → FOV | X02 | covered area / visible-target fraction vs yaw & position |
| Target state → measurements | X03 | r̂, P_D vs target range/orientation; skew distortion vs speed |
| Target features → measurements | X04, X07 | P_r, σ_R, P_D vs ρ × θ; mixed-pixel error vs ρ contrast |
| Target → FOV/visibility | X02, X03 | occluded-beam mask; target visibility vs pose |
| Environment → measurements | X05, X06 | P_D(r), RMSE(r), σ_n and threshold γ vs V and P_amb |
| Environment → FOV | X05, X06 | effective-FOV area / max range vs V, P_amb, ρ |

### 6.2 Registry (details & hypotheses in `sim/registry.yaml`)

| ID | Title | Sweep | Primary outputs |
|---|---|---|---|
| X01 | Sensor pose | translate/rotate sensor in `room_asym` | scan overlays; residuals; P_r & SNR vs pose |
| X02 | FOV & LOS | yaw sweep; post crossing wall | S_geo vs S_eff map; occlusion mask |
| X03 | Target state | box range, yaw, speed (0–5 m/s) | r̂, P_D vs target pose; scan-skew distortion |
| X04 | Surface | ρ ∈ [0.02,1] × θ ∈ [0°,85°] | P_r, σ_R, P_D heatmaps — analytic vs MC |
| X05 | Fog | V: 20 km → 20 m | P_D(r), RMSE(r), effective r_max vs V |
| X06 | Ambient light | P_amb sweep | σ_n, γ, σ_R, P_D vs P_amb (P_FA held at P_FA,0 by CFAR); S_eff shrink |
| X07 | Divergence / mixed pixels | post edge across beam; K = 1 vs 5; ρ contrast | apparent range vs edge offset (phantom points) |
| X08 | Phase ranging | SNR sweep (single wall) | σ_R vs CRLB (E17); outlier rate vs E18 |
| X09 | Pose estimation | MC pose MLE at 3 poses × 3 SNRs | Cov vs CRB ellipses (E22/E24); NEES |
| X10 | Model validity | all canonical scenes, regimes | analytic vs simulator: KS, Wilson CIs, log-score |
| M01 | Media | robot path in `room_asym`, fog front, moving box | MP4/GIF: scan, S_geo/S_eff, P_D overlay |
| X11* | Specular (stretch) | `mirror_room` | phantom returns behind mirror |

---

## 7. Configuration, reproducibility, performance
- Frozen dataclasses `SensorConfig`, `NoiseConfig`, `EnvConfig`, `ModelConfig`; defaults from §4;
  experiments override via `registry.yaml`. Each run writes `sim/out/E##/run.json`
  (resolved config, seed, git SHA + dirty flag, package versions, timestamp, wall time).
- Vectorized NumPy; target ≤ 10 ms per scan (682 × 5 sub-rays × ≤ 50 segments) and ≤ 5 min per
  experiment on a laptop.
- `pyproject.toml` at repo root; package `src/lidar_sim`. Runtime: `numpy, scipy, matplotlib, pyyaml`;
  dev: `pytest, hypothesis, ruff`; media: `imageio[ffmpeg]` (optional). Python ≥ 3.12.

## 8. 3D extension (final project, not now)
Adds `geometry/scene3d.py` (triangles, Möller–Trumbore, BVH), `Multilayer3DPattern`, SE(3) pose
parameterization (p = 6) and its Jacobian, and 2D Gauss–Hermite cubature for sub-rays. `physics/`,
`models/`, `estimation/fisher.py` unchanged and must pass `tests/contract/` at d = 3. (`fisher_information`'s default `param=Planar2DParam()` is a 2D convenience; 3D callers pass an SE(3) `PoseParam`.) Planned uses:
particle-filter target tracking via `log_likelihood`, information-driven planning via `fisher_information`.

## 9. Risks
| Risk | Mitigation |
|---|---|
| Model sprawl | Core = E1–E8, E10–E25. Specular, range-walk, Oren–Nayar are flags/stretch. |
| Analytic model ≠ simulator | Mismatch is a reported result (X10); independence enforced (ADR-0005). |
| Non-smooth likelihood (edges, wraps) | CRB on `smooth` beams only; local MLE init; basin study. |
| Parameter degeneracy (ρ, V, P₀) | One-factor-at-a-time sweeps; no joint fitting in core scope. |
| Over-claiming fidelity | "physics-inspired virtual sensor" framing; assumptions table in report. |
| Report > 10 pages | 6–8 figures in body; rest to appendix. |
