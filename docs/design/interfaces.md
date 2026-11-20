# Move to See — frozen interfaces (v0.2)

Signatures every module codes against. Shapes use N particles, P hypotheses, M support points, B beams,
B_c candidate beams, K sub-rays. Arrays are `float64` unless stated. Equation ids refer to `final-spec.md`.
Tracks: **A** = geometry / robot / simulation, **B** = inference / planning. Each module's independent oracle
tests belong to the other track and are written from the specification, not from the code.

## mts/config.py (A)

```python
@dataclass(frozen=True)
class RobotConfig:
    dt: float = 0.5; substep: float = 0.02; a_max: float = 1.0; w_max: float = 1.0
    v_max: float = 1.0; v_rough: float = 0.3; radius: float = 0.30; margin: float = 0.05
    h_s: float = 0.40; lookahead: float = 0.6; k_heading: float = 2.0; brake_margin: float = 0.05
    goal_tol: float = 0.15

@dataclass(frozen=True)
class BeliefConfig:
    n_particles: int = 2000; ess_frac: float = 0.5; a: float = 0.98; sigma_j: float = 0.003
    min_increment: float = 1e-6; chunk: int = 200_000      # max (hypothesis, beam) pairs per predictor chunk

@dataclass(frozen=True)
class PlannerConfig:
    M: int = 24; lambda_t: float = 0.05; lattice: float = 0.5; radius: float = 4.0; cell: float = 0.1
    sigma_ref: float = 0.012; switch_margin: float = 0.05

@dataclass(frozen=True)
class Condition:
    name: str; env: EnvConfig; target_material: int

RASTER: Raster3DConfig      # Θ=240°, n_az=121, ε∈[−10°,40°], n_el=11, T_scan=0.1, θ0=5 mrad
SENSOR: SensorConfig        # RASTER.sensor_config(n_subrays=9)
NOISE: NoiseConfig; MODEL: ModelConfig
CONDITIONS: dict[str, Condition]     # "C1".."C4" (spec §9)
T_STEPS = 60; ACC_TOL = 0.10; DWELL = 5
def make_sensor(n_subrays: int = 9) -> LidarSensor      # LidarSensor with Raster3DPattern(RASTER)
def make_model(n_subrays: int = 9) -> BeamModel
```

## mts/world.py (A)

```python
TARGET_OBJECT = 99          # object id of the target box in truth scenes
@dataclass(frozen=True)
class World:
    arena: tuple[float, float, float, float]            # x0, x1, y0, y1
    box_lo: np.ndarray; box_hi: np.ndarray              # (n, 3) obstacles incl. walls
    box_material: np.ndarray                            # (n,) int
    rough: np.ndarray                                   # (R, 4) rectangles x0, x1, y0, y1
    prior_lo: np.ndarray; prior_hi: np.ndarray          # (3,) region R_T of target centres
    target_half: np.ndarray                             # (3,) half-sizes s_T
    target_material: int
    start: np.ndarray                                   # (4,) x_0
    ground_material: int
    def background_scene(self) -> Scene3D               # obstacles (object ids 1..n) + ground (object 0)
    def truth_scene(self, q) -> Scene3D                 # background + target as the LAST box (TARGET_OBJECT)
    def target_box(self, q) -> tuple[np.ndarray, np.ndarray]   # (…,3) lo, hi for q (…,3)
    def envelope(self) -> tuple[np.ndarray, np.ndarray]        # (3,), (3,) box containing every possible target
    def keepouts(self, cfg: RobotConfig) -> np.ndarray         # (Q, 4) inflated rectangles (spec §4.2)
    def feasible(self, xy, cfg) -> np.ndarray                  # (…,) bool for xy (…,2)
    def clearance(self, xy, cfg) -> np.ndarray                 # (…,) distance to nearest keep-out, 0 inside
    def v_lim(self, xy, cfg) -> np.ndarray                     # (…,) F8
    def with_target_material(self, material: int) -> World
def yard(target_material: int = BLACK, rough: bool = True, dark_background: bool = False) -> World
    # also: World.in_rough(xy), World.n_boxes; GROUND_OBJECT = 0
```

## mts/robot.py (A)

```python
@dataclass(frozen=True)
class Trajectory:                                   # one decision step
    t0: float; states: np.ndarray                   # (S+1, 4) substep states, states[0] = start
    u: np.ndarray; substep: float; feasible: bool; world: World; cfg: RobotConfig
    def state(self, t: float) -> np.ndarray         # (4,), t absolute in [t0, t0 + S·substep]
    def pose(self, t: float) -> Pose                # F1 (uses h_s given at construction)
    @property
    def end(self) -> np.ndarray                     # (4,)
def step(world, cfg, x, u, t0: float = 0.0) -> Trajectory                     # F7–F8
def sensor_pose(x, cfg) -> Pose; def safe_speed(cfg, clearance); def end_speed_bound(cfg, v, d) -> float   # F19
def follow(world, cfg, x, path: np.ndarray, face: np.ndarray | None = None) -> tuple[np.ndarray, bool]
    # F19 with fallbacks; path (n, 2) polyline from the robot to the goal; returns (u (2,), fallback)
class NavGrid:
    def __init__(self, world, cfg: RobotConfig, cell: float = 0.1, v_min: float = 0.15)
    free: np.ndarray; speed: np.ndarray; xs: np.ndarray; ys: np.ndarray   # (ny, nx); cell-centre coordinates
    def centre(self, cell) -> np.ndarray                       # (2,)
    def cell_of(self, xy) -> tuple[int, int]                   # nearest free cell (iy, ix)
    def travel_time(self, xy) -> tuple[np.ndarray, np.ndarray] # (ny, nx) time [s] (inf unreachable), predecessors
    def path(self, pred, goal: tuple[int, int]) -> np.ndarray  # (n, 2) cell centres from start cell to goal
```

## mts/predict.py (B)

```python
@dataclass(frozen=True)
class Observation:
    detected: np.ndarray   # (B,) bool
    range: np.ndarray      # (B,) m, NaN where not detected
    fire_time: np.ndarray  # (B,) s
    @classmethod
    def from_scan(cls, scan: Scan) -> Observation   # copies the three public fields only

@dataclass(frozen=True)
class ScanGeometry:
    pose: PoseLike; fire_time: np.ndarray          # (B,)
    dir_world: np.ndarray; w: np.ndarray           # (B, K, 3), (B, K)
    origins: np.ndarray                            # (B·K, 3)
    background: HitBatch                           # nearest background hit per sub-ray (ids remapped, spec §5.2)
    beams: np.ndarray                              # (B_c,) int, candidate beam indices (envelope cull)

@dataclass(frozen=True)
class BeamBatch:                                   # E16–E19 for (hypothesis, candidate beam)
    beams: np.ndarray                              # (B_c,)
    P_D: np.ndarray; P_FA: np.ndarray; r_bar: np.ndarray; sigma_R: np.ndarray   # (P, B_c)
    hit_any: np.ndarray                            # (P, B_c) bool
    out_prob: np.ndarray; out_shift: np.ndarray    # (P, B_c, 6)
    on_target: np.ndarray                          # (P, B_c) bool: some sub-ray hits the hypothesised target

@dataclass(frozen=True)
class DesignBatch:                                 # F14 (or blind model)
    beams: np.ndarray                              # (B_c,)
    pi: np.ndarray; mu: np.ndarray; sigma: np.ndarray; Z: np.ndarray   # (M, B_c)
    uniform: np.ndarray                            # (M, B_c) bool (no optical return)

class Predictor:
    def __init__(self, world: World, env: EnvConfig, chunk: int = 200_000)   # uses world.background_scene()
    def geometry(self, pose: PoseLike, t0: float = 0.0, K: int = 9) -> ScanGeometry
    def trace(self, geom: ScanGeometry, q) -> RadiometricScan    # single hypothesis, all B beams (tests, §5.2)
    def predict(self, geom: ScanGeometry, q) -> BeamBatch        # q (P, 3)
    def loglik(self, geom: ScanGeometry, obs: Observation, q) -> np.ndarray   # (P,), F9
    def design(self, geom: ScanGeometry, q, kind: str = "aware") -> DesignBatch   # "aware" | "blind"
```

The per-beam log-likelihood must be the E20 likelihood of `lidar_sim.models.beam_model` (a thin array-level
entry point may be added there, provided the existing methods keep their results).

## mts/info.py (B)

```python
def log_bc(design: DesignBatch, r_min: float, r_max: float) -> np.ndarray   # (M, M, B_c) log BC_i, F15
def pair_distance(design, r_min, r_max) -> np.ndarray                       # (M, M) D, F16
def mi_lower_bound(D: np.ndarray, weights: np.ndarray | None = None) -> float   # F17 (nats)
```

## mts/belief.py (B)

```python
@dataclass(frozen=True)
class UpdateInfo:
    stages: int; resamples: int; ess: float; skipped: bool
class ParticleBelief:
    def __init__(self, world: World, cfg: BeliefConfig, rng: np.random.Generator)
    particles: np.ndarray; weights: np.ndarray                  # (N, 3), (N,) normalised
    def update(self, loglik: Callable[[np.ndarray], np.ndarray], rng) -> UpdateInfo   # F10–F12
    def mean(self) -> np.ndarray; def cov(self) -> np.ndarray; def ess(self) -> float
    def support(self, M: int, rng) -> np.ndarray                # (M, 3) systematic resample
```

## mts/control.py (A: Action, Controller), mts/planner.py (B), mts/route.py (A: FixedRoute)

```python
@dataclass(frozen=True)
class Action:
    u: np.ndarray                  # (2,)
    goal: np.ndarray | None        # (2,) chosen viewpoint / look-ahead target
    fallback: bool
    info: dict                     # planner diagnostics (scores, n candidates, Î at goal, …)
class Controller(Protocol):
    name: str
    def act(self, k: int, x: np.ndarray, belief: ParticleBelief, rng) -> Action
class ViewpointPlanner:            # spec §7.3–7.5
    def __init__(self, world, robot_cfg, planner_cfg, predictor: Predictor, kind: str = "aware")
class FixedRoute:                  # spec §8, in mts/route.py
    def __init__(self, world, robot_cfg, waypoints: np.ndarray | None = None, horizon: float = 3.0)
```

## mts/sim.py (A)

```python
def stream(seed, trial, purpose, k=None) -> Generator   # purpose: "target" | "filter" | "planner" | "noise" (with scan k)
def draw_target(world, seed, trial) -> np.ndarray
def make_controller(name, world, predictor, robot_cfg, planner_cfg) -> Controller   # "info" | "blind" | "fixed"
def run_trial(world: World, condition: Condition, controller: str, seed: int, trial: int,
              robot_cfg=RobotConfig(), belief_cfg=BeliefConfig(), planner_cfg=PlannerConfig(),
              steps: int = T_STEPS, q_true=None, keep: bool = False) -> Trial   # keep: store particles for media
@dataclass
class Trial:   # per-step arrays of length steps (+1 for states)
    q_true; states (T+1, 4); controls (T, 2); goals (T, 2); fallback (T,); n_detected (T,);
    n_target_detected (T,)  # evaluation only, from Scan.truth; infeasible (T,)
    q_hat (T+1, 3); cov (T+1, 3, 3); ess (T,); stages (T,); score (T,); wall_time (T, 2)  # planning, belief update [s]
```

## mts/metrics.py (B)

```python
def position_error(q_hat, q_true) -> np.ndarray                    # F20
def dwell_time(err, dt, tol=0.10, dwell=5) -> float                # F21 (inf if none)
def time_score(err, dt, tol=0.10, dwell=5) -> float                # min(τ, T·dt)
def entropy_proxy(cov) -> np.ndarray; def nees(q_hat, cov, q_true) -> np.ndarray   # F22
def wilson(k, n, conf=0.99) -> tuple[float, float]
def paired_wilcoxon(a, b, alternative="less") -> float             # p-value (scipy.stats.wilcoxon, zero_method="pratt")
def holm(pvalues) -> np.ndarray
def bootstrap_ci(x, stat=np.median, n=10_000, conf=0.95, rng=None) -> tuple[float, float]
```
