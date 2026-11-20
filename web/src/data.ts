// ==========================================================
// web/src/data.ts
// ==========================================================
// Types of the JSON files in public/data/, written by scripts/build_data.py
// and scripts/export_decisions.py, with the loader and formatters. Everything
// physical is computed in Python; the browser only selects, formats and draws.

// ------------------------------------------
// Shared types
// ------------------------------------------
// Condition and controller ids, and vectors in metres.
export type Cond = "C1" | "C2" | "C3" | "C4";
export type Ctrl = "info" | "blind" | "fixed";
export type V2 = [number, number];
export type V3 = [number, number, number];
export type Rect = [number, number, number, number]; // xmin, xmax, ymin, ymax

// ------------------------------------------
// Conditions and controllers
// ------------------------------------------
// Display order and the words used for each controller.
export const CONDS: Cond[] = ["C1", "C2", "C3", "C4"];
export const CTRLS: Ctrl[] = ["info", "blind", "fixed"];

export const CTRL_LABEL: Record<Ctrl, string> = {
  info: "Information-driven",
  blind: "Physics-blind",
  fixed: "Fixed route",
};
export const CTRL_SHORT: Record<Ctrl, string> = { info: "Info", blind: "Blind", fixed: "Fixed" };
export const CTRL_NOTE: Record<Ctrl, string> = {
  info: "Plans each move with the full sensor physics.",
  blind: "Same planner, but assumes every visible point is seen.",
  fixed: "Drives a preset inspection loop.",
};

// ------------------------------------------
// World
// ------------------------------------------
// The yard, the fixed route and the four conditions (world.json).
export type ConditionInfo = { label: string; rho: number; visibilityM: number; sun: number };

export type World = {
  version: 1;
  arena: Rect;
  boxes: { lo: V3; hi: V3 }[];
  rough: Rect[];
  prior: { lo: V3; hi: V3 };
  targetHalf: V3;
  start: [number, number, number, number];
  fixedRoute: V2[];
  routeClosed: boolean;
  dt: number;
  steps: number;
  toleranceM: number;
  dwellSamples: number;
  conditions: Record<Cond, ConditionInfo>;
  conditionsOrder: Cond[];
};

// ------------------------------------------
// Replay
// ------------------------------------------
// One trial: every controller's states, belief and error (replay/Cx/tt.json).
export type ReplayCtrl = {
  states: [number, number, number][];
  mean: V3[];
  covXY: [number, number, number][];
  error: number[];
  dwellStart: number | null;
  timeScoreS: number;
  goals: (V2 | null)[] | null;
  nDetected: number[];
  nTargetDetected: number[];
};

export type Replay = {
  version: 1;
  condition: Cond;
  trial: number;
  seed: number;
  dt: number;
  qTrue: V3;
  controllers: Record<Ctrl, ReplayCtrl>;
};

// ------------------------------------------
// SensorGrid
// ------------------------------------------
// Detection and range-noise curves over the parameter grid (sensor.json).
export type SensorGrid = {
  version: 1;
  rangeM: number[];
  rho: number[];
  visibilityM: number[];
  sun: number[];
  extinctionPerM: number[];
  reportingWindow: V2;
  curves: { pd: number[]; sigmaM: number[]; r50M: number }[];
};

// ------------------------------------------
// Decisions
// ------------------------------------------
// Three planning steps of fog trial 0 (decisions.json).
export type Candidate = {
  id: number;
  xy: V2;
  heading: number;
  informationNat: number;
  travelS: number;
  scoreNat: number;
  evaluatedByPolicy: boolean;
};

export type Decision = {
  k: number;
  state: [number, number, number, number];
  mean: V3;
  previousId: number | null;
  bestId: number;
  selectedId: number;
  goalKept: boolean;
  candidates: Candidate[];
  particlesXY?: V2[];
  weights?: number[];
  support?: V2[];
};

export type Decisions = {
  version: 1;
  condition: Cond;
  trial: number;
  seed: number;
  controller: Ctrl;
  lambda: number;
  switchMargin: number;
  supportSize: number;
  steps: Decision[];
};

// ------------------------------------------
// Results
// ------------------------------------------
// Paired time scores, summaries and the registered tests (results.json).
export type Summary = {
  n: number;
  successCount: number;
  successRate: number;
  successWilson95: V2;
  medianTimeScoreS: number;
  timeScoreCi95: V2;
  meanTimeScoreS: number;
  medianFinalErrorM: number;
  finalErrorCi95: V2;
  maxFinalErrorM: number;
  meanPathLengthM: number;
  noTargetReturnFraction: number;
  fallbackSteps?: number;
  infeasibleSteps?: number;
  meanPlanTimeS?: number;
  meanUpdateTimeS?: number;
};

export type Pair = {
  trial: number;
  timeScoreS: [number, number, number];
  dwellStart: [number | null, number | null, number | null];
  finalErrorM: [number, number, number];
};

export type Comparison = {
  hypothesis: "H1" | "H2";
  condition: Cond;
  against: "fixed" | "blind";
  pRaw: number;
  pHolm: number;
  medianPairedDifferenceS: number;
  differenceCi95: V2;
  infoFaster: number;
  ties: number;
  infoSlower: number;
  infoOnlySuccess?: number;
  otherOnlySuccess?: number;
};

export type Results = {
  version: 1;
  n: number;
  steps: number;
  dt: number;
  paired: Record<Cond, Pair[]>;
  summary: Record<Cond, Record<Ctrl, Summary>>;
  comparisons: Comparison[];
  control: { medianPairedDifferenceS: number; differenceCi95: V2; marginS: number };
  verdicts: { H1: boolean; H2: boolean; H3: boolean };
};

// ------------------------------------------
// Evidence
// ------------------------------------------
// Registered checks, deviations and the test run (evidence.json).
export type Experiment = {
  id: string;
  title: string;
  exploratory: boolean;
  source?: string;
  criteria: { key: string; text: string; supported: boolean }[];
  observations: { label: string; value: number; unit: string; source?: string }[];
  provenance: { seed: number; sourceSha256: string; timestamp: string };
};

export type Evidence = {
  version: 1;
  experiments: Experiment[];
  source?: string;
  deviations: { text: string; source: string }[];
  tests: {
    command: string;
    sourceSha256: string;
    collected: number;
    passed: number | null;
    failed: number | null;
    skipped: number | null;
    deselected: number | null;
    executedAt: string | null;
    statCollected?: number;
    statCommand?: string;
  };
};

// ------------------------------------------
// Manifest
// ------------------------------------------
// File list and sizes of the data (manifest.json).
export type Manifest = { version: 1; decisionAvailable: boolean; totalGzipBytes: number };

// ------------------------------------------
// load
// ------------------------------------------
// Fetches a data file once; a failed request is forgotten so a retry refetches.
const cache = new Map<string, Promise<unknown>>();

/** Fetch a data file once; a failed request is forgotten so a retry refetches. */
export function load<T>(path: string): Promise<T> {
  let p = cache.get(path) as Promise<T> | undefined;
  if (!p) {
    p = fetch(`data/${path}`).then(r => {
      if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
      return r.json() as Promise<T>;
    });
    p.catch(() => cache.delete(path));
    cache.set(path, p);
  }
  return p;
}

// ------------------------------------------
// replayPath
// ------------------------------------------
// Path of one trial's replay file.
export const replayPath = (c: Cond, trial: number) => `replay/${c}/${String(trial).padStart(2, "0")}.json`;

// ------------------------------------------
// Formatters
// ------------------------------------------
// Seconds, metres and p-values for display. Time scores are multiples of
// 0.25 s, so two decimals are exact; lengths switch to mm below 0.1 m.
export const fmtS = (s: number) => `${s < 0 ? "−" : ""}${Number(Math.abs(s).toFixed(2))} s`;

/** Metres with a unit that suits the size: mm below 0.1 m. */
export function fmtM(m: number): string {
  if (m < 0.1) return `${(m * 1000).toFixed(m < 0.01 ? 1 : 0)} mm`;
  return `${m.toFixed(2)} m`;
}

export function fmtP(p: number): string {
  if (p >= 0.001) return p.toFixed(4);
  const [mant, exp] = p.toExponential(1).split("e");
  return `${mant}×10${superscript(Number(exp))}`;
}

// ------------------------------------------
// superscript
// ------------------------------------------
// Unicode superscript digits for exponents.
const SUP: Record<string, string> = {
  "-": "⁻",
  "0": "⁰",
  "1": "¹",
  "2": "²",
  "3": "³",
  "4": "⁴",
  "5": "⁵",
  "6": "⁶",
  "7": "⁷",
  "8": "⁸",
  "9": "⁹",
};
const superscript = (n: number) =>
  String(n)
    .split("")
    .map(c => SUP[c] ?? c)
    .join("");
