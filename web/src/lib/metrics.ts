// ==========================================================
// web/src/lib/metrics.ts
// ==========================================================
// Metric helpers shared by the page and scripts/check_models.mjs; they
// mirror src/mts/metrics.py exactly.

// ------------------------------------------
// dwellStart
// ------------------------------------------
// Index where the first run of `dwell` errors within `tol` starts, or null.
export function dwellStart(error: readonly number[], tol = 0.1, dwell = 5): number | null {
  if (!Number.isInteger(dwell) || dwell < 1) throw new Error("Expected positive integer dwell");
  let count = 0;
  for (let i = 0; i < error.length; i++) {
    count = error[i] <= tol ? count + 1 : 0;
    if (count >= dwell) return i - dwell + 1;
  }
  return null;
}

// ------------------------------------------
// timeScore
// ------------------------------------------
// Time to localize: the dwell start in seconds, capped at the trial length.
export function timeScore(error: readonly number[], dt = 0.5, tol = 0.1, dwell = 5): number {
  if (!(dt > 0)) throw new Error("Expected positive dt");
  const start = dwellStart(error, tol, dwell);
  return Math.min(start === null ? Infinity : start * dt, Math.max(0, error.length - 1) * dt);
}

// ------------------------------------------
// sensorIndex
// ------------------------------------------
// Index of a curve in the 6 × 5 × 5 sensor grid (reflectivity, fog, sun).
export function sensorIndex(rhoIdx: number, visIdx: number, sunIdx: number): number {
  if (
    ![rhoIdx, visIdx, sunIdx].every(Number.isInteger) ||
    rhoIdx < 0 ||
    rhoIdx >= 6 ||
    visIdx < 0 ||
    visIdx >= 5 ||
    sunIdx < 0 ||
    sunIdx >= 5
  )
    throw new Error("Invalid sensor grid index");
  return (rhoIdx * 5 + visIdx) * 5 + sunIdx;
}
