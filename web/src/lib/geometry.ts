// ==========================================================
// web/src/lib/geometry.ts
// ==========================================================
// Geometry helpers shared by the page and scripts/check_models.mjs.

// ------------------------------------------
// ellipse
// ------------------------------------------
// Two-standard-deviation semi-axes of an xy covariance, and the angle of
// the major axis, counter-clockwise from +x.
export function ellipse(covXY: readonly number[]): { a: number; b: number; angle: number } {
  if (covXY.length !== 3 || !covXY.every(Number.isFinite)) throw new Error("Invalid XY covariance");
  const [xx, xy, yy] = covXY;
  const gap = Math.hypot(xx - yy, 2 * xy);
  const hi = (xx + yy + gap) / 2;
  const lo = (xx + yy - gap) / 2;
  if (lo < -1e-10 || hi < -1e-10) throw new Error("Covariance is not positive semidefinite");
  return {
    a: 2 * Math.sqrt(Math.max(0, hi)),
    b: 2 * Math.sqrt(Math.max(0, lo)),
    angle: gap === 0 ? 0 : Math.atan2(2 * xy, xx - yy) / 2,
  };
}
