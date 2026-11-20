#!/usr/bin/env node
// ==========================================================
// web/scripts/check_models.mjs
// ==========================================================
// Checks the browser helpers against Python. Node 26 imports the same
// TypeScript files the page uses.
//
//   npm run test:models

import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { resolve } from "node:path";
import { ellipse } from "../src/lib/geometry.ts";
import { dwellStart, timeScore, sensorIndex } from "../src/lib/metrics.ts";

// ------------------------------------------
// Helpers
// ------------------------------------------
// Paths, JSON reading and a tolerance check.
const web = fileURLToPath(new URL("..", import.meta.url));
const read = path => JSON.parse(readFileSync(resolve(web, path), "utf8"));
const close = (actual, expected, tol = 1e-10) =>
  assert.ok(Number.isFinite(actual) && Math.abs(actual - expected) <= tol, `${actual} != ${expected} (atol=${tol})`);

// ------------------------------------------
// Ellipses
// ------------------------------------------
// Semi-axes and angle against the Python oracle, and the covariance rebuilt
// from them.
const cases = read("tests/fixtures/ellipses.json");
for (const row of cases) {
  const { a, b, angle } = ellipse(row.covXY);
  close(a, row.a);
  close(b, row.b);
  if (row.a !== row.b) close(Math.sin(angle - row.angle), 0);
  const c = Math.cos(angle),
    s = Math.sin(angle);
  const reconstructed = [
    (a * a * c * c + b * b * s * s) / 4,
    ((a * a - b * b) * c * s) / 4,
    (a * a * s * s + b * b * c * c) / 4,
  ];
  reconstructed.forEach((v, i) => close(v, row.covXY[i]));
}
assert.throws(() => ellipse([-1, 0, 1]));

// ------------------------------------------
// Time score and sensor index
// ------------------------------------------
// Boundary cases of the dwell rule and every index of the sensor grid.
for (const row of read("tests/fixtures/metrics.json")) {
  assert.equal(dwellStart(row.error, row.tol, row.dwell), row.dwellStart);
  close(timeScore(row.error, row.dt, row.tol, row.dwell), row.timeScoreS, 0);
}
for (const row of read("tests/fixtures/sensor.json"))
  assert.equal(sensorIndex(row.rhoIdx, row.visIdx, row.sunIdx), row.index);
assert.throws(() => sensorIndex(6, 0, 0));

// ------------------------------------------
// Replay files
// ------------------------------------------
// Every exported trial: its time score recomputed and every covariance valid.
let replays = 0,
  controllers = 0,
  ellipses = 0;
for (const condition of ["C1", "C2", "C3", "C4"]) {
  const names = readdirSync(resolve(web, `public/data/replay/${condition}`)).filter(n => n.endsWith(".json"));
  assert.equal(names.length, 40);
  for (const name of names) {
    const replay = read(`public/data/replay/${condition}/${name}`);
    assert.equal(replay.condition, condition);
    assert.equal(name, `${String(replay.trial).padStart(2, "0")}.json`);
    for (const row of Object.values(replay.controllers)) {
      assert.equal(row.error.length, 61);
      assert.equal(dwellStart(row.error), row.dwellStart);
      close(timeScore(row.error, replay.dt), row.timeScoreS, 0);
      for (const cov of row.covXY) {
        ellipse(cov);
        ellipses++;
      }
      controllers++;
    }
    replays++;
  }
}
assert.equal(replays, 160);
assert.equal(controllers, 480);
assert.equal(ellipses, 29280);

// ------------------------------------------
// Report
// ------------------------------------------
// One line per group of checks.
console.log(`PASS: ${cases.length} Python ellipse oracles; dwell/time boundary cases; 150 sensor indices.`);
console.log(
  `PASS: ${replays} replay files, ${controllers} controller records, ${ellipses} covariances; all dwellStart/timeScoreS match.`,
);
