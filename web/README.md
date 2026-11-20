# Move to See — interactive site

An interactive companion to the project: replay any of the 480 recorded trials, change the target and the weather to
see what the sensor model predicts, step through the planner's choices, and inspect every paired result and
registered check.

**Live:** https://robot.josephjuniormensah.com

## Sections

| Section | What it does |
| --- | --- |
| Replay | Top-down yard with the three controllers of one trial: paths, belief mean and 2σ ellipse, field of view, the true target, and the error over time. Pick any condition and trial; play, pause or scrub. The selection is kept in the URL. |
| Sensor | Detection probability and range noise against range for six reflectivities, five fog levels and five sunlight levels, straight from the analytic sensor model. |
| Decision | Three planning steps of fog trial 0: every candidate viewpoint coloured by its score, the belief particles and the 24 hypotheses the information bound compares. Click a viewpoint for its information, travel cost and score. |
| Results | All 40 paired trials of each condition on one chart, success rates with Wilson intervals, median time scores with bootstrap intervals, and the registered hypothesis tests. Any trial opens in the replay. |
| Evidence | Every registered experiment with its criteria and verdict, the documented deviations, and the test-suite run. |

## Run it

```bash
npm ci
npm run dev            # http://localhost:5173
npm run typecheck
npm run test:models    # browser helpers against the Python oracles and every replay file
npm run build          # static site in dist/
npm run smoke          # headless Chrome at 390 px and 1280 px, light and dark (needs Chrome installed)
```

## Rebuild the data

`public/data/` is generated from the experiment outputs in `sim/out/` (run the experiments first, see the main README):

```bash
npm run data -- --results-dir ../sim/out        # also runs the Python test suite and records its counts
npm run decisions -- --results-dir ../sim/out   # replays fog trial 0 and exports three planning steps
npm run test:models
```

`build_data.py` recomputes every F05 summary statistic, the five paired comparisons, the control interval and the
verdicts from the raw trial records with the registered code, and stops if any differs from `metrics.json` (floating
values to 1e-10, p-values to 1e-12). It also re-evaluates every sensor curve and checks the rounding of every exported
array. `export_decisions.py` reruns the first 21 steps of the trial with a read-only hook on the planner and publishes the
planning steps only if the rerun reproduces the recorded states, goals, scores and belief mean and covariance (within
1e-8, 1e-9 for scores; the published export matches to zero).

The browser draws and selects; it computes nothing physical. Its few helpers (ellipse axes, time score, curve index)
are checked by `test:models` against Python oracle fixtures; it also recomputes the time score of every exported
trial and checks all 29,280 recorded covariances are valid.

## Deploy

Vercel with the project root set to `web/`; `vercel.json` runs the type check and the model checks before building.
The data is committed, so the deployment does not need Python.

## Stack

React, TypeScript and Vite. Charts are hand-drawn SVG using `d3-scale` and `d3-shape`. Space Grotesk, IBM Plex Sans
and IBM Plex Mono are bundled locally.
