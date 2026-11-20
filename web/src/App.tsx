// ==========================================================
// web/src/App.tsx
// ==========================================================
// The page: top bar, the replay hero and the five sections below it. Holds
// the selected condition and trial, kept in the URL.

import { useCallback, useEffect, useState } from "react";
import Replay from "./Replay";
import Sensor from "./Sensor";
import Decision from "./Decision";
import Results from "./Results";
import Evidence from "./Evidence";
import ThemeToggle from "./ThemeToggle";
import { useScrolled } from "./hooks";
import {
  CONDS,
  load,
  replayPath,
  type Cond,
  type Replay as ReplayData,
  type Results as ResultsData,
  type World,
} from "./data";

// ------------------------------------------
// Links
// ------------------------------------------
// Portfolio and repository.
const SITE = "https://josephjuniormensah.com";
const REPO = "https://github.com/jjmensah/move-to-see";

// ------------------------------------------
// readUrl
// ------------------------------------------
// Condition and trial from the URL, falling back to fog trial 3.
function readUrl(): { condition: Cond; trial: number } {
  const q = new URLSearchParams(window.location.search);
  const c = q.get("condition") as Cond;
  const t = q.has("trial") ? Number(q.get("trial")) : NaN;
  return { condition: CONDS.includes(c) ? c : "C2", trial: Number.isInteger(t) && t >= 0 && t < 40 ? t : 3 };
}

// ------------------------------------------
// App
// ------------------------------------------
// Loads the world and results, then lays out every section.
export default function App() {
  const [world, setWorld] = useState<World | null>(null);
  const [results, setResults] = useState<ResultsData | null>(null);
  const [failed, setFailed] = useState(false);
  const [sel, setSel] = useState(readUrl);
  const [playSignal, setPlaySignal] = useState(0);
  const [decisionTruth, setDecisionTruth] = useState<number[] | null>(null);
  const scrolled = useScrolled();

  useEffect(() => {
    Promise.all([load<World>("world.json"), load<ResultsData>("results.json")])
      .then(([w, r]) => {
        setWorld(w);
        setResults(r);
      })
      .catch(() => setFailed(true));
    // The decision section shows fog trial 0; its true target comes from that trial's replay.
    load<ReplayData>(replayPath("C2", 0))
      .then(d => setDecisionTruth(d.qTrue))
      .catch(() => {});
  }, []);

  useEffect(() => {
    const onPop = () => setSel(readUrl());
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  const pick = useCallback((condition: Cond, trial: number) => {
    setSel({ condition, trial });
    const url = new URL(window.location.href);
    url.searchParams.set("condition", condition);
    url.searchParams.set("trial", String(trial));
    window.history.replaceState(null, "", url);
  }, []);

  const openFromResults = useCallback(
    (condition: Cond, trial: number) => {
      pick(condition, trial);
      const url = new URL(window.location.href);
      url.hash = "replay";
      window.history.pushState(null, "", url);
      const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      document.getElementById("replay")?.scrollIntoView({ behavior: still ? "auto" : "smooth" });
      document.getElementById("replay-title")?.focus({ preventScroll: true });
      setPlaySignal(s => s + 1);
    },
    [pick],
  );

  if (failed)
    return <div className="loading boot">The trial records could not be loaded. Refresh the page to try again.</div>;
  if (!world || !results) return <div className="loading boot">Loading 480 recorded trials…</div>;

  const s = results.summary;
  const fog = s.C2,
    sun = s.C3;

  return (
    <>
      <a className="skip" href="#replay">
        Skip to the replay
      </a>
      <div className={`bar${scrolled ? " solid" : ""}`}>
        <header className="topbar">
          <a className="brand" href="#top">
            <img src="./icon.svg" alt="" />
            Move to See
          </a>
          <div className="topbar-end">
            <nav aria-label="Sections">
              <a href="#replay">Replay</a>
              <a href="#sensor">Sensor</a>
              <a href="#decision">Decision</a>
              <a href="#results">Results</a>
              <a href="#evidence">Evidence</a>
              <a href="#limits">Limits</a>
            </nav>
            <ThemeToggle />
            <a className="button" href={REPO}>
              Source
            </a>
          </div>
        </header>
      </div>

      <main id="top">
        <section id="replay" className="wrap hero" aria-labelledby="replay-title">
          <div className="hero-top">
            <div className="stack" style={{ gap: 18 }}>
              <span className="kicker">Active perception · 3D LiDAR · simulation</span>
              <h1 id="replay-title" tabIndex={-1}>
                Where should a robot drive to <em>see</em> what it is looking for?
              </h1>
            </div>
            <p className="lede">
              A ground robot with a simulated 3D LiDAR has to find a dark box hidden behind crates, in fog and in
              sunlight. It decides where to drive by predicting what each viewpoint would let it measure. Replay any of
              the 480 recorded trials.
            </p>
            <p className="small muted" style={{ gridColumn: "1 / -1" }}>
              Each half second the robot scans, updates its belief about where the target is, and picks the next
              viewpoint to drive to. Time to localize: the moment from which the robot's estimate stays within 0.10 m of
              the target for five consecutive scans (30 s if it never does).{" "}
              <a href={`${REPO}#how-it-works`}>How it works</a> ·{" "}
              <a href={`${REPO}/blob/main/docs/design/final-spec.md`}>Design specification</a> ·{" "}
              <a href={REPO}>Source</a>
            </p>
          </div>
          <div className="stats" aria-label="Headline results in fog">
            <div className="stat" style={{ ["--c" as string]: "var(--info)" }}>
              <b className="num">{fog.info.medianTimeScoreS.toFixed(1)} s</b>
              <span>
                median time to localize the target in fog with the physics-aware planner ({fog.info.successCount}/40
                localized)
              </span>
            </div>
            <div className="stat" style={{ ["--c" as string]: "var(--fixed)" }}>
              <b className="num">{fog.fixed.medianTimeScoreS.toFixed(1)} s</b>
              <span>
                median for a fixed inspection route in the same trials ({fog.fixed.successCount}/40 localized)
              </span>
            </div>
            <div className="stat" style={{ ["--c" as string]: "var(--blind)" }}>
              <b className="num">{fog.blind.successCount + sun.blind.successCount}/80</b>
              <span>fog and sunlight trials localized by the same planner when it ignores the sensor physics</span>
            </div>
          </div>
          <Replay world={world} condition={sel.condition} trial={sel.trial} onPick={pick} playSignal={playSignal} />
        </section>

        <section id="sensor" className="section" aria-labelledby="sensor-title">
          <div className="wrap">
            <div className="section-head">
              <span className="kicker">The measurement</span>
              <h2 id="sensor-title">A target you can see is not always one you can measure</h2>
              <p>
                The LiDAR measures range from the phase of an amplitude-modulated laser. A dark target returns little
                light; fog absorbs it on the way out and back, and sunlight adds noise that raises the detection
                threshold. These curves come straight from the sensor model the robot plans with.
              </p>
            </div>
            <Sensor world={world} replayCondition={sel.condition} />
          </div>
        </section>

        <section id="decision" className="section" aria-labelledby="decision-title">
          <div className="wrap">
            <div className="section-head">
              <span className="kicker">The next view</span>
              <h2 id="decision-title">Why drive there?</h2>
              <p>
                Every half second the planner weighs a lattice of reachable viewpoints: how much it expects to learn
                about the target there, minus a cost for the time it takes to drive. Step through three moments of one
                fog trial and click any viewpoint to see its score.
              </p>
            </div>
            <Decision world={world} qTrue={decisionTruth} />
          </div>
        </section>

        <section id="results" className="section" aria-labelledby="results-title">
          <div className="wrap">
            <div className="section-head">
              <span className="kicker">The paired trials</span>
              <h2 id="results-title">Same target, same noise, three ways to search</h2>
              <p>
                In each of 40 trials per condition the three controllers face the same target position and the same
                random noise, so every trial is a fair head-to-head. Pick a trial to watch it.
              </p>
            </div>
            <Results world={world} onOpen={openFromResults} />
          </div>
        </section>

        <section id="evidence" className="section" aria-labelledby="evidence-title">
          <div className="wrap">
            <div className="section-head">
              <span className="kicker">The checks</span>
              <h2 id="evidence-title">Decided before the runs, reported as they came out</h2>
              <p>
                Each experiment's hypothesis, sample size and seed, and its pass criteria apart from the documented
                changes below, were written into a registry before its first full run; tuning used separate pilot seeds.
                Here is every registered check and its outcome.
              </p>
            </div>
            <Evidence />
          </div>
        </section>

        <section id="limits" className="section" aria-labelledby="limits-title">
          <div className="wrap grid-2" style={{ alignItems: "start" }}>
            <div className="section-head" style={{ marginBottom: 0 }}>
              <span className="kicker">Read the results</span>
              <h2 id="limits-title">What this shows, and what it does not</h2>
              <p>
                These are simulation results for one sensor model in one known yard. The model is carefully checked
                against a waveform-level simulation of the same sensor, but no hardware was involved.
              </p>
              <div className="row" style={{ marginTop: 8 }}>
                <a className="button" href={REPO}>
                  Source code and data
                </a>
                <a className="button ghost" href={`${REPO}/blob/main/docs/verification/results.md`}>
                  Full results
                </a>
              </div>
            </div>
            <div className="panel">
              <ul className="caveats">
                <li>
                  <b>The robot knows the map.</b> Its own position, the obstacles, the terrain, the weather and the
                  target's size and reflectivity are known; only the target's position is unknown.
                </li>
                <li>
                  <b>One stationary box.</b> The target does not move and is a box, like everything else in the yard.
                </li>
                <li>
                  <b>The physics-blind planner is deliberately mis-specified.</b> It shows the cost of ignoring the
                  physics, not the best a simpler planner could do.
                </li>
                <li>
                  <b>One fixed route.</b> A different inspection loop could do better or worse; in sunlight the fixed
                  route was faster in{" "}
                  {results.comparisons.find(c => c.condition === "C3" && c.against === "fixed")?.infoSlower ?? "some"}{" "}
                  of 40 pairs.
                </li>
                <li>
                  <b>Fog dims, it does not glow.</b> The fog model attenuates the beam but leaves out backscatter.
                </li>
                <li>
                  <b>Time is simulated time.</b> The time scores count robot seconds, not computation. On a laptop the
                  information-driven controller took on average 0.1–0.4 s to plan and 0.7–3.2 s to update its belief per
                  half-second step, depending on the condition, so this implementation does not yet run in real time.
                </li>
                <li>
                  <b>The ellipses are summaries.</b> They show two standard deviations of the belief's horizontal
                  spread, not a calibrated credible region.
                </li>
              </ul>
            </div>
          </div>
        </section>
      </main>

      <footer className="wrap footer">
        <p>
          Designed, built and verified by <a href={SITE}>Joseph Junior Mensah</a> as the final project for ECE 4240
          (Robot Perception) at Cornell, building on the midterm AMCW LiDAR simulator. <a href={REPO}>The repository</a>{" "}
          has the code, the registered experiments and every result shown here.
        </p>
      </footer>
    </>
  );
}
