// ==========================================================
// web/src/App.tsx
// ==========================================================
// The page: top bar, the replay hero and the five sections below it. Holds
// the selected condition and trial, kept in the URL.

import { useCallback, useEffect, useState } from "react";
import Replay from "./Replay";
import HowItWorks from "./HowItWorks";
import Sensor from "./Sensor";
import Decision from "./Decision";
import Results from "./Results";
import Evidence from "./Evidence";
import ThemeToggle from "./ThemeToggle";
import { useReveal, useScrolled } from "./hooks";
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
  useReveal(Boolean(world && results));

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

  const watch = () => {
    const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    document.getElementById("stage")?.scrollIntoView({ behavior: still ? "auto" : "smooth", block: "start" });
    setPlaySignal(s => s + 1);
  };

  const slower = results?.comparisons.find(c => c.condition === "C3" && c.against === "fixed")?.infoSlower;
  const limits: [string, string][] = [
    [
      "Does the robot know the map?",
      "Yes. Its own position, the obstacles, the terrain, the weather and the target's size and reflectivity are known; only the target's position is unknown.",
    ],
    [
      "What kind of target is it?",
      "One stationary box. It does not move, and like everything else in the yard it is a box.",
    ],
    [
      "Is the physics-blind planner a fair baseline?",
      "It is deliberately mis-specified: the same planner with a sensor model that assumes every visible point is measured. It shows the cost of ignoring the physics, not the best a simpler planner could do.",
    ],
    [
      "Would a different fixed route do better?",
      `Possibly; this is one inspection loop of many. Even this one was faster than the information-driven planner in ${slower ?? "some"} of the 40 sunlight pairs.`,
    ],
    [
      "Does the fog glow?",
      "No. The fog model attenuates the beam on the way out and back, but leaves out backscatter.",
    ],
    [
      "Does it run in real time?",
      "Not yet. The time scores count simulated seconds. On a laptop the information-driven controller took on average 0.1–0.4 s to plan and 0.7–3.2 s to update its belief per half-second step, depending on the condition.",
    ],
    [
      "What do the ellipses mean?",
      "They show two standard deviations of the belief's horizontal spread: a summary, not a calibrated credible region.",
    ],
  ];

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
              <a href="#how">How it works</a>
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
        <section id="replay" className="hero" aria-labelledby="replay-title">
          <div className="wrap">
            <h1 id="replay-title" className="reveal" tabIndex={-1}>
              Where should
              <br />a robot drive
              <br />
              <span className="hand">to see?</span>
            </h1>
            <p className="lede reveal">
              A ground robot with a simulated 3D LiDAR has to find a dark box hidden behind crates, in fog and in
              sunlight. It drives to wherever its sensor will actually see it, by predicting what each viewpoint would
              let it measure.
            </p>
            <div className="cta-row reveal">
              <button type="button" className="button lg" onClick={watch}>
                Watch a trial
              </button>
              <a className="button lg ghost" href={REPO}>
                Read the code
              </a>
            </div>
            <p className="fine reveal">480 recorded trials · 666 tests · every experiment registered before it ran</p>
            <div className="stats reveal" aria-label="Headline results in fog">
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
            <div id="stage" className="stage-wrap reveal">
              <div className="scribble" aria-hidden="true">
                <span className="hand">pick a trial, press play</span>
                <svg viewBox="0 0 120 90">
                  <path d="M8 6 C 40 4, 78 18, 92 52 M78 44 L 93 55 L 99 37" />
                </svg>
              </div>
              <div className="stage">
                <Replay
                  world={world}
                  condition={sel.condition}
                  trial={sel.trial}
                  onPick={pick}
                  playSignal={playSignal}
                />
              </div>
              <p className="fine">
                Time to localize: the moment from which the estimate stays within 0.10 m of the target for five scans in
                a row (30 s if it never does).
              </p>
            </div>
          </div>
        </section>

        <section id="how" className="band vivid" aria-labelledby="how-title">
          <div className="wrap">
            <div className="section-head reveal">
              <span className="kicker">how it works</span>
              <h2 id="how-title">Scan. Update. Move.</h2>
            </div>
            <HowItWorks />
          </div>
        </section>

        <section id="sensor" className="band paper" aria-labelledby="sensor-title">
          <div className="wrap">
            <div className="section-head reveal">
              <span className="kicker">the measurement</span>
              <h2 id="sensor-title">A target you can see is not always one you can measure</h2>
              <p>
                The LiDAR measures range from the phase of an amplitude-modulated laser. A dark target returns little
                light; fog attenuates it on the way out and back, and sunlight adds noise that raises the detection
                threshold. These curves come straight from the sensor model the robot plans with.
              </p>
            </div>
            <Sensor world={world} replayCondition={sel.condition} />
          </div>
        </section>

        <section id="decision" className="band vivid" aria-labelledby="decision-title">
          <div className="wrap">
            <div className="section-head reveal">
              <span className="kicker">the next view</span>
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

        <section id="results" className="band paper" aria-labelledby="results-title">
          <div className="wrap">
            <div className="section-head reveal">
              <span className="kicker">the paired trials</span>
              <h2 id="results-title">Same target, same noise, three ways to search</h2>
              <p>
                In each of 40 trials per condition the three controllers face the same target position and the same
                random noise, so every trial is a fair head-to-head. Pick a trial to watch it.
              </p>
            </div>
            <Results world={world} onOpen={openFromResults} />
          </div>
        </section>

        <section id="evidence" className="band vivid" aria-labelledby="evidence-title">
          <div className="wrap">
            <div className="section-head reveal">
              <span className="kicker">the checks</span>
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

        <section id="limits" className="band paper" aria-labelledby="limits-title">
          <div className="wrap" style={{ maxWidth: 820 }}>
            <div className="section-head reveal">
              <span className="kicker">read the results</span>
              <h2 id="limits-title">What this shows, and what it does not</h2>
              <p>
                These are simulation results for one sensor model in one known yard. The model is checked against a
                waveform-level simulation of the same sensor, but no hardware was involved.
              </p>
            </div>
            <div className="faq reveal">
              {limits.map(([q, a]) => (
                <details key={q}>
                  <summary>{q}</summary>
                  <p>{a}</p>
                </details>
              ))}
            </div>
          </div>
        </section>
        <section className="final" aria-labelledby="final-title">
          <div className="wrap">
            <img className="reveal" src="./icon.svg" alt="" />
            <h2 id="final-title" className="reveal">
              Move to <span className="hand">see.</span>
            </h2>
            <p className="lede reveal">
              The code, the registered experiments and every result on this page are in the repository.
            </p>
            <div className="cta-row reveal">
              <a className="button lg" href={REPO}>
                Read the code
              </a>
              <a className="button lg ghost" href={`${REPO}/blob/main/docs/verification/results.md`}>
                Full results
              </a>
            </div>
          </div>
        </section>
      </main>

      <footer className="footer">
        <a className="brand" href="#top">
          <img src="./icon.svg" alt="" />
          Move to See
        </a>
        <p>
          Designed, built and verified by <a href={SITE}>Joseph Junior Mensah</a> for ECE 4240 (Robot Perception) at
          Cornell, building on the midterm AMCW LiDAR simulator.
        </p>
        <a href={REPO}>GitHub</a>
      </footer>
    </>
  );
}
