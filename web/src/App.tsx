// ==========================================================
// web/src/App.tsx
// ==========================================================
// The page: top bar, the replay hero and the five sections below it. Holds
// the selected condition and trial, kept in the URL.

import { useCallback, useEffect, useState } from "react";
import Replay from "./Replay";
import ThemeToggle from "./ThemeToggle";
import {
  CONDS,
  load,
  type Cond,
  type Results as ResultsData,
  type World,
} from "./data";

// ------------------------------------------
// Links
// ------------------------------------------
// The repository.
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
  const [playSignal] = useState(0);

  useEffect(() => {
    Promise.all([load<World>("world.json"), load<ResultsData>("results.json")])
      .then(([w, r]) => {
        setWorld(w);
        setResults(r);
      })
      .catch(() => setFailed(true));
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
      <div className="bar">
        <header className="wrap topbar">
          <a className="brand" href="#replay">
            <img src="./icon.svg" alt="" />
            Move to See
          </a>
          <div className="topbar-end">
            <nav aria-label="Sections">
              <a href="#replay">Replay</a>
            </nav>
            <ThemeToggle />
          </div>
        </header>
      </div>

      <main>
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





      </main>

    </>
  );
}
