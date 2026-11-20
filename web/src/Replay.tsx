// ==========================================================
// web/src/Replay.tsx
// ==========================================================
// The hero: a replay of one recorded trial with all three controllers.
// Condition and trial live in the URL, so a replay can be shared.

import { useEffect, useMemo, useRef, useState } from "react";
import { scaleLinear, scaleLog } from "d3-scale";
import { line } from "d3-shape";
import {
  CONDS,
  CTRLS,
  CTRL_LABEL,
  CTRL_NOTE,
  fmtM,
  fmtS,
  load,
  replayPath,
  type Cond,
  type Ctrl,
  type Replay as ReplayData,
  type World,
} from "./data";
import { Belief, Fov, Robot, Target, YardSvg } from "./Yard";

// ------------------------------------------
// Colours and speeds
// ------------------------------------------
// Controller colours, and playback speeds as multiples of real time.
const COLOR: Record<Ctrl, string> = { info: "var(--info)", blind: "var(--blind)", fixed: "var(--fixed)" };
const SPEEDS = [1, 2.5, 5] as const; // multiples of real time

// ------------------------------------------
// ReplayProps
// ------------------------------------------
// Selected trial, a callback to change it, and a signal to start playing.
export type ReplayProps = {
  world: World;
  condition: Cond;
  trial: number;
  onPick: (c: Cond, trial: number) => void;
  playSignal: number;
};

// ------------------------------------------
// confirmedBy
// ------------------------------------------
// True once the five-scan window that starts at `dwellStart` has been played.
const confirmedBy = (dwellStart: number | null, step: number, dwell: number) =>
  dwellStart !== null && step >= dwellStart + dwell - 1;

// ------------------------------------------
// ErrorInset
// ------------------------------------------
// Distance from belief to target over time, log scale, up to the current step.
function ErrorInset({
  data,
  step,
  shown,
  world,
}: {
  data: ReplayData;
  step: number;
  shown: Record<Ctrl, boolean>;
  world: World;
}) {
  const W = 360,
    H = 150,
    m = { l: 44, r: 10, t: 10, b: 24 };
  const x = scaleLinear()
    .domain([0, world.steps * world.dt])
    .range([m.l, W - m.r]);
  const y = scaleLog()
    .domain([0.001, 3])
    .range([H - m.b, m.t])
    .clamp(true);
  const path = (err: number[]) =>
    line<number>()
      .x((_, i) => x(i * world.dt))
      .y(e => y(Math.max(e, 0.001)))(err.slice(0, step + 1)) ?? "";
  return (
    <svg
      className="chart"
      viewBox={`0 0 ${W} ${H}`}
      role="img"
      aria-label="Position error against time for each controller, log scale"
    >
      <g className="grid">
        {[0.001, 0.01, 0.1, 1].map(v => (
          <line key={v} x1={m.l} x2={W - m.r} y1={y(v)} y2={y(v)} />
        ))}
      </g>
      {[0.001, 0.01, 0.1, 1].map(v => (
        <text key={v} x={m.l - 6} y={y(v) + 3} textAnchor="end">
          {v >= 0.1 ? `${v} m` : `${v * 1000} mm`}
        </text>
      ))}
      {[0, 10, 20, 30].map(t => (
        <text key={t} x={x(t)} y={H - 6} textAnchor="middle">
          {t} s
        </text>
      ))}
      <line
        x1={m.l}
        x2={W - m.r}
        y1={y(world.toleranceM)}
        y2={y(world.toleranceM)}
        stroke="var(--ink)"
        strokeDasharray="4 3"
        strokeOpacity="0.6"
      />
      <text x={W - m.r} y={y(world.toleranceM) - 4} textAnchor="end" style={{ fill: "var(--ink)" }}>
        0.10 m goal
      </text>
      {CTRLS.filter(c => shown[c]).map(c => (
        <path
          key={c}
          d={path(data.controllers[c].error)}
          fill="none"
          stroke={COLOR[c]}
          strokeWidth="2"
          strokeLinejoin="round"
        />
      ))}
      <line
        x1={x(step * world.dt)}
        x2={x(step * world.dt)}
        y1={m.t}
        y2={H - m.b}
        stroke="var(--ink)"
        strokeOpacity="0.35"
      />
    </svg>
  );
}

// ------------------------------------------
// Replay
// ------------------------------------------
// Loads the selected trial and plays, scrubs or steps through it.
// A newer selection always wins over a slower, older request.
export default function Replay({ world, condition, trial, onPick, playSignal }: ReplayProps) {
  const [data, setData] = useState<ReplayData | null>(null);
  const [failed, setFailed] = useState<string | null>(null);
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState<(typeof SPEEDS)[number]>(2.5);
  const [shown, setShown] = useState<Record<Ctrl, boolean>>({ info: true, blind: true, fixed: true });
  const [showBelief, setShowBelief] = useState(true);
  const [showTruth, setShowTruth] = useState(true);
  const [focus, setFocus] = useState<Ctrl>("info");
  const [attempt, setAttempt] = useState(0);
  const last = world.steps;

  useEffect(() => {
    let live = true;
    setFailed(null);
    setPlaying(false);
    setStep(0);
    load<ReplayData>(replayPath(condition, trial))
      .then(d => {
        if (live) setData(d);
      })
      .catch(e => {
        if (live) setFailed(String(e.message ?? e));
      });
    return () => {
      live = false;
    };
  }, [condition, trial, attempt]);

  // An outside request to play (for example from the results section).
  // It waits until the requested trial has loaded.
  const firstSignal = useRef(playSignal);
  const [pendingPlay, setPendingPlay] = useState<string | null>(null);
  const key = `${condition}/${trial}`;
  useEffect(() => {
    if (playSignal !== firstSignal.current) setPendingPlay(key);
    // Only a new signal queues playback; the key is read at that moment.
  }, [playSignal]);

  const ready = data && data.condition === condition && data.trial === trial;
  useEffect(() => {
    if (pendingPlay === null) return;
    // The visitor picked another trial meanwhile: drop the request.
    if (pendingPlay !== key) {
      setPendingPlay(null);
      return;
    }
    if (ready) {
      setPendingPlay(null);
      setStep(0);
      setPlaying(true);
    }
  }, [pendingPlay, key, ready]);

  useEffect(() => {
    if (!playing || !ready) return;
    const id = window.setInterval(
      () =>
        setStep(s => {
          if (s >= last) {
            setPlaying(false);
            return s;
          }
          return s + 1;
        }),
      (world.dt * 1000) / speed,
    );
    return () => window.clearInterval(id);
  }, [playing, ready, speed, last, world.dt]);

  useEffect(() => {
    const onHide = () => {
      if (document.hidden) setPlaying(false);
    };
    document.addEventListener("visibilitychange", onHide);
    return () => document.removeEventListener("visibilitychange", onHide);
  }, []);

  const cond = world.conditions[condition];
  const paths = useMemo(() => {
    if (!data) return null;
    return Object.fromEntries(CTRLS.map(c => [c, data.controllers[c].states])) as Record<
      Ctrl,
      [number, number, number][]
    >;
  }, [data]);

  return (
    <div className="stack">
      <div className="controls">
        <div className="segmented" role="group" aria-label="Replay condition">
          {CONDS.map(c => (
            <button key={c} type="button" aria-pressed={c === condition} onClick={() => onPick(c, trial)}>
              {world.conditions[c].label}
            </button>
          ))}
        </div>
        <label className="field">
          Trial
          <select value={trial} onChange={e => onPick(condition, Number(e.target.value))}>
            {Array.from({ length: 40 }, (_, i) => (
              <option key={i} value={i}>
                {i}
              </option>
            ))}
          </select>
        </label>
        <button
          type="button"
          className="button ghost"
          onClick={() => onPick(condition, (trial + 1 + Math.floor(Math.random() * 39)) % 40)}
        >
          Another trial
        </button>
      </div>

      <div className="replay">
        <div className="plot yard">
          {failed ? (
            <div className="error-box">
              <p>This trial could not be loaded ({failed}).</p>
              <button type="button" className="button" onClick={() => setAttempt(a => a + 1)}>
                Try again
              </button>
            </div>
          ) : !ready || !paths ? (
            <div className="loading">Loading trial {trial}…</div>
          ) : (
            <YardSvg
              world={world}
              label={`Top-down replay of trial ${trial} in ${cond.label}, step ${step} of ${last}`}
              desc="Grey boxes are crates and walls, the hatched area is rough ground that limits speed, the dotted rectangle is where the target may be, and the dashed loop is the fixed route."
            >
              {f => (
                <>
                  {shown[focus] &&
                    (() => {
                      const s = paths[focus][step];
                      return <Fov f={f} x={s[0]} y={s[1]} theta={s[2]} color={COLOR[focus]} />;
                    })()}
                  {showTruth && <Target f={f} world={world} q={data!.qTrue} />}
                  {CTRLS.filter(c => shown[c]).map(c => (
                    <polyline
                      key={`p${c}`}
                      points={paths[c]
                        .slice(0, step + 1)
                        .map(s => `${f.sx(s[0])},${f.sy(s[1])}`)
                        .join(" ")}
                      fill="none"
                      stroke={COLOR[c]}
                      strokeWidth="2.4"
                      strokeLinejoin="round"
                      strokeLinecap="round"
                      strokeOpacity="0.85"
                    />
                  ))}
                  {showBelief &&
                    CTRLS.filter(c => shown[c]).map(c => {
                      const r = data!.controllers[c];
                      return (
                        <Belief
                          key={`b${c}`}
                          f={f}
                          mean={[r.mean[step][0], r.mean[step][1]]}
                          cov={r.covXY[step]}
                          color={COLOR[c]}
                        />
                      );
                    })}
                  {CTRLS.filter(c => shown[c]).map(c => {
                    const s = paths[c][step];
                    return (
                      <Robot key={`r${c}`} f={f} x={s[0]} y={s[1]} theta={s[2]} color={COLOR[c]} dim={c !== focus} />
                    );
                  })}
                </>
              )}
            </YardSvg>
          )}
          <div className="scrub">
            <button
              type="button"
              className="button icon"
              disabled={!ready}
              aria-label={playing ? "Pause" : step >= last ? "Replay from the start" : "Play"}
              onClick={() => {
                if (step >= last) setStep(0);
                setPlaying(p => !p);
              }}
            >
              {playing ? (
                <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">
                  <rect x="2" y="1" width="3.5" height="12" fill="currentColor" />
                  <rect x="8.5" y="1" width="3.5" height="12" fill="currentColor" />
                </svg>
              ) : (
                <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">
                  <path d="M3 1l10 6-10 6z" fill="currentColor" />
                </svg>
              )}
            </button>
            <div className="segmented" role="group" aria-label="Playback speed, as a multiple of real time">
              {SPEEDS.map(s => (
                <button key={s} type="button" aria-pressed={speed === s} onClick={() => setSpeed(s)}>
                  {s}×
                </button>
              ))}
            </div>
            <input
              type="range"
              min={0}
              max={last}
              step={1}
              value={step}
              disabled={!ready}
              aria-label="Time step"
              aria-valuetext={`${fmtS(step * world.dt)} of 30 s`}
              onChange={e => {
                setPlaying(false);
                setStep(Number(e.target.value));
              }}
            />
            <span className="time" aria-hidden="true">
              {(step * world.dt).toFixed(1)} s
            </span>
          </div>
          <div className="legend" style={{ marginTop: 10 }}>
            <span>
              <svg width="12" height="12">
                <rect width="12" height="12" rx="1.5" fill="var(--truth)" />
              </svg>
              target (truth, never seen by the robot)
            </span>
            <span>
              <svg width="16" height="12">
                <ellipse cx="8" cy="6" rx="7" ry="4.5" fill="none" stroke="var(--soft)" />
              </svg>
              belief: mean and 2σ ellipse
            </span>
            <span>
              <svg width="18" height="10">
                <rect width="18" height="10" fill="none" stroke="var(--prior)" strokeDasharray="2 3" />
              </svg>
              where the target may be
            </span>
            <span>
              <svg width="14" height="14">
                <rect width="14" height="14" fill="url(#rough-hatch)" opacity="0.7" />
              </svg>
              rough ground (slow)
            </span>
            <span>
              <svg width="18" height="10">
                <path d="M1 9L9 1L17 9" fill="none" stroke="var(--soft)" strokeDasharray="2 3" />
              </svg>
              field of view, 5.6 m
            </span>
          </div>
        </div>

        <aside className="readout" aria-label="Controllers">
          {CTRLS.map(c => {
            const r = ready ? data!.controllers[c] : null;
            const done = r ? confirmedBy(r.dwellStart, step, world.dwellSamples) : false;
            const beams = r && step > 0 ? r.nTargetDetected[step - 1] : null;
            return (
              <div key={c} className="ctrl-card" data-off={!shown[c]}>
                <input
                  type="checkbox"
                  className="check"
                  checked={shown[c]}
                  aria-label={`Show ${CTRL_LABEL[c]}`}
                  onChange={e => setShown(s => ({ ...s, [c]: e.target.checked }))}
                  style={{ accentColor: COLOR[c] }}
                />
                <button
                  type="button"
                  className="plain"
                  onClick={() => setFocus(c)}
                  aria-pressed={focus === c}
                  title="Show this robot's field of view"
                  style={{ display: "flex", alignItems: "center", gap: 8 }}
                >
                  <i className="dot" style={{ background: COLOR[c] }} />
                  <span className="name">{CTRL_LABEL[c]}</span>
                </button>
                <span className="err" title="Distance from the belief mean to the true target">
                  {r ? fmtM(r.error[step]) : "–"}
                </span>
                <span className="note">{CTRL_NOTE[c]}</span>
                <div className="status">
                  {!r ? (
                    <span>–</span>
                  ) : done ? (
                    <span>
                      Within 0.10 m from <b>{fmtS(r.timeScoreS)}</b>, confirmed at{" "}
                      {fmtS((r.dwellStart! + world.dwellSamples - 1) * world.dt)}
                    </span>
                  ) : step >= last ? (
                    <span>Not localized within 30 s</span>
                  ) : (
                    <span className="muted">Not yet within 0.10 m for 5 scans</span>
                  )}
                  <span className="muted" title="Beams whose strongest return was the target in this scan">
                    {beams === null ? "" : `${beams} beams on target`}
                  </span>
                </div>
              </div>
            );
          })}
          <div className="plot">
            <div className="chart-title">Distance from belief to target</div>
            {ready ? (
              <ErrorInset data={data!} step={step} shown={shown} world={world} />
            ) : (
              <div className="loading" style={{ minHeight: 150 }}>
                …
              </div>
            )}
          </div>
          <div className="row">
            <label className="check">
              <input type="checkbox" checked={showBelief} onChange={e => setShowBelief(e.target.checked)} />
              Belief ellipses
            </label>
            <label className="check">
              <input type="checkbox" checked={showTruth} onChange={e => setShowTruth(e.target.checked)} />
              True target
            </label>
          </div>
          <p className="small muted">
            {cond.label}: target reflectivity {cond.rho},{" "}
            {cond.visibilityM >= 10000 ? "clear air" : `visibility ${cond.visibilityM} m`},{" "}
            {cond.sun ? `${cond.sun} sun` : "no sunlight"}. Seed 5105, trial {trial}; the three robots see the same
            target and the same noise streams.
          </p>
        </aside>
      </div>
    </div>
  );
}
