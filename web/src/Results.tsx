// ==========================================================
// web/src/Results.tsx
// ==========================================================
// All paired trials of a condition, the success and time summaries, and the
// registered tests H1 to H3.

import { useEffect, useId, useState } from "react";
import {
  CONDS,
  CTRLS,
  CTRL_LABEL,
  CTRL_SHORT,
  fmtM,
  fmtP,
  fmtS,
  load,
  type Cond,
  type Pair,
  type Results as ResultsData,
  type V2,
  type World,
} from "./data";
import { useNarrow } from "./hooks";

// ------------------------------------------
// Formatters
// ------------------------------------------
// Intervals, percentages and verdict pills.
const interval = (v: V2) => `[${fmtS(v[0])}, ${fmtS(v[1])}]`;
const percent = (v: number) => `${(100 * v).toFixed(1)}%`;
const verdict = (supported: boolean) => (
  <span className={`verdict${supported ? "" : " no"}`}>{supported ? "supported" : "not supported"}</span>
);

// ------------------------------------------
// jitter
// ------------------------------------------
// Spreads coincident marks sideways; the time coordinate stays exact.
const jitter = (trial: number) => (((trial * 17) % 41) - 20) * 1.6;

// ------------------------------------------
// LAYOUT
// ------------------------------------------
// Chart geometry for wide screens and for phones, where the three columns
// sit closer together so all of them fit without scrolling.
const LAYOUT = {
  wide: { w: 900, cols: [240, 470, 700], x0: 90, x1: 860, tick: 78, spread: 1.4 },
  narrow: { w: 300, cols: [85, 170, 255], x0: 40, x1: 290, tick: 32, spread: 0.35 },
};

// ------------------------------------------
// Triplets
// ------------------------------------------
// One line per trial joining its three controllers' time scores.
function Triplets({
  pairs,
  selected,
  select,
}: {
  pairs: Pair[];
  selected: number | null;
  select: (trial: number) => void;
}) {
  const id = useId();
  const g = LAYOUT[useNarrow() ? "narrow" : "wide"];
  const x = (i: number, trial: number) => g.cols[i] + jitter(trial) * g.spread;
  const y = (seconds: number) => 48 + seconds * 8;
  return (
    <svg
      className="chart"
      viewBox={`0 0 ${g.w} 330`}
      role="group"
      aria-labelledby={`${id}-title ${id}-desc`}
      style={{ maxWidth: 1080, marginInline: "auto" }}
    >
      <title id={`${id}-title`}>Paired time scores by controller</title>
      <desc id={`${id}-desc`}>
        Each line joins one trial's three controllers. Time runs from zero at the top to 30 seconds at the bottom.
        Hollow squares mean not localized. Focus or activate a trial for its exact values; a table follows.
      </desc>
      <text x="8" y="16">
        Time score (s)
      </text>
      <g className="grid">
        {[0, 5, 10, 15, 20, 25, 30].map(t => (
          <g key={t}>
            <line x1={g.x0} x2={g.x1} y1={y(t)} y2={y(t)} />
            <text x={g.tick} y={y(t) + 4} textAnchor="end">
              {t}
            </text>
          </g>
        ))}
      </g>
      {CTRLS.map((c, i) => (
        <text key={c} className="label" x={g.cols[i]} y="36" textAnchor="middle" style={{ fill: `var(--${c})` }}>
          {CTRL_SHORT[c]}
        </text>
      ))}
      {pairs.map(p => {
        const active = p.trial === selected;
        const points = CTRLS.map(
          (_, i) => `${x(i, p.trial)},${y(p.dwellStart[i] === null ? 30 : p.timeScoreS[i])}`,
        ).join(" ");
        return (
          <g
            key={p.trial}
            role="button"
            tabIndex={0}
            aria-pressed={active}
            aria-label={`Trial ${p.trial}: ${CTRLS.map((c, i) => `${CTRL_SHORT[c]} ${fmtS(p.timeScoreS[i])}${p.dwellStart[i] === null ? ", not localized" : ""}`).join("; ")}`}
            onFocus={() => select(p.trial)}
            onClick={() => select(p.trial)}
            onKeyDown={e => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                select(p.trial);
              }
            }}
            style={{ cursor: "pointer" }}
          >
            <polyline points={points} fill="none" stroke="transparent" strokeWidth="10" />
            <polyline
              points={points}
              fill="none"
              stroke={active ? "var(--ink)" : "var(--faint)"}
              strokeWidth={active ? 2 : 0.7}
              opacity={active ? 1 : 0.3}
              pointerEvents="none"
            />
            {CTRLS.map((c, i) =>
              p.dwellStart[i] === null ? (
                <rect
                  key={c}
                  x={x(i, p.trial) - 3.5}
                  y={y(30) - 3.5}
                  width="7"
                  height="7"
                  fill="var(--surface)"
                  stroke={`var(--${c})`}
                  strokeWidth={active ? 2.5 : 1}
                >
                  <title>Not localized — 30 s time score</title>
                </rect>
              ) : (
                <circle
                  key={c}
                  cx={x(i, p.trial)}
                  cy={y(p.timeScoreS[i])}
                  r={active ? 4.5 : 2.5}
                  fill={`var(--${c})`}
                />
              ),
            )}
          </g>
        );
      })}
      {pairs
        .filter(p => p.trial === selected)
        .map(p => (
          <g key={`highlight-${p.trial}`} aria-hidden="true" pointerEvents="none">
            <polyline
              points={CTRLS.map(
                (_, i) => `${x(i, p.trial)},${y(p.dwellStart[i] === null ? 30 : p.timeScoreS[i])}`,
              ).join(" ")}
              fill="none"
              stroke="var(--ink)"
              strokeWidth="2"
            />
            {CTRLS.map((c, i) =>
              p.dwellStart[i] === null ? (
                <rect
                  key={c}
                  x={x(i, p.trial) - 4}
                  y={y(30) - 4}
                  width="8"
                  height="8"
                  fill="var(--surface)"
                  stroke={`var(--${c})`}
                  strokeWidth="2.5"
                />
              ) : (
                <circle
                  key={c}
                  cx={x(i, p.trial)}
                  cy={y(p.timeScoreS[i])}
                  r="4.5"
                  fill={`var(--${c})`}
                  stroke="var(--surface)"
                />
              ),
            )}
          </g>
        ))}
      <text x={g.w / 2 + 20} y="316" textAnchor="middle">
        □ not localized · 30 s score
      </text>
    </svg>
  );
}

// ------------------------------------------
// Results
// ------------------------------------------
// Condition tabs, the paired chart, summaries and the hypothesis tables.
export default function Results({ world, onOpen }: { world: World; onOpen: (c: Cond, trial: number) => void }) {
  const [data, setData] = useState<ResultsData | null>(null);
  const [error, setError] = useState(false);
  const [attempt, setAttempt] = useState(0);
  const [condition, setCondition] = useState<Cond>("C1");
  const [trial, setTrial] = useState<number | null>(null);
  useEffect(() => {
    let live = true;
    setError(false);
    load<ResultsData>("results.json")
      .then(value => {
        if (live) setData(value);
      })
      .catch(() => {
        if (live) setError(true);
      });
    return () => {
      live = false;
    };
  }, [attempt]);
  const pairs = data?.paired[condition] ?? [];
  const picked = pairs.find(p => p.trial === trial);
  return (
    <div>
      {error ? (
        <div className="callout" role="alert">
          Results could not be loaded.{" "}
          <button className="button ghost" onClick={() => setAttempt(v => v + 1)}>
            Retry results
          </button>
        </div>
      ) : !data ? (
        <p role="status">Loading paired results…</p>
      ) : (
        <div className="stack">
          <div className="segmented" role="group" aria-label="Results condition">
            {CONDS.map(c => (
              <button
                key={c}
                style={{ minHeight: 44 }}
                aria-pressed={c === condition}
                onClick={() => {
                  setCondition(c);
                  setTrial(null);
                }}
              >
                {world.conditions[c].label}
              </button>
            ))}
          </div>
          <div className="panel stack">
            <div>
              <h3>{world.conditions[condition].label}</h3>
              <p className="chart-sub">
                {pairs.length} paired trials. Lower is faster: the time from which the estimate stays within 0.10 m for
                five scans, 30 s if never. Select a line or use the trial table for exact values.
              </p>
            </div>
            <div className="plot wide-chart">
              <Triplets pairs={pairs} selected={trial} select={setTrial} />
            </div>
            <div className="legend">
              {CTRLS.map(c => (
                <span key={c}>
                  <i className="dot" style={{ background: `var(--${c})` }} />
                  {CTRL_LABEL[c]}
                </span>
              ))}
            </div>
            <div className="callout" aria-live="polite">
              {picked ? (
                <div className="stack">
                  <strong>Trial {picked.trial}</strong>
                  {CTRLS.map((c, i) => (
                    <p key={c}>
                      {CTRL_LABEL[c]}: <span className="num">{fmtS(picked.timeScoreS[i])}</span>
                      {picked.dwellStart[i] === null ? " · not localized" : " · localized"}; final error{" "}
                      <span className="num">{fmtM(picked.finalErrorM[i])}</span>.
                    </p>
                  ))}
                  <div>
                    <button
                      className="button"
                      style={{ minHeight: 44 }}
                      onClick={() => onOpen(condition, picked.trial)}
                    >
                      Open in the replay
                    </button>
                  </div>
                </div>
              ) : (
                "Select a trial to compare its three time scores and final errors."
              )}
            </div>
            <div className="stats">
              {CTRLS.map(c => {
                const s = data.summary[condition][c];
                return (
                  <div className="stat" key={c} style={{ borderTopColor: `var(--${c})` }}>
                    <h3>{CTRL_LABEL[c]}</h3>
                    <b>
                      {s.successCount}/{s.n} localized
                    </b>
                    <span>
                      95% Wilson interval: {percent(s.successWilson95[0])}–{percent(s.successWilson95[1])}
                    </span>
                    <strong className="num">{fmtS(s.medianTimeScoreS)} median</strong>
                    <span>95% CI: {interval(s.timeScoreCi95)}</span>
                  </div>
                );
              })}
            </div>
            <details>
              <summary style={{ cursor: "pointer", paddingBlock: 12 }}>Per-trial time scores and replay</summary>
              <div className="table-scroll" tabIndex={0} role="region" aria-label="Per-trial time scores">
                <table className="data">
                  <caption>Time scores; not localized runs receive 30 s.</caption>
                  <thead>
                    <tr>
                      <th scope="col">Trial</th>
                      {CTRLS.map(c => (
                        <th scope="col" key={c}>
                          {CTRL_LABEL[c]}
                        </th>
                      ))}
                      <th scope="col">Replay</th>
                    </tr>
                  </thead>
                  <tbody>
                    {pairs.map(p => (
                      <tr key={p.trial} style={{ background: p.trial === trial ? "var(--grid)" : undefined }}>
                        <th scope="row">
                          <button
                            className="button ghost"
                            style={{ minHeight: 44 }}
                            aria-pressed={p.trial === trial}
                            onClick={() => setTrial(p.trial)}
                          >
                            Trial {p.trial}
                          </button>
                        </th>
                        {CTRLS.map((c, i) => (
                          <td key={c}>
                            <span className="num">{fmtS(p.timeScoreS[i])}</span>
                            {p.dwellStart[i] === null && <span> · not localized</span>}
                          </td>
                        ))}
                        <td>
                          <button
                            className="button ghost"
                            style={{ minHeight: 44 }}
                            aria-label={`Open trial ${p.trial} in the replay`}
                            onClick={() => onOpen(condition, p.trial)}
                          >
                            Open replay
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </details>
          </div>
          <div className="panel stack">
            <h3>H1 / H2 · Paired comparisons</h3>
            <p className="small muted">
              Difference = information-driven − comparison controller. Negative values favor information-driven. Faster
              / tie / slower counts use paired time scores; Holm adjustment covers all five comparisons.
            </p>
            <p className="small muted scroll-hint" aria-hidden="true">
              Swipe sideways for the intervals and verdicts →
            </p>
            <div className="table-scroll" tabIndex={0} role="region" aria-label="Hypothesis comparisons">
              <table className="data comparison-table">
                <thead>
                  <tr>
                    {[
                      "Hypothesis",
                      "Condition",
                      "Versus",
                      "Median difference (95% CI)",
                      "Faster / tie / slower",
                      "Holm p",
                      "Hypothesis verdict",
                    ].map(s => (
                      <th scope="col" key={s}>
                        {s}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {data.comparisons.map(c => (
                    <tr key={`${c.hypothesis}-${c.condition}-${c.against}`}>
                      <th scope="row">{c.hypothesis}</th>
                      <td>{world.conditions[c.condition].label}</td>
                      <td>{CTRL_LABEL[c.against]}</td>
                      <td className="num">
                        {fmtS(c.medianPairedDifferenceS)} {interval(c.differenceCi95)}
                      </td>
                      <td className="num">
                        {c.infoFaster} / {c.ties} / {c.infoSlower}
                      </td>
                      <td className="num">{fmtP(c.pHolm)}</td>
                      <td>{verdict(data.verdicts[c.hypothesis])}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="callout stack">
              <div className="row">
                <strong>H3 · control: white target in clear air</strong>
                {verdict(data.verdicts.H3)}
              </div>
              <p>
                Information-driven − physics-blind: median difference{" "}
                <span className="num">{fmtS(data.control.medianPairedDifferenceS)}</span>; 95% CI{" "}
                <span className="num">{interval(data.control.differenceCi95)}</span>; equivalence margin{" "}
                <span className="num">±{fmtS(data.control.marginS)}</span>.
              </p>
              <p>Equivalence concerns the median paired difference, not identical behavior in every trial.</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
