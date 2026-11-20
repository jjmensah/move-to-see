// ==========================================================
// web/src/Decision.tsx
// ==========================================================
// One planning step of one recorded trial, exported by
// scripts/export_decisions.py: every candidate viewpoint with the information
// bound and travel time the planner weighs. Nothing is recomputed here.

import { useEffect, useMemo, useState } from "react";
import { scaleLinear, scaleSequential } from "d3-scale";
import { load, type Candidate, type Decisions, type World } from "./data";
import { Belief, Robot, Target, YardSvg } from "./Yard";

// One planning step of one recorded trial, exported offline by
// scripts/export_decisions.py: every reachable viewpoint with the information
// bound and travel time the planner weighs. Nothing is recomputed here.

// ------------------------------------------
// interpolateScore
// ------------------------------------------
// Score colour, mixed in CSS so it follows the theme.
function interpolateScore(t: number) {
  return `color-mix(in srgb, var(--accent) ${Math.round(8 + 92 * t)}%, var(--surface))`;
}

// ------------------------------------------
// Decision
// ------------------------------------------
// Candidate map, the chosen viewpoint's score and a ranked table.
export default function Decision({ world, qTrue }: { world: World; qTrue: number[] | null }) {
  const [data, setData] = useState<Decisions | null>(null);
  const [failed, setFailed] = useState(false);
  const [stepIdx, setStepIdx] = useState(0);
  const [picked, setPicked] = useState<number | null>(null);

  useEffect(() => {
    load<Decisions>("decisions.json")
      .then(setData)
      .catch(() => setFailed(true));
  }, []);

  const step = data?.steps[stepIdx];
  const ranked = useMemo(
    () =>
      step ? [...step.candidates].sort((a, b) => b.scoreNat - a.scoreNat || a.travelS - b.travelS || a.id - b.id) : [],
    [step],
  );

  if (failed)
    return <div className="error-box">The planning steps could not be loaded. Refresh the page to try again.</div>;
  if (!data || !step) return <div className="loading">Loading the planner's choices…</div>;

  const byId = new Map(step.candidates.map(c => [c.id, c]));
  const selected = byId.get(step.selectedId)!;
  const best = byId.get(step.bestId)!;
  const shown: Candidate = (picked !== null && byId.get(picked)) || selected;
  const lo = Math.min(...step.candidates.map(c => c.scoreNat));
  const hi = Math.max(...step.candidates.map(c => c.scoreNat));
  const colour = scaleSequential(interpolateScore).domain([lo, hi]);
  const maxInfo = Math.max(...step.candidates.map(c => c.informationNat), Math.log(data.supportSize));
  const bar = scaleLinear().domain([0, maxInfo]).range([0, 100]);
  const t = step.k * world.dt;
  const kq = qTrue ?? null;

  return (
    <div className="grid-side">
      <div className="plot yard">
        <YardSvg
          world={world}
          label={`Viewpoints the planner considered ${t} s into fog trial ${data.trial}, coloured by score`}
          desc="Each dot is a reachable viewpoint coloured by its score; small grey dots are the belief particles; rings mark the chosen goal."
        >
          {f => {
            const sp = Math.min(
              ...step.candidates
                .slice(1)
                .map((c, i) => Math.hypot(c.xy[0] - step.candidates[i].xy[0], c.xy[1] - step.candidates[i].xy[1]))
                .filter(d => d > 1e-6),
            );
            const r = Math.max(3, Math.min(9, (sp * 56) / 2.4));
            return (
              <>
                {step.candidates.map(c => (
                  <circle
                    key={c.id}
                    cx={f.sx(c.xy[0])}
                    cy={f.sy(c.xy[1])}
                    r={r}
                    fill={colour(c.scoreNat)}
                    stroke={c.id === shown.id ? "var(--ink)" : "none"}
                    strokeWidth="2"
                    style={{ cursor: "pointer" }}
                    onClick={() => setPicked(c.id)}
                  >
                    <title>{`${c.informationNat.toFixed(3)} nat − ${data.lambda} × ${c.travelS.toFixed(1)} s = ${c.scoreNat.toFixed(3)}`}</title>
                  </circle>
                ))}
                {step.particlesXY?.map((p, i) => (
                  <circle key={`p${i}`} cx={f.sx(p[0])} cy={f.sy(p[1])} r="1.4" fill="var(--ink)" opacity="0.35" />
                ))}
                {step.support?.map((p, i) => (
                  <circle
                    key={`s${i}`}
                    cx={f.sx(p[0])}
                    cy={f.sy(p[1])}
                    r="3"
                    fill="none"
                    stroke="var(--blind)"
                    strokeWidth="1.4"
                  />
                ))}
                {kq && (
                  <g opacity="0.85">
                    <Target f={f} world={world} q={kq} />
                  </g>
                )}
                <Belief f={f} mean={[step.mean[0], step.mean[1]]} cov={[0, 0, 0]} color="var(--ink)" />
                <line
                  x1={f.sx(step.state[0])}
                  y1={f.sy(step.state[1])}
                  x2={f.sx(selected.xy[0])}
                  y2={f.sy(selected.xy[1])}
                  stroke="var(--info)"
                  strokeWidth="2"
                  strokeDasharray="5 4"
                />
                <circle
                  cx={f.sx(selected.xy[0])}
                  cy={f.sy(selected.xy[1])}
                  r={r + 6}
                  fill="none"
                  stroke="var(--info)"
                  strokeWidth="2.5"
                />
                {best.id !== selected.id && (
                  <circle
                    cx={f.sx(best.xy[0])}
                    cy={f.sy(best.xy[1])}
                    r={r + 6}
                    fill="none"
                    stroke="var(--ink)"
                    strokeDasharray="3 3"
                    strokeWidth="1.5"
                  />
                )}
                <Robot f={f} x={step.state[0]} y={step.state[1]} theta={step.state[2]} color="var(--info)" />
              </>
            );
          }}
        </YardSvg>
        <div className="legend" style={{ marginTop: 10 }}>
          <span>
            <svg width="40" height="10">
              <defs>
                <linearGradient id="sg">
                  <stop offset="0" stopColor="var(--surface)" />
                  <stop offset="1" stopColor="var(--accent)" />
                </linearGradient>
              </defs>
              <rect width="40" height="10" rx="2" fill="url(#sg)" stroke="var(--rule)" />
            </svg>
            score, low to high
          </span>
          <span>
            <svg width="10" height="10">
              <circle cx="5" cy="5" r="1.6" fill="var(--ink)" opacity="0.5" />
            </svg>
            belief particles (a sample)
          </span>
          <span>
            <svg width="10" height="10">
              <circle cx="5" cy="5" r="3.5" fill="none" stroke="var(--blind)" strokeWidth="1.4" />
            </svg>
            {data.supportSize} hypotheses the bound compares
          </span>
          <span>
            <svg width="16" height="16">
              <circle cx="8" cy="8" r="6.5" fill="none" stroke="var(--info)" strokeWidth="2.5" />
            </svg>
            chosen goal
          </span>
        </div>
      </div>

      <div className="stack">
        <div className="segmented" role="group" aria-label="Planning step">
          {data.steps.map((s, i) => (
            <button
              key={s.k}
              type="button"
              aria-pressed={i === stepIdx}
              onClick={() => {
                setStepIdx(i);
                setPicked(null);
              }}
            >
              t = {s.k * world.dt} s
            </button>
          ))}
        </div>
        <div className="panel stack">
          <div>
            <div className="chart-title">{picked === null ? "The chosen viewpoint" : "Selected viewpoint"}</div>
            <div className="chart-sub">
              ({shown.xy[0].toFixed(2)}, {shown.xy[1].toFixed(2)}) m, {shown.travelS.toFixed(1)} s away.
              {shown.id === selected.id &&
                step.goalKept &&
                " Kept from the previous step: no other viewpoint beat it by the switching margin."}
              {shown.id === selected.id && !step.goalKept && " The highest score this step."}
            </div>
          </div>
          <Bars
            label="Information (lower bound)"
            value={shown.informationNat}
            width={bar(shown.informationNat)}
            color="var(--accent)"
            unit="nat"
          />
          <Bars
            label={`Travel cost (${data.lambda} nat/s × ${shown.travelS.toFixed(1)} s)`}
            value={-data.lambda * shown.travelS}
            width={bar(data.lambda * shown.travelS)}
            color="var(--fixed)"
            unit="nat"
          />
          <Bars
            label="Score J"
            value={shown.scoreNat}
            width={bar(Math.max(shown.scoreNat, 0))}
            color="var(--ink)"
            unit="nat"
            strong
          />
          <p className="small muted">
            The bound cannot exceed ln {data.supportSize} = {Math.log(data.supportSize).toFixed(2)} nat: perfectly
            telling apart all {data.supportSize} hypotheses.
          </p>
          {picked !== null && (
            <button type="button" className="button ghost" onClick={() => setPicked(null)}>
              Back to the chosen viewpoint
            </button>
          )}
        </div>
        <div className="table-scroll">
          <table className="data">
            <caption className="sr-only">Top viewpoints by score</caption>
            <thead>
              <tr>
                <th>Top viewpoints</th>
                <th className="num">info</th>
                <th className="num">travel</th>
                <th className="num">score</th>
              </tr>
            </thead>
            <tbody>
              {ranked.slice(0, 6).map((c, i) => (
                <tr
                  key={c.id}
                  onClick={() => setPicked(c.id)}
                  style={{ cursor: "pointer", fontWeight: c.id === shown.id ? 650 : undefined }}
                >
                  <td>
                    <button type="button" className="plain" onClick={() => setPicked(c.id)}>
                      #{i + 1} ({c.xy[0].toFixed(1)}, {c.xy[1].toFixed(1)}){c.id === selected.id ? " · chosen" : ""}
                    </button>
                  </td>
                  <td className="num">{c.informationNat.toFixed(2)}</td>
                  <td className="num">{c.travelS.toFixed(1)} s</td>
                  <td className="num">{c.scoreNat.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <details>
          <summary className="small" style={{ cursor: "pointer" }}>
            All {step.candidates.length} viewpoints
          </summary>
          <div className="table-scroll" style={{ maxHeight: 320, overflowY: "auto", marginTop: 8 }}>
            <table className="data">
              <caption className="sr-only">Every reachable viewpoint at this step, by score</caption>
              <thead>
                <tr>
                  <th>Viewpoint</th>
                  <th className="num">info</th>
                  <th className="num">travel</th>
                  <th className="num">score</th>
                </tr>
              </thead>
              <tbody>
                {ranked.map((c, i) => (
                  <tr key={c.id} style={{ fontWeight: c.id === shown.id ? 650 : undefined }}>
                    <td>
                      <button
                        type="button"
                        className="plain"
                        aria-pressed={c.id === shown.id}
                        onClick={() => setPicked(c.id)}
                      >
                        #{i + 1} ({c.xy[0].toFixed(2)}, {c.xy[1].toFixed(2)}){c.id === selected.id ? " · chosen" : ""}
                      </button>
                    </td>
                    <td className="num">{c.informationNat.toFixed(2)}</td>
                    <td className="num">{c.travelS.toFixed(1)} s</td>
                    <td className="num">{c.scoreNat.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
        <p className="small muted">
          Fog trial {data.trial}, information-driven controller, {step.candidates.length} reachable viewpoints on the
          planner's lattice at t = {t} s. While planning, the robot skips viewpoints too far away to win; the scores
          shown for those were computed afterwards with the same {data.supportSize} hypotheses, and the rerun was
          checked to reproduce the recorded trial exactly.
        </p>
      </div>
    </div>
  );
}

// ------------------------------------------
// Bars
// ------------------------------------------
// A labelled bar for one term of the score.
function Bars({
  label,
  value,
  width,
  color,
  unit,
  strong = false,
}: {
  label: string;
  value: number;
  width: number;
  color: string;
  unit: string;
  strong?: boolean;
}) {
  return (
    <div style={{ display: "grid", gap: 4 }}>
      <div className="row" style={{ justifyContent: "space-between", fontSize: "var(--step--1)" }}>
        <span style={{ fontWeight: strong ? 650 : 400 }}>{label}</span>
        <b className="num" style={{ fontWeight: strong ? 600 : 500 }}>
          {value >= 0 ? "" : "−"}
          {Math.abs(value).toFixed(3)} {unit}
        </b>
      </div>
      <div style={{ height: 8, borderRadius: 4, background: "var(--grid)" }}>
        <div
          style={{ width: `${Math.min(100, Math.max(0, width))}%`, height: "100%", borderRadius: 4, background: color }}
        />
      </div>
    </div>
  );
}
