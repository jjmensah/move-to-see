// ==========================================================
// web/src/Sensor.tsx
// ==========================================================
// The sensor playground: detection probability and range noise against
// range, for any reflectivity, fog and sunlight on the exported grid.

import { useEffect, useState } from "react";
import { scaleLinear } from "d3-scale";
import { line } from "d3-shape";
import { CONDS, load, type Cond, type SensorGrid, type World } from "./data";
import { sensorIndex } from "./lib/metrics";

// ------------------------------------------
// Pick
// ------------------------------------------
// Grid indices of the chosen reflectivity, visibility and sunlight.
type Pick = { rho: number; vis: number; sun: number };

// ------------------------------------------
// nearestIndex
// ------------------------------------------
// Index of the grid value closest to v.
function nearestIndex(values: number[], v: number) {
  return values.reduce((best, x, i) => (Math.abs(x - v) < Math.abs(values[best] - v) ? i : best), 0);
}

// ------------------------------------------
// Chart
// ------------------------------------------
// One curve against range, with a white-target reference and a cursor.
function Chart({
  grid,
  curve,
  refCurve,
  kind,
  cursor,
  setCursor,
}: {
  grid: SensorGrid;
  curve: SensorGrid["curves"][number];
  refCurve: SensorGrid["curves"][number];
  kind: "pd" | "sigma";
  cursor: number;
  setCursor: (i: number) => void;
}) {
  const W = 520,
    H = 230,
    m = { l: 48, r: 14, t: 14, b: 30 };
  const r = grid.rangeM;
  const x = scaleLinear()
    .domain([0, r[r.length - 1]])
    .range([m.l, W - m.r]);
  const yMax = kind === "pd" ? 1 : 0.1;
  const y = scaleLinear()
    .domain([0, yMax])
    .range([H - m.b, m.t])
    .clamp(true);
  const vals = (c: typeof curve) => (kind === "pd" ? c.pd : c.sigmaM);
  const usable = (c: typeof curve, i: number) => kind === "pd" || c.pd[i] >= 0.5;
  const path = (c: typeof curve) =>
    line<number>()
      .defined((_, i) => usable(c, i))
      .x((_, i) => x(r[i]))
      .y(v => y(Math.min(v, yMax)))(vals(c)) ?? "";
  const ticks = kind === "pd" ? [0, 0.25, 0.5, 0.75, 1] : [0, 0.025, 0.05, 0.075, 0.1];
  const v = vals(curve)[cursor];
  const pick = (clientX: number, el: SVGSVGElement) => {
    const b = el.getBoundingClientRect();
    const px = ((clientX - b.left) / b.width) * W;
    const d = x.invert(px);
    setCursor(Math.max(0, Math.min(r.length - 1, nearestIndex(r, d))));
  };
  return (
    <svg
      className="chart sensor-chart"
      viewBox={`0 0 ${W} ${H}`}
      role="img"
      style={{ touchAction: "pan-y" }}
      aria-label={kind === "pd" ? "Detection probability against range" : "Range noise against range"}
      onPointerMove={e => pick(e.clientX, e.currentTarget)}
      onPointerDown={e => pick(e.clientX, e.currentTarget)}
    >
      <g className="grid">
        {ticks.map(t => (
          <line key={t} x1={m.l} x2={W - m.r} y1={y(t)} y2={y(t)} />
        ))}
      </g>
      {ticks.map(t => (
        <text key={t} x={m.l - 6} y={y(t) + 3} textAnchor="end">
          {kind === "pd" ? t : `${Math.round(t * 1000)}`}
        </text>
      ))}
      {[0, 1, 2, 3, 4, 5].map(t => (
        <text key={t} x={x(t)} y={H - 10} textAnchor="middle">
          {t} m
        </text>
      ))}
      {kind === "pd" && (
        <line
          x1={m.l}
          x2={W - m.r}
          y1={y(0.5)}
          y2={y(0.5)}
          stroke="var(--ink)"
          strokeOpacity="0.4"
          strokeDasharray="4 3"
        />
      )}
      <path d={path(refCurve)} fill="none" stroke="var(--faint)" strokeWidth="1.5" strokeDasharray="5 4" />
      <path d={path(curve)} fill="none" stroke="var(--accent)" strokeWidth="2.6" strokeLinejoin="round" />
      {kind === "pd" && curve.r50M <= r[r.length - 1] && (
        <g>
          <line
            x1={x(curve.r50M)}
            x2={x(curve.r50M)}
            y1={y(0)}
            y2={y(0.5)}
            stroke="var(--accent)"
            strokeDasharray="2 3"
          />
          <text x={x(curve.r50M) + 4} y={y(0) - 6} style={{ fill: "var(--accent)" }}>
            r₅₀
          </text>
        </g>
      )}
      <line x1={x(r[cursor])} x2={x(r[cursor])} y1={m.t} y2={H - m.b} stroke="var(--ink)" strokeOpacity="0.35" />
      {usable(curve, cursor) && (
        <circle
          cx={x(r[cursor])}
          cy={y(Math.min(v, yMax))}
          r="4.5"
          fill="var(--accent)"
          stroke="var(--surface)"
          strokeWidth="2"
        />
      )}
    </svg>
  );
}

// ------------------------------------------
// Sensor
// ------------------------------------------
// Presets for the four tested conditions, three sliders and both charts.
export default function Sensor({ world, replayCondition }: { world: World; replayCondition: Cond }) {
  const [grid, setGrid] = useState<SensorGrid | null>(null);
  const [failed, setFailed] = useState(false);
  const [p, setP] = useState<Pick | null>(null);
  const [cursor, setCursor] = useState(100);

  useEffect(() => {
    load<SensorGrid>("sensor.json")
      .then(setGrid)
      .catch(() => setFailed(true));
  }, []);

  const preset = (c: Cond, g: SensorGrid): Pick => {
    const k = world.conditions[c];
    return {
      rho: nearestIndex(g.rho, k.rho),
      vis: nearestIndex(g.visibilityM, k.visibilityM),
      sun: nearestIndex(g.sun, k.sun),
    };
  };
  useEffect(() => {
    if (grid && !p) setP(preset("C2", grid));
  });

  if (failed)
    return <div className="error-box">The sensor curves could not be loaded. Refresh the page to try again.</div>;
  if (!grid || !p) return <div className="loading">Loading the sensor model…</div>;

  const curve = grid.curves[sensorIndex(p.rho, p.vis, p.sun)];
  const ref = grid.curves[sensorIndex(nearestIndex(grid.rho, 0.9), grid.visibilityM.length - 1, 0)];
  const r = grid.rangeM[cursor];
  const lastR = grid.rangeM[grid.rangeM.length - 1];
  const visLabel = (v: number) => (v >= 10000 ? "clear" : `${v} m`);
  const active = CONDS.find(c => {
    const q = preset(c, grid);
    return q.rho === p.rho && q.vis === p.vis && q.sun === p.sun;
  });

  return (
    <div className="sensor-layout">
      <div className="panel sensor-controls">
        <div className="stack">
          <div className="chart-title">Conditions</div>
          <div className="chart-sub">Start from one of the four tested conditions, then move the sliders.</div>
          <div className="segmented" role="group" aria-label="Tested condition">
            {CONDS.map(c => (
              <button key={c} type="button" aria-pressed={active === c} onClick={() => setP(preset(c, grid))}>
                {world.conditions[c].label}
              </button>
            ))}
          </div>
          {replayCondition !== active && (
            <button type="button" className="button ghost" onClick={() => setP(preset(replayCondition, grid))}>
              Use the replay's condition ({world.conditions[replayCondition].label})
            </button>
          )}
        </div>
        <div className="stack">
          <Slider
            label="Target reflectivity"
            value={grid.rho[p.rho].toFixed(2)}
            hint={p.rho === 0 ? "black" : p.rho === grid.rho.length - 1 ? "white" : ""}
            max={grid.rho.length - 1}
            v={p.rho}
            onChange={v => setP({ ...p, rho: v })}
          />
          <Slider
            label="Visibility (fog)"
            value={visLabel(grid.visibilityM[p.vis])}
            hint={`extinction ${grid.extinctionPerM[p.vis].toFixed(3)} /m`}
            max={grid.visibilityM.length - 1}
            v={p.vis}
            onChange={v => setP({ ...p, vis: v })}
          />
          <Slider
            label="Sunlight"
            value={`${grid.sun[p.sun]} sun`}
            hint={p.sun === 0 ? "none" : ""}
            max={grid.sun.length - 1}
            v={p.sun}
            onChange={v => setP({ ...p, sun: v })}
          />
        </div>
        <div className="stats sensor-stats">
          <div className="stat" style={{ ["--c" as string]: "var(--accent)" }}>
            <b className="num">{curve.r50M <= lastR ? `${curve.r50M.toFixed(2)} m` : `> ${lastR} m`}</b>
            <span>
              {curve.r50M <= lastR
                ? "50% detection range (r₅₀)"
                : `r₅₀ = ${curve.r50M.toFixed(1)} m, past the 5.6 m reporting limit`}
            </span>
          </div>
          <div className="stat">
            <b className="num">{curve.pd[cursor].toFixed(2)}</b>
            <span>
              detection probability at <span className="nowrap">{r.toFixed(2)} m</span>
            </span>
          </div>
        </div>
      </div>
      <div className="plot">
        <div className="chart-title">Will a beam detect the target?</div>
        <div className="chart-sub">
          Probability that one beam's return from a facing target, with nothing in the way, crosses the detection
          threshold. Dashed grey: a white target in clear air.
        </div>
        <Chart grid={grid} curve={curve} refCurve={ref} kind="pd" cursor={cursor} setCursor={setCursor} />
      </div>
      <div className="plot">
        <div className="chart-title">How noisy is the range it reports?</div>
        <div className="chart-sub">
          Predicted range-noise scale, in mm, before any phase-unwrapping errors; drawn where detection probability is
          at least 0.5, and least reliable close to that threshold.
        </div>
        <Chart grid={grid} curve={curve} refCurve={ref} kind="sigma" cursor={cursor} setCursor={setCursor} />
        <input
          type="range"
          min={0}
          max={grid.rangeM.length - 1}
          value={cursor}
          aria-label="Range"
          aria-valuetext={`${r.toFixed(2)} m: detection probability ${curve.pd[cursor].toFixed(2)}, range noise ${(curve.sigmaM[cursor] * 1000).toFixed(1)} mm`}
          onChange={e => setCursor(Number(e.target.value))}
        />
        <p className="small muted">
          At {r.toFixed(2)} m: detection probability <span className="num">{curve.pd[cursor].toFixed(3)}</span>, range
          noise <span className="num">{(curve.sigmaM[cursor] * 1000).toFixed(1)} mm</span>.
        </p>
      </div>
    </div>
  );
}

// ------------------------------------------
// Slider
// ------------------------------------------
// A labelled slider over grid indices.
function Slider({
  label,
  value,
  hint,
  max,
  v,
  onChange,
}: {
  label: string;
  value: string;
  hint: string;
  max: number;
  v: number;
  onChange: (v: number) => void;
}) {
  return (
    <label className="stack" style={{ gap: 2 }}>
      <span className="row" style={{ justifyContent: "space-between", fontSize: 14.5 }}>
        <span>{label}</span>
        <span>
          <b className="num">{value}</b>
          {hint && <span className="muted"> · {hint}</span>}
        </span>
      </span>
      <input
        type="range"
        min={0}
        max={max}
        step={1}
        value={v}
        onChange={e => onChange(Number(e.target.value))}
        aria-valuetext={`${value}${hint ? `, ${hint}` : ""}`}
      />
    </label>
  );
}
