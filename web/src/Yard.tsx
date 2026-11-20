// ==========================================================
// web/src/Yard.tsx
// ==========================================================
// Top-down view of the yard in world metres, y up. Everything drawn on it
// goes through `sx` and `sy`, so a metre has the same length on both axes.

import type { ReactNode } from "react";
import type { V2, World } from "./data";
import { ellipse } from "./lib/geometry";

// Top-down view of the yard in world metres, y up. Everything drawn on it goes
// through `sx`/`sy`, so one metre is the same length on both axes.

// ------------------------------------------
// Scale
// ------------------------------------------
// Pixels per metre in the view box, and the margin around the yard.
export const PX = 56; // pixels per metre in the viewBox
const PAD = 26;

// ------------------------------------------
// Frame
// ------------------------------------------
// World-to-SVG mapping of the yard.
export type Frame = { sx: (x: number) => number; sy: (y: number) => number; w: number; h: number };

// ------------------------------------------
// frame
// ------------------------------------------
// Builds the mapping for a world.
export function frame(world: World): Frame {
  const [x0, x1, y0, y1] = world.arena;
  return {
    sx: x => PAD + (x - x0) * PX,
    sy: y => PAD + (y1 - y) * PX,
    w: (x1 - x0) * PX + 2 * PAD,
    h: (y1 - y0) * PX + 2 * PAD,
  };
}

// ------------------------------------------
// YardBase
// ------------------------------------------
// Arena, grid, rough ground, prior region, fixed route and crates.
export function YardBase({ world, f, showRoute = true }: { world: World; f: Frame; showRoute?: boolean }) {
  const [x0, x1, y0, y1] = world.arena;
  const ticksX = Array.from({ length: Math.floor(x1 - x0) + 1 }, (_, i) => x0 + i);
  const ticksY = Array.from({ length: Math.floor(y1 - y0) + 1 }, (_, i) => y0 + i);
  const route = world.fixedRoute.map(([x, y]) => `${f.sx(x)},${f.sy(y)}`).join(" ");
  return (
    <g>
      <defs>
        <clipPath id="yard-clip">
          <rect x={f.sx(x0)} y={f.sy(y1)} width={(x1 - x0) * PX} height={(y1 - y0) * PX} />
        </clipPath>
        <pattern id="rough-hatch" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <line x1="0" y1="0" x2="0" y2="7" stroke="var(--rough)" strokeWidth="2" />
        </pattern>
      </defs>
      <rect
        x={f.sx(x0)}
        y={f.sy(y1)}
        width={(x1 - x0) * PX}
        height={(y1 - y0) * PX}
        fill="var(--surface)"
        stroke="var(--rule)"
      />
      <g stroke="var(--grid)" strokeWidth="1">
        {ticksX.slice(1, -1).map(x => (
          <line key={`gx${x}`} x1={f.sx(x)} x2={f.sx(x)} y1={f.sy(y0)} y2={f.sy(y1)} />
        ))}
        {ticksY.slice(1, -1).map(y => (
          <line key={`gy${y}`} x1={f.sx(x0)} x2={f.sx(x1)} y1={f.sy(y)} y2={f.sy(y)} />
        ))}
      </g>
      {ticksX
        .filter(x => x % 2 === 0)
        .map(x => (
          <text key={`tx${x}`} x={f.sx(x)} y={f.sy(y0) + 15} textAnchor="middle">
            {x} m
          </text>
        ))}
      {ticksY
        .filter(y => y % 2 === 0 && y > y0)
        .map(y => (
          <text key={`ty${y}`} x={f.sx(x0) - 6} y={f.sy(y) + 3} textAnchor="end">
            {y}
          </text>
        ))}
      {world.rough.map(([a, b, c, d], i) => (
        <rect
          key={`r${i}`}
          x={f.sx(a)}
          y={f.sy(d)}
          width={(b - a) * PX}
          height={(d - c) * PX}
          fill="url(#rough-hatch)"
          opacity="0.55"
        />
      ))}
      <rect
        x={f.sx(world.prior.lo[0])}
        y={f.sy(world.prior.hi[1])}
        width={(world.prior.hi[0] - world.prior.lo[0]) * PX}
        height={(world.prior.hi[1] - world.prior.lo[1]) * PX}
        fill="none"
        stroke="var(--prior)"
        strokeDasharray="2 4"
        strokeWidth="1.4"
      />
      {showRoute && (
        <polyline
          points={route}
          fill="none"
          stroke="var(--fixed)"
          strokeOpacity="0.28"
          strokeWidth="1.5"
          strokeDasharray="6 5"
          strokeLinejoin="round"
        />
      )}
      {world.boxes.map((b, i) => (
        <rect
          key={`b${i}`}
          x={f.sx(b.lo[0])}
          y={f.sy(b.hi[1])}
          width={(b.hi[0] - b.lo[0]) * PX}
          height={(b.hi[1] - b.lo[1]) * PX}
          fill="var(--obstacle)"
          stroke="var(--obstacle-edge)"
          fillOpacity="0.85"
          rx="1.5"
        />
      ))}
    </g>
  );
}

// ------------------------------------------
// Robot
// ------------------------------------------
// Robot marker: a disc with a heading tick.
export function Robot({
  f,
  x,
  y,
  theta,
  color,
  dim = false,
}: {
  f: Frame;
  x: number;
  y: number;
  theta: number;
  color: string;
  dim?: boolean;
}) {
  const cx = f.sx(x),
    cy = f.sy(y);
  const hx = cx + Math.cos(theta) * 15,
    hy = cy - Math.sin(theta) * 15;
  return (
    <g opacity={dim ? 0.45 : 1}>
      <circle cx={cx} cy={cy} r="8" fill={color} stroke="var(--surface)" strokeWidth="2" />
      <line x1={cx} y1={cy} x2={hx} y2={hy} stroke={color} strokeWidth="3" strokeLinecap="round" />
    </g>
  );
}

// ------------------------------------------
// Belief
// ------------------------------------------
// Belief mean and the 2σ ellipse of its xy marginal.
export function Belief({ f, mean, cov, color }: { f: Frame; mean: V2; cov: [number, number, number]; color: string }) {
  const e = ellipse(cov);
  const cx = f.sx(mean[0]),
    cy = f.sy(mean[1]);
  const deg = (-e.angle * 180) / Math.PI;
  return (
    <g>
      <ellipse
        cx={cx}
        cy={cy}
        rx={Math.max(e.a * PX, 1.5)}
        ry={Math.max(e.b * PX, 1.5)}
        transform={`rotate(${deg} ${cx} ${cy})`}
        fill={color}
        fillOpacity="0.1"
        stroke={color}
        strokeWidth="1.4"
      />
      <path
        d={`M${cx - 5},${cy - 5}L${cx + 5},${cy + 5}M${cx - 5},${cy + 5}L${cx + 5},${cy - 5}`}
        stroke={color}
        strokeWidth="2"
        strokeLinecap="round"
      />
    </g>
  );
}

// ------------------------------------------
// Fov
// ------------------------------------------
// Nominal field of view: 240° wide, out to the 5.6 m reporting limit.
export function Fov({ f, x, y, theta, color }: { f: Frame; x: number; y: number; theta: number; color: string }) {
  const r = 5.6 * PX,
    half = (2 * Math.PI) / 3;
  const cx = f.sx(x),
    cy = f.sy(y);
  const p = (a: number) => `${cx + Math.cos(a) * r},${cy - Math.sin(a) * r}`;
  return (
    <path
      d={`M${cx},${cy}L${p(theta - half)}A${r},${r} 0 1 0 ${p(theta + half)}Z`}
      fill={color}
      fillOpacity="0.05"
      stroke={color}
      strokeOpacity="0.2"
      strokeDasharray="3 4"
    />
  );
}

// ------------------------------------------
// Target
// ------------------------------------------
// Footprint of the true target.
export function Target({ f, world, q }: { f: Frame; world: World; q: number[] }) {
  const [hx, hy] = world.targetHalf;
  return (
    <rect
      x={f.sx(q[0] - hx)}
      y={f.sy(q[1] + hy)}
      width={2 * hx * PX}
      height={2 * hy * PX}
      fill="var(--truth)"
      stroke="var(--surface)"
      strokeWidth="1.5"
      rx="1.5"
    />
  );
}

// ------------------------------------------
// YardSvg
// ------------------------------------------
// The yard as an accessible SVG; children are clipped to the arena.
export function YardSvg({
  world,
  label,
  children,
  desc,
}: {
  world: World;
  label: string;
  desc?: string;
  children: (f: Frame) => ReactNode;
}) {
  const f = frame(world);
  return (
    <svg viewBox={`0 0 ${f.w} ${f.h}`} role="img" aria-label={label}>
      {desc && <desc>{desc}</desc>}
      <YardBase world={world} f={f} />
      <g clipPath="url(#yard-clip)">{children(f)}</g>
    </svg>
  );
}
