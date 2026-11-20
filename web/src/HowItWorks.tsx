// ==========================================================
// web/src/HowItWorks.tsx
// ==========================================================
// The loop in three cards: scan, update, move. Each card has a small drawing
// in theme colours; the numbers in the text come from the project's settings.

// ------------------------------------------
// dots
// ------------------------------------------
// Deterministic scatter for the drawings, so every render looks the same.
function dots(n: number, seed: number, cx: number, cy: number, sx: number, sy: number): [number, number][] {
  let s = seed;
  const next = () => ((s = (s * 16807) % 2147483647) - 1) / 2147483646;
  return Array.from({ length: n }, () => [cx + (next() - 0.5) * 2 * sx, cy + (next() - 0.5) * 2 * sy]);
}

// ------------------------------------------
// ScanArt
// ------------------------------------------
// A robot fanning beams through fog; only a few reach the dark box.
function ScanArt() {
  const angles = [-38, -26, -14, -2, 10, 22, 34];
  return (
    <svg viewBox="0 0 240 150" aria-hidden="true">
      <ellipse cx="150" cy="60" rx="70" ry="26" fill="var(--surface)" opacity="0.8" />
      <ellipse cx="175" cy="96" rx="58" ry="20" fill="var(--surface)" opacity="0.6" />
      {angles.map(a => {
        const r = (a * Math.PI) / 180;
        const hit = Math.abs(a + 2) < 8;
        return (
          <line
            key={a}
            x1="44"
            y1="102"
            x2={44 + Math.cos(r) * (hit ? 132 : 104)}
            y2={102 + Math.sin(r) * (hit ? 132 : 104)}
            stroke="var(--info)"
            strokeWidth="2"
            strokeDasharray="3 5"
            strokeLinecap="round"
            opacity={hit ? 1 : 0.35}
          />
        );
      })}
      <rect x="168" y="84" width="22" height="22" rx="3" fill="var(--truth)" />
      <circle cx="44" cy="102" r="13" fill="var(--coral-lo)" stroke="var(--surface)" strokeWidth="3" />
      <line x1="44" y1="102" x2="62" y2="102" stroke="var(--surface)" strokeWidth="3" strokeLinecap="round" />
    </svg>
  );
}

// ------------------------------------------
// UpdateArt
// ------------------------------------------
// A wide particle cloud on the left narrows to a tight ellipse on the right.
function UpdateArt() {
  const wide = dots(46, 7, 62, 75, 40, 42);
  const tight = dots(30, 11, 178, 75, 12, 9);
  return (
    <svg viewBox="0 0 240 150" aria-hidden="true">
      {wide.map(([x, y], i) => (
        <circle key={`w${i}`} cx={x} cy={y} r="2.6" fill="var(--soft)" opacity="0.55" />
      ))}
      <path d="M108 75 H138" stroke="var(--ink)" strokeWidth="2.4" strokeLinecap="round" />
      <path d="M131 68 L139 75 L131 82" fill="none" stroke="var(--ink)" strokeWidth="2.4" strokeLinecap="round" />
      <ellipse
        cx="178"
        cy="75"
        rx="26"
        ry="20"
        fill="var(--info)"
        fillOpacity="0.12"
        stroke="var(--info)"
        strokeWidth="2"
      />
      {tight.map(([x, y], i) => (
        <circle key={`t${i}`} cx={x} cy={y} r="2.4" fill="var(--ink)" opacity="0.7" />
      ))}
      <rect x="183" y="64" width="12" height="12" rx="2" fill="var(--truth)" stroke="var(--surface)" strokeWidth="2" />
    </svg>
  );
}

// ------------------------------------------
// MoveArt
// ------------------------------------------
// A lattice of candidate viewpoints shaded by score, and the dashed drive to the best.
function MoveArt() {
  const cells: { x: number; y: number; score: number }[] = [];
  for (let i = 0; i < 9; i++) {
    for (let j = 0; j < 5; j++) {
      const x = 32 + i * 22;
      const y = 30 + j * 22;
      const score = Math.max(0, 1 - Math.hypot(x - 164, y - 74) / 120);
      cells.push({ x, y, score });
    }
  }
  return (
    <svg viewBox="0 0 240 150" aria-hidden="true">
      {cells.map(c => (
        <circle key={`${c.x}-${c.y}`} cx={c.x} cy={c.y} r="5" fill="var(--info)" opacity={0.15 + 0.85 * c.score} />
      ))}
      <path
        d="M32 118 C 70 120, 110 96, 142 76"
        fill="none"
        stroke="var(--ink)"
        strokeWidth="2.4"
        strokeDasharray="5 5"
      />
      <circle cx="142" cy="74" r="12" fill="none" stroke="var(--coral-lo)" strokeWidth="3" />
      <circle cx="32" cy="118" r="10" fill="var(--coral-lo)" stroke="var(--surface)" strokeWidth="3" />
    </svg>
  );
}

// ------------------------------------------
// HowItWorks
// ------------------------------------------
// Three step cards under one heading.
export default function HowItWorks() {
  const steps = [
    {
      art: <ScanArt />,
      title: "Scan",
      text: "Every half second the LiDAR fires 1,331 beams across 240°. A dark box in fog answers only a few of them, and the sensor model predicts how many.",
    },
    {
      art: <UpdateArt />,
      title: "Update",
      text: "2,000 particles hold every place the target could be. Each scan reweights them with the physics-based likelihood, counting the beams that came back empty too.",
    },
    {
      art: <MoveArt />,
      title: "Move",
      text: "The planner weighs a lattice of reachable viewpoints: what it expects to learn there, minus a cost for the drive. It heads for the best, then scans again.",
    },
  ];
  return (
    <div className="steps3">
      {steps.map(s => (
        <article key={s.title} className="step reveal">
          <div className="step-art">{s.art}</div>
          <h3>{s.title}</h3>
          <p>{s.text}</p>
        </article>
      ))}
    </div>
  );
}
