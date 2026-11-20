// ==========================================================
// web/src/Evidence.tsx
// ==========================================================
// The verification ledger: every registered experiment with its criteria,
// verdict and provenance, the documented deviations and the test run.

import { useEffect, useState } from "react";
import { fmtM, fmtS, load, type Evidence as EvidenceData } from "./data";

// ------------------------------------------
// observation
// ------------------------------------------
// Formats a recorded observation with a unit that suits it.
function observation(value: number, unit: string, label: string) {
  if (unit === "s") return fmtS(value);
  if (unit === "m") return fmtM(value);
  if (/coverage|success.rate/i.test(label)) return `${(value * 100).toFixed(1)}%`;
  const number = Number.isInteger(value)
    ? value.toLocaleString("en-US")
    : value !== 0 && Math.abs(value) < 0.001
      ? value.toExponential(2)
      : Number(value.toPrecision(4)).toString();
  return `${number}${unit ? ` ${unit}` : ""}`;
}

// ------------------------------------------
// Evidence
// ------------------------------------------
// Loads evidence.json and lists every check.
export default function Evidence() {
  const [data, setData] = useState<EvidenceData | null>(null);
  const [error, setError] = useState(false);
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    let live = true;
    setError(false);
    load<EvidenceData>("evidence.json")
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
  return (
    <div>
      {error ? (
        <div className="callout" role="alert">
          Evidence could not be loaded.{" "}
          <button className="button ghost" onClick={() => setAttempt(v => v + 1)}>
            Retry evidence
          </button>
        </div>
      ) : !data ? (
        <p role="status">Loading experiment evidence…</p>
      ) : (
        <div className="stack">
          <div className="ledger">
            {data.experiments.map(e => {
              const supported = e.criteria.length > 0 && e.criteria.every(c => c.supported);
              return (
                <details key={e.id}>
                  <summary style={{ gridTemplateColumns: "3.4em minmax(0, 1fr)" }}>
                    <span className="id">{e.id}</span>
                    <span>{e.title}</span>
                    <span
                      className={`verdict${e.exploratory ? " neutral" : supported ? "" : " no"}`}
                      style={{ gridColumn: 2, justifySelf: "start", whiteSpace: "normal" }}
                    >
                      {e.exploratory ? "exploratory — no gate" : supported ? "supported" : "not supported"}
                    </span>
                  </summary>
                  <div className="body">
                    {e.criteria.length > 0 ? (
                      <ul>
                        {e.criteria.map(c => (
                          <li key={c.key}>
                            <strong>{c.key}</strong> · {c.text}{" "}
                            <span className={`verdict${c.supported ? "" : " no"}`}>
                              {c.supported ? "supported" : "not supported"}
                            </span>
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <p>No registered acceptance gate; observations are descriptive.</p>
                    )}
                    <dl style={{ margin: 0, display: "grid", gap: 8 }}>
                      {e.observations.map((o, i) => (
                        <div
                          key={`${o.label}-${i}`}
                          className="row"
                          style={{ justifyContent: "space-between", alignItems: "baseline" }}
                        >
                          <dt style={{ overflowWrap: "anywhere" }}>{o.label.replaceAll("_", " ")}</dt>
                          <dd className="num" style={{ margin: 0 }}>
                            {observation(o.value, o.unit, o.label)}
                          </dd>
                        </div>
                      ))}
                    </dl>
                    <p className="prov" style={{ overflowWrap: "anywhere" }}>
                      Seed {e.provenance.seed} · source{" "}
                      <span title={e.provenance.sourceSha256}>{e.provenance.sourceSha256.slice(0, 10)}</span> ·{" "}
                      <time dateTime={e.provenance.timestamp}>{e.provenance.timestamp.slice(0, 10)}</time>
                    </p>
                  </div>
                </details>
              );
            })}
          </div>
          <details className="panel">
            <summary style={{ cursor: "pointer", paddingBlock: 8 }}>
              Deviations from the registration ({data.deviations.length})
            </summary>
            <ul className="stack">
              {data.deviations.map((d, i) => (
                <li key={i} style={{ overflowWrap: "anywhere" }}>
                  {d.text}
                  <p className="prov">Source: {d.source}</p>
                </li>
              ))}
            </ul>
          </details>
          <div className="panel stack">
            <h3>Test suite</h3>
            <div className="stats">
              <div className="stat">
                <b>{data.tests.passed ?? "Not recorded"}</b>
                <span>tests passed</span>
              </div>
              <div className="stat">
                <b>{data.tests.deselected ?? "Not recorded"}</b>
                <span>statistical tests deselected (not run)</span>
              </div>
              <div className="stat">
                <b>{data.tests.collected}</b>
                <span>tests collected</span>
              </div>
            </div>
            <p className="small muted">
              Failed: {data.tests.failed ?? "not recorded"} · Skipped: {data.tests.skipped ?? "not recorded"}.
              Deselected tests are not included in the passing count.
            </p>
            <p className="small">
              Command: <code style={{ overflowWrap: "anywhere", whiteSpace: "pre-wrap" }}>{data.tests.command}</code>
            </p>
            <p className="prov">
              Source <span title={data.tests.sourceSha256}>{data.tests.sourceSha256.slice(0, 10)}</span> ·{" "}
              {data.tests.executedAt ? (
                <time dateTime={data.tests.executedAt}>{data.tests.executedAt.slice(0, 10)}</time>
              ) : (
                "Execution date not recorded"
              )}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
