import React from "react";

// Small hand-built SVG/HTML charts (no chart library). Colours come from the
// --viz-* custom properties in styles.css, which were checked together for
// colour-blind separation. Every chart labels its values directly, so
// nothing depends on colour alone, and marks carry a <title> for hover.

const pct = (v, digits = 0) => `${(v * 100).toFixed(digits)}%`;

/**
 * The 0-100% probability scale with the three confidence zones marked and
 * this patient's estimate placed on it. Low confidence is the middle of the
 * scale (near 50%); High is at either end.
 */
export function ProbabilityScale({ probability, bands, compact = false }) {
  const moderate = bands?.find((b) => b.level === "Moderate")?.min_distance ?? 0.15;
  const high = bands?.find((b) => b.level === "High")?.min_distance ?? 0.3;
  const zones = [
    { from: 0, to: 0.5 - high, level: "high" },
    { from: 0.5 - high, to: 0.5 - moderate, level: "moderate" },
    { from: 0.5 - moderate, to: 0.5 + moderate, level: "low" },
    { from: 0.5 + moderate, to: 0.5 + high, level: "moderate" },
    { from: 0.5 + high, to: 1, level: "high" },
  ];
  const side = probability >= 0.5 ? "likely" : "unlikely";
  return (
    <div className={`prob-scale ${compact ? "compact" : ""}`}
         role="img" aria-label={`Estimated probability of return to work ${pct(probability)}`}>
      <div className="prob-scale-track">
        {zones.map((z) => (
          <div key={z.from} className={`zone ${z.level} ${z.from < 0.5 ? "unlikely" : "likely"}`}
               style={{ left: `${z.from * 100}%`, width: `${(z.to - z.from) * 100}%` }} />
        ))}
        <div className="midline" />
        <div className={`marker ${side}`} style={{ left: `${probability * 100}%` }}>
          <span className="marker-label">{pct(probability)}</span>
        </div>
      </div>
      {!compact && (
        <>
          <div className="prob-scale-zones" aria-hidden="true">
            <span style={{ width: `${(0.5 - high) * 100}%` }}>High</span>
            <span style={{ width: `${(high - moderate) * 100}%` }}>Moderate</span>
            <span style={{ width: `${moderate * 200}%` }}>Low confidence</span>
            <span style={{ width: `${(high - moderate) * 100}%` }}>Moderate</span>
            <span style={{ width: `${(0.5 - high) * 100}%` }}>High</span>
          </div>
          <div className="prob-scale-ends" aria-hidden="true">
            <span>0% unlikely to return</span>
            <span>50%</span>
            <span>likely to return 100%</span>
          </div>
        </>
      )}
    </div>
  );
}

/** One bar split into labelled parts of a whole, e.g. RTW vs Non-RTW. */
export function SplitBar({ parts }) {
  const total = parts.reduce((sum, p) => sum + p.value, 0);
  if (!total) return null;
  return (
    <div className="split-bar-wrap">
      <div className="split-bar" role="img"
           aria-label={parts.map((p) => `${p.label} ${p.value}`).join(", ")}>
        {parts.filter((p) => p.value > 0).map((p) => (
          <div key={p.label} className={`seg ${p.tone}`} style={{ flexGrow: p.value }}
               title={`${p.label}: ${p.value} (${pct(p.value / total)})`} />
        ))}
      </div>
      <div className="split-legend">
        {parts.map((p) => (
          <div key={p.label} className="split-legend-item">
            <div className="split-figure">
              <span className={`swatch ${p.tone}`} />
              <span className="split-value">{p.value}</span>
            </div>
            <span className="split-label">{p.label}, {pct(p.value / total)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

/** Horizontal bars for a few ordered or named categories, each labelled. */
export function BarList({ items }) {
  const max = Math.max(...items.map((i) => i.value), 1);
  return (
    <div className="bar-list">
      {items.map((item) => (
        <div className="bar-list-row" key={item.label} title={`${item.label}: ${item.value}`}>
          <span className="bar-list-label">{item.label}</span>
          <div className="bar-list-track">
            <div className={`bar-list-fill ${item.tone || ""}`} style={{ width: `${(item.value / max) * 100}%` }} />
          </div>
          <span className="bar-list-value">{item.value}</span>
          {item.note && <span className="bar-list-note">{item.note}</span>}
        </div>
      ))}
    </div>
  );
}

/** Columns over time. `points` is [{label, value, title}] in time order. */
export function ColumnChart({ points, height = 132 }) {
  // Whole-number gridlines: at most four steps, with the top of the axis
  // rounded up to a multiple of the step so labels are evenly spaced.
  const peak = Math.max(...points.map((p) => p.value), 1);
  const step = Math.ceil(peak / 4);
  const ticks = Math.ceil(peak / step);
  const max = ticks * step;
  return (
    <div className="column-chart" style={{ "--plot-height": `${height}px` }}
         role="img" aria-label={points.map((p) => `${p.title}: ${p.value}`).join(", ")}>
      <div className="column-plot">
        {Array.from({ length: ticks + 1 }, (_, i) => (
          <div key={i} className="gridline" style={{ bottom: `${(i / ticks) * 100}%` }}>
            <span>{i * step}</span>
          </div>
        ))}
        <div className="columns">
          {points.map((p) => (
            <div className="column-slot" key={p.title} title={`${p.title}: ${p.value}`}>
              {p.value > 0 && <span className="column-value">{p.value}</span>}
              <div className="column" style={{ height: `${(p.value / max) * 100}%` }} />
            </div>
          ))}
        </div>
      </div>
      <div className="column-labels" aria-hidden="true">
        {points.map((p) => <span key={p.title}>{p.label}</span>)}
      </div>
    </div>
  );
}

/**
 * Dot-and-interval plot: each row's value with its 95% interval, against a
 * reference line. Used to show model accuracy against the majority-class
 * guess, where overlapping intervals mean "no real difference".
 */
export function IntervalPlot({ rows, reference, min, max }) {
  const x = (v) => `${((v - min) / (max - min)) * 100}%`;
  const ticks = [];
  for (let t = Math.ceil(min * 20) / 20; t <= max + 1e-9; t += 0.05) ticks.push(Number(t.toFixed(2)));
  return (
    <div className="interval-plot">
      {rows.map((row) => (
        <div className="interval-row" key={row.label}>
          <span className="interval-label">{row.label}</span>
          <div className="interval-track"
               title={`${row.label}: ${pct(row.value, 1)} (95% interval ${pct(row.low, 1)} to ${pct(row.high, 1)})`}>
            {ticks.map((t) => <div key={t} className="tick" style={{ left: x(t) }} />)}
            <div className="reference" style={{ left: x(reference.value) }} />
            <div className="interval" style={{ left: x(row.low), width: `calc(${x(row.high)} - ${x(row.low)})` }} />
            <div className="dot" style={{ left: x(row.value) }} />
          </div>
          <span className="interval-value">{pct(row.value, 1)}</span>
        </div>
      ))}
      <div className="interval-row axis" aria-hidden="true">
        <span className="interval-label" />
        <div className="interval-track">
          {ticks.map((t) => <span key={t} className="tick-label" style={{ left: x(t) }}>{pct(t)}</span>)}
        </div>
        <span className="interval-value" />
      </div>
      <p className="interval-key">
        <span className="key-dot" /> test accuracy
        <span className="key-line" /> 95% interval
        <span className="key-ref" /> {reference.label} ({pct(reference.value, 1)})
      </p>
    </div>
  );
}
