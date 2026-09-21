"use client";

import { useState } from "react";
import { pct } from "@/lib/format";

/**
 * Grouped horizontal bars: detection rate per category, one bar per arm.
 *
 * Horizontal because the category labels are text ("brightness 0.8", "-30°"),
 * and bars because the job is comparing magnitudes across a small set of
 * categories. Every bar is labelled with its value and every arm is named in
 * the legend, so identity never rests on colour alone. Bars start at zero and
 * share one 0-100% scale - there is no second axis.
 *
 * The control arm is omitted from the legend and the rows when a run did not
 * measure it, rather than drawn as an empty bar.
 */

export type Row = {
  label: string;
  baseline: number | null;
  control?: number | null;
  candidate: number | null;
  samples?: number;
};

const SERIES = [
  { key: "baseline" as const, name: "Baseline (no pattern)", color: "var(--series-baseline)" },
  { key: "control" as const, name: "Control (unoptimized pattern)", color: "var(--series-control)" },
  { key: "candidate" as const, name: "Candidate (optimized pattern)", color: "var(--series-candidate)" },
];

export function ComparisonBars({ rows, caption }: { rows: Row[]; caption?: string }) {
  const [hover, setHover] = useState<string | null>(null);
  if (rows.length === 0) {
    return <p className="text-sm" style={{ color: "var(--ink-secondary)" }}>Nothing measured on this axis.</p>;
  }
  const hasControl = rows.some((r) => r.control !== null && r.control !== undefined);
  const series = SERIES.filter((s) => s.key !== "control" || hasControl);

  return (
    <figure className="m-0">
      <ul className="mb-4 flex flex-wrap gap-4 text-xs" style={{ color: "var(--ink-secondary)" }}>
        {series.map((s) => (
          <li key={s.key} className="flex items-center gap-2">
            <span aria-hidden className="inline-block h-2 w-4 rounded-sm" style={{ background: s.color }} />
            {s.name}
          </li>
        ))}
      </ul>

      <div className="space-y-4">
        {rows.map((row) => (
          <div key={row.label}>
            <div className="mb-1 flex justify-between text-xs" style={{ color: "var(--ink-secondary)" }}>
              <span>{row.label}</span>
              {row.samples !== undefined && <span>n={row.samples}</span>}
            </div>
            {series.map((s) => {
              const value = row[s.key] ?? null;
              const id = `${row.label}-${s.key}`;
              return (
                <div
                  key={s.key}
                  className="mb-[2px] flex items-center gap-3"
                  onMouseEnter={() => setHover(id)}
                  onMouseLeave={() => setHover(null)}
                  title={`${s.name} — ${row.label}: ${pct(value)}`}
                >
                  <div className="h-3 flex-1 overflow-hidden rounded-sm" style={{ background: "var(--surface)" }}>
                    {value !== null && (
                      <div
                        className="h-full rounded-r"
                        style={{
                          width: `${Math.max(value * 100, value > 0 ? 1 : 0)}%`,
                          background: s.color,
                          opacity: hover && hover !== id ? 0.55 : 1,
                        }}
                      />
                    )}
                  </div>
                  <span className="w-28 shrink-0 text-right text-xs tabular-nums"
                        style={{ color: "var(--ink-secondary)" }}>
                    {pct(value)}
                  </span>
                </div>
              );
            })}
          </div>
        ))}
      </div>

      {caption && (
        <figcaption className="mt-4 text-xs" style={{ color: "var(--ink-muted)" }}>{caption}</figcaption>
      )}
    </figure>
  );
}

/**
 * A robustness component. A meter, not a chart: one value, shown against its
 * full range, with the sample count that produced it. Renders "not measured"
 * rather than an empty bar when the axis was never exercised.
 */
export function ScoreMeter({ label, value, samples }: { label: string; value: number | null; samples?: number }) {
  return (
    <div>
      <div className="mb-1 flex justify-between text-xs">
        <span style={{ color: "var(--ink-secondary)" }}>{label}</span>
        <span className="tabular-nums" style={{ color: value === null ? "var(--ink-muted)" : "var(--ink)" }}>
          {pct(value)}
          {samples !== undefined && value !== null && (
            <span style={{ color: "var(--ink-muted)" }}> · n={samples}</span>
          )}
        </span>
      </div>
      <div className="h-2 overflow-hidden rounded-sm" style={{ background: "var(--surface)" }}>
        {value !== null && (
          <div className="h-full rounded-r" style={{ width: `${value * 100}%`, background: "var(--series-candidate)" }} />
        )}
      </div>
    </div>
  );
}

/**
 * How the measured drop splits between the patch covering the subject and
 * the pattern itself. A stacked bar, because the two parts sum to the whole
 * and that relationship is the entire point.
 */
export function AttributionBar({
  attribution,
  candidateRate,
}: {
  attribution: Record<string, any>;
  candidateRate?: number | null;
}) {
  const spread = attribution?.control_spread;
  if (!attribution?.available) {
    return (
      <div>
        <p className="text-sm" style={{ color: "var(--warning)" }}>
          Not attributable — {attribution?.reason ?? "no control arm was run"}.
        </p>
        <p className="mt-2 text-xs" style={{ color: "var(--ink-muted)" }}>
          Without a control arm, the drop below cannot be separated from the patch simply
          covering part of the subject. Re-run with <code>control: true</code>.
        </p>
      </div>
    );
  }
  // A positive margin is only a win if it exceeds the spread between the
  // random control patterns - otherwise it must not be styled like one.
  const significance = attribution.control_spread ? attribution.significance : null;
  const vsBest = attribution.attributable_vs_best_control;
  const clears =
    vsBest !== null && vsBest !== undefined && vsBest > 0 &&
    (!significance?.available || significance.significant_against_all);
  const total = Math.max(attribution.total_drop, 1e-9);
  const occlusion = Math.max(attribution.occlusion_drop, 0) / total;
  const pattern = Math.max(attribution.attributable_drop, 0) / total;

  return (
    <div>
      <div className="mb-2 flex h-4 overflow-hidden rounded-sm" style={{ background: "var(--surface)" }}>
        <div style={{ width: `${occlusion * 100}%`, background: "var(--series-control)" }} />
        <div className="ml-[2px]" style={{ width: `${pattern * 100}%`, background: "var(--series-candidate)" }} />
      </div>
      <dl className="grid grid-cols-2 gap-y-1 text-xs" style={{ color: "var(--ink-secondary)" }}>
        <dt>total drop</dt>
        <dd className="tabular-nums text-right">{pct(attribution.total_drop)}</dd>
        <dt>
          <span aria-hidden className="mr-2 inline-block h-2 w-3 rounded-sm"
                style={{ background: "var(--series-control)" }} />
          occlusion (the patch)
        </dt>
        <dd className="tabular-nums text-right">{pct(attribution.occlusion_drop)}</dd>
        <dt>
          <span aria-hidden className="mr-2 inline-block h-2 w-3 rounded-sm"
                style={{ background: "var(--series-candidate)" }} />
          the pattern itself
        </dt>
        <dd className="tabular-nums text-right">{pct(attribution.attributable_drop)}</dd>
        {attribution.attributable_vs_best_control !== null &&
          attribution.attributable_vs_best_control !== undefined && (
            <>
              <dt>vs. the best control draw</dt>
              <dd className="tabular-nums text-right" style={{ color: clears ? "var(--ink-secondary)" : "var(--warning)" }}>
                {pct(attribution.attributable_vs_best_control)}
                {!clears && <span style={{ color: "var(--ink-muted)" }}> (within noise)</span>}
              </dd>
            </>
          )}
        {spread?.draws > 1 && (
          <>
            <dt>spread across {spread.draws} control draws</dt>
            <dd className="tabular-nums text-right">{pct(spread.range)}</dd>
          </>
        )}
        {significance?.available && (
          <>
            <dt>worst p-value vs. {significance.draws} control draw(s)</dt>
            <dd className="tabular-nums text-right"
                style={{ color: significance.significant_against_all
                  ? "var(--ink-secondary)" : "var(--warning)" }}>
              {significance.worst_p_value.toFixed(4)}
            </dd>
          </>
        )}
        <dt>samples per arm</dt>
        <dd className="tabular-nums text-right">{attribution.samples_per_arm}</dd>
      </dl>

      {spread?.draws > 1 && (
        <div className="mt-3">
          <div className="mb-1 text-xs" style={{ color: "var(--ink-muted)" }}>
            control draws (each an unoptimized pattern)
          </div>
          <div className="flex flex-wrap gap-2 text-xs tabular-nums">
            {[...spread.rates].sort((a: number, b: number) => a - b).map((rate: number, i: number) => (
              <span key={i} className="rounded px-2 py-1"
                    style={{ background: "var(--surface)", color: "var(--ink-secondary)" }}>
                {pct(rate)}
              </span>
            ))}
            <span className="rounded px-2 py-1"
                  style={{ background: "var(--surface)", color: "var(--series-candidate)" }}>
              candidate {pct(candidateRate)}
            </span>
          </div>
        </div>
      )}
      <p className="mt-3 text-xs leading-relaxed"
         style={{ color: clears ? "var(--ink-secondary)" : "var(--warning)" }}>
        {attribution.verdict}.
      </p>
      {significance?.available && (
        <p className="mt-2 text-xs" style={{ color: "var(--ink-muted)" }}>
          {significance.test}; the worst of the per-draw p-values is shown, because the
          claim must hold against every unoptimized control rather than the weakest one.
        </p>
      )}
    </div>
  );
}
