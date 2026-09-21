"use client";

import Link from "next/link";
import { use } from "react";
import { PatternImage } from "@/components/pattern-image";
import { Card, Loading, PageHeader, Shell, useApi } from "@/components/ui";
import type { Pattern } from "@/lib/api";
import { num, shortId, when } from "@/lib/format";

export default function PatternDetail({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const { data, error, loading } = useApi<Pattern>(`/patterns/${id}`);
  const placement = data?.generation_parameters?.placement;

  return (
    <Shell>
      <PageHeader title={`Pattern ${shortId(id)}`} sub={data ? `version ${data.version}` : undefined} />
      <Loading error={error} loading={loading} />
      {data && (
        <div className="grid gap-6 lg:grid-cols-2">
          <Card title="Artifact">
            <PatternImage patternId={id} className="w-full max-w-xs rounded border" />
            <p className="mt-3 text-xs" style={{ color: "var(--ink-muted)" }}>
              This PNG is exactly what a printer receives — colours are snapped to the ink
              palette before export, so nothing degrades between the measurement and the print.
            </p>
          </Card>

          <div className="space-y-6">
            <Card title="Optimization">
              <dl className="grid grid-cols-2 gap-y-1 text-sm" style={{ color: "var(--ink-secondary)" }}>
                <dt>strategy</dt><dd>{data.metrics?.strategy}</dd>
                <dt>iterations</dt><dd className="tabular-nums">{data.metrics?.iterations}</dd>
                <dt>initial loss</dt><dd className="tabular-nums">{num(data.metrics?.initial_loss, 5)}</dd>
                <dt>final loss</dt><dd className="tabular-nums">{num(data.metrics?.final_loss, 5)}</dd>
                <dt>improvement</dt><dd className="tabular-nums">{num(data.metrics?.improvement, 5)}</dd>
                <dt>created</dt><dd>{when(data.created_at)}</dd>
              </dl>
            </Card>

            <Card title="Generation parameters">
              <dl className="grid grid-cols-2 gap-y-1 text-xs" style={{ color: "var(--ink-secondary)" }}>
                {Object.entries(data.generation_parameters ?? {})
                  .filter(([k]) => k !== "transform_spec" && k !== "placement")
                  .map(([k, v]) => (
                    <div key={k} className="contents">
                      <dt>{k}</dt>
                      <dd className="tabular-nums">{String(v)}</dd>
                    </div>
                  ))}
              </dl>
              {placement && (
                <p className="mt-3 text-xs" style={{ color: "var(--ink-muted)" }}>
                  placement: centre ({placement.cx?.toFixed(2)}, {placement.cy?.toFixed(2)}), size{" "}
                  {placement.width?.toFixed(2)} × {placement.height?.toFixed(2)} of the frame
                </p>
              )}
            </Card>

            <Link href={`/experiments/${data.experiment_id}`} className="block text-sm underline">
              ← Experiment that produced it
            </Link>
          </div>
        </div>
      )}
    </Shell>
  );
}
