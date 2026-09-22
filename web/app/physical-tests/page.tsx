"use client";

import Link from "next/link";
import { Card, Loading, PageHeader, Shell, Table, useApi } from "@/components/ui";
import type { PhysicalTest } from "@/lib/api";
import { pct, shortId, when } from "@/lib/format";

export default function PhysicalTests() {
  const { data, error, loading } = useApi<PhysicalTest[]>("/physical-tests");
  return (
    <Shell>
      <PageHeader
        title="Physical tests"
        sub="Real cameras, real prints, recorded conditions. Never merged with simulated results."
      />
      <Loading error={error} loading={loading} />
      {!error && (
        <div className="space-y-6">
          <Card>
            <Table
              head={["Experiment", "Arm", "Camera", "Resolution", "Distance", "Angle", "Lighting", "Environment", "Frames", "Detection rate", "Recorded"]}
              empty="No physical tests recorded. Run scripts/physical_test.py against a printed pattern."
              rows={(data ?? []).map((t) => [
                <Link key="e" href={`/experiments/${t.experiment_id}`} className="underline">
                  {shortId(t.experiment_id)}
                </Link>,
                <strong key="arm">{t.arm || "unspecified"}</strong>,
                t.camera,
                <span key="r" style={{ color: "var(--ink-secondary)" }}>{t.resolution}</span>,
                <span key="d" className="tabular-nums">{t.distance_m ?? "—"} m</span>,
                <span key="a" className="tabular-nums">{t.angle_deg ?? "—"}°</span>,
                t.lighting,
                t.environment,
                <span key="f" className="tabular-nums">{t.frame_count}</span>,
                <span key="p" className="tabular-nums">{pct(t.result?.detection_rate)}</span>,
                <span key="w" style={{ color: "var(--ink-muted)" }}>{when(t.created_at)}</span>,
              ])}
            />
          </Card>
          <Card title="Recording one">
            <pre className="overflow-x-auto text-xs leading-relaxed" style={{ color: "var(--ink-secondary)" }}>
{`python scripts/physical_test.py \\
  --api-key "$VEIL_API_KEY" \\
  --experiment "$EXPERIMENT_ID" --pattern "$PATTERN_ID" \\
  --frames 60 --distance 3.0 --angle 15 \\
  --lighting "office fluorescent, ~400 lux" \\
  --environment "lab, white wall"`}
            </pre>
            <p className="mt-3 text-xs" style={{ color: "var(--ink-muted)" }}>
              Distance, angle, lighting and environment are required. A physical number without its
              conditions is not a result.
            </p>
          </Card>
        </div>
      )}
    </Shell>
  );
}
