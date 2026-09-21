"use client";

import Link from "next/link";
import { Card, Loading, PageHeader, Shell, Status, Table, useApi } from "@/components/ui";
import type { Experiment } from "@/lib/api";
import { when } from "@/lib/format";

export default function Experiments() {
  const { data, error, loading } = useApi<Experiment[]>("/experiments");
  return (
    <Shell>
      <PageHeader title="Experiments" sub="Each row is a reproducible measurement, not a claim." />
      <Loading error={error} loading={loading} />
      {!error && (
        <Card>
          <Table
            head={["#", "Name", "Detector", "Target", "Status", "Created"]}
            empty="No experiments yet."
            rows={(data ?? []).map((e) => [
              <span key="n" className="tabular-nums">{String(e.number).padStart(3, "0")}</span>,
              <Link key="l" href={`/experiments/${e.id}`} className="underline">{e.name}</Link>,
              <span key="d" style={{ color: "var(--ink-secondary)" }}>{e.detector_id}</span>,
              <span key="t" style={{ color: "var(--ink-secondary)" }}>
                {e.configuration?.target_label ?? "—"}
              </span>,
              <Status key="s" value={e.status} />,
              <span key="c" style={{ color: "var(--ink-muted)" }}>{when(e.created_at)}</span>,
            ])}
          />
        </Card>
      )}
    </Shell>
  );
}
