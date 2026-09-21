"use client";

import Link from "next/link";
import { Card, Loading, PageHeader, Shell, Table, useApi } from "@/components/ui";
import type { Pattern } from "@/lib/api";
import { num, shortId, when } from "@/lib/format";

export default function Patterns() {
  const { data, error, loading } = useApi<Pattern[]>("/patterns");
  return (
    <Shell>
      <PageHeader title="Patterns" sub="Candidate prints, quantized to the printable ink palette on export." />
      <Loading error={error} loading={loading} />
      {!error && (
        <Card>
          <Table
            head={["ID", "Version", "Strategy", "Iterations", "Loss improvement", "Created"]}
            empty="No patterns yet. A pattern is produced by a completed experiment run."
            rows={(data ?? []).map((p) => [
              <Link key="l" href={`/patterns/${p.id}`} className="underline">{shortId(p.id)}</Link>,
              <span key="v" className="tabular-nums">v{p.version}</span>,
              <span key="s" style={{ color: "var(--ink-secondary)" }}>{p.metrics?.strategy ?? "—"}</span>,
              <span key="i" className="tabular-nums">{p.metrics?.iterations ?? "—"}</span>,
              <span key="d" className="tabular-nums">{num(p.metrics?.improvement, 5)}</span>,
              <span key="c" style={{ color: "var(--ink-muted)" }}>{when(p.created_at)}</span>,
            ])}
          />
        </Card>
      )}
    </Shell>
  );
}
