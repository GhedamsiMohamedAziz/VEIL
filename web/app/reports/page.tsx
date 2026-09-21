"use client";

import Link from "next/link";
import { useState } from "react";
import { Card, Loading, PageHeader, Shell, Table, useApi } from "@/components/ui";
import { apiBase, getApiKey, type Report } from "@/lib/api";
import { pct, shortId, when } from "@/lib/format";

export default function Reports() {
  const { data, error, loading } = useApi<Report[]>("/reports");
  const [open, setOpen] = useState<Report | null>(null);
  const [problem, setProblem] = useState<string | null>(null);

  async function download(report: Report) {
    setProblem(null);
    try {
      const response = await fetch(`${apiBase}/reports/${report.id}/pdf`, {
        headers: { "X-API-Key": getApiKey() },
      });
      // Without this an error body opens in a new tab dressed up as a PDF.
      if (!response.ok) throw new Error(`PDF unavailable (${response.status})`);
      const url = URL.createObjectURL(await response.blob());
      window.open(url, "_blank");
      setTimeout(() => URL.revokeObjectURL(url), 30_000);
    } catch (e) {
      setProblem((e as Error).message);
    }
  }

  return (
    <Shell>
      <PageHeader title="Reports" sub="Every report carries its conditions and a limitations section." />
      <Loading error={error} loading={loading} />
      {problem && <p className="mb-4 text-sm" style={{ color: "var(--series-candidate)" }}>{problem}</p>}
      {!error && (
        <div className="space-y-6">
          <Card>
            <Table
              head={["Report", "Experiment", "Baseline", "Candidate", "Generated", ""]}
              empty="No reports yet. Generate one from a completed experiment."
              rows={(data ?? []).map((r) => [
                <button key="b" onClick={() => setOpen(r)} className="underline">{shortId(r.id)}</button>,
                <Link key="e" href={`/experiments/${r.experiment_id}`} className="underline">
                  {r.payload?.experiment?.name ?? shortId(r.experiment_id)}
                </Link>,
                <span key="x" className="tabular-nums">{pct(r.payload?.results?.baseline?.detection_rate)}</span>,
                <span key="y" className="tabular-nums">{pct(r.payload?.results?.candidate?.detection_rate)}</span>,
                <span key="w" style={{ color: "var(--ink-muted)" }}>{when(r.created_at)}</span>,
                <button key="p" onClick={() => download(r)} className="underline">PDF</button>,
              ])}
            />
          </Card>

          {open && (
            <Card title={open.payload?.title ?? "Report"}>
              <section className="mb-5">
                <h3 className="mb-2 text-xs uppercase tracking-[0.18em]" style={{ color: "var(--ink-muted)" }}>
                  Limitations
                </h3>
                <ul className="space-y-2 text-xs leading-relaxed" style={{ color: "var(--ink-secondary)" }}>
                  {(open.payload?.limitations ?? []).map((line: string) => <li key={line}>— {line}</li>)}
                </ul>
              </section>
              <details>
                <summary className="cursor-pointer text-xs" style={{ color: "var(--ink-muted)" }}>
                  Full JSON payload
                </summary>
                <pre className="mt-3 max-h-96 overflow-auto text-xs" style={{ color: "var(--ink-secondary)" }}>
                  {JSON.stringify(open.payload, null, 2)}
                </pre>
              </details>
            </Card>
          )}
        </div>
      )}
    </Shell>
  );
}
