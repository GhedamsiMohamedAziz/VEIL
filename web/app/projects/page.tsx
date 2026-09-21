"use client";

import Link from "next/link";
import { useState } from "react";
import { Card, Loading, PageHeader, Shell, Table, useApi } from "@/components/ui";
import { api, type Project } from "@/lib/api";
import { shortId, when } from "@/lib/format";

export default function Projects() {
  const { data, error, loading } = useApi<Project[]>("/projects");
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);

  async function create() {
    if (!name.trim()) return;
    setBusy(true);
    try {
      await api("/projects", { method: "POST", body: JSON.stringify({ name, description: "" }) });
      location.reload();
    } catch (e) {
      setProblem((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <Shell>
      <PageHeader title="Projects" sub="A project groups datasets, experiments and physical tests." />
      <Loading error={error} loading={loading} />
      {!error && (
        <div className="space-y-6">
          <Card title="New project">
            <div className="flex flex-wrap gap-3">
              <input
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Project name"
                className="min-w-0 flex-1 rounded border px-3 py-2 text-sm"
                style={{ borderColor: "var(--border)", background: "var(--surface)", color: "var(--ink)" }}
              />
              <button onClick={create} disabled={busy} className="rounded border px-4 py-2 text-sm"
                      style={{ borderColor: "var(--border)" }}>
                {busy ? "creating…" : "Create"}
              </button>
            </div>
            {problem && <p className="mt-3 text-sm" style={{ color: "var(--series-candidate)" }}>{problem}</p>}
          </Card>

          <Card>
            <Table
              head={["Name", "ID", "Description", "Created"]}
              empty="No projects yet."
              rows={(data ?? []).map((p) => [
                <Link key="l" href={`/projects/${p.id}`} className="underline">{p.name}</Link>,
                <code key="i" style={{ color: "var(--ink-muted)" }}>{shortId(p.id)}</code>,
                <span key="d" style={{ color: "var(--ink-secondary)" }}>{p.description || "—"}</span>,
                <span key="c" style={{ color: "var(--ink-muted)" }}>{when(p.created_at)}</span>,
              ])}
            />
          </Card>
        </div>
      )}
    </Shell>
  );
}
