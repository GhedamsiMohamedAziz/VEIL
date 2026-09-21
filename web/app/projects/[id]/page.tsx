"use client";

import Link from "next/link";
import { use } from "react";
import { Card, Loading, PageHeader, Shell, Status, Table, useApi } from "@/components/ui";
import type { Dataset, Experiment, Project } from "@/lib/api";
import { shortId, when } from "@/lib/format";

export default function ProjectDetail({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const project = useApi<Project>(`/projects/${id}`);
  const datasets = useApi<Dataset[]>(`/datasets?project_id=${id}`);
  const experiments = useApi<Experiment[]>(`/experiments?project_id=${id}`);

  return (
    <Shell>
      <PageHeader title={project.data?.name ?? "Project"} sub={project.data?.description || undefined} />
      <Loading error={project.error} loading={project.loading} />
      {project.data && (
        <div className="space-y-6">
          <Card title="Datasets">
            <Table
              head={["Name", "Version", "Items", "ID"]}
              empty="No datasets. Upload artifacts via the API, then group them into a dataset."
              rows={(datasets.data ?? []).map((d) => [
                d.name,
                <span key="v" className="tabular-nums">v{d.version}</span>,
                <span key="n" className="tabular-nums">{d.artifact_ids.length}</span>,
                <code key="i" style={{ color: "var(--ink-muted)" }}>{shortId(d.id)}</code>,
              ])}
            />
          </Card>

          <Card title="Experiments">
            <Table
              head={["#", "Name", "Detector", "Status", "Completed"]}
              empty="No experiments in this project."
              rows={(experiments.data ?? []).map((e) => [
                <span key="n" className="tabular-nums">{String(e.number).padStart(3, "0")}</span>,
                <Link key="l" href={`/experiments/${e.id}`} className="underline">{e.name}</Link>,
                <span key="d" style={{ color: "var(--ink-secondary)" }}>{e.detector_id}</span>,
                <Status key="s" value={e.status} />,
                <span key="c" style={{ color: "var(--ink-muted)" }}>{when(e.completed_at)}</span>,
              ])}
            />
          </Card>
        </div>
      )}
    </Shell>
  );
}
