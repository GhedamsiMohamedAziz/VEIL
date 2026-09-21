"use client";

import Link from "next/link";
import { Card, Loading, PageHeader, Shell, Stat, Status, Table, useApi } from "@/components/ui";
import type { Detector, Experiment, PhysicalTest, Project } from "@/lib/api";
import { shortId, when } from "@/lib/format";

export default function Dashboard() {
  const projects = useApi<Project[]>("/projects");
  const experiments = useApi<Experiment[]>("/experiments");
  const physical = useApi<PhysicalTest[]>("/physical-tests");
  const detectors = useApi<Detector[]>("/detectors");

  const error = projects.error ?? experiments.error;
  const completed = (experiments.data ?? []).filter((e) => e.status === "completed").length;

  return (
    <Shell>
      <PageHeader title="Dashboard" sub="Experimental measurements under recorded conditions." />
      <Loading error={error} loading={projects.loading} />

      {!error && (
        <>
          <div className="mb-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <Stat label="Projects" value={String(projects.data?.length ?? 0)} />
            <Stat label="Experiments" value={String(experiments.data?.length ?? 0)} note={`${completed} completed`} />
            <Stat label="Physical tests" value={String(physical.data?.length ?? 0)} note="real camera measurements" />
            <Stat label="Detectors" value={String(detectors.data?.length ?? 0)} note="registered in code" />
          </div>

          <Card title="Recent experiments">
            <Table
              head={["#", "Name", "Detector", "Status", "Created"]}
              empty="No experiments yet. Create a project, upload authorized imagery, then define an experiment."
              rows={(experiments.data ?? []).slice(0, 10).map((e) => [
                <span key="n" className="tabular-nums">{String(e.number).padStart(3, "0")}</span>,
                <Link key="l" href={`/experiments/${e.id}`} className="underline">{e.name}</Link>,
                <span key="d" style={{ color: "var(--ink-secondary)" }}>{e.detector_id}</span>,
                <Status key="s" value={e.status} />,
                <span key="c" style={{ color: "var(--ink-muted)" }}>{when(e.created_at)}</span>,
              ])}
            />
          </Card>

          <div className="mt-6">
            <Card title="Projects">
              <Table
                head={["Name", "ID", "Created"]}
                empty="No projects."
                rows={(projects.data ?? []).map((p) => [
                  <Link key="l" href={`/projects/${p.id}`} className="underline">{p.name}</Link>,
                  <code key="i" style={{ color: "var(--ink-muted)" }}>{shortId(p.id)}</code>,
                  <span key="c" style={{ color: "var(--ink-muted)" }}>{when(p.created_at)}</span>,
                ])}
              />
            </Card>
          </div>
        </>
      )}
    </Shell>
  );
}
