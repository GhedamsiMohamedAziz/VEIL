"use client";

import Link from "next/link";
import { use, useEffect, useRef, useState } from "react";
import { AttributionBar, ComparisonBars, ScoreMeter, type Row } from "@/components/chart";
import { PatternImage } from "@/components/pattern-image";
import { Card, Loading, PageHeader, Shell, Stat, Status, Table, useApi } from "@/components/ui";
import { api, type Evaluation, type Experiment, type Pattern, type PhysicalTest, type Run } from "@/lib/api";
import { num, pct, shortId, when } from "@/lib/format";

const AXES = ["angle", "lighting", "scale", "camera_noise", "deformation"] as const;

export default function ExperimentDetail({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const experiment = useApi<Experiment>(`/experiments/${id}`);
  const runs = useApi<Run[]>(`/experiments/${id}/runs`);
  const results = useApi<Evaluation[]>(`/experiments/${id}/results`);
  const patterns = useApi<Pattern[]>(`/experiments/${id}/patterns`);
  const physical = useApi<PhysicalTest[]>(`/physical-tests?experiment_id=${id}`);
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; };  // stops run() polling once the page is left
  }, []);

  const evaluation = results.data?.[0];
  const score = evaluation?.metrics?.veil_score ?? {};
  const perAxis = evaluation?.metrics?.comparison?.per_axis ?? {};
  const byRotation = evaluation?.metrics?.comparison?.by_rotation;
  const attribution = evaluation?.metrics?.comparison?.attribution;
  const transfer: Record<string, any> = evaluation?.metrics?.transfer ?? {};
  const latestRun = runs.data?.[0];
  const pattern = patterns.data?.[0];

  async function run() {
    setBusy(true);
    setProblem(null);
    try {
      const started = await api<Run>(`/experiments/${id}/run`, { method: "POST", body: JSON.stringify({ seed: 42 }) });
      runs.reload();
      // A run takes minutes: follow it until it ends instead of guessing a delay.
      let current = started;
      const deadline = Date.now() + 60 * 60 * 1000;
      while (current.status === "queued" || current.status === "running") {
        await new Promise((resolve) => setTimeout(resolve, 2000));
        if (!mounted.current) return;
        if (Date.now() > deadline) {
          setProblem("still running after an hour; reload this page to check on it");
          break;
        }
        current = await api<Run>(`/runs/${started.id}`);
      }
      if (current.status === "failed") setProblem(current.error || "run failed");
      for (const hook of [experiment, runs, results, patterns]) hook.reload();
    } catch (e) {
      setProblem((e as Error).message);
    } finally {
      if (mounted.current) setBusy(false);
    }
  }

  async function report() {
    setBusy(true);
    setProblem(null);
    try {
      await api(`/experiments/${id}/report`, { method: "POST" });
      location.href = "/reports";
    } catch (e) {
      setProblem((e as Error).message);
      setBusy(false);
    }
  }

  const axisRows: Row[] = AXES.map((axis) => ({
    label: axis.replace("_", " "),
    baseline: perAxis[axis]?.baseline_detection_rate ?? null,
    control: perAxis[axis]?.control_detection_rate ?? null,
    candidate: perAxis[axis]?.candidate_detection_rate ?? null,
    samples: perAxis[axis]?.samples,
  })).filter((r) => r.samples);

  const rotationRows: Row[] = (byRotation?.candidate ?? []).map((row: any, i: number) => ({
    label: `${row.value}°`,
    baseline: byRotation.baseline?.[i]?.detection_rate ?? null,
    control: byRotation.control?.[i]?.detection_rate ?? null,
    candidate: row.detection_rate,
    samples: row.samples,
  }));

  return (
    <Shell>
      <PageHeader
        title={experiment.data ? `Experiment #${String(experiment.data.number).padStart(3, "0")}` : "Experiment"}
        sub={experiment.data?.name}
        actions={
          <div className="flex gap-2">
            <button onClick={run} disabled={busy} className="rounded border px-4 py-2 text-sm"
                    style={{ borderColor: "var(--border)" }}>
              {busy ? "…" : "Run"}
            </button>
            <button onClick={report} disabled={busy || !evaluation} className="rounded border px-4 py-2 text-sm"
                    style={{ borderColor: "var(--border)", opacity: evaluation ? 1 : 0.4 }}>
              Generate report
            </button>
          </div>
        }
      />
      <Loading error={experiment.error} loading={experiment.loading} />
      {problem && <p className="mb-4 text-sm" style={{ color: "var(--series-candidate)" }}>{problem}</p>}

      {experiment.data && (
        <div className="space-y-6">
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5">
            <Stat label="Status" value={experiment.data.status} />
            <Stat label="Detector" value={experiment.data.detector_id}
                  note={`target: ${experiment.data.configuration?.target_label ?? "—"}`} />
            <Stat label="Baseline detection" value={pct(evaluation?.baseline_metrics?.detection_rate)}
                  note={evaluation ? `n=${evaluation.baseline_metrics.samples}` : "no completed run"} />
            <Stat label="Control detection"
                  value={pct(evaluation?.control_metrics?.detection_rate)}
                  note="unoptimized pattern, same size" />
            <Stat label="Candidate detection" value={pct(evaluation?.candidate_metrics?.detection_rate)}
                  note={evaluation ? `n=${evaluation.candidate_metrics.samples}` : undefined} />
          </div>

          {evaluation && (
            <>
              <Card title="Effect attribution">
                <div className="grid gap-6 md:grid-cols-2">
                  <AttributionBar attribution={attribution}
                                  candidateRate={evaluation?.candidate_metrics?.detection_rate} />
                  <p className="text-xs leading-relaxed" style={{ color: "var(--ink-muted)" }}>
                    A patch covering part of the subject reduces detection on its own. The
                    control arm is a set of <em>unoptimized</em> patterns of identical size,
                    palette and placement, so the difference between control and candidate is
                    the only part of the drop attributable to the pattern. Whether that
                    difference is real is a separate question, answered by a paired test
                    against every draw. Read both before the score below.
                  </p>
                </div>
              </Card>

              <Card title="VEIL score (experimental)">
                <div className="grid gap-5 md:grid-cols-2">
                  <div className="space-y-4">
                    <ScoreMeter label="Digital robustness" value={score.digital_robustness}
                                samples={score.sample_counts?.digital} />
                    <ScoreMeter label="Angle robustness" value={score.angle_robustness}
                                samples={score.sample_counts?.axis_angle} />
                    <ScoreMeter label="Lighting robustness" value={score.lighting_robustness}
                                samples={score.sample_counts?.axis_lighting} />
                    <ScoreMeter label="Physical robustness" value={score.physical_robustness}
                                samples={score.sample_counts?.physical} />
                  </div>
                  <p className="text-xs leading-relaxed" style={{ color: "var(--ink-muted)" }}>
                    {score.definition}. A component reads &quot;not measured&quot; — never 0% — when no
                    sample exercised that axis. These are experimental measurements for{" "}
                    <span style={{ color: "var(--ink-secondary)" }}>{experiment.data.detector_id}</span>{" "}
                    under the transformation grid below, on the dataset version pinned by the run.
                    They are not guarantees and are not expected to transfer to other models,
                    cameras or environments.
                  </p>
                </div>
              </Card>

              <Card title="Detection rate by transformation axis">
                <ComparisonBars rows={axisRows}
                  caption="Lower bars mean the detector found the target less often. Compare candidate against control, not against baseline." />
              </Card>

              {Object.keys(transfer).length > 0 && (
                <Card title="Transfer to detectors not optimized against">
                  <Table
                    head={["Detector", "Baseline", "Control", "Candidate", "Verdict", "p across detectors (Holm)"]}
                    empty="No transfer detectors configured."
                    rows={Object.entries(transfer).map(([id, block]) => [
                      <span key="d">{id}</span>,
                      <span key="b" className="tabular-nums">{pct(block?.baseline?.detection_rate)}</span>,
                      <span key="c" className="tabular-nums">{pct(block?.control?.detection_rate)}</span>,
                      <span key="x" className="tabular-nums">{pct(block?.candidate?.detection_rate)}</span>,
                      <span key="v" style={{
                        color: block?.attribution?.has_headroom === false
                          ? "var(--warning)" : "var(--ink-secondary)",
                      }}>
                        {block?.available
                          ? (block?.attribution?.verdict ?? block?.attribution?.reason)
                          : `not measured — ${block?.reason}`}
                      </span>,
                      // Each detector is one more chance of a spurious "significant":
                      // this is the p-value corrected for how many were tested.
                      <span key="h" className="tabular-nums">
                        {block?.family
                          ? `${num(block.family.holm_p_value, 3)} — ${block.family.significant_after_correction ? "significant" : "not significant"}`
                          : "—"}
                      </span>,
                    ])}
                  />
                  <p className="mt-4 text-xs leading-relaxed" style={{ color: "var(--ink-muted)" }}>
                    The same pattern and placement, measured against models it was never
                    optimized against, with the same three-arm discipline. A detector whose
                    baseline is near zero never saw the target to begin with, so no transfer
                    conclusion can be drawn from it — the verdict says so rather than reporting
                    a number. These results describe only the detectors listed.
                  </p>
                </Card>
              )}

              {rotationRows.length > 0 && (
                <Card title="Detection rate by rotation">
                  <ComparisonBars rows={rotationRows} />
                </Card>
              )}
            </>
          )}

          <div className="grid gap-6 lg:grid-cols-2">
            <Card title="Pattern">
              {pattern ? (
                <div className="space-y-3">
                  <PatternImage patternId={pattern.id} className="w-40 rounded border" />
                  <dl className="text-xs" style={{ color: "var(--ink-secondary)" }}>
                    <div>strategy: {pattern.metrics?.strategy}</div>
                    <div>iterations: {pattern.metrics?.iterations}</div>
                    <div>loss improvement: {num(pattern.metrics?.improvement, 5)}</div>
                  </dl>
                  {pattern.metrics?.improvement === 0 && (
                    <p className="text-xs" style={{ color: "var(--warning)" }}>
                      The optimizer made no progress — any measured drop is attributable to
                      occlusion, not to the pattern. See docs/experiments.md.
                    </p>
                  )}
                  <Link href={`/patterns/${pattern.id}`} className="block text-xs underline">Pattern detail →</Link>
                </div>
              ) : (
                <p className="text-sm" style={{ color: "var(--ink-secondary)" }}>No pattern generated yet.</p>
              )}
            </Card>

            <Card title="Transformations">
              <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs" style={{ color: "var(--ink-secondary)" }}>
                {Object.entries(evaluation?.transformations ?? experiment.data.configuration?.transforms ?? {}).map(
                  ([key, value]) => (
                    <div key={key} className="contents">
                      <dt>{key}</dt>
                      <dd className="tabular-nums">{JSON.stringify(value)}</dd>
                    </div>
                  ),
                )}
              </dl>
            </Card>
          </div>

          <Card title="Runs">
            <Table
              head={["Run", "Status", "Seed", "Code", "Detector version", "Finished"]}
              empty="No runs yet."
              rows={(runs.data ?? []).map((r) => [
                <code key="i">{shortId(r.id)}</code>,
                <Status key="s" value={r.status} />,
                <span key="d" className="tabular-nums">{r.seed}</span>,
                <code key="c" style={{ color: "var(--ink-muted)" }}>{r.code_version || "—"}</code>,
                <span key="v" style={{ color: "var(--ink-muted)" }}>{r.detector_version || "—"}</span>,
                <span key="f" style={{ color: "var(--ink-muted)" }}>{when(r.finished_at)}</span>,
              ])}
            />
            {latestRun?.error && (
              <p className="mt-3 text-sm" style={{ color: "var(--series-candidate)" }}>{latestRun.error}</p>
            )}
          </Card>

          <Card title="Physical tests">
            <Table
              head={["Arm", "Camera", "Distance", "Angle", "Lighting", "Frames", "Detected"]}
              empty="No physical tests recorded. Physical robustness stays 'not measured' until one is."
              rows={(physical.data ?? []).map((t) => [
                <strong key="arm">{t.arm || "unspecified"}</strong>,
                t.camera,
                <span key="d" className="tabular-nums">{t.distance_m ?? "—"} m</span>,
                <span key="a" className="tabular-nums">{t.angle_deg ?? "—"}°</span>,
                t.lighting,
                <span key="f" className="tabular-nums">{t.frame_count}</span>,
                <span key="r">{pct(t.result?.detection_rate)}</span>,
              ])}
            />
          </Card>

          <Card title="Log">
            <pre className="overflow-x-auto text-xs leading-relaxed" style={{ color: "var(--ink-secondary)" }}>
              {(latestRun?.log ?? []).join("\n") || "no log"}
            </pre>
          </Card>
        </div>
      )}
    </Shell>
  );
}
