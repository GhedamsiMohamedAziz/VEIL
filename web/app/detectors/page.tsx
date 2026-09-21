"use client";

import { Card, Loading, PageHeader, Shell, Table, useApi } from "@/components/ui";
import type { Detector } from "@/lib/api";

export default function Detectors() {
  const { data, error, loading } = useApi<Detector[]>("/detectors");
  return (
    <Shell>
      <PageHeader
        title="Detectors"
        sub="The models VEIL is authorized to test against. This list is fixed in code."
      />
      <Loading error={error} loading={loading} />
      {!error && (
        <div className="space-y-6">
          <Card>
            <Table
              head={["ID", "Version", "Labels", "Gradients", "Licence"]}
              empty="No detectors registered."
              rows={(data ?? []).map((d) => [
                <span key="i">{d.id}</span>,
                <span key="v" style={{ color: "var(--ink-secondary)" }}>{d.version}</span>,
                <span key="l" className="tabular-nums">{d.label_count}</span>,
                <span key="g" style={{ color: "var(--ink-secondary)" }}>{d.differentiable ? "yes" : "no"}</span>,
                <span key="c" style={{ color: "var(--ink-muted)" }}>{d.license}</span>,
              ])}
            />
          </Card>
          <Card title="Why there is no 'add detector' button">
            <p className="text-sm leading-relaxed" style={{ color: "var(--ink-secondary)" }}>
              Loading a user-supplied checkpoint means deserializing untrusted code. VEIL keeps the
              detector registry in source, so adding a model is a reviewed change with a checked
              licence — not an upload. Customer models are a future feature requiring ONNX or a
              sandboxed process.
            </p>
          </Card>
        </div>
      )}
    </Shell>
  );
}
