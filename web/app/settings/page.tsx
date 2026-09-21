"use client";

import { useEffect, useState } from "react";
import { Card, PageHeader, Shell } from "@/components/ui";
import { apiBase, getApiKey, setApiKey } from "@/lib/api";

export default function Settings() {
  const [key, setKey] = useState("");
  const [saved, setSaved] = useState(false);

  useEffect(() => setKey(getApiKey()), []);

  return (
    <Shell>
      <PageHeader title="Settings" sub="The dashboard talks to the VEIL API as a normal client." />
      <div className="max-w-xl space-y-6">
        <Card title="API key">
          <p className="mb-4 text-sm" style={{ color: "var(--ink-secondary)" }}>
            Create one with <code>veil create-org &quot;Name&quot; mohamed-aziz.ghedamsi@epitech.eu</code>. It is stored in this
            browser&apos;s localStorage only and is sent as <code>X-API-Key</code>.
          </p>
          <input
            type="password"
            value={key}
            onChange={(e) => { setKey(e.target.value); setSaved(false); }}
            placeholder="veil_…"
            className="w-full rounded border px-3 py-2 text-sm"
            style={{ borderColor: "var(--border)", background: "var(--surface)", color: "var(--ink)" }}
          />
          <button
            onClick={() => { setApiKey(key); setSaved(true); }}
            className="mt-3 rounded border px-4 py-2 text-sm"
            style={{ borderColor: "var(--border)" }}
          >
            Save
          </button>
          {saved && <span className="ml-3 text-sm" style={{ color: "var(--good)" }}>saved</span>}
        </Card>

        <Card title="API endpoint">
          <code className="text-sm" style={{ color: "var(--ink-secondary)" }}>{apiBase}</code>
          <p className="mt-3 text-xs" style={{ color: "var(--ink-muted)" }}>
            Set <code>NEXT_PUBLIC_VEIL_API</code> at build time to point elsewhere.
          </p>
        </Card>
      </div>
    </Shell>
  );
}
