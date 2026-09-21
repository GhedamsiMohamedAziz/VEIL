"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { ApiError, api } from "@/lib/api";

const NAV = [
  ["/dashboard", "Dashboard"],
  ["/projects", "Projects"],
  ["/experiments", "Experiments"],
  ["/patterns", "Patterns"],
  ["/detectors", "Detectors"],
  ["/physical-tests", "Physical tests"],
  ["/reports", "Reports"],
  ["/settings", "Settings"],
] as const;

export function Shell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  return (
    <div className="flex min-h-screen flex-col md:flex-row">
      <nav
        className="shrink-0 border-b md:min-h-screen md:w-56 md:border-r md:border-b-0"
        style={{ borderColor: "var(--border)" }}
      >
        <Link href="/" className="block px-5 py-6 text-lg tracking-[0.3em]">
          VEIL
        </Link>
        <ul className="flex flex-wrap gap-1 px-3 pb-4 md:block">
          {NAV.map(([href, label]) => {
            const active = pathname === href || pathname.startsWith(`${href}/`);
            return (
              <li key={href}>
                <Link
                  href={href}
                  className="block rounded px-3 py-2 text-sm"
                  style={{
                    background: active ? "var(--surface-raised)" : "transparent",
                    color: active ? "var(--ink)" : "var(--ink-secondary)",
                  }}
                >
                  {label}
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>
      <main className="min-w-0 flex-1 px-5 py-8 md:px-10">{children}</main>
    </div>
  );
}

export function PageHeader({ title, sub, actions }: { title: string; sub?: string; actions?: React.ReactNode }) {
  return (
    <header className="mb-8 flex flex-wrap items-start justify-between gap-4">
      <div>
        <h1 className="text-2xl">{title}</h1>
        {sub && <p className="mt-1 text-sm" style={{ color: "var(--ink-secondary)" }}>{sub}</p>}
      </div>
      {actions}
    </header>
  );
}

export function Card({ title, children }: { title?: string; children: React.ReactNode }) {
  return (
    <section
      className="rounded-lg border p-5"
      style={{ borderColor: "var(--border)", background: "var(--surface-raised)" }}
    >
      {title && (
        <h2 className="mb-4 text-xs uppercase tracking-[0.18em]" style={{ color: "var(--ink-muted)" }}>
          {title}
        </h2>
      )}
      {children}
    </section>
  );
}

/** A single headline number. Not a chart - one value has no shape to plot. */
export function Stat({ label, value, note }: { label: string; value: string; note?: string }) {
  return (
    <div
      className="rounded-lg border p-4"
      style={{ borderColor: "var(--border)", background: "var(--surface-raised)" }}
    >
      <div className="text-xs uppercase tracking-[0.15em]" style={{ color: "var(--ink-muted)" }}>{label}</div>
      <div className="mt-2 text-2xl tabular-nums">{value}</div>
      {note && <div className="mt-1 text-xs" style={{ color: "var(--ink-secondary)" }}>{note}</div>}
    </div>
  );
}

export function Status({ value }: { value: string }) {
  const color =
    value === "completed" ? "var(--good)" :
    value === "failed" ? "var(--series-candidate)" :
    value === "running" || value === "queued" ? "var(--warning)" : "var(--ink-muted)";
  return (
    <span className="inline-flex items-center gap-2 text-xs uppercase tracking-wider">
      <span aria-hidden className="inline-block h-2 w-2 rounded-full" style={{ background: color }} />
      {value}
    </span>
  );
}

export function Table({ head, rows, empty }: { head: string[]; rows: React.ReactNode[][]; empty: string }) {
  if (rows.length === 0) {
    return <p className="text-sm" style={{ color: "var(--ink-secondary)" }}>{empty}</p>;
  }
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[32rem] border-collapse text-sm">
        <thead>
          <tr style={{ color: "var(--ink-muted)" }}>
            {head.map((h) => (
              <th key={h} className="border-b px-2 py-2 text-left font-normal text-xs uppercase tracking-wider"
                  style={{ borderColor: "var(--border)" }}>
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>
              {row.map((cell, j) => (
                <td key={j} className="border-b px-2 py-3 align-top" style={{ borderColor: "var(--border)" }}>
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Data loading with the three states that actually happen. */
export function useApi<T>(path: string | null, deps: unknown[] = []) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!path) return;
    let live = true;
    setLoading(true);
    api<T>(path)
      .then((value) => live && (setData(value), setError(null)))
      .catch((e: ApiError) => live && setError(e.message))
      .finally(() => live && setLoading(false));
    return () => { live = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [path, ...deps]);

  return { data, error, loading, reload: () => setData(null) };
}

export function Loading({ error, loading }: { error: string | null; loading: boolean }) {
  if (error) {
    return (
      <p className="text-sm" style={{ color: "var(--series-candidate)" }}>
        {error}{" "}
        {error.includes("API key") && <Link href="/settings" className="underline">Set one →</Link>}
      </p>
    );
  }
  if (loading) return <p className="text-sm" style={{ color: "var(--ink-muted)" }}>loading…</p>;
  return null;
}
