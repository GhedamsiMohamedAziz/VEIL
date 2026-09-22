"use client";

import { useEffect, useState } from "react";
import { apiBase, getApiKey } from "@/lib/api";

/**
 * An <img> cannot send an X-API-Key header, so the bytes are fetched and
 * turned into an object URL. The alternative - a signed query parameter -
 * would put a credential in browser history and server logs.
 */
export function PatternImage({ patternId, className }: { patternId: string; className?: string }) {
  const [url, setUrl] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let objectUrl: string | null = null;
    let cancelled = false;
    // The router reuses this component across /patterns/[id]: start clean, or
    // one missing image leaves every later pattern "unavailable".
    setUrl(null);
    setFailed(false);
    fetch(`${apiBase}/patterns/${patternId}/image`, { headers: { "X-API-Key": getApiKey() } })
      .then((r) => (r.ok ? r.blob() : Promise.reject(new Error(String(r.status)))))
      .then((blob) => {
        if (cancelled) return;
        objectUrl = URL.createObjectURL(blob);
        setUrl(objectUrl);
      })
      .catch(() => { if (!cancelled) setFailed(true); });
    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [patternId]);

  if (failed) return <p className="text-xs" style={{ color: "var(--ink-muted)" }}>image unavailable</p>;
  if (!url) return <div className={className} style={{ background: "var(--surface)" }} />;
  // eslint-disable-next-line @next/next/no-img-element
  return (
    <img
      src={url}
      alt={`Generated pattern ${patternId.slice(0, 8)}`}
      className={className}
      style={{ imageRendering: "pixelated" }}
    />
  );
}
