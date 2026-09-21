/** Formatting rules. "not measured" is never rendered as 0. */

export function pct(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined) return "not measured";
  return `${(value * 100).toFixed(digits)}%`;
}

export function num(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined) return "—";
  return value.toFixed(digits);
}

export function when(iso: string | null | undefined): string {
  if (!iso) return "—";
  return new Date(iso).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

export function shortId(id: string): string {
  return id.slice(0, 8);
}
