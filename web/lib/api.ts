"use client";

/**
 * Thin fetch wrapper. The API key lives in localStorage and is set on
 * /settings - the dashboard is a client of the API like any other, with no
 * session of its own and no server-side secret.
 */
const BASE = process.env.NEXT_PUBLIC_VEIL_API ?? "http://localhost:8000/api/v1";

export const KEY_STORAGE = "veil.apiKey";

export function getApiKey(): string {
  if (typeof window === "undefined") return "";
  return window.localStorage.getItem(KEY_STORAGE) ?? "";
}

export function setApiKey(key: string) {
  window.localStorage.setItem(KEY_STORAGE, key.trim());
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const key = getApiKey();
  if (!key) throw new ApiError(401, "No API key set. Add one on Settings.");
  const response = await fetch(`${BASE}${path}`, {
    ...init,
    headers: {
      "X-API-Key": key,
      ...(init?.body ? { "content-type": "application/json" } : {}),
      ...init?.headers,
    },
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}) as any);
    throw new ApiError(response.status, detail.detail ?? `${response.status} ${response.statusText}`);
  }
  return response.status === 204 ? (undefined as T) : ((await response.json()) as T);
}

export const apiBase = BASE;

// --- shapes the dashboard reads (a subset of the OpenAPI schema) ---

export type Project = { id: string; name: string; description: string; created_at: string };
export type Dataset = { id: string; project_id: string; name: string; version: number; artifact_ids: string[] };
export type Experiment = {
  id: string; project_id: string; number: number; name: string; description: string;
  status: string; detector_id: string; dataset_id: string;
  configuration: Record<string, any>; created_at: string; completed_at: string | null;
};
export type Run = {
  id: string; experiment_id: string; status: string; seed: number; code_version: string;
  detector_version: string; dataset_version: number; error: string; log: string[];
  created_at: string; started_at: string | null; finished_at: string | null;
};
export type Metrics = {
  samples: number; detections?: number; detection_rate: number | null;
  detection_rate_ci95?: [number, number]; mean_confidence: number | null;
  max_confidence?: number;
};
export type Evaluation = {
  id: string; experiment_id: string; run_id: string | null; pattern_id: string | null;
  baseline_metrics: Metrics; control_metrics: Metrics; candidate_metrics: Metrics;
  transformations: Record<string, any>; sample_count: number; metrics: Record<string, any>;
  created_at: string;
};
export type Pattern = {
  id: string; experiment_id: string; run_id: string | null; version: number;
  artifact_id: string | null; generation_parameters: Record<string, any>;
  metrics: Record<string, any>; created_at: string;
};
export type Detector = {
  id: string; name: string; version: string; labels: string[]; label_count: number;
  differentiable: boolean; license: string; source: string; notes: string;
};
export type PhysicalTest = {
  id: string; experiment_id: string; pattern_id: string | null;
  arm: string; // baseline | control | candidate | unspecified - what was on the subject
  camera: string; resolution: string; distance_m: number | null; angle_deg: number | null;
  lighting: string; environment: string; frame_count: number;
  result: Record<string, any>; created_at: string;
};
export type Report = {
  id: string; experiment_id: string; run_id: string | null;
  payload: Record<string, any>; pdf_artifact_id: string | null; created_at: string;
};
