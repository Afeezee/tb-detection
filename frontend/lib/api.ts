// Thin fetch wrappers around the FastAPI backend.
// Base URL is picked up from NEXT_PUBLIC_API_BASE_URL at build time.

const RAW_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
export const API_BASE = RAW_BASE.replace(/\/$/, "");

export type ModelName = "densenet121" | "hybrid";

export interface PredictionResponse {
  id: number | null;
  label: "Normal" | "TB-positive";
  label_index: 0 | 1;
  confidence: number;
  probabilities: Record<string, number>;
  model_used: ModelName;
  gradcam_base64: string;
  image_filename: string;
  patient_ref: string | null;
  created_at: string;
}

export interface HistoryRow {
  id: number;
  patient_ref: string | null;
  image_filename: string;
  prediction: string;
  confidence: number;
  model_name: string;
  gradcam_path: string | null;
  clinician_notes: string | null;
  created_at: string;
}

export interface HistoryResponse {
  count: number;
  items: HistoryRow[];
}

export interface BenchmarkRow {
  model: string;
  training_regime: "single-source" | "multi-source";
  test_set: "internal" | "external_tbx11k";
  sensitivity: number;
  specificity: number;
  f1: number;
  auc_roc: number;
  notes: string | null;
}

export interface BenchmarkResponse {
  rows: BenchmarkRow[];
  generated_at: string;
}

async function jsonOrThrow<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API ${res.status}: ${text || res.statusText}`);
  }
  return (await res.json()) as T;
}

export async function postPrediction(params: {
  file: File;
  model: ModelName;
  patientRef?: string;
  clinicianNotes?: string;
  persist?: boolean;
}): Promise<PredictionResponse> {
  const form = new FormData();
  form.append("file", params.file);
  if (params.patientRef) form.append("patient_ref", params.patientRef);
  if (params.clinicianNotes) form.append("clinician_notes", params.clinicianNotes);
  form.append("persist", String(params.persist ?? true));

  const res = await fetch(
    `${API_BASE}/predict?model=${encodeURIComponent(params.model)}`,
    { method: "POST", body: form }
  );
  return jsonOrThrow<PredictionResponse>(res);
}

export async function fetchHistory(limit = 50): Promise<HistoryResponse> {
  const res = await fetch(`${API_BASE}/history?limit=${limit}`, { cache: "no-store" });
  return jsonOrThrow<HistoryResponse>(res);
}

export async function fetchBenchmarks(): Promise<BenchmarkResponse> {
  const res = await fetch(`${API_BASE}/metrics`, { cache: "no-store" });
  return jsonOrThrow<BenchmarkResponse>(res);
}
