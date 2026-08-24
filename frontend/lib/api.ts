import type { ReviewLine, ReviewPatch } from "./geometry";

export const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { ...(init?.body instanceof FormData ? {} : { "Content-Type": "application/json" }), ...(init?.headers ?? {}) },
    cache: "no-store",
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export type ImageAsset = {
  id: string;
  filename: string;
  nm_per_pixel: number | null;
  content_url: string;
  input_mode: "raw_sem" | "visionflux_annotated";
  imported_measurements: number;
};

export type AnalysisJob = {
  id: string;
  image_id: string;
  status: "QUEUED" | "ANALYZING" | "POSTPROCESSING" | "DONE" | "FAILED";
  progress: number;
  error_message: string | null;
  model_version: string;
  summary: Record<string, unknown>;
};

export type AnalysisResult = {
  analysis_id: string;
  image_id: string;
  image_url: string;
  summary: Record<string, unknown>;
  measurements: Array<{
    id: string; external_id: string | null;
    x1: number; y1: number; x2: number; y2: number;
    width_px: number; width_nm: number | null; angle_deg: number;
    confidence: number; source: string; metadata: Record<string, unknown>;
  }>;
};

export type ReviewResponse = {
  id: string;
  analysis_id: string;
  status: "OPEN" | "APPROVED";
  measurements: Array<ReviewLine & { width_px: number; width_nm: number | null; angle_deg: number; edited: boolean; source: string }>;
};

export async function uploadImage(file: File, nmPerPixel?: number): Promise<ImageAsset> {
  const body = new FormData();
  body.append("file", file);
  if (nmPerPixel !== undefined && Number.isFinite(nmPerPixel)) body.append("nm_per_pixel", String(nmPerPixel));
  return request<ImageAsset>("/api/images", { method: "POST", body });
}

export function createAnalysis(imageId: string) {
  return request<AnalysisJob>("/api/analyses", { method: "POST", body: JSON.stringify({ image_id: imageId }) });
}
export function getAnalysis(id: string) { return request<AnalysisJob>(`/api/analyses/${id}`); }
export function getResult(id: string) { return request<AnalysisResult>(`/api/analyses/${id}/result`); }
export function getReview(id: string) { return request<ReviewResponse>(`/api/analyses/${id}/review`); }
export function saveReview(reviewId: string, patch: ReviewPatch) {
  return request<ReviewResponse>(`/api/reviews/${reviewId}`, { method: "PATCH", body: JSON.stringify(patch) });
}
export function approveReview(reviewId: string) {
  return request<{ review_id: string; status: string; training_examples: number }>(`/api/reviews/${reviewId}/approve`, { method: "POST" });
}
export function absoluteImageUrl(path: string) { return path.startsWith("http") ? path : `${API_BASE}${path}`; }
