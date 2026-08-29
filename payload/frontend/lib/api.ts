import type { ReviewLine, ReviewPatch } from "./geometry";

export const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  status: number;
  code: string | null;

  constructor(status: number, code: string | null, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
  }
}

export function parseApiErrorPayload(status: number, text: string) {
  let code: string | null = null;
  let message = text || `Request failed: ${status}`;
  try {
    const parsed = JSON.parse(text) as { detail?: unknown };
    if (typeof parsed.detail === "string") {
      code = parsed.detail;
      message = parsed.detail;
    }
  } catch {
    // Plain-text response: keep the original body.
  }
  return { status, code, message };
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    credentials: "include",
    headers: {
      ...(init?.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
      ...(init?.headers ?? {}),
    },
    cache: "no-store",
  });
  if (!response.ok) {
    const parsed = parseApiErrorPayload(response.status, await response.text());
    throw new ApiError(parsed.status, parsed.code, parsed.message);
  }
  return response.json() as Promise<T>;
}

export type AuthUser = {
  id: string;
  email: string;
  must_change_password: boolean;
};

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
    id: string;
    external_id: string | null;
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    width_px: number;
    width_nm: number | null;
    angle_deg: number;
    confidence: number;
    source: string;
    metadata: Record<string, unknown>;
  }>;
};

export type ReviewMeasurement = ReviewLine & {
  width_px: number;
  width_nm: number | null;
  angle_deg: number;
  edited: boolean;
  source: string;
  confidence: number | null;
};

export type ReviewResponse = {
  id: string;
  analysis_id: string;
  status: "OPEN" | "APPROVED";
  measurements: ReviewMeasurement[];
};

export function login(email: string, password: string) {
  return request<AuthUser>("/api/auth/login", { method: "POST", body: JSON.stringify({ email, password }) });
}

export function getCurrentUser() {
  return request<AuthUser>("/api/auth/me");
}

export function changePassword(newPassword: string) {
  return request<AuthUser>("/api/auth/change-password", { method: "POST", body: JSON.stringify({ new_password: newPassword }) });
}

export function logout() {
  return request<{ status: string }>("/api/auth/logout", { method: "POST" });
}

export async function uploadImage(file: File, nmPerPixel?: number): Promise<ImageAsset> {
  const body = new FormData();
  body.append("file", file);
  if (nmPerPixel !== undefined && Number.isFinite(nmPerPixel)) body.append("nm_per_pixel", String(nmPerPixel));
  return request<ImageAsset>("/api/images", { method: "POST", body });
}

export function createAnalysis(imageId: string) {
  return request<AnalysisJob>("/api/analyses", { method: "POST", body: JSON.stringify({ image_id: imageId }) });
}

export function getAnalysis(id: string) {
  return request<AnalysisJob>(`/api/analyses/${id}`);
}

export function getResult(id: string) {
  return request<AnalysisResult>(`/api/analyses/${id}/result`);
}

export function getReview(id: string) {
  return request<ReviewResponse>(`/api/analyses/${id}/review`);
}

export function saveReview(reviewId: string, patch: ReviewPatch) {
  return request<ReviewResponse>(`/api/reviews/${reviewId}`, { method: "PATCH", body: JSON.stringify(patch) });
}

export function approveReview(reviewId: string) {
  return request<{ review_id: string; status: string; training_examples: number }>(`/api/reviews/${reviewId}/approve`, { method: "POST" });
}

export function absoluteImageUrl(path: string) {
  return path.startsWith("http") ? path : `${API_BASE}${path}`;
}

export function exportUrl(analysisId: string, kind: "csv" | "overlay" | "labeled" | "bundle") {
  return `${API_BASE}/api/analyses/${analysisId}/exports/${kind}`;
}
