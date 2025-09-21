import { fallbackAnalysis, fallbackDashboard, fallbackPatientDetail, fallbackPatients } from "./mock-data";
import type { AnalysisDetail, DashboardOverview, ImageUploadResponse, PatientDetail, PatientListResponse } from "./types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

async function fetchJson<T>(path: string, fallback: T): Promise<T> {
  try {
    const response = await fetch(`${API_BASE_URL}${path}`, {
      cache: "no-store",
      headers: {
        "Content-Type": "application/json"
      }
    });

    if (!response.ok) {
      throw new Error(`Request failed: ${response.status}`);
    }

    return (await response.json()) as T;
  } catch (error) {
    console.warn(`Falling back to mock data for ${path}`, error);
    return fallback;
  }
}

export async function fetchDashboardOverview(): Promise<DashboardOverview> {
  return fetchJson<DashboardOverview>("/api/dashboard/overview", fallbackDashboard);
}

export async function fetchPatients(search?: string): Promise<PatientListResponse> {
  const query = search ? `?search=${encodeURIComponent(search)}` : "";
  return fetchJson<PatientListResponse>(`/api/patients${query}`, fallbackPatients);
}

export async function fetchPatientDetail(patientId: string): Promise<PatientDetail> {
  return fetchJson<PatientDetail>(`/api/patients/${patientId}`, fallbackPatientDetail);
}

export async function fetchAnalysisDetail(analysisId: string): Promise<AnalysisDetail> {
  return fetchJson<AnalysisDetail>(`/api/analyses/${analysisId}`, fallbackAnalysis);
}

export async function uploadImage(formData: FormData): Promise<ImageUploadResponse> {
  const response = await fetch(`${API_BASE_URL}/api/uploads/images`, {
    method: "POST",
    body: formData,
    cache: "no-store"
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Upload failed: ${response.status}`);
  }

  return (await response.json()) as ImageUploadResponse;
}
