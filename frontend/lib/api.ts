import { fallbackAnalysis, fallbackDashboard, fallbackPatientDetail, fallbackPatients } from "./mock-data";
import type {
  AnalysisDetail,
  ChangePasswordRequest,
  DashboardOverview,
  ImageUploadResponse,
  PatientDetail,
  PatientListResponse,
  TokenResponse,
  UserCreateRequest,
  UserProfile,
  UserUpdateRequest
} from "./types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

async function fetchJson<T>(path: string, fallback: T): Promise<T> {
  try {
    const response = await fetch(`${API_BASE_URL}${path}`, {
      cache: "no-store",
    headers,
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

export async function uploadImage(formData: FormData, token?: string): Promise<ImageUploadResponse> {
  const headers: HeadersInit = token ? { Authorization: `Bearer ${token}` } : {};

  const response = await fetch(`${API_BASE_URL}/api/uploads/images`, {
    method: "POST",
    body: formData,
    cache: "no-store",
    headers,
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Upload failed: ${response.status}`);
  }

  return (await response.json()) as ImageUploadResponse;
}


async function extractErrorMessage(response: Response): Promise<string> {
  try {
    const data = await response.json();
    if (typeof data === "string") {
      return data;
    }
    if (data && typeof data.detail === "string") {
      return data.detail;
    }
    if (data && Array.isArray(data.detail) && data.detail.length > 0) {
      const first = data.detail[0];
      if (first?.msg) {
        return String(first.msg);
      }
    }
    return JSON.stringify(data);
  } catch {
    try {
      return await response.text();
    } catch {
      return `Request failed: ${response.status}`;
    }
  }
}

function buildAuthHeaders(token: string, base: HeadersInit = {}): HeadersInit {
  return {
    Authorization: `Bearer ${token}`,
    ...base
  };
}

export async function registerUser(payload: UserCreateRequest): Promise<UserProfile> {
  const response = await fetch(`${API_BASE_URL}/api/auth/register`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  });

  if (!response.ok) {
    throw new Error(await extractErrorMessage(response));
  }

  return (await response.json()) as UserProfile;
}

export async function loginUser(email: string, password: string): Promise<TokenResponse> {
  const params = new URLSearchParams();
  params.set("username", email);
  params.set("password", password);

  const response = await fetch(`${API_BASE_URL}/api/auth/login`, {
    method: "POST",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded"
    },
    body: params.toString()
  });

  if (!response.ok) {
    throw new Error(await extractErrorMessage(response));
  }

  return (await response.json()) as TokenResponse;
}

export async function fetchCurrentUser(token: string): Promise<UserProfile> {
  const response = await fetch(`${API_BASE_URL}/api/auth/me`, {
    headers: buildAuthHeaders(token)
  });

  if (!response.ok) {
    throw new Error(await extractErrorMessage(response));
  }

  return (await response.json()) as UserProfile;
}

export async function updateProfile(token: string, payload: UserUpdateRequest): Promise<UserProfile> {
  const response = await fetch(`${API_BASE_URL}/api/auth/me`, {
    method: "PATCH",
    headers: buildAuthHeaders(token, {
      "Content-Type": "application/json"
    }),
    body: JSON.stringify(payload)
  });

  if (!response.ok) {
    throw new Error(await extractErrorMessage(response));
  }

  return (await response.json()) as UserProfile;
}

export async function changePassword(token: string, payload: ChangePasswordRequest): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/auth/change-password`, {
    method: "POST",
    headers: buildAuthHeaders(token, {
      "Content-Type": "application/json"
    }),
    body: JSON.stringify(payload)
  });

  if (!response.ok) {
    throw new Error(await extractErrorMessage(response));
  }
}

export async function logoutUser(token: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/auth/logout`, {
    method: "POST",
    headers: buildAuthHeaders(token)
  });

  if (!response.ok && response.status !== 204) {
    throw new Error(await extractErrorMessage(response));
  }
}
