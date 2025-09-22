import { fallbackAnalysis, fallbackAnalysesSummary, fallbackDashboard, fallbackModels, fallbackPatientDetail, fallbackPatients } from "./mock-data";
import type {
  AnalysisDetail,
  AnalysisSummary,
  ChangePasswordRequest,
  DashboardOverview,
  ImageUploadResponse,
  ModelConfig,
  ModelConfigCreateRequest,
  ModelConfigUpdateRequest,
  PatientDetail,
  PatientListResponse,
  TokenResponse,
  UserCreateRequest,
  UserProfile,
  UserUpdateRequest
} from "./types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

async function fetchJson<T>(path: string, fallback: T, token?: string): Promise<T> {
  try {
    const headers: HeadersInit = token
      ? {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`
        }
      : {
          "Content-Type": "application/json"
        };

    const response = await fetch(`${API_BASE_URL}${path}`, {
      cache: "no-store",
      headers
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

export async function fetchDashboardOverview(token?: string): Promise<DashboardOverview> {
  return fetchJson<DashboardOverview>("/api/dashboard/overview", fallbackDashboard, token);
}

export async function fetchPatients(search?: string, token?: string): Promise<PatientListResponse> {
  const query = search ? `?search=${encodeURIComponent(search)}` : "";
  return fetchJson<PatientListResponse>(`/api/patients${query}`, fallbackPatients, token);
}

export async function fetchPatientDetail(patientId: string, token?: string): Promise<PatientDetail> {
  return fetchJson<PatientDetail>(`/api/patients/${patientId}`, fallbackPatientDetail, token);
}

export async function fetchAnalysisDetail(analysisId: string, token?: string): Promise<AnalysisDetail> {
  return fetchJson<AnalysisDetail>(`/api/analyses/${analysisId}`, fallbackAnalysis, token);
}

export async function fetchAnalyses(token?: string, status?: string): Promise<AnalysisSummary[]> {
  const query = status ? `?status=${encodeURIComponent(status)}` : "";
  return fetchJson<AnalysisSummary[]>(`/api/analyses${query}`, fallbackAnalysesSummary, token);
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

export async function fetchModels(token?: string): Promise<ModelConfig[]> {
  return fetchJson<ModelConfig[]>("/api/models", fallbackModels, token);
}

export async function createModelConfig(token: string, payload: ModelConfigCreateRequest): Promise<ModelConfig> {
  const response = await fetch(`${API_BASE_URL}/api/models`, {
    method: "POST",
    headers: buildAuthHeaders(token, {
      "Content-Type": "application/json"
    }),
    body: JSON.stringify(payload)
  });

  if (!response.ok) {
    throw new Error(await extractErrorMessage(response));
  }

  return (await response.json()) as ModelConfig;
}

export async function updateModelConfig(token: string, modelId: string, payload: ModelConfigUpdateRequest): Promise<ModelConfig> {
  const response = await fetch(`${API_BASE_URL}/api/models/${modelId}`, {
    method: "PATCH",
    headers: buildAuthHeaders(token, {
      "Content-Type": "application/json"
    }),
    body: JSON.stringify(payload)
  });

  if (!response.ok) {
    throw new Error(await extractErrorMessage(response));
  }

  return (await response.json()) as ModelConfig;
}

export async function activateModelConfig(token: string, modelId: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/models/${modelId}/activate`, {
    method: "POST",
    headers: buildAuthHeaders(token)
  });

  if (!response.ok && response.status !== 204) {
    throw new Error(await extractErrorMessage(response));
  }
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
