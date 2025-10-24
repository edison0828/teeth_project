import type { DemoInferenceResult, DemoSampleListResponse } from "./types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

async function fetchJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    cache: "no-store",
  });

  if (!response.ok) {
    throw new Error(await extractErrorMessage(response));
  }

  return (await response.json()) as T;
}

export async function fetchDemoSamples(): Promise<DemoSampleListResponse> {
  return fetchJson<DemoSampleListResponse>("/demo/samples");
}

export interface DemoInferenceOptions {
  file?: File;
  sampleId?: string;
  onlyPositive?: boolean;
}

export async function submitDemoInference(options: DemoInferenceOptions): Promise<DemoInferenceResult> {
  const formData = new FormData();
  if (options.sampleId) {
    formData.append("sample_id", options.sampleId);
  }
  if (options.file) {
    formData.append("file", options.file);
  }
  if (options.onlyPositive !== undefined) {
    formData.append("only_positive", String(options.onlyPositive));
  }

  const response = await fetch(`${API_BASE_URL}/demo/infer`, {
    method: "POST",
    body: formData,
    cache: "no-store",
  });

  if (!response.ok) {
    throw new Error(await extractErrorMessage(response));
  }

  return (await response.json()) as DemoInferenceResult;
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
