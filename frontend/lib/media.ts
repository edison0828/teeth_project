const API_BASE_MEDIA_URL = (process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000").replace(/\/$/, "");

export function resolveMediaUrl(uri?: string | null): string | undefined {
  if (!uri) {
    return undefined;
  }
  if (/^https?:\/\//i.test(uri)) {
    return uri;
  }
  if (uri.startsWith("/")) {
    return `${API_BASE_MEDIA_URL}${uri}`;
  }
  return uri;
}

export { API_BASE_MEDIA_URL };
