const STORAGE_KEY = "oral-xray-auth";

type StoredToken = {
  accessToken: string;
  expiresAt: number;
};

function isBrowser(): boolean {
  return typeof window !== "undefined" && typeof window.localStorage !== "undefined";
}

export function persistToken(accessToken: string, expiresInSeconds: number): void {
  if (!isBrowser()) {
    return;
  }

  const expiresAt = Date.now() + expiresInSeconds * 1000;
  const payload: StoredToken = { accessToken, expiresAt };
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
}

export function readToken(): string | null {
  if (!isBrowser()) {
    return null;
  }

  const raw = window.localStorage.getItem(STORAGE_KEY);
  if (!raw) {
    return null;
  }

  try {
    const payload = JSON.parse(raw) as StoredToken;
    if (Date.now() > payload.expiresAt) {
      window.localStorage.removeItem(STORAGE_KEY);
      return null;
    }
    return payload.accessToken;
  } catch (error) {
    window.localStorage.removeItem(STORAGE_KEY);
    return null;
  }
}

export function clearToken(): void {
  if (!isBrowser()) {
    return;
  }
  window.localStorage.removeItem(STORAGE_KEY);
}
