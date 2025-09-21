import { AUTH_COOKIE_KEY, AUTH_STORAGE_KEY } from "./auth-keys";

type StoredToken = {
  accessToken: string;
  expiresAt: number;
};

function isBrowser(): boolean {
  return typeof window !== "undefined" && typeof window.document !== "undefined";
}

function persistCookie(accessToken: string, expiresInSeconds: number): void {
  if (!isBrowser()) {
    return;
  }

  const maxAge = Math.max(1, Math.floor(expiresInSeconds));
  const secure = window.location.protocol === "https:" ? "; Secure" : "";
  document.cookie = `${AUTH_COOKIE_KEY}=${encodeURIComponent(accessToken)}; Path=/; Max-Age=${maxAge}; SameSite=Lax${secure}`;
}

function readCookie(): string | null {
  if (!isBrowser()) {
    return null;
  }

  const cookies = document.cookie ? document.cookie.split(/;\s*/) : [];
  for (const entry of cookies) {
    if (entry.startsWith(`${AUTH_COOKIE_KEY}=`)) {
      return decodeURIComponent(entry.substring(AUTH_COOKIE_KEY.length + 1));
    }
  }
  return null;
}

function clearCookie(): void {
  if (!isBrowser()) {
    return;
  }
  document.cookie = `${AUTH_COOKIE_KEY}=; Path=/; Max-Age=0; SameSite=Lax`;
}

export function persistToken(accessToken: string, expiresInSeconds: number): void {
  if (!isBrowser()) {
    return;
  }

  const expiresAt = Date.now() + expiresInSeconds * 1000;
  const payload: StoredToken = { accessToken, expiresAt };
  window.localStorage.setItem(AUTH_STORAGE_KEY, JSON.stringify(payload));
  persistCookie(accessToken, expiresInSeconds);
}

export function readToken(): string | null {
  if (!isBrowser()) {
    return null;
  }

  const raw = window.localStorage.getItem(AUTH_STORAGE_KEY);
  if (raw) {
    try {
      const payload = JSON.parse(raw) as StoredToken;
      if (Date.now() > payload.expiresAt) {
        window.localStorage.removeItem(AUTH_STORAGE_KEY);
        clearCookie();
      } else {
        return payload.accessToken;
      }
    } catch (error) {
      window.localStorage.removeItem(AUTH_STORAGE_KEY);
      clearCookie();
    }
  }

  return readCookie();
}

export function clearToken(): void {
  if (!isBrowser()) {
    return;
  }
  window.localStorage.removeItem(AUTH_STORAGE_KEY);
  clearCookie();
}
