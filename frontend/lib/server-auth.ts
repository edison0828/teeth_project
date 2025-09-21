import { cookies } from "next/headers";

import { AUTH_COOKIE_KEY } from "./auth-keys";

export function readServerToken(): string | null {
  const cookieStore = cookies();
  const tokenCookie = cookieStore.get(AUTH_COOKIE_KEY);
  return tokenCookie?.value ?? null;
}