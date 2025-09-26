"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { usePathname, useRouter } from "next/navigation";
import { fetchCurrentUser, logoutUser } from "../lib/api";
import { clearToken, readToken } from "../lib/auth-storage";
import type { UserProfile } from "../lib/types";

type AuthContextValue = {
  token: string | null;
  user: UserProfile | null;
  loading: boolean;
  refresh: () => Promise<UserProfile | null>;
  logout: () => Promise<void>;
  setToken: (accessToken: string | null) => void;
  setUser: (profile: UserProfile | null) => void;
  guestMode: boolean;
  enterGuestMode: () => void;
  exitGuestMode: () => void;
};

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

const AUTH_ROUTES = ["/login", "/register"];
const GUEST_MODE_KEY = "dentamind-guest-mode";

function isAuthRoute(pathname: string): boolean {
  return AUTH_ROUTES.some((route) => pathname === route || pathname.startsWith(`${route}/`));
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [token, setToken] = useState<string | null>(null);
  const [user, setUser] = useState<UserProfile | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [guestMode, setGuestMode] = useState<boolean>(false);

  const router = useRouter();
  const pathname = usePathname();

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const persisted = window.localStorage.getItem(GUEST_MODE_KEY) === "1";
    if (persisted) {
      setGuestMode(true);
    }
  }, []);

  const refresh = useCallback(async (): Promise<UserProfile | null> => {
    const persistedGuest =
      typeof window !== "undefined" && window.localStorage.getItem(GUEST_MODE_KEY) === "1";

    if (persistedGuest && !guestMode) {
      setGuestMode(true);
    }

    if (guestMode || persistedGuest) {
      setToken(null);
      setUser(null);
      setLoading(false);
      return null;
    }

    const stored = readToken();

    if (!stored) {
      setToken(null);
      setUser(null);
      setLoading(false);
      return null;
    }

    if (stored === token && user) {
      setLoading(false);
      return user;
    }

    setLoading(true);
    setToken(stored);

    try {
      const profile = await fetchCurrentUser(stored);
      setUser(profile);
      return profile;
    } catch (error) {
      clearToken();
      setToken(null);
      setUser(null);
      return null;
    } finally {
      setLoading(false);
    }
  }, [token, user, guestMode]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  useEffect(() => {
    if (loading) {
      return;
    }

    if (!token && !guestMode && !isAuthRoute(pathname)) {
      router.replace("/login");
    } else if ((token || guestMode) && isAuthRoute(pathname)) {
      router.replace("/");
    }
  }, [token, guestMode, loading, pathname, router]);

  const enterGuestMode = useCallback(() => {
    clearToken();
    setToken(null);
    setUser(null);
    setGuestMode(true);
    if (typeof window !== "undefined") {
      window.localStorage.setItem(GUEST_MODE_KEY, "1");
    }
    setLoading(false);
  }, []);

  const exitGuestMode = useCallback(() => {
    setGuestMode(false);
    if (typeof window !== "undefined") {
      window.localStorage.removeItem(GUEST_MODE_KEY);
    }
  }, []);

  const logout = useCallback(async () => {
    if (token) {
      try {
        await logoutUser(token);
      } catch (error) {
        console.warn("logout warning", error);
      }
    }
    clearToken();
    setToken(null);
    setUser(null);
    if (guestMode) {
      exitGuestMode();
    }
    router.push("/login");
  }, [token, guestMode, exitGuestMode, router]);

  const value = useMemo<AuthContextValue>(
    () => ({
      token,
      user,
      loading,
      refresh,
      logout,
      setUser,
      setToken,
      guestMode,
      enterGuestMode,
      exitGuestMode,
    }),
    [token, user, loading, refresh, logout, guestMode, enterGuestMode, exitGuestMode]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}