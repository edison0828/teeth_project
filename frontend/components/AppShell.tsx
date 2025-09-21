"use client";

import { usePathname } from "next/navigation";
import type { ReactNode } from "react";
import { useAuth } from "../contexts/AuthContext";
import Sidebar from "./Sidebar";
import TopNav from "./TopNav";

const AUTH_ROUTES = ["/login", "/register"];

function isAuthRoute(pathname: string): boolean {
  return AUTH_ROUTES.some((route) => pathname === route || pathname.startsWith(`${route}/`));
}

export default function AppShell({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const { loading, user } = useAuth();

  if (isAuthRoute(pathname)) {
    return (
      <main className="flex min-h-screen flex-col items-center justify-center px-6 py-12">
        {children}
      </main>
    );
  }

  if (loading || !user) {
    return (
      <div className="flex min-h-screen items-center justify-center text-sm text-slate-300">
        驗證中...
      </div>
    );
  }

  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <div className="flex flex-1 flex-col">
        <TopNav />
        <main className="flex-1 overflow-y-auto p-8 lg:p-10 xl:p-12">{children}</main>
      </div>
    </div>
  );
}