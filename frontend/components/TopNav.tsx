"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { clsx } from "clsx";
import { useEffect, useMemo, useRef, useState } from "react";

import { useAuth } from "../contexts/AuthContext";
import { getTopNavItems } from "../lib/navigation";

function getInitials(nameOrEmail: string | undefined | null): string {
  if (!nameOrEmail) {
    return "U";
  }

  const trimmed = nameOrEmail.trim();
  if (!trimmed) {
    return "U";
  }

  const parts = trimmed.split(/\s+/).filter(Boolean);
  if (parts.length >= 2) {
    return `${parts[0][0]}${parts[parts.length - 1][0]}`.toUpperCase();
  }

  const single = parts[0];
  if (!single.includes("@")) {
    return single.slice(0, 2).toUpperCase();
  }

  const alias = single.split("@")[0];
  return alias.slice(0, 2).toUpperCase() || "U";
}

export default function TopNav() {
  const pathname = usePathname();
  const { user, logout } = useAuth();
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setMenuOpen(false);
      }
    }

    if (menuOpen) {
      document.addEventListener("mousedown", handleClickOutside);
    }
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [menuOpen]);

  const navItems = useMemo(() => getTopNavItems(user), [user]);

  const displayName = user?.full_name?.trim() || user?.email || "已登入使用者";
  const subtitle =
    user?.full_name && user?.email
      ? user.email
      : user
        ? user.is_active
          ? "帳號已啟用"
          : "帳號停用"
        : "";
  const initials = getInitials(user?.full_name ?? user?.email ?? undefined);

  return (
    <header className="sticky top-0 z-20 flex items-center justify-between border-b border-white/5 bg-[#0B142A]/80 px-6 py-5 backdrop-blur lg:px-10">
      <div className="hidden items-center gap-3 text-sm font-medium text-slate-300 md:flex">
        {navItems.map((item) => {
          const isActive = item.activeStartsWith
            ? pathname === item.href || pathname.startsWith(item.activeStartsWith)
            : pathname === item.href;
          return (
            <Link
              key={item.label}
              href={item.href}
              className={clsx(
                "rounded-full px-4 py-2 transition-colors",
                isActive ? "bg-white/10 text-white" : "hover:bg-white/5"
              )}
            >
              {item.label}
            </Link>
          );
        })}
      </div>
      <div className="ml-auto flex items-center gap-6 text-sm">
        <div className="hidden items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-2 text-slate-300 md:flex">
          <span className="text-xs uppercase tracking-wide text-slate-400">Models</span>
          <span className="rounded-full bg-primary/20 px-2 py-0.5 text-primary-subtle">4 active</span>
        </div>
        <div className="relative" ref={menuRef}>
          <button
            type="button"
            onClick={() => setMenuOpen((open) => !open)}
            className="flex items-center gap-3 rounded-full border border-white/10 bg-white/5 px-3 py-2 text-left text-xs text-slate-400 transition hover:bg-white/10 hover:text-white"
          >
            <span className="hidden text-right md:block">
              <span className="block text-sm font-semibold text-white">{displayName}</span>
              {subtitle}
            </span>
            <span className="flex h-10 w-10 items-center justify-center rounded-full bg-gradient-to-br from-primary to-accent text-sm font-semibold text-slate-900">
              {initials}
            </span>
          </button>
          {menuOpen ? (
            <div className="absolute right-0 mt-3 w-48 rounded-2xl border border-white/10 bg-[#050B1C]/95 p-2 text-sm text-slate-200 shadow-xl">
              <Link
                href="/account"
                className="block rounded-xl px-3 py-2 hover:bg-white/10"
                onClick={() => setMenuOpen(false)}
              >
                帳戶設定
              </Link>
              <Link
                href="/profile"
                className="block rounded-xl px-3 py-2 hover:bg-white/10"
                onClick={() => setMenuOpen(false)}
              >
                個人檔案
              </Link>
              <Link
                href="/settings"
                className="block rounded-xl px-3 py-2 hover:bg-white/10"
                onClick={() => setMenuOpen(false)}
              >
                平台設定
              </Link>
              <button
                type="button"
                onClick={async () => {
                  setMenuOpen(false);
                  await logout();
                }}
                className="mt-2 w-full rounded-xl bg-red-500/10 px-3 py-2 text-left text-red-300 hover:bg-red-500/20"
              >
                登出
              </button>
            </div>
          ) : null}
        </div>
      </div>
    </header>
  );
}