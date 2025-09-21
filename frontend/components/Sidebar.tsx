"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import { clsx } from "clsx";

import { useAuth } from "../contexts/AuthContext";
import { fetchDashboardOverview } from "../lib/api";
import { getSidebarNavItems, type NavigationMetrics } from "../lib/navigation";

export default function Sidebar() {
  const pathname = usePathname();
  const { token } = useAuth();
  const [metrics, setMetrics] = useState<NavigationMetrics>({});

  useEffect(() => {
    let cancelled = false;

    async function loadMetrics() {
      if (!token) {
        setMetrics({});
        return;
      }

      try {
        const overview = await fetchDashboardOverview(token);
        if (cancelled) {
          return;
        }
        setMetrics({
          pendingImages: overview.system_status.pending_images,
          reportsReady: overview.system_status.new_reports,
          recentPatients: overview.recent_patients.length,
        });
      } catch (error) {
        if (!cancelled) {
          setMetrics({});
        }
        console.warn("Unable to load navigation metrics", error);
      }
    }

    loadMetrics();

    return () => {
      cancelled = true;
    };
  }, [token]);

  const navigation = useMemo(() => getSidebarNavItems(metrics), [metrics]);

  return (
    <aside className="hidden min-h-screen w-64 flex-col border-r border-white/5 bg-[#050B1C]/80 p-6 backdrop-blur xl:flex">
      <div className="mb-10 flex items-center gap-2 text-lg font-semibold tracking-wide text-primary">
        <span className="flex h-10 w-10 items-center justify-center rounded-2xl bg-primary/10 text-xl font-bold text-primary">
          DM
        </span>
        <div>
          <p className="text-sm uppercase text-slate-300">DentaMind AI</p>
          <p className="text-xs text-slate-500">Intelligence Suite</p>
        </div>
      </div>
      <nav className="flex flex-1 flex-col gap-2">
        {navigation.map((item) => {
          const isActive = pathname === item.href || (item.href !== "/" && pathname.startsWith(item.href));
          return (
            <Link
              key={item.label}
              href={item.href}
              className={clsx(
                "group flex items-center justify-between rounded-xl px-4 py-3 text-sm font-medium transition-all",
                isActive
                  ? "bg-primary/10 text-white shadow-card"
                  : "text-slate-400 hover:bg-white/5 hover:text-white"
              )}
            >
              <span>{item.label}</span>
              {typeof item.badge === "number" && item.badge > 0 ? (
                <span className="rounded-full bg-primary/30 px-2 py-0.5 text-xs text-primary-subtle">
                  {item.badge}
                </span>
              ) : null}
            </Link>
          );
        })}
      </nav>
      <div className="mt-auto rounded-2xl bg-white/5 p-4 text-xs text-slate-400">
        <p className="font-semibold text-white">Analysis Capacity</p>
        <p className="mt-2 text-3xl font-bold text-white">68%</p>
        <p className="mt-1 leading-relaxed text-slate-400">
          AI cluster is operating within optimal load. Scheduled maintenance on Nov 25.
        </p>
      </div>
    </aside>
  );
}