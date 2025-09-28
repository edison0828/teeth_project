import type { UserProfile } from "./types";

export type NavigationMetrics = {
  pendingImages?: number;
  reportsReady?: number;
  recentPatients?: number;
};

export type SidebarNavItem = {
  label: string;
  href: string;
  badge?: number | null;
};

export type TopNavItem = {
  label: string;
  href: string;
  activeStartsWith?: string;
};

export function getSidebarNavItems(metrics: NavigationMetrics = {}): SidebarNavItem[] {
  const { pendingImages, reportsReady, recentPatients } = metrics;
  return [
    { label: "Dashboard", href: "/" },
    {
      label: "Pending Images",
      href: "/image-upload",
      badge: pendingImages && pendingImages > 0 ? pendingImages : null,
    },
    {
      label: "Analyses",
      href: "/analyses",
      badge: reportsReady && reportsReady > 0 ? reportsReady : null,
    },
    {
      label: "Patients",
      href: "/patients",
      badge: recentPatients && recentPatients > 0 ? recentPatients : null,
    },
    { label: "Demo", href: "/demo" },
    { label: "Notifications", href: "/notifications" },
    { label: "Reports", href: "/reports" },
    { label: "Models", href: "/models" },
  ];
}

export function getTopNavItems(user?: UserProfile | null): TopNavItem[] {
  return [
    { label: "Dashboard", href: "/" },
    { label: "Patients", href: "/patients", activeStartsWith: "/patients" },
    { label: "Image Upload", href: "/image-upload", activeStartsWith: "/image-upload" },
    { label: "Analyses", href: "/analyses", activeStartsWith: "/analysis" },
    { label: "Models", href: "/models", activeStartsWith: "/models" },
    { label: "Demo", href: "/demo", activeStartsWith: "/demo" },
    { label: "Settings", href: "/settings", activeStartsWith: "/settings" },
    { label: "Help", href: "/help", activeStartsWith: "/help" },
  ];
}