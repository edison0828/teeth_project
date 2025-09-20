import { ReactNode } from "react";

interface StatCardProps {
  title: string;
  value: string | number;
  description?: string;
  icon?: ReactNode;
  tone?: "default" | "accent" | "warning" | "danger";
}

const toneStyles: Record<NonNullable<StatCardProps["tone"]>, string> = {
  default: "bg-white/5 text-white",
  accent: "bg-accent/10 text-accent",
  warning: "bg-warning/10 text-warning",
  danger: "bg-danger/10 text-danger"
};

export default function StatCard({ title, value, description, icon, tone = "default" }: StatCardProps) {
  return (
    <div className="rounded-2xl bg-white/5 p-5 shadow-card transition hover:-translate-y-0.5 hover:shadow-lg">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs uppercase tracking-wide text-slate-400">{title}</p>
          <p className="mt-2 text-3xl font-semibold text-white">{value}</p>
        </div>
        {icon ? <div className={`rounded-full p-3 ${toneStyles[tone]}`}>{icon}</div> : null}
      </div>
      {description ? <p className="mt-3 text-sm text-slate-400">{description}</p> : null}
    </div>
  );
}
