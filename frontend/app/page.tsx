import Link from "next/link";
import ProgressCircle from "../components/ProgressCircle";
import { readServerToken } from "../lib/server-auth";
import StatCard from "../components/StatCard";
import { fetchDashboardOverview } from "../lib/api";

export default async function DashboardPage() {
  const token = readServerToken();

  if (!token) {
    return null;
  }

  const overview = await fetchDashboardOverview(token);

  return (
    <div className="flex flex-col gap-8">
      <section className="grid gap-6 lg:grid-cols-[2fr,1fr]">
        <div className="rounded-3xl bg-white/5 p-6 shadow-card">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <h1 className="text-2xl font-semibold text-white">Quick Actions</h1>
              <p className="mt-1 text-sm text-slate-400">
                Upload new imaging studies or jump back into recent cases.
              </p>
            </div>
            <div className="flex gap-3">
              {overview.quick_actions.map((action) => (
                <Link
                  key={action.id}
                  href={action.action}
                  className="rounded-full bg-primary px-5 py-2 text-sm font-semibold text-slate-900 shadow-lg shadow-primary/40 transition hover:-translate-y-0.5"
                >
                  {action.label}
                </Link>
              ))}
            </div>
          </div>
          <div className="mt-8 grid gap-4 md:grid-cols-3">
            <StatCard
              title="Pending Images"
              value={overview.system_status.pending_images}
              description="Awaiting preprocessing or manual review"
            />
            <StatCard
              title="Reports Ready"
              value={overview.system_status.new_reports}
              description="Completed analyses awaiting sign-off"
              tone="accent"
            />
            <StatCard
              title="Models Active"
              value={overview.system_status.models_active}
              description={`Last synced ${new Date(overview.system_status.last_synced).toLocaleString()}`}
            />
          </div>
        </div>
        <div className="flex flex-col gap-4 rounded-3xl bg-white/5 p-6 shadow-card">
          <h2 className="text-lg font-semibold text-white">Statistics Overview</h2>
          <div className="flex flex-1 flex-col items-center justify-center gap-6">
            <ProgressCircle value={overview.statistics.detection_rate} label="Accuracy" />
            <div className="grid w-full grid-cols-2 gap-3 text-sm text-slate-300">
              <div>
                <p className="text-xs uppercase tracking-wide text-slate-400">Weekly Volume</p>
                <p className="mt-1 text-xl font-semibold text-white">{overview.statistics.weekly_volume}</p>
                <p className="text-xs text-accent">+{Math.round(overview.statistics.week_over_week_change * 100)}% vs last week</p>
              </div>
              <div>
                <p className="text-xs uppercase tracking-wide text-slate-400">Avg Processing</p>
                <p className="mt-1 text-xl font-semibold text-white">{overview.statistics.average_processing_time} mins</p>
                <p className="text-xs text-slate-400">Uptime {(overview.statistics.uptime_percentage * 100).toFixed(1)}%</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="grid gap-6 lg:grid-cols-3">
        <div className="rounded-3xl bg-white/5 p-6 shadow-card lg:col-span-2">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-white">Detected Conditions</h2>
            <span className="text-xs uppercase tracking-wide text-slate-400">Last 30 days</span>
          </div>
          <div className="mt-6 grid gap-4 md:grid-cols-3">
            {overview.detected_conditions.map((condition) => (
              <div key={condition.label} className="rounded-2xl bg-white/5 p-4">
                <p className="text-sm text-slate-400">{condition.label}</p>
                <p className="mt-2 text-3xl font-semibold text-white">{condition.count}</p>
                <div className="mt-3 space-y-2 text-xs text-slate-400">
                  {condition.severity_breakdown.map((slice) => (
                    <div key={slice.level} className="flex items-center justify-between">
                      <span className="capitalize">{slice.level}</span>
                      <span className="font-medium text-white">{Math.round(slice.percentage * 100)}%</span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
        <div className="rounded-3xl bg-white/5 p-6 shadow-card">
          <h2 className="text-lg font-semibold text-white">Recent Patients</h2>
          <ul className="mt-5 space-y-4 text-sm">
            {overview.recent_patients.map((patient) => (
              <li key={patient.id} className="flex items-center justify-between">
                <div>
                  <p className="font-medium text-white">{patient.name}</p>
                  <p className="text-xs text-slate-400">Last visit {patient.last_visit}</p>
                </div>
                <span className="rounded-full bg-white/10 px-3 py-1 text-xs text-slate-300">
                  {patient.most_recent_study ?? "--"}
                </span>
              </li>
            ))}
          </ul>
          <div className="mt-6 rounded-2xl bg-primary/10 p-4 text-xs text-primary-subtle">
            <p className="font-semibold text-white">Queue Monitoring</p>
            <p className="mt-1 leading-relaxed">
              {overview.pending_images.length} imaging studies are pending review. Auto-assign to radiologists?
            </p>
          </div>
        </div>
      </section>

      <section className="rounded-3xl bg-white/5 p-6 shadow-card">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-white">Pending Uploads</h2>
          <span className="text-xs uppercase tracking-wide text-slate-400">Realtime feed</span>
        </div>
        <div className="mt-4 space-y-3 text-sm">
          {overview.pending_images.map((image) => (
            <div key={image.id} className="flex items-center justify-between rounded-2xl bg-white/5 px-4 py-3">
              <div>
                <p className="font-medium text-white">{image.patient_name}</p>
                <p className="text-xs text-slate-400">
                  {image.image_type} · {new Date(image.submitted_at).toLocaleString()}
                </p>
              </div>
              <span className="rounded-full bg-warning/20 px-3 py-1 text-xs text-warning">{image.status}</span>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
