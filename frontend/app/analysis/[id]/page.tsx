import { notFound } from "next/navigation";
import { fetchAnalysisDetail } from "../../../lib/api";
import { readServerToken } from "../../../lib/server-auth";

interface AnalysisPageProps {
  params: { id: string };
}

export default async function AnalysisPage({ params }: AnalysisPageProps) {
  const { id } = params;
  const token = readServerToken();

  if (!token) {
    return null;
  }

  const analysis = await fetchAnalysisDetail(id, token);

  if (!analysis) {
    notFound();
  }

  return (
    <div className="grid gap-8 xl:grid-cols-[1.4fr,1fr]">
      <section className="rounded-3xl bg-white/5 p-6 shadow-card">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-semibold text-white">Image Analysis Result</h1>
            <p className="text-sm text-slate-400">
              {analysis.overall_assessment ?? "Analysis in progress. Findings will appear shortly."}
            </p>
          </div>
          <span className="rounded-full bg-accent/10 px-4 py-1 text-sm font-medium text-accent">{analysis.status}</span>
        </div>

        <div className="mt-6 rounded-3xl border border-white/5 bg-[#050B1C] p-4">
          <div className="relative flex h-[420px] items-center justify-center rounded-2xl border border-white/10 bg-gradient-to-br from-[#0F172A] to-[#101936]">
            <div className="absolute inset-6 rounded-2xl border border-primary/30"></div>
            <p className="text-sm text-slate-400">Radiograph preview placeholder</p>
          </div>
          <div className="mt-4 flex flex-wrap items-center gap-3 text-xs text-slate-300">
            <span className="rounded-full bg-white/5 px-3 py-1">Zoom</span>
            <span className="rounded-full bg-white/5 px-3 py-1">Contrast</span>
            <span className="rounded-full bg-white/5 px-3 py-1">Mask Overlay</span>
            <span className="rounded-full bg-white/5 px-3 py-1">Annotations</span>
          </div>
        </div>

        <div className="mt-6 grid gap-4 md:grid-cols-2">
          {analysis.findings.map((finding) => (
            <div key={finding.finding_id} className="rounded-2xl bg-white/5 p-4">
              <div className="flex items-center justify-between">
                <p className="text-sm font-semibold text-white capitalize">{finding.type}</p>
                <span className="rounded-full bg-white/10 px-3 py-1 text-xs text-slate-300">
                  {finding.tooth_label ?? "--"}
                </span>
              </div>
              <p className="mt-2 text-xs text-slate-400">Severity: {finding.severity}</p>
              <p className="mt-1 text-xs text-slate-400">Confidence: {(finding.confidence * 100).toFixed(1)}%</p>
              <p className="mt-1 text-xs text-slate-500">Model {finding.model_key} · {finding.model_version}</p>
              {finding.note ? <p className="mt-2 text-sm text-slate-300">Note: {finding.note}</p> : null}
            </div>
          ))}
        </div>
      </section>

      <section className="space-y-6">
        <div className="rounded-3xl bg-white/5 p-6 shadow-card">
          <h2 className="text-lg font-semibold text-white">Patient Overview</h2>
          <div className="mt-3 space-y-2 text-sm text-slate-300">
            <p className="flex justify-between"><span className="text-slate-400">Patient</span><span className="text-white">{analysis.patient.name}</span></p>
            <p className="flex justify-between"><span className="text-slate-400">Image Type</span><span className="text-white">{analysis.image.type}</span></p>
            <p className="flex justify-between"><span className="text-slate-400">Captured</span><span className="text-white">{new Date(analysis.image.captured_at).toLocaleString()}</span></p>
          </div>
        </div>

        <div className="rounded-3xl bg-white/5 p-6 shadow-card">
          <h2 className="text-lg font-semibold text-white">Detected Conditions</h2>
          <ul className="mt-4 space-y-3 text-sm">
            {analysis.detected_conditions.map((condition) => (
              <li key={condition.label} className="flex items-center justify-between rounded-xl bg-white/5 px-3 py-2">
                <span className="text-white">{condition.label}</span>
                <span className="rounded-full bg-primary/20 px-3 py-1 text-xs text-primary-subtle">{condition.count} findings</span>
              </li>
            ))}
          </ul>
        </div>

        <div className="rounded-3xl bg-white/5 p-6 shadow-card">
          <h2 className="text-lg font-semibold text-white">Pipeline Progress</h2>
          <p className="text-xs text-slate-400">Overall confidence {(analysis.progress.overall_confidence * 100).toFixed(1)}%</p>
          <ul className="mt-4 space-y-4 text-sm text-slate-300">
            {analysis.progress.steps.map((step) => (
              <li key={step.title} className="rounded-2xl bg-white/5 p-3">
                <p className="font-medium text-white">{step.title}</p>
                <p className="text-xs text-slate-400">{new Date(step.timestamp).toLocaleString()} · {step.status}</p>
                <p className="mt-1 text-xs text-slate-400">{step.description}</p>
              </li>
            ))}
          </ul>
        </div>

        <div className="rounded-3xl bg-gradient-to-br from-primary/20 to-accent/20 p-6 text-sm text-white">
          <h2 className="text-lg font-semibold">Report Actions</h2>
          <div className="mt-4 flex flex-wrap gap-3">
            {analysis.report_actions.map((action) => (
              <a
                key={action.label}
                href={action.href}
                className="rounded-full bg-white/10 px-4 py-2 text-sm font-semibold text-white transition hover:bg-white/20"
              >
                {action.label}
              </a>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}
