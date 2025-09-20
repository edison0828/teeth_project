import { fetchPatients } from "../../lib/api";

export default async function ImageUploadPage() {
  const patients = await fetchPatients();

  return (
    <div className="grid gap-8 xl:grid-cols-[1.2fr,1fr]">
      <section className="rounded-3xl bg-white/5 p-6 shadow-card">
        <h1 className="text-2xl font-semibold text-white">Upload &amp; Preprocessing</h1>
        <p className="mt-1 text-sm text-slate-400">
          Drag and drop CBCT, panoramic, or bitewing studies. The system pre-validates DICOM metadata before
          preprocessing.
        </p>
        <div className="mt-8 flex h-64 flex-col items-center justify-center rounded-3xl border border-dashed border-primary/40 bg-white/5 text-center">
          <p className="text-lg font-semibold text-primary-subtle">Drop X-ray images here</p>
          <p className="mt-2 text-sm text-slate-400">or click to browse files</p>
          <button className="mt-6 rounded-full bg-primary px-5 py-2 text-sm font-semibold text-slate-900 shadow-lg shadow-primary/40">
            Select Files
          </button>
        </div>
        <div className="mt-6 grid gap-4 md:grid-cols-2">
          <div className="rounded-2xl bg-white/5 p-4">
            <p className="text-xs uppercase tracking-wide text-slate-400">Preprocessing</p>
            <ul className="mt-3 space-y-2 text-sm">
              <li className="flex items-center justify-between">
                <span>Auto orientation</span>
                <span className="rounded-full bg-accent/20 px-2 py-0.5 text-xs text-accent">Enabled</span>
              </li>
              <li className="flex items-center justify-between">
                <span>Noise reduction</span>
                <span className="rounded-full bg-white/10 px-2 py-0.5 text-xs text-slate-300">Adaptive</span>
              </li>
              <li className="flex items-center justify-between">
                <span>Contrast harmonization</span>
                <span className="rounded-full bg-white/10 px-2 py-0.5 text-xs text-slate-300">AI optimized</span>
              </li>
            </ul>
          </div>
          <div className="rounded-2xl bg-white/5 p-4">
            <p className="text-xs uppercase tracking-wide text-slate-400">Upload Status</p>
            <div className="mt-3 space-y-3 text-sm">
              <div className="rounded-xl bg-white/5 px-3 py-2">
                <p className="font-medium text-white">pending_caries_20231103.dcm</p>
                <p className="text-xs text-slate-400">Queued for preprocessing</p>
              </div>
              <div className="rounded-xl bg-white/5 px-3 py-2">
                <p className="font-medium text-white">johnsmith_bw_20231031.dcm</p>
                <p className="text-xs text-accent">Analyzing (75%)</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="rounded-3xl bg-white/5 p-6 shadow-card">
        <h2 className="text-lg font-semibold text-white">Associated Patient</h2>
        <div className="mt-4 space-y-4 text-sm text-slate-300">
          <div>
            <label className="text-xs uppercase tracking-wide text-slate-400">Patient</label>
            <select className="mt-2 w-full rounded-xl border border-white/10 bg-[#0B142A] px-3 py-2">
              {patients.items.map((patient) => (
                <option key={patient.id} value={patient.id}>
                  {patient.name}
                </option>
              ))}
            </select>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-xs uppercase tracking-wide text-slate-400">Study Type</label>
              <select className="mt-2 w-full rounded-xl border border-white/10 bg-[#0B142A] px-3 py-2">
                <option>Panoramic Scan</option>
                <option>Bitewing</option>
                <option>Full-mouth X-ray</option>
                <option>CBCT</option>
              </select>
            </div>
            <div>
              <label className="text-xs uppercase tracking-wide text-slate-400">Priority</label>
              <select className="mt-2 w-full rounded-xl border border-white/10 bg-[#0B142A] px-3 py-2">
                <option>Standard</option>
                <option>Urgent</option>
              </select>
            </div>
          </div>
          <div>
            <label className="text-xs uppercase tracking-wide text-slate-400">Immediate AI Analysis</label>
            <div className="mt-2 flex items-center gap-3">
              <input type="checkbox" defaultChecked className="h-4 w-4 rounded border-white/20 bg-transparent" />
              <span>Trigger AI pipeline after upload</span>
            </div>
          </div>
          <div>
            <label className="text-xs uppercase tracking-wide text-slate-400">Clinical Notes</label>
            <textarea
              rows={4}
              className="mt-2 w-full rounded-xl border border-white/10 bg-[#0B142A] px-3 py-2"
              placeholder="Highlight findings or instructions for radiologist review"
            />
          </div>
          <button className="w-full rounded-full bg-gradient-to-r from-primary to-accent px-4 py-3 text-sm font-semibold text-slate-900 shadow-lg shadow-primary/40">
            Confirm &amp; Analyze
          </button>
        </div>
      </section>
    </div>
  );
}
