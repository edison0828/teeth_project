import Link from "next/link";
import { fetchPatientDetail, fetchPatients } from "../../lib/api";

export default async function PatientsPage() {
  const patients = await fetchPatients();
  const selectedPatientId = patients.items[0]?.id;
  const patientDetail = selectedPatientId ? await fetchPatientDetail(selectedPatientId) : null;

  return (
    <div className="grid gap-8 xl:grid-cols-[1.2fr,1fr]">
      <section className="rounded-3xl bg-white/5 p-6 shadow-card">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-semibold text-white">Patient Registry</h1>
            <p className="text-sm text-slate-400">Manage patient profiles, visits, and imaging history.</p>
          </div>
          <Link
            href="#add"
            className="rounded-full bg-primary px-4 py-2 text-sm font-semibold text-slate-900 shadow-lg shadow-primary/40"
          >
            + New Patient
          </Link>
        </div>
        <div className="mt-6 overflow-hidden rounded-2xl border border-white/5">
          <table className="min-w-full divide-y divide-white/5 text-sm">
            <thead className="bg-white/5 text-left text-xs uppercase tracking-wide text-slate-400">
              <tr>
                <th className="px-4 py-3">Patient</th>
                <th className="px-4 py-3">Last Visit</th>
                <th className="px-4 py-3">Most Recent Study</th>
                <th className="px-4 py-3 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {patients.items.map((patient) => (
                <tr key={patient.id} className="hover:bg-white/5">
                  <td className="px-4 py-3">
                    <p className="font-medium text-white">{patient.name}</p>
                    <p className="text-xs text-slate-400">ID: {patient.id}</p>
                  </td>
                  <td className="px-4 py-3 text-slate-300">{patient.last_visit ?? "--"}</td>
                  <td className="px-4 py-3 text-slate-300">{patient.most_recent_study ?? "--"}</td>
                  <td className="px-4 py-3 text-right">
                    <Link href={`/analysis/AN-901`} className="text-sm text-primary-subtle">
                      View Analyses
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="rounded-3xl bg-white/5 p-6 shadow-card">
        <h2 className="text-lg font-semibold text-white">Patient Detail</h2>
        {patientDetail ? (
          <div className="mt-4 space-y-6 text-sm text-slate-300">
            <div className="rounded-2xl bg-white/5 p-4">
              <p className="text-xs uppercase tracking-wide text-slate-400">Basic Information</p>
              <div className="mt-3 grid grid-cols-2 gap-3 text-sm">
                <div>
                  <p className="text-slate-400">Name</p>
                  <p className="text-white">{patientDetail.name}</p>
                </div>
                <div>
                  <p className="text-slate-400">DOB</p>
                  <p className="text-white">{patientDetail.dob}</p>
                </div>
                <div>
                  <p className="text-slate-400">Contact</p>
                  <p className="text-white">{patientDetail.contact ?? "--"}</p>
                </div>
                <div>
                  <p className="text-slate-400">Last Visit</p>
                  <p className="text-white">{patientDetail.last_visit ?? "--"}</p>
                </div>
              </div>
            </div>

            <div className="rounded-2xl bg-white/5 p-4">
              <p className="text-xs uppercase tracking-wide text-slate-400">Medical History</p>
              <p className="mt-3 leading-relaxed text-slate-300">
                {patientDetail.medical_history ?? "No medical history recorded."}
              </p>
            </div>

            <div className="rounded-2xl bg-white/5 p-4">
              <p className="text-xs uppercase tracking-wide text-slate-400">Recent Analyses</p>
              <ul className="mt-3 space-y-3 text-sm">
                {patientDetail.recent_analyses.map((analysis) => (
                  <li key={analysis.id} className="flex items-center justify-between rounded-xl bg-white/5 px-3 py-2">
                    <div>
                      <p className="font-medium text-white">{analysis.id}</p>
                      <p className="text-xs text-slate-400">{analysis.overall_assessment ?? "In progress"}</p>
                    </div>
                    <span className="rounded-full bg-accent/10 px-3 py-1 text-xs text-accent">{analysis.status}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        ) : (
          <p className="mt-8 text-sm text-slate-400">Select a patient to view details.</p>
        )}
      </section>
    </div>
  );
}
