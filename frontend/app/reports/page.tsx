import Link from "next/link";

import { readServerToken } from "../../lib/server-auth";

const demoReports = [
  {
    id: "REP-202311-01",
    analysisId: "AN-901",
    patient: "Jane Doe",
    generatedAt: new Date().toISOString(),
    author: "Dr. Lee"
  },
  {
    id: "REP-202311-02",
    analysisId: "AN-902",
    patient: "John Smith",
    generatedAt: new Date().toISOString(),
    author: "Dr. Wu"
  }
];

export default async function ReportsPage() {
  const token = readServerToken();

  if (!token) {
    return null;
  }

  return (
    <div className="space-y-8">
      <header className="rounded-3xl bg-white/5 px-8 py-10 text-white shadow-card">
        <p className="text-xs uppercase tracking-[0.3em] text-primary">Reports</p>
        <h1 className="mt-4 text-3xl font-semibold">報告匣</h1>
        <p className="mt-3 text-sm text-slate-300 lg:max-w-3xl">
          集中管理已產出的分析報告，支援下載、分享與審核紀錄，使跨團隊協作更加順暢。
        </p>
      </header>

      <section className="rounded-3xl bg-white/5 p-6 shadow-card">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-white">最新報告</h2>
          <button className="rounded-full border border-white/10 px-4 py-2 text-xs font-semibold text-white hover:bg-white/10">
            建立客製報告
          </button>
        </div>
        <div className="mt-6 overflow-hidden rounded-2xl border border-white/5">
          <table className="min-w-full divide-y divide-white/5 text-sm">
            <thead className="bg-white/5 text-left text-xs uppercase tracking-wide text-slate-400">
              <tr>
                <th className="px-4 py-3">報告編號</th>
                <th className="px-4 py-3">分析案件</th>
                <th className="px-4 py-3">患者</th>
                <th className="px-4 py-3">產出時間</th>
                <th className="px-4 py-3">作者</th>
                <th className="px-4 py-3 text-right">操作</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {demoReports.map((report) => (
                <tr key={report.id} className="hover:bg-white/5">
                  <td className="px-4 py-3 font-medium text-white">{report.id}</td>
                  <td className="px-4 py-3">
                    <Link href={`/analysis/${report.analysisId}`} className="text-primary hover:text-primary-subtle">
                      {report.analysisId}
                    </Link>
                  </td>
                  <td className="px-4 py-3 text-slate-300">{report.patient}</td>
                  <td className="px-4 py-3 text-xs text-slate-400">
                    {new Date(report.generatedAt).toLocaleString()}
                  </td>
                  <td className="px-4 py-3 text-slate-300">{report.author}</td>
                  <td className="px-4 py-3 text-right">
                    <div className="flex justify-end gap-2 text-xs">
                      <button className="rounded-full border border-white/10 px-3 py-1 text-slate-300 hover:bg-white/10">
                        下載
                      </button>
                      <button className="rounded-full border border-white/10 px-3 py-1 text-slate-300 hover:bg-white/10">
                        分享
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}