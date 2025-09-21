import Link from "next/link";

import { fetchAnalyses } from "../../lib/api";
import { readServerToken } from "../../lib/server-auth";

const STATUS_LABEL: Record<string, string> = {
  completed: "已完成",
  processing: "處理中",
  queued: "排程中"
};

export default async function AnalysesPage() {
  const token = readServerToken();

  if (!token) {
    return null;
  }

  const analyses = await fetchAnalyses(token);

  return (
    <div className="space-y-8">
      <header className="rounded-3xl bg-white/5 px-8 py-10 text-white shadow-card">
        <p className="text-xs uppercase tracking-[0.3em] text-primary">Analyses</p>
        <h1 className="mt-4 text-3xl font-semibold">影像分析總覽</h1>
        <p className="mt-3 text-sm text-slate-300 lg:max-w-3xl">
          追蹤所有生成中的 AI 分析、匯出結果與確認影像報告。點選個別案例即可查看詳細資訊與模型輸出。
        </p>
      </header>

      <section className="rounded-3xl bg-white/5 p-6 shadow-card">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-white">近期分析案件</h2>
          <Link
            href="/image-upload"
            className="rounded-full bg-primary px-4 py-2 text-sm font-semibold text-slate-900 shadow-lg shadow-primary/40"
          >
            + 新增分析
          </Link>
        </div>
        <div className="mt-6 overflow-hidden rounded-2xl border border-white/5">
          <table className="min-w-full divide-y divide-white/5 text-sm">
            <thead className="bg-white/5 text-left text-xs uppercase tracking-wide text-slate-400">
              <tr>
                <th className="px-4 py-3">分析 ID</th>
                <th className="px-4 py-3">影像編號</th>
                <th className="px-4 py-3">申請人</th>
                <th className="px-4 py-3">狀態</th>
                <th className="px-4 py-3">最新更新</th>
                <th className="px-4 py-3 text-right">操作</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {analyses.map((analysis) => (
                <tr key={analysis.id} className="hover:bg-white/5">
                  <td className="px-4 py-3 font-medium text-white">{analysis.id}</td>
                  <td className="px-4 py-3 text-slate-300">{analysis.image_id}</td>
                  <td className="px-4 py-3 text-slate-300">{analysis.requested_by}</td>
                  <td className="px-4 py-3">
                    <span className="rounded-full bg-primary/10 px-3 py-1 text-xs font-semibold text-primary">
                      {STATUS_LABEL[analysis.status] ?? analysis.status}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-xs text-slate-400">
                    {analysis.completed_at
                      ? new Date(analysis.completed_at).toLocaleString()
                      : new Date(analysis.triggered_at).toLocaleString()}
                  </td>
                  <td className="px-4 py-3 text-right">
                    <Link
                      href={`/analysis/${analysis.id}`}
                      className="text-sm font-semibold text-primary hover:text-primary-subtle"
                    >
                      查看
                    </Link>
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