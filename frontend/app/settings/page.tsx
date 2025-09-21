import { readServerToken } from "../../lib/server-auth";

const toggles = [
  { id: "notify-email", label: "電子郵件通知", description: "有新的 AI 分析或報告時透過 email 提醒。" },
  { id: "notify-sms", label: "簡訊通知", description: "高優先權告警時發送簡訊提醒。" },
  { id: "auto-assign", label: "自動指派分析", description: "新影像上傳後自動指派至待命醫師。" }
];

export default async function SettingsPage() {
  const token = readServerToken();

  if (!token) {
    return null;
  }

  return (
    <div className="space-y-8">
      <header className="rounded-3xl bg-white/5 px-8 py-10 text-white shadow-card">
        <p className="text-xs uppercase tracking-[0.3em] text-primary">Settings</p>
        <h1 className="mt-4 text-3xl font-semibold">平台設定</h1>
        <p className="mt-3 text-sm text-slate-300 lg:max-w-3xl">
          管理通知、分析流程與模型行為，針對醫療團隊需求調整 AI 平台的運作方式。
        </p>
      </header>

      <section className="grid gap-6 lg:grid-cols-2">
        <div className="rounded-3xl bg-white/5 p-6 shadow-card">
          <h2 className="text-lg font-semibold text-white">通知偏好</h2>
          <p className="mt-1 text-sm text-slate-300">選擇 AI 平台推播與提醒的方式。</p>
          <form className="mt-5 space-y-4 text-sm text-slate-200">
            {toggles.map((toggle) => (
              <label key={toggle.id} className="flex items-start gap-3 rounded-2xl border border-white/10 bg-[#0B142A] p-4">
                <input type="checkbox" className="mt-1 h-4 w-4 rounded border-white/20 bg-transparent" defaultChecked />
                <span>
                  <span className="text-sm font-semibold text-white">{toggle.label}</span>
                  <p className="text-xs text-slate-400">{toggle.description}</p>
                </span>
              </label>
            ))}
            <button className="w-full rounded-full bg-gradient-to-r from-primary to-accent px-4 py-3 text-sm font-semibold text-slate-900 shadow-lg shadow-primary/40">
              儲存偏好
            </button>
          </form>
        </div>
        <div className="rounded-3xl bg-white/5 p-6 shadow-card">
          <h2 className="text-lg font-semibold text-white">模型與流程</h2>
          <ul className="mt-5 space-y-4 text-sm text-slate-200">
            <li className="rounded-2xl border border-white/10 bg-[#0B142A] p-4">
              <p className="text-sm font-semibold text-white">模型版本</p>
              <p className="text-xs text-slate-400">Caries Detector v1.3.0 · Periodontal Segmenter v0.9.1</p>
            </li>
            <li className="rounded-2xl border border-white/10 bg-[#0B142A] p-4">
              <p className="text-sm font-semibold text-white">分析 SLA</p>
              <p className="text-xs text-slate-400">標準分析需於 15 分鐘內完成，逾時將發送通知。</p>
            </li>
            <li className="rounded-2xl border border-white/10 bg-[#0B142A] p-4">
              <p className="text-sm font-semibold text-white">維運排程</p>
              <p className="text-xs text-slate-400">下次系統保養：11 月 25 日 03:00-04:00 (GMT+8)</p>
            </li>
          </ul>
        </div>
      </section>
    </div>
  );
}