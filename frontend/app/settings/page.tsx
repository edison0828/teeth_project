import Link from "next/link";

const toggles = [
  { id: "notify-email", label: "電子郵件通知", description: "有新的 AI 分析或報告時透過 email 提醒。" },
  { id: "notify-sms", label: "簡訊通知", description: "高優先權告警時發送簡訊提醒。" },
  { id: "auto-assign", label: "自動指派分析", description: "新影像上傳後自動指派至待命醫師。" }
];

export default function SettingsPage() {
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
          <h2 className="text-lg font-semibold text-white">模型管理</h2>
          <p className="mt-1 text-sm text-slate-300">
            管理牙位與齲齒偵測模型，切換不同權重或調整門檻設定。支援 Cross-Attn 及直接 YOLO 偵測流程。
          </p>
          <ul className="mt-4 space-y-2 text-xs text-slate-300">
            <li className="flex items-start gap-2 rounded-2xl border border-white/10 bg-[#0B142A] px-3 py-2">
              <span className="mt-1 h-2 w-2 rounded-full bg-primary"></span>
              <span>Cross-Attn：牙位 YOLO 偵測 + 單牙交叉注意力分類。</span>
            </li>
            <li className="flex items-start gap-2 rounded-2xl border border-white/10 bg-[#0B142A] px-3 py-2">
              <span className="mt-1 h-2 w-2 rounded-full bg-accent"></span>
              <span>YOLO 齲齒：直接輸出病灶框，並自動對應 FDI 牙位。</span>
            </li>
          </ul>
          <Link
            href="/models"
            className="mt-5 inline-flex items-center justify-center rounded-full bg-gradient-to-r from-primary to-accent px-4 py-3 text-sm font-semibold text-slate-900 shadow-lg shadow-primary/40"
          >
            開啟模型管理中心
          </Link>
        </div>
      </section>
    </div>
  );
}



