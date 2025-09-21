import { readServerToken } from "../../lib/server-auth";

const demoNotifications = [
  {
    id: "notify-1",
    title: "新分析報告已完成",
    body: "案例 AN-901 的報告已生成，等待醫師確認。",
    createdAt: new Date().toISOString(),
    level: "info"
  },
  {
    id: "notify-2",
    title: "模型版本更新",
    body: "Caries Detector v1.3.0 已部署，精準度提升 2%。",
    createdAt: new Date().toISOString(),
    level: "success"
  },
  {
    id: "notify-3",
    title: "影像上傳失敗",
    body: "患者 P84021 的 CBCT 影像上傳中斷，請重新嘗試。",
    createdAt: new Date().toISOString(),
    level: "warning"
  }
];

const levelStyles: Record<string, string> = {
  info: "bg-primary/20 text-primary",
  success: "bg-emerald-500/20 text-emerald-300",
  warning: "bg-amber-500/20 text-amber-200",
  danger: "bg-red-500/20 text-red-300"
};

export default async function NotificationsPage() {
  const token = readServerToken();

  if (!token) {
    return null;
  }

  return (
    <div className="space-y-8">
      <header className="rounded-3xl bg-white/5 px-8 py-10 text-white shadow-card">
        <p className="text-xs uppercase tracking-[0.3em] text-primary">Notifications</p>
        <h1 className="mt-4 text-3xl font-semibold">通知中心</h1>
        <p className="mt-3 text-sm text-slate-300 lg:max-w-3xl">
          追蹤 AI 模型、影像分析及平台系統的即時提醒。高優先權通知將優先顯示，協助團隊快速掌握狀態。
        </p>
      </header>

      <section className="rounded-3xl bg-white/5 p-6 shadow-card">
        <h2 className="text-lg font-semibold text-white">最近提醒</h2>
        <ul className="mt-6 space-y-4 text-sm">
          {demoNotifications.map((notification) => (
            <li key={notification.id} className="rounded-2xl border border-white/5 bg-white/5 p-4">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <div className="flex items-center gap-3">
                    <span className={`rounded-full px-3 py-1 text-xs font-semibold ${levelStyles[notification.level] ?? levelStyles.info}`}>
                      {notification.level.toUpperCase()}
                    </span>
                    <span className="text-xs text-slate-400">
                      {new Date(notification.createdAt).toLocaleString()}
                    </span>
                  </div>
                  <p className="mt-2 text-base font-semibold text-white">{notification.title}</p>
                  <p className="mt-1 text-slate-300">{notification.body}</p>
                </div>
                <button className="rounded-full border border-white/10 px-3 py-2 text-xs text-slate-300 hover:bg-white/10">
                  標記已讀
                </button>
              </div>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}