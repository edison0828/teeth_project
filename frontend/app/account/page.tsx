import AccountPanel from "./AccountPanel";

export default function AccountPage() {
  return (
    <div className="mx-auto max-w-5xl">
      <header className="rounded-3xl bg-white/5 px-8 py-10 text-white shadow-card">
        <p className="text-xs uppercase tracking-[0.3em] text-primary">Profile</p>
        <h1 className="mt-4 text-3xl font-semibold">個人資料與安全設定</h1>
        <p className="mt-3 text-sm text-slate-300 lg:max-w-2xl">
          管理您的帳號資訊、變更登入密碼並檢視平台的安全狀態。所有變更會即時同步到 AI 分析儀表板與病歷紀錄。
        </p>
      </header>
      <AccountPanel />
    </div>
  );
}
