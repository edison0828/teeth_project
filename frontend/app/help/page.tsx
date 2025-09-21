import Link from "next/link";

import { readServerToken } from "../../lib/server-auth";

const faqs = [
  {
    question: "如何建立新的 AI 分析?",
    answer: "至影像上傳頁面選擇患者後上傳影像，系統將自動排程分析並於通知中心提供結果。"
  },
  {
    question: "忘記密碼怎麼辦?",
    answer: "請洽系統管理員或醫院 IT 部門重設密碼，目前版本尚未開放自助重設功能。"
  },
  {
    question: "模型準確率如何量測?",
    answer: "在 Dashboard 可查看最近四週的模型整體偵測率，也可匯出分析報告進行審核。"
  }
];

export default async function HelpPage() {
  const token = readServerToken();

  if (!token) {
    return null;
  }

  return (
    <div className="space-y-8">
      <header className="rounded-3xl bg-white/5 px-8 py-10 text-white shadow-card">
        <p className="text-xs uppercase tracking-[0.3em] text-primary">Help Center</p>
        <h1 className="mt-4 text-3xl font-semibold">支援與常見問題</h1>
        <p className="mt-3 text-sm text-slate-300 lg:max-w-3xl">
          使用 DentaMind AI 過程遇到問題嗎？以下提供快速指引與常見問答，也可直接提交支援單協助您排解狀況。
        </p>
      </header>

      <section className="rounded-3xl bg-white/5 p-6 shadow-card">
        <h2 className="text-lg font-semibold text-white">常見問題</h2>
        <dl className="mt-6 space-y-4 text-sm text-slate-200">
          {faqs.map((item) => (
            <div key={item.question} className="rounded-2xl border border-white/10 bg-[#0B142A] p-4">
              <dt className="text-base font-semibold text-white">{item.question}</dt>
              <dd className="mt-2 text-slate-300">{item.answer}</dd>
            </div>
          ))}
        </dl>
      </section>

      <section className="rounded-3xl bg-gradient-to-br from-primary/20 to-accent/20 p-6 text-sm text-white">
        <h2 className="text-lg font-semibold">仍需要協助嗎？</h2>
        <p className="mt-2 text-slate-100">
          您可以提交支援單或預約線上培訓，我們的臨床專家將協助完成部署與教育訓練。
        </p>
        <div className="mt-4 flex flex-wrap gap-3">
          <Link href="#support" className="rounded-full bg-white/20 px-4 py-2 text-sm font-semibold text-white hover:bg-white/30">
            提交支援單
          </Link>
          <Link href="#training" className="rounded-full bg-white/10 px-4 py-2 text-sm font-semibold text-white hover:bg-white/20">
            預約培訓
          </Link>
        </div>
      </section>
    </div>
  );
}