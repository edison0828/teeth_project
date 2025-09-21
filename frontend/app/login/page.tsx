import LoginForm from "./LoginForm";

const highlights = [
  "跨模型分析一次整合",
  "AI 協助撰寫口腔影像報告",
  "全口病歷追蹤與即時提醒"
];

export default function LoginPage() {
  return (
    <div className="mx-auto flex max-w-6xl flex-col gap-10 pt-12 text-white lg:flex-row lg:items-center lg:pt-0">
      <section className="flex-1 text-center lg:text-left">
        <p className="inline-flex items-center rounded-full bg-primary/10 px-4 py-1 text-xs font-semibold uppercase tracking-[0.3em] text-primary">
          DentaMind AI Suite
        </p>
        <h1 className="mt-6 text-4xl font-semibold leading-tight lg:text-5xl">
          登入您的智慧口腔影像工作站
        </h1>
        <p className="mt-4 text-base text-slate-300 lg:max-w-xl">
          即時串接放射影像、病歷與 AI 分析結果，協助醫師快速完成診斷與溝通。登入後即可檢視待處理病例、產出報告與追蹤模型效能。
        </p>
        <ul className="mt-6 flex flex-col gap-3 text-sm text-slate-300">
          {highlights.map((item) => (
            <li key={item} className="inline-flex items-center gap-3 rounded-2xl border border-white/5 bg-white/5 px-4 py-3 backdrop-blur">
              <span className="flex h-8 w-8 items-center justify-center rounded-full bg-primary/20 text-sm font-semibold text-primary">
                ✓
              </span>
              <span>{item}</span>
            </li>
          ))}
        </ul>
      </section>
      <div className="flex-1 lg:max-w-sm">
        <LoginForm />
      </div>
    </div>
  );
}
