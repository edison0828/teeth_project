import { clsx } from "clsx";

const menu = ["Patient Management", "Image Upload", "Settings", "Help"];

export default function TopNav() {
  return (
    <header className="sticky top-0 z-20 flex items-center justify-between border-b border-white/5 bg-[#0B142A]/80 px-6 py-5 backdrop-blur lg:px-10">
      <div className="hidden items-center gap-6 text-sm font-medium text-slate-300 md:flex">
        {menu.map((item) => (
          <span
            key={item}
            className={clsx(
              "cursor-pointer rounded-full px-4 py-2 transition-colors",
              item === "Patient Management" ? "bg-white/10 text-white" : "hover:bg-white/5"
            )}
          >
            {item}
          </span>
        ))}
      </div>
      <div className="ml-auto flex items-center gap-6 text-sm">
        <div className="hidden items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-2 text-slate-300 md:flex">
          <span className="text-xs uppercase tracking-wide text-slate-400">Models</span>
          <span className="rounded-full bg-primary/20 px-2 py-0.5 text-primary-subtle">4 active</span>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-right text-xs text-slate-400">
            <span className="block text-sm font-semibold text-white">Dr. Lee</span>
            Lead Radiologist
          </span>
          <span className="flex h-10 w-10 items-center justify-center rounded-full bg-gradient-to-br from-primary to-accent text-sm font-semibold text-slate-900">
            DL
          </span>
        </div>
      </div>
    </header>
  );
}
