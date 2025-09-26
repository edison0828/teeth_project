import { fetchModels } from "../../lib/api";
import { readServerToken } from "../../lib/server-auth";
import ModelManagementWorkspace from "./ModelManagementWorkspace";

export default async function ModelsPage() {
  const token = readServerToken();
  const models = await fetchModels(token ?? undefined);

  return (
    <div className="space-y-8">
      <header className="rounded-3xl bg-white/5 px-8 py-10 text-white shadow-card">
        <p className="text-xs uppercase tracking-[0.3em] text-primary">Model Hub</p>
        <h1 className="mt-4 text-3xl font-semibold">模型管理中心</h1>
        <p className="mt-3 text-sm text-slate-300 lg:max-w-3xl">
          檢視、切換與調整牙位與齲齒偵測模型。支援多種推論流程，協助團隊快速驗證與部署新權重。
        </p>
      </header>

      <ModelManagementWorkspace initialModels={models} tokenAvailable={Boolean(token)} />
    </div>
  );
}
