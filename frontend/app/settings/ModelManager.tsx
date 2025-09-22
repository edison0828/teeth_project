"use client";

import { FormEvent, useState } from "react";

import { activateModelConfig, createModelConfig } from "../../lib/api";
import { readToken } from "../../lib/auth-storage";
import type { ModelConfig, ModelConfigCreateRequest } from "../../lib/types";

type ModelManagerProps = {
  initialModels: ModelConfig[];
};

const DEFAULT_FORM: ModelConfigCreateRequest = {
  name: "",
  description: "",
  detector_path: "models/fdi_all seg.pt",
  classifier_path: "models/cross_attn_fdi_camAlignA.pth",
  detector_threshold: 0.25,
  classification_threshold: 0.5,
  max_teeth: 64,
  is_active: false,
};

export default function ModelManager({ initialModels }: ModelManagerProps) {
  const [models, setModels] = useState<ModelConfig[]>(initialModels);
  const [form, setForm] = useState<ModelConfigCreateRequest>(DEFAULT_FORM);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  const activeId = models.find((model) => model.is_active)?.id ?? null;

  const handleActivate = async (modelId: string) => {
    const token = readToken();
    if (!token) {
      setError("請先登入後再調整模型。");
      return;
    }

    setSubmitting(true);
    setError(null);
    setMessage(null);
    try {
      await activateModelConfig(token, modelId);
      setModels((previous) =>
        previous.map((model) => ({
          ...model,
          is_active: model.id === modelId,
        }))
      );
      setMessage("已切換為指定模型。");
    } catch (err) {
      const detail = err instanceof Error ? err.message : "模型啟用失敗。";
      setError(detail);
    } finally {
      setSubmitting(false);
    }
  };

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const token = readToken();
    if (!token) {
      setError("請先登入後再新增模型。");
      return;
    }

    setSubmitting(true);
    setError(null);
    setMessage(null);
    try {
      const payload: ModelConfigCreateRequest = {
        ...form,
        detector_threshold: Number(form.detector_threshold),
        classification_threshold: Number(form.classification_threshold),
        max_teeth: Number(form.max_teeth),
      };
      const created = await createModelConfig(token, payload);
      setModels((previous) => {
        const next = payload.is_active
          ? previous.map((model) => ({ ...model, is_active: false }))
          : previous.slice();
        next.push(created);
        return next;
      });
      setForm({ ...DEFAULT_FORM });
      setMessage("模型設定已建立。");
    } catch (err) {
      const detail = err instanceof Error ? err.message : "新增模型時發生錯誤。";
      setError(detail);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="space-y-6">
      {error && <div className="rounded-xl bg-red-500/10 px-4 py-3 text-xs text-red-300">{error}</div>}
      {message && <div className="rounded-xl bg-primary/10 px-4 py-3 text-xs text-primary-subtle">{message}</div>}

      <div className="space-y-4">
        {models.length === 0 ? (
          <p className="text-sm text-slate-300">尚未建立任何模型設定。</p>
        ) : (
          models.map((model) => (
            <div
              key={model.id}
              className={`rounded-2xl border px-4 py-3 text-sm transition ${
                model.is_active ? "border-primary/60 bg-primary/5" : "border-white/10 bg-white/5"
              }`}
            >
              <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
                <div>
                  <p className="text-base font-semibold text-white">{model.name}</p>
                  {model.description ? <p className="mt-1 text-xs text-slate-400">{model.description}</p> : null}
                  <p className="mt-2 text-xs text-slate-400">
                    Detector: <span className="text-slate-200">{model.detector_path}</span>
                  </p>
                  <p className="mt-1 text-xs text-slate-400">
                    Classifier: <span className="text-slate-200">{model.classifier_path}</span>
                  </p>
                  <p className="mt-1 text-xs text-slate-500">
                    門檻 {model.detector_threshold.toFixed(2)} / {model.classification_threshold.toFixed(2)} · 每張上限 {model.max_teeth} 顆
                  </p>
                  <p className="mt-1 text-xs text-slate-500">更新時間 {new Date(model.updated_at).toLocaleString()}</p>
                </div>
                <div className="flex items-center gap-3 sm:flex-col">
                  <span
                    className={`rounded-full px-3 py-1 text-xs ${
                      model.is_active ? "bg-primary/20 text-primary" : "bg-white/10 text-slate-300"
                    }`}
                  >
                    {model.is_active ? "使用中" : "待命"}
                  </span>
                  <button
                    type="button"
                    className="rounded-full bg-primary px-4 py-2 text-xs font-semibold text-slate-900 disabled:opacity-50"
                    onClick={() => handleActivate(model.id)}
                    disabled={model.is_active || submitting}
                  >
                    {model.is_active ? "目前使用中" : submitting ? "處理中..." : "設為主要模型"}
                  </button>
                </div>
              </div>
            </div>
          ))
        )}
      </div>

      <form
        onSubmit={handleSubmit}
        className="space-y-4 rounded-2xl border border-white/10 bg-[#0B142A] p-4 text-sm text-slate-200"
      >
        <h3 className="text-base font-semibold text-white">新增模型設定</h3>
        <div className="grid gap-4 md:grid-cols-2">
          <label className="flex flex-col gap-1">
            <span className="text-xs uppercase tracking-wide text-slate-400">名稱</span>
            <input
              required
              value={form.name}
              onChange={(event) => setForm((previous) => ({ ...previous, name: event.target.value }))}
              className="rounded-xl border border-white/10 bg-[#050B1C] px-3 py-2"
              placeholder="例如 Cross-Attn Caries"
            />
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-xs uppercase tracking-wide text-slate-400">偵測模型路徑</span>
            <input
              required
              value={form.detector_path}
              onChange={(event) => setForm((previous) => ({ ...previous, detector_path: event.target.value }))}
              className="rounded-xl border border-white/10 bg-[#050B1C] px-3 py-2"
            />
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-xs uppercase tracking-wide text-slate-400">分類模型路徑</span>
            <input
              required
              value={form.classifier_path}
              onChange={(event) => setForm((previous) => ({ ...previous, classifier_path: event.target.value }))}
              className="rounded-xl border border-white/10 bg-[#050B1C] px-3 py-2"
            />
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-xs uppercase tracking-wide text-slate-400">偵測閾值</span>
            <input
              type="number"
              min={0}
              max={1}
              step={0.01}
              value={form.detector_threshold}
              onChange={(event) => setForm((previous) => ({ ...previous, detector_threshold: Number(event.target.value) }))}
              className="rounded-xl border border-white/10 bg-[#050B1C] px-3 py-2"
            />
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-xs uppercase tracking-wide text-slate-400">分類閾值</span>
            <input
              type="number"
              min={0}
              max={1}
              step={0.01}
              value={form.classification_threshold}
              onChange={(event) =>
                setForm((previous) => ({ ...previous, classification_threshold: Number(event.target.value) }))
              }
              className="rounded-xl border border-white/10 bg-[#050B1C] px-3 py-2"
            />
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-xs uppercase tracking-wide text-slate-400">最大牙齒數</span>
            <input
              type="number"
              min={1}
              max={128}
              value={form.max_teeth}
              onChange={(event) => setForm((previous) => ({ ...previous, max_teeth: Number(event.target.value) }))}
              className="rounded-xl border border-white/10 bg-[#050B1C] px-3 py-2"
            />
          </label>
          <label className="md:col-span-2 flex flex-col gap-1">
            <span className="text-xs uppercase tracking-wide text-slate-400">描述</span>
            <textarea
              rows={2}
              value={form.description ?? ""}
              onChange={(event) => setForm((previous) => ({ ...previous, description: event.target.value }))}
              className="rounded-xl border border-white/10 bg-[#050B1C] px-3 py-2"
              placeholder="說明此模型的用途或資料來源"
            />
          </label>
          <label className="flex items-center gap-2 text-xs text-slate-300">
            <input
              type="checkbox"
              checked={form.is_active ?? false}
              onChange={(event) => setForm((previous) => ({ ...previous, is_active: event.target.checked }))}
              className="h-4 w-4 rounded border-white/20 bg-transparent"
            />
            建立後立即啟用此模型
          </label>
        </div>
        <button
          type="submit"
          disabled={submitting}
          className="w-full rounded-full bg-gradient-to-r from-primary to-accent px-4 py-3 text-sm font-semibold text-slate-900 disabled:opacity-50"
        >
          {submitting ? "Saving..." : "新增模型"}
        </button>
      </form>
    </div>
  );
}
