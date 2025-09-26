"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";

import { activateModelConfig, createModelConfig, updateModelConfig } from "../../lib/api";
import { readToken } from "../../lib/auth-storage";
import { useAuth } from "../../contexts/AuthContext";
import type {
  ModelConfig,
  ModelConfigCreateRequest,
  ModelConfigUpdateRequest,
  ModelType,
} from "../../lib/types";

type ModelManagementWorkspaceProps = {
  initialModels: ModelConfig[];
  tokenAvailable: boolean;
};

const MODEL_TYPE_OPTIONS: Array<{
  value: ModelType;
  label: string;
  description: string;
  classifierLabel: string;
  thresholdLabel: string;
}> = [
  {
    value: "cross_attn",
    label: "Tooth classifier pipeline",
    description: "使用牙位 YOLO 偵測搭配 Cross-Attention 單牙分類。",
    classifierLabel: "分類模型路徑",
    thresholdLabel: "分類閾值",
  },
  {
    value: "yolo_caries",
    label: "Direct YOLO caries",
    description: "以齲齒偵測 YOLO 直接輸出病灶並結合牙位偵測。",
    classifierLabel: "齲齒偵測模型路徑",
    thresholdLabel: "齲齒信心閾值",
  },
];

const DEFAULT_FORM: ModelConfigCreateRequest = {
  name: "",
  description: "",
  model_type: "cross_attn",
  detector_path: "models/fdi_all seg.pt",
  classifier_path: "models/cross_attn_fdi_camAlignA.pth",
  detector_threshold: 0.25,
  classification_threshold: 0.5,
  max_teeth: 64,
  is_active: false,
};

function modelTypeMeta(type: ModelType) {
  return MODEL_TYPE_OPTIONS.find((option) => option.value === type) ?? MODEL_TYPE_OPTIONS[0];
}

type ModelCardProps = {
  model: ModelConfig;
  canMutate: boolean;
  onActivate: (modelId: string) => Promise<void>;
  onUpdate: (modelId: string, payload: ModelConfigUpdateRequest) => Promise<void>;
};

function ModelCard({ model, canMutate, onActivate, onUpdate }: ModelCardProps) {
  const [editing, setEditing] = useState(false);
  const [local, setLocal] = useState<ModelConfigUpdateRequest>({
    name: model.name,
    description: model.description ?? "",
    model_type: model.model_type,
    detector_path: model.detector_path,
    classifier_path: model.classifier_path,
    detector_threshold: model.detector_threshold,
    classification_threshold: model.classification_threshold,
    max_teeth: model.max_teeth,
  });
  const [saving, setSaving] = useState(false);
  const meta = modelTypeMeta((local.model_type ?? model.model_type) as ModelType);

  useEffect(() => {
    setLocal({
      name: model.name,
      description: model.description ?? "",
      model_type: model.model_type,
      detector_path: model.detector_path,
      classifier_path: model.classifier_path,
      detector_threshold: model.detector_threshold,
      classification_threshold: model.classification_threshold,
      max_teeth: model.max_teeth,
    });
  }, [model]);

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!canMutate) {
      return;
    }
    setSaving(true);
    try {
      await onUpdate(model.id, {
        ...local,
        detector_threshold: Number(local.detector_threshold),
        classification_threshold: Number(local.classification_threshold),
        max_teeth: Number(local.max_teeth),
      });
      setEditing(false);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div
      className={`rounded-3xl border px-5 py-4 transition ${
        model.is_active ? "border-primary/60 bg-primary/5" : "border-white/10 bg-white/5"
      }`}
    >
      <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
        <div className="space-y-1">
          <span className="inline-flex items-center rounded-full bg-white/10 px-2 py-0.5 text-[10px] uppercase tracking-wide text-slate-400">
            {meta.label}
          </span>
          <h3 className="text-lg font-semibold text-white">{model.name}</h3>
          {model.description ? <p className="text-sm text-slate-300">{model.description}</p> : null}
          <p className="text-xs text-slate-400">
            偵測模型：<span className="text-slate-200">{model.detector_path}</span>
          </p>
          <p className="text-xs text-slate-400">
            {meta.classifierLabel}：<span className="text-slate-200">{model.classifier_path}</span>
          </p>
          <p className="text-xs text-slate-500">
            門檻 {model.detector_threshold.toFixed(2)} / {model.classification_threshold.toFixed(2)} · 上限 {model.max_teeth} 顆
          </p>
          <p className="text-xs text-slate-500">更新時間 {new Date(model.updated_at).toLocaleString()}</p>
        </div>
        <div className="flex items-center gap-3 md:flex-col">
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
            onClick={() => onActivate(model.id)}
            disabled={!canMutate || model.is_active}
          >
            {model.is_active ? "目前使用中" : "設為主要模型"}
          </button>
          <button
            type="button"
            className="text-xs text-primary-subtle underline-offset-2 hover:underline disabled:opacity-50"
            onClick={() => setEditing((value) => !value)}
            disabled={!canMutate}
          >
            {editing ? "取消" : "編輯設定"}
          </button>
        </div>
      </div>
      {editing ? (
        <form onSubmit={handleSubmit} className="mt-4 grid gap-3 rounded-2xl border border-white/10 bg-[#0B142A] p-4 text-sm">
          <div className="grid gap-3 md:grid-cols-2">
            <label className="flex flex-col gap-1">
              <span className="text-xs uppercase tracking-wide text-slate-400">名稱</span>
              <input
                value={local.name ?? ""}
                onChange={(event) => setLocal((prev) => ({ ...prev, name: event.target.value }))}
                className="rounded-xl border border-white/10 bg-[#050B1C] px-3 py-2"
              />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs uppercase tracking-wide text-slate-400">模型類型</span>
              <select
                value={local.model_type ?? model.model_type}
                onChange={(event) =>
                  setLocal((prev) => ({ ...prev, model_type: event.target.value as ModelType }))
                }
                className="rounded-xl border border-white/10 bg-[#050B1C] px-3 py-2"
              >
                {MODEL_TYPE_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs uppercase tracking-wide text-slate-400">偵測模型路徑</span>
              <input
                value={local.detector_path ?? ""}
                onChange={(event) => setLocal((prev) => ({ ...prev, detector_path: event.target.value }))}
                className="rounded-xl border border-white/10 bg-[#050B1C] px-3 py-2"
              />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs uppercase tracking-wide text-slate-400">{meta.classifierLabel}</span>
              <input
                value={local.classifier_path ?? ""}
                onChange={(event) => setLocal((prev) => ({ ...prev, classifier_path: event.target.value }))}
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
                value={local.detector_threshold ?? 0}
                onChange={(event) =>
                  setLocal((prev) => ({ ...prev, detector_threshold: Number(event.target.value) }))
                }
                className="rounded-xl border border-white/10 bg-[#050B1C] px-3 py-2"
              />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs uppercase tracking-wide text-slate-400">{meta.thresholdLabel}</span>
              <input
                type="number"
                min={0}
                max={1}
                step={0.01}
                value={local.classification_threshold ?? 0}
                onChange={(event) =>
                  setLocal((prev) => ({ ...prev, classification_threshold: Number(event.target.value) }))
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
                value={local.max_teeth ?? model.max_teeth}
                onChange={(event) => setLocal((prev) => ({ ...prev, max_teeth: Number(event.target.value) }))}
                className="rounded-xl border border-white/10 bg-[#050B1C] px-3 py-2"
              />
            </label>
            <label className="md:col-span-2 flex flex-col gap-1">
              <span className="text-xs uppercase tracking-wide text-slate-400">描述</span>
              <textarea
                rows={2}
                value={local.description ?? ""}
                onChange={(event) => setLocal((prev) => ({ ...prev, description: event.target.value }))}
                className="rounded-xl border border-white/10 bg-[#050B1C] px-3 py-2"
              />
            </label>
          </div>
          <div className="flex items-center justify-end gap-3">
            <button
              type="button"
              className="rounded-full border border-white/20 px-4 py-2 text-xs text-slate-300"
              onClick={() => setEditing(false)}
              disabled={saving}
            >
              取消
            </button>
            <button
              type="submit"
              className="rounded-full bg-primary px-4 py-2 text-xs font-semibold text-slate-900 disabled:opacity-50"
              disabled={saving}
            >
              {saving ? "儲存中..." : "儲存調整"}
            </button>
          </div>
        </form>
      ) : null}
    </div>
  );
}

export default function ModelManagementWorkspace({
  initialModels,
  tokenAvailable,
}: ModelManagementWorkspaceProps) {
  const { guestMode } = useAuth();
  const [models, setModels] = useState<ModelConfig[]>(initialModels);
  const [form, setForm] = useState<ModelConfigCreateRequest>({ ...DEFAULT_FORM });
  const [filterType, setFilterType] = useState<ModelType | "all">("all");
  const [submitting, setSubmitting] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const activeModel = useMemo(() => models.find((model) => model.is_active) ?? null, [models]);
  const counts = useMemo(() => {
    return models.reduce(
      (acc, model) => {
        acc[model.model_type] = (acc[model.model_type] ?? 0) + 1;
        return acc;
      },
      {} as Record<ModelType, number>
    );
  }, [models]);

  const visibleModels = useMemo(() => {
    if (filterType === "all") {
      return models;
    }
    return models.filter((model) => model.model_type === filterType);
  }, [models, filterType]);

  const canMutate = tokenAvailable && !guestMode;

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

  const handleCreate = async (event: FormEvent<HTMLFormElement>) => {
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
        const next = payload.is_active ? previous.map((model) => ({ ...model, is_active: false })) : previous.slice();
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

  const handleUpdate = async (modelId: string, payload: ModelConfigUpdateRequest) => {
    const token = readToken();
    if (!token) {
      setError("請先登入後再更新模型。");
      return;
    }
    try {
      const updated = await updateModelConfig(token, modelId, payload);
      setModels((previous) => previous.map((model) => (model.id === updated.id ? updated : model)));
      setMessage("模型設定已更新。");
    } catch (err) {
      const detail = err instanceof Error ? err.message : "更新模型時發生錯誤。";
      setError(detail);
    }
  };

  return (
    <div className="space-y-6">
      {error && <div className="rounded-xl bg-red-500/10 px-4 py-3 text-xs text-red-300">{error}</div>}
      {message && <div className="rounded-xl bg-primary/10 px-4 py-3 text-xs text-primary-subtle">{message}</div>}
      {guestMode && (
        <div className="rounded-xl border border-white/10 bg-white/5 px-4 py-3 text-xs text-slate-300">
          目前為示範模式，僅能瀏覽預設模型設定。登入後即可儲存自訂模型。
        </div>
      )}

      <section className="grid gap-6 lg:grid-cols-[2fr,1fr]">
        <div className="rounded-3xl bg-white/5 p-6 shadow-card">
          <h2 className="text-lg font-semibold text-white">模型總覽</h2>
          <p className="mt-1 text-sm text-slate-300">
            共 {models.length} 組設定 · Cross-Attn {counts.cross_attn ?? 0} · YOLO {counts.yolo_caries ?? 0}
          </p>
          <div className="mt-4 rounded-2xl border border-white/10 bg-[#0B142A] p-4 text-sm text-slate-200">
            <p className="text-xs uppercase tracking-wide text-slate-400">當前使用</p>
            <p className="mt-1 text-lg font-semibold text-white">
              {activeModel ? activeModel.name : guestMode ? "Demo 模型" : "尚未啟用"}
            </p>
            <p className="mt-1 text-xs text-slate-400">
              {activeModel
                ? modelTypeMeta(activeModel.model_type).description
                : "請建立並啟用一組模型設定。"}
            </p>
          </div>
          <div className="mt-4 flex flex-wrap items-center gap-3 text-xs text-slate-300">
            <label className="flex items-center gap-2 rounded-full border border-white/10 bg-[#0B142A] px-3 py-2">
              <span>篩選</span>
              <select
                value={filterType}
                onChange={(event) => setFilterType(event.target.value as ModelType | "all")}
                className="bg-transparent text-slate-100 focus:outline-none"
              >
                <option value="all">全部</option>
                {MODEL_TYPE_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>
            <span className="rounded-full border border-white/10 px-3 py-2">{visibleModels.length} 組符合條件</span>
          </div>
        </div>

        <form
          onSubmit={handleCreate}
          className="space-y-3 rounded-3xl border border-white/10 bg-[#0B142A] p-5 text-sm text-slate-200"
        >
          <h2 className="text-lg font-semibold text-white">新增模型設定</h2>
          <p className="text-xs text-slate-400">
            指定偵測與分類權重，並設定閾值與是否立即啟用。
          </p>
          <div className="grid gap-3 md:grid-cols-2">
            <label className="flex flex-col gap-1">
              <span className="text-xs uppercase tracking-wide text-slate-400">名稱</span>
              <input
                required
                value={form.name}
                onChange={(event) => setForm((prev) => ({ ...prev, name: event.target.value }))}
                className="rounded-xl border border-white/10 bg-[#050B1C] px-3 py-2"
                placeholder="例如 Cross-Attn CamAlignA"
              />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs uppercase tracking-wide text-slate-400">模型類型</span>
              <select
                value={form.model_type}
                onChange={(event) => {
                  const nextType = event.target.value as ModelType;
                  const meta = modelTypeMeta(nextType);
                  setForm((prev) => ({
                    ...prev,
                    model_type: nextType,
                    classifier_path:
                      nextType === "yolo_caries"
                        ? "models/yolo_caries.pt"
                        : "models/cross_attn_fdi_camAlignA.pth",
                  }));
                }}
                className="rounded-xl border border-white/10 bg-[#050B1C] px-3 py-2"
              >
                {MODEL_TYPE_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs uppercase tracking-wide text-slate-400">偵測模型路徑</span>
              <input
                required
                value={form.detector_path}
                onChange={(event) => setForm((prev) => ({ ...prev, detector_path: event.target.value }))}
                className="rounded-xl border border-white/10 bg-[#050B1C] px-3 py-2"
              />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs uppercase tracking-wide text-slate-400">{modelTypeMeta(form.model_type).classifierLabel}</span>
              <input
                required
                value={form.classifier_path}
                onChange={(event) => setForm((prev) => ({ ...prev, classifier_path: event.target.value }))}
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
                onChange={(event) => setForm((prev) => ({ ...prev, detector_threshold: Number(event.target.value) }))}
                className="rounded-xl border border-white/10 bg-[#050B1C] px-3 py-2"
              />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs uppercase tracking-wide text-slate-400">{modelTypeMeta(form.model_type).thresholdLabel}</span>
              <input
                type="number"
                min={0}
                max={1}
                step={0.01}
                value={form.classification_threshold}
                onChange={(event) =>
                  setForm((prev) => ({ ...prev, classification_threshold: Number(event.target.value) }))
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
                onChange={(event) => setForm((prev) => ({ ...prev, max_teeth: Number(event.target.value) }))}
                className="rounded-xl border border-white/10 bg-[#050B1C] px-3 py-2"
              />
            </label>
            <label className="md:col-span-2 flex flex-col gap-1">
              <span className="text-xs uppercase tracking-wide text-slate-400">描述</span>
              <textarea
                rows={2}
                value={form.description ?? ""}
                onChange={(event) => setForm((prev) => ({ ...prev, description: event.target.value }))}
                className="rounded-xl border border-white/10 bg-[#050B1C] px-3 py-2"
                placeholder="說明此模型的用途或資料來源"
              />
            </label>
            <label className="flex items-center gap-2 text-xs text-slate-300">
              <input
                type="checkbox"
                checked={form.is_active ?? false}
                onChange={(event) => setForm((prev) => ({ ...prev, is_active: event.target.checked }))}
                className="h-4 w-4 rounded border-white/20 bg-transparent"
              />
              建立後立即啟用此模型
            </label>
          </div>
          <button
            type="submit"
            disabled={submitting || !canMutate}
            className="w-full rounded-full bg-gradient-to-r from-primary to-accent px-4 py-3 text-sm font-semibold text-slate-900 disabled:opacity-50"
          >
            {submitting ? "Saving..." : "新增模型"}
          </button>
        </form>
      </section>

      <section className="space-y-4">
        {visibleModels.length === 0 ? (
          <p className="text-sm text-slate-300">尚未建立符合條件的模型設定。</p>
        ) : (
          visibleModels.map((model) => (
            <ModelCard
              key={model.id}
              model={model}
              canMutate={canMutate}
              onActivate={handleActivate}
              onUpdate={handleUpdate}
            />
          ))
        )}
      </section>
    </div>
  );
}
