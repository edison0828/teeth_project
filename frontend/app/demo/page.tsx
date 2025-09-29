"use client";

import { useEffect, useMemo, useState, type ChangeEvent } from "react";
import { fetchDemoSamples, submitDemoInference } from "../../lib/api";
import type {
  DemoInferenceResult,
  DemoSampleSummary,
  DemoToothFinding,
} from "../../lib/types";
import { resolveMediaUrl } from "../../lib/media";

interface PreviewSource {
  url: string;
  disposable: boolean;
}

function formatProbability(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function pickDefaultFinding(findings: DemoToothFinding[]): DemoToothFinding | null {
  if (findings.length === 0) {
    return null;
  }
  const positive = findings.find((item) => item.pred);
  return positive ?? findings[0];
}

export default function DemoPage(): JSX.Element {
  const [samples, setSamples] = useState<DemoSampleSummary[]>([]);
  const [selectedSampleId, setSelectedSampleId] = useState<string | null>(null);
  const [result, setResult] = useState<DemoInferenceResult | null>(null);
  const [activeFinding, setActiveFinding] = useState<DemoToothFinding | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [preview, setPreview] = useState<PreviewSource | null>(null);
  const [inputKey, setInputKey] = useState(() => Date.now());

  useEffect(() => {
    void (async () => {
      try {
        const response = await fetchDemoSamples();
        setSamples(response.items);
      } catch (err) {
        console.error("Failed to load demo samples", err);
        setError("無法載入預設樣本，將改用內建範例資料。");
      }
    })();
  }, []);

  useEffect(() => {
    return () => {
      if (preview?.disposable) {
        URL.revokeObjectURL(preview.url);
      }
    };
  }, [preview]);

  const viewSources = useMemo(() => {
    const overlay = result ? resolveMediaUrl(result.overlay_url) : null;
    const cam = activeFinding?.cam_path
      ? resolveMediaUrl(activeFinding.cam_path)
      : null;

    const views: Array<{ key: string; label: string; url: string }> = [];

    if (preview?.url) {
      views.push({ key: "original", label: "原始影像", url: preview.url });
    }

    if (overlay) {
      views.push({ key: "overlay", label: "推論疊圖", url: overlay });
    }

    if (cam) {
      views.push({
        key: "cam",
        label: activeFinding ? `Grad-CAM（FDI ${activeFinding.fdi}）` : "Grad-CAM",
        url: cam,
      });
    }

    return views;
  }, [preview, result, activeFinding]);

  const [activeView, setActiveView] = useState<string>("original");

  useEffect(() => {
    if (viewSources.length === 0) {
      setActiveView("original");
      return;
    }
    if (!viewSources.some((view) => view.key === activeView)) {
      setActiveView(viewSources[0].key);
    }
  }, [viewSources, activeView]);

  const updatePreview = (url: string | null, disposable = false): void => {
    setPreview((current) => {
      if (current?.disposable) {
        URL.revokeObjectURL(current.url);
      }
      if (!url) {
        return null;
      }
      return { url, disposable };
    });
  };

  const resetState = (): void => {
    setResult(null);
    setActiveFinding(null);
    setActiveView("original");
  };

  const handleSample = async (sample: DemoSampleSummary): Promise<void> => {
    setError(null);
    setSelectedSampleId(sample.id);
    updatePreview(resolveMediaUrl(sample.image_path) ?? sample.image_path);
    setLoading(true);
    resetState();

    try {
      const response = await submitDemoInference({ sampleId: sample.id });
      setResult(response);
      const finding = pickDefaultFinding(response.findings);
      setActiveFinding(finding);
      if (finding?.cam_path) {
        setActiveView("cam");
      } else if (response.overlay_url) {
        setActiveView("overlay");
      }
    } catch (err) {
      console.error("Demo inference failed", err);
      setError(err instanceof Error ? err.message : "推論失敗");
    } finally {
      setLoading(false);
    }
  };

  const handleUpload = async (event: ChangeEvent<HTMLInputElement>): Promise<void> => {
    if (!event.target.files || event.target.files.length === 0) {
      return;
    }

    const file = event.target.files[0];
    const localUrl = URL.createObjectURL(file);

    setError(null);
    setSelectedSampleId(null);
    updatePreview(localUrl, true);
    setLoading(true);
    resetState();

    try {
      const response = await submitDemoInference({ file });
      setResult(response);
      const finding = pickDefaultFinding(response.findings);
      setActiveFinding(finding);
      if (finding?.cam_path) {
        setActiveView("cam");
      } else if (response.overlay_url) {
        setActiveView("overlay");
      }
    } catch (err) {
      console.error("Demo upload failed", err);
      setError(err instanceof Error ? err.message : "上傳失敗");
    } finally {
      setLoading(false);
      setInputKey(Date.now());
    }
  };

  const activePreview = viewSources.find((view) => view.key === activeView);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <div className="mx-auto flex min-h-screen max-w-6xl flex-col px-6 py-10">
        <header className="space-y-4 text-center lg:text-left">
          <p className="text-sm uppercase tracking-[0.3em] text-cyan-300">Cross-Attention Grad-CAM Demo</p>
          <h1 className="text-4xl font-bold tracking-tight text-white sm:text-5xl">
            雲端即時體驗交叉注意力齲齒可視化
          </h1>
          <p className="text-base leading-relaxed text-slate-300 sm:text-lg">
            上傳全口 X 光或選擇預載樣本，即可呼叫 FastAPI 推論服務，查看逐齒置信度、偵測框與 Grad-CAM 熱力圖。
          </p>
        </header>

        <div className="mt-10 grid flex-1 gap-8 lg:grid-cols-[320px,1fr]">
          <aside className="space-y-6 rounded-2xl border border-slate-800 bg-slate-900/40 p-6 shadow-xl shadow-slate-900/40 backdrop-blur">
            <div className="space-y-3">
              <h2 className="text-xl font-semibold text-white">預設樣本</h2>
              <p className="text-sm text-slate-300">
                專案會自動掃描 <code className="rounded bg-slate-800 px-1.5 py-0.5 text-xs">demo_backend/static/samples</code> 內的影像。
                點擊即可立即送出推論請求。
              </p>
              <div className="space-y-3">
                {samples.map((sample) => {
                  const previewUrl = resolveMediaUrl(sample.image_path) ?? sample.image_path;
                  const isActive = sample.id === selectedSampleId;
                  return (
                    <button
                      key={sample.id}
                      type="button"
                      onClick={() => void handleSample(sample)}
                      disabled={loading}
                      className={`group w-full rounded-xl border p-3 text-left transition hover:border-cyan-400/80 hover:bg-slate-800/70 focus:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400/80 ${
                        isActive ? "border-cyan-400/80 bg-slate-800/70" : "border-slate-800/80 bg-slate-900/30"
                      } disabled:cursor-not-allowed disabled:opacity-60`}
                    >
                      {previewUrl ? (
                        // eslint-disable-next-line @next/next/no-img-element
                        <img
                          src={previewUrl}
                          alt={sample.title}
                          className="mb-3 h-32 w-full rounded-lg object-cover"
                        />
                      ) : null}
                      <h3 className="text-base font-medium text-white">{sample.title}</h3>
                      <p className="mt-1 text-xs text-slate-300/80">{sample.description}</p>
                    </button>
                  );
                })}
                {samples.length === 0 ? (
                  <p className="text-sm text-slate-400">尚未放入任何示範影像。</p>
                ) : null}
              </div>
            </div>

            <div className="space-y-3">
              <h2 className="text-xl font-semibold text-white">上傳影像</h2>
              <p className="text-sm text-slate-300">
                支援單張 PNG 或 JPG。推論完成後，系統會產生疊圖與 CSV，並於頁面右側顯示逐齒結果。
              </p>
              <label className="flex h-32 cursor-pointer items-center justify-center rounded-xl border border-dashed border-cyan-400/60 bg-slate-900/30 text-center text-sm font-medium text-cyan-200 transition hover:bg-slate-800/50">
                <input
                  key={inputKey}
                  type="file"
                  accept="image/png,image/jpeg,image/jpg"
                  onChange={handleUpload}
                  className="hidden"
                  disabled={loading}
                />
                {loading ? "處理中..." : "點擊選擇影像或拖曳檔案"}
              </label>
            </div>

            <div className="rounded-xl border border-slate-800/60 bg-slate-900/20 p-4 text-xs text-slate-300">
              <p className="font-semibold text-white">使用小提醒</p>
              <ul className="mt-2 space-y-1 list-disc pl-4">
                <li>確保模型權重已放入 <code>models/</code> 目錄。</li>
                <li>若使用 CPU，請設定 <code>DEMO_DEVICE=cpu</code>。</li>
                <li>推論結果將儲存在 <code>demo_backend/outputs/</code>。</li>
              </ul>
            </div>
          </aside>

          <section className="flex flex-col gap-6">
            {error ? (
              <p className="rounded-xl border border-red-500/50 bg-red-500/10 px-4 py-3 text-sm text-red-200">{error}</p>
            ) : null}
            {loading ? (
              <p className="text-sm text-cyan-200">模型推論中，請稍候...</p>
            ) : null}

            <div className="rounded-2xl border border-slate-800 bg-slate-900/30 p-6 shadow-xl shadow-slate-900/30">
              <h2 className="text-2xl font-semibold text-white">視覺化結果</h2>
              <p className="mt-1 text-sm text-slate-300">
                依序切換原圖、推論疊圖與 Grad-CAM 熱力圖，觀察模型在各齒位的注意力分布。
              </p>

              <div className="mt-4 flex flex-wrap gap-3">
                {viewSources.map((view) => (
                  <button
                    key={view.key}
                    type="button"
                    onClick={() => setActiveView(view.key)}
                    className={`rounded-full px-4 py-1.5 text-sm font-medium transition focus:outline-none focus-visible:ring-2 focus-visible:ring-cyan-300 ${
                      activeView === view.key
                        ? "bg-cyan-400 text-slate-900"
                        : "bg-slate-800 text-slate-200 hover:bg-slate-700"
                    }`}
                  >
                    {view.label}
                  </button>
                ))}
              </div>

              <div className="mt-4 min-h-[320px] overflow-hidden rounded-xl border border-slate-800 bg-slate-950/60">
                {activePreview ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={activePreview.url}
                    alt={activePreview.label}
                    className="h-full w-full object-contain"
                  />
                ) : (
                  <div className="flex h-full items-center justify-center px-6 py-12 text-sm text-slate-400">
                    尚未選擇樣本或上傳影像。
                  </div>
                )}
              </div>

              {result ? (
                <div className="mt-6 space-y-4">
                  <div className="flex flex-col gap-2 text-sm text-slate-300 lg:flex-row lg:items-center lg:justify-between">
                    <div>
                      <span className="font-semibold text-white">請求 ID：</span>
                      <span className="font-mono text-cyan-200">{result.request_id}</span>
                    </div>
                    <div className="text-xs text-slate-400">
                      推論產出與 CSV 可透過 <code>/demo-outputs</code> 路徑下載。
                    </div>
                  </div>
                  {result.warnings.length > 0 ? (
                    <ul className="space-y-1 rounded-xl border border-yellow-400/40 bg-yellow-400/10 p-3 text-xs text-yellow-200">
                      {result.warnings.map((warning) => (
                        <li key={warning}>{warning}</li>
                      ))}
                    </ul>
                  ) : null}

                  <div className="overflow-x-auto">
                    <table className="min-w-full divide-y divide-slate-800 text-sm">
                      <thead>
                        <tr className="bg-slate-900/60 text-left text-slate-200">
                          <th className="px-3 py-2 font-medium">FDI</th>
                          <th className="px-3 py-2 font-medium">蛀牙機率</th>
                          <th className="px-3 py-2 font-medium">門檻</th>
                          <th className="px-3 py-2 font-medium">判定</th>
                          <th className="px-3 py-2 font-medium">偵測框</th>
                          <th className="px-3 py-2 font-medium">Grad-CAM</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800/60">
                        {result.findings.map((finding) => {
                          const isSelected = activeFinding?.fdi === finding.fdi &&
                            activeFinding?.bbox.x1 === finding.bbox.x1 &&
                            activeFinding?.bbox.y1 === finding.bbox.y1;
                          const camUrl = finding.cam_path ? resolveMediaUrl(finding.cam_path) : null;
                          return (
                            <tr
                              key={`${result.request_id}-${finding.fdi}-${finding.bbox.x1}-${finding.bbox.y1}`}
                              onClick={() => {
                                setActiveFinding(finding);
                                if (finding.cam_path) {
                                  setActiveView("cam");
                                }
                              }}
                              className={`cursor-pointer transition hover:bg-slate-900/60 ${
                                isSelected ? "bg-slate-900/80" : ""
                              }`}
                            >
                              <td className="px-3 py-2 font-semibold text-white">{finding.fdi}</td>
                              <td className="px-3 py-2 text-slate-200">{formatProbability(finding.prob_caries)}</td>
                              <td className="px-3 py-2 text-slate-200">{formatProbability(finding.thr_used)}</td>
                              <td className="px-3 py-2">
                                <span
                                  className={`rounded-full px-3 py-1 text-xs font-semibold ${
                                    finding.pred
                                      ? "bg-red-500/20 text-red-200"
                                      : "bg-slate-800 text-slate-200"
                                  }`}
                                >
                                  {finding.pred ? "疑似蛀牙" : "正常"}
                                </span>
                              </td>
                              <td className="px-3 py-2 font-mono text-xs text-slate-300">
                                [{finding.bbox.x1}, {finding.bbox.y1}] → [{finding.bbox.x2}, {finding.bbox.y2}]
                              </td>
                              <td className="px-3 py-2">
                                {camUrl ? (
                                  // eslint-disable-next-line @next/next/no-img-element
                                  <img
                                    src={camUrl}
                                    alt={`Grad-CAM ${finding.fdi}`}
                                    className="h-20 w-20 rounded-lg object-cover"
                                  />
                                ) : (
                                  <span className="text-xs text-slate-400">無可用熱力圖</span>
                                )}
                              </td>
                            </tr>
                          );
                        })}
                        {result.findings.length === 0 ? (
                          <tr>
                            <td colSpan={6} className="px-3 py-4 text-center text-sm text-slate-400">
                              無任何牙齒偵測結果。
                            </td>
                          </tr>
                        ) : null}
                      </tbody>
                    </table>
                  </div>
                </div>
              ) : null}
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
