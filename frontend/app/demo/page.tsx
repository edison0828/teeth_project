"use client";

import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
  type PointerEvent,
} from "react";
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

function pickDefaultFinding(
  findings: DemoToothFinding[],
): DemoToothFinding | null {
  if (findings.length === 0) {
    return null;
  }
  const positive = findings.find((item) => item.pred);
  return positive ?? findings[0];
}

interface ZoomableImageProps {
  src: string;
  alt: string;
  sharedView?: [number, number, number, number] | null;
  onViewChange?: (viewBox: [number, number, number, number] | null) => void;
}

function ZoomableImage({
  src,
  alt,
  sharedView,
  onViewChange,
}: ZoomableImageProps): JSX.Element {
  const containerRef = useRef<HTMLDivElement>(null);
  const [imageSize, setImageSize] = useState<{
    width: number;
    height: number;
  } | null>(null);
  const [viewBox, setViewBox] = useState<
    [number, number, number, number] | null
  >(null);
  const [selectionRect, setSelectionRect] = useState<{
    x: number;
    y: number;
    width: number;
    height: number;
  } | null>(null);
  const [isSelecting, setIsSelecting] = useState(false);
  const initialViewBoxRef = useRef<[number, number, number, number] | null>(
    null,
  );
  const viewBoxRef = useRef<[number, number, number, number] | null>(null);
  const selectionStartRef = useRef<{ x: number; y: number } | null>(null);

  useEffect(() => {
    let cancelled = false;
    const img = new Image();
    img.onload = () => {
      if (cancelled) {
        return;
      }
      const naturalWidth = img.naturalWidth || 1;
      const naturalHeight = img.naturalHeight || 1;
      const baseView: [number, number, number, number] = [
        0,
        0,
        naturalWidth,
        naturalHeight,
      ];
      setImageSize({ width: naturalWidth, height: naturalHeight });
      initialViewBoxRef.current = [...baseView];
      const nextView = sharedView ?? baseView;
      const clonedView: [number, number, number, number] = [...nextView];
      setViewBox(clonedView);
      viewBoxRef.current = [...clonedView];
    };
    img.src = src;
    return () => {
      cancelled = true;
    };
  }, [src]);

  useEffect(() => {
    if (sharedView === undefined) {
      return;
    }
    if (sharedView === null) {
      const base = initialViewBoxRef.current;
      if (!base) {
        return;
      }
      const current = viewBoxRef.current;
      if (
        !current ||
        base.some((value, index) => Math.abs(value - current[index]) > 1e-3)
      ) {
        const baseClone: [number, number, number, number] = [...base];
        setViewBox(baseClone);
      }
      return;
    }
    const target: [number, number, number, number] = [...sharedView];
    const current = viewBoxRef.current;
    if (
      !current ||
      target.some((value, index) => Math.abs(value - current[index]) > 1e-3)
    ) {
      setViewBox(target);
    }
  }, [sharedView]);

  useEffect(() => {
    if (viewBox) {
      viewBoxRef.current = [...viewBox];
    }
  }, [viewBox]);

  const beginSelection = (event: PointerEvent<HTMLDivElement>): void => {
    if (event.button !== 0 || !containerRef.current || !viewBoxRef.current) {
      return;
    }
    const targetElement = event.target as HTMLElement | null;
    if (targetElement && targetElement.closest("[data-zoom-ignore]")) {
      return;
    }
    event.preventDefault();
    const rect = containerRef.current.getBoundingClientRect();
    const start = {
      x: Math.min(Math.max(event.clientX - rect.left, 0), rect.width),
      y: Math.min(Math.max(event.clientY - rect.top, 0), rect.height),
    };
    selectionStartRef.current = start;
    setSelectionRect({ x: start.x, y: start.y, width: 0, height: 0 });
    setIsSelecting(true);
    event.currentTarget.setPointerCapture?.(event.pointerId);
  };

  const updateSelection = (event: PointerEvent<HTMLDivElement>): void => {
    if (!isSelecting || !containerRef.current || !selectionStartRef.current) {
      return;
    }
    event.preventDefault();
    const rect = containerRef.current.getBoundingClientRect();
    const current = {
      x: Math.min(Math.max(event.clientX - rect.left, 0), rect.width),
      y: Math.min(Math.max(event.clientY - rect.top, 0), rect.height),
    };
    const start = selectionStartRef.current;
    const x = Math.min(start.x, current.x);
    const y = Math.min(start.y, current.y);
    const width = Math.abs(current.x - start.x);
    const height = Math.abs(current.y - start.y);
    setSelectionRect({ x, y, width, height });
  };

  const finishSelection = (event: PointerEvent<HTMLDivElement>): void => {
    if (
      !isSelecting ||
      !containerRef.current ||
      !selectionStartRef.current ||
      !viewBoxRef.current
    ) {
      return;
    }
    event.preventDefault();
    try {
      event.currentTarget.releasePointerCapture?.(event.pointerId);
    } catch {
      // ignore
    }
    setIsSelecting(false);
    const currentRect = selectionRect;
    setSelectionRect(null);
    selectionStartRef.current = null;
    if (!currentRect) {
      return;
    }
    const { width, height } = currentRect;
    if (width < 8 || height < 8) {
      return;
    }
    const containerRect = containerRef.current.getBoundingClientRect();
    const view = viewBoxRef.current;

    const clamp = (value: number) => Math.min(Math.max(value, 0), 1);
    const minXNorm = clamp(currentRect.x / containerRect.width);
    const maxXNorm = clamp((currentRect.x + width) / containerRect.width);
    const minYNorm = clamp(currentRect.y / containerRect.height);
    const maxYNorm = clamp((currentRect.y + height) / containerRect.height);

    const spanX = Math.max(maxXNorm - minXNorm, 0.02);
    const spanY = Math.max(maxYNorm - minYNorm, 0.02);
    const minWidth = initialViewBoxRef.current
      ? initialViewBoxRef.current[2] * 0.01
      : view[2] * 0.01;
    const minHeight = initialViewBoxRef.current
      ? initialViewBoxRef.current[3] * 0.01
      : view[3] * 0.01;

    const newWidth = Math.max(view[2] * spanX, Math.min(view[2], minWidth));
    const newHeight = Math.max(view[3] * spanY, Math.min(view[3], minHeight));
    let newX = view[0] + view[2] * minXNorm;
    let newY = view[1] + view[3] * minYNorm;
    const maxX = view[0] + view[2] - newWidth;
    const maxY = view[1] + view[3] - newHeight;
    newX = Math.min(Math.max(newX, view[0]), Math.max(maxX, view[0]));
    newY = Math.min(Math.max(newY, view[1]), Math.max(maxY, view[1]));

    const newView: [number, number, number, number] = [
      newX,
      newY,
      newWidth,
      newHeight,
    ];
    setViewBox(newView);
    onViewChange?.(newView);
  };

  const cancelSelection = (event: PointerEvent<HTMLDivElement>): void => {
    if (!isSelecting) {
      return;
    }
    finishSelection(event);
  };

  const handleReset = (): void => {
    const base = initialViewBoxRef.current;
    if (!base) {
      return;
    }
    const baseClone: [number, number, number, number] = [...base];
    setViewBox(baseClone);
    onViewChange?.(baseClone);
  };

  const isAtInitial = (() => {
    if (!viewBox || !initialViewBoxRef.current) {
      return true;
    }
    return viewBox.every(
      (value, index) =>
        Math.abs(value - initialViewBoxRef.current![index]) < 1e-3,
    );
  })();

  return (
    <div
      ref={containerRef}
      className="relative h-full w-full select-none"
      onPointerDown={beginSelection}
      onPointerMove={updateSelection}
      onPointerUp={finishSelection}
      onPointerLeave={cancelSelection}
      onContextMenu={(event) => event.preventDefault()}
    >
      {imageSize && viewBox ? (
        <svg
          className="h-full w-full"
          viewBox={`${viewBox[0]} ${viewBox[1]} ${viewBox[2]} ${viewBox[3]}`}
          preserveAspectRatio="xMidYMid meet"
        >
          <title>{alt}</title>
          <image href={src} width={imageSize.width} height={imageSize.height} />
        </svg>
      ) : (
        <div className="flex h-full items-center justify-center text-xs text-slate-500">
          載入影像中...
        </div>
      )}
      <button
        type="button"
        data-zoom-ignore
        onClick={handleReset}
        disabled={isAtInitial}
        className="absolute right-3 top-3 rounded-lg bg-slate-900/80 px-2 py-1 text-xs text-slate-200 shadow disabled:opacity-40"
      >
        重置視圖
      </button>
      {selectionRect ? (
        <div
          className="pointer-events-none absolute rounded border border-cyan-300 bg-cyan-400/10"
          style={{
            left: selectionRect.x,
            top: selectionRect.y,
            width: selectionRect.width,
            height: selectionRect.height,
          }}
        />
      ) : null}
    </div>
  );
}

export default function DemoPage(): JSX.Element {
  const [samples, setSamples] = useState<DemoSampleSummary[]>([]);
  const [selectedSampleId, setSelectedSampleId] = useState<string | null>(null);
  const [result, setResult] = useState<DemoInferenceResult | null>(null);
  const [activeFinding, setActiveFinding] = useState<DemoToothFinding | null>(
    null,
  );
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [preview, setPreview] = useState<PreviewSource | null>(null);
  const [inputKey, setInputKey] = useState(() => Date.now());
  const [onlyPositives, setOnlyPositives] = useState(true);
  const [modelType, setModelType] = useState<string>("cross");
  const [sharedZoomView, setSharedZoomView] = useState<
    [number, number, number, number] | null
  >(null);
  const [isSampleModalOpen, setSampleModalOpen] = useState(false);
  const [toothSearch, setToothSearch] = useState("");
  const selectedSample = useMemo(() => {
    if (!selectedSampleId) {
      return null;
    }
    return samples.find((sample) => sample.id === selectedSampleId) ?? null;
  }, [samples, selectedSampleId]);

  const filteredFindings = useMemo(() => {
    if (!result) {
      return [];
    }

    const safeParse = (value: string | undefined, fallback = 0) => {
      const parsed = Number.parseInt(value ?? "", 10);
      return Number.isNaN(parsed) ? fallback : parsed;
    };

    const sorted = [...result.findings].sort((a, b) => {
      const headA = safeParse(a.fdi?.charAt(0), 0);
      const headB = safeParse(b.fdi?.charAt(0), 0);
      if (headA !== headB) {
        return headA - headB;
      }
      const fullA = safeParse(a.fdi, headA);
      const fullB = safeParse(b.fdi, headB);
      return fullA - fullB;
    });

    const term = toothSearch.trim();
    if (!term) {
      return sorted;
    }
    const normalized = term.replace(/\s+/g, "").toLowerCase();
    return sorted.filter((finding) =>
      finding.fdi.toLowerCase().includes(normalized),
    );
  }, [result, toothSearch]);

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

    const views: Array<{ key: string; label: string; url: string }> = [];

    if (preview?.url) {
      views.push({ key: "original", label: "原始影像", url: preview.url });
    }

    if (overlay) {
      views.push({ key: "overlay", label: "推論疊圖", url: overlay });
    }

    return views;
  }, [preview, result]);

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
    setSharedZoomView(null);
    setToothSearch("");
  };

  const handleSample = async (sample: DemoSampleSummary): Promise<void> => {
    setError(null);
    setSelectedSampleId(sample.id);
    updatePreview(resolveMediaUrl(sample.image_path) ?? sample.image_path);
    setLoading(true);
    resetState();

    try {
      const response = await submitDemoInference({
        sampleId: sample.id,
        onlyPositive: onlyPositives,
        modelType,
      });
      setResult(response);
      const finding = pickDefaultFinding(response.findings);
      setActiveFinding(finding);
      setSharedZoomView(null);
      setToothSearch("");
      if (response.overlay_url) {
        setActiveView("overlay");
      } else {
        setActiveView("original");
      }
      setSampleModalOpen(false);
    } catch (err) {
      console.error("Demo inference failed", err);
      setError(err instanceof Error ? err.message : "推論失敗");
    } finally {
      setLoading(false);
    }
  };

  const handleUpload = async (
    event: ChangeEvent<HTMLInputElement>,
  ): Promise<void> => {
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
      const response = await submitDemoInference({
        file,
        onlyPositive: onlyPositives,
        modelType,
      });
      setResult(response);
      const finding = pickDefaultFinding(response.findings);
      setActiveFinding(finding);
      setSharedZoomView(null);
      setToothSearch("");
      if (response.overlay_url) {
        setActiveView("overlay");
      } else {
        setActiveView("original");
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
  const activeCamUrl = activeFinding?.cam_path
    ? resolveMediaUrl(activeFinding.cam_path)
    : null;
  const activeRoiUrl = activeFinding?.roi_path
    ? resolveMediaUrl(activeFinding.roi_path)
    : null;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <div className="mx-auto flex min-h-screen max-w-6xl flex-col px-6 py-10">
        <header className="space-y-4 text-center lg:text-left">
          <p className="text-sm uppercase tracking-[0.3em] text-cyan-300">
            Cross-Attention Grad-CAM Demo
          </p>
          <h1 className="text-4xl font-bold tracking-tight text-white sm:text-5xl">
            即時體驗交叉注意力齲齒可視化
          </h1>
          <p className="text-base leading-relaxed text-slate-300 sm:text-lg">
            上傳全口 X 光或選擇預載樣本，即可呼叫 FastAPI
            推論服務，查看逐齒置信度、偵測框與 Grad-CAM 熱力圖。
          </p>
        </header>

        <div className="mt-10 grid flex-1 gap-8 lg:grid-cols-[320px,1fr]">
          <aside className="space-y-6 rounded-2xl border border-slate-800 bg-slate-900/40 p-6 shadow-xl shadow-slate-900/40 backdrop-blur">
            <div className="space-y-3">
              <h2 className="text-xl font-semibold text-white">預設樣本</h2>
              <p className="text-sm text-slate-300">
                專案會自動掃描{" "}
                <code className="rounded bg-slate-800 px-1.5 py-0.5 text-xs">
                  demo_backend/static/samples
                </code>{" "}
                內的影像。
              </p>
              <button
                type="button"
                onClick={() => setSampleModalOpen(true)}
                disabled={samples.length === 0 || loading}
                className="w-full rounded-xl border border-cyan-400/60 bg-slate-900/40 px-4 py-3 text-sm font-semibold text-cyan-200 transition hover:bg-slate-800/60 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {samples.length === 0 ? "尚無可用樣本" : "瀏覽預設樣本"}
              </button>
              {selectedSample ? (
                <div className="rounded-lg border border-slate-800 bg-slate-900/60 p-3 text-xs text-slate-300">
                  <div className="font-semibold text-white">
                    目前選取：{selectedSample.title}
                  </div>
                  <div className="mt-1 text-slate-400">
                    {selectedSample.description}
                  </div>
                </div>
              ) : null}
            </div>

            <div className="space-y-3">
              <h2 className="text-xl font-semibold text-white">上傳影像</h2>
              <p className="text-sm text-slate-300">
                支援單張 PNG 或 JPG。推論完成後，系統會產生疊圖與
                CSV，並於頁面右側顯示逐齒結果。
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

            <div className="rounded-xl border border-slate-800/60 bg-slate-900/30 p-4">
              <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                <div>
                  <p className="text-sm font-semibold text-white">推論設定</p>
                  <p className="text-xs text-slate-400">
                    選擇使用的模型架構與視覺化選項。
                  </p>
                </div>
                <div className="flex flex-col gap-2">
                  <label className="flex items-center gap-2 text-sm text-slate-200">
                    <span className="text-xs text-slate-400">模型：</span>
                    <select
                      value={modelType}
                      onChange={(e) => setModelType(e.target.value)}
                      disabled={loading}
                      className="rounded bg-slate-950 border-slate-700 px-2 py-1 text-xs text-slate-200 focus:ring-cyan-400"
                    >
                      <option value="cross">Cross Attention</option>
                      <option value="swin">Swin Transformer</option>
                    </select>
                  </label>
                  <label className="flex items-center gap-2 text-sm text-slate-200">
                    <input
                      type="checkbox"
                      className="h-4 w-4 rounded border-slate-600 bg-slate-900 text-cyan-400 focus:ring-cyan-400"
                      checked={onlyPositives}
                      onChange={(event) =>
                        setOnlyPositives(event.target.checked)
                      }
                      disabled={loading}
                    />
                    <span>只顯示疑似蛀牙</span>
                  </label>
                </div>
              </div>
            </div>

            <div className="rounded-xl border border-slate-800/60 bg-slate-900/20 p-4 text-xs text-slate-300">
              <p className="font-semibold text-white">使用小提醒</p>
              <ul className="mt-2 space-y-1 list-disc pl-4">
                <li>
                  確保模型權重已放入 <code>models/</code> 目錄。
                </li>
                <li>
                  若使用 CPU，請設定 <code>DEMO_DEVICE=cpu</code>。
                </li>
                <li>
                  推論結果將儲存在 <code>demo_backend/outputs/</code>。
                </li>
              </ul>
            </div>
          </aside>

          <section className="flex flex-col gap-6">
            {error ? (
              <p className="rounded-xl border border-red-500/50 bg-red-500/10 px-4 py-3 text-sm text-red-200">
                {error}
              </p>
            ) : null}
            {loading ? (
              <p className="text-sm text-cyan-200">模型推論中，請稍候...</p>
            ) : null}

            <div className="rounded-2xl border border-slate-800 bg-slate-900/30 p-6 shadow-xl shadow-slate-900/30">
              <h2 className="text-2xl font-semibold text-white">視覺化結果</h2>
              <p className="mt-1 text-sm text-slate-300">
                依序切換原圖與推論疊圖掌握整體差異，右側可進一步檢視選定牙齒的
                ROI 與 Grad-CAM。
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
                  <ZoomableImage
                    src={activePreview.url}
                    alt={activePreview.label}
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
                      <span className="font-semibold text-white">
                        請求 ID：
                      </span>
                      <span className="font-mono text-cyan-200">
                        {result.request_id}
                      </span>
                    </div>
                    <div className="text-xs text-slate-400">
                      推論產出與 CSV 可透過 <code>/demo-outputs</code>{" "}
                      路徑下載。
                    </div>
                  </div>
                  {result.warnings.length > 0 ? (
                    <ul className="space-y-1 rounded-xl border border-yellow-400/40 bg-yellow-400/10 p-3 text-xs text-yellow-200">
                      {result.warnings.map((warning) => (
                        <li key={warning}>{warning}</li>
                      ))}
                    </ul>
                  ) : null}

                  <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr),260px]">
                    <div className="space-y-4">
                      {activeFinding ? (
                        <div className="space-y-4 rounded-xl border border-slate-800 bg-slate-900/45 p-4">
                          <div className="flex flex-col gap-2 text-sm text-slate-300 sm:flex-row sm:items-center sm:justify-between">
                            <div>
                              <span className="text-base font-semibold text-white">
                                選取牙齒視圖
                              </span>
                              <p className="text-xs text-slate-400">
                                顯示原始 ROI 與對應的 Grad-CAM
                                熱力圖，可雙擊重置縮放。
                              </p>
                            </div>
                            <div className="text-xs font-mono text-cyan-200">
                              {`FDI ${activeFinding.fdi} · ${formatProbability(activeFinding.prob_caries)}`}
                            </div>
                          </div>
                          <div className="grid gap-3 sm:grid-cols-2">
                            <div className="rounded-lg border border-slate-800 bg-slate-950/60">
                              <div className="border-b border-slate-800 px-3 py-2 text-xs font-semibold uppercase tracking-wide text-slate-300">
                                原始 ROI
                              </div>
                              <div className="h-56 p-2">
                                {activeRoiUrl ? (
                                  <ZoomableImage
                                    src={activeRoiUrl}
                                    alt={`ROI ${activeFinding.fdi}`}
                                    sharedView={sharedZoomView}
                                    onViewChange={setSharedZoomView}
                                  />
                                ) : (
                                  <div className="flex h-full items-center justify-center text-xs text-slate-500">
                                    暫無裁切影像
                                  </div>
                                )}
                              </div>
                            </div>
                            <div className="rounded-lg border border-slate-800 bg-slate-950/60">
                              <div className="border-b border-slate-800 px-3 py-2 text-xs font-semibold uppercase tracking-wide text-slate-300">
                                Grad-CAM
                              </div>
                              <div className="h-56 p-2">
                                {activeCamUrl ? (
                                  <ZoomableImage
                                    src={activeCamUrl}
                                    alt={`Grad-CAM ${activeFinding.fdi}`}
                                    sharedView={sharedZoomView}
                                    onViewChange={setSharedZoomView}
                                  />
                                ) : (
                                  <div className="flex h-full items-center justify-center text-xs text-slate-500">
                                    暫無熱力圖
                                  </div>
                                )}
                              </div>
                            </div>
                          </div>
                        </div>
                      ) : (
                        <div className="rounded-xl border border-dashed border-slate-800 bg-slate-900/30 p-6 text-sm text-slate-400">
                          請於右側快速列表或下方表格點選牙齒，以查看對應的 ROI
                          與 Grad-CAM。
                        </div>
                      )}

                      <div className="overflow-x-auto">
                        <table className="min-w-full divide-y divide-slate-800 text-sm">
                          <thead>
                            <tr className="bg-slate-900/60 text-left text-slate-200">
                              <th className="px-3 py-2 font-medium">FDI</th>
                              <th className="px-3 py-2 font-medium">
                                蛀牙機率
                              </th>
                              <th className="px-3 py-2 font-medium">門檻</th>
                              <th className="px-3 py-2 font-medium">判定</th>
                              <th className="px-3 py-2 font-medium">偵測框</th>
                              <th className="px-3 py-2 font-medium">
                                Grad-CAM
                              </th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-slate-800/60">
                            {result.findings.map((finding) => {
                              const isSelected =
                                activeFinding?.fdi === finding.fdi &&
                                activeFinding?.bbox.x1 === finding.bbox.x1 &&
                                activeFinding?.bbox.y1 === finding.bbox.y1;
                              const camUrl = finding.cam_path
                                ? resolveMediaUrl(finding.cam_path)
                                : null;
                              return (
                                <tr
                                  key={`${result.request_id}-${finding.fdi}-${finding.bbox.x1}-${finding.bbox.y1}`}
                                  onClick={() => {
                                    setSharedZoomView(null);
                                    setActiveFinding(finding);
                                    if (finding.cam_path) {
                                      setActiveView("overlay");
                                    }
                                  }}
                                  className={`cursor-pointer transition hover:bg-slate-900/60 ${
                                    isSelected ? "bg-slate-900/80" : ""
                                  }`}
                                >
                                  <td className="px-3 py-2 font-semibold text-white">
                                    {finding.fdi}
                                  </td>
                                  <td className="px-3 py-2 text-slate-200">
                                    {formatProbability(finding.prob_caries)}
                                  </td>
                                  <td className="px-3 py-2 text-slate-200">
                                    {formatProbability(finding.thr_used)}
                                  </td>
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
                                    [{finding.bbox.x1}, {finding.bbox.y1}] → [
                                    {finding.bbox.x2}, {finding.bbox.y2}]
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
                                      <span className="text-xs text-slate-400">
                                        無可用熱力圖
                                      </span>
                                    )}
                                  </td>
                                </tr>
                              );
                            })}
                            {result.findings.length === 0 ? (
                              <tr>
                                <td
                                  colSpan={6}
                                  className="px-3 py-4 text-center text-sm text-slate-400"
                                >
                                  無任何牙齒偵測結果。
                                </td>
                              </tr>
                            ) : null}
                          </tbody>
                        </table>
                      </div>
                    </div>
                    <aside className="flex h-full flex-col rounded-xl border border-slate-800 bg-slate-900/30 p-4">
                      <div className="flex items-center justify-between gap-3">
                        <h3 className="text-sm font-semibold text-white">
                          牙位快速選擇
                        </h3>
                        <span className="text-xs text-slate-400">
                          {filteredFindings.length}/{result.findings.length}
                        </span>
                      </div>
                      <div className="mt-3">
                        <label className="sr-only" htmlFor="tooth-search">
                          搜尋牙位
                        </label>
                        <input
                          id="tooth-search"
                          type="text"
                          value={toothSearch}
                          onChange={(event) =>
                            setToothSearch(event.target.value)
                          }
                          placeholder="輸入 FDI 牙位"
                          className="w-full rounded-lg border border-slate-700 bg-slate-900/60 px-3 py-2 text-sm text-slate-100 placeholder:text-slate-500 focus:border-cyan-400 focus:outline-none focus:ring-2 focus:ring-cyan-400/40"
                        />
                      </div>
                      <div className="mt-4 flex-1 space-y-2 overflow-y-auto pr-1">
                        {filteredFindings.length > 0 ? (
                          filteredFindings.map((finding) => {
                            const isSelected =
                              activeFinding?.fdi === finding.fdi &&
                              activeFinding?.bbox.x1 === finding.bbox.x1 &&
                              activeFinding?.bbox.y1 === finding.bbox.y1;
                            return (
                              <button
                                key={`${result.request_id}-quick-${finding.fdi}-${finding.bbox.x1}-${finding.bbox.y1}`}
                                type="button"
                                onClick={() => {
                                  setSharedZoomView(null);
                                  setActiveFinding(finding);
                                  if (finding.cam_path) {
                                    setActiveView("overlay");
                                  }
                                }}
                                className={`w-full rounded-lg border px-3 py-2 text-left transition focus:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400/80 ${
                                  isSelected
                                    ? "border-cyan-400/80 bg-slate-800/70"
                                    : "border-slate-800 bg-slate-950/60 hover:border-cyan-400/60"
                                }`}
                              >
                                <div className="flex items-center justify-between text-sm">
                                  <span className="font-semibold text-white">
                                    FDI {finding.fdi}
                                  </span>
                                  <span
                                    className={`text-xs ${
                                      finding.pred
                                        ? "text-red-300"
                                        : "text-slate-400"
                                    }`}
                                  >
                                    {finding.pred ? "疑似蛀牙" : "正常"}
                                  </span>
                                </div>
                                <div className="mt-1 text-xs text-slate-300">
                                  機率 {formatProbability(finding.prob_caries)}
                                </div>
                              </button>
                            );
                          })
                        ) : (
                          <p className="text-sm text-slate-400">
                            找不到符合的牙位
                          </p>
                        )}
                      </div>
                    </aside>
                  </div>
                </div>
              ) : null}
            </div>
          </section>
        </div>
      </div>
      {isSampleModalOpen ? (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 px-4 py-8"
          onClick={(event) => {
            if (event.target === event.currentTarget) {
              setSampleModalOpen(false);
            }
          }}
        >
          <div className="relative w-full max-w-5xl max-h-[85vh] overflow-y-auto rounded-2xl border border-slate-800 bg-slate-900/95 p-6 shadow-2xl shadow-slate-950">
            <div className="flex items-center justify-between gap-3">
              <h3 className="text-lg font-semibold text-white">選擇預設樣本</h3>
              <button
                type="button"
                onClick={() => setSampleModalOpen(false)}
                className="rounded-full border border-slate-700 bg-slate-800/80 p-2 text-slate-300 transition hover:border-cyan-400/60 hover:text-cyan-200"
                aria-label="關閉預設樣本選單"
              >
                ×
              </button>
            </div>
            <p className="mt-1 text-sm text-slate-300">
              請挑選要套用的預設樣本，系統會立即觸發推論。
            </p>
            <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {samples.length > 0 ? (
                samples.map((sample) => {
                  const previewUrl =
                    resolveMediaUrl(sample.image_path) ?? sample.image_path;
                  const isActive = sample.id === selectedSampleId;
                  return (
                    <button
                      key={`modal-${sample.id}`}
                      type="button"
                      onClick={() => void handleSample(sample)}
                      disabled={loading}
                      className={`group flex flex-col overflow-hidden rounded-xl border p-3 text-left transition focus:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400/80 ${
                        isActive
                          ? "border-cyan-400/80 bg-slate-800/70"
                          : "border-slate-800 bg-slate-900/40 hover:border-cyan-400/60"
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
                      <span className="text-sm font-semibold text-white">
                        {sample.title}
                      </span>
                      <span className="mt-1 text-xs text-slate-300">
                        {sample.description}
                      </span>
                    </button>
                  );
                })
              ) : (
                <p className="text-sm text-slate-400">尚未放入任何示範影像。</p>
              )}
            </div>
            <div className="mt-6 flex justify-end">
              <button
                type="button"
                onClick={() => setSampleModalOpen(false)}
                className="rounded-lg border border-slate-700 bg-slate-800/80 px-4 py-2 text-sm font-medium text-slate-200 transition hover:border-cyan-400/60 hover:text-cyan-200"
              >
                關閉
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
