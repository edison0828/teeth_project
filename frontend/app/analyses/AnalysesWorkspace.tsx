"use client";

import { clsx } from "clsx";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import type {
  AnalysisPreview,
  AnalysisPreviewFinding,
  AnalysisSummary,
} from "../../lib/types";

import { resolveMediaUrl } from "../../lib/media";

type AnalysesWorkspaceProps = {
  analyses: AnalysisSummary[];
};

const STATUS_LABEL: Record<string, string> = {
  completed: "Completed",
  processing: "Processing",
  queued: "Queued",
};

const SEVERITY_LABEL: Record<string, string> = {
  mild: "Mild",
  moderate: "Moderate",
  severe: "Severe",
};

const SEVERITY_TONE: Record<string, string> = {
  mild: "border-emerald-300/80 bg-emerald-500/15 text-emerald-200",
  moderate: "border-amber-300/80 bg-amber-500/15 text-amber-200",
  severe: "border-rose-400/80 bg-rose-500/15 text-rose-200",
};

const NO_VIS_MESSAGE = "No visualization yet. Run an analysis first.";
const MISSING_IMAGE_MESSAGE = "Preview image missing.";

function toCssColor(color?: number[] | null): string | undefined {
  if (!color || color.length < 3) {
    return undefined;
  }
  const [b, g, r] = color;
  return `rgb(${Math.round(r)}, ${Math.round(g)}, ${Math.round(b)})`;
}

function formatDate(value?: string | null): string {
  if (!value) {
    return "N/A";
  }
  try {
    return new Date(value).toLocaleString();
  } catch {
    return value;
  }
}

function ensurePreview(
  analysis: AnalysisSummary | null | undefined
): AnalysisPreview | undefined {
  return analysis?.preview ?? undefined;
}

export default function AnalysesWorkspace({
  analyses,
}: AnalysesWorkspaceProps) {
  const [selectedId, setSelectedId] = useState<string | null>(
    analyses[0]?.id ?? null
  );
  const selectedAnalysis = useMemo(() => {
    if (!analyses.length) {
      return null;
    }
    return analyses.find((item) => item.id === selectedId) ?? analyses[0];
  }, [analyses, selectedId]);

  const preview = ensurePreview(selectedAnalysis);
  const [selectedFindingId, setSelectedFindingId] = useState<string | null>(
    preview?.findings[0]?.finding_id ?? null
  );
  const [hoveredFindingId, setHoveredFindingId] = useState<string | null>(null);
  const [useOverlay, setUseOverlay] = useState(true);

  useEffect(() => {
    setSelectedFindingId(preview?.findings[0]?.finding_id ?? null);
    setUseOverlay(true);
  }, [preview]);

  const selectedFinding = useMemo<AnalysisPreviewFinding | null>(() => {
    if (!preview) {
      return null;
    }
    return (
      preview.findings.find((item) => item.finding_id === selectedFindingId) ??
      preview.findings[0] ??
      null
    );
  }, [preview, selectedFindingId]);

  const selectedAssets = useMemo<Record<string, string>>(() => {
    if (!selectedFinding?.assets) {
      return {};
    }
    return Object.entries(selectedFinding.assets).reduce((acc, [key, value]) => {
      const resolved = resolveMediaUrl(value ?? undefined);
      if (resolved) {
        acc[key] = resolved;
      }
      return acc;
    }, {} as Record<string, string>);
  }, [selectedFinding?.assets]);

  const imgRef = useRef<HTMLImageElement | null>(null);
  const [renderSize, setRenderSize] = useState({ width: 1, height: 1 });
  const [naturalSize, setNaturalSize] = useState({
    width: preview?.image_size?.[0] ?? 1,
    height: preview?.image_size?.[1] ?? 1,
  });

  useEffect(() => {
    setNaturalSize({
      width: preview?.image_size?.[0] ?? 1,
      height: preview?.image_size?.[1] ?? 1,
    });
  }, [preview?.image_size]);

  const updateRenderSize = useCallback(() => {
    const node = imgRef.current;
    if (!node) {
      return;
    }
    const rect = node.getBoundingClientRect();
    setRenderSize({
      width: Math.max(rect.width, 1),
      height: Math.max(rect.height, 1),
    });
  }, []);

  useEffect(() => {
    updateRenderSize();
    window.addEventListener("resize", updateRenderSize);
    return () => window.removeEventListener("resize", updateRenderSize);
  }, [updateRenderSize]);

  const handleImageLoad = useCallback(
    (event: React.SyntheticEvent<HTMLImageElement>) => {
      const target = event.currentTarget;
      setNaturalSize({
        width: target.naturalWidth || 1,
        height: target.naturalHeight || 1,
      });
      requestAnimationFrame(updateRenderSize);
    },
    [updateRenderSize]
  );

  const baseWidth = (preview?.image_size?.[0] ?? naturalSize.width) || 1;
  const baseHeight = (preview?.image_size?.[1] ?? naturalSize.height) || 1;

  const scaleX = renderSize.width / baseWidth;
  const scaleY = renderSize.height / baseHeight;

  const rawImageSrc = useOverlay
    ? preview?.overlay_uri ?? preview?.image_uri
    : preview?.image_uri ?? preview?.overlay_uri;
  const imageSrc = resolveMediaUrl(rawImageSrc);

  const findingCount = preview?.findings.length ?? 0;

  if (!analyses.length) {
    return (
      <div className="space-y-6">
        <header className="rounded-3xl bg-white/5 px-8 py-10 text-white shadow-card">
          <p className="text-xs uppercase tracking-[0.3em] text-primary">
            Analyses
          </p>
          <h1 className="mt-4 text-3xl font-semibold">
            Interactive Analysis Workspace
          </h1>
          <p className="mt-3 text-sm text-slate-300 lg:max-w-3xl">
            Pick an analysis on the left, inspect the overlay in the middle, and
            open Grad-CAM details on the right.
          </p>
        </header>
        <div className="rounded-3xl border border-dashed border-white/10 px-6 py-16 text-center text-slate-300">
          No analysis records yet. Upload an image to get started.
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <header className="rounded-3xl bg-white/5 px-8 py-10 text-white shadow-card">
        <p className="text-xs uppercase tracking-[0.3em] text-primary">
          Analyses
        </p>
        <h1 className="mt-4 text-3xl font-semibold">
          Interactive Analysis Workspace
        </h1>
        <p className="mt-3 text-sm text-slate-300 lg:max-w-3xl">
          Pick an analysis on the left, inspect the overlay in the middle, and
          open Grad-CAM details on the right.
        </p>
      </header>

      <div className="grid gap-6 xl:grid-cols-[minmax(280px,360px)_minmax(320px,1fr)_minmax(260px,320px)]">
        <aside className="space-y-4">
          <h2 className="text-sm font-semibold text-slate-200">Analyses</h2>
          <div className="space-y-3 overflow-hidden rounded-3xl border border-white/5 bg-white/5 p-4">
            {analyses.map((analysis) => {
              const previewCount = analysis.preview?.findings.length ?? 0;
              const isSelected = analysis.id === selectedAnalysis?.id;
              return (
                <button
                  key={analysis.id}
                  type="button"
                  onClick={() => setSelectedId(analysis.id)}
                  className={clsx(
                    "w-full rounded-2xl border px-4 py-3 text-left transition",
                    isSelected
                      ? "border-primary/70 bg-primary/10 text-white shadow-lg shadow-primary/10"
                      : "border-white/5 bg-slate-900/40 text-slate-200 hover:border-primary/40 hover:bg-primary/10/20"
                  )}
                >
                  <div className="flex items-center justify-between text-xs text-slate-300">
                    <span>{analysis.id}</span>
                    <span className="rounded-full bg-white/10 px-2 py-0.5 text-[11px] uppercase tracking-wider">
                      {STATUS_LABEL[analysis.status] ?? analysis.status}
                    </span>
                  </div>
                  <p className="mt-2 text-sm font-semibold text-white">
                    {analysis.overall_assessment ?? "Awaiting model conclusion"}
                  </p>
                  <div className="mt-3 flex items-center justify-between text-xs text-slate-400">
                    <span>
                      {previewCount
                        ? `${previewCount} detections`
                        : "No visualization yet"}
                    </span>
                    <span>
                      {formatDate(
                        analysis.completed_at ?? analysis.triggered_at
                      )}
                    </span>
                  </div>
                </button>
              );
            })}
          </div>
        </aside>

        <section className="space-y-4">
          <div className="flex items-center justify-between gap-3">
            <div>
              <h2 className="text-lg font-semibold text-white">
                Model Visualization
              </h2>
              <p className="text-xs text-slate-400">
                Bounding boxes and tooth predictions
              </p>
            </div>
            <div className="flex items-center gap-2 text-xs">
              <span
                className={clsx(
                  "rounded-full px-3 py-1",
                  useOverlay
                    ? "bg-primary/20 text-primary"
                    : "bg-white/10 text-slate-200"
                )}
              >
                Overlay
              </span>
              <label className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-slate-900/60 px-3 py-1">
                <span className="text-slate-300">Image layer</span>
                <button
                  type="button"
                  onClick={() => setUseOverlay((prev) => !prev)}
                  className="rounded-full bg-primary/20 px-3 py-1 text-[11px] font-semibold text-primary hover:bg-primary/30"
                >
                  {useOverlay ? "Show original" : "Show overlay"}
                </button>
              </label>
            </div>
          </div>

          <div className="relative overflow-hidden rounded-3xl border border-white/10 bg-[#050B1C]/80 p-4">
            {imageSrc && preview ? (
              <div className="relative overflow-hidden rounded-2xl border border-white/10 bg-[#0F172A] p-2">
                <div className="relative" style={{ minHeight: 320 }}>
                  <img
                    ref={imgRef}
                    src={imageSrc}
                    alt="analysis preview"
                    className="h-auto w-full select-none rounded-xl object-contain"
                    onLoad={handleImageLoad}
                  />
                  <div className="pointer-events-none absolute inset-0">
                    {preview.findings.map((finding) => {
                      const [x, y, w, h] = finding.bbox;
                      const left = x * scaleX;
                      const top = y * scaleY;
                      const width = w * scaleX;
                      const height = h * scaleY;
                      const isActive =
                        finding.finding_id === selectedFinding?.finding_id;
                      const isHovered = finding.finding_id === hoveredFindingId;
                      const strokeColor = toCssColor(finding.color_bgr);
                      return (
                        <button
                          key={finding.finding_id}
                          type="button"
                          className="pointer-events-auto absolute"
                          style={{ left, top, width, height }}
                          onClick={() =>
                            setSelectedFindingId(finding.finding_id)
                          }
                          onMouseEnter={() =>
                            setHoveredFindingId(finding.finding_id)
                          }
                          onMouseLeave={() =>
                            setHoveredFindingId((current) =>
                              current === finding.finding_id ? null : current
                            )
                          }
                        >
                          <span
                            className={clsx(
                              "block h-full w-full rounded-xl border-2 bg-primary/5 transition",
                              isActive ? "ring-4 ring-primary/40" : "ring-0",
                              isHovered && !isActive
                                ? "border-primary/70"
                                : "border-white/40"
                            )}
                            style={
                              strokeColor
                                ? {
                                    borderColor: strokeColor,
                                    backgroundColor: isActive
                                      ? `${strokeColor}33`
                                      : undefined,
                                  }
                                : undefined
                            }
                          />
                        </button>
                      );
                    })}
                  </div>
                </div>
              </div>
            ) : (
              <div className="flex h-[360px] items-center justify-center rounded-2xl border border-dashed border-white/15 text-sm text-slate-400">
                {findingCount === 0 ? NO_VIS_MESSAGE : MISSING_IMAGE_MESSAGE}
              </div>
            )}
            <div className="mt-4 flex flex-wrap items-center gap-3 text-xs text-slate-300">
              <span className="rounded-full bg-white/10 px-3 py-1">
                Detections: {findingCount}
              </span>
              <span className="rounded-full bg-white/10 px-3 py-1">
                Viewport {Math.round(renderSize.width)} × {Math.round(renderSize.height)}
              </span>
            </div>
          </div>
        </section>

        <aside className="space-y-4">
          <h2 className="text-sm font-semibold text-slate-200">
            Tooth Details
          </h2>
          <div className="rounded-3xl border border-white/5 bg-white/5 p-4">
            {selectedFinding ? (
              <div className="space-y-4">
                <div>
                  <p className="text-xs uppercase tracking-[0.2em] text-slate-400">
                    Tooth
                  </p>
                  <h3 className="text-xl font-semibold text-white">
                    {selectedFinding.tooth_label ?? "Unlabeled"}
                  </h3>
                </div>
                <div className="flex flex-wrap items-center gap-2 text-xs">
                  <span
                    className={clsx(
                      "rounded-full border px-2 py-1",
                      SEVERITY_TONE[selectedFinding.severity] ??
                        "border-white/20 text-slate-200"
                    )}
                  >
                    {SEVERITY_LABEL[selectedFinding.severity] ??
                      selectedFinding.severity}
                  </span>
                  <span className="rounded-full border border-white/10 px-2 py-1 text-slate-200">
                    Confidence {(selectedFinding.confidence * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="space-y-3 text-xs text-slate-300">
                  <p>
                    Centroid:{" "}
                    {selectedFinding.centroid
                      ? selectedFinding.centroid
                          .map((v) => (v * 100).toFixed(1) + '%')
                          .join(' / ')
                      : 'N/A'}
                  </p>
                  <p>
                    Bounding box: [
                    {selectedFinding.bbox.map((v) => Math.round(v)).join(", ")}]
                  </p>
                </div>
                <div className="space-y-3">
                  <p className="text-xs uppercase tracking-[0.2em] text-slate-400">
                    Grad-CAM
                  </p>
                  {selectedAssets.gradcam ? (
                    <img
                      src={selectedAssets.gradcam}
                      alt="Grad-CAM heatmap"
                      className="w-full rounded-2xl border border-white/10"
                    />
                  ) : selectedAssets.heatmap ? (
                    <img
                      src={selectedAssets.heatmap}
                      alt="Heatmap"
                      className="w-full rounded-2xl border border-white/10"
                    />
                  ) : (
                    <div className="flex h-40 items-center justify-center rounded-2xl border border-dashed border-white/15 text-xs text-slate-400">
                      Grad-CAM preview not available.
                    </div>
                  )}
                </div>
                {selectedAssets.crop ? (
                  <div className="space-y-2">
                    <p className="text-xs uppercase tracking-[0.2em] text-slate-400">
                      Tooth crop
                    </p>
                    <img
                      src={selectedAssets.crop}
                      alt="Tooth crop"
                      className="w-full rounded-2xl border border-white/10"
                    />
                  </div>
                ) : null}
              </div>
            ) : (
              <div className="flex h-64 items-center justify-center text-sm text-slate-400">
                Select a detection to inspect.
              </div>
            )}
          </div>
        </aside>
      </div>
    </div>
  );
}






