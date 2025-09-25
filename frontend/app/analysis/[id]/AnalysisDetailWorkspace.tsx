"use client";

import { clsx } from "clsx";
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
  type SyntheticEvent,
  type WheelEvent,
} from "react";

import type {
  AnalysisDetail,
  AnalysisPreviewFinding,
} from "../../../lib/types";
import { resolveMediaUrl } from "../../../lib/media";

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

const MIN_SCALE = 0.35;
const MAX_SCALE = 6;
const MIN_ASSET_ZOOM = 0.6;
const MAX_ASSET_ZOOM = 2.5;

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

function toCssColor(color?: number[] | null): string | undefined {
  if (!color || color.length < 3) {
    return undefined;
  }
  const [b, g, r] = color;
  return `rgb(${Math.round(r)}, ${Math.round(g)}, ${Math.round(b)})`;
}

function formatTimestamp(value?: string | null): string {
  if (!value) {
    return "N/A";
  }
  try {
    return new Date(value).toLocaleString();
  } catch {
    return value;
  }
}

function formatPercentPair(values?: number[] | null): string {
  if (!values || values.length < 2) {
    return "N/A";
  }
  return values.map((v) => `${(v * 100).toFixed(1)}%`).join(" / ");
}

function formatExtraValue(value: unknown): string {
  if (value == null) {
    return "—";
  }
  if (typeof value === "number") {
    return Number.isFinite(value) ? value.toFixed(2) : String(value);
  }
  if (typeof value === "object") {
    try {
      return JSON.stringify(value);
    } catch {
      return "[object]";
    }
  }
  return String(value);
}

type ViewportState = {
  scale: number;
  x: number;
  y: number;
};

type AnalysisDetailWorkspaceProps = {
  analysis: AnalysisDetail;
};

export default function AnalysisDetailWorkspace({
  analysis,
}: AnalysisDetailWorkspaceProps) {
  const preview = analysis.preview ?? null;

  const overlayIds = useMemo(() => {
    if (!preview) {
      return new Set<string>();
    }
    return new Set(preview.findings.map((finding) => finding.finding_id));
  }, [preview]);
  const [selectedFindingId, setSelectedFindingId] = useState<string | null>(() => {
    return (
      preview?.findings[0]?.finding_id ??
      analysis.findings[0]?.finding_id ??
      null
    );
  });
  const [hoveredFindingId, setHoveredFindingId] = useState<string | null>(null);
  const [useOverlay, setUseOverlay] = useState(() => Boolean(preview?.overlay_uri));

  useEffect(() => {
    if (!preview?.overlay_uri) {
      setUseOverlay(false);
    }
  }, [preview?.overlay_uri]);
  useEffect(() => {
    setSelectedFindingId((current) => {
      if (
        current &&
        analysis.findings.some((finding) => finding.finding_id === current)
      ) {
        return current;
      }
      return (
        preview?.findings[0]?.finding_id ??
        analysis.findings[0]?.finding_id ??
        null
      );
    });
  }, [analysis.findings, preview]);

  const selectedPreviewFinding = useMemo<AnalysisPreviewFinding | null>(() => {
    if (!preview || !selectedFindingId) {
      return null;
    }
    return (
      preview.findings.find(
        (finding) => finding.finding_id === selectedFindingId
      ) ?? null
    );
  }, [preview, selectedFindingId]);

  const selectedDetailFinding = useMemo(() => {
    if (!selectedFindingId) {
      return null;
    }
    return (
      analysis.findings.find(
        (finding) => finding.finding_id === selectedFindingId
      ) ?? null
    );
  }, [analysis.findings, selectedFindingId]);

  const selectedAssets = useMemo<Record<string, string>>(() => {
    if (!selectedPreviewFinding?.assets) {
      return {};
    }
    return Object.entries(selectedPreviewFinding.assets).reduce(
      (acc, [key, value]) => {
        const resolved = resolveMediaUrl(value ?? undefined);
        if (resolved) {
          acc[key] = resolved;
        }
        return acc;
      },
      {} as Record<string, string>
    );
  }, [selectedPreviewFinding?.assets]);
  const [assetZoom, setAssetZoom] = useState(1);

  useEffect(() => {
    setAssetZoom(1);
  }, [selectedFindingId]);

  const overlaySrc = resolveMediaUrl(preview?.overlay_uri ?? undefined);
  const baseImageCandidate =
    preview?.image_uri ?? analysis.image.public_url ?? undefined;
  const baseImageSrc = resolveMediaUrl(baseImageCandidate);
  const displayImageSrc =
    resolveMediaUrl(
      useOverlay
        ? preview?.overlay_uri ?? baseImageCandidate
        : baseImageCandidate ?? preview?.overlay_uri
    ) ?? baseImageSrc;

  const canToggleOverlay = Boolean(overlaySrc && baseImageSrc);

  const containerRef = useRef<HTMLDivElement | null>(null);
  const draggingRef = useRef(false);
  const pointerRef = useRef({ x: 0, y: 0 });

  const [isDragging, setIsDragging] = useState(false);
  const [hasInteracted, setHasInteracted] = useState(false);
  const [containerSize, setContainerSize] = useState({ width: 1, height: 1 });
  const [imageSize, setImageSize] = useState(() => {
    const [width, height] = preview?.image_size ?? [];
    return {
      width: width && width > 0 ? width : 1024,
      height: height && height > 0 ? height : 1024,
    };
  });
  const [viewport, setViewport] = useState<ViewportState>({
    scale: 1,
    x: 0,
    y: 0,
  });
  useEffect(() => {
    const node = containerRef.current;
    if (!node) {
      return;
    }

    const rect = node.getBoundingClientRect();
    setContainerSize({ width: rect.width, height: rect.height });

    if (typeof ResizeObserver === "undefined") {
      return;
    }

    const observer = new ResizeObserver((entries) => {
      if (!entries.length) {
        return;
      }
      const entry = entries[0];
      setContainerSize({
        width: entry.contentRect.width,
        height: entry.contentRect.height,
      });
    });
    observer.observe(node);

    return () => {
      observer.disconnect();
    };
  }, []);

  useEffect(() => {
    if (preview?.image_size && preview.image_size.length >= 2) {
      const [width, height] = preview.image_size;
      if (width && height) {
        setImageSize({
          width,
          height,
        });
      }
    }
  }, [preview]);

  const clampOffsets = useCallback(
    (x: number, y: number, scale: number) => {
      const scaledWidth = imageSize.width * scale;
      const scaledHeight = imageSize.height * scale;

      if (
        !Number.isFinite(scaledWidth) ||
        !Number.isFinite(scaledHeight) ||
        !containerSize.width ||
        !containerSize.height
      ) {
        return { x, y };
      }

      const minX = Math.min(0, containerSize.width - scaledWidth);
      const maxX = Math.max(0, containerSize.width - scaledWidth);
      const minY = Math.min(0, containerSize.height - scaledHeight);
      const maxY = Math.max(0, containerSize.height - scaledHeight);

      return {
        x: clamp(x, minX, maxX),
        y: clamp(y, minY, maxY),
      };
    },
    [containerSize, imageSize]
  );

  const fitToScreen = useCallback(() => {
    if (!containerSize.width || !containerSize.height) {
      return;
    }
    const scaleX = containerSize.width / imageSize.width;
    const scaleY = containerSize.height / imageSize.height;
    const baseScale = Math.min(scaleX, scaleY);

    if (!Number.isFinite(baseScale) || baseScale <= 0) {
      return;
    }

    const nextScale = clamp(baseScale, MIN_SCALE, MAX_SCALE);
    const scaledWidth = imageSize.width * nextScale;
    const scaledHeight = imageSize.height * nextScale;
    const x = (containerSize.width - scaledWidth) / 2;
    const y = (containerSize.height - scaledHeight) / 2;

    setViewport({ scale: nextScale, x, y });
  }, [containerSize, imageSize]);

  useEffect(() => {
    if (!hasInteracted) {
      fitToScreen();
    }
  }, [fitToScreen, hasInteracted, imageSize]);

  const handleImageLoad = useCallback(
    (event: SyntheticEvent<HTMLImageElement>) => {
      const target = event.currentTarget;
      if (target.naturalWidth && target.naturalHeight) {
        setImageSize({
          width: target.naturalWidth,
          height: target.naturalHeight,
        });
        if (!hasInteracted) {
          requestAnimationFrame(() => {
            fitToScreen();
          });
        }
      }
    },
    [fitToScreen, hasInteracted]
  );

  const handleAssetZoomChange = useCallback((value: number) => {
    setAssetZoom((current) => {
      if (!Number.isFinite(value)) {
        return current;
      }
      return clamp(value, MIN_ASSET_ZOOM, MAX_ASSET_ZOOM);
    });
  }, []);

  const adjustAssetZoom = useCallback((delta: number) => {
    setAssetZoom((current) => {
      const next = clamp(current + delta, MIN_ASSET_ZOOM, MAX_ASSET_ZOOM);
      return Number.isFinite(next) ? Number(next.toFixed(2)) : current;
    });
  }, []);

  const handleWheel = useCallback(
    (event: WheelEvent<HTMLDivElement>) => {
      event.preventDefault();
      const node = containerRef.current;
      if (!node) {
        return;
      }
      const rect = node.getBoundingClientRect();
      const pointerX = event.clientX - rect.left;
      const pointerY = event.clientY - rect.top;
      const zoomFactor = event.deltaY < 0 ? 1.1 : 0.9;

      setHasInteracted(true);
      setViewport((prev) => {
        const nextScale = clamp(prev.scale * zoomFactor, MIN_SCALE, MAX_SCALE);
        if (nextScale === prev.scale) {
          return prev;
        }
        const scaleRatio = nextScale / prev.scale;
        const offsetX = pointerX - prev.x;
        const offsetY = pointerY - prev.y;
        const nextX = pointerX - offsetX * scaleRatio;
        const nextY = pointerY - offsetY * scaleRatio;
        const clamped = clampOffsets(nextX, nextY, nextScale);
        return { scale: nextScale, x: clamped.x, y: clamped.y };
      });
    },
    [clampOffsets]
  );

  const handlePointerDown = useCallback((event: ReactPointerEvent<HTMLDivElement>) => {
    const target = event.target as HTMLElement;
    if (target.closest("button")) {
      return;
    }
    event.preventDefault();
    draggingRef.current = true;
    pointerRef.current = { x: event.clientX, y: event.clientY };
    setIsDragging(true);
    setHasInteracted(true);
    event.currentTarget.setPointerCapture(event.pointerId);
  }, []);

  const handlePointerMove = useCallback((event: ReactPointerEvent<HTMLDivElement>) => {
    if (!draggingRef.current) {
      return;
    }
    event.preventDefault();
    const { x, y } = pointerRef.current;
    const dx = event.clientX - x;
    const dy = event.clientY - y;
    pointerRef.current = { x: event.clientX, y: event.clientY };
    setViewport((prev) => {
      const nextX = prev.x + dx;
      const nextY = prev.y + dy;
      const clamped = clampOffsets(nextX, nextY, prev.scale);
      return { ...prev, x: clamped.x, y: clamped.y };
    });
  }, [clampOffsets]);

  const endPan = useCallback((event: ReactPointerEvent<HTMLDivElement>) => {
    if (!draggingRef.current) {
      return;
    }
    draggingRef.current = false;
    setIsDragging(false);
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }, []);

  const handlePointerUp = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      endPan(event);
    },
    [endPan]
  );

  const handlePointerLeave = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      endPan(event);
    },
    [endPan]
  );

  const adjustZoom = useCallback(
    (factor: number) => {
      if (!Number.isFinite(factor) || factor === 0) {
        return;
      }
      setHasInteracted(true);
      const centerX = containerSize.width / 2;
      const centerY = containerSize.height / 2;
      setViewport((prev) => {
        const nextScale = clamp(prev.scale * factor, MIN_SCALE, MAX_SCALE);
        if (nextScale === prev.scale) {
          return prev;
        }
        const scaleRatio = nextScale / prev.scale;
        const offsetX = centerX - prev.x;
        const offsetY = centerY - prev.y;
        const nextX = centerX - offsetX * scaleRatio;
        const nextY = centerY - offsetY * scaleRatio;
        const clamped = clampOffsets(nextX, nextY, nextScale);
        return { scale: nextScale, x: clamped.x, y: clamped.y };
      });
    },
    [clampOffsets, containerSize]
  );

  const zoomIn = useCallback(() => adjustZoom(1.2), [adjustZoom]);
  const zoomOut = useCallback(() => adjustZoom(1 / 1.2), [adjustZoom]);

  const resetView = useCallback(() => {
    setHasInteracted(false);
    fitToScreen();
  }, [fitToScreen]);

  const zoomPercent = Math.round(viewport.scale * 100);
  const scaledWidth = Math.round(imageSize.width * viewport.scale);
  const scaledHeight = Math.round(imageSize.height * viewport.scale);
  const overlayCount = preview?.findings.length ?? 0;
  return (
    <div className="grid gap-8 xl:grid-cols-[1.4fr,1fr]">
      <section className="space-y-6 rounded-3xl bg-white/5 p-6 shadow-card">
        <div className="flex flex-col gap-4 text-white lg:flex-row lg:items-start lg:justify-between">
          <div>
            <h1 className="text-2xl font-semibold">Analysis {analysis.id}</h1>
            <p className="text-sm text-slate-400">
              {analysis.overall_assessment ??
                "Analysis in progress. Findings will appear shortly."}
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <span className="rounded-full bg-accent/10 px-3 py-1 font-medium text-accent">
              {STATUS_LABEL[analysis.status] ?? analysis.status}
            </span>
            <span className="rounded-full bg-white/10 px-3 py-1 text-slate-200">
              {analysis.image.type}
            </span>
            <span className="rounded-full bg-white/10 px-3 py-1 text-slate-200">
              Captured {formatTimestamp(analysis.image.captured_at)}
            </span>
          </div>
        </div>

        <div className="rounded-3xl border border-white/10 bg-[#050B1C]/80 p-4">
          {displayImageSrc ? (
            <>
              <div
                ref={containerRef}
                className={clsx(
                  "relative h-[460px] overflow-hidden rounded-2xl border border-white/10 bg-[#0F172A]",
                  isDragging ? "cursor-grabbing" : "cursor-grab"
                )}
                onWheel={handleWheel}
                onPointerDown={handlePointerDown}
                onPointerMove={handlePointerMove}
                onPointerUp={handlePointerUp}
                onPointerLeave={handlePointerLeave}
                onPointerCancel={handlePointerUp}
                style={{ touchAction: "none" }}
              >
                <div
                  className="absolute left-0 top-0 select-none"
                  style={{
                    width: imageSize.width,
                    height: imageSize.height,
                    transform: `translate(${viewport.x}px, ${viewport.y}px) scale(${viewport.scale})`,
                    transformOrigin: "top left",
                  }}
                >
                  <img
                    src={displayImageSrc}
                    alt="Radiograph preview"
                    onLoad={handleImageLoad}
                    className="absolute left-0 top-0 h-full w-full select-none object-contain"
                    draggable={false}
                  />
                  {overlayCount > 0 ? (
                    <div className="pointer-events-none absolute inset-0">
                      {preview?.findings.map((finding) => {
                        const [x, y, w, h] = finding.bbox;
                        const isActive = finding.finding_id === selectedFindingId;
                        const isHovered = finding.finding_id === hoveredFindingId;
                        const strokeColor = toCssColor(finding.color_bgr);
                        return (
                          <button
                            key={finding.finding_id}
                            type="button"
                            className="pointer-events-auto absolute"
                            style={{
                              left: x,
                              top: y,
                              width: w,
                              height: h,
                            }}
                            onClick={() => setSelectedFindingId(finding.finding_id)}
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
                  ) : null}
                </div>
              </div>

              <div className="mt-4 flex flex-wrap items-center gap-3 text-xs text-slate-300">
                <span className="rounded-full bg-white/10 px-3 py-1">
                  Zoom {zoomPercent}%
                </span>
                <span className="rounded-full bg-white/10 px-3 py-1">
                  Viewport {scaledWidth} × {scaledHeight}
                </span>
                <span className="rounded-full bg-white/10 px-3 py-1">
                  Detections {overlayCount}
                </span>
                <span className="rounded-full bg-white/10 px-3 py-1">
                  Scroll to zoom · drag to pan
                </span>
              </div>

              <div className="mt-4 flex flex-wrap items-center gap-3 text-xs text-slate-200">
                <div className="inline-flex items-center gap-2">
                  <button
                    type="button"
                    onClick={zoomOut}
                    className="rounded-full border border-white/10 px-3 py-1 text-xs text-slate-200 transition hover:border-primary/50 hover:text-primary"
                  >
                    - Zoom out
                  </button>
                  <button
                    type="button"
                    onClick={zoomIn}
                    className="rounded-full border border-white/10 px-3 py-1 text-xs text-slate-200 transition hover:border-primary/50 hover:text-primary"
                  >
                    + Zoom in
                  </button>
                </div>
                <button
                  type="button"
                  onClick={resetView}
                  className="rounded-full border border-white/10 px-3 py-1 text-xs text-slate-200 transition hover:border-primary/50 hover:text-primary"
                >
                  Reset view
                </button>
                <label
                  className={clsx(
                    "inline-flex items-center gap-2 rounded-full border px-3 py-1",
                    canToggleOverlay
                      ? "border-white/10 bg-slate-900/60"
                      : "border-white/5 bg-slate-900/40 text-slate-500"
                  )}
                >
                  <span>Overlay</span>
                  <button
                    type="button"
                    onClick={() => setUseOverlay((current) => !current)}
                    disabled={!canToggleOverlay}
                    className={clsx(
                      "rounded-full px-3 py-1 text-[11px] font-semibold transition",
                      useOverlay
                        ? "bg-primary/20 text-primary hover:bg-primary/30"
                        : "bg-white/10 text-slate-200 hover:bg-white/20",
                      !canToggleOverlay && "cursor-not-allowed opacity-50"
                    )}
                  >
                    {useOverlay ? "Showing overlay" : "Original image"}
                  </button>
                </label>
              </div>
            </>
          ) : (
            <div className="flex h-[420px] items-center justify-center rounded-2xl border border-dashed border-white/15 text-sm text-slate-400">
              No preview image available for this analysis.
            </div>
          )}
        </div>

        <div className="rounded-3xl border border-white/5 bg-white/5 p-4">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-sm font-semibold text-white">Detections</h2>
              <p className="text-xs text-slate-400">
                {analysis.findings.length} total findings
              </p>
            </div>
          </div>
          <div className="mt-4 space-y-3">
            {analysis.findings.map((finding) => {
              const isActive = finding.finding_id === selectedFindingId;
              const severityTone =
                SEVERITY_TONE[finding.severity] ?? "border-white/20 text-slate-200";
              return (
                <button
                  key={finding.finding_id}
                  type="button"
                  onClick={() => setSelectedFindingId(finding.finding_id)}
                  className={clsx(
                    "w-full rounded-2xl border px-4 py-3 text-left transition",
                    isActive
                      ? "border-primary/70 bg-primary/10 text-white shadow-lg shadow-primary/10"
                      : "border-white/5 bg-slate-900/40 text-slate-200 hover:border-primary/40 hover:bg-primary/10/20"
                  )}
                >
                  <div className="flex items-center justify-between text-xs text-slate-400">
                    <span className="font-mono uppercase tracking-[0.2em]">
                      {finding.finding_id}
                    </span>
                    <span
                      className={clsx(
                        "rounded-full border px-2 py-0.5 text-[11px]",
                        severityTone
                      )}
                    >
                      {SEVERITY_LABEL[finding.severity] ?? finding.severity}
                    </span>
                  </div>
                  <div className="mt-2 flex items-center justify-between">
                    <p className="text-sm font-semibold text-white capitalize">
                      {finding.type}
                    </p>
                    <span className="text-xs text-slate-300">
                      Confidence {(finding.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-slate-400">
                    <span>
                      Tooth {finding.tooth_label ?? "—"}
                    </span>
                    <span>
                      BBox [{finding.region.bbox.map((v) => Math.round(v)).join(", ")}]
                    </span>
                    {overlayIds.has(finding.finding_id) ? (
                      <span className="rounded-full bg-primary/20 px-2 py-0.5 text-[11px] text-primary">
                        Overlay linked
                      </span>
                    ) : null}
                    {finding.confirmed != null ? (
                      <span className="rounded-full bg-white/10 px-2 py-0.5 text-[11px] text-slate-200">
                        {finding.confirmed ? "Confirmed" : "Pending review"}
                      </span>
                    ) : null}
                  </div>
                </button>
              );
            })}
            {analysis.findings.length === 0 ? (
              <div className="flex h-32 items-center justify-center rounded-2xl border border-dashed border-white/10 text-sm text-slate-400">
                No detections available for this analysis yet.
              </div>
            ) : null}
          </div>
        </div>
      </section>
      <section className="space-y-6">
        <div className="rounded-3xl bg-white/5 p-6 shadow-card">
          <h2 className="text-lg font-semibold text-white">Finding Details</h2>
          {selectedDetailFinding ? (
            <div className="mt-4 space-y-4 text-sm text-slate-200">
              <div>
                <p className="text-xs uppercase tracking-[0.2em] text-slate-400">
                  Tooth
                </p>
                <p className="text-xl font-semibold text-white">
                  {selectedDetailFinding.tooth_label ?? "Unlabeled"}
                </p>
                <p className="text-xs text-slate-400">
                  {selectedDetailFinding.type} ·{" "}
                  {SEVERITY_LABEL[selectedDetailFinding.severity] ??
                    selectedDetailFinding.severity}
                </p>
              </div>
              <div className="flex flex-wrap items-center gap-2 text-xs">
                <span
                  className={clsx(
                    "rounded-full border px-2 py-1",
                    SEVERITY_TONE[selectedDetailFinding.severity] ??
                      "border-white/20 text-slate-200"
                  )}
                >
                  Severity{" "}
                  {SEVERITY_LABEL[selectedDetailFinding.severity] ??
                    selectedDetailFinding.severity}
                </span>
                <span className="rounded-full border border-white/10 px-2 py-1 text-slate-200">
                  Confidence {(selectedDetailFinding.confidence * 100).toFixed(1)}%
                </span>
                <span className="rounded-full border border-white/10 px-2 py-1 text-slate-200">
                  Model {selectedDetailFinding.model_key} ·{" "}
                  {selectedDetailFinding.model_version}
                </span>
              </div>
              <div className="space-y-2 text-xs text-slate-300">
                <p>
                  Bounding box [
                  {selectedDetailFinding.region.bbox
                    .map((v) => Math.round(v))
                    .join(", ")}
                  ]
                </p>
                <p>
                  Centroid {formatPercentPair(selectedPreviewFinding?.centroid)}
                </p>
                {selectedDetailFinding.note ? (
                  <p>Note: {selectedDetailFinding.note}</p>
                ) : null}
              </div>
              {Object.keys(selectedDetailFinding.extra ?? {}).length ? (
                <div className="space-y-2 text-xs text-slate-300">
                  <p className="text-xs uppercase tracking-[0.2em] text-slate-400">
                    Model extra
                  </p>
                  <ul className="space-y-1">
                    {Object.entries(selectedDetailFinding.extra).map(
                      ([key, value]) => (
                        <li
                          key={key}
                          className="flex items-center justify-between gap-3"
                        >
                          <span className="text-slate-400">{key}</span>
                          <span className="text-white">
                            {formatExtraValue(value)}
                          </span>
                        </li>
                      )
                    )}
                  </ul>
                </div>
              ) : null}
              <div className="space-y-3">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <p className="text-xs uppercase tracking-[0.2em] text-slate-400">
                    Visual assets
                  </p>
                  <div className="flex items-center gap-2 text-xs text-slate-300">
                    <button
                      type="button"
                      onClick={() => adjustAssetZoom(-0.2)}
                      className="rounded-full border border-white/15 px-2 py-1 text-slate-200 transition hover:border-primary/50 hover:text-primary"
                    >
                      -
                    </button>
                    <input
                      type="range"
                      min={MIN_ASSET_ZOOM}
                      max={MAX_ASSET_ZOOM}
                      step={0.05}
                      value={assetZoom}
                      onChange={(event) =>
                        handleAssetZoomChange(Number(event.currentTarget.value))
                      }
                      className="h-1.5 w-32 accent-primary"
                    />
                    <button
                      type="button"
                      onClick={() => adjustAssetZoom(0.2)}
                      className="rounded-full border border-white/15 px-2 py-1 text-slate-200 transition hover:border-primary/50 hover:text-primary"
                    >
                      +
                    </button>
                    <span className="w-12 text-right text-slate-200">
                      {Math.round(assetZoom * 100)}%
                    </span>
                  </div>
                </div>
                {selectedAssets.gradcam ? (
                  <div className="relative mx-auto max-h-[360px] w-full overflow-auto rounded-2xl border border-white/10 bg-slate-950/40 p-3">
                    <img
                      src={selectedAssets.gradcam}
                      alt="Grad-CAM heatmap"
                      className="mx-auto h-auto w-auto max-w-full rounded-xl"
                      style={{
                        transform: `scale(${assetZoom})`,
                        transformOrigin: "top center",
                      }}
                    />
                  </div>
                ) : null}
                {!selectedAssets.gradcam && selectedAssets.heatmap ? (
                  <div className="relative mx-auto max-h-[360px] w-full overflow-auto rounded-2xl border border-white/10 bg-slate-950/40 p-3">
                    <img
                      src={selectedAssets.heatmap}
                      alt="Heatmap"
                      className="mx-auto h-auto w-auto max-w-full rounded-xl"
                      style={{
                        transform: `scale(${assetZoom})`,
                        transformOrigin: "top center",
                      }}
                    />
                  </div>
                ) : null}
                {selectedAssets.mask ? (
                  <div className="relative mx-auto max-h-[360px] w-full overflow-auto rounded-2xl border border-white/10 bg-slate-950/40 p-3">
                    <img
                      src={selectedAssets.mask}
                      alt="Segmentation mask"
                      className="mx-auto h-auto w-auto max-w-full rounded-xl"
                      style={{
                        transform: `scale(${assetZoom})`,
                        transformOrigin: "top center",
                      }}
                    />
                  </div>
                ) : null}
                {selectedAssets.crop ? (
                  <div className="relative mx-auto max-h-[360px] w-full overflow-auto rounded-2xl border border-white/10 bg-slate-950/40 p-3">
                    <img
                      src={selectedAssets.crop}
                      alt="Tooth crop"
                      className="mx-auto h-auto w-auto max-w-full rounded-xl"
                      style={{
                        transform: `scale(${assetZoom})`,
                        transformOrigin: "top center",
                      }}
                    />
                  </div>
                ) : null}
                {!selectedAssets.gradcam &&
                !selectedAssets.heatmap &&
                !selectedAssets.mask &&
                !selectedAssets.crop ? (
                  <div className="flex h-32 items-center justify-center rounded-2xl border border-dashed border-white/15 text-xs text-slate-400">
                    No auxiliary visualizations available.
                  </div>
                ) : null}
              </div>
            </div>
          ) : (
            <p className="mt-4 text-sm text-slate-400">
              Select a detection to inspect model outputs and notes.
            </p>
          )}
        </div>

        <div className="rounded-3xl bg-white/5 p-6 shadow-card">
          <h2 className="text-lg font-semibold text-white">Patient Overview</h2>
          <div className="mt-3 space-y-2 text-sm text-slate-300">
            <p className="flex justify-between">
              <span className="text-slate-400">Patient</span>
              <span className="text-white">{analysis.patient.name}</span>
            </p>
            <p className="flex justify-between">
              <span className="text-slate-400">Image Type</span>
              <span className="text-white">{analysis.image.type}</span>
            </p>
            <p className="flex justify-between">
              <span className="text-slate-400">Captured</span>
              <span className="text-white">
                {formatTimestamp(analysis.image.captured_at)}
              </span>
            </p>
          </div>
        </div>

        <div className="rounded-3xl bg-white/5 p-6 shadow-card">
          <h2 className="text-lg font-semibold text-white">
            Detected Conditions
          </h2>
          <ul className="mt-4 space-y-3 text-sm">
            {analysis.detected_conditions.map((condition) => (
              <li
                key={condition.label}
                className="flex items-center justify-between rounded-xl bg-white/5 px-3 py-2"
              >
                <span className="text-white">{condition.label}</span>
                <span className="rounded-full bg-primary/20 px-3 py-1 text-xs text-primary-subtle">
                  {condition.count} findings
                </span>
              </li>
            ))}
          </ul>
        </div>

        <div className="rounded-3xl bg-white/5 p-6 shadow-card">
          <h2 className="text-lg font-semibold text-white">Pipeline Progress</h2>
          <p className="text-xs text-slate-400">
            Overall confidence {(analysis.progress.overall_confidence * 100).toFixed(1)}%
          </p>
          <ul className="mt-4 space-y-4 text-sm text-slate-300">
            {analysis.progress.steps.map((step) => (
              <li key={step.title} className="rounded-2xl bg-white/5 p-3">
                <p className="font-medium text-white">{step.title}</p>
                <p className="text-xs text-slate-400">
                  {formatTimestamp(step.timestamp)} · {step.status}
                </p>
                <p className="mt-1 text-xs text-slate-400">{step.description}</p>
              </li>
            ))}
          </ul>
        </div>

        <div className="rounded-3xl bg-gradient-to-br from-primary/20 to-accent/20 p-6 text-sm text-white">
          <h2 className="text-lg font-semibold">Report Actions</h2>
          <div className="mt-4 flex flex-wrap gap-3">
            {analysis.report_actions.map((action) => (
              <a
                key={action.label}
                href={action.href}
                className="rounded-full bg-white/10 px-4 py-2 text-sm font-semibold text-white transition hover:bg-white/20"
              >
                {action.label}
              </a>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}
