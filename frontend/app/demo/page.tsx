"use client";

import { useEffect, useState, type ChangeEvent } from "react";
import { fetchDemoSamples, submitDemoInference } from "../../lib/api";
import type { DemoInferenceResult, DemoSampleSummary } from "../../lib/types";
import { resolveMediaUrl } from "../../lib/media";

function formatProbability(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

export default function DemoPage(): JSX.Element {
  const [samples, setSamples] = useState<DemoSampleSummary[]>([]);
  const [selectedSample, setSelectedSample] = useState<string | null>(null);
  const [result, setResult] = useState<DemoInferenceResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [inputKey, setInputKey] = useState(() => Date.now());

  useEffect(() => {
    void (async () => {
      try {
        const response = await fetchDemoSamples();
        setSamples(response.items);
      } catch (err) {
        console.error("Failed to load demo samples", err);
        setError("Unable to load demo samples. Fallback data will be used.");
      }
    })();
  }, []);

  const handleSample = async (sampleId: string, rerun = false): Promise<void> => {
    setError(null);
    setSelectedSample(sampleId);
    setLoading(true);
    try {
      const response = await submitDemoInference({ sampleId, rerun });
      const sample = rerun ? null : samples.find((item) => item.id === sampleId) ?? null;
      setResult({
        ...response,
        findings: response.findings.map((finding) => ({
          ...finding,
          cam_path: finding.cam_path ?? sample?.cam_paths[finding.fdi] ?? null
        }))
      });
    } catch (err) {
      console.error("Demo inference failed", err);
      setError(err instanceof Error ? err.message : "Inference failed");
    } finally {
      setLoading(false);
    }
  };

  const handleUpload = async (event: ChangeEvent<HTMLInputElement>): Promise<void> => {
    if (!event.target.files || event.target.files.length === 0) {
      return;
    }
    const file = event.target.files[0];
    setError(null);
    setSelectedSample(null);
    setLoading(true);
    try {
      const response = await submitDemoInference({ file });
      setResult(response);
    } catch (err) {
      console.error("Demo upload failed", err);
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setLoading(false);
      setInputKey(Date.now());
    }
  };

  return (
    <div className="space-y-8 p-6">
      <div className="space-y-2">
        <h1 className="text-3xl font-semibold">Cross-Attention Grad-CAM Demo</h1>
        <p className="text-muted-foreground max-w-3xl">
          Explore the lightweight inference API that powers the cross-attention Grad-CAM workflow. Choose a bundled
          sample or upload your own panoramic radiograph to generate per-tooth predictions, overlay visualisations, and
          Grad-CAM heatmaps.
        </p>
      </div>

      <div className="grid gap-8 lg:grid-cols-[320px,1fr]">
        <aside className="space-y-6 rounded-lg border bg-card p-4 shadow-sm">
          <div className="space-y-3">
            <h2 className="text-xl font-medium">Sample Gallery</h2>
            <p className="text-sm text-muted-foreground">
              Use a curated case to instantly preview inference results or rerun the full pipeline on the server.
            </p>
            <div className="space-y-3">
              {samples.map((sample) => {
                const isActive = sample.id === selectedSample;
                const previewUrl = resolveMediaUrl(sample.image_path);
                return (
                  <div key={sample.id} className={`rounded-md border p-3 ${isActive ? "border-primary" : "border-border"}`}>
                    {previewUrl ? (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img
                        src={previewUrl}
                        alt={sample.title}
                        className="mb-2 h-32 w-full rounded object-cover"
                      />
                    ) : null}
                    <h3 className="font-medium">{sample.title}</h3>
                    <p className="text-xs text-muted-foreground">{sample.description}</p>
                    <div className="mt-3 flex flex-wrap gap-2">
                      <button
                        type="button"
                        className="rounded bg-primary px-3 py-1 text-sm font-medium text-primary-foreground disabled:opacity-50"
                        onClick={() => void handleSample(sample.id, false)}
                        disabled={loading}
                      >
                        View Demo
                      </button>
                      <button
                        type="button"
                        className="rounded border px-3 py-1 text-sm font-medium disabled:opacity-50"
                        onClick={() => void handleSample(sample.id, true)}
                        disabled={loading}
                      >
                        Rerun Model
                      </button>
                    </div>
                  </div>
                );
              })}
              {samples.length === 0 ? (
                <p className="text-sm text-muted-foreground">No samples available. Add entries to the manifest.</p>
              ) : null}
            </div>
          </div>

          <div className="space-y-3">
            <h2 className="text-xl font-medium">Upload</h2>
            <p className="text-sm text-muted-foreground">
              Upload a single PNG or JPG panoramic radiograph. Files are processed in-memory and removed after
              inference.
            </p>
            <input
              key={inputKey}
              type="file"
              accept="image/png,image/jpeg,image/jpg"
              onChange={handleUpload}
              className="w-full text-sm"
              disabled={loading}
            />
          </div>
        </aside>

        <section className="space-y-6">
          {error ? <p className="rounded bg-destructive/10 p-3 text-sm text-destructive">{error}</p> : null}
          {loading ? <p className="text-sm text-muted-foreground">Running inference...</p> : null}

          {result ? (
            <div className="space-y-4 rounded-lg border bg-card p-4 shadow-sm">
              <div className="flex flex-col gap-4 lg:flex-row">
                <div className="flex-1 space-y-2">
                  <h2 className="text-2xl font-semibold">Inference Output</h2>
                  <p className="text-sm text-muted-foreground">
                    Request ID: <span className="font-mono">{result.request_id}</span>
                  </p>
                  {result.warnings.length > 0 ? (
                    <ul className="list-disc space-y-1 pl-5 text-sm text-yellow-600">
                      {result.warnings.map((warning) => (
                        <li key={warning}>{warning}</li>
                      ))}
                    </ul>
                  ) : null}
                </div>
                {result.overlay_url ? (
                  <div className="relative min-h-[220px] w-full overflow-hidden rounded-md border lg:w-1/2">
                    {resolveMediaUrl(result.overlay_url) ? (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img
                        src={resolveMediaUrl(result.overlay_url)}
                        alt="Inference overlay"
                        className="h-full w-full object-cover"
                      />
                    ) : null}
                  </div>
                ) : null}
              </div>

              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-border text-sm">
                  <thead>
                    <tr className="bg-muted/40 text-left">
                      <th className="px-3 py-2 font-medium">FDI</th>
                      <th className="px-3 py-2 font-medium">Probability</th>
                      <th className="px-3 py-2 font-medium">Threshold</th>
                      <th className="px-3 py-2 font-medium">Prediction</th>
                      <th className="px-3 py-2 font-medium">Bounding Box</th>
                      <th className="px-3 py-2 font-medium">Grad-CAM</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border">
                    {result.findings.map((finding) => {
                      const camUrl = finding.cam_path ? resolveMediaUrl(finding.cam_path) : undefined;
                      return (
                        <tr key={`${result.request_id}-${finding.fdi}-${finding.bbox.x1}`} className="align-top">
                          <td className="px-3 py-2 font-medium">{finding.fdi}</td>
                          <td className="px-3 py-2">{formatProbability(finding.prob_caries)}</td>
                          <td className="px-3 py-2">{formatProbability(finding.thr_used)}</td>
                          <td className="px-3 py-2">
                            <span
                              className={`rounded px-2 py-1 text-xs font-semibold ${
                                finding.pred ? "bg-destructive/10 text-destructive" : "bg-muted text-muted-foreground"
                              }`}
                            >
                              {finding.pred ? "Caries" : "Normal"}
                            </span>
                          </td>
                          <td className="px-3 py-2 text-xs font-mono">
                            [{finding.bbox.x1}, {finding.bbox.y1}] → [{finding.bbox.x2}, {finding.bbox.y2}]
                          </td>
                          <td className="px-3 py-2">
                            {camUrl ? (
                              // eslint-disable-next-line @next/next/no-img-element
                              <img
                                src={camUrl}
                                alt={`Grad-CAM for ${finding.fdi}`}
                                className="h-20 w-20 rounded object-cover"
                              />
                            ) : (
                              <span className="text-xs text-muted-foreground">Not available</span>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          ) : (
            <div className="rounded-lg border bg-muted/20 p-6 text-sm text-muted-foreground">
              Select a sample or upload an image to generate predictions.
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
