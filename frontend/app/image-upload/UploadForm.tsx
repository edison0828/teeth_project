'use client';

import { ChangeEvent, DragEvent, FormEvent, useMemo, useRef, useState } from "react";

import { uploadImage } from "../../lib/api";
import { readToken } from "../../lib/auth-storage";
import type { ImageUploadResponse, PatientSummary } from "../../lib/types";

type UploadState =
  | { state: "idle" }
  | { state: "uploading" }
  | { state: "success"; payload: ImageUploadResponse }
  | { state: "error"; message: string };

type UploadFormProps = {
  patients: PatientSummary[];
};

const STUDY_TYPES = [
  "Panoramic Scan",
  "Bitewing",
  "Full-mouth X-ray",
  "CBCT"
];

const PRIORITY_OPTIONS: Array<{ value: string; label: string }> = [
  { value: "standard", label: "Standard" },
  { value: "urgent", label: "Urgent" }
];

function formatDateTime(value?: Date): string {
  const date = value ?? new Date();
  const offset = date.getTimezoneOffset();
  const local = new Date(date.getTime() - offset * 60 * 1000);
  return local.toISOString().slice(0, 16);
}

export default function UploadForm({ patients }: UploadFormProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [patientId, setPatientId] = useState<string>(patients[0]?.id ?? "");
  const [studyType, setStudyType] = useState<string>(STUDY_TYPES[0]);
  const [priority, setPriority] = useState<string>(PRIORITY_OPTIONS[0].value);
  const [autoAnalyze, setAutoAnalyze] = useState<boolean>(true);
  const [capturedAt, setCapturedAt] = useState<string>(formatDateTime());
  const [notes, setNotes] = useState<string>("");
  const [status, setStatus] = useState<UploadState>({ state: "idle" });

  const patientOptions = useMemo(() => patients.map((patient) => ({ value: patient.id, label: patient.name })), [patients]);
  const hasPatients = patientOptions.length > 0;

  const selectedFileName = file?.name ?? "No file selected";

  const handleFileSelection = (selected: File | null) => {
    if (selected) {
      setFile(selected);
      setStatus({ state: "idle" });
    }
  };

  const onFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const selected = event.target.files?.[0] ?? null;
    handleFileSelection(selected);
  };

  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    const dropped = event.dataTransfer.files?.[0] ?? null;
    handleFileSelection(dropped);
  };

  const triggerFileDialog = () => {
    fileInputRef.current?.click();
  };

  const submitForm = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    if (!file) {
      setStatus({ state: "error", message: "Please select an image or DICOM file to upload." });
      return;
    }

    if (!hasPatients) {
      setStatus({ state: "error", message: "Please add a patient before uploading." });
      return;
    }

    if (!patientId) {
      setStatus({ state: "error", message: "Please choose a patient before uploading." });
      return;
    }

    const token = readToken();
    if (!token) {
      setStatus({ state: "error", message: "請先登入後再上傳影像。" });
      return;
    }

    const capturedValue = capturedAt.length === 16 ? `${capturedAt}:00` : capturedAt;

    const formData = new FormData();
    formData.append("patient_id", patientId);
    formData.append("type", studyType);
    formData.append("captured_at", capturedValue);
    formData.append("auto_analyze", autoAnalyze ? "true" : "false");
    formData.append("priority", priority);
    if (notes.trim()) {
      formData.append("notes", notes.trim());
    }
    formData.append("file", file);

    setStatus({ state: "uploading" });
    try {
      const response = await uploadImage(formData, token);
      setStatus({ state: "success", payload: response });
      setFile(null);
      setNotes("");
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : "Upload failed. Please try again.";
      setStatus({ state: "error", message });
    }
  };

  const statusLabel = (() => {
    switch (status.state) {
      case "uploading":
        return "Uploading image to the analysis queue...";
      case "success":
        return `Upload completed. Image ID: ${status.payload.image.id}`;
      case "error":
        return status.message;
      default:
        return "Select a file and submit to trigger preprocessing.";
    }
  })();

  return (
    <form onSubmit={submitForm} className="grid gap-8 xl:grid-cols-[1.2fr,1fr]">
      <section className="rounded-3xl bg-white/5 p-6 shadow-card">
        <h1 className="text-2xl font-semibold text-white">Upload &amp; Preprocessing</h1>
        <p className="mt-1 text-sm text-slate-400">
          Drag and drop CBCT, panoramic, or bitewing studies. The system pre-validates DICOM metadata before preprocessing.
        </p>
        <div
          className="mt-8 flex h-64 flex-col items-center justify-center rounded-3xl border border-dashed border-primary/40 bg-white/5 text-center"
          onClick={triggerFileDialog}
          onDragOver={(event) => event.preventDefault()}
          onDrop={onDrop}
          onKeyDown={(event) => {
            if (event.key === "Enter" || event.key === " ") {
              event.preventDefault();
              triggerFileDialog();
            }
          }}
          role="button"
          tabIndex={0}
        >
          <p className="text-lg font-semibold text-primary-subtle">{file ? selectedFileName : "Drop X-ray images here"}</p>
          <p className="mt-2 text-sm text-slate-400">or click to browse files</p>
          <button
            type="button"
            onClick={triggerFileDialog}
            className="mt-6 rounded-full bg-primary px-5 py-2 text-sm font-semibold text-slate-900 shadow-lg shadow-primary/40"
          >
            Select Files
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".dcm,.jpg,.jpeg,.png"
            className="hidden"
            onChange={onFileChange}
          />
        </div>
        <div className="mt-6 grid gap-4 md:grid-cols-2">
          <div className="rounded-2xl bg-white/5 p-4">
            <p className="text-xs uppercase tracking-wide text-slate-400">Preprocessing</p>
            <ul className="mt-3 space-y-2 text-sm">
              <li className="flex items-center justify-between">
                <span>Auto orientation</span>
                <span className="rounded-full bg-accent/20 px-2 py-0.5 text-xs text-accent">Enabled</span>
              </li>
              <li className="flex items-center justify-between">
                <span>Noise reduction</span>
                <span className="rounded-full bg-white/10 px-2 py-0.5 text-xs text-slate-300">Adaptive</span>
              </li>
              <li className="flex items-center justify-between">
                <span>Contrast harmonization</span>
                <span className="rounded-full bg-white/10 px-2 py-0.5 text-xs text-slate-300">AI optimized</span>
              </li>
            </ul>
          </div>
          <div className="rounded-2xl bg-white/5 p-4">
            <p className="text-xs uppercase tracking-wide text-slate-400">Upload Status</p>
            <div className="mt-3 space-y-3 text-sm">
              <div className="rounded-xl bg-white/5 px-3 py-2">
                <p className="font-medium text-white">{selectedFileName}</p>
                <p className={`text-xs ${status.state === "error" ? "text-red-400" : status.state === "success" ? "text-accent" : "text-slate-400"}`}>
                  {statusLabel}
                </p>
              </div>
              {status.state === "success" && (
                <div className="rounded-xl bg-white/5 px-3 py-2 text-xs text-slate-300">
                  <p>Stored at: {status.payload.image.storage_uri ?? "local storage"}</p>
                  {status.payload.auto_analyze ? (
                    <p>Analysis job queued automatically.</p>
                  ) : (
                    <p>Auto analysis not requested.</p>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      </section>

      <section className="rounded-3xl bg-white/5 p-6 shadow-card">
        <h2 className="text-lg font-semibold text-white">Associated Patient</h2>
        <div className="mt-4 space-y-4 text-sm text-slate-300">
          <div>
            <label className="text-xs uppercase tracking-wide text-slate-400">Patient</label>
            <select
              value={patientId}
              onChange={(event) => setPatientId(event.target.value)}
              disabled={!hasPatients}
              className="mt-2 w-full rounded-xl border border-white/10 bg-[#0B142A] px-3 py-2 disabled:opacity-60"
            >
              {patientOptions.map((patient) => (
                <option key={patient.value} value={patient.value}>
                  {patient.label}
                </option>
              ))}
            </select>
            {!hasPatients && (
              <p className="mt-2 text-xs text-red-300">Add a patient from the Patients page before uploading.</p>
            )}
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-xs uppercase tracking-wide text-slate-400">Study Type</label>
              <select
                value={studyType}
                onChange={(event) => setStudyType(event.target.value)}
                className="mt-2 w-full rounded-xl border border-white/10 bg-[#0B142A] px-3 py-2"
              >
                {STUDY_TYPES.map((type) => (
                  <option key={type} value={type}>
                    {type}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="text-xs uppercase tracking-wide text-slate-400">Priority</label>
              <select
                value={priority}
                onChange={(event) => setPriority(event.target.value)}
                className="mt-2 w-full rounded-xl border border-white/10 bg-[#0B142A] px-3 py-2"
              >
                {PRIORITY_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>
          </div>
          <div>
            <label className="text-xs uppercase tracking-wide text-slate-400">Captured At</label>
            <input
              type="datetime-local"
              value={capturedAt}
              onChange={(event) => setCapturedAt(event.target.value)}
              className="mt-2 w-full rounded-xl border border-white/10 bg-[#0B142A] px-3 py-2"
            />
          </div>
          <div>
            <label className="text-xs uppercase tracking-wide text-slate-400">Immediate AI Analysis</label>
            <div className="mt-2 flex items-center gap-3">
              <input
                type="checkbox"
                checked={autoAnalyze}
                onChange={(event) => setAutoAnalyze(event.target.checked)}
                className="h-4 w-4 rounded border-white/20 bg-transparent"
              />
              <span>Trigger AI pipeline after upload</span>
            </div>
          </div>
          <div>
            <label className="text-xs uppercase tracking-wide text-slate-400">Clinical Notes</label>
            <textarea
              rows={4}
              value={notes}
              onChange={(event) => setNotes(event.target.value)}
              className="mt-2 w-full rounded-xl border border-white/10 bg-[#0B142A] px-3 py-2"
              placeholder="Highlight findings or instructions for radiologist review"
            />
          </div>
          <button
            type="submit"
            disabled={status.state === "uploading" || !hasPatients}
            className="w-full rounded-full bg-gradient-to-r from-primary to-accent px-4 py-3 text-sm font-semibold text-slate-900 shadow-lg shadow-primary/40 disabled:opacity-60"
          >
            {status.state === "uploading" ? "Uploading..." : "Confirm & Analyze"}
          </button>
          {status.state === "error" && (
            <p className="text-xs text-red-400">{status.message}</p>
          )}
        </div>
      </section>
    </form>
  );
}






