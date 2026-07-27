"use client";

import { useMemo, useState } from "react";
import ImageDropzone from "@/components/ImageDropzone";
import ResultBadge from "@/components/ResultBadge";
import { postPrediction, type ModelName, type PredictionResponse } from "@/lib/api";
import { exportReport } from "@/lib/report";

export default function PredictPage() {
  const [file, setFile] = useState<File | null>(null);
  const [model, setModel] = useState<ModelName>("densenet121");
  const [patientRef, setPatientRef] = useState("");
  const [clinicianNotes, setClinicianNotes] = useState("");
  const [persist, setPersist] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);

  const previewUrl = useMemo(() => (file ? URL.createObjectURL(file) : null), [file]);

  async function runInference() {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await postPrediction({
        file,
        model,
        patientRef: patientRef.trim() || undefined,
        clinicianNotes: clinicianNotes.trim() || undefined,
        persist,
      });
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prediction failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold text-clinical-ink">Analyse a chest X-ray</h1>
        <p className="text-sm text-clinical-muted">
          Upload a PA chest radiograph. The image is CLAHE-enhanced, resized to 224×224, and
          passed through the selected model. Grad-CAM highlights the lung regions driving the
          prediction.
        </p>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <div className="space-y-4 rounded-xl border border-clinical-border bg-clinical-surface p-6">
          <ImageDropzone onFile={setFile} file={file} disabled={loading} />

          <div className="grid grid-cols-2 gap-3">
            <label className="text-sm">
              <span className="text-clinical-muted">Model</span>
              <select
                value={model}
                onChange={(e) => setModel(e.target.value as ModelName)}
                disabled={loading}
                className="mt-1 w-full rounded-md border border-clinical-border bg-white px-3 py-2 text-sm"
              >
                <option value="densenet121">DenseNet121 (baseline, faster)</option>
                <option value="hybrid">Hybrid CNN+ViT (novelty)</option>
              </select>
            </label>
            <label className="text-sm">
              <span className="text-clinical-muted">Patient reference</span>
              <input
                value={patientRef}
                onChange={(e) => setPatientRef(e.target.value)}
                placeholder="e.g. PT-00123"
                disabled={loading}
                className="mt-1 w-full rounded-md border border-clinical-border bg-white px-3 py-2 text-sm"
              />
            </label>
          </div>

          <label className="block text-sm">
            <span className="text-clinical-muted">Clinician notes (optional)</span>
            <textarea
              value={clinicianNotes}
              onChange={(e) => setClinicianNotes(e.target.value)}
              rows={3}
              disabled={loading}
              className="mt-1 w-full rounded-md border border-clinical-border bg-white px-3 py-2 text-sm"
            />
          </label>

          <label className="flex items-center gap-2 text-sm text-clinical-muted">
            <input
              type="checkbox"
              checked={persist}
              onChange={(e) => setPersist(e.target.checked)}
              disabled={loading}
            />
            Save this prediction to the patient history
          </label>

          <button
            type="button"
            onClick={runInference}
            disabled={!file || loading}
            className="w-full rounded-md bg-clinical-accent px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
          >
            {loading ? "Analysing…" : "Run analysis"}
          </button>

          {error && (
            <p className="rounded border border-clinical-positive/40 bg-clinical-positive/5 p-3 text-sm text-clinical-positive">
              {error}
            </p>
          )}
        </div>

        <div className="space-y-4 rounded-xl border border-clinical-border bg-clinical-surface p-6">
          {!result && (
            <div className="grid grid-cols-2 gap-4">
              <div>
                <div className="text-xs uppercase tracking-wide text-clinical-muted">
                  Uploaded image
                </div>
                <div className="mt-2 flex h-56 items-center justify-center rounded-lg border border-dashed border-clinical-border bg-clinical-bg">
                  {previewUrl ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img
                      src={previewUrl}
                      alt="uploaded X-ray"
                      className="max-h-full max-w-full rounded"
                    />
                  ) : (
                    <span className="text-sm text-clinical-muted">No image selected</span>
                  )}
                </div>
              </div>
              <div>
                <div className="text-xs uppercase tracking-wide text-clinical-muted">
                  Grad-CAM overlay
                </div>
                <div className="mt-2 flex h-56 items-center justify-center rounded-lg border border-dashed border-clinical-border bg-clinical-bg">
                  <span className="text-sm text-clinical-muted">Run analysis to view</span>
                </div>
              </div>
            </div>
          )}

          {result && (
            <>
              <ResultBadge label={result.label} confidence={result.confidence} />

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <div className="text-xs uppercase tracking-wide text-clinical-muted">
                    Uploaded image
                  </div>
                  {previewUrl && (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img
                      src={previewUrl}
                      alt="uploaded X-ray"
                      className="mt-2 max-h-56 w-full rounded object-contain"
                    />
                  )}
                </div>
                <div>
                  <div className="text-xs uppercase tracking-wide text-clinical-muted">
                    Grad-CAM overlay
                  </div>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={`data:image/png;base64,${result.gradcam_base64}`}
                    alt="Grad-CAM heatmap"
                    className="mt-2 max-h-56 w-full rounded object-contain"
                  />
                </div>
              </div>

              <dl className="grid grid-cols-2 gap-3 rounded-md border border-clinical-border p-3 text-sm">
                <div>
                  <dt className="text-clinical-muted">Model</dt>
                  <dd className="text-clinical-ink">{result.model_used}</dd>
                </div>
                <div>
                  <dt className="text-clinical-muted">Reference</dt>
                  <dd className="text-clinical-ink">{result.patient_ref || "—"}</dd>
                </div>
                <div>
                  <dt className="text-clinical-muted">P(Normal)</dt>
                  <dd className="text-clinical-ink">
                    {(result.probabilities["Normal"] * 100).toFixed(1)}%
                  </dd>
                </div>
                <div>
                  <dt className="text-clinical-muted">P(TB-positive)</dt>
                  <dd className="text-clinical-ink">
                    {(result.probabilities["TB-positive"] * 100).toFixed(1)}%
                  </dd>
                </div>
              </dl>

              <button
                type="button"
                onClick={() => exportReport("report-source")}
                className="w-full rounded-md border border-clinical-border px-4 py-2 text-sm font-medium text-clinical-ink hover:bg-clinical-bg"
              >
                Export PDF report
              </button>

              {/* Hidden source for the print/PDF window. */}
              <div id="report-source" className="hidden">
                <h1>Tuberculosis screening report</h1>
                <p className="muted">
                  Generated {new Date(result.created_at).toLocaleString("en-GB")} · Reference{" "}
                  {result.patient_ref || "—"}
                </p>
                <div className="card">
                  <div>
                    Prediction:{" "}
                    <span className={result.label === "TB-positive" ? "positive" : "negative"}>
                      {result.label}
                    </span>{" "}
                    (confidence {(result.confidence * 100).toFixed(1)}%)
                  </div>
                  <div>Model: {result.model_used}</div>
                </div>
                <div className="row">
                  <div className="card">
                    <div className="muted">Uploaded image</div>
                    {previewUrl && <img src={previewUrl} alt="upload" />}
                  </div>
                  <div className="card">
                    <div className="muted">Grad-CAM overlay</div>
                    <img
                      src={`data:image/png;base64,${result.gradcam_base64}`}
                      alt="gradcam"
                    />
                  </div>
                </div>
                <table>
                  <tbody>
                    <tr>
                      <th>P(Normal)</th>
                      <td>{(result.probabilities["Normal"] * 100).toFixed(1)}%</td>
                    </tr>
                    <tr>
                      <th>P(TB-positive)</th>
                      <td>{(result.probabilities["TB-positive"] * 100).toFixed(1)}%</td>
                    </tr>
                    {clinicianNotes && (
                      <tr>
                        <th>Notes</th>
                        <td>{clinicianNotes}</td>
                      </tr>
                    )}
                  </tbody>
                </table>
                <p className="muted" style={{ marginTop: 16 }}>
                  Research prototype — not a licensed medical device. Results must be reviewed by
                  a qualified clinician.
                </p>
              </div>
            </>
          )}
        </div>
      </div>
    </section>
  );
}
