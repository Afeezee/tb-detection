"use client";

import { useEffect, useState } from "react";
import { fetchBenchmarks, type BenchmarkRow } from "@/lib/api";

function pct(x: number) {
  return (x * 100).toFixed(2) + "%";
}

export default function BenchmarkPage() {
  const [rows, setRows] = useState<BenchmarkRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const data = await fetchBenchmarks();
        if (!cancelled) setRows(data.rows);
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : "Failed to load metrics");
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <section className="space-y-8">
      <div>
        <h1 className="text-2xl font-semibold text-clinical-ink">Benchmark results</h1>
        <p className="text-sm text-clinical-muted">
          Comparison of DenseNet121 and Hybrid CNN+ViT, single-source versus multi-source
          training, on the internal test split and on the external TBX11K generalisation split.
        </p>
      </div>

      {loading && <p className="text-sm text-clinical-muted">Loading metrics…</p>}
      {error && (
        <p className="rounded border border-clinical-positive/40 bg-clinical-positive/5 p-3 text-sm text-clinical-positive">
          {error}
        </p>
      )}

      {!loading && !error && (
        <div className="overflow-hidden rounded-xl border border-clinical-border bg-clinical-surface">
          <table className="w-full text-sm">
            <thead className="bg-clinical-bg text-left text-xs uppercase tracking-wide text-clinical-muted">
              <tr>
                <th className="px-4 py-3">Model</th>
                <th className="px-4 py-3">Training regime</th>
                <th className="px-4 py-3">Test set</th>
                <th className="px-4 py-3">Sensitivity</th>
                <th className="px-4 py-3">Specificity</th>
                <th className="px-4 py-3">F1</th>
                <th className="px-4 py-3">AUC-ROC</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={i} className="border-t border-clinical-border align-top">
                  <td className="px-4 py-3 font-medium text-clinical-ink">{r.model}</td>
                  <td className="px-4 py-3 text-clinical-muted">{r.training_regime}</td>
                  <td className="px-4 py-3 text-clinical-muted">{r.test_set}</td>
                  <td className="px-4 py-3">{pct(r.sensitivity)}</td>
                  <td className="px-4 py-3">{pct(r.specificity)}</td>
                  <td className="px-4 py-3">{pct(r.f1)}</td>
                  <td className="px-4 py-3">{r.auc_roc.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="grid gap-4 md:grid-cols-2">
        <figure className="rounded-xl border border-clinical-border bg-clinical-surface p-4">
          <figcaption className="mb-2 text-xs uppercase tracking-wide text-clinical-muted">
            DenseNet121 — ROC curve
          </figcaption>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src="/roc/densenet121_roc.png"
            alt="DenseNet121 ROC curve"
            className="w-full rounded"
          />
        </figure>
        <figure className="rounded-xl border border-clinical-border bg-clinical-surface p-4">
          <figcaption className="mb-2 text-xs uppercase tracking-wide text-clinical-muted">
            Hybrid CNN+ViT — ROC curve
          </figcaption>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src="/roc/hybrid_roc.png"
            alt="Hybrid CNN+ViT ROC curve"
            className="w-full rounded"
          />
        </figure>
      </div>

      <p className="text-xs text-clinical-muted">
        ROC images are served from <code>frontend/public/roc/</code>. Copy your saved plots as
        <code> densenet121_roc.png</code> and <code>hybrid_roc.png</code> into that folder.
      </p>
    </section>
  );
}
