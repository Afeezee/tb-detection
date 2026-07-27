"use client";

import { useEffect, useMemo, useState } from "react";
import { fetchHistory, type HistoryRow } from "@/lib/api";

const PAGE_SIZE = 15;

export default function HistoryPage() {
  const [rows, setRows] = useState<HistoryRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(0);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const data = await fetchHistory(200);
        if (!cancelled) setRows(data.items);
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : "Failed to load history");
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const pageCount = Math.max(1, Math.ceil(rows.length / PAGE_SIZE));
  const paged = useMemo(
    () => rows.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE),
    [rows, page]
  );

  return (
    <section className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold text-clinical-ink">Patient history</h1>
        <p className="text-sm text-clinical-muted">
          Every prediction saved through the /predict page is persisted to Neon Postgres and
          appears here. Sorted by most recent first.
        </p>
      </div>

      {loading && <p className="text-sm text-clinical-muted">Loading history…</p>}
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
                <th className="px-4 py-3">Timestamp</th>
                <th className="px-4 py-3">Patient ref</th>
                <th className="px-4 py-3">Result</th>
                <th className="px-4 py-3">Confidence</th>
                <th className="px-4 py-3">Model</th>
                <th className="px-4 py-3">Image</th>
              </tr>
            </thead>
            <tbody>
              {paged.length === 0 && (
                <tr>
                  <td colSpan={6} className="px-4 py-6 text-center text-clinical-muted">
                    No predictions recorded yet.
                  </td>
                </tr>
              )}
              {paged.map((row) => (
                <tr key={row.id} className="border-t border-clinical-border">
                  <td className="px-4 py-3 text-clinical-muted">
                    {new Date(row.created_at).toLocaleString("en-GB")}
                  </td>
                  <td className="px-4 py-3">{row.patient_ref || "—"}</td>
                  <td
                    className={
                      row.prediction === "TB-positive"
                        ? "px-4 py-3 font-medium text-clinical-positive"
                        : "px-4 py-3 font-medium text-clinical-negative"
                    }
                  >
                    {row.prediction}
                  </td>
                  <td className="px-4 py-3">{(row.confidence * 100).toFixed(1)}%</td>
                  <td className="px-4 py-3 text-clinical-muted">{row.model_name}</td>
                  <td className="px-4 py-3 text-clinical-muted">{row.image_filename}</td>
                </tr>
              ))}
            </tbody>
          </table>

          <div className="flex items-center justify-between border-t border-clinical-border px-4 py-3 text-sm">
            <span className="text-clinical-muted">
              Page {page + 1} of {pageCount} · {rows.length} records
            </span>
            <div className="flex gap-2">
              <button
                type="button"
                onClick={() => setPage((p) => Math.max(0, p - 1))}
                disabled={page === 0}
                className="rounded border border-clinical-border px-3 py-1 disabled:opacity-40"
              >
                Previous
              </button>
              <button
                type="button"
                onClick={() => setPage((p) => Math.min(pageCount - 1, p + 1))}
                disabled={page >= pageCount - 1}
                className="rounded border border-clinical-border px-3 py-1 disabled:opacity-40"
              >
                Next
              </button>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
