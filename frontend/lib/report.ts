// Client-side PDF export via the browser's print pipeline — no extra deps.
// The /predict page renders a hidden print-only view containing the summary
// plus the Grad-CAM overlay; calling this opens the print dialog which the
// user can save as PDF from any modern browser.

export function exportReport(elementId: string) {
  const source = document.getElementById(elementId);
  if (!source) return;

  const win = window.open("", "_blank", "noopener,noreferrer,width=900,height=1200");
  if (!win) return;

  win.document.write(`<!doctype html>
    <html lang="en-GB">
      <head>
        <meta charset="utf-8" />
        <title>TB Detection Report</title>
        <style>
          body {
            font-family: ui-sans-serif, system-ui, "Segoe UI", Arial, sans-serif;
            color: #0f1e2e;
            padding: 32px;
            line-height: 1.5;
          }
          h1 { font-size: 20px; margin: 0 0 4px; }
          .muted { color: #5b6b7c; font-size: 12px; }
          .row { display: flex; gap: 24px; margin-top: 16px; }
          .card { border: 1px solid #e5e9ef; border-radius: 8px; padding: 16px; flex: 1; }
          img { max-width: 100%; border-radius: 6px; }
          .positive { color: #c1121f; font-weight: 600; }
          .negative { color: #118a3c; font-weight: 600; }
          table { border-collapse: collapse; width: 100%; margin-top: 12px; }
          th, td { border-bottom: 1px solid #e5e9ef; padding: 6px 8px; text-align: left; font-size: 13px; }
          @media print { .no-print { display: none; } }
        </style>
      </head>
      <body>${source.innerHTML}
        <div class="no-print" style="margin-top:24px;">
          <button onclick="window.print()">Print / Save as PDF</button>
        </div>
      </body>
    </html>`);
  win.document.close();
}
