// Client-side PDF and DOCX generation for the TB screening report.
// No server round-trip, no popups (the old print-window pipeline was
// unreliable because blob URLs and document.write don't survive popup
// blockers). Downloads are triggered directly via file-saver.

import { jsPDF } from "jspdf";
import {
  AlignmentType,
  Document,
  HeadingLevel,
  ImageRun,
  Packer,
  Paragraph,
  Table,
  TableCell,
  TableRow,
  TextRun,
  WidthType,
} from "docx";
import { saveAs } from "file-saver";

export interface ReportData {
  patientRef: string;
  timestamp: Date;
  label: "Normal" | "TB-positive";
  confidence: number;
  modelUsed: string;
  probabilities: Record<string, number>;
  clinicianNotes?: string;
  originalImageBase64: string; // raw base64, no data: prefix
  gradcamBase64: string; // raw base64, no data: prefix
}

// --------------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------------

export function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(reader.error);
    reader.onload = () => {
      const result = reader.result as string;
      // Strip the "data:image/...;base64," prefix so callers get raw base64.
      const comma = result.indexOf(",");
      resolve(comma >= 0 ? result.slice(comma + 1) : result);
    };
    reader.readAsDataURL(file);
  });
}

function base64ToUint8Array(b64: string): Uint8Array {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

function reportFilename(data: ReportData, ext: "pdf" | "docx"): string {
  const ref = data.patientRef?.trim() || "report";
  const stamp = data.timestamp
    .toISOString()
    .replace(/[:.]/g, "-")
    .slice(0, 19);
  const safe = ref.replace(/[^a-z0-9-_]/gi, "_");
  return `tb-screening-${safe}-${stamp}.${ext}`;
}

// --------------------------------------------------------------------------
// PDF
// --------------------------------------------------------------------------

export function exportPdf(data: ReportData): void {
  const doc = new jsPDF({ unit: "pt", format: "a4" });
  const pageWidth = doc.internal.pageSize.getWidth();
  const margin = 40;
  let y = margin;

  const positive = data.label === "TB-positive";
  const accent: [number, number, number] = positive ? [193, 18, 31] : [17, 138, 60];
  const ink: [number, number, number] = [15, 30, 46];
  const muted: [number, number, number] = [91, 107, 124];

  // Header
  doc.setFont("helvetica", "bold");
  doc.setFontSize(16);
  doc.setTextColor(...ink);
  doc.text("Tuberculosis screening report", margin, y);
  y += 22;

  doc.setFont("helvetica", "normal");
  doc.setFontSize(9);
  doc.setTextColor(...muted);
  doc.text(
    `Generated ${data.timestamp.toLocaleString("en-GB")}  ·  Reference ${
      data.patientRef?.trim() || "—"
    }`,
    margin,
    y
  );
  y += 8;
  doc.setDrawColor(229, 233, 239);
  doc.line(margin, y, pageWidth - margin, y);
  y += 20;

  // Prediction banner
  doc.setFont("helvetica", "bold");
  doc.setFontSize(11);
  doc.setTextColor(...muted);
  doc.text("PREDICTION", margin, y);
  y += 16;

  doc.setFontSize(18);
  doc.setTextColor(...accent);
  doc.text(data.label, margin, y);

  doc.setFontSize(11);
  doc.setTextColor(...ink);
  doc.text(
    `Confidence ${(data.confidence * 100).toFixed(1)}%`,
    margin + 200,
    y
  );
  y += 22;

  doc.setFont("helvetica", "normal");
  doc.setFontSize(10);
  doc.setTextColor(...muted);
  doc.text(`Model: ${data.modelUsed}`, margin, y);
  y += 24;

  // Images side by side
  const imgSlot = (pageWidth - margin * 2 - 20) / 2;
  const imgH = imgSlot;
  drawImage(doc, data.originalImageBase64, margin, y, imgSlot, imgH);
  drawImage(
    doc,
    data.gradcamBase64,
    margin + imgSlot + 20,
    y,
    imgSlot,
    imgH
  );

  doc.setFontSize(9);
  doc.setTextColor(...muted);
  doc.text("Uploaded image", margin, y + imgH + 12);
  doc.text("Grad-CAM overlay", margin + imgSlot + 20, y + imgH + 12);
  y += imgH + 30;

  // Probability table
  doc.setFont("helvetica", "bold");
  doc.setFontSize(11);
  doc.setTextColor(...muted);
  doc.text("CLASS PROBABILITIES", margin, y);
  y += 14;

  doc.setFont("helvetica", "normal");
  doc.setFontSize(11);
  doc.setTextColor(...ink);
  for (const [cls, p] of Object.entries(data.probabilities)) {
    doc.text(`${cls}`, margin, y);
    doc.text(`${(p * 100).toFixed(1)}%`, margin + 200, y);
    y += 16;
  }
  y += 8;

  // Notes
  if (data.clinicianNotes && data.clinicianNotes.trim()) {
    doc.setFont("helvetica", "bold");
    doc.setFontSize(11);
    doc.setTextColor(...muted);
    doc.text("CLINICIAN NOTES", margin, y);
    y += 14;

    doc.setFont("helvetica", "normal");
    doc.setFontSize(10);
    doc.setTextColor(...ink);
    const lines = doc.splitTextToSize(
      data.clinicianNotes.trim(),
      pageWidth - margin * 2
    );
    doc.text(lines, margin, y);
    y += lines.length * 12 + 12;
  }

  // Disclaimer footer
  const footer =
    "Research prototype — not a licensed medical device. Results must be reviewed by a qualified clinician.";
  doc.setFontSize(8);
  doc.setTextColor(...muted);
  const footerLines = doc.splitTextToSize(footer, pageWidth - margin * 2);
  doc.text(footerLines, margin, doc.internal.pageSize.getHeight() - margin);

  doc.save(reportFilename(data, "pdf"));
}

function drawImage(
  doc: jsPDF,
  b64: string,
  x: number,
  y: number,
  w: number,
  h: number
) {
  try {
    // jsPDF auto-detects PNG/JPEG from the base64; use PNG since our Grad-CAM
    // is PNG-encoded and the uploaded file could be either.
    doc.addImage(`data:image/png;base64,${b64}`, "PNG", x, y, w, h, undefined, "FAST");
  } catch {
    try {
      doc.addImage(`data:image/jpeg;base64,${b64}`, "JPEG", x, y, w, h, undefined, "FAST");
    } catch {
      // Fallback: draw a placeholder rectangle so the layout does not collapse.
      doc.setDrawColor(229, 233, 239);
      doc.rect(x, y, w, h);
    }
  }
}

// --------------------------------------------------------------------------
// DOCX
// --------------------------------------------------------------------------

export async function exportDocx(data: ReportData): Promise<void> {
  const positive = data.label === "TB-positive";

  const originalPng = base64ToUint8Array(data.originalImageBase64);
  const gradcamPng = base64ToUint8Array(data.gradcamBase64);

  const probRows: TableRow[] = [
    new TableRow({
      children: [
        headerCell("Class"),
        headerCell("Probability"),
      ],
    }),
    ...Object.entries(data.probabilities).map(
      ([cls, p]) =>
        new TableRow({
          children: [cell(cls), cell(`${(p * 100).toFixed(1)}%`)],
        })
    ),
  ];

  const doc = new Document({
    creator: "TB Detection",
    title: "TB screening report",
    styles: {
      default: {
        document: {
          run: { font: "Calibri", size: 22 }, // 11pt
        },
      },
    },
    sections: [
      {
        properties: {},
        children: [
          new Paragraph({
            heading: HeadingLevel.HEADING_1,
            children: [
              new TextRun({ text: "Tuberculosis screening report", bold: true }),
            ],
          }),
          new Paragraph({
            children: [
              new TextRun({
                text: `Generated ${data.timestamp.toLocaleString("en-GB")}  ·  Reference ${
                  data.patientRef?.trim() || "—"
                }`,
                color: "5b6b7c",
                size: 18,
              }),
            ],
          }),
          new Paragraph({ text: "" }),

          new Paragraph({
            children: [
              new TextRun({ text: "Prediction: ", bold: true }),
              new TextRun({
                text: data.label,
                bold: true,
                color: positive ? "c1121f" : "118a3c",
              }),
              new TextRun({
                text: `  (confidence ${(data.confidence * 100).toFixed(1)}%)`,
              }),
            ],
          }),
          new Paragraph({
            children: [new TextRun({ text: `Model: ${data.modelUsed}` })],
          }),
          new Paragraph({ text: "" }),

          new Paragraph({
            alignment: AlignmentType.CENTER,
            children: [
              new ImageRun({
                data: originalPng,
                transformation: { width: 260, height: 260 },
              }),
              new TextRun({ text: "  " }),
              new ImageRun({
                data: gradcamPng,
                transformation: { width: 260, height: 260 },
              }),
            ],
          }),
          new Paragraph({
            alignment: AlignmentType.CENTER,
            children: [
              new TextRun({
                text: "Uploaded image                                       Grad-CAM overlay",
                color: "5b6b7c",
                size: 18,
              }),
            ],
          }),
          new Paragraph({ text: "" }),

          new Paragraph({
            heading: HeadingLevel.HEADING_2,
            children: [new TextRun({ text: "Class probabilities" })],
          }),
          new Table({
            width: { size: 100, type: WidthType.PERCENTAGE },
            rows: probRows,
          }),
          new Paragraph({ text: "" }),

          ...(data.clinicianNotes && data.clinicianNotes.trim()
            ? [
                new Paragraph({
                  heading: HeadingLevel.HEADING_2,
                  children: [new TextRun({ text: "Clinician notes" })],
                }),
                new Paragraph({
                  children: [
                    new TextRun({ text: data.clinicianNotes.trim() }),
                  ],
                }),
                new Paragraph({ text: "" }),
              ]
            : []),

          new Paragraph({
            children: [
              new TextRun({
                text: "Research prototype — not a licensed medical device. Results must be reviewed by a qualified clinician.",
                color: "5b6b7c",
                size: 16,
                italics: true,
              }),
            ],
          }),
        ],
      },
    ],
  });

  const blob = await Packer.toBlob(doc);
  saveAs(blob, reportFilename(data, "docx"));
}

function cell(text: string): TableCell {
  return new TableCell({
    children: [new Paragraph({ children: [new TextRun({ text })] })],
  });
}

function headerCell(text: string): TableCell {
  return new TableCell({
    children: [
      new Paragraph({
        children: [new TextRun({ text, bold: true, color: "5b6b7c" })],
      }),
    ],
  });
}
