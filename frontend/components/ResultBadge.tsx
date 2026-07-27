import clsx from "clsx";

export default function ResultBadge({
  label,
  confidence,
}: {
  label: "Normal" | "TB-positive";
  confidence: number;
}) {
  const positive = label === "TB-positive";
  return (
    <div
      className={clsx(
        "rounded-lg border p-4",
        positive
          ? "border-clinical-positive/40 bg-clinical-positive/5 text-clinical-positive"
          : "border-clinical-negative/40 bg-clinical-negative/5 text-clinical-negative"
      )}
    >
      <div className="text-xs uppercase tracking-wide">Prediction</div>
      <div className="mt-1 text-2xl font-semibold">{label}</div>
      <div className="mt-1 text-sm">Confidence {(confidence * 100).toFixed(1)}%</div>
    </div>
  );
}
