import Link from "next/link";

export default function Home() {
  return (
    <section className="grid gap-6 md:grid-cols-2">
      <div className="rounded-xl border border-clinical-border bg-clinical-surface p-8">
        <h1 className="text-3xl font-semibold text-clinical-ink">
          Screening chest X-rays for pulmonary tuberculosis
        </h1>
        <p className="mt-4 text-clinical-muted">
          Two deep-learning models trained on a multi-source pooled dataset of
          approximately 12,600 chest radiographs. A DenseNet121 baseline
          establishes the reference point; a Hybrid CNN+ViT model fuses local
          CNN features with global attention features for the novelty
          contribution.
        </p>
        <div className="mt-6 flex gap-3">
          <Link
            href="/predict"
            className="rounded-md bg-clinical-accent px-4 py-2 text-sm font-medium text-white hover:opacity-90"
          >
            Analyse an X-ray
          </Link>
          <Link
            href="/benchmark"
            className="rounded-md border border-clinical-border px-4 py-2 text-sm font-medium text-clinical-ink hover:bg-clinical-bg"
          >
            View benchmarks
          </Link>
        </div>
      </div>

      <div className="rounded-xl border border-clinical-border bg-clinical-surface p-8">
        <h2 className="text-lg font-semibold text-clinical-ink">Headline results</h2>
        <p className="mt-1 text-sm text-clinical-muted">Multi-source pooled test set.</p>
        <ul className="mt-4 space-y-3 text-sm">
          <li className="flex items-baseline justify-between border-b border-clinical-border pb-2">
            <span className="text-clinical-ink">DenseNet121</span>
            <span className="text-clinical-muted">AUC 0.9987 · Sens 0.9911 · Spec 0.9842</span>
          </li>
          <li className="flex items-baseline justify-between">
            <span className="text-clinical-ink">Hybrid CNN+ViT</span>
            <span className="text-clinical-muted">AUC 0.9980 · Sens 0.9911 · Spec 0.9812</span>
          </li>
        </ul>
        <p className="mt-4 text-xs text-clinical-muted">
          Both models were validated for generalisation on an external TBX11K split
          after evidence of shortcut learning in the single-source baseline.
        </p>
      </div>
    </section>
  );
}
