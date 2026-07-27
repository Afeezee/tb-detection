export default function AboutPage() {
  return (
    <section className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold text-clinical-ink">About this project</h1>
        <p className="text-sm text-clinical-muted">
          Final-year project by Adesanlu Martins (U/22/CS/0011) under the supervision of Miss
          Shadare. Chest X-ray screening for pulmonary tuberculosis using two deep-learning
          models.
        </p>
      </div>

      <article className="prose prose-slate max-w-none rounded-xl border border-clinical-border bg-clinical-surface p-6 text-clinical-ink">
        <h2 className="text-lg font-semibold">Architectures</h2>
        <p>
          The <strong>DenseNet121</strong> baseline is a densely-connected CNN pretrained on
          ImageNet with a two-class classification head. It matches the architecture used in
          the majority of published TB-CXR papers, so it serves as a directly comparable
          reference point.
        </p>
        <p>
          The <strong>Hybrid CNN+ViT</strong> model is the novelty contribution. Local texture
          features from a DenseNet121 backbone are concatenated with the class-token features
          from a Vision Transformer (ViT-B/16); a small MLP classifier is trained on top of the
          fused representation. The intuition is that TB signs range from fine-grained infiltrates
          (well-captured by CNNs) to whole-lung consolidation patterns (better captured by
          attention over the full field of view).
        </p>

        <h2 className="mt-6 text-lg font-semibold">Datasets</h2>
        <ul>
          <li>Kaggle TB Chest X-ray dataset (Tawsifur Rahman et al.) — approximately 7,000 images.</li>
          <li>TBX11K — approximately 11,200 images with distinct acquisition characteristics.</li>
          <li>Pooled after de-duplication: about 12,600 chest radiographs.</li>
        </ul>

        <h2 className="mt-6 text-lg font-semibold">Multi-source generalisation</h2>
        <p>
          An initial single-source DenseNet121 baseline reached an AUC of 0.9999 on its own
          test split but collapsed to an AUC of 0.5581 on external TBX11K — near-chance
          performance, and a textbook symptom of shortcut learning. Retraining on the pooled
          multi-source corpus recovered generalisation while preserving high sensitivity. This
          finding is the main motivation for the deployed models and is highlighted on the
          benchmark page.
        </p>

        <h2 className="mt-6 text-lg font-semibold">Explainability</h2>
        <p>
          Every prediction is accompanied by a Grad-CAM overlay computed against the last
          convolutional layer of DenseNet121 (or the CNN branch of the hybrid model). Visual
          inspection during validation confirmed that activations land on lung fields rather
          than on the mediastinum or image borders.
        </p>

        <h2 className="mt-6 text-lg font-semibold">Stack</h2>
        <ul>
          <li>Model training in PyTorch (torchvision) with albumentations for augmentation.</li>
          <li>Serving via FastAPI, wrapping the training-time modules without modification.</li>
          <li>Frontend in Next.js 14 (App Router) with Tailwind CSS.</li>
          <li>Persistence in Neon Postgres via psycopg2.</li>
          <li>Deployment as a two-service monorepo on Railway.</li>
        </ul>

        <p className="mt-6 text-xs text-clinical-muted">
          Research prototype only. Not a licensed medical device and not to be used for clinical
          decision-making without a qualified clinician in the loop.
        </p>
      </article>
    </section>
  );
}
