"""Hardcoded benchmark table for the /metrics endpoint.

Numbers reported by the trained checkpoints in models/. Update this file when
retraining rather than reading a JSON blob so the deployed service does not
depend on a mutable file being present.
"""
from datetime import datetime, timezone

from schemas import BenchmarkResponse, ModelBenchmark


def get_benchmarks() -> BenchmarkResponse:
    rows = [
        ModelBenchmark(
            model="DenseNet121",
            training_regime="single-source",
            test_set="internal",
            sensitivity=0.9999,
            specificity=0.9999,
            f1=0.9999,
            auc_roc=0.9999,
            notes="Single-source baseline — near-perfect internal metrics that did not generalise.",
        ),
        ModelBenchmark(
            model="DenseNet121",
            training_regime="single-source",
            test_set="external_tbx11k",
            sensitivity=0.0,
            specificity=0.0,
            f1=0.0,
            auc_roc=0.5581,
            notes="Collapsed on external TBX11K — evidence of shortcut learning.",
        ),
        ModelBenchmark(
            model="DenseNet121",
            training_regime="multi-source",
            test_set="internal",
            sensitivity=0.9911,
            specificity=0.9842,
            f1=0.9407,
            auc_roc=0.9987,
            notes="Baseline, multi-source pooled training set (Kaggle tawsifurrahman + TBX11K, ~12.6k images).",
        ),
        ModelBenchmark(
            model="Hybrid CNN+ViT",
            training_regime="multi-source",
            test_set="internal",
            sensitivity=0.9911,
            specificity=0.9812,
            f1=0.9308,
            auc_roc=0.9980,
            notes="Novelty model — DenseNet121 features fused with ViT-B/16 features.",
        ),
    ]
    return BenchmarkResponse(rows=rows, generated_at=datetime.now(timezone.utc))
