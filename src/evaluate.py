"""
Evaluation script: computes the full metric suite expected for a TB
detection thesis, on both the held-out internal test set and the
external (cross-dataset) test set for a generalisation check.

Run:
    python -m src.evaluate --model densenet121 --split test
    python -m src.evaluate --model densenet121 --split external_test
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

import config
from src.dataset import TBChestXrayDataset, get_eval_transforms
from src.model import get_model
from src.train import get_device


def sensitivity_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # recall on TB-positive
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return sensitivity, specificity


def evaluate(model_name: str, split: str):
    device = get_device()
    metadata_csv = config.PROCESSED_DIR / "metadata.csv"
    test_ds = TBChestXrayDataset(metadata_csv, split, get_eval_transforms())
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    checkpoint = torch.load(config.MODELS_DIR / f"{model_name}_best.pt", map_location=device)
    model = get_model(model_name, num_classes=config.NUM_CLASSES)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # P(TB-positive)
            preds = outputs.argmax(dim=1)

            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    sensitivity, specificity = sensitivity_specificity(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    print(f"\n=== Evaluation: model={model_name} split={split} ===")
    print(f"Sensitivity (Recall, TB+): {sensitivity:.4f}")
    print(f"Specificity:               {specificity:.4f}")
    print(f"Precision:                 {precision:.4f}")
    print(f"F1-score:                  {f1:.4f}")
    print(f"AUC-ROC:                   {roc_auc:.4f}")
    print(f"Confusion matrix:\n{confusion_matrix(all_labels, all_preds)}")

    # Save ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title(f"ROC Curve — {model_name} ({split})")
    plt.legend()
    out_path = config.MODELS_DIR / f"{model_name}_{split}_roc.png"
    plt.savefig(out_path, bbox_inches="tight")
    print(f"ROC curve saved to {out_path}")

    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "auc_roc": roc_auc,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=config.MODEL_NAME)
    parser.add_argument("--split", default="test", choices=["test", "external_test"])
    args = parser.parse_args()
    evaluate(args.model, args.split)
