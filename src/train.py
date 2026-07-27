"""
Training loop for the TB detection model.

Run:
    python -m src.train --model densenet121
    python -m src.train --model hybrid
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from src.dataset import TBChestXrayDataset, get_train_transforms, get_eval_transforms
from src.model import get_model


def get_device():
    if config.DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def main(model_name: str):
    device = get_device()
    print(f"Training '{model_name}' on {device}")

    metadata_csv = config.PROCESSED_DIR / "metadata.csv"
    train_ds = TBChestXrayDataset(metadata_csv, "train", get_train_transforms())
    val_ds = TBChestXrayDataset(metadata_csv, "val", get_eval_transforms())

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)

    model = get_model(model_name, num_classes=config.NUM_CLASSES).to(device)

    class_weights = train_ds.class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    checkpoint_path = config.MODELS_DIR / f"{model_name}_best.pt"

    for epoch in range(1, config.NUM_EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"| val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(
                {"model_state_dict": model.state_dict(), "model_name": model_name, "epoch": epoch},
                checkpoint_path,
            )
            print(f"  -> saved new best checkpoint to {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    print(f"Training complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=config.MODEL_NAME, choices=[
        "densenet121", "efficientnet_b0", "mobilenet_v3", "hybrid"
    ])
    args = parser.parse_args()
    main(args.model)
