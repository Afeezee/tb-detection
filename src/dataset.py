"""
PyTorch Dataset for the TB chest X-ray classifier.
Reads data/processed/metadata.csv produced by data_preprocessing.py.
"""
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset

import config


def get_train_transforms():
    return A.Compose(
        [
            A.Rotate(limit=10, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.ElasticTransform(alpha=1, sigma=15, p=0.2),
            # Horizontal flip deliberately omitted by default: chest X-rays are
            # not left-right symmetric in a way that's clinically meaningless
            # (cardiac silhouette, aortic arch). Enable only if you've checked
            # your label convention doesn't encode laterality info.
            A.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_eval_transforms():
    return A.Compose(
        [
            A.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
            ToTensorV2(),
        ]
    )


class TBChestXrayDataset(Dataset):
    def __init__(self, metadata_csv: Path, split: str, transform=None):
        df = pd.read_csv(metadata_csv)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.transform = transform or get_eval_transforms()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(row["filepath"], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        augmented = self.transform(image=img)
        img_tensor = augmented["image"]
        label = torch.tensor(row["label"], dtype=torch.long)
        return img_tensor, label

    def class_weights(self):
        """Inverse-frequency weights for the loss function (handles class imbalance)."""
        counts = self.df["label"].value_counts().sort_index()
        weights = 1.0 / counts
        weights = weights / weights.sum() * len(counts)
        return torch.tensor(weights.values, dtype=torch.float32)
