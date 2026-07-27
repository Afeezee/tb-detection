"""
Data preprocessing pipeline for TB chest X-ray detection.

Responsibilities:
  1. Walk the raw dataset folders (Montgomery, Shenzhen, TBX11K) and build a
     single metadata table (filepath, label, source_dataset).
  2. Apply standard preprocessing: grayscale -> CLAHE -> resize -> 3-channel
     stack (so pretrained ImageNet backbones can consume it).
  3. Save processed images + a metadata.csv used by dataset.py for
     train/val/test loading.

Run:
    python -m src.data_preprocessing
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import config


def build_montgomery_shenzhen_metadata() -> pd.DataFrame:
    """
    Montgomery and Shenzhen filenames encode the label in the filename itself:
    e.g. 'MCUCXR_0001_0.png' -> label 0 (normal), 'MCUCXR_0002_1.png' -> label 1 (TB).
    This matches the standard convention used in both public releases.
    """
    records = []
    for source, subdir in [("montgomery", "montgomery"), ("shenzhen", "shenzhen")]:
        img_dir = config.RAW_DIR / subdir / "CXR_png"
        if not img_dir.exists():
            print(f"[skip] {img_dir} not found — download instructions in README.md")
            continue
        for img_path in img_dir.glob("*.png"):
            stem = img_path.stem
            try:
                label = int(stem.split("_")[-1])
            except ValueError:
                continue
            records.append({"filepath": str(img_path), "label": label, "source": source})
    return pd.DataFrame(records)


def build_tbx11k_metadata() -> pd.DataFrame:
    """
    TBX11K ships with a directory-per-class layout under imgs/{health,sick,tb}.
    We map: health -> 0, sick (non-TB abnormal) -> 0 (treated as TB-negative
    for this binary task; revisit if you extend to multi-class), tb -> 1.
    """
    records = []
    img_root = config.RAW_DIR / "tbx11k" / "imgs"
    if not img_root.exists():
        print(f"[skip] {img_root} not found — download instructions in README.md")
        return pd.DataFrame(records)

    label_map = {"health": 0, "sick": 0, "tb": 1}
    for class_dir, label in label_map.items():
        class_path = img_root / class_dir
        if not class_path.exists():
            continue
        for img_path in class_path.glob("*.png"):
            records.append({"filepath": str(img_path), "label": label, "source": "tbx11k"})
    return pd.DataFrame(records)


def apply_clahe(gray_img: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(
        clipLimit=config.CLAHE_CLIP_LIMIT, tileGridSize=config.CLAHE_TILE_GRID_SIZE
    )
    return clahe.apply(gray_img)


def preprocess_single_image(src_path: str, dst_path: Path) -> bool:
    img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False

    if config.APPLY_CLAHE:
        img = apply_clahe(img)

    img = cv2.resize(img, (config.IMG_SIZE, config.IMG_SIZE), interpolation=cv2.INTER_AREA)

    # Stack to 3 channels so ImageNet-pretrained backbones (expecting RGB) work
    # without modifying the first conv layer.
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    return cv2.imwrite(str(dst_path), img_rgb)


def run_pipeline():
    print("Scanning raw datasets ...")
    df = pd.concat(
        [build_montgomery_shenzhen_metadata(), build_tbx11k_metadata()],
        ignore_index=True,
    )

    if df.empty:
        raise RuntimeError(
            "No raw images found. Download the datasets and place them under "
            "data/raw/ as described in README.md before running this pipeline."
        )

    print(f"Found {len(df)} raw images across {df['source'].nunique()} source datasets.")
    print(df.groupby(["source", "label"]).size())

    # Stratified split by label, keeping the TBX11K held-out for external
    # validation is handled separately in evaluate.py -- for the main
    # train/val/test split we stratify across the pooled Montgomery+Shenzhen
    # data and treat TBX11K as a distinct cross-dataset generalisation set.
    pooled = df[df["source"] != "tbx11k"].reset_index(drop=True)
    external = df[df["source"] == "tbx11k"].reset_index(drop=True)

    train_df, temp_df = train_test_split(
        pooled,
        test_size=(config.VAL_SPLIT + config.TEST_SPLIT),
        stratify=pooled["label"],
        random_state=config.RANDOM_SEED,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=config.TEST_SPLIT / (config.VAL_SPLIT + config.TEST_SPLIT),
        stratify=temp_df["label"],
        random_state=config.RANDOM_SEED,
    )

    splits = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
        "external_test": external,  # cross-dataset generalisation benchmark
    }

    all_rows = []
    for split_name, split_df in splits.items():
        print(f"\nProcessing split: {split_name} ({len(split_df)} images)")
        for _, row in tqdm(split_df.iterrows(), total=len(split_df)):
            src = row["filepath"]
            dst = config.PROCESSED_DIR / split_name / f"{Path(src).stem}.png"
            if preprocess_single_image(src, dst):
                all_rows.append(
                    {
                        "filepath": str(dst),
                        "label": row["label"],
                        "source": row["source"],
                        "split": split_name,
                    }
                )

    metadata_df = pd.DataFrame(all_rows)
    metadata_path = config.PROCESSED_DIR / "metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)
    print(f"\nDone. Metadata written to {metadata_path}")
    print(metadata_df.groupby(["split", "label"]).size())


if __name__ == "__main__":
    run_pipeline()
