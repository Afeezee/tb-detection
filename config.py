"""
Central configuration for the TB detection project.
Adjust paths and hyperparameters here rather than scattering magic numbers
through the codebase.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"          # place downloaded datasets here (see README)
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
GRADCAM_OUTPUT_DIR = DATA_DIR / "gradcam_outputs"

for d in [RAW_DIR, PROCESSED_DIR, MODELS_DIR, GRADCAM_OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset sources expected under data/raw/ (download instructions in README.md)
#   data/raw/montgomery/{CXR_png, ClinicalReadings}
#   data/raw/shenzhen/{CXR_png, ClinicalReadings}
#   data/raw/tbx11k/{imgs, annotations}
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------
IMG_SIZE = 224                 # DenseNet121 / EfficientNet default input
APPLY_CLAHE = True
APPLY_LUNG_SEGMENTATION = False   # flip on once the segmentation U-Net is trained
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
MODEL_NAME = "densenet121"     # baseline first; swap to "hybrid" once baseline works
NUM_CLASSES = 2                # 0 = Normal, 1 = TB-positive
BATCH_SIZE = 16
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 6
K_FOLDS = 5                    # set to 1 to skip cross-validation
DEVICE = "cuda"  # falls back to cpu automatically in train.py if unavailable

# ---------------------------------------------------------------------------
# Neon Postgres persistence
# Set NEON_DATABASE_URL in a .env file, e.g.:
#   NEON_DATABASE_URL=postgresql://user:password@ep-xxxx.neon.tech/tbdetect?sslmode=require
# Never commit the real .env file or the connection string to version control.
# ---------------------------------------------------------------------------
NEON_DATABASE_URL = os.getenv("NEON_DATABASE_URL", "")
