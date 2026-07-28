"""Inference service — wraps src/model.py, src/gradcam.py, and the CLAHE
preprocessing from src/data_preprocessing.py.

The wrapping is deliberately thin: we do not reimplement anything, we import
the existing functions and add I/O boilerplate that FastAPI needs (BytesIO,
base64 output, model caching).
"""
from __future__ import annotations

import base64
import io
import logging
import os
import sys
import threading
import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

logger = logging.getLogger("tb_backend.inference")

# Make project root importable so `import config` and `from src.<mod>` both
# resolve — the existing modules assume this layout.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config  # noqa: E402
from src.model import get_model  # noqa: E402
from src.gradcam import generate_gradcam_overlay  # noqa: E402
from src.dataset import get_eval_transforms  # noqa: E402
from src.data_preprocessing import apply_clahe  # noqa: E402


CLASS_NAMES = {0: "Normal", 1: "TB-positive"}
SUPPORTED_MODELS = ("densenet121", "hybrid")

# LFS pointer files are ~130 bytes and start with this magic string.
_LFS_POINTER_MAGIC = b"version https://git-lfs.github.com/spec/"
_LFS_POINTER_MAX_BYTES = 4096


def _is_lfs_pointer(path: Path) -> bool:
    """A checkpoint file is a Git LFS pointer stub if it's tiny and starts with
    the LFS spec header. Real .pt files are megabytes and start with a torch
    pickle magic byte."""
    try:
        if path.stat().st_size > _LFS_POINTER_MAX_BYTES:
            return False
        with path.open("rb") as f:
            head = f.read(len(_LFS_POINTER_MAGIC))
        return head == _LFS_POINTER_MAGIC
    except OSError:
        return False


def _download_checkpoint(url: str, dest: Path) -> None:
    """Stream a checkpoint from a public URL to disk. Used to recover from LFS
    pointer stubs when Railway's build did not resolve LFS content."""
    tmp = dest.with_suffix(dest.suffix + ".partial")
    logger.info("Downloading checkpoint from %s to %s", url, dest)
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as out:
        # 4 MiB chunks — small enough to stream, large enough to be efficient.
        while True:
            chunk = resp.read(4 * 1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    tmp.replace(dest)
    logger.info("Downloaded %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelRegistry:
    """Lazily loads and caches checkpointed models. Thread-safe for uvicorn workers."""

    def __init__(self, models_dir: Path, device: torch.device):
        self.models_dir = models_dir
        self.device = device
        self._cache: dict[str, torch.nn.Module] = {}
        self._lock = threading.Lock()

    def _checkpoint_path(self, name: str) -> Path:
        return self.models_dir / f"{name}_best.pt"

    def available(self) -> list[str]:
        """A checkpoint is "available" if it is on disk AND is a real .pt file.
        LFS pointer stubs are treated as missing so we don't attempt torch.load
        on them; the URL-based recovery in get() handles the download."""
        return [
            m for m in SUPPORTED_MODELS
            if self._checkpoint_path(m).exists() and not _is_lfs_pointer(self._checkpoint_path(m))
        ]

    def get(self, name: str) -> torch.nn.Module:
        if name not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model '{name}'. Choose from {SUPPORTED_MODELS}.")
        with self._lock:
            if name in self._cache:
                return self._cache[name]

            ckpt_path = self._checkpoint_path(name)

            # Recover from Railway's non-LFS clone: if the file is missing or is
            # a Git LFS pointer stub, download the real weights from a public URL
            # supplied via MODEL_<NAME>_URL (e.g. a GitHub Release asset).
            if not ckpt_path.exists() or _is_lfs_pointer(ckpt_path):
                url = os.getenv(f"MODEL_{name.upper()}_URL", "").strip()
                if not url:
                    raise FileNotFoundError(
                        f"Checkpoint for '{name}' is missing or is a Git LFS pointer "
                        f"and no MODEL_{name.upper()}_URL is set. Upload the .pt file "
                        f"as a GitHub Release asset and set the env var to its "
                        f"browser_download_url."
                    )
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                _download_checkpoint(url, ckpt_path)

            model = get_model(name, num_classes=config.NUM_CLASSES, pretrained=False)
            # weights_only=False because our checkpoints wrap model_state_dict in
            # a training-time dict (optimizer state, epoch, metrics). PyTorch 2.6+
            # defaults to True which rejects non-tensor pickle entries. Safe here
            # because these files are produced by our own training runs.
            state = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            state_dict = state.get("model_state_dict", state) if isinstance(state, dict) else state
            model.load_state_dict(state_dict)
            model.to(self.device).eval()
            self._cache[name] = model
            return model


def _preprocess_bytes(image_bytes: bytes) -> tuple[torch.Tensor, np.ndarray]:
    """Mirror the CLAHE + resize pipeline used at training time.

    Returns (normalised tensor of shape (1,3,H,W), unnormalised RGB uint8 image
    of shape (H,W,3)). The second value is what Grad-CAM overlays onto.
    """
    file_arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(file_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode image. Upload a PNG or JPEG chest X-ray.")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if config.APPLY_CLAHE:
        gray = apply_clahe(gray)
    resized_gray = cv2.resize(gray, (config.IMG_SIZE, config.IMG_SIZE), interpolation=cv2.INTER_AREA)
    resized_rgb = cv2.cvtColor(resized_gray, cv2.COLOR_GRAY2RGB)

    transform = get_eval_transforms()
    tensor = transform(image=resized_rgb)["image"].unsqueeze(0)
    return tensor, resized_rgb


def _encode_png_base64(rgb_uint8: np.ndarray) -> str:
    bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("Failed to PNG-encode the Grad-CAM overlay.")
    return base64.b64encode(buf.tobytes()).decode("ascii")


class InferenceService:
    def __init__(self, models_dir: Optional[Path] = None):
        self.device = get_device()
        self.registry = ModelRegistry(
            models_dir or Path(config.MODELS_DIR), self.device
        )

    def predict(self, image_bytes: bytes, model_name: str):
        model = self.registry.get(model_name)
        input_tensor, display_rgb = _preprocess_bytes(image_bytes)

        with torch.no_grad():
            logits = model(input_tensor.to(self.device))
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])

        overlay = generate_gradcam_overlay(
            model=model,
            model_name=model_name,
            input_tensor=input_tensor,
            original_rgb_uint8=display_rgb,
            device=self.device,
            target_class=1,
        )
        gradcam_b64 = _encode_png_base64(overlay)

        return {
            "label_index": pred_class,
            "label": CLASS_NAMES[pred_class],
            "confidence": confidence,
            "probabilities": {CLASS_NAMES[i]: float(p) for i, p in enumerate(probs)},
            "gradcam_base64": gradcam_b64,
        }
