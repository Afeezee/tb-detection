"""Inference service — wraps src/model.py, src/gradcam.py, and the CLAHE
preprocessing from src/data_preprocessing.py.

The wrapping is deliberately thin: we do not reimplement anything, we import
the existing functions and add I/O boilerplate that FastAPI needs (BytesIO,
base64 output, model caching).
"""
from __future__ import annotations

import base64
import io
import sys
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

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
        return [m for m in SUPPORTED_MODELS if self._checkpoint_path(m).exists()]

    def get(self, name: str) -> torch.nn.Module:
        if name not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model '{name}'. Choose from {SUPPORTED_MODELS}.")
        with self._lock:
            if name in self._cache:
                return self._cache[name]
            ckpt_path = self._checkpoint_path(name)
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"Checkpoint missing: {ckpt_path}. Place the trained .pt file there before starting the API."
                )
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
