"""
Grad-CAM explainability for the TB classifier.
Wraps the `grad-cam` library so the Streamlit app can request a heatmap
overlay for any prediction — this is the figure your thesis and the demo
interface both lean on heavily.
"""
from pathlib import Path

import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import config


def _get_target_layer(model, model_name: str):
    """Last convolutional layer per architecture — this is what Grad-CAM hooks into."""
    if model_name == "densenet121":
        return [model.features.denseblock4.denselayer16.conv2]
    if model_name == "efficientnet_b0":
        return [model.features[-1]]
    if model_name == "mobilenet_v3":
        return [model.features[-1]]
    if model_name == "hybrid":
        # Grad-CAM on the CNN branch only; the ViT branch needs attention
        # rollout instead, which is a separate (optional) enhancement.
        return [model.cnn_features.denseblock4.denselayer16.conv2]
    raise ValueError(f"No target layer mapping defined for '{model_name}'")


def generate_gradcam_overlay(
    model: torch.nn.Module,
    model_name: str,
    input_tensor: torch.Tensor,
    original_rgb_uint8: np.ndarray,
    device: torch.device,
    target_class: int = 1,
) -> np.ndarray:
    """
    input_tensor: normalised, preprocessed tensor, shape (1, 3, H, W)
    original_rgb_uint8: the *unnormalised* RGB image (0-255) to overlay onto,
                         resized to the same H, W as input_tensor
    Returns an RGB uint8 heatmap-overlaid image ready to display or save.
    """
    model.eval()
    target_layers = _get_target_layer(model, model_name)

    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(target_class)]

    grayscale_cam = cam(input_tensor=input_tensor.to(device), targets=targets)[0]

    rgb_float = original_rgb_uint8.astype(np.float32) / 255.0
    overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)
    return overlay


def save_overlay(overlay_rgb: np.ndarray, filename: str) -> Path:
    out_path = config.GRADCAM_OUTPUT_DIR / filename
    cv2.imwrite(str(out_path), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
    return out_path
