"""
Model architectures for TB detection.

Start with `densenet121` to get a working, defensible baseline (matches the
architecture used in most published TB-CXR papers, so your benchmarking
table is directly comparable). Once that trains and evaluates cleanly, wire
up `hybrid_cnn_vit` for the novelty contribution.
"""
import torch
import torch.nn as nn
import torchvision.models as tvm


def build_densenet121(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    weights = tvm.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    model = tvm.densenet121(weights=weights)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model


def build_efficientnet_b0(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    weights = tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = tvm.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def build_mobilenet_v3(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """Lightweight variant for the edge-deployment / low-resource-clinic story."""
    weights = tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = tvm.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


class HybridCNNViT(nn.Module):
    """
    Feature-fusion hybrid: DenseNet121 (local texture features, good for TB's
    fine-grained lung opacities) concatenated with a ViT-Small's global
    attention features (good for overall lung field context), then a small
    classifier head on the fused representation.

    This is the novelty module for the thesis once the DenseNet121 baseline
    is validated -- train and evaluate this separately, and report the
    ablation (CNN-only vs ViT-only vs fused) as a results table.
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        densenet_weights = tvm.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        vit_weights = tvm.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None

        densenet = tvm.densenet121(weights=densenet_weights)
        self.cnn_features = densenet.features
        self.cnn_pool = nn.AdaptiveAvgPool2d(1)
        cnn_out_dim = densenet.classifier.in_features  # 1024

        vit = tvm.vit_b_16(weights=vit_weights)
        # Strip the classification head, keep the encoder for feature extraction
        self.vit = vit
        self.vit.heads = nn.Identity()
        vit_out_dim = 768

        self.classifier = nn.Sequential(
            nn.Linear(cnn_out_dim + vit_out_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # CNN branch
        cnn_feat = self.cnn_features(x)
        cnn_feat = torch.relu(cnn_feat)
        cnn_feat = self.cnn_pool(cnn_feat).flatten(1)

        # ViT branch expects 224x224 input -- ensure your DataLoader resizes
        # to a common size that satisfies both branches.
        vit_feat = self.vit(x)

        fused = torch.cat([cnn_feat, vit_feat], dim=1)
        return self.classifier(fused)


MODEL_REGISTRY = {
    "densenet121": build_densenet121,
    "efficientnet_b0": build_efficientnet_b0,
    "mobilenet_v3": build_mobilenet_v3,
    "hybrid": lambda num_classes, pretrained: HybridCNNViT(num_classes, pretrained),
}


def get_model(name: str, num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Options: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](num_classes=num_classes, pretrained=pretrained)
