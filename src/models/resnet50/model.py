"""Định nghĩa model ResNet-50 cho bài toán phân loại đa nhãn."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


def build_resnet50(
    num_classes: int = 20,
    pretrained: bool = True,
    weights: str | None = "ResNet50_Weights.IMAGENET1K_V2",
) -> nn.Module:
    """Khởi tạo mô hình ResNet-50 từ torchvision với đầu ra theo số lớp tùy chọn."""
    if pretrained:
        weights_enum = (
            ResNet50_Weights.IMAGENET1K_V2
            if weights == "ResNet50_Weights.IMAGENET1K_V2"
            else ResNet50_Weights.DEFAULT
        )
    else:
        weights_enum = None

    model = resnet50(weights=weights_enum)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
