"""Kiểm thử tính đúng đắn của model builder và metrics."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.evaluation.metrics import compute_multilabel_metrics
from src.models.builder import build_model


def test_build_resnet50_output_shape() -> None:
    """Kiểm tra mô hình ResNet-50 cho ra tensor shape chính xác (B, num_classes)."""
    model = build_model("resnet50", overrides={"pretrained": False, "num_classes": 20})
    model.eval()

    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (2, 20)
    assert not torch.isnan(output).any()


def test_compute_multilabel_metrics_perfect_prediction() -> None:
    """Kiểm tra hàm tính metrics khi dự đoán khớp hoàn toàn 100%."""
    targets = np.array([[1, 0, 1], [0, 1, 0]])
    probs = np.array([[0.9, 0.1, 0.8], [0.2, 0.7, 0.3]])

    results = compute_multilabel_metrics(targets, probs, threshold=0.5)

    assert results["macro_f1"] == 1.0
    assert results["micro_f1"] == 1.0
    assert results["subset_accuracy"] == 1.0
