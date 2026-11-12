"""Đọc cấu hình và khởi tạo mô hình tương ứng."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch.nn as nn

from src.models.registry import get_model_fn
from src.utils.config import load_yaml


def build_model(model_name: str, overrides: dict[str, Any] | None = None) -> nn.Module:
    """Khởi tạo mô hình dựa trên tên và tệp cấu hình của mô hình đó."""
    config_path = Path(__file__).parent / model_name / "config.yaml"
    cfg: dict[str, Any] = {}
    if config_path.exists():
        cfg = load_yaml(config_path)

    if overrides:
        cfg.update(overrides)

    model_fn = get_model_fn(model_name)
    num_classes = int(cfg.get("num_classes", 20))
    pretrained = bool(cfg.get("pretrained", True))
    weights = cfg.get("weights")

    return model_fn(num_classes=num_classes, pretrained=pretrained, weights=weights)
