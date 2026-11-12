"""Quản lý danh sách các mô hình được hỗ trợ trong dự án."""

from __future__ import annotations

from typing import Callable

import torch.nn as nn

from src.models.resnet50.model import build_resnet50

MODEL_REGISTRY: dict[str, Callable[..., nn.Module]] = {
    "resnet50": build_resnet50,
}


def register_model(name: str) -> Callable[[Callable[..., nn.Module]], Callable[..., nn.Module]]:
    """Bộ trang trí đăng ký hàm tạo mô hình mới vào registry."""

    def decorator(fn: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
        MODEL_REGISTRY[name] = fn
        return fn

    return decorator


def get_model_fn(name: str) -> Callable[..., nn.Module]:
    """Lấy hàm tạo mô hình theo tên đã đăng ký."""
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise KeyError(f"Không tìm thấy model '{name}'. Các model hiện có: {available}")
    return MODEL_REGISTRY[name]
