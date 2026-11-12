"""Khởi tạo Optimizer và Learning Rate Scheduler cho quá trình huấn luyện."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler, StepLR


def build_optimizer(model: nn.Module, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    """Tạo bộ tối ưu hóa dựa trên cấu hình."""
    opt_name = str(cfg.get("optimizer", "adamw")).lower()
    lr = float(cfg.get("learning_rate", 1e-4))
    weight_decay = float(cfg.get("weight_decay", 1e-2))

    if opt_name == "adamw":
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if opt_name == "adam":
        return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if opt_name == "sgd":
        momentum = float(cfg.get("momentum", 0.9))
        return SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    raise ValueError(f"Optimizer không được hỗ trợ: {opt_name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
    epochs: int,
) -> LRScheduler | None:
    """Tạo bộ điều chỉnh tốc độ học (scheduler)."""
    sched_name = str(cfg.get("scheduler", "cosine")).lower()
    min_lr = float(cfg.get("min_lr", 1e-6))

    if sched_name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    if sched_name == "step":
        step_size = int(cfg.get("step_size", 10))
        gamma = float(cfg.get("gamma", 0.1))
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    if sched_name in {"none", ""}:
        return None

    raise ValueError(f"Scheduler không được hỗ trợ: {sched_name}")
