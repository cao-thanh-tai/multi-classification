"""Thực hiện đánh giá mô hình trên tập validation/test."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_multilabel_metrics


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Đánh giá mô hình trên DataLoader và trả về các chỉ số độ đo kèm thời gian suy luận."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_targets: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []

    # Đo thời gian suy luận thuần túy
    inference_time_sec = 0.0

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            start_t = time.perf_counter()
            outputs = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            inference_time_sec += time.perf_counter() - start_t

            loss = criterion(outputs, targets)
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            probs = torch.sigmoid(outputs)
            all_targets.append(targets.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    if total_samples == 0:
        return {"val_loss": 0.0, "infer_ms": 0.0}

    targets_arr = np.concatenate(all_targets, axis=0)
    probs_arr = np.concatenate(all_probs, axis=0)

    metrics = compute_multilabel_metrics(targets_arr, probs_arr, threshold=threshold)
    metrics["val_loss"] = round(total_loss / total_samples, 4)
    metrics["infer_ms"] = round((inference_time_sec / total_samples) * 1000.0, 2)

    return metrics
