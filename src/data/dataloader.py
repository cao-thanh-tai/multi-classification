"""Khởi tạo DataLoader cho tập train và val từ cấu hình dữ liệu."""

from __future__ import annotations

from typing import Any

from torch.utils.data import DataLoader, Subset

from src.data.dataset import PascalVOCMultiLabelDataset
from src.data.transforms import build_transforms


def build_dataloaders(
    data_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    limit: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Tạo DataLoader cho tập huấn luyện và kiểm định."""
    image_size = int(data_cfg.get("image_size", 224))
    mean = tuple(data_cfg.get("normalization", {}).get("mean", [0.485, 0.456, 0.406]))
    std = tuple(data_cfg.get("normalization", {}).get("std", [0.229, 0.224, 0.225]))

    train_transform = build_transforms(image_size=image_size, is_train=True, mean=mean, std=std)
    val_transform = build_transforms(image_size=image_size, is_train=False, mean=mean, std=std)

    data_roots = data_cfg["data_roots"]
    splits_root = data_cfg["splits_root"]
    classes_path = data_cfg["classes_path"]

    train_dataset = PascalVOCMultiLabelDataset(
        split="train",
        data_roots=data_roots,
        splits_root=splits_root,
        classes_path=classes_path,
        transform=train_transform,
    )
    val_dataset = PascalVOCMultiLabelDataset(
        split="val",
        data_roots=data_roots,
        splits_root=splits_root,
        classes_path=classes_path,
        transform=val_transform,
    )

    if limit is not None and limit > 0:
        train_dataset = Subset(train_dataset, range(min(limit, len(train_dataset))))
        val_dataset = Subset(val_dataset, range(min(limit, len(val_dataset))))

    batch_size = int(train_cfg.get("batch_size", 32))
    num_workers = int(train_cfg.get("num_workers", 2))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
