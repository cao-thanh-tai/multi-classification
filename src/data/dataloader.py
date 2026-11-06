"""Factory tạo DataLoader cho các split VOC."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from torch.utils.data import DataLoader

from src.data.dataset import PascalVOCMultiLabelDataset
from src.data.transforms import build_transforms


def build_dataloader(
    split: str,
    data_roots: Mapping[str, str | Path],
    splits_root: str | Path,
    classes_path: str | Path,
    image_size: int,
    batch_size: int,
    num_workers: int = 0,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> DataLoader:
    """Tạo DataLoader với shuffle chỉ bật cho train."""
    if batch_size <= 0:
        raise ValueError("batch_size phải lớn hơn 0")
    if num_workers < 0:
        raise ValueError("num_workers không được âm")
    dataset = PascalVOCMultiLabelDataset(
        split=split,
        data_roots=data_roots,
        splits_root=splits_root,
        classes_path=classes_path,
        transform=build_transforms(
            image_size=image_size,
            train=split == "train",
            mean=mean,
            std=std,
        ),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train",
        num_workers=num_workers,
        pin_memory=True,
    )