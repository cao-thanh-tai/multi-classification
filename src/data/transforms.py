"""Xây dựng các phép biến đổi ảnh (transforms) cho tập huấn luyện và đánh giá."""

from __future__ import annotations

from typing import Sequence

from torchvision import transforms


def build_transforms(
    image_size: int = 224,
    is_train: bool = True,
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std: Sequence[float] = (0.229, 0.224, 0.225),
) -> transforms.Compose:
    """Tạo chuỗi biến đổi ảnh cho train hoặc val/test."""
    if is_train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
