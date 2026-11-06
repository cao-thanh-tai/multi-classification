"""Transform dùng chung cho dataset PASCAL VOC."""

from __future__ import annotations

from typing import Callable

from torchvision import transforms


def build_transforms(
    image_size: int,
    train: bool,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> Callable:
    """Tạo transform train hoặc deterministic transform cho val/test."""
    if image_size <= 0:
        raise ValueError("image_size phải lớn hơn 0")
    operations: list[Callable] = [transforms.Resize((image_size, image_size))]
    if train:
        operations.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.05,
                ),
            ]
        )
    operations.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    return transforms.Compose(operations)