"""Dataset multi-label cho PASCAL VOC 2007."""

from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Callable, Mapping

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


ImageTransform = Callable[[Image.Image], Tensor]


class PascalVOCMultiLabelDataset(Dataset[tuple[Tensor, Tensor]]):
    """Đọc ảnh VOC và mã hóa nhãn object thành vector multi-hot."""

    def __init__(
        self,
        split: str,
        data_roots: Mapping[str, str | Path],
        splits_root: str | Path,
        classes_path: str | Path,
        transform: ImageTransform | None = None,
    ) -> None:
        """Khởi tạo dataset từ split, class mapping và dữ liệu VOC raw."""
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Split không hợp lệ: {split!r}; cần train, val hoặc test")
        if split not in data_roots:
            raise ValueError(f"Thiếu data root cho split {split!r}")

        self.split = split
        self.voc_root = Path(data_roots[split])
        self.splits_root = Path(splits_root)
        self.classes_path = Path(classes_path)
        self.transform = transform
        self.classes = self._read_classes()
        self.class_to_index = {name: index for index, name in enumerate(self.classes)}
        self.image_ids = self._read_split_ids()
        self.annotation_dir = self.voc_root / "Annotations"
        self.image_dir = self.voc_root / "JPEGImages"
        self._validate_image_ids()

    def __len__(self) -> int:
        """Trả về số ảnh trong split."""
        return len(self.image_ids)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """Đọc một ảnh và vector nhãn multi-hot tương ứng."""
        image_id = self.image_ids[index]
        image_path = self.image_dir / f"{image_id}.jpg"
        annotation_path = self.annotation_dir / f"{image_id}.xml"
        image = Image.open(image_path).convert("RGB")
        labels = self._read_labels(annotation_path)

        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            image_tensor = self._to_tensor(image)

        target = torch.zeros(len(self.classes), dtype=torch.float32)
        for label in labels:
            target[self.class_to_index[label]] = 1.0
        return image_tensor, target

    def _read_classes(self) -> list[str]:
        if not self.classes_path.is_file():
            raise FileNotFoundError(
                f"Không tìm thấy class mapping: {self.classes_path}. "
                "Hãy chạy cell EDA tạo classes.json trước."
            )
        with self.classes_path.open(encoding="utf-8") as file:
            payload = json.load(file)
        classes = payload.get("classes")
        if not isinstance(classes, list) or not classes or not all(
            isinstance(name, str) and name for name in classes
        ):
            raise ValueError(f"classes.json không có danh sách class hợp lệ: {self.classes_path}")
        if len(set(classes)) != len(classes):
            raise ValueError(f"classes.json có class trùng: {self.classes_path}")
        return classes

    def _read_split_ids(self) -> list[str]:
        split_path = self.splits_root / f"{self.split}.txt"
        if not split_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy split: {split_path}")
        image_ids = [
            line.strip()
            for line in split_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not image_ids:
            raise ValueError(f"Split rỗng: {split_path}")
        if len(set(image_ids)) != len(image_ids):
            raise ValueError(f"Split có image ID trùng: {split_path}")
        return image_ids

    def _validate_image_ids(self) -> None:
        for image_id in self.image_ids:
            image_path = self.image_dir / f"{image_id}.jpg"
            annotation_path = self.annotation_dir / f"{image_id}.xml"
            if not image_path.is_file():
                raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")
            if not annotation_path.is_file():
                raise FileNotFoundError(f"Không tìm thấy annotation: {annotation_path}")

    def _read_labels(self, annotation_path: Path) -> set[str]:
        root = ET.parse(annotation_path).getroot()
        labels = {
            node.findtext("name", "").strip()
            for node in root.findall("object")
            if node.findtext("name", "").strip()
        }
        unknown = labels - self.class_to_index.keys()
        if unknown:
            raise ValueError(
                f"Class không có trong mapping tại {annotation_path}: {sorted(unknown)}"
            )
        return labels

    @staticmethod
    def _to_tensor(image: Image.Image) -> Tensor:
        """Chuyển ảnh PIL RGB thành tensor float CHW trong khoảng [0, 1]."""
        values = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        values = values.reshape(image.height, image.width, len(image.getbands()))
        return values.permute(2, 0, 1).float() / 255.0