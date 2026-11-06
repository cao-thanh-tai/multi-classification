import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from src.data.dataset import PascalVOCMultiLabelDataset


def _create_fixture(root: Path) -> tuple[dict[str, Path], Path, Path]:
    voc_root = root / "raw" / "VOC2007"
    (voc_root / "Annotations").mkdir(parents=True)
    (voc_root / "JPEGImages").mkdir()
    splits_root = root / "splits"
    splits_root.mkdir()
    classes_path = root / "classes.json"

    (splits_root / "train.txt").write_text("000001\n", encoding="utf-8")
    classes_path.write_text(
        json.dumps({"classes": ["cat", "dog"]}),
        encoding="utf-8",
    )
    Image.new("RGB", (2, 3), color=(255, 0, 0)).save(voc_root / "JPEGImages" / "000001.jpg")
    (voc_root / "Annotations" / "000001.xml").write_text(
        """<annotation><object><name>dog</name></object>
        <object><name>dog</name></object><object><name>cat</name></object>
        </annotation>""",
        encoding="utf-8",
    )
    data_roots = {"train": voc_root, "val": voc_root, "test": voc_root}
    return data_roots, splits_root, classes_path


def test_dataset_returns_rgb_tensor_and_deduplicated_multihot(tmp_path: Path) -> None:
    data_roots, splits_root, classes_path = _create_fixture(tmp_path)
    dataset = PascalVOCMultiLabelDataset(
        "train", data_roots, splits_root, classes_path
    )

    image, target = dataset[0]

    assert image.shape == (3, 3, 2)
    assert image.dtype == torch.float32
    assert torch.equal(target, torch.tensor([1.0, 1.0]))


def test_dataset_rejects_missing_classes_file(tmp_path: Path) -> None:
    data_roots, splits_root, _ = _create_fixture(tmp_path)

    with pytest.raises(FileNotFoundError, match="class mapping"):
        PascalVOCMultiLabelDataset(
            "train", data_roots, splits_root, tmp_path / "missing.json"
        )
