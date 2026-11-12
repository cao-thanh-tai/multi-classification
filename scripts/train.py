"""Vỏ dòng lệnh để thực hiện huấn luyện mô hình."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Thêm thư mục gốc vào sys.path để hỗ trợ import src.*
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch

from src.data.dataloader import build_dataloaders
from src.models.builder import build_model
from src.training.trainer import Trainer
from src.utils.config import load_yaml
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    """Phân tích các đối số dòng lệnh."""
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình phân loại đa nhãn PASCAL VOC.")
    parser.add_argument("--model", type=str, required=True, help="Tên model cần huấn luyện (vd: resnet50)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (mặc định đọc từ config)")
    parser.add_argument("--epochs", type=int, default=None, help="Số epoch ghi đè")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size ghi đè")
    parser.add_argument("--limit", type=int, default=None, help="Giới hạn số mẫu (dành cho smoke test)")
    parser.add_argument("--config-train", type=str, default="config/train.yaml", help="Đường dẫn file train.yaml")
    parser.add_argument("--config-data", type=str, default="config/data.yaml", help="Đường dẫn file data.yaml")
    return parser.parse_args()


def main() -> None:
    """Điểm nhập thực thi huấn luyện mô hình."""
    args = parse_args()

    train_cfg = load_yaml(args.config_train)
    data_cfg = load_yaml(args.config_data)

    seed = args.seed if args.seed is not None else int(train_cfg.get("seed", 42))
    set_seed(seed)

    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = build_dataloaders(
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        limit=args.limit,
    )

    model = build_model(args.model)

    trainer = Trainer(
        model=model,
        model_name=args.model,
        seed=seed,
        train_loader=train_loader,
        val_loader=val_loader,
        train_cfg=train_cfg,
        device=device,
    )

    trainer.fit()


if __name__ == "__main__":
    main()
