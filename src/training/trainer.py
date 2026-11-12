"""Triển khai vòng lặp huấn luyện dùng chung cho toàn bộ các mô hình."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.evaluator import evaluate
from src.training.optim import build_optimizer, build_scheduler
from src.utils.logger import setup_logger


class Trainer:
    """Quản lý toàn bộ quy trình huấn luyện, đánh giá và lưu trữ checkpoint."""

    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        seed: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        train_cfg: dict[str, Any],
        device: torch.device,
        exp_dir: Path | None = None,
    ) -> None:
        """Khởi tạo Trainer với model, dataloaders và các tham số huấn luyện."""
        self.model = model.to(device)
        self.model_name = model_name
        self.seed = seed
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_cfg = train_cfg
        self.device = device

        if exp_dir is None:
            base_exp = Path(train_cfg.get("output", {}).get("experiments_dir", "experiments"))
            self.exp_dir = base_exp / f"{model_name}_seed{seed}"
        else:
            self.exp_dir = Path(exp_dir)

        self.ckpt_dir = self.exp_dir / "checkpoints"
        self.log_dir = self.exp_dir / "logs"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = setup_logger(
            name=f"{model_name}_seed{seed}",
            log_file=self.log_dir / "train.log",
        )

        self.epochs = int(train_cfg.get("epochs", 20))
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = build_optimizer(self.model, train_cfg)
        self.scheduler = build_scheduler(self.optimizer, train_cfg, self.epochs)
        self.threshold = float(train_cfg.get("threshold", 0.5))
        self.save_best_metric = str(train_cfg.get("save_best_metric", "macro_f1"))

        self.params_m = round(sum(p.numel() for p in self.model.parameters()) / 1e6, 2)

    def train_epoch(self, epoch: int) -> float:
        """Huấn luyện mô hình trong 1 epoch và trả về loss trung bình."""
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for images, targets in self.train_loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        if self.scheduler is not None:
            self.scheduler.step()

        return total_loss / max(total_samples, 1)

    def fit(self) -> dict[str, Any]:
        """Thực hiện toàn bộ quá trình huấn luyện qua các epoch và lưu kết quả."""
        self.logger.info(
            f"Bắt đầu huấn luyện {self.model_name} (Seed {self.seed}) | "
            f"Tham số: {self.params_m}M | Thiết bị: {self.device.type}"
        )

        best_score = -1.0
        best_metrics: dict[str, Any] = {}
        start_time = time.perf_counter()

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.perf_counter()
            train_loss = self.train_epoch(epoch)
            val_metrics = evaluate(
                self.model,
                self.val_loader,
                self.criterion,
                self.device,
                threshold=self.threshold,
            )
            epoch_duration = round(time.perf_counter() - epoch_start, 1)

            current_score = val_metrics.get(self.save_best_metric, 0.0)
            self.logger.info(
                f"Epoch [{epoch:02d}/{self.epochs:02d}] ({epoch_duration}s) | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['val_loss']:.4f} | "
                f"Macro F1: {val_metrics['macro_f1']:.4f} | mAP: {val_metrics['mAP']:.4f}"
            )

            # Lưu checkpoint tốt nhất
            if current_score > best_score:
                best_score = current_score
                best_metrics = val_metrics.copy()
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "metrics": val_metrics,
                    },
                    self.ckpt_dir / "best.pt",
                )

        total_duration_min = round((time.perf_counter() - start_time) / 60.0, 2)
        self.logger.info(
            f"Huấn luyện hoàn tất sau {total_duration_min} phút | "
            f"Best {self.save_best_metric}: {best_score:.4f}"
        )

        # Lưu checkpoint cuối cùng
        torch.save(
            {
                "epoch": self.epochs,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": val_metrics,
            },
            self.ckpt_dir / "last.pt",
        )

        final_summary = {
            "model": self.model_name,
            "seed": self.seed,
            "params_m": self.params_m,
            "train_min": total_duration_min,
            "infer_ms": best_metrics.get("infer_ms", 0.0),
            "macro_f1": best_metrics.get("macro_f1", 0.0),
            "micro_f1": best_metrics.get("micro_f1", 0.0),
            "mAP": best_metrics.get("mAP", 0.0),
            "subset_accuracy": best_metrics.get("subset_accuracy", 0.0),
            "val_loss": best_metrics.get("val_loss", 0.0),
        }

        metrics_file = self.exp_dir / "metrics.json"
        with metrics_file.open("w", encoding="utf-8") as f:
            json.dump(final_summary, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Đã lưu kết quả tóm tắt vào: {metrics_file}")
        return final_summary
