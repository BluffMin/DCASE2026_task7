from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from runners.common import SimpleRunLogger, multilabel_onehot_to_index


@dataclass
class TrainConfig:
    epochs: int = 5
    lr: float = 1e-3
    batch_size: int = 16
    weight_decay: float = 0.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class GeometryTrainer:
    def __init__(self, model, train_cfg: TrainConfig, save_dir: Path):
        self.model = model.to(train_cfg.device)
        self.cfg = train_cfg
        self.logger = SimpleRunLogger(save_dir)
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
        )

    def train_one_epoch(self, loader: DataLoader, task_id: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_count = 0

        for waveforms, targets, _ in loader:
            waveforms = waveforms.to(self.cfg.device)
            targets = targets.to(self.cfg.device)
            target_index = multilabel_onehot_to_index(targets)

            logits = self.model(waveforms, task_id=task_id)
            loss = F.cross_entropy(logits, target_index)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * waveforms.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == target_index).sum().item()
            total_count += waveforms.size(0)

        return {
            "loss": total_loss / max(total_count, 1),
            "acc": total_correct / max(total_count, 1),
        }

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, task_id: int) -> float:
        self.model.eval()
        total_correct = 0
        total_count = 0

        for waveforms, targets, _ in loader:
            waveforms = waveforms.to(self.cfg.device)
            targets = targets.to(self.cfg.device)
            target_index = multilabel_onehot_to_index(targets)

            logits = self.model(waveforms, task_id=task_id)
            preds = logits.argmax(dim=1)
            total_correct += (preds == target_index).sum().item()
            total_count += waveforms.size(0)

        return total_correct / max(total_count, 1)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, task_id: int) -> None:
        for epoch in range(1, self.cfg.epochs + 1):
            train_metrics = self.train_one_epoch(train_loader, task_id=task_id)
            val_acc = self.evaluate(val_loader, task_id=task_id)
            payload = {
                "epoch": epoch,
                "task_id": task_id,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["acc"],
                "val_acc": val_acc,
            }
            self.logger.log(payload)

        self.logger.flush_jsonl()
