from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import torch


@dataclass
class EpochLog:
    epoch: int
    train_loss: float
    train_acc: float
    val_acc: float


class SimpleRunLogger:
    def __init__(self, save_dir: Path):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.records: List[Dict] = []

    def log(self, payload: Dict) -> None:
        self.records.append(payload)

    def flush_jsonl(self, filename: str = "train_log.jsonl") -> None:
        path = self.save_dir / filename
        with path.open("w", encoding="utf-8") as f:
            for row in self.records:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def multilabel_onehot_to_index(targets: torch.Tensor) -> torch.Tensor:
    return targets.argmax(dim=1)
