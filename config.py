from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd


@dataclass
class AudioConfig:
    sample_rate: int = 32000
    clip_seconds: int = 4
    mel_bins: int = 64
    fmin: int = 50
    fmax: int = 14000
    window_size: int = 1024
    hop_size: int = 320

    @property
    def clip_samples(self) -> int:
        return self.sample_rate * self.clip_seconds


@dataclass
class DataConfig:
    data_root: Path
    train_split_relpath: str = "evaluation_setup/development_train.txt"
    test_split_relpath: str = "evaluation_setup/development_test.txt"

    @property
    def train_split_path(self) -> Path:
        return self.data_root / self.train_split_relpath

    @property
    def test_split_path(self) -> Path:
        return self.data_root / self.test_split_relpath


CLASS_TO_INDEX: Dict[str, int] = {
    "alarm": 0,
    "baby_cry": 1,
    "dog_bark": 2,
    "engine": 3,
    "fire": 4,
    "footsteps": 5,
    "knocking": 6,
    "telephone_ringing": 7,
    "piano": 8,
    "speech": 9,
}


def load_split_dataframe(split_path: Path) -> pd.DataFrame:
    return pd.read_csv(
        split_path,
        sep="\t",
        names=["filename", "target", "domain", "new_target"],
    )
