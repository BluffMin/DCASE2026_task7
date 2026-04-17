"""
single_file_geometry_task7.py

DCASE 2026 Task 7 auxiliary-study 브랜치의 geometry 실험을
"한 파일로 바로 테스트"할 수 있게 묶은 실행 스크립트.

원본에서 유지한 핵심:
- sample_rate = 32000
- clip_samples = 4초
- mel_bins = 64
- development_train.txt / development_test.txt 사용
- shared CNN + task-specific BN + task-specific residual adapter
- D1 checkpoint 부분 로드 가능
- current task의 BN / adapter / fc / proj / anchors만 학습

예시:
python single_file_geometry_task7.py \
  --data-root /workspace/DCASE/data \
  --task-id 0 \
  --epochs 5 \
  --batch-size 16 \
  --lr 1e-3 \
  --checkpoint /workspace/DCASE/checkpoint_D1.pth \
  --save-dir /workspace/DCASE/runs/geometry_single
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from torchlibrosa.stft import Spectrogram, LogmelFilterBank
except Exception as e:
    raise ImportError(
        "torchlibrosa가 필요합니다. `pip install torchlibrosa` 후 다시 실행하세요."
    ) from e


# =========================================================
# 1) 원본 config_task7.py에 맞춘 설정
# =========================================================

@dataclass
class Config:
    sample_rate: int = 32000
    clip_seconds: int = 4
    mel_bins: int = 64
    fmin: int = 50
    fmax: int = 14000
    window_size: int = 1024
    hop_size: int = 320
    window: str = "hann"
    pad_mode: str = "reflect"
    center: bool = True
    ref: float = 1.0
    amin: float = 1e-10
    top_db: float | None = None
    classes_num: int = 10

    @property
    def clip_samples(self) -> int:
        return self.sample_rate * self.clip_seconds


DICT_CLASS_LABELS = {
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


# =========================================================
# 2) datasetfactory_task7.py에 맞춘 데이터셋
# =========================================================

def to_one_hot(k: int, classes_num: int) -> np.ndarray:
    target = np.zeros(classes_num, dtype=np.float32)
    target[k] = 1.0
    return target


def pad_sequence(x: np.ndarray, max_len: int) -> np.ndarray:
    if len(x) < max_len:
        return np.concatenate((x, np.zeros(max_len - len(x), dtype=x.dtype)))
    return x


def pad_truncate_sequence(x: np.ndarray, max_len: int) -> np.ndarray:
    if len(x) < max_len:
        return np.concatenate((x, np.zeros(max_len - len(x), dtype=x.dtype)))
    return x[:max_len]


class DILDatasetInc(Dataset):
    def __init__(self, df: pd.DataFrame, audio_folder: str | Path, cfg: Config):
        self.df = df.reset_index(drop=True)
        self.audio_folder = Path(audio_folder)
        self.cfg = cfg
        self.data_files: List[np.ndarray] = []
        self.labels: List[np.ndarray] = []
        self.audio_files: List[str] = []
        self._load_dataset()

    def _load_dataset(self) -> None:
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            file_name = row["filename"]
            label = int(row["new_target"])
            file_path = self.audio_folder / file_name

            audio, _ = librosa.core.load(
                str(file_path),
                sr=self.cfg.sample_rate,
                mono=True,
            )
            waveform = pad_sequence(audio, self.cfg.clip_samples)
            target = to_one_hot(label, self.cfg.classes_num)

            self.data_files.append(waveform.astype(np.float32))
            self.labels.append(target.astype(np.float32))
            self.audio_files.append(file_name)

    def __len__(self) -> int:
        return len(self.data_files)

    def __getitem__(self, idx: int):
        data = torch.tensor(self.data_files[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        audio_file = self.audio_files[idx]
        return data, label, audio_file


# =========================================================
# 3) domain_net_geometry.py 기반 모델
# =========================================================

def init_layer(layer: nn.Module) -> None:
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(layer.weight)
        if getattr(layer, "bias", None) is not None:
            layer.bias.data.fill_(0.0)


def init_bn(bn: nn.Module) -> None:
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


class ResidualAdapter2d(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.down = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.up = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)
        self.scale = nn.Parameter(torch.tensor(0.1))
        init_layer(self.down)
        init_layer(self.up)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.up(F.relu_(self.down(x)))
        return x + self.scale * residual


class ConvBlockGeometry(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, nb_tasks: int):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.bnF = nn.ModuleList([nn.BatchNorm2d(out_channels) for _ in range(nb_tasks)])
        self.bnS = nn.ModuleList([nn.BatchNorm2d(out_channels) for _ in range(nb_tasks)])
        self.adapter = nn.ModuleList([ResidualAdapter2d(out_channels) for _ in range(nb_tasks)])

        init_layer(self.conv1)
        init_layer(self.conv2)
        for bn in self.bnF:
            init_bn(bn)
        for bn in self.bnS:
            init_bn(bn)

    def forward(
        self,
        x: torch.Tensor,
        task: int,
        pool_size=(2, 2),
        pool_type: str = "avg",
    ) -> torch.Tensor:
        x = F.relu_(self.bnF[task](self.conv1(x)))
        x = F.relu_(self.bnS[task](self.conv2(x)))
        x = self.adapter[task](x)

        if pool_type == "max":
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg":
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg+max":
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise ValueError("Incorrect pool_type")

        return x


class GeometryCnn14(nn.Module):
    """
    원본 geometry 모델 핵심 구조:
    - shared conv
    - task-specific BN
    - task-specific residual adapter
    - projection + class anchors
    """

    def __init__(
        self,
        sample_rate: int,
        window_size: int,
        hop_size: int,
        mel_bins: int,
        fmin: int,
        fmax: int,
        classes_num: int,
        nb_tasks: int = 3,
        embed_dim: int = 256,
    ):
        super().__init__()
        self.nb_tasks = nb_tasks
        self.classes_num = classes_num
        self.embed_dim = embed_dim

        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window="hann",
            center=True,
            pad_mode="reflect",
            freeze_parameters=True,
        )

        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax,
            ref=1.0,
            amin=1e-10,
            top_db=None,
            freeze_parameters=True,
        )

        self.bn0 = nn.ModuleList([nn.BatchNorm2d(mel_bins) for _ in range(nb_tasks)])
        self.conv_block1 = ConvBlockGeometry(1, 64, nb_tasks)
        self.conv_block2 = ConvBlockGeometry(64, 128, nb_tasks)
        self.conv_block3 = ConvBlockGeometry(128, 256, nb_tasks)
        self.conv_block4 = ConvBlockGeometry(256, 512, nb_tasks)
        self.conv_block5 = ConvBlockGeometry(512, 1024, nb_tasks)
        self.conv_block6 = ConvBlockGeometry(1024, 2048, nb_tasks)

        self.fc = nn.Linear(2048, classes_num)
        self.proj = nn.Linear(2048, embed_dim)
        self.class_anchors = nn.Parameter(torch.randn(classes_num, embed_dim))

        init_layer(self.fc)
        init_layer(self.proj)
        for bn in self.bn0:
            init_bn(bn)

    def init_anchors_from_fc(self) -> None:
        with torch.no_grad():
            fc_w = self.fc.weight.data
            pseudo = self.proj(fc_w)
            self.class_anchors.data.copy_(F.normalize(pseudo, dim=-1))

    def freeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_for_task(
        self,
        task_id: int,
        train_fc: bool = True,
        train_proj: bool = True,
        train_anchors: bool = True,
    ) -> None:
        self.freeze_all()

        for p in self.bn0[task_id].parameters():
            p.requires_grad = True

        blocks = [
            self.conv_block1,
            self.conv_block2,
            self.conv_block3,
            self.conv_block4,
            self.conv_block5,
            self.conv_block6,
        ]
        for block in blocks:
            for p in block.bnF[task_id].parameters():
                p.requires_grad = True
            for p in block.bnS[task_id].parameters():
                p.requires_grad = True
            for p in block.adapter[task_id].parameters():
                p.requires_grad = True

        if train_fc:
            for p in self.fc.parameters():
                p.requires_grad = True
        if train_proj:
            for p in self.proj.parameters():
                p.requires_grad = True
        if train_anchors:
            self.class_anchors.requires_grad = True

    def partial_load_from_baseline(self, ckpt: Dict[str, torch.Tensor]) -> Tuple[List[str], List[str]]:
        model_state = self.state_dict()
        loaded, skipped = [], []
        for k, v in ckpt.items():
            if k in model_state and model_state[k].shape == v.shape:
                model_state[k] = v
                loaded.append(k)
            else:
                skipped.append(k)
        self.load_state_dict(model_state, strict=False)
        return loaded, skipped

    def extract_backbone_feature(self, waveform: torch.Tensor, task: int) -> torch.Tensor:
        x = self.spectrogram_extractor(waveform)
        x = self.logmel_extractor(x)
        x = x.transpose(1, 3)
        x = self.bn0[task](x)
        x = x.transpose(1, 3)

        x = self.conv_block1(x, task=task, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, task=task, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, task=task, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, task=task, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, task=task, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, task=task, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)

        x = torch.mean(x, dim=3)
        x1, _ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        feat = x1 + x2
        return feat

    def extract_embedding(self, waveform: torch.Tensor, task: int) -> torch.Tensor:
        feat = self.extract_backbone_feature(waveform, task)
        emb = self.proj(feat)
        emb = F.normalize(emb, dim=-1)
        return emb

    def forward(self, waveform: torch.Tensor, task: int, return_embedding: bool = False):
        feat = self.extract_backbone_feature(waveform, task)
        logits = self.fc(feat)
        emb = F.normalize(self.proj(feat), dim=-1)
        if return_embedding:
            return logits, emb
        return logits


# =========================================================
# 4) geometry 러너: "테스트해볼 수 있는" 최소 재현
# =========================================================

def onehot_to_index(y: torch.Tensor) -> torch.Tensor:
    return y.argmax(dim=1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    task_id: int,
    device: torch.device,
    epoch: int | None = None,
    epochs: int | None = None,
) -> float:
    model.eval()
    total = 0
    correct = 0

    progress_desc = f"[Eval][Task {task_id}]"
    if epoch is not None and epochs is not None:
        progress_desc = f"[Eval][Task {task_id}][Epoch {epoch}/{epochs}]"

    pbar = tqdm(loader, desc=progress_desc, leave=False)

    for waveforms, targets, _ in pbar:
        waveforms = waveforms.to(device)
        targets = targets.to(device)
        target_idx = onehot_to_index(targets)

        logits = model(waveforms, task=task_id)
        preds = logits.argmax(dim=1)

        total += waveforms.size(0)
        correct += (preds == target_idx).sum().item()
        pbar.set_postfix(acc=f"{(correct / max(total, 1)):.4f}")

    return correct / max(total, 1)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    task_id: int,
    device: torch.device,
    anchor_pull_w: float = 0.0,
    anchor_sep_w: float = 0.0,
    epoch: int | None = None,
    epochs: int | None = None,
    log_interval: int = 20,
) -> Dict[str, float]:
    model.train()

    total = 0
    correct = 0
    total_loss = 0.0
    total_ce = 0.0
    total_pull = 0.0
    total_sep = 0.0

    progress_desc = f"[Train][Task {task_id}]"
    if epoch is not None and epochs is not None:
        progress_desc = f"[Train][Task {task_id}][Epoch {epoch}/{epochs}]"

    pbar = tqdm(loader, desc=progress_desc, leave=False)

    for batch_idx, (waveforms, targets, _) in enumerate(pbar, start=1):
        waveforms = waveforms.to(device)
        targets = targets.to(device)
        target_idx = onehot_to_index(targets)

        logits, emb = model(waveforms, task=task_id, return_embedding=True)

        ce_loss = F.cross_entropy(logits, target_idx)

        # positive anchor pull
        anchors = F.normalize(model.class_anchors, dim=-1)
        pos_anchor = anchors[target_idx]
        pull_loss = 1.0 - (emb * pos_anchor).sum(dim=-1).mean()

        # simple anchor separation
        sim = anchors @ anchors.t()
        eye = torch.eye(sim.size(0), device=sim.device, dtype=torch.bool)
        off_diag = sim.masked_fill(eye, 0.0)
        sep_loss = off_diag.pow(2).mean()

        loss = ce_loss + anchor_pull_w * pull_loss + anchor_sep_w * sep_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        batch_size = waveforms.size(0)

        total += batch_size
        correct += (preds == target_idx).sum().item()
        total_loss += loss.item() * batch_size
        total_ce += ce_loss.item() * batch_size
        total_pull += pull_loss.item() * batch_size
        total_sep += sep_loss.item() * batch_size

        avg_loss = total_loss / max(total, 1)
        avg_acc = correct / max(total, 1)

        if batch_idx == 1 or batch_idx % log_interval == 0 or batch_idx == len(loader):
            pbar.set_postfix(
                loss=f"{avg_loss:.4f}",
                acc=f"{avg_acc:.4f}",
                ce=f"{(total_ce / max(total,1)):.4f}",
                pull=f"{(total_pull / max(total,1)):.4f}",
                sep=f"{(total_sep / max(total,1)):.4f}",
            )

    return {
        "loss": total_loss / max(total, 1),
        "ce_loss": total_ce / max(total, 1),
        "anchor_pull_loss": total_pull / max(total, 1),
        "anchor_sep_loss": total_sep / max(total, 1),
        "acc": correct / max(total, 1),
    }


def load_split_dataframe(split_path: Path) -> pd.DataFrame:
    return pd.read_csv(
        split_path,
        sep="\t",
        names=["filename", "target", "domain", "new_target"],
    )


def save_jsonl(records: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# =========================================================
# 5) main
# =========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True,
                        help="task7_data 상위 경로. 예: /workspace/DCASE/data")
    parser.add_argument("--train-split", type=str, default="evaluation_setup/development_train.txt")
    parser.add_argument("--test-split", type=str, default="evaluation_setup/development_test.txt")
    parser.add_argument("--task-id", type=int, default=0, choices=[0, 1, 2],
                        help="0=D1, 1=D2, 2=D3")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--anchor-pull-w", type=float, default=0.0)
    parser.add_argument("--anchor-sep-w", type=float, default=0.0)
    parser.add_argument("--checkpoint", type=str, default="",
                        help="checkpoint_D1.pth 같은 baseline checkpoint 경로")
    parser.add_argument("--save-dir", type=str, default="runs/geometry_single")
    parser.add_argument("--log-interval", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(args.data_root)
    train_split_path = data_root / args.train_split
    test_split_path = data_root / args.test_split

    if not train_split_path.exists():
        raise FileNotFoundError(f"train split을 찾을 수 없습니다: {train_split_path}")
    if not test_split_path.exists():
        raise FileNotFoundError(f"test split을 찾을 수 없습니다: {test_split_path}")

    train_df = load_split_dataframe(train_split_path)
    test_df = load_split_dataframe(test_split_path)

    train_ds = DILDatasetInc(train_df, audio_folder=data_root, cfg=cfg)
    test_ds = DILDatasetInc(test_df, audio_folder=data_root, cfg=cfg)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = GeometryCnn14(
        sample_rate=cfg.sample_rate,
        window_size=cfg.window_size,
        hop_size=cfg.hop_size,
        mel_bins=cfg.mel_bins,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
        classes_num=cfg.classes_num,
        nb_tasks=3,
        embed_dim=args.embed_dim,
    )

    # baseline checkpoint partial load
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"checkpoint를 찾을 수 없습니다: {ckpt_path}")

        ckpt_obj = torch.load(str(ckpt_path), map_location="cpu")
        if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj:
            ckpt_state = ckpt_obj["state_dict"]
        else:
            ckpt_state = ckpt_obj

        loaded, skipped = model.partial_load_from_baseline(ckpt_state)
        print(f"[checkpoint] loaded={len(loaded)}, skipped={len(skipped)}")
        model.init_anchors_from_fc()

    model.unfreeze_for_task(args.task_id)
    model = model.to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    logs = []
    best_acc = -1.0
    best_path = save_dir / f"best_task{args.task_id}.pth"

    print("=" * 80)
    print("[Start]")
    print(f"device        : {device}")
    print(f"data_root     : {data_root}")
    print(f"train_split   : {train_split_path}")
    print(f"test_split    : {test_split_path}")
    print(f"train_samples : {len(train_ds)}")
    print(f"test_samples  : {len(test_ds)}")
    print(f"task_id       : {args.task_id}")
    print(f"epochs        : {args.epochs}")
    print(f"batch_size    : {args.batch_size}")
    print(f"lr            : {args.lr}")
    print(f"save_dir      : {save_dir}")
    print("=" * 80)

    for epoch in range(1, args.epochs + 1):
        print(f"[Epoch {epoch}/{args.epochs}] training...")
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            task_id=args.task_id,
            device=device,
            anchor_pull_w=args.anchor_pull_w,
            anchor_sep_w=args.anchor_sep_w,
            epoch=epoch,
            epochs=args.epochs,
            log_interval=args.log_interval,
        )

        print(f"[Epoch {epoch}/{args.epochs}] evaluating...")
        test_acc = evaluate(
            model=model,
            loader=test_loader,
            task_id=args.task_id,
            device=device,
            epoch=epoch,
            epochs=args.epochs,
        )

        row = {
            "epoch": epoch,
            "task_id": args.task_id,
            **train_metrics,
            "test_acc": test_acc,
        }
        logs.append(row)

        print(
            "[Epoch Summary] "
            f"epoch={epoch} "
            f"loss={row['loss']:.4f} "
            f"ce={row['ce_loss']:.4f} "
            f"pull={row['anchor_pull_loss']:.4f} "
            f"sep={row['anchor_sep_loss']:.4f} "
            f"train_acc={row['acc']:.4f} "
            f"test_acc={row['test_acc']:.4f}"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), best_path)
            print(f"[Checkpoint] best model saved to: {best_path}")

    save_jsonl(logs, save_dir / "train_log.jsonl")

    summary = {
        "task_id": args.task_id,
        "best_test_acc": best_acc,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "checkpoint": args.checkpoint,
        "save_dir": str(save_dir),
    }
    with (save_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[done] best_test_acc={best_acc:.4f}")
    print(f"[saved] {best_path}")
    print(f"[saved] {save_dir / 'train_log.jsonl'}")
    print(f"[saved] {save_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
