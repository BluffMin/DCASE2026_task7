#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Domain-Specific Augmentation Training + TTA Routing for DCASE Task7 D2/D3.

핵심 목적
---------
1) D1 checkpoint에서 시작한다.
2) D2 expert는 D2에 지정한 augmentation으로 학습한다.
3) D3 expert는 D3에 지정한 augmentation으로 학습한다.
4) Inference 때는 각 expert에 해당 domain-specific TTA augmentation set을 모두 적용한다.
5) 각 expert의 TTA 결과에서 entropy / mean entropy / consistency score를 계산한다.
6) score가 가장 좋은 expert를 선택하거나, top-k expert를 MoE처럼 섞어 최종 예측한다.

예시 실행
---------
python train_domain_aug_tta_routing.py \
  --data_root ./data/task7_data \
  --d1_checkpoint ./checkpoints/BN/checkpoint_D1.pth \
  --save_dir ./runs/domain_aug_tta_routing \
  --epochs 120 \
  --batch_size 32 \
  --num_workers 6 \
  --lr 1e-4 \
  --domain_train_aug D2:device,D3:noise \
  --domain_tta_aug D2:device,D3:noise \
  --score_types entropy_mean_probs mean_entropy consistency \
  --top_ks 2 \
  --taus 0.5 1.0 2.0 \
  --cuda

학습을 이미 끝낸 checkpoint로 routing만 다시 평가
----------------------------------------------
python train_domain_aug_tta_routing.py \
  --data_root ./data/task7_data \
  --save_dir ./runs/domain_aug_tta_routing_eval \
  --d2_checkpoint ./runs/domain_aug_tta_routing/checkpoints/best_D2.pth \
  --d3_checkpoint ./runs/domain_aug_tta_routing/checkpoints/best_D3.pth \
  --skip_train \
  --domain_tta_aug D2:device,D3:noise \
  --cuda

주의
----
- split 파일은 data_root/evaluation_setup/development_train.txt,
  data_root/evaluation_setup/development_test.txt 를 사용한다.
- wav 경로는 data_root / filename 으로 읽는다.
- 모델은 domain_net.py의 MCnn14를 사용한다고 가정한다.
- 현재 코드는 D2/D3만 대상으로 한다. D1은 시작 checkpoint로만 사용한다.
"""

from __future__ import annotations

import os
import sys
import json
import math
import time
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import librosa
import numpy as np
import pandas as pd
from sklearn import metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------
# Repo-root import
# ---------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
for parent in THIS_FILE.parents:
    if (parent / "config_task7.py").exists() or (parent / "domain_net.py").exists():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

try:
    from domain_net import MCnn14
except ImportError:
    from baseline.domain_net import MCnn14


# ---------------------------------------------------------
# Fixed setup
# ---------------------------------------------------------
SAMPLE_RATE = 32000
CLIP_SECONDS = 4
CLIP_SAMPLES = SAMPLE_RATE * CLIP_SECONDS

MEL_BINS = 64
FMIN = 50
FMAX = 14000
WINDOW_SIZE = 1024
HOP_SIZE = 320

CLASSES_NUM = 10
NB_TASKS = 3

TASK_TO_DOMAIN = {0: "D1", 1: "D2", 2: "D3"}
DOMAIN_TO_TASK = {"D1": 0, "D2": 1, "D3": 2}
TARGET_DOMAINS = ["D2", "D3"]
TARGET_TASKS = [1, 2]


# ---------------------------------------------------------
# Basic utils
# ---------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def safe_name(x: str) -> str:
    return x.replace(".", "p").replace("/", "_").replace(" ", "_").replace(",", "_").replace(":", "-")


def load_split_df(data_root: str, split_name: str) -> pd.DataFrame:
    split_path = Path(data_root) / "evaluation_setup" / split_name
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    df = pd.read_csv(
        split_path,
        sep="\t",
        header=None,
        names=["filename", "target", "domain", "new_target"],
    )
    return df


def get_class_names(df: pd.DataFrame) -> Dict[int, str]:
    mapping = {}
    for _, row in df.iterrows():
        mapping[int(row["new_target"])] = str(row["target"])
    return mapping


def pad_or_truncate(x: np.ndarray, max_len: int = CLIP_SAMPLES) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if len(x) < max_len:
        return np.concatenate([x, np.zeros(max_len - len(x), dtype=np.float32)], axis=0)
    return x[:max_len].astype(np.float32)


def random_crop_or_pad(x: np.ndarray, max_len: int = CLIP_SAMPLES) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if len(x) < max_len:
        return np.concatenate([x, np.zeros(max_len - len(x), dtype=np.float32)], axis=0)
    if len(x) == max_len:
        return x.astype(np.float32)
    start = np.random.randint(0, len(x) - max_len + 1)
    return x[start:start + max_len].astype(np.float32)


def split_into_chunks(x: np.ndarray, chunk_size: int = CLIP_SAMPLES) -> List[np.ndarray]:
    x = np.asarray(x, dtype=np.float32)

    if len(x) <= chunk_size:
        return [pad_or_truncate(x, chunk_size)]

    chunks = []
    start = 0
    while start < len(x):
        chunks.append(pad_or_truncate(x[start:start + chunk_size], chunk_size))
        start += chunk_size

    return chunks


# ---------------------------------------------------------
# Audio augmentation
# ---------------------------------------------------------
def normalize_peak(audio: np.ndarray, peak: float = 0.99) -> np.ndarray:
    audio = audio.astype(np.float32)
    m = np.max(np.abs(audio))
    if m > peak:
        audio = audio / (m + 1e-12) * peak
    return audio.astype(np.float32)


def aug_identity(audio: np.ndarray) -> np.ndarray:
    return audio.astype(np.float32)


def aug_gain(audio: np.ndarray, gain_db: float) -> np.ndarray:
    gain = 10.0 ** (gain_db / 20.0)
    return normalize_peak(audio * gain)


def aug_time_shift(audio: np.ndarray, shift_seconds: float) -> np.ndarray:
    shift = int(round(shift_seconds * SAMPLE_RATE))
    return np.roll(audio, shift).astype(np.float32)


def aug_noise(audio: np.ndarray, std: float, seed: Optional[int] = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, std, size=len(audio)).astype(np.float32)
    return normalize_peak(audio + noise)


def aug_random_crop_shift(audio: np.ndarray, max_shift_seconds: float = 0.12) -> np.ndarray:
    shift = np.random.uniform(-max_shift_seconds, max_shift_seconds)
    return aug_time_shift(audio, shift)


def aug_fft_tilt(audio: np.ndarray, mode: str) -> np.ndarray:
    """
    device/channel-like perturbation.
    scipy 없이 FFT scale만 사용한다.
    """
    x = audio.astype(np.float32)
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / SAMPLE_RATE)

    if mode == "lowpass_like":
        start, end = 7000.0, 15000.0
        ramp = np.clip((freqs - start) / max(end - start, 1.0), 0.0, 1.0)
        scale = 1.0 - 0.45 * ramp
    elif mode == "highpass_like":
        start, end = 50.0, 800.0
        ramp = np.clip((freqs - start) / max(end - start, 1.0), 0.0, 1.0)
        scale = 0.65 + 0.35 * ramp
    else:
        raise ValueError(f"Unknown fft tilt mode: {mode}")

    y = np.fft.irfft(spec * scale, n=len(x)).astype(np.float32)
    return normalize_peak(y)


def apply_train_aug(audio: np.ndarray, aug_mode: str) -> np.ndarray:
    """
    학습용 augmentation.
    매 sample마다 stochastic하게 적용된다.
    """
    x = audio.astype(np.float32)

    if aug_mode in ["none", "identity"]:
        return x

    if aug_mode == "light":
        # 약한 gain + 약한 shift
        if np.random.rand() < 0.7:
            x = aug_gain(x, np.random.uniform(-3.0, 3.0))
        if np.random.rand() < 0.5:
            x = aug_random_crop_shift(x, 0.08)
        return x.astype(np.float32)

    if aug_mode == "device":
        # device/channel mismatch 흉내
        r = np.random.rand()
        if r < 0.35:
            x = aug_fft_tilt(x, "lowpass_like")
        elif r < 0.70:
            x = aug_fft_tilt(x, "highpass_like")
        if np.random.rand() < 0.7:
            x = aug_gain(x, np.random.uniform(-4.0, 4.0))
        return x.astype(np.float32)

    if aug_mode == "noise":
        if np.random.rand() < 0.8:
            x = aug_noise(x, std=np.random.uniform(0.0002, 0.0008), seed=None)
        if np.random.rand() < 0.5:
            x = aug_gain(x, np.random.uniform(-2.0, 2.0))
        return x.astype(np.float32)

    if aug_mode == "gain_shift":
        if np.random.rand() < 0.85:
            x = aug_gain(x, np.random.uniform(-4.0, 4.0))
        if np.random.rand() < 0.7:
            x = aug_random_crop_shift(x, 0.10)
        return x.astype(np.float32)

    if aug_mode == "device_light":
        # light augmentation + weak device/channel perturbation.
        if np.random.rand() < 0.7:
            x = aug_gain(x, np.random.uniform(-3.0, 3.0))
        if np.random.rand() < 0.5:
            x = aug_random_crop_shift(x, 0.08)
        if np.random.rand() < 0.35:
            x = aug_fft_tilt(x, "lowpass_like" if np.random.rand() < 0.5 else "highpass_like")
            x = 0.75 * audio.astype(np.float32) + 0.25 * x.astype(np.float32)
        return normalize_peak(x).astype(np.float32)

    if aug_mode == "strong":
        if np.random.rand() < 0.8:
            x = aug_gain(x, np.random.uniform(-5.0, 5.0))
        if np.random.rand() < 0.6:
            x = aug_random_crop_shift(x, 0.12)
        if np.random.rand() < 0.5:
            x = aug_noise(x, std=np.random.uniform(0.0002, 0.0008), seed=None)
        if np.random.rand() < 0.5:
            x = aug_fft_tilt(x, "lowpass_like" if np.random.rand() < 0.5 else "highpass_like")
        return x.astype(np.float32)

    if aug_mode == "crop_gain_shift_noise":
        if np.random.rand() < 0.8:
            x = aug_gain(x, np.random.uniform(-4.0, 4.0))
        if np.random.rand() < 0.7:
            x = aug_random_crop_shift(x, 0.12)
        if np.random.rand() < 0.5:
            x = aug_noise(x, std=np.random.uniform(0.0002, 0.0007), seed=None)
        return x.astype(np.float32)

    raise ValueError(f"Unknown train aug mode: {aug_mode}")


def make_tta_audios(audio: np.ndarray, aug_set: str, base_seed: int = 0) -> List[np.ndarray]:
    """
    Inference/TTA용 deterministic augmentation set.
    routing score가 랜덤성에 흔들리지 않도록 seed를 고정한다.
    """
    audio = audio.astype(np.float32)

    if aug_set in ["none", "identity"]:
        return [aug_identity(audio)]

    if aug_set == "light":
        return [
            aug_identity(audio),
            aug_gain(audio, -3.0),
            aug_gain(audio, 3.0),
            aug_time_shift(audio, -0.08),
            aug_time_shift(audio, 0.08),
        ]

    if aug_set == "device":
        return [
            aug_identity(audio),
            aug_gain(audio, -3.0),
            aug_gain(audio, 3.0),
            aug_fft_tilt(audio, "lowpass_like"),
            aug_fft_tilt(audio, "highpass_like"),
        ]

    if aug_set == "noise":
        return [
            aug_identity(audio),
            aug_noise(audio, 0.0003, base_seed + 11),
            aug_noise(audio, 0.0006, base_seed + 17),
            aug_gain(audio, -2.0),
            aug_gain(audio, 2.0),
        ]

    if aug_set == "strong":
        return [
            aug_identity(audio),
            aug_gain(audio, -4.0),
            aug_gain(audio, 4.0),
            aug_time_shift(audio, -0.12),
            aug_time_shift(audio, 0.12),
            aug_fft_tilt(audio, "lowpass_like"),
            aug_fft_tilt(audio, "highpass_like"),
            aug_noise(audio, 0.0005, base_seed + 23),
        ]

    if aug_set == "crop_gain_shift_noise":
        return [
            aug_identity(audio),
            aug_gain(audio, -4.0),
            aug_gain(audio, 4.0),
            aug_time_shift(audio, -0.12),
            aug_time_shift(audio, 0.12),
            aug_noise(audio, 0.0003, base_seed + 31),
            aug_noise(audio, 0.0006, base_seed + 37),
        ]

    if aug_set == "gain_shift":
        return [
            aug_identity(audio),
            aug_gain(audio, -4.0),
            aug_gain(audio, 4.0),
            aug_time_shift(audio, -0.10),
            aug_time_shift(audio, 0.10),
        ]

    if aug_set == "device_light":
        low = 0.75 * audio + 0.25 * aug_fft_tilt(audio, "lowpass_like")
        high = 0.75 * audio + 0.25 * aug_fft_tilt(audio, "highpass_like")
        return [
            aug_identity(audio),
            aug_gain(audio, -3.0),
            aug_gain(audio, 3.0),
            aug_time_shift(audio, -0.08),
            aug_time_shift(audio, 0.08),
            normalize_peak(low),
            normalize_peak(high),
        ]

    raise ValueError(f"Unknown TTA aug set: {aug_set}")


def parse_domain_aug_map(
    domain_aug: str,
    default_aug: str,
    include_d1: bool = False,
) -> Dict[int, str]:
    """
    예:
      --domain_train_aug "D2:device,D3:noise"
      --domain_tta_aug "D1:identity,D2:device,D3:noise"

    include_d1=False이면 D2/D3만 기본 map에 포함한다.
    include_d1=True이면 D1/D2/D3 모두 기본 map에 포함한다.

    주의:
    - D1은 학습 augmentation 대상이 아니라 routing 후보용 TTA augmentation 대상이다.
    - D1은 보통 identity 또는 light를 권장한다.
    """
    domains = ["D1", "D2", "D3"] if include_d1 else TARGET_DOMAINS
    out = {DOMAIN_TO_TASK[d]: default_aug for d in domains}

    if domain_aug is None or domain_aug.strip() == "":
        return out

    for item in domain_aug.split(","):
        item = item.strip()
        if not item:
            continue
        domain, aug = item.split(":")
        domain = domain.strip()
        aug = aug.strip()
        if domain not in DOMAIN_TO_TASK:
            raise ValueError(f"Unknown domain in augmentation map: {domain}")
        out[DOMAIN_TO_TASK[domain]] = aug

    return out


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------
class DomainAudioDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        data_root: str,
        domain: str,
        train: bool,
        aug_mode: str = "identity",
        clip_samples: int = CLIP_SAMPLES,
    ) -> None:
        self.df = df[df["domain"] == domain].copy().reset_index(drop=True)
        self.data_root = Path(data_root)
        self.domain = domain
        self.task_id = DOMAIN_TO_TASK[domain]
        self.train = train
        self.aug_mode = aug_mode
        self.clip_samples = clip_samples

        if len(self.df) == 0:
            raise ValueError(f"No samples for domain={domain}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        wav_path = self.data_root / row["filename"]

        if not wav_path.exists():
            raise FileNotFoundError(f"Audio file not found: {wav_path}")

        audio, _ = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
        audio = audio.astype(np.float32)

        if self.train:
            audio = random_crop_or_pad(audio, self.clip_samples)
            audio = apply_train_aug(audio, self.aug_mode)
        else:
            audio = pad_or_truncate(audio, self.clip_samples)

        target = int(row["new_target"])
        x = torch.from_numpy(audio.astype(np.float32))
        y = torch.tensor(target, dtype=torch.long)
        return x, y


def make_loader(
    df: pd.DataFrame,
    data_root: str,
    domain: str,
    train: bool,
    aug_mode: str,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    ds = DomainAudioDataset(
        df=df,
        data_root=data_root,
        domain=domain,
        train=train,
        aug_mode=aug_mode,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


# ---------------------------------------------------------
# Model loading/saving
# ---------------------------------------------------------
def build_model(device: str) -> MCnn14:
    model = MCnn14(
        sample_rate=SAMPLE_RATE,
        window_size=WINDOW_SIZE,
        hop_size=HOP_SIZE,
        mel_bins=MEL_BINS,
        fmin=FMIN,
        fmax=FMAX,
        classes_num=CLASSES_NUM,
        nb_tasks=NB_TASKS,
    ).to(device)
    return model


def extract_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        for key in ["state_dict", "model_state_dict", "model", "net", "network"]:
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    return obj


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            out[k[len("module."):]] = v
        else:
            out[k] = v
    return out


def load_model_checkpoint(model: nn.Module, ckpt_path: str, device: str, strict: bool = False) -> None:
    ckpt_path = str(ckpt_path)
    state = torch.load(ckpt_path, map_location=device)
    state = extract_state_dict(state)
    state = strip_module_prefix(state)

    missing, unexpected = model.load_state_dict(state, strict=strict)
    print(f"[load] {ckpt_path}")
    print(f"[load] missing={len(missing)}, unexpected={len(unexpected)}")
    if len(missing) > 0:
        print(f"[missing example] {missing[:10]}")
    if len(unexpected) > 0:
        print(f"[unexpected example] {unexpected[:10]}")


def save_checkpoint(
    model: nn.Module,
    path: Path,
    domain: str,
    task_id: int,
    epoch: int,
    best_acc: float,
    train_aug: str,
    extra: Optional[Dict] = None,
) -> None:
    ensure_dir(path.parent)
    payload = {
        "model_state_dict": model.state_dict(),
        "domain": domain,
        "task_id": task_id,
        "epoch": epoch,
        "best_acc": best_acc,
        "train_aug": train_aug,
    }
    if extra is not None:
        payload.update(extra)
    torch.save(payload, path)


# ---------------------------------------------------------
# Training / validation
# ---------------------------------------------------------
@torch.no_grad()
def evaluate_single_model_loader(
    model: nn.Module,
    loader: DataLoader,
    task_id: int,
    device: str,
) -> Dict[str, float]:
    model.eval()

    total = 0
    correct = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x, task=task_id)
        loss = criterion(logits, y)

        pred = torch.argmax(logits, dim=-1)
        total += y.numel()
        correct += int((pred == y).sum().item())
        loss_sum += float(loss.item()) * y.numel()

    acc = 100.0 * correct / max(total, 1)
    avg_loss = loss_sum / max(total, 1)
    return {"acc": round(acc, 4), "loss": round(avg_loss, 6)}


def train_one_domain_expert(
    args: argparse.Namespace,
    domain: str,
    task_id: int,
    train_aug: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    device: str,
    save_dir: Path,
) -> Tuple[nn.Module, Path, Dict]:
    print("\n" + "=" * 100)
    print(f"[TRAIN EXPERT] domain={domain}, task_id={task_id}, train_aug={train_aug}")
    print("=" * 100)

    model = build_model(device)

    if args.d1_checkpoint:
        print(f"[init] load D1 checkpoint: {args.d1_checkpoint}")
        load_model_checkpoint(model, args.d1_checkpoint, device=device, strict=False)

    train_loader = make_loader(
        df=train_df,
        data_root=args.data_root,
        domain=domain,
        train=True,
        aug_mode=train_aug,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    val_loader = make_loader(
        df=test_df,
        data_root=args.data_root,
        domain=domain,
        train=False,
        aug_mode="identity",
        batch_size=args.batch_size_eval,
        num_workers=args.num_workers,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
        eta_min=args.min_lr,
    )

    ckpt_dir = save_dir / "checkpoints"
    best_path = ckpt_dir / f"best_{domain}.pth"
    last_path = ckpt_dir / f"last_{domain}.pth"

    best_acc = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()

        total = 0
        correct = 0
        loss_sum = 0.0
        start_time = time.time()

        for step, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x, task=task_id)
            loss = criterion(logits, y)
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

            pred = torch.argmax(logits, dim=-1)
            total += y.numel()
            correct += int((pred == y).sum().item())
            loss_sum += float(loss.item()) * y.numel()

            if getattr(args, "max_train_batches", -1) > 0 and step >= args.max_train_batches:
                print(f"[smoke] stop train epoch after {step} batch(es)")
                break

        scheduler.step()

        train_acc = 100.0 * correct / max(total, 1)
        train_loss = loss_sum / max(total, 1)
        val_metrics = evaluate_single_model_loader(
            model=model,
            loader=val_loader,
            task_id=task_id,
            device=device,
        )

        row = {
            "epoch": epoch,
            "domain": domain,
            "task_id": task_id,
            "train_aug": train_aug,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 4),
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_sec": round(time.time() - start_time, 2),
        }
        history.append(row)

        print(
            f"[{domain}] epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={row['train_loss']:.4f}, train_acc={row['train_acc']:.2f} | "
            f"val_loss={row['val_loss']:.4f}, val_acc={row['val_acc']:.2f} | "
            f"lr={row['lr']:.2e}"
        )

        if val_metrics["acc"] > best_acc:
            best_acc = val_metrics["acc"]
            save_checkpoint(
                model=model,
                path=best_path,
                domain=domain,
                task_id=task_id,
                epoch=epoch,
                best_acc=best_acc,
                train_aug=train_aug,
                extra={"args": vars(args)},
            )
            print(f"[save best] {best_path} | best_acc={best_acc:.2f}")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_checkpoint(
                model=model,
                path=last_path,
                domain=domain,
                task_id=task_id,
                epoch=epoch,
                best_acc=best_acc,
                train_aug=train_aug,
                extra={"args": vars(args)},
            )

    hist_df = pd.DataFrame(history)
    hist_path = save_dir / "logs" / f"train_history_{domain}.csv"
    ensure_dir(hist_path.parent)
    hist_df.to_csv(hist_path, index=False, encoding="utf-8-sig")

    # best checkpoint reload
    best_model = build_model(device)
    load_model_checkpoint(best_model, str(best_path), device=device, strict=False)
    best_model.eval()

    summary = {
        "domain": domain,
        "task_id": task_id,
        "train_aug": train_aug,
        "best_acc": best_acc,
        "best_checkpoint": str(best_path),
        "history_csv": str(hist_path),
    }
    save_json(summary, save_dir / "logs" / f"train_summary_{domain}.json")

    return best_model, best_path, summary


def train_or_load_experts(
    args: argparse.Namespace,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    device: str,
    save_dir: Path,
    train_aug_map: Dict[int, str],
) -> Tuple[Dict[int, nn.Module], Dict[str, Any]]:
    models = {}
    summaries = {}

    # Optional D1 router expert.
    # D1은 추가 학습하지 않고, D1 checkpoint를 그대로 routing 후보에 넣는다.
    # 즉 D1/D2/D3가 모두 inference candidate expert가 된다.
    if args.include_d1_router:
        if not args.d1_checkpoint:
            raise ValueError("--include_d1_router 사용 시 --d1_checkpoint가 필요합니다.")

        print("\n" + "=" * 100)
        print("[LOAD D1 ROUTER EXPERT]")
        print("=" * 100)

        d1_model = build_model(device)
        load_model_checkpoint(d1_model, args.d1_checkpoint, device=device, strict=False)
        d1_model.eval()

        models[0] = d1_model
        summaries["D1"] = {
            "domain": "D1",
            "task_id": 0,
            "loaded_checkpoint": args.d1_checkpoint,
            "train_aug": "pretrained_D1_only",
            "note": "D1 is used only as a routing candidate and is not additionally trained.",
        }

    if args.skip_train:
        ckpt_map = {
            1: args.d2_checkpoint,
            2: args.d3_checkpoint,
        }

        for task_id, ckpt in ckpt_map.items():
            domain = TASK_TO_DOMAIN[task_id]
            if not ckpt:
                raise ValueError(f"--skip_train 사용 시 --{domain.lower()}_checkpoint가 필요합니다.")

            print("\n" + "=" * 100)
            print(f"[LOAD EXPERT] {domain}: {ckpt}")
            print("=" * 100)

            model = build_model(device)
            load_model_checkpoint(model, ckpt, device=device, strict=False)
            model.eval()
            models[task_id] = model
            summaries[domain] = {
                "domain": domain,
                "task_id": task_id,
                "loaded_checkpoint": ckpt,
                "train_aug": train_aug_map.get(task_id, "unknown"),
            }

        return models, summaries

    for task_id in TARGET_TASKS:
        domain = TASK_TO_DOMAIN[task_id]
        train_aug = train_aug_map[task_id]

        model, ckpt_path, summary = train_one_domain_expert(
            args=args,
            domain=domain,
            task_id=task_id,
            train_aug=train_aug,
            train_df=train_df,
            test_df=test_df,
            device=device,
            save_dir=save_dir,
        )

        models[task_id] = model
        summaries[domain] = summary

    return models, summaries


# ---------------------------------------------------------
# TTA routing scores
# ---------------------------------------------------------
def entropy_from_probs(probs: torch.Tensor) -> torch.Tensor:
    eps = 1e-12
    return -(probs * torch.log(probs + eps)).sum(dim=-1)


def kl_to_mean(probs: torch.Tensor, mean_probs: torch.Tensor) -> torch.Tensor:
    eps = 1e-12
    return (probs * (torch.log(probs + eps) - torch.log(mean_probs.unsqueeze(0) + eps))).sum(dim=-1).mean()


@torch.no_grad()
def forward_one_model_one_audio(
    model: nn.Module,
    audio: np.ndarray,
    task_id: int,
    device: str,
) -> torch.Tensor:
    """
    긴 오디오는 4초 chunk로 나누고 logits 평균.
    return logits: [1, C]
    """
    chunks = split_into_chunks(audio, CLIP_SAMPLES)
    logits_list = []

    for chunk in chunks:
        x = torch.from_numpy(chunk.astype(np.float32)).unsqueeze(0).to(device)
        logits = model(x, task=task_id)
        logits_list.append(logits)

    logits = torch.stack(logits_list, dim=0).mean(dim=0)
    return logits


@torch.no_grad()
def compute_tta_outputs_for_task(
    model: nn.Module,
    audio: np.ndarray,
    task_id: int,
    device: str,
    tta_aug_set: str,
    base_seed: int,
) -> Dict[str, Any]:
    tta_audios = make_tta_audios(audio, aug_set=tta_aug_set, base_seed=base_seed)

    logits_list = []
    probs_list = []

    for aug_audio in tta_audios:
        logits = forward_one_model_one_audio(
            model=model,
            audio=aug_audio,
            task_id=task_id,
            device=device,
        )
        probs = torch.softmax(logits, dim=-1)

        logits_list.append(logits.squeeze(0))
        probs_list.append(probs.squeeze(0))

    logits_stack = torch.stack(logits_list, dim=0)  # [K, C]
    probs_stack = torch.stack(probs_list, dim=0)    # [K, C]

    mean_logits = logits_stack.mean(dim=0)
    mean_probs = probs_stack.mean(dim=0)
    mean_probs = mean_probs / mean_probs.sum().clamp_min(1e-12)

    entropy_mean_probs = float(entropy_from_probs(mean_probs.unsqueeze(0)).item())
    mean_entropy = float(entropy_from_probs(probs_stack).mean().item())
    consistency_kl = float(kl_to_mean(probs_stack, mean_probs).item())
    confidence = float(torch.max(mean_probs).item())
    margin = float(torch.topk(mean_probs, k=2).values.diff().abs().item())

    return {
        "logits_stack": logits_stack,
        "probs_stack": probs_stack,
        "mean_logits": mean_logits,
        "mean_probs": mean_probs,
        "entropy_mean_probs": entropy_mean_probs,
        "mean_entropy": mean_entropy,
        "consistency_kl": consistency_kl,
        "confidence": confidence,
        "margin": margin,
        "num_aug": len(tta_audios),
        "tta_aug_set": tta_aug_set,
    }


@torch.no_grad()
def forward_all_experts_with_domain_tta(
    models: Dict[int, nn.Module],
    audio: np.ndarray,
    device: str,
    tta_aug_map: Dict[int, str],
    base_seed: int,
) -> Dict[int, Dict[str, Any]]:
    outputs = {}

    for task_id, model in models.items():
        tta_aug_set = tta_aug_map[task_id]
        outputs[task_id] = compute_tta_outputs_for_task(
            model=model,
            audio=audio,
            task_id=task_id,
            device=device,
            tta_aug_set=tta_aug_set,
            base_seed=base_seed + task_id * 1000,
        )

    return outputs


def score_task(out: Dict[str, Any], score_type: str, lambda_consistency: float) -> float:
    """
    낮을수록 좋은 score로 통일.
    """
    if score_type == "entropy_mean_probs":
        return float(out["entropy_mean_probs"])

    if score_type == "mean_entropy":
        return float(out["mean_entropy"])

    if score_type == "consistency":
        return float(out["entropy_mean_probs"] + lambda_consistency * out["consistency_kl"])

    if score_type == "kl_only":
        return float(out["consistency_kl"])

    if score_type == "neg_confidence":
        return -float(out["confidence"])

    if score_type == "neg_margin":
        return -float(out["margin"])

    raise ValueError(f"Unknown score_type: {score_type}")


def select_topk_by_score(scores: Dict[int, float], top_k: int) -> List[int]:
    top_k = min(top_k, len(scores))
    return [t for t, _ in sorted(scores.items(), key=lambda x: x[1])[:top_k]]


def mix_probs_by_scores(
    outputs: Dict[int, Dict[str, Any]],
    selected_tasks: List[int],
    scores: Dict[int, float],
    tau: float,
) -> torch.Tensor:
    score_tensor = torch.tensor([scores[t] for t in selected_tasks], dtype=torch.float32)
    weights = torch.softmax(-score_tensor / max(tau, 1e-6), dim=0).to(outputs[selected_tasks[0]]["mean_probs"].device)

    final_probs = torch.zeros_like(outputs[selected_tasks[0]]["mean_probs"])
    for w, t in zip(weights, selected_tasks):
        final_probs = final_probs + w * outputs[t]["mean_probs"]

    final_probs = final_probs / final_probs.sum().clamp_min(1e-12)
    return final_probs


def get_final_probs_and_route(
    outputs: Dict[int, Dict[str, Any]],
    policy: str,
    true_task: Optional[int],
    score_type: str,
    lambda_consistency: float,
    top_k: int,
    tau: float,
) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
    tasks = sorted(outputs.keys())

    if policy == "fixed_D2":
        chosen = 1
        return outputs[chosen]["mean_probs"], chosen, {"scores": {}}

    if policy == "fixed_D3":
        chosen = 2
        return outputs[chosen]["mean_probs"], chosen, {"scores": {}}

    if policy == "oracle":
        if true_task is None:
            raise ValueError("oracle policy requires true_task")
        chosen = true_task
        return outputs[chosen]["mean_probs"], chosen, {"scores": {}}

    if policy == "all_mean":
        final_probs = torch.stack([outputs[t]["mean_probs"] for t in tasks], dim=0).mean(dim=0)
        final_probs = final_probs / final_probs.sum().clamp_min(1e-12)
        return final_probs, -1, {"scores": {}}

    scores = {t: score_task(outputs[t], score_type, lambda_consistency) for t in tasks}

    if policy == "hard":
        chosen = min(scores, key=scores.get)
        return outputs[chosen]["mean_probs"], chosen, {"scores": scores}

    if policy == "moe":
        selected = select_topk_by_score(scores, top_k=top_k)
        final_probs = mix_probs_by_scores(outputs, selected, scores, tau=tau)
        return final_probs, selected[0], {"scores": scores, "selected_tasks": selected}

    raise ValueError(f"Unknown policy: {policy}")


# ---------------------------------------------------------
# Routing evaluation
# ---------------------------------------------------------
@torch.no_grad()
def evaluate_routing_policy(
    models: Dict[int, nn.Module],
    df: pd.DataFrame,
    data_root: str,
    device: str,
    policy: str,
    score_type: str,
    lambda_consistency: float,
    top_k: int,
    tau: float,
    tta_aug_map: Dict[int, str],
    seed: int,
    max_eval_samples: int,
) -> Dict[str, Any]:
    for model in models.values():
        model.eval()

    if max_eval_samples > 0 and len(df) > max_eval_samples:
        df = df.sample(n=max_eval_samples, random_state=seed).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    y_true = []
    y_pred = []
    chosen_tasks = []
    rows = []

    for idx, row in df.iterrows():
        wav_path = Path(data_root) / row["filename"]
        if not wav_path.exists():
            raise FileNotFoundError(f"Audio file not found: {wav_path}")

        audio, _ = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
        audio = audio.astype(np.float32)

        target = int(row["new_target"])
        domain = str(row["domain"])
        true_task = DOMAIN_TO_TASK.get(domain, None)

        outputs = forward_all_experts_with_domain_tta(
            models=models,
            audio=audio,
            device=device,
            tta_aug_map=tta_aug_map,
            base_seed=seed + idx * 31,
        )

        final_probs, chosen_task, aux = get_final_probs_and_route(
            outputs=outputs,
            policy=policy,
            true_task=true_task,
            score_type=score_type,
            lambda_consistency=lambda_consistency,
            top_k=top_k,
            tau=tau,
        )

        pred = int(torch.argmax(final_probs, dim=-1).item())
        conf = float(torch.max(final_probs, dim=-1).values.item())

        y_true.append(target)
        y_pred.append(pred)
        chosen_tasks.append(chosen_task)

        row_out = {
            "filename": row["filename"],
            "domain": domain,
            "target_name": row["target"],
            "target": target,
            "pred": pred,
            "correct": int(pred == target),
            "confidence": conf,
            "true_task": true_task,
            "true_domain": domain,
            "chosen_task": chosen_task,
            "chosen_domain": TASK_TO_DOMAIN.get(chosen_task, "MIX"),
            "route_correct": int(chosen_task == true_task)
            if chosen_task in TASK_TO_DOMAIN and true_task is not None
            else np.nan,
        }

        for task_id, out in outputs.items():
            d = TASK_TO_DOMAIN[task_id]
            row_out[f"tta_aug_{d}"] = out["tta_aug_set"]
            row_out[f"num_aug_{d}"] = out["num_aug"]
            row_out[f"entropy_mean_probs_{d}"] = out["entropy_mean_probs"]
            row_out[f"mean_entropy_{d}"] = out["mean_entropy"]
            row_out[f"consistency_kl_{d}"] = out["consistency_kl"]
            row_out[f"confidence_{d}"] = out["confidence"]
            row_out[f"margin_{d}"] = out["margin"]

        if "scores" in aux:
            for task_id, score in aux["scores"].items():
                row_out[f"score_{TASK_TO_DOMAIN[task_id]}"] = float(score)

        if "selected_tasks" in aux:
            row_out["selected_domains"] = ",".join([TASK_TO_DOMAIN[t] for t in aux["selected_tasks"]])

        rows.append(row_out)

        if (idx + 1) % 100 == 0:
            print(
                f"[eval] policy={policy}, score={score_type}, "
                f"top_k={top_k}, tau={tau} | {idx + 1}/{len(df)}"
            )

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    chosen_tasks = np.array(chosen_tasks)

    acc = round(float((y_true == y_pred).mean() * 100.0), 2)
    cm = metrics.confusion_matrix(y_true, y_pred, labels=list(range(CLASSES_NUM)))

    pred_df = pd.DataFrame(rows)
    valid_route = pred_df["route_correct"].dropna()
    router_acc = round(float(valid_route.mean() * 100.0), 2) if len(valid_route) > 0 else np.nan

    route_hist = {}
    for task_id in sorted(models.keys()):
        route_hist[TASK_TO_DOMAIN[task_id]] = int(np.sum(chosen_tasks == task_id))
    route_hist["MIX"] = int(np.sum(chosen_tasks == -1))

    return {
        "acc": acc,
        "router_acc": router_acc,
        "route_hist": route_hist,
        "cm": cm,
        "pred_df": pred_df,
    }


def classwise_accuracy(pred_df: pd.DataFrame, class_name_map: Dict[int, str]) -> pd.DataFrame:
    rows = []
    for c in range(CLASSES_NUM):
        sub = pred_df[pred_df["target"] == c]
        n = len(sub)
        correct = int(sub["correct"].sum()) if n > 0 else 0
        acc = round(correct / max(n, 1) * 100.0, 2) if n > 0 else np.nan

        rows.append({
            "class_id": c,
            "class_name": class_name_map.get(c, str(c)),
            "n_samples": n,
            "correct": correct,
            "accuracy": acc,
        })

    return pd.DataFrame(rows)


def save_eval_outputs(result: Dict[str, Any], out_dir: Path, prefix: str, class_name_map: Dict[int, str]) -> Dict[str, Any]:
    ensure_dir(out_dir)

    pred_df = result["pred_df"]
    pred_df.to_csv(out_dir / f"{prefix}_predictions.csv", index=False, encoding="utf-8-sig")

    cw = classwise_accuracy(pred_df, class_name_map)
    cw.to_csv(out_dir / f"{prefix}_classwise.csv", index=False, encoding="utf-8-sig")

    np.savetxt(out_dir / f"{prefix}_confusion.csv", result["cm"], delimiter=",", fmt="%d")

    compact = {
        "acc": result["acc"],
        "router_acc": result["router_acc"],
        "route_hist": result["route_hist"],
    }
    save_json(compact, out_dir / f"{prefix}_summary.json")
    return compact


def make_policy_grid(args: argparse.Namespace) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Fast mode: only run the best policy observed in the previous sweep.

    Previous result 기준:
    - oracle 제외
    - moe + entropy_mean_probs + top2 + tau1.0 계열이 가장 안정적
    """
    return [
        ("best_moe_entropy_top2_tau1p0", {
            "policy": "moe",
            "score_type": "entropy_mean_probs",
            "top_k": 2,
            "tau": 1.0,
        })
    ]


def evaluate_all_policies(
    args: argparse.Namespace,
    models: Dict[int, nn.Module],
    test_df: pd.DataFrame,
    device: str,
    save_dir: Path,
    tta_aug_map: Dict[int, str],
) -> None:
    print("\n" + "=" * 100)
    print("[ROUTING EVALUATION]")
    print("=" * 100)

    class_name_map = get_class_names(test_df)
    policy_configs = make_policy_grid(args)

    all_results = {}
    summary_rows = []

    eval_groups = TARGET_DOMAINS + ["ALL"]

    for policy_name, cfg in policy_configs:
        print("\n" + "=" * 100)
        print(f"[POLICY] {policy_name}")
        print("=" * 100)

        policy_dir = save_dir / "routing" / policy_name
        ensure_dir(policy_dir)

        policy_result = {}

        for group in eval_groups:
            if group == "ALL":
                eval_df = test_df[test_df["domain"].isin(TARGET_DOMAINS)].copy().reset_index(drop=True)
            else:
                eval_df = test_df[test_df["domain"] == group].copy().reset_index(drop=True)

            print(f"[evaluate] policy={policy_name}, group={group}, n={len(eval_df)}")

            result = evaluate_routing_policy(
                models=models,
                df=eval_df,
                data_root=args.data_root,
                device=device,
                policy=cfg["policy"],
                score_type=cfg["score_type"],
                lambda_consistency=args.lambda_consistency,
                top_k=cfg["top_k"],
                tau=cfg["tau"],
                tta_aug_map=tta_aug_map,
                seed=args.seed + len(policy_name),
                max_eval_samples=args.max_eval_samples,
            )

            compact = save_eval_outputs(
                result=result,
                out_dir=policy_dir,
                prefix=group,
                class_name_map=class_name_map,
            )
            policy_result[group] = compact

        avg = round(float(np.mean([policy_result[d]["acc"] for d in TARGET_DOMAINS])), 2)
        avg_router = round(float(np.nanmean([policy_result[d]["router_acc"] for d in TARGET_DOMAINS])), 2)

        policy_result["Avg_D2D3"] = avg
        policy_result["Avg_router_acc_D2D3"] = avg_router
        all_results[policy_name] = policy_result

        row = {
            "policy": policy_name,
            "Avg_D2D3": avg,
            "Avg_router_acc_D2D3": avg_router,
            "ALL_acc": policy_result["ALL"]["acc"],
            "ALL_router_acc": policy_result["ALL"]["router_acc"],
            "score_type": cfg["score_type"],
            "top_k": cfg["top_k"],
            "tau": cfg["tau"],
            "lambda_consistency": args.lambda_consistency,
            "tta_aug_D2": tta_aug_map[1],
            "tta_aug_D3": tta_aug_map[2],
        }

        for d in TARGET_DOMAINS:
            row[f"{d}_acc"] = policy_result[d]["acc"]
            row[f"{d}_router_acc"] = policy_result[d]["router_acc"]
            for route_domain, count in policy_result[d]["route_hist"].items():
                row[f"{d}_route_{route_domain}"] = count

        summary_rows.append(row)
        print(f"[RESULT] {policy_name} | Avg_D2D3={avg} | Avg_router_acc={avg_router}")

    save_json(all_results, save_dir / "routing" / "all_results.json")

    summary_df = pd.DataFrame(summary_rows)
    summary_by_avg = summary_df.sort_values("Avg_D2D3", ascending=False)
    summary_by_router = summary_df.sort_values("Avg_router_acc_D2D3", ascending=False)

    summary_by_avg.to_csv(save_dir / "routing" / "summary_by_avg.csv", index=False, encoding="utf-8-sig")
    summary_by_router.to_csv(save_dir / "routing" / "summary_by_router_acc.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print("[SUMMARY BY AVG]")
    print(summary_by_avg.to_string(index=False))

    print("\n[SUMMARY BY ROUTER ACC]")
    print(summary_by_router.to_string(index=False))


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    device = "cuda" if (args.cuda and torch.cuda.is_available()) else "cpu"
    print(f"[device] {device}")

    save_dir = Path(args.save_dir)
    ensure_dir(save_dir)

    train_df = load_split_df(args.data_root, "development_train.txt")
    test_df = load_split_df(args.data_root, "development_test.txt")

    print(f"[data_root] {args.data_root}")
    print(f"[split] train={len(train_df)}, test={len(test_df)}")
    for d in TARGET_DOMAINS:
        print(
            f"[data] {d} "
            f"train={len(train_df[train_df['domain'] == d])}, "
            f"test={len(test_df[test_df['domain'] == d])}"
        )

    # train_aug_map은 D2/D3 학습용이다. D1은 학습하지 않는다.
    train_aug_map = parse_domain_aug_map(
        args.domain_train_aug,
        default_aug=args.default_train_aug,
        include_d1=False,
    )

    # tta_aug_map은 routing 후보용이다. include_d1_router=True이면 D1도 포함한다.
    tta_aug_map = parse_domain_aug_map(
        args.domain_tta_aug,
        default_aug=args.default_tta_aug,
        include_d1=args.include_d1_router,
    )

    config = vars(args).copy()
    config["train_aug_map"] = {TASK_TO_DOMAIN[k]: v for k, v in train_aug_map.items()}
    config["tta_aug_map"] = {TASK_TO_DOMAIN[k]: v for k, v in tta_aug_map.items()}
    save_json(config, save_dir / "config.json")

    print("[train_aug_map]", config["train_aug_map"])
    print("[tta_aug_map]", config["tta_aug_map"])

    models, train_summaries = train_or_load_experts(
        args=args,
        train_df=train_df,
        test_df=test_df,
        device=device,
        save_dir=save_dir,
        train_aug_map=train_aug_map,
    )
    save_json(train_summaries, save_dir / "train_summaries.json")

    evaluate_all_policies(
        args=args,
        models=models,
        test_df=test_df,
        device=device,
        save_dir=save_dir,
        tta_aug_map=tta_aug_map,
    )

    print("\n[DONE]")
    print(f"Saved to: {save_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()

    # paths
    parser.add_argument("--data_root", type=str, default="./data/task7_data")
    parser.add_argument("--d1_checkpoint", type=str, default="./checkpoints/BN/checkpoint_D1.pth")
    parser.add_argument("--d2_checkpoint", type=str, default="")
    parser.add_argument("--d3_checkpoint", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="./runs/domain_aug_tta_routing")

    # train
    parser.add_argument("--skip_train", action="store_true", help="D2/D3 checkpoint를 불러와 routing만 평가한다.")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--batch_size_eval", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--max_train_batches", type=int, default=-1, help="Debug/smoke-test limit for train batches per epoch.")

    # augmentation
    parser.add_argument(
        "--default_train_aug",
        type=str,
        default="identity",
        choices=["none", "identity", "light", "device", "noise", "gain_shift", "strong", "crop_gain_shift_noise", "device_light"],
        help="domain_train_aug가 비어있을 때 사용할 학습 augmentation.",
    )
    parser.add_argument(
        "--domain_train_aug",
        type=str,
        default="D2:device,D3:noise",
        help='도메인별 학습 augmentation. 예: "D2:device,D3:noise"',
    )
    parser.add_argument(
        "--default_tta_aug",
        type=str,
        default="identity",
        choices=["none", "identity", "light", "device", "noise", "gain_shift", "strong", "crop_gain_shift_noise", "device_light"],
        help="domain_tta_aug가 비어있을 때 사용할 TTA augmentation.",
    )
    parser.add_argument(
        "--domain_tta_aug",
        type=str,
        default="D2:device,D3:noise",
        help='도메인별 inference TTA augmentation. 예: "D2:device,D3:noise"',
    )

    # routing
    parser.add_argument(
        "--score_types",
        type=str,
        nargs="+",
        default=["entropy_mean_probs", "mean_entropy", "consistency"],
        choices=[
            "entropy_mean_probs",
            "mean_entropy",
            "consistency",
            "kl_only",
            "neg_confidence",
            "neg_margin",
        ],
    )
    parser.add_argument("--lambda_consistency", type=float, default=1.0)
    parser.add_argument("--top_ks", type=int, nargs="+", default=[2])
    parser.add_argument("--taus", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--max_eval_samples", type=int, default=-1, help="debug용. -1이면 전체 평가.")
    parser.add_argument(
        "--include_d1_router",
        action="store_true",
        help="D1 checkpoint를 routing 후보 expert로 포함한다. D1은 추가 학습하지 않는다.",
    )

    # misc
    parser.add_argument("--seed", type=int, default=1193)
    parser.add_argument("--cuda", action="store_true", default=False)

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
