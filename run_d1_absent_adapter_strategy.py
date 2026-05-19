#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D1 test가 없는 조건에서 수행하는 D2/D3 adapter ablation + routing study.

핵심 조건
- D1 train/test data는 prototype, calibration, router 학습/평가에 사용하지 않는다.
- D1 checkpoint는 frozen anchor/fallback 후보로만 사용한다.
- 실제 평가는 기존 실험처럼 D2/D3 test만 수행한다.
- main result는 D2/D3 hybrid MoE이며, D1 anchor/fallback은 ablation으로 기록한다.

예상 실행 위치
- baseline 코드(domain_net.py)가 있는 폴더 안에서 실행.
"""

import os
import sys
import json
import copy
import random
import argparse
from pathlib import Path
from itertools import chain
from typing import Dict, List, Tuple, Optional, Any

import librosa
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn import metrics

sys.path.insert(0, os.path.dirname(__file__))
from domain_net import MCnn14


# =========================================================
# Fixed setup
# =========================================================
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

ANCHOR_TASK = 0
EXPERT_TASKS = [1, 2]
FORWARD_TASKS = [0, 1, 2]
DEFAULT_EVAL_DOMAINS = ["D2", "D3"]


# =========================================================
# Utils
# =========================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj: Dict, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_split_df(data_root: str, split_name: str) -> pd.DataFrame:
    split_path = Path(data_root) / "evaluation_setup" / split_name
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    return pd.read_csv(
        split_path,
        sep="\t",
        header=None,
        names=["filename", "target", "domain", "new_target"],
    )


def pad_truncate_sequence(x: np.ndarray, max_len: int = CLIP_SAMPLES) -> np.ndarray:
    if len(x) < max_len:
        return np.concatenate((x, np.zeros(max_len - len(x), dtype=x.dtype)))
    return x[:max_len]


def split_into_chunks(x: np.ndarray, chunk_size: int = CLIP_SAMPLES) -> List[np.ndarray]:
    if len(x) <= chunk_size:
        return [pad_truncate_sequence(x, chunk_size)]
    chunks = []
    start = 0
    while start < len(x):
        chunks.append(pad_truncate_sequence(x[start:start + chunk_size], chunk_size))
        start += chunk_size
    return chunks


def entropy_np(p: np.ndarray, eps: float = 1e-12) -> float:
    return float(-(p * np.log(p + eps)).sum())


def entropy_from_probs(probs: torch.Tensor) -> torch.Tensor:
    eps = 1e-12
    return -(probs * torch.log(probs + eps)).sum(dim=-1)


def maxprob_np(p: np.ndarray) -> float:
    return float(np.max(p))


def margin_np(p: np.ndarray) -> float:
    top2 = np.sort(p)[-2:]
    return float(top2[-1] - top2[-2])


def energy_np(logits: np.ndarray, temperature: float = 1.0) -> float:
    z = logits / max(temperature, 1e-6)
    m = np.max(z)
    return float(-temperature * (m + np.log(np.exp(z - m).sum())))


def l2_normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / (np.linalg.norm(x) + eps)


def cosine_distance_np(a: np.ndarray, b: np.ndarray) -> float:
    a = l2_normalize_np(a)
    b = l2_normalize_np(b)
    return float(1.0 - np.sum(a * b))


def diag_mahalanobis_np(x: np.ndarray, mu: np.ndarray, var: np.ndarray) -> float:
    z = l2_normalize_np(x)
    return float(np.mean(((z - mu) ** 2) / var))


def probs_from_logits_np(logits: np.ndarray) -> np.ndarray:
    z = logits.astype(np.float64)
    z = z - np.max(z)
    e = np.exp(z)
    return (e / np.sum(e)).astype(np.float32)


def softmax_np(scores: np.ndarray) -> np.ndarray:
    scores = scores - np.max(scores)
    exp = np.exp(scores)
    return exp / np.sum(exp)


def copy_bn_state(src_bn: nn.BatchNorm2d, dst_bn: nn.BatchNorm2d) -> None:
    dst_bn.weight.data.copy_(src_bn.weight.data)
    dst_bn.bias.data.copy_(src_bn.bias.data)
    dst_bn.running_mean.data.copy_(src_bn.running_mean.data)
    dst_bn.running_var.data.copy_(src_bn.running_var.data)
    if hasattr(src_bn, "num_batches_tracked") and hasattr(dst_bn, "num_batches_tracked"):
        dst_bn.num_batches_tracked.data.copy_(src_bn.num_batches_tracked.data)


def get_class_names(df: pd.DataFrame) -> Dict[int, str]:
    mapping = {}
    for _, row in df.iterrows():
        mapping[int(row["new_target"])] = str(row["target"])
    return mapping


# =========================================================
# Adapter experiment map
# =========================================================
def make_contiguous_adapter_experiments() -> Dict[int, List[int]]:
    """21개 contiguous adapter setting.
    id 1~6: single block
    id 7~11: length-2 contiguous
    id 12~15: length-3 contiguous, id15=[4,5,6]
    id 16~18: length-4
    id 19~20: length-5
    id 21: all [1,2,3,4,5,6]
    """
    exp_map = {}
    idx = 1
    for length in range(1, 7):
        for start in range(1, 7 - length + 1):
            exp_map[idx] = list(range(start, start + length))
            idx += 1
    return exp_map


def resolve_exp_ids(adapter_mode: str, exp_ids: List[int]) -> Dict[int, List[int]]:
    all_map = make_contiguous_adapter_experiments()
    if adapter_mode == "all":
        selected = exp_ids if exp_ids else list(all_map.keys())
        return {eid: all_map[eid] for eid in selected if eid in all_map}
    if adapter_mode == "late456":
        return {15: [4, 5, 6]}
    if adapter_mode == "single":
        selected = exp_ids if exp_ids else list(range(1, 7))
        return {eid: all_map[eid] for eid in selected if eid in all_map and len(all_map[eid]) == 1}
    if adapter_mode == "custom":
        if not exp_ids:
            raise ValueError("adapter_mode=custom requires --custom_blocks or --exp_ids")
        return {eid: all_map[eid] for eid in exp_ids if eid in all_map}
    raise ValueError(f"Unknown adapter_mode: {adapter_mode}")


# =========================================================
# Dataset
# =========================================================
class Task7AugDataset(Dataset):
    def __init__(self, df: pd.DataFrame, data_root: str, train: bool = False, aug_mode: str = "none"):
        self.df = df.reset_index(drop=True)
        self.data_root = Path(data_root)
        self.train = train
        self.aug_mode = aug_mode

    def __len__(self) -> int:
        return len(self.df)

    def _crop_or_pad(self, audio: np.ndarray) -> np.ndarray:
        if len(audio) <= CLIP_SAMPLES:
            return pad_truncate_sequence(audio, CLIP_SAMPLES).astype(np.float32)
        if self.train and ("crop" in self.aug_mode):
            start = np.random.randint(0, len(audio) - CLIP_SAMPLES + 1)
            return audio[start:start + CLIP_SAMPLES].astype(np.float32)
        return audio[:CLIP_SAMPLES].astype(np.float32)

    def _random_gain(self, audio: np.ndarray) -> np.ndarray:
        gain_db = np.random.uniform(-4.0, 4.0)
        return audio * (10.0 ** (gain_db / 20.0))

    def _time_shift(self, audio: np.ndarray) -> np.ndarray:
        max_shift = int(0.15 * SAMPLE_RATE)
        shift = np.random.randint(-max_shift, max_shift + 1)
        return np.roll(audio, shift)

    def _small_noise(self, audio: np.ndarray) -> np.ndarray:
        std = np.random.uniform(0.0002, 0.0010)
        return audio + np.random.randn(len(audio)).astype(np.float32) * std

    def _augment_audio(self, audio: np.ndarray) -> np.ndarray:
        if self.aug_mode == "none":
            return audio.astype(np.float32)
        if self.aug_mode in ["gain_shift", "crop_gain_shift", "crop_gain_shift_noise"]:
            if np.random.rand() < 0.5:
                audio = self._random_gain(audio)
            if np.random.rand() < 0.5:
                audio = self._time_shift(audio)
        if self.aug_mode in ["noise", "crop_gain_shift_noise"]:
            if np.random.rand() < 0.3:
                audio = self._small_noise(audio)
        return audio.astype(np.float32)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        wav_path = self.data_root / row["filename"]
        if not wav_path.exists():
            raise FileNotFoundError(f"Audio file not found: {wav_path}")
        audio, _ = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
        audio = self._crop_or_pad(audio.astype(np.float32))
        if self.train:
            audio = self._augment_audio(audio)
        label = int(row["new_target"])
        return torch.from_numpy(audio.astype(np.float32)), torch.tensor(label, dtype=torch.long), row["filename"]


# =========================================================
# Model with adapters
# =========================================================
class Adapter2d(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.down = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.up = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.0))
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.xavier_uniform_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.alpha * self.up(self.act(self.down(x)))


class MCnn14Adapter(MCnn14):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num, nb_tasks=3, adapter_reduction=16):
        super().__init__(sample_rate=sample_rate, window_size=window_size, hop_size=hop_size,
                         mel_bins=mel_bins, fmin=fmin, fmax=fmax, classes_num=classes_num, nb_tasks=nb_tasks)
        self.block1_adapters = nn.ModuleList([Adapter2d(64, adapter_reduction) for _ in range(nb_tasks)])
        self.block2_adapters = nn.ModuleList([Adapter2d(128, adapter_reduction) for _ in range(nb_tasks)])
        self.block3_adapters = nn.ModuleList([Adapter2d(256, adapter_reduction) for _ in range(nb_tasks)])
        self.block4_adapters = nn.ModuleList([Adapter2d(512, adapter_reduction) for _ in range(nb_tasks)])
        self.block5_adapters = nn.ModuleList([Adapter2d(1024, adapter_reduction) for _ in range(nb_tasks)])
        self.block6_adapters = nn.ModuleList([Adapter2d(2048, adapter_reduction) for _ in range(nb_tasks)])
        self._active_adapter_blocks = set()

    def set_active_adapters(self, blocks: List[int]) -> None:
        self._active_adapter_blocks = set(blocks)

    def _apply_adapter(self, x: torch.Tensor, task: int, block_idx: int) -> torch.Tensor:
        if block_idx not in self._active_adapter_blocks:
            return x
        return getattr(self, f"block{block_idx}_adapters")[task](x)

    def _forward_block(self, block: nn.Module, x: torch.Tensor, task: int, block_idx: int) -> torch.Tensor:
        x = F.relu_(block.bnF[task](block.conv1(x)))
        x = F.relu_(block.bnS[task](block.conv2(x)))
        x = self._apply_adapter(x, task, block_idx)
        x = F.avg_pool2d(x, kernel_size=(2, 2))
        return x

    def forward_features(self, input: torch.Tensor, task: int = 1) -> torch.Tensor:
        x = self.spectrogram_extractor(input)
        x = self.logmel_extractor(x)
        x = x.transpose(1, 3)
        x = self.bn0[task](x)
        x = x.transpose(1, 3)
        x = self._forward_block(self.conv_block1, x, task, 1)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self._forward_block(self.conv_block2, x, task, 2)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self._forward_block(self.conv_block3, x, task, 3)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self._forward_block(self.conv_block4, x, task, 4)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self._forward_block(self.conv_block5, x, task, 5)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self._forward_block(self.conv_block6, x, task, 6)
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        x1, _ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        return x1 + x2

    def forward(self, input: torch.Tensor, task: int = 1) -> torch.Tensor:
        feat = self.forward_features(input, task=task)
        return self.fc(feat)


def build_model(adapter_reduction: int, active_blocks: List[int], device: str) -> MCnn14Adapter:
    model = MCnn14Adapter(SAMPLE_RATE, WINDOW_SIZE, HOP_SIZE, MEL_BINS, FMIN, FMAX,
                          CLASSES_NUM, NB_TASKS, adapter_reduction).to(device)
    model.set_active_adapters(active_blocks)
    return model


def load_checkpoint(model: nn.Module, ckpt_path: str, device: str, strict: bool = False) -> None:
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=strict)
    print(f"[load] {ckpt_path}")
    print(f"[load] missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print("[missing example]", missing[:10])
    if unexpected:
        print("[unexpected example]", unexpected[:10])


def freeze_all_params(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def get_adapter_module(model: MCnn14Adapter, task_id: int, block_idx: int) -> nn.Module:
    return getattr(model, f"block{block_idx}_adapters")[task_id]


def configure_task_adapter_trainable(model: MCnn14Adapter, task_id: int, adapter_blocks: List[int]) -> List[nn.Parameter]:
    freeze_all_params(model)
    params = []
    for block_idx in adapter_blocks:
        module = get_adapter_module(model, task_id, block_idx)
        for p in module.parameters():
            p.requires_grad = True
            params.append(p)
    return params


def initialize_task_bn_from_d1(model: MCnn14Adapter, task_id: int) -> None:
    copy_bn_state(model.bn0[0], model.bn0[task_id])
    for block in [model.conv_block1, model.conv_block2, model.conv_block3, model.conv_block4, model.conv_block5, model.conv_block6]:
        copy_bn_state(block.bnF[0], block.bnF[task_id])
        copy_bn_state(block.bnS[0], block.bnS[task_id])


def set_bn_eval(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()


def clone_teacher(model: nn.Module) -> nn.Module:
    teacher = copy.deepcopy(model)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =========================================================
# Training
# =========================================================
def kd_kl_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    T = temperature
    log_p_student = F.log_softmax(student_logits / T, dim=-1)
    p_teacher = F.softmax(teacher_logits / T, dim=-1)
    return F.kl_div(log_p_student, p_teacher, reduction="batchmean") * (T * T)


def train_one_epoch_adapter(
    model: MCnn14Adapter,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str,
    task_id: int,
    active_blocks: List[int],
    freeze_bn_stats: bool,
    teacher_model: Optional[MCnn14Adapter] = None,
    teacher_task_id: Optional[int] = None,
    lwf_lambda: float = 0.0,
    kd_temperature: float = 2.0,
) -> Tuple[float, float, float, float]:
    model.train()
    model.set_active_adapters(active_blocks)
    if freeze_bn_stats:
        set_bn_eval(model)
    if teacher_model is not None:
        teacher_model.eval()
        teacher_model.set_active_adapters(active_blocks)

    total_loss, total_ce, total_kd, total, correct = 0.0, 0.0, 0.0, 0, 0
    for audio, target, _ in loader:
        audio = audio.float().to(device)
        target = target.long().to(device)
        optimizer.zero_grad()
        logits = model(audio, task=task_id)
        ce_loss = criterion(logits, target)
        pred = logits.argmax(dim=-1)
        correct += int((pred == target).sum().item())
        kd_loss = torch.tensor(0.0, device=device)
        if teacher_model is not None and teacher_task_id is not None and lwf_lambda > 0:
            with torch.no_grad():
                teacher_logits = teacher_model(audio, task=teacher_task_id)
            kd_loss = kd_kl_loss(logits, teacher_logits, kd_temperature)
        loss = ce_loss + lwf_lambda * kd_loss
        loss.backward()
        optimizer.step()
        total += int(target.size(0))
        total_loss += float(loss.item())
        total_ce += float(ce_loss.item())
        total_kd += float(kd_loss.item())

    n = max(len(loader), 1)
    return round(total_loss / n, 6), round(total_ce / n, 6), round(total_kd / n, 6), round(correct / max(total, 1) * 100.0, 2)


@torch.no_grad()
def evaluate_fixed_task(model: MCnn14Adapter, df: pd.DataFrame, data_root: str, device: str, task_id: int, active_blocks: List[int]) -> float:
    model.eval()
    model.set_active_adapters(active_blocks)
    y_true, y_pred = [], []
    for _, row in df.reset_index(drop=True).iterrows():
        wav_path = Path(data_root) / row["filename"]
        target = int(row["new_target"])
        audio, _ = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
        chunks = split_into_chunks(audio.astype(np.float32), CLIP_SAMPLES)
        chunk_logits = []
        for chunk in chunks:
            x = torch.from_numpy(chunk.astype(np.float32)).unsqueeze(0).to(device)
            chunk_logits.append(model(x, task=task_id))
        logits = torch.stack(chunk_logits, dim=0).mean(dim=0)
        y_true.append(target)
        y_pred.append(int(logits.argmax(dim=-1).item()))
    if len(y_true) == 0:
        return float("nan")
    return round(float((np.array(y_true) == np.array(y_pred)).mean() * 100.0), 2)


def train_adapter_for_domain(
    model: MCnn14Adapter,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    args: argparse.Namespace,
    device: str,
    domain: str,
    out_dir: Path,
    active_blocks: List[int],
    teacher_model: Optional[MCnn14Adapter] = None,
    teacher_task_id: Optional[int] = None,
) -> Dict[str, Any]:
    ensure_dir(out_dir)
    task_id = DOMAIN_TO_TASK[domain]
    domain_train = train_df[train_df["domain"] == domain].copy().reset_index(drop=True)
    domain_test = test_df[test_df["domain"] == domain].copy().reset_index(drop=True)
    print("\n" + "=" * 100)
    print(f"[TRAIN] {domain} adapter | blocks={active_blocks} | n_train={len(domain_train)} | n_test={len(domain_test)}")
    print("=" * 100)

    if args.init_bn_from_d1:
        initialize_task_bn_from_d1(model, task_id)

    params = configure_task_adapter_trainable(model, task_id, active_blocks)
    print(f"[trainable params] {count_trainable_params(model)}")

    dataset = Task7AugDataset(domain_train, args.data_root, train=True, aug_mode=args.aug_mode)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                        pin_memory=True, drop_last=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params, lr=args.lr_incremental, betas=(0.9, 0.999), eps=1e-8,
                           weight_decay=args.weight_decay, amsgrad=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)

    best_metric, best_epoch, best_state = -1.0, -1, None
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, ce_loss, kd_loss, train_acc = train_one_epoch_adapter(
            model=model, loader=loader, optimizer=optimizer, criterion=criterion, device=device,
            task_id=task_id, active_blocks=active_blocks, freeze_bn_stats=args.freeze_bn_stats,
            teacher_model=teacher_model, teacher_task_id=teacher_task_id,
            lwf_lambda=args.lwf_lambda, kd_temperature=args.kd_temperature,
        )
        fixed_acc = evaluate_fixed_task(model, domain_test, args.data_root, device, task_id, active_blocks)
        scheduler.step()
        row = {"epoch": epoch, "domain": domain, "train_loss": train_loss, "ce_loss": ce_loss,
               "kd_loss": kd_loss, "train_acc": train_acc, f"{domain}_fixed_acc": fixed_acc,
               "lr": optimizer.param_groups[0]["lr"]}
        history.append(row)
        print(row)
        if fixed_acc > best_metric:
            best_metric = fixed_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, out_dir / f"best_{domain}.pth")
    pd.DataFrame(history).to_csv(out_dir / f"history_{domain}.csv", index=False, encoding="utf-8-sig")
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
    after_path = out_dir / f"after_{domain}.pth"
    torch.save(model.state_dict(), after_path)
    result = {"domain": domain, "task_id": task_id, "best_epoch": best_epoch, "best_metric": best_metric,
              "checkpoint": str(after_path)}
    save_json(result, out_dir / f"summary_{domain}.json")
    return result


# =========================================================
# Forward cache and descriptors
# =========================================================
@torch.no_grad()
def forward_one_audio_all_tasks(model: MCnn14Adapter, wav_path: Path, device: str, active_blocks: List[int], tasks: List[int] = FORWARD_TASKS) -> Dict[str, Any]:
    audio, _ = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
    chunks = split_into_chunks(audio.astype(np.float32), CLIP_SAMPLES)
    model.eval()
    model.set_active_adapters(active_blocks)
    out = {"logits": {}, "probs": {}, "features": {}, "entropy": {}, "maxprob": {}, "margin": {}, "energy": {}, "pred": {}}
    for task_id in tasks:
        chunk_logits, chunk_feats = [], []
        for chunk in chunks:
            x = torch.from_numpy(chunk.astype(np.float32)).unsqueeze(0).to(device)
            feat = model.forward_features(x, task=task_id)
            logits = model.fc(feat)
            chunk_feats.append(feat)
            chunk_logits.append(logits)
        logits = torch.stack(chunk_logits, dim=0).mean(dim=0).squeeze(0)
        feat = torch.stack(chunk_feats, dim=0).mean(dim=0).squeeze(0)
        probs = torch.softmax(logits, dim=-1)
        z = logits.detach().cpu().numpy().astype(np.float32)
        p = probs.detach().cpu().numpy().astype(np.float32)
        f = feat.detach().cpu().numpy().astype(np.float32)
        out["logits"][task_id] = z
        out["probs"][task_id] = p
        out["features"][task_id] = f
        out["entropy"][task_id] = entropy_np(p)
        out["maxprob"][task_id] = maxprob_np(p)
        out["margin"][task_id] = margin_np(p)
        out["energy"][task_id] = energy_np(z)
        out["pred"][task_id] = int(np.argmax(p))
    return out


def build_forward_cache(model: MCnn14Adapter, df: pd.DataFrame, data_root: str, device: str, active_blocks: List[int], cache_name: str) -> List[Dict[str, Any]]:
    rows = []
    for idx, row in df.reset_index(drop=True).iterrows():
        wav_path = Path(data_root) / row["filename"]
        outputs = forward_one_audio_all_tasks(model, wav_path, device, active_blocks, FORWARD_TASKS)
        rows.append({"filename": row["filename"], "domain": row["domain"], "target_name": row["target"],
                     "target": int(row["new_target"]), "true_task": DOMAIN_TO_TASK[str(row["domain"])],
                     "outputs": outputs})
        if (idx + 1) % 100 == 0:
            print(f"[{cache_name}] {idx + 1}/{len(df)}")
    return rows


def fit_temperature_for_logits(logits: torch.Tensor, labels: torch.Tensor, device: str) -> float:
    logits = logits.to(device)
    labels = labels.to(device)
    log_T = torch.zeros(1, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_T], lr=0.05, max_iter=100)
    def closure():
        optimizer.zero_grad()
        T = torch.exp(log_T).clamp(min=0.05, max=10.0)
        loss = F.cross_entropy(logits / T, labels)
        loss.backward()
        return loss
    optimizer.step(closure)
    T = float(torch.exp(log_T).detach().cpu().item())
    return max(min(T, 10.0), 0.05)


def fit_temperatures_from_cache(cache: List[Dict[str, Any]], device: str, tasks: List[int] = EXPERT_TASKS) -> Dict[int, float]:
    temperatures = {}
    for task_id in tasks:
        domain = TASK_TO_DOMAIN[task_id]
        items = [x for x in cache if x["domain"] == domain]
        logits = torch.tensor(np.stack([x["outputs"]["logits"][task_id] for x in items]), dtype=torch.float32)
        labels = torch.tensor([x["target"] for x in items], dtype=torch.long)
        T = fit_temperature_for_logits(logits, labels, device)
        temperatures[task_id] = T
        print(f"[temperature] {domain}: T={T:.4f}, n={len(items)}")
    return temperatures


def add_calibrated_outputs(cache: List[Dict[str, Any]], temperatures: Dict[int, float], tasks: List[int] = EXPERT_TASKS) -> None:
    for item in cache:
        out = item["outputs"]
        out["cal_probs"] = {}
        out["cal_entropy"] = {}
        for task_id in tasks:
            logits = out["logits"][task_id]
            T = temperatures.get(task_id, 1.0)
            p = probs_from_logits_np(logits / max(T, 1e-6))
            out["cal_probs"][task_id] = p
            out["cal_entropy"][task_id] = entropy_np(p)


def build_prototypes_and_descriptors(cache: List[Dict[str, Any]], tasks: List[int] = EXPERT_TASKS, eps: float = 1e-5) -> Dict[str, Any]:
    domain_proto, domain_desc = {}, {}
    class_proto, class_desc = {t: {} for t in tasks}, {t: {} for t in tasks}
    for task_id in tasks:
        domain = TASK_TO_DOMAIN[task_id]
        items = [x for x in cache if x["domain"] == domain]
        feats = np.stack([l2_normalize_np(x["outputs"]["features"][task_id]) for x in items])
        mu = feats.mean(axis=0)
        var = feats.var(axis=0) + eps
        domain_proto[task_id] = l2_normalize_np(mu).astype(np.float32)
        domain_desc[task_id] = {"mu": mu.astype(np.float32), "var": var.astype(np.float32)}
        for c in range(CLASSES_NUM):
            c_items = [x for x in items if x["target"] == c]
            if not c_items:
                continue
            c_feats = np.stack([l2_normalize_np(x["outputs"]["features"][task_id]) for x in c_items])
            c_mu = c_feats.mean(axis=0)
            c_var = c_feats.var(axis=0) + eps
            class_proto[task_id][c] = l2_normalize_np(c_mu).astype(np.float32)
            class_desc[task_id][c] = {"mu": c_mu.astype(np.float32), "var": c_var.astype(np.float32), "n": len(c_items)}
        print(f"[proto/desc] {domain}: n={len(items)}, class_count={len(class_proto[task_id])}")
    return {"domain_proto": domain_proto, "class_proto": class_proto, "domain_desc": domain_desc, "class_desc": class_desc}


def add_distance_features(cache: List[Dict[str, Any]], proto_desc: Dict[str, Any], tasks: List[int] = EXPERT_TASKS) -> None:
    for item in cache:
        out = item["outputs"]
        out["proto_domain_dist"] = {}
        out["proto_class_dist"] = {}
        out["maha_domain_dist"] = {}
        out["maha_class_dist"] = {}
        for task_id in tasks:
            feat = out["features"][task_id]
            out["proto_domain_dist"][task_id] = cosine_distance_np(feat, proto_desc["domain_proto"][task_id])
            min_cdist = 999.0
            for proto in proto_desc["class_proto"][task_id].values():
                min_cdist = min(min_cdist, cosine_distance_np(feat, proto))
            out["proto_class_dist"][task_id] = float(min_cdist)
            dd = proto_desc["domain_desc"][task_id]
            out["maha_domain_dist"][task_id] = diag_mahalanobis_np(feat, dd["mu"], dd["var"])
            min_mdist = 999.0
            for desc in proto_desc["class_desc"][task_id].values():
                min_mdist = min(min_mdist, diag_mahalanobis_np(feat, desc["mu"], desc["var"]))
            out["maha_class_dist"][task_id] = float(min_mdist)


def fit_score_stats(cache: List[Dict[str, Any]], score_keys: List[str], tasks: List[int] = EXPERT_TASKS) -> Dict[str, Dict[int, Dict[str, float]]]:
    stats = {}
    for key in score_keys:
        stats[key] = {}
        for task_id in tasks:
            values = np.array([x["outputs"][key][task_id] for x in cache], dtype=np.float32)
            stats[key][task_id] = {"mean": float(values.mean()), "std": float(values.std() + 1e-6)}
    return stats


def zscore_value(value: float, stats: Dict[str, Any], key: str, task_id: int) -> float:
    return float((value - stats[key][task_id]["mean"]) / stats[key][task_id]["std"])


# =========================================================
# Policy evaluation
# =========================================================
def weighted_probs(item: Dict[str, Any], weights_by_task: Dict[int, float], prob_key: str = "probs") -> np.ndarray:
    final = None
    for task_id, w in weights_by_task.items():
        p = item["outputs"][prob_key][task_id]
        final = w * p if final is None else final + w * p
    return final


def hard_probs(item: Dict[str, Any], task_id: int, prob_key: str = "probs") -> np.ndarray:
    return item["outputs"][prob_key][task_id]


def hybrid_expert_weights(item: Dict[str, Any], tau: float, d3_bias: float, alpha: float, beta: float, delta: float, score_stats: Dict[str, Any]) -> Tuple[Dict[int, float], Dict[int, float]]:
    out = item["outputs"]
    raw_scores = {}
    for t in EXPERT_TASKS:
        e = zscore_value(out["entropy"][t], score_stats, "entropy", t)
        pc = zscore_value(out["proto_class_dist"][t], score_stats, "proto_class_dist", t)
        mc = zscore_value(out["maha_class_dist"][t], score_stats, "maha_class_dist", t)
        raw_scores[t] = alpha * e + beta * pc + delta * mc
    raw_scores[2] = raw_scores[2] - d3_bias
    arr = np.array([raw_scores[t] for t in EXPERT_TASKS], dtype=np.float32)
    weights = softmax_np(-arr / max(tau, 1e-6))
    return {t: float(w) for t, w in zip(EXPERT_TASKS, weights)}, raw_scores


def anchor_residual_probs(item: Dict[str, Any], weights_by_task: Dict[int, float], residual_scale: float) -> np.ndarray:
    out = item["outputs"]
    z_anchor = out["logits"][ANCHOR_TASK]
    z_final = z_anchor.copy()
    for t, w in weights_by_task.items():
        z_final = z_final + residual_scale * w * (out["logits"][t] - z_anchor)
    return probs_from_logits_np(z_final)


def eval_cache_policy(
    cache: List[Dict[str, Any]],
    policy_name: str,
    policy_kind: str,
    tau: float = 1.0,
    d3_bias: float = 0.0,
    alpha: float = 1.0,
    beta: float = 0.0,
    delta: float = 0.0,
    residual_scale: float = 1.0,
    fallback_threshold: float = 0.70,
    fallback_lambda: float = 0.25,
    score_stats: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    y_true, y_pred, chosen_tasks = [], [], []
    rows = []
    for item in cache:
        out = item["outputs"]
        target = item["target"]
        true_task = item["true_task"]
        weights_by_task = None
        chosen_task = -1
        if policy_kind == "fixed_D1_anchor":
            chosen_task = 0
            final_probs = hard_probs(item, 0)
        elif policy_kind == "fixed_D2":
            chosen_task = 1
            final_probs = hard_probs(item, 1)
        elif policy_kind == "fixed_D3":
            chosen_task = 2
            final_probs = hard_probs(item, 2)
        elif policy_kind == "oracle":
            chosen_task = true_task
            final_probs = hard_probs(item, true_task)
        elif policy_kind == "all_mean":
            weights_by_task = {1: 0.5, 2: 0.5}
            final_probs = weighted_probs(item, weights_by_task)
        elif policy_kind == "all_mean_with_D1":
            weights_by_task = {0: 1/3, 1: 1/3, 2: 1/3}
            final_probs = weighted_probs(item, weights_by_task)
        elif policy_kind == "logit_mean":
            final_probs = probs_from_logits_np(np.stack([out["logits"][1], out["logits"][2]], axis=0).mean(axis=0))
        elif policy_kind == "logit_mean_with_D1":
            final_probs = probs_from_logits_np(np.stack([out["logits"][0], out["logits"][1], out["logits"][2]], axis=0).mean(axis=0))
        elif policy_kind == "entropy_hard":
            scores = {t: out["entropy"][t] for t in EXPERT_TASKS}
            chosen_task = min(EXPERT_TASKS, key=lambda t: scores[t])
            final_probs = hard_probs(item, chosen_task)
        elif policy_kind == "cal_entropy_hard":
            scores = {t: out["cal_entropy"][t] for t in EXPERT_TASKS}
            chosen_task = min(EXPERT_TASKS, key=lambda t: scores[t])
            final_probs = hard_probs(item, chosen_task, "cal_probs")
        elif policy_kind == "proto_class_hard":
            scores = {t: out["proto_class_dist"][t] for t in EXPERT_TASKS}
            chosen_task = min(EXPERT_TASKS, key=lambda t: scores[t])
            final_probs = hard_probs(item, chosen_task)
        elif policy_kind == "maha_class_hard":
            scores = {t: out["maha_class_dist"][t] for t in EXPERT_TASKS}
            chosen_task = min(EXPERT_TASKS, key=lambda t: scores[t])
            final_probs = hard_probs(item, chosen_task)
        elif policy_kind in ["entropy_moe", "proto_class_moe", "maha_class_moe"]:
            if policy_kind == "entropy_moe":
                scores = {t: out["entropy"][t] for t in EXPERT_TASKS}
            elif policy_kind == "proto_class_moe":
                scores = {t: out["proto_class_dist"][t] for t in EXPERT_TASKS}
            else:
                scores = {t: out["maha_class_dist"][t] for t in EXPERT_TASKS}
            arr = np.array([scores[t] for t in EXPERT_TASKS], dtype=np.float32)
            ws = softmax_np(-arr / max(tau, 1e-6))
            weights_by_task = {t: float(w) for t, w in zip(EXPERT_TASKS, ws)}
            chosen_task = EXPERT_TASKS[int(np.argmin(arr))]
            final_probs = weighted_probs(item, weights_by_task)
        elif policy_kind in ["hybrid_moe", "anchor_residual_fallback"]:
            if score_stats is None:
                raise ValueError(f"{policy_kind} requires score_stats")
            weights_by_task, raw_scores = hybrid_expert_weights(item, tau, d3_bias, alpha, beta, delta, score_stats)
            chosen_task = max(weights_by_task, key=lambda t: weights_by_task[t])
            if policy_kind == "hybrid_moe":
                final_probs = weighted_probs(item, weights_by_task)
            else:
                final_probs = anchor_residual_probs(item, weights_by_task, residual_scale)
                if max(weights_by_task.values()) < fallback_threshold:
                    d1_probs = hard_probs(item, 0)
                    final_probs = (1.0 - fallback_lambda) * final_probs + fallback_lambda * d1_probs
        else:
            raise ValueError(policy_kind)

        pred = int(np.argmax(final_probs))
        y_true.append(target)
        y_pred.append(pred)
        chosen_tasks.append(chosen_task)
        row = {"filename": item["filename"], "domain": item["domain"], "target": target, "pred": pred,
               "correct": int(pred == target), "true_task": true_task, "chosen_task": chosen_task,
               "chosen_domain": TASK_TO_DOMAIN.get(chosen_task, "MIX"),
               "route_correct": int(chosen_task == true_task) if chosen_task in TASK_TO_DOMAIN else np.nan,
               "confidence": float(np.max(final_probs))}
        for t in [0, 1, 2]:
            d = TASK_TO_DOMAIN[t]
            row[f"entropy_{d}"] = out["entropy"].get(t, np.nan)
            if weights_by_task is not None:
                row[f"weight_{d}"] = weights_by_task.get(t, np.nan)
        rows.append(row)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = round(float((y_true == y_pred).mean() * 100.0), 2) if len(y_true) else np.nan
    cm = metrics.confusion_matrix(y_true, y_pred, labels=list(range(CLASSES_NUM))) if len(y_true) else np.zeros((CLASSES_NUM, CLASSES_NUM), dtype=int)
    pred_df = pd.DataFrame(rows)
    valid_route = pred_df["route_correct"].dropna() if len(pred_df) else pd.Series(dtype=float)
    router_acc = round(float(valid_route.mean() * 100.0), 2) if len(valid_route) else np.nan
    chosen_arr = np.array(chosen_tasks)
    route_hist = {TASK_TO_DOMAIN[t]: int(np.sum(chosen_arr == t)) for t in [0, 1, 2]}
    route_hist["MIX"] = int(np.sum(chosen_arr == -1))
    return {"policy": policy_name, "acc": acc, "router_acc": router_acc, "route_hist": route_hist, "cm": cm}, pred_df


def classwise_accuracy(pred_df: pd.DataFrame, class_name_map: Dict[int, str]) -> pd.DataFrame:
    rows = []
    for c in range(CLASSES_NUM):
        sub = pred_df[pred_df["target"] == c]
        n = len(sub)
        correct = int(sub["correct"].sum()) if n > 0 else 0
        acc = round(correct / max(n, 1) * 100.0, 2) if n > 0 else np.nan
        rows.append({"class_id": c, "class_name": class_name_map.get(c, str(c)), "n_samples": n, "correct": correct, "accuracy": acc})
    return pd.DataFrame(rows)


def save_policy_result(result: Dict[str, Any], pred_df: pd.DataFrame, out_dir: Path, policy_name: str, domain: str, class_name_map: Dict[int, str]) -> Dict[str, Any]:
    pdir = out_dir / policy_name
    ensure_dir(pdir)
    pred_df.to_csv(pdir / f"{domain}_predictions.csv", index=False, encoding="utf-8-sig")
    classwise_accuracy(pred_df, class_name_map).to_csv(pdir / f"{domain}_classwise.csv", index=False, encoding="utf-8-sig")
    np.savetxt(pdir / f"{domain}_confusion.csv", result["cm"], delimiter=",", fmt="%d")
    compact = {"acc": result["acc"], "router_acc": result["router_acc"], "route_hist": result["route_hist"]}
    save_json(compact, pdir / f"{domain}_summary.json")
    return compact


def build_policy_grid(args: argparse.Namespace) -> List[Tuple[str, Dict[str, Any]]]:
    configs = [
        ("fixed_D1_anchor", {"policy_kind": "fixed_D1_anchor"}),
        ("fixed_D2", {"policy_kind": "fixed_D2"}),
        ("fixed_D3", {"policy_kind": "fixed_D3"}),
        ("oracle", {"policy_kind": "oracle"}),
        ("entropy_hard", {"policy_kind": "entropy_hard"}),
        ("cal_entropy_hard", {"policy_kind": "cal_entropy_hard"}),
        ("proto_class_hard", {"policy_kind": "proto_class_hard"}),
        ("maha_class_hard", {"policy_kind": "maha_class_hard"}),
        ("all_mean", {"policy_kind": "all_mean"}),
        ("logit_mean", {"policy_kind": "logit_mean"}),
        ("all_mean_with_D1", {"policy_kind": "all_mean_with_D1"}),
        ("logit_mean_with_D1", {"policy_kind": "logit_mean_with_D1"}),
    ]
    for tau in args.taus:
        tn = str(tau).replace(".", "p")
        configs.append((f"entropy_moe_tau{tn}", {"policy_kind": "entropy_moe", "tau": tau}))
        configs.append((f"proto_class_moe_tau{tn}", {"policy_kind": "proto_class_moe", "tau": tau}))
        configs.append((f"maha_class_moe_tau{tn}", {"policy_kind": "maha_class_moe", "tau": tau}))
    for a in args.hybrid_alphas:
        for b in args.hybrid_betas:
            for d in args.hybrid_deltas:
                for g in args.hybrid_gammas:
                    for t in args.hybrid_taus:
                        name = f"hybrid_moe_a{a}_b{b}_d{d}_g{g}_t{t}".replace(".", "p")
                        cfg = {"policy_kind": "hybrid_moe", "alpha": a, "beta": b, "delta": d, "d3_bias": g, "tau": t}
                        configs.append((name, cfg))
                        for rs in args.anchor_residual_scales:
                            for th in args.fallback_thresholds:
                                for fl in args.fallback_lambdas:
                                    fname = f"anchor_fallback_a{a}_b{b}_d{d}_g{g}_t{t}_rs{rs}_th{th}_fl{fl}".replace(".", "p")
                                    fcfg = dict(cfg)
                                    fcfg.update({"policy_kind": "anchor_residual_fallback", "residual_scale": rs,
                                                "fallback_threshold": th, "fallback_lambda": fl})
                                    configs.append((fname, fcfg))
    return configs


def run_policy_suite(model: MCnn14Adapter, train_df: pd.DataFrame, test_df: pd.DataFrame, args: argparse.Namespace,
                     device: str, active_blocks: List[int], out_dir: Path) -> pd.DataFrame:
    ensure_dir(out_dir)
    class_name_map = get_class_names(test_df)
    train_df = train_df[train_df["domain"].isin(DEFAULT_EVAL_DOMAINS)].copy().reset_index(drop=True)
    test_df = test_df[test_df["domain"].isin(DEFAULT_EVAL_DOMAINS)].copy().reset_index(drop=True)

    print("\n" + "=" * 100)
    print("[1] Build train cache: D2/D3 train only, D1 output only as anchor")
    print("=" * 100)
    train_cache = build_forward_cache(model, train_df, args.data_root, device, active_blocks, "train_cache")

    print("\n" + "=" * 100)
    print("[2] Temperature scaling: D2/D3 only")
    print("=" * 100)
    temperatures = fit_temperatures_from_cache(train_cache, device, EXPERT_TASKS)
    save_json({str(k): v for k, v in temperatures.items()}, out_dir / "temperatures.json")
    add_calibrated_outputs(train_cache, temperatures, EXPERT_TASKS)

    print("\n" + "=" * 100)
    print("[3] Prototype + Mahalanobis descriptors: D2/D3 only")
    print("=" * 100)
    proto_desc = build_prototypes_and_descriptors(train_cache, EXPERT_TASKS)
    add_distance_features(train_cache, proto_desc, EXPERT_TASKS)
    score_stats = fit_score_stats(train_cache, ["entropy", "proto_class_dist", "maha_class_dist"], EXPERT_TASKS)
    save_json(score_stats, out_dir / "score_stats.json")

    print("\n" + "=" * 100)
    print("[4] Build test cache: evaluate D2/D3 only")
    print("=" * 100)
    test_cache = build_forward_cache(model, test_df, args.data_root, device, active_blocks, "test_cache")
    add_calibrated_outputs(test_cache, temperatures, EXPERT_TASKS)
    add_distance_features(test_cache, proto_desc, EXPERT_TASKS)
    save_json({
        "train_cache_n": len(train_cache), "test_cache_n": len(test_cache),
        "D1_train_n_used": 0,
        "D1_test_n_evaluated": 0,
        "D2_train_n": int((train_df["domain"] == "D2").sum()),
        "D3_train_n": int((train_df["domain"] == "D3").sum()),
        "D2_test_n": int((test_df["domain"] == "D2").sum()),
        "D3_test_n": int((test_df["domain"] == "D3").sum()),
    }, out_dir / "cache_info.json")

    print("\n" + "=" * 100)
    print("[5] Evaluate policies")
    print("=" * 100)
    rows = []
    for policy_name, cfg in build_policy_grid(args):
        print("\n" + "-" * 100)
        print(f"[POLICY] {policy_name}")
        policy_result = {}
        for domain in DEFAULT_EVAL_DOMAINS:
            domain_cache = [x for x in test_cache if x["domain"] == domain]
            result, pred_df = eval_cache_policy(
                domain_cache, policy_name, cfg["policy_kind"],
                tau=cfg.get("tau", 1.0), d3_bias=cfg.get("d3_bias", 0.0),
                alpha=cfg.get("alpha", 1.0), beta=cfg.get("beta", 0.0), delta=cfg.get("delta", 0.0),
                residual_scale=cfg.get("residual_scale", 1.0),
                fallback_threshold=cfg.get("fallback_threshold", 0.70),
                fallback_lambda=cfg.get("fallback_lambda", 0.25),
                score_stats=score_stats,
            )
            compact = save_policy_result(result, pred_df, out_dir, policy_name, domain, class_name_map)
            policy_result[domain] = compact
            print(f"[{policy_name}] {domain}: acc={compact['acc']}, router_acc={compact['router_acc']}, routes={compact['route_hist']}")
        avg = round(np.mean([policy_result[d]["acc"] for d in DEFAULT_EVAL_DOMAINS]), 2)
        avg_router_acc = round(np.nanmean([policy_result[d]["router_acc"] for d in DEFAULT_EVAL_DOMAINS]), 2)
        row = {"policy": policy_name, "Avg": avg, "Avg_router_acc": avg_router_acc}
        for dname in DEFAULT_EVAL_DOMAINS:
            row[f"{dname}_acc"] = policy_result[dname]["acc"]
            row[f"{dname}_router_acc"] = policy_result[dname]["router_acc"]
            for rname in ["D1", "D2", "D3", "MIX"]:
                row[f"{dname}_route_{rname}"] = policy_result[dname]["route_hist"].get(rname, 0)
        rows.append(row)
        summary_df = pd.DataFrame(rows)
        summary_df.sort_values("Avg", ascending=False).to_csv(out_dir / "summary_by_avg_live.csv", index=False, encoding="utf-8-sig")
        summary_df.sort_values("D3_acc", ascending=False).to_csv(out_dir / "summary_by_d3_live.csv", index=False, encoding="utf-8-sig")
        print(f"[RESULT] {policy_name}: Avg={avg}, Avg_router_acc={avg_router_acc}")
    summary_df = pd.DataFrame(rows)
    summary_df.sort_values("Avg", ascending=False).to_csv(out_dir / "summary_by_avg.csv", index=False, encoding="utf-8-sig")
    summary_df.sort_values("D3_acc", ascending=False).to_csv(out_dir / "summary_by_d3.csv", index=False, encoding="utf-8-sig")
    summary_df.sort_values("Avg_router_acc", ascending=False).to_csv(out_dir / "summary_by_router_acc.csv", index=False, encoding="utf-8-sig")
    print("\n[SUMMARY BY AVG]")
    print(summary_df.sort_values("Avg", ascending=False).head(30).to_string(index=False))
    return summary_df


# =========================================================
# Experiment runner
# =========================================================
def run_one_experiment(args: argparse.Namespace, train_df: pd.DataFrame, test_df: pd.DataFrame, device: str,
                       exp_id: int, active_blocks: List[int]) -> Dict[str, Any]:
    exp_name = f"exp{exp_id:02d}_blocks_{''.join(map(str, active_blocks))}"
    out_dir = Path(args.save_dir) / exp_name
    ensure_dir(out_dir)
    print("\n" + "#" * 120)
    print(f"[EXPERIMENT] {exp_name} | blocks={active_blocks}")
    print("#" * 120)
    save_json({"exp_id": exp_id, "exp_name": exp_name, "adapter_blocks": active_blocks,
               "D1_data_policy": "D1 checkpoint only; no D1 train/prototype/calibration/test evaluation"}, out_dir / "config.json")

    model = build_model(args.adapter_reduction, active_blocks, device)
    load_checkpoint(model, args.d1_checkpoint, device, strict=False)

    teacher_d1 = clone_teacher(model) if args.lwf_lambda > 0 else None
    d2_result = train_adapter_for_domain(model, train_df, test_df, args, device, "D2", out_dir / "step1_train_D2",
                                         active_blocks, teacher_model=teacher_d1, teacher_task_id=0 if teacher_d1 else None)
    torch.save(model.state_dict(), out_dir / "after_D2.pth")
    del teacher_d1
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    teacher_after_d2 = clone_teacher(model) if args.lwf_lambda > 0 else None
    d3_result = train_adapter_for_domain(model, train_df, test_df, args, device, "D3", out_dir / "step2_train_D3",
                                         active_blocks, teacher_model=teacher_after_d2, teacher_task_id=1 if teacher_after_d2 else None)
    torch.save(model.state_dict(), out_dir / "after_D3.pth")
    torch.save(model.state_dict(), out_dir / "best.pth")
    del teacher_after_d2
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    save_json({"D2": d2_result, "D3": d3_result}, out_dir / "train_summary.json")

    summary_df = run_policy_suite(model, train_df, test_df, args, device, active_blocks, out_dir / "final_policy_eval")
    summary_df.insert(0, "exp_id", exp_id)
    summary_df.insert(1, "exp_name", exp_name)
    summary_df.insert(2, "adapter_blocks", "+".join(map(str, active_blocks)))
    summary_df.to_csv(out_dir / "compact_policy_summary.csv", index=False, encoding="utf-8-sig")
    best_row = summary_df.sort_values("Avg", ascending=False).iloc[0].to_dict()
    save_json({"best_by_avg": {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in best_row.items()}}, out_dir / "best_summary.json")
    return {"exp_id": exp_id, "exp_name": exp_name, "adapter_blocks": active_blocks, "summary_df": summary_df}


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = "cuda" if (args.cuda and torch.cuda.is_available()) else "cpu"
    print(f"[device] {device}")
    save_dir = Path(args.save_dir)
    ensure_dir(save_dir)

    train_df = load_split_df(args.data_root, "development_train.txt")
    test_df = load_split_df(args.data_root, "development_test.txt")
    print(f"[data] train total: {len(train_df)}")
    print(f"[data] test total : {len(test_df)}")
    for d in ["D1", "D2", "D3"]:
        print(f"[data] {d} train={len(train_df[train_df['domain'] == d])}, test={len(test_df[test_df['domain'] == d])}")

    exp_map = resolve_exp_ids(args.adapter_mode, args.exp_ids)
    if args.custom_blocks:
        exp_map = {999: args.custom_blocks}
    save_json({
        "data_root": args.data_root,
        "d1_checkpoint": args.d1_checkpoint,
        "save_dir": args.save_dir,
        "adapter_mode": args.adapter_mode,
        "selected_experiments": {str(k): v for k, v in exp_map.items()},
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr_incremental": args.lr_incremental,
        "aug_mode": args.aug_mode,
        "lwf_lambda": args.lwf_lambda,
        "D1_policy": "D1 checkpoint only; no D1 train/prototype/calibration/test evaluation",
    }, save_dir / "global_config.json")

    all_rows = []
    for exp_id, blocks in exp_map.items():
        result = run_one_experiment(args, train_df, test_df, device, exp_id, blocks)
        df = result["summary_df"].copy()
        all_rows.append(df)
        final_df = pd.concat(all_rows, ignore_index=True)
        final_df.sort_values("Avg", ascending=False).to_csv(save_dir / "summary_all_by_avg.csv", index=False, encoding="utf-8-sig")
        final_df.sort_values("D3_acc", ascending=False).to_csv(save_dir / "summary_all_by_d3.csv", index=False, encoding="utf-8-sig")
        final_df.sort_values("Avg_router_acc", ascending=False).to_csv(save_dir / "summary_all_by_router_acc.csv", index=False, encoding="utf-8-sig")

    if all_rows:
        final_df = pd.concat(all_rows, ignore_index=True)
        print("\n" + "=" * 120)
        print("[FINAL SUMMARY TOP-30 BY AVG]")
        print(final_df.sort_values("Avg", ascending=False).head(30).to_string(index=False))
        print("=" * 120)
    print(f"[DONE] Saved to: {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--d1_checkpoint", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)

    parser.add_argument("--adapter_mode", type=str, default="all", choices=["all", "single", "late456", "custom"])
    parser.add_argument("--exp_ids", type=int, nargs="*", default=[])
    parser.add_argument("--custom_blocks", type=int, nargs="*", default=[])

    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr_incremental", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adapter_reduction", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1193)

    parser.add_argument("--aug_mode", type=str, default="crop_gain_shift_noise", choices=["none", "crop", "gain_shift", "crop_gain_shift", "crop_gain_shift_noise", "noise"])
    parser.add_argument("--lwf_lambda", type=float, default=0.0)
    parser.add_argument("--kd_temperature", type=float, default=2.0)
    parser.add_argument("--freeze_bn_stats", action="store_true", default=False)
    parser.add_argument("--init_bn_from_d1", action="store_true", default=False)

    parser.add_argument("--taus", type=float, nargs="+", default=[0.7, 1.0, 1.3])
    parser.add_argument("--hybrid_alphas", type=float, nargs="+", default=[0.15, 0.20, 0.25])
    parser.add_argument("--hybrid_betas", type=float, nargs="+", default=[0.40, 0.50])
    parser.add_argument("--hybrid_deltas", type=float, nargs="+", default=[0.60, 0.75])
    parser.add_argument("--hybrid_gammas", type=float, nargs="+", default=[0.15, 0.20])
    parser.add_argument("--hybrid_taus", type=float, nargs="+", default=[0.7, 1.0, 1.3])
    parser.add_argument("--anchor_residual_scales", type=float, nargs="+", default=[1.0])
    parser.add_argument("--fallback_thresholds", type=float, nargs="+", default=[0.70])
    parser.add_argument("--fallback_lambdas", type=float, nargs="+", default=[0.25])

    parser.add_argument("--cuda", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
