import os
import sys
import time
import json
import math
import copy
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn import metrics

# baseline 폴더 안에서 실행한다고 가정
# baseline/domain_net.py의 MCnn14를 그대로 사용
sys.path.insert(0, os.path.dirname(__file__))
from domain_net import MCnn14


# =========================================================
# Fixed task setup (official baseline setting)
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
NB_TASKS = 3   # D1, D2, D3

DOMAIN_TO_TASK = {"D1": 0, "D2": 1, "D3": 2}
TASK_TO_DOMAIN = {0: "D1", 1: "D2", 2: "D3"}

EXPERIMENTS = {
    1: {"name": "baseline_all_bn",       "bn_blocks": [0, 1, 2, 3, 4, 5, 6], "adapter_blocks": []},
    2: {"name": "late_bn_56",            "bn_blocks": [5, 6],                 "adapter_blocks": []},
    3: {"name": "late_bn_456",           "bn_blocks": [4, 5, 6],              "adapter_blocks": []},
    4: {"name": "late_adapter_56",       "bn_blocks": [],                     "adapter_blocks": [5, 6]},
    5: {"name": "late_adapter_456",      "bn_blocks": [],                     "adapter_blocks": [4, 5, 6]},
    6: {"name": "late_bn_adapter_56",    "bn_blocks": [5, 6],                 "adapter_blocks": [5, 6]},
}


# =========================================================
# Utils
# =========================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj: Dict, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_split_df(data_root: str, split_name: str) -> pd.DataFrame:
    split_path = Path(data_root) / "evaluation_setup" / split_name
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
        chunk = x[start:start + chunk_size]
        chunks.append(pad_truncate_sequence(chunk, chunk_size))
        start += chunk_size
    return chunks


def entropy_from_probs(probs: torch.Tensor) -> torch.Tensor:
    eps = 1e-12
    return -(probs * torch.log(probs + eps)).sum(dim=-1)


def copy_bn_state(src_bn: nn.BatchNorm2d, dst_bn: nn.BatchNorm2d) -> None:
    dst_bn.weight.data.copy_(src_bn.weight.data)
    dst_bn.bias.data.copy_(src_bn.bias.data)
    dst_bn.running_mean.data.copy_(src_bn.running_mean.data)
    dst_bn.running_var.data.copy_(src_bn.running_var.data)
    if hasattr(src_bn, "num_batches_tracked") and hasattr(dst_bn, "num_batches_tracked"):
        dst_bn.num_batches_tracked.data.copy_(src_bn.num_batches_tracked.data)


# =========================================================
# Dataset
# =========================================================
class Task7TrainDataset(Dataset):
    def __init__(self, df: pd.DataFrame, data_root: str):
        self.df = df.reset_index(drop=True)
        self.data_root = Path(data_root)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        wav_path = self.data_root / row["filename"]
        audio, _ = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
        audio = pad_truncate_sequence(audio, CLIP_SAMPLES).astype(np.float32)

        label = int(row["new_target"])
        onehot = np.zeros(CLASSES_NUM, dtype=np.float32)
        onehot[label] = 1.0
        return torch.from_numpy(audio), torch.from_numpy(onehot), row["filename"]


# =========================================================
# Adapter modules
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


# =========================================================
# Baseline-compatible extension
# =========================================================
class MCnn14LateBNAdapter(MCnn14):
    """
    baseline MCnn14를 그대로 상속.
    기존 파라미터 이름은 유지하고, adapter만 추가.
    checkpoint는 strict=False로 로드하면 baseline 키는 정확히 붙고
    adapter 키만 새로 남는다.
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
        adapter_reduction: int = 16,
    ):
        super().__init__(
            sample_rate=sample_rate,
            window_size=window_size,
            hop_size=hop_size,
            mel_bins=mel_bins,
            fmin=fmin,
            fmax=fmax,
            classes_num=classes_num,
            nb_tasks=nb_tasks,
        )

        # block별 task-specific adapters
        self.block1_adapters = nn.ModuleList([Adapter2d(64, reduction=adapter_reduction) for _ in range(nb_tasks)])
        self.block2_adapters = nn.ModuleList([Adapter2d(128, reduction=adapter_reduction) for _ in range(nb_tasks)])
        self.block3_adapters = nn.ModuleList([Adapter2d(256, reduction=adapter_reduction) for _ in range(nb_tasks)])
        self.block4_adapters = nn.ModuleList([Adapter2d(512, reduction=adapter_reduction) for _ in range(nb_tasks)])
        self.block5_adapters = nn.ModuleList([Adapter2d(1024, reduction=adapter_reduction) for _ in range(nb_tasks)])
        self.block6_adapters = nn.ModuleList([Adapter2d(2048, reduction=adapter_reduction) for _ in range(nb_tasks)])

        self._active_adapter_blocks = set()

    def set_active_adapters(self, blocks: List[int]) -> None:
        self._active_adapter_blocks = set(blocks)

    def _apply_adapter(self, x: torch.Tensor, task: int, block_idx: int) -> torch.Tensor:
        if block_idx not in self._active_adapter_blocks:
            return x
        if block_idx == 1:
            return self.block1_adapters[task](x)
        elif block_idx == 2:
            return self.block2_adapters[task](x)
        elif block_idx == 3:
            return self.block3_adapters[task](x)
        elif block_idx == 4:
            return self.block4_adapters[task](x)
        elif block_idx == 5:
            return self.block5_adapters[task](x)
        elif block_idx == 6:
            return self.block6_adapters[task](x)
        return x

    def _forward_block(self, block: nn.Module, x: torch.Tensor, task: int, block_idx: int) -> torch.Tensor:
        # baseline ConvBlock forward를 그대로 풀어서 adapter hook 삽입
        x = F.relu_(block.bnF[task](block.conv1(x)))
        x = F.relu_(block.bnS[task](block.conv2(x)))
        x = self._apply_adapter(x, task, block_idx)
        x = F.avg_pool2d(x, kernel_size=(2, 2))
        return x

    def forward(self, input: torch.Tensor, task: int = 1) -> torch.Tensor:
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
        x = x1 + x2
        x = self.fc(x)
        return x


# =========================================================
# Parameter control
# =========================================================
def initialize_new_task_from_d1(model: MCnn14LateBNAdapter, task_id: int) -> None:
    """
    D2/D3 시작 시 BN을 D1 값으로 복사해서 출발.
    baseline은 task-specific BN을 갖고 있고, 여기서는 초기화를 안정적으로 맞추기 위해 복사.
    """
    copy_bn_state(model.bn0[0], model.bn0[task_id])

    blocks = [
        model.conv_block1,
        model.conv_block2,
        model.conv_block3,
        model.conv_block4,
        model.conv_block5,
        model.conv_block6,
    ]
    for block in blocks:
        copy_bn_state(block.bnF[0], block.bnF[task_id])
        copy_bn_state(block.bnS[0], block.bnS[task_id])


def freeze_all_params(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def get_bn_modules(model: MCnn14LateBNAdapter, task_id: int) -> Dict[int, List[nn.BatchNorm2d]]:
    return {
        0: [model.bn0[task_id]],
        1: [model.conv_block1.bnF[task_id], model.conv_block1.bnS[task_id]],
        2: [model.conv_block2.bnF[task_id], model.conv_block2.bnS[task_id]],
        3: [model.conv_block3.bnF[task_id], model.conv_block3.bnS[task_id]],
        4: [model.conv_block4.bnF[task_id], model.conv_block4.bnS[task_id]],
        5: [model.conv_block5.bnF[task_id], model.conv_block5.bnS[task_id]],
        6: [model.conv_block6.bnF[task_id], model.conv_block6.bnS[task_id]],
    }


def get_adapter_modules(model: MCnn14LateBNAdapter, task_id: int) -> Dict[int, nn.Module]:
    return {
        1: model.block1_adapters[task_id],
        2: model.block2_adapters[task_id],
        3: model.block3_adapters[task_id],
        4: model.block4_adapters[task_id],
        5: model.block5_adapters[task_id],
        6: model.block6_adapters[task_id],
    }


def configure_trainable_params(
    model: MCnn14LateBNAdapter,
    task_id: int,
    bn_blocks: List[int],
    adapter_blocks: List[int],
) -> List[nn.Parameter]:
    freeze_all_params(model)

    params = []
    bn_module_map = get_bn_modules(model, task_id)
    adapter_module_map = get_adapter_modules(model, task_id)

    for block_idx in bn_blocks:
        for bn in bn_module_map[block_idx]:
            for p in bn.parameters():
                p.requires_grad = True
                params.append(p)

    for block_idx in adapter_blocks:
        module = adapter_module_map[block_idx]
        for p in module.parameters():
            p.requires_grad = True
            params.append(p)

    return params


def configure_bn_train_eval(
    model: MCnn14LateBNAdapter,
    task_id: int,
    bn_blocks_to_train: List[int],
) -> None:
    """
    model.train() 후 호출.
    late-BN 실험에서 선택되지 않은 current-task BN은 eval로 묶어서
    running stats까지 고정.
    """
    bn_blocks_to_train = set(bn_blocks_to_train)
    bn_module_map = get_bn_modules(model, task_id)

    for block_idx, modules in bn_module_map.items():
        for bn in modules:
            if block_idx in bn_blocks_to_train:
                bn.train()
            else:
                bn.eval()


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =========================================================
# Eval
# =========================================================
@torch.no_grad()
def evaluate_domain_agnostic(
    model: MCnn14LateBNAdapter,
    df: pd.DataFrame,
    data_root: str,
    seen_tasks: List[int],
    adapter_blocks: List[int],
    device: str,
) -> Tuple[float, np.ndarray, Dict[str, int]]:
    """
    baseline 방식:
    seen task 각각으로 forward -> entropy 최소 task 선택 -> 그 logits로 분류
    """
    model.eval()
    model.set_active_adapters(adapter_blocks)

    y_true, y_pred = [], []
    routed_tasks = []

    for _, row in df.iterrows():
        wav_path = Path(data_root) / row["filename"]
        target = int(row["new_target"])

        audio, _ = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
        chunks = split_into_chunks(audio, CLIP_SAMPLES)

        task_logits = []
        task_entropies = []

        for task_id in seen_tasks:
            chunk_logits = []
            for chunk in chunks:
                x = torch.from_numpy(chunk.astype(np.float32)).unsqueeze(0).to(device)
                logits = model(x, task=task_id)
                chunk_logits.append(logits)

            avg_logits = torch.stack(chunk_logits, dim=0).mean(dim=0)  # (1, C)
            probs = torch.softmax(avg_logits, dim=-1)
            ent = entropy_from_probs(probs)  # (1,)
            task_logits.append(avg_logits)
            task_entropies.append(ent)

        task_entropies = torch.cat(task_entropies, dim=0)  # (num_seen,)
        best_idx = torch.argmin(task_entropies).item()
        chosen_task = seen_tasks[best_idx]
        pred = torch.argmax(task_logits[best_idx], dim=-1).item()

        y_true.append(target)
        y_pred.append(pred)
        routed_tasks.append(chosen_task)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = float((y_true == y_pred).mean() * 100.0)
    cm = metrics.confusion_matrix(y_true, y_pred)

    route_hist = {}
    for task_id in seen_tasks:
        route_hist[TASK_TO_DOMAIN[task_id]] = int(np.sum(np.array(routed_tasks) == task_id))

    return round(acc, 2), cm, route_hist


# =========================================================
# Train
# =========================================================
def train_one_epoch(
    model: MCnn14LateBNAdapter,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    task_id: int,
    bn_blocks: List[int],
    adapter_blocks: List[int],
    device: str,
) -> Tuple[float, float]:
    model.train()
    model.set_active_adapters(adapter_blocks)
    configure_bn_train_eval(model, task_id, bn_blocks)

    total_loss = 0.0
    correct = 0
    total = 0

    for audio, target, _ in loader:
        audio = audio.float().to(device)
        target = target.float().to(device)
        target_idx = torch.argmax(target, dim=-1)

        optimizer.zero_grad()
        logits = model(audio, task=task_id)
        loss = criterion(logits, target_idx)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = torch.argmax(logits, dim=-1)
        correct += (pred == target_idx).sum().item()
        total += target_idx.size(0)

    return round(total_loss / max(len(loader), 1), 6), round(100.0 * correct / max(total, 1), 2)


# =========================================================
# Experiment runner
# =========================================================
def load_d1_checkpoint(model: MCnn14LateBNAdapter, ckpt_path: str, device: str) -> None:
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[checkpoint] loaded from: {ckpt_path}")
    print(f"[checkpoint] missing={len(missing)} unexpected={len(unexpected)}")


def run_one_experiment(
    exp_id: int,
    args: argparse.Namespace,
    train_df_all: pd.DataFrame,
    test_df_all: pd.DataFrame,
    device: str,
) -> Dict:
    exp_cfg = EXPERIMENTS[exp_id]
    exp_name = exp_cfg["name"]
    bn_blocks = exp_cfg["bn_blocks"]
    adapter_blocks = exp_cfg["adapter_blocks"]

    print("\n" + "=" * 100)
    print(f"[EXP {exp_id}] {exp_name}")
    print(f"bn_blocks={bn_blocks}, adapter_blocks={adapter_blocks}")

    exp_dir = Path(args.save_dir) / exp_name
    ensure_dir(exp_dir)

    model = MCnn14LateBNAdapter(
        sample_rate=SAMPLE_RATE,
        window_size=WINDOW_SIZE,
        hop_size=HOP_SIZE,
        mel_bins=MEL_BINS,
        fmin=FMIN,
        fmax=FMAX,
        classes_num=CLASSES_NUM,
        nb_tasks=NB_TASKS,
        adapter_reduction=args.adapter_reduction,
    ).to(device)

    load_d1_checkpoint(model, args.d1_checkpoint, device)

    criterion = nn.CrossEntropyLoss()

    # D2 / D3 split
    d2_train = train_df_all[train_df_all["domain"] == "D2"].copy()
    d3_train = train_df_all[train_df_all["domain"] == "D3"].copy()
    d2_test = test_df_all[test_df_all["domain"] == "D2"].copy()
    d3_test = test_df_all[test_df_all["domain"] == "D3"].copy()

    d2_loader = DataLoader(
        Task7TrainDataset(d2_train, args.data_root),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    d3_loader = DataLoader(
        Task7TrainDataset(d3_train, args.data_root),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # -------------------------
    # Stage 1: adapt to D2
    # -------------------------
    initialize_new_task_from_d1(model, task_id=1)
    params = configure_trainable_params(model, task_id=1, bn_blocks=bn_blocks, adapter_blocks=adapter_blocks)
    print(f"[D2] trainable params = {count_trainable_params(model)}")

    optimizer = torch.optim.Adam(
        params,
        lr=args.lr_incremental,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        amsgrad=True,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr,
    )

    best_d2_acc = -1.0
    best_d2_epoch = -1
    best_d2_state = None
    hist_d2 = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=d2_loader,
            optimizer=optimizer,
            criterion=criterion,
            task_id=1,
            bn_blocks=bn_blocks,
            adapter_blocks=adapter_blocks,
            device=device,
        )
        eval_d2_acc, _, route_hist = evaluate_domain_agnostic(
            model=model,
            df=d2_test,
            data_root=args.data_root,
            seen_tasks=[0, 1],
            adapter_blocks=adapter_blocks,
            device=device,
        )
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "eval_d2_acc": eval_d2_acc,
            "lr": optimizer.param_groups[0]["lr"],
            "route_hist": route_hist,
        }
        hist_d2.append(row)
        print(row)

        if eval_d2_acc > best_d2_acc:
            best_d2_acc = eval_d2_acc
            best_d2_epoch = epoch
            best_d2_state = copy.deepcopy(model.state_dict())
            torch.save(best_d2_state, exp_dir / "best_after_D2.pth")

    if best_d2_state is not None:
        model.load_state_dict(best_d2_state, strict=False)

    # -------------------------
    # Stage 2: adapt to D3
    # -------------------------
    initialize_new_task_from_d1(model, task_id=2)
    params = configure_trainable_params(model, task_id=2, bn_blocks=bn_blocks, adapter_blocks=adapter_blocks)
    print(f"[D3] trainable params = {count_trainable_params(model)}")

    optimizer = torch.optim.Adam(
        params,
        lr=args.lr_incremental,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        amsgrad=True,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr,
    )

    best_avg = -1.0
    best_d3_epoch = -1
    best_d3_state = None
    hist_d3 = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=d3_loader,
            optimizer=optimizer,
            criterion=criterion,
            task_id=2,
            bn_blocks=bn_blocks,
            adapter_blocks=adapter_blocks,
            device=device,
        )

        d2_after_d3, _, d2_route_hist = evaluate_domain_agnostic(
            model=model,
            df=d2_test,
            data_root=args.data_root,
            seen_tasks=[0, 1, 2],
            adapter_blocks=adapter_blocks,
            device=device,
        )
        d3_after_d3, _, d3_route_hist = evaluate_domain_agnostic(
            model=model,
            df=d3_test,
            data_root=args.data_root,
            seen_tasks=[0, 1, 2],
            adapter_blocks=adapter_blocks,
            device=device,
        )
        avg_after_d3 = round((d2_after_d3 + d3_after_d3) / 2.0, 2)
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "D2_after_D3": d2_after_d3,
            "D3_after_D3": d3_after_d3,
            "Avg_after_D3": avg_after_d3,
            "lr": optimizer.param_groups[0]["lr"],
            "route_hist_d2": d2_route_hist,
            "route_hist_d3": d3_route_hist,
        }
        hist_d3.append(row)
        print(row)

        if avg_after_d3 > best_avg:
            best_avg = avg_after_d3
            best_d3_epoch = epoch
            best_d3_state = copy.deepcopy(model.state_dict())
            torch.save(best_d3_state, exp_dir / "best_after_D3.pth")

    if best_d3_state is not None:
        model.load_state_dict(best_d3_state, strict=False)

    final_d2_after_d3, _, final_route_d2 = evaluate_domain_agnostic(
        model=model,
        df=d2_test,
        data_root=args.data_root,
        seen_tasks=[0, 1, 2],
        adapter_blocks=adapter_blocks,
        device=device,
    )
    final_d3_after_d3, _, final_route_d3 = evaluate_domain_agnostic(
        model=model,
        df=d3_test,
        data_root=args.data_root,
        seen_tasks=[0, 1, 2],
        adapter_blocks=adapter_blocks,
        device=device,
    )

    result = {
        "exp_id": exp_id,
        "exp_name": exp_name,
        "bn_blocks": bn_blocks,
        "adapter_blocks": adapter_blocks,
        "D2_stage": {
            "best_epoch": best_d2_epoch,
            "best_D2_after_D2": best_d2_acc,
            "history": hist_d2,
        },
        "D3_stage": {
            "best_epoch": best_d3_epoch,
            "history": hist_d3,
        },
        "final": {
            "D2_after_D2": best_d2_acc,
            "D2_after_D3": final_d2_after_d3,
            "D3_after_D3": final_d3_after_d3,
            "Avg_after_D3": round((final_d2_after_d3 + final_d3_after_d3) / 2.0, 2),
            "Forgetting_on_D2": round(best_d2_acc - final_d2_after_d3, 2),
            "route_hist_d2_after_d3": final_route_d2,
            "route_hist_d3_after_d3": final_route_d3,
        },
    }

    save_json(result, exp_dir / "summary.json")
    return result


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = "cuda" if (args.cuda and torch.cuda.is_available()) else "cpu"
    print(f"[device] {device}")

    train_df_all = load_split_df(args.data_root, "development_train.txt")
    test_df_all = load_split_df(args.data_root, "development_test.txt")

    all_results = []
    for exp_id in args.exp_ids:
        result = run_one_experiment(
            exp_id=exp_id,
            args=args,
            train_df_all=train_df_all,
            test_df_all=test_df_all,
            device=device,
        )
        all_results.append(result)

    rows = []
    for r in all_results:
        f = r["final"]
        rows.append({
            "exp_id": r["exp_id"],
            "exp_name": r["exp_name"],
            "D2_after_D2": f["D2_after_D2"],
            "D2_after_D3": f["D2_after_D3"],
            "D3_after_D3": f["D3_after_D3"],
            "Avg_after_D3": f["Avg_after_D3"],
            "Forgetting_on_D2": f["Forgetting_on_D2"],
        })

    summary_df = pd.DataFrame(rows)
    out_dir = Path(args.save_dir)
    ensure_dir(out_dir)
    summary_df.to_csv(out_dir / "summary_table.csv", index=False, encoding="utf-8-sig")
    save_json({"results": all_results}, out_dir / "all_results.json")

    print("\n" + "=" * 100)
    print(summary_df.to_string(index=False))
    print("=" * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--d1_checkpoint", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./runs/task7_bn_adapter_ablation")

    parser.add_argument("--exp_ids", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr_incremental", type=float, default=1e-5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--adapter_reduction", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1193)
    parser.add_argument("--cuda", action="store_true", default=False)

    args = parser.parse_args()
    main(args)