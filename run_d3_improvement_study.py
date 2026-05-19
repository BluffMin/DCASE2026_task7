import os
import sys
import json
import copy
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import librosa
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn import metrics

# baseline 폴더 안에서 실행한다고 가정
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

ADAPTER_BLOCKS = [4, 5, 6]

# 현재 분석 결과 기반 class weight
# class_id:
# 0 alarm
# 1 baby_cry
# 2 dog_bark
# 3 engine
# 4 fire
# 5 footsteps
# 6 knocking
# 7 telephone_ringing
# 8 piano
# 9 speech
DEFAULT_CLASS_WEIGHTS = {
    0: 1.0,
    1: 2.0,
    2: 1.5,
    3: 1.0,
    4: 3.0,
    5: 1.5,
    6: 1.0,
    7: 4.0,
    8: 1.0,
    9: 2.0,
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


def get_class_names(df: pd.DataFrame) -> Dict[int, str]:
    mapping = {}
    for _, row in df.iterrows():
        mapping[int(row["new_target"])] = str(row["target"])
    return mapping


# =========================================================
# Dataset
# =========================================================
class Task7Dataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        data_root: str,
        train: bool = False,
        augment: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.data_root = Path(data_root)
        self.train = train
        self.augment = augment

    def __len__(self) -> int:
        return len(self.df)

    def _random_gain(self, audio: np.ndarray) -> np.ndarray:
        gain_db = np.random.uniform(-6.0, 6.0)
        gain = 10 ** (gain_db / 20.0)
        return audio * gain

    def _time_shift(self, audio: np.ndarray) -> np.ndarray:
        max_shift = int(0.2 * SAMPLE_RATE)
        shift = np.random.randint(-max_shift, max_shift + 1)
        return np.roll(audio, shift)

    def _add_small_noise(self, audio: np.ndarray) -> np.ndarray:
        # 외부 노이즈를 쓰지 않고, 아주 작은 Gaussian noise만 사용
        noise_std = np.random.uniform(0.0005, 0.002)
        noise = np.random.randn(len(audio)).astype(np.float32) * noise_std
        return audio + noise

    def _augment_audio(self, audio: np.ndarray, label: int) -> np.ndarray:
        # hard class 중심으로 조금 더 자주 augmentation
        hard_classes = {1, 4, 5, 7, 9}
        p = 0.7 if label in hard_classes else 0.3

        if np.random.rand() < p:
            audio = self._random_gain(audio)

        if np.random.rand() < p:
            audio = self._time_shift(audio)

        if np.random.rand() < 0.25:
            audio = self._add_small_noise(audio)

        return audio.astype(np.float32)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        wav_path = self.data_root / row["filename"]

        audio, _ = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
        audio = pad_truncate_sequence(audio, CLIP_SAMPLES).astype(np.float32)

        label = int(row["new_target"])

        if self.train and self.augment:
            audio = self._augment_audio(audio, label)

        return torch.from_numpy(audio), torch.tensor(label, dtype=torch.long), row["filename"]


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
# Model
# =========================================================
class MCnn14LateBNAdapter(MCnn14):
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
        if block_idx == 2:
            return self.block2_adapters[task](x)
        if block_idx == 3:
            return self.block3_adapters[task](x)
        if block_idx == 4:
            return self.block4_adapters[task](x)
        if block_idx == 5:
            return self.block5_adapters[task](x)
        if block_idx == 6:
            return self.block6_adapters[task](x)

        return x

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
        x = x1 + x2

        return x

    def forward(self, input: torch.Tensor, task: int = 1) -> torch.Tensor:
        x = self.forward_features(input, task=task)
        x = self.fc(x)
        return x


# =========================================================
# Parameter control
# =========================================================
def initialize_new_task_from_d1(model: MCnn14LateBNAdapter, task_id: int) -> None:
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


def get_adapter_modules(model: MCnn14LateBNAdapter, task_id: int) -> Dict[int, nn.Module]:
    return {
        1: model.block1_adapters[task_id],
        2: model.block2_adapters[task_id],
        3: model.block3_adapters[task_id],
        4: model.block4_adapters[task_id],
        5: model.block5_adapters[task_id],
        6: model.block6_adapters[task_id],
    }


def configure_d3_adapter_trainable(
    model: MCnn14LateBNAdapter,
    adapter_blocks: List[int],
) -> List[nn.Parameter]:
    freeze_all_params(model)

    params = []
    task_id = 2
    adapter_map = get_adapter_modules(model, task_id)

    for block_idx in adapter_blocks:
        module = adapter_map[block_idx]
        for p in module.parameters():
            p.requires_grad = True
            params.append(p)

    return params


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =========================================================
# Load model
# =========================================================
def build_model(adapter_reduction: int, device: str) -> MCnn14LateBNAdapter:
    model = MCnn14LateBNAdapter(
        sample_rate=SAMPLE_RATE,
        window_size=WINDOW_SIZE,
        hop_size=HOP_SIZE,
        mel_bins=MEL_BINS,
        fmin=FMIN,
        fmax=FMAX,
        classes_num=CLASSES_NUM,
        nb_tasks=NB_TASKS,
        adapter_reduction=adapter_reduction,
    ).to(device)

    model.set_active_adapters(ADAPTER_BLOCKS)
    return model


def load_checkpoint(
    model: nn.Module,
    ckpt_path: str,
    device: str,
    strict: bool = False,
) -> None:
    state = torch.load(ckpt_path, map_location=device)

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    missing, unexpected = model.load_state_dict(state, strict=strict)

    print(f"[load] {ckpt_path}")
    print(f"[load] missing={len(missing)}, unexpected={len(unexpected)}")
    if len(missing) > 0:
        print("[missing example]", missing[:10])
    if len(unexpected) > 0:
        print("[unexpected example]", unexpected[:10])


# =========================================================
# Evaluation
# =========================================================
@torch.no_grad()
def get_logits_by_tasks(
    model: MCnn14LateBNAdapter,
    audio: np.ndarray,
    seen_tasks: List[int],
    device: str,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    chunks = split_into_chunks(audio, CLIP_SAMPLES)

    task_logits = []
    task_entropies = []

    for task_id in seen_tasks:
        chunk_logits = []

        for chunk in chunks:
            x = torch.from_numpy(chunk.astype(np.float32)).unsqueeze(0).to(device)
            logits = model(x, task=task_id)
            chunk_logits.append(logits)

        avg_logits = torch.stack(chunk_logits, dim=0).mean(dim=0)
        probs = torch.softmax(avg_logits, dim=-1)
        ent = entropy_from_probs(probs)

        task_logits.append(avg_logits)
        task_entropies.append(ent)

    task_entropies = torch.cat(task_entropies, dim=0)
    return task_logits, task_entropies


@torch.no_grad()
def evaluate_policy(
    model: MCnn14LateBNAdapter,
    df: pd.DataFrame,
    data_root: str,
    device: str,
    policy: str,
    fixed_task: Optional[int] = None,
    d3_bias: float = 0.0,
    top_k: int = 2,
    tau: float = 1.0,
) -> Dict:
    model.eval()
    model.set_active_adapters(ADAPTER_BLOCKS)

    seen_tasks = [0, 1, 2]

    y_true = []
    y_pred = []
    chosen_tasks = []

    rows = []

    for idx, row in df.iterrows():
        wav_path = Path(data_root) / row["filename"]
        target = int(row["new_target"])

        audio, _ = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
        audio = audio.astype(np.float32)

        task_logits, entropies = get_logits_by_tasks(model, audio, seen_tasks, device)

        if policy == "fixed":
            assert fixed_task is not None
            chosen_idx = seen_tasks.index(fixed_task)
            final_logits = task_logits[chosen_idx]
            chosen_task = fixed_task

        elif policy == "entropy":
            chosen_idx = torch.argmin(entropies).item()
            final_logits = task_logits[chosen_idx]
            chosen_task = seen_tasks[chosen_idx]

        elif policy == "d3_bias":
            adjusted = entropies.clone()
            d3_idx = seen_tasks.index(2)
            adjusted[d3_idx] = adjusted[d3_idx] - d3_bias
            chosen_idx = torch.argmin(adjusted).item()
            final_logits = task_logits[chosen_idx]
            chosen_task = seen_tasks[chosen_idx]

        elif policy == "topk_mean":
            k = min(top_k, len(seen_tasks))
            idxs = torch.argsort(entropies)[:k].tolist()
            final_logits = torch.stack([task_logits[i] for i in idxs], dim=0).mean(dim=0)
            chosen_task = seen_tasks[idxs[0]]

        elif policy == "entropy_weighted":
            k = min(top_k, len(seen_tasks))
            idxs = torch.argsort(entropies)[:k].tolist()
            selected_ent = entropies[idxs]
            weights = torch.softmax(-selected_ent / tau, dim=0)

            final_logits = 0.0
            for w, i in zip(weights, idxs):
                final_logits = final_logits + w * task_logits[i]

            chosen_task = seen_tasks[idxs[0]]

        else:
            raise ValueError(f"Unknown policy: {policy}")

        probs = torch.softmax(final_logits, dim=-1)
        pred = torch.argmax(final_logits, dim=-1).item()
        conf = torch.max(probs, dim=-1).values.item()

        y_true.append(target)
        y_pred.append(pred)
        chosen_tasks.append(chosen_task)

        rows.append({
            "filename": row["filename"],
            "domain": row["domain"],
            "target_name": row["target"],
            "target": target,
            "pred": pred,
            "correct": int(pred == target),
            "chosen_task": chosen_task,
            "chosen_domain": TASK_TO_DOMAIN[chosen_task],
            "confidence": float(conf),
            "entropy_D1": float(entropies[0].item()),
            "entropy_D2": float(entropies[1].item()),
            "entropy_D3": float(entropies[2].item()),
        })

        if (idx + 1) % 100 == 0:
            print(f"[eval {policy}] {idx + 1}/{len(df)}")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    chosen_tasks = np.array(chosen_tasks)

    acc = round(float((y_true == y_pred).mean() * 100.0), 2)
    cm = metrics.confusion_matrix(y_true, y_pred, labels=list(range(CLASSES_NUM)))

    route_hist = {}
    for task_id in seen_tasks:
        route_hist[TASK_TO_DOMAIN[task_id]] = int(np.sum(chosen_tasks == task_id))

    pred_df = pd.DataFrame(rows)

    return {
        "acc": acc,
        "cm": cm,
        "route_hist": route_hist,
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


# =========================================================
# Train D3 variants
# =========================================================
def make_weighted_sampler(df: pd.DataFrame, weight_map: Dict[int, float]) -> WeightedRandomSampler:
    weights = []
    for _, row in df.iterrows():
        label = int(row["new_target"])
        weights.append(float(weight_map.get(label, 1.0)))

    weights = torch.DoubleTensor(weights)
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )
    return sampler


def train_one_epoch_d3(
    model: MCnn14LateBNAdapter,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, float]:
    model.train()
    model.set_active_adapters(ADAPTER_BLOCKS)

    total_loss = 0.0
    total = 0
    correct = 0

    for audio, target, _ in loader:
        audio = audio.float().to(device)
        target = target.long().to(device)

        optimizer.zero_grad()

        logits = model(audio, task=2)
        loss = criterion(logits, target)

        loss.backward()
        optimizer.step()

        pred = torch.argmax(logits, dim=-1)
        correct += int((pred == target).sum().item())
        total += int(target.size(0))
        total_loss += float(loss.item())

    return round(total_loss / max(len(loader), 1), 6), round(correct / max(total, 1) * 100.0, 2)


def run_d3_training_variant(
    variant_name: str,
    args: argparse.Namespace,
    train_df_all: pd.DataFrame,
    test_df_all: pd.DataFrame,
    class_name_map: Dict[int, str],
    device: str,
    use_weighted_ce: bool,
    use_sampler: bool,
    use_aug: bool,
) -> Dict:
    print("\n" + "=" * 100)
    print(f"[TRAIN VARIANT] {variant_name}")
    print(f"use_weighted_ce={use_weighted_ce}, use_sampler={use_sampler}, use_aug={use_aug}")

    out_dir = Path(args.save_dir) / variant_name
    ensure_dir(out_dir)

    model = build_model(args.adapter_reduction, device)
    load_checkpoint(model, args.after_d2_checkpoint, device, strict=False)

    # D3는 D1 BN으로 초기화해서 기존 실험과 최대한 맞춤
    initialize_new_task_from_d1(model, task_id=2)

    params = configure_d3_adapter_trainable(model, ADAPTER_BLOCKS)
    print(f"[trainable params] {count_trainable_params(model)}")

    d3_train = train_df_all[train_df_all["domain"] == "D3"].copy()
    d2_test = test_df_all[test_df_all["domain"] == "D2"].copy()
    d3_test = test_df_all[test_df_all["domain"] == "D3"].copy()

    dataset = Task7Dataset(
        df=d3_train,
        data_root=args.data_root,
        train=True,
        augment=use_aug,
    )

    if use_sampler:
        sampler = make_weighted_sampler(d3_train, DEFAULT_CLASS_WEIGHTS)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    if use_weighted_ce:
        weights = torch.ones(CLASSES_NUM, dtype=torch.float32)
        for k, v in DEFAULT_CLASS_WEIGHTS.items():
            weights[int(k)] = float(v)
        weights = weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        params,
        lr=args.lr_incremental,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay,
        amsgrad=True,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr,
    )

    best_metric = -1.0
    best_state = None
    best_epoch = -1
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch_d3(
            model=model,
            loader=loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        d2_eval = evaluate_policy(
            model=model,
            df=d2_test,
            data_root=args.data_root,
            device=device,
            policy=args.train_eval_policy,
            top_k=args.train_eval_topk,
            tau=args.train_eval_tau,
        )

        d3_eval = evaluate_policy(
            model=model,
            df=d3_test,
            data_root=args.data_root,
            device=device,
            policy=args.train_eval_policy,
            top_k=args.train_eval_topk,
            tau=args.train_eval_tau,
        )

        d2_acc = d2_eval["acc"]
        d3_acc = d3_eval["acc"]
        avg_acc = round((d2_acc + d3_acc) / 2.0, 2)

        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "D2_acc": d2_acc,
            "D3_acc": d3_acc,
            "Avg_acc": avg_acc,
            "lr": optimizer.param_groups[0]["lr"],
            "D3_route_hist": d3_eval["route_hist"],
        }
        history.append(row)

        print(row)

        if args.best_metric == "D3":
            current = d3_acc
        elif args.best_metric == "AVG":
            current = avg_acc
        else:
            raise ValueError(args.best_metric)

        if current > best_metric:
            best_metric = current
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, out_dir / "best.pth")

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)

    # 최종은 여러 policy로 다시 평가
    final_policies = run_inference_policy_suite(
        model=model,
        d2_test=d2_test,
        d3_test=d3_test,
        data_root=args.data_root,
        device=device,
        class_name_map=class_name_map,
        out_dir=out_dir / "final_policy_eval",
    )

    result = {
        "variant_name": variant_name,
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "history": history,
        "final_policies": final_policies,
    }

    save_json(result, out_dir / "summary.json")
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False, encoding="utf-8-sig")

    return result


# =========================================================
# Policy suite
# =========================================================
def save_policy_outputs(
    result: Dict,
    out_dir: Path,
    prefix: str,
    class_name_map: Dict[int, str],
) -> Dict:
    ensure_dir(out_dir)

    pred_df = result["pred_df"]
    pred_df.to_csv(out_dir / f"{prefix}_predictions.csv", index=False, encoding="utf-8-sig")

    cw = classwise_accuracy(pred_df, class_name_map)
    cw.to_csv(out_dir / f"{prefix}_classwise.csv", index=False, encoding="utf-8-sig")

    np.savetxt(out_dir / f"{prefix}_confusion.csv", result["cm"], delimiter=",", fmt="%d")

    compact = {
        "acc": result["acc"],
        "route_hist": result["route_hist"],
    }
    save_json(compact, out_dir / f"{prefix}_summary.json")

    return compact


def run_inference_policy_suite(
    model: MCnn14LateBNAdapter,
    d2_test: pd.DataFrame,
    d3_test: pd.DataFrame,
    data_root: str,
    device: str,
    class_name_map: Dict[int, str],
    out_dir: Path,
) -> Dict:
    ensure_dir(out_dir)

    policy_results = {}

    # 1. Oracle/fixed routing
    fixed_settings = [
        ("fixed_D1", 0),
        ("fixed_D2", 1),
        ("fixed_D3", 2),
    ]

    for name, task_id in fixed_settings:
        print(f"[policy suite] {name}")

        d2_res = evaluate_policy(model, d2_test, data_root, device, policy="fixed", fixed_task=task_id)
        d3_res = evaluate_policy(model, d3_test, data_root, device, policy="fixed", fixed_task=task_id)

        d2_compact = save_policy_outputs(d2_res, out_dir / name, "D2", class_name_map)
        d3_compact = save_policy_outputs(d3_res, out_dir / name, "D3", class_name_map)

        policy_results[name] = {
            "D2": d2_compact,
            "D3": d3_compact,
            "Avg": round((d2_compact["acc"] + d3_compact["acc"]) / 2.0, 2),
        }

    # baseline entropy
    print("[policy suite] entropy")
    d2_res = evaluate_policy(model, d2_test, data_root, device, policy="entropy")
    d3_res = evaluate_policy(model, d3_test, data_root, device, policy="entropy")

    d2_compact = save_policy_outputs(d2_res, out_dir / "entropy", "D2", class_name_map)
    d3_compact = save_policy_outputs(d3_res, out_dir / "entropy", "D3", class_name_map)

    policy_results["entropy"] = {
        "D2": d2_compact,
        "D3": d3_compact,
        "Avg": round((d2_compact["acc"] + d3_compact["acc"]) / 2.0, 2),
    }

    # 2. D3 entropy bias
    for bias in [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]:
        name = f"d3_bias_{bias:.2f}".replace(".", "p")
        print(f"[policy suite] {name}")

        d2_res = evaluate_policy(model, d2_test, data_root, device, policy="d3_bias", d3_bias=bias)
        d3_res = evaluate_policy(model, d3_test, data_root, device, policy="d3_bias", d3_bias=bias)

        d2_compact = save_policy_outputs(d2_res, out_dir / name, "D2", class_name_map)
        d3_compact = save_policy_outputs(d3_res, out_dir / name, "D3", class_name_map)

        policy_results[name] = {
            "D2": d2_compact,
            "D3": d3_compact,
            "Avg": round((d2_compact["acc"] + d3_compact["acc"]) / 2.0, 2),
        }

    # 3. Top-k mean ensemble
    for k in [2, 3]:
        name = f"top{k}_mean"
        print(f"[policy suite] {name}")

        d2_res = evaluate_policy(model, d2_test, data_root, device, policy="topk_mean", top_k=k)
        d3_res = evaluate_policy(model, d3_test, data_root, device, policy="topk_mean", top_k=k)

        d2_compact = save_policy_outputs(d2_res, out_dir / name, "D2", class_name_map)
        d3_compact = save_policy_outputs(d3_res, out_dir / name, "D3", class_name_map)

        policy_results[name] = {
            "D2": d2_compact,
            "D3": d3_compact,
            "Avg": round((d2_compact["acc"] + d3_compact["acc"]) / 2.0, 2),
        }

    # 4. Entropy-weighted ensemble
    for k in [2, 3]:
        for tau in [0.5, 1.0, 2.0]:
            name = f"top{k}_ew_tau{tau}".replace(".", "p")
            print(f"[policy suite] {name}")

            d2_res = evaluate_policy(
                model, d2_test, data_root, device,
                policy="entropy_weighted",
                top_k=k,
                tau=tau,
            )
            d3_res = evaluate_policy(
                model, d3_test, data_root, device,
                policy="entropy_weighted",
                top_k=k,
                tau=tau,
            )

            d2_compact = save_policy_outputs(d2_res, out_dir / name, "D2", class_name_map)
            d3_compact = save_policy_outputs(d3_res, out_dir / name, "D3", class_name_map)

            policy_results[name] = {
                "D2": d2_compact,
                "D3": d3_compact,
                "Avg": round((d2_compact["acc"] + d3_compact["acc"]) / 2.0, 2),
            }

    rows = []
    for name, r in policy_results.items():
        rows.append({
            "policy": name,
            "D2_acc": r["D2"]["acc"],
            "D3_acc": r["D3"]["acc"],
            "Avg": r["Avg"],
            "D3_route_D1": r["D3"]["route_hist"].get("D1", 0),
            "D3_route_D2": r["D3"]["route_hist"].get("D2", 0),
            "D3_route_D3": r["D3"]["route_hist"].get("D3", 0),
        })

    table = pd.DataFrame(rows).sort_values("D3_acc", ascending=False)
    table.to_csv(out_dir / "policy_summary_table.csv", index=False, encoding="utf-8-sig")

    save_json(policy_results, out_dir / "policy_results.json")

    print("\n[Policy summary]")
    print(table.to_string(index=False))

    return policy_results


# =========================================================
# Main
# =========================================================
def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    device = "cuda" if (args.cuda and torch.cuda.is_available()) else "cpu"
    print(f"[device] {device}")

    save_dir = Path(args.save_dir)
    ensure_dir(save_dir)

    train_df_all = load_split_df(args.data_root, "development_train.txt")
    test_df_all = load_split_df(args.data_root, "development_test.txt")

    d2_test = test_df_all[test_df_all["domain"] == "D2"].copy()
    d3_test = test_df_all[test_df_all["domain"] == "D3"].copy()

    class_name_map = get_class_names(test_df_all)

    all_results = {}

    # =====================================================
    # 1, 2. 기존 best_after_D3 checkpoint로 oracle/routing/ensemble 평가
    # =====================================================
    if args.run_inference:
        print("\n" + "#" * 100)
        print("[1,2] Inference policy study on existing best_after_D3")
        print("#" * 100)

        model = build_model(args.adapter_reduction, device)
        load_checkpoint(model, args.after_d3_checkpoint, device, strict=False)
        model.eval()

        policy_results = run_inference_policy_suite(
            model=model,
            d2_test=d2_test,
            d3_test=d3_test,
            data_root=args.data_root,
            device=device,
            class_name_map=class_name_map,
            out_dir=save_dir / "01_02_inference_policy_study",
        )

        all_results["01_02_inference_policy_study"] = policy_results

    # =====================================================
    # 3. Weighted CE
    # =====================================================
    if args.run_weighted_ce:
        result = run_d3_training_variant(
            variant_name="03_weighted_ce",
            args=args,
            train_df_all=train_df_all,
            test_df_all=test_df_all,
            class_name_map=class_name_map,
            device=device,
            use_weighted_ce=True,
            use_sampler=False,
            use_aug=False,
        )
        all_results["03_weighted_ce"] = result

    # =====================================================
    # 4. Hard-class oversampling
    # =====================================================
    if args.run_sampler:
        result = run_d3_training_variant(
            variant_name="04_hard_sampler",
            args=args,
            train_df_all=train_df_all,
            test_df_all=test_df_all,
            class_name_map=class_name_map,
            device=device,
            use_weighted_ce=False,
            use_sampler=True,
            use_aug=False,
        )
        all_results["04_hard_sampler"] = result

    # =====================================================
    # 5. Hard-class oversampling + augmentation
    # =====================================================
    if args.run_sampler_aug:
        result = run_d3_training_variant(
            variant_name="05_hard_sampler_aug",
            args=args,
            train_df_all=train_df_all,
            test_df_all=test_df_all,
            class_name_map=class_name_map,
            device=device,
            use_weighted_ce=False,
            use_sampler=True,
            use_aug=True,
        )
        all_results["05_hard_sampler_aug"] = result

    save_json(all_results, save_dir / "all_results.json")

    print("\n" + "=" * 100)
    print("[DONE]")
    print(f"Saved to: {save_dir}")
    print("=" * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, required=True)

    parser.add_argument("--after_d2_checkpoint", type=str, required=True)
    parser.add_argument("--after_d3_checkpoint", type=str, required=True)

    parser.add_argument("--save_dir", type=str, required=True)

    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr_incremental", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adapter_reduction", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1193)

    parser.add_argument("--cuda", action="store_true", default=False)

    parser.add_argument("--best_metric", type=str, default="D3", choices=["D3", "AVG"])

    parser.add_argument(
        "--train_eval_policy",
        type=str,
        default="entropy",
        choices=["entropy", "topk_mean", "entropy_weighted"],
    )
    parser.add_argument("--train_eval_topk", type=int, default=2)
    parser.add_argument("--train_eval_tau", type=float, default=1.0)

    parser.add_argument("--run_inference", action="store_true", default=False)
    parser.add_argument("--run_weighted_ce", action="store_true", default=False)
    parser.add_argument("--run_sampler", action="store_true", default=False)
    parser.add_argument("--run_sampler_aug", action="store_true", default=False)

    parser.add_argument("--run_all", action="store_true", default=False)

    args = parser.parse_args()

    if args.run_all:
        args.run_inference = True
        args.run_weighted_ce = True
        args.run_sampler = True
        args.run_sampler_aug = True

    main(args)