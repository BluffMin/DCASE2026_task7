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
from torch.utils.data import Dataset, DataLoader
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

# D1 데이터가 없으므로 routing / calibration / prototype 후보는 D2, D3만 사용
ROUTING_TASKS = [1, 2]

# late_adapter_456
ADAPTER_BLOCKS = [4, 5, 6]


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
        aug_mode: str = "none",
    ):
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
        gain = 10.0 ** (gain_db / 20.0)
        return audio * gain

    def _time_shift(self, audio: np.ndarray) -> np.ndarray:
        max_shift = int(0.15 * SAMPLE_RATE)
        shift = np.random.randint(-max_shift, max_shift + 1)
        return np.roll(audio, shift)

    def _small_noise(self, audio: np.ndarray) -> np.ndarray:
        std = np.random.uniform(0.0002, 0.0010)
        noise = np.random.randn(len(audio)).astype(np.float32) * std
        return audio + noise

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
        label = int(row["new_target"])

        audio = self._crop_or_pad(audio)

        if self.train:
            audio = self._augment_audio(audio)

        return (
            torch.from_numpy(audio.astype(np.float32)),
            torch.tensor(label, dtype=torch.long),
            row["filename"],
        )


# =========================================================
# Adapter
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

        # adapter 위치: conv1/conv2 + BN + ReLU 이후, pooling 이전
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
        feat = self.forward_features(input, task=task)
        logits = self.fc(feat)
        return logits


# =========================================================
# Model helpers
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


def load_checkpoint(model: nn.Module, ckpt_path: str, device: str, strict: bool = False) -> None:
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


def configure_task_adapter_trainable(
    model: MCnn14LateBNAdapter,
    task_id: int,
    adapter_blocks: List[int],
) -> List[nn.Parameter]:
    freeze_all_params(model)

    params = []
    adapter_map = get_adapter_modules(model, task_id)

    for block_idx in adapter_blocks:
        module = adapter_map[block_idx]
        for p in module.parameters():
            p.requires_grad = True
            params.append(p)

    return params


def initialize_task_bn_from_d1(model: MCnn14LateBNAdapter, task_id: int) -> None:
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


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_bn_eval(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()


# =========================================================
# Mixup
# =========================================================
def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    if alpha <= 0:
        return x, y, y, 1.0

    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1.0 - lam) * x[index]
    y_a = y
    y_b = y[index]

    return mixed_x, y_a, y_b, lam


# =========================================================
# Training
# =========================================================
def train_one_epoch_adapter(
    model: MCnn14LateBNAdapter,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str,
    task_id: int,
    use_mixup: bool = False,
    mixup_alpha: float = 0.2,
    freeze_bn_stats: bool = True,
) -> Tuple[float, float]:
    model.train()
    model.set_active_adapters(ADAPTER_BLOCKS)

    if freeze_bn_stats:
        set_bn_eval(model)

    total_loss = 0.0
    total = 0
    correct = 0

    for audio, target, _ in loader:
        audio = audio.float().to(device)
        target = target.long().to(device)

        optimizer.zero_grad()

        if use_mixup:
            mixed_audio, y_a, y_b, lam = mixup_data(audio, target, alpha=mixup_alpha)
            logits = model(mixed_audio, task=task_id)
            loss = lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
        else:
            logits = model(audio, task=task_id)
            loss = criterion(logits, target)

        pred = torch.argmax(logits, dim=-1)
        correct += int((pred == target).sum().item())

        loss.backward()
        optimizer.step()

        total += int(target.size(0))
        total_loss += float(loss.item())

    avg_loss = round(total_loss / max(len(loader), 1), 6)
    acc = round(correct / max(total, 1) * 100.0, 2)

    return avg_loss, acc


@torch.no_grad()
def evaluate_fixed_task(
    model: MCnn14LateBNAdapter,
    df: pd.DataFrame,
    data_root: str,
    device: str,
    task_id: int,
) -> float:
    model.eval()
    model.set_active_adapters(ADAPTER_BLOCKS)

    y_true = []
    y_pred = []

    df = df.reset_index(drop=True)

    for _, row in df.iterrows():
        wav_path = Path(data_root) / row["filename"]
        target = int(row["new_target"])

        audio, _ = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
        audio = audio.astype(np.float32)

        chunks = split_into_chunks(audio, CLIP_SAMPLES)
        chunk_logits = []

        for chunk in chunks:
            x = torch.from_numpy(chunk.astype(np.float32)).unsqueeze(0).to(device)
            logits = model(x, task=task_id)
            chunk_logits.append(logits)

        logits = torch.stack(chunk_logits, dim=0).mean(dim=0)
        pred = int(torch.argmax(logits, dim=-1).item())

        y_true.append(target)
        y_pred.append(pred)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return round(float((y_true == y_pred).mean() * 100.0), 2)


def train_adapter_for_domain(
    model: MCnn14LateBNAdapter,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    args: argparse.Namespace,
    device: str,
    domain: str,
    out_dir: Path,
) -> Dict:
    ensure_dir(out_dir)

    task_id = DOMAIN_TO_TASK[domain]
    domain_train = train_df[train_df["domain"] == domain].copy()
    domain_test = test_df[test_df["domain"] == domain].copy()

    print("\n" + "=" * 100)
    print(f"[TRAIN] {domain} adapter only")
    print(f"[data] {domain} train: {len(domain_train)}")
    print(f"[data] {domain} test : {len(domain_test)}")
    print("=" * 100)

    if args.init_bn_from_d1:
        print(f"[init] {domain} BN <- D1 BN")
        initialize_task_bn_from_d1(model, task_id=task_id)

    params = configure_task_adapter_trainable(
        model=model,
        task_id=task_id,
        adapter_blocks=ADAPTER_BLOCKS,
    )

    print(f"[trainable params] {count_trainable_params(model)}")

    dataset = Task7Dataset(
        df=domain_train,
        data_root=args.data_root,
        train=True,
        aug_mode=args.aug_mode,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

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
    best_epoch = -1
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch_adapter(
            model=model,
            loader=loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            task_id=task_id,
            use_mixup=args.use_mixup,
            mixup_alpha=args.mixup_alpha,
            freeze_bn_stats=args.freeze_bn_stats,
        )

        fixed_acc = evaluate_fixed_task(
            model=model,
            df=domain_test,
            data_root=args.data_root,
            device=device,
            task_id=task_id,
        )

        scheduler.step()

        row = {
            "epoch": epoch,
            "domain": domain,
            "task_id": task_id,
            "train_loss": train_loss,
            "train_acc": train_acc,
            f"{domain}_fixed_acc": fixed_acc,
            "lr": optimizer.param_groups[0]["lr"],
        }

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

    result = {
        "domain": domain,
        "task_id": task_id,
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "checkpoint": str(after_path),
    }

    save_json(result, out_dir / f"summary_{domain}.json")
    return result


# =========================================================
# Forward helpers for policy evaluation
# =========================================================
@torch.no_grad()
def forward_audio_all_tasks(
    model: MCnn14LateBNAdapter,
    audio: np.ndarray,
    seen_tasks: List[int],
    device: str,
    temperatures: Optional[Dict[int, float]] = None,
) -> Dict:
    chunks = split_into_chunks(audio, CLIP_SAMPLES)

    logits_by_task = {}
    probs_by_task = {}
    cal_probs_by_task = {}
    features_by_task = {}
    entropies = {}
    cal_entropies = {}

    for task_id in seen_tasks:
        chunk_logits = []
        chunk_features = []

        for chunk in chunks:
            x = torch.from_numpy(chunk.astype(np.float32)).unsqueeze(0).to(device)

            feat = model.forward_features(x, task=task_id)
            logits = model.fc(feat)

            chunk_features.append(feat)
            chunk_logits.append(logits)

        logits = torch.stack(chunk_logits, dim=0).mean(dim=0)
        feat = torch.stack(chunk_features, dim=0).mean(dim=0)

        probs = torch.softmax(logits, dim=-1)
        ent = entropy_from_probs(probs)

        T = 1.0
        if temperatures is not None:
            T = float(temperatures.get(task_id, 1.0))

        cal_logits = logits / T
        cal_probs = torch.softmax(cal_logits, dim=-1)
        cal_ent = entropy_from_probs(cal_probs)

        logits_by_task[task_id] = logits
        probs_by_task[task_id] = probs
        cal_probs_by_task[task_id] = cal_probs
        features_by_task[task_id] = feat
        entropies[task_id] = float(ent.item())
        cal_entropies[task_id] = float(cal_ent.item())

    return {
        "logits": logits_by_task,
        "probs": probs_by_task,
        "cal_probs": cal_probs_by_task,
        "features": features_by_task,
        "entropies": entropies,
        "cal_entropies": cal_entropies,
    }


# =========================================================
# Temperature scaling
# =========================================================
@torch.no_grad()
def collect_logits_for_temperature(
    model: MCnn14LateBNAdapter,
    df: pd.DataFrame,
    data_root: str,
    task_id: int,
    device: str,
    max_samples: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()

    domain_name = TASK_TO_DOMAIN[task_id]
    rows = df[df["domain"] == domain_name].copy().reset_index(drop=True)

    if max_samples > 0 and len(rows) > max_samples:
        rows = rows.sample(n=max_samples, random_state=task_id).reset_index(drop=True)

    logits_list = []
    labels_list = []

    for idx, row in rows.iterrows():
        wav_path = Path(data_root) / row["filename"]
        audio, _ = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
        audio = audio.astype(np.float32)

        chunks = split_into_chunks(audio, CLIP_SAMPLES)
        chunk_logits = []

        for chunk in chunks:
            x = torch.from_numpy(chunk.astype(np.float32)).unsqueeze(0).to(device)
            logits = model(x, task=task_id)
            chunk_logits.append(logits)

        avg_logits = torch.stack(chunk_logits, dim=0).mean(dim=0)

        logits_list.append(avg_logits.detach().cpu())
        labels_list.append(int(row["new_target"]))

        if (idx + 1) % 100 == 0:
            print(f"[collect temperature] {domain_name} {idx + 1}/{len(rows)}")

    if len(logits_list) == 0:
        raise RuntimeError(f"No samples for temperature fitting: task={task_id}, domain={domain_name}")

    logits_all = torch.cat(logits_list, dim=0)
    labels_all = torch.tensor(labels_list, dtype=torch.long)

    return logits_all, labels_all


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
    T = max(min(T, 10.0), 0.05)

    return T


def fit_branch_temperatures(
    model: MCnn14LateBNAdapter,
    train_df: pd.DataFrame,
    data_root: str,
    seen_tasks: List[int],
    device: str,
    max_calib_samples: int = -1,
) -> Dict[int, float]:
    temperatures = {}

    for task_id in seen_tasks:
        domain_name = TASK_TO_DOMAIN[task_id]
        print("\n" + "-" * 80)
        print(f"[temperature] fitting task={task_id}, domain={domain_name}")
        print("-" * 80)

        logits, labels = collect_logits_for_temperature(
            model=model,
            df=train_df,
            data_root=data_root,
            task_id=task_id,
            device=device,
            max_samples=max_calib_samples,
        )

        T = fit_temperature_for_logits(logits, labels, device=device)
        temperatures[task_id] = T

        print(f"[temperature] {domain_name}: T={T:.4f}")

    return temperatures


# =========================================================
# Prototype
# =========================================================
@torch.no_grad()
def build_prototypes(
    model: MCnn14LateBNAdapter,
    train_df: pd.DataFrame,
    data_root: str,
    seen_tasks: List[int],
    device: str,
    max_proto_samples: int = -1,
    normalize: bool = True,
) -> Dict:
    model.eval()

    domain_feats = {task_id: [] for task_id in seen_tasks}
    class_feats = {task_id: {c: [] for c in range(CLASSES_NUM)} for task_id in seen_tasks}

    for task_id in seen_tasks:
        domain_name = TASK_TO_DOMAIN[task_id]
        rows = train_df[train_df["domain"] == domain_name].copy().reset_index(drop=True)

        if max_proto_samples > 0 and len(rows) > max_proto_samples:
            rows = rows.sample(n=max_proto_samples, random_state=task_id).reset_index(drop=True)

        print("\n" + "-" * 80)
        print(f"[prototype] building {domain_name}, n={len(rows)}")
        print("-" * 80)

        for idx, row in rows.iterrows():
            wav_path = Path(data_root) / row["filename"]
            label = int(row["new_target"])

            audio, _ = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
            audio = audio.astype(np.float32)

            chunks = split_into_chunks(audio, CLIP_SAMPLES)
            feats = []

            for chunk in chunks:
                x = torch.from_numpy(chunk.astype(np.float32)).unsqueeze(0).to(device)
                feat = model.forward_features(x, task=task_id)
                feats.append(feat)

            feat = torch.stack(feats, dim=0).mean(dim=0).squeeze(0)

            if normalize:
                feat = F.normalize(feat, dim=0)

            feat_cpu = feat.detach().cpu()

            domain_feats[task_id].append(feat_cpu)
            class_feats[task_id][label].append(feat_cpu)

            if (idx + 1) % 100 == 0:
                print(f"[prototype] {domain_name} {idx + 1}/{len(rows)}")

    domain_proto = {}
    class_proto = {}

    for task_id in seen_tasks:
        if len(domain_feats[task_id]) == 0:
            raise RuntimeError(f"No prototype features for task={task_id}")

        proto = torch.stack(domain_feats[task_id], dim=0).mean(dim=0)

        if normalize:
            proto = F.normalize(proto, dim=0)

        domain_proto[task_id] = proto

        class_proto[task_id] = {}

        for c in range(CLASSES_NUM):
            if len(class_feats[task_id][c]) == 0:
                continue

            c_proto = torch.stack(class_feats[task_id][c], dim=0).mean(dim=0)

            if normalize:
                c_proto = F.normalize(c_proto, dim=0)

            class_proto[task_id][c] = c_proto

    return {
        "domain_proto": domain_proto,
        "class_proto": class_proto,
        "normalize": normalize,
    }


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().cpu()
    b = b.detach().cpu()

    a = F.normalize(a, dim=0)
    b = F.normalize(b, dim=0)

    return float(1.0 - torch.sum(a * b).item())


def compute_proto_distances(
    features_by_task: Dict[int, torch.Tensor],
    prototypes: Dict,
    seen_tasks: List[int],
    proto_type: str,
) -> Dict[int, float]:
    distances = {}

    if proto_type == "domain":
        for task_id in seen_tasks:
            feat = features_by_task[task_id].squeeze(0).detach().cpu()
            proto = prototypes["domain_proto"][task_id]
            distances[task_id] = cosine_distance(feat, proto)

    elif proto_type == "class":
        for task_id in seen_tasks:
            feat = features_by_task[task_id].squeeze(0).detach().cpu()

            min_dist = None

            for _, proto in prototypes["class_proto"][task_id].items():
                dist = cosine_distance(feat, proto)
                if min_dist is None or dist < min_dist:
                    min_dist = dist

            if min_dist is None:
                min_dist = 999.0

            distances[task_id] = float(min_dist)

    else:
        raise ValueError(f"Unknown proto_type: {proto_type}")

    return distances


# =========================================================
# Policy evaluation
# =========================================================
def get_final_probs_and_route(
    outputs: Dict,
    policy: str,
    seen_tasks: List[int],
    prototypes: Optional[Dict] = None,
    top_k: int = 2,
    tau: float = 1.0,
    true_task: Optional[int] = None,
) -> Tuple[torch.Tensor, int, Dict]:
    probs = outputs["probs"]
    cal_probs = outputs["cal_probs"]
    ent = outputs["entropies"]
    cal_ent = outputs["cal_entropies"]

    aux = {}

    if policy.startswith("fixed_"):
        domain = policy.replace("fixed_", "")
        chosen_task = DOMAIN_TO_TASK[domain]
        final_probs = probs[chosen_task]
        aux["score"] = {}

    elif policy == "oracle":
        if true_task is None:
            raise ValueError("oracle policy requires true_task")
        chosen_task = true_task
        final_probs = probs[chosen_task]
        aux["score"] = {}

    elif policy == "entropy_hard":
        chosen_task = min(seen_tasks, key=lambda t: ent[t])
        final_probs = probs[chosen_task]
        aux["score"] = {TASK_TO_DOMAIN[t]: ent[t] for t in seen_tasks}

    elif policy == "cal_entropy_hard":
        chosen_task = min(seen_tasks, key=lambda t: cal_ent[t])
        final_probs = cal_probs[chosen_task]
        aux["score"] = {TASK_TO_DOMAIN[t]: cal_ent[t] for t in seen_tasks}

    elif policy == "proto_domain_hard":
        if prototypes is None:
            raise ValueError("prototype policy requires prototypes")

        distances = compute_proto_distances(
            features_by_task=outputs["features"],
            prototypes=prototypes,
            seen_tasks=seen_tasks,
            proto_type="domain",
        )

        chosen_task = min(seen_tasks, key=lambda t: distances[t])
        final_probs = probs[chosen_task]
        aux["score"] = {TASK_TO_DOMAIN[t]: distances[t] for t in seen_tasks}

    elif policy == "proto_class_hard":
        if prototypes is None:
            raise ValueError("prototype policy requires prototypes")

        distances = compute_proto_distances(
            features_by_task=outputs["features"],
            prototypes=prototypes,
            seen_tasks=seen_tasks,
            proto_type="class",
        )

        chosen_task = min(seen_tasks, key=lambda t: distances[t])
        final_probs = probs[chosen_task]
        aux["score"] = {TASK_TO_DOMAIN[t]: distances[t] for t in seen_tasks}

    elif policy == "all_mean":
        chosen_task = -1
        final_probs = torch.stack([probs[t] for t in seen_tasks], dim=0).mean(dim=0)
        aux["score"] = {}

    elif policy == "entropy_moe":
        scores = torch.tensor([ent[t] for t in seen_tasks])
        order = torch.argsort(scores)[:min(top_k, len(seen_tasks))]
        selected_tasks = [seen_tasks[i] for i in order.tolist()]
        weights = torch.softmax(-scores[order] / tau, dim=0).to(probs[seen_tasks[0]].device)

        final_probs = 0.0
        for w, t in zip(weights, selected_tasks):
            final_probs = final_probs + w * probs[t]

        chosen_task = selected_tasks[0]
        aux["score"] = {TASK_TO_DOMAIN[t]: ent[t] for t in seen_tasks}
        aux["weights"] = {TASK_TO_DOMAIN[t]: float(w.item()) for w, t in zip(weights, selected_tasks)}

    elif policy == "cal_entropy_moe":
        scores = torch.tensor([cal_ent[t] for t in seen_tasks])
        order = torch.argsort(scores)[:min(top_k, len(seen_tasks))]
        selected_tasks = [seen_tasks[i] for i in order.tolist()]
        weights = torch.softmax(-scores[order] / tau, dim=0).to(cal_probs[seen_tasks[0]].device)

        final_probs = 0.0
        for w, t in zip(weights, selected_tasks):
            final_probs = final_probs + w * cal_probs[t]

        chosen_task = selected_tasks[0]
        aux["score"] = {TASK_TO_DOMAIN[t]: cal_ent[t] for t in seen_tasks}
        aux["weights"] = {TASK_TO_DOMAIN[t]: float(w.item()) for w, t in zip(weights, selected_tasks)}

    elif policy == "proto_domain_moe":
        if prototypes is None:
            raise ValueError("prototype policy requires prototypes")

        distances = compute_proto_distances(
            features_by_task=outputs["features"],
            prototypes=prototypes,
            seen_tasks=seen_tasks,
            proto_type="domain",
        )

        scores = torch.tensor([distances[t] for t in seen_tasks])
        order = torch.argsort(scores)[:min(top_k, len(seen_tasks))]
        selected_tasks = [seen_tasks[i] for i in order.tolist()]
        weights = torch.softmax(-scores[order] / tau, dim=0).to(probs[seen_tasks[0]].device)

        final_probs = 0.0
        for w, t in zip(weights, selected_tasks):
            final_probs = final_probs + w * probs[t]

        chosen_task = selected_tasks[0]
        aux["score"] = {TASK_TO_DOMAIN[t]: distances[t] for t in seen_tasks}
        aux["weights"] = {TASK_TO_DOMAIN[t]: float(w.item()) for w, t in zip(weights, selected_tasks)}

    elif policy == "proto_class_moe":
        if prototypes is None:
            raise ValueError("prototype policy requires prototypes")

        distances = compute_proto_distances(
            features_by_task=outputs["features"],
            prototypes=prototypes,
            seen_tasks=seen_tasks,
            proto_type="class",
        )

        scores = torch.tensor([distances[t] for t in seen_tasks])
        order = torch.argsort(scores)[:min(top_k, len(seen_tasks))]
        selected_tasks = [seen_tasks[i] for i in order.tolist()]
        weights = torch.softmax(-scores[order] / tau, dim=0).to(probs[seen_tasks[0]].device)

        final_probs = 0.0
        for w, t in zip(weights, selected_tasks):
            final_probs = final_probs + w * probs[t]

        chosen_task = selected_tasks[0]
        aux["score"] = {TASK_TO_DOMAIN[t]: distances[t] for t in seen_tasks}
        aux["weights"] = {TASK_TO_DOMAIN[t]: float(w.item()) for w, t in zip(weights, selected_tasks)}

    else:
        raise ValueError(f"Unknown policy: {policy}")

    return final_probs, chosen_task, aux


@torch.no_grad()
def evaluate_policy(
    model: MCnn14LateBNAdapter,
    df: pd.DataFrame,
    data_root: str,
    device: str,
    policy: str,
    seen_tasks: List[int],
    temperatures: Optional[Dict[int, float]] = None,
    prototypes: Optional[Dict] = None,
    top_k: int = 2,
    tau: float = 1.0,
) -> Dict:
    model.eval()
    model.set_active_adapters(ADAPTER_BLOCKS)

    y_true = []
    y_pred = []
    chosen_tasks = []
    rows = []

    df = df.reset_index(drop=True)

    for idx, row in df.iterrows():
        wav_path = Path(data_root) / row["filename"]
        target = int(row["new_target"])
        domain = str(row["domain"])
        true_task = DOMAIN_TO_TASK.get(domain, None)

        audio, _ = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
        audio = audio.astype(np.float32)

        outputs = forward_audio_all_tasks(
            model=model,
            audio=audio,
            seen_tasks=seen_tasks,
            device=device,
            temperatures=temperatures,
        )

        final_probs, chosen_task, aux = get_final_probs_and_route(
            outputs=outputs,
            policy=policy,
            seen_tasks=seen_tasks,
            prototypes=prototypes,
            top_k=top_k,
            tau=tau,
            true_task=true_task,
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
            "chosen_task": chosen_task,
            "chosen_domain": TASK_TO_DOMAIN.get(chosen_task, "MIX"),
            "true_task": true_task,
            "true_domain": domain,
            "route_correct": int(chosen_task == true_task)
            if chosen_task in TASK_TO_DOMAIN and true_task is not None
            else np.nan,
            "confidence": conf,
        }

        for task_id in seen_tasks:
            d = TASK_TO_DOMAIN[task_id]
            row_out[f"entropy_{d}"] = outputs["entropies"][task_id]
            row_out[f"cal_entropy_{d}"] = outputs["cal_entropies"][task_id]

        if "score" in aux:
            for k, v in aux["score"].items():
                row_out[f"score_{k}"] = v

        if "weights" in aux:
            for k, v in aux["weights"].items():
                row_out[f"weight_{k}"] = v

        rows.append(row_out)

        if (idx + 1) % 100 == 0:
            print(f"[eval] policy={policy}, top_k={top_k}, tau={tau} | {idx + 1}/{len(df)}")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    chosen_tasks = np.array(chosen_tasks)

    acc = round(float((y_true == y_pred).mean() * 100.0), 2)
    cm = metrics.confusion_matrix(y_true, y_pred, labels=list(range(CLASSES_NUM)))

    route_hist = {}
    for task_id in seen_tasks:
        route_hist[TASK_TO_DOMAIN[task_id]] = int(np.sum(chosen_tasks == task_id))
    route_hist["MIX"] = int(np.sum(chosen_tasks == -1))

    pred_df = pd.DataFrame(rows)

    valid_route = pred_df["route_correct"].dropna()
    router_acc = round(float(valid_route.mean() * 100.0), 2) if len(valid_route) > 0 else np.nan

    return {
        "acc": acc,
        "cm": cm,
        "route_hist": route_hist,
        "router_acc": router_acc,
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


def save_eval_outputs(
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
        "router_acc": result["router_acc"],
        "route_hist": result["route_hist"],
    }

    save_json(compact, out_dir / f"{prefix}_summary.json")
    return compact


def make_policy_configs(args: argparse.Namespace) -> List[Tuple[str, Dict]]:
    configs = []

    # Fixed branches
    configs.append(("fixed_D2", {"policy": "fixed_D2"}))
    configs.append(("fixed_D3", {"policy": "fixed_D3"}))

    # Oracle upper bound
    configs.append(("oracle", {"policy": "oracle"}))

    # Hard routing
    configs.append(("entropy_hard", {"policy": "entropy_hard"}))
    configs.append(("cal_entropy_hard", {"policy": "cal_entropy_hard"}))
    configs.append(("proto_domain_hard", {"policy": "proto_domain_hard"}))
    configs.append(("proto_class_hard", {"policy": "proto_class_hard"}))

    # Uniform ensemble baseline
    configs.append(("all_mean", {"policy": "all_mean"}))

    # MoE-style branch mixture
    for k in args.top_ks:
        for tau in args.taus:
            tau_name = str(tau).replace(".", "p")

            configs.append((
                f"entropy_moe_top{k}_tau{tau_name}",
                {"policy": "entropy_moe", "top_k": k, "tau": tau},
            ))

            configs.append((
                f"cal_entropy_moe_top{k}_tau{tau_name}",
                {"policy": "cal_entropy_moe", "top_k": k, "tau": tau},
            ))

            configs.append((
                f"proto_domain_moe_top{k}_tau{tau_name}",
                {"policy": "proto_domain_moe", "top_k": k, "tau": tau},
            ))

            configs.append((
                f"proto_class_moe_top{k}_tau{tau_name}",
                {"policy": "proto_class_moe", "top_k": k, "tau": tau},
            ))

    return configs


def run_policy_suite(
    model: MCnn14LateBNAdapter,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    data_root: str,
    device: str,
    args: argparse.Namespace,
    out_dir: Path,
) -> Dict:
    ensure_dir(out_dir)

    seen_tasks = ROUTING_TASKS
    eval_domains = ["D2", "D3"]
    class_name_map = get_class_names(test_df)

    print("\n" + "=" * 100)
    print("[1] Fit branch-wise temperatures: D2, D3 only")
    print("=" * 100)

    temperatures = fit_branch_temperatures(
        model=model,
        train_df=train_df,
        data_root=data_root,
        seen_tasks=seen_tasks,
        device=device,
        max_calib_samples=args.max_calib_samples,
    )

    save_json({str(k): v for k, v in temperatures.items()}, out_dir / "temperatures.json")

    print("\n" + "=" * 100)
    print("[2] Build prototypes: D2, D3 only")
    print("=" * 100)

    prototypes = build_prototypes(
        model=model,
        train_df=train_df,
        data_root=data_root,
        seen_tasks=seen_tasks,
        device=device,
        max_proto_samples=args.max_proto_samples,
        normalize=True,
    )

    prototype_info = {
        "domain_proto_norm": {
            TASK_TO_DOMAIN[k]: float(torch.norm(v).item())
            for k, v in prototypes["domain_proto"].items()
        },
        "class_proto_count": {
            TASK_TO_DOMAIN[k]: len(v)
            for k, v in prototypes["class_proto"].items()
        },
    }
    save_json(prototype_info, out_dir / "prototype_info.json")

    print("\n" + "=" * 100)
    print("[3] Evaluate policies")
    print("=" * 100)

    policy_configs = make_policy_configs(args)

    all_results = {}
    summary_rows = []

    for policy_name, cfg in policy_configs:
        print("\n" + "-" * 100)
        print(f"[POLICY] {policy_name}")
        print("-" * 100)

        policy_dir = out_dir / policy_name
        ensure_dir(policy_dir)

        policy_result = {}

        for domain in eval_domains:
            domain_df = test_df[test_df["domain"] == domain].copy()

            print(f"[evaluate] policy={policy_name}, domain={domain}, n={len(domain_df)}")

            res = evaluate_policy(
                model=model,
                df=domain_df,
                data_root=data_root,
                device=device,
                policy=cfg["policy"],
                seen_tasks=seen_tasks,
                temperatures=temperatures,
                prototypes=prototypes,
                top_k=cfg.get("top_k", 2),
                tau=cfg.get("tau", 1.0),
            )

            compact = save_eval_outputs(
                result=res,
                out_dir=policy_dir,
                prefix=domain,
                class_name_map=class_name_map,
            )

            policy_result[domain] = compact

        avg = round(np.mean([policy_result[d]["acc"] for d in eval_domains]), 2)
        avg_router_acc = round(np.nanmean([policy_result[d]["router_acc"] for d in eval_domains]), 2)

        policy_result["Avg"] = avg
        policy_result["Avg_router_acc"] = avg_router_acc

        all_results[policy_name] = policy_result

        row = {
            "policy": policy_name,
            "Avg": avg,
            "Avg_router_acc": avg_router_acc,
        }

        for domain in eval_domains:
            row[f"{domain}_acc"] = policy_result[domain]["acc"]
            row[f"{domain}_router_acc"] = policy_result[domain]["router_acc"]

            for route_domain, count in policy_result[domain]["route_hist"].items():
                row[f"{domain}_route_{route_domain}"] = count

        summary_rows.append(row)

        print(f"[RESULT] {policy_name} | Avg={avg} | Avg_router_acc={avg_router_acc}")

    save_json(all_results, out_dir / "all_policy_results.json")

    summary_df = pd.DataFrame(summary_rows)

    summary_by_avg = summary_df.sort_values("Avg", ascending=False)
    summary_by_router = summary_df.sort_values("Avg_router_acc", ascending=False)

    summary_by_avg.to_csv(out_dir / "summary_by_avg.csv", index=False, encoding="utf-8-sig")
    summary_by_router.to_csv(out_dir / "summary_by_router_acc.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print("[SUMMARY BY AVG]")
    print(summary_by_avg.to_string(index=False))

    print("\n[SUMMARY BY ROUTER ACC]")
    print(summary_by_router.to_string(index=False))
    print("=" * 100)

    return all_results


# =========================================================
# Main experiment
# =========================================================
def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    device = "cuda" if (args.cuda and torch.cuda.is_available()) else "cpu"
    print(f"[device] {device}")

    save_dir = Path(args.save_dir)
    ensure_dir(save_dir)

    train_df = load_split_df(args.data_root, "development_train.txt")
    test_df = load_split_df(args.data_root, "development_test.txt")

    d2_train = train_df[train_df["domain"] == "D2"].copy()
    d3_train = train_df[train_df["domain"] == "D3"].copy()
    d2_test = test_df[test_df["domain"] == "D2"].copy()
    d3_test = test_df[test_df["domain"] == "D3"].copy()

    print(f"[data] D2 train: {len(d2_train)}")
    print(f"[data] D3 train: {len(d3_train)}")
    print(f"[data] D2 test : {len(d2_test)}")
    print(f"[data] D3 test : {len(d3_test)}")

    save_json(
        {
            "data_root": args.data_root,
            "d1_checkpoint": args.d1_checkpoint,
            "save_dir": args.save_dir,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr_incremental": args.lr_incremental,
            "min_lr": args.min_lr,
            "weight_decay": args.weight_decay,
            "adapter_reduction": args.adapter_reduction,
            "num_workers": args.num_workers,
            "seed": args.seed,
            "aug_mode": args.aug_mode,
            "use_mixup": args.use_mixup,
            "mixup_alpha": args.mixup_alpha,
            "freeze_bn_stats": args.freeze_bn_stats,
            "init_bn_from_d1": args.init_bn_from_d1,
            "adapter_blocks": ADAPTER_BLOCKS,
            "routing_tasks": [TASK_TO_DOMAIN[t] for t in ROUTING_TASKS],
            "top_ks": args.top_ks,
            "taus": args.taus,
            "max_calib_samples": args.max_calib_samples,
            "max_proto_samples": args.max_proto_samples,
        },
        save_dir / "config.json",
    )

    model = build_model(args.adapter_reduction, device)

    print("\n" + "=" * 100)
    print("[LOAD] D1 checkpoint")
    print("=" * 100)
    load_checkpoint(model, args.d1_checkpoint, device, strict=False)

    # =====================================================
    # STEP 1. Train D2 adapter from D1 checkpoint
    # =====================================================
    d2_out_dir = save_dir / "step1_train_D2"

    d2_result = train_adapter_for_domain(
        model=model,
        train_df=train_df,
        test_df=test_df,
        args=args,
        device=device,
        domain="D2",
        out_dir=d2_out_dir,
    )

    print("\n" + "=" * 100)
    print("[STEP 1 DONE] after D2 checkpoint saved")
    print(d2_result)
    print("=" * 100)

    torch.save(model.state_dict(), save_dir / "after_D2.pth")

    # =====================================================
    # STEP 2. Train D3 adapter from after_D2 model
    # =====================================================
    d3_out_dir = save_dir / "step2_train_D3"

    d3_result = train_adapter_for_domain(
        model=model,
        train_df=train_df,
        test_df=test_df,
        args=args,
        device=device,
        domain="D3",
        out_dir=d3_out_dir,
    )

    print("\n" + "=" * 100)
    print("[STEP 2 DONE] after D3 checkpoint saved")
    print(d3_result)
    print("=" * 100)

    torch.save(model.state_dict(), save_dir / "after_D3.pth")
    torch.save(model.state_dict(), save_dir / "best.pth")

    save_json(
        {
            "D2": d2_result,
            "D3": d3_result,
        },
        save_dir / "train_summary.json",
    )

    # =====================================================
    # STEP 3. Final D2/D3 routing + MoE evaluation
    # =====================================================
    print("\n" + "=" * 100)
    print("[FINAL POLICY EVALUATION]")
    print("=" * 100)

    final_results = run_policy_suite(
        model=model,
        train_df=train_df,
        test_df=test_df,
        data_root=args.data_root,
        device=device,
        args=args,
        out_dir=save_dir / "final_policy_eval",
    )

    save_json(final_results, save_dir / "final_results.json")

    print("\n[DONE]")
    print(f"Saved to: {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--d1_checkpoint", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)

    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr_incremental", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adapter_reduction", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1193)

    parser.add_argument(
        "--aug_mode",
        type=str,
        default="none",
        choices=[
            "none",
            "crop",
            "gain_shift",
            "crop_gain_shift",
            "crop_gain_shift_noise",
            "noise",
        ],
    )

    parser.add_argument("--use_mixup", action="store_true", default=False)
    parser.add_argument("--mixup_alpha", type=float, default=0.2)

    parser.add_argument(
        "--freeze_bn_stats",
        action="store_true",
        default=False,
        help="If set, BN running mean/var are frozen during adapter training.",
    )

    parser.add_argument(
        "--init_bn_from_d1",
        action="store_true",
        default=False,
        help="If set, initialize D2/D3 BN state from D1 BN state in checkpoint.",
    )

    parser.add_argument(
        "--top_ks",
        type=int,
        nargs="+",
        default=[2],
        help="Top-k values for MoE-style branch mixture. With D2/D3 only, top_k=2 means both branches are mixed.",
    )

    parser.add_argument(
        "--taus",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 2.0],
        help="Softmax temperature for MoE-style weighting.",
    )

    parser.add_argument(
        "--max_calib_samples",
        type=int,
        default=-1,
        help="Max samples per domain for temperature scaling. -1 means all.",
    )

    parser.add_argument(
        "--max_proto_samples",
        type=int,
        default=-1,
        help="Max samples per domain for prototype building. -1 means all.",
    )

    parser.add_argument("--cuda", action="store_true", default=False)

    args = parser.parse_args()
    main(args)
