import os
import sys
import time
import json
import random
import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import librosa

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn import metrics

sys.path.insert(0, os.path.dirname(__file__))
from domain_net import MCnn14


SAMPLE_RATE = 32000
CLIP_SECONDS = 4
CLIP_SAMPLES = SAMPLE_RATE * CLIP_SECONDS

MEL_BINS = 64
FMIN = 50
FMAX = 14000
WINDOW_SIZE = 1024
HOP_SIZE = 320
CLASSES_NUM = 10

DOMAIN_TO_TASK = {
    "D1": 0,
    "D2": 1,
    "D3": 2,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pad_or_truncate(x: np.ndarray, max_len: int) -> np.ndarray:
    if len(x) < max_len:
        pad = np.zeros(max_len - len(x), dtype=x.dtype)
        x = np.concatenate([x, pad], axis=0)
    else:
        x = x[:max_len]
    return x


def load_split_df(data_root: str, split_name: str) -> pd.DataFrame:
    split_path = Path(data_root) / "evaluation_setup" / split_name
    df = pd.read_csv(
        split_path,
        sep="\t",
        header=None,
        names=["filename", "target", "domain", "new_target"],
    )
    return df


def load_checkpoint_flexible(model: nn.Module, ckpt_path: str, device: str) -> None:
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=False)


def make_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float = 0.0,
) -> optim.Optimizer:
    return torch.optim.Adam(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=weight_decay,
        amsgrad=True,
    )


class Task7WaveDataset(Dataset):
    def __init__(self, df: pd.DataFrame, data_root: str):
        self.df = df.reset_index(drop=True)
        self.data_root = Path(data_root)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        rel_path = row["filename"]
        wav_path = self.data_root / rel_path

        audio, _ = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
        audio = pad_or_truncate(audio, CLIP_SAMPLES).astype(np.float32)

        label = int(row["new_target"])
        onehot = np.zeros(CLASSES_NUM, dtype=np.float32)
        onehot[label] = 1.0

        return torch.from_numpy(audio), torch.from_numpy(onehot), rel_path


def compute_accuracy(
    model: nn.Module,
    loader: DataLoader,
    task_idx: int,
    device: str,
) -> Tuple[float, np.ndarray]:
    model.eval()

    total = 0
    correct = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for audio, target, _ in loader:
            audio = audio.to(device, non_blocking=True).float()
            target = target.to(device, non_blocking=True).float()

            logits = model(audio, task_idx)
            pred = torch.argmax(logits, dim=1)
            gt = torch.argmax(target, dim=1)

            total += gt.size(0)
            correct += (pred == gt).sum().item()

            y_true.append(gt.cpu().numpy())
            y_pred.append(pred.cpu().numpy())

    y_true = np.concatenate(y_true, axis=0) if y_true else np.array([])
    y_pred = np.concatenate(y_pred, axis=0) if y_pred else np.array([])

    cm = metrics.confusion_matrix(
        y_true, y_pred, labels=list(range(CLASSES_NUM))
    ) if len(y_true) > 0 else np.zeros((CLASSES_NUM, CLASSES_NUM), dtype=np.int64)

    acc = 100.0 * correct / max(total, 1)
    return round(acc, 2), cm


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    task_idx: int,
    device: str,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for audio, target, _ in loader:
        audio = audio.to(device, non_blocking=True).float()
        target = target.to(device, non_blocking=True).float()
        target_idx = torch.argmax(target, dim=1)

        optimizer.zero_grad()
        logits = model(audio, task_idx)
        loss = criterion(logits, target_idx)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        pred = torch.argmax(logits, dim=1)
        correct += (pred == target_idx).sum().item()
        total += target_idx.size(0)

    epoch_loss = running_loss / max(len(loader), 1)
    epoch_acc = 100.0 * correct / max(total, 1)

    return round(epoch_loss, 6), round(epoch_acc, 2)


def run_independent_domain_experiment(
    domain_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    args: argparse.Namespace,
    device: str,
) -> Dict:
    task_idx = DOMAIN_TO_TASK[domain_name]
    save_dir = Path(args.save_dir) / domain_name
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print(f"[domain] {domain_name}")
    print(f"[samples] train={len(train_df)}, test={len(test_df)}")
    print(f"[task_idx] {task_idx}")

    train_dataset = Task7WaveDataset(train_df, args.data_root)
    test_dataset = Task7WaveDataset(test_df, args.data_root)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = MCnn14(
        sample_rate=SAMPLE_RATE,
        window_size=WINDOW_SIZE,
        hop_size=HOP_SIZE,
        mel_bins=MEL_BINS,
        fmin=FMIN,
        fmax=FMAX,
        classes_num=CLASSES_NUM,
        nb_tasks=3,
    ).to(device)

    load_checkpoint_flexible(model, args.d1_checkpoint, device)

    for p in model.parameters():
        p.requires_grad = True

    optimizer = make_optimizer(
        model,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr,
    )

    criterion = nn.CrossEntropyLoss()

    best_acc = -1.0
    best_epoch = -1
    history = []

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            task_idx=task_idx,
            device=device,
        )

        val_acc, cm = compute_accuracy(model, test_loader, task_idx, device)
        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), save_dir / f"best_{domain_name}.pth")
            np.save(save_dir / f"best_{domain_name}_cm.npy", cm)

        row = {
            "domain": domain_name,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_acc": round(val_acc, 2),
            "best_acc": round(best_acc, 2),
            "best_epoch": best_epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_sec": round(time.time() - start_time, 2),
        }
        history.append(row)
        print(row)

    summary = {
        "domain": domain_name,
        "best_acc": round(best_acc, 2),
        "best_epoch": best_epoch,
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "history": history,
        "best_ckpt": str(save_dir / f"best_{domain_name}.pth"),
    }

    with open(save_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[best] {domain_name} | best_acc={best_acc:.2f} @ epoch {best_epoch}")
    return summary


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = "cuda" if (args.cuda and torch.cuda.is_available()) else "cpu"
    print(f"[device] {device}")

    full_train_df = load_split_df(args.data_root, "development_train.txt")
    full_test_df = load_split_df(args.data_root, "development_test.txt")

    results = []

    for domain_name in args.domains:
        train_df = full_train_df[full_train_df["domain"] == domain_name].copy()
        test_df = full_test_df[full_test_df["domain"] == domain_name].copy()

        if len(train_df) == 0 or len(test_df) == 0:
            raise ValueError(f"{domain_name} data is empty. Check split files and data_root.")

        result = run_independent_domain_experiment(
            domain_name=domain_name,
            train_df=train_df,
            test_df=test_df,
            args=args,
            device=device,
        )
        results.append(result)

    final_summary = {
        "experiment_name": "independent_domain_full_finetuning_upperbound",
        "domains": args.domains,
        "results": results,
    }

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.save_dir) / "all_results.json", "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 90)
    print("[FINAL SUMMARY]")
    for r in results:
        print(f"{r['domain']}: best_acc={r['best_acc']} @ epoch {r['best_epoch']}")
    print("=" * 90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--d1_checkpoint", type=str, required=True)
    parser.add_argument("--domains", type=str, nargs="+", default=["D2", "D3"], choices=["D2", "D3"])

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1193)
    parser.add_argument("--save_dir", type=str, default="./runs/independent_upperbound")
    parser.add_argument("--cuda", action="store_true", default=False)

    args = parser.parse_args()
    main(args)