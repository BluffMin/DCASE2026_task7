from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from config import AudioConfig, DataConfig, load_split_dataframe
from data.dataset import DILDataset, collate_audio_samples
from models.geometry import GeometryTask7Cnn14
from runners.geometry_runner import GeometryTrainer, TrainConfig


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--save-dir", type=str, default="runs/geometry")
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    audio_cfg = AudioConfig()
    data_cfg = DataConfig(data_root=Path(args.data_root))

    train_df = load_split_dataframe(data_cfg.train_split_path)
    val_df = load_split_dataframe(data_cfg.test_split_path)

    train_ds = DILDataset(train_df, audio_root=data_cfg.data_root, audio_cfg=audio_cfg, classes_num=10)
    val_ds = DILDataset(val_df, audio_root=data_cfg.data_root, audio_cfg=audio_cfg, classes_num=10)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_audio_samples,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_audio_samples,
    )

    model = GeometryTask7Cnn14(
        sample_rate=audio_cfg.sample_rate,
        window_size=audio_cfg.window_size,
        hop_size=audio_cfg.hop_size,
        mel_bins=audio_cfg.mel_bins,
        fmin=audio_cfg.fmin,
        fmax=audio_cfg.fmax,
        classes_num=10,
        num_tasks=3,
        embed_dim=256,
    )
    model.unfreeze_for_task(args.task_id)

    trainer = GeometryTrainer(
        model=model,
        train_cfg=TrainConfig(
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
        ),
        save_dir=Path(args.save_dir),
    )
    trainer.fit(train_loader=train_loader, val_loader=val_loader, task_id=args.task_id)


if __name__ == "__main__":
    main()
