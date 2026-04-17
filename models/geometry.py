from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_cnn import Task7BaseCnn14, init_layer


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


class GeometryTask7Cnn14(Task7BaseCnn14):
    """
    원본 geometry 실험에서 핵심이었던 구조:
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
        num_tasks: int = 3,
        embed_dim: int = 256,
    ):
        super().__init__(
            sample_rate=sample_rate,
            window_size=window_size,
            hop_size=hop_size,
            mel_bins=mel_bins,
            fmin=fmin,
            fmax=fmax,
            classes_num=classes_num,
            num_tasks=num_tasks,
        )
        self.embed_dim = embed_dim
        block_out_channels = [64, 128, 256, 512, 1024, 2048]
        self.adapters = nn.ModuleList([
            nn.ModuleList([ResidualAdapter2d(ch, reduction=4) for _ in range(num_tasks)])
            for ch in block_out_channels
        ])
        self.proj = nn.Linear(2048, embed_dim)
        self.class_anchors = nn.Parameter(torch.randn(classes_num, embed_dim))
        init_layer(self.proj)

    def extract_backbone_feature(self, waveform: torch.Tensor, task_id: int) -> torch.Tensor:
        x = self.spectrogram_extractor(waveform)
        x = self.logmel_extractor(x)
        x = x.transpose(1, 3)
        x = self.bn0[task_id](x)
        x = x.transpose(1, 3)

        for block, block_adapters in zip(self.blocks, self.adapters):
            x = block.conv1(x)
            x = block.bnF[task_id](x)
            x = F.relu_(x)

            x = block.conv2(x)
            x = block.bnS[task_id](x)
            x = F.relu_(x)

            x = block_adapters[task_id](x)
            x = F.avg_pool2d(x, kernel_size=(2, 2))
            x = F.dropout(x, p=0.2, training=self.training)

        x = torch.mean(x, dim=3)
        x1, _ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        return x1 + x2

    def extract_embedding(self, waveform: torch.Tensor, task_id: int) -> torch.Tensor:
        feat = self.extract_backbone_feature(waveform, task_id)
        emb = self.proj(feat)
        return F.normalize(emb, dim=-1)

    def forward(self, waveform: torch.Tensor, task_id: int, return_embedding: bool = False):
        feat = self.extract_backbone_feature(waveform, task_id)
        logits = self.fc(feat)
        emb = F.normalize(self.proj(feat), dim=-1)
        if return_embedding:
            return logits, emb
        return logits

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

        for block, block_adapters in zip(self.blocks, self.adapters):
            for p in block.bnF[task_id].parameters():
                p.requires_grad = True
            for p in block.bnS[task_id].parameters():
                p.requires_grad = True
            for p in block_adapters[task_id].parameters():
                p.requires_grad = True

        if train_fc:
            for p in self.fc.parameters():
                p.requires_grad = True
        if train_proj:
            for p in self.proj.parameters():
                p.requires_grad = True
        if train_anchors:
            self.class_anchors.requires_grad = True

    def partial_load_from_checkpoint(self, checkpoint_state: Dict[str, torch.Tensor]) -> Tuple[List[str], List[str]]:
        model_state = self.state_dict()
        loaded, skipped = [], []
        for key, value in checkpoint_state.items():
            if key in model_state and model_state[key].shape == value.shape:
                model_state[key] = value
                loaded.append(key)
            else:
                skipped.append(key)
        self.load_state_dict(model_state, strict=False)
        return loaded, skipped
