"""Geometry-preserving domain-incremental audio model for DCASE2026 Task7.

Drop-in alternative to the baseline domain_net.py.
Main ideas:
1) shared CNN backbone
2) lightweight task-specific residual adapters
3) learnable class anchors
4) optional retrieval of embeddings for auxiliary losses
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank


def init_layer(layer: nn.Module) -> None:
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(layer.weight)
        if getattr(layer, "bias", None) is not None:
            layer.bias.data.fill_(0.0)


def init_bn(bn: nn.Module) -> None:
    if isinstance(bn, (nn.BatchNorm1d, nn.BatchNorm2d)):
        bn.bias.data.fill_(0.0)
        bn.weight.data.fill_(1.0)


class ResidualAdapter(nn.Module):
    def __init__(self, channels: int, bottleneck_ratio: int = 4):
        super().__init__()
        hidden = max(channels // bottleneck_ratio, 8)
        self.down = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.up = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)
        self.scale = nn.Parameter(torch.tensor(0.1))
        init_layer(self.down)
        init_layer(self.up)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * self.up(self.act(self.down(x)))


class ConvBlockGeometry(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_tasks: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.adapters = nn.ModuleList([ResidualAdapter(out_ch) for _ in range(num_tasks)])
        self.dropout = nn.Dropout(p=0.2)

        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = x + self.adapters[task_id](x)
        x = F.avg_pool2d(x, kernel_size=(2, 2))
        x = self.dropout(x)
        return x


class GeometryCnn14(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        window_size: int,
        hop_size: int,
        mel_bins: int,
        fmin: int,
        fmax: int,
        classes_num: int,
        num_tasks: int,
        embedding_dim: int = 512,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.classes_num = classes_num
        self.embedding_dim = embedding_dim

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

        self.bn0 = nn.BatchNorm2d(mel_bins)
        init_bn(self.bn0)

        channels = [1, 64, 128, 256, 512, 1024, 2048]
        self.blocks = nn.ModuleList(
            [ConvBlockGeometry(channels[i], channels[i + 1], num_tasks) for i in range(len(channels) - 1)]
        )

        self.fc1 = nn.Linear(2048, embedding_dim)
        self.fc_out = nn.Linear(embedding_dim, classes_num)
        self.class_anchors = nn.Parameter(torch.randn(classes_num, embedding_dim))
        self.anchor_scale = nn.Parameter(torch.tensor(10.0))

        init_layer(self.fc1)
        init_layer(self.fc_out)
        nn.init.xavier_uniform_(self.class_anchors)

    def freeze_shared(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_task(self, task_id: int, train_backbone_for_first_task: bool = True) -> None:
        self.freeze_shared()

        if train_backbone_for_first_task and task_id == 0:
            for module in [self.bn0, self.fc1, self.fc_out]:
                for p in module.parameters():
                    p.requires_grad = True
            self.class_anchors.requires_grad = True
            for block in self.blocks:
                for p in block.conv1.parameters():
                    p.requires_grad = True
                for p in block.conv2.parameters():
                    p.requires_grad = True
                for p in block.bn1.parameters():
                    p.requires_grad = True
                for p in block.bn2.parameters():
                    p.requires_grad = True

        for block in self.blocks:
            for p in block.adapters[task_id].parameters():
                p.requires_grad = True

        # keep classifier and anchors trainable in every phase
        for module in [self.fc1, self.fc_out]:
            for p in module.parameters():
                p.requires_grad = True
        self.class_anchors.requires_grad = True
        self.anchor_scale.requires_grad = True

    def extract_embedding(self, waveform: torch.Tensor, task_id: int) -> torch.Tensor:
        x = self.spectrogram_extractor(waveform)
        x = self.logmel_extractor(x)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        for block in self.blocks:
            x = block(x, task_id)

        x = torch.mean(x, dim=3)
        x1, _ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.normalize(x, dim=-1)
        return x

    def compute_anchor_logits(self, embedding: torch.Tensor) -> torch.Tensor:
        anchors = F.normalize(self.class_anchors, dim=-1)
        return self.anchor_scale * embedding @ anchors.t()

    def forward(
        self, waveform: torch.Tensor, task_id: int, return_embedding: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        embedding = self.extract_embedding(waveform, task_id)
        classifier_logits = self.fc_out(embedding)
        anchor_logits = self.compute_anchor_logits(embedding)
        logits = 0.5 * classifier_logits + 0.5 * anchor_logits
        if return_embedding:
            return logits, embedding
        return logits
