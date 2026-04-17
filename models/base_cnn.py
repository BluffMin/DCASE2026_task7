from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import LogmelFilterBank, Spectrogram


def init_layer(layer: nn.Module) -> None:
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(layer.weight)
        if getattr(layer, "bias", None) is not None:
            layer.bias.data.fill_(0.0)


def init_bn(bn: nn.Module) -> None:
    if isinstance(bn, (nn.BatchNorm1d, nn.BatchNorm2d)):
        bn.bias.data.fill_(0.0)
        bn.weight.data.fill_(1.0)


class TaskConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_tasks: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bnF = nn.ModuleList([nn.BatchNorm2d(out_channels) for _ in range(num_tasks)])
        self.bnS = nn.ModuleList([nn.BatchNorm2d(out_channels) for _ in range(num_tasks)])

        init_layer(self.conv1)
        init_layer(self.conv2)
        for bn in self.bnF:
            init_bn(bn)
        for bn in self.bnS:
            init_bn(bn)

    def forward(self, x: torch.Tensor, task_id: int, pool_size=(2, 2), pool_type: str = "avg") -> torch.Tensor:
        x = F.relu_(self.bnF[task_id](self.conv1(x)))
        x = F.relu_(self.bnS[task_id](self.conv2(x)))

        if pool_type == "max":
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg":
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg+max":
            x = F.avg_pool2d(x, kernel_size=pool_size) + F.max_pool2d(x, kernel_size=pool_size)
        else:
            raise ValueError(f"Unsupported pool_type: {pool_type}")
        return x


class Task7BaseCnn14(nn.Module):
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
    ):
        super().__init__()
        self.classes_num = classes_num
        self.num_tasks = num_tasks

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

        self.bn0 = nn.ModuleList([nn.BatchNorm2d(mel_bins) for _ in range(num_tasks)])
        for bn in self.bn0:
            init_bn(bn)

        channels = [64, 128, 256, 512, 1024, 2048]
        self.blocks = nn.ModuleList([
            TaskConvBlock(1, channels[0], num_tasks),
            TaskConvBlock(channels[0], channels[1], num_tasks),
            TaskConvBlock(channels[1], channels[2], num_tasks),
            TaskConvBlock(channels[2], channels[3], num_tasks),
            TaskConvBlock(channels[3], channels[4], num_tasks),
            TaskConvBlock(channels[4], channels[5], num_tasks),
        ])
        self.fc = nn.Linear(2048, classes_num)
        init_layer(self.fc)

    def extract_backbone_feature(self, waveform: torch.Tensor, task_id: int) -> torch.Tensor:
        x = self.spectrogram_extractor(waveform)
        x = self.logmel_extractor(x)
        x = x.transpose(1, 3)
        x = self.bn0[task_id](x)
        x = x.transpose(1, 3)

        for block in self.blocks:
            x = block(x, task_id=task_id, pool_size=(2, 2), pool_type="avg")
            x = F.dropout(x, p=0.2, training=self.training)

        x = torch.mean(x, dim=3)
        x1, _ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        return x1 + x2

    def forward(self, waveform: torch.Tensor, task_id: int) -> torch.Tensor:
        feat = self.extract_backbone_feature(waveform, task_id=task_id)
        return self.fc(feat)
