from __future__ import annotations

import torch
import torch.nn as nn

from models.base_cnn import init_layer


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
        return x + self.scale * self.up(self.act(self.down(x)))


class LoRAConv2d(nn.Module):
    """
    원본 auxiliary-study 브랜치의 LoRA-style conv 아이디어를
    분리해 놓은 최소 구현 버전.
    """
    def __init__(self, in_channels: int, out_channels: int, rank: int = 8, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.a = nn.Conv2d(in_channels, rank, kernel_size=1, bias=False)
        self.b = nn.Conv2d(rank, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.scale = nn.Parameter(torch.tensor(0.1))
        init_layer(self.a)
        init_layer(self.b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * self.b(self.a(x))
