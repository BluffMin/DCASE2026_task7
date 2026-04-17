# domain_net_geometry.py
"""
Geometry-preserving extension of the DCASE 2026 Task 7 baseline.

Key ideas:
- keep the baseline-compatible shared CNN + domain-specific BN layout
- add lightweight task-specific residual adapters
- add normalized embedding projection and learnable class anchors
- expose helper functions for partial loading from baseline checkpoint_D1.pth
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank


def init_layer(layer: nn.Module):
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(layer.weight)
        if getattr(layer, "bias", None) is not None:
            layer.bias.data.fill_(0.0)


def init_bn(bn: nn.BatchNorm2d):
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


class ResidualAdapter2d(nn.Module):
    """
    Lightweight residual adapter:
    x + conv1x1(relu(conv1x1(x))) * scale
    """

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


class ConvBlockGeometry(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, nb_tasks: int):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )

        # baseline-compatible task-specific BN
        self.bnF = nn.ModuleList([nn.BatchNorm2d(out_channels) for _ in range(nb_tasks)])
        self.bnS = nn.ModuleList([nn.BatchNorm2d(out_channels) for _ in range(nb_tasks)])

        # new task-specific adapters
        self.adapter = nn.ModuleList([ResidualAdapter2d(out_channels) for _ in range(nb_tasks)])

        init_layer(self.conv1)
        init_layer(self.conv2)
        for bn in self.bnF:
            init_bn(bn)
        for bn in self.bnS:
            init_bn(bn)

    def forward(
        self,
        x: torch.Tensor,
        task: int,
        pool_size=(2, 2),
        pool_type: str = "avg",
    ) -> torch.Tensor:
        x = F.relu_(self.bnF[task](self.conv1(x)))
        x = F.relu_(self.bnS[task](self.conv2(x)))
        x = self.adapter[task](x)

        if pool_type == "max":
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg":
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg+max":
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise ValueError("Incorrect pool_type")

        return x


class GeometryCnn14(nn.Module):
    """
    Baseline-compatible backbone + geometry head.

    Task mapping:
    0 -> D1
    1 -> D2
    2 -> D3
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
        embed_dim: int = 256,
    ):
        super().__init__()
        self.nb_tasks = nb_tasks
        self.classes_num = classes_num
        self.embed_dim = embed_dim

        window = "hann"
        center = True
        pad_mode = "reflect"
        ref = 1.0
        amin = 1e-10
        top_db = None

        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax,
            ref=ref,
            amin=amin,
            top_db=top_db,
            freeze_parameters=True,
        )

        # IMPORTANT: channel is mel_bins after transpose(1, 3)
        self.bn0 = nn.ModuleList([nn.BatchNorm2d(mel_bins) for _ in range(nb_tasks)])

        self.conv_block1 = ConvBlockGeometry(1, 64, nb_tasks)
        self.conv_block2 = ConvBlockGeometry(64, 128, nb_tasks)
        self.conv_block3 = ConvBlockGeometry(128, 256, nb_tasks)
        self.conv_block4 = ConvBlockGeometry(256, 512, nb_tasks)
        self.conv_block5 = ConvBlockGeometry(512, 1024, nb_tasks)
        self.conv_block6 = ConvBlockGeometry(1024, 2048, nb_tasks)

        self.fc = nn.Linear(2048, classes_num)

        # geometry head
        self.proj = nn.Linear(2048, embed_dim)
        self.class_anchors = nn.Parameter(torch.randn(classes_num, embed_dim))

        init_layer(self.fc)
        init_layer(self.proj)
        for bn in self.bn0:
            init_bn(bn)

    # -------------------------------------------------
    # loading / freezing helpers
    # -------------------------------------------------
    def init_anchors_from_fc(self):
        """
        Initialize anchors from fc weights if dimensions allow,
        otherwise keep random initialization.
        """
        with torch.no_grad():
            if self.fc.weight.shape[1] == 2048 and self.proj.out_features == self.embed_dim:
                # project class weights into anchor space in a simple way
                fc_w = self.fc.weight.data  # [C, 2048]
                pseudo = self.proj(fc_w)    # [C, embed_dim]
                self.class_anchors.data.copy_(F.normalize(pseudo, dim=-1))

    def freeze_all(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_for_task(self, task_id: int, train_fc: bool = True, train_proj: bool = True, train_anchors: bool = True):
        """
        Fine-tuning policy:
        - keep shared conv weights frozen after D1 init
        - train only:
          * current task BN
          * current task adapters
          * fc / proj / anchors
        """
        self.freeze_all()

        # current-task bn0
        for p in self.bn0[task_id].parameters():
            p.requires_grad = True

        # current-task block-specific BN + adapters
        blocks = [
            self.conv_block1,
            self.conv_block2,
            self.conv_block3,
            self.conv_block4,
            self.conv_block5,
            self.conv_block6,
        ]
        for block in blocks:
            for p in block.bnF[task_id].parameters():
                p.requires_grad = True
            for p in block.bnS[task_id].parameters():
                p.requires_grad = True
            for p in block.adapter[task_id].parameters():
                p.requires_grad = True

        if train_fc:
            for p in self.fc.parameters():
                p.requires_grad = True

        if train_proj:
            for p in self.proj.parameters():
                p.requires_grad = True

        if train_anchors:
            self.class_anchors.requires_grad = True

    def partial_load_from_baseline(self, ckpt: Dict[str, torch.Tensor]) -> Tuple[List[str], List[str]]:
        """
        Load baseline-compatible weights from checkpoint_D1.pth.

        Compatible names from baseline:
        - bn0.{task}
        - conv_block{n}.conv{1|2}.*
        - conv_block{n}.bnF.{task}.*
        - conv_block{n}.bnS.{task}.*
        - fc.*
        """
        model_state = self.state_dict()
        loaded, skipped = [], []

        for k, v in ckpt.items():
            if k in model_state and model_state[k].shape == v.shape:
                model_state[k] = v
                loaded.append(k)
            else:
                skipped.append(k)

        self.load_state_dict(model_state, strict=False)
        return loaded, skipped

    # -------------------------------------------------
    # forward
    # -------------------------------------------------
    def extract_backbone_feature(self, waveform: torch.Tensor, task: int) -> torch.Tensor:
        x = self.spectrogram_extractor(waveform)   # [B, 1, T, F]
        x = self.logmel_extractor(x)               # [B, 1, T, mel]
        x = x.transpose(1, 3)                      # [B, mel, T, 1]
        x = self.bn0[task](x)
        x = x.transpose(1, 3)                      # [B, 1, T, mel]

        x = self.conv_block1(x, task=task, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, task=task, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, task=task, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, task=task, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, task=task, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, task=task, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)

        x = torch.mean(x, dim=3)     # [B, C, T]
        x1, _ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        feat = x1 + x2               # [B, 2048]
        return feat

    def extract_embedding(self, waveform: torch.Tensor, task: int) -> torch.Tensor:
        feat = self.extract_backbone_feature(waveform, task)
        emb = self.proj(feat)
        emb = F.normalize(emb, dim=-1)
        return emb

    def forward(self, waveform: torch.Tensor, task: int, return_embedding: bool = False):
        feat = self.extract_backbone_feature(waveform, task)
        logits = self.fc(feat)
        emb = F.normalize(self.proj(feat), dim=-1)

        if return_embedding:
            return logits, emb
        return logits