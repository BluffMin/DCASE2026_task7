from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

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


class LoRAConv2d(nn.Module):
    """A lightweight LoRA-style residual for Conv2d.

    Base conv is always present. The low-rank branch is:
        x -> A(1x1) -> B(kxk) -> scale
    which is added to the frozen/full conv output.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
        rank: int = 8,
        lora_alpha: float = 8.0,
        num_tasks: int = 1,
    ):
        super().__init__()
        self.base = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        init_layer(self.base)
        self.rank = rank
        self.scaling = lora_alpha / max(rank, 1)
        self.lora_A = nn.ModuleList(
            [nn.Conv2d(in_channels, rank, kernel_size=1, bias=False) for _ in range(num_tasks)]
        )
        self.lora_B = nn.ModuleList(
            [
                nn.Conv2d(rank, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
                for _ in range(num_tasks)
            ]
        )
        for a in self.lora_A:
            nn.init.kaiming_uniform_(a.weight, a=5**0.5)
        for b in self.lora_B:
            nn.init.zeros_(b.weight)

    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        out = self.base(x)
        if self.rank > 0:
            out = out + self.scaling * self.lora_B[task_id](self.lora_A[task_id](x))
        return out

    def task_parameters(self, task_id: int) -> List[nn.Parameter]:
        return list(self.lora_A[task_id].parameters()) + list(self.lora_B[task_id].parameters())


class TaskBatchNorm2d(nn.Module):
    """Per-task BN with optional adaptive interpolation or EMA memory stats.

    mode:
      - domain: ordinary task-specific BN
      - adaptive: interpolate running stats and current-batch stats at eval
      - memory: use EMA memory stats gathered online during train/eval
    """

    def __init__(
        self,
        num_features: int,
        num_tasks: int,
        mode: str = "domain",
        alpha: float = 0.7,
        memory_momentum: float = 0.9,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.mode = mode
        self.alpha = alpha
        self.memory_momentum = memory_momentum
        self.eps = eps
        self.task_bns = nn.ModuleList([nn.BatchNorm2d(num_features) for _ in range(num_tasks)])
        for bn in self.task_bns:
            init_bn(bn)
        self.register_buffer("memory_mean", torch.zeros(num_tasks, num_features))
        self.register_buffer("memory_var", torch.ones(num_tasks, num_features))
        self.register_buffer("memory_ready", torch.zeros(num_tasks, dtype=torch.bool))

    def _compute_batch_stats(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = x.mean(dim=(0, 2, 3))
        var = x.var(dim=(0, 2, 3), unbiased=False)
        return mean, var

    def _update_memory(self, task_id: int, mean: torch.Tensor, var: torch.Tensor) -> None:
        if not self.memory_ready[task_id]:
            self.memory_mean[task_id].copy_(mean.detach())
            self.memory_var[task_id].copy_(var.detach())
            self.memory_ready[task_id] = True
            return
        mm = self.memory_momentum
        self.memory_mean[task_id].mul_(mm).add_(mean.detach() * (1.0 - mm))
        self.memory_var[task_id].mul_(mm).add_(var.detach() * (1.0 - mm))

    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        bn = self.task_bns[task_id]
        if self.mode == "domain":
            return bn(x)

        if self.training:
            out = bn(x)
            mean, var = self._compute_batch_stats(x)
            if self.mode == "memory":
                self._update_memory(task_id, mean, var)
            return out

        # eval path
        current_mean, current_var = self._compute_batch_stats(x)
        if self.mode == "adaptive":
            use_mean = self.alpha * bn.running_mean + (1.0 - self.alpha) * current_mean
            use_var = self.alpha * bn.running_var + (1.0 - self.alpha) * current_var
        elif self.mode == "memory" and self.memory_ready[task_id]:
            use_mean = self.alpha * self.memory_mean[task_id] + (1.0 - self.alpha) * current_mean
            use_var = self.alpha * self.memory_var[task_id] + (1.0 - self.alpha) * current_var
        else:
            use_mean = bn.running_mean
            use_var = bn.running_var

        return F.batch_norm(
            x,
            running_mean=use_mean,
            running_var=use_var,
            weight=bn.weight,
            bias=bn.bias,
            training=False,
            momentum=0.0,
            eps=self.eps,
        )

    def task_parameters(self, task_id: int) -> List[nn.Parameter]:
        return list(self.task_bns[task_id].parameters())


class ConvBlockPEFT(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        num_tasks: int,
        peft_mode: str = "residual",
        bn_mode: str = "domain",
        lora_rank: int = 8,
        lora_alpha: float = 8.0,
        bn_alpha: float = 0.7,
        memory_momentum: float = 0.9,
    ):
        super().__init__()
        self.peft_mode = peft_mode
        self.use_lora = peft_mode in {"lora", "hybrid"}
        self.use_adapter = peft_mode in {"residual", "hybrid"}

        if self.use_lora:
            self.conv1 = LoRAConv2d(
                in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False,
                rank=lora_rank, lora_alpha=lora_alpha, num_tasks=num_tasks
            )
            self.conv2 = LoRAConv2d(
                out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False,
                rank=lora_rank, lora_alpha=lora_alpha, num_tasks=num_tasks
            )
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
            init_layer(self.conv1)
            init_layer(self.conv2)

        self.bn1 = TaskBatchNorm2d(
            out_ch, num_tasks=num_tasks, mode=bn_mode, alpha=bn_alpha, memory_momentum=memory_momentum
        )
        self.bn2 = TaskBatchNorm2d(
            out_ch, num_tasks=num_tasks, mode=bn_mode, alpha=bn_alpha, memory_momentum=memory_momentum
        )
        self.adapters = nn.ModuleList([ResidualAdapter(out_ch) for _ in range(num_tasks)])
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        if self.use_lora:
            x = F.relu_(self.bn1(self.conv1(x, task_id), task_id))
            x = F.relu_(self.bn2(self.conv2(x, task_id), task_id))
        else:
            x = F.relu_(self.bn1(self.conv1(x), task_id))
            x = F.relu_(self.bn2(self.conv2(x), task_id))
        if self.use_adapter:
            x = x + self.adapters[task_id](x)
        x = F.avg_pool2d(x, kernel_size=(2, 2))
        x = self.dropout(x)
        return x

    def task_parameters(self, task_id: int) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        params += self.bn1.task_parameters(task_id)
        params += self.bn2.task_parameters(task_id)
        if self.use_adapter:
            params += list(self.adapters[task_id].parameters())
        if self.use_lora:
            params += self.conv1.task_parameters(task_id)
            params += self.conv2.task_parameters(task_id)
        return params

    def shared_parameters(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        if not self.use_lora:
            params += list(self.conv1.parameters()) + list(self.conv2.parameters())
        return params


class TaskHeadRouter(nn.Module):
    """Optional light router over embeddings.

    Not used for training by default, but can be enabled to compare against entropy routing.
    """

    def __init__(self, embedding_dim: int, num_tasks: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim // 2, num_tasks),
        )
        for m in self.mlp:
            init_layer(m)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.mlp(emb)


class Task7PeftCnn14(nn.Module):
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
        peft_mode: str = "residual",
        bn_mode: str = "domain",
        lora_rank: int = 8,
        lora_alpha: float = 8.0,
        bn_alpha: float = 0.7,
        memory_momentum: float = 0.9,
        use_router: bool = False,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.classes_num = classes_num
        self.embedding_dim = embedding_dim
        self.peft_mode = peft_mode
        self.bn_mode = bn_mode

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
            [
                ConvBlockPEFT(
                    channels[i],
                    channels[i + 1],
                    num_tasks=num_tasks,
                    peft_mode=peft_mode,
                    bn_mode=bn_mode,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    bn_alpha=bn_alpha,
                    memory_momentum=memory_momentum,
                )
                for i in range(len(channels) - 1)
            ]
        )
        self.fc1 = nn.Linear(2048, embedding_dim)
        self.fc_out = nn.Linear(embedding_dim, classes_num)
        self.class_anchors = nn.Parameter(torch.randn(classes_num, embedding_dim))
        self.anchor_scale = nn.Parameter(torch.tensor(10.0))
        init_layer(self.fc1)
        init_layer(self.fc_out)
        nn.init.xavier_uniform_(self.class_anchors)
        self.router = TaskHeadRouter(embedding_dim, num_tasks) if use_router else None

    def freeze_shared(self) -> None:
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_task(self, task_id: int, train_backbone_for_first_task: bool = True, tune_router: bool = False) -> None:
        self.freeze_shared()
        if train_backbone_for_first_task and task_id == 0:
            for module in [self.bn0, self.fc1, self.fc_out]:
                for p in module.parameters():
                    p.requires_grad = True
            self.class_anchors.requires_grad = True
            self.anchor_scale.requires_grad = True
            for block in self.blocks:
                for p in block.shared_parameters():
                    p.requires_grad = True
                for p in block.task_parameters(task_id):
                    p.requires_grad = True
        else:
            for module in [self.fc1, self.fc_out]:
                for p in module.parameters():
                    p.requires_grad = True
            self.class_anchors.requires_grad = True
            self.anchor_scale.requires_grad = True
            for block in self.blocks:
                for p in block.task_parameters(task_id):
                    p.requires_grad = True
        if tune_router and self.router is not None:
            for p in self.router.parameters():
                p.requires_grad = True

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

    def classify_with_task(self, waveform: torch.Tensor, task_id: int, return_embedding: bool = False):
        emb = self.extract_embedding(waveform, task_id)
        classifier_logits = self.fc_out(emb)
        anchor_logits = self.compute_anchor_logits(emb)
        logits = 0.5 * classifier_logits + 0.5 * anchor_logits
        if return_embedding:
            return logits, emb
        return logits

    def route_task(self, waveform: torch.Tensor, seen_task_ids: List[int], strategy: str = "entropy") -> torch.Tensor:
        if strategy == "entropy":
            entropy_scores: List[torch.Tensor] = []
            for task_id in seen_task_ids:
                logits, _ = self.classify_with_task(waveform, task_id, return_embedding=True)
                probs = torch.softmax(logits, dim=-1)
                ent = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)
                entropy_scores.append(ent.unsqueeze(1))
            score = torch.cat(entropy_scores, dim=1)
            best_local_idx = torch.argmin(score, dim=1)
            return torch.tensor(seen_task_ids, device=waveform.device)[best_local_idx]

        if strategy == "router":
            if self.router is None:
                raise ValueError("Router was not enabled in the model.")
            # use task 0 embedding as generic shared summary
            emb = self.extract_embedding(waveform, task_id=seen_task_ids[0])
            logits = self.router(emb)
            local = torch.argmax(logits[:, seen_task_ids], dim=1)
            return torch.tensor(seen_task_ids, device=waveform.device)[local]

        raise ValueError(f"Unknown routing strategy: {strategy}")

    def forward(self, waveform: torch.Tensor, task_id: int, return_embedding: bool = False):
        return self.classify_with_task(waveform, task_id, return_embedding=return_embedding)
