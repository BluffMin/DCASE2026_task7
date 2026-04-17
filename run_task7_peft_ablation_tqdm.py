from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm.auto import tqdm

import config_task7 as config
from config_task7 import sample_rate, mel_bins, fmin, fmax, window_size, hop_size
from datasetfactory_task7 import DILDatasetInc as DILDataset
from domain_net_task7_peft import Task7PeftCnn14


# ------------------------------
# Utilities
# ------------------------------
def now_str() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def seed_everything(seed: int = 1193) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_1d_float_tensor(x):
    if isinstance(x, torch.Tensor):
        t = x.detach().float().flatten()
    else:
        t = torch.as_tensor(x, dtype=torch.float32).flatten()
    return t


def _infer_target_num_samples() -> int:
    # Prefer explicit config if available; otherwise default to 10 s at 32 kHz.
    for name in ('clip_samples', 'clip_length', 'max_audio_samples'):
        if hasattr(config, name):
            try:
                v = int(getattr(config, name))
                if v > 0:
                    return v
            except Exception:
                pass
    return int(sample_rate * 10)


def pad_collate_audio(batch):
    """Collate variable-length waveforms by zero-padding/truncating to a fixed length.

    Expected item format from DILDataset: (audio, target, meta).
    This avoids DataLoader stack errors when raw waveform lengths differ.
    """
    audios, targets, metas = zip(*batch)
    audio_tensors = [_to_1d_float_tensor(a) for a in audios]
    max_len_in_batch = max(t.numel() for t in audio_tensors)
    target_len = _infer_target_num_samples()
    final_len = min(max_len_in_batch, target_len) if max_len_in_batch > 0 else target_len

    padded = []
    for t in audio_tensors:
        n = t.numel()
        if n >= final_len:
            padded.append(t[:final_len])
        else:
            out = torch.zeros(final_len, dtype=torch.float32)
            out[:n] = t
            padded.append(out)

    target_tensors = []
    for y in targets:
        if isinstance(y, torch.Tensor):
            target_tensors.append(y.detach().float())
        else:
            target_tensors.append(torch.as_tensor(y, dtype=torch.float32))

    return torch.stack(padded, dim=0), torch.stack(target_tensors, dim=0), list(metas)


@dataclass
class LossWeights:
    ce: float = 1.0
    anchor_pull: float = 0.2
    anchor_sep: float = 0.05
    distill: float = 0.5
    structure: float = 0.1
    router: float = 0.0


class RunLogger:
    def __init__(self, args_dict: dict, exp_id: Optional[str] = None, base_dir: str = "logs_ablation"):
        self.timestamp = now_str()
        self.exp_id = exp_id if exp_id is not None else f"ABL-{self.timestamp}"
        self.run_dir = os.path.join(base_dir, self.exp_id)
        ensure_dir(self.run_dir)
        self.train_log_path = os.path.join(self.run_dir, "train_log.jsonl")
        self.summary_path = os.path.join(self.run_dir, "summary.json")
        self.result_csv_path = os.path.join(base_dir, "result.csv")
        self.config_path = os.path.join(self.run_dir, "config.json")
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(args_dict, f, indent=2, ensure_ascii=False)
        self.summary = {
            "exp_id": self.exp_id,
            "created_at": self.timestamp,
            "status": "running",
            "avg_acc": None,
            "per_domain_acc": {},
            "router_acc": {},
            "args": args_dict,
        }

    def log_epoch(self, payload: dict) -> None:
        with open(self.train_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def set_domain_acc(self, name: str, value: float) -> None:
        self.summary["per_domain_acc"][name] = float(value)

    def set_router_acc(self, name: str, value: float) -> None:
        self.summary["router_acc"][name] = float(value)

    def set_avg_acc(self, value: float) -> None:
        self.summary["avg_acc"] = float(value)

    def set_status(self, status: str) -> None:
        self.summary["status"] = status

    def flush(self) -> None:
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(self.summary, f, indent=2, ensure_ascii=False)
        row = {
            "exp_id": self.summary["exp_id"],
            "status": self.summary["status"],
            "avg_acc": self.summary["avg_acc"],
        }
        row.update(self.summary["per_domain_acc"])
        row.update({f"router_{k}": v for k, v in self.summary["router_acc"].items()})
        write_header = not os.path.exists(self.result_csv_path)
        with open(self.result_csv_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)


# ------------------------------
# Presets for ablation
# ------------------------------
PRESETS: Dict[str, Dict] = {
    "residual": {
        "peft_mode": "residual",
        "bn_mode": "domain",
        "use_router": False,
        "routing": "entropy",
    },
    "bn_domain": {
        "peft_mode": "none",
        "bn_mode": "domain",
        "use_router": False,
        "routing": "entropy",
    },
    "bn_adaptive": {
        "peft_mode": "none",
        "bn_mode": "adaptive",
        "use_router": False,
        "routing": "entropy",
    },
    "bn_memory": {
        "peft_mode": "none",
        "bn_mode": "memory",
        "use_router": False,
        "routing": "entropy",
    },
    "lora": {
        "peft_mode": "lora",
        "bn_mode": "domain",
        "use_router": False,
        "routing": "entropy",
    },
    "hybrid_bn_lora": {
        "peft_mode": "hybrid",
        "bn_mode": "adaptive",
        "use_router": False,
        "routing": "entropy",
    },
    "hybrid_mem_lora": {
        "peft_mode": "hybrid",
        "bn_mode": "memory",
        "use_router": False,
        "routing": "entropy",
    },
    "hybrid_router": {
        "peft_mode": "hybrid",
        "bn_mode": "adaptive",
        "use_router": True,
        "routing": "router",
    },
}


# ------------------------------
# Geometry losses
# ------------------------------
def pairwise_distance_matrix(x: torch.Tensor) -> torch.Tensor:
    return torch.cdist(x, x, p=2)


def compute_anchor_losses(embedding: torch.Tensor, target_idx: torch.Tensor, model: Task7PeftCnn14) -> Dict[str, torch.Tensor]:
    anchors = torch.nn.functional.normalize(model.class_anchors, dim=-1)
    target_anchors = anchors[target_idx]
    anchor_pull = ((embedding - target_anchors) ** 2).sum(dim=-1).mean()
    dist_mat = pairwise_distance_matrix(anchors)
    eye = torch.eye(dist_mat.size(0), device=dist_mat.device)
    off_diag = dist_mat + eye * 1e6
    min_inter = off_diag.min(dim=1).values
    anchor_sep = torch.relu(1.0 - min_inter).mean()
    return {"anchor_pull": anchor_pull, "anchor_sep": anchor_sep}


def accuracy_from_logits(logits: torch.Tensor, targets_onehot: torch.Tensor) -> float:
    pred = torch.argmax(logits, dim=1)
    gt = torch.argmax(targets_onehot, dim=1)
    return 100.0 * (pred == gt).sum().item() / max(gt.numel(), 1)


# ------------------------------
# Learner
# ------------------------------
def _progress(iterable, desc: str, total=None, leave=False):
    return tqdm(iterable, desc=desc, total=total, leave=leave, dynamic_ncols=True, mininterval=0.5)


class AblationLearner:
    def __init__(self, args, logger: RunLogger):
        self.args = args
        self.logger = logger
        self.loss_weights = LossWeights(
            ce=args.w_ce,
            anchor_pull=args.w_anchor_pull,
            anchor_sep=args.w_anchor_sep,
            distill=args.w_distill,
            structure=args.w_structure,
            router=args.w_router,
        )
        self.model = Task7PeftCnn14(
            sample_rate=sample_rate,
            window_size=window_size,
            hop_size=hop_size,
            mel_bins=mel_bins,
            fmin=fmin,
            fmax=fmax,
            classes_num=config.classes_num_DIL,
            num_tasks=args.num_tasks,
            embedding_dim=args.embedding_dim,
            peft_mode=args.peft_mode,
            bn_mode=args.bn_mode,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            bn_alpha=args.bn_alpha,
            memory_momentum=args.memory_momentum,
            use_router=args.use_router,
        )
        self.cur_task = -1
        self.prev_model: Optional[Task7PeftCnn14] = None
        self.reference_anchor_dist: Optional[torch.Tensor] = None

    def _make_loader(self, df, batch_size: int, shuffle: bool, num_workers: int):
        dataset = DILDataset(df, config.audio_folder_DIL)
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=pad_collate_audio,
        )

    def _freeze_snapshot(self):
        self.prev_model = copy.deepcopy(self.model).eval().cpu()
        for p in self.prev_model.parameters():
            p.requires_grad = False

    def _update_reference_geometry(self):
        anchors = torch.nn.functional.normalize(self.model.class_anchors.detach(), dim=-1)
        self.reference_anchor_dist = pairwise_distance_matrix(anchors).detach().cpu()

    def _split_df_by_domain(self, df, domain_name: str):
        return df[df["domain"] == domain_name].reset_index(drop=True)

    def incremental_train(self, train_loader, task_name: str, device: str):
        self.model.to(device)
        self.model.unfreeze_task(self.cur_task, train_backbone_for_first_task=True, tune_router=self.args.use_router)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        lr = self.args.learning_rate if self.cur_task == 0 else self.args.learning_rate / self.args.lr_decay_after_first
        optimizer = torch.optim.Adam(trainable_params, lr=lr, betas=(0.9, 0.999), eps=1e-8)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epoch, eta_min=1e-6)
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        if self.prev_model is not None:
            self.prev_model.to(device)
            self.prev_model.eval()

        for epoch_idx in range(1, self.args.epoch + 1):
            self.model.train()
            stats = {
                "total_loss": 0.0,
                "ce": 0.0,
                "pull": 0.0,
                "sep": 0.0,
                "distill": 0.0,
                "structure": 0.0,
                "router": 0.0,
                "batch_acc": 0.0,
                "n": 0,
            }
            pbar = _progress(train_loader, desc=f"train {task_name} ep {epoch_idx}/{self.args.epoch}", total=len(train_loader), leave=False)
            for audio, target, _ in pbar:
                audio = audio.float().to(device)
                target = target.float().to(device)
                target_idx = torch.argmax(target, dim=-1)
                optimizer.zero_grad()
                logits, embedding = self.model(audio, self.cur_task, return_embedding=True)
                ce_loss = criterion(logits, target_idx)
                anchor_dict = compute_anchor_losses(embedding, target_idx, self.model)
                total_loss = (
                    self.loss_weights.ce * ce_loss
                    + self.loss_weights.anchor_pull * anchor_dict["anchor_pull"]
                    + self.loss_weights.anchor_sep * anchor_dict["anchor_sep"]
                )

                distill_loss = torch.tensor(0.0, device=device)
                if self.prev_model is not None:
                    with torch.no_grad():
                        prev_logits, _ = self.prev_model(audio, max(self.cur_task - 1, 0), return_embedding=True)
                    distill_loss = F.mse_loss(logits, prev_logits)
                    total_loss = total_loss + self.loss_weights.distill * distill_loss

                structure_loss = torch.tensor(0.0, device=device)
                if self.reference_anchor_dist is not None:
                    cur_anchor_dist = pairwise_distance_matrix(
                        torch.nn.functional.normalize(self.model.class_anchors, dim=-1)
                    )
                    ref_anchor_dist = self.reference_anchor_dist.to(cur_anchor_dist.device)
                    structure_loss = F.mse_loss(cur_anchor_dist, ref_anchor_dist)
                    total_loss = total_loss + self.loss_weights.structure * structure_loss

                router_loss = torch.tensor(0.0, device=device)
                if self.args.use_router and self.model.router is not None:
                    router_logits = self.model.router(embedding.detach())
                    router_targets = torch.full_like(target_idx, fill_value=self.cur_task)
                    router_loss = criterion(router_logits, router_targets)
                    total_loss = total_loss + self.loss_weights.router * router_loss

                total_loss.backward()
                if self.args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, self.args.grad_clip)
                optimizer.step()

                stats["total_loss"] += total_loss.item()
                stats["ce"] += ce_loss.item()
                stats["pull"] += anchor_dict["anchor_pull"].item()
                stats["sep"] += anchor_dict["anchor_sep"].item()
                stats["distill"] += distill_loss.item()
                stats["structure"] += structure_loss.item()
                stats["router"] += router_loss.item()
                stats["batch_acc"] += accuracy_from_logits(logits, target)
                stats["n"] += 1

                denom_inner = max(stats["n"], 1)
                pbar.set_postfix({
                    "loss": f"{stats['total_loss']/denom_inner:.4f}",
                    "ce": f"{stats['ce']/denom_inner:.4f}",
                    "acc": f"{stats['batch_acc']/denom_inner:.2f}",
                })

            pbar.close()
            scheduler.step()
            denom = max(stats["n"], 1)
            payload = {
                "task_name": task_name,
                "task_id": self.cur_task,
                "epoch": epoch_idx,
                "total_loss": stats["total_loss"] / denom,
                "ce_loss": stats["ce"] / denom,
                "anchor_pull": stats["pull"] / denom,
                "anchor_sep": stats["sep"] / denom,
                "distill_loss": stats["distill"] / denom,
                "structure_loss": stats["structure"] / denom,
                "router_loss": stats["router"] / denom,
                "batch_acc": stats["batch_acc"] / denom,
            }
            print(
                f"[{task_name}] ep={epoch_idx:03d} loss={payload['total_loss']:.4f} "
                f"ce={payload['ce_loss']:.4f} pull={payload['anchor_pull']:.4f} "
                f"sep={payload['anchor_sep']:.4f} distill={payload['distill_loss']:.4f} "
                f"struct={payload['structure_loss']:.4f} router={payload['router_loss']:.4f} "
                f"acc={payload['batch_acc']:.2f}"
            )
            self.logger.log_epoch(payload)

        self._freeze_snapshot()
        self._update_reference_geometry()

    @torch.no_grad()
    def evaluate_domain(self, loader, task_id: int, device: str, desc: Optional[str] = None) -> float:
        self.model.to(device)
        self.model.eval()
        correct, total = 0, 0
        iterator = _progress(loader, desc=desc or f"eval oracle t{task_id}", total=len(loader), leave=False)
        for inputs, targets, _ in iterator:
            inputs = inputs.float().to(device)
            targets = targets.float().to(device)
            logits = self.model(inputs, task_id)
            pred = torch.argmax(logits, dim=-1)
            gt = torch.argmax(targets, dim=-1)
            correct += (pred == gt).sum().item()
            total += gt.numel()
            iterator.set_postfix({"acc": f"{100.0 * correct / max(total,1):.2f}"})
        iterator.close()
        return 100.0 * correct / max(total, 1)

    @torch.no_grad()
    def evaluate_routed(self, loader, seen_task_ids: List[int], true_task_id: int, device: str, desc: Optional[str] = None) -> Tuple[float, float]:
        self.model.to(device)
        self.model.eval()
        correct, total = 0, 0
        route_correct = 0
        iterator = _progress(loader, desc=desc or f"eval routed true={true_task_id}", total=len(loader), leave=False)
        for inputs, targets, _ in iterator:
            inputs = inputs.float().to(device)
            targets = targets.float().to(device)
            routed_task_ids = self.model.route_task(inputs, seen_task_ids=seen_task_ids, strategy=self.args.routing)
            per_sample_logits = []
            for i in range(inputs.size(0)):
                logits = self.model(inputs[i : i + 1], int(routed_task_ids[i].item()))
                per_sample_logits.append(logits)
            logits = torch.cat(per_sample_logits, dim=0)
            pred = torch.argmax(logits, dim=-1)
            gt = torch.argmax(targets, dim=-1)
            correct += (pred == gt).sum().item()
            total += gt.numel()
            route_correct += (routed_task_ids == true_task_id).sum().item()
            iterator.set_postfix({
                "cls": f"{100.0 * correct / max(total,1):.2f}",
                "route": f"{100.0 * route_correct / max(total,1):.2f}",
            })
        iterator.close()
        acc = 100.0 * correct / max(total, 1)
        route_acc = 100.0 * route_correct / max(total, 1)
        return acc, route_acc

    def run(self, task_order: List[str], device: str) -> None:
        train_df_all = config.df_DIL_dev_train
        test_df_all = config.df_DIL_dev_test

        seen_task_ids: List[int] = []
        routed_accs: List[float] = []
        task_accs: List[float] = []

        for domain_name in task_order:
            self.cur_task += 1
            seen_task_ids.append(self.cur_task)
            print(f"\n[Train] task_id={self.cur_task} domain={domain_name}")
            train_df = self._split_df_by_domain(train_df_all, domain_name)
            train_loader = self._make_loader(train_df, self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)
            self.incremental_train(train_loader, task_name=domain_name, device=device)

            print(f"[Eval after {domain_name}]")
            for eval_tid, eval_domain in enumerate(task_order[: self.cur_task + 1]):
                test_df = self._split_df_by_domain(test_df_all, eval_domain)
                test_loader = self._make_loader(test_df, self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)
                oracle_acc = self.evaluate_domain(test_loader, eval_tid, device=device, desc=f"oracle {eval_domain}")
                self.logger.set_domain_acc(eval_domain, oracle_acc)
                task_accs.append(oracle_acc)
                print(f"  oracle/{eval_domain}: {oracle_acc:.2f}")
                routed_acc, router_acc = self.evaluate_routed(test_loader, seen_task_ids, true_task_id=eval_tid, device=device, desc=f"routed {eval_domain}")
                self.logger.set_router_acc(eval_domain, router_acc)
                routed_accs.append(routed_acc)
                print(f"  routed/{eval_domain}: cls={routed_acc:.2f}, route={router_acc:.2f}")

        avg = float(np.mean(routed_accs)) if routed_accs else 0.0
        self.logger.set_avg_acc(avg)
        self.logger.set_status("done")
        self.logger.flush()
        print(f"\n[Done] routed-average={avg:.2f}")


# ------------------------------
# CLI
# ------------------------------
def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DCASE2026 Task7 PEFT/BN/LoRA ablation runner")
    parser.add_argument("--preset", type=str, default="residual", choices=list(PRESETS.keys()) + ["all"])
    parser.add_argument("--task_order", type=str, default="D2,D3", help="Comma-separated domain order, e.g. D2,D3")
    parser.add_argument("--num_tasks", type=int, default=2)
    parser.add_argument("--epoch", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--lr_decay_after_first", type=float, default=5.0)
    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=8.0)
    parser.add_argument("--bn_alpha", type=float, default=0.7)
    parser.add_argument("--memory_momentum", type=float, default=0.9)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--routing", type=str, default="entropy", choices=["entropy", "router"])
    parser.add_argument("--seed", type=int, default=1193)

    # loss weights
    parser.add_argument("--w_ce", type=float, default=1.0)
    parser.add_argument("--w_anchor_pull", type=float, default=0.2)
    parser.add_argument("--w_anchor_sep", type=float, default=0.05)
    parser.add_argument("--w_distill", type=float, default=0.5)
    parser.add_argument("--w_structure", type=float, default=0.1)
    parser.add_argument("--w_router", type=float, default=0.2)

    # filled by preset
    parser.add_argument("--peft_mode", type=str, default="residual", choices=["none", "residual", "lora", "hybrid"])
    parser.add_argument("--bn_mode", type=str, default="domain", choices=["domain", "adaptive", "memory"])
    parser.add_argument("--use_router", action="store_true")
    return parser.parse_args()


def run_one(args: argparse.Namespace, preset_name: str) -> None:
    preset = PRESETS[preset_name]
    args.peft_mode = preset["peft_mode"]
    args.bn_mode = preset["bn_mode"]
    args.use_router = preset["use_router"]
    args.routing = preset["routing"]

    task_order = [x.strip() for x in args.task_order.split(",") if x.strip()]
    if len(task_order) != args.num_tasks:
        raise ValueError(f"num_tasks={args.num_tasks} but task_order={task_order}")

    exp_id = f"{preset_name}-{now_str()}"
    print(" " + "#" * 100)
    print(f"[Run start] preset={preset_name} exp_id={exp_id}")
    print(f"  peft_mode={args.peft_mode} bn_mode={args.bn_mode} use_router={args.use_router} routing={args.routing}")
    print(f"  task_order={task_order} seed={args.seed} batch_size={args.batch_size} epoch={args.epoch}")
    print("#" * 100)
    logger = RunLogger(args_dict=vars(args), exp_id=exp_id)
    try:
        learner = AblationLearner(args, logger)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Device] {device}")
        learner.run(task_order=task_order, device=device)
        print(f"[Run done] preset={preset_name} exp_id={exp_id} avg_acc={logger.summary.get('avg_acc')}")
    except Exception as e:
        logger.set_status("failed")
        logger.summary["error"] = repr(e)
        logger.flush()
        print(f"[Run failed] preset={preset_name} exp_id={exp_id} error={e!r}")
        raise


if __name__ == "__main__":
    args = build_args()
    seed_everything(args.seed)
    if args.preset == "all":
        for name in PRESETS:
            local_args = copy.deepcopy(args)
            run_one(local_args, name)
    else:
        run_one(args, args.preset)
