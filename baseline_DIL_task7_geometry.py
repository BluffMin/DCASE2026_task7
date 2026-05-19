# baseline_DIL_task7_geometry.py
"""
D1 checkpoint initialized geometry experiment for DCASE 2026 Task 7.

Flow:
- load baseline-compatible weights from checkpoint_D1.pth into GeometryCnn14
- do NOT train on D1 (D1 dev data is unavailable)
- incrementally adapt on D2, then D3
- during evaluation/inference, consider D1/D2/D3 branches all together
- route by energy + optional anchor distance score
- save logs and outputs
"""

import argparse
import copy
import json
import os
import random
import time
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import config_task7 as config
from datasetfactory_task7 import DILDatasetInc as DILDataset
from domain_net_geometry import GeometryCnn14
from utilities import get_filename


# =========================================================
# misc
# =========================================================

def now_str() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def seed_everything(seed: int = 1193):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# logger
# =========================================================

class RunLogger:
    def __init__(self, args_dict: dict, exp_id: Optional[str] = None, base_dir: str = "logs"):
        self.timestamp = now_str()
        self.exp_id = exp_id if exp_id is not None else f"EXP-{self.timestamp}"
        self.run_dir = os.path.join(base_dir, self.exp_id)
        ensure_dir(self.run_dir)

        self.train_log_path = os.path.join(self.run_dir, "train_log.jsonl")
        self.summary_path = os.path.join(self.run_dir, "summary.json")
        self.note_path = os.path.join(self.run_dir, "notes.txt")
        self.result_csv_path = os.path.join(self.run_dir, "result.csv")
        self.config_path = os.path.join(self.run_dir, "config.json")

        self.summary = {
            "exp_id": self.exp_id,
            "created_at": self.timestamp,
            "status": "running",
            "avg_acc": None,
            "per_domain_acc": {},
            "best_epoch_by_task": {},
            "args": args_dict,
        }

        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(args_dict, f, indent=2, ensure_ascii=False)

    def write_note(self, text: str):
        with open(self.note_path, "a", encoding="utf-8") as f:
            f.write(text + "\n")

    def log_epoch(self, payload: dict):
        with open(self.train_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def set_best_epoch(self, task_name: str, epoch: int):
        self.summary["best_epoch_by_task"][task_name] = int(epoch)

    def set_domain_acc(self, domain_name: str, acc: float):
        self.summary["per_domain_acc"][domain_name] = float(acc)

    def set_avg_acc(self, avg_acc: float):
        self.summary["avg_acc"] = float(avg_acc)

    def set_status(self, status: str):
        self.summary["status"] = status

    def save_summary(self):
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(self.summary, f, indent=2, ensure_ascii=False)

    def append_result_csv(self):
        row = {
            "exp_id": self.summary["exp_id"],
            "status": self.summary["status"],
            "avg_acc": self.summary["avg_acc"],
        }
        for k, v in self.summary["per_domain_acc"].items():
            row[k] = v

        keys = list(row.keys())
        write_header = not os.path.exists(self.result_csv_path)

        with open(self.result_csv_path, "a", encoding="utf-8") as f:
            if write_header:
                f.write(",".join(keys) + "\n")
            f.write(",".join([str(row[k]) for k in keys]) + "\n")

    def export_notion_row(
        self,
        architecture: str = "GeometryCnn14 (D1-init) + Residual Adapters",
        cl_method: str = "D1 checkpoint initialization + Structure Preservation",
        augmentation: str = "None",
    ):
        d1 = self.summary["per_domain_acc"].get("D1", "-")
        d2 = self.summary["per_domain_acc"].get("D2", "-")
        d3 = self.summary["per_domain_acc"].get("D3", "-")
        avg = self.summary["avg_acc"] if self.summary["avg_acc"] is not None else "-"

        line = (
            f"{self.summary['exp_id']} | "
            f"{self.summary['status'].capitalize()} | "
            f"{architecture} | "
            f"{cl_method} | "
            f"{augmentation} | "
            f"{d1} | "
            f"{d2} | "
            f"{d3} | "
            f"{avg}"
        )

        with open(os.path.join(self.run_dir, "notion_row.txt"), "w", encoding="utf-8") as f:
            f.write(line + "\n")


# =========================================================
# loss helpers
# =========================================================

@dataclass
class LossWeights:
    ce: float = 1.0
    anchor_pull: float = 0.2
    anchor_sep: float = 0.05
    distill: float = 0.5
    structure: float = 0.1


def pairwise_distance_matrix(x: torch.Tensor) -> torch.Tensor:
    return torch.cdist(x, x, p=2)


def compute_anchor_losses(embedding: torch.Tensor, targets: torch.Tensor, model: GeometryCnn14) -> Dict[str, torch.Tensor]:
    anchors = torch.nn.functional.normalize(model.class_anchors, dim=-1)
    target_anchors = anchors[targets]

    anchor_pull = ((embedding - target_anchors) ** 2).sum(dim=-1).mean()

    dist_mat = pairwise_distance_matrix(anchors)
    eye = torch.eye(dist_mat.size(0), device=dist_mat.device)
    off_diag = dist_mat + eye * 1e6
    min_inter = off_diag.min(dim=1).values
    anchor_sep = torch.relu(1.0 - min_inter).mean()

    return {
        "anchor_pull": anchor_pull,
        "anchor_sep": anchor_sep,
    }


def accuracy_from_logits(logits: torch.Tensor, targets_onehot: torch.Tensor) -> float:
    pred = torch.argmax(logits, dim=1)
    gt = torch.argmax(targets_onehot, dim=1)
    correct = (pred == gt).sum().item()
    total = gt.numel()
    return 100.0 * correct / max(total, 1)


# =========================================================
# learner
# =========================================================

class Learner:
    """
    Task mapping:
    0 -> D1
    1 -> D2
    2 -> D3
    """

    def __init__(
        self,
        logger: RunLogger,
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
        self.logger = logger
        self.loss_weights = LossWeights()

        self.model = GeometryCnn14(
            sample_rate=sample_rate,
            window_size=window_size,
            hop_size=hop_size,
            mel_bins=mel_bins,
            fmin=fmin,
            fmax=fmax,
            classes_num=classes_num,
            nb_tasks=num_tasks,
            embed_dim=embed_dim,
        )

        self.prev_model: Optional[GeometryCnn14] = None
        self.reference_anchor_dist: Optional[torch.Tensor] = None

        self.domain_to_task = {"D1": 0, "D2": 1, "D3": 2}
        self.seen_domains = ["D1"]  # D1 is seen from checkpoint init

    # -------------------------------------------------
    # io helpers
    # -------------------------------------------------
    def _make_loader(self, df, batch_size: int, shuffle: bool, num_workers: int):
        dataset = DILDataset(df, config.audio_folder_DIL)
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

    def _geometry_ckpt_path(self, domain_name: str):
        ensure_dir(config.save_resume_path)
        return os.path.join(config.save_resume_path, f"geometry_checkpoint_{domain_name}.pth")

    def save_geometry_checkpoint(self, domain_name: str):
        ckpt_path = self._geometry_ckpt_path(domain_name)
        torch.save(self.model.state_dict(), ckpt_path)
        print(f"[Checkpoint] saved: {ckpt_path}")

    def load_geometry_checkpoint(self, domain_name: str, device: str):
        ckpt_path = self._geometry_ckpt_path(domain_name)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device)
        self.model.load_state_dict(state, strict=True)
        print(f"[Checkpoint] loaded: {ckpt_path}")

    # -------------------------------------------------
    # D1 init
    # -------------------------------------------------
    def initialize_from_d1_checkpoint(self, device: str, ckpt_name: str = "checkpoint_D1.pth"):
        d1_ckpt_path = os.path.join(config.save_resume_path, ckpt_name)
        if not os.path.exists(d1_ckpt_path):
            raise FileNotFoundError(
                f"D1 checkpoint not found: {d1_ckpt_path}\n"
                f"Expected path based on config.save_resume_path={config.save_resume_path}"
            )

        ckpt = torch.load(d1_ckpt_path, map_location=device)
        loaded, skipped = self.model.partial_load_from_baseline(ckpt)
        self.model.init_anchors_from_fc()

        self.logger.write_note(f"D1 partial init from: {d1_ckpt_path}")
        self.logger.write_note(f"Loaded keys: {len(loaded)}")
        self.logger.write_note(f"Skipped keys: {len(skipped)}")

        print(f"[D1 Init] checkpoint: {d1_ckpt_path}")
        print(f"[D1 Init] loaded={len(loaded)}, skipped={len(skipped)}")

        self._freeze_snapshot()
        self._update_reference_geometry()

    def _freeze_snapshot(self):
        self.prev_model = copy.deepcopy(self.model).eval().cpu()
        for p in self.prev_model.parameters():
            p.requires_grad = False

    def _update_reference_geometry(self):
        with torch.no_grad():
            anchors = torch.nn.functional.normalize(self.model.class_anchors.detach(), dim=-1)
            self.reference_anchor_dist = pairwise_distance_matrix(anchors).detach().cpu()

    # -------------------------------------------------
    # train
    # -------------------------------------------------
    def incremental_train(self, train_loader, device: str, args, domain_name: str):
        task_id = self.domain_to_task[domain_name]
        self.model.to(device)
        self.model.unfreeze_for_task(task_id)

        if self.prev_model is not None:
            self.prev_model.to(device)
            self.prev_model.eval()

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        lr = args.learning_rate

        optimizer = optim.Adam(
            trainable_params,
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epoch,
            eta_min=1e-6,
        )
        criterion = nn.CrossEntropyLoss(ignore_index=-1)

        best_epoch = 0
        best_loss = float("inf")

        for epoch_idx in range(1, args.epoch + 1):
            self.model.train()

            epoch_total_loss = 0.0
            epoch_ce = 0.0
            epoch_anchor_pull = 0.0
            epoch_anchor_sep = 0.0
            epoch_distill = 0.0
            epoch_structure = 0.0
            epoch_batch_acc = 0.0
            num_batches = 0

            for audio, target, _ in train_loader:
                audio = audio.float().to(device)
                target = target.float().to(device)
                target_idx = torch.argmax(target, dim=-1)

                optimizer.zero_grad()

                logits, emb = self.model(audio, task_id, return_embedding=True)

                ce_loss = criterion(logits, target_idx)
                anchor_dict = compute_anchor_losses(emb, target_idx, self.model)
                anchor_pull = anchor_dict["anchor_pull"]
                anchor_sep = anchor_dict["anchor_sep"]

                total_loss = (
                    self.loss_weights.ce * ce_loss
                    + self.loss_weights.anchor_pull * anchor_pull
                    + self.loss_weights.anchor_sep * anchor_sep
                )

                distill_loss = torch.tensor(0.0, device=device)
                if self.prev_model is not None:
                    with torch.no_grad():
                        prev_logits = self.prev_model(audio, 0 if domain_name == "D2" else 1 if "D2" in self.seen_domains else 0)
                    distill_loss = torch.nn.functional.mse_loss(logits, prev_logits)
                    total_loss = total_loss + self.loss_weights.distill * distill_loss

                structure_loss = torch.tensor(0.0, device=device)
                if self.reference_anchor_dist is not None:
                    cur_anchor_dist = pairwise_distance_matrix(
                        torch.nn.functional.normalize(self.model.class_anchors, dim=-1)
                    )
                    ref_anchor_dist = self.reference_anchor_dist.to(cur_anchor_dist.device)
                    structure_loss = torch.nn.functional.mse_loss(cur_anchor_dist, ref_anchor_dist)
                    total_loss = total_loss + self.loss_weights.structure * structure_loss

                total_loss.backward()
                optimizer.step()

                batch_acc = accuracy_from_logits(logits.detach(), target.detach())

                epoch_total_loss += total_loss.item()
                epoch_ce += ce_loss.item()
                epoch_anchor_pull += anchor_pull.item()
                epoch_anchor_sep += anchor_sep.item()
                epoch_distill += distill_loss.item()
                epoch_structure += structure_loss.item()
                epoch_batch_acc += batch_acc
                num_batches += 1

            scheduler.step()

            avg_total_loss = epoch_total_loss / max(1, num_batches)
            avg_ce = epoch_ce / max(1, num_batches)
            avg_anchor_pull = epoch_anchor_pull / max(1, num_batches)
            avg_anchor_sep = epoch_anchor_sep / max(1, num_batches)
            avg_distill = epoch_distill / max(1, num_batches)
            avg_structure = epoch_structure / max(1, num_batches)
            avg_batch_acc = epoch_batch_acc / max(1, num_batches)

            print(
                f"[Task {domain_name}] "
                f"epoch={epoch_idx:03d} "
                f"loss={avg_total_loss:.4f} "
                f"ce={avg_ce:.4f} "
                f"pull={avg_anchor_pull:.4f} "
                f"sep={avg_anchor_sep:.4f} "
                f"distill={avg_distill:.4f} "
                f"struct={avg_structure:.4f} "
                f"batch_acc={avg_batch_acc:.2f}"
            )

            self.logger.log_epoch({
                "task_name": domain_name,
                "task_id": task_id,
                "epoch": epoch_idx,
                "total_loss": float(avg_total_loss),
                "ce_loss": float(avg_ce),
                "anchor_pull_loss": float(avg_anchor_pull),
                "anchor_sep_loss": float(avg_anchor_sep),
                "distill_loss": float(avg_distill),
                "structure_loss": float(avg_structure),
                "avg_batch_acc": float(avg_batch_acc),
                "lr": float(optimizer.param_groups[0]["lr"]),
            })

            if avg_total_loss < best_loss:
                best_loss = avg_total_loss
                best_epoch = epoch_idx

        self.logger.set_best_epoch(domain_name, best_epoch)

        if args.save:
            self.save_geometry_checkpoint(domain_name)

        self._freeze_snapshot()
        self._update_reference_geometry()

    # -------------------------------------------------
    # routing / inference
    # -------------------------------------------------
    def route_and_predict(
        self,
        audio: torch.Tensor,
        candidate_domains: List[str],
        device: str,
        alpha_energy: float = 1.0,
        beta_anchor: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.model = self.model.to(device)
        self.model.eval()
        audio = audio.to(device)

        branch_logits = []
        branch_scores = []

        anchors = torch.nn.functional.normalize(
            self.model.class_anchors.to(device), dim=-1
        )

        with torch.no_grad():
            for domain_name in candidate_domains:
                task_id = self.domain_to_task[domain_name]
                logits, emb = self.model(audio, task_id, return_embedding=True)

                energy = -torch.logsumexp(logits, dim=1)
                anchor_dist = torch.cdist(emb, anchors).min(dim=1).values
                score = alpha_energy * energy + beta_anchor * anchor_dist

                branch_logits.append(logits)
                branch_scores.append(score)

        branch_logits = torch.stack(branch_logits, dim=1)
        branch_scores = torch.stack(branch_scores, dim=1)
        best_branch_idx = torch.argmin(branch_scores, dim=1)

        batch_indices = torch.arange(branch_logits.size(0), device=device)
        routed_logits = branch_logits[batch_indices, best_branch_idx]
        return routed_logits, best_branch_idx

    def evaluate_domains(
        self,
        df_dev_test,
        candidate_domains: List[str],
        batch_size: int,
        num_workers: int,
        device: str,
    ) -> float:
        self.model = self.model.to(device)
        self.model.eval()

        id_to_class = {v: k for k, v in config.dict_class_labels.items()}
        ensure_dir(config.output_folder)

        available_domains = sorted(list(df_dev_test["domain"].unique()))
        eval_domains = [d for d in ["D1", "D2", "D3"] if d in available_domains]

        all_acc = []

        for domain_name in eval_domains:
            valid_df = df_dev_test[df_dev_test["domain"].isin([domain_name])]
            loader = self._make_loader(valid_df, batch_size=1, shuffle=False, num_workers=num_workers)

            correct = 0
            total = 0

            out_file = os.path.join(
                config.output_folder,
                f"output_geometry_{domain_name}_{now_str()}.txt"
            )

            with open(out_file, "w", encoding="utf-8") as f:
                for inputs, targets, audio_files in loader:
                    inputs = inputs.float().to(device)
                    targets = targets.float().to(device)
                    gt = torch.argmax(targets, dim=-1)

                    routed_logits, best_branch_idx = self.route_and_predict(
                        inputs,
                        candidate_domains=candidate_domains,
                        device=device,
                        alpha_energy=1.0,
                        beta_anchor=0.1,
                    )
                    pred = torch.argmax(routed_logits, dim=1)

                    correct += (pred == gt).sum().item()
                    total += gt.numel()

                    pred_labels = [id_to_class[int(p)] for p in pred.detach().cpu().tolist()]
                    routed_domains = [candidate_domains[int(i)] for i in best_branch_idx.detach().cpu().tolist()]

                    for file_name, label, routed_domain in zip(audio_files, pred_labels, routed_domains):
                        f.write(file_name + "\t" + label + "\t" + routed_domain + "\n")

            acc = round(100.0 * correct / max(total, 1), 2)
            print(f"eval domain: {domain_name} | candidate branches: {candidate_domains} | acc: {acc}")
            self.logger.set_domain_acc(domain_name, acc)
            all_acc.append(acc)

        avg_acc = float(np.mean(all_acc)) if len(all_acc) > 0 else 0.0
        self.logger.set_avg_acc(avg_acc)
        return avg_acc


# =========================================================
# train entry
# =========================================================

def train(args):
    seed_everything(args.seed)

    device = "cuda" if (args.cuda and torch.cuda.is_available()) else "cpu"
    logger = RunLogger(args_dict=vars(args).copy(), exp_id=args.exp_id, base_dir=args.log_dir)
    logger.write_note("D1 checkpoint initialized geometry experiment started.")
    logger.write_note(f"Device: {device}")

    try:
        df_dev_train = config.df_DIL_dev_train
        df_dev_test = config.df_DIL_dev_test

        learner = Learner(
            logger=logger,
            sample_rate=config.sample_rate,
            window_size=config.window_size,
            hop_size=config.hop_size,
            mel_bins=config.mel_bins,
            fmin=config.fmin,
            fmax=config.fmax,
            classes_num=config.classes_num_DIL,
            num_tasks=3,
            embed_dim=args.embed_dim,
        )

        # 1) initialize from D1 checkpoint
        learner.initialize_from_d1_checkpoint(device=device, ckpt_name=args.d1_checkpoint_name)

        # optional: save D1-initialized geometry state
        if args.save_d1_init:
            learner.save_geometry_checkpoint("D1")

        # 2) evaluate initial state with D1 as only candidate if any D1 exists in dev_test
        if args.eval_before_incremental:
            avg_acc = learner.evaluate_domains(
                df_dev_test=df_dev_test,
                candidate_domains=["D1"],
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
            )
            print(f"[Before incremental] Average Accuracy: {avg_acc:.2f}")

        # 3) incremental training on D2 then D3
        incremental_domains = ["D2", "D3"]

        for domain_name in incremental_domains:
            train_df = df_dev_train[df_dev_train["domain"].isin([domain_name])]
            print(f"\n[Incremental setup] domain={domain_name}, train={len(train_df)}")

            if len(train_df) == 0:
                logger.write_note(f"Skipped {domain_name}: no training samples found.")
                continue

            if args.resume:
                learner.load_geometry_checkpoint(domain_name, device=device)
                if domain_name not in learner.seen_domains:
                    learner.seen_domains.append(domain_name)
            else:
                train_loader = learner._make_loader(
                    train_df,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=args.num_workers,
                )
                learner.incremental_train(train_loader, device=device, args=args, domain_name=domain_name)
                if domain_name not in learner.seen_domains:
                    learner.seen_domains.append(domain_name)

            # evaluate with all seen branches including D1
            avg_acc = learner.evaluate_domains(
                df_dev_test=df_dev_test,
                candidate_domains=learner.seen_domains,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
            )
            print(f"[After {domain_name}] Average Accuracy: {avg_acc:.2f}")

        logger.set_status("done")
        logger.save_summary()
        logger.append_result_csv()
        logger.export_notion_row(
            architecture="GeometryCnn14 (D1-init) + Residual Adapters",
            cl_method="D1 checkpoint initialization + Structure Preservation",
            augmentation=str(args.augmentation),
        )
        print(f"[Logger] saved to: {logger.run_dir}")

    except Exception as e:
        logger.set_status("failed")
        logger.write_note(f"Exception: {str(e)}")
        logger.write_note(traceback.format_exc())
        logger.save_summary()
        print(f"[Logger] failed run saved to: {logger.run_dir}")
        raise


# =========================================================
# main
# =========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="D1-initialized Geometry DCASE2026 Task7")
    subparsers = parser.add_subparsers(dest="mode")

    parser_train = subparsers.add_parser("train")
    parser_train.add_argument("--augmentation", type=str, choices=["none", "mixup"], default="none")
    parser_train.add_argument("--learning_rate", type=float, default=1e-4)
    parser_train.add_argument("--batch_size", type=int, default=32)
    parser_train.add_argument("--num_workers", type=int, default=8)
    parser_train.add_argument("--cuda", action="store_true", default=False)
    parser_train.add_argument("--epoch", type=int, default=120)
    parser_train.add_argument("--resume", action="store_true", default=False)
    parser_train.add_argument("--save", action="store_true", default=False)

    parser_train.add_argument("--seed", type=int, default=1193)
    parser_train.add_argument("--exp_id", type=str, default=None)
    parser_train.add_argument("--log_dir", type=str, default="logs")
    parser_train.add_argument("--embed_dim", type=int, default=256)

    parser_train.add_argument("--d1_checkpoint_name", type=str, default="checkpoint_D1.pth")
    parser_train.add_argument("--save_d1_init", action="store_true", default=False)
    parser_train.add_argument("--eval_before_incremental", action="store_true", default=False)

    args = parser.parse_args()
    args.filename = get_filename(__file__)

    if args.mode == "train":
        train(args)
    else:
        raise ValueError("Error argument!")