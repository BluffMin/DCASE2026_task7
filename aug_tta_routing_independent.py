#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Augmentation-TTA Routing for DCASE Task7 D2/D3 independent upperbound checkpoints.

예상 사용 흐름
-------------
1) 먼저 기존 independent upperbound 학습 실행:
   python baseline/baseline_upperbound_independent.py \
     --data_root /workspace/DCASE/task7_data \
     --d1_checkpoint /workspace/DCASE/checkpoints/BN/checkpoint_D1.pth \
     --domains D2 D3 \
     --epochs 120 \
     --batch_size 32 \
     --num_workers 8 \
     --learning_rate 1e-4 \
     --cuda \
     --save_dir /workspace/DCASE/runs/independent_upperbound

2) 이 코드로 D2/D3 독립 checkpoint를 불러와 augmentation-TTA routing 평가:
   python baseline/aug_tta_routing_independent.py \
     --data_root /workspace/DCASE/task7_data \
     --independent_root /workspace/DCASE/runs/independent_upperbound \
     --save_dir /workspace/DCASE/runs/aug_tta_routing_independent \
     --batch_size_eval 1 \
     --num_workers 0 \
     --cuda

명시적으로 checkpoint를 지정하고 싶으면:
   python baseline/aug_tta_routing_independent.py \
     --data_root /workspace/DCASE/task7_data \
     --d2_checkpoint /workspace/DCASE/runs/independent_upperbound/.../best_D2.pth \
     --d3_checkpoint /workspace/DCASE/runs/independent_upperbound/.../best_D3.pth \
     --save_dir /workspace/DCASE/runs/aug_tta_routing_independent \
     --cuda

주의
----
- data_root는 /workspace/DCASE/task7_data 기준.
- split 파일은 data_root/evaluation_setup/development_train.txt,
  data_root/evaluation_setup/development_test.txt 를 사용.
- wav 경로는 data_root / filename 으로 읽음.
- baseline/domain_net.py 의 MCnn14를 사용한다고 가정.
"""

import os
import sys
import json
import math
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import librosa
import numpy as np
import pandas as pd
from sklearn import metrics

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------
# Import baseline model
# ---------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
THIS_DIR = THIS_FILE.parent
REPO_ROOT = THIS_DIR.parent

# 실행 위치가 baseline/ 내부든 repo root든 모두 대응
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(REPO_ROOT))

try:
    from domain_net import MCnn14
except ImportError:
    from baseline.domain_net import MCnn14


# ---------------------------------------------------------
# Fixed setup
# ---------------------------------------------------------
SAMPLE_RATE = 32000
CLIP_SECONDS = 4
CLIP_SAMPLES = SAMPLE_RATE * CLIP_SECONDS

MEL_BINS = 64
FMIN = 50
FMAX = 14000
WINDOW_SIZE = 1024
HOP_SIZE = 320

CLASSES_NUM = 10
NB_TASKS = 3

TASK_TO_DOMAIN = {0: "D1", 1: "D2", 2: "D3"}
DOMAIN_TO_TASK = {"D1": 0, "D2": 1, "D3": 2}
EVAL_DOMAINS = ["D2", "D3"]
SEEN_TASKS = [1, 2]


# ---------------------------------------------------------
# Utils
# ---------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_split_df(data_root: str, split_name: str) -> pd.DataFrame:
    split_path = Path(data_root) / "evaluation_setup" / split_name
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    df = pd.read_csv(
        split_path,
        sep="\t",
        header=None,
        names=["filename", "target", "domain", "new_target"],
    )
    return df


def pad_truncate_sequence(x: np.ndarray, max_len: int = CLIP_SAMPLES) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if len(x) < max_len:
        return np.concatenate([x, np.zeros(max_len - len(x), dtype=np.float32)], axis=0)
    return x[:max_len].astype(np.float32)


def split_into_chunks(x: np.ndarray, chunk_size: int = CLIP_SAMPLES) -> List[np.ndarray]:
    x = np.asarray(x, dtype=np.float32)

    if len(x) <= chunk_size:
        return [pad_truncate_sequence(x, chunk_size)]

    chunks = []
    start = 0
    while start < len(x):
        chunks.append(pad_truncate_sequence(x[start:start + chunk_size], chunk_size))
        start += chunk_size

    return chunks


def entropy_from_probs(probs: torch.Tensor) -> torch.Tensor:
    eps = 1e-12
    return -(probs * torch.log(probs + eps)).sum(dim=-1)


def kl_to_mean(probs: torch.Tensor, mean_probs: torch.Tensor) -> torch.Tensor:
    """
    probs: [K, C]
    mean_probs: [C]
    return: scalar mean KL(p_k || mean)
    """
    eps = 1e-12
    return (probs * (torch.log(probs + eps) - torch.log(mean_probs.unsqueeze(0) + eps))).sum(dim=-1).mean()


def get_class_names(df: pd.DataFrame) -> Dict[int, str]:
    mapping = {}
    for _, row in df.iterrows():
        mapping[int(row["new_target"])] = str(row["target"])
    return mapping


def safe_name(x: str) -> str:
    return x.replace(".", "p").replace("/", "_").replace(" ", "_")


# ---------------------------------------------------------
# Audio TTA augmentations
# ---------------------------------------------------------
def aug_identity(audio: np.ndarray) -> np.ndarray:
    return audio.astype(np.float32)


def aug_gain(audio: np.ndarray, gain_db: float) -> np.ndarray:
    gain = 10.0 ** (gain_db / 20.0)
    return (audio * gain).astype(np.float32)


def aug_time_shift(audio: np.ndarray, shift_seconds: float) -> np.ndarray:
    shift = int(round(shift_seconds * SAMPLE_RATE))
    return np.roll(audio, shift).astype(np.float32)


def aug_noise(audio: np.ndarray, std: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, std, size=len(audio)).astype(np.float32)
    return (audio + noise).astype(np.float32)


def aug_fft_tilt(audio: np.ndarray, mode: str) -> np.ndarray:
    """
    간단한 device/channel-like perturbation.
    scipy 없이 FFT scale만 사용.
    - lowpass_like: 높은 주파수 성분을 약하게 줄임
    - highpass_like: 낮은 주파수 성분을 약하게 줄임
    """
    x = audio.astype(np.float32)
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / SAMPLE_RATE)

    scale = np.ones_like(freqs, dtype=np.float32)

    if mode == "lowpass_like":
        # 8kHz 이후를 부드럽게 감쇠
        start, end = 7000.0, 15000.0
        ramp = np.clip((freqs - start) / max(end - start, 1.0), 0.0, 1.0)
        scale = 1.0 - 0.45 * ramp
    elif mode == "highpass_like":
        # 300Hz 이하를 부드럽게 감쇠
        start, end = 50.0, 800.0
        ramp = np.clip((freqs - start) / max(end - start, 1.0), 0.0, 1.0)
        scale = 0.65 + 0.35 * ramp
    else:
        raise ValueError(f"Unknown fft tilt mode: {mode}")

    y = np.fft.irfft(spec * scale, n=len(x)).astype(np.float32)
    return y


def make_tta_audios(audio: np.ndarray, aug_set: str, base_seed: int = 0) -> List[np.ndarray]:
    """
    TTA용 deterministic augmentation set.
    routing에서 랜덤성이 너무 크면 비교가 흔들리므로 seed 기반으로 고정.
    """
    audio = audio.astype(np.float32)

    if aug_set == "identity":
        return [aug_identity(audio)]

    if aug_set == "light":
        return [
            aug_identity(audio),
            aug_gain(audio, -3.0),
            aug_gain(audio, 3.0),
            aug_time_shift(audio, -0.08),
            aug_time_shift(audio, 0.08),
        ]

    if aug_set == "device":
        return [
            aug_identity(audio),
            aug_gain(audio, -3.0),
            aug_gain(audio, 3.0),
            aug_fft_tilt(audio, "lowpass_like"),
            aug_fft_tilt(audio, "highpass_like"),
        ]

    if aug_set == "noise":
        return [
            aug_identity(audio),
            aug_noise(audio, 0.0003, base_seed + 11),
            aug_noise(audio, 0.0006, base_seed + 17),
            aug_gain(audio, -2.0),
            aug_gain(audio, 2.0),
        ]

    if aug_set == "strong":
        return [
            aug_identity(audio),
            aug_gain(audio, -4.0),
            aug_gain(audio, 4.0),
            aug_time_shift(audio, -0.12),
            aug_time_shift(audio, 0.12),
            aug_fft_tilt(audio, "lowpass_like"),
            aug_fft_tilt(audio, "highpass_like"),
            aug_noise(audio, 0.0005, base_seed + 23),
        ]

    raise ValueError(f"Unknown aug_set: {aug_set}")


# ---------------------------------------------------------
# Model loading
# ---------------------------------------------------------
def build_model(device: str) -> MCnn14:
    model = MCnn14(
        sample_rate=SAMPLE_RATE,
        window_size=WINDOW_SIZE,
        hop_size=HOP_SIZE,
        mel_bins=MEL_BINS,
        fmin=FMIN,
        fmax=FMAX,
        classes_num=CLASSES_NUM,
        nb_tasks=NB_TASKS,
    ).to(device)
    return model


def extract_state_dict(obj):
    if isinstance(obj, dict):
        for key in ["state_dict", "model_state_dict", "model", "net", "network"]:
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    return obj


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            out[k[len("module."):]] = v
        else:
            out[k] = v
    return out


def load_model_checkpoint(model: nn.Module, ckpt_path: str, device: str, strict: bool = False) -> None:
    ckpt_path = str(ckpt_path)
    state = torch.load(ckpt_path, map_location=device)
    state = extract_state_dict(state)
    state = strip_module_prefix(state)

    missing, unexpected = model.load_state_dict(state, strict=strict)
    print(f"[load] {ckpt_path}")
    print(f"[load] missing={len(missing)}, unexpected={len(unexpected)}")
    if len(missing) > 0:
        print(f"[missing example] {missing[:10]}")
    if len(unexpected) > 0:
        print(f"[unexpected example] {unexpected[:10]}")


def find_domain_checkpoint(independent_root: str, domain: str) -> Optional[Path]:
    """
    baseline_upperbound_independent.py의 저장 이름을 정확히 몰라도
    흔한 패턴을 최대한 자동 탐색.
    """
    root = Path(independent_root)
    if not root.exists():
        return None

    preferred_names = [
        f"best_{domain}.pth",
        f"{domain}_best.pth",
        f"checkpoint_{domain}.pth",
        f"{domain}.pth",
        f"after_{domain}.pth",
        "best.pth",
        "checkpoint.pth",
        "model.pth",
    ]

    candidate_dirs = [
        root,
        root / domain,
        root / f"domain_{domain}",
        root / f"train_{domain}",
        root / f"independent_{domain}",
        root / f"{domain}_independent",
    ]

    for d in candidate_dirs:
        if not d.exists():
            continue
        for name in preferred_names:
            p = d / name
            if p.exists():
                return p

    # recursive fallback: domain 문자열이 들어간 pth 우선
    pths = list(root.rglob("*.pth")) + list(root.rglob("*.pt"))
    domain_pths = [p for p in pths if domain.lower() in str(p).lower()]

    if len(domain_pths) > 0:
        domain_pths = sorted(domain_pths, key=lambda p: (len(str(p)), str(p)))
        return domain_pths[0]

    return None


def load_domain_models(args: argparse.Namespace, device: str) -> Dict[int, MCnn14]:
    d2_ckpt = Path(args.d2_checkpoint) if args.d2_checkpoint else find_domain_checkpoint(args.independent_root, "D2")
    d3_ckpt = Path(args.d3_checkpoint) if args.d3_checkpoint else find_domain_checkpoint(args.independent_root, "D3")

    if d2_ckpt is None or not d2_ckpt.exists():
        raise FileNotFoundError(
            "D2 checkpoint를 찾지 못했습니다. --d2_checkpoint로 직접 지정하세요. "
            f"searched independent_root={args.independent_root}"
        )

    if d3_ckpt is None or not d3_ckpt.exists():
        raise FileNotFoundError(
            "D3 checkpoint를 찾지 못했습니다. --d3_checkpoint로 직접 지정하세요. "
            f"searched independent_root={args.independent_root}"
        )

    models = {}
    ckpt_map = {1: d2_ckpt, 2: d3_ckpt}

    for task_id, ckpt in ckpt_map.items():
        domain = TASK_TO_DOMAIN[task_id]
        print("\n" + "=" * 100)
        print(f"[LOAD MODEL] {domain} independent checkpoint")
        print("=" * 100)

        model = build_model(device)
        load_model_checkpoint(model, str(ckpt), device=device, strict=False)
        model.eval()
        models[task_id] = model

    return models


# ---------------------------------------------------------
# Forward / scoring
# ---------------------------------------------------------
@torch.no_grad()
def forward_one_model_one_audio(
    model: MCnn14,
    audio: np.ndarray,
    task_id: int,
    device: str,
) -> torch.Tensor:
    """
    긴 오디오는 4초 chunk로 나누고 logits를 평균.
    return logits: [1, C]
    """
    chunks = split_into_chunks(audio, CLIP_SAMPLES)
    logits_list = []

    for chunk in chunks:
        x = torch.from_numpy(chunk.astype(np.float32)).unsqueeze(0).to(device)
        logits = model(x, task=task_id)
        logits_list.append(logits)

    logits = torch.stack(logits_list, dim=0).mean(dim=0)
    return logits


@torch.no_grad()
def compute_aug_outputs_for_task(
    model: MCnn14,
    audio: np.ndarray,
    task_id: int,
    device: str,
    aug_set: str,
    base_seed: int,
) -> Dict:
    tta_audios = make_tta_audios(audio, aug_set=aug_set, base_seed=base_seed)

    logits_list = []
    probs_list = []

    for aug_audio in tta_audios:
        logits = forward_one_model_one_audio(
            model=model,
            audio=aug_audio,
            task_id=task_id,
            device=device,
        )
        probs = torch.softmax(logits, dim=-1)

        logits_list.append(logits.squeeze(0))
        probs_list.append(probs.squeeze(0))

    logits_stack = torch.stack(logits_list, dim=0)  # [K, C]
    probs_stack = torch.stack(probs_list, dim=0)    # [K, C]

    mean_logits = logits_stack.mean(dim=0)
    mean_probs = probs_stack.mean(dim=0)
    mean_probs = mean_probs / mean_probs.sum().clamp_min(1e-12)

    entropy_mean_probs = float(entropy_from_probs(mean_probs.unsqueeze(0)).item())
    mean_entropy = float(entropy_from_probs(probs_stack).mean().item())
    consistency_kl = float(kl_to_mean(probs_stack, mean_probs).item())

    return {
        "logits_stack": logits_stack,
        "probs_stack": probs_stack,
        "mean_logits": mean_logits,
        "mean_probs": mean_probs,
        "entropy_mean_probs": entropy_mean_probs,
        "mean_entropy": mean_entropy,
        "consistency_kl": consistency_kl,
        "num_aug": len(tta_audios),
    }


@torch.no_grad()
def forward_all_candidate_tasks(
    models: Dict[int, MCnn14],
    audio: np.ndarray,
    device: str,
    aug_set: str,
    domain_aug_map: Optional[Dict[int, str]],
    base_seed: int,
) -> Dict[int, Dict]:
    outputs = {}

    for task_id, model in models.items():
        task_aug_set = aug_set
        if domain_aug_map is not None and task_id in domain_aug_map:
            task_aug_set = domain_aug_map[task_id]

        outputs[task_id] = compute_aug_outputs_for_task(
            model=model,
            audio=audio,
            task_id=task_id,
            device=device,
            aug_set=task_aug_set,
            base_seed=base_seed + task_id * 1000,
        )
        outputs[task_id]["aug_set"] = task_aug_set

    return outputs


def score_task(out: Dict, score_type: str, lambda_consistency: float) -> float:
    if score_type == "entropy_mean_probs":
        return float(out["entropy_mean_probs"])

    if score_type == "mean_entropy":
        return float(out["mean_entropy"])

    if score_type == "consistency":
        return float(out["entropy_mean_probs"] + lambda_consistency * out["consistency_kl"])

    if score_type == "kl_only":
        return float(out["consistency_kl"])

    raise ValueError(f"Unknown score_type: {score_type}")


def select_topk_by_score(scores: Dict[int, float], top_k: int) -> List[int]:
    return [t for t, _ in sorted(scores.items(), key=lambda x: x[1])[:top_k]]


def mix_probs_by_scores(
    outputs: Dict[int, Dict],
    selected_tasks: List[int],
    scores: Dict[int, float],
    tau: float,
) -> torch.Tensor:
    score_tensor = torch.tensor([scores[t] for t in selected_tasks], dtype=torch.float32)
    weights = torch.softmax(-score_tensor / tau, dim=0).to(outputs[selected_tasks[0]]["mean_probs"].device)

    final_probs = torch.zeros_like(outputs[selected_tasks[0]]["mean_probs"])

    for w, t in zip(weights, selected_tasks):
        final_probs = final_probs + w * outputs[t]["mean_probs"]

    final_probs = final_probs / final_probs.sum().clamp_min(1e-12)
    return final_probs


def get_final_probs_and_route(
    outputs: Dict[int, Dict],
    policy: str,
    true_task: Optional[int],
    score_type: str,
    lambda_consistency: float,
    top_k: int,
    tau: float,
) -> Tuple[torch.Tensor, int, Dict]:
    tasks = sorted(outputs.keys())

    if policy == "fixed_D2":
        chosen = 1
        return outputs[chosen]["mean_probs"], chosen, {"scores": {}}

    if policy == "fixed_D3":
        chosen = 2
        return outputs[chosen]["mean_probs"], chosen, {"scores": {}}

    if policy == "oracle":
        if true_task is None:
            raise ValueError("oracle policy requires true_task")
        chosen = true_task
        return outputs[chosen]["mean_probs"], chosen, {"scores": {}}

    if policy == "all_mean":
        final_probs = torch.stack([outputs[t]["mean_probs"] for t in tasks], dim=0).mean(dim=0)
        final_probs = final_probs / final_probs.sum().clamp_min(1e-12)
        return final_probs, -1, {"scores": {}}

    scores = {t: score_task(outputs[t], score_type, lambda_consistency) for t in tasks}

    if policy == "aug_hard":
        chosen = min(scores, key=scores.get)
        return outputs[chosen]["mean_probs"], chosen, {"scores": scores}

    if policy == "aug_moe":
        selected = select_topk_by_score(scores, top_k=top_k)
        final_probs = mix_probs_by_scores(outputs, selected, scores, tau=tau)
        return final_probs, selected[0], {"scores": scores, "selected_tasks": selected}

    raise ValueError(f"Unknown policy: {policy}")


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------
@torch.no_grad()
def evaluate_policy(
    models: Dict[int, MCnn14],
    df: pd.DataFrame,
    data_root: str,
    device: str,
    policy: str,
    score_type: str,
    lambda_consistency: float,
    top_k: int,
    tau: float,
    aug_set: str,
    domain_aug_map: Optional[Dict[int, str]],
    seed: int,
) -> Dict:
    for model in models.values():
        model.eval()

    y_true = []
    y_pred = []
    chosen_tasks = []
    rows = []

    df = df.reset_index(drop=True)

    for idx, row in df.iterrows():
        wav_path = Path(data_root) / row["filename"]
        if not wav_path.exists():
            raise FileNotFoundError(f"Audio file not found: {wav_path}")

        audio, _ = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
        audio = audio.astype(np.float32)

        target = int(row["new_target"])
        domain = str(row["domain"])
        true_task = DOMAIN_TO_TASK.get(domain, None)

        outputs = forward_all_candidate_tasks(
            models=models,
            audio=audio,
            device=device,
            aug_set=aug_set,
            domain_aug_map=domain_aug_map,
            base_seed=seed + idx * 31,
        )

        final_probs, chosen_task, aux = get_final_probs_and_route(
            outputs=outputs,
            policy=policy,
            true_task=true_task,
            score_type=score_type,
            lambda_consistency=lambda_consistency,
            top_k=top_k,
            tau=tau,
        )

        pred = int(torch.argmax(final_probs, dim=-1).item())
        conf = float(torch.max(final_probs, dim=-1).values.item())

        y_true.append(target)
        y_pred.append(pred)
        chosen_tasks.append(chosen_task)

        row_out = {
            "filename": row["filename"],
            "domain": domain,
            "target_name": row["target"],
            "target": target,
            "pred": pred,
            "correct": int(pred == target),
            "confidence": conf,
            "true_task": true_task,
            "true_domain": domain,
            "chosen_task": chosen_task,
            "chosen_domain": TASK_TO_DOMAIN.get(chosen_task, "MIX"),
            "route_correct": int(chosen_task == true_task)
            if chosen_task in TASK_TO_DOMAIN and true_task is not None
            else np.nan,
        }

        for task_id, out in outputs.items():
            d = TASK_TO_DOMAIN[task_id]
            row_out[f"aug_set_{d}"] = out["aug_set"]
            row_out[f"num_aug_{d}"] = out["num_aug"]
            row_out[f"entropy_mean_probs_{d}"] = out["entropy_mean_probs"]
            row_out[f"mean_entropy_{d}"] = out["mean_entropy"]
            row_out[f"consistency_kl_{d}"] = out["consistency_kl"]

        if "scores" in aux:
            for task_id, score in aux["scores"].items():
                row_out[f"score_{TASK_TO_DOMAIN[task_id]}"] = float(score)

        if "selected_tasks" in aux:
            row_out["selected_domains"] = ",".join([TASK_TO_DOMAIN[t] for t in aux["selected_tasks"]])

        rows.append(row_out)

        if (idx + 1) % 100 == 0:
            print(
                f"[eval] policy={policy}, score={score_type}, aug={aug_set}, "
                f"top_k={top_k}, tau={tau} | {idx + 1}/{len(df)}"
            )

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    chosen_tasks = np.array(chosen_tasks)

    acc = round(float((y_true == y_pred).mean() * 100.0), 2)
    cm = metrics.confusion_matrix(y_true, y_pred, labels=list(range(CLASSES_NUM)))

    pred_df = pd.DataFrame(rows)
    valid_route = pred_df["route_correct"].dropna()
    router_acc = round(float(valid_route.mean() * 100.0), 2) if len(valid_route) > 0 else np.nan

    route_hist = {}
    for task_id in SEEN_TASKS:
        route_hist[TASK_TO_DOMAIN[task_id]] = int(np.sum(chosen_tasks == task_id))
    route_hist["MIX"] = int(np.sum(chosen_tasks == -1))

    return {
        "acc": acc,
        "router_acc": router_acc,
        "route_hist": route_hist,
        "cm": cm,
        "pred_df": pred_df,
    }


def classwise_accuracy(pred_df: pd.DataFrame, class_name_map: Dict[int, str]) -> pd.DataFrame:
    rows = []
    for c in range(CLASSES_NUM):
        sub = pred_df[pred_df["target"] == c]
        n = len(sub)
        correct = int(sub["correct"].sum()) if n > 0 else 0
        acc = round(correct / max(n, 1) * 100.0, 2) if n > 0 else np.nan

        rows.append({
            "class_id": c,
            "class_name": class_name_map.get(c, str(c)),
            "n_samples": n,
            "correct": correct,
            "accuracy": acc,
        })

    return pd.DataFrame(rows)


def save_eval_outputs(result: Dict, out_dir: Path, prefix: str, class_name_map: Dict[int, str]) -> Dict:
    ensure_dir(out_dir)

    pred_df = result["pred_df"]
    pred_df.to_csv(out_dir / f"{prefix}_predictions.csv", index=False, encoding="utf-8-sig")

    cw = classwise_accuracy(pred_df, class_name_map)
    cw.to_csv(out_dir / f"{prefix}_classwise.csv", index=False, encoding="utf-8-sig")

    np.savetxt(out_dir / f"{prefix}_confusion.csv", result["cm"], delimiter=",", fmt="%d")

    compact = {
        "acc": result["acc"],
        "router_acc": result["router_acc"],
        "route_hist": result["route_hist"],
    }
    save_json(compact, out_dir / f"{prefix}_summary.json")
    return compact


def parse_domain_aug_map(domain_aug: str) -> Optional[Dict[int, str]]:
    """
    예:
      --domain_aug "D2:device,D3:noise"
    """
    if domain_aug is None or domain_aug.strip() == "":
        return None

    out = {}
    for item in domain_aug.split(","):
        item = item.strip()
        if not item:
            continue
        domain, aug = item.split(":")
        domain = domain.strip()
        aug = aug.strip()
        if domain not in DOMAIN_TO_TASK:
            raise ValueError(f"Unknown domain in --domain_aug: {domain}")
        out[DOMAIN_TO_TASK[domain]] = aug
    return out


def make_policy_grid(args: argparse.Namespace) -> List[Tuple[str, Dict]]:
    configs = []

    # 기본 비교군
    configs.append(("fixed_D2", {"policy": "fixed_D2", "score_type": "entropy_mean_probs", "top_k": 1, "tau": 1.0}))
    configs.append(("fixed_D3", {"policy": "fixed_D3", "score_type": "entropy_mean_probs", "top_k": 1, "tau": 1.0}))
    configs.append(("oracle", {"policy": "oracle", "score_type": "entropy_mean_probs", "top_k": 1, "tau": 1.0}))
    configs.append(("all_mean", {"policy": "all_mean", "score_type": "entropy_mean_probs", "top_k": 2, "tau": 1.0}))

    # Aug-TTA hard routing
    for score_type in args.score_types:
        configs.append((
            f"aug_hard_{score_type}",
            {
                "policy": "aug_hard",
                "score_type": score_type,
                "top_k": 1,
                "tau": 1.0,
            },
        ))

    # Aug-TTA MoE/top-k
    for score_type in args.score_types:
        for k in args.top_ks:
            for tau in args.taus:
                configs.append((
                    f"aug_moe_{score_type}_top{k}_tau{safe_name(str(tau))}",
                    {
                        "policy": "aug_moe",
                        "score_type": score_type,
                        "top_k": k,
                        "tau": tau,
                    },
                ))

    return configs


def run_suite(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    device = "cuda" if (args.cuda and torch.cuda.is_available()) else "cpu"
    print(f"[device] {device}")

    save_dir = Path(args.save_dir)
    ensure_dir(save_dir)

    train_df = load_split_df(args.data_root, "development_train.txt")
    test_df = load_split_df(args.data_root, "development_test.txt")

    print(f"[data_root] {args.data_root}")
    print(f"[split] train={len(train_df)}, test={len(test_df)}")
    for d in EVAL_DOMAINS:
        print(f"[data] {d} train={len(train_df[train_df['domain'] == d])}, test={len(test_df[test_df['domain'] == d])}")

    models = load_domain_models(args, device=device)

    domain_aug_map = parse_domain_aug_map(args.domain_aug)

    config_to_save = vars(args).copy()
    config_to_save["domain_aug_map"] = {
        TASK_TO_DOMAIN[k]: v for k, v in domain_aug_map.items()
    } if domain_aug_map is not None else None
    save_json(config_to_save, save_dir / "config.json")

    class_name_map = get_class_names(test_df)

    policy_configs = make_policy_grid(args)

    all_results = {}
    summary_rows = []

    for policy_name, cfg in policy_configs:
        print("\n" + "=" * 100)
        print(f"[POLICY] {policy_name}")
        print("=" * 100)

        policy_dir = save_dir / policy_name
        ensure_dir(policy_dir)

        policy_result = {}

        for domain in EVAL_DOMAINS:
            domain_df = test_df[test_df["domain"] == domain].copy().reset_index(drop=True)
            if args.max_eval_samples > 0 and len(domain_df) > args.max_eval_samples:
                domain_df = domain_df.sample(
                    n=args.max_eval_samples,
                    random_state=args.seed + DOMAIN_TO_TASK[domain],
                ).reset_index(drop=True)

            print(f"[evaluate] policy={policy_name}, domain={domain}, n={len(domain_df)}")

            result = evaluate_policy(
                models=models,
                df=domain_df,
                data_root=args.data_root,
                device=device,
                policy=cfg["policy"],
                score_type=cfg["score_type"],
                lambda_consistency=args.lambda_consistency,
                top_k=cfg["top_k"],
                tau=cfg["tau"],
                aug_set=args.aug_set,
                domain_aug_map=domain_aug_map,
                seed=args.seed,
            )

            compact = save_eval_outputs(
                result=result,
                out_dir=policy_dir,
                prefix=domain,
                class_name_map=class_name_map,
            )

            policy_result[domain] = compact

        avg = round(float(np.mean([policy_result[d]["acc"] for d in EVAL_DOMAINS])), 2)
        avg_router = round(float(np.nanmean([policy_result[d]["router_acc"] for d in EVAL_DOMAINS])), 2)

        policy_result["Avg"] = avg
        policy_result["Avg_router_acc"] = avg_router
        all_results[policy_name] = policy_result

        row = {
            "policy": policy_name,
            "Avg": avg,
            "Avg_router_acc": avg_router,
            "aug_set": args.aug_set,
            "domain_aug": args.domain_aug,
            "lambda_consistency": args.lambda_consistency,
        }

        for d in EVAL_DOMAINS:
            row[f"{d}_acc"] = policy_result[d]["acc"]
            row[f"{d}_router_acc"] = policy_result[d]["router_acc"]
            for route_domain, count in policy_result[d]["route_hist"].items():
                row[f"{d}_route_{route_domain}"] = count

        summary_rows.append(row)

        print(f"[RESULT] {policy_name} | Avg={avg} | Avg_router_acc={avg_router}")

    save_json(all_results, save_dir / "all_results.json")

    summary_df = pd.DataFrame(summary_rows)
    summary_by_avg = summary_df.sort_values("Avg", ascending=False)
    summary_by_router = summary_df.sort_values("Avg_router_acc", ascending=False)

    summary_by_avg.to_csv(save_dir / "summary_by_avg.csv", index=False, encoding="utf-8-sig")
    summary_by_router.to_csv(save_dir / "summary_by_router_acc.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print("[SUMMARY BY AVG]")
    print(summary_by_avg.to_string(index=False))

    print("\n[SUMMARY BY ROUTER ACC]")
    print(summary_by_router.to_string(index=False))

    print("\n[DONE]")
    print(f"Saved to: {save_dir}")


def main():
    parser = argparse.ArgumentParser()

    # root/path
    parser.add_argument("--data_root", type=str, default="/workspace/DCASE/task7_data")
    parser.add_argument("--independent_root", type=str, default="/workspace/DCASE/runs/independent_upperbound")
    parser.add_argument("--d2_checkpoint", type=str, default="")
    parser.add_argument("--d3_checkpoint", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="/workspace/DCASE/runs/aug_tta_routing_independent")

    # eval
    parser.add_argument("--batch_size_eval", type=int, default=1, help="현재 코드는 파일 단위 평가라 기록용 인자입니다.")
    parser.add_argument("--num_workers", type=int, default=0, help="현재 코드는 librosa 파일 단위 평가라 기록용 인자입니다.")
    parser.add_argument("--max_eval_samples", type=int, default=-1, help="debug용. -1이면 전체 평가.")
    parser.add_argument("--seed", type=int, default=1193)
    parser.add_argument("--cuda", action="store_true", default=False)

    # TTA
    parser.add_argument(
        "--aug_set",
        type=str,
        default="device",
        choices=["identity", "light", "device", "noise", "strong"],
        help="모든 후보 branch에 공통으로 적용할 TTA augmentation set.",
    )
    parser.add_argument(
        "--domain_aug",
        type=str,
        default="",
        help='도메인별 다른 TTA를 쓰고 싶을 때. 예: "D2:device,D3:noise". 비우면 --aug_set 공통 사용.',
    )
    parser.add_argument(
        "--score_types",
        type=str,
        nargs="+",
        default=["entropy_mean_probs", "mean_entropy", "consistency"],
        choices=["entropy_mean_probs", "mean_entropy", "consistency", "kl_only"],
    )
    parser.add_argument("--lambda_consistency", type=float, default=1.0)

    # MoE
    parser.add_argument("--top_ks", type=int, nargs="+", default=[2])
    parser.add_argument("--taus", type=float, nargs="+", default=[0.5, 1.0, 2.0])

    args = parser.parse_args()
    run_suite(args)


if __name__ == "__main__":
    main()
