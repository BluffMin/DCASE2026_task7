import os
import sys
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import librosa
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

# baseline 폴더 안에서 실행한다고 가정
sys.path.insert(0, os.path.dirname(__file__))
from domain_net import MCnn14


# =========================================================
# Fixed setup
# =========================================================
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

# late_adapter_456
ADAPTER_BLOCKS = [4, 5, 6]

# D1 데이터가 없다는 조건: 최종 routing/gating 후보는 D2, D3만 사용
ROUTING_TASKS = [1, 2]
EVAL_DOMAINS = ["D2", "D3"]


# =========================================================
# Utils
# =========================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj: Dict, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_split_df(data_root: str, split_name: str) -> pd.DataFrame:
    split_path = Path(data_root) / "evaluation_setup" / split_name
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    return pd.read_csv(
        split_path,
        sep="\t",
        header=None,
        names=["filename", "target", "domain", "new_target"],
    )


def pad_truncate_sequence(x: np.ndarray, max_len: int = CLIP_SAMPLES) -> np.ndarray:
    if len(x) < max_len:
        return np.concatenate((x, np.zeros(max_len - len(x), dtype=x.dtype)))
    return x[:max_len]


def split_into_chunks(x: np.ndarray, chunk_size: int = CLIP_SAMPLES) -> List[np.ndarray]:
    if len(x) <= chunk_size:
        return [pad_truncate_sequence(x, chunk_size)]

    chunks = []
    start = 0
    while start < len(x):
        chunk = x[start:start + chunk_size]
        chunks.append(pad_truncate_sequence(chunk, chunk_size))
        start += chunk_size
    return chunks


def entropy_np(p: np.ndarray, eps: float = 1e-12) -> float:
    return float(-(p * np.log(p + eps)).sum())


def entropy_from_probs(probs: torch.Tensor) -> torch.Tensor:
    eps = 1e-12
    return -(probs * torch.log(probs + eps)).sum(dim=-1)


def margin_np(p: np.ndarray) -> float:
    top2 = np.sort(p)[-2:]
    return float(top2[-1] - top2[-2])


def energy_np(logits: np.ndarray, temperature: float = 1.0) -> float:
    # Lower energy can indicate higher confidence.
    z = logits / max(temperature, 1e-6)
    m = np.max(z)
    return float(-temperature * (m + np.log(np.exp(z - m).sum())))


def get_class_names(df: pd.DataFrame) -> Dict[int, str]:
    mapping = {}
    for _, row in df.iterrows():
        mapping[int(row["new_target"])] = str(row["target"])
    return mapping


# =========================================================
# Adapter model, same as previous study
# =========================================================
class Adapter2d(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.down = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.up = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.0))
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.xavier_uniform_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.alpha * self.up(self.act(self.down(x)))


class MCnn14LateBNAdapter(MCnn14):
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
        adapter_reduction: int = 16,
    ):
        super().__init__(
            sample_rate=sample_rate,
            window_size=window_size,
            hop_size=hop_size,
            mel_bins=mel_bins,
            fmin=fmin,
            fmax=fmax,
            classes_num=classes_num,
            nb_tasks=nb_tasks,
        )
        self.block1_adapters = nn.ModuleList([Adapter2d(64, reduction=adapter_reduction) for _ in range(nb_tasks)])
        self.block2_adapters = nn.ModuleList([Adapter2d(128, reduction=adapter_reduction) for _ in range(nb_tasks)])
        self.block3_adapters = nn.ModuleList([Adapter2d(256, reduction=adapter_reduction) for _ in range(nb_tasks)])
        self.block4_adapters = nn.ModuleList([Adapter2d(512, reduction=adapter_reduction) for _ in range(nb_tasks)])
        self.block5_adapters = nn.ModuleList([Adapter2d(1024, reduction=adapter_reduction) for _ in range(nb_tasks)])
        self.block6_adapters = nn.ModuleList([Adapter2d(2048, reduction=adapter_reduction) for _ in range(nb_tasks)])
        self._active_adapter_blocks = set()

    def set_active_adapters(self, blocks: List[int]) -> None:
        self._active_adapter_blocks = set(blocks)

    def _apply_adapter(self, x: torch.Tensor, task: int, block_idx: int) -> torch.Tensor:
        if block_idx not in self._active_adapter_blocks:
            return x
        if block_idx == 1:
            return self.block1_adapters[task](x)
        if block_idx == 2:
            return self.block2_adapters[task](x)
        if block_idx == 3:
            return self.block3_adapters[task](x)
        if block_idx == 4:
            return self.block4_adapters[task](x)
        if block_idx == 5:
            return self.block5_adapters[task](x)
        if block_idx == 6:
            return self.block6_adapters[task](x)
        return x

    def _forward_block(self, block: nn.Module, x: torch.Tensor, task: int, block_idx: int) -> torch.Tensor:
        x = F.relu_(block.bnF[task](block.conv1(x)))
        x = F.relu_(block.bnS[task](block.conv2(x)))
        x = self._apply_adapter(x, task, block_idx)
        x = F.avg_pool2d(x, kernel_size=(2, 2))
        return x

    def forward_features(self, input: torch.Tensor, task: int = 1) -> torch.Tensor:
        x = self.spectrogram_extractor(input)
        x = self.logmel_extractor(x)
        x = x.transpose(1, 3)
        x = self.bn0[task](x)
        x = x.transpose(1, 3)

        x = self._forward_block(self.conv_block1, x, task, 1)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self._forward_block(self.conv_block2, x, task, 2)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self._forward_block(self.conv_block3, x, task, 3)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self._forward_block(self.conv_block4, x, task, 4)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self._forward_block(self.conv_block5, x, task, 5)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self._forward_block(self.conv_block6, x, task, 6)
        x = F.dropout(x, p=0.2, training=self.training)

        x = torch.mean(x, dim=3)
        x1, _ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        return x1 + x2

    def forward(self, input: torch.Tensor, task: int = 1) -> torch.Tensor:
        feat = self.forward_features(input, task=task)
        return self.fc(feat)


def build_model(adapter_reduction: int, device: str) -> MCnn14LateBNAdapter:
    model = MCnn14LateBNAdapter(
        sample_rate=SAMPLE_RATE,
        window_size=WINDOW_SIZE,
        hop_size=HOP_SIZE,
        mel_bins=MEL_BINS,
        fmin=FMIN,
        fmax=FMAX,
        classes_num=CLASSES_NUM,
        nb_tasks=NB_TASKS,
        adapter_reduction=adapter_reduction,
    ).to(device)
    model.set_active_adapters(ADAPTER_BLOCKS)
    return model


def load_checkpoint(model: nn.Module, ckpt_path: str, device: str, strict: bool = False) -> None:
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=strict)
    print(f"[load] {ckpt_path}")
    print(f"[load] missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print("[missing example]", missing[:10])
    if unexpected:
        print("[unexpected example]", unexpected[:10])


# =========================================================
# Forward cache
# =========================================================
@torch.no_grad()
def forward_one_audio_all_branches(
    model: MCnn14LateBNAdapter,
    wav_path: Path,
    device: str,
    seen_tasks: List[int] = ROUTING_TASKS,
) -> Dict[str, Any]:
    audio, _ = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
    audio = audio.astype(np.float32)
    chunks = split_into_chunks(audio, CLIP_SAMPLES)

    out = {
        "logits": {},
        "probs": {},
        "features": {},
        "entropy": {},
        "maxprob": {},
        "margin": {},
        "energy": {},
        "pred": {},
    }

    for task_id in seen_tasks:
        chunk_logits = []
        chunk_feats = []
        for chunk in chunks:
            x = torch.from_numpy(chunk.astype(np.float32)).unsqueeze(0).to(device)
            feat = model.forward_features(x, task=task_id)
            logits = model.fc(feat)
            chunk_feats.append(feat)
            chunk_logits.append(logits)
        logits = torch.stack(chunk_logits, dim=0).mean(dim=0).squeeze(0)
        feat = torch.stack(chunk_feats, dim=0).mean(dim=0).squeeze(0)
        probs = torch.softmax(logits, dim=-1)

        z = logits.detach().cpu().numpy().astype(np.float32)
        p = probs.detach().cpu().numpy().astype(np.float32)
        f = feat.detach().cpu().numpy().astype(np.float32)

        out["logits"][task_id] = z
        out["probs"][task_id] = p
        out["features"][task_id] = f
        out["entropy"][task_id] = entropy_np(p)
        out["maxprob"][task_id] = float(np.max(p))
        out["margin"][task_id] = margin_np(p)
        out["energy"][task_id] = energy_np(z)
        out["pred"][task_id] = int(np.argmax(p))

    return out


def build_forward_cache(
    model: MCnn14LateBNAdapter,
    df: pd.DataFrame,
    data_root: str,
    device: str,
    seen_tasks: List[int] = ROUTING_TASKS,
    cache_name: str = "cache",
) -> List[Dict[str, Any]]:
    rows = []
    df = df.reset_index(drop=True)
    for idx, row in df.iterrows():
        wav_path = Path(data_root) / row["filename"]
        outputs = forward_one_audio_all_branches(model, wav_path, device, seen_tasks)
        rows.append({
            "filename": row["filename"],
            "domain": row["domain"],
            "target_name": row["target"],
            "target": int(row["new_target"]),
            "true_task": DOMAIN_TO_TASK[str(row["domain"])],
            "outputs": outputs,
        })
        if (idx + 1) % 100 == 0:
            print(f"[{cache_name}] {idx + 1}/{len(df)}")
    return rows


# =========================================================
# Temperature calibration
# =========================================================
def fit_temperature_for_logits(logits: torch.Tensor, labels: torch.Tensor, device: str) -> float:
    logits = logits.to(device)
    labels = labels.to(device)
    log_T = torch.zeros(1, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_T], lr=0.05, max_iter=100)

    def closure():
        optimizer.zero_grad()
        T = torch.exp(log_T).clamp(min=0.05, max=10.0)
        loss = F.cross_entropy(logits / T, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    T = float(torch.exp(log_T).detach().cpu().item())
    return max(min(T, 10.0), 0.05)


def fit_temperatures_from_cache(cache: List[Dict[str, Any]], device: str, seen_tasks: List[int] = ROUTING_TASKS) -> Dict[int, float]:
    temperatures = {}
    for task_id in seen_tasks:
        domain_name = TASK_TO_DOMAIN[task_id]
        items = [x for x in cache if x["domain"] == domain_name]
        logits = torch.tensor(np.stack([x["outputs"]["logits"][task_id] for x in items]), dtype=torch.float32)
        labels = torch.tensor([x["target"] for x in items], dtype=torch.long)
        T = fit_temperature_for_logits(logits, labels, device)
        temperatures[task_id] = T
        print(f"[temperature] {domain_name}: T={T:.4f}, n={len(items)}")
    return temperatures


def add_calibrated_outputs(cache: List[Dict[str, Any]], temperatures: Dict[int, float], seen_tasks: List[int] = ROUTING_TASKS) -> None:
    for item in cache:
        item["outputs"]["cal_probs"] = {}
        item["outputs"]["cal_entropy"] = {}
        for task_id in seen_tasks:
            logits = item["outputs"]["logits"][task_id]
            T = temperatures.get(task_id, 1.0)
            z = logits / max(T, 1e-6)
            z = z - np.max(z)
            p = np.exp(z)
            p = p / np.sum(p)
            item["outputs"]["cal_probs"][task_id] = p.astype(np.float32)
            item["outputs"]["cal_entropy"][task_id] = entropy_np(p)


# =========================================================
# Prototype / descriptor
# =========================================================
def l2_normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / (np.linalg.norm(x) + eps)


def build_prototypes_and_descriptors(
    cache: List[Dict[str, Any]],
    seen_tasks: List[int] = ROUTING_TASKS,
    eps: float = 1e-5,
) -> Dict[str, Any]:
    domain_proto = {}
    class_proto = {t: {} for t in seen_tasks}
    domain_desc = {}
    class_desc = {t: {} for t in seen_tasks}

    for task_id in seen_tasks:
        domain = TASK_TO_DOMAIN[task_id]
        items = [x for x in cache if x["domain"] == domain]
        feats = np.stack([l2_normalize_np(x["outputs"]["features"][task_id]) for x in items])
        mu = feats.mean(axis=0)
        var = feats.var(axis=0) + eps
        domain_proto[task_id] = l2_normalize_np(mu).astype(np.float32)
        domain_desc[task_id] = {"mu": mu.astype(np.float32), "var": var.astype(np.float32)}

        for c in range(CLASSES_NUM):
            c_items = [x for x in items if x["target"] == c]
            if len(c_items) == 0:
                continue
            c_feats = np.stack([l2_normalize_np(x["outputs"]["features"][task_id]) for x in c_items])
            c_mu = c_feats.mean(axis=0)
            c_var = c_feats.var(axis=0) + eps
            class_proto[task_id][c] = l2_normalize_np(c_mu).astype(np.float32)
            class_desc[task_id][c] = {"mu": c_mu.astype(np.float32), "var": c_var.astype(np.float32), "n": len(c_items)}

        print(f"[proto/desc] {domain}: n={len(items)}, class_count={len(class_proto[task_id])}")

    return {
        "domain_proto": domain_proto,
        "class_proto": class_proto,
        "domain_desc": domain_desc,
        "class_desc": class_desc,
    }


def cosine_distance_np(a: np.ndarray, b: np.ndarray) -> float:
    a = l2_normalize_np(a)
    b = l2_normalize_np(b)
    return float(1.0 - np.sum(a * b))


def diag_mahalanobis_np(x: np.ndarray, mu: np.ndarray, var: np.ndarray) -> float:
    z = l2_normalize_np(x)
    return float(np.mean(((z - mu) ** 2) / var))


def add_distance_features(cache: List[Dict[str, Any]], proto_desc: Dict[str, Any], seen_tasks: List[int] = ROUTING_TASKS) -> None:
    for item in cache:
        out = item["outputs"]
        out["proto_domain_dist"] = {}
        out["proto_class_dist"] = {}
        out["maha_domain_dist"] = {}
        out["maha_class_dist"] = {}

        for task_id in seen_tasks:
            feat = out["features"][task_id]
            out["proto_domain_dist"][task_id] = cosine_distance_np(feat, proto_desc["domain_proto"][task_id])

            min_cdist = 999.0
            for _, proto in proto_desc["class_proto"][task_id].items():
                min_cdist = min(min_cdist, cosine_distance_np(feat, proto))
            out["proto_class_dist"][task_id] = float(min_cdist)

            dd = proto_desc["domain_desc"][task_id]
            out["maha_domain_dist"][task_id] = diag_mahalanobis_np(feat, dd["mu"], dd["var"])

            min_mdist = 999.0
            for _, desc in proto_desc["class_desc"][task_id].items():
                min_mdist = min(min_mdist, diag_mahalanobis_np(feat, desc["mu"], desc["var"]))
            out["maha_class_dist"][task_id] = float(min_mdist)


# =========================================================
# Score normalization
# =========================================================
def fit_score_stats(cache: List[Dict[str, Any]], score_keys: List[str], seen_tasks: List[int] = ROUTING_TASKS) -> Dict[str, Dict[int, Dict[str, float]]]:
    stats = {}
    for key in score_keys:
        stats[key] = {}
        for task_id in seen_tasks:
            values = np.array([x["outputs"][key][task_id] for x in cache], dtype=np.float32)
            stats[key][task_id] = {
                "mean": float(values.mean()),
                "std": float(values.std() + 1e-6),
            }
    return stats


def zscore_value(value: float, stats: Dict[str, Any], key: str, task_id: int) -> float:
    m = stats[key][task_id]["mean"]
    s = stats[key][task_id]["std"]
    return float((value - m) / s)


# =========================================================
# Evaluation policies
# =========================================================
def softmax_np(scores: np.ndarray) -> np.ndarray:
    scores = scores - np.max(scores)
    exp = np.exp(scores)
    return exp / np.sum(exp)


def weighted_probs(cache_item: Dict[str, Any], weights_by_task: Dict[int, float], prob_key: str = "probs") -> np.ndarray:
    final = None
    for task_id, w in weights_by_task.items():
        p = cache_item["outputs"][prob_key][task_id]
        if final is None:
            final = w * p
        else:
            final = final + w * p
    return final


def hard_probs(cache_item: Dict[str, Any], task_id: int, prob_key: str = "probs") -> np.ndarray:
    return cache_item["outputs"][prob_key][task_id]


def eval_cache_policy(
    cache: List[Dict[str, Any]],
    policy_name: str,
    policy_kind: str,
    seen_tasks: List[int] = ROUTING_TASKS,
    tau: float = 1.0,
    d3_bias: float = 0.0,
    alpha: float = 1.0,
    beta: float = 0.0,
    delta: float = 0.0,
    score_stats: Optional[Dict[str, Any]] = None,
    router_model: Optional[nn.Module] = None,
    router_scaler: Optional[StandardScaler] = None,
    device: str = "cpu",
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    y_true, y_pred, chosen_tasks = [], [], []
    rows = []

    for item in cache:
        out = item["outputs"]
        true_task = item["true_task"]
        target = item["target"]

        weights_by_task = None
        chosen_task = -1
        prob_key = "probs"

        if policy_kind == "fixed_D2":
            chosen_task = 1
            final_probs = hard_probs(item, 1)
        elif policy_kind == "fixed_D3":
            chosen_task = 2
            final_probs = hard_probs(item, 2)
        elif policy_kind == "oracle":
            chosen_task = true_task
            final_probs = hard_probs(item, true_task)
        elif policy_kind == "entropy_hard":
            scores = {t: out["entropy"][t] for t in seen_tasks}
            chosen_task = min(seen_tasks, key=lambda t: scores[t])
            final_probs = hard_probs(item, chosen_task)
        elif policy_kind == "cal_entropy_hard":
            scores = {t: out["cal_entropy"][t] for t in seen_tasks}
            chosen_task = min(seen_tasks, key=lambda t: scores[t])
            final_probs = hard_probs(item, chosen_task, prob_key="cal_probs")
        elif policy_kind == "proto_class_hard":
            scores = {t: out["proto_class_dist"][t] for t in seen_tasks}
            chosen_task = min(seen_tasks, key=lambda t: scores[t])
            final_probs = hard_probs(item, chosen_task)
        elif policy_kind == "maha_class_hard":
            scores = {t: out["maha_class_dist"][t] for t in seen_tasks}
            chosen_task = min(seen_tasks, key=lambda t: scores[t])
            final_probs = hard_probs(item, chosen_task)
        elif policy_kind == "all_mean":
            weights_by_task = {t: 1.0 / len(seen_tasks) for t in seen_tasks}
            final_probs = weighted_probs(item, weights_by_task)
        elif policy_kind == "logit_mean":
            logits = np.stack([out["logits"][t] for t in seen_tasks], axis=0).mean(axis=0)
            logits = logits - logits.max()
            final_probs = np.exp(logits) / np.exp(logits).sum()
        elif policy_kind in ["entropy_moe", "biased_entropy_moe", "proto_class_moe", "maha_class_moe", "hybrid_moe"]:
            raw_scores = {}
            for t in seen_tasks:
                if policy_kind == "entropy_moe":
                    raw_scores[t] = out["entropy"][t]
                elif policy_kind == "biased_entropy_moe":
                    raw_scores[t] = out["entropy"][t]
                elif policy_kind == "proto_class_moe":
                    raw_scores[t] = out["proto_class_dist"][t]
                elif policy_kind == "maha_class_moe":
                    raw_scores[t] = out["maha_class_dist"][t]
                elif policy_kind == "hybrid_moe":
                    if score_stats is None:
                        raise ValueError("hybrid_moe requires score_stats")
                    e = zscore_value(out["entropy"][t], score_stats, "entropy", t)
                    pc = zscore_value(out["proto_class_dist"][t], score_stats, "proto_class_dist", t)
                    mc = zscore_value(out["maha_class_dist"][t], score_stats, "maha_class_dist", t)
                    raw_scores[t] = alpha * e + beta * pc + delta * mc
                else:
                    raise ValueError(policy_kind)
            if policy_kind in ["biased_entropy_moe", "hybrid_moe"] and 2 in raw_scores:
                raw_scores[2] = raw_scores[2] - d3_bias
            score_arr = np.array([raw_scores[t] for t in seen_tasks], dtype=np.float32)
            weights = softmax_np(-score_arr / max(tau, 1e-6))
            weights_by_task = {t: float(w) for t, w in zip(seen_tasks, weights)}
            chosen_task = seen_tasks[int(np.argmin(score_arr))]
            final_probs = weighted_probs(item, weights_by_task)
        elif policy_kind == "learned_router":
            if router_model is None or router_scaler is None:
                raise ValueError("learned_router requires router_model and router_scaler")
            x = make_router_feature_vector(item).reshape(1, -1)
            x = router_scaler.transform(x)
            xt = torch.tensor(x, dtype=torch.float32, device=device)
            with torch.no_grad():
                w = torch.softmax(router_model(xt), dim=-1).detach().cpu().numpy()[0]
            weights_by_task = {1: float(w[0]), 2: float(w[1])}
            chosen_task = 1 if w[0] >= w[1] else 2
            final_probs = weighted_probs(item, weights_by_task)
        else:
            raise ValueError(policy_kind)

        pred = int(np.argmax(final_probs))
        conf = float(np.max(final_probs))

        y_true.append(target)
        y_pred.append(pred)
        chosen_tasks.append(chosen_task)

        row = {
            "filename": item["filename"],
            "domain": item["domain"],
            "target": target,
            "pred": pred,
            "correct": int(pred == target),
            "true_task": true_task,
            "chosen_task": chosen_task,
            "chosen_domain": TASK_TO_DOMAIN.get(chosen_task, "MIX"),
            "route_correct": int(chosen_task == true_task) if chosen_task in TASK_TO_DOMAIN else np.nan,
            "confidence": conf,
        }
        for t in seen_tasks:
            d = TASK_TO_DOMAIN[t]
            row[f"entropy_{d}"] = out["entropy"][t]
            row[f"cal_entropy_{d}"] = out["cal_entropy"][t]
            row[f"proto_class_dist_{d}"] = out["proto_class_dist"][t]
            row[f"maha_class_dist_{d}"] = out["maha_class_dist"][t]
            if weights_by_task is not None:
                row[f"weight_{d}"] = weights_by_task.get(t, np.nan)
        rows.append(row)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = round(float((y_true == y_pred).mean() * 100.0), 2)
    cm = metrics.confusion_matrix(y_true, y_pred, labels=list(range(CLASSES_NUM)))
    pred_df = pd.DataFrame(rows)
    valid_route = pred_df["route_correct"].dropna()
    router_acc = round(float(valid_route.mean() * 100.0), 2) if len(valid_route) > 0 else np.nan

    route_hist = {}
    chosen_arr = np.array(chosen_tasks)
    for t in seen_tasks:
        route_hist[TASK_TO_DOMAIN[t]] = int(np.sum(chosen_arr == t))
    route_hist["MIX"] = int(np.sum(chosen_arr == -1))

    return {"policy": policy_name, "acc": acc, "router_acc": router_acc, "route_hist": route_hist, "cm": cm}, pred_df


# =========================================================
# Learned router
# =========================================================
def make_router_feature_vector(item: Dict[str, Any]) -> np.ndarray:
    out = item["outputs"]
    vals = []
    for key in ["entropy", "cal_entropy", "maxprob", "margin", "energy", "proto_domain_dist", "proto_class_dist", "maha_domain_dist", "maha_class_dist"]:
        vals.append(float(out[key][1]))
        vals.append(float(out[key][2]))
        vals.append(float(out[key][1] - out[key][2]))
    return np.array(vals, dtype=np.float32)


class RouterMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_router_dataset(cache: List[Dict[str, Any]], target_mode: str = "domain") -> Tuple[np.ndarray, np.ndarray, List[int]]:
    X, y, keep_idx = [], [], []
    for idx, item in enumerate(cache):
        target = item["target"]
        if target_mode == "domain":
            label = 0 if item["domain"] == "D2" else 1
        elif target_mode == "oracle_correct":
            pred_d2 = item["outputs"]["pred"][1]
            pred_d3 = item["outputs"]["pred"][2]
            ok_d2 = int(pred_d2 == target)
            ok_d3 = int(pred_d3 == target)
            if ok_d2 and not ok_d3:
                label = 0
            elif ok_d3 and not ok_d2:
                label = 1
            elif ok_d2 and ok_d3:
                # 둘 다 맞으면 더 confident한 branch를 target으로 둔다.
                label = 0 if item["outputs"]["maxprob"][1] >= item["outputs"]["maxprob"][2] else 1
            else:
                # 둘 다 틀린 sample은 router supervised target으로 애매하므로 제외
                continue
        else:
            raise ValueError(target_mode)
        X.append(make_router_feature_vector(item))
        y.append(label)
        keep_idx.append(idx)
    return np.stack(X), np.array(y, dtype=np.int64), keep_idx


def train_router(
    train_cache: List[Dict[str, Any]],
    device: str,
    target_mode: str = "domain",
    hidden_dim: int = 64,
    dropout: float = 0.1,
    lr: float = 1e-3,
    epochs: int = 200,
    batch_size: int = 64,
    seed: int = 1193,
) -> Tuple[RouterMLP, StandardScaler, Dict[str, Any]]:
    set_seed(seed)
    X, y, keep_idx = make_router_dataset(train_cache, target_mode=target_mode)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = RouterMLP(input_dim=Xs.shape[1], hidden_dim=hidden_dim, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    X_t = torch.tensor(Xs, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)

    n = len(y)
    indices = np.arange(n)
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        np.random.shuffle(indices)
        total_loss, total_correct = 0.0, 0
        for start in range(0, n, batch_size):
            idx = indices[start:start + batch_size]
            xb = X_t[idx].to(device)
            yb = y_t[idx].to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(idx)
            total_correct += int((logits.argmax(dim=-1) == yb).sum().item())
        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            train_acc = round(total_correct / max(n, 1) * 100.0, 2)
            avg_loss = round(total_loss / max(n, 1), 6)
            row = {"epoch": epoch, "loss": avg_loss, "router_train_acc": train_acc}
            history.append(row)
            print(f"[router:{target_mode}] {row}")

    info = {
        "target_mode": target_mode,
        "n_train": int(n),
        "class_count": {"D2": int(np.sum(y == 0)), "D3": int(np.sum(y == 1))},
        "history": history,
    }
    return model, scaler, info


# =========================================================
# Save helpers
# =========================================================
def classwise_accuracy(pred_df: pd.DataFrame, class_name_map: Dict[int, str]) -> pd.DataFrame:
    rows = []
    for c in range(CLASSES_NUM):
        sub = pred_df[pred_df["target"] == c]
        n = len(sub)
        correct = int(sub["correct"].sum()) if n > 0 else 0
        acc = round(correct / max(n, 1) * 100.0, 2) if n > 0 else np.nan
        rows.append({"class_id": c, "class_name": class_name_map.get(c, str(c)), "n_samples": n, "correct": correct, "accuracy": acc})
    return pd.DataFrame(rows)


def save_policy_result(
    result: Dict[str, Any],
    pred_df: pd.DataFrame,
    out_dir: Path,
    policy_name: str,
    domain: str,
    class_name_map: Dict[int, str],
) -> Dict[str, Any]:
    pdir = out_dir / policy_name
    ensure_dir(pdir)
    pred_df.to_csv(pdir / f"{domain}_predictions.csv", index=False, encoding="utf-8-sig")
    classwise_accuracy(pred_df, class_name_map).to_csv(pdir / f"{domain}_classwise.csv", index=False, encoding="utf-8-sig")
    np.savetxt(pdir / f"{domain}_confusion.csv", result["cm"], delimiter=",", fmt="%d")
    compact = {"acc": result["acc"], "router_acc": result["router_acc"], "route_hist": result["route_hist"]}
    save_json(compact, pdir / f"{domain}_summary.json")
    return compact


# =========================================================
# Main evaluation suite
# =========================================================
def build_policy_grid(args: argparse.Namespace) -> List[Tuple[str, Dict[str, Any]]]:
    configs = []
    base = [
        ("fixed_D2", {"policy_kind": "fixed_D2"}),
        ("fixed_D3", {"policy_kind": "fixed_D3"}),
        ("oracle", {"policy_kind": "oracle"}),
        ("entropy_hard", {"policy_kind": "entropy_hard"}),
        ("cal_entropy_hard", {"policy_kind": "cal_entropy_hard"}),
        ("proto_class_hard", {"policy_kind": "proto_class_hard"}),
        ("maha_class_hard", {"policy_kind": "maha_class_hard"}),
        ("all_mean", {"policy_kind": "all_mean"}),
        ("logit_mean", {"policy_kind": "logit_mean"}),
    ]
    configs.extend(base)

    for tau in args.taus:
        tn = str(tau).replace(".", "p")
        configs.append((f"entropy_moe_tau{tn}", {"policy_kind": "entropy_moe", "tau": tau}))
        configs.append((f"proto_class_moe_tau{tn}", {"policy_kind": "proto_class_moe", "tau": tau}))
        configs.append((f"maha_class_moe_tau{tn}", {"policy_kind": "maha_class_moe", "tau": tau}))

    for gamma in args.d3_biases:
        gn = str(gamma).replace(".", "p")
        configs.append((f"biased_entropy_moe_gamma{gn}", {"policy_kind": "biased_entropy_moe", "tau": args.default_tau, "d3_bias": gamma}))

    for alpha in args.hybrid_alphas:
        for beta in args.hybrid_betas:
            for delta in args.hybrid_deltas:
                for gamma in args.hybrid_gammas:
                    for tau in args.hybrid_taus:
                        name = (
                            f"hybrid_moe_a{alpha}_b{beta}_d{delta}_g{gamma}_t{tau}"
                            .replace(".", "p")
                        )
                        configs.append((name, {
                            "policy_kind": "hybrid_moe",
                            "alpha": alpha,
                            "beta": beta,
                            "delta": delta,
                            "d3_bias": gamma,
                            "tau": tau,
                        }))

    return configs


def evaluate_all_policies(
    train_cache: List[Dict[str, Any]],
    test_cache: List[Dict[str, Any]],
    args: argparse.Namespace,
    device: str,
    out_dir: Path,
    class_name_map: Dict[int, str],
    score_stats: Dict[str, Any],
) -> pd.DataFrame:
    ensure_dir(out_dir)
    rows = []
    configs = build_policy_grid(args)

    router_models = {}
    if args.train_learned_router:
        for target_mode in args.router_target_modes:
            model, scaler, info = train_router(
                train_cache=train_cache,
                device=device,
                target_mode=target_mode,
                hidden_dim=args.router_hidden_dim,
                dropout=args.router_dropout,
                lr=args.router_lr,
                epochs=args.router_epochs,
                batch_size=args.router_batch_size,
                seed=args.seed,
            )
            router_models[target_mode] = (model, scaler, info)
            save_json(info, out_dir / f"router_{target_mode}_info.json")
            torch.save(model.state_dict(), out_dir / f"router_{target_mode}.pth")
            configs.append((f"learned_router_{target_mode}", {"policy_kind": "learned_router", "router_target_mode": target_mode}))

    for policy_name, cfg in configs:
        print("\n" + "-" * 100)
        print(f"[POLICY] {policy_name}")
        print("-" * 100)

        policy_result = {}
        for domain in EVAL_DOMAINS:
            domain_cache = [x for x in test_cache if x["domain"] == domain]
            router_model = None
            router_scaler = None
            if cfg["policy_kind"] == "learned_router":
                target_mode = cfg["router_target_mode"]
                router_model, router_scaler, _ = router_models[target_mode]

            result, pred_df = eval_cache_policy(
                cache=domain_cache,
                policy_name=policy_name,
                policy_kind=cfg["policy_kind"],
                tau=cfg.get("tau", 1.0),
                d3_bias=cfg.get("d3_bias", 0.0),
                alpha=cfg.get("alpha", 1.0),
                beta=cfg.get("beta", 0.0),
                delta=cfg.get("delta", 0.0),
                score_stats=score_stats,
                router_model=router_model,
                router_scaler=router_scaler,
                device=device,
            )
            compact = save_policy_result(result, pred_df, out_dir, policy_name, domain, class_name_map)
            policy_result[domain] = compact
            print(f"[{policy_name}] {domain}: acc={compact['acc']}, router_acc={compact['router_acc']}, routes={compact['route_hist']}")

        avg = round(np.mean([policy_result[d]["acc"] for d in EVAL_DOMAINS]), 2)
        avg_router_acc = round(np.nanmean([policy_result[d]["router_acc"] for d in EVAL_DOMAINS]), 2)
        row = {
            "policy": policy_name,
            "Avg": avg,
            "Avg_router_acc": avg_router_acc,
            "D2_acc": policy_result["D2"]["acc"],
            "D3_acc": policy_result["D3"]["acc"],
            "D2_router_acc": policy_result["D2"]["router_acc"],
            "D3_router_acc": policy_result["D3"]["router_acc"],
            "D2_route_D2": policy_result["D2"]["route_hist"].get("D2", 0),
            "D2_route_D3": policy_result["D2"]["route_hist"].get("D3", 0),
            "D2_route_MIX": policy_result["D2"]["route_hist"].get("MIX", 0),
            "D3_route_D2": policy_result["D3"]["route_hist"].get("D2", 0),
            "D3_route_D3": policy_result["D3"]["route_hist"].get("D3", 0),
            "D3_route_MIX": policy_result["D3"]["route_hist"].get("MIX", 0),
        }
        rows.append(row)
        print(f"[RESULT] {policy_name}: Avg={avg}, Avg_router_acc={avg_router_acc}")

        summary_df = pd.DataFrame(rows)
        summary_df.sort_values("Avg", ascending=False).to_csv(out_dir / "summary_by_avg_live.csv", index=False, encoding="utf-8-sig")
        summary_df.sort_values("D3_acc", ascending=False).to_csv(out_dir / "summary_by_d3_live.csv", index=False, encoding="utf-8-sig")

    summary_df = pd.DataFrame(rows)
    summary_df.sort_values("Avg", ascending=False).to_csv(out_dir / "summary_by_avg.csv", index=False, encoding="utf-8-sig")
    summary_df.sort_values("D3_acc", ascending=False).to_csv(out_dir / "summary_by_d3.csv", index=False, encoding="utf-8-sig")
    summary_df.sort_values("Avg_router_acc", ascending=False).to_csv(out_dir / "summary_by_router_acc.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print("[SUMMARY BY AVG]")
    print(summary_df.sort_values("Avg", ascending=False).head(30).to_string(index=False))
    print("=" * 100)
    return summary_df


# =========================================================
# Main
# =========================================================
def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = "cuda" if (args.cuda and torch.cuda.is_available()) else "cpu"
    print(f"[device] {device}")

    out_dir = Path(args.save_dir)
    ensure_dir(out_dir)

    train_df_all = load_split_df(args.data_root, "development_train.txt")
    test_df_all = load_split_df(args.data_root, "development_test.txt")

    # D1 데이터는 없다고 가정하므로 train/test 모두 D2/D3만 사용
    train_df = train_df_all[train_df_all["domain"].isin(EVAL_DOMAINS)].copy().reset_index(drop=True)
    test_df = test_df_all[test_df_all["domain"].isin(EVAL_DOMAINS)].copy().reset_index(drop=True)

    class_name_map = get_class_names(test_df_all)

    config = vars(args).copy()
    config["routing_tasks"] = [TASK_TO_DOMAIN[t] for t in ROUTING_TASKS]
    config["eval_domains"] = EVAL_DOMAINS
    save_json(config, out_dir / "config.json")

    model = build_model(args.adapter_reduction, device)
    load_checkpoint(model, args.checkpoint, device=device, strict=False)
    model.eval()
    model.set_active_adapters(ADAPTER_BLOCKS)

    print("\n" + "=" * 100)
    print("[1] Build train forward cache for D2/D3")
    print("=" * 100)
    train_cache = build_forward_cache(model, train_df, args.data_root, device, ROUTING_TASKS, cache_name="train_cache")

    print("\n" + "=" * 100)
    print("[2] Fit branch-wise temperatures using D2/D3 train only")
    print("=" * 100)
    temperatures = fit_temperatures_from_cache(train_cache, device=device, seen_tasks=ROUTING_TASKS)
    save_json({str(k): v for k, v in temperatures.items()}, out_dir / "temperatures.json")
    add_calibrated_outputs(train_cache, temperatures, ROUTING_TASKS)

    print("\n" + "=" * 100)
    print("[3] Build prototype and mean-variance descriptors using D2/D3 train only")
    print("=" * 100)
    proto_desc = build_prototypes_and_descriptors(train_cache, ROUTING_TASKS)
    add_distance_features(train_cache, proto_desc, ROUTING_TASKS)

    score_stats = fit_score_stats(
        train_cache,
        score_keys=["entropy", "proto_class_dist", "maha_class_dist"],
        seen_tasks=ROUTING_TASKS,
    )
    save_json(score_stats, out_dir / "score_stats.json")

    print("\n" + "=" * 100)
    print("[4] Build test forward cache for D2/D3")
    print("=" * 100)
    test_cache = build_forward_cache(model, test_df, args.data_root, device, ROUTING_TASKS, cache_name="test_cache")
    add_calibrated_outputs(test_cache, temperatures, ROUTING_TASKS)
    add_distance_features(test_cache, proto_desc, ROUTING_TASKS)

    # Light cache metadata only. Full arrays are not saved to avoid huge files.
    save_json({
        "train_cache_n": len(train_cache),
        "test_cache_n": len(test_cache),
        "D2_train_n": int((train_df["domain"] == "D2").sum()),
        "D3_train_n": int((train_df["domain"] == "D3").sum()),
        "D2_test_n": int((test_df["domain"] == "D2").sum()),
        "D3_test_n": int((test_df["domain"] == "D3").sum()),
    }, out_dir / "cache_info.json")

    print("\n" + "=" * 100)
    print("[5] Evaluate hybrid MoE, descriptor routing, biased MoE, learned router")
    print("=" * 100)
    summary_df = evaluate_all_policies(
        train_cache=train_cache,
        test_cache=test_cache,
        args=args,
        device=device,
        out_dir=out_dir / "policy_eval",
        class_name_map=class_name_map,
        score_stats=score_stats,
    )

    summary_df.to_csv(out_dir / "summary_all_policies_by_avg.csv", index=False, encoding="utf-8-sig")
    print("\n[DONE]")
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True, help="after_D3.pth or best.pth from previous symmetric augmentation study")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--adapter_reduction", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1193)

    # Basic MoE / bias policies
    parser.add_argument("--taus", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--default_tau", type=float, default=0.5)
    parser.add_argument("--d3_biases", type=float, nargs="+", default=[0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50])

    # Hybrid score grid
    # score_i = alpha * z(entropy_i) + beta * z(proto_class_dist_i) + delta * z(maha_class_dist_i) - gamma_i
    parser.add_argument("--hybrid_alphas", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    parser.add_argument("--hybrid_betas", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    parser.add_argument("--hybrid_deltas", type=float, nargs="+", default=[0.0, 0.25, 0.5])
    parser.add_argument("--hybrid_gammas", type=float, nargs="+", default=[0.0, 0.05, 0.10, 0.15, 0.20])
    parser.add_argument("--hybrid_taus", type=float, nargs="+", default=[0.5, 1.0, 2.0])

    # Learned router
    parser.add_argument("--train_learned_router", action="store_true", default=False)
    parser.add_argument("--router_target_modes", type=str, nargs="+", default=["domain", "oracle_correct"], choices=["domain", "oracle_correct"])
    parser.add_argument("--router_hidden_dim", type=int, default=64)
    parser.add_argument("--router_dropout", type=float, default=0.1)
    parser.add_argument("--router_lr", type=float, default=1e-3)
    parser.add_argument("--router_epochs", type=int, default=200)
    parser.add_argument("--router_batch_size", type=int, default=64)

    parser.add_argument("--cuda", action="store_true", default=False)

    args = parser.parse_args()
    main(args)
