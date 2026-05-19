import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List

import librosa
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn import metrics
import matplotlib.pyplot as plt

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


# =========================================================
# Utils
# =========================================================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj: Dict, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_split_df(data_root: str, split_name: str) -> pd.DataFrame:
    split_path = Path(data_root) / "evaluation_setup" / split_name
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


def entropy_from_probs(probs: torch.Tensor) -> torch.Tensor:
    eps = 1e-12
    return -(probs * torch.log(probs + eps)).sum(dim=-1)


def get_class_names(df: pd.DataFrame) -> Dict[int, str]:
    mapping = {}
    for _, row in df.iterrows():
        mapping[int(row["new_target"])] = str(row["target"])
    return mapping


# =========================================================
# Adapter
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


# =========================================================
# Model
# =========================================================
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
        x = x1 + x2

        return x

    def forward(self, input: torch.Tensor, task: int = 1) -> torch.Tensor:
        x = self.forward_features(input, task=task)
        x = self.fc(x)
        return x


# =========================================================
# Load model
# =========================================================
def load_model(
    ckpt_path: str,
    adapter_blocks: List[int],
    adapter_reduction: int,
    device: str,
) -> MCnn14LateBNAdapter:
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

    state = torch.load(ckpt_path, map_location=device)

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    missing, unexpected = model.load_state_dict(state, strict=False)

    print(f"[load] checkpoint: {ckpt_path}")
    print(f"[load] missing={len(missing)}, unexpected={len(unexpected)}")

    if len(unexpected) > 0:
        print("[warning] unexpected keys example:", unexpected[:10])
    if len(missing) > 0:
        print("[warning] missing keys example:", missing[:10])

    model.set_active_adapters(adapter_blocks)
    model.eval()

    return model


# =========================================================
# Prediction
# =========================================================
@torch.no_grad()
def predict_domain_agnostic(
    model: MCnn14LateBNAdapter,
    df: pd.DataFrame,
    data_root: str,
    seen_tasks: List[int],
    adapter_blocks: List[int],
    device: str,
) -> pd.DataFrame:
    model.eval()
    model.set_active_adapters(adapter_blocks)

    rows = []

    for idx, row in df.iterrows():
        wav_path = Path(data_root) / row["filename"]
        target = int(row["new_target"])

        audio, _ = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
        chunks = split_into_chunks(audio, CLIP_SAMPLES)

        task_logits = []
        task_probs = []
        task_entropies = []

        for task_id in seen_tasks:
            chunk_logits = []

            for chunk in chunks:
                x = torch.from_numpy(chunk.astype(np.float32)).unsqueeze(0).to(device)
                logits = model(x, task=task_id)
                chunk_logits.append(logits)

            avg_logits = torch.stack(chunk_logits, dim=0).mean(dim=0)
            probs = torch.softmax(avg_logits, dim=-1)
            ent = entropy_from_probs(probs)

            task_logits.append(avg_logits)
            task_probs.append(probs)
            task_entropies.append(ent)

        task_entropies_tensor = torch.cat(task_entropies, dim=0)
        best_idx = torch.argmin(task_entropies_tensor).item()
        chosen_task = seen_tasks[best_idx]

        chosen_logits = task_logits[best_idx]
        chosen_probs = task_probs[best_idx]

        pred = torch.argmax(chosen_logits, dim=-1).item()
        confidence = torch.max(chosen_probs, dim=-1).values.item()

        out = {
            "filename": row["filename"],
            "domain": row["domain"],
            "target_name": row["target"],
            "target": target,
            "pred": pred,
            "correct": int(pred == target),
            "chosen_task": chosen_task,
            "chosen_domain": TASK_TO_DOMAIN[chosen_task],
            "confidence": float(confidence),
            "chosen_entropy": float(task_entropies_tensor[best_idx].item()),
        }

        for i, task_id in enumerate(seen_tasks):
            out[f"entropy_{TASK_TO_DOMAIN[task_id]}"] = float(task_entropies_tensor[i].item())

        rows.append(out)

        if (idx + 1) % 100 == 0:
            print(f"[predict] {idx + 1}/{len(df)}")

    return pd.DataFrame(rows)


# =========================================================
# Analysis tables
# =========================================================
def make_classwise_accuracy(pred_df: pd.DataFrame, class_name_map: Dict[int, str]) -> pd.DataFrame:
    rows = []

    for c in range(CLASSES_NUM):
        sub = pred_df[pred_df["target"] == c]
        n = len(sub)

        if n == 0:
            correct = 0
            acc = np.nan
        else:
            correct = int(sub["correct"].sum())
            acc = correct / n * 100.0

        rows.append({
            "class_id": c,
            "class_name": class_name_map.get(c, str(c)),
            "n_samples": n,
            "correct": correct,
            "accuracy": round(acc, 2) if not np.isnan(acc) else np.nan,
        })

    return pd.DataFrame(rows)


def make_confusion_matrix(pred_df: pd.DataFrame) -> np.ndarray:
    return metrics.confusion_matrix(
        pred_df["target"].values,
        pred_df["pred"].values,
        labels=list(range(CLASSES_NUM)),
    )


def make_confusion_pairs(pred_df: pd.DataFrame, class_name_map: Dict[int, str]) -> pd.DataFrame:
    wrong = pred_df[pred_df["correct"] == 0].copy()

    rows = []
    grouped = wrong.groupby(["target", "pred"]).size().reset_index(name="count")
    grouped = grouped.sort_values("count", ascending=False)

    for _, row in grouped.iterrows():
        t = int(row["target"])
        p = int(row["pred"])
        rows.append({
            "true_id": t,
            "true_name": class_name_map.get(t, str(t)),
            "pred_id": p,
            "pred_name": class_name_map.get(p, str(p)),
            "count": int(row["count"]),
        })

    return pd.DataFrame(rows)


def make_routing_summary(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for target_domain in sorted(pred_df["domain"].unique()):
        sub = pred_df[pred_df["domain"] == target_domain]

        for chosen_domain in ["D1", "D2", "D3"]:
            n = int((sub["chosen_domain"] == chosen_domain).sum())
            ratio = n / max(len(sub), 1) * 100.0

            rows.append({
                "target_domain": target_domain,
                "chosen_domain": chosen_domain,
                "n": n,
                "ratio": round(ratio, 2),
            })

    return pd.DataFrame(rows)


def make_classwise_routing(pred_df: pd.DataFrame, class_name_map: Dict[int, str]) -> pd.DataFrame:
    rows = []

    for c in range(CLASSES_NUM):
        sub = pred_df[pred_df["target"] == c]
        total = len(sub)

        row = {
            "class_id": c,
            "class_name": class_name_map.get(c, str(c)),
            "n_samples": total,
        }

        for d in ["D1", "D2", "D3"]:
            n = int((sub["chosen_domain"] == d).sum())
            row[f"route_to_{d}"] = n
            row[f"route_to_{d}_ratio"] = round(n / max(total, 1) * 100.0, 2)

        rows.append(row)

    return pd.DataFrame(rows)


# =========================================================
# Plots
# =========================================================
def save_confusion_plot(
    cm: np.ndarray,
    class_labels: List[str],
    path: Path,
    normalize: bool = False,
) -> None:
    if normalize:
        denom = cm.sum(axis=1, keepdims=True)
        denom[denom == 0] = 1
        mat = cm / denom
    else:
        mat = cm

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(mat)

    ax.set_xticks(np.arange(len(class_labels)))
    ax.set_yticks(np.arange(len(class_labels)))
    ax.set_xticklabels(class_labels, rotation=45, ha="right")
    ax.set_yticklabels(class_labels)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("D3 Confusion Matrix" + (" Normalized" if normalize else ""))

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            text = f"{mat[i, j]:.2f}" if normalize else str(int(mat[i, j]))
            ax.text(j, i, text, ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_bar_plot(df: pd.DataFrame, x_col: str, y_col: str, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df[x_col].astype(str), df[y_col])
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# =========================================================
# Audio feature stats
# =========================================================
def extract_audio_feature_stats(df: pd.DataFrame, data_root: str) -> pd.DataFrame:
    rows = []

    for idx, row in df.iterrows():
        wav_path = Path(data_root) / row["filename"]
        audio, sr = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)

        duration = len(audio) / sr
        rms = librosa.feature.rms(y=audio)[0]
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]

        rows.append({
            "filename": row["filename"],
            "domain": row["domain"],
            "target_name": row["target"],
            "target": int(row["new_target"]),
            "duration": float(duration),
            "rms_mean": float(np.mean(rms)),
            "rms_std": float(np.std(rms)),
            "zcr_mean": float(np.mean(zcr)),
            "zcr_std": float(np.std(zcr)),
            "centroid_mean": float(np.mean(centroid)),
            "centroid_std": float(np.std(centroid)),
            "bandwidth_mean": float(np.mean(bandwidth)),
            "bandwidth_std": float(np.std(bandwidth)),
            "rolloff_mean": float(np.mean(rolloff)),
            "rolloff_std": float(np.std(rolloff)),
        })

        if (idx + 1) % 100 == 0:
            print(f"[audio stats] {idx + 1}/{len(df)}")

    return pd.DataFrame(rows)


def summarize_audio_stats(audio_df: pd.DataFrame) -> pd.DataFrame:
    features = [
        "duration",
        "rms_mean",
        "rms_std",
        "zcr_mean",
        "zcr_std",
        "centroid_mean",
        "centroid_std",
        "bandwidth_mean",
        "bandwidth_std",
        "rolloff_mean",
        "rolloff_std",
    ]

    summary = audio_df.groupby(["domain", "target_name", "target"])[features].agg(["mean", "std", "min", "max"])
    summary.columns = ["_".join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()

    return summary


# =========================================================
# Main analysis
# =========================================================
def analyze(args: argparse.Namespace) -> None:
    device = "cuda" if (args.cuda and torch.cuda.is_available()) else "cpu"
    print(f"[device] {device}")

    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    adapter_blocks = args.adapter_blocks
    print(f"[adapter_blocks] {adapter_blocks}")

    model = load_model(
        ckpt_path=args.checkpoint,
        adapter_blocks=adapter_blocks,
        adapter_reduction=args.adapter_reduction,
        device=device,
    )

    train_df_all = load_split_df(args.data_root, "development_train.txt")
    test_df_all = load_split_df(args.data_root, "development_test.txt")

    d3_test = test_df_all[test_df_all["domain"] == "D3"].copy()
    d2_test = test_df_all[test_df_all["domain"] == "D2"].copy()

    class_name_map = get_class_names(test_df_all)
    class_labels = [class_name_map.get(i, str(i)) for i in range(CLASSES_NUM)]

    save_json(
        {
            "checkpoint": args.checkpoint,
            "adapter_blocks": adapter_blocks,
            "data_root": args.data_root,
            "output_dir": args.output_dir,
        },
        out_dir / "analysis_config.json",
    )

    # -------------------------
    # D3 prediction
    # -------------------------
    print("[analysis] D3 prediction")
    d3_pred = predict_domain_agnostic(
        model=model,
        df=d3_test,
        data_root=args.data_root,
        seen_tasks=[0, 1, 2],
        adapter_blocks=adapter_blocks,
        device=device,
    )

    d3_pred.to_csv(out_dir / "D3_predictions.csv", index=False, encoding="utf-8-sig")

    d3_acc = round(d3_pred["correct"].mean() * 100.0, 2)
    print(f"[D3 acc] {d3_acc}")

    d3_classwise = make_classwise_accuracy(d3_pred, class_name_map)
    d3_classwise.to_csv(out_dir / "D3_classwise_accuracy.csv", index=False, encoding="utf-8-sig")

    d3_cm = make_confusion_matrix(d3_pred)
    np.savetxt(out_dir / "D3_confusion_matrix.csv", d3_cm, delimiter=",", fmt="%d")

    save_confusion_plot(
        cm=d3_cm,
        class_labels=class_labels,
        path=out_dir / "D3_confusion_matrix.png",
        normalize=False,
    )

    save_confusion_plot(
        cm=d3_cm,
        class_labels=class_labels,
        path=out_dir / "D3_confusion_matrix_normalized.png",
        normalize=True,
    )

    d3_confusion_pairs = make_confusion_pairs(d3_pred, class_name_map)
    d3_confusion_pairs.to_csv(out_dir / "D3_confusion_pairs.csv", index=False, encoding="utf-8-sig")

    d3_routing = make_routing_summary(d3_pred)
    d3_routing.to_csv(out_dir / "D3_routing_summary.csv", index=False, encoding="utf-8-sig")

    d3_classwise_routing = make_classwise_routing(d3_pred, class_name_map)
    d3_classwise_routing.to_csv(out_dir / "D3_classwise_routing.csv", index=False, encoding="utf-8-sig")

    d3_wrong = d3_pred[d3_pred["correct"] == 0].copy()
    d3_wrong.to_csv(out_dir / "D3_wrong_samples.csv", index=False, encoding="utf-8-sig")

    d3_wrong_high_conf = d3_wrong.sort_values("confidence", ascending=False)
    d3_wrong_high_conf.to_csv(out_dir / "D3_wrong_high_confidence_samples.csv", index=False, encoding="utf-8-sig")

    d3_wrong_low_entropy = d3_wrong.sort_values("chosen_entropy", ascending=True)
    d3_wrong_low_entropy.to_csv(out_dir / "D3_wrong_low_entropy_samples.csv", index=False, encoding="utf-8-sig")

    save_bar_plot(
        df=d3_classwise,
        x_col="class_name",
        y_col="accuracy",
        title="D3 Class-wise Accuracy",
        path=out_dir / "D3_classwise_accuracy.png",
    )

    # -------------------------
    # D2 optional prediction
    # -------------------------
    if args.analyze_d2:
        print("[analysis] D2 prediction")

        d2_pred = predict_domain_agnostic(
            model=model,
            df=d2_test,
            data_root=args.data_root,
            seen_tasks=[0, 1, 2],
            adapter_blocks=adapter_blocks,
            device=device,
        )

        d2_pred.to_csv(out_dir / "D2_predictions.csv", index=False, encoding="utf-8-sig")

        d2_acc = round(d2_pred["correct"].mean() * 100.0, 2)

        d2_classwise = make_classwise_accuracy(d2_pred, class_name_map)
        d2_classwise.to_csv(out_dir / "D2_classwise_accuracy.csv", index=False, encoding="utf-8-sig")

        d2_routing = make_routing_summary(d2_pred)
        d2_routing.to_csv(out_dir / "D2_routing_summary.csv", index=False, encoding="utf-8-sig")

        d2_cm = make_confusion_matrix(d2_pred)
        np.savetxt(out_dir / "D2_confusion_matrix.csv", d2_cm, delimiter=",", fmt="%d")

        save_confusion_plot(
            cm=d2_cm,
            class_labels=class_labels,
            path=out_dir / "D2_confusion_matrix.png",
            normalize=False,
        )

        save_confusion_plot(
            cm=d2_cm,
            class_labels=class_labels,
            path=out_dir / "D2_confusion_matrix_normalized.png",
            normalize=True,
        )
    else:
        d2_acc = None

    # -------------------------
    # Audio feature stats
    # -------------------------
    if args.audio_stats:
        print("[analysis] D3 audio feature stats")

        d3_audio_stats = extract_audio_feature_stats(d3_test, args.data_root)
        d3_audio_stats.to_csv(out_dir / "D3_audio_feature_stats_per_file.csv", index=False, encoding="utf-8-sig")

        d3_audio_summary = summarize_audio_stats(d3_audio_stats)
        d3_audio_summary.to_csv(out_dir / "D3_audio_feature_stats_summary.csv", index=False, encoding="utf-8-sig")

    # -------------------------
    # Final report
    # -------------------------
    report = {
        "checkpoint": args.checkpoint,
        "adapter_blocks": adapter_blocks,
        "D3_acc": d3_acc,
        "D2_acc": d2_acc,
        "D3_num_samples": int(len(d3_pred)),
        "D3_num_correct": int(d3_pred["correct"].sum()),
        "D3_num_wrong": int((1 - d3_pred["correct"]).sum()),
        "D3_routing_counts": d3_pred["chosen_domain"].value_counts().to_dict(),
        "D3_classwise_accuracy": d3_classwise.to_dict(orient="records"),
        "D3_top_confusion_pairs": d3_confusion_pairs.head(20).to_dict(orient="records"),
    }

    save_json(report, out_dir / "analysis_report.json")

    print("\n" + "=" * 100)
    print("[D3 class-wise accuracy]")
    print(d3_classwise.to_string(index=False))

    print("\n[D3 routing summary]")
    print(d3_routing.to_string(index=False))

    print("\n[D3 top confusion pairs]")
    print(d3_confusion_pairs.head(20).to_string(index=False))

    print("\n[Saved outputs]")
    print(out_dir)
    print("=" * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--adapter_blocks", type=int, nargs="+", default=[4, 5, 6])
    parser.add_argument("--adapter_reduction", type=int, default=16)

    parser.add_argument("--cuda", action="store_true", default=False)
    parser.add_argument("--analyze_d2", action="store_true", default=False)
    parser.add_argument("--audio_stats", action="store_true", default=False)

    args = parser.parse_args()
    analyze(args)