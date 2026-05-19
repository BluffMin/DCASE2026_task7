"""
D1-absent setting: inference-only routing ablation launcher.

This file runs the existing experiment script without a bash wrapper.
It keeps the D2/D3 adapted checkpoint fixed and evaluates only D2/D3 test data.
D1 checkpoint is used only as a frozen anchor/fallback expert.

Run:
  python -u run_d1_anchor_fallback_on_best_cgsn.py

Optional:
  python -u run_d1_anchor_fallback_on_best_cgsn.py --gpu 1 --out /workspace/DCASE/runs/d1_anchor_fallback_on_best_cgsn
"""

import argparse
import os
import runpy
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="1", help="GPU id for CUDA_VISIBLE_DEVICES. Use empty string to disable setting it.")
    parser.add_argument("--script", type=str, default="run_anchor_residual_hybrid_moe_study.py")
    parser.add_argument("--data_root", type=str, default="/workspace/DCASE/task7_data")
    parser.add_argument("--checkpoint", type=str, default="/workspace/DCASE/runs/symaug_lwf_moe_study/symaug_crop_gain_shift_noise_lwf0p0/best.pth")
    parser.add_argument("--d1_checkpoint", type=str, default="/workspace/DCASE/checkpoints/BN/checkpoint_D1.pth")
    parser.add_argument("--out", type=str, default="/workspace/DCASE/runs/d1_anchor_fallback_on_best_cgsn")
    parser.add_argument("--seed", type=str, default="1193")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.gpu != "":
        # Must be set before the target script initializes CUDA.
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    script_path = Path(args.script)
    if not script_path.exists():
        # Try resolving relative to the current file location.
        alt = Path(__file__).resolve().parent / args.script
        if alt.exists():
            script_path = alt
        else:
            raise FileNotFoundError(
                f"Cannot find target script: {args.script}\n"
                f"Place run_anchor_residual_hybrid_moe_study.py in the same directory, "
                f"or pass --script /path/to/run_anchor_residual_hybrid_moe_study.py"
            )

    # Arguments passed to run_anchor_residual_hybrid_moe_study.py
    sys.argv = [
        str(script_path),
        "--data_root", args.data_root,
        "--checkpoint", args.checkpoint,
        "--d1_checkpoint", args.d1_checkpoint,
        "--save_dir", args.out,
        "--eval_domains", "D2", "D3",
        "--taus", "0.7", "1.0", "1.3",
        "--d3_biases", "0.15", "0.20", "0.25", "0.30",
        "--hybrid_alphas", "0.15", "0.20", "0.25",
        "--hybrid_betas", "0.40", "0.50",
        "--hybrid_deltas", "0.60", "0.75",
        "--hybrid_gammas", "0.15", "0.20",
        "--hybrid_taus", "0.7", "1.0", "1.3",
        "--anchor_residual_scales", "1.0",
        "--fallback_thresholds", "0.70",
        "--fallback_lambdas", "0.25",
        "--adapter_reduction", "16",
        "--seed", args.seed,
        "--cuda",
    ]

    print("=" * 100)
    print("[D1-absent inference-only experiment]")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(not set)')}")
    print(f"data_root     = {args.data_root}")
    print(f"checkpoint    = {args.checkpoint}")
    print(f"d1_checkpoint = {args.d1_checkpoint}")
    print(f"save_dir      = {args.out}")
    print("eval_domains  = D2 D3")
    print("=" * 100)

    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()
