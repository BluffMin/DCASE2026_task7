#!/usr/bin/env bash
set -euo pipefail

# =========================================================
# Anchor-residual Hybrid MoE study with D1/D2/D3 evaluation
# =========================================================

DATA_ROOT=/workspace/DCASE/task7_data

# Frozen D1 checkpoint
# D1 data is NOT used for prototype / calibration / routing-score fitting.
# This checkpoint is used only as anchor/fallback expert.
D1_CKPT=/workspace/DCASE/checkpoints/BN/checkpoint_D1.pth

# D2 -> D3 adaptation이 끝난 checkpoint
# 우선 crop_gain_shift_noise + LwF 0.0 실험의 best checkpoint 사용
ADAPTED_CKPT=/workspace/DCASE/runs/symaug_lwf_moe_study/symaug_crop_gain_shift_noise_lwf0p0/best.pth

OUT_ROOT=/workspace/DCASE/runs/anchor_residual_hybrid_moe_eval_cgsn_d1d2d3

mkdir -p "${OUT_ROOT}"

CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 python -u run_anchor_residual_hybrid_moe_study_d2d3.py \
  --data_root "${DATA_ROOT}" \
  --checkpoint "${ADAPTED_CKPT}" \
  --d1_checkpoint "${D1_CKPT}" \
  --save_dir "${OUT_ROOT}" \
  --eval_domains D1 D2 D3 \
  --taus 0.7 1.0 1.3 \
  --d3_biases 0.10 0.15 0.20 0.30 0.50 \
  --hybrid_alphas 0.15 0.20 0.25 \
  --hybrid_betas 0.40 0.50 \
  --hybrid_deltas 0.60 0.75 \
  --hybrid_gammas 0.15 0.20 \
  --hybrid_taus 0.7 1.0 1.3 \
  --anchor_residual_scales 0.25 0.50 0.75 1.0 \
  --fallback_thresholds 0.55 0.60 0.70 \
  --fallback_lambdas 0.25 0.50 0.75 \
  --adapter_reduction 16 \
  --seed 1193 \
  --cuda \
  > "${OUT_ROOT}/run.log" 2>&1

echo "Anchor-residual hybrid MoE study finished."
echo "Results: ${OUT_ROOT}"