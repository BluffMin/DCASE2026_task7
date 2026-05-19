#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=/workspace/DCASE_ed/organized_paper_setup/fold1
D1_CKPT=/workspace/DCASE_ed/outputs/ablation_tau/base/base_best.pt
OUT_ROOT=/workspace/DCASE_ed/outputs/d1_absent_adapter_strategy_all

mkdir -p "${OUT_ROOT}"

PYTHONUNBUFFERED=1 python -u run_d1_absent_adapter_strategy.py \
  --data_root "${DATA_ROOT}" \
  --d1_checkpoint "${D1_CKPT}" \
  --save_dir "${OUT_ROOT}" \
  --adapter_mode all \
  --exp_ids $(seq 1 21) \
  --epochs 120 \
  --batch_size 32 \
  --num_workers 6 \
  --aug_mode crop_gain_shift_noise \
  --lwf_lambda 0.0 \
  --taus 0.7 1.0 1.3 \
  --hybrid_alphas 0.15 0.20 0.25 \
  --hybrid_betas 0.40 0.50 \
  --hybrid_deltas 0.60 0.75 \
  --hybrid_gammas 0.15 0.20 \
  --hybrid_taus 0.7 1.0 1.3 \
  --anchor_residual_scales 1.0 \
  --fallback_thresholds 0.70 \
  --fallback_lambdas 0.25 \
  --cuda \
  2>&1 | tee "${OUT_ROOT}/run.log"

echo "Done. Results: ${OUT_ROOT}"
