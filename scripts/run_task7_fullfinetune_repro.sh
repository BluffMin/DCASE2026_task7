#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/task7_data}"
export D1_CKPT="${D1_CKPT:-${REPO_ROOT}/checkpoints/BN/checkpoint_D1.pth}"
export EVAL_ROOT="${EVAL_ROOT:-${REPO_ROOT}/evaluation_audio}"
export OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/runs/task7_fullfinetune_repro}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

SMOKE_TEST="${SMOKE_TEST:-0}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "[error] DATA_ROOT not found: ${DATA_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${D1_CKPT}" ]]; then
  echo "[error] D1_CKPT not found: ${D1_CKPT}" >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/checkpoints" "${OUT_ROOT}/results" "${OUT_ROOT}/configs" "${OUT_ROOT}/submission"

if [[ "${SMOKE_TEST}" == "1" ]]; then
  EPOCHS="${EPOCHS:-1}"
  NUM_WORKERS="${NUM_WORKERS:-0}"
  MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-1}"
  SKIP_SUBMISSION_FLAG="--skip_submission"
  SMOKE_FLAG="--smoke_test"
else
  EPOCHS="${EPOCHS:-120}"
  NUM_WORKERS="${NUM_WORKERS:-6}"
  MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:--1}"
  SKIP_SUBMISSION_FLAG=""
  SMOKE_FLAG=""
fi

BATCH_SIZE="${BATCH_SIZE:-32}"
BATCH_SIZE_EVAL="${BATCH_SIZE_EVAL:-64}"
LR="${LR:-0.0001}"
MIN_LR="${MIN_LR:-0.000001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
GRAD_CLIP="${GRAD_CLIP:-5.0}"
SEED="${SEED:-1193}"
TRAIN_WORK="${OUT_ROOT}/train_work"
D2_SRC="${TRAIN_WORK}/checkpoints/best_D2.pth"
D3_SRC="${TRAIN_WORK}/checkpoints/best_D3.pth"
D2_DST="${OUT_ROOT}/checkpoints/D2_fullfinetune_device_best.pth"
D3_DST="${OUT_ROOT}/checkpoints/D3_fullfinetune_gain_shift_best.pth"

{
  echo "# Full Fine-Tuning Preflight Inspection"
  echo
  echo "- REPO_ROOT: \`${REPO_ROOT}\`"
  echo "- DATA_ROOT: \`${DATA_ROOT}\`"
  echo "- D1_CKPT: \`${D1_CKPT}\`"
  echo "- OUT_ROOT: \`${OUT_ROOT}\`"
  echo "- Model: \`domain_net.py::MCnn14\`"
  echo "- Training script: \`train_domain_aug_tta_routing_d1_router_fast.py\`"
  echo "- Training loop: \`train_one_domain_expert\`"
  echo "- Validation loop: \`evaluate_single_model_loader\`"
  echo "- Augmentation: \`apply_train_aug\`"
  echo "- Dataset/splits: official \`development_train.txt\` and \`development_test.txt\`"
  echo "- Adapter modules: not used by this runner."
  echo "- Optimizer parameters: full \`model.parameters()\`, so this is full fine-tuning."
} > "${OUT_ROOT}/inspection_fullfinetune.md"

if [[ "${FORCE_RETRAIN}" == "1" ]]; then
  echo "[force] removing previous train_work and reproduced checkpoints under ${OUT_ROOT}" | tee "${OUT_ROOT}/logs/force_retrain.log"
  rm -rf "${TRAIN_WORK}"
  rm -f "${D2_DST}" "${D3_DST}" "${OUT_ROOT}/checkpoints/D2_dictionary.pth" "${OUT_ROOT}/checkpoints/D3_dictionary.pth"
fi

if [[ ! -f "${D2_DST}" || ! -f "${D3_DST}" ]]; then
  echo "[train] full MCnn14 D2/D3 experts" | tee "${OUT_ROOT}/logs/train_fullfinetune.log"
  PYTHONUNBUFFERED=1 python -u train_domain_aug_tta_routing_d1_router_fast.py \
    --data_root "${DATA_ROOT}" \
    --d1_checkpoint "${D1_CKPT}" \
    --save_dir "${TRAIN_WORK}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --batch_size_eval "${BATCH_SIZE_EVAL}" \
    --num_workers "${NUM_WORKERS}" \
    --lr "${LR}" \
    --min_lr "${MIN_LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --grad_clip "${GRAD_CLIP}" \
    --domain_train_aug D2:device,D3:gain_shift \
    --domain_tta_aug D1:identity,D2:identity,D3:identity \
    --score_types entropy_mean_probs \
    --top_ks 2 \
    --taus 3.0 4.0 \
    --max_train_batches "${MAX_TRAIN_BATCHES}" \
    --seed "${SEED}" \
    --cuda 2>&1 | tee -a "${OUT_ROOT}/logs/train_fullfinetune.log"

  cp -a "${D2_SRC}" "${D2_DST}"
  cp -a "${D3_SRC}" "${D3_DST}"
else
  echo "[reuse] ${D2_DST} and ${D3_DST}" | tee "${OUT_ROOT}/logs/train_fullfinetune.log"
fi

echo "[artifacts] generating metrics, tables, and submissions" | tee "${OUT_ROOT}/logs/generate_artifacts.log"
PYTHONUNBUFFERED=1 python -u generate_fullfinetune_repro_artifacts.py \
  --repo_root "${REPO_ROOT}" \
  --data_root "${DATA_ROOT}" \
  --d1_checkpoint "${D1_CKPT}" \
  --eval_root "${EVAL_ROOT}" \
  --out_root "${OUT_ROOT}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --batch_size_eval "${BATCH_SIZE_EVAL}" \
  --num_workers "${NUM_WORKERS}" \
  --lr "${LR}" \
  --min_lr "${MIN_LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --grad_clip "${GRAD_CLIP}" \
  --seed "${SEED}" \
  --cuda \
  ${SKIP_SUBMISSION_FLAG} \
  ${SMOKE_FLAG} 2>&1 | tee -a "${OUT_ROOT}/logs/generate_artifacts.log"

if [[ -f "${OUT_ROOT}/submission/Heo_SeoulTech_task7_model.py" ]]; then
  echo "[smoke] testing submission model loader" | tee "${OUT_ROOT}/logs/model_smoke_test.log"
  PYTHONPATH="${REPO_ROOT}:${OUT_ROOT}/submission:${PYTHONPATH:-}" python - <<'PY' 2>&1 | tee -a "${OUT_ROOT}/logs/model_smoke_test.log"
import os
import sys
from pathlib import Path
import torch
out = Path(os.environ["OUT_ROOT"]) / "submission"
sys.path.insert(0, str(out))
import Heo_SeoulTech_task7_model as model_py
for sid in range(1, 5):
    model = model_py.load_model(sid)
    x = torch.zeros(1, 128000)
    y = model(x)
    print(f"submission={sid} keys={list(y.keys())} D2_shape={tuple(y['D2'].shape)} D3_shape={tuple(y['D3'].shape)}")
PY
else
  echo "[smoke] submission model loader skipped; submission model file was not generated." | tee "${OUT_ROOT}/logs/model_smoke_test.log"
fi

echo "[DONE] command: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} SMOKE_TEST=${SMOKE_TEST} FORCE_RETRAIN=${FORCE_RETRAIN} bash scripts/run_task7_fullfinetune_repro.sh"
echo "[DONE] D2 checkpoint: ${D2_DST}"
echo "[DONE] D3 checkpoint: ${D3_DST}"
echo "[DONE] summary: ${OUT_ROOT}/results/all_systems_summary.csv"
echo "[DONE] submission: ${OUT_ROOT}/submission"
echo "[DONE] latex: ${OUT_ROOT}/results/report_table_system_configs.tex ${OUT_ROOT}/results/report_table_development_results.tex ${OUT_ROOT}/results/report_table_classwise_S1.tex"
echo "[DONE] consistency: ${OUT_ROOT}/report_consistency_check_fullfinetune.md"
