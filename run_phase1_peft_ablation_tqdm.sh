#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=python
SCRIPT_PATH="./run_task7_peft_ablation_tqdm.py"
TASK_ORDER="D2,D3"
NUM_TASKS=2
SEEDS=(0)
PRESETS=(
  residual
  bn_domain
  bn_adaptive
  bn_memory
  lora
  hybrid_bn_lora
  hybrid_mem_lora
  hybrid_router
)
LOG_ROOT="/workspace/DCASE/logs_ablation_bash"
mkdir -p "${LOG_ROOT}"

for seed in "${SEEDS[@]}"; do
  for preset in "${PRESETS[@]}"; do
    RUN_NAME="${preset}_seed${seed}"
    LOG_FILE="${LOG_ROOT}/${RUN_NAME}.log"
    echo "[$(date '+%F %T')] START ${RUN_NAME}" | tee -a "${LOG_FILE}"
    stdbuf -oL -eL ${PYTHON_BIN} "${SCRIPT_PATH}" \
      --preset "${preset}" \
      --task_order "${TASK_ORDER}" \
      --num_tasks "${NUM_TASKS}" \
      --seed "${seed}" \
      --num_workers 0 \
      2>&1 | tee -a "${LOG_FILE}"
    echo "[$(date '+%F %T')] END ${RUN_NAME}" | tee -a "${LOG_FILE}"
  done
done
