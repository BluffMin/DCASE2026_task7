#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=python
SCRIPT_PATH="run_task7_residual_focus.py"
LOG_ROOT="logs_residual_focus_bash"
mkdir -p "${LOG_ROOT}"

TASK_ORDER="${TASK_ORDER:-D2,D3}"
NUM_TASKS="${NUM_TASKS:-2}"
EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SEEDS="${SEEDS:-0 1 2}"

PRESETS=(
  bn_domain
  residual_base
  residual_late2
  residual_late1
  residual_gated_late2
  residual_gated_late1
  lora_last1_r4
  lora_last1_r8
  lora_last1_r16
)

echo "=== Residual-focus DCASE experiments ==="
echo "TASK_ORDER=${TASK_ORDER} NUM_TASKS=${NUM_TASKS} EPOCHS=${EPOCHS} BATCH_SIZE=${BATCH_SIZE}"

for seed in ${SEEDS}; do
  for preset in "${PRESETS[@]}"; do
    RUN_NAME="${preset}_seed${seed}"
    LOG_FILE="${LOG_ROOT}/${RUN_NAME}.log"

    echo "--------------------------------------------------"
    echo "Running ${RUN_NAME}"

    stdbuf -oL -eL ${PYTHON_BIN} "${SCRIPT_PATH}" \
      --preset "${preset}" \
      --task_order "${TASK_ORDER}" \
      --num_tasks "${NUM_TASKS}" \
      --epoch "${EPOCHS}" \
      --batch_size "${BATCH_SIZE}" \
      --num_workers "${NUM_WORKERS}" \
      --seed "${seed}" \
      2>&1 | tee "${LOG_FILE}"
  done
done

echo "All residual-focus runs finished."