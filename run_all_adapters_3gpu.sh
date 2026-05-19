#!/bin/bash

DATA_ROOT=/workspace/DCASE/task7_data
D1_CKPT=/workspace/DCASE/checkpoints/BN/checkpoint_D1.pth
OUT_ROOT=/workspace/DCASE/runs/task7_all_adapter_parallel

mkdir -p ${OUT_ROOT}

# GPU 1: exp 1~21
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python -u run_all_adapter_ablation.py \
  --data_root ${DATA_ROOT} \
  --d1_checkpoint ${D1_CKPT} \
  --save_dir ${OUT_ROOT}/gpu1 \
  --adapter_mode all \
  --exp_ids $(seq 1 21) \
  --epochs 120 \
  --batch_size 32 \
  --lr_incremental 1e-3 \
  --num_workers 6 \
  --cuda \
  > ${OUT_ROOT}/gpu1.log 2>&1 &

# GPU 2: exp 22~42
CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 python -u run_all_adapter_ablation.py \
  --data_root ${DATA_ROOT} \
  --d1_checkpoint ${D1_CKPT} \
  --save_dir ${OUT_ROOT}/gpu2 \
  --adapter_mode all \
  --exp_ids $(seq 22 42) \
  --epochs 120 \
  --batch_size 32 \
  --lr_incremental 1e-3 \
  --num_workers 6 \
  --cuda \
  > ${OUT_ROOT}/gpu2.log 2>&1 &

# GPU 3: exp 43~63
CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 python -u run_all_adapter_ablation.py \
  --data_root ${DATA_ROOT} \
  --d1_checkpoint ${D1_CKPT} \
  --save_dir ${OUT_ROOT}/gpu3 \
  --adapter_mode all \
  --exp_ids $(seq 43 63) \
  --epochs 120 \
  --batch_size 32 \
  --lr_incremental 1e-3 \
  --num_workers 6 \
  --cuda \
  > ${OUT_ROOT}/gpu3.log 2>&1 &

wait

echo "All adapter experiments finished."