#!/bin/bash

DATA_ROOT=/workspace/DCASE/task7_data
D1_CKPT=/workspace/DCASE/checkpoints/BN/checkpoint_D1.pth
OUT_ROOT=/workspace/DCASE/runs/symaug_lwf_moe_study

mkdir -p ${OUT_ROOT}

CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 python -u run_d2_to_d3_adapter_moe_study2.py \
  --data_root ${DATA_ROOT} \
  --d1_checkpoint ${D1_CKPT} \
  --save_dir ${OUT_ROOT} \
  --epochs 120 \
  --batch_size 32 \
  --lr_incremental 1e-3 \
  --num_workers 6 \
  --aug_modes none crop gain_shift crop_gain_shift crop_gain_shift_noise \
  --lwf_lambdas 0.0 0.3 0.5 1.0 \
  --kd_temperature 2.0 \
  --top_ks 2 \
  --taus 0.5 1.0 2.0 \
  --d3_biases 0.10 0.15 0.20 0.30 0.50 \
  --max_calib_samples -1 \
  --max_proto_samples -1 \
  --init_bn_from_d1 \
  --cuda \
  > ${OUT_ROOT}/run.log 2>&1

echo "Symmetric augmentation + LwF + MoE study finished."