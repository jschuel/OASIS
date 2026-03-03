#!/usr/bin/env bash
set -euo pipefail

echo "[INFO] host=$(hostname)"
echo "[INFO] pwd=$(pwd)"
echo "[INFO] torchrun=$TORCHRUN"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-}"

# Run DDP (2 processes on 2 GPUs)
"$TORCHRUN" --nproc_per_node=2 -m OASIS.train_entry \
  --rc "$RC" --rn "$RN" --ro "$RO" --run_id "$RUN_ID" \
  --data_root "$DATA_ROOT" --model_root "$MODEL_ROOT" \
  --seed "$SEED" --epochs "$EPOCHS" --patience "$PATIENCE" \
  --batch_size "$BATCH_SIZE" --lr "$LR"
