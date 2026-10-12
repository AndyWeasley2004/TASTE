#!/usr/bin/env bash
# Usage: bash scripts/run_ar_pretrain.sh [run_name] [extra train args]
# Pretrains the AR generator from configs/ar/pretrain.json on exactly 4 GPUs.
# Data and checkpoints resolve against $TASTE_DATA_ROOT and $TASTE_CKPT_ROOT.
set -euo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_NAME=${1:-taste_ar_pretrain}; shift || true
NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
if [ "$NGPU" -ne 4 ]; then
  echo "ERROR: the pretraining recipe requires exactly 4 GPUs, got $NGPU" >&2
  exit 1
fi
mkdir -p "runs/${RUN_NAME}"
torchrun --standalone --nproc_per_node="$NGPU" \
  -m taste.training.train_ar_pretrain \
  --config configs/ar/pretrain.json \
  --run-name "$RUN_NAME" "$@" \
  2>&1 | tee -a "runs/${RUN_NAME}/console.log"
