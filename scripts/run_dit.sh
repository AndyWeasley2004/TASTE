#!/usr/bin/env bash
# Usage: bash scripts/run_dit.sh <pretrain|large_public|curated_a|curated_b|high_quality> [run_name] [extra train args]
# Trains one latent DiT stage from configs/dit/<stage>.json on 2, 4 or 8 GPUs;
# the trainer derives gradient accumulation from the GPU count and the config's
# global batch. Data and checkpoints resolve against $TASTE_DATA_ROOT and
# $TASTE_CKPT_ROOT.
set -euo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export OMP_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STAGE=${1:?usage: run_dit.sh <pretrain|large_public|curated_a|curated_b|high_quality> [run_name] [extra train args]}; shift
case "$STAGE" in pretrain|large_public|curated_a|curated_b|high_quality) ;; *) echo "ERROR: unknown stage '$STAGE'" >&2; exit 1;; esac
RUN_NAME=${1:-taste_dit_${STAGE}}; shift || true
NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
case "$NGPU" in
  2|4|8) ;;
  *) echo "ERROR: the DiT recipe runs on 2, 4 or 8 GPUs; got $NGPU" >&2; exit 1;;
esac
mkdir -p "runs/${RUN_NAME}"
torchrun --standalone --nproc_per_node="$NGPU" \
  -m taste.training.train_dit \
  --config "configs/dit/${STAGE}.json" \
  --run-name "$RUN_NAME" "$@" \
  2>&1 | tee -a "runs/${RUN_NAME}/console.log"
