#!/usr/bin/env bash
# Usage: bash scripts/run_ar_stage.sh <large_public|curated|high_quality> [run_name] [extra train args]
# Trains one conditional AR stage from configs/ar/<stage>.json on 4 or 8 GPUs;
# gradient accumulation is derived from the GPU count so the global token batch
# is the same either way. Data and checkpoints resolve against $TASTE_DATA_ROOT
# and $TASTE_CKPT_ROOT.
set -euo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STAGE=${1:?usage: run_ar_stage.sh <large_public|curated|high_quality> [run_name] [extra train args]}; shift
case "$STAGE" in large_public|curated|high_quality) ;; *) echo "ERROR: unknown stage '$STAGE'" >&2; exit 1;; esac
RUN_NAME=${1:-taste_ar_${STAGE}}; shift || true
NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
if [ "$NGPU" -ne 4 ] && [ "$NGPU" -ne 8 ]; then
  echo "ERROR: the ${STAGE} stage requires 4 or 8 GPUs, got $NGPU" >&2
  exit 1
fi
mkdir -p "runs/${RUN_NAME}"
torchrun --standalone --nproc_per_node="$NGPU" \
  -m taste.training.train_ar_stage \
  --config "configs/ar/${STAGE}.json" \
  --run-name "$RUN_NAME" "$@" \
  2>&1 | tee -a "runs/${RUN_NAME}/console.log"
