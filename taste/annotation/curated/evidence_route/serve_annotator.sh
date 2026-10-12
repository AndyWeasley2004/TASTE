#!/usr/bin/env bash
# Serve Qwen3.6-35B-A3B with vLLM for the evidence route (one replica per PORT).
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export HF_HUB_CACHE=${TASTE_CKPT_ROOT}/hf_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
if [ "$NGPU" -gt 4 ]; then
    echo "ERROR: at most 4 GPUs per replica" >&2
    exit 1
fi

MODEL_PATH="$HF_HUB_CACHE/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0"

vllm serve "$MODEL_PATH" \
    --served-model-name Qwen/Qwen3.6-35B-A3B \
    --host 127.0.0.1 \
    --port "${PORT:-8000}" \
    --tensor-parallel-size "$NGPU" \
    --max-model-len 16384 \
    --max-num-seqs 8 \
    --gpu-memory-utilization 0.90 \
    --enable-prefix-caching \
    --language-model-only
