#!/usr/bin/env bash
# Usage: bash scripts/train_codec.sh
# Trains the window-VAE codec from configs/codec/tokenizer.yaml. Assumes TASTE_DATA_ROOT/
# TASTE_CKPT_ROOT and the codec token rows at $TASTE_DATA_ROOT/codec_tokens/tokens/<tier>/tokens-*.parquet
# (one full-piece token row per piece, columns piece_id and input_ids).
set -euo pipefail
cd "$(dirname "$0")/.."

# Pack each tier's token rows into a ragged store of 5 s windows (codec_tokens/windows/<tier>).
python -m taste.codec.build_windows --stage all

# Train the codec; checkpoints go to $TASTE_CKPT_ROOT/codec/<run name>.
python -m taste.codec.train_tokenizer configs/codec/tokenizer.yaml
