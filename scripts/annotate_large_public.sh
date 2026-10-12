#!/usr/bin/env bash
# Usage: bash scripts/annotate_large_public.sh
# Reference pipeline that reproduces the large-public labels shipped by import_release.sh.
# Assumes TASTE_DATA_ROOT/TASTE_CKPT_ROOT, the CLaMP 3 repository at TASTE_CLAMP3_ROOT,
# Aria-MIDI deduped under $TASTE_DATA_ROOT/aria_midi/deduped, the benchmark manifest from
# import_release.sh, and the structure probe trained by annotate_curated.sh.
set -euo pipefail
cd "$(dirname "$0")/.."

L=${TASTE_DATA_ROOT:?set TASTE_DATA_ROOT}/large_public_annotation

# Manifest of the 371,053 deduped pieces and their global keys.
python -m taste.annotation.large_public.manifests deduped
python -m taste.representation.key --dataset aria --root "$TASTE_DATA_ROOT/aria_midi/deduped"

# Symbolic, Aria and CLaMP 3 features for the benchmark (default manifest) and the
# deduped corpus, plus CLaMP 3 text embeddings of the benchmark imagery captions.
for m in symbolic aria; do
  python -m taste.annotation.large_public.features $m
  python -m taste.annotation.large_public.features $m \
    --manifest "$L/deduped_manifest.parquet" --output "$L/deduped_${m}_features.npz"
done
python -m taste.annotation.large_public.features clamp3 --gpus 0
python -m taste.annotation.large_public.features clamp3 --gpus 0 \
  --manifest "$L/deduped_manifest.parquet" --output "$L/deduped_clamp3_features.npz"
python -m taste.annotation.large_public.features clamp3-text --gpus 0

# Fit the tag probes, gates and caption-space projection on the benchmark, then label
# the deduped corpus (attributes, tempo, retrieved imagery).
python -m taste.annotation.large_public.tag_probes fit \
  --benchmark-features clamp3_c2="$L/clamp3_features.npz" \
  --benchmark-features aria="$L/aria_features.npz" \
  --benchmark-features symbolic="$L/symbolic_features.npz"
python -m taste.annotation.large_public.tag_probes apply \
  --features clamp3_c2="$L/deduped_clamp3_features.npz" \
  --features aria="$L/deduped_aria_features.npz" \
  --features symbolic="$L/deduped_symbolic_features.npz" \
  --tempo-features "$L/deduped_symbolic_features.npz"

# Rewrite each retrieved imagery caption with a local Qwen3.5-27B-FP8, verify, and merge.
vllm serve Qwen/Qwen3.5-27B-FP8 --port 8100 &
SERVER=$!
until curl -sf http://127.0.0.1:8100/v1/models >/dev/null; do sleep 10; done
python -m taste.annotation.large_public.imagery_captions rewrite --ports 8100
kill $SERVER
python -m taste.annotation.large_public.imagery_captions verify
python -m taste.annotation.large_public.imagery_captions merge --backup-suffix pre_rewrite

# Structure-probe sections for every deduped piece.
python -m taste.annotation.structure_probe.extract_features --source large_public_full
python -m taste.annotation.structure_probe.infer --source large_public_full \
  --out-dir "$TASTE_DATA_ROOT/aria_midi/deduped/structure_labels"
