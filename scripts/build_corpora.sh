#!/usr/bin/env bash
# Usage: bash scripts/build_corpora.sh
# Builds the four tokenized training corpora, the imagery states and the DiT latent stores.
# Assumes TASTE_DATA_ROOT/TASTE_CKPT_ROOT, Aria-MIDI under $TASTE_DATA_ROOT/aria_midi/{pruned,deduped},
# the imported release (import_release.sh), the structure probe (annotate_curated.sh) and
# the window-VAE codec at $TASTE_CKPT_ROOT/codec/window_vae.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${TASTE_DATA_ROOT:?set TASTE_DATA_ROOT}"

# Pretrain labels: global key, onset tempo (written by the feature sweep) and probe sections.
python -m taste.representation.key --dataset aria
python -m taste.annotation.structure_probe.extract_features --source aria
python -m taste.annotation.structure_probe.infer --source aria \
  --out-dir "$TASTE_DATA_ROOT/aria_midi/pruned/structure_labels"

# Token corpora, each followed by its verify gate.
for t in pretrain large_public curated high_quality; do
  python -m taste.corpus.build_$t
  python -m taste.corpus.verify_$t
done

# Qwen3-Embedding-4B states of the imagery descriptions for the three conditional corpora.
for t in large_public curated high_quality; do
  python -m taste.representation.imagery_states --data-dir "$TASTE_DATA_ROOT/corpora/$t"
  python -m taste.representation.imagery_states --data-dir "$TASTE_DATA_ROOT/corpora/$t" --finalize
done

# DiT latent stores at 0.5 s hop: the splits the configs/dit/*.json read.
latents() { for s in "${@:2}"; do python -m taste.corpus.build_dit_latents --stage "$1" --split "$s" --hop 0.5 --gpu 0; done; }
latents pretrain epoch0 epoch1 val
latents large_public pass0 pass1 pass5 val
latents curated pass0 pass1 pass2 val
latents high_quality pass0 pass1 pass4 val

# Latent statistics come from a 1 s-hop pass over large_public/pass0; every DiT config reads them.
python -m taste.corpus.build_dit_latents --stage large_public --split pass0 --gpu 0 \
  --out-root "$TASTE_DATA_ROOT/dit_latents_hop1"
cp "$TASTE_DATA_ROOT/dit_latents_hop1/large_public/latent_stats.npz" "$TASTE_DATA_ROOT/dit_latents/large_public/"
