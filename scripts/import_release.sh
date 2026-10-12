#!/usr/bin/env bash
# Usage: bash scripts/import_release.sh
# Downloads the TASTE dataset and expands every tier into the pipeline layout under
# $TASTE_DATA_ROOT. Assumes TASTE_DATA_ROOT is set and the `hf` CLI is installed.
# Aria-MIDI (https://huggingface.co/datasets/loubb/aria-midi) is fetched separately:
# its pruned and deduped subsets go to $TASTE_DATA_ROOT/aria_midi/{pruned,deduped}/,
# each holding data/ and metadata.json; the large-public labels refer to the deduped files.
set -euo pipefail
cd "$(dirname "$0")/.."

R=${TASTE_DATA_ROOT:?set TASTE_DATA_ROOT}/release

# Released MIDI and labels of the curated and high-quality tiers, the large-public
# labels and the tag-probe benchmark manifest.
hf download Itsuki-music/TASTE --repo-type dataset --local-dir "$R"

# Curated -> curated/pieces/<id>/, high-quality -> high_quality/{midi,metadata,prompts}/,
# large public -> aria_midi/deduped/{large_public_labels,structure_labels,key_estimates}/,
# benchmark -> large_public_annotation/benchmark.parquet.
for t in curated high_quality large_public large_public_annotation; do
  python -m taste.corpus.import_release --tier "$t" --release-dir "$R"
done
