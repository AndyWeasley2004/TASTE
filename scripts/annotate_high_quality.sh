#!/usr/bin/env bash
# Usage: AUDIO_DIR=<performance audio> bash scripts/annotate_high_quality.sh
# Reference pipeline that reproduces the high-quality section timing, key and tempo shipped
# by import_release.sh. Assumes TASTE_DATA_ROOT/TASTE_CKPT_ROOT, the Aria repository at
# TASTE_ARIA_ROOT, LD_PRELOAD for pretty_midi, and the structure probe trained by annotate_curated.sh.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${TASTE_DATA_ROOT:?set TASTE_DATA_ROOT}"
: "${AUDIO_DIR:?set AUDIO_DIR}"

# Section timing: Aria layer-7 frame features, then constrained clustering of the
# prompt's section letters onto the performance.
python -m taste.annotation.high_quality.section_timing.process --stage all --device cuda:0

# Snap each section boundary to a nearby structure-probe peak.
python -m taste.annotation.structure_probe.write_labels --source high_quality --apply

# Global key from the MIDI.
python -m taste.representation.key --dataset high_quality

# Beat tracking of the performance audio, then beat_bpm and tempo class.
python -m taste.annotation.curated.structure_analysis.run_batch "$AUDIO_DIR" \
  --output "$TASTE_DATA_ROOT/high_quality" --stage beats --gpus 0
python -m taste.representation.tempo --dataset high_quality
