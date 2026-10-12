#!/usr/bin/env bash
# Usage: AUDIO_DIR=<audio of audio-only sources> bash scripts/annotate_curated.sh
# Reference pipeline that reproduces the curated labels shipped by import_release.sh.
# Assumes TASTE_DATA_ROOT/TASTE_CKPT_ROOT/TASTE_CLAMP3_ROOT, a Python 3.7 conda env with
# musicsections (TASTE_SECTION_FUSION_ENV), JINA_API_KEY for the evidence route,
# GEMINI_API_KEY for the audio route, and the high-quality teacher sections in place
# (from import_release.sh). Run before annotate_high_quality.sh, which needs the probe trained here.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${TASTE_DATA_ROOT:?set TASTE_DATA_ROOT}"
: "${AUDIO_DIR:?set AUDIO_DIR}"

# Transcribe the audio-only sources (PIAST-YT, Pop2Piano covers) to MIDI with Transkun.
python -m taste.transcription.audio_to_midi "$AUDIO_DIR" --devices cuda:0

# Beat tracking and section fusion on each piece's rendered audio; writes
# audio_beats.txt and meta_archive/structure_section_fusion.{json,lab} per piece.
python -m taste.annotation.curated.structure_analysis.run_batch "$TASTE_DATA_ROOT/curated/pieces" \
  --piece-layout --output "$TASTE_DATA_ROOT/curated/structure_analysis_state" --gpus 0

# Structure probe: features and teacher sections of the curated and high-quality tiers, then training.
python -m taste.annotation.structure_probe.extract_features --source teacher
python -m taste.annotation.structure_probe.train

# Tempo (beat_bpm + tempo class) and global key per piece.
python -m taste.representation.tempo --dataset curated
python -m taste.representation.key --dataset curated

# Evidence route: Qwen3.6-35B-A3B served locally with vLLM, web evidence through Jina.
PORT=8000 bash taste/annotation/curated/evidence_route/serve_annotator.sh &
SERVER=$!
until curl -sf http://127.0.0.1:8000/v1/models >/dev/null; do sleep 10; done
python -m taste.annotation.curated.evidence_route.run_batch --base-urls http://127.0.0.1:8000/v1
kill $(pgrep -P $SERVER) $SERVER

# Audio route: Gemini batch captions of the rendered audio, normalized to the vocabulary
# by a local Qwen3.6 server that normalize.py starts and stops itself.
python -m taste.annotation.curated.audio_route.build_manifest
python -m taste.annotation.curated.audio_route.prepare_batches plan
python -m taste.annotation.curated.audio_route.run_batch_sequence
python -m taste.annotation.curated.audio_route.normalize

# Replace the section-fusion structure with the probe's lettered sections.
python -m taste.annotation.structure_probe.write_labels --source curated --apply

# Consolidate each route's attribute labels with confident tag-probe predictions
# (reads the probe features under curated/features/).
for r in evidence_route audio_route; do
  python -m taste.annotation.consolidation --scope "$r" --execute
done
