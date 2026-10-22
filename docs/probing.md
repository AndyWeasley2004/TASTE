# Probing

[Back to README](../README.md) · [Generation](generation.md) · [Research](research.md) · [Code map](code-map.md)

Estimate attributes or section structure from MIDI. Run commands from the repository root. The [dependency guide](dependencies.md) covers all probe stages and external feature/teacher workers.

```bash
pip install -e ".[pipeline]"
export TASTE_DATA_ROOT=/path/to/data
export TASTE_CKPT_ROOT=/path/to/checkpoints
hf download Itsuki-music/TASTE --include "probes/*" --local-dir "$TASTE_CKPT_ROOT"
```

| Probe | Input | Output | Code |
|---|---|---|---|
| Musical tags | CLaMP 3, Aria and symbolic feature vectors | Emotion, genre, accompaniment and performance predictions | [tag_probes.py](../taste/annotation/large_public/tag_probes.py) |
| Section structure | Frame-level symbolic MIDI features | Section boundaries and letters A–H | [structure_probe/](../taste/annotation/structure_probe/) |

## Musical tags

Tag extraction needs the external CLaMP 3 checkout and its worker environment; set `TASTE_CLAMP3_ROOT` and `TASTE_CLAMP3_PYTHON` as described in the dependency guide. The Aria tag extractor loads `loubb/aria-medium-embedding` through Transformers with the revision pinned in `features.py`. `TASTE_ARIA_ROOT` is used by the separate high-quality section-timing pipeline.

For your own MIDI collection, use a fresh `TASTE_DATA_ROOT` to isolate feature caches (cached vectors are keyed by piece ID). Create a manifest with unique, filename-safe `piece_id` strings and absolute `midi_path` strings:

```python
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

files = sorted(Path("/absolute/path/to/midi").rglob("*.mid"))
assert files, "No MIDI files found"
rows = [{"piece_id": f"piece_{i:06d}", "midi_path": str(p)} for i, p in enumerate(files)]
pq.write_table(pa.Table.from_pylist(rows), "manifest.parquet")
```

Extract the three feature families:

```bash
python -m taste.annotation.large_public.features symbolic --manifest manifest.parquet --output features/symbolic.npz
python -m taste.annotation.large_public.features aria --manifest manifest.parquet --output features/aria.npz
python -m taste.annotation.large_public.features clamp3 --manifest manifest.parquet --output features/clamp3.npz --gpus 0
```

Score the aligned vectors using the released probes:

```python
from pathlib import Path
import pyarrow.parquet as pq
from taste.annotation.large_public.tag_probes import concat_features, score_attributes

ids = pq.read_table("manifest.parquet", columns=["piece_id"])["piece_id"].to_pylist()
x = concat_features({
    "clamp3_c2": Path("features/clamp3.npz"),
    "aria": Path("features/aria.npz"),
    "symbolic": Path("features/symbolic.npz"),
}, ids)
predictions, confidences = score_attributes(x)
for i, piece_id in enumerate(ids):
    print(piece_id, {field: str(values[i]) for field, values in predictions.items()})
```

Each archive contains `ids` and an aligned `features` matrix; `concat_features` reorders by ID and concatenates in **CLaMP 3 → Aria → symbolic** order. Keep the released feature models and preprocessing when using released probe weights. Validate archives before scoring, for example:

```bash
python -m taste.annotation.large_public.features verify \
    --manifest manifest.parquet --archive features/symbolic.npz --expected-dim 32
```

The verifier checks shape, unique IDs, manifest order and finite values; repeat for the embedding archives with their model's output dimension.

These are ungated predictions. `tag_probes apply` adds corpus-specific gates and metadata overrides and requires the full deduplicated corpus; use the API above for smaller collections.

## Section structure

The current CLI operates on the supported corpus layouts. For the deduplicated Aria-MIDI release at `$TASTE_DATA_ROOT/aria_midi/deduped/data/`:

```bash
python -m taste.annotation.structure_probe.extract_features --source large_public_full
python -m taste.annotation.structure_probe.infer --source large_public_full --out-dir sections
```

Requires CUDA. Predictions go to `$TASTE_DATA_ROOT/structure_probe/sections/`. See `--help` for other corpus sources; arbitrary single-file input is not supported.

## Reproduce on TASTE data

First follow the [research environment and release import](research.md). Importing the release supplies labels and the tag benchmark; running probes alone does not require rerunning the evidence/audio annotation services.

| Task | Entry point | Main artifacts |
|---|---|---|
| Refit tags and regenerate large-public labels | [annotate_large_public.sh](../scripts/annotate_large_public.sh): manifest → features → `tag_probes fit` → `apply` | `$TASTE_CKPT_ROOT/probes/tag_probes/{*.joblib,meta.json,imagery_retrieval.npz}`; `$TASTE_DATA_ROOT/aria_midi/deduped/large_public_labels/labels.parquet` |
| Retrain section boundaries and grouping | Commands below | `$TASTE_CKPT_ROOT/probes/structure_probe/{best.pt,eval.json}` |
| Infer sections on the deduplicated release | `extract_features` and `infer --source large_public_full` above | `sections/{labels.parquet,stats.json}` under `$TASTE_DATA_ROOT/structure_probe` |
| Rebuild all annotation routes | [Research guide](research.md) and the three `annotate_*.sh` launchers | Curated labels, high-quality metadata and large-public labels |

```bash
python -m taste.annotation.structure_probe.extract_features --source teacher
python -m taste.annotation.structure_probe.train --epochs 10 --batch-size 16 --lr 0.0003
```

The teacher source reads curated `pieces/<id>/performance.mid` with archived section-fusion labels, plus high-quality `midi/<id>.mid` and `metadata/<id>.json`. Training uses a seeded 5% validation split and repeats high-quality training rows three times. Inspect `eval.json` for boundary and grouping metrics. Retraining writes into the configured checkpoint root; use a separate root to retain the downloaded weights.

The large-public launcher also performs imagery retrieval, starts a local language model for caption rewriting, and runs section inference. Run only its feature/fit blocks when retraining tag probes. `apply` expects the full 371,053-piece corpus, metadata and key estimates; it is not a generic dataset scorer.

## Adapt training or structure probing to another dataset

**Tag training.** Extract the same three feature archives, plus `features clamp3-text --manifest benchmark.parquet --output features/text.npz --gpus 0`. Your benchmark needs `piece_id`, `midi_path`, `split_group`, `split` (`train`/`val`), `emotion`, `genre`, `accompaniment`, `expression`, `articulation`, `performance`, and `imagery`. Use the [tag vocabulary](../attributes.json); performance is the space-separated expression/articulation pair. Keep performances of the same work in one split group. Missing labels may be null, but each fitted head needs class coverage and enough groups for five-fold pruning; performance calibration requires `flexible` and `detached` classes, and imagery fitting needs nonempty captions.

```bash
python -m taste.annotation.large_public.tag_probes fit \
    --benchmark benchmark.parquet \
    --benchmark-features clamp3_c2=features/clamp3.npz \
    --benchmark-features aria=features/aria.npz \
    --benchmark-features symbolic=features/symbolic.npz \
    --text-features features/text.npz
```

`fit` produces deployment heads trained on all available benchmark labels after pruning; separate train-side models calibrate gates on validation rows. For an independent evaluation, exclude test pieces from the input benchmark entirely. Use `score_attributes` above on the new collection; recalibrate confidence gates on target-domain labels before using them to accept annotations.

**Structure inference.** There is no arbitrary-manifest CLI yet. A small adapter can reuse `read_notes_and_pedal` and `frame_features` in [extract_features.py](../taste/annotation/structure_probe/extract_features.py): write each nonempty MIDI's `(n_frames, 26)` array at the existing 0.5-second hop to `structure_probe/large_public_frames/shard_000.npz`, keyed by piece ID. Write `structure_probe/manifest_large_public.parquet` with `piece_id`, `shard` (0), `duration_sec` and `n_frames`. In a fresh data root, these artifacts can be passed directly to:

```bash
python -m taste.annotation.structure_probe.infer --source large_public --out-dir sections
```

Here `large_public` selects the artifact layout; it does not require Aria metadata at inference time. Split larger collections across shards. Inspect skipped/empty MIDI and long pieces before batching; the full-release extractor excludes pieces over one hour.

**Structure training.** A custom teacher adapter must additionally provide `source`, `boundaries_sec`, `section_letters`, `section_starts_sec`, and `section_ends_sec` in `structure_probe/manifest.parquet`, with arrays in `structure_probe/frames/shard_*.npz`. Times are seconds; repeated letters denote recurring sections within a piece. Adapt `train.py`'s source-specific weighting, validation reporting and piece-level split for your dataset, especially when multiple performances share a work. Keep the feature definition unchanged to reuse released weights.

For the implementation of every step, see the [code map](code-map.md).
