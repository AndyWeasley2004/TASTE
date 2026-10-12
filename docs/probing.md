# Probing

[Back to README](../README.md) · [Generation](generation.md) · [Research](research.md)

Use probes to estimate attributes or section structure from existing MIDI. These tools are separate from generation. Run commands from the repository root.

```bash
pip install -e ".[probes]"
export TASTE_DATA_ROOT=/path/to/data
export TASTE_CKPT_ROOT=/path/to/checkpoints
hf download Itsuki-music/TASTE --include "probes/*" --local-dir "$TASTE_CKPT_ROOT"
```

| Probe | Input | Output | Code |
|---|---|---|---|
| Musical tags | CLaMP 3, Aria and symbolic feature vectors | Emotion, genre, accompaniment and performance predictions | [tag_probes.py](../taste/annotation/large_public/tag_probes.py) |
| Section structure | Frame-level symbolic MIDI features | Section boundaries and letters A–H | [structure_probe/](../taste/annotation/structure_probe/) |

## Musical tags

Tag extraction needs the external [CLaMP 3](https://github.com/sanderwood/clamp3) and [Aria](https://github.com/EleutherAI/aria) repositories and their model dependencies. Set `TASTE_CLAMP3_ROOT` and `TASTE_ARIA_ROOT` to those checkouts.

For your own MIDI collection, prepare a Parquet manifest with unique `piece_id` strings and `midi_path` strings pointing to local MIDI files. Extract the three feature families:

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

These are raw predictions and confidence scores. The corpus pipeline additionally applies confidence gates and metadata overrides. Its `tag_probes apply` command is specific to the full deduplicated corpus and validates its size; use the Python API above for smaller collections. Imagery retrieval and caption rewriting belong to the [annotation workflow](research.md), not this tag-scoring step.

## Section structure

The current CLI operates on the supported corpus layouts. For the deduplicated Aria-MIDI release at `$TASTE_DATA_ROOT/aria_midi/deduped/data/`:

```bash
python -m taste.annotation.structure_probe.extract_features --source large_public_full
python -m taste.annotation.structure_probe.infer --source large_public_full --out-dir sections
```

Inference uses CUDA. Features are stored under `$TASTE_DATA_ROOT/structure_probe/`; predictions go to its `sections/` directory. Other supported sources are listed by `--help`. There is no standalone `--midi` CLI for arbitrary files yet.

To retrain probes or reproduce the complete labels, see the [research guide](research.md) and [annotation launcher](../scripts/annotate_large_public.sh).
