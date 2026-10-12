# TASTE: Taxonomy-Aware Symbolic Piano Generation from Textual Description

Official implementation of **TASTE**: generate expressive solo-piano MIDI from musical tags and a text description.

[Demo](https://taste-piano.github.io/) | [Dataset](https://huggingface.co/datasets/Itsuki-music/TASTE) | [Checkpoints](https://huggingface.co/Itsuki-music/TASTE)

## Generate your first MIDI

Start with the released **AR model**. It generates its own section plan from your tags and imagery description. You need Python 3.11+ and a CUDA GPU with bfloat16 support. The first run also downloads Qwen3-Embedding-4B to encode the text.

### 1. Install

```bash
git clone https://github.com/AndyWeasley2004/TASTE.git
cd TASTE
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The default install covers AR generation. No dataset, annotation service or API key is needed.

### 2. Download and generate

```bash
hf download Itsuki-music/TASTE --include "ar/high_quality/*" --local-dir checkpoints
python -m taste.inference.sample_ar \
    --model-dir checkpoints/ar/high_quality \
    --prompts-json prompts/example.json \
    --n-seeds 1 --rows-per-batch 1 --output-dir out/ar
```

Open **`out/ar/generated/library_k0.mid`** in your DAW or MIDI player with a piano instrument. The matching JSON file records the prompt, generated section plan and sampling details. The sampler produces MIDI, not rendered audio.

### 3. Make it yours

Edit [prompts/example.json](prompts/example.json):

```json
[
  {
    "name": "library",
    "emotion": "sad",
    "genre": "classical",
    "accompaniment": "sustained",
    "performance": "steady detached",
    "key": "A minor",
    "tempo": "moderate",
    "imagery": "An abandoned library in autumn, sunlight slanting through tall windows."
  }
]
```

The file is a JSON **list**: add more objects for more prompts. Keep all fields; set any of `emotion`, `genre`, `accompaniment` or `performance` to `null` to leave that attribute unspecified. See [the tag vocabulary](attributes.json) for accepted values. Increase `--n-seeds` for more variants.

## Other uses

| I want to… | Start here |
|---|---|
| Adjust generation, use another AR checkpoint, or run DiT | [Generation guide](docs/generation.md) |
| Predict musical attributes or section structure | [Probing guide](docs/probing.md) |
| Use the dataset, reproduce annotations, or train models | [Research guide](docs/research.md) |

Generation code lives in [`taste/inference/`](taste/inference/); probes live in [`taste/annotation/`](taste/annotation/). The research guide contains the full code map and checkpoint inventory.

## Citation

```bibtex
% TODO: arXiv id and final author list
@misc{yao2026taste,
  title         = {TASTE: Taxonomy-Aware Symbolic Piano Generation from Textual Description},
  author        = {Mingyang Yao and others},
  year          = {2026},
  eprint        = {TODO},
  archivePrefix = {arXiv}
}
```

## License

MIT; see [LICENSE](LICENSE).
