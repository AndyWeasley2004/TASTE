# TASTE: Taxonomy-Aware Symbolic Piano Generation from Textual Description

Official implementation of **TASTE**: generate expressive solo-piano MIDI from musical tags and a text description.

[Demo](https://taste-piano.github.io/) | [Dataset](https://huggingface.co/datasets/Itsuki-music/TASTE) | [Checkpoints](https://huggingface.co/Itsuki-music/TASTE)

## Generate your first MIDI

Requires Python 3.11+ and a CUDA GPU with bfloat16 support. The first run downloads Qwen3-Embedding-4B for text encoding.

### 1. Install

```bash
git clone https://github.com/AndyWeasley2004/TASTE.git
cd TASTE
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Download and generate

```bash
hf download Itsuki-music/TASTE --include "ar/high_quality/*" --local-dir checkpoints
python -m taste.inference.sample_ar \
    --model-dir checkpoints/ar/high_quality \
    --prompts-json prompts/example.json \
    --n-seeds 1 --rows-per-batch 1 --output-dir out/ar
```

Output: **`out/ar/generated/library_k0.mid`**, plus a JSON file with sampling details. Open the MIDI in a DAW or piano player.

### 3. Make it yours

Edit [prompts/example.json](prompts/example.json) with your tags and imagery description. See [attributes.json](attributes.json) for tag values; use `null` for unspecified attributes. Add prompts to the JSON list or increase `--n-seeds` for more outputs. AR generates its own section plan.

## Other uses

| I want to… | Start here |
|---|---|
| Adjust generation, use another AR checkpoint, or run DiT | [Generation guide](docs/generation.md) |
| Predict musical attributes or section structure | [Probing guide](docs/probing.md) |
| Use the dataset, reproduce annotations, or train models | [Research guide](docs/research.md) |

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
