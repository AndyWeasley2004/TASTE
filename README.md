# TASTE: Taxonomy-Aware Symbolic Piano Generation from Textual Description

This is the official implementation of "TASTE: Taxonomy-Aware Symbolic Piano Generation from Textual Description".

[Paper](TODO: arXiv) | [Demo](https://taste-piano.github.io/) | [Dataset](https://huggingface.co/datasets/Itsuki-music/TASTE) | [Checkpoints](https://huggingface.co/Itsuki-music/TASTE)

TASTE generates expressive solo-piano MIDI from six tags (key, tempo, emotion, genre, accompaniment, performance), a section plan (letters A–H) and a free-text imagery description. Two model families are released, each trained in four stages (Pretrain → Large public → Curated → High-quality):

- **TASTE (AR)**: a 0.89B autoregressive decoder (1.235B with the imagery cross-attention), with AdaLN-Zero tag conditioning and cross-attention over Qwen3-Embedding-4B states of the imagery description.
- **TASTE (DiT)**: a 397.6M rectified-flow latent diffusion transformer over a window-VAE codec (1 s windows, 0.5 s hop, 128-d latents).

## Environment

```bash
conda create -n taste python=3.11 && conda activate taste
pip install -r requirements.txt && pip install -e .
export TASTE_DATA_ROOT=/path/to/data          # datasets, corpora, latent stores
export TASTE_CKPT_ROOT=/path/to/checkpoints   # downloaded and trained checkpoints, HF cache
export TASTE_CLAMP3_ROOT=/path/to/clamp3      # CLaMP 3 repository (tag probes)
export TASTE_ARIA_ROOT=/path/to/aria          # Aria repository (high-quality section timing)
```

- Where the conda `libffi` shadows the system one, modules that import `pretty_midi` need `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7`.
- The section-fusion worker of the curated structure analysis runs in a Python 3.7 conda environment with `musicsections`, named by `TASTE_SECTION_FUSION_ENV` (default `structure`).

## External Models and Tools

| Name | Used for | Link |
|---|---|---|
| Qwen3-Embedding-4B | imagery states (AR cross-attention, DiT conditioning) | [HF](https://huggingface.co/Qwen/Qwen3-Embedding-4B) |
| Qwen3.6-35B-A3B | curated evidence route and audio-caption normalization, served with vLLM | [HF](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) |
| Gemini 3.5 Flash | curated audio route (`GEMINI_API_KEY`) | [Google AI](https://ai.google.dev/) |
| Qwen3.5-27B-FP8 | large-public imagery rewrite | [HF](https://huggingface.co/Qwen/Qwen3.5-27B-FP8) |
| CLaMP 3 | tag probes, imagery retrieval, curated consolidation | [GitHub](https://github.com/sanderwood/clamp3) |
| Aria / aria-medium-embedding | probe features, high-quality section timing | [GitHub](https://github.com/EleutherAI/aria), [HF](https://huggingface.co/loubb/aria-medium-embedding) |
| Beat This! | beat tracking and tempo | [GitHub](https://github.com/CPJKU/beat_this) |
| Transkun | audio-to-MIDI transcription | [GitHub](https://github.com/Yujia-Yan/Transkun) |
| Jina | web search for the evidence route (`JINA_API_KEY`) | [Jina](https://jina.ai/) |
| MuseTok | window-codec architecture | MuseTok: Symbolic Music Tokenization for Generation and Semantic Understanding |

## Quick Start

Download the main model and sample from a prompt file:

```bash
hf download Itsuki-music/TASTE --include "ar/high_quality/*" --local-dir $TASTE_CKPT_ROOT
python -m taste.inference.sample_ar --model-dir $TASTE_CKPT_ROOT/ar/high_quality \
    --prompts-json prompts.json --n-seeds 4 --output-dir out/ar
```

`prompts.json` is a list of prompts. `key` is one of the 24 keys (`C major` … `B minor`, flats as `Db`, `Eb`, `Ab`, `Bb`), `tempo` is `slow`, `moderate` or `fast`, and any of the four attributes may be `null`. The model writes its own section plan.

```json
[{"name": "library", "emotion": "sad", "genre": "classical", "accompaniment": "sustained",
  "performance": "steady detached", "key": "A minor", "tempo": "moderate",
  "imagery": "An abandoned library in autumn, leaves crunch underfoot as sunlight slants through tall windows."}]
```

Output: `out/ar/generated/<name>_k<i>.mid` plus `<name>_k<i>.json`, which records the prompt, seed, sampled section plan (`plan`), the emitted per-window sections, `ended_naturally` and the token ids. `prompts/high_quality_val.json` holds the 100 held-out high-quality prompts in the same format.

TASTE (DiT) takes its section plan and length from the high-quality latent-store index (`$TASTE_DATA_ROOT/dit_latents/high_quality/{val,pass0}/index.parquet`, built by `scripts/build_corpora.sh`), so prompt names are high-quality piece ids:

```bash
hf download Itsuki-music/TASTE --include "dit/high_quality/*" "codec/window_vae/*" --local-dir $TASTE_CKPT_ROOT
python -m taste.inference.sample_dit --model-dir $TASTE_CKPT_ROOT/dit/high_quality --prompts-json prompts/high_quality_val.json --output-dir out/dit
```

## Checkpoints

Download with `hf download Itsuki-music/TASTE --include "<path>/*" --local-dir $TASTE_CKPT_ROOT`.

| Path | Contents | Size |
|---|---|---|
| `ar/pretrain` | TASTE (AR) after pretraining, unconditional (safetensors) | 1.7 GB |
| `ar/large_public`, `ar/curated` | TASTE (AR) after the large-public and curated stages | 2.4 GB each |
| `ar/high_quality` | TASTE (AR) after the high-quality stage (main model) | 2.4 GB |
| `dit/high_quality` | TASTE (DiT) EMA weights (`ema.pt`), config, latent statistics | 1.5 GB |
| `codec/window_vae` | window-VAE codec and its token vocabulary | 351 MB |
| `probes/structure_probe` | structure probe (section boundaries and letters) | 7 MB |
| `probes/tag_probes` | large-public tag probes and imagery retrieval bank | 254 MB |
| `transcription` | Transkun weights for audio-to-MIDI transcription | 54 MB |

Training writes `<out_dir>/final`, and each stage config initialises from the previous stage's `final` (e.g. `ar/curated/final`); to start a stage from a released checkpoint, download it into `<stage>/final`.

## Data

| Tier | Pieces | Released | Source and labels |
|---|---|---|---|
| Pretrain | 817K | none | [Aria-MIDI](https://huggingface.co/datasets/loubb/aria-midi) pruned; key and tempo estimation, structure probe |
| Large public | 371,053 | labels | [Aria-MIDI](https://huggingface.co/datasets/loubb/aria-midi) deduped; tag probes, imagery retrieval and rewrite |
| Curated | 51,130 | MIDI + labels | LLM evidence or audio-caption routes, consolidated with the tag probes |
| High-quality | 9,949 | MIDI + labels | internally collected and annotated |

Curated sources: the [Aria-MIDI](https://huggingface.co/datasets/loubb/aria-midi) named subset, [PIAST-YT](https://github.com/Hayeonbang/PIAST), [Pop2Piano](https://github.com/sweetcocoa/pop2piano) covers, [PiJAMA](https://github.com/almostimplemented/PiJAMA), [POP909](https://github.com/music-x-lab/POP909-Dataset) and [EMOPIA](https://annahung31.github.io/EMOPIA/).

`bash scripts/import_release.sh` downloads the release and expands every tier into the pipeline layout under `$TASTE_DATA_ROOT` (Aria-MIDI goes to `$TASTE_DATA_ROOT/aria_midi/{pruned,deduped}`). The released labels are then in place; `scripts/annotate_curated.sh`, `scripts/annotate_high_quality.sh` and `scripts/annotate_large_public.sh` reproduce them (in that order), and `scripts/build_corpora.sh` builds the pretrain labels, the four token corpora with their verify gates, the imagery states and the DiT latent stores.

## Training

Launchers live in `scripts/`; configs in `configs/`. Run the stages in order, each initialised from the previous stage's `final`:

- **AR**: `configs/ar/{pretrain,large_public,curated,high_quality}.json`; `bash scripts/run_ar_pretrain.sh`, then `bash scripts/run_ar_stage.sh <stage>`. 4 GPUs (48 GB) for pretraining, 4 or 8 for the conditional stages; the per-device token batch in the config can be adjusted to the GPU count.
- **DiT**: `configs/dit/{pretrain,large_public,curated_a,curated_b,high_quality}.json`; `bash scripts/run_dit.sh <stage>` on 2, 4 or 8 GPUs.
- **Codec**: `configs/codec/tokenizer.yaml`; `bash scripts/train_codec.sh` builds the window store and trains the window VAE, and `scripts/build_corpora.sh` encodes the corpora into the DiT latent stores.

## Repository Layout

```
taste/
  representation/      MIDI tokenizer, vocabulary, key/tempo estimation, imagery states
  transcription/       Transkun audio-to-MIDI transcription
  annotation/
    high_quality/      constrained-clustering section timing
    curated/           source acquisition, evidence route, audio route, structure analysis
    large_public/      manifests, features, tag probes, imagery captions
    structure_probe/   section-structure probe: features, training, inference, label writing
    consolidation.py   tag-probe consolidation of curated labels
  corpus/              release import, corpus builders and verifiers, DiT latent stores
  models/              AR decoder, conditional AR wrapper, latent DiT
  codec/               window-VAE codec: windows, dataset, training
  training/            AR pretrain, AR conditional stage and DiT trainers
  inference/           AR and DiT samplers, latent-to-MIDI decoding
configs/               AR, DiT and codec configs
prompts/               the 100 held-out high-quality prompts
scripts/               release import, annotation, corpus, codec and training pipelines
```

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
