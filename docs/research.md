# Training and data pipelines

[Back to README](../README.md) · [Generation](generation.md) · [Probing](probing.md)

Run commands from the repository root.

## Environment

```bash
conda create -n taste python=3.11 && conda activate taste
pip install -e ".[research]"
export TASTE_DATA_ROOT=/path/to/data          # datasets, corpora, latent stores
export TASTE_CKPT_ROOT=/path/to/checkpoints   # downloaded and trained checkpoints, HF cache
export TASTE_CLAMP3_ROOT=/path/to/clamp3      # CLaMP 3 repository (tag probes)
export TASTE_ARIA_ROOT=/path/to/aria          # Aria repository (high-quality section timing)
```

- The section-fusion worker of the curated structure analysis runs in a Python 3.7 conda environment with `musicsections`, named by `TASTE_SECTION_FUSION_ENV` (default `structure`).

Annotation launchers list their model and service prerequisites. The evidence route uses `JINA_API_KEY`; the audio route uses `GEMINI_API_KEY`.

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

`bash scripts/import_release.sh` downloads the release and expands every tier into the pipeline layout under `$TASTE_DATA_ROOT` (Aria-MIDI goes to `$TASTE_DATA_ROOT/aria_midi/{pruned,deduped}`). The released labels are then in place; `scripts/annotate_curated.sh`, `scripts/annotate_high_quality.sh` and `scripts/annotate_large_public.sh` reproduce them (in that order), and `scripts/build_corpora.sh` builds the pretrain labels, the four token corpora with their verify gates, the imagery states and the DiT latent stores.

## Training

Launchers live in `scripts/`; configs in `configs/`. Run the stages in order, each initialised from the previous stage's `final`:

- **AR**: `configs/ar/{pretrain,large_public,curated,high_quality}.json`; `bash scripts/run_ar_pretrain.sh`, then `bash scripts/run_ar_stage.sh <stage>`. 4 GPUs (48 GB) for pretraining, 4 or 8 for the conditional stages; the per-device token batch in the config can be adjusted to the GPU count.
- **DiT**: `configs/dit/{pretrain,large_public,curated_a,curated_b,high_quality}.json`; `bash scripts/run_dit.sh <stage>` on 2, 4 or 8 GPUs.
- **Codec**: `configs/codec/tokenizer.yaml`; `bash scripts/train_codec.sh` builds the window store and trains the window VAE, and `scripts/build_corpora.sh` encodes the corpora into the DiT latent stores.

## Code

`taste/inference/` contains samplers, `annotation/` contains probes and labeling tools, `corpus/` builds training data, and `training/` contains trainers. Model definitions live in `models/`; the window VAE is in `codec/`. See the [code map](code-map.md) for each module and the [probing guide](probing.md) for probe replication and adaptation to another dataset.
