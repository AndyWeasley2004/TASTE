# Generation

[Back to README](../README.md) · [Probing](probing.md) · [Research](research.md)

Run commands from the repository root after the README installation.

## AR prompts and sampling

Use `prompts/example.json` for a single example or `prompts/high_quality_val.json` for the 100 held-out high-quality prompts. Each prompt needs the four attribute fields, `key`, `tempo` and `imagery`; `name` is optional. Use simple, unique names for output filenames.

- **Tags:** accepted attribute values are in [attributes.json](../attributes.json); `null` leaves an attribute unspecified.
- **Key:** one of the 24 major/minor keys, such as `C major` or `A minor` (flat spellings include `Db`, `Eb`, `Ab`, `Bb`).
- **Tempo:** `slow`, `moderate` or `fast`.
- **Imagery:** a free-text description. AR generates the section plan itself.

```bash
python -m taste.inference.sample_ar --help
```

| Option | Purpose |
|---|---|
| `--n-seeds 4` | Generate four variants per prompt |
| `--rows-per-batch 1` | Sample one output at a time; increase for throughput |
| `--temperature 0.98` | Sampling temperature (default) |
| `--top-p 0.98` | Nucleus sampling cutoff (default) |
| `--max-new-tokens 4094` | Maximum generation length in tokens, not seconds |

Outputs are `generated/<name>_k<i>.mid` and matching JSON files under `--output-dir`. JSON includes `plan`, `emitted_sections`, `ended_naturally` and token IDs. The generated plan and emitted sections can differ. If `ended_naturally` is false, sampling reached the token limit.

Reproduce an output with the same prompt order, number of variants and batch layout: the recorded seed alone does not reproduce an individual row. `--rows-per-batch` limits MIDI sampling batches; text descriptions are encoded together. For large prompt lists, split the files to reduce text-encoding memory.

## Other AR checkpoints

Replace `ar/high_quality` with `ar/curated` or `ar/large_public` in both the download and sampling commands. The pretraining checkpoint uses a separate unconditional sampler:

```bash
hf download Itsuki-music/TASTE --include "ar/pretrain/*" --local-dir checkpoints
python -m taste.inference.sample_ar_pretrain \
    --model-dir checkpoints/ar/pretrain --num-samples 1 --output-dir out/pretrain
```

## DiT

The released DiT sampler requires the high-quality latent-store index to obtain each piece's section plan and length. It does not currently accept arbitrary text prompts alone. Build the index through the [research workflow](research.md); prompt names must match its piece IDs.

```bash
pip install -e ".[research]"
pip install vector-quantize-pytorch
export TASTE_DATA_ROOT=/path/to/data
export TASTE_CKPT_ROOT=/path/to/checkpoints
hf download Itsuki-music/TASTE --include "dit/high_quality/*" "codec/window_vae/*" --local-dir "$TASTE_CKPT_ROOT"
python -m taste.inference.sample_dit \
    --model-dir "$TASTE_CKPT_ROOT/dit/high_quality" \
    --prompts-json prompts/high_quality_val.json \
    --n-seeds 1 --batch 1 --output-dir out/dit
```

The default index location is `$TASTE_DATA_ROOT/dit_latents/high_quality/{val,pass0}/index.parquet`; use `--latent-index-root` to select another root. Codec imports require `vector-quantize-pytorch`, including when loading the window VAE.

## Local setup

`TASTE_CKPT_ROOT` defaults to `checkpoints`; the imagery encoder cache is stored in its `hf_cache` directory unless `HF_HUB_CACHE` is already set. Keep the repository checkout when using the editable install: the tokenizer and attribute definitions are loaded from it.

For CUDA errors, check that your PyTorch installation detects your GPU. No minimum VRAM figure is established here; both the MIDI model and the text encoder need GPU memory. Start with the single-prompt example.
