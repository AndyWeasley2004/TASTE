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

Outputs are `generated/<name>_k<i>.mid` and matching JSON files under `--output-dir`. `ended_naturally: false` indicates the token limit was reached; the emitted sections may differ from the generated plan.

Reproducibility requires the same prompt order, variant count and batch layout. `--rows-per-batch` limits MIDI sampling batches; split large prompt files to reduce text-encoding memory.

## Other AR checkpoints

Replace `ar/high_quality` with `ar/curated` or `ar/large_public` in both the download and sampling commands. The pretraining checkpoint uses a separate unconditional sampler:

```bash
hf download Itsuki-music/TASTE --include "ar/pretrain/*" --local-dir checkpoints
python -m taste.inference.sample_ar_pretrain \
    --model-dir checkpoints/ar/pretrain --num-samples 1 --output-dir out/pretrain
```

## DiT

DiT takes section plans and lengths from the high-quality latent-store index. Build it through the [research workflow](research.md); prompt names must match its piece IDs.

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

The default index location is `$TASTE_DATA_ROOT/dit_latents/high_quality/{val,pass0}/index.parquet`; use `--latent-index-root` to select another root.

`TASTE_CKPT_ROOT` defaults to `checkpoints`; `HF_HUB_CACHE` overrides the default text-encoder cache at `checkpoints/hf_cache`. Keep the checkout for the editable install's tokenizer and attribute files.
