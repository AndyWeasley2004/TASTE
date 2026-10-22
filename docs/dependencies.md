# Dependencies

[README](../README.md) · [Probing](probing.md) · [Research](research.md)

Run from the release repository root. Choose one profile in a Python 3.11 environment:

| Scope | Requirements | Editable install |
|---|---|---|
| Generation only: AR and DiT, including codec decoding | [requirements-generation.txt](../requirements-generation.txt) | `pip install -e .` |
| Data pipeline: acquisition, transcription, annotations, all probe training/inference, corpus and latent construction | [requirements-pipeline.txt](../requirements-pipeline.txt) | `pip install -e ".[pipeline]"` |
| All TASTE packages, including generator and codec training | [requirements.txt](../requirements.txt) | `pip install -e ".[all]"` |

For a requirements-file install, run `pip install -r <file>` followed by `pip install -e . --no-deps`. The `probes` and `research` extras remain aliases for `pipeline` and `all`. Aria utilities are installed from a pinned upstream Git revision because `ariautils` is not published on PyPI. Git must be installed.

## External workers for the data pipeline

Both the pipeline and all-packages profiles need the following workers for full annotation reproduction. These run as subprocesses: their upstream versions conflict with the main environment, so do not combine their requirement files into one pip install. MIDI-only structure-probe training/inference needs none of these workers.

### CLaMP 3 tag and imagery features

The symbolic/text extractor uses the [official CLaMP 3 implementation](https://github.com/sanderwood/clamp3). Its Transformers 4 API is isolated from TASTE's Transformers 5 installation.

```bash
python3.11 -m venv external/clamp3-env
external/clamp3-env/bin/python -m pip install -r requirements/workers/clamp3.txt
git clone https://github.com/sanderwood/clamp3.git external/clamp3
git -C external/clamp3 checkout 9016d2b0c8d12d1aa79c2e0ab201e6822bdc83a8
export TASTE_CLAMP3_ROOT="$PWD/external/clamp3"
export TASTE_CLAMP3_PYTHON="$PWD/external/clamp3-env/bin/python"
```

`features clamp3` and `features clamp3-text` use this interpreter by default; `--python` can override it. The worker file covers the symbolic/text routes used here, not CLaMP 3's optional image/audio routes. Download the upstream checkpoints as described in its README.

### High-quality Aria frame features

The [Aria model source](https://github.com/EleutherAI/aria) is loaded from a checkout; its PyTorch, safetensors and aria-utils dependencies are in the pipeline profile. The tag embedding route loads its pinned model directly through Transformers.

```bash
git clone https://github.com/EleutherAI/aria.git external/aria
export TASTE_ARIA_ROOT="$PWD/external/aria"
```

### Section fusion / DEEPSIM teacher

[Salamon, Nieto and Bryan's section-fusion implementation](https://github.com/justinsalamon/musicseg_deepemb) uses Python 3.7, TensorFlow/Keras 2.6 and legacy PyTorch. This is an upstream compatibility requirement, not a named local conda environment. Install a Python 3.7 interpreter separately, then:

```bash
python3.7 -m venv external/section-fusion-env
external/section-fusion-env/bin/python -m pip install "pip<24.1" "setuptools<68" wheel "Cython<3" numpy==1.19.2
external/section-fusion-env/bin/python -m pip install --no-build-isolation -r requirements/workers/section-fusion.txt
external/section-fusion-env/bin/python -m pip install --no-deps \
    "musicsections @ git+https://github.com/justinsalamon/musicseg_deepemb.git@13aed85d30459603b4ed5024c3a84c28b70bc6eb"
export TASTE_SECTION_FUSION_PYTHON="$PWD/external/section-fusion-env/bin/python"
```

The dependency file follows upstream's package metadata. Upstream also documents a post-install NumPy 1.21.5 workaround; that overrides its declared NumPy pin and is not part of the default dependency set here. Legacy wheels and GPU support depend on the host platform/CUDA version. Place the upstream DEEPSIM and few-shot weights in `$TASTE_CKPT_ROOT/section_fusion/models/{deepsim,fewshot}` or pass `run_batch --model-root`. The launcher uses the configured interpreter directly; no conda environment name is assumed.

### Annotation language-model serving

vLLM 0.19 requires Transformers <5, so run it separately from TASTE:

```bash
python3.11 -m venv external/vllm-env
external/vllm-env/bin/python -m pip install -r requirements/workers/vllm.txt
export TASTE_VLLM_BIN="$PWD/external/vllm-env/bin/vllm"
```

The annotation launchers and audio normalization use this executable (`normalize --vllm-bin` can override it). Model weights and service credentials are still required; see [research](research.md).

## Non-Python dependencies

Install **FFmpeg**, **SoX with MP3 support**, and **libsndfile** for transcription/audio segmentation, plus **curl** for the annotation launchers. Source acquisition uses the `yt-dlp` package in the pipeline profile; YouTube extraction may additionally require a supported JavaScript runtime, such as Deno. GPU stages need compatible CUDA drivers. Dependency installation does not download datasets or all third-party checkpoints.
