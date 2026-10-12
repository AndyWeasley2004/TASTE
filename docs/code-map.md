# Code map

[README](../README.md) · [Probing and replication](probing.md) · [Research](research.md)

## Data annotation and probes

The annotation flow is **source evidence / audio → teacher labels → MIDI features → probes → accepted labels → training corpora**. Imagery retrieval and rewriting provide weak supervision; probe predictions are not human-verified labels.

| Component | Implementation | Responsibility |
|---|---|---|
| Dataset layout and paths | [paths.py](../taste/paths.py), [import_release.py](../taste/corpus/import_release.py) | Resolve data/checkpoint roots and unpack the release into pipeline layouts. |
| Curated sources | [source_acquisition/](../taste/annotation/curated/source_acquisition/), [transcription/](../taste/transcription/) | Acquire source material and transcribe audio-only performances to MIDI. |
| Evidence and audio teachers | [evidence_route/](../taste/annotation/curated/evidence_route/), [audio_route/](../taste/annotation/curated/audio_route/) | Produce attribute/imagery annotations from retrieved evidence or audio; prompts live beside the code. |
| Structural teachers | [structure_analysis/](../taste/annotation/curated/structure_analysis/), [section_timing/](../taste/annotation/high_quality/section_timing/) | Curated beat tracking/section fusion; high-quality Aria features and constrained timing of supplied section letters. |
| Tag manifests and splits | [manifests.py](../taste/annotation/large_public/manifests.py) | Benchmark work-group splits and deduplicated-corpus manifests; builders enforce release-specific counts. |
| Tag features | [features.py](../taste/annotation/large_public/features.py), [midi_features.py](../taste/representation/midi_features.py) | Extract/verify CLaMP 3, Aria and 32 symbolic statistics; embed imagery text. |
| Tag learning and scoring | [tag_probes.py](../taste/annotation/large_public/tag_probes.py) | `fit_models`: grouped pruning, classifiers, gates and imagery projection; `concat_features` / `score_attributes`: reusable inference; `apply_models`: full-corpus gates, metadata overrides and retrieval. |
| Structure features | [extract_features.py](../taste/annotation/structure_probe/extract_features.py) | Parse notes/pedal, compute 26-dimensional frames, attach teacher spans, and write sharded arrays/manifests. |
| Structure model and training | [train.py](../taste/annotation/structure_probe/train.py) | `StructureProbe`, boundary and same-section losses, splitting, checkpoint selection and evaluation. |
| Structure decoding | [infer.py](../taste/annotation/structure_probe/infer.py) | Predict boundary peaks, cluster sections into letters A–H, merge spans and export labels/statistics. |
| Structure integration | [write_labels.py](../taste/annotation/structure_probe/write_labels.py) | Write curated lettered sections or snap high-quality boundaries; `--pilot-dir` previews, `--apply` updates dataset files. |
| Imagery and label consolidation | [imagery_captions.py](../taste/annotation/large_public/imagery_captions.py), [consolidation.py](../taste/annotation/consolidation.py) | Rewrite/verify retrieved captions and consolidate curated attribute evidence with probe predictions. |
| Key and tempo | [key.py](../taste/representation/key.py), [tempo.py](../taste/representation/tempo.py) | Separate estimators for global key and tempo labels. |

Use the [probing guide](probing.md) for input schemas, commands, outputs and dataset adaptation. Full annotation launchers are [annotate_curated.sh](../scripts/annotate_curated.sh), [annotate_high_quality.sh](../scripts/annotate_high_quality.sh) and [annotate_large_public.sh](../scripts/annotate_large_public.sh); each lists its prerequisites.

## Generation and training

| Module | Implementation | Responsibility |
|---|---|---|
| MIDI representation | [representation/](../taste/representation/) | Event tokenizer/vocabulary, tempo/key utilities and imagery states. |
| Training data | [corpus/](../taste/corpus/) | Tier-specific corpus builders, formats, verification and DiT latent stores. |
| Window codec | [codec/](../taste/codec/) | Window datasets, VAE training, reconstruction and latent representation. |
| Generators | [models/](../taste/models/) | Unconditional/conditional AR and DiT architectures. |
| Training loops | [training/](../taste/training/), [configs/](../configs/) | AR pretraining, conditional stages and DiT training/configuration. |
| Sampling | [inference/](../taste/inference/) | AR/DiT sampling and latent decoding to MIDI; see [generation](generation.md). |

The [research guide](research.md) gives the release import, corpus-building and staged-training order.
