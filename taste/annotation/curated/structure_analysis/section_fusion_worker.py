#!/usr/bin/env python3
"""Persistent section-fusion worker: beat-synchronous deep features, spectral clustering and section fusion per audio file.

It runs in a separate Python 3.7 conda env (`structure`) and is launched by run_batch.py.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import musicsections
from musicsections import core
import tensorflow as tf


def _alpha_label(index):
    value = index + 1
    result = ""
    while value:
        value, remainder = divmod(value - 1, 26)
        result = chr(65 + remainder) + result
    return result


def _serialize_level(level_index, level):
    intervals, native_labels = level
    canonical = {}
    sections = []
    for interval, native_label in zip(intervals, native_labels):
        key = str(native_label)
        if key not in canonical:
            canonical[key] = _alpha_label(len(canonical))
        sections.append({
            "start_sec": round(float(interval[0]), 3),
            "end_sec": round(float(interval[1]), 3),
            "label": canonical[key],
            "native_cluster_id": key,
        })
    return {
        "level": level_index,
        "section_count": len(sections),
        "unique_label_count": len(canonical),
        "structure": " ".join(section["label"] for section in sections),
        "sections": sections,
    }


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(".{}.{}.tmp".format(path.name, os.getpid()))
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _write_lab(path, sections):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(".{}.{}.tmp".format(path.name, os.getpid()))
    tmp.write_text(
        "".join(
            "{start_sec:.3f}\t{end_sec:.3f}\t{label}\n".format(**section)
            for section in sections
        ),
        encoding="utf-8",
    )
    tmp.replace(path)


def _segment_features_to_level(features, hierarchy_levels, min_duration):
    """Run the released section-fusion analysis through the requested hierarchy depth."""
    Csync = features["Csync"]
    Msync = features["Msync"]
    Hsync = features["Hsync"]
    beat_times = features["beat_times"]
    audio_duration = features["audio_duration"]
    if Csync is None or Msync is None or Hsync is None:
        return core.make_single_section(audio_duration, hierarchy_levels)

    embedding = core.embed_beats(
        Csync,
        Msync,
        Hsync,
        mu=0.5,
        gamma=0.5,
        recsmooth=9,
        recwidth=9,
        evecsmooth=9,
        normalize_matrices=core.NORMALIZE_MATRICES,
        distance=core.DISTANCE,
        maxnorm=core.MAXNORM,
    )
    Cnorm = np.cumsum(embedding ** 2, axis=1) ** 0.5
    segmentations = [
        core.cluster(embedding, Cnorm, k, beat_times)
        for k in range(1, min(hierarchy_levels + 1, embedding.shape[0] + 1))
    ]
    levels = core.reindex(segmentations)
    if min_duration is None:
        return levels
    cleaned = [
        core.clean_segments(levels, min_duration=min_duration, fix_level=level)
        for level in range(1, len(levels) + 1)
    ]
    return core.segments_to_levels(cleaned)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--ready", required=True, type=Path)
    parser.add_argument("--start", required=True, type=Path)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--min-duration", type=float, default=12.0)
    parser.add_argument("--output-level", type=int, default=4)
    parser.add_argument("--hierarchy-levels", type=int, default=4)
    parser.add_argument("--deepsim-batch-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    gpu_requested = bool(os.environ.get("CUDA_VISIBLE_DEVICES", "").strip())
    tensorflow_gpus = tf.config.list_physical_devices("GPU")
    if gpu_requested and not tensorflow_gpus:
        raise RuntimeError(
            "A GPU section-fusion worker was requested, but TensorFlow registered no GPU. "
            "Check the structure environment's CUDA libraries and LD_LIBRARY_PATH."
        )

    load_start = time.perf_counter()
    deepsim = musicsections.load_deepsim_model(str(args.model_root / "deepsim"))
    fewshot = musicsections.load_fewshot_model(str(args.model_root / "fewshot"))
    model_load_sec = time.perf_counter() - load_start
    if args.deepsim_batch_size:
        original_predict = deepsim.base_model.predict

        def predict_with_batch_size(inputs, *predict_args, **predict_kwargs):
            predict_kwargs.setdefault("batch_size", args.deepsim_batch_size)
            return original_predict(inputs, *predict_args, **predict_kwargs)

        deepsim.base_model.predict = predict_with_batch_size
    if gpu_requested:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("A GPU section-fusion worker was requested, but PyTorch registered no GPU.")
    _write_json(args.ready, {
        "model_load_sec": model_load_sec,
        "gpu_requested": gpu_requested,
        "tensorflow_gpus": [gpu.name for gpu in tensorflow_gpus],
    })
    while not args.start.exists():
        time.sleep(0.05)

    items = json.loads(args.manifest.read_text(encoding="utf-8"))
    timings = []
    failures = []
    for item in items:
        try:
            feature_start = time.perf_counter()
            with tempfile.TemporaryDirectory(prefix="section_fusion_audio_") as temp_dir:
                analysis_audio = Path(temp_dir) / (item["id"] + ".wav")
                subprocess.run([
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", item["audio"], "-map_metadata", "-1", "-vn",
                    "-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le",
                    str(analysis_audio),
                ], check=True)
                Csync, Msync, Hsync, beat_times, audio_duration = core.make_beat_sync_features(
                    str(analysis_audio),
                    deepsim,
                    fewshot,
                    beats_file=item["beats"],
                    magicnorm=core.MAGICNORM,
                )
            feature_sec = time.perf_counter() - feature_start
            features = {
                "Csync": Csync,
                "Msync": Msync,
                "Hsync": Hsync,
                "beat_times": beat_times,
                "audio_duration": audio_duration,
            }

            np.random.seed(args.seed)
            analysis_start = time.perf_counter()
            segmentations = _segment_features_to_level(
                features,
                hierarchy_levels=args.hierarchy_levels,
                min_duration=args.min_duration,
            )
            native_hierarchy_levels = len(segmentations)
            if segmentations and native_hierarchy_levels < args.hierarchy_levels:
                segmentations.extend(
                    [segmentations[-1]] * (args.hierarchy_levels - native_hierarchy_levels)
                )
            analysis_sec = time.perf_counter() - analysis_start

            levels = [
                _serialize_level(index + 1, level)
                for index, level in enumerate(segmentations)
            ]
            if not 1 <= args.output_level <= len(levels):
                raise ValueError(
                    "Requested level {} but section fusion produced {} levels".format(
                        args.output_level, len(levels)
                    )
                )
            selected = levels[args.output_level - 1]
            beat_grid = [round(float(value), 6) for value in beat_times]
            output = {
                "audio": str(Path(item["audio"]).resolve()),
                "method": "Section fusion (deep embeddings and section fusion) with Beat This beats",
                "models": {
                    "beat_tracker": "Beat This final0",
                    "deepsim": str((args.model_root / "deepsim" / "best.h5").resolve()),
                    "fewshot": str((args.model_root / "fewshot" / "best_model.tar").resolve()),
                },
                "parameters": {
                    "min_duration_sec": args.min_duration,
                    "mu": 0.5,
                    "gamma": 0.5,
                    "beat_tracker": "Beat This, minimal postprocessor, no DBN",
                    "beat_cache": str(Path(item["beats"]).resolve()),
                    "lab_level_policy": "fixed user-selected section-fusion hierarchy level",
                    "output_level": args.output_level,
                    "hierarchy_levels": args.hierarchy_levels,
                    "deepsim_batch_size": args.deepsim_batch_size or "keras_default",
                    "seed": args.seed,
                },
                "audio_duration_sec": round(float(audio_duration), 3),
                "beat_grid_times_sec": beat_grid,
                "beat_grid_note": "Section fusion adds time 0 and audio duration around Beat This detections when needed.",
                "recommended_level": selected["level"],
                "recommended_structure": selected["structure"],
                "timing_sec_excluding_model_load": {
                    "section_fusion_features": feature_sec,
                    "analysis": analysis_sec,
                    "total": feature_sec + analysis_sec,
                },
                "levels": levels,
            }
            if native_hierarchy_levels < args.hierarchy_levels:
                output["parameters"]["hierarchy_fallback"] = (
                    "Repeated deepest valid section-fusion segmentation because the beat grid "
                    "supported only {} native levels.".format(native_hierarchy_levels)
                )
            _write_lab(Path(item["lab"]), selected["sections"])
            _write_json(Path(item["json"]), output)
            timings.append({
                "id": item["id"],
                "section_fusion_features_sec": feature_sec,
                "analysis_sec": analysis_sec,
                "section_count": selected["section_count"],
            })
        except Exception:
            failure_traceback = traceback.format_exc()
            failures.append({"id": item["id"], "traceback": failure_traceback})
            print(
                "[section fusion failed] {}\n{}".format(item["audio"], failure_traceback),
                file=sys.stderr,
                flush=True,
            )

    result = {
        "worker": "section_fusion",
        "model_load_sec": model_load_sec,
        "processed": len(timings),
        "failed": len(failures),
        "timings": timings,
        "failures": failures,
    }
    _write_json(args.result, result)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
