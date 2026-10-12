"""Batch constrained-clustering section timing for the high-quality dataset.

Stage `cache` stores Aria layer-7 frame features for every MIDI; stage `annotate` runs the
constrained clustering from the cache and writes `section_timing` and `section_sequence_timed`
into each metadata JSON (atomically).

    python -m taste.annotation.high_quality.section_timing.process --stage all --device cuda:0
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import traceback
from pathlib import Path

import numpy as np

from taste import paths
from taste.annotation.high_quality.section_timing.constrained_clustering import (
    FPS as CASCADE_FPS, LAM_DUR, LAM_NOV_STAGE1, LAM_NOV_STAGE2, MIN_SEC, _prep)
from taste.annotation.high_quality.section_timing.infer import (
    build_matrix, extract_features, section_letter, segment)

FPS = 2.0
ARIA_LAYER = 7
METHOD_NAME = "constrained_clustering"


def _parse_indexed_cuda(device):
    if device == "cuda":
        return None
    if device.startswith("cuda:"):
        idx = device.split(":", 1)[1]
        if idx.isdigit():
            return idx
    return None


def _confine_single_cuda(device):
    """Return local device after limiting visibility to one physical GPU."""
    idx = _parse_indexed_cuda(device)
    if idx is None:
        return device
    existing = os.environ.get("CUDA_VISIBLE_DEVICES")
    if existing and existing != idx:
        raise ValueError(
            f"CUDA_VISIBLE_DEVICES is already {existing!r}, "
            f"but --device requests {idx!r}"
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = idx
    return "cuda:0"


def _sample_id(path):
    return path.stem


def _metadata_path(metadata_dir, sid):
    return metadata_dir / f"{sid}.json"


def _midi_path(midi_dir, sid):
    return midi_dir / f"{sid}.mid"


def _cache_path(cache_dir, sid):
    return cache_dir / f"{sid}.npz"


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json_atomic(path, obj):
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp.replace(path)


def collect_samples(dataset_dir):
    metadata_dir = dataset_dir / "metadata"
    midi_dir = dataset_dir / "midi"
    samples = []
    for meta_path in sorted(metadata_dir.glob("*.json")):
        sid = _sample_id(meta_path)
        midi_path = _midi_path(midi_dir, sid)
        if midi_path.exists():
            samples.append((sid, meta_path, midi_path))
    return samples


def cache_aria_features(samples, cache_dir, *, device, force=False):
    from taste.annotation.high_quality.section_timing.aria_features import (
        extract_aria_features)

    cache_dir.mkdir(parents=True, exist_ok=True)
    pending = [
        (sid, _midi, _cache_path(cache_dir, sid))
        for sid, _meta, _midi in samples
        if force or not _cache_path(cache_dir, sid).exists()
    ]

    print(
        f"Aria cache: {len(samples)} selected, "
        f"{len(samples) - len(pending)} existing, {len(pending)} pending",
        flush=True,
    )
    for i, (sid, midi_path, out_path) in enumerate(pending, 1):
        print(f"[aria {i}/{len(pending)}] {sid}", flush=True)
        try:
            feat = extract_aria_features(midi_path, fps=FPS, device=device)
            x = feat["X_all"][ARIA_LAYER].astype(np.float16)
            tmp = out_path.with_name(f".{out_path.name}.tmp")
            with open(tmp, "wb") as f:
                np.savez(
                    f,
                    X=x,
                    fps=np.float32(feat["fps"]),
                    n_frames=np.int32(feat["n_frames"]),
                    duration=np.float32(feat["duration"]),
                    aria_layer=np.int32(ARIA_LAYER),
                    sample_id=sid,
                )
            tmp.replace(out_path)
        except Exception:
            print(f"[aria FAILED] {sid}", file=sys.stderr, flush=True)
            traceback.print_exc()


def _infer_one(args):
    sid, meta_path, midi_path, cache_path, force = args
    try:
        meta = _read_json(meta_path)
        if not force and meta.get("section_timing", {}).get("method") == METHOD_NAME:
            return sid, "skip", None

        seq = meta.get("section_sequence")
        if not seq:
            return sid, "fail", "missing section_sequence"
        if not cache_path.exists():
            return sid, "fail", f"missing aria cache: {cache_path}"

        labels = [str(s) for s in seq]
        letters = [section_letter(s) for s in labels]

        x_hc = build_matrix(extract_features(midi_path, fps=CASCADE_FPS))
        bnds_stage1 = segment(
            x_hc,
            letters,
            CASCADE_FPS,
            min_sec=MIN_SEC,
            lam_dur=LAM_DUR,
            lam_nov=LAM_NOV_STAGE1,
        )

        cached = np.load(cache_path)
        x_aria = _prep(cached["X"].astype(np.float32))
        bnds = segment(
            x_aria,
            letters,
            CASCADE_FPS,
            min_sec=MIN_SEC,
            lam_dur=LAM_DUR,
            lam_nov=LAM_NOV_STAGE2,
            init=bnds_stage1,
        )

        hop = 1.0 / CASCADE_FPS
        starts = [round(float(b) * hop, 3) for b in bnds[:-1]]
        starts[0] = 0.0
        duration = round(float(cached["duration"]), 3)
        sections = []
        for i, (name, letter, start) in enumerate(zip(labels, letters, starts)):
            end = starts[i + 1] if i + 1 < len(starts) else duration
            sections.append(
                {
                    "index": i,
                    "name": name,
                    "letter": letter,
                    "start_sec": start,
                    "end_sec": round(float(end), 3),
                }
            )

        meta["section_timing"] = {
            "method": METHOD_NAME,
            "source": "transcribed_midi_and_section_sequence",
            "fps": CASCADE_FPS,
            "aria_model": "loubb/aria-medium-base",
            "aria_layer": ARIA_LAYER,
            "aria_cache": str(cache_path.relative_to(paths.DATA_ROOT)),
            "duration_sec": duration,
            "sections": sections,
        }
        # flat field consumed by the high-quality corpus builder
        meta["section_sequence_timed"] = [
            {"name": s["name"], "start_sec": s["start_sec"]} for s in sections
        ]
        _write_json_atomic(meta_path, meta)
        return sid, "ok", None
    except Exception:
        return sid, "fail", traceback.format_exc()


def annotate_metadata(samples, cache_dir, *, workers, force=False):
    tasks = [
        (sid, meta_path, midi_path, _cache_path(cache_dir, sid), force)
        for sid, meta_path, midi_path in samples
    ]

    print(f"Annotate: {len(tasks)} selected with {workers} workers", flush=True)
    ok = skipped = 0
    failed = []
    with mp.Pool(processes=workers) as pool:
        for sid, status, err in pool.imap_unordered(_infer_one, tasks, chunksize=8):
            if status == "ok":
                ok += 1
            elif status == "skip":
                skipped += 1
            else:
                failed.append((sid, err))
            done = ok + skipped + len(failed)
            if done == len(tasks) or done % 100 == 0:
                print(
                    f"[annotate] done={done}/{len(tasks)} ok={ok} "
                    f"skip={skipped} fail={len(failed)}",
                    flush=True,
                )

    if failed:
        fail_path = cache_dir / "metadata_update_failures.jsonl"
        with open(fail_path, "w", encoding="utf-8") as f:
            for sid, err in failed:
                f.write(json.dumps({"sample_id": sid, "error": err}) + "\n")
        print(f"Wrote failures to {fail_path}", file=sys.stderr)
        return 1
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", type=Path, default=paths.data("high_quality"))
    ap.add_argument("--cache-dir", type=Path,
                    default=paths.data("high_quality", "section_timing_cache"))
    ap.add_argument("--stage", choices=["all", "cache", "annotate"], default="all")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force-cache", action="store_true")
    ap.add_argument("--force-metadata", action="store_true")
    args = ap.parse_args()

    if args.workers < 1 or args.workers > 8:
        print("Error: --workers must be between 1 and 8", file=sys.stderr)
        return 2

    device = args.device
    if args.stage in {"all", "cache"}:
        device = _confine_single_cuda(device)

    samples = collect_samples(args.dataset_dir)
    print(f"Dataset: {args.dataset_dir}", flush=True)
    print(f"Samples with metadata+MIDI: {len(samples)}", flush=True)
    print(f"Cache dir: {args.cache_dir}", flush=True)

    if args.stage in {"all", "cache"}:
        print(f"Aria device: {device}", flush=True)
        cache_aria_features(
            samples,
            args.cache_dir,
            device=device,
            force=args.force_cache,
        )

    if args.stage in {"all", "annotate"}:
        return annotate_metadata(
            samples,
            args.cache_dir,
            workers=args.workers,
            force=args.force_metadata,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
