#!/usr/bin/env python3
# Uses Beat This! by Foscarin, Schlueter and Widmer (ISMIR 2024).
# Upstream: https://github.com/CPJKU/beat_this (MIT).
"""Persistent Beat This worker: writes one beat-time file per audio item of its shard; launched by run_batch.py."""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from beat_this.inference import File2Beats


def _write_json(path, value):
    tmp = path.with_name(".{}.{}.tmp".format(path.name, os.getpid()))
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _write_times(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(".{}.{}.tmp".format(path.name, os.getpid()))
    tmp.write_text("".join("{:.6f}\n".format(value) for value in values), encoding="utf-8")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--ready", required=True, type=Path)
    parser.add_argument("--start", required=True, type=Path)
    parser.add_argument("--model", default="final0")
    parser.add_argument("--float16", action="store_true")
    args = parser.parse_args()

    gpu_requested = bool(os.environ.get("CUDA_VISIBLE_DEVICES"))
    if gpu_requested and not torch.cuda.is_available():
        raise RuntimeError("A GPU was assigned, but PyTorch cannot access CUDA")
    device = "cuda:0" if gpu_requested else "cpu"

    load_start = time.perf_counter()
    tracker = File2Beats(
        checkpoint_path=args.model,
        device=device,
        float16=args.float16 and gpu_requested,
        dbn=False,
    )
    model_load_sec = time.perf_counter() - load_start
    _write_json(args.ready, {"device": device, "model_load_sec": model_load_sec})
    while not args.start.exists():
        time.sleep(0.05)

    items = json.loads(args.manifest.read_text(encoding="utf-8"))
    timings = []
    failures = []
    for item in items:
        started = time.perf_counter()
        try:
            beats, downbeats = tracker(item["audio"])
            beats = np.asarray(beats, dtype=np.float64).reshape(-1)
            beats = np.unique(beats[np.isfinite(beats) & (beats >= 0)])
            if beats.size < 2:
                raise ValueError("Beat This returned fewer than two beats")
            _write_times(Path(item["beats"]), beats)
            timings.append({
                "id": item["id"],
                "beat_compute_sec": time.perf_counter() - started,
                "beat_count": int(beats.size),
                "downbeat_count": int(np.asarray(downbeats).size),
            })
        except Exception:
            failures.append({"id": item["id"], "traceback": traceback.format_exc()})
            print("[Beat This failed] {}".format(item["audio"]), file=sys.stderr, flush=True)

    result = {
        "worker": "beat_this",
        "device": device,
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
