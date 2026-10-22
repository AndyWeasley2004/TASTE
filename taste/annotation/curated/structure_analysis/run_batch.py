#!/usr/bin/env python3
"""Structure analysis: resumable Beat This beat tracking followed by section fusion over an audio collection.

Uses Beat This! (Foscarin et al., ISMIR 2024; https://github.com/CPJKU/beat_this)
and section fusion (Salamon et al., ISMIR 2021;
https://github.com/justinsalamon/musicseg_deepemb).
Set TASTE_SECTION_FUSION_PYTHON to the interpreter with the legacy worker dependencies.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

from taste import paths


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_ROOT = paths.ckpt("section_fusion", "models")
DEFAULT_TORCH_HOME = paths.ckpt("beat_this")
SECTION_FUSION_PYTHON = os.environ.get("TASTE_SECTION_FUSION_PYTHON", sys.executable)
AUDIO_EXTENSIONS = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".wav"}
PIECE_AUDIO_NAME = "rendered_audio.mp3"
PIECE_BEATS_NAME = "audio_beats.txt"
PIECE_JSON_NAME = "meta_archive/structure_section_fusion.json"
PIECE_LAB_NAME = "meta_archive/structure_section_fusion.lab"


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(".{}.{}.tmp".format(path.name, os.getpid()))
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _collect_audio(inputs, piece_layout=False):
    found = []
    for raw in inputs:
        path = raw.expanduser().resolve()
        if path.is_file():
            if path.suffix.lower() in AUDIO_EXTENSIONS:
                found.append(path)
        elif path.is_dir():
            found.extend(
                child.resolve()
                for child in path.rglob("*")
                if child.is_file() and child.suffix.lower() in AUDIO_EXTENSIONS
            )
        else:
            raise FileNotFoundError(path)

    paths = sorted(set(found), key=lambda value: str(value))
    if piece_layout:
        paths = [path for path in paths if path.name == PIECE_AUDIO_NAME]
    if not paths:
        raise ValueError("No supported audio files found")

    id_counts = {}
    for path in paths:
        # piece layout: audio sits at <piece_id>/meta_archive/rendered_audio.mp3
        sample_id = path.parent.parent.name if piece_layout else path.stem
        id_counts[sample_id] = id_counts.get(sample_id, 0) + 1

    items = []
    for path in paths:
        # piece layout: audio sits at <piece_id>/meta_archive/rendered_audio.mp3
        sample_id = path.parent.parent.name if piece_layout else path.stem
        if piece_layout:
            if id_counts[sample_id] != 1:
                raise ValueError(
                    "Piece layout requires exactly one rendered_audio.mp3 per piece ID; "
                    "found duplicate ID {}".format(sample_id)
                )
        elif id_counts[sample_id] == 1:
            sample_id = path.stem
        else:
            digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:10]
            sample_id = "{}__{}".format(path.stem, digest)
        items.append({"id": sample_id, "audio": str(path), "bytes": path.stat().st_size})
    return items


def _add_output_paths(items, output_dir, piece_layout=False):
    for item in items:
        sample_id = item["id"]
        if piece_layout:
            piece_dir = Path(item["audio"]).parent.parent
            if piece_dir.name != sample_id:
                raise ValueError(
                    "Audio ID {} does not match piece folder {}".format(sample_id, piece_dir)
                )
            item["beats"] = str((piece_dir / PIECE_BEATS_NAME).resolve())
            item["json"] = str((piece_dir / PIECE_JSON_NAME).resolve())
            item["lab"] = str((piece_dir / PIECE_LAB_NAME).resolve())
        else:
            item["beats"] = str((output_dir / "beats" / (sample_id + ".beats.txt")).resolve())
            item["json"] = str((output_dir / "labels" / (sample_id + ".json")).resolve())
            item["lab"] = str((output_dir / "labels" / (sample_id + ".lab")).resolve())


def _parse_devices(text):
    if text.strip().lower() == "cpu":
        return ["cpu"]
    devices = [value.strip() for value in text.split(",") if value.strip()]
    if not devices or any(not value.isdigit() for value in devices):
        raise ValueError("--gpus must be 'cpu' or comma-separated GPU indices")
    if len(devices) > 4:
        raise ValueError("This project permits at most four GPUs")
    return devices


def _balanced_shards(items, count):
    shards = [[] for _ in range(count)]
    loads = [0 for _ in range(count)]
    for item in sorted(items, key=lambda value: value["bytes"], reverse=True):
        index = min(range(count), key=lambda value: loads[value])
        shards[index].append(item)
        loads[index] += item["bytes"]
    return [shard for shard in shards if shard]


def _log_tail(path, lines=80):
    if not path.exists():
        return "(worker log was not created)"
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])


def _aggregate_results(worker_results):
    aggregate = {
        "processed": 0,
        "failed": 0,
        "model_load_sec_max": 0.0,
        "beat_compute_sec_sum": 0.0,
        "section_fusion_features_sec_sum": 0.0,
        "analysis_sec_sum": 0.0,
    }
    for result in worker_results:
        aggregate["processed"] += int(result.get("processed", 0))
        aggregate["failed"] += int(result.get("failed", 0))
        aggregate["model_load_sec_max"] = max(
            aggregate["model_load_sec_max"], float(result.get("model_load_sec", 0.0))
        )
        for item in result.get("timings", []):
            aggregate["beat_compute_sec_sum"] += float(item.get("beat_compute_sec", 0.0))
            aggregate["section_fusion_features_sec_sum"] += float(item.get("section_fusion_features_sec", 0.0))
            aggregate["analysis_sec_sum"] += float(item.get("analysis_sec", 0.0))
    processed = aggregate["processed"]
    for key in ("beat_compute_sec", "section_fusion_features_sec", "analysis_sec"):
        aggregate[key + "_mean"] = (
            aggregate[key + "_sum"] / processed if processed else 0.0
        )
    return aggregate


def _run_workers(
    stage,
    pending,
    devices,
    workers_per_device,
    output_dir,
    args,
):
    if not pending:
        return {
            "stage": stage,
            "workers": 0,
            "processed": 0,
            "failed": 0,
            "wall_sec_excluding_model_load": 0.0,
        }

    assignments = []
    for device in devices:
        for _ in range(workers_per_device):
            assignments.append(device)
    shards = _balanced_shards(pending, min(len(assignments), len(pending)))
    assignments = assignments[: len(shards)]

    run_dir = output_dir / (".run_{}_{}".format(stage, uuid.uuid4().hex[:10]))
    ready_dir = run_dir / "ready"
    ready_dir.mkdir(parents=True)
    start_file = run_dir / "START"

    processes = []
    log_handles = []
    try:
        for index, (shard, device) in enumerate(zip(shards, assignments)):
            shard_path = run_dir / "shard_{:03d}.json".format(index)
            result_path = run_dir / "result_{:03d}.json".format(index)
            ready_path = ready_dir / "worker_{:03d}.json".format(index)
            log_path = run_dir / "worker_{:03d}.log".format(index)
            _write_json(shard_path, shard)

            if stage == "beats":
                worker = MODULE_DIR / "beat_worker.py"
                command = [
                    sys.executable, str(worker),
                    "--manifest", str(shard_path),
                    "--result", str(result_path),
                    "--ready", str(ready_path),
                    "--start", str(start_file),
                    "--model", args.beat_model,
                ]
                if args.beat_fp16 and device != "cpu":
                    command.append("--float16")
            else:
                worker = MODULE_DIR / "section_fusion_worker.py"
                command = [
                    SECTION_FUSION_PYTHON, str(worker),
                    "--manifest", str(shard_path),
                    "--result", str(result_path),
                    "--ready", str(ready_path),
                    "--start", str(start_file),
                    "--model-root", str(args.model_root),
                    "--min-duration", str(args.min_duration),
                    "--output-level", str(args.output_level),
                    "--hierarchy-levels", str(args.hierarchy_levels),
                    "--deepsim-batch-size", str(args.deepsim_batch_size),
                    "--seed", str(args.seed),
                ]

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = "" if device == "cpu" else device
            env["PYTHONUNBUFFERED"] = "1"
            env["TORCH_HOME"] = str(args.torch_home)
            env["TF_CPP_MIN_LOG_LEVEL"] = "2"
            thread_count = max(1, (os.cpu_count() or 1) // max(1, len(shards)))
            env["OMP_NUM_THREADS"] = str(thread_count)
            env["MKL_NUM_THREADS"] = str(thread_count)
            env["NUMBA_CACHE_DIR"] = str(run_dir / "numba_cache_{:03d}".format(index))
            env["MPLCONFIGDIR"] = str(run_dir / "mpl_cache_{:03d}".format(index))

            log_handle = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                command,
                cwd=str(MODULE_DIR.parent),
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            processes.append((process, result_path, ready_path, log_path))
            log_handles.append(log_handle)

        deadline = time.monotonic() + args.startup_timeout
        while True:
            if all(ready.exists() for _process, _result, ready, _log in processes):
                break
            for process, _result, _ready, log_path in processes:
                if process.poll() is not None:
                    raise RuntimeError(
                        "{} worker exited during model loading:\n{}".format(stage, _log_tail(log_path))
                    )
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for {} workers to load models".format(stage))
            time.sleep(0.1)

        start = time.perf_counter()
        start_file.touch()
        failures = []
        for process, _result, _ready, log_path in processes:
            code = process.wait()
            if code:
                failures.append((code, log_path))
        wall_sec = time.perf_counter() - start
        for handle in log_handles:
            handle.close()
        log_handles = []

        if failures:
            details = "\n\n".join(
                "exit {}: {}\n{}".format(code, path, _log_tail(path))
                for code, path in failures
            )
            raise RuntimeError("{} worker failure(s):\n{}".format(stage, details))

        worker_results = [json.loads(result.read_text(encoding="utf-8")) for _, result, _, _ in processes]
        aggregate = _aggregate_results(worker_results)
        aggregate.update({
            "stage": stage,
            "workers": len(processes),
            "devices": assignments,
            "wall_sec_excluding_model_load": wall_sec,
        })
        shutil.rmtree(str(run_dir))
        return aggregate
    except Exception:
        for process, _result, _ready, _log in processes:
            if process.poll() is None:
                process.terminate()
        for process, _result, _ready, _log in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        for handle in log_handles:
            handle.close()
        print("Preserved failed worker state at {}".format(run_dir), file=sys.stderr)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Audio files or directories")
    parser.add_argument("--output", required=True, type=Path, help="Output/cache directory")
    parser.add_argument(
        "--piece-layout",
        action="store_true",
        help=(
            "Process only <piece_id>/meta_archive/rendered_audio.mp3 files and write "
            "audio_beats.txt and meta_archive/structure_section_fusion.{json,lab} in each piece directory; "
            "--output stores only run state and the manifest"
        ),
    )
    parser.add_argument("--stage", choices=("all", "beats", "section_fusion"), default="all")
    parser.add_argument("--gpus", default="0,1,2,3", help="GPU indices, or 'cpu'")
    parser.add_argument("--beat-workers-per-gpu", type=int, default=2,
                        help="Beat This workers per GPU; total workers are capped at 8")
    parser.add_argument("--section-fusion-workers-per-gpu", type=int, default=2,
                        help="Section-fusion workers per GPU; total workers are capped at 8")
    parser.add_argument("--deepsim-batch-size", type=int, default=500,
                        help="Keras DeepSim inference batch size; use 0 for its default")
    parser.add_argument("--force-beats", action="store_true")
    parser.add_argument("--force-section-fusion", action="store_true")
    parser.add_argument("--beat-model", default="final0")
    beat_precision = parser.add_mutually_exclusive_group()
    beat_precision.add_argument(
        "--beat-fp16", dest="beat_fp16", action="store_true",
        help="Run Beat This in FP16 (default)",
    )
    beat_precision.add_argument(
        "--beat-fp32", dest="beat_fp16", action="store_false",
        help="Run Beat This in FP32 for numerical conservatism",
    )
    parser.set_defaults(beat_fp16=True)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--torch-home", type=Path, default=DEFAULT_TORCH_HOME)
    parser.add_argument("--min-duration", type=float, default=12.0)
    parser.add_argument("--output-level", type=int, default=4)
    parser.add_argument("--hierarchy-levels", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--startup-timeout", type=float, default=1800.0)
    args = parser.parse_args()

    devices = _parse_devices(args.gpus)
    if devices == ["cpu"]:
        args.beat_workers_per_gpu = 1
        args.section_fusion_workers_per_gpu = 1
    if args.section_fusion_workers_per_gpu < 1:
        parser.error("--section-fusion-workers-per-gpu must be at least 1")
    if args.beat_workers_per_gpu < 1:
        parser.error("--beat-workers-per-gpu must be at least 1")
    if devices != ["cpu"] and args.beat_workers_per_gpu * len(devices) > 8:
        parser.error("Beat This multiprocessing is capped at 8 total workers")
    if devices != ["cpu"] and args.section_fusion_workers_per_gpu * len(devices) > 8:
        parser.error("Section-fusion multiprocessing is capped at 8 total workers")
    if args.deepsim_batch_size < 0:
        parser.error("--deepsim-batch-size cannot be negative")
    if args.hierarchy_levels < args.output_level:
        parser.error("--hierarchy-levels must be at least --output-level")
    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=True)

    items = _collect_audio(args.inputs, args.piece_layout)
    _add_output_paths(items, args.output, args.piece_layout)
    timings = {}

    if args.stage in ("all", "beats"):
        pending = [
            item for item in items
            if args.force_beats or not (Path(item["beats"]).exists() and Path(item["beats"]).stat().st_size)
        ]
        beat_worker_copies = 1 if devices == ["cpu"] else args.beat_workers_per_gpu
        timings["beats"] = _run_workers(
            "beats", pending, devices, beat_worker_copies, args.output, args
        )
        timings["beats"]["skipped_existing"] = len(items) - len(pending)

    if args.stage in ("all", "section_fusion"):
        missing = [item["audio"] for item in items if not Path(item["beats"]).exists()]
        if missing:
            raise RuntimeError(
                "Section-fusion stage requires cached Beat This results; {} are missing (first: {})".format(
                    len(missing), missing[0]
                )
            )
        pending = [
            item for item in items
            if args.force_section_fusion or not (Path(item["json"]).exists() and Path(item["lab"]).exists())
        ]
        worker_copies = 1 if devices == ["cpu"] else args.section_fusion_workers_per_gpu
        timings["section_fusion"] = _run_workers(
            "section_fusion", pending, devices, worker_copies, args.output, args
        )
        timings["section_fusion"]["skipped_existing"] = len(items) - len(pending)

    total_wall = sum(value["wall_sec_excluding_model_load"] for value in timings.values())
    manifest = {
        "method": "Beat This final0 -> section-fusion features -> section-fusion spectral analysis",
        "piece_count": len(items),
        "parameters": {
            "beat_environment": sys.executable,
            "section_fusion_environment": SECTION_FUSION_ENV,
            "devices": devices,
            "beat_workers_per_gpu": args.beat_workers_per_gpu,
            "section_fusion_workers_per_gpu": args.section_fusion_workers_per_gpu,
            "deepsim_batch_size": args.deepsim_batch_size or "keras_default",
            "beat_model": args.beat_model,
            "beat_float16": args.beat_fp16,
            "min_duration_sec": args.min_duration,
            "output_level": args.output_level,
            "hierarchy_levels": args.hierarchy_levels,
            "piece_layout": args.piece_layout,
            "seed": args.seed,
        },
        "timings": timings,
        "total_wall_sec_excluding_model_load": total_wall,
        "pieces": items,
    }
    _write_json(args.output / "manifest.json", manifest)
    print(json.dumps({
        "piece_count": len(items),
        "timings": timings,
        "total_wall_sec_excluding_model_load": total_wall,
        "manifest": str((args.output / "manifest.json").resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
