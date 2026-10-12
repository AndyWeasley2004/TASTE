#!/usr/bin/env python3
# Inference code from Transkun (Yujia Yan, https://github.com/Yujia-Yan/Transkun), MIT license; the 2.0 weights load from the transcription checkpoint directory.
"""Transcribe piano audio to MIDI with Transkun V2, one file per worker, resuming past existing outputs.

Each audio file under INPUT is written to a sibling `midi/` folder (or --output-dir), preserving relative paths with
--recursive. Workers split the file list into disjoint duration-balanced shards; a file is always transcribed start to
finish by one worker because Transkun carries decoder state across its overlapping segments.

    python -m taste.transcription.audio_to_midi INPUT --devices cuda:0,cuda:1 --workers 8
"""

import argparse
from collections import Counter
import multiprocessing as mp
import os
import queue
import sys
import wave
from pathlib import Path

from taste import paths


DEFAULT_WEIGHT = paths.ckpt("transcription") / "2.0.pt"
DEFAULT_CONF = paths.ckpt("transcription") / "2.0.conf"

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}
FAILURE_LOG_NAME = "transcribe_failures.txt"


def read_audio(path: str, normalize: bool = True):
    """Read an audio file via pydub (supports mp3, wav, flac, ogg, etc.)."""
    import numpy as np

    # SoundFile reads PCM WAV without spawning ffmpeg per item; pydub handles compressed inputs.
    if Path(path).suffix.lower() == ".wav":
        import soundfile as sf

        y, fs = sf.read(path, dtype="float32" if normalize else "int16",
                        always_2d=True)
        return fs, y

    import pydub
    audio = pydub.AudioSegment.from_file(path)
    y = np.array(audio.get_array_of_samples())
    y = y.reshape(-1, audio.channels)
    if normalize:
        y = np.float32(y) / 2**15
    return audio.frame_rate, y


def load_model(weight_path: str, conf_path: str, device: str):
    """Load the Transkun model from a checkpoint and config."""
    import moduleconf
    import torch

    conf_manager = moduleconf.parseFromFile(conf_path)
    TransKun = conf_manager["Model"].module.TransKun
    conf = conf_manager["Model"].config

    # Load on CPU before transferring the completed model. This avoids a large
    # temporary allocation on the GPU when several replicas start together.
    checkpoint = torch.load(weight_path, map_location="cpu")
    model = TransKun(conf=conf)

    key = "best_state_dict" if "best_state_dict" in checkpoint else "state_dict"
    model.load_state_dict(checkpoint[key], strict=False)
    model = model.to(device)
    model.eval()
    return model


def transcribe_file(model, audio_path: Path, midi_path: Path, device: str,
                    tmp_tag: str):
    """Transcribe one audio file to MIDI using an already-loaded model."""
    import soxr
    import torch
    from taste.transcription.data import writeMidi

    fs, audio = read_audio(str(audio_path))
    if fs != model.fs:
        audio = soxr.resample(audio, fs, model.fs)

    x = torch.from_numpy(audio).to(device)
    with torch.inference_mode():
        notes_est = model.transcribe(x, discardSecondHalf=False)

    midi_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = midi_path.with_name(f".{midi_path.name}.{tmp_tag}.tmp")
    try:
        writeMidi(notes_est).write(str(tmp_path))
        tmp_path.replace(midi_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _run_shard(shard, device: str, weight: str, conf: str,
               wid: int, result_queue=None):
    """Transcribe one disjoint shard of files (one worker process)."""
    tag = f"[w{wid}]"
    print(f"{tag} loading model on {device} ({len(shard)} files) ...", flush=True)
    failed = []
    try:
        # Each process performs GPU inference and light audio I/O; allowing
        # every replica to create a large CPU thread pool hurts shared hosts.
        import torch
        torch.set_num_threads(1)
        if device.startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        model = load_model(weight, conf, device)
    except Exception as e:
        msg = f"MODEL_LOAD_FAILED on {device}: {e}"
        print(f"{tag} {msg}", file=sys.stderr, flush=True)
        failed.append(msg)
        if result_queue is not None:
            result_queue.put({"wid": wid, "ok": 0, "failed": failed})
        return failed

    for i, (audio_path, midi_path) in enumerate(shard, 1):
        if midi_path.exists():
            print(f"{tag} [{i}/{len(shard)}] skip existing {midi_path.name}",
                  flush=True)
            continue

        if i == 1 or i % 25 == 0 or i == len(shard):
            print(f"{tag} progress {i}/{len(shard)}", flush=True)
        try:
            tmp_tag = f"w{wid}.pid{os.getpid()}"
            transcribe_file(model, audio_path, midi_path, device, tmp_tag)
        except Exception as e:
            # Skip a bad file rather than abort; a re-run picks up the rest.
            print(f"{tag} FAILED {audio_path.name}: {e}", file=sys.stderr, flush=True)
            failed.append(audio_path.name)

    print(f"{tag} done — {len(shard) - len(failed)} ok, {len(failed)} failed",
          flush=True)
    if result_queue is not None:
        result_queue.put({"wid": wid, "ok": len(shard) - len(failed),
                          "failed": failed})
    return failed


def _parse_devices(single_device: str | None, devices_arg: str | None):
    if devices_arg:
        devices = [d.strip() for d in devices_arg.split(",") if d.strip()]
    else:
        devices = [single_device or "cpu"]

    if not devices:
        raise ValueError("no devices were specified")

    repeated = [d for d, n in Counter(devices).items() if n > 1]
    if repeated:
        raise ValueError(f"duplicate device(s): {', '.join(repeated)}")

    cuda_devices = [d for d in devices if d.startswith("cuda")]
    if len(cuda_devices) > 2:
        raise ValueError("at most 2 CUDA devices are allowed")

    return devices


def _cuda_index(device: str):
    if device == "cuda":
        return None
    if device.startswith("cuda:"):
        suffix = device.split(":", 1)[1]
        if suffix.isdigit():
            return suffix
    return None


def _confine_indexed_cuda_devices(devices):
    """Restrict visibility to listed CUDA indices and remap to local ids.

    Passing absolute devices like cuda:2,cuda:3 while all GPUs are visible can
    still let dependencies create incidental contexts on cuda:0. Confining
    CUDA_VISIBLE_DEVICES before worker processes import torch avoids that.
    """
    indices = [_cuda_index(d) for d in devices]
    if not indices or any(i is None for i in indices):
        return devices

    visible = ",".join(indices)
    existing = os.environ.get("CUDA_VISIBLE_DEVICES")
    if existing and existing != visible:
        raise ValueError(
            f"CUDA_VISIBLE_DEVICES is already {existing!r}, "
            f"but --devices requests {visible!r}"
        )

    os.environ["CUDA_VISIBLE_DEVICES"] = visible
    return [f"cuda:{i}" for i in range(len(devices))]


def _resolve_audio_input(audio_arg: Path, recursive: bool = False):
    if audio_arg.is_dir():
        if audio_arg.name != "audio" and (audio_arg / "audio").is_dir():
            audio_dir = audio_arg / "audio"
        else:
            audio_dir = audio_arg
        iterator = audio_dir.rglob("*") if recursive else audio_dir.iterdir()
        audio_files = sorted(
            (p for p in iterator if p.is_file() and p.suffix.lower() in AUDIO_EXTS),
            key=lambda p: p.name,
        )
        return audio_dir, audio_files

    if audio_arg.is_file():
        return audio_arg.parent, [audio_arg]

    raise FileNotFoundError(audio_arg)


def _output_tasks(audio_dir: Path, audio_files, midi_dir: Path):
    tasks = [
        (path, (midi_dir / path.relative_to(audio_dir)).with_suffix(".mid"))
        for path in audio_files
    ]
    outputs = Counter(midi for _, midi in tasks)
    duplicates = [str(path) for path, n in outputs.items() if n > 1]
    if duplicates:
        raise ValueError("multiple audio files map to the same MIDI: " + ", ".join(duplicates[:10]))
    return tasks


def _duration_seconds(path: Path) -> float:
    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as audio:
                return audio.getnframes() / audio.getframerate()
        except (wave.Error, OSError, ZeroDivisionError):
            pass
    return float(path.stat().st_size)


def _balanced_shards(tasks, n_workers: int):
    """Greedy longest-first assignment reduces the final straggler tail."""
    shards = [[] for _ in range(n_workers)]
    loads = [0.0] * n_workers
    for task in sorted(tasks, key=lambda item: _duration_seconds(item[0]), reverse=True):
        worker = min(range(n_workers), key=loads.__getitem__)
        shards[worker].append(task)
        loads[worker] += _duration_seconds(task[0])
    return shards, loads


def _write_failure_log(midi_dir: Path, failed):
    if not failed:
        return
    failure_log = midi_dir / FAILURE_LOG_NAME
    failure_log.write_text("\n".join(failed) + "\n", encoding="utf-8")
    print(f"Wrote failure list to {failure_log}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe piano audio to MIDI using Transkun V2."
    )
    parser.add_argument(
        "audio",
        help="An audio file, dataset audio/ folder, or dataset root.",
    )
    parser.add_argument(
        "--device", default=None,
        help="Single compute device: 'cpu', 'cuda', 'cuda:0', ... "
             "(default: cpu unless --devices is set)",
    )
    parser.add_argument(
        "--devices", default=None,
        help="Comma-separated devices for one worker per device, e.g. "
             "'cuda:0,cuda:1'. At most 2 CUDA devices are allowed.",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Parallel worker processes. Default is the number of --devices, "
             "or 1 for a single --device. If --workers exceeds --devices, "
             "workers are assigned to devices round-robin.",
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="Discover audio recursively and preserve its relative directory tree in MIDI output.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="MIDI root (default: a midi/ directory beside the input audio root).",
    )
    parser.add_argument("--weight", default=None, help="Path to model weights")
    parser.add_argument("--conf", default=None, help="Path to model config")
    args = parser.parse_args()

    audio_arg = Path(args.audio)
    try:
        audio_dir, audio_files = _resolve_audio_input(audio_arg, args.recursive)
        devices = _parse_devices(args.device, args.devices)
        devices = _confine_indexed_cuda_devices(devices)
    except FileNotFoundError:
        print(f"Error: path not found: {audio_arg}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if not audio_files:
        print(f"Error: no audio files in {audio_dir}", file=sys.stderr)
        sys.exit(1)

    # By default MIDI goes into a `midi/` folder sibling to the audio root,
    # preserving relative subdirectories for recursive datasets.
    midi_dir = args.output_dir or (audio_dir.parent / "midi")
    tasks = _output_tasks(audio_dir, audio_files, midi_dir)

    pending = [task for task in tasks if not task[1].exists()]
    todo = pending

    print(f"{len(audio_files)} audio file(s) in {audio_dir} — "
          f"{len(audio_files) - len(pending)} already transcribed, "
          f"{len(pending)} pending, {len(todo)} selected.")
    if not todo:
        return

    weight = args.weight or str(DEFAULT_WEIGHT)
    conf = args.conf or str(DEFAULT_CONF)
    requested_workers = args.workers or len(devices)
    if requested_workers < 1:
        print("Error: --workers must be >= 1", file=sys.stderr)
        sys.exit(1)

    n_workers = max(1, min(requested_workers, len(todo)))
    worker_devices = [
        devices[k % len(devices)]
        for k in range(n_workers)
    ]

    print(f"Devices: {', '.join(worker_devices)}")
    print(f"Workers: {n_workers}")
    print(f"MIDI output: {midi_dir}")
    midi_dir.mkdir(parents=True, exist_ok=True)

    if n_workers == 1:
        failed = _run_shard(todo, worker_devices[0], weight, conf, wid=0)
    else:
        # Duration-balanced shards are disjoint, so workers never touch the
        # same file and outputs cannot race.
        mp.set_start_method("spawn", force=True)
        result_queue = mp.Queue()
        procs = []
        shards, loads = _balanced_shards(todo, n_workers)
        print("Estimated audio seconds per worker: " +
              ", ".join(str(round(load)) for load in loads))
        for k in range(n_workers):
            shard = shards[k]
            p = mp.Process(
                target=_run_shard,
                args=(shard, worker_devices[k], weight, conf, k,
                      result_queue),
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
        failed = []
        for _ in procs:
            try:
                result = result_queue.get(timeout=1)
            except queue.Empty:
                continue
            failed.extend(result["failed"])
        for k, p in enumerate(procs):
            if p.exitcode != 0:
                failed.append(f"worker_{k}_exitcode_{p.exitcode}")

    _write_failure_log(midi_dir, failed)
    if failed:
        print(f"Finished with {len(failed)} failure(s). Re-run after fixing them.",
              file=sys.stderr)
        sys.exit(1)
    print(f"All done. MIDI written to {midi_dir}")


if __name__ == "__main__":
    main()
