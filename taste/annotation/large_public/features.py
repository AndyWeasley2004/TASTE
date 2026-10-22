#!/usr/bin/env python3
# Uses CLaMP 3 (https://github.com/sanderwood/clamp3) and Aria
# (https://github.com/EleutherAI/aria; loubb/aria-medium-embedding).
# The symbolic feature extractor is implemented in taste.representation.midi_features.
"""Extract the three feature families of the tag probes for a manifest.

`symbolic` computes the 32 interpretable MIDI statistics; `aria` and `clamp3`
run the Aria embedding model and the official CLaMP 3 symbolic-C2 extractor;
`clamp3-text` embeds the per-piece imagery captions in the same CLaMP 3 space
(the target of the caption-space ridge projection fitted by `tag_probes fit`);
`verify` checks a consolidated archive against its manifest.
"""
from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from taste import paths
from taste.representation.midi_features import FEATURE_NAMES, extract_symbolic_features

ROOT = paths.data("large_public_annotation")
MANIFEST = ROOT / "benchmark.parquet"
ARIA_REVISION = "0b6896f797763b23bee4676db365f07485757dbb"
ARIA_WINDOW = 2048

_ARIA_WORKER_TOKENIZER = None


def manifest_rows(path: Path) -> list[tuple[str, str]]:
    table = pq.read_table(path, columns=["piece_id", "midi_path"])
    return list(zip(table["piece_id"].to_pylist(), table["midi_path"].to_pylist()))


def consolidate(cache: Path, rows: list[tuple[str, str]], output: Path) -> None:
    by_stem = {path.stem: path for path in cache.rglob("*.npy")}
    ids, features = [], []
    for piece_id, _ in rows:
        path = by_stem.get(piece_id)
        if path is None:
            raise FileNotFoundError(f"missing feature for {piece_id} under {cache}")
        vector = np.asarray(np.load(path), dtype=np.float32).squeeze()
        if vector.ndim != 1:
            raise ValueError(f"expected global vector for {piece_id}, found {vector.shape}")
        ids.append(piece_id)
        features.append(vector)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            ids=np.asarray(ids, dtype=object),
            features=np.stack(features),
        )
    os.replace(temporary, output)


def consolidate_directory(cache: Path, output: Path) -> None:
    paths = sorted(cache.rglob("*.npy"))
    if not paths:
        raise FileNotFoundError(f"no .npy features under {cache}")
    ids, features = [], []
    for path in paths:
        vector = np.asarray(np.load(path), dtype=np.float32).squeeze()
        if vector.ndim != 1:
            raise ValueError(f"expected global vector in {path}, found {vector.shape}")
        ids.append(path.stem)
        features.append(vector)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            ids=np.asarray(ids, dtype=object),
            features=np.stack(features),
        )
    os.replace(temporary, output)


# --- symbolic statistics ----------------------------------------------------


def _feature_worker(payload: tuple[str, str]) -> tuple[str, np.ndarray | None, str | None]:
    piece_id, path = payload
    try:
        return piece_id, extract_symbolic_features(path), None
    except Exception as error:
        return piece_id, None, repr(error)


def extract_symbolic(manifest: Path, workers: int, output: Path) -> None:
    table = pq.read_table(manifest, columns=["piece_id", "midi_path"])
    payloads = list(zip(table["piece_id"].to_pylist(), table["midi_path"].to_pylist()))
    features_by_id = {}
    errors = []
    with mp.Pool(workers) as pool:
        for index, (piece_id, vector, error) in enumerate(
            pool.imap_unordered(_feature_worker, payloads, chunksize=8), 1
        ):
            if vector is None:
                errors.append({"piece_id": piece_id, "error": error})
            else:
                features_by_id[piece_id] = vector
            if index % 1000 == 0 or index == len(payloads):
                print(
                    f"[{index}/{len(payloads)}] ok={len(features_by_id)} "
                    f"errors={len(errors)}",
                    flush=True,
                )
    if errors:
        (output.parent / "symbolic_errors.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in errors)
        )
    ordered_ids = [piece_id for piece_id, _ in payloads]
    missing = np.full(len(FEATURE_NAMES), np.nan, dtype=np.float32)
    np.savez_compressed(
        output,
        ids=np.asarray(ordered_ids, dtype=object),
        features=np.stack([
            features_by_id.get(piece_id, missing) for piece_id in ordered_ids
        ]),
        feature_names=np.asarray(FEATURE_NAMES, dtype=object),
        error_count=np.asarray(len(errors), dtype=np.int64),
    )


# --- Aria embeddings --------------------------------------------------------


def _init_aria_worker(model_name: str, revision: str) -> None:
    import transformers.tokenization_utils as tokenization_utils
    from transformers import AutoTokenizer
    from transformers.tokenization_utils_base import BatchEncoding

    if not hasattr(tokenization_utils, "BatchEncoding"):
        tokenization_utils.BatchEncoding = BatchEncoding
    global _ARIA_WORKER_TOKENIZER
    _ARIA_WORKER_TOKENIZER = AutoTokenizer.from_pretrained(
        model_name, revision=revision, trust_remote_code=True
    )


def _aria_tokenize_worker(payload: tuple[str, str]) -> tuple[str, list[int]]:
    piece_id, midi_path = payload
    ids = _ARIA_WORKER_TOKENIZER.encode_from_file(
        midi_path, return_tensors="pt"
    ).input_ids[0, :2048]
    return piece_id, ids.tolist()


def extract_aria(
    manifest: Path,
    cache: Path,
    output: Path,
    model_name: str,
    revision: str,
    device: str,
    batch_size: int,
    workers: int,
    num_shards: int,
    shard_index: int,
) -> None:
    import torch
    import transformers.tokenization_utils as tokenization_utils
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.tokenization_utils_base import BatchEncoding

    # The official remote tokenizer imports BatchEncoding from its
    # Transformers-4 location; Transformers 5 moved the public class.
    if not hasattr(tokenization_utils, "BatchEncoding"):
        tokenization_utils.BatchEncoding = BatchEncoding

    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise ValueError("expected 0 <= shard_index < num_shards")
    rows = manifest_rows(manifest)[shard_index::num_shards]
    cache.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, revision=revision, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, revision=revision, trust_remote_code=True, dtype=torch.bfloat16
    ).to(device).eval()
    eos = tokenizer._convert_token_to_id(tokenizer.eos_token)
    pad = tokenizer._convert_token_to_id(tokenizer.pad_token)
    pending = [
        (piece_id, midi_path)
        for piece_id, midi_path in rows
        if not (cache / f"{piece_id}.npy").is_file()
    ]
    completed = len(rows) - len(pending)
    pool = None
    if workers > 1:
        pool = mp.get_context("spawn").Pool(
            workers,
            initializer=_init_aria_worker,
            initargs=(model_name, revision),
        )
        encoded = pool.imap(_aria_tokenize_worker, pending, chunksize=2)
    else:
        encoded = (
            (
                piece_id,
                tokenizer.encode_from_file(midi_path, return_tensors="pt")
                .input_ids[0, :2048]
                .tolist(),
            )
            for piece_id, midi_path in pending
        )
    while True:
        batch = list(itertools.islice(encoded, batch_size))
        if not batch:
            break
        sequences = []
        for _, values in batch:
            ids = torch.tensor(values, dtype=torch.long)
            ids[-1] = eos
            sequences.append(ids)
        padded = torch.nn.utils.rnn.pad_sequence(
            sequences, batch_first=True, padding_value=pad
        )
        with torch.inference_mode():
            vectors = model(input_ids=padded.to(device))[0]
        for (piece_id, _), vector in zip(batch, vectors):
            np.save(cache / f"{piece_id}.npy", vector.float().cpu().numpy())
        completed += len(batch)
        if completed % 100 < len(batch) or completed == len(rows):
            print(f"[{completed}/{len(rows)}]", flush=True)
    if pool is not None:
        pool.close()
        pool.join()
    consolidate(cache, rows, output)


def _aria_tokenize_windows_worker(payload: tuple[str, str]) -> tuple[str, list[list[int]]]:
    piece_id, midi_path = payload
    ids = _ARIA_WORKER_TOKENIZER.encode_from_file(
        midi_path, return_tensors="pt"
    ).input_ids[0].tolist()
    return piece_id, [ids[start:start + ARIA_WINDOW]
                      for start in range(0, len(ids), ARIA_WINDOW)]


def extract_aria_windowed(
    manifest: Path,
    cache: Path,
    output: Path,
    model_name: str,
    revision: str,
    device: str,
    batch_size: int,
    workers: int,
    num_shards: int,
    shard_index: int,
) -> None:
    """`extract_aria` over the whole piece: the mean of its per-window embeddings.

    The token sequence is split into consecutive windows of at most
    ARIA_WINDOW tokens (the last one may be shorter), every window is embedded
    exactly as `extract_aria` embeds its single truncated window (EOS forced at
    the last position), and the stored vector is the mean over windows. A piece
    that fits in one window therefore reproduces `extract_aria` exactly. The
    per-piece window count of this run is written to
    `cache/window_counts_<shard_index>_<num_shards>.json`.
    """
    import torch
    import transformers.tokenization_utils as tokenization_utils
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.tokenization_utils_base import BatchEncoding

    if not hasattr(tokenization_utils, "BatchEncoding"):
        tokenization_utils.BatchEncoding = BatchEncoding

    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise ValueError("expected 0 <= shard_index < num_shards")
    rows = manifest_rows(manifest)[shard_index::num_shards]
    cache.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, revision=revision, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, revision=revision, trust_remote_code=True, dtype=torch.bfloat16
    ).to(device).eval()
    eos = tokenizer._convert_token_to_id(tokenizer.eos_token)
    pad = tokenizer._convert_token_to_id(tokenizer.pad_token)
    pending = [
        (piece_id, midi_path)
        for piece_id, midi_path in rows
        if not (cache / f"{piece_id}.npy").is_file()
    ]
    completed = len(rows) - len(pending)
    pool = None
    if workers > 1:
        pool = mp.get_context("spawn").Pool(
            workers,
            initializer=_init_aria_worker,
            initargs=(model_name, revision),
        )
        encoded = pool.imap(_aria_tokenize_windows_worker, pending, chunksize=2)
    else:
        encoded = (
            (
                piece_id,
                [values[start:start + ARIA_WINDOW]
                 for start in range(0, len(values), ARIA_WINDOW)],
            )
            for piece_id, values in (
                (
                    piece_id,
                    tokenizer.encode_from_file(midi_path, return_tensors="pt")
                    .input_ids[0]
                    .tolist(),
                )
                for piece_id, midi_path in pending
            )
        )

    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    buffer: list[tuple[str, list[int]]] = []

    def run_batch(batch: list[tuple[str, list[int]]]) -> None:
        sequences = []
        for _, values in batch:
            ids = torch.tensor(values, dtype=torch.long)
            ids[-1] = eos
            sequences.append(ids)
        padded = torch.nn.utils.rnn.pad_sequence(
            sequences, batch_first=True, padding_value=pad
        )
        with torch.inference_mode():
            vectors = model(input_ids=padded.to(device))[0]
        for (piece_id, _), vector in zip(batch, vectors):
            value = vector.float().cpu().numpy()
            sums[piece_id] = sums.get(piece_id, 0.0) + value
            counts[piece_id] = counts.get(piece_id, 0) + 1

    window_counts: dict[str, int] = {}
    for piece_id, windows in encoded:
        window_counts[piece_id] = len(windows)
        buffer.extend((piece_id, window) for window in windows)
        while len(buffer) >= batch_size:
            run_batch(buffer[:batch_size])
            del buffer[:batch_size]
        done = [name for name in sums if counts[name] == window_counts[name]]
        for name in done:
            np.save(cache / f"{name}.npy", sums.pop(name) / window_counts[name])
            completed += 1
        if done and (completed % 100 < len(done) or completed == len(rows)):
            print(f"[{completed}/{len(rows)}]", flush=True)
    while buffer:
        run_batch(buffer[:batch_size])
        del buffer[:batch_size]
    for name, vector in sums.items():
        np.save(cache / f"{name}.npy", vector / window_counts[name])
    print(f"[{len(rows)}/{len(rows)}]", flush=True)
    if pool is not None:
        pool.close()
        pool.join()
    (cache / f"window_counts_{shard_index}_{num_shards}.json").write_text(
        json.dumps(window_counts) + "\n"
    )
    consolidate(cache, rows, output)


# --- CLaMP 3 ---------------------------------------------------------------


def _clamp3_script(repo: Path) -> Path:
    script = repo / "code" / "extract_clamp3.py"
    if not script.is_file():
        raise FileNotFoundError(f"official CLaMP 3 entry point missing: {script}")
    config_text = (repo / "code" / "config.py").read_text()
    if '"weights_clamp3_c2"' not in config_text:
        raise ValueError("CLaMP 3 must be configured with its symbolic C2 checkpoint")
    return script


def _shard_symlinks(sources: list[Path], shard_root: Path, count: int) -> list[Path]:
    """Round-robin the inputs into `count` disjoint symlink directories."""
    shard_dirs = [shard_root / f"shard_{index:02d}" for index in range(count)]
    for directory in shard_dirs:
        directory.mkdir(parents=True, exist_ok=True)
    for index, source in enumerate(sources):
        if not source.is_file():
            raise FileNotFoundError(f"missing CLaMP staging input: {source}")
        destination = shard_dirs[index % count] / source.name
        if not destination.exists():
            destination.symlink_to(source)
    return shard_dirs


def _run_extractors(
    script: Path,
    repo: Path,
    python: str,
    shard_dirs: list[Path],
    cache: Path,
    gpus: list[str],
    log_dir: Path,
) -> None:
    """Run the untouched official extractor once per GPU on disjoint shards."""
    cache.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[subprocess.Popen, object, str]] = []
    for index, (gpu, shard_dir) in enumerate(zip(gpus, shard_dirs)):
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu
        environment["OMP_NUM_THREADS"] = "1"
        environment["MKL_NUM_THREADS"] = "1"
        environment["OPENBLAS_NUM_THREADS"] = "1"
        environment["NUMEXPR_NUM_THREADS"] = "1"
        environment["TOKENIZERS_PARALLELISM"] = "false"
        log_path = log_dir / f"clamp3_shard_{index:02d}_gpu_{gpu}.log"
        handle = log_path.open("a")
        process = subprocess.Popen(
            [python, str(script), str(shard_dir), str(cache), "--get_global"],
            cwd=repo / "code",
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((process, handle, str(log_path)))
        print(
            f"started CLaMP shard {index + 1}/{len(gpus)} on GPU {gpu}: {log_path}",
            flush=True,
        )

    failures = []
    for process, handle, log_path in processes:
        return_code = process.wait()
        handle.close()
        if return_code:
            failures.append(f"{log_path} exited {return_code}")
    if failures:
        raise RuntimeError("; ".join(failures))


def _midi_to_mtf(payload: tuple[str, str, str]) -> str | None:
    """CLaMP 3's official MTF semantics with bounded multiprocessing."""
    piece_id, midi_path, output_path = payload
    destination = Path(output_path)
    if destination.is_file():
        return None
    try:
        import mido

        midi = mido.MidiFile(midi_path)
        messages = [f"ticks_per_beat {midi.ticks_per_beat}"]
        ignored_meta = {
            "text", "copyright", "track_name", "instrument_name", "lyrics",
            "marker", "cue_marker", "device_name",
        }
        for message in midi.merged_track:
            if message.is_meta and message.type in ignored_meta:
                continue
            text = " ".join(str(value) for value in message.dict().values())
            messages.append(text.encode("unicode_escape").decode("utf-8"))
        temporary = destination.with_suffix(".tmp.mtf")
        temporary.write_text("\n".join(messages), encoding="utf-8")
        os.replace(temporary, destination)
        return None
    except Exception as error:
        return f"{piece_id}: {error!r}"


def extract_clamp3(
    manifest: Path,
    repo: Path,
    staging: Path,
    shard_root: Path,
    cache: Path,
    output: Path,
    python: str,
    workers: int,
    gpus: list[str],
    log_dir: Path,
) -> None:
    script = _clamp3_script(repo)
    rows = manifest_rows(manifest)
    staging.mkdir(parents=True, exist_ok=True)
    payloads = [
        (piece_id, midi_path, str(staging / f"{piece_id}.mtf"))
        for piece_id, midi_path in rows
    ]
    with mp.Pool(workers) as pool:
        for index, error in enumerate(
            pool.imap(_midi_to_mtf, payloads, chunksize=16), 1
        ):
            if error:
                raise ValueError(error)
            if index % 1000 == 0 or index == len(payloads):
                print(f"[MTF {index}/{len(payloads)}]", flush=True)
    shard_dirs = _shard_symlinks(
        [staging / f"{piece_id}.mtf" for piece_id, _ in rows], shard_root, len(gpus)
    )
    _run_extractors(script, repo, python, shard_dirs, cache, gpus, log_dir)
    consolidate(cache, rows, output)


def extract_clamp3_text(
    manifest: Path,
    repo: Path,
    staging: Path,
    shard_root: Path,
    cache: Path,
    output: Path,
    python: str,
    gpus: list[str],
    log_dir: Path,
) -> None:
    script = _clamp3_script(repo)
    table = pq.read_table(manifest, columns=["piece_id", "imagery"])
    staging.mkdir(parents=True, exist_ok=True)
    written = []
    for piece_id, imagery in zip(
        table["piece_id"].to_pylist(), table["imagery"].to_pylist()
    ):
        value = str(imagery or "").strip()
        if not value:
            continue
        path = staging / f"{piece_id}.txt"
        path.write_text(value + "\n")
        written.append(path)
    print(f"wrote {len(written):,} imagery texts", flush=True)
    shard_dirs = _shard_symlinks(written, shard_root, len(gpus))
    _run_extractors(script, repo, python, shard_dirs, cache, gpus, log_dir)
    consolidate_directory(cache, output)


# --- verification -----------------------------------------------------------


def verify_archive(
    manifest: Path,
    archive: Path,
    expected_dim: int,
    shards: list[Path],
) -> None:
    data = np.load(archive, allow_pickle=True)
    ids = [str(value) for value in data["ids"]]
    features = np.asarray(data["features"])
    manifest_ids = [piece_id for piece_id, _ in manifest_rows(manifest)]
    finite = all(
        np.isfinite(features[start:start + 4096]).all()
        for start in range(0, len(features), 4096)
    )
    report = {
        "archive": str(archive),
        "rows": len(ids),
        "unique_ids": len(set(ids)),
        "shape": list(features.shape),
        "dtype": str(features.dtype),
        "finite": bool(finite),
        "exact_manifest_order": ids == manifest_ids,
    }
    if shards:
        shard_ids = []
        for shard in shards:
            shard_data = np.load(shard, allow_pickle=True)
            shard_ids.extend(str(value) for value in shard_data["ids"])
        report.update({
            "shard_rows": len(shard_ids),
            "unique_shard_ids": len(set(shard_ids)),
            "shard_union_exact": set(shard_ids) == set(ids),
        })
    expected_shape = (len(manifest_ids), expected_dim)
    if (
        features.shape != expected_shape
        or len(set(ids)) != len(ids)
        or not finite
        or ids != manifest_ids
        or (shards and (
            len(shard_ids) != len(ids)
            or len(set(shard_ids)) != len(ids)
            or set(shard_ids) != set(ids)
        ))
    ):
        raise ValueError(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


def parse_gpus(value: str) -> list[str]:
    gpus = [item.strip() for item in value.split(",") if item.strip()]
    if not 1 <= len(gpus) <= 16:
        raise ValueError("--gpus must specify between 1 and 16 workers")
    return gpus


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    symbolic = sub.add_parser("symbolic")
    symbolic.add_argument("--manifest", type=Path, default=MANIFEST)
    symbolic.add_argument("--workers", type=int, default=8)
    symbolic.add_argument("--output", type=Path, default=ROOT / "symbolic_features.npz")

    aria = sub.add_parser("aria")
    aria.add_argument("--manifest", type=Path, default=MANIFEST)
    aria.add_argument("--cache", type=Path, default=ROOT / "aria_cache")
    aria.add_argument("--output", type=Path, default=ROOT / "aria_features.npz")
    aria.add_argument("--model", default="loubb/aria-medium-embedding")
    aria.add_argument("--revision", default=ARIA_REVISION)
    aria.add_argument("--device", default="cuda")
    aria.add_argument("--batch-size", type=int, default=4)
    aria.add_argument("--workers", type=int, default=1)
    aria.add_argument("--num-shards", type=int, default=1)
    aria.add_argument("--shard-index", type=int, default=0)

    clamp = sub.add_parser("clamp3")
    clamp.add_argument("--manifest", type=Path, default=MANIFEST)
    clamp.add_argument("--repo", type=Path, default=paths.CLAMP3_ROOT)
    clamp.add_argument("--staging", type=Path, default=ROOT / "clamp3_midi_staging")
    clamp.add_argument("--shard-root", type=Path, default=ROOT / "clamp3_midi_shards")
    clamp.add_argument("--cache", type=Path, default=ROOT / "clamp3_cache")
    clamp.add_argument("--output", type=Path, default=ROOT / "clamp3_features.npz")
    clamp.add_argument("--python", default=os.environ.get("TASTE_CLAMP3_PYTHON", sys.executable))
    clamp.add_argument("--workers", type=int, default=8)
    clamp.add_argument(
        "--gpus", required=True,
        help="comma-separated physical GPU IDs; repeat an ID for multiple workers",
    )
    clamp.add_argument("--log-dir", type=Path, default=ROOT / "logs")

    clamp_text = sub.add_parser("clamp3-text")
    clamp_text.add_argument("--manifest", type=Path, default=MANIFEST)
    clamp_text.add_argument("--repo", type=Path, default=paths.CLAMP3_ROOT)
    clamp_text.add_argument("--staging", type=Path, default=ROOT / "clamp3_imagery_text")
    clamp_text.add_argument("--shard-root", type=Path, default=ROOT / "clamp3_text_shards")
    clamp_text.add_argument("--cache", type=Path, default=ROOT / "clamp3_text_cache")
    clamp_text.add_argument("--output", type=Path, default=ROOT / "clamp3_text_features.npz")
    clamp_text.add_argument("--python", default=os.environ.get("TASTE_CLAMP3_PYTHON", sys.executable))
    clamp_text.add_argument(
        "--gpus", required=True,
        help="comma-separated physical GPU IDs; repeat an ID for multiple workers",
    )
    clamp_text.add_argument("--log-dir", type=Path, default=ROOT / "logs")

    verify = sub.add_parser("verify")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--archive", type=Path, required=True)
    verify.add_argument("--expected-dim", type=int, required=True)
    verify.add_argument("--shard", action="append", type=Path, default=[])

    args = parser.parse_args()
    if hasattr(args, "workers") and not 1 <= args.workers <= 8:
        raise ValueError("--workers must be between 1 and 8")
    if hasattr(args, "batch_size") and args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.command == "symbolic":
        extract_symbolic(args.manifest, args.workers, args.output)
    elif args.command == "aria":
        extract_aria(
            args.manifest, args.cache, args.output, args.model,
            args.revision, args.device, args.batch_size, args.workers,
            args.num_shards, args.shard_index,
        )
    elif args.command == "clamp3":
        extract_clamp3(
            args.manifest, args.repo, args.staging, args.shard_root,
            args.cache, args.output, args.python, args.workers,
            parse_gpus(args.gpus), args.log_dir,
        )
    elif args.command == "clamp3-text":
        extract_clamp3_text(
            args.manifest, args.repo, args.staging, args.shard_root,
            args.cache, args.output, args.python,
            parse_gpus(args.gpus), args.log_dir,
        )
    else:
        verify_archive(args.manifest, args.archive, args.expected_dim, args.shard)


if __name__ == "__main__":
    main()
