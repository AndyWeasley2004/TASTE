#!/usr/bin/env python3
"""Estimate one global major/minor key per MIDI file (Kostka-Payne profiles over performed note durations).

``--dataset curated`` and ``--dataset high_quality`` write the key into each
piece's label/metadata JSON; ``--dataset aria`` writes deterministic Parquet
shards next to an Aria MIDI subset.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from bisect import bisect_right
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import mido
import pyarrow.parquet as pq

from taste import paths


DEFAULT_ROOT = paths.data("curated")
DEFAULT_HIGH_QUALITY_ROOT = paths.data("high_quality")
DEFAULT_ARIA_ROOT = paths.data("aria_midi", "pruned")
EXPECTED_PIECES = 51_130
EXPECTED_ARIA_MIDIS_BY_ROOT = {
    "pruned": 820_944,
    "deduped": 371_053,
}
FIELD_SOURCE = "kostka_payne"

# Kostka-Payne profiles from Music and Probability, Table 6.1. These are
# also the profiles exposed by Partitura as key_profiles="kostka_payne".
MAJOR_PROFILE = (
    0.748, 0.060, 0.488, 0.082, 0.670, 0.460,
    0.096, 0.715, 0.104, 0.366, 0.057, 0.400,
)
MINOR_PROFILE = (
    0.712, 0.048, 0.474, 0.618, 0.049, 0.460,
    0.105, 0.747, 0.404, 0.067, 0.133, 0.330,
)
MAJOR_NAMES = (
    "C major", "Db major", "D major", "Eb major", "E major", "F major",
    "F# major", "G major", "Ab major", "A major", "Bb major", "B major",
)
MINOR_NAMES = (
    "C minor", "C# minor", "D minor", "D# minor", "E minor", "F minor",
    "F# minor", "G minor", "G# minor", "A minor", "Bb minor", "B minor",
)
KEY_NAMES = MAJOR_NAMES + MINOR_NAMES


def key_token(key: str) -> str:
    return f"<KEY_{key.replace('#', 'S').replace('b', 'B').replace(' ', '_').upper()}>"


def rotated_profile(profile: tuple[float, ...], tonic: int) -> tuple[float, ...]:
    return tuple(profile[(pitch_class - tonic) % 12] for pitch_class in range(12))


def centered_unit(profile: tuple[float, ...]) -> tuple[float, ...]:
    mean = sum(profile) / len(profile)
    centered = tuple(value - mean for value in profile)
    norm = math.sqrt(sum(value * value for value in centered))
    return tuple(value / norm for value in centered)


KEY_PROFILES = tuple(
    centered_unit(rotated_profile(profile, tonic))
    for profile in (MAJOR_PROFILE, MINOR_PROFILE)
    for tonic in range(12)
)


def build_tick_to_seconds(midi: mido.MidiFile):
    """Build a converter from absolute ticks using the performance tempo map."""
    tempo_events: list[tuple[int, int, int, int]] = []
    for track_index, track in enumerate(midi.tracks):
        absolute_tick = 0
        for message_index, message in enumerate(track):
            absolute_tick += message.time
            if message.type == "set_tempo":
                tempo_events.append(
                    (absolute_tick, track_index, message_index, message.tempo)
                )
    tempo_events.sort()

    ticks = [0]
    seconds = [0.0]
    tempos = [500_000]
    for tick, _, _, tempo in tempo_events:
        if tick == ticks[-1]:
            tempos[-1] = tempo
            continue
        elapsed = mido.tick2second(
            tick - ticks[-1], midi.ticks_per_beat, tempos[-1]
        )
        ticks.append(tick)
        seconds.append(seconds[-1] + elapsed)
        tempos.append(tempo)

    def tick_to_seconds(tick: int) -> float:
        index = bisect_right(ticks, tick) - 1
        return seconds[index] + mido.tick2second(
            tick - ticks[index], midi.ticks_per_beat, tempos[index]
        )

    return tick_to_seconds


def pitch_class_durations(midi_path: Path) -> tuple[list[float], int]:
    """Return performance-time note durations in seconds by pitch class."""
    midi = mido.MidiFile(midi_path, clip=True)
    if midi.ticks_per_beat <= 0:
        raise ValueError("SMPTE-time MIDI is not supported")
    tick_to_seconds = build_tick_to_seconds(midi)
    durations = [0.0] * 12
    note_count = 0

    for track in midi.tracks:
        absolute_tick = 0
        active: dict[tuple[int, int], deque[int]] = defaultdict(deque)
        for message in track:
            absolute_tick += message.time
            if message.type not in ("note_on", "note_off"):
                continue
            channel = getattr(message, "channel", 0)
            note = message.note
            if channel == 9:
                continue
            key = (channel, note)
            is_on = message.type == "note_on" and message.velocity > 0
            if is_on:
                active[key].append(absolute_tick)
                continue
            if not active[key]:
                continue
            start_tick = active[key].popleft()
            duration = tick_to_seconds(absolute_tick) - tick_to_seconds(start_tick)
            if duration > 0:
                durations[note % 12] += duration
                note_count += 1

        for (_, note), starts in active.items():
            for start_tick in starts:
                duration = (
                    tick_to_seconds(absolute_tick) - tick_to_seconds(start_tick)
                )
                if duration > 0:
                    durations[note % 12] += duration
                    note_count += 1

    if note_count == 0 or sum(durations) <= 0:
        raise ValueError("MIDI contains no positive-duration pitched notes")
    return durations, note_count


def estimate_key(midi_path: Path) -> dict[str, Any]:
    durations, note_count = pitch_class_durations(midi_path)
    mean = sum(durations) / 12
    centered = tuple(value - mean for value in durations)
    norm = math.sqrt(sum(value * value for value in centered))
    if norm == 0:
        raise ValueError("MIDI has a uniform pitch-class distribution")
    unit_input = tuple(value / norm for value in centered)
    scores = [
        sum(value * reference for value, reference in zip(unit_input, profile))
        for profile in KEY_PROFILES
    ]
    ranking = sorted(range(24), key=lambda index: (-scores[index], index))
    best, second = ranking[:2]
    return {
        "key": KEY_NAMES[best],
        "score": scores[best],
        "runner_up": KEY_NAMES[second],
        "runner_up_score": scores[second],
        "margin": scores[best] - scores[second],
        "note_count": note_count,
    }


def process_piece(root: str, piece_id: str) -> dict[str, Any]:
    piece_dir = Path(root) / "pieces" / piece_id
    result = estimate_key(piece_dir / "performance.mid")
    result["piece_id"] = piece_id
    return result


def process_high_quality_sample(root: str, sample_id: str) -> dict[str, Any]:
    result = estimate_key(Path(root) / "midi" / f"{sample_id}.mid")
    result["piece_id"] = sample_id
    return result


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".key-estimation-tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def process_aria_shard(
    root: str,
    relative_paths: list[str],
    output_path: str,
) -> dict[str, Any]:
    """Estimate one deterministic Aria shard and atomically write Parquet."""
    import pyarrow as pa
    import pyarrow.parquet as parquet

    destination = Path(output_path)
    if destination.exists():
        metadata = parquet.read_metadata(destination)
        error_path = destination.with_suffix(".errors.json")
        errors = (
            len(json.loads(error_path.read_text(encoding="utf-8"))["errors"])
            if error_path.exists()
            else 0
        )
        return {
            "status": "skipped",
            "rows": metadata.num_rows,
            "errors": errors,
            "output": destination.name,
        }

    rows = []
    errors = []
    for relative_path in relative_paths:
        midi_path = Path(root) / relative_path
        try:
            result = estimate_key(midi_path)
            stem = midi_path.stem
            composition_id, separator, segment_id = stem.rpartition("_")
            if not separator or not composition_id or not segment_id.isdigit():
                raise ValueError(f"unexpected Aria filename: {midi_path.name}")
            rows.append({
                "midi_relpath": relative_path,
                "piece_id": stem,
                "composition_id": composition_id,
                "segment_id": int(segment_id),
                "key": result["key"],
                "key_token": key_token(result["key"]),
                "score": result["score"],
                "runner_up": result["runner_up"],
                "runner_up_score": result["runner_up_score"],
                "margin": result["margin"],
                "note_count": result["note_count"],
            })
        except Exception as error:
            errors.append({
                "midi_relpath": relative_path,
                "error": f"{type(error).__name__}: {error}",
            })

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".parquet.key-estimation-tmp")
    parquet.write_table(
        pa.Table.from_pylist(rows),
        temporary,
        compression="zstd",
        row_group_size=2_048,
    )
    os.replace(temporary, destination)
    error_path = destination.with_suffix(".errors.json")
    if errors:
        atomic_write_json(error_path, {"errors": errors})
    return {
        "status": "written",
        "rows": len(rows),
        "errors": len(errors),
        "output": destination.name,
    }


def run_aria(
    root: Path,
    output_dir: Path,
    workers: int,
    shard_size: int,
    log_every: int,
) -> None:
    midi_paths = sorted((root / "data").glob("*/*.mid"))
    expected_midi_count = EXPECTED_ARIA_MIDIS_BY_ROOT.get(root.name)
    if expected_midi_count is None:
        raise ValueError(
            f"unknown Aria subset root {root.name!r}; add its expected count "
            "before running a complete estimation"
        )
    if len(midi_paths) != expected_midi_count:
        raise ValueError(
            f"expected {expected_midi_count} Aria MIDIs, "
            f"found {len(midi_paths)}"
        )
    relative_paths = [path.relative_to(root).as_posix() for path in midi_paths]
    if len(relative_paths) != len(set(relative_paths)):
        raise ValueError("duplicate Aria relative MIDI paths")

    shards = [
        relative_paths[start:start + shard_size]
        for start in range(0, len(relative_paths), shard_size)
    ]
    started = time.monotonic()
    written = skipped = rows = errors = 0
    output_dir.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                process_aria_shard,
                str(root),
                shard,
                str(output_dir / f"part-{index:05d}.parquet"),
            ): index
            for index, shard in enumerate(shards)
        }
        for completed, future in enumerate(as_completed(futures), 1):
            result = future.result()
            written += result["status"] == "written"
            skipped += result["status"] == "skipped"
            rows += result["rows"]
            errors += result["errors"]
            if completed % log_every == 0 or completed == len(shards):
                elapsed = time.monotonic() - started
                print(
                    f"[{completed}/{len(shards)} shards] rows={rows} "
                    f"errors={errors} written={written} skipped={skipped} "
                    f"midi_per_sec={(rows + errors) / max(elapsed, 1e-9):.2f}",
                    flush=True,
                )

    manifest = {
        "dataset": "aria",
        "root": str(root),
        "method": FIELD_SOURCE,
        "expected_midi_count": expected_midi_count,
        "successful_rows": rows,
        "errors": errors,
        "shard_size": shard_size,
        "shards": len(shards),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    if errors:
        raise SystemExit(1)


def update_label(
    root: Path,
    result: dict[str, Any],
    overwrite: bool,
) -> str:
    piece_id = result["piece_id"]
    label_path = root / "pieces" / piece_id / "label.json"
    label = json.loads(label_path.read_text(encoding="utf-8"))
    if label.get("piece_id", label.get("work_id")) != piece_id:
        raise ValueError(f"label ID mismatch: {label_path}")

    fields = label.get("accepted_fields")
    sources = label.get("field_sources")
    if not isinstance(fields, dict) or not isinstance(sources, dict):
        raise ValueError(f"invalid label schema: {label_path}")
    existing = fields.get("key")
    existing_source = sources.get("key")
    if existing is not None and (
        existing != result["key"] or existing_source != FIELD_SOURCE
    ) and not overwrite:
        raise ValueError(
            f"existing key differs for {piece_id}: "
            f"{existing!r} ({existing_source!r})"
        )
    if existing == result["key"] and existing_source == FIELD_SOURCE:
        return "skipped"

    fields["key"] = result["key"]
    sources["key"] = FIELD_SOURCE
    atomic_write_json(label_path, label)
    return "updated"


def update_high_quality_metadata(
    root: Path,
    result: dict[str, Any],
    overwrite: bool,
) -> str:
    sample_id = result["piece_id"]
    path = root / "metadata" / f"{sample_id}.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("sample_id") != sample_id:
        raise ValueError(f"metadata ID mismatch: {path}")
    existing = metadata.get("key")
    existing_source = metadata.get("key_source")
    if existing is not None and (
        existing != result["key"] or existing_source != FIELD_SOURCE
    ) and not overwrite:
        raise ValueError(
            f"existing key differs for {sample_id}: "
            f"{existing!r} ({existing_source!r})"
        )
    if existing == result["key"] and existing_source == FIELD_SOURCE:
        return "skipped"
    metadata["key"] = result["key"]
    metadata["key_source"] = FIELD_SOURCE
    atomic_write_json(path, metadata)
    return "updated"


def load_piece_ids(
    root: Path,
    requested: list[str] | None,
) -> list[str]:
    rows = pq.read_table(
        root / "manifests/pieces.parquet", columns=["piece_id"]
    ).to_pylist()
    manifest_ids = sorted(row["piece_id"] for row in rows)
    if len(manifest_ids) != EXPECTED_PIECES or len(set(manifest_ids)) != EXPECTED_PIECES:
        raise ValueError(
            f"expected {EXPECTED_PIECES} unique manifest IDs, found "
            f"{len(manifest_ids)} rows and {len(set(manifest_ids))} IDs"
        )
    if requested:
        unknown = sorted(set(requested) - set(manifest_ids))
        if unknown:
            raise ValueError(f"unknown piece IDs: {unknown}")
        piece_ids = sorted(set(requested))
    else:
        piece_ids = manifest_ids
    return piece_ids


def load_high_quality_ids(
    root: Path,
    requested: list[str] | None,
) -> list[str]:
    metadata_ids = sorted(path.stem for path in (root / "metadata").glob("*.json"))
    midi_ids = sorted(path.stem for path in (root / "midi").glob("*.mid"))
    if metadata_ids != midi_ids:
        missing_midi = sorted(set(metadata_ids) - set(midi_ids))
        missing_metadata = sorted(set(midi_ids) - set(metadata_ids))
        raise ValueError(
            f"high-quality ID mismatch: missing_midi={missing_midi[:10]}, "
            f"missing_metadata={missing_metadata[:10]}"
        )
    if requested:
        unknown = sorted(set(requested) - set(metadata_ids))
        if unknown:
            raise ValueError(f"unknown high-quality sample IDs: {unknown}")
        sample_ids = sorted(set(requested))
    else:
        sample_ids = metadata_ids
    return sample_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=("curated", "high_quality", "aria"), default="curated"
    )
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--piece-id", action="append", dest="piece_ids")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--shard-size", type=int, default=10_000)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("--workers must be between 1 and 8")

    if args.dataset == "aria":
        root = args.root or DEFAULT_ARIA_ROOT
        output_dir = args.output_dir or (
            root / "key_estimates"
        )
        if args.piece_ids or args.dry_run or args.overwrite:
            raise ValueError(
                "--piece-id/--dry-run/--overwrite are not used for Aria; "
                "delete a deterministic part file to recompute that shard"
            )
        if args.shard_size <= 0:
            raise ValueError("--shard-size must be positive")
        run_aria(
            root=root,
            output_dir=output_dir,
            workers=args.workers,
            shard_size=args.shard_size,
            log_every=args.log_every,
        )
        return

    root = args.root or (
        DEFAULT_ROOT if args.dataset == "curated" else DEFAULT_HIGH_QUALITY_ROOT
    )
    piece_ids = (
        load_piece_ids(root, args.piece_ids)
        if args.dataset == "curated"
        else load_high_quality_ids(root, args.piece_ids)
    )
    process = process_piece if args.dataset == "curated" else process_high_quality_sample
    update = update_label if args.dataset == "curated" else update_high_quality_metadata
    started = time.monotonic()
    updated = skipped = 0
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process, str(root), piece_id): piece_id
            for piece_id in piece_ids
        }
        for completed, future in enumerate(as_completed(futures), 1):
            piece_id = futures[future]
            try:
                result = future.result()
                if args.dry_run:
                    results.append(result)
                else:
                    action = update(root, result, args.overwrite)
                    updated += action == "updated"
                    skipped += action == "skipped"
            except Exception as error:
                errors.append({
                    "piece_id": piece_id,
                    "error": f"{type(error).__name__}: {error}",
                })
            if (
                completed % args.log_every == 0
                or completed == len(piece_ids)
            ):
                elapsed = time.monotonic() - started
                print(
                    f"[{completed}/{len(piece_ids)}] updated={updated} "
                    f"skipped={skipped} errors={len(errors)} "
                    f"pieces_per_sec={completed / max(elapsed, 1e-9):.2f}",
                    flush=True,
                )

    summary = {
        "dataset": args.dataset,
        "root": str(root),
        "method": FIELD_SOURCE,
        "piece_count": len(piece_ids),
        "updated": updated,
        "skipped": skipped,
        "errors": errors,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    if args.dry_run:
        summary["results"] = sorted(results, key=lambda result: result["piece_id"])
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
