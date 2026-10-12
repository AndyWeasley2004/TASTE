#!/usr/bin/env python3
"""Build the pretraining corpus: five augmented passes plus a held-out val split of section-annotated token rows.

Row shape ``<KEY_*> <TEMPO_*> [plan] <piano prefix> <S|S_MID> <SEC_*> ... <T> <SEC_*> ... <E>``:
the active section state follows the start token and every ``<T>``, and the
plan is the run-length encoding of exactly the emitted window states. A row is
a contiguous run of whole sections whose start is drawn per
``sha1(seed:piece_id:epoch)``; ``epoch0`` and ``val`` are unaugmented, the other
epochs carry one pitch shift in [-5, +5] and one tempo factor in [0.90, 1.10].
The tokenizer's tempo transform maps ``t -> t * tempo_factor``, so the
augmented BPM is ``bpm / tempo_factor`` (checked by ``assert_tempo_direction``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import random
import shutil
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from taste import paths
from taste.representation.tokenizer import get_tokenizer
from taste.representation.key import KEY_NAMES, MAJOR_NAMES, MINOR_NAMES
from taste.representation.midi_features import tempo_class
from taste.representation.vocab import (
    S_MID_ID,
    key_token_id,
    plan_count_ids,
    save_vocab_meta,
    tempo_token_id,
)
from taste.corpus.sft_format import canonical_section_ids
from taste.corpus.pretrain_format import FORMAT_VERSION, N_EPOCHS, ROW_SCHEMA, SEQ_LEN

EXT_ROOT = paths.data("aria_midi", "pruned")
KEY_ROOT = EXT_ROOT / "key_estimates"
TEMPO_PATH = EXT_ROOT / "tempo_estimates" / "tempo.parquet"
LABELS_PATH = EXT_ROOT / "structure_labels" / "labels.parquet"
OUT_ROOT = paths.data("corpora", "pretrain")

MAX_PITCH_AUG = 5
MAX_TEMPO_AUG = 0.10
N_VAL_PIECES = 1_500
N_SHARDS = 64
ROWS_PER_GROUP = 256

INDEX_SCHEMA = pa.schema([
    pa.field("piece_id", pa.string()),
    pa.field("midi_relpath", pa.string()),
    pa.field("key", pa.string()),
    pa.field("key_aug", pa.string()),
    pa.field("tempo_bpm", pa.float64()),
    pa.field("tempo_bpm_aug", pa.float64()),
    pa.field("tempo_class_aug", pa.string()),
    pa.field("pitch_shift", pa.int8()),
    # float64: the recorded draw must reproduce the row bit-for-bit, and a
    # float32 round trip moves onsets across the 25 ms quantization boundary
    pa.field("tempo_factor", pa.float64()),
    pa.field("n_tokens", pa.int32()),
    pa.field("truncated", pa.bool_()),
    pa.field("has_end", pa.bool_()),
    pa.field("crop_start_section", pa.int32()),
    pa.field("crop_end_section", pa.int32()),
    pa.field("n_sections_total", pa.int32()),
    pa.field("n_sections_emitted", pa.int32()),
    pa.field("n_plan_tokens", pa.int32()),
])

MAX_PLAN_FIXPOINT_STEPS = 8

_STATE: dict = {}
_SEED = 0
_OUT_ROOT = OUT_ROOT


def worker_state() -> dict:
    """Per-process tokenizer plus the tokenizer's own augmentation transforms."""
    if not _STATE:
        tokenizer = get_tokenizer()
        _STATE["tokenizer"] = tokenizer
        # the exported wrappers clamp the output to the input length, which
        # would truncate or pad a time-scaled sequence; the inner transform is
        # the one that is length-correct for a single piece
        _STATE["pitch_aug"] = tokenizer.export_pitch_aug(
            MAX_PITCH_AUG
        ).keywords["_aug_fn"]
        _STATE["tempo_aug"] = tokenizer.export_tempo_aug(
            MAX_TEMPO_AUG, mixup=False
        ).keywords["_aug_fn"]
    return _STATE


def transpose_key(key: str, shift: int) -> str:
    for names in (MAJOR_NAMES, MINOR_NAMES):
        if key in names:
            return names[(names.index(key) + shift) % 12]
    raise ValueError(f"unsupported key: {key}")


def draw(piece_id: str, epoch: int) -> tuple[int, float, float]:
    """Pitch shift, tempo factor, and the crop-start uniform for one pass.

    Pass 0 is clean and discards the first two values, but consumes them so all
    passes share one derivation.
    """
    digest = hashlib.sha1(f"{_SEED}:{piece_id}:{epoch}".encode()).hexdigest()
    rng = random.Random(int(digest[:16], 16))
    return (
        rng.randint(-MAX_PITCH_AUG, MAX_PITCH_AUG),
        rng.uniform(1.0 - MAX_TEMPO_AUG, 1.0 + MAX_TEMPO_AUG),
        rng.random(),
    )


def pass_draw(piece_id: str, epoch: int) -> tuple[int, float, float]:
    """The pass's effective draw: pass 0 is clean but still draws its crop."""
    pitch_shift, tempo_factor, crop_uniform = draw(piece_id, epoch)
    if epoch == 0:
        return 0, 1.0, crop_uniform
    return pitch_shift, tempo_factor, crop_uniform


def section_states(starts_sec, letters, tempo_factor: float) -> dict[int, int]:
    """Window index -> ``<SEC_*>`` local ID, in augmented time, floor-quantized."""
    if len(starts_sec) != len(letters) or not letters:
        raise ValueError("structure label has no usable sections")
    ordered = sorted(
        ({"letter": letter, "start_sec": float(start) * tempo_factor}
         for letter, start in zip(letters, starts_sec)),
        key=lambda item: item["start_sec"],
    )
    markers = canonical_section_ids(ordered)
    changes = {
        max(0, int(section["start_sec"] * 1000) // 5000): marker
        for section, marker in zip(ordered, markers)
    }
    changes.setdefault(0, markers[0])
    return changes


def repeat_section_states(local: list[int], changes: dict[int, int]) -> list[int]:
    """Insert the active state after ``<S>`` and after every ``<T>``."""
    tokenizer = worker_state()["tokenizer"]
    if int(tokenizer.abs_time_step_ms) != 5000:
        raise ValueError("section-state representation requires 5-second windows")
    current = changes[0]
    window = 0
    inserted = 0
    output: list[int] = []
    for token_id in local:
        output.append(token_id)
        if token_id == tokenizer.start_id:
            current = changes.get(0, current)
            output.append(current)
            inserted += 1
        elif token_id == tokenizer.time_id:
            window += 1
            current = changes.get(window, current)
            output.append(current)
            inserted += 1
    if not inserted:
        raise ValueError("MIDI has no <S>")
    return output


def truncate_local(local: list[int], max_len: int, time_id: int) -> tuple[list[int], bool]:
    """Cut before the ``<T>`` opening the first window that does not fully fit."""
    if len(local) <= max_len:
        return local, False
    cut = max(
        (index for index, value in enumerate(local[: max_len + 1]) if value == time_id),
        default=0,
    )
    if not cut:
        raise ValueError("first 5-second window does not fit in the context")
    return local[:cut], True


def emitted_states(stream: list[int], start_id: int, time_id: int) -> list[int]:
    """The ``<SEC_*>`` state of every 5-second window, in emission order."""
    return [
        stream[position + 1]
        for position, token_id in enumerate(stream)
        if token_id in (start_id, time_id)
    ]


def section_plan(states: list[int]) -> list[int]:
    """Header plan: per run of ``states``, the ``<SEC_*>`` then its digit tokens."""
    runs: list[list[int]] = []
    for state in states:
        if runs and runs[-1][0] == state:
            runs[-1][1] += 1
        else:
            runs.append([state, 1])
    plan: list[int] = []
    for state, count in runs:
        plan.append(state)
        plan.extend(plan_count_ids(count))
    return plan


def n_section_runs(local: list[int], section_ids: frozenset[int]) -> int:
    runs = 0
    previous = None
    for token_id in local:
        if token_id in section_ids:
            if token_id != previous:
                runs += 1
            previous = token_id
    return runs


def window_positions(stream: list[int], start_id: int, time_id: int) -> list[int]:
    """Index in ``stream`` of the token opening each 5-second window."""
    return [
        position
        for position, token_id in enumerate(stream)
        if token_id in (start_id, time_id)
    ]


def section_boundaries(changes: dict[int, int], n_windows: int) -> list[int]:
    """Window index of each section start, dropping labels past the stream end."""
    return [window for window in sorted(changes) if window < n_windows]


def crop_window_span(
    boundaries: list[int], n_windows: int, start: int, end: int
) -> tuple[int, int]:
    """Half-open window range of the crop covering sections ``start..end``."""
    first = boundaries[start]
    last = boundaries[end + 1] if end + 1 < len(boundaries) else n_windows
    return first, last


def crop_cost(
    states: list[int],
    positions: list[int],
    boundaries: list[int],
    n_windows: int,
    stream_len: int,
    prefix_len: int,
    header_len: int,
    start: int,
    end: int,
) -> int:
    """Row length of the crop covering sections ``start..end``, plan included."""
    first, last = crop_window_span(boundaries, n_windows, start, end)
    end_position = positions[last] if last < n_windows else stream_len
    n_stream = prefix_len + end_position - positions[first]
    return header_len + len(section_plan(states[first:last])) + n_stream


def take_crop(
    stream: list[int],
    positions: list[int],
    n_windows: int,
    prefix_len: int,
    start_id: int,
    first_window: int,
    last_window: int,
) -> list[int]:
    """Windows ``[first_window, last_window)`` as a row stream re-based to ``<S>``.

    The ``<T>`` opening the crop becomes ``<S>``, which re-bases onsets to the
    crop start; notes that onset earlier are dropped with the tokens before it.
    """
    end_position = positions[last_window] if last_window < n_windows else len(stream)
    return (
        stream[:prefix_len]
        + [start_id]
        + stream[positions[first_window] + 1 : end_position]
    )


def piece_stream(
    midi_path: Path, starts_sec, letters, pitch_shift: int, tempo_factor: float
) -> dict:
    """The whole augmented, section-annotated stream plus its window geometry."""
    state = worker_state()
    tokenizer = state["tokenizer"]
    local = tokenizer.tokenize_midi(midi_path)
    if pitch_shift or tempo_factor != 1.0:
        augmented = state["tempo_aug"](
            state["pitch_aug"](tokenizer.decode(local), pitch_aug=pitch_shift),
            tempo_aug=tempo_factor,
        )
        # piano pitches are 21-108, so a +-5 shift can never leave the
        # library's 0-127 range; treat an <U> as a violated invariant
        if tokenizer.unk_tok in augmented:
            raise ValueError(f"pitch augmentation left <U> in {midi_path}")
        local = tokenizer.encode(augmented)

    changes = section_states(starts_sec, letters, tempo_factor)
    stream = repeat_section_states(local, changes)
    positions = window_positions(stream, tokenizer.start_id, tokenizer.time_id)
    return {
        "stream": stream,
        "positions": positions,
        "prefix_len": positions[0],
        "states": emitted_states(stream, tokenizer.start_id, tokenizer.time_id),
        "boundaries": section_boundaries(changes, len(positions)),
    }


def build_row(
    midi_path: Path,
    key: str,
    tempo_bpm: float,
    starts_sec,
    letters,
    pitch_shift: int,
    tempo_factor: float,
    crop_uniform: float,
) -> dict:
    """Build one row's ``input_ids`` plus the index record fields."""
    tokenizer = worker_state()["tokenizer"]

    key_aug = transpose_key(key, pitch_shift)
    bpm_aug = tempo_bpm / tempo_factor
    class_aug = tempo_class(bpm_aug)
    header = [key_token_id(key_aug), tempo_token_id(class_aug)]

    piece = piece_stream(midi_path, starts_sec, letters, pitch_shift, tempo_factor)
    stream = piece["stream"]
    positions = piece["positions"]
    n_windows = len(positions)
    prefix_len = piece["prefix_len"]
    states = piece["states"]
    boundaries = piece["boundaries"]
    n_sections = len(boundaries)

    def cost(start: int, end: int) -> int:
        return crop_cost(
            states, positions, boundaries, n_windows, len(stream), prefix_len,
            len(header), start, end,
        )

    # smallest start whose tail still fits; if even the last section alone is
    # over budget the fallback below handles it
    s_end = next(
        (start for start in range(n_sections) if cost(start, n_sections - 1) <= SEQ_LEN),
        n_sections - 1,
    )
    start = min(int(crop_uniform * (s_end + 1)), s_end)
    # a crop whose first window is not window 0 of the piece opens with <S_MID>
    row_start_id = tokenizer.start_id if boundaries[start] == 0 else S_MID_ID

    if cost(start, start) <= SEQ_LEN:
        end = start
        while end + 1 < n_sections and cost(start, end + 1) <= SEQ_LEN:
            end += 1
        first, last = crop_window_span(boundaries, n_windows, start, end)
        crop = take_crop(
            stream, positions, n_windows, prefix_len, row_start_id, first, last
        )
        plan = section_plan(states[first:last])
        truncated = False
    else:
        # single over-budget section: window-boundary truncation fixpoint,
        # front-anchored at that section's first window
        first = boundaries[start]
        crop = take_crop(
            stream, positions, n_windows, prefix_len, row_start_id,
            first, n_windows,
        )
        plan = section_plan(states[first:])
        for _ in range(MAX_PLAN_FIXPOINT_STEPS):
            if len(header) + len(plan) + len(crop) <= SEQ_LEN:
                break
            crop, _ = truncate_local(
                crop, SEQ_LEN - len(header) - len(plan), tokenizer.time_id
            )
            plan = section_plan(
                emitted_states(crop, row_start_id, tokenizer.time_id)
            )
        else:
            raise ValueError(f"plan/truncation fixpoint did not converge for {midi_path}")
        end = start
        truncated = True

    has_end = crop[-1] == tokenizer.end_id
    if has_end != (not truncated and end == n_sections - 1):
        raise ValueError(f"<E> does not agree with the crop end for {midi_path}")

    return {
        "input_ids": header + plan + crop,
        "n_plan_tokens": len(plan),
        "key_aug": key_aug,
        "tempo_bpm_aug": bpm_aug,
        "tempo_class_aug": class_aug,
        "truncated": truncated,
        "has_end": has_end,
        "crop_start_section": start,
        "crop_end_section": end,
        "n_sections_total": n_sections,
        "n_sections_emitted": n_section_runs(
            crop, frozenset(tokenizer.section_ids)
        ),
    }


class ShardWriter:
    """Row + index parquet writers for one shard, flushed every row group."""

    def __init__(self, row_path: Path, index_path: Path, epoch_idx: int):
        row_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        self.row_path = row_path
        self.index_path = index_path
        self.epoch_idx = epoch_idx
        self.rows: list[dict] = []
        self.index: list[dict] = []
        self.n_rows = 0
        self.n_tokens = 0
        self.n_truncated = 0
        self.n_cropped = 0
        self.n_plan_tokens = 0
        self._row_writer = pq.ParquetWriter(row_path, ROW_SCHEMA, compression="zstd")
        self._index_writer = pq.ParquetWriter(
            index_path, INDEX_SCHEMA, compression="zstd"
        )

    def add(self, input_ids: list[int], record: dict) -> None:
        self.rows.append(input_ids)
        self.index.append(record)
        self.n_rows += 1
        self.n_tokens += len(input_ids)
        self.n_truncated += bool(record["truncated"])
        self.n_cropped += not (record["crop_start_section"] == 0 and record["has_end"])
        self.n_plan_tokens += record["n_plan_tokens"]
        if len(self.rows) >= ROWS_PER_GROUP:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        self._row_writer.write_table(pa.table({
            "input_ids": self.rows,
            "n_tokens": [len(ids) for ids in self.rows],
            "epoch_idx": [self.epoch_idx] * len(self.rows),
        }, schema=ROW_SCHEMA))
        self._index_writer.write_table(pa.table(
            {name: [record[name] for record in self.index] for name in INDEX_SCHEMA.names},
            schema=INDEX_SCHEMA,
        ))
        self.rows, self.index = [], []

    def close(self) -> None:
        self.flush()
        self._row_writer.close()
        self._index_writer.close()
        if self.n_rows == 0:
            self.row_path.unlink()
            self.index_path.unlink()


def worker(job: tuple[int, list[dict]]) -> dict:
    shard, pieces = job
    tag = f"{shard:05d}"
    epochs = list(range(N_EPOCHS))
    writers = {
        epoch: ShardWriter(
            _OUT_ROOT / f"epoch{epoch}" / f"train-{tag}.parquet",
            _OUT_ROOT / "index" / f"epoch{epoch}" / f"train-{tag}.parquet",
            epoch,
        )
        for epoch in epochs
    }
    writers["val"] = ShardWriter(
        _OUT_ROOT / "val" / f"val-{tag}.parquet",
        _OUT_ROOT / "index" / "val" / f"val-{tag}.parquet",
        0,
    )

    for piece in pieces:
        if piece["is_val"]:
            passes = [("val", 0, 1.0, 0.0)]
        else:
            passes = [
                (epoch, *pass_draw(piece["piece_id"], epoch))
                for epoch in epochs
            ]
        for split, pitch_shift, tempo_factor, crop_uniform in passes:
            row = build_row(
                EXT_ROOT / piece["midi_relpath"],
                piece["key"],
                piece["tempo_bpm"],
                piece["section_starts_sec"],
                piece["section_letters"],
                pitch_shift,
                tempo_factor,
                crop_uniform,
            )
            writers[split].add(row["input_ids"], {
                "piece_id": piece["piece_id"],
                "midi_relpath": piece["midi_relpath"],
                "key": piece["key"],
                "key_aug": row["key_aug"],
                "tempo_bpm": piece["tempo_bpm"],
                "tempo_bpm_aug": row["tempo_bpm_aug"],
                "tempo_class_aug": row["tempo_class_aug"],
                "pitch_shift": pitch_shift,
                "tempo_factor": tempo_factor,
                "n_tokens": len(row["input_ids"]),
                "truncated": row["truncated"],
                "has_end": row["has_end"],
                "crop_start_section": row["crop_start_section"],
                "crop_end_section": row["crop_end_section"],
                "n_sections_total": row["n_sections_total"],
                "n_sections_emitted": row["n_sections_emitted"],
                "n_plan_tokens": row["n_plan_tokens"],
            })

    stats = {
        str(name): {
            "rows": writer.n_rows,
            "tokens": writer.n_tokens,
            "truncated": writer.n_truncated,
            "cropped": writer.n_cropped,
            "plan_tokens": writer.n_plan_tokens,
        }
        for name, writer in writers.items()
    }
    for writer in writers.values():
        writer.close()
    return {"pieces": len(pieces), "splits": stats}


def load_inputs(key_root: Path, tempo_path: Path, labels_path: Path) -> tuple[list[dict], int]:
    """Join keys, tempo, and structure labels; return per-piece build inputs."""
    key_table = ds.dataset(
        sorted(key_root.glob("part-*.parquet")), format="parquet"
    ).to_table(columns=["piece_id", "composition_id", "midi_relpath", "key"])
    unsupported = sorted(set(key_table["key"].to_pylist()) - set(KEY_NAMES))
    if unsupported:
        raise ValueError(f"key estimates contain unsupported spellings: {unsupported}")

    tempo_table = pq.read_table(tempo_path, columns=["piece_id", "tempo_bpm"])
    tempo = dict(zip(
        tempo_table["piece_id"].to_pylist(), tempo_table["tempo_bpm"].to_pylist()
    ))
    labels_table = pq.read_table(
        labels_path, columns=["piece_id", "section_letters", "section_starts_sec"]
    )
    labels = {
        piece_id: (letters, starts)
        for piece_id, letters, starts in zip(
            labels_table["piece_id"].to_pylist(),
            labels_table["section_letters"].to_pylist(),
            labels_table["section_starts_sec"].to_pylist(),
        )
    }

    pieces: list[dict] = []
    skipped = 0
    for piece_id, composition_id, relpath, key in zip(
        key_table["piece_id"].to_pylist(),
        key_table["composition_id"].to_pylist(),
        key_table["midi_relpath"].to_pylist(),
        key_table["key"].to_pylist(),
    ):
        # extraction deliberately skipped empty MIDIs and >3600 s outliers
        if piece_id not in labels:
            skipped += 1
            continue
        if piece_id not in tempo:
            raise KeyError(f"{piece_id} has structure labels but no tempo estimate")
        letters, starts = labels[piece_id]
        pieces.append({
            "piece_id": piece_id,
            "composition_id": composition_id,
            "midi_relpath": relpath,
            "key": key,
            "tempo_bpm": float(tempo[piece_id]),
            "section_letters": letters,
            "section_starts_sec": starts,
        })
    return pieces, skipped


def mark_val(pieces: list[dict], n_val: int, seed: int) -> int:
    """Hold out whole compositions until ``n_val`` pieces are covered."""
    by_composition: dict[str, list[dict]] = {}
    for piece in pieces:
        by_composition.setdefault(piece["composition_id"], []).append(piece)
    order = sorted(by_composition)
    random.Random(seed).shuffle(order)
    held: set[str] = set()
    covered = 0
    for composition_id in order:
        if covered >= n_val:
            break
        held.add(composition_id)
        covered += len(by_composition[composition_id])
    for piece in pieces:
        piece["is_val"] = piece["composition_id"] in held
    return covered


def assert_tempo_direction(pieces: list[dict]) -> str:
    """Verify empirically which way the tokenizer's tempo factor scales time."""
    state = worker_state()
    tokenizer = state["tokenizer"]
    tokens = tokenizer.decode(
        tokenizer.tokenize_midi(EXT_ROOT / pieces[0]["midi_relpath"])
    )
    factor = 1.10
    stretched = tokenizer.encode(state["tempo_aug"](tokens, tempo_aug=factor))
    base_windows = sum(1 for token in tokens if token == tokenizer.time_tok)
    aug_windows = sum(1 for value in stretched if value == tokenizer.time_id)
    if base_windows < 5:
        raise ValueError("probe piece is too short to measure the tempo direction")
    ratio = aug_windows / base_windows
    if not 1.05 < ratio < 1.15:
        raise AssertionError(
            f"tempo_aug={factor} changed duration by {ratio:.3f}x; expected ~{factor}x "
            "(the recorded BPM/section-time convention depends on this)"
        )
    return (
        f"tempo_aug scales event time by the factor (x{factor} -> {ratio:.3f}x "
        "duration); augmented bpm = bpm / factor, boundary times x factor"
    )


def main() -> None:
    global _SEED, _OUT_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--key-root", type=Path, default=KEY_ROOT)
    parser.add_argument("--tempo-path", type=Path, default=TEMPO_PATH)
    parser.add_argument("--labels-path", type=Path, default=LABELS_PATH)
    parser.add_argument("--shards", type=int, default=N_SHARDS)
    parser.add_argument("--n-val-pieces", type=int, default=N_VAL_PIECES)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 48:
        raise ValueError("--workers must be between 1 and 48")
    _SEED = args.seed
    _OUT_ROOT = args.out_root

    pieces, skipped_no_labels = load_inputs(
        args.key_root, args.tempo_path, args.labels_path
    )
    if not pieces:
        raise SystemExit("no pieces to build")
    tempo_direction = assert_tempo_direction(pieces)
    print(tempo_direction, flush=True)

    n_val_pieces = mark_val(pieces, args.n_val_pieces, args.seed)
    n_val_compositions = len({
        piece["composition_id"] for piece in pieces if piece["is_val"]
    })

    if args.out_root.exists() and any(args.out_root.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"{args.out_root} is not empty; pass --overwrite")
        shutil.rmtree(args.out_root)
    args.out_root.mkdir(parents=True, exist_ok=True)

    jobs = [
        (shard, pieces[shard::args.shards])
        for shard in range(min(args.shards, len(pieces)))
    ]
    started = time.monotonic()
    split_names = (*(str(epoch) for epoch in range(N_EPOCHS)), "val")
    totals = {
        name: {"rows": 0, "tokens": 0, "truncated": 0, "cropped": 0, "plan_tokens": 0}
        for name in split_names
    }
    done_pieces = 0
    with mp.Pool(
        args.workers,
        initializer=_init_worker,
        initargs=(_SEED, _OUT_ROOT),
    ) as pool:
        for done, result in enumerate(pool.imap_unordered(worker, jobs), 1):
            done_pieces += result["pieces"]
            for name, stats in result["splits"].items():
                for field, value in stats.items():
                    totals[name][field] += value
            print(
                f"[{done}/{len(jobs)}] pieces={done_pieces:,} "
                + " ".join(f"{name}:{stats['rows']:,}rows" for name, stats in totals.items()),
                flush=True,
            )

    summary = {
        "format_version": FORMAT_VERSION,
        "seq_len": SEQ_LEN,
        "n_epochs": N_EPOCHS,
        "time_step_ms": worker_state()["tokenizer"].time_step_ms,
        "seed": args.seed,
        "source": {
            "midi_root": str(EXT_ROOT),
            "keys": str(args.key_root),
            "tempo": str(args.tempo_path),
            "structure_labels": str(args.labels_path),
        },
        "row_stream": (
            "<KEY_*> <TEMPO_*> [plan] then the aria stream (opened by <S> for a "
            "row starting at the piece's window 0, else <S_MID>) with the active "
            "<SEC_*> repeated after the start token and every <T>; no <D>; complete pieces "
            "keep <E>"
        ),
        "section_plan": (
            "run-length encoding of exactly the emitted window states: per run "
            "the <SEC_*> token then the run's 5-second-window count as "
            "<PLAN_0>-<PLAN_9> decimal digits, most significant first"
        ),
        "segment_sampling": (
            "a row is a contiguous run of whole sections; s_end is the smallest "
            "section whose tail still fits, the start is drawn uniformly from "
            "{0..s_end} and the crop extends greedily to the most whole sections "
            "that fit; <E> iff the crop reaches the true last window; the crop's "
            "first window becomes window 0 of the row (<T> -> <S> if it is the "
            "piece's window 0, else <T> -> <S_MID>)"
        ),
        "truncation": (
            "fallback for a single section that alone exceeds the budget: cut "
            "immediately before the <T> opening the first 5-second window that "
            "does not fully fit in the budget left by the plan, then the plan is "
            "recomputed from the surviving windows until the row fits; no <E>"
        ),
        "augmentation": {
            "path": "token-level (AbsTokenizer.export_pitch_aug/export_tempo_aug)",
            "epoch0": "unaugmented",
            f"epoch1_{N_EPOCHS - 1}": "one independent draw per piece",
            "pitch_semitones": [-MAX_PITCH_AUG, MAX_PITCH_AUG],
            "tempo_factor": [1.0 - MAX_TEMPO_AUG, 1.0 + MAX_TEMPO_AUG],
            "tempo_direction": tempo_direction,
            "draw_rng": (
                "sha1(seed:piece_id:epoch) -> random.Random; per pass: pitch "
                "shift, tempo factor, then the crop-start uniform"
            ),
        },
        "pieces_built": done_pieces,
        "pieces_skipped_no_structure_labels": skipped_no_labels,
        "val": {
            "held_out_compositions": n_val_compositions,
            "held_out_pieces": n_val_pieces,
            "augmented": False,
            "crop": "front-anchored (start section 0)",
        },
        "splits": totals,
        "size_bytes": sum(
            path.stat().st_size for path in args.out_root.rglob("*.parquet")
        ),
        "elapsed_sec": round(time.monotonic() - started, 3),
    }
    save_vocab_meta(args.out_root)
    (args.out_root / "meta.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def _init_worker(seed: int, out_root: Path) -> None:
    global _SEED, _OUT_ROOT
    _SEED = seed
    _OUT_ROOT = out_root
    worker_state()


if __name__ == "__main__":
    main()
