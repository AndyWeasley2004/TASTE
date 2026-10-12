#!/usr/bin/env python3
"""Build the large-public SFT corpus: pretraining-grammar token rows of the deduped Aria subset plus their text prompts.

Rows reuse ``build_pretrain.build_row`` with the crop pinned to section 0, so
every row opens with ``<S>`` and keeps ``<E>`` iff the whole piece fits. Passes
``pass0``-``pass4`` each carry one ``sha1(seed:piece_id:pass)`` draw of pitch
shift and tempo factor and ``pass5`` is clean; the third value of the same
stream is the prefix-drop Bernoulli. ``pass0`` is packed in stratum order
(1 = emotion only, 3 = emotion + accompaniment + performance, 2 = the rest),
the other passes are shuffled. The val split holds out whole compositions,
unaugmented and never prefix-dropped.
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
from taste.representation.key import KEY_NAMES
from taste.corpus.build_pretrain import build_row, mark_val, worker_state
from taste.corpus.large_public_format import (
    FORMAT_VERSION,
    N_PASSES,
    PREFIX_DROP_P,
    PROMPT_FIELDS,
    ROW_SCHEMA,
    SEQ_LEN,
)

EXT_ROOT = paths.data("aria_midi", "deduped")
KEY_ROOT = EXT_ROOT / "key_estimates"
TEXT_LABELS_PATH = EXT_ROOT / "large_public_labels" / "labels.parquet"
LABELS_PATH = EXT_ROOT / "structure_labels" / "labels.parquet"
OUT_ROOT = paths.data("corpora", "large_public")

MAX_PITCH_AUG = 5
MAX_TEMPO_AUG = 0.10
N_VAL_PIECES = 1_500
N_SHARDS = 64
N_VAL_SHARDS = 8
ROWS_PER_GROUP = 256
KEY_SAMPLE = 500
KEY_DISAGREEMENT_LIMIT = 0.01

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
    # front-anchored rows stop at a whole-section boundary without being
    # truncated, so <E> follows has_end, not `not truncated`
    pa.field("has_end", pa.bool_()),
    pa.field("n_sections_emitted", pa.int32()),
    pa.field("n_plan_tokens", pa.int32()),
    pa.field("stratum", pa.int8()),
    pa.field("prefix_dropped", pa.bool_()),
])

_SEED = 0
_OUT_ROOT = OUT_ROOT


def draw(piece_id: str, pass_idx: int) -> tuple[int, float, bool]:
    """Pitch shift, tempo factor and prefix-drop flag for one (piece, pass).

    Drawn for every pass, the clean last one included, so the prefix-drop flag
    is always the third value of the stream.
    """
    digest = hashlib.sha1(f"{_SEED}:{piece_id}:{pass_idx}".encode()).hexdigest()
    rng = random.Random(int(digest[:16], 16))
    pitch = rng.randint(-MAX_PITCH_AUG, MAX_PITCH_AUG)
    factor = rng.uniform(1.0 - MAX_TEMPO_AUG, 1.0 + MAX_TEMPO_AUG)
    dropped = rng.random() < PREFIX_DROP_P
    return pitch, factor, dropped


def pass_draw(piece_id: str, pass_idx: int) -> tuple[int, float, bool]:
    pitch, factor, dropped = draw(piece_id, pass_idx)
    if pass_idx == N_PASSES - 1:
        return 0, 1.0, dropped
    return pitch, factor, dropped


def stratum_of(piece: dict) -> int:
    has_emotion = piece["emotion"] is not None
    has_accompaniment = piece["accompaniment"] is not None
    has_performance = piece["performance"] is not None
    if has_emotion and not has_accompaniment and not has_performance:
        return 1
    if has_emotion and has_accompaniment and has_performance:
        return 3
    return 2


class ShardWriter:
    """Row + index parquet writers for one shard, flushed every row group."""

    def __init__(self, row_path: Path, index_path: Path, pass_idx: int):
        row_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        self.row_path = row_path
        self.index_path = index_path
        self.pass_idx = pass_idx
        self.rows: list[dict] = []
        self.index: list[dict] = []
        self.n_rows = 0
        self.n_tokens = 0
        self.n_truncated = 0
        self.n_complete = 0
        self.n_plan_tokens = 0
        self.n_dropped = 0
        self.strata = {1: 0, 2: 0, 3: 0}
        self._row_writer = pq.ParquetWriter(row_path, ROW_SCHEMA, compression="zstd")
        self._index_writer = pq.ParquetWriter(
            index_path, INDEX_SCHEMA, compression="zstd"
        )

    def add(self, input_ids: list[int], prompt: dict, record: dict) -> None:
        self.rows.append({"input_ids": input_ids, **prompt})
        self.index.append(record)
        self.n_rows += 1
        self.n_tokens += len(input_ids)
        self.n_truncated += bool(record["truncated"])
        self.n_complete += bool(record["has_end"])
        self.n_plan_tokens += record["n_plan_tokens"]
        self.n_dropped += bool(record["prefix_dropped"])
        self.strata[record["stratum"]] += 1
        if len(self.rows) >= ROWS_PER_GROUP:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        self._row_writer.write_table(pa.table({
            "input_ids": [row["input_ids"] for row in self.rows],
            "n_tokens": [len(row["input_ids"]) for row in self.rows],
            "pass_idx": [self.pass_idx] * len(self.rows),
            "stratum": [record["stratum"] for record in self.index],
            "prefix_dropped": [record["prefix_dropped"] for record in self.index],
            **{name: [row[name] for row in self.rows] for name in PROMPT_FIELDS},
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


def worker(job: tuple[str, int, list[dict]]) -> dict:
    """Build one (split, shard) block, in the order the block was handed over."""
    split, shard, pieces = job
    is_val = split == "val"
    pass_idx = 0 if is_val else int(split[-1])
    prefix = "val" if is_val else "train"
    tag = f"{shard:05d}"
    writer = ShardWriter(
        _OUT_ROOT / split / f"{prefix}-{tag}.parquet",
        _OUT_ROOT / "index" / split / f"{prefix}-{tag}.parquet",
        pass_idx,
    )

    for piece in pieces:
        if is_val:
            pitch_shift, tempo_factor, dropped = 0, 1.0, False
        else:
            pitch_shift, tempo_factor, dropped = pass_draw(
                piece["piece_id"], pass_idx
            )
        row = build_row(
            EXT_ROOT / piece["midi_relpath"],
            piece["key"],
            piece["tempo_bpm"],
            piece["section_starts_sec"],
            piece["section_letters"],
            pitch_shift,
            tempo_factor,
            0.0,  # front-anchored: crop starts at section 0 for every row
        )
        writer.add(
            row["input_ids"],
            {name: piece[name] for name in PROMPT_FIELDS},
            {
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
                "n_sections_emitted": row["n_sections_emitted"],
                "n_plan_tokens": row["n_plan_tokens"],
                "stratum": piece["stratum"],
                "prefix_dropped": dropped,
            },
        )

    stats = {
        "split": split,
        "rows": writer.n_rows,
        "tokens": writer.n_tokens,
        "truncated": writer.n_truncated,
        "complete": writer.n_complete,
        "plan_tokens": writer.n_plan_tokens,
        "dropped": writer.n_dropped,
        "strata": writer.strata,
    }
    writer.close()
    return stats


def check_key_agreement(key: dict[str, str], label_key: dict[str, str], seed: int) -> float:
    """The text-label table carries its own ``key``; it must agree with the key estimates."""
    shared = sorted(set(key) & set(label_key))
    sample = random.Random(seed).sample(shared, min(KEY_SAMPLE, len(shared)))
    disagree = sum(1 for piece_id in sample if key[piece_id] != label_key[piece_id])
    rate = disagree / max(1, len(sample))
    if rate > KEY_DISAGREEMENT_LIMIT:
        raise SystemExit(
            f"key estimates disagree with large_public_labels on {disagree}/{len(sample)} "
            f"sampled pieces ({rate:.3%} > {KEY_DISAGREEMENT_LIMIT:.1%})"
        )
    return rate


def load_inputs(
    key_root: Path, text_labels_path: Path, labels_path: Path, seed: int
) -> tuple[list[dict], int, float]:
    """Join keys, text labels/tempo, and structure labels."""
    key_table = ds.dataset(
        sorted(key_root.glob("part-*.parquet")), format="parquet"
    ).to_table(columns=["piece_id", "composition_id", "midi_relpath", "key"])
    unsupported = sorted(set(key_table["key"].to_pylist()) - set(KEY_NAMES))
    if unsupported:
        raise ValueError(f"key estimates contain unsupported spellings: {unsupported}")

    text_table = pq.read_table(
        text_labels_path, columns=["piece_id", "key", "tempo_bpm", *PROMPT_FIELDS]
    )
    text_labels = {
        record["piece_id"]: record for record in text_table.to_pylist()
    }
    key_rate = check_key_agreement(
        dict(zip(key_table["piece_id"].to_pylist(), key_table["key"].to_pylist())),
        {piece_id: record["key"] for piece_id, record in text_labels.items()},
        seed,
    )

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
        record = text_labels[piece_id]
        letters, starts = labels[piece_id]
        piece = {
            "piece_id": piece_id,
            "composition_id": composition_id,
            "midi_relpath": relpath,
            "key": key,
            "tempo_bpm": float(record["tempo_bpm"]),
            "section_letters": letters,
            "section_starts_sec": starts,
            **{name: (record[name] or None) for name in PROMPT_FIELDS},
        }
        piece["stratum"] = stratum_of(piece)
        pieces.append(piece)
    return pieces, skipped, key_rate


def blocks(order: list[dict], n_shards: int) -> list[list[dict]]:
    """Split an ordered piece list into ``n_shards`` contiguous blocks."""
    size = -(-len(order) // n_shards)
    return [order[start:start + size] for start in range(0, len(order), size)]


def pass_order(pieces: list[dict], pass_idx: int, seed: int) -> list[dict]:
    if pass_idx == 0:
        return sorted(pieces, key=lambda piece: (piece["stratum"], piece["piece_id"]))
    order = list(pieces)
    random.Random(seed * 1000 + pass_idx).shuffle(order)
    return order


def stratum_layout(order: list[dict], size: int) -> dict:
    """Row and shard boundaries of each stratum in the pass-0 packing."""
    layout: dict[str, dict] = {}
    start = 0
    for stratum in (1, 2, 3):
        count = sum(1 for piece in order if piece["stratum"] == stratum)
        end = start + count
        layout[f"stratum{stratum}"] = {
            "rows": count,
            "row_start": start,
            "row_end": end,
            "shard_start": start // size,
            "shard_end": max(start, end - 1) // size,
        }
        start = end
    return layout


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
    parser.add_argument("--text-labels-path", type=Path, default=TEXT_LABELS_PATH)
    parser.add_argument("--labels-path", type=Path, default=LABELS_PATH)
    parser.add_argument("--shards", type=int, default=N_SHARDS)
    parser.add_argument("--val-shards", type=int, default=N_VAL_SHARDS)
    parser.add_argument("--n-val-pieces", type=int, default=N_VAL_PIECES)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        raise ValueError("--workers must be between 1 and 32")
    _SEED = args.seed
    _OUT_ROOT = args.out_root

    pieces, skipped_no_labels, key_rate = load_inputs(
        args.key_root, args.text_labels_path, args.labels_path, args.seed
    )
    if not pieces:
        raise SystemExit("no pieces to build")
    tempo_direction = assert_tempo_direction(pieces)
    print(tempo_direction, flush=True)
    print(f"key disagreement with large_public_labels: {key_rate:.3%}", flush=True)

    n_val_pieces = mark_val(pieces, args.n_val_pieces, args.seed)
    n_val_compositions = len({
        piece["composition_id"] for piece in pieces if piece["is_val"]
    })
    train = [piece for piece in pieces if not piece["is_val"]]
    val = [piece for piece in pieces if piece["is_val"]]

    if args.out_root.exists() and any(args.out_root.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"{args.out_root} is not empty; pass --overwrite")
        shutil.rmtree(args.out_root)
    args.out_root.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[str, int, list[dict]]] = []
    orders = {
        pass_idx: pass_order(train, pass_idx, args.seed)
        for pass_idx in range(N_PASSES)
    }
    for pass_idx, order in orders.items():
        jobs += [
            (f"pass{pass_idx}", shard, block)
            for shard, block in enumerate(blocks(order, args.shards))
        ]
    jobs += [
        ("val", shard, block)
        for shard, block in enumerate(
            blocks(sorted(val, key=lambda piece: piece["piece_id"]), args.val_shards)
        )
    ]

    started = time.monotonic()
    split_names = (*(f"pass{index}" for index in range(N_PASSES)), "val")
    totals = {
        name: {"rows": 0, "tokens": 0, "truncated": 0, "complete": 0,
               "plan_tokens": 0, "dropped": 0, "strata": {1: 0, 2: 0, 3: 0}}
        for name in split_names
    }
    with mp.Pool(
        args.workers, initializer=_init_worker, initargs=(_SEED, _OUT_ROOT)
    ) as pool:
        for done, result in enumerate(pool.imap_unordered(worker, jobs), 1):
            entry = totals[result["split"]]
            for field in ("rows", "tokens", "truncated", "complete",
                          "plan_tokens", "dropped"):
                entry[field] += result[field]
            for stratum, count in result["strata"].items():
                entry["strata"][stratum] += count
            print(
                f"[{done}/{len(jobs)}] "
                + " ".join(f"{name}:{stats['rows']:,}rows"
                           for name, stats in totals.items()),
                flush=True,
            )

    for stats in totals.values():
        stats["strata"] = {str(key): value for key, value in stats["strata"].items()}
        stats["prefix_drop_rate"] = round(
            stats["dropped"] / max(1, stats["rows"]), 5
        )

    summary = {
        "format_version": FORMAT_VERSION,
        "seq_len": SEQ_LEN,
        "n_passes": N_PASSES,
        "time_step_ms": get_tokenizer().time_step_ms,
        "seed": args.seed,
        "source": {
            "midi_root": str(EXT_ROOT),
            "keys": str(args.key_root),
            "text_labels": str(args.text_labels_path),
            "structure_labels": str(args.labels_path),
            "key_disagreement_with_text_labels": key_rate,
        },
        "row_stream": (
            "<KEY_*> <TEMPO_*> [plan] then the aria stream, always opened by "
            "<S> (never <S_MID>), with the active <SEC_*> repeated after <S> "
            "and every <T>; no <D>; complete pieces keep <E>"
        ),
        "segment_sampling": (
            "front-anchored: every row crops from section 0 and extends "
            "greedily to the most whole sections that fit; <E> iff the whole "
            "piece fits"
        ),
        "section_plan": (
            "run-length encoding of exactly the emitted window states: per run "
            "the <SEC_*> token then the run's 5-second-window count as "
            "<PLAN_0>-<PLAN_9> decimal digits, most significant first"
        ),
        "truncation": (
            "fallback for a first section that alone exceeds the budget: cut "
            "immediately before the <T> opening the first 5-second window "
            "that does not fully fit in the budget left by the plan, then the "
            "plan is recomputed from the surviving windows until the row fits; "
            "no <E>"
        ),
        "augmentation": {
            "path": "token-level (AbsTokenizer.export_pitch_aug/export_tempo_aug)",
            f"pass0_{N_PASSES - 2}": "one independent draw per piece and pass",
            f"pass{N_PASSES - 1}": "unaugmented (clean pass last)",
            "pitch_semitones": [-MAX_PITCH_AUG, MAX_PITCH_AUG],
            "tempo_factor": [1.0 - MAX_TEMPO_AUG, 1.0 + MAX_TEMPO_AUG],
            "velocity": "not augmented",
            "tempo_direction": tempo_direction,
            "draw_rng": (
                "sha1(seed:piece_id:pass) -> random.Random; pitch, tempo factor, "
                "then the prefix-drop Bernoulli, drawn for every pass"
            ),
        },
        "prompt": {
            "fields": list(PROMPT_FIELDS),
            "storage": "raw strings on the row; empty in the label table -> null",
            "prefix_drop_p": PREFIX_DROP_P,
        },
        "strata": {
            "definition": {
                "1": "emotion present, accompaniment and performance absent",
                "3": "emotion, accompaniment and performance all present",
                "2": "everything else",
            },
            "pass0_order": "S1 -> S2 -> S3, piece_id ascending within a stratum",
            f"pass1_{N_PASSES - 1}_order": "uniform shuffle seeded by seed*1000 + pass",
            "pass0_layout": stratum_layout(
                orders[0], -(-len(orders[0]) // args.shards)
            ),
        },
        "pieces_built": len(train),
        "pieces_skipped_no_structure_labels": skipped_no_labels,
        "val": {
            "held_out_compositions": n_val_compositions,
            "held_out_pieces": n_val_pieces,
            "augmented": False,
            "prefix_dropped": False,
            "crop": "front-anchored (start section 0), as in every pass",
        },
        "splits": totals,
        "size_bytes": sum(
            path.stat().st_size for path in args.out_root.rglob("*.parquet")
        ),
        "elapsed_sec": round(time.monotonic() - started, 3),
    }
    (args.out_root / "meta.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def _init_worker(seed: int, out_root: Path) -> None:
    global _SEED, _OUT_ROOT
    _SEED = seed
    _OUT_ROOT = out_root
    worker_state()
    get_tokenizer()


if __name__ == "__main__":
    main()
