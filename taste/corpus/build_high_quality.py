#!/usr/bin/env python3
"""Build the high-quality SFT corpus: pretraining-grammar token rows of the generated set plus their generation prompts.

Per sample the builder reads ``midi/<id>.mid``, the ``key`` and
``section_sequence_timed`` of ``metadata/<id>.json`` (a section's letter is the
first character of its ``name``) and the ``tempo_bpm`` written by
``taste.representation.tempo --dataset high_quality``, and ``prompts/<id>.json`` (the four
attributes of ``input_attributes`` and the ``metaphor`` as the imagery
caption). Every field must be present and in vocabulary. ``pass0``-``pass3``
each carry one ``sha1(seed:sample_id:pass)`` draw of pitch shift and tempo
factor, ``pass4`` is clean, and the val split is 100 sample ids drawn
uniformly.
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
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]

from taste import paths
from taste.representation.tokenizer import get_tokenizer
from taste.representation.key import KEY_NAMES
from taste.corpus.build_pretrain import build_row, worker_state
from taste.corpus.high_quality_format import (
    FORMAT_VERSION,
    N_PASSES,
    PREFIX_DROP_P,
    PROMPT_FIELDS,
    ROW_SCHEMA,
    SEQ_LEN,
)

HIGH_QUALITY_ROOT = paths.data("high_quality")
ATTRIBUTES_PATH = REPO / "attributes.json"
OUT_ROOT = paths.data("corpora", "high_quality")

# the controlled attributes; "imagery" is free text and is not class-checked
ATTRIBUTE_FIELDS = ("emotion", "genre", "accompaniment", "performance")

MAX_PITCH_AUG = 5
MAX_TEMPO_AUG = 0.10
N_VAL_PIECES = 100
N_SHARDS = 16
N_VAL_SHARDS = 1
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
    # front-anchored rows stop at a whole-section boundary without being
    # truncated, so <E> follows has_end, not `not truncated`
    pa.field("has_end", pa.bool_()),
    pa.field("n_sections_emitted", pa.int32()),
    pa.field("n_plan_tokens", pa.int32()),
    pa.field("prefix_dropped", pa.bool_()),
])

_SEED = 0
_OUT_ROOT = OUT_ROOT
_HIGH_QUALITY_ROOT = HIGH_QUALITY_ROOT
_CLASSES: dict[str, list[str]] = {}


def draw(piece_id: str, pass_idx: int) -> tuple[int, float, bool]:
    """Pitch shift, tempo factor and prefix-drop flag for one (sample, pass).

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


def attribute_classes(path: Path = ATTRIBUTES_PATH) -> dict[str, list[str]]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_piece(piece_id: str) -> dict:
    """The metadata, prompt and tempo inputs of one sample; a missing or out-of-vocabulary field raises."""
    metadata = json.loads(
        (_HIGH_QUALITY_ROOT / "metadata" / f"{piece_id}.json").read_text(encoding="utf-8")
    )
    prompt = json.loads(
        (_HIGH_QUALITY_ROOT / "prompts" / f"{piece_id}.json").read_text(encoding="utf-8")
    )
    sections = metadata["section_sequence_timed"]
    if not sections:
        raise ValueError(f"{piece_id}: section_sequence_timed is empty")

    attributes = prompt["input_attributes"]
    fields: dict[str, str] = {}
    for name in ATTRIBUTE_FIELDS:
        value = attributes.get(name)
        if value not in _CLASSES[name]:
            raise ValueError(
                f"{piece_id}: {name}={value!r} is not one of {_CLASSES[name]}"
            )
        fields[name] = value
    metaphor = prompt.get("metaphor")
    if not metaphor:
        raise ValueError(f"{piece_id}: prompts/*.json has no non-empty metaphor")
    fields["imagery"] = metaphor

    return {
        "piece_id": piece_id,
        "midi_relpath": f"midi/{piece_id}.mid",
        "key": metadata["key"],
        "tempo_bpm": metadata["tempo_bpm"],
        "section_letters": [section["name"][0] for section in sections],
        "section_starts_sec": [float(section["start_sec"]) for section in sections],
        **fields,
    }


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
        self.lengths: list[int] = []
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
        self.lengths.append(len(input_ids))
        if len(self.rows) >= ROWS_PER_GROUP:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        self._row_writer.write_table(pa.table({
            "input_ids": [row["input_ids"] for row in self.rows],
            "n_tokens": [len(row["input_ids"]) for row in self.rows],
            "pass_idx": [self.pass_idx] * len(self.rows),
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
            _HIGH_QUALITY_ROOT / piece["midi_relpath"],
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
        "lengths": writer.lengths,
    }
    writer.close()
    return stats


# one bad generation batch: metaphor null on all 50, genre "blues" (out of
# vocabulary) on 14 of them
EXCLUDED_IDS = frozenset(f"sample_{index:06d}" for index in range(397, 447))


def sample_ids(high_quality_root: Path) -> list[str]:
    ids = sorted(path.stem for path in (high_quality_root / "midi").glob("*.mid"))
    return [piece_id for piece_id in ids if piece_id not in EXCLUDED_IDS]


def load_inputs(high_quality_root: Path, workers: int) -> tuple[list[dict], int]:
    """Read every sample's metadata, prompt and tempo estimate."""
    ids = sample_ids(high_quality_root)
    with mp.Pool(
        workers, initializer=_init_reader, initargs=(high_quality_root, _CLASSES)
    ) as pool:
        records = pool.map(read_piece, ids, chunksize=32)

    unsupported = sorted({record["key"] for record in records} - set(KEY_NAMES))
    if unsupported:
        raise ValueError(f"metadata keys contain unsupported spellings: {unsupported}")

    pieces: list[dict] = []
    skipped_no_tempo = 0
    for record in records:
        if record["tempo_bpm"] is None:
            skipped_no_tempo += 1
            continue
        pieces.append({**record, "tempo_bpm": float(record["tempo_bpm"])})
    return pieces, skipped_no_tempo


def blocks(order: list[dict], n_shards: int) -> list[list[dict]]:
    """Split an ordered piece list into ``n_shards`` contiguous blocks."""
    size = -(-len(order) // n_shards)
    return [order[start:start + size] for start in range(0, len(order), size)]


def pass_order(pieces: list[dict], pass_idx: int, seed: int) -> list[dict]:
    order = list(pieces)
    random.Random(seed * 1000 + pass_idx).shuffle(order)
    return order


def val_ids(pieces: list[dict], n_val: int, seed: int) -> set[str]:
    """Uniform hold-out by ``sample_id``: generated samples share no composition."""
    ids = sorted(piece["piece_id"] for piece in pieces)
    return set(random.Random(seed).sample(ids, min(n_val, len(ids))))


def assert_tempo_direction(pieces: list[dict]) -> str:
    """Verify empirically which way the tokenizer's tempo factor scales time."""
    state = worker_state()
    tokenizer = state["tokenizer"]
    probe = max(pieces[:64], key=lambda piece: piece["section_starts_sec"][-1])
    tokens = tokenizer.decode(
        tokenizer.tokenize_midi(_HIGH_QUALITY_ROOT / probe["midi_relpath"])
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


def length_stats(lengths: list[int]) -> dict:
    ordered = sorted(lengths)
    if not ordered:
        return {}
    return {
        "mean": round(sum(ordered) / len(ordered), 1),
        "p50": ordered[len(ordered) // 2],
        "p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "max": ordered[-1],
    }


def main() -> None:
    global _SEED, _OUT_ROOT, _HIGH_QUALITY_ROOT, _CLASSES
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--high-quality-root", type=Path, default=HIGH_QUALITY_ROOT)
    parser.add_argument("--attributes-path", type=Path, default=ATTRIBUTES_PATH)
    parser.add_argument("--shards", type=int, default=N_SHARDS)
    parser.add_argument("--val-shards", type=int, default=N_VAL_SHARDS)
    parser.add_argument("--n-val-pieces", type=int, default=N_VAL_PIECES)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        raise ValueError("--workers must be between 1 and 32")
    _SEED = args.seed
    _OUT_ROOT = args.out_root
    _HIGH_QUALITY_ROOT = args.high_quality_root
    _CLASSES = attribute_classes(args.attributes_path)

    pieces, skipped_no_tempo = load_inputs(args.high_quality_root, args.workers)
    if not pieces:
        raise SystemExit("no pieces to build")
    tempo_direction = assert_tempo_direction(pieces)
    print(tempo_direction, flush=True)
    print(f"samples skipped for a missing tempo: {skipped_no_tempo}", flush=True)

    held_out = val_ids(pieces, args.n_val_pieces, args.seed)
    train = [piece for piece in pieces if piece["piece_id"] not in held_out]
    val = [piece for piece in pieces if piece["piece_id"] in held_out]

    if args.out_root.exists() and any(args.out_root.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"{args.out_root} is not empty; pass --overwrite")
        shutil.rmtree(args.out_root)
    args.out_root.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[str, int, list[dict]]] = []
    for pass_idx in range(N_PASSES):
        jobs += [
            (f"pass{pass_idx}", shard, block)
            for shard, block in enumerate(
                blocks(pass_order(train, pass_idx, args.seed), args.shards)
            )
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
               "plan_tokens": 0, "dropped": 0}
        for name in split_names
    }
    lengths = {name: [] for name in split_names}
    with mp.Pool(
        args.workers, initializer=_init_worker,
        initargs=(_SEED, _OUT_ROOT, _HIGH_QUALITY_ROOT),
    ) as pool:
        for done, result in enumerate(pool.imap_unordered(worker, jobs), 1):
            entry = totals[result["split"]]
            for field in ("rows", "tokens", "truncated", "complete",
                          "plan_tokens", "dropped"):
                entry[field] += result[field]
            lengths[result["split"]] += result["lengths"]
            print(
                f"[{done}/{len(jobs)}] "
                + " ".join(f"{name}:{stats['rows']:,}rows"
                           for name, stats in totals.items()),
                flush=True,
            )

    for name, stats in totals.items():
        stats["prefix_drop_rate"] = round(stats["dropped"] / max(1, stats["rows"]), 5)
        stats["truncation_rate"] = round(stats["truncated"] / max(1, stats["rows"]), 5)
        stats["row_length"] = length_stats(lengths[name])

    train_tokens = sum(
        totals[f"pass{index}"]["tokens"] for index in range(N_PASSES)
    )
    summary = {
        "format_version": FORMAT_VERSION,
        "seq_len": SEQ_LEN,
        "n_passes": N_PASSES,
        "time_step_ms": get_tokenizer().time_step_ms,
        "seed": args.seed,
        "source": {
            "corpus_root": str(args.high_quality_root),
            "midi": "midi/<sample_id>.mid",
            "labels": "prompts/<sample_id>.json input_attributes + metaphor",
            "structure_labels": "metadata/<sample_id>.json section_sequence_timed",
            "key": "metadata/<sample_id>.json key",
            "tempo": "metadata/<sample_id>.json tempo_bpm",
            "tempo_method": (
                "beat tracking: 60 / median inter-beat interval "
                "(taste.representation.tempo.beat_bpm)"
            ),
            "attributes": str(args.attributes_path),
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
            "storage": (
                "raw strings on the row; the tier is fully labeled, so no field "
                "is ever null and a gap aborts the build"
            ),
            "prefix_drop_p": PREFIX_DROP_P,
        },
        "order": "every pass is one uniform shuffle seeded by seed*1000 + pass",
        "pieces_built": len(train),
        "pieces_skipped_no_tempo_estimate": skipped_no_tempo,
        "excluded_bad_batch": {
            "ids": "sample_000397..sample_000446",
            "count": len(EXCLUDED_IDS),
            "reason": "metaphor null on all 50; genre 'blues' (out of vocabulary) on 14",
        },
        "val": {
            "held_out_ids": len(val),
            "selection": "random.Random(seed).sample over the sorted sample ids",
            "augmented": False,
            "prefix_dropped": False,
            "crop": "front-anchored (start section 0), as in every pass",
        },
        "splits": totals,
        "total_train_tokens": train_tokens,
        "size_bytes": sum(
            path.stat().st_size for path in args.out_root.rglob("*.parquet")
        ),
        "elapsed_sec": round(time.monotonic() - started, 3),
    }
    (args.out_root / "meta.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def _init_reader(high_quality_root: Path, classes: dict[str, list[str]]) -> None:
    global _HIGH_QUALITY_ROOT, _CLASSES
    _HIGH_QUALITY_ROOT = high_quality_root
    _CLASSES = classes


def _init_worker(seed: int, out_root: Path, high_quality_root: Path) -> None:
    global _SEED, _OUT_ROOT, _HIGH_QUALITY_ROOT
    _SEED = seed
    _OUT_ROOT = out_root
    _HIGH_QUALITY_ROOT = high_quality_root
    worker_state()
    get_tokenizer()


if __name__ == "__main__":
    main()
