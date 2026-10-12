#!/usr/bin/env python3
"""Pre-launch gate for the curated SFT corpus; exits non-zero on any failure.

Runs the pretraining checks on the token rows (grammar, section plan, ``<E>``,
split accounting, sampled re-derivation, section states, tempo, key-histogram
shift) plus the SFT checks: prompt columns and key against ``label.json``,
prefix-drop rate per pass, a clean last pass, front-anchored rows, val rows
neither augmented nor prefix-dropped, and a val set equal to the ``val`` pieces
of ``manifests/split.parquet``. Shards are streamed in name order.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import random
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from taste.representation.tokenizer import get_tokenizer
from taste.representation.midi_features import tempo_class
from taste.representation.vocab import (
    KEY_TO_ID,
    MODEL_VOCAB_SIZE,
    PLAN_BASE,
    PLAN_IDS,
    S_MID_ID,
    TEMPO_TO_ID,
    UNUSED_IDS,
)
from taste.corpus.build_pretrain import (
    build_row,
    emitted_states,
    n_section_runs,
    section_plan,
    section_states,
)
import taste.corpus.build_curated as builder
from taste.corpus.build_curated import (
    CURATED_ROOT,
    OUT_ROOT,
    SPLIT_PATH,
    load_split,
    read_piece,
)
from taste.corpus.curated_format import (
    FORMAT_VERSION,
    N_PASSES,
    PREFIX_DROP_P,
    PROMPT_FIELDS,
    SEQ_LEN,
)

SPLITS = (*(f"pass{index}" for index in range(N_PASSES)), "val")
# the clean pass comes last, so it is the transposition baseline
CLEAN_SPLIT = f"pass{N_PASSES - 1}"
SAMPLE_ROWS = 200
DROP_RATE_BAND = (0.09, 0.11)

_FAILURES: list[str] = []
# the exhaustive per-row checks are O(total tokens) in Python, so shards are
# verified in a pool; the annotation table is read once and inherited by fork
_LABELS: dict = {}


def check(condition: bool, message: str) -> bool:
    if not condition:
        _FAILURES.append(message)
    return bool(condition)


def shard_paths(root: Path, split: str) -> list[tuple[Path, Path]]:
    prefix = "val" if split == "val" else "train"
    rows = sorted((root / split).glob(f"{prefix}-*.parquet"))
    index = sorted((root / "index" / split).glob(f"{prefix}-*.parquet"))
    if not rows:
        raise SystemExit(f"missing {split} shards under {root}")
    if [path.name for path in rows] != [path.name for path in index]:
        raise SystemExit(f"{split}: row and index shards do not correspond")
    return list(zip(rows, index))


def check_row_grammar(
    split: str, rows: list[dict], index: list[dict], key_histogram: Counter
) -> None:
    tokenizer = get_tokenizer()
    section_ids = frozenset(tokenizer.section_ids)
    # canonical_section_ids maps the earliest section to the first <SEC_*>, so a
    # front-anchored row always opens on this state
    section_0 = tokenizer.section_ids[0]
    plan_ids = frozenset(PLAN_IDS)
    unused = frozenset(UNUSED_IDS)
    key_ids = frozenset(KEY_TO_ID.values())
    tempo_ids = frozenset(TEMPO_TO_ID.values())
    expected_pass = 0 if split == "val" else int(split[-1])

    for row, record in zip(rows, index):
        ids = row["input_ids"]
        tag = f"{split}/{record['piece_id']}"
        if not check(row["n_tokens"] == len(ids), f"{tag}: n_tokens != len(input_ids)"):
            continue
        if not check(len(ids) <= SEQ_LEN, f"{tag}: {len(ids)} tokens over the context"):
            continue
        check(row["pass_idx"] == expected_pass, f"{tag}: wrong pass_idx")
        check(
            row["prefix_dropped"] == record["prefix_dropped"],
            f"{tag}: row/index prefix_dropped differ",
        )
        bad = [value for value in ids if not 0 <= value < MODEL_VOCAB_SIZE]
        if not check(not bad, f"{tag}: ids outside the vocabulary: {bad[:4]}"):
            continue
        if not check(
            not any(value in unused for value in ids), f"{tag}: uses a padding-range id"
        ):
            continue

        if not check(ids[0] in key_ids and ids[1] in tempo_ids, f"{tag}: bad header"):
            continue
        check(ids[0] == KEY_TO_ID[record["key_aug"]], f"{tag}: key token != index key_aug")
        check(
            ids[1] == TEMPO_TO_ID[record["tempo_class_aug"]],
            f"{tag}: tempo token != index tempo_class_aug",
        )
        key_histogram[ids[0]] += 1

        cursor = 2
        plan_ok = True
        while cursor < len(ids) and ids[cursor] in section_ids:
            cursor += 1
            digits = ""
            while cursor < len(ids) and ids[cursor] in plan_ids:
                digits += str(ids[cursor] - PLAN_BASE)
                cursor += 1
            if not digits or digits[0] == "0":
                plan_ok = check(False, f"{tag}: plan run has no valid digit count")
                break
        if not plan_ok:
            continue
        plan = ids[2:cursor]
        check(
            len(plan) == record["n_plan_tokens"],
            f"{tag}: n_plan_tokens != the plan in the row",
        )

        prefix_start = cursor
        while cursor < len(ids) and ids[cursor] == tokenizer.piano_prefix_id:
            cursor += 1
        if not check(
            cursor > prefix_start
            and cursor + 1 < len(ids)
            and ids[cursor] == tokenizer.start_id,
            f"{tag}: header is not [key, tempo, plan, piano prefix..., <S>]",
        ):
            continue
        check(S_MID_ID not in ids, f"{tag}: row is not front-anchored (<S_MID>)")
        check(
            ids[cursor + 1] == section_0,
            f"{tag}: first in-stream state is not the piece's section-0 letter",
        )

        stream = ids[prefix_start:]
        check(
            not any(value in plan_ids for value in stream),
            f"{tag}: <PLAN_*> digit outside the header",
        )

        grammar_ok = True
        for position, value in enumerate(stream):
            after_state = position > 0 and stream[position - 1] in {
                tokenizer.start_id, tokenizer.time_id
            }
            if value in section_ids and not after_state:
                grammar_ok = check(False, f"{tag}: <SEC_*> at {position} not after <S>/<T>")
                break
            if value == tokenizer.time_id:
                if position + 1 >= len(stream) or stream[position + 1] not in section_ids:
                    grammar_ok = check(False, f"{tag}: <T> at {position} without a state")
                    break
        if not grammar_ok:
            continue

        check(
            plan == section_plan(
                emitted_states(stream, tokenizer.start_id, tokenizer.time_id)
            ),
            f"{tag}: plan is not the run-length encoding of the emitted states",
        )

        check(tokenizer.end_approach_id not in ids, f"{tag}: contains <D>")
        has_end = ids[-1] == tokenizer.end_id
        check(
            has_end == bool(record["has_end"]),
            f"{tag}: <E>={has_end} but has_end={record['has_end']}",
        )
        check(
            not (has_end and record["truncated"]),
            f"{tag}: truncated row keeps <E>",
        )
        check(
            ids.count(tokenizer.end_id) == (1 if has_end else 0),
            f"{tag}: unexpected <E> count",
        )
        check(
            n_section_runs(stream, section_ids) == record["n_sections_emitted"],
            f"{tag}: n_sections_emitted mismatch",
        )
        if split in (CLEAN_SPLIT, "val"):
            check(
                record["pitch_shift"] == 0 and record["tempo_factor"] == 1.0,
                f"{tag}: unaugmented split carries a draw",
            )
            check(
                record["tempo_class_aug"] == tempo_class(record["tempo_bpm"]),
                f"{tag}: unaugmented split carries an augmented tempo class",
            )
        if split == "val":
            check(not record["prefix_dropped"], f"{tag}: val row is prefix-dropped")


def check_prompts(split: str, rows: list[dict], index: list[dict]) -> None:
    for row, record in zip(rows, index):
        tag = f"{split}/{record['piece_id']}"
        expected = _LABELS[record["piece_id"]]
        for name in PROMPT_FIELDS:
            if row[name] != expected[name]:
                check(False, f"{tag}: prompt field {name} != label.json")
                break
        check(
            record["key"] == expected["key"],
            f"{tag}: index key != label.json accepted_fields.key",
        )


def verify_shard(job: tuple[str, Path, Path, int, frozenset]) -> dict:
    """All exhaustive per-row checks for one shard, plus its sampled rows."""
    global _FAILURES
    split, row_path, index_path, offset, wanted = job
    _FAILURES = []
    rows = pq.read_table(row_path).to_pylist()
    index = pq.read_table(index_path).to_pylist()
    if len(rows) != len(index):
        raise SystemExit(f"{row_path.name}: rows vs index length mismatch")

    histogram: Counter = Counter()
    check_row_grammar(split, rows, index, histogram)
    check_prompts(split, rows, index)

    sample: list[tuple[dict, dict]] = []
    ids: set[str] = set()
    lengths: list[int] = []
    n_tokens = n_truncated = n_complete = n_plan = n_dropped = n_transposed = 0
    for position, (row, record) in enumerate(zip(rows, index)):
        if offset + position in wanted:
            sample.append((row, record))
        n_tokens += row["n_tokens"]
        lengths.append(row["n_tokens"])
        n_truncated += bool(record["truncated"])
        n_complete += bool(record["has_end"])
        n_plan += record["n_plan_tokens"]
        n_dropped += bool(record["prefix_dropped"])
        n_transposed += record["key_aug"] != record["key"]
        ids.add(record["piece_id"])
    return {
        "rows": len(rows),
        "tokens": n_tokens,
        "truncated": n_truncated,
        "complete": n_complete,
        "plan_tokens": n_plan,
        "dropped": n_dropped,
        "transposed": n_transposed,
        "lengths": lengths,
        "ids": ids,
        "histogram": histogram,
        "sample": sample,
        "failures": _FAILURES,
    }


def check_tempo_recompute(split: str, sample: list[tuple[dict, dict]]) -> None:
    for _, record in sample:
        expected = record["tempo_bpm"] / record["tempo_factor"]
        tag = f"{split}/{record['piece_id']}"
        check(
            abs(record["tempo_bpm_aug"] - expected) < 1e-9,
            f"{tag}: tempo_bpm_aug != tempo_bpm / tempo_factor",
        )
        check(
            record["tempo_class_aug"] == tempo_class(record["tempo_bpm_aug"]),
            f"{tag}: tempo_class_aug does not match the augmented BPM",
        )


def check_rederivation(split: str, sample: list[tuple[dict, dict]], root: Path) -> None:
    for row, record in sample:
        tag = f"{split}/{record['piece_id']}"
        pitch, factor = record["pitch_shift"], record["tempo_factor"]
        if split != "val":
            expected = builder.pass_draw(record["piece_id"], int(split[-1]))
            check(
                (pitch, factor, record["prefix_dropped"]) == expected,
                f"{tag}: recorded draw/prefix_dropped is not the seeded draw",
            )
        label = _LABELS[record["piece_id"]]
        rebuilt = build_row(
            root / record["midi_relpath"],
            record["key"],
            record["tempo_bpm"],
            label["section_starts_sec"],
            label["section_letters"],
            pitch,
            factor,
            0.0,
        )
        check(
            rebuilt["input_ids"] == row["input_ids"],
            f"{tag}: re-derived row differs from the stored row",
        )


def check_section_states(split: str, sample: list[tuple[dict, dict]]) -> None:
    tokenizer = get_tokenizer()
    for row, record in sample:
        tag = f"{split}/{record['piece_id']}"
        label = _LABELS[record["piece_id"]]
        changes = section_states(
            label["section_starts_sec"], label["section_letters"], record["tempo_factor"]
        )
        current = changes[0]
        window = 0
        ids = row["input_ids"]
        for cursor, value in enumerate(ids[:-1]):
            if value == tokenizer.start_id:
                current = changes.get(0, current)
            elif value == tokenizer.time_id:
                window += 1
                current = changes.get(window, current)
            else:
                continue
            if ids[cursor + 1] != current:
                check(False, f"{tag}: state at window {window} is not the label state")
                break


def check_key_shift(histograms: dict[str, Counter], transposed: dict[str, float]) -> None:
    base = histograms[CLEAN_SPLIT]
    base_total = sum(base.values())
    for index in range(N_PASSES - 1):
        split = f"pass{index}"
        other = histograms[split]
        check(
            sum(other.values()) == base_total,
            f"{split}: row count differs from {CLEAN_SPLIT}",
        )
        overlap = sum(min(base[key], other[key]) for key in set(base) | set(other))
        share = overlap / max(1, base_total)
        check(
            share < 0.98,
            f"{split}: key histogram is indistinguishable from {CLEAN_SPLIT} "
            f"(overlap {share:.3f}); transposition did not move mass",
        )
        # a uniform +-5 semitone draw leaves 1/11 of pieces at the original key;
        # 5 binomial sigmas keeps the band honest at any corpus size
        expected = 10 / 11
        tolerance = 5 * math.sqrt(expected * (1 - expected) / max(1, base_total))
        check(
            abs(transposed[split] - expected) < tolerance,
            f"{split}: {transposed[split]:.3f} of rows are transposed, expected "
            f"{expected:.3f} +- {tolerance:.3f}",
        )
        print(
            f"  {split}: key-histogram overlap with {CLEAN_SPLIT} = {share:.3f}, "
            f"transposed fraction = {transposed[split]:.3f}"
        )


def length_stats(lengths: list[int]) -> dict:
    ordered = sorted(lengths)
    return {
        "mean": round(sum(ordered) / len(ordered), 1),
        "p50": ordered[len(ordered) // 2],
        "p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "max": ordered[-1],
    }


def load_labels(root: Path, workers: int) -> dict:
    piece_ids = sorted(path.name for path in (root / "pieces").iterdir())
    builder._CURATED_ROOT = root
    with mp.Pool(
        workers, initializer=builder._init_reader, initargs=(root,)
    ) as pool:
        records = pool.map(read_piece, piece_ids, chunksize=64)
    return {record["piece_id"]: record for record in records}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=OUT_ROOT)
    parser.add_argument("--curated-root", type=Path, default=CURATED_ROOT)
    parser.add_argument("--split-path", type=Path, default=SPLIT_PATH)
    parser.add_argument("--sample-rows", type=int, default=SAMPLE_ROWS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("--workers must be between 1 and 8")

    meta = json.loads((args.root / "meta.json").read_text(encoding="utf-8"))
    check(
        meta["format_version"] == FORMAT_VERSION,
        f"meta.json format_version {meta['format_version']!r} != {FORMAT_VERSION!r}",
    )
    check(meta["seq_len"] == SEQ_LEN, f"meta.json seq_len {meta['seq_len']} != {SEQ_LEN}")
    check(meta["n_passes"] == N_PASSES, f"meta.json n_passes {meta['n_passes']} != {N_PASSES}")
    check(
        meta["time_step_ms"] == get_tokenizer().time_step_ms,
        f"meta.json time_step_ms {meta['time_step_ms']} != the tokenizer's "
        f"{get_tokenizer().time_step_ms}",
    )
    builder._SEED = meta["seed"]

    global _LABELS
    _LABELS = load_labels(args.curated_root, args.workers)

    rng = random.Random(args.seed)
    histograms: dict[str, Counter] = {}
    transposed: dict[str, float] = {}
    piece_ids: dict[str, set[str]] = {}
    summary: dict[str, dict] = {}

    pool = mp.Pool(args.workers)
    for split in SPLITS:
        shards = shard_paths(args.root, split)
        counts = [pq.ParquetFile(index_path).metadata.num_rows
                  for _, index_path in shards]
        n_rows = sum(counts)
        wanted = frozenset(
            rng.sample(range(n_rows), min(args.sample_rows, n_rows))
        )
        offsets: list[int] = []
        running = 0
        for count in counts:
            offsets.append(running)
            running += count

        results = pool.map(verify_shard, [
            (split, row_path, index_path, offset, wanted)
            for (row_path, index_path), offset in zip(shards, offsets)
        ])

        histogram: Counter = Counter()
        ids: set[str] = set()
        sample: list[tuple[dict, dict]] = []
        lengths: list[int] = []
        n_tokens = n_truncated = n_complete = n_plan = n_dropped = n_transposed = 0
        for result in results:
            _FAILURES.extend(result["failures"])
            histogram.update(result["histogram"])
            ids |= result["ids"]
            sample += result["sample"]
            lengths += result["lengths"]
            n_tokens += result["tokens"]
            n_truncated += result["truncated"]
            n_complete += result["complete"]
            n_plan += result["plan_tokens"]
            n_dropped += result["dropped"]
            n_transposed += result["transposed"]

        check_tempo_recompute(split, sample)
        check_rederivation(split, sample, args.curated_root)
        check_section_states(split, sample)

        drop_rate = n_dropped / max(1, n_rows)
        if split == "val":
            check(n_dropped == 0, f"val: {n_dropped} rows are prefix-dropped")
        else:
            check(
                DROP_RATE_BAND[0] <= drop_rate <= DROP_RATE_BAND[1],
                f"{split}: prefix-drop rate {drop_rate:.4f} outside "
                f"[{DROP_RATE_BAND[0]}, {DROP_RATE_BAND[1]}] (p={PREFIX_DROP_P})",
            )
        histograms[split] = histogram
        transposed[split] = n_transposed / max(1, n_rows)
        piece_ids[split] = ids
        summary[split] = {
            "rows": n_rows,
            "tokens": n_tokens,
            "truncated": n_truncated,
            "truncation_rate": round(n_truncated / max(1, n_rows), 5),
            "complete": n_complete,
            "plan_tokens": n_plan,
            "prefix_dropped": n_dropped,
            "prefix_drop_rate": round(drop_rate, 5),
            "row_length": length_stats(lengths),
            "sampled": len(sample),
        }
        print(f"{split}: {json.dumps(summary[split])}", flush=True)

    pool.close()
    pool.join()

    for index in range(1, N_PASSES):
        check(
            piece_ids[f"pass{index}"] == piece_ids["pass0"],
            f"pass{index} covers a different piece set than pass0",
        )
    check(
        not (piece_ids["pass0"] & piece_ids["val"]),
        "val piece ids also appear in the training passes",
    )
    split = load_split(args.split_path)
    built = piece_ids["pass0"] | piece_ids["val"]
    check(
        piece_ids["val"] == {piece for piece in built if split[piece] == "val"},
        "val shards disagree with the val pieces of the split manifest",
    )
    check(
        len(piece_ids["val"]) == meta["val"]["held_out_pieces"],
        "meta.json val.held_out_pieces disagrees with the val shards",
    )
    check_key_shift(histograms, transposed)

    total_train_tokens = sum(
        summary[f"pass{index}"]["tokens"] for index in range(N_PASSES)
    )
    check(
        total_train_tokens == meta["total_train_tokens"],
        "meta.json total_train_tokens disagrees with the shards",
    )
    print(json.dumps({
        "root": str(args.root),
        "train_pieces": len(piece_ids["pass0"]),
        "val_pieces": len(piece_ids["val"]),
        "total_train_tokens": total_train_tokens,
        "prompt_check": "exhaustive (every row against its label.json)",
        "splits": summary,
        "failures": len(_FAILURES),
    }, indent=2))
    if _FAILURES:
        print(f"\nFAILED ({len(_FAILURES)}):", file=sys.stderr)
        for message in _FAILURES[:50]:
            print(f"  - {message}", file=sys.stderr)
        raise SystemExit(1)
    print("\nverify_curated: all checks passed")


if __name__ == "__main__":
    main()
