#!/usr/bin/env python3
"""Pre-launch gate for the pretraining corpus; exits non-zero on any failure.

Exhaustive checks: vocabulary range, row grammar and section plan, ``<E>`` and
crop agreement, split accounting, val disjointness by ``composition_id`` and
the vocabulary metadata. Sampled checks: bit-for-bit row re-derivation, section
states, tempo classes, crop round trips and the augmentation round trip.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import pyarrow.dataset as ds
import pyarrow.parquet as pq

from ariautils.midi import MidiDict, get_duration_ms

from taste.representation.tokenizer import get_tokenizer
from taste.representation.midi_features import tempo_class
from taste.representation.vocab import (
    KEY_BASE,
    KEY_TO_ID,
    MODEL_VOCAB_SIZE,
    PLAN_BASE,
    PLAN_IDS,
    S_MID_ID,
    TEMPO_BASE,
    TEMPO_TO_ID,
    UNUSED_IDS,
    VOCAB_META_NAME,
    load_vocab_meta,
    vocab_meta,
)
from taste.corpus.build_pretrain import (
    EXT_ROOT,
    KEY_ROOT,
    LABELS_PATH,
    MAX_PITCH_AUG,
    N_EPOCHS,
    OUT_ROOT,
    build_row,
    crop_window_span,
    emitted_states,
    n_section_runs,
    pass_draw,
    piece_stream,
    section_plan,
    section_states,
    take_crop,
    worker_state,
)
from taste.corpus.pretrain_format import FORMAT_VERSION, SEQ_LEN

# epoch0 is the unaugmented pass; the augmented passes share one algorithm and
# one fresh draw each, so the gate checks epoch0, val and the first
# AUGMENTED_PASSES augmented epochs (--augmented-passes 4 = every pass).
AUGMENTED_PASSES = 2


def splits(augmented_passes: int) -> tuple[str, ...]:
    return (*(f"epoch{epoch}" for epoch in range(augmented_passes + 1)), "val")
SAMPLE_ROWS = 200
SAMPLE_CROPS = 50
MAX_USED_ID = S_MID_ID + 1
WINDOW_MS = 5000

_FAILURES: list[str] = []


def check(condition: bool, message: str) -> bool:
    if not condition:
        _FAILURES.append(message)
    return bool(condition)


def read_split(root: Path, split: str) -> tuple[list[dict], list[dict]]:
    prefix = "val" if split == "val" else "train"
    rows = sorted(glob.glob(str(root / split / f"{prefix}-*.parquet")))
    index = sorted(glob.glob(str(root / "index" / split / f"{prefix}-*.parquet")))
    if not rows:
        raise SystemExit(f"missing {split} shards under {root}")
    if [Path(path).name for path in rows] != [Path(path).name for path in index]:
        raise SystemExit(f"{split}: row and index shards do not correspond")
    row_records, index_records = [], []
    for row_path, index_path in zip(rows, index):
        row_records.extend(pq.read_table(row_path).to_pylist())
        index_records.extend(pq.read_table(index_path).to_pylist())
    if len(row_records) != len(index_records):
        raise SystemExit(f"{split}: {len(row_records)} rows vs {len(index_records)} index rows")
    return row_records, index_records


def check_row_grammar(split: str, rows: list[dict], index: list[dict]) -> Counter:
    tokenizer = get_tokenizer()
    section_ids = frozenset(tokenizer.section_ids)
    plan_ids = frozenset(PLAN_IDS)
    unused = frozenset(UNUSED_IDS)
    key_ids = frozenset(KEY_TO_ID.values())
    tempo_ids = frozenset(TEMPO_TO_ID.values())
    expected_epoch = 0 if split == "val" else int(split[-1])
    key_histogram: Counter = Counter()

    for row, record in zip(rows, index):
        ids = row["input_ids"]
        tag = f"{split}/{record['piece_id']}"
        if not check(row["n_tokens"] == len(ids), f"{tag}: n_tokens != len(input_ids)"):
            continue
        if not check(len(ids) <= SEQ_LEN, f"{tag}: {len(ids)} tokens over the context"):
            continue
        check(row["epoch_idx"] == expected_epoch, f"{tag}: wrong epoch_idx")
        bad = [value for value in ids if not 0 <= value < MAX_USED_ID]
        if not check(not bad, f"{tag}: ids outside the used vocabulary: {bad[:4]}"):
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
            and ids[cursor] in (tokenizer.start_id, S_MID_ID),
            f"{tag}: header is not [key, tempo, plan, piano prefix..., <S>/<S_MID>]",
        ):
            continue
        row_start_id = ids[cursor]
        check(ids[cursor + 1] in section_ids, f"{tag}: no <SEC_*> after the start token")
        # letters are piece-global, so a row that starts at the piece's window 0
        # necessarily opens with the piece's first section, <SEC_A>
        if row_start_id == tokenizer.start_id:
            check(
                ids[cursor + 1] == tokenizer.section_ids[0],
                f"{tag}: <S> row does not open with <SEC_A>",
            )
        check(
            (row_start_id == S_MID_ID) == (record["crop_start_section"] > 0),
            f"{tag}: start token {row_start_id} disagrees with crop_start_section "
            f"{record['crop_start_section']}",
        )

        stream = ids[prefix_start:]
        check(
            not any(value in plan_ids for value in stream),
            f"{tag}: <PLAN_*> digit outside the header",
        )

        grammar_ok = True
        for position, value in enumerate(stream):
            after_state = position > 0 and stream[position - 1] in {
                tokenizer.start_id, S_MID_ID, tokenizer.time_id
            }
            if value in section_ids and not after_state:
                grammar_ok = check(
                    False, f"{tag}: <SEC_*> at {position} not after <S>/<S_MID>/<T>"
                )
                break
            if value == tokenizer.time_id:
                if position + 1 >= len(stream) or stream[position + 1] not in section_ids:
                    grammar_ok = check(False, f"{tag}: <T> at {position} without a state")
                    break
        if not grammar_ok:
            continue

        check(
            plan == section_plan(
                emitted_states(stream, row_start_id, tokenizer.time_id)
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
            has_end == (record["crop_end_section"] == record["n_sections_total"] - 1
                        and not record["truncated"]),
            f"{tag}: <E> does not agree with the crop end",
        )
        check(
            not (has_end and record["truncated"]),
            f"{tag}: truncated row carries <E>",
        )
        check(
            0 <= record["crop_start_section"] <= record["crop_end_section"]
            < record["n_sections_total"],
            f"{tag}: crop section range is not ordered inside the piece",
        )
        check(
            ids.count(tokenizer.end_id) == (1 if has_end else 0),
            f"{tag}: unexpected <E> count",
        )
        check(
            n_section_runs(stream, section_ids) == record["n_sections_emitted"],
            f"{tag}: n_sections_emitted mismatch",
        )
        if split in ("epoch0", "val"):
            check(
                record["pitch_shift"] == 0 and record["tempo_factor"] == 1.0,
                f"{tag}: unaugmented split carries a draw",
            )
        if split == "val":
            check(
                record["crop_start_section"] == 0,
                f"{tag}: val row is not front-anchored",
            )
    return key_histogram


def check_tempo_recompute(split: str, index: list[dict], sample: list[int]) -> None:
    for position in sample:
        record = index[position]
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


def check_rederivation(
    split: str,
    rows: list[dict],
    index: list[dict],
    sample: list[int],
    labels: dict,
) -> None:
    for position in sample:
        record = index[position]
        tag = f"{split}/{record['piece_id']}"
        pitch, factor = record["pitch_shift"], record["tempo_factor"]
        crop_uniform = 0.0 if split == "val" else pass_draw(
            record["piece_id"], int(split[-1])
        )[2]
        if split != "val":
            expected = pass_draw(record["piece_id"], int(split[-1]))
            check(
                (pitch, factor) == expected[:2],
                f"{tag}: recorded draw is not the seeded draw",
            )
        letters, starts = labels[record["piece_id"]]
        rebuilt = build_row(
            EXT_ROOT / record["midi_relpath"],
            record["key"],
            record["tempo_bpm"],
            starts,
            letters,
            pitch,
            factor,
            crop_uniform,
        )
        check(
            rebuilt["input_ids"] == rows[position]["input_ids"],
            f"{tag}: re-derived row differs from the stored row",
        )


def check_section_states(
    split: str,
    rows: list[dict],
    index: list[dict],
    sample: list[int],
    labels: dict,
) -> None:
    tokenizer = get_tokenizer()
    for position in sample:
        record = index[position]
        tag = f"{split}/{record['piece_id']}"
        letters, starts = labels[record["piece_id"]]
        changes = section_states(starts, letters, record["tempo_factor"])
        # the row's window 0 is the crop's first section boundary in the piece
        window = sorted(changes)[record["crop_start_section"]] - 1
        current = None
        ids = rows[position]["input_ids"]
        for cursor, value in enumerate(ids[:-1]):
            if value not in (tokenizer.start_id, S_MID_ID, tokenizer.time_id):
                continue
            window += 1
            current = changes.get(window, current)
            if ids[cursor + 1] != current:
                check(False, f"{tag}: state at window {window} is not the label state")
                break


def note_events(midi_dict: MidiDict) -> list[tuple[int, int, int]]:
    """``(onset_ms, pitch, velocity)`` of every note, in onset order."""
    events = [
        (
            get_duration_ms(
                start_tick=0,
                end_tick=note["data"]["start"],
                tempo_msgs=midi_dict.tempo_msgs,
                ticks_per_beat=midi_dict.ticks_per_beat,
            ),
            note["data"]["pitch"],
            note["data"]["velocity"],
        )
        for note in midi_dict.note_msgs
    ]
    return sorted(events)


def check_crops(
    split: str,
    rows: list[dict],
    index: list[dict],
    sample: list[int],
    labels: dict,
) -> int:
    """Segment invariants and the 25 ms crop round trip on over-budget rows.

    The row's stream must be exactly the whole-section crop its index record
    claims, and detokenizing it must reproduce the notes of the corresponding
    time slice of the source piece, re-based to the crop start.
    """
    tokenizer = get_tokenizer()
    grid_ms = tokenizer.time_step_ms
    for position in sample:
        record = index[position]
        tag = f"{split}/{record['piece_id']}"
        letters, starts = labels[record["piece_id"]]
        piece = piece_stream(
            EXT_ROOT / record["midi_relpath"],
            starts,
            letters,
            record["pitch_shift"],
            record["tempo_factor"],
        )
        positions = piece["positions"]
        n_windows = len(positions)
        boundaries = piece["boundaries"]
        if not check(
            len(boundaries) == record["n_sections_total"],
            f"{tag}: n_sections_total does not match the piece",
        ):
            continue
        first, last = crop_window_span(
            boundaries, n_windows, record["crop_start_section"], record["crop_end_section"]
        )
        ids = rows[position]["input_ids"]
        stream = ids[2 + record["n_plan_tokens"]:]
        row_start_id = (
            S_MID_ID if record["crop_start_section"] > 0 else tokenizer.start_id
        )
        if record["truncated"]:
            # fallback path: front-anchored at the section start, cut at a
            # window boundary inside it, so only the left edge is a section
            check(
                stream[: piece["prefix_len"] + 1]
                == piece["stream"][: piece["prefix_len"]] + [row_start_id],
                f"{tag}: truncated crop does not start at its section boundary",
            )
            expected_states = piece["states"][first : first + stream.count(tokenizer.time_id) + 1]
        else:
            expected = take_crop(
                piece["stream"], positions, n_windows, piece["prefix_len"],
                row_start_id, first, last,
            )
            check(
                stream == expected,
                f"{tag}: row stream is not the whole-section crop it records",
            )
            expected_states = piece["states"][first:last]
        check(
            ids[2 : 2 + record["n_plan_tokens"]] == section_plan(expected_states),
            f"{tag}: plan is not the RLE of the emitted windows",
        )
        check(len(ids) <= SEQ_LEN, f"{tag}: crop exceeds the context")

        n_crop_windows = stream.count(tokenizer.time_id) + 1
        start_ms = first * WINDOW_MS
        end_ms = (first + n_crop_windows) * WINDOW_MS
        source = note_events(tokenizer.detokenize_midi(piece["stream"]))
        expected_notes = [
            (onset - start_ms, pitch, velocity)
            for onset, pitch, velocity in source
            if start_ms <= onset < end_ms
        ]
        # <S_MID> is not an Aria id; the Aria decoder only knows <S>
        actual = note_events(tokenizer.detokenize_midi(
            [tokenizer.start_id if value == S_MID_ID else value for value in stream]
        ))
        if not check(
            len(actual) == len(expected_notes),
            f"{tag}: crop round trip has {len(actual)} notes, slice has "
            f"{len(expected_notes)}",
        ):
            continue
        worst = max(
            (abs(a[0] - b[0]) for a, b in zip(actual, expected_notes)), default=0
        )
        check(
            worst <= grid_ms
            and all(a[1:] == b[1:] for a, b in zip(actual, expected_notes)),
            f"{tag}: crop round trip differs from the source slice "
            f"(worst onset delta {worst} ms)",
        )
    return len(sample)


def check_augmentation_grid(pieces: list[tuple[str, list, list]]) -> None:
    """``export_pitch_aug``/``export_tempo_aug`` round trip on the 25 ms grid."""
    state = worker_state()
    tokenizer = state["tokenizer"]
    grid_ms = tokenizer.time_step_ms
    for relpath, _letters, _starts in pieces:
        tokens = tokenizer.decode(tokenizer.tokenize_midi(EXT_ROOT / relpath))
        base = note_events(tokenizer.detokenize_midi(tokenizer.encode(tokens)))

        shift = MAX_PITCH_AUG
        shifted = state["pitch_aug"](tokens, pitch_aug=shift)
        moved = note_events(tokenizer.detokenize_midi(tokenizer.encode(shifted)))
        check(
            len(moved) == len(base)
            and all(
                after[1] == before[1] + shift and after[0] == before[0]
                for before, after in zip(base, moved)
            ),
            f"{relpath}: pitch_aug did not shift every pitch by {shift} at fixed time",
        )

        factor = 1.10
        stretched = state["tempo_aug"](tokens, tempo_aug=factor)
        scaled = note_events(tokenizer.detokenize_midi(tokenizer.encode(stretched)))
        if not check(
            len(scaled) == len(base),
            f"{relpath}: tempo_aug changed the note count",
        ):
            continue
        worst = max(
            (abs(after[0] - round(before[0] * factor)) for before, after in zip(base, scaled)),
            default=0,
        )
        check(
            worst <= grid_ms
            and all(after[1] == before[1] for before, after in zip(base, scaled)),
            f"{relpath}: tempo_aug is not a x{factor} time scaling on the "
            f"{grid_ms} ms grid (worst delta {worst} ms)",
        )


def check_key_shift(histograms: dict[str, Counter], transposed: dict[str, float]) -> None:
    base = histograms["epoch0"]
    base_total = sum(base.values())
    for split in histograms:
        if split in ("epoch0", "val"):
            continue
        other = histograms[split]
        check(sum(other.values()) == base_total, f"{split}: row count differs from epoch0")
        overlap = sum(min(base[key], other[key]) for key in set(base) | set(other))
        share = overlap / max(1, base_total)
        check(
            share < 0.98,
            f"{split}: key histogram is indistinguishable from epoch0 "
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
            f"  {split}: key-histogram overlap with epoch0 = {share:.3f}, "
            f"transposed fraction = {transposed[split]:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=OUT_ROOT)
    parser.add_argument("--key-root", type=Path, default=KEY_ROOT)
    parser.add_argument("--labels-path", type=Path, default=LABELS_PATH)
    parser.add_argument("--sample-rows", type=int, default=SAMPLE_ROWS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--augmented-passes", type=int, default=AUGMENTED_PASSES,
        help=f"augmented epochs to check after epoch0 (1..{N_EPOCHS - 1})",
    )
    args = parser.parse_args()
    if not 1 <= args.augmented_passes < N_EPOCHS:
        raise ValueError(f"--augmented-passes must be in [1, {N_EPOCHS})")

    meta = json.loads((args.root / "meta.json").read_text(encoding="utf-8"))
    check(
        meta["format_version"] == FORMAT_VERSION,
        f"meta.json format_version {meta['format_version']!r} != {FORMAT_VERSION!r}",
    )
    check(
        meta["time_step_ms"] == get_tokenizer().time_step_ms,
        f"meta.json time_step_ms {meta['time_step_ms']} != the production tokenizer",
    )
    check(
        load_vocab_meta(args.root) == vocab_meta(),
        f"{args.root / VOCAB_META_NAME} does not match taste.representation.vocab",
    )
    check(
        vocab_meta()["model_vocab_size"] == MODEL_VOCAB_SIZE
        and MAX_USED_ID <= MODEL_VOCAB_SIZE,
        f"{MAX_USED_ID} used ids do not fit the model vocab {MODEL_VOCAB_SIZE}",
    )

    labels_table = pq.read_table(
        args.labels_path, columns=["piece_id", "section_letters", "section_starts_sec"]
    )
    labels = {
        piece_id: (letters, starts)
        for piece_id, letters, starts in zip(
            labels_table["piece_id"].to_pylist(),
            labels_table["section_letters"].to_pylist(),
            labels_table["section_starts_sec"].to_pylist(),
        )
    }
    key_table = ds.dataset(
        sorted(args.key_root.glob("part-*.parquet")), format="parquet"
    ).to_table(columns=["piece_id", "composition_id"])
    composition = dict(zip(
        key_table["piece_id"].to_pylist(), key_table["composition_id"].to_pylist()
    ))

    rng = random.Random(args.seed)
    histograms: dict[str, Counter] = {}
    transposed: dict[str, float] = {}
    piece_ids: dict[str, set[str]] = {}
    summary: dict[str, dict] = {}

    for split in splits(args.augmented_passes):
        rows, index = read_split(args.root, split)
        histograms[split] = check_row_grammar(split, rows, index)
        transposed[split] = sum(
            record["key_aug"] != record["key"] for record in index
        ) / max(1, len(index))
        piece_ids[split] = {record["piece_id"] for record in index}
        sample = rng.sample(range(len(rows)), min(args.sample_rows, len(rows)))
        check_tempo_recompute(split, index, sample)
        check_rederivation(split, rows, index, sample, labels)
        check_section_states(split, rows, index, sample, labels)
        over_budget = [
            position for position, record in enumerate(index)
            if record["crop_start_section"] > 0 or not record["has_end"]
        ]
        crop_sample = rng.sample(
            over_budget, min(SAMPLE_CROPS, len(over_budget))
        )
        check_crops(split, rows, index, crop_sample, labels)
        summary[split] = {
            "rows": len(rows),
            "tokens": sum(row["n_tokens"] for row in rows),
            "truncated": sum(bool(record["truncated"]) for record in index),
            "cropped": len(over_budget),
            "plan_tokens": sum(record["n_plan_tokens"] for record in index),
            "sampled": len(sample),
            "crop_sampled": len(crop_sample),
        }
        if split == "epoch0":
            aug_pieces = [
                (index[position]["midi_relpath"], *labels[index[position]["piece_id"]])
                for position in sample[:3]
            ]
        print(f"{split}: {json.dumps(summary[split])}", flush=True)

    check_augmentation_grid(aug_pieces)

    for epoch in range(1, args.augmented_passes + 1):
        check(
            piece_ids[f"epoch{epoch}"] == piece_ids["epoch0"],
            f"epoch{epoch} covers a different piece set than epoch0",
        )
    train_compositions = {
        composition[piece_id] for piece_id in piece_ids["epoch0"]
    }
    val_compositions = {composition[piece_id] for piece_id in piece_ids["val"]}
    check(
        not (train_compositions & val_compositions),
        f"{len(train_compositions & val_compositions)} compositions appear in both splits",
    )
    check_key_shift(histograms, transposed)

    print(json.dumps({
        "root": str(args.root),
        "train_compositions": len(train_compositions),
        "val_compositions": len(val_compositions),
        "splits": summary,
        "failures": len(_FAILURES),
    }, indent=2))
    if _FAILURES:
        print(f"\nFAILED ({len(_FAILURES)}):", file=sys.stderr)
        for message in _FAILURES[:50]:
            print(f"  - {message}", file=sys.stderr)
        raise SystemExit(1)
    print("\nverify_pretrain: all checks passed")


if __name__ == "__main__":
    main()
