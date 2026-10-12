#!/usr/bin/env python3
"""Pack a stage's token shards into a ragged store of 5-second windows.

    python -m taste.codec.build_windows --stage all

Reads ``tokens/<stage>/tokens-*.parquet`` under the codec token root, splits
each row on ``<T>``, drops the header and in-stream ``<SEC_*>`` tokens, and
writes ``<out>/<stage>/`` with ``tokens.u16`` (flat window tokens),
``index.npz`` (``offsets``, ``piece_row``, ``window_idx``, ``section``),
``pieces.npz`` (``piece_id``, ``key_id``, ``tempo_id``, ``n_windows``) and
``meta.json``. The 5 s to 1 s split happens online in ``WindowGroupDataset``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from taste import paths

from . import vocab
from .windows import split_piece

ROOT = paths.data("codec_tokens")
STAGES = ["large_public", "curated", "high_quality"]


def build(
    root: Path,
    stage: str,
    out_dir: Path,
    max_tokens: int | None,
    drop_empty: bool,
    shard_limit: int | None,
    piece_limit: int | None,
    batch_size: int,
) -> dict:
    vocab.verify()
    shards = sorted((root / "tokens" / stage).glob("tokens-*.parquet"))
    if not shards:
        raise SystemExit(f"no shards under {root / 'tokens' / stage}")
    if shard_limit is not None:
        shards = shards[:shard_limit]

    out_dir.mkdir(parents=True, exist_ok=True)
    token_path = out_dir / "tokens.u16"

    offsets = [0]
    piece_row: list[int] = []
    window_idx: list[int] = []
    section: list[int] = []
    piece_ids: list[str] = []
    key_ids: list[int] = []
    tempo_ids: list[int] = []
    n_windows: list[int] = []

    started = time.perf_counter()
    cursor = 0
    dropped_empty = 0
    with token_path.open("wb") as sink:
        for shard in shards:
            for batch in pq.ParquetFile(shard).iter_batches(batch_size=batch_size):
                rows = batch.to_pydict()
                for piece_id, input_ids in zip(rows["piece_id"], rows["input_ids"]):
                    if piece_limit is not None and len(piece_ids) >= piece_limit:
                        break
                    piece = split_piece(piece_id, input_ids, max_tokens=max_tokens)
                    row = len(piece_ids)
                    kept = 0
                    for index, window in enumerate(piece.windows):
                        if drop_empty and window.size == 0:
                            dropped_empty += 1
                            continue
                        sink.write(np.asarray(window, dtype=np.uint16).tobytes())
                        cursor += window.size
                        offsets.append(cursor)
                        piece_row.append(row)
                        window_idx.append(index)
                        section.append(int(piece.sections[index]))
                        kept += 1
                    piece_ids.append(piece.piece_id)
                    key_ids.append(piece.key_id)
                    tempo_ids.append(piece.tempo_id)
                    n_windows.append(kept)
                if piece_limit is not None and len(piece_ids) >= piece_limit:
                    break
            if piece_limit is not None and len(piece_ids) >= piece_limit:
                break
            print(f"  {shard.name}: {len(piece_ids)} pieces, {len(piece_row)} windows")

    np.savez(
        out_dir / "index.npz",
        offsets=np.asarray(offsets, dtype=np.int64),
        piece_row=np.asarray(piece_row, dtype=np.int32),
        window_idx=np.asarray(window_idx, dtype=np.int32),
        section=np.asarray(section, dtype=np.uint8),
    )
    np.savez(
        out_dir / "pieces.npz",
        piece_id=np.asarray(piece_ids),
        key_id=np.asarray(key_ids, dtype=np.int16),
        tempo_id=np.asarray(tempo_ids, dtype=np.int16),
        n_windows=np.asarray(n_windows, dtype=np.int32),
    )
    meta = {
        "stage": stage,
        "pieces": len(piece_ids),
        "windows": len(piece_row),
        "tokens": cursor,
        "window_ms": vocab.WINDOW_MS,
        "subwindow_ms": vocab.SUBWINDOW_MS,
        "time_step_ms": vocab.TIME_STEP_MS,
        "vocab_format": vocab.FORMAT_VERSION,
        "vocab_size": vocab.VOCAB_SIZE,
        "note_event": ["piano", "onset", "dur"],
        "build_max_tokens": max_tokens,
        "drop_empty": drop_empty,
        "dropped_empty_windows": dropped_empty,
        "token_dtype": "uint16",
        "shards": [shard.name for shard in shards],
        "elapsed_sec": round(time.perf_counter() - started, 2),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--stage",
        nargs="+",
        default=["high_quality"],
        choices=[*STAGES, "all"],
        help="one or more stages, or 'all'",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "windows")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="truncate 5 s windows at build time; default keeps them full",
    )
    parser.add_argument("--drop-empty", action="store_true", help="skip silent windows")
    parser.add_argument("--shards", type=int, default=None, help="use only the first N shards")
    parser.add_argument("--pieces", type=int, default=None, help="stop after N pieces")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    stages = STAGES if "all" in args.stage else args.stage
    for stage in stages:
        print(f"[info] building {stage} (joint-piano 5 s windows) ...")
        meta = build(
            root=args.root,
            stage=stage,
            out_dir=args.out / stage,
            max_tokens=args.max_tokens,
            drop_empty=args.drop_empty,
            shard_limit=args.shards,
            piece_limit=args.pieces,
            batch_size=args.batch_size,
        )
        print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
