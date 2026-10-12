#!/usr/bin/env python3
"""Encode the token corpora into window-VAE latents for the latent DiT.

One store per (stage, split) under ``<out_root>/<stage>/<split>/``: a flat
float16 ``mu.f16`` memmap of shape ``(n_latents_total, 128)``, an
``index.parquet`` with one row per kept corpus row, and ``meta.json``. Each row
is split into 5 s windows, cropped to at most 38 windows on a section boundary,
cut into 1 s encoder windows spaced ``--hop`` seconds apart and encoded to the
posterior mean. At hop 0.5 the store holds two latents per second, interleaved
integer offset first; ``--from-v1`` reuses the integer-offset latents of an
existing hop-1.0 store and encodes only the half-offset windows.
``--shard K/N`` encodes the split's source files with ``i % N == K`` and
``--merge N`` stitches the shard directories back in source-file order.

    python -m taste.corpus.build_dit_latents --stage high_quality --split val --gpu 0
    python -m taste.corpus.build_dit_latents --stage high_quality --split val --hop 0.5 \
        --from-v1 <hop-1.0 store>/high_quality/val
    python -m taste.corpus.build_dit_latents --stage large_public --split pass0 --shard 2/6 --gpu 2
    python -m taste.corpus.build_dit_latents --stage large_public --split pass0 --merge 6
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import time
from collections import deque
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from taste import paths
from taste.codec import vocab
from taste.codec.infer_reconstruct import load_run
from taste.codec.windows import split_5s_to_1s, split_piece, truncate_to_events

CKPT_DIR = paths.ckpt("codec", "window_vae")
OUT_ROOT = paths.data("dit_latents")

MAX_WINDOWS = 38
MIN_WINDOWS_AFTER_CROP = 20
ENC_SEQLEN = 96
D_LATENT = 128
PROMPT_FIELDS = ("emotion", "genre", "accompaniment", "performance", "imagery")

# stage -> number of SFT passes in its token corpus
STAGES = {
    "large_public": 6,
    "curated": 3,
    "high_quality": 5,
    "pretrain": 0,
}
PRETRAIN_SPLITS = ("epoch0", "epoch1", "val")

INDEX_SCHEMA = pa.schema([
    pa.field("piece_id", pa.string()),
    pa.field("source_file", pa.string()),
    pa.field("source_row", pa.int32()),
    pa.field("pass_idx", pa.int8()),
    pa.field("offset", pa.int64()),
    pa.field("n_seconds", pa.int32()),
    pa.field("n_latents", pa.int32()),
    pa.field("n_windows", pa.int32()),
    pa.field("complete", pa.bool_()),
    pa.field("start_mid", pa.bool_()),
    pa.field("key_id", pa.int8()),
    pa.field("tempo_id", pa.int8()),
    pa.field("sections", pa.list_(pa.int8())),
    *(pa.field(name, pa.string()) for name in PROMPT_FIELDS),
    pa.field("prefix_dropped", pa.bool_()),
    pa.field("context_shard", pa.string()),
    pa.field("context_length", pa.int32()),
])


def parse_shard(text: str) -> tuple[int, int]:
    shard, n_shards = (int(part) for part in text.split("/"))
    if not 0 <= shard < n_shards:
        raise argparse.ArgumentTypeError(f"--shard {text}: need 0 <= K < N")
    return shard, n_shards


# ---------------------------------------------------------------- worker side

def crop_windows(windows: list[np.ndarray], sections: np.ndarray) -> int:
    """Number of leading 5 s windows to keep, cutting on a section boundary."""
    if len(windows) <= MAX_WINDOWS:
        return len(windows)
    starts = np.concatenate(([0], np.flatnonzero(sections[1:] != sections[:-1]) + 1))
    run_start = int(starts[np.searchsorted(starts, MAX_WINDOWS, side="right") - 1])
    return MAX_WINDOWS if run_start < MIN_WINDOWS_AFTER_CROP else run_start


def pack_row(windows_1s: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """1 s windows → ``(tokens (n, 96) int16, lengths (n,) int16)``."""
    tokens = np.full((len(windows_1s), ENC_SEQLEN), vocab.PAD_ID, dtype=np.int16)
    lengths = np.empty(len(windows_1s), dtype=np.int16)
    tokens[:, 0] = vocab.T_ID
    for row, window in enumerate(windows_1s):
        window = truncate_to_events(window, ENC_SEQLEN - 1)
        tokens[row, 1 : window.size + 1] = window
        lengths[row] = window.size + 1
    return tokens, lengths


def piece_events(windows: list[np.ndarray]) -> tuple[np.ndarray, list[tuple[int, int | None]]]:
    """The kept 5 s windows as one event list with piece-absolute onsets."""
    onsets: list[int] = []
    payload: list[tuple[int, int | None]] = []
    for index, window in enumerate(windows):
        base = index * vocab.WINDOW_MS
        cursor = 0
        while cursor < window.size:
            head = int(window[cursor])
            length = vocab.event_length(head)
            onsets.append(base + vocab.onset_ms(int(window[cursor + 1])))
            payload.append((head, int(window[cursor + 2]) if length == 3 else None))
            cursor += length
    return np.asarray(onsets, dtype=np.int64), payload


def cut_window(onsets: np.ndarray, payload: list, start_ms: int) -> np.ndarray:
    """The 1 s window starting at ``start_ms``, onsets rebased to its start.

    An event goes whole into the window its onset falls in; a window reaching
    past the piece end simply holds fewer events.
    """
    lo = int(np.searchsorted(onsets, start_ms))
    hi = int(np.searchsorted(onsets, start_ms + vocab.SUBWINDOW_MS))
    tokens: list[int] = []
    for index in range(lo, hi):
        head, duration = payload[index]
        tokens.append(head)
        tokens.append(vocab.onset_id(int(onsets[index]) - start_ms))
        if duration is not None:
            tokens.append(duration)
    return np.asarray(tokens, dtype=np.int64)


def encoder_windows(piece, keep: int, hop: float, from_v1: bool) -> tuple[int, list]:
    """``(n_seconds, windows to encode)`` for one cropped piece.

    At hop 1.0 the integer-offset windows tile the piece. At hop 0.5 the
    half-offset windows are interleaved after each integer one, or returned
    alone when the integer latents are reused from an existing hop-1.0 store.
    """
    integer: list[np.ndarray] = []
    for window in piece.windows[:keep]:
        integer.extend(split_5s_to_1s(window))
    if hop == 1.0:
        return len(integer), integer
    onsets, payload = piece_events(piece.windows[:keep])
    half = [
        cut_window(onsets, payload, 1000 * second + 500) for second in range(len(integer))
    ]
    if from_v1:
        return len(integer), half
    return len(integer), [window for pair in zip(integer, half) for window in pair]


def process_group(task: tuple) -> tuple[list, int, int]:
    """One parquet row group → packed encoder inputs plus index fields."""
    data_file, group, row_offset, piece_ids, is_pretrain, hop, from_v1 = task
    columns = ["input_ids", "epoch_idx"] if is_pretrain else [
        "input_ids", "pass_idx", "prefix_dropped", *PROMPT_FIELDS
    ]
    table = pq.ParquetFile(data_file).read_row_group(group, columns=columns).to_pydict()

    out, skipped, cropped = [], 0, 0
    for local, piece_id in enumerate(piece_ids):
        ids = np.asarray(table["input_ids"][local], dtype=np.int32)
        has_end = bool(ids[-1] == vocab.E_ID)
        body_start = int(
            np.flatnonzero((ids == vocab.S_ID) | (ids == vocab.S_MID_ID))[0]
        )
        piece = split_piece(piece_id, ids)
        keep = crop_windows(piece.windows, piece.sections)
        if keep < 2:
            skipped += 1
            continue
        cropped += keep < len(piece.windows)
        n_seconds, windows = encoder_windows(piece, keep, hop, from_v1)
        tokens, lengths = pack_row(windows)
        prompts = (
            {"prefix_dropped": False, **{name: "" for name in PROMPT_FIELDS}}
            if is_pretrain
            else {
                "prefix_dropped": bool(table["prefix_dropped"][local]),
                **{name: (table[name][local] or "") for name in PROMPT_FIELDS},
            }
        )
        record = {
            "piece_id": piece_id,
            "source_file": os.path.basename(data_file),
            "source_row": row_offset + local,
            "pass_idx": int(table["epoch_idx" if is_pretrain else "pass_idx"][local]),
            "n_seconds": n_seconds,
            "n_latents": int(n_seconds * round(1 / hop)),
            "n_windows": keep,
            "complete": has_end and keep == len(piece.windows),
            "start_mid": bool(ids[body_start] == vocab.S_MID_ID),
            "key_id": piece.key_id - vocab.KEY_LO if piece.key_id >= 0 else -1,
            "tempo_id": piece.tempo_id - vocab.TEMPO_LO if piece.tempo_id >= 0 else -1,
            "sections": (piece.sections[:keep].astype(np.int16) - 1).astype(np.int8).tolist(),
            **prompts,
        }
        out.append((record, tokens, lengths))
    return out, skipped, cropped


# ------------------------------------------------------------------ GPU side

@torch.no_grad()
def encode_chunk(model, tokens: np.ndarray, lengths: np.ndarray, device: str) -> np.ndarray:
    """``(B, 96)`` packed 1 s windows → ``(B, 128)`` posterior means, float16."""
    enc = torch.from_numpy(tokens.astype(np.int64)).to(device)
    lens = torch.from_numpy(lengths.astype(np.int64)).to(device)
    mask = torch.arange(ENC_SEQLEN, device=device)[None, :] >= lens[:, None]
    # bf16 autocast: |dmu| <= 0.024 against fp32 (2% of the per-dim latent
    # std), well inside the float16 store's resolution.
    with torch.autocast(device.split(":")[0], dtype=torch.bfloat16):
        _z, mu, _logvar = model.encode(enc.T.unsqueeze(-1), padding_mask=mask.unsqueeze(1))
    return mu[:, 0].float().cpu().numpy().astype(np.float16)


class MuWriter:
    """Streams encoded 1 s latents into ``mu.f16`` row by row.

    Encoded chunks arrive at the encoder's batch granularity, rows are announced
    in store order; a row is written once all of its own windows are encoded. In
    ``--from-v1`` mode the row's half-offset latents are interleaved with the
    integer-offset latents of the hop-1.0 store, which the caller supplies.
    """

    def __init__(self, path: Path):
        self.file = open(path, "wb")
        self.chunks: deque = deque()
        self.have = 0
        self.rows: deque = deque()

    def add_row(self, n_encoded: int, integer: np.ndarray | None) -> None:
        self.rows.append((n_encoded, integer))

    def add_mu(self, mu: np.ndarray) -> None:
        self.chunks.append(mu)
        self.have += mu.shape[0]
        while self.rows and self.have >= self.rows[0][0]:
            count, integer = self.rows.popleft()
            block = self._take(count)
            if integer is None:
                self.file.write(block.tobytes())
                continue
            out = np.empty((2 * count, D_LATENT), dtype=np.float16)
            out[0::2] = integer
            out[1::2] = block
            self.file.write(out.tobytes())

    def _take(self, count: int) -> np.ndarray:
        taken: list[np.ndarray] = []
        remaining = count
        while remaining:
            chunk = self.chunks[0]
            if chunk.shape[0] <= remaining:
                taken.append(self.chunks.popleft())
                remaining -= chunk.shape[0]
            else:
                taken.append(chunk[:remaining])
                self.chunks[0] = chunk[remaining:]
                remaining = 0
        self.have -= count
        return taken[0] if len(taken) == 1 else np.concatenate(taken)

    def close(self) -> None:
        if self.rows or self.have:
            raise ValueError(f"writer closed with {len(self.rows)} rows pending")
        self.file.close()


def split_files(corpus: Path, split: str) -> list[Path]:
    prefix = "val-" if split == "val" else "train-"
    return sorted((corpus / split).glob(f"{prefix}*.parquet"))


def load_v1_store(path: Path, keep_files: set[str] | None = None) -> tuple[dict, np.memmap]:
    """``(index columns, mu memmap)`` of the hop-1.0 store that ``--from-v1`` reuses.

    ``keep_files`` keeps only the rows coming from those source parquet files
    (shard mode); the hop-1.0 store was built from the same files in the same
    order, so the kept rows align positionally with this build's rows.
    """
    columns = pq.read_table(
        path / "index.parquet",
        columns=["piece_id", "source_file", "offset", "n_seconds"],
    ).to_pydict()
    total = columns["offset"][-1] + columns["n_seconds"][-1]
    mu = np.memmap(path / "mu.f16", dtype=np.float16, mode="r", shape=(total, D_LATENT))
    if keep_files is not None:
        pick = [i for i, name in enumerate(columns["source_file"]) if name in keep_files]
        columns = {key: [value[i] for i in pick] for key, value in columns.items()}
    return columns, mu


def file_blocks(records: list[dict]) -> list[dict]:
    """Per-source-file row/latent runs of a shard's index, in store order."""
    blocks: list[dict] = []
    for record in records:
        if blocks and blocks[-1]["file"] == record["source_file"]:
            blocks[-1]["n_rows"] += 1
            blocks[-1]["n_latents"] += record["n_latents"]
        else:
            blocks.append({
                "file": record["source_file"],
                "n_rows": 1,
                "n_latents": record["n_latents"],
            })
    return blocks


def merge_shards(out_root: Path, stage: str, split: str, n_shards: int) -> None:
    """Stitch ``<split>__shard{k}_{n_shards}`` dirs into ``<split>/``.

    Shard ``k`` holds source files ``k, k+N, ...``, so the merge walks the
    split's full source-file list and copies each file's block out of the shard
    that owns it: the store comes out in source-file order, not shard order.
    """
    base = out_root / stage
    dirs = [base / f"{split}__shard{k}_{n_shards}" for k in range(n_shards)]
    for path in dirs:
        if not (path / "meta.json").exists():
            raise SystemExit(f"missing shard {path}")
    metas = [json.loads((path / "meta.json").read_text(encoding="utf-8")) for path in dirs]
    for key in ("source_files", "hop", "latents_per_second", "from_v1", "max_windows"):
        if len({json.dumps(meta[key]) for meta in metas}) != 1:
            raise SystemExit(f"shards disagree on {key}")
    out_dir = base / split
    if (out_dir / "meta.json").exists():
        raise SystemExit(f"{out_dir} already built")

    where: dict[str, tuple[int, int, int, int, int]] = {}
    for shard, meta in enumerate(metas):
        latent = row = 0
        for block in meta["file_blocks"]:
            where[block["file"]] = (
                shard, latent, block["n_latents"], row, block["n_rows"]
            )
            latent += block["n_latents"]
            row += block["n_rows"]
    indexes = [pq.read_table(path / "index.parquet") for path in dirs]

    out_dir.mkdir(parents=True, exist_ok=True)
    row_bytes = D_LATENT * 2
    tables = []
    with open(out_dir / "mu.f16", "wb") as dst:
        for name in metas[0]["source_files"]:
            if name not in where:
                continue
            shard, latent, n_latents, row, n_rows = where[name]
            with open(dirs[shard] / "mu.f16", "rb") as src:
                src.seek(latent * row_bytes)
                remaining = n_latents * row_bytes
                while remaining:
                    chunk = src.read(min(1 << 24, remaining))
                    if not chunk:
                        raise ValueError(f"{dirs[shard]}/mu.f16 ended early on {name}")
                    dst.write(chunk)
                    remaining -= len(chunk)
            tables.append(indexes[shard].slice(row, n_rows))

    table = pa.concat_tables(tables)
    counts = np.asarray(table.column("n_latents"), dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(counts)[:-1])) if counts.size else counts
    table = table.set_column(
        table.schema.get_field_index("offset"),
        INDEX_SCHEMA.field("offset"),
        pa.array(offsets, type=pa.int64()),
    )
    total_latents = int(counts.sum())
    written = (out_dir / "mu.f16").stat().st_size
    if total_latents * row_bytes != written:
        raise ValueError(
            f"index sums to {total_latents} latents ({total_latents * row_bytes} B) "
            f"but mu.f16 is {written} B")
    pq.write_table(table, out_dir / "index.parquet")

    meta = dict(metas[0])
    meta.pop("shard")
    meta.pop("file_blocks")
    for key in ("n_rows", "n_seconds", "skipped_lt2_windows", "cropped", "missing_context"):
        meta[key] = sum(shard_meta[key] for shard_meta in metas)
    meta["n_latents"] = total_latents
    if meta["n_rows"] != table.num_rows:
        raise ValueError(f"shards report {meta['n_rows']} rows, merged {table.num_rows}")
    meta["merged_shards"] = n_shards
    meta["shard_elapsed_s"] = [shard_meta["elapsed_s"] for shard_meta in metas]
    meta["elapsed_s"] = round(max(meta["shard_elapsed_s"]), 1)
    meta["seconds_per_second"] = round(meta["n_seconds"] / meta["elapsed_s"], 1)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    for path in dirs:
        shutil.rmtree(path)
    print(
        f"[ok] merged {stage}/{split} from {n_shards} shards: {meta['n_rows']} rows, "
        f"{meta['n_seconds']} seconds, {total_latents} latents"
    )


def build_split(args, model, device: str, stage: str, split: str) -> None:
    corpus = paths.data("corpora", stage)
    all_files = split_files(corpus, split)
    if args.shard is None:
        data_files = all_files
        out_dir = args.out_root / stage / split
    else:
        shard, n_shards = args.shard
        data_files = [
            path for index, path in enumerate(all_files) if index % n_shards == shard
        ]
        out_dir = args.out_root / stage / f"{split}__shard{shard}_{n_shards}"
    if (out_dir / "meta.json").exists():
        print(f"[skip] {stage}/{split}: meta.json exists")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    is_pretrain = stage == "pretrain"
    lps = round(1 / args.hop)

    v1_columns, v1_mu = (None, None)
    if args.from_v1:
        v1_columns, v1_mu = load_v1_store(
            args.from_v1,
            None if args.shard is None else {path.name for path in data_files},
        )

    context: dict[str, tuple[str, int]] = {}
    if not is_pretrain:
        manifest = pq.read_table(
            corpus / "qwen3_4b_imagery_states" / "manifest.parquet"
        ).to_pydict()
        context = {
            str(piece): (str(shard), int(length))
            for piece, shard, length in zip(
                manifest["piece_id"], manifest["shard_file"], manifest["length"]
            )
        }

    tasks = []
    total_rows = 0
    for data_file in data_files:
        pieces = pq.read_table(
            corpus / "index" / split / data_file.name, columns=["piece_id"]
        ).column("piece_id").to_pylist()
        meta = pq.ParquetFile(data_file).metadata
        row_offset = 0
        for group in range(meta.num_row_groups):
            n_rows = meta.row_group(group).num_rows
            group_pieces = [str(p) for p in pieces[row_offset : row_offset + n_rows]]
            if group_pieces:
                tasks.append((
                    str(data_file), group, row_offset, group_pieces,
                    is_pretrain, args.hop, bool(args.from_v1),
                ))
                total_rows += len(group_pieces)
            row_offset += n_rows
        if row_offset != meta.num_rows or len(pieces) != meta.num_rows:
            raise ValueError(f"{data_file}: index and data disagree on row count")

    records: list[dict] = []
    buf_tokens: list[np.ndarray] = []
    buf_lengths: list[np.ndarray] = []
    buffered = 0
    n_latents_total = 0
    n_seconds_total = 0
    skipped = 0
    cropped = 0
    missing_context = 0
    stats_sum = np.zeros(D_LATENT, dtype=np.float64)
    stats_sq = np.zeros(D_LATENT, dtype=np.float64)
    stats_n = 0
    # the hop-1.0 pass0 statistics serve the store at every hop
    collect_stats = args.hop == 1.0 and split == "pass0" and args.shard is None
    start_time = time.time()
    last_report = start_time
    rows_seen = 0

    writer = MuWriter(out_dir / "mu.f16")

    def flush(force: bool) -> None:
        nonlocal buffered, stats_n
        while buffered >= args.batch_windows or (force and buffered):
            take = min(args.batch_windows, buffered)
            tokens = np.concatenate(buf_tokens)
            lengths = np.concatenate(buf_lengths)
            mu = encode_chunk(model, tokens[:take], lengths[:take], device)
            writer.add_mu(mu)
            if collect_stats:
                block = mu.astype(np.float64)
                stats_sum[:] += block.sum(axis=0)
                stats_sq[:] += (block * block).sum(axis=0)
                stats_n += block.shape[0]
            buf_tokens[:] = [tokens[take:]] if take < tokens.shape[0] else []
            buf_lengths[:] = [lengths[take:]] if take < lengths.shape[0] else []
            buffered -= take

    def harvest(result) -> None:
        nonlocal buffered, n_latents_total, n_seconds_total, skipped, cropped
        nonlocal missing_context, last_report, rows_seen
        rows, n_skipped, n_cropped = result
        rows_seen += len(rows) + n_skipped
        skipped += n_skipped
        cropped += n_cropped
        for record, tokens, lengths in rows:
            shard, length = context.get(record["piece_id"], ("", 0))
            if shard == "":
                missing_context += 1
            record["context_shard"] = shard
            record["context_length"] = length
            record["offset"] = n_latents_total
            n_latents_total += record["n_latents"]
            n_seconds_total += record["n_seconds"]
            integer = None
            if v1_columns is not None:
                position = len(records)
                if (v1_columns["piece_id"][position] != record["piece_id"]
                        or v1_columns["n_seconds"][position] != record["n_seconds"]):
                    raise ValueError(
                        f"row {position}: hop-1.0 store holds "
                        f"{v1_columns['piece_id'][position]}/"
                        f"{v1_columns['n_seconds'][position]} s, this build has "
                        f"{record['piece_id']}/{record['n_seconds']} s")
                start = v1_columns["offset"][position]
                integer = v1_mu[start : start + record["n_seconds"]]
            records.append(record)
            writer.add_row(tokens.shape[0], integer)
            buf_tokens.append(tokens)
            buf_lengths.append(lengths)
            buffered += tokens.shape[0]
        flush(force=False)
        now = time.time()
        if now - last_report >= 60:
            last_report = now
            done = now - start_time
            rate = n_seconds_total / done
            eta = done * (total_rows - rows_seen) / rows_seen / 60
            print(
                f"[run] {stage}/{split} {rows_seen}/{total_rows} rows "
                f"{n_seconds_total / 1e6:.2f}M seconds {rate:.0f} s/s "
                f"eta {eta:.0f} min",
                flush=True,
            )

    with mp.get_context("spawn").Pool(args.workers) as pool:
        pending: deque = deque()
        for task in tasks:
            while len(pending) >= 2 * args.workers:
                harvest(pending.popleft().get())
            pending.append(pool.apply_async(process_group, (task,)))
        while pending:
            harvest(pending.popleft().get())
    flush(force=True)
    writer.close()
    if v1_columns is not None and len(records) != len(v1_columns["piece_id"]):
        raise ValueError(
            f"hop-1.0 store has {len(v1_columns['piece_id'])} rows, this build kept "
            f"{len(records)}")

    elapsed = time.time() - start_time
    columns = {
        field.name: pa.array([record[field.name] for record in records], type=field.type)
        for field in INDEX_SCHEMA
    }
    pq.write_table(pa.table(columns, schema=INDEX_SCHEMA), out_dir / "index.parquet")

    if collect_stats:
        mean = stats_sum / stats_n
        std = np.sqrt(stats_sq / stats_n - mean * mean)
        np.savez(
            args.out_root / stage / "latent_stats.npz",
            mean=mean.astype(np.float32),
            std=std.astype(np.float32),
            n=np.int64(stats_n),
        )

    meta = {
        "stage": stage,
        "split": split,
        "source_dir": str(corpus / split),
        "source_files": [path.name for path in split_files(corpus, split)],
        "codec_checkpoint": str(CKPT_DIR),
        "max_windows": MAX_WINDOWS,
        "d_latent": D_LATENT,
        "hop": args.hop,
        "latents_per_second": lps,
        "from_v1": str(args.from_v1) if args.from_v1 else None,
        "encode_dtype": "bfloat16 autocast, mu stored float16",
        "n_rows": len(records),
        "n_seconds": n_seconds_total,
        "n_latents": n_latents_total,
        "skipped_lt2_windows": skipped,
        "cropped": cropped,
        "missing_context": missing_context,
        "elapsed_s": round(elapsed, 1),
        "seconds_per_second": round(n_seconds_total / elapsed, 1),
    }
    if args.shard is not None:
        meta["shard"] = list(args.shard)
        meta["file_blocks"] = file_blocks(records)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(
        f"[ok] {stage}/{split}: {len(records)} rows, {n_seconds_total} seconds, "
        f"{n_latents_total} latents, skipped {skipped}, cropped {cropped}, "
        f"{elapsed / 60:.1f} min, {n_seconds_total / elapsed:.0f} s of music / s"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=sorted(STAGES))
    parser.add_argument("--split", required=True)
    parser.add_argument("--hop", type=float, default=1.0, choices=[1.0, 0.5])
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--from-v1", type=Path,
                        help="hop-1.0 store split dir (hop 0.5 only): reuse its "
                             "integer-offset latents and encode the half-offset "
                             "windows only")
    parser.add_argument("--shard", type=parse_shard,
                        help="K/N: encode only the split's source files with "
                             "i %% N == K into <split>__shard{K}_{N}/")
    parser.add_argument("--merge", type=int,
                        help="N: merge <split>__shard{0..N-1}_{N}/ into <split>/ "
                             "and remove the shard dirs")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-windows", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.shard is not None and args.merge is not None:
        raise SystemExit("--shard builds one shard, --merge stitches them")
    if args.workers > 8:
        raise SystemExit("workers is capped at 8 repo-wide")
    if args.from_v1 and args.hop != 0.5:
        raise SystemExit("--from-v1 applies to hop 0.5 only")

    n_passes = STAGES[args.stage]
    known = (
        set(PRETRAIN_SPLITS) if args.stage == "pretrain"
        else {f"pass{i}" for i in range(n_passes)} | {"val"}
    )
    splits = sorted(known) if args.split == "all" else [args.split]
    for split in splits:
        if split not in known:
            raise SystemExit(f"{args.stage} has no split {split}")
    if args.from_v1 and len(splits) != 1:
        raise SystemExit("--from-v1 names one split directory, so build one split")

    if args.merge is not None:
        for split in splits:
            merge_shards(args.out_root, args.stage, split, args.merge)
        return

    vocab.verify()
    device = f"cuda:{args.gpu}"
    _config, model, ckpt_path = load_run(CKPT_DIR, device)
    model.eval()
    print(f"[info] codec {ckpt_path.name}")

    for split in splits:
        build_split(args, model, device, args.stage, split)


if __name__ == "__main__":
    main()
