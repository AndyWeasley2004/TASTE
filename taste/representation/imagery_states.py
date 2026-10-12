"""Cache frozen Qwen3 imagery states as per-piece [length, 2560] bf16 arrays.

Padding is removed; empty captions have length zero and use the trainer's null
context. Run --finalize after all distributed extraction shards finish.
"""

import argparse
import json
import os
import time
from datetime import date
from pathlib import Path

import pyarrow.parquet as pq

MODEL_NAME = "Qwen/Qwen3-Embedding-4B"
MODEL_REVISION = "5cf2132abc99cad020ac570b19d031efec650f2b"
STATES_DIRNAME = "qwen3_4b_imagery_states"
D_MODEL = 2560
MAX_LEN = 64


def row_shards(data_dir: Path) -> list[tuple[str, Path, Path]]:
    """(split, data shard, index shard) for every row shard, in a fixed order."""
    splits = sorted(p.name for p in data_dir.iterdir() if p.is_dir() and p.name != "index")
    shards = []
    for split in splits:
        data_files = sorted((data_dir / split).glob("*.parquet"))  # train-* and val-*
        for data_file in data_files:
            index_file = data_dir / "index" / split / data_file.name
            if not index_file.is_file():
                raise FileNotFoundError(f"missing index shard for {data_file}")
            shards.append((split, data_file, index_file))
    if not shards:
        raise FileNotFoundError(f"no row shards under {data_dir}")
    return shards


def first_occurrences(shards: list[tuple[str, Path, Path]]) -> list[list[int]]:
    """Row positions of the pieces each shard owns (first occurrence wins)."""
    seen: set[str] = set()
    owned = []
    for _, _, index_file in shards:
        pieces = pq.read_table(index_file, columns=["piece_id"]).column("piece_id").to_pylist()
        rows = []
        for row, piece in enumerate(pieces):
            if piece not in seen:
                seen.add(piece)
                rows.append(row)
        owned.append(rows)
    return owned


def encode_shard(model, tok, device, captions: list[str], batch_size: int):
    """[len, 2560] bf16 states, real-token length, and truncation flag per row."""
    import torch

    states, lengths, truncated = [], [], []
    for start in range(0, len(captions), batch_size):
        chunk = captions[start : start + batch_size]
        raw = tok(chunk, add_special_tokens=True)["input_ids"]
        enc = tok(
            chunk,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        mask = enc["attention_mask"].to(device)
        with torch.inference_mode():
            out = model(
                input_ids=enc["input_ids"].to(device), attention_mask=mask
            ).last_hidden_state
        lens = mask.sum(1).tolist()
        out = out.cpu()
        for j, caption in enumerate(chunk):
            states.append(out[j, : lens[j]].clone())
            lengths.append(0 if not caption.strip() else lens[j])
            truncated.append(len(raw[j]) > MAX_LEN)
    return states, lengths, truncated


def run_split(
    data_dir: Path,
    out_dir: Path,
    device_index: int,
    num_workers_split: int,
    split_rank: int,
    batch_size: int,
):
    import pandas as pd
    import torch
    from safetensors.torch import save_file
    from transformers import AutoModel, AutoTokenizer

    out_dir.mkdir(parents=True, exist_ok=True)
    shards = row_shards(data_dir)
    owned_rows = first_occurrences(shards)

    device = "cuda:0"  # CUDA_VISIBLE_DEVICES pins this to --device-index
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    tok.padding_side = "right"  # causal model: trailing pads cannot leak into real tokens
    model = AutoModel.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION, dtype=torch.bfloat16
    ).to(device)
    model.eval()

    rows = []
    t0 = time.time()
    n_done = 0
    for shard_index, ((split, data_file, index_file), row_positions) in enumerate(
        zip(shards, owned_rows)
    ):
        if shard_index % num_workers_split != split_rank:
            continue
        name = f"states_{split}_{data_file.stem}.safetensors"
        pieces = pq.read_table(index_file, columns=["piece_id"]).column("piece_id").to_pylist()
        captions = pq.read_table(data_file, columns=["imagery"]).column("imagery").to_pylist()
        pieces = [pieces[row] for row in row_positions]
        captions = [captions[row] or "" for row in row_positions]

        states, lengths, truncated = encode_shard(model, tok, device, captions, batch_size)
        save_file(dict(zip(pieces, states)), str(out_dir / name))
        rows.extend(zip(pieces, [name] * len(pieces), lengths, truncated))

        n_done += len(pieces)
        elapsed = time.time() - t0
        print(
            f"[split {split_rank}] shard {shard_index} {name} "
            f"{len(pieces)} pieces | {n_done} total | {n_done/max(elapsed,1e-9):.1f} pieces/s",
            flush=True,
        )

    manifest = pd.DataFrame(rows, columns=["piece_id", "shard_file", "length", "truncated"])
    manifest.to_parquet(out_dir / f"manifest_part{split_rank}.parquet", index=False)
    print(
        f"[split {split_rank}] done {len(manifest)} pieces in {time.time()-t0:.0f}s "
        f"| peak VRAM {torch.cuda.max_memory_allocated()/2**30:.1f} GiB",
        flush=True,
    )


def finalize(data_dir: Path, out_dir: Path):
    import pandas as pd

    parts = sorted(out_dir.glob("manifest_part*.parquet"))
    manifest = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    if manifest["piece_id"].duplicated().any():
        raise ValueError("duplicate piece_id across manifest parts")
    manifest = manifest.sort_values("piece_id").reset_index(drop=True)
    manifest[["piece_id", "shard_file", "length"]].to_parquet(
        out_dir / "manifest.parquet", index=False
    )

    meta = {
        "model": MODEL_NAME,
        "revision": MODEL_REVISION,
        "layer": "final (last_hidden_state), per token",
        "d_model": D_MODEL,
        "dtype": "bfloat16",
        "max_len": MAX_LEN,
        "n_pieces": int(len(manifest)),
        "null_context_pieces": int((manifest["length"] == 0).sum()),
        "truncated": int(manifest["truncated"].sum()),
        "source_corpus": str(data_dir),
        "date": date.today().isoformat(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--device-index", type=int, default=0)
    ap.add_argument("--num-workers-split", type=int, default=1)
    ap.add_argument("--split-rank", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--finalize", action="store_true")
    args = ap.parse_args()
    out_dir = args.out_dir or args.data_dir / STATES_DIRNAME

    if args.finalize:
        finalize(args.data_dir, out_dir)
        return
    if not 0 <= args.split_rank < args.num_workers_split:
        raise ValueError("--split-rank must lie in [0, --num-workers-split)")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_index)
    run_split(
        args.data_dir,
        out_dir,
        args.device_index,
        args.num_workers_split,
        args.split_rank,
        args.batch_size,
    )


if __name__ == "__main__":
    main()
