"""Conditional training of the AR generator on one paired (prompt, MIDI) stage.

The same trainer runs the large-public, curated and high-quality stages: rows
are batched under a per-device token budget, the corpus passes are emitted as
ordered blocks under one cosine schedule, the model is the conditional wrapper
from ``taste.models.ar_conditional`` initialised from the previous stage's
checkpoint, and every batch carries attribute class indices plus precomputed
Qwen3-Embedding-4B imagery states.

The run may switch between 4 and 8 GPUs. Gradient accumulation scales so the
global token batch is constant, and a cursor records which rows each block has
already produced so the remaining rows can be replanned after a switch.

``format_version`` comes from the config and is checked against the corpus
``meta.json``. The pass count and block layout come from ``meta.json``: a corpus
with a ``strata`` entry splits pass 0 into one block per stratum, otherwise
each pass is one block. The cursor carries a stage tag derived from
``format_version`` so one stage's cursor can never replay on another's corpus.
"""
from __future__ import annotations

import argparse
import base64
import glob
import json
import math
import os
import re
import shutil
import time
import zlib
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import partial
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
REPO = Path(__file__).resolve().parents[2]

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from safetensors import safe_open
from torch.utils.data import BatchSampler, DataLoader
from transformers import Trainer, TrainerCallback, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint, seed_worker

from taste import paths
from taste.models.ar_conditional import (
    ATTRIBUTE_FIELDS,
    CONTEXT_DIM,
    load_attribute_classes,
    load_conditional_model,
    null_indices,
    parameter_report,
)
from taste.representation.vocab import PAD_ID, save_vocab_meta

SEQ_LEN = 4096
# the LR horizon is defined at a canonical world size so that 4- and 8-GPU
# runs share one schedule; only tail padding depends on the world size
CANONICAL_WORLD_SIZE = 4

# RTX A6000 dense BF16 tensor peak, used only for the logged approximate MFU
A6000_BF16_DENSE_FLOPS = 154.8e12

IMAGERY_STATES_DIRNAME = "qwen3_4b_imagery_states"


@dataclass
class Cfg:
    format_version: str = "taste_4k_large_public_25ms"
    data_dir: str = "corpora/large_public"  # relative to TASTE_DATA_ROOT
    init_checkpoint: str = "ar/pretrain/final"  # relative to TASTE_CKPT_ROOT
    seq_len: int = SEQ_LEN
    lr: float = 1.5e-4
    min_lr_rate: float = 0.1
    weight_decay: float = 0.1
    warmup_ratio: float = 0.01
    max_grad_norm: float = 1.0
    # grad_accum is derived so that every world size sees the same
    # global_tokens_per_step optimizer step
    global_tokens_per_step: int = 589_824
    tokens_per_device_batch: int = 12_288
    pad_to_multiple: int = 256
    per_device_eval_batch: int = 4
    out_dir: str = "ar/large_public"  # relative to TASTE_CKPT_ROOT
    run_dir: str = "./runs/taste_ar_large_public"
    logging_steps: int = 20
    eval_steps: int = 500
    eval_on_start: bool = True
    save_steps: int = 500
    archive_every: int = 5000
    dataloader_workers: int = 4
    dataloader_prefetch_factor: int = 4
    seed: int = 0
    max_steps: int = -1
    allowed_world_sizes: tuple = (4, 8)


def stage_name(format_version: str) -> str:
    """``taste_4k_large_public_25ms`` -> ``large_public``."""
    match = re.fullmatch(r"taste_\d+k_(\w+)_\d+ms", format_version)
    if match is None:
        raise ValueError(f"cannot read a stage name out of {format_version!r}")
    return match.group(1)


def cursor_format(format_version: str) -> str:
    """Per-stage cursor tag: one stage's cursor must not replay on another."""
    return f"{stage_name(format_version)}_cursor"


def hms(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def rounded(value):
    return round(value, 6) if isinstance(value, float) else value


# --------------------------------------------------------------------------
# deterministic sampler plan
# --------------------------------------------------------------------------


@dataclass
class SamplerPlan:
    """The full emission order as flat per-rank batches.

    ``indices[offsets[k]:offsets[k+1]]`` is the k-th per-rank batch; Accelerate
    hands consecutive groups of ``world_size`` batches to the ranks, so batch
    ``k`` belongs to optimizer step ``k // (world_size * grad_accum)``.
    """

    indices: np.ndarray
    offsets: np.ndarray
    blocks: np.ndarray

    def __len__(self) -> int:
        return len(self.offsets) - 1


def block_row_index(block_ids: np.ndarray, n_blocks: int) -> list[np.ndarray]:
    """Row indices of each block, ascending — the bitmap coordinate system."""
    order = np.argsort(block_ids, kind="stable")
    bounds = np.searchsorted(block_ids[order], np.arange(n_blocks + 1))
    return [order[bounds[block] : bounds[block + 1]] for block in range(n_blocks)]


def block_seed(seed: int, block: int, generation: int) -> list[int]:
    return [int(seed), int(block), int(generation)]


def build_plan(
    block_rows: list[np.ndarray],
    padded_lengths: np.ndarray,
    allowed: list[np.ndarray],
    *,
    tokens_per_device_batch: int,
    world_size: int,
    seed: int,
    generation: int,
) -> SamplerPlan:
    """Blocks in fixed order; inside a block, same-shape batches in random order.

    ``allowed[block]`` is a boolean mask over ``block_rows[block]`` positions.
    Rows are grouped by padded length so a batch has one shape, chopped into
    ``world_size``-sized groups of equal-length rank batches, and a short tail
    is filled by repeating rows of the same group.
    """
    batches: list[np.ndarray] = []
    batch_blocks: list[int] = []
    for block, rows in enumerate(block_rows):
        rows = rows[allowed[block]]
        rng = np.random.default_rng(block_seed(seed, block, generation))
        pads = padded_lengths[rows]
        order = np.argsort(pads, kind="stable")
        rows, pads = rows[order], pads[order]
        unique_pads, starts = np.unique(pads, return_index=True)
        starts = np.append(starts, len(rows))

        block_batches: list[np.ndarray] = []
        for group, pad in enumerate(unique_pads):
            indices = rows[starts[group] : starts[group + 1]]
            indices = indices[rng.permutation(len(indices))]
            local_batch = max(1, tokens_per_device_batch // int(pad))
            global_block = local_batch * world_size
            n_groups = math.ceil(len(indices) / global_block)
            short = n_groups * global_block - len(indices)
            if short:
                repeats = math.ceil(short / len(indices))
                indices = np.concatenate((indices, np.tile(indices, repeats)[:short]))
            block_batches.extend(
                indices.reshape(n_groups, world_size, local_batch)
            )
        order = rng.permutation(len(block_batches))
        emitted = [
            block_batches[group][rank] for group in order for rank in range(world_size)
        ]
        batches.extend(emitted)
        batch_blocks.extend([block] * len(emitted))

    sizes = np.fromiter((len(batch) for batch in batches), dtype=np.int64, count=len(batches))
    offsets = np.zeros(len(batches) + 1, dtype=np.int64)
    np.cumsum(sizes, out=offsets[1:])
    return SamplerPlan(
        indices=np.concatenate(batches).astype(np.int64),
        offsets=offsets,
        blocks=np.asarray(batch_blocks, dtype=np.int16),
    )


def replay_consumed(
    plan: SamplerPlan,
    block_rows: list[np.ndarray],
    n_batches: int,
) -> list[np.ndarray]:
    """Rows already emitted after ``n_batches`` per-rank batches, per block."""
    n_batches = min(int(n_batches), len(plan))
    seen = np.zeros(sum(len(rows) for rows in block_rows), dtype=bool)
    seen[plan.indices[: plan.offsets[n_batches]]] = True
    return [seen[rows] for rows in block_rows]


def encode_bitmap(mask: np.ndarray) -> str:
    return base64.b64encode(zlib.compress(np.packbits(mask).tobytes(), 6)).decode()


def decode_bitmap(payload: str, n_rows: int) -> np.ndarray:
    raw = zlib.decompress(base64.b64decode(payload))
    return np.unpackbits(np.frombuffer(raw, dtype=np.uint8))[:n_rows].astype(bool)


class _EpochSetter:
    def __init__(self, batch_sampler: "PlanBatchSampler"):
        self.batch_sampler = batch_sampler

    def set_epoch(self, epoch: int) -> None:
        self.batch_sampler.set_epoch(epoch)


class PlanBatchSampler(BatchSampler):
    """Replays a fixed ``SamplerPlan``; the plan is the run's whole data order."""

    batch_size = None
    drop_last = False

    def __init__(self, plan: SamplerPlan):
        self.plan = plan
        self.epoch = 0
        self.sampler = _EpochSetter(self)

    def __len__(self) -> int:
        return len(self.plan)

    def set_epoch(self, epoch: int) -> None:
        # the passes are already one ordered plan; there is no reshuffle
        self.epoch = int(epoch)

    def __iter__(self):
        indices, offsets = self.plan.indices, self.plan.offsets
        for batch in range(len(self.plan)):
            yield indices[offsets[batch] : offsets[batch + 1]].tolist()


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

# safetensors handles are opened lazily and kept per dataloader worker process
_STATE_HANDLES: dict[int, object] = {}


def context_handle(context_dir: str, shard_files: list[str], shard: int):
    handle = _STATE_HANDLES.get(shard)
    if handle is None:
        handle = safe_open(os.path.join(context_dir, shard_files[shard]), framework="pt")
        _STATE_HANDLES[shard] = handle
    return handle


def read_context_manifest(context_dir: str) -> tuple[list[str], dict[str, tuple[int, int]]]:
    table = pq.read_table(os.path.join(context_dir, "manifest.parquet")).to_pydict()
    shard_files = sorted(set(table["shard_file"]))
    shard_id = {name: index for index, name in enumerate(shard_files)}
    entries = {
        piece: (shard_id[shard], length)
        for piece, shard, length in zip(
            table["piece_id"], table["shard_file"], table["length"]
        )
    }
    return shard_files, entries


def prompt_columns(
    data_files: list[str],
    index_files: list[str],
    n_rows: int,
    classes: dict[str, list[str]],
    nulls: dict[str, int],
    context_entries: dict[str, tuple[int, int]],
    block_ids: np.ndarray,
) -> pa.Table:
    """Per-row conditioning columns, read without touching the token payload."""
    prompts = pq.read_table(
        data_files, columns=[*ATTRIBUTE_FIELDS, "prefix_dropped"]
    ).to_pydict()
    pieces = pq.read_table(index_files, columns=["piece_id"]).column("piece_id").to_pylist()
    if len(pieces) != n_rows or len(prompts["prefix_dropped"]) != n_rows:
        raise ValueError("index and data shards disagree on row count")

    dropped = np.asarray(prompts["prefix_dropped"], dtype=bool)
    arrays = {"piece_id": pa.array(pieces, type=pa.string())}
    for field_name in ATTRIBUTE_FIELDS:
        lookup = {name: index for index, name in enumerate(classes[field_name])}
        null = nulls[field_name]
        arrays[f"{field_name}_id"] = pa.array(
            [
                null if (value is None or is_dropped) else lookup[value]
                for value, is_dropped in zip(prompts[field_name], dropped)
            ],
            type=pa.int16(),
        )
    # a dropped prefix keeps only the learned null context token; so does a
    # piece whose caption was empty, which the manifest records as length 0
    arrays["context_len"] = pa.array(
        [0 if is_dropped else context_entries[piece][1] for piece, is_dropped in zip(pieces, dropped)],
        type=pa.int16(),
    )
    arrays["context_shard"] = pa.array(
        [context_entries[piece][0] for piece in pieces], type=pa.int16()
    )
    arrays["block"] = pa.array(block_ids, type=pa.int16())
    return pa.table(arrays)


def strata_layout(meta: dict) -> list[str] | None:
    """The pass-0 stratum names in block order, or None for a flat corpus."""
    if "strata" not in meta:
        return None
    layout = meta["strata"]["pass0_layout"]
    return sorted(layout, key=lambda name: layout[name]["row_start"])


def block_layout(meta: dict) -> tuple[int, list[str] | None, int]:
    """Passes, pass-0 strata (or None) and the resulting block count.

    A stratified pass 0 becomes one block per stratum; a corpus without a
    ``strata`` entry gets one block per pass.
    """
    n_passes = int(meta["n_passes"])
    strata = strata_layout(meta)
    n_strata = len(strata) if strata else 1
    return n_passes, strata, n_strata + (n_passes - 1)


def pass0_block_ids(meta: dict, n_rows: int) -> np.ndarray:
    """pass0 is written stratum-ordered; the boundaries are declared in meta."""
    block_ids = np.full(n_rows, -1, dtype=np.int16)
    layout = meta["strata"]["pass0_layout"]
    for block, name in enumerate(strata_layout(meta)):
        span = layout[name]
        block_ids[span["row_start"] : span["row_end"]] = block
    if (block_ids < 0).any():
        raise ValueError("pass0 stratum layout does not cover every row")
    return block_ids


def build_datasets(cfg: Cfg, meta: dict):
    classes = load_attribute_classes()
    nulls = null_indices(classes)
    context_dir = os.path.join(cfg.data_dir, IMAGERY_STATES_DIRNAME)
    shard_files, context_entries = read_context_manifest(context_dir)

    n_passes, strata, n_blocks = block_layout(meta)
    n_strata = n_blocks - n_passes + 1

    parts = []
    for pass_index in range(n_passes):
        name = f"pass{pass_index}"
        data_files = sorted(glob.glob(os.path.join(cfg.data_dir, name, "train-*.parquet")))
        index_files = sorted(glob.glob(os.path.join(cfg.data_dir, "index", name, "train-*.parquet")))
        if not data_files or len(data_files) != len(index_files):
            raise FileNotFoundError(f"missing or mismatched shards for {name}")
        rows = load_dataset("parquet", data_files=data_files, split="train")
        if pass_index == 0 and strata:
            block_ids = pass0_block_ids(meta, len(rows))
            if (np.asarray(rows["stratum"], dtype=np.int16) - 1 != block_ids).any():
                raise ValueError("pass0 stratum column disagrees with the meta layout")
        else:
            block_ids = np.full(len(rows), n_strata - 1 + pass_index, dtype=np.int16)
        extra = prompt_columns(
            data_files, index_files, len(rows), classes, nulls, context_entries, block_ids,
        )
        parts.append(concatenate_datasets([rows, Dataset(extra)], axis=1))
    train = concatenate_datasets(parts)

    val_data = sorted(glob.glob(os.path.join(cfg.data_dir, "val", "val-*.parquet")))
    val_index = sorted(glob.glob(os.path.join(cfg.data_dir, "index", "val", "val-*.parquet")))
    if not val_data or len(val_data) != len(val_index):
        raise FileNotFoundError("missing or mismatched val shards")
    val_rows = load_dataset("parquet", data_files=val_data, split="train")
    val = concatenate_datasets(
        [
            val_rows,
            Dataset(
                prompt_columns(
                    val_data, val_index, len(val_rows), classes, nulls, context_entries,
                    np.zeros(len(val_rows), dtype=np.int16),
                )
            ),
        ],
        axis=1,
    ).sort("n_tokens")

    keep = [
        "input_ids", "n_tokens", "block", "piece_id", "context_len", "context_shard",
        *[f"{field_name}_id" for field_name in ATTRIBUTE_FIELDS],
    ]
    train = train.select_columns(keep)
    val = val.select_columns(keep)
    if max(train["n_tokens"]) > cfg.seq_len:
        raise ValueError(f"row longer than the {cfg.seq_len}-token context")
    return train, val, context_dir, shard_files, n_blocks


def collate(features, *, pad_to_multiple: int, seq_len: int, context_dir: str, shard_files: list[str]):
    longest = max(len(feature["input_ids"]) for feature in features)
    target = math.ceil(longest / pad_to_multiple) * pad_to_multiple
    if target > seq_len:
        raise ValueError(f"padded length {target} exceeds the {seq_len}-token context")

    input_ids, labels, attention_mask = [], [], []
    for feature in features:
        ids = torch.as_tensor(feature["input_ids"], dtype=torch.long)
        padding = target - len(ids)
        input_ids.append(torch.cat((ids, torch.full((padding,), PAD_ID, dtype=torch.long))))
        labels.append(torch.cat((ids, torch.full((padding,), -100, dtype=torch.long))))
        attention_mask.append(torch.cat((
            torch.ones(len(ids), dtype=torch.long),
            torch.zeros(padding, dtype=torch.long),
        )))

    context_lengths = torch.tensor(
        [int(feature["context_len"]) for feature in features], dtype=torch.long
    )
    context = torch.zeros(
        (len(features), max(1, int(context_lengths.max())), CONTEXT_DIM),
        dtype=torch.bfloat16,
    )
    for row, feature in enumerate(features):
        length = int(feature["context_len"])
        if length:
            handle = context_handle(context_dir, shard_files, int(feature["context_shard"]))
            context[row, :length] = handle.get_tensor(feature["piece_id"])

    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_mask),
        "attribute_ids": torch.tensor(
            [[int(feature[f"{name}_id"]) for name in ATTRIBUTE_FIELDS] for feature in features],
            dtype=torch.long,
        ),
        "context_states": context,
        "context_lengths": context_lengths,
    }


# --------------------------------------------------------------------------
# callbacks
# --------------------------------------------------------------------------


class ConciseLogger(TrainerCallback):
    """One human line per log plus a metrics.jsonl record, main process only."""

    def __init__(self, run_dir: str, total_steps: int, token_counter: dict, stage: str):
        self.stage = stage
        self.metrics_path = os.path.join(run_dir, "metrics.jsonl")
        self.log_path = os.path.join(run_dir, "train_log.txt")
        self.total = max(1, total_steps)
        self.counter = token_counter
        self.started = self.last_time = None
        self.last_tokens = 0
        self.last_flos = 0.0

    def write(self, line: str) -> None:
        print(line, flush=True)
        with open(self.log_path, "a") as handle:
            handle.write(line + "\n")

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        self.started = self.last_time = time.time()
        self.last_tokens = self.counter["tokens"]
        self.last_flos = state.total_flos
        self.write(
            f"=== {self.stage} stage start {datetime.now():%Y-%m-%d %H:%M:%S} | "
            f"total_steps {self.total} ==="
        )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or not logs:
            return
        now, step = time.time(), state.global_step
        with open(self.metrics_path, "a") as handle:
            handle.write(json.dumps({
                "step": step,
                "t": round(now - self.started, 1),
                **{name: rounded(value) for name, value in logs.items()},
            }) + "\n")

        if "loss" in logs:
            world = int(os.environ.get("WORLD_SIZE", 1))
            delta_time = max(now - self.last_time, 1e-6)
            delta_tokens = self.counter["tokens"] - self.last_tokens
            tok_s = delta_tokens * world / delta_time
            step_mfu = (
                max(0.0, state.total_flos - self.last_flos)
                / delta_time / world / A6000_BF16_DENSE_FLOPS
            )
            self.last_time, self.last_tokens = now, self.counter["tokens"]
            self.last_flos = state.total_flos
            self.write(
                f"[{step:>6}/{self.total} {100 * step / self.total:4.1f}%] "
                f"loss {logs['loss']:.3f} "
                f"ppl {math.exp(min(20.0, logs['loss'])):7.1f} "
                f"lr {logs.get('learning_rate', 0):.2e} "
                f"gnorm {logs.get('grad_norm', 0):.2f} | "
                f"{tok_s / 1e3:.0f}k tok/s approx-MFU {100 * step_mfu:.1f}% | "
                f"{hms(now - self.started)}"
            )

        if "eval_loss" in logs:
            loss = logs["eval_loss"]
            self.write(
                f"   >> eval @ {step}: {loss:.4f} (ppl {math.exp(min(20.0, loss)):.1f})"
            )

    def on_train_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero or self.started is None:
            return
        elapsed = max(time.time() - self.started, 1e-6)
        world = int(os.environ.get("WORLD_SIZE", 1))
        mfu = state.total_flos / elapsed / world / A6000_BF16_DENSE_FLOPS
        peak_gib = torch.cuda.max_memory_allocated() / 2**30
        self.write(
            f"=== {self.stage} stage end | peak {peak_gib:.2f} GiB/GPU | "
            f"overall approx MFU {100 * mfu:.1f}% | {hms(elapsed)} ==="
        )


class DataCursorCallback(TrainerCallback):
    """Data cursor: a per-block consumed bitmap replayed from the sampler plan.

    The bitmaps are the only thing a world-size switch needs: the plan is a
    pure function of ``(seed, block, generation, allowed)``, so the surviving
    rows can be replanned at the new world size while the optimizer and
    scheduler carry over untouched.
    """

    def __init__(
        self,
        run_dir: str,
        cfg: Cfg,
        world_size: int,
        grad_accum: int,
        generation: int,
        plan_start_step: int,
        plan: SamplerPlan,
        block_rows: list[np.ndarray],
        allowed: list[np.ndarray],
    ):
        self.path = Path(run_dir) / "data_cursor.json"
        self.cfg = cfg
        self.world_size = world_size
        self.grad_accum = grad_accum
        self.generation = generation
        self.plan_start_step = plan_start_step
        self.plan = plan
        self.block_rows = block_rows
        self.allowed = allowed

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        # the plan begins at plan_start_step (0 for generation 0); global_step
        # is cumulative across generations, so only the delta indexes the plan
        steps_into_plan = state.global_step - self.plan_start_step
        if steps_into_plan < 0:
            raise ValueError(
                f"global_step {state.global_step} precedes plan start "
                f"{self.plan_start_step}"
            )
        consumed = replay_consumed(
            self.plan, self.block_rows, steps_into_plan * self.grad_accum * self.world_size
        )
        # rows consumed in earlier generations stay consumed
        n_blocks = len(self.block_rows)
        for block in range(n_blocks):
            consumed[block] |= ~self.allowed[block]
        payload = {
            "format": cursor_format(self.cfg.format_version),
            "global_step": state.global_step,
            "seed": self.cfg.seed,
            "world_size": self.world_size,
            "grad_accum": self.grad_accum,
            "generation": self.generation,
            "plan_start_step": self.plan_start_step,
            "checkpoint": str(Path(args.output_dir) / f"checkpoint-{state.global_step}"),
            "blocks": [
                {
                    "block": block,
                    "n_rows": int(len(self.block_rows[block])),
                    "n_allowed": int(self.allowed[block].sum()),
                    "n_consumed": int(consumed[block].sum()),
                    "allowed": encode_bitmap(self.allowed[block]),
                    "consumed": encode_bitmap(consumed[block]),
                }
                for block in range(n_blocks)
            ],
        }
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload) + "\n")
        temporary.replace(self.path)


RESUME_ONLY_FILES = {
    "optimizer.pt",
    "scheduler.pt",
    "scaler.pt",
    "trainer_state.json",
    "training_args.bin",
}


def save_stage_meta(destination: str | Path, cfg: Cfg) -> None:
    """Everything a sampling script needs beyond the HF weights."""
    destination = Path(destination)
    save_vocab_meta(destination)
    shutil.copy2(REPO / "attributes.json", destination / "attributes.json")
    context_dir = Path(cfg.data_dir) / IMAGERY_STATES_DIRNAME
    (destination / "imagery_encoder.json").write_text(
        json.dumps(
            {
                "attribute_fields": list(ATTRIBUTE_FIELDS),
                "imagery_states_dir": str(context_dir),
                "imagery_states_meta": json.load(open(context_dir / "meta.json")),
            },
            indent=2,
        )
        + "\n"
    )


class CheckpointRetention(TrainerCallback):
    """Custom retention: this callback is the only deleter of checkpoints."""

    def __init__(self, out_dir: str, archive_every: int, cfg: Cfg):
        if archive_every <= 0:
            raise ValueError("archive_every must be positive")
        self.out_dir = Path(out_dir)
        self.archive_every = int(archive_every)
        self.cfg = cfg

    # rolling window: the RETAIN_LAST most recent saves are kept as full
    # resume checkpoints; model-only archives at archive_every multiples are
    # written before deletion and do not count toward the window
    RETAIN_LAST = 3

    def archive(self, step: int) -> None:
        source = self.out_dir / f"checkpoint-{step}"
        destination = self.out_dir / "model_only" / f"step{step}"
        destination.mkdir(parents=True, exist_ok=True)
        for path in sorted(source.iterdir()):
            if not path.is_file() or path.name.startswith("rng_state"):
                continue
            if path.name not in RESUME_ONLY_FILES:
                shutil.copy2(path, destination / path.name)
        save_stage_meta(destination, self.cfg)
        print(f"[retention] archived model-only weights to {destination}", flush=True)

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        current_step = state.global_step
        save_stage_meta(self.out_dir / f"checkpoint-{current_step}", self.cfg)
        if current_step % self.archive_every == 0:
            self.archive(current_step)
        steps = sorted(
            int(path.name.split("-")[1])
            for path in self.out_dir.glob("checkpoint-*")
            if path.is_dir()
        )
        for step in steps[: -self.RETAIN_LAST]:
            if step % self.archive_every == 0:
                archived = self.out_dir / "model_only" / f"step{step}"
                if not archived.is_dir():
                    raise RuntimeError(
                        f"refusing to delete checkpoint-{step}: expected "
                        f"model-only archive {archived} is missing"
                    )
            shutil.rmtree(self.out_dir / f"checkpoint-{step}", ignore_errors=True)
            print(f"[retention] deleted checkpoint-{step}", flush=True)


# --------------------------------------------------------------------------
# trainer
# --------------------------------------------------------------------------


class PlanBlockTrainer(Trainer):
    def __init__(self, *args, cfg: Cfg, plan: SamplerPlan, token_counter: dict, **kwargs):
        self.cfg = cfg
        self.plan = plan
        self.token_counter = token_counter
        super().__init__(*args, **kwargs)

    def get_train_dataloader(self) -> DataLoader:
        dataloader_kwargs = {
            "batch_sampler": PlanBatchSampler(self.plan),
            "collate_fn": self.data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "worker_init_fn": partial(
                seed_worker,
                num_workers=self.args.dataloader_num_workers,
                rank=self.args.process_index,
            ),
        }
        if self.args.dataloader_num_workers > 0:
            dataloader_kwargs["prefetch_factor"] = self.args.dataloader_prefetch_factor
        self.accelerator.even_batches = False
        return self.accelerator.prepare(DataLoader(self.train_dataset, **dataloader_kwargs))

    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        previous_even_batches = self.accelerator.even_batches
        self.accelerator.even_batches = True
        try:
            return super().get_eval_dataloader(eval_dataset)
        finally:
            self.accelerator.even_batches = previous_even_batches

    def training_step(self, model, inputs, num_items_in_batch=None):
        self.token_counter["tokens"] += inputs["input_ids"].numel()
        return super().training_step(model, inputs, num_items_in_batch)


def build_model(cfg: Cfg):
    from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_qwen3

    # the model's own fused linear CE must not be replaced by Liger's CE patches
    apply_liger_kernel_to_qwen3(
        rope=True,
        cross_entropy=False,
        fused_linear_cross_entropy=False,
        rms_norm=True,
        swiglu=True,
    )
    model = load_conditional_model(cfg.init_checkpoint, attn_implementation="sdpa")
    if int(model.config.max_position_embeddings) != cfg.seq_len:
        raise ValueError("model and corpus context lengths differ")
    model.config.use_cache = False
    return model


def resolve_plan(cfg: Cfg, train, world_size: int, run_dir: str, n_blocks: int):
    """Pick the resume path and produce the plan the run will follow.

    Fresh runs and same-world resumes replay generation ``g`` unchanged, so a
    same-world resume is bit-exact through Trainer's deterministic data skip. A
    4<->8 switch starts generation ``g+1`` over the rows no block has consumed.
    """
    block_ids = np.asarray(train["block"], dtype=np.int16)
    block_rows = block_row_index(block_ids, n_blocks)
    padded = (
        np.ceil(np.asarray(train["n_tokens"], dtype=np.int64) / cfg.pad_to_multiple)
        * cfg.pad_to_multiple
    ).astype(np.int64)

    cursor_path = Path(run_dir) / "data_cursor.json"
    if not cursor_path.exists():
        mode = "fresh"
        generation = 0
        plan_start_step = 0
        allowed = [np.ones(len(rows), dtype=bool) for rows in block_rows]
    else:
        cursor = json.load(open(cursor_path))
        expected = cursor_format(cfg.format_version)
        if cursor["format"] != expected:
            raise ValueError(
                f"cursor format {cursor['format']!r} is not this stage's {expected!r}"
            )
        if cursor["seed"] != cfg.seed:
            raise ValueError("cursor seed differs from the configured seed")
        blocks = cursor["blocks"]
        if cursor["world_size"] == world_size and cursor["generation"] == 0:
            # generation 0 at the original world size: HF's deterministic data
            # skip replays the identical plan, so the resume is bit-exact
            mode = "same-world"
            generation = 0
            plan_start_step = 0
            allowed = [
                decode_bitmap(entry["allowed"], entry["n_rows"]) for entry in blocks
            ]
        else:
            # any resume past a world switch starts a fresh generation over the
            # unconsumed rows: global_step is cumulative across generations, so
            # a later-generation plan cannot be replayed by HF's step-indexed
            # skip (statistically clean, no longer bit-exact)
            mode = (
                f"world-switch {cursor['world_size']} -> {world_size}"
                if cursor["world_size"] != world_size
                else f"regen-resume (generation {cursor['generation']})"
            )
            generation = cursor["generation"] + 1
            plan_start_step = int(cursor["global_step"])
            allowed = [
                ~decode_bitmap(entry["consumed"], entry["n_rows"]) for entry in blocks
            ]
        if len(blocks) != n_blocks:
            raise ValueError(
                f"cursor has {len(blocks)} blocks; this corpus has {n_blocks}"
            )
        for block, entry in enumerate(blocks):
            if entry["n_rows"] != len(block_rows[block]):
                raise ValueError(f"cursor block {block} row count differs from the corpus")

    plan = build_plan(
        block_rows, padded, allowed,
        tokens_per_device_batch=cfg.tokens_per_device_batch,
        world_size=world_size,
        seed=cfg.seed,
        generation=generation,
    )
    return mode, generation, plan_start_step, block_rows, padded, allowed, plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one conditional AR stage from a config.")
    parser.add_argument("--run-name", default="taste_ar_large_public",
                        help="logs and the data cursor go to runs/<run-name> under the working directory")
    parser.add_argument("--config", help="e.g. configs/ar/large_public.json")
    args = parser.parse_args()

    cfg = Cfg()
    if args.config:
        for name, value in json.load(open(args.config)).items():
            if not hasattr(cfg, name):
                raise ValueError(f"unknown config field: {name}")
            setattr(cfg, name, value)
    cfg.run_dir = f"./runs/{args.run_name}"
    cfg.data_dir = str(paths.data(cfg.data_dir))
    cfg.init_checkpoint = str(paths.ckpt(cfg.init_checkpoint))
    cfg.out_dir = str(paths.ckpt(cfg.out_dir))

    data_meta = json.load(open(os.path.join(cfg.data_dir, "meta.json")))
    if data_meta["format_version"] != cfg.format_version:
        raise ValueError(
            f"{cfg.data_dir} uses format {data_meta['format_version']!r}; "
            f"the config asks for {cfg.format_version!r}"
        )
    stage = stage_name(cfg.format_version)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size not in tuple(cfg.allowed_world_sizes):
        raise ValueError(
            f"the AR stage trainer supports world sizes {tuple(cfg.allowed_world_sizes)}; "
            f"found {world_size}"
        )
    grad_accum, remainder = divmod(
        cfg.global_tokens_per_step, cfg.tokens_per_device_batch * world_size
    )
    if remainder or grad_accum < 1:
        raise ValueError(
            f"global_tokens_per_step {cfg.global_tokens_per_step} is not an integer "
            f"multiple of {cfg.tokens_per_device_batch} x world {world_size}"
        )

    is_main = int(os.environ.get("RANK", 0)) == 0
    if is_main:
        os.makedirs(cfg.run_dir, exist_ok=True)
        os.makedirs(cfg.out_dir, exist_ok=True)
        save_stage_meta(cfg.out_dir, cfg)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = build_model(cfg)
    train, val, context_dir, shard_files, n_blocks = build_datasets(cfg, data_meta)

    mode, generation, plan_start_step, block_rows, padded, allowed, plan = resolve_plan(
        cfg, train, world_size, cfg.run_dir, n_blocks
    )
    resume_checkpoint = None if mode == "fresh" else get_last_checkpoint(cfg.out_dir)
    if mode != "fresh":
        if resume_checkpoint is None:
            raise ValueError("cursor exists but no checkpoint found to resume from")
        checkpoint_step = json.load(
            open(os.path.join(resume_checkpoint, "trainer_state.json"))
        )["global_step"]
        if generation > 0 and checkpoint_step != plan_start_step:
            raise ValueError(
                f"latest checkpoint is at step {checkpoint_step} but the cursor "
                f"was written at step {plan_start_step}; refusing to resume with "
                f"a misaligned plan offset"
            )
    # the LR horizon is fixed at the canonical world size so a 4<->8 switch
    # inherits an unchanged schedule
    horizon_plan = (
        plan
        if world_size == CANONICAL_WORLD_SIZE and mode == "fresh"
        else build_plan(
            block_rows, padded,
            [np.ones(len(rows), dtype=bool) for rows in block_rows],
            tokens_per_device_batch=cfg.tokens_per_device_batch,
            world_size=CANONICAL_WORLD_SIZE,
            seed=cfg.seed,
            generation=0,
        )
    )
    horizon_steps = len(horizon_plan) // (
        CANONICAL_WORLD_SIZE * (cfg.global_tokens_per_step // (cfg.tokens_per_device_batch * CANONICAL_WORLD_SIZE))
    )
    total_steps = cfg.max_steps if cfg.max_steps > 0 else horizon_steps

    summary = {
        "stage": stage,
        "format_version": cfg.format_version,
        "n_passes": int(data_meta["n_passes"]),
        "n_blocks": n_blocks,
        "resume_mode": mode,
        "generation": generation,
        "plan_start_step": plan_start_step,
        "world_size": world_size,
        "grad_accum": grad_accum,
        "global_tokens_per_step": cfg.global_tokens_per_step,
        "plan_batches": len(plan),
        "plan_steps": len(plan) // (world_size * grad_accum),
        "horizon_steps_at_world4": horizon_steps,
        "planned_total_steps": total_steps,
        # the nominal figure ignores the pad-to-multiple overhead and the floor
        # in tokens_per_device_batch // padded_length
        "nominal_steps_from_token_count": round(
            int(np.asarray(train["n_tokens"], dtype=np.int64).sum())
            / cfg.global_tokens_per_step
        ),
        "corpus_tokens": int(np.asarray(train["n_tokens"], dtype=np.int64).sum()),
        "n_train_rows": len(train),
        "n_val_rows": len(val),
        "rows_allowed_per_block": [int(mask.sum()) for mask in allowed],
    }
    if is_main:
        print(f"[resume] path: {mode} (generation {generation})", flush=True)
        print(json.dumps(summary, indent=2), flush=True)
        print(json.dumps(parameter_report(model), indent=2), flush=True)

    training_args = TrainingArguments(
        output_dir=cfg.out_dir,
        num_train_epochs=1.0,
        max_steps=total_steps,
        per_device_train_batch_size=1,  # unused: the token-budget plan sizes batches
        per_device_eval_batch_size=cfg.per_device_eval_batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        max_grad_norm=cfg.max_grad_norm,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": cfg.min_lr_rate},
        optim="adamw_torch_fused",
        bf16=True,
        gradient_checkpointing=False,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        train_sampling_strategy="random",
        length_column_name="n_tokens",
        dataloader_num_workers=cfg.dataloader_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=cfg.dataloader_workers > 0,
        dataloader_prefetch_factor=(
            cfg.dataloader_prefetch_factor if cfg.dataloader_workers > 0 else None
        ),
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        eval_on_start=cfg.eval_on_start,
        prediction_loss_only=True,
        disable_tqdm=True,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=None,  # CheckpointRetention is the only deleter
        report_to=[],
        seed=cfg.seed,
        data_seed=cfg.seed,
        # a later-generation plan has already dropped the consumed rows, so
        # replaying HF's step-indexed skip prefix would be wrong twice over
        ignore_data_skip=generation > 0,
    )
    if is_main:
        json.dump(
            {**asdict(cfg), **summary, **parameter_report(model)},
            open(os.path.join(cfg.run_dir, "config.json"), "w"),
            indent=2,
        )

    token_counter = {"tokens": 0}
    trainer = PlanBlockTrainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=val,
        data_collator=partial(
            collate,
            pad_to_multiple=cfg.pad_to_multiple,
            seq_len=cfg.seq_len,
            context_dir=context_dir,
            shard_files=shard_files,
        ),
        callbacks=[
            ConciseLogger(cfg.run_dir, total_steps, token_counter, stage),
            DataCursorCallback(
                cfg.run_dir, cfg, world_size, grad_accum, generation,
                plan_start_step, plan, block_rows, allowed,
            ),
            CheckpointRetention(cfg.out_dir, cfg.archive_every, cfg),
        ],
        cfg=cfg,
        plan=plan,
        token_counter=token_counter,
    )
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    final_dir = os.path.join(cfg.out_dir, "final")
    trainer.save_model(final_dir)
    if is_main:
        save_stage_meta(final_dir, cfg)


if __name__ == "__main__":
    main()
