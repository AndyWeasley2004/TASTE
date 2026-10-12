"""Pretraining of the AR MIDI generator from random initialisation.

Piece-aligned variable-length rows are batched under a per-device token
budget, and the corpus's augmented epoch copies are trained in order (all of
epoch 0, then epoch 1, ...) as sampler blocks under one cosine schedule.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import partial
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from datasets import concatenate_datasets, load_dataset
from torch.utils.data import BatchSampler, DataLoader
from transformers import Trainer, TrainerCallback, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint, seed_worker

from taste import paths
from taste.corpus.pretrain_format import FORMAT_VERSION, N_EPOCHS, SEQ_LEN
from taste.models.ar import new_model
from taste.representation.vocab import PAD_ID, save_vocab_meta

# RTX A6000 dense BF16 tensor peak, used only for the logged approximate MFU
A6000_BF16_DENSE_FLOPS = 154.8e12


@dataclass
class Cfg:
    data_dir: str = "corpora/pretrain"  # relative to TASTE_DATA_ROOT
    seq_len: int = SEQ_LEN
    lr: float = 2e-4
    min_lr_rate: float = 0.1
    weight_decay: float = 0.1
    warmup_ratio: float = 0.01
    max_grad_norm: float = 1.0
    tokens_per_device_batch: int = 12_288
    pad_to_multiple: int = 256
    per_device_eval_batch: int = 4
    grad_accum: int = 6
    out_dir: str = "ar/pretrain"  # relative to TASTE_CKPT_ROOT
    run_dir: str = "./runs/taste_ar_pretrain"
    logging_steps: int = 20
    eval_steps: int = 1000
    eval_on_start: bool = True
    save_steps: int = 1000
    # retention: keep the latest checkpoint, keep archive_every-multiples for
    # retain_window steps after they are written, archive model-only weights
    archive_every: int = 5000
    retain_window: int = 10000
    dataloader_workers: int = 2
    dataloader_prefetch_factor: int = 4
    seed: int = 0
    max_steps: int = -1
    expected_world_size: int = 4


def hms(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def rounded(value):
    return round(value, 6) if isinstance(value, float) else value


class ConciseLogger(TrainerCallback):
    """One human line per log plus a metrics.jsonl record, main process only.

    Token throughput counts the padded tokens actually processed: rows are variable
    length and the sampler pads per batch, so the padded tokens pushed through the
    model differ from the nominal budget. Consecutive world_size batches share
    one shape by construction, so rank 0's count x world is exact.
    """

    def __init__(self, run_dir: str, total_steps: int, token_counter: dict):
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
            f"=== AR pretrain start {datetime.now():%Y-%m-%d %H:%M:%S} | "
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
                f"[{step:>6}/{self.total} {100 * step / self.total:4.1f}% "
                f"ep{logs.get('epoch', 0):.2f}] loss {logs['loss']:.3f} "
                f"ppl {math.exp(min(20.0, logs['loss'])):7.1f} "
                f"lr {logs.get('learning_rate', 0):.2e} "
                f"gnorm {logs.get('grad_norm', 0):.2f} | "
                f"{tok_s / 1e3:.0f}k tok/s approx-MFU {100 * step_mfu:.1f}% | "
                f"{hms(now - self.started)}"
            )

        if "eval_loss" in logs:
            loss = logs["eval_loss"]
            self.write(
                f"   >> eval @ {step}: {loss:.4f} "
                f"(ppl {math.exp(min(20.0, loss)):.1f})"
            )

    def on_train_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero or self.started is None:
            return
        elapsed = max(time.time() - self.started, 1e-6)
        world = int(os.environ.get("WORLD_SIZE", 1))
        mfu = state.total_flos / elapsed / world / A6000_BF16_DENSE_FLOPS
        peak_gib = torch.cuda.max_memory_allocated() / 2**30
        self.write(
            f"=== AR pretrain end | peak {peak_gib:.2f} GiB/GPU | "
            f"overall approx MFU {100 * mfu:.1f}% | {hms(elapsed)} ==="
        )


class DataCursorCallback(TrainerCallback):
    def __init__(self, run_dir: str, seed: int, world_size: int):
        self.path = Path(run_dir) / "data_cursor.json"
        self.seed = seed
        self.world_size = world_size

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        payload = {
            "global_step": state.global_step,
            "epoch": state.epoch,
            "seed": self.seed,
            "world_size": self.world_size,
            "checkpoint": str(Path(args.output_dir) / f"checkpoint-{state.global_step}"),
            "resume_semantics": "Trainer RNG + optimizer + scheduler + deterministic data skip",
        }
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(self.path)


# everything a full checkpoint carries beyond the model itself
RESUME_ONLY_FILES = {
    "optimizer.pt",
    "scheduler.pt",
    "scaler.pt",
    "trainer_state.json",
    "training_args.bin",
}


class CheckpointRetention(TrainerCallback):
    """Custom retention: this callback is the only deleter of checkpoints.

    Full checkpoints are written every ``save_steps`` so a crash can resume.
    Retention keeps the most recent checkpoint plus every
    ``archive_every``-multiple whose age is under ``retain_window`` steps, and
    deletes everything else. Independently, each ``archive_every``-multiple
    save also copies model-only weights to ``model_only/step{N}/``, which is
    kept permanently for later sampling comparisons.
    """

    def __init__(self, out_dir: str, archive_every: int, retain_window: int):
        if archive_every <= 0 or retain_window <= 0:
            raise ValueError("archive_every and retain_window must be positive")
        self.out_dir = Path(out_dir)
        self.archive_every = int(archive_every)
        self.retain_window = int(retain_window)

    def keep(self, step: int, current_step: int, latest: int) -> bool:
        if step == latest:
            return True
        return (
            step % self.archive_every == 0
            and current_step - step < self.retain_window
        )

    def archive(self, step: int) -> None:
        source = self.out_dir / f"checkpoint-{step}"
        destination = self.out_dir / "model_only" / f"step{step}"
        destination.mkdir(parents=True, exist_ok=True)
        for path in sorted(source.iterdir()):
            if not path.is_file() or path.name.startswith("rng_state"):
                continue
            if path.name not in RESUME_ONLY_FILES:
                shutil.copy2(path, destination / path.name)
        # sampling from an archive needs the MIDI vocab sidecar
        save_vocab_meta(destination)
        print(f"[retention] archived model-only weights to {destination}", flush=True)

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        current_step = state.global_step
        if current_step % self.archive_every == 0:
            self.archive(current_step)
        steps = sorted(
            int(path.name.split("-")[1])
            for path in self.out_dir.glob("checkpoint-*")
            if path.is_dir()
        )
        latest = max(steps)
        for step in steps:
            if not self.keep(step, current_step, latest):
                # external boundary: a concurrent/interrupted pass may have
                # already removed it
                shutil.rmtree(self.out_dir / f"checkpoint-{step}", ignore_errors=True)
                print(f"[retention] deleted checkpoint-{step}", flush=True)


class _EpochSetter:
    def __init__(self, batch_sampler: "TokenBudgetBatchSampler"):
        self.batch_sampler = batch_sampler

    def set_epoch(self, epoch: int) -> None:
        self.batch_sampler.set_epoch(epoch)


class TokenBudgetBatchSampler(BatchSampler):
    """Build same-shaped per-rank batches capped by a token budget.

    Consecutive groups of ``world_size`` batches share one padded length and
    local batch size. Accelerate then assigns one batch from each group to each
    DDP rank, keeping ranks balanced and synchronized.
    """

    batch_size = None
    drop_last = False

    def __init__(
        self,
        lengths,
        tokens_per_device_batch: int,
        pad_to_multiple: int,
        world_size: int,
        seed: int,
        strata,
    ):
        if tokens_per_device_batch <= 0:
            raise ValueError("tokens_per_device_batch must be positive")
        if pad_to_multiple <= 0:
            raise ValueError("pad_to_multiple must be positive")
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        self.lengths = [int(length) for length in lengths]
        self.tokens_per_device_batch = int(tokens_per_device_batch)
        self.pad_to_multiple = int(pad_to_multiple)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.strata = [int(value) for value in strata]
        if len(self.strata) != len(self.lengths):
            raise ValueError("strata and lengths differ")
        self.epoch = 0
        self.sampler = _EpochSetter(self)

        self._groups: dict[tuple[int, int], list[int]] = {}
        for index, (length, stratum) in enumerate(zip(self.lengths, self.strata)):
            padded = math.ceil(length / self.pad_to_multiple) * self.pad_to_multiple
            self._groups.setdefault((stratum, padded), []).append(index)

        self.n_batches = 0
        self.n_emitted_examples = 0
        for (_, padded), indices in self._groups.items():
            local_batch = max(1, self.tokens_per_device_batch // padded)
            global_block = local_batch * self.world_size
            n_blocks = math.ceil(len(indices) / global_block)
            self.n_batches += n_blocks * self.world_size
            self.n_emitted_examples += n_blocks * global_block

    def __len__(self) -> int:
        return self.n_batches

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        blocks_by_stratum: dict[int, list[list[list[int]]]] = {}
        for (stratum, padded), original_indices in self._groups.items():
            indices = list(original_indices)
            rng.shuffle(indices)
            local_batch = max(1, self.tokens_per_device_batch // padded)
            global_block = local_batch * self.world_size
            for start in range(0, len(indices), global_block):
                block = indices[start : start + global_block]
                if len(block) < global_block:
                    repeats = math.ceil((global_block - len(block)) / len(indices))
                    block.extend((indices * repeats)[: global_block - len(block)])
                blocks_by_stratum.setdefault(stratum, []).append([
                    block[offset : offset + local_batch]
                    for offset in range(0, global_block, local_batch)
                ])
        for stratum in sorted(blocks_by_stratum):
            blocks = blocks_by_stratum[stratum]
            rng.shuffle(blocks)
            for block in blocks:
                yield from block


class EpochBlockTrainer(Trainer):
    """Token-budget batching with the epoch copies as ordered sampler blocks."""

    def __init__(self, *args, cfg: Cfg, token_counter: dict, **kwargs):
        self.cfg = cfg
        self.token_counter = token_counter
        super().__init__(*args, **kwargs)

    def get_train_dataloader(self) -> DataLoader:
        batch_sampler = build_epoch_block_sampler(
            self.train_dataset, self.cfg, self.args.world_size,
        )
        dataloader_kwargs = {
            "batch_sampler": batch_sampler,
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

        # The custom sampler already emits a multiple of world_size batches.
        # Let Accelerate shard consecutive same-shaped batches without trying
        # to force a single fixed batch size.
        self.accelerator.even_batches = False
        return self.accelerator.prepare(
            DataLoader(self.train_dataset, **dataloader_kwargs)
        )

    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        # The variable-size training sampler needs uneven Accelerate batches,
        # but evaluation collectives require every rank to run the same number
        # of steps. Scope even batching to the prepared evaluation loader.
        previous_even_batches = self.accelerator.even_batches
        self.accelerator.even_batches = True
        try:
            return super().get_eval_dataloader(eval_dataset)
        finally:
            self.accelerator.even_batches = previous_even_batches

    def training_step(self, model, inputs, num_items_in_batch=None):
        self.token_counter["tokens"] += inputs["input_ids"].numel()
        return super().training_step(model, inputs, num_items_in_batch)


def build_epoch_block_sampler(dataset, cfg: Cfg, world_size: int) -> TokenBudgetBatchSampler:
    """Sequential epoch copies, shuffled within each copy.

    ``TokenBudgetBatchSampler`` walks strata in sorted order and shuffles rows
    and blocks inside a stratum; using the epoch index as the stratum gives
    exactly the "all of epoch0, then epoch1, then epoch2" order.
    """
    return TokenBudgetBatchSampler(
        lengths=dataset["n_tokens"],
        tokens_per_device_batch=cfg.tokens_per_device_batch,
        pad_to_multiple=cfg.pad_to_multiple,
        world_size=world_size,
        seed=cfg.seed,
        strata=dataset["epoch_idx"],
    )


def build_model(cfg: Cfg):
    from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_qwen3

    # Liger RMSNorm + SwiGLU only; the model's own fused linear CE must not be
    # replaced by Liger's CE patches
    apply_liger_kernel_to_qwen3(
        rope=False,
        cross_entropy=False,
        fused_linear_cross_entropy=False,
        rms_norm=True,
        swiglu=True,
    )
    model = new_model(attn_implementation="sdpa")
    if int(model.config.max_position_embeddings) != cfg.seq_len:
        raise ValueError("model and corpus context lengths differ")
    model.config.use_cache = False
    return model


def build_datasets(cfg: Cfg):
    parts = []
    for epoch in range(N_EPOCHS):
        files = sorted(glob.glob(os.path.join(cfg.data_dir, f"epoch{epoch}", "train-*.parquet")))
        if not files:
            raise FileNotFoundError(f"missing epoch{epoch} shards in {cfg.data_dir}")
        parts.append(load_dataset("parquet", data_files=files, split="train"))
    train = concatenate_datasets(parts).with_format(
        "torch", columns=["input_ids", "n_tokens", "epoch_idx"]
    )
    if max(train["n_tokens"]) > cfg.seq_len:
        raise ValueError(f"row longer than the {cfg.seq_len}-token context")

    val_files = sorted(glob.glob(os.path.join(cfg.data_dir, "val", "val-*.parquet")))
    if not val_files:
        raise FileNotFoundError(f"missing val shards in {cfg.data_dir}")
    val = (
        load_dataset("parquet", data_files=val_files, split="train")
        .sort("n_tokens")
        .with_format("torch", columns=["input_ids", "n_tokens"])
    )
    return train, val


def collate(features, pad_to_multiple: int, seq_len: int):
    longest = max(len(feature["input_ids"]) for feature in features)
    target = math.ceil(longest / pad_to_multiple) * pad_to_multiple
    if target > seq_len:
        raise ValueError(f"padded length {target} exceeds the {seq_len}-token context")
    input_ids, labels, attention_mask = [], [], []
    for feature in features:
        ids = feature["input_ids"].long()
        padding = target - len(ids)
        input_ids.append(torch.cat((ids, torch.full((padding,), PAD_ID, dtype=torch.long))))
        # full-sequence supervision; only padding is ignored
        labels.append(torch.cat((ids, torch.full((padding,), -100, dtype=torch.long))))
        attention_mask.append(torch.cat((
            torch.ones(len(ids), dtype=torch.long),
            torch.zeros(padding, dtype=torch.long),
        )))
    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_mask),
    }


def assert_epoch_blocks(train, cfg: Cfg, world_size: int, n_batches: int = 64) -> None:
    """The first batches must draw only from epoch0."""
    sampler = build_epoch_block_sampler(train, cfg, world_size)
    epoch_idx = train["epoch_idx"]
    seen = set()
    for count, batch in enumerate(sampler):
        if count >= n_batches:
            break
        seen.update(int(epoch_idx[index]) for index in batch)
    if seen != {0}:
        raise AssertionError(f"first {n_batches} batches span epoch copies {sorted(seen)}")
    print(f"epoch-block order check: first {n_batches} batches are epoch0 only", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain the AR MIDI generator.")
    parser.add_argument("--run-name", default="taste_ar_pretrain",
                        help="logs go to runs/<run-name> under the working directory")
    parser.add_argument("--config", help="e.g. configs/ar/pretrain.json")
    parser.add_argument("--fresh", action="store_true",
                        help="ignore existing checkpoints in out_dir instead of resuming")
    args = parser.parse_args()

    cfg = Cfg()
    if args.config:
        for name, value in json.load(open(args.config)).items():
            if not hasattr(cfg, name):
                raise ValueError(f"unknown config field: {name}")
            setattr(cfg, name, value)
    cfg.run_dir = f"./runs/{args.run_name}"
    cfg.data_dir = str(paths.data(cfg.data_dir))
    cfg.out_dir = str(paths.ckpt(cfg.out_dir))

    if not os.path.isdir(cfg.data_dir):
        raise FileNotFoundError(f"corpus not built: {cfg.data_dir}")
    data_meta = json.load(open(os.path.join(cfg.data_dir, "meta.json")))
    if data_meta["format_version"] != FORMAT_VERSION:
        raise ValueError(
            f"{cfg.data_dir} uses format {data_meta['format_version']!r}; "
            f"expected {FORMAT_VERSION!r}"
        )

    is_main = int(os.environ.get("RANK", 0)) == 0
    if is_main:
        os.makedirs(cfg.run_dir, exist_ok=True)
        os.makedirs(cfg.out_dir, exist_ok=True)
        save_vocab_meta(cfg.out_dir)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = build_model(cfg)
    train, val = build_datasets(cfg)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size != cfg.expected_world_size:
        raise ValueError(
            f"approved pretraining recipe requires {cfg.expected_world_size} GPUs; "
            f"found {world_size}"
        )
    if is_main:
        assert_epoch_blocks(train, cfg, world_size)

    sampler_probe = build_epoch_block_sampler(train, cfg, world_size)
    microsteps_per_rank = len(sampler_probe) // world_size
    horizon_steps = math.ceil(microsteps_per_rank / cfg.grad_accum)
    corpus_tokens = int(sum(train["n_tokens"]))
    sampler_summary = {
        "microsteps_per_rank": microsteps_per_rank,
        "optimizer_steps_over_3_epoch_copies": horizon_steps,
        "emitted_examples": sampler_probe.n_emitted_examples,
        "repeated_examples_for_even_ranks": sampler_probe.n_emitted_examples - len(train),
        "corpus_tokens": corpus_tokens,
        "nominal_steps_at_full_budget": round(
            corpus_tokens / (cfg.tokens_per_device_batch * world_size * cfg.grad_accum)
        ),
    }
    del sampler_probe
    total_steps = cfg.max_steps if cfg.max_steps > 0 else horizon_steps
    if is_main:
        print(json.dumps(sampler_summary, indent=2), flush=True)

    training_args = TrainingArguments(
        output_dir=cfg.out_dir,
        num_train_epochs=1.0,  # the 3 epoch copies are already concatenated
        max_steps=cfg.max_steps,
        per_device_train_batch_size=1,  # unused: the token-budget sampler sizes batches
        per_device_eval_batch_size=cfg.per_device_eval_batch,
        gradient_accumulation_steps=cfg.grad_accum,
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
        ignore_data_skip=False,
    )
    if is_main:
        json.dump(
            {
                **asdict(cfg),
                "world_size": world_size,
                "n_params": sum(p.numel() for p in model.parameters()),
                "planned_total_steps": total_steps,
                "n_train_rows": len(train),
                "n_val_rows": len(val),
                "token_budget_sampler": sampler_summary,
            },
            open(os.path.join(cfg.run_dir, "config.json"), "w"),
            indent=2,
        )

    token_counter = {"tokens": 0}
    trainer = EpochBlockTrainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=val,
        data_collator=partial(
            collate, pad_to_multiple=cfg.pad_to_multiple, seq_len=cfg.seq_len,
        ),
        callbacks=[
            ConciseLogger(cfg.run_dir, total_steps, token_counter),
            DataCursorCallback(cfg.run_dir, cfg.seed, world_size),
            CheckpointRetention(cfg.out_dir, cfg.archive_every, cfg.retain_window),
        ],
        cfg=cfg,
        token_counter=token_counter,
    )
    resume = None if args.fresh else get_last_checkpoint(cfg.out_dir)
    trainer.train(resume_from_checkpoint=resume)
    final_dir = os.path.join(cfg.out_dir, "final")
    trainer.save_model(final_dir)
    if is_main:
        save_vocab_meta(final_dir)


if __name__ == "__main__":
    main()
