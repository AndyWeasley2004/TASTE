"""Rectified-flow training of the latent DiT, one stage per config.

The config names the stage's latent stores (walked one at a time, in list
order, each for an equal share of ``max_steps``), the standardisation
statistics, the initial checkpoint (the previous stage's EMA weights) and the
schedule. Objective: ``t = sigmoid(n + logit_normal_mu)`` with ``n ~ N(0, 1)``,
``x_t = (1 - t) x0 + t eps``, the model predicts ``v = eps - x0``, MSE over the
non-padded latents. Batches are length-bucketed and padded to their own longest
row.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import bf16_compress_hook
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset

from taste import paths
from taste.models.dit import DiTConfig, LatentDiT, load_stats
from taste.training.dit_data import (
    DropoutProbs,
    LatentStoreDataset,
    LengthBucketedBatchSampler,
    attribute_null_ids,
    collate,
)

REPO = Path(__file__).resolve().parents[2]
T_BINS = (0.0, 0.25, 0.5, 0.75, 1.0)
VAL_T_GRID = (0.1, 0.3, 0.5, 0.7, 0.9)
VAL_MAX_ROWS = 2000


@dataclass
class Cfg:
    # store and corpus directories are relative to TASTE_DATA_ROOT; init_checkpoint
    # and out_dir to TASTE_CKPT_ROOT
    stores: list = field(default_factory=list)  # [[store_dir, corpus_dir], ...]
    val_store: list = field(default_factory=list)  # [store_dir, corpus_dir]
    stats_path: str = ""
    init_checkpoint: str | None = None
    out_dir: str = ""
    run_dir: str = ""

    d_model: int = 1024
    n_layers: int = 16
    n_heads: int = 16
    mlp_ratio: int = 4
    cond_dim: int = 512
    max_len: int = 384

    per_device_batch: int = 24
    global_batch: int = 384
    lr: float = 1e-4
    betas: tuple = (0.9, 0.95)
    weight_decay: float = 0.01
    warmup_steps: int = 2000
    min_lr_rate: float = 0.1
    max_steps: int = 100_000
    grad_clip: float = 1.0
    ema_decay: float = 0.9995
    seed: int = 0

    # mean of the logit-normal noise-level distribution; a negative value moves
    # training mass toward low noise, where the note-level detail is decided
    logit_normal_mu: float = 0.0

    dropout_attributes: float = 0.1
    dropout_globals: float = 0.1
    dropout_plan: float = 0.1

    grad_accum: int = 1
    loader_workers: int = 2
    logging_steps: int = 20
    eval_steps: int = 1000
    save_steps: int = 2500
    archive_every: int = 10_000


def hms(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def lr_multiplier(step: int, cfg: Cfg) -> float:
    if step < cfg.warmup_steps:
        return (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    progress = min(1.0, progress)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return cfg.min_lr_rate + (1 - cfg.min_lr_rate) * cosine


def param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    """No decay on embeddings, norms and biases (everything below 2 dims)."""
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if parameter.ndim < 2 or "embedding" in name or "null_context" in name:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_model(cfg: Cfg) -> LatentDiT:
    return LatentDiT(
        DiTConfig(
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            mlp_ratio=cfg.mlp_ratio,
            cond_dim=cfg.cond_dim,
            max_len=cfg.max_len,
        )
    )


def make_loader(cfg: Cfg, stores: list, stats, rank: int, world: int, seed: int) -> DataLoader:
    dataset = LatentStoreDataset(stores, stats)
    sampler = LengthBucketedBatchSampler(
        dataset.row_lengths(), cfg.per_device_batch, world, rank, seed
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=cfg.loader_workers,
        pin_memory=True,
        persistent_workers=cfg.loader_workers > 0,
        prefetch_factor=4 if cfg.loader_workers > 0 else None,
        collate_fn=partial(
            collate,
            max_len=cfg.max_len,
            dropout=DropoutProbs(
                cfg.dropout_attributes, cfg.dropout_globals, cfg.dropout_plan
            ),
            attribute_nulls=attribute_null_ids(),
        ),
    )


def loader_cycle(loader: DataLoader):
    epoch = 0
    # the train loader carries a batch_sampler, the val loader a plain sampler
    sampler = loader.batch_sampler if hasattr(loader.batch_sampler, "set_epoch") else loader.sampler
    while True:
        sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


def to_device(batch: dict, device) -> dict:
    return {name: tensor.to(device, non_blocking=True) for name, tensor in batch.items()}


def flow_loss(model, batch: dict, t: torch.Tensor):
    """Returns (summed squared error, element count, per-sequence loss) for noise levels ``t`` of shape ``[B]``."""
    x0 = batch["latents"]
    eps = batch["eps"]
    t_shaped = t[:, None, None]
    x_t = (1 - t_shaped) * x0 + t_shaped * eps
    target = eps - x0
    with torch.autocast("cuda", dtype=torch.bfloat16):
        prediction = model(
            x_t, t, batch["valid_mask"], batch["attrs"], batch["key"], batch["tempo"],
            batch["complete"], batch["start_mid"], batch["sections"], batch["sec_index"],
            batch["context"], batch["context_mask"],
        )
    valid = batch["valid_mask"]
    error = (prediction.float() - target) ** 2 * valid[:, :, None]
    per_sequence = error.sum(dim=(1, 2)) / (valid.sum(1) * x0.shape[-1])
    count = valid.sum() * x0.shape[-1]
    return error.sum(), count, per_sequence


def draw_t(cfg: Cfg, size, device) -> torch.Tensor:
    """Training noise levels: logit-normal with mean ``logit_normal_mu``."""
    return torch.sigmoid(torch.randn(size, device=device) + cfg.logit_normal_mu).to(device)


# --------------------------------------------------------------------------
# EMA and checkpoints
# --------------------------------------------------------------------------


class Ema:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = {
            name: parameter.detach().clone().float()
            for name, parameter in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        live = model.state_dict()
        names = list(self.shadow)
        # one fused multi-tensor lerp over all ~300 tensors
        torch._foreach_lerp_(
            [self.shadow[name] for name in names],
            [live[name].float() for name in names],
            1 - self.decay,
        )

    def load(self, state: dict) -> None:
        self.shadow = {name: tensor.float() for name, tensor in state.items()}


def save_resume(out_dir: Path, step: int, model, ema: Ema, optimizer, cfg: Cfg, rank: int,
                world: int):
    """Rank 0 writes the run state; every rank adds its own RNG state."""
    destination = out_dir / f"resume-{step}"
    if rank == 0:
        destination.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), destination / "model.pt")
        torch.save(ema.shadow, destination / "ema.pt")
        torch.save(optimizer.state_dict(), destination / "optimizer.pt")
        (destination / "state.json").write_text(json.dumps({"step": step}) + "\n")
    if world > 1:
        dist.barrier()
    torch.save(
        {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()},
        destination / f"rng-{rank}.pt",
    )
    if world > 1:
        dist.barrier()
    if rank != 0:
        return
    if cfg.archive_every > 0 and step % cfg.archive_every == 0:
        archive = out_dir / "model_only" / f"step{step}"
        archive.mkdir(parents=True, exist_ok=True)
        shutil.copy2(destination / "model.pt", archive / "model.pt")
        shutil.copy2(destination / "ema.pt", archive / "ema.pt")
    steps = sorted(
        int(path.name.split("-")[1]) for path in out_dir.glob("resume-*") if path.is_dir()
    )
    # keep only the newest resume checkpoint; the weight-only archives every
    # archive_every steps are the permanent record
    for stale in steps[:-1]:
        shutil.rmtree(out_dir / f"resume-{stale}", ignore_errors=True)


def latest_resume(out_dir: Path) -> Path | None:
    steps = sorted(
        int(path.name.split("-")[1]) for path in out_dir.glob("resume-*") if path.is_dir()
    )
    return out_dir / f"resume-{steps[-1]}" if steps else None


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


def build_val_loader(cfg: Cfg, stats, rank: int, world: int) -> DataLoader:
    dataset = LatentStoreDataset([cfg.val_store], stats)
    indices = np.arange(min(len(dataset), VAL_MAX_ROWS))[rank::world]
    return DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=cfg.per_device_batch,
        shuffle=False,
        num_workers=0,
        collate_fn=partial(
            collate,
            max_len=cfg.max_len,
            dropout=DropoutProbs(0.0, 0.0, 0.0),
            attribute_nulls=attribute_null_ids(),
        ),
    )


@torch.no_grad()
def validate(model, loader: DataLoader, device, world: int, seed: int) -> dict:
    model.eval()
    results = {}
    for t_value in VAL_T_GRID:
        total = torch.zeros(2, device=device)
        for index, batch in enumerate(loader):
            batch = to_device(batch, device)
            shape = batch["latents"].shape
            generator = torch.Generator().manual_seed(
                seed * 1_000_003 + index * 97 + int(round(t_value * 10))
            )
            batch["eps"] = torch.randn(shape, generator=generator).to(device)
            t = torch.full((shape[0],), t_value, device=device)
            error, count, _ = flow_loss(model, batch, t)
            total += torch.stack((error, count.float()))
        if world > 1:
            dist.all_reduce(total)
        results[f"val_t{t_value}"] = (total[0] / total[1]).item()
    results["val_mean"] = float(np.mean(list(results.values())))
    model.train()
    return results


# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one latent DiT stage from a config.")
    parser.add_argument("--config", help="stage config, e.g. configs/dit/large_public.json")
    parser.add_argument("--run-name", default="taste_dit_pretrain",
                        help="logs go to runs/<run-name> under the working directory")
    parser.add_argument("--resume", action="store_true",
                        help="continue from the newest resume-* checkpoint in out_dir")
    args = parser.parse_args()

    cfg = Cfg()
    provided: set[str] = set()
    if args.config:
        for name, value in json.load(open(args.config)).items():
            if not hasattr(cfg, name):
                raise ValueError(f"unknown config field: {name}")
            setattr(cfg, name, value)
            provided.add(name)
    cfg.run_dir = str(Path("runs") / args.run_name)
    cfg.stores = [[str(paths.data(store)), str(paths.data(corpus))] for store, corpus in cfg.stores]
    cfg.val_store = [str(paths.data(part)) for part in cfg.val_store]
    cfg.stats_path = str(paths.data(cfg.stats_path))
    cfg.out_dir = str(paths.ckpt(cfg.out_dir))
    if cfg.init_checkpoint:
        cfg.init_checkpoint = str(paths.ckpt(cfg.init_checkpoint))

    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    # a config that names grad_accum keeps it; otherwise it follows the GPU count
    if "grad_accum" not in provided:
        per_step = cfg.per_device_batch * world
        if cfg.global_batch % per_step:
            raise ValueError(
                f"global_batch {cfg.global_batch} is not divisible by "
                f"per_device_batch {cfg.per_device_batch} x world {world}")
        cfg.grad_accum = cfg.global_batch // per_step
    if world > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(cfg.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    out_dir, run_dir = Path(cfg.out_dir), Path(cfg.run_dir)
    is_main = rank == 0
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(parents=True, exist_ok=True)

    stats = load_stats(cfg.stats_path)
    model = build_model(cfg).to(device)
    if cfg.init_checkpoint:
        state = torch.load(Path(cfg.init_checkpoint) / "ema.pt", map_location="cpu")
        model.load_state_dict({name: tensor.float() for name, tensor in state.items()})
    n_parameters = model.count_parameters()
    ema = Ema(model, cfg.ema_decay)

    optimizer = torch.optim.AdamW(
        param_groups(model, cfg.weight_decay), lr=cfg.lr, betas=tuple(cfg.betas), fused=True
    )
    start_step = 0
    if args.resume:
        checkpoint = latest_resume(out_dir)
        if checkpoint is None:
            raise FileNotFoundError(f"--resume: no resume-* checkpoint in {out_dir}")
        model.load_state_dict(torch.load(checkpoint / "model.pt", map_location="cpu"))
        ema.load(torch.load(checkpoint / "ema.pt", map_location="cpu"))
        optimizer.load_state_dict(torch.load(checkpoint / "optimizer.pt", map_location=device))
        start_step = json.loads((checkpoint / "state.json").read_text())["step"]
        rng_path = checkpoint / f"rng-{rank}.pt"
        if rng_path.exists():
            rng = torch.load(rng_path, map_location="cpu")
            torch.set_rng_state(rng["cpu"])
            torch.cuda.set_rng_state(rng["cuda"])
        if is_main:
            print(f"[resume] from {checkpoint} at step {start_step}", flush=True)

    ema.shadow = {name: tensor.to(device) for name, tensor in ema.shadow.items()}
    wrapped = (
        DistributedDataParallel(model, device_ids=[local_rank]) if world > 1 else model
    )
    if world > 1:
        wrapped.register_comm_hook(None, bf16_compress_hook)

    # store phases (first step, end step, stores): one store per phase, in list
    # order; the last-listed store is the corpus's unaugmented pass, so it closes
    # the run in the low-LR tail
    n_phases = len(cfg.stores)
    edges = [round(k * cfg.max_steps / n_phases) for k in range(n_phases + 1)]
    phases = [(edges[k], edges[k + 1], [cfg.stores[k]]) for k in range(n_phases)]
    phase = next(k for k, (first, end, _) in enumerate(phases) if first <= start_step < end)
    loader = make_loader(cfg, phases[phase][2], stats, rank, world, cfg.seed + phase)
    batches = loader_cycle(loader)
    val_loader = build_val_loader(cfg, stats, rank, world)

    if is_main:
        summary = {
            "run": args.run_name,
            "parameters": n_parameters,
            "world_size": world,
            "grad_accum": cfg.grad_accum,
            "n_train_rows": len(loader.dataset),
            "n_val_rows": len(val_loader.dataset),
            "store_phases": [[first, end, [store for store, _ in stores]] for first, end, stores in phases],
            "start_step": start_step,
        }
        print(json.dumps(summary, indent=2), flush=True)
        json.dump({**asdict(cfg), **summary}, open(run_dir / "config.json", "w"), indent=2)

    metrics_path = run_dir / "metrics.jsonl"
    log_path = run_dir / "train_log.txt"

    def write(line: str) -> None:
        if is_main:
            print(line, flush=True)
            with open(log_path, "a") as handle:
                handle.write(line + "\n")

    write(
        f"=== DiT {args.run_name} start {datetime.now():%Y-%m-%d %H:%M:%S} | "
        f"{n_parameters / 1e6:.1f}M params | steps {start_step}->{cfg.max_steps} ==="
    )

    # running sums: loss, count, then (sum, count) per t-bin
    accumulator = torch.zeros(2 + 2 * (len(T_BINS) - 1), device=device)
    seconds_of_music = torch.zeros(1, device=device)
    started = last_time = time.time()
    model.train()

    for step in range(start_step, cfg.max_steps):
        if step >= phases[phase][1]:
            phase += 1
            loader = make_loader(cfg, phases[phase][2], stats, rank, world, cfg.seed + phase)
            batches = loader_cycle(loader)
            write(
                f"[phase {phase + 1}/{len(phases)}] step {step}: loader rebuilt over "
                + ", ".join(store for store, _ in phases[phase][2])
            )

        for micro in range(cfg.grad_accum):
            batch = to_device(next(batches), device)
            t = draw_t(cfg, batch["latents"].shape[:1], device)
            batch["eps"] = torch.randn_like(batch["latents"])
            last_micro = micro == cfg.grad_accum - 1
            if world > 1 and not last_micro:
                with wrapped.no_sync():
                    error, count, per_sequence = flow_loss(wrapped, batch, t)
                    loss = error / count
                    (loss / cfg.grad_accum).backward()
            else:
                error, count, per_sequence = flow_loss(wrapped, batch, t)
                loss = error / count
                (loss / cfg.grad_accum).backward()
            accumulator[0] += error.detach()
            accumulator[1] += count
            # bin the per-sequence loss by noise level
            binned_loss, binned_t = per_sequence.detach(), t
            bin_index = torch.clamp(
                (binned_t * (len(T_BINS) - 1)).long(), max=len(T_BINS) - 2
            )
            for index in range(len(T_BINS) - 1):
                selected = bin_index == index
                accumulator[2 + 2 * index] += binned_loss[selected].sum()
                accumulator[3 + 2 * index] += selected.sum()
            seconds_of_music += batch["valid_mask"].sum()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        if not (torch.isfinite(loss) and torch.isfinite(grad_norm)):
            raise RuntimeError(f"step {step}: non-finite loss {loss.item()} / grad norm {float(grad_norm)}")
        for group in optimizer.param_groups:
            group["lr"] = cfg.lr * lr_multiplier(step, cfg)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        ema.update(model)

        completed = step + 1
        if completed % cfg.logging_steps == 0:
            totals = torch.cat((accumulator, seconds_of_music))
            if world > 1:
                dist.all_reduce(totals)
            now = time.time()
            elapsed = max(now - last_time, 1e-6)
            record = {
                "step": completed,
                "t": round(now - started, 1),
                "loss": (totals[0] / totals[1]).item(),
                "lr": optimizer.param_groups[0]["lr"],
                "grad_norm": float(grad_norm),
                "sec_per_step": elapsed / cfg.logging_steps,
                "music_seconds_per_s": (totals[-1] / elapsed).item(),
            }
            for index in range(len(T_BINS) - 1):
                name = f"loss_t{T_BINS[index]:.2f}_{T_BINS[index + 1]:.2f}"
                counted = totals[3 + 2 * index]
                record[name] = (totals[2 + 2 * index] / counted).item() if counted > 0 else float("nan")
            if is_main:
                with open(metrics_path, "a") as handle:
                    handle.write(json.dumps(record) + "\n")
            bins = " ".join(
                f"{record[f'loss_t{T_BINS[i]:.2f}_{T_BINS[i + 1]:.2f}']:.3f}"
                for i in range(len(T_BINS) - 1)
            )
            write(
                f"[{completed:>6}/{cfg.max_steps} {100 * completed / cfg.max_steps:4.1f}%] "
                f"loss {record['loss']:.4f} | t-bins {bins} | "
                f"lr {record['lr']:.2e} gnorm {record['grad_norm']:.2f} | "
                f"{record['sec_per_step']:.3f}s/step "
                f"{record['music_seconds_per_s'] / 1e3:.1f}k music-s/s | "
                f"{hms(now - started)}"
            )
            accumulator.zero_()
            seconds_of_music.zero_()
            last_time = time.time()

        if completed % cfg.eval_steps == 0 or completed == cfg.max_steps:
            results = validate(wrapped, val_loader, device, world, cfg.seed)
            if is_main:
                with open(metrics_path, "a") as handle:
                    handle.write(json.dumps({"step": completed, **results}) + "\n")
            write(
                f"   >> eval @ {completed}: mean {results['val_mean']:.4f} | "
                + " ".join(f"{name.replace('val_t', 't')} {value:.4f}"
                           for name, value in results.items() if name.startswith("val_t"))
            )
            last_time = time.time()

        if completed % cfg.save_steps == 0 or completed == cfg.max_steps:
            save_resume(out_dir, completed, model, ema, optimizer, cfg, rank, world)
            last_time = time.time()

    if is_main:
        final = out_dir / "final"
        final.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), final / "model.pt")
        torch.save(ema.shadow, final / "ema.pt")
        json.dump({**asdict(cfg), "parameters": n_parameters},
                  open(final / "config.json", "w"), indent=2)
        shutil.copy2(cfg.stats_path, final / "latent_stats.npz")
        shutil.copy2(REPO / "attributes.json", final / "attributes.json")
        write(
            f"=== DiT {args.run_name} end | peak "
            f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB/GPU | "
            f"{hms(time.time() - started)} ==="
        )
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
