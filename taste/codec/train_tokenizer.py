#!/usr/bin/env python3
# Adapted from MuseTok, "Symbolic Music Tokenization for Generation and
# Semantic Understanding" (ICASSP 2026): https://github.com/Yuer867/MuseTok.
"""Train the window codec (VAE or residual VQ), with MuseTok's schedule.

    python -m taste.codec.train_tokenizer configs/codec/tokenizer.yaml

Linear warmup to ``max_lr``, then cosine decay to ``min_lr`` over
``training.total_steps``, Adam, gradient clipping and EMA-smoothed loss
reporting. Relative ``store_dir`` entries resolve under
``$TASTE_DATA_ROOT/codec_tokens``; checkpoints go to
``$TASTE_CKPT_ROOT/codec/<run name>`` unless ``training.ckpt_dir`` is set.
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from taste import paths

from . import vocab
from .models.musetok import build_codec
from .window_group_dataset import build_split_datasets, stage_sample_weights

DATA_ROOT = paths.data("codec_tokens")


def compute_loss_ema(ema: float, batch_loss: float, decay: float = 0.95) -> float:
    return batch_loss if ema == 0.0 else batch_loss * (1 - decay) + ema * decay


def batch_to_model_inputs(batch, device):
    """Encoder input is ``(L, batch, W)``, decoder input ``(len, batch)``."""
    return {
        "enc_inp": batch["enc_input"].permute(2, 0, 1).to(device),
        "dec_inp": batch["dec_input"].permute(1, 0).to(device),
        "dec_tgt": batch["dec_target"].permute(1, 0).to(device),
        "dec_window_idx": batch["dec_window_idx"].to(device),
        "padding_mask": batch["enc_padding_mask"].to(device),
    }


def reconstruction_accuracy(dec_logits, dec_tgt, pad_id):
    correct = torch.argmax(dec_logits, dim=-1) == dec_tgt
    scored = dec_tgt != pad_id
    total = int(scored.sum().item())
    return (correct & scored).sum().item() / max(total, 1)


def codebook_usage(counts: np.ndarray) -> dict:
    """Fraction of codes ever selected, and the usage distribution's perplexity.

    Perplexity should stay within an order of magnitude of the codebook size;
    a collapse onto a few codes leaves the latents nearly uninformative.
    """
    total = counts.sum()
    if total == 0:
        return {"used_fraction": 0.0, "perplexity": 0.0}
    probabilities = counts[counts > 0] / total
    return {
        "used_fraction": float((counts > 0).mean()),
        "perplexity": float(np.exp(-(probabilities * np.log(probabilities)).sum())),
    }


def validate(model, dloader, device, max_batches, pad_id):
    model.eval()
    recons_losses, reg_losses, accuracies = [], [], []
    counts = (
        np.zeros((model.num_quantizers, model.codebook_size), dtype=np.int64)
        if model.supports_codebook_usage
        else None
    )

    print("[info] validating ...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dloader):
            if batch_idx >= max_batches:
                break
            inputs = batch_to_model_inputs(batch, device)
            dec_logits, reg_aux = model(
                inputs["enc_inp"],
                inputs["dec_inp"],
                inputs["dec_window_idx"],
                padding_mask=inputs["padding_mask"],
            )
            losses = model.compute_loss(reg_aux, 0.0, dec_logits, inputs["dec_tgt"])
            accuracy = reconstruction_accuracy(dec_logits, inputs["dec_tgt"], pad_id)

            if counts is not None:
                indices = model.code_indices(
                    inputs["enc_inp"], padding_mask=inputs["padding_mask"]
                )
                flat = indices.reshape(-1, indices.size(-1)).cpu().numpy()
                for quantizer in range(min(counts.shape[0], flat.shape[1])):
                    np.add.at(counts[quantizer], flat[:, quantizer], 1)

            recons_losses.append(losses["recons_loss"].item())
            reg_losses.append(losses["reg_loss"].item())
            accuracies.append(accuracy)

    usage = codebook_usage(counts) if counts is not None else None
    return recons_losses, reg_losses, accuracies, usage


class Trainer:
    def __init__(self, config: dict, use_wandb: bool = False):
        self.config = config
        self.data_conf = config["data"]
        self.model_conf = config["model"]
        self.train_conf = config["training"]
        self.total_steps = self.train_conf["total_steps"]
        self.device = self.train_conf["device"]
        self.trained_steps = self.train_conf["trained_steps"]
        self.recons_loss_ema = 0.0
        self.reg_loss_ema = 0.0
        self.use_wandb = use_wandb
        self.wandb = None

    def run_name(self) -> str:
        model_type = self.model_conf.get("model_type", "ResidualVQ")
        stem = "{}_{}L-{}win-seqlen_{}-d{}".format(
            model_type,
            self.model_conf["enc_n_layer"],
            self.data_conf["max_windows"],
            self.data_conf["dec_seqlen"],
            self.model_conf["d_latent"],
        )
        if model_type == "ResidualVQ":
            stem += "-n{}-s{}".format(
                self.model_conf["num_quantizers"], self.model_conf["codebook_size"]
            )
        return stem

    def prepare_dirs(self, config_path: Path) -> None:
        ckpt_dir = self.train_conf.get("ckpt_dir") or paths.ckpt("codec", self.run_name())
        self.ckpt_dir = Path(ckpt_dir)
        self.params_dir = self.ckpt_dir / "params"
        self.optim_dir = self.ckpt_dir / "optim"
        self.params_dir.mkdir(parents=True, exist_ok=True)
        self.optim_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(config_path, self.ckpt_dir / "config.yaml")

    def store_dirs(self) -> list[Path]:
        """``store_dir`` accepts one path or a list of them, one per stage."""
        configured = self.data_conf["store_dir"]
        if isinstance(configured, (str, Path)):
            configured = [configured]
        stores = []
        for entry in configured:
            store = Path(entry)
            stores.append(store if store.is_absolute() else DATA_ROOT / store)
        return stores

    def build_data(self):
        stores = self.store_dirs()
        self.dset, self.dset_val = build_split_datasets(
            stores,
            val_fraction=self.data_conf["val_fraction"],
            split_seed=self.data_conf["split_seed"],
            val_cap=self.data_conf["val_cap"],
            do_augment=self.data_conf["do_augment"],
            model_enc_seqlen=self.data_conf["enc_seqlen"],
            model_dec_seqlen=self.data_conf["dec_seqlen"],
            model_max_windows=self.data_conf["max_windows"],
        )
        for name, size in zip(self.dset.names, self.dset.sizes):
            print(f"[info] store {name}: {size} training pieces")
        print("[info] # training pieces:", len(self.dset))
        print("[info] # validation pieces:", len(self.dset_val))

        weights = self.data_conf.get("stage_weights")
        sampler = None
        if weights:
            per_item = stage_sample_weights(self.dset, weights)
            sampler = WeightedRandomSampler(
                torch.as_tensor(per_item, dtype=torch.double),
                num_samples=len(self.dset),
                replacement=True,
            )
            share = np.asarray(weights, dtype=float)
            print("[info] stage mixture:", dict(zip(self.dset.names, share / share.sum())))

        workers = self.data_conf["num_workers"]
        self.dloader = DataLoader(
            self.dset,
            batch_size=self.data_conf["batch_size"],
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=workers,
            drop_last=True,
            persistent_workers=workers > 0,
        )
        self.dloader_val = DataLoader(
            self.dset_val,
            batch_size=self.data_conf["batch_size"],
            shuffle=False,
            num_workers=workers,
            persistent_workers=workers > 0,
        )
        self.report_budget()

    def report_budget(self) -> None:
        """Say what ``total_steps`` buys, in both senses of "epoch".

        An epoch here is a pass over *pieces*, but each sample reads only
        ``max_windows`` windows out of a piece's ~45, so a piece epoch is far
        from a pass over the data. Printing both keeps the schedule honest.
        """
        batch_size = self.data_conf["batch_size"]
        steps_per_epoch = max(1, len(self.dset) // batch_size)
        seen = self.total_steps * batch_size * self.data_conf["max_windows"]
        print(
            "[info] budget: {:,} steps at batch {} = {:,} steps/epoch".format(
                self.total_steps, batch_size, steps_per_epoch
            )
        )
        print(
            "[info] budget: {:.1f} passes over pieces, {:.2f} over all {:,} windows".format(
                self.total_steps / steps_per_epoch, seen / self.dset.n_windows,
                self.dset.n_windows,
            )
        )

    def build_model(self):
        self.model = build_codec(
            self.model_conf, n_token=self.dset.vocab_size, pad_id=vocab.PAD_ID
        ).to(self.device)
        print("[info] codec: {} | regulariser: {}".format(
            type(self.model).__name__, self.model.reg_name
        ))

        if self.model_conf["pretrained_params_path"]:
            self.model.load_state_dict(torch.load(self.model_conf["pretrained_params_path"]))
            print("[info] load pretrained params")

        self.model.train()
        self.n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("[info] model # params: {} ({:.1f}M)".format(self.n_params, self.n_params / 1e6))

        opt_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = optim.Adam(opt_params, lr=self.train_conf["max_lr"])
        if self.model_conf["pretrained_optim_path"]:
            self.optimizer.load_state_dict(torch.load(self.model_conf["pretrained_optim_path"]))
            print("[info] load pretrained optim")
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, self.total_steps, eta_min=self.train_conf["min_lr"]
        )

    def maybe_start_wandb(self):
        if not self.use_wandb:
            return
        import wandb

        self.wandb = wandb
        wandb.init(
            config=dict(**{"n_parameters": self.n_params}, **self.config),
            resume="allow",
            project=self.train_conf["wandb_project"],
            name=self.run_name(),
            id=self.run_name(),
        )

    def log(self, values: dict) -> None:
        if self.wandb is not None:
            self.wandb.log(values, step=self.trained_steps)

    def save_checkpoint(self) -> None:
        tag = "step_{:d}-RC_{:.3f}-REG_{:.3f}".format(
            self.trained_steps, self.recons_loss_ema, self.reg_loss_ema
        )
        torch.save(self.model.state_dict(), self.params_dir / f"{tag}-model.pt")
        torch.save(self.optimizer.state_dict(), self.optim_dir / f"{tag}-optim.pt")

    def train_epoch(self, epoch: int) -> bool:
        """Runs one epoch; returns False once ``total_steps`` has been reached."""
        self.model.train()
        print(f"[epoch {epoch:03d}] training ...")
        print(f"[epoch {epoch:03d}] # batches = {len(self.dloader)}")
        started = time.time()

        for batch_idx, batch in enumerate(self.dloader):
            self.model.zero_grad()
            inputs = batch_to_model_inputs(batch, self.device)
            self.trained_steps += 1

            dec_logits, reg_aux = self.model(
                inputs["enc_inp"],
                inputs["dec_inp"],
                inputs["dec_window_idx"],
                padding_mask=inputs["padding_mask"],
            )
            losses = self.model.compute_loss(
                reg_aux, self.train_conf["beta"], dec_logits, inputs["dec_tgt"]
            )
            accuracy = reconstruction_accuracy(dec_logits, inputs["dec_tgt"], vocab.PAD_ID)

            warmup = self.train_conf["lr_warmup_steps"]
            if self.trained_steps < warmup:
                self.optimizer.param_groups[0]["lr"] = (
                    self.train_conf["max_lr"] * self.trained_steps / warmup
                )
            else:
                self.scheduler.step(self.trained_steps - warmup)

            losses["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_conf["grad_clip"])
            self.optimizer.step()

            self.recons_loss_ema = compute_loss_ema(
                self.recons_loss_ema, losses["recons_loss"].item()
            )
            self.reg_loss_ema = compute_loss_ema(self.reg_loss_ema, losses["reg_loss"].item())

            if not self.trained_steps % self.train_conf["log_interval"]:
                print(
                    " -- epoch {:03d} | batch {:04d}: loss = (RC: {:.4f} | {}: {:.4f}), "
                    "acc = {:.4f}, step = {}, time = {:.1f}s".format(
                        epoch,
                        batch_idx,
                        self.recons_loss_ema,
                        self.model.reg_name,
                        self.reg_loss_ema,
                        accuracy,
                        self.trained_steps,
                        time.time() - started,
                    )
                )
            self.log(
                {
                    "recons_loss": self.recons_loss_ema,
                    self.model.reg_name: self.reg_loss_ema,
                    "acc": accuracy,
                    "lr": self.optimizer.param_groups[0]["lr"],
                }
            )

            if not self.trained_steps % self.train_conf["val_interval"]:
                vallosses = validate(
                    self.model,
                    self.dloader_val,
                    self.device,
                    self.train_conf["val_batches"],
                    vocab.PAD_ID,
                )
                usage = vallosses[3]
                message = "[info] validation: RC {:.4f} | {} {:.4f} | acc {:.4f}".format(
                    np.mean(vallosses[0]),
                    self.model.reg_name,
                    np.mean(vallosses[1]),
                    np.mean(vallosses[2]),
                )
                metrics = {
                    "recons_loss_val": np.mean(vallosses[0]),
                    f"{self.model.reg_name}_val": np.mean(vallosses[1]),
                    "acc_val": np.mean(vallosses[2]),
                }
                if usage is not None:
                    message += " | codes used {:.1%} | perplexity {:.1f}".format(
                        usage["used_fraction"], usage["perplexity"]
                    )
                    metrics["codebook_used_fraction"] = usage["used_fraction"]
                    metrics["codebook_perplexity"] = usage["perplexity"]
                print(message)
                self.log(metrics)
                self.model.train()

            if not self.trained_steps % self.train_conf["ckpt_interval"]:
                self.save_checkpoint()

            if self.trained_steps >= self.total_steps:
                print(f"[info] reached total_steps {self.total_steps}")
                self.save_checkpoint()
                return False

        print(
            "[epoch {:03d}] completed -- loss = (RC: {:.4f} | {}: {:.4f}), time = {:.1f}s".format(
                epoch, self.recons_loss_ema, self.model.reg_name, self.reg_loss_ema,
                time.time() - started,
            )
        )
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--device", default=None, help="override training.device")
    parser.add_argument(
        "--model-type",
        default=None,
        choices=["ResidualVQ", "VAE"],
        help="override model.model_type",
    )
    args = parser.parse_args()

    config = yaml.load(args.config.read_text(encoding="utf-8"), Loader=yaml.FullLoader)
    if args.device:
        config["training"]["device"] = args.device
    if args.model_type:
        config["model"]["model_type"] = args.model_type

    trainer = Trainer(config, use_wandb=config["training"]["use_wandb"])
    trainer.build_data()
    trainer.build_model()
    trainer.prepare_dirs(args.config)
    trainer.maybe_start_wandb()

    for epoch in range(config["training"]["max_epochs"]):
        if not trainer.train_epoch(epoch + 1):
            break

    if trainer.wandb is not None:
        trainer.wandb.finish()


if __name__ == "__main__":
    main()
