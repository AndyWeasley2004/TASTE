#!/usr/bin/env python3
"""Reconstruct token streams with a trained window codec (VAE or residual VQ).

    python -m taste.codec.infer_reconstruct --examples 3 --device cuda:0

Loads the run's ``config.yaml`` and the latest (or chosen) ``*-model.pt``,
encodes each sample's 1 s windows to latents, teacher-forced-decodes back to
tokens, reports token accuracy and prints target vs prediction for a few
examples. Each predicted token sees the previous ground-truth tokens plus the
latent segment embeddings, as in training.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from taste import paths

from . import vocab
from .models.musetok import build_codec
from .window_group_dataset import WindowGroupDataset

DATA_ROOT = paths.data("codec_tokens")


def latest_checkpoint(ckpt_dir: Path) -> Path:
    params = sorted(ckpt_dir.glob("params/*-model.pt"))
    if not params:
        params = sorted(ckpt_dir.glob("*-model.pt"))
    if not params:
        raise SystemExit(f"no *-model.pt under {ckpt_dir}")
    # prefer the highest step_N in ``step_N-...-model.pt`` names
    def step_key(path: Path) -> int:
        name = path.name
        if name.startswith("step_"):
            try:
                return int(name.split("-", 1)[0].removeprefix("step_"))
            except ValueError:
                return -1
        return -1

    return max(params, key=lambda p: (step_key(p), p.stat().st_mtime))


def load_run(ckpt_dir: Path, device: str, checkpoint: Path | None = None):
    ckpt_dir = Path(ckpt_dir)
    config_path = ckpt_dir / "config.yaml"
    if not config_path.exists():
        raise SystemExit(f"missing {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    ckpt_path = Path(checkpoint) if checkpoint else latest_checkpoint(ckpt_dir)

    model = build_codec(config["model"], n_token=vocab.VOCAB_SIZE, pad_id=vocab.PAD_ID)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"[info] loaded {ckpt_path}")
    print(f"[info] codec {type(model).__name__} | device {device}")
    return config, model, ckpt_path


def batch_to_device(batch, device):
    return {
        "enc_inp": batch["enc_input"].permute(2, 0, 1).to(device),
        "dec_inp": batch["dec_input"].permute(1, 0).to(device),
        "dec_tgt": batch["dec_target"].permute(1, 0).to(device),
        "dec_window_idx": batch["dec_window_idx"].to(device),
        "padding_mask": batch["enc_padding_mask"].to(device),
        "piece_id": batch["piece_id"],
        "st_window_id": batch["st_window_id"],
        "enc_n_windows": batch["enc_n_windows"],
        "dec_length": batch["dec_length"],
    }


@torch.no_grad()
def reconstruct(model, inputs):
    """Encode → latent → teacher-forced decode; returns logits and latent."""
    extras = {}
    if type(model).__name__ == "TransformerVAE":
        _sampled, mu, logvar = model.encode(
            inputs["enc_inp"], padding_mask=inputs["padding_mask"]
        )
        latent = mu  # deterministic reconstruction (same as eval reparameterize)
        extras = {"mu": mu, "logvar": logvar}
    else:
        latent, indices, _commit = model.encode(
            inputs["enc_inp"], padding_mask=inputs["padding_mask"]
        )
        extras = {"indices": indices}
    logits = model.decode(inputs["dec_inp"], latent, inputs["dec_window_idx"])
    return logits, latent, extras


def token_accuracy(pred: torch.Tensor, target: torch.Tensor, pad_id: int) -> tuple[float, int, int]:
    """``pred``/``target`` shaped ``(dec_len, batch)``."""
    scored = target != pad_id
    correct = (pred == target) & scored
    n_ok = int(correct.sum().item())
    n_tot = int(scored.sum().item())
    return n_ok / max(n_tot, 1), n_ok, n_tot


def per_sample_accuracy(pred: torch.Tensor, target: torch.Tensor, pad_id: int) -> list[float]:
    scored = target != pad_id
    correct = (pred == target) & scored
    out = []
    for b in range(target.size(1)):
        n = int(scored[:, b].sum().item())
        out.append(float(correct[:, b].sum().item()) / max(n, 1))
    return out


def format_stream(ids, id_to_token, limit: int = 48) -> str:
    parts = []
    for i, tid in enumerate(ids):
        if i >= limit:
            parts.append("...")
            break
        tid = int(tid)
        if tid == vocab.PAD_ID:
            break
        parts.append(vocab.describe(tid, id_to_token))
    return " | ".join(parts)


def align_diff(target, pred, id_to_token, limit: int = 40) -> str:
    """Mark mismatches with ``target→pred``; matches as the shared token."""
    parts = []
    n = min(len(target), len(pred), limit)
    for i in range(n):
        t, p = int(target[i]), int(pred[i])
        if t == vocab.PAD_ID:
            break
        td, pd = vocab.describe(t, id_to_token), vocab.describe(p, id_to_token)
        parts.append(td if t == p else f"{td}→{pd}")
    if min(len(target), len(pred)) > limit:
        parts.append("...")
    return " | ".join(parts)


def strip_pad(ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(ids, dtype=np.int64)
    keep = ids != vocab.PAD_ID
    if not keep.any():
        return ids[:0]
    return ids[: int(np.where(keep)[0][-1]) + 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=paths.ckpt("codec", "window_vae"),
        help="run directory with config.yaml and params/",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="specific *-model.pt")
    parser.add_argument("--store", default=None, help="override data store (default: first in config)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=64, help="how many groups to score")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--examples", type=int, default=3, help="pretty-print this many samples")
    parser.add_argument("--example-tokens", type=int, default=48, help="tokens shown per example")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--appoint-start",
        type=int,
        default=0,
        help="fixed 1 s start index inside each piece (None-like: use 0)",
    )
    args = parser.parse_args()

    config, model, ckpt_path = load_run(args.ckpt, args.device, args.checkpoint)
    data_conf = config["data"]
    stores = data_conf["store_dir"]
    if isinstance(stores, (str, Path)):
        stores = [stores]
    store = Path(args.store) if args.store else Path(stores[0])
    if not store.is_absolute():
        store = DATA_ROOT / store

    dataset = WindowGroupDataset(
        store,
        model_max_windows=data_conf["max_windows"],
        model_enc_seqlen=data_conf["enc_seqlen"],
        model_dec_seqlen=data_conf["dec_seqlen"],
        do_augment=False,
        appoint_st_window=args.appoint_start,
    )
    rng = np.random.default_rng(args.seed)
    n = min(args.samples, len(dataset))
    indices = rng.choice(len(dataset), size=n, replace=False).tolist()
    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    id_to_token = vocab.load_id_to_token()
    total_ok = total_n = 0
    example_budget = args.examples
    latent_norms = []

    print(f"[info] store {store} | scoring {n} groups | max_windows={data_conf['max_windows']}")
    print()

    for batch in loader:
        inputs = batch_to_device(batch, args.device)
        logits, latent, extras = reconstruct(model, inputs)
        pred = logits.argmax(dim=-1)
        acc, n_ok, n_tot = token_accuracy(pred, inputs["dec_tgt"], vocab.PAD_ID)
        total_ok += n_ok
        total_n += n_tot
        latent_norms.append(latent.float().norm(dim=-1).mean().item())

        sample_acc = per_sample_accuracy(pred, inputs["dec_tgt"], vocab.PAD_ID)
        for b in range(pred.size(1)):
            if example_budget <= 0:
                break
            example_budget -= 1
            length = int(inputs["dec_length"][b].item())
            tgt = strip_pad(inputs["dec_tgt"][: length + 1, b].cpu().numpy())
            prd = strip_pad(pred[: length + 1, b].cpu().numpy())
            # align to the same scored span (teacher-forced targets)
            span = min(tgt.size, prd.size, length)
            tgt, prd = tgt[:span], prd[:span]

            piece = inputs["piece_id"][b]
            start = int(inputs["st_window_id"][b].item())
            n_win = int(inputs["enc_n_windows"][b].item())
            print("=" * 78)
            print(
                f"example piece={piece}  1s_start={start}  n_windows={n_win}  "
                f"acc={sample_acc[b]:.4f}  tokens={span}"
            )
            if "mu" in extras:
                mu = extras["mu"][b, :n_win]
                logvar = extras["logvar"][b, :n_win]
                print(
                    f"  latent: shape={tuple(mu.shape)}  "
                    f"|mu|={mu.float().norm(dim=-1).mean().item():.3f}  "
                    f"mean_sigma={torch.exp(0.5 * logvar).mean().item():.3f}"
                )
            elif "indices" in extras:
                idx = extras["indices"][b, :n_win]
                print(f"  latent: shape={tuple(latent[b, :n_win].shape)}  "
                      f"codes[0]={idx[0].tolist() if n_win else []}")
            else:
                print(f"  latent: shape={tuple(latent[b, :n_win].shape)}")
            print()
            print("  TARGET:")
            print("   ", format_stream(tgt, id_to_token, args.example_tokens))
            print()
            print("  PREDICTED:")
            print("   ", format_stream(prd, id_to_token, args.example_tokens))
            print()
            print("  DIFF (target→pred on mismatches):")
            print("   ", align_diff(tgt, prd, id_to_token, args.example_tokens))
            print()

    print("=" * 78)
    print(
        f"[result] token accuracy = {total_ok / max(total_n, 1):.4f}  "
        f"({total_ok}/{total_n})  over {n} groups"
    )
    print(f"[result] mean |latent| per window = {float(np.mean(latent_norms)):.3f}")
    print(f"[result] checkpoint {ckpt_path}")


if __name__ == "__main__":
    main()
