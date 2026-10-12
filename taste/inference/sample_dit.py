#!/usr/bin/env python3
"""Sample DiT latents and decode to MIDI.

Prompt IDs select section plans and lengths from the high-quality latent index.
Euler sampling runs noise-to-data with classifier-free guidance; only integer-
second latents are passed to the codec.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from taste import paths

os.environ.setdefault("HF_HUB_CACHE", str(paths.ckpt("hf_cache")))

import numpy as np
import pyarrow.parquet as pq
import torch

from taste.inference import latent_decode as ld
from taste.inference.sample_ar import encode_imagery
from taste.models.ar_conditional import ATTRIBUTE_FIELDS, load_attribute_classes
from taste.models.dit import CONTEXT_DIM, D_LATENT, DiTConfig, LatentDiT, destandardize, load_stats
from taste.representation.key import KEY_NAMES
from taste.representation.vocab import TEMPO_CLASSES, key_token_id, tempo_token_id
from taste.training.dit_data import (
    COMPLETE_NULL,
    KEY_NULL,
    SECONDS_PER_WINDOW,
    SECTION_NULL,
    START_MID_NULL,
    TEMPO_NULL,
    section_index,
)

LATENT_INDEX_ROOT = paths.data("dit_latents", "high_quality")
INDEX_SPLITS = ("val", "pass0")
SEED_BASE = 101
MODEL_FIELDS = ("d_model", "n_layers", "n_heads", "mlp_ratio", "cond_dim", "max_len")
KNOWN_PROMPT_KEYS = {*ATTRIBUTE_FIELDS, "key", "tempo", "imagery", "name"}



def load_dit(model_dir: Path, device: str, config_path: Path | None = None,
             stats_path: Path | None = None) -> tuple[LatentDiT, tuple, int]:
    """``ema.pt`` from ``model_dir``; its config and latent stats default to the
    same directory."""
    config = json.loads(Path(config_path or model_dir / "config.json").read_text())
    model = LatentDiT(DiTConfig(**{name: config[name] for name in MODEL_FIELDS}))
    model.load_state_dict(torch.load(model_dir / "ema.pt", map_location="cpu"))
    model.to(device).eval()
    stats = load_stats(stats_path or model_dir / "latent_stats.npz")
    return model, stats, config["max_len"]


def load_plans(root: Path) -> dict[str, dict]:
    """``piece_id -> {n_latents, n_windows, sections, complete, lps}`` from the store."""
    plans: dict[str, dict] = {}
    found = []
    for split in INDEX_SPLITS:
        path = root / split / "index.parquet"
        if not path.is_file():
            continue
        found.append(split)
        lps = int(json.loads((root / split / "meta.json").read_text())
                  .get("latents_per_second", 1))
        columns = pq.read_table(
            path, columns=["piece_id", "n_latents", "n_windows", "sections", "complete"]
        ).to_pydict()
        for row in range(len(columns["piece_id"])):
            plans.setdefault(columns["piece_id"][row], {
                "n_latents": int(columns["n_latents"][row]),
                "n_windows": int(columns["n_windows"][row]),
                "sections": [int(value) for value in columns["sections"][row]],
                "complete": bool(columns["complete"][row]),
                "lps": lps,
            })
    if not found:
        raise FileNotFoundError(f"no latent-store index under {root} "
                                f"(looked for {list(INDEX_SPLITS)})")
    return plans


def plan_for(prompt: dict, plans: dict[str, dict]) -> dict:
    name = prompt["name"]
    if name not in plans:
        raise KeyError(f"{name} has no row in the latent store index")
    plan = plans[name]
    if len(plan["sections"]) != plan["n_windows"]:
        raise ValueError(f"{name}: {len(plan['sections'])} sections for "
                         f"{plan['n_windows']} windows")
    per_window = ld.SUBWINDOWS * plan["lps"]
    if plan["n_latents"] != per_window * plan["n_windows"]:
        raise ValueError(f"{name}: n_latents {plan['n_latents']} is not "
                         f"{per_window}x n_windows {plan['n_windows']}")
    if min(plan["sections"]) < 0:
        raise ValueError(f"{name}: piece carries no section plan")
    return plan



def condition_tensors(rows, classes, contexts, length, device):
    """Conditioning for one batch of ``(prompt_index, prompt, plan, k)`` rows."""
    batch = len(rows)
    null_attrs = [len(classes[name]) for name in ATTRIBUTE_FIELDS]
    attrs = torch.tensor(
        [[classes[name].index(prompt[name]) for name in ATTRIBUTE_FIELDS]
         for _, prompt, _, _ in rows],
        dtype=torch.long, device=device,
    )
    key = torch.tensor([KEY_NAMES.index(prompt["key"]) for _, prompt, _, _ in rows],
                       dtype=torch.long, device=device)
    tempo = torch.tensor([TEMPO_CLASSES.index(prompt["tempo"]) for _, prompt, _, _ in rows],
                         dtype=torch.long, device=device)
    complete = torch.tensor([int(plan["complete"]) for _, _, plan, _ in rows],
                            dtype=torch.long, device=device)
    # generation always starts a piece at its beginning
    start_mid = torch.zeros(batch, dtype=torch.long, device=device)

    valid_mask = torch.zeros(batch, length, dtype=torch.bool, device=device)
    sections = torch.full((batch, length), SECTION_NULL, dtype=torch.long, device=device)
    sec_index = torch.zeros(batch, length, dtype=torch.long, device=device)
    for row, (_, _, plan, _) in enumerate(rows):
        per_latent = np.repeat(np.asarray(plan["sections"], dtype=np.int64),
                               SECONDS_PER_WINDOW * plan["lps"])
        valid_mask[row, :len(per_latent)] = True
        sections[row, :len(per_latent)] = torch.from_numpy(per_latent).to(device)
        sec_index[row, :len(per_latent)] = torch.from_numpy(
            section_index(per_latent)).to(device)

    lengths = [contexts[index][1] for index, _, _, _ in rows]
    context = torch.zeros(batch, max(lengths), CONTEXT_DIM,
                          dtype=torch.bfloat16, device=device)
    for row, (index, _, _, _) in enumerate(rows):
        state, count = contexts[index]
        if count:
            context[row, :count] = state[0].to(torch.bfloat16)
    context_mask = (torch.arange(max(lengths), device=device)[None, :]
                    < torch.tensor(lengths, device=device)[:, None])

    cond = {"attrs": attrs, "key": key, "tempo": tempo, "complete": complete,
            "start_mid": start_mid, "sections": sections, "sec_index": sec_index,
            "context": context, "context_mask": context_mask}
    null = {
        "attrs": torch.tensor(null_attrs, dtype=torch.long,
                              device=device).expand(batch, -1),
        "key": torch.full((batch,), KEY_NULL, dtype=torch.long, device=device),
        "tempo": torch.full((batch,), TEMPO_NULL, dtype=torch.long, device=device),
        "complete": torch.full((batch,), COMPLETE_NULL, dtype=torch.long, device=device),
        "start_mid": torch.full((batch,), START_MID_NULL, dtype=torch.long, device=device),
        "sections": torch.full((batch, length), SECTION_NULL, dtype=torch.long, device=device),
        "sec_index": torch.zeros(batch, length, dtype=torch.long, device=device),
        "context": torch.zeros(batch, 0, CONTEXT_DIM, dtype=torch.bfloat16, device=device),
        "context_mask": torch.zeros(batch, 0, dtype=torch.bool, device=device),
    }
    return valid_mask, cond, null


@torch.no_grad()
def sample_batch(model, rows, classes, contexts, stats, args, device) -> list[np.ndarray]:
    """Euler integration of the rectified-flow ODE; returns per-row ``μ``."""
    batch = len(rows)
    length = max(plan["n_latents"] for _, _, plan, _ in rows)
    valid_mask, cond, null = condition_tensors(rows, classes, contexts, length, device)

    x = torch.zeros(batch, length, D_LATENT, device=device)
    for row, (_, _, _, seed) in enumerate(rows):
        generator = torch.Generator(device=device).manual_seed(seed)
        x[row] = torch.randn(length, D_LATENT, generator=generator, device=device)

    dt = 1.0 / args.steps
    for step in range(args.steps):
        t = torch.full((batch,), 1.0 - step * dt, device=device)
        v = model(x, t, valid_mask, **cond)
        if args.cfg != 1.0:
            v_null = model(x, t, valid_mask, **null)
            v = v_null + args.cfg * (v - v_null)
        x = x - dt * v

    latents = destandardize(x.float().cpu().numpy(), stats)
    return [latents[row, :plan["n_latents"]] for row, (_, _, plan, _) in enumerate(rows)]



def sample_latents(model, stats, max_len: int, prompts: list[dict],
                   plans: dict[str, dict], contexts, args, device):
    """Sample every ``prompt x seed`` row; returns the rows and their ``μ``."""
    classes = load_attribute_classes()
    rows = []
    for index, prompt in enumerate(prompts):
        plan = plan_for(prompt, plans)
        if plan["n_latents"] > max_len:
            raise ValueError(f"{prompt['name']}: {plan['n_latents']} latents exceed "
                             f"the model's max_len {max_len}")
        rows += [(index, prompt, plan, SEED_BASE + k) for k in range(args.n_seeds)]

    mus, order = [], []
    for start in range(0, len(rows), args.batch):
        chunk = rows[start:start + args.batch]
        mus += sample_batch(model, chunk, classes, contexts, stats, args, device)
        order += chunk
    return order, mus


def decode_and_write(order, mus, args, device):
    """Latents → MIDI + sidecars under ``<output_dir>/generated``.

    Returns the per-piece per-second token arrays (the decoder's own output,
    from which cheap symbolic statistics are read) and the decode stats.
    """
    _, codec = ld.load_codec(device)
    # the decoder consumes one latent per second, so a half-hop store's
    # half-offset latents are dropped before decoding
    per_second = [mu[:: plan["lps"]] for mu, (_, _, plan, _) in zip(mus, order)]
    seconds, decode_stats = ld.decode_pieces(codec, per_second, device,
                                             batch=args.batch)
    print(f"[decode] {decode_stats}", flush=True)

    midi_dir = args.output_dir / "generated"
    midi_dir.mkdir(parents=True, exist_ok=True)
    for (index, prompt, plan, seed), piece in zip(order, seconds):
        sections = [value + 1 for value in plan["sections"]]
        row = ld.rebuild_row(piece, key_token_id(prompt["key"]),
                             tempo_token_id(prompt["tempo"]), sections,
                             complete=plan["complete"])
        name = f"{prompt['name']}_k{seed - SEED_BASE}"
        ld.write_midi(row, midi_dir / f"{name}.mid")
        record = {
            **{key: value for key, value in prompt.items()
               if key not in KNOWN_PROMPT_KEYS},
            "name": name,
            "prompt_index": index + 1,
            **{field: prompt.get(field) for field in ATTRIBUTE_FIELDS},
            "key": prompt["key"],
            "tempo": prompt["tempo"],
            "imagery": prompt.get("imagery"),
            "seed": seed,
            "plan": [[chr(ord("A") + value - 1), length]
                     for value, length in _runs(sections)],
            "n_windows": plan["n_windows"],
            "complete": plan["complete"],
            "cfg": args.cfg,
            "steps": args.steps,
            "n_generated_tokens": len(row),
            "ended_naturally": plan["complete"],
        }
        (midi_dir / f"{name}.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({k: record[k] for k in
                          ["name", "n_windows", "n_generated_tokens", "plan"]}), flush=True)
    return seconds, decode_stats



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--prompts-json", type=Path, required=True)
    parser.add_argument("--latent-index-root", type=Path, default=LATENT_INDEX_ROOT)
    parser.add_argument("--n-seeds", type=int, default=8)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = f"cuda:{args.gpu}"
    prompts = json.loads(args.prompts_json.read_text())
    plans = load_plans(args.latent_index_root)
    model, stats, max_len = load_dit(args.model_dir, device)

    contexts = encode_imagery([prompt["imagery"] for prompt in prompts], device)
    order, mus = sample_latents(model, stats, max_len, prompts, plans,
                                contexts, args, device)
    del model
    torch.cuda.empty_cache()
    decode_and_write(order, mus, args, device)


def _runs(sections: list[int]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    for value in sections:
        if runs and runs[-1][0] == value:
            runs[-1] = (value, runs[-1][1] + 1)
        else:
            runs.append((value, 1))
    return runs


if __name__ == "__main__":
    main()
