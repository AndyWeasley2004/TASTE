#!/usr/bin/env python3
"""Conditional sampling from a TASTE-AR conditional-stage checkpoint.

Each prompt carries the four controlled attributes, a key/tempo header and a
free-text imagery description, which is encoded with frozen Qwen3-Embedding-4B
using the same recipe as ``taste.representation.imagery_states``. Sampling is
T 0.98 / top_p 0.98 within the 4,096-token context.

Rows are ordered prompt-major, seed-minor and sampled in batched ``generate``
calls. One RNG seed (the first seed) drives every batch, so the per-row
``seed`` in the sidecar labels the variant; a row is reproducible only by
re-running the same batch layout.

    python -m taste.inference.sample_ar \
        --model-dir $TASTE_CKPT_ROOT/ar/high_quality/final \
        --prompts-json prompts.json --n-seeds 8 --rows-per-batch 32 \
        --output-dir out/ar_high_quality
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from taste import paths

os.environ.setdefault("HF_HUB_CACHE", str(paths.ckpt("hf_cache")))

import torch
from transformers import AutoModel, AutoTokenizer

from taste.inference.sample_ar_pretrain import MidiGrammar, PROMPT_LEN, parse_plan
from taste.models.ar_conditional import (
    ATTRIBUTE_FIELDS,
    load_attribute_classes,
    load_conditional_model,
)
from taste.representation.imagery_states import MAX_LEN, MODEL_NAME, MODEL_REVISION
from taste.representation.tokenizer import get_tokenizer
from taste.representation.vocab import PAD_ID, key_token_id, tempo_token_id

MAX_CONTEXT = 4096

SEED_BASE = 101

# prompt keys the sidecar records explicitly; anything else in a prompts.json
# entry (piece_id, reference_midi, ...) is copied into the sidecar verbatim
KNOWN_PROMPT_KEYS = {*ATTRIBUTE_FIELDS, "key", "tempo", "imagery", "name"}


def attribute_ids(condition: dict[str, str | None], classes) -> list[int]:
    return [
        len(classes[field]) if condition[field] is None
        else classes[field].index(condition[field])
        for field in ATTRIBUTE_FIELDS
    ]


def encode_imagery(texts: list[str], device) -> list[tuple[torch.Tensor, int]]:
    """Qwen3-Embedding-4B per-token final-layer states, pad positions dropped."""
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    tok.padding_side = "right"  # causal model: trailing pads cannot leak in
    encoder = AutoModel.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION, dtype=torch.bfloat16
    ).to(device)
    encoder.eval()
    enc = tok(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)
    with torch.inference_mode():
        out = encoder(input_ids=input_ids, attention_mask=mask).last_hidden_state
    lengths = mask.sum(1).tolist()
    states = [(out[i : i + 1, : lengths[i]].clone(), int(lengths[i])) for i in range(len(texts))]
    del encoder
    torch.cuda.empty_cache()
    return states


def emitted_sections(stream_tokens, midi) -> str:
    """Per-window section letters emitted in-stream (after ``<S>`` and each
    ``<T>``), in order. The grammar does not force these to follow the plan
    header, so they are recorded separately from it."""
    id_to_letter = {
        token_id: token[5] for token, token_id in midi.section_to_id.items()
    }
    return "".join(
        id_to_letter[token] for token in stream_tokens if token in id_to_letter
    )


def build_inputs(chunk, classes, contexts):
    """Sampler tensors for one chunk of rows (rows are ``(position, ...)``)."""
    specs = [spec for _, _, spec, _, _ in chunk]
    prompt_ids = torch.tensor(
        [[key_token_id(spec["key"]), tempo_token_id(spec["tempo"])] for spec in specs],
        device="cuda",
    )
    ids = torch.tensor([attribute_ids(spec, classes) for spec in specs], device="cuda")
    lengths = torch.tensor(
        [contexts[position][1] for position, _, _, _, _ in chunk], device="cuda"
    )
    context = torch.zeros(
        (len(chunk), int(lengths.max()), contexts[0][0].shape[-1]),
        dtype=contexts[0][0].dtype,
        device="cuda",
    )
    for row, (position, _, _, _, _) in enumerate(chunk):
        state, length = contexts[position]
        context[row, :length] = state[0]
    return prompt_ids, ids, context, lengths


def trim_rows(output, midi):
    """``(tokens, ended)`` per row: trailing pads dropped, cut after ``<E>``."""
    rows = []
    for row in output[:, PROMPT_LEN:].tolist():
        while row and row[-1] == PAD_ID:
            row.pop()
        ended = midi.end_id in row
        if ended:
            row = row[: row.index(midi.end_id) + 1]
        rows.append((row, ended))
    return rows


def generate_batch(model, prompt_ids, attributes, context, lengths, args):
    """One batched sampling call; returns ``(tokens, ended)`` per row."""
    midi = get_tokenizer()
    with torch.inference_mode():
        output = model.generate(
            input_ids=prompt_ids,
            attention_mask=torch.ones_like(prompt_ids),
            attribute_ids=attributes,
            context_states=context,
            context_lengths=lengths,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=0,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=midi.end_id,
            pad_token_id=PAD_ID,
            use_cache=True,
            logits_processor=[MidiGrammar(prompt_ids.shape[0], prompt_ids.device)],
        )
    return trim_rows(output, midi)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--prompts-json", type=Path, required=True,
        help="JSON list of prompts (attributes, key, tempo, imagery, optional name)",
    )
    parser.add_argument("--temperature", type=float, default=0.98)
    parser.add_argument("--top-p", type=float, default=0.98)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_CONTEXT - PROMPT_LEN)
    parser.add_argument(
        "--n-seeds", type=int, default=3,
        help=f"seeds are {SEED_BASE}..{SEED_BASE}+K-1",
    )
    parser.add_argument(
        "--rows-per-batch", type=int, default=0,
        help="0 samples every row in one generate call; >0 chunks the row list",
    )
    args = parser.parse_args()

    args.max_new_tokens = min(args.max_new_tokens, MAX_CONTEXT - PROMPT_LEN)
    midi_dir = args.output_dir / "generated"
    midi_dir.mkdir(parents=True, exist_ok=True)

    prompts = json.loads(args.prompts_json.read_text())
    prompts = list(enumerate(prompts, start=1))  # 1-based prompt_index
    seeds = list(range(SEED_BASE, SEED_BASE + args.n_seeds))

    classes = load_attribute_classes()
    contexts = encode_imagery([spec["imagery"] for _, spec in prompts], "cuda")
    model = load_conditional_model(args.model_dir, attn_implementation="sdpa").to("cuda")
    model.eval()
    model.config.use_cache = True
    midi = get_tokenizer()

    # row order: prompt-major, seed-minor
    rows = [(position, index, prompt_spec, seed_index, seed)
            for position, (index, prompt_spec) in enumerate(prompts)
            for seed_index, seed in enumerate(seeds)]
    size = args.rows_per_batch if args.rows_per_batch > 0 else len(rows)
    chunks = [rows[start:start + size] for start in range(0, len(rows), size)]

    torch.manual_seed(seeds[0])
    torch.cuda.manual_seed_all(seeds[0])
    outputs, batch_of_row = [], []
    for batch_index, chunk in enumerate(chunks):
        outputs += generate_batch(
            model, *build_inputs(chunk, classes, contexts), args
        )
        batch_of_row += [batch_index] * len(chunk)

    for row, ((_, index, prompt_spec, seed_index, seed), (generated, ended)) in enumerate(
        zip(rows, outputs)
    ):
        plan_len = generated.index(midi.piano_prefix_id)
        name = (f"{prompt_spec['name']}_k{seed_index}" if "name" in prompt_spec
                else f"prompt{index}_seed{seed}")
        path = midi_dir / f"{name}.mid"
        music = midi.detokenize_midi(generated[plan_len:])
        music.to_midi().save(path)
        extras = {key: value for key, value in prompt_spec.items()
                  if key not in KNOWN_PROMPT_KEYS}
        record = {
            **extras,
            "name": name,
            "prompt_index": index,
            **{field: prompt_spec[field] for field in ATTRIBUTE_FIELDS},
            "key": prompt_spec["key"],
            "tempo": prompt_spec["tempo"],
            "imagery": prompt_spec["imagery"],
            "seed": seed,
            "batch_row": row,
            "batch_index": batch_of_row[row],
            "batched": True,
            "n_generated_tokens": len(generated),
            "ended_naturally": ended,
            "plan": parse_plan(generated[:plan_len], midi),
            "emitted_sections": emitted_sections(generated[plan_len:], midi),
            "tokens": generated,
        }
        (midi_dir / f"{name}.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
        print(
            json.dumps({k: record[k] for k in
                        ["name", "n_generated_tokens", "ended_naturally", "plan"]}),
            flush=True,
        )


if __name__ == "__main__":
    main()
