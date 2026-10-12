#!/usr/bin/env python3
"""Unconditional sampling from the TASTE-AR pretrain checkpoint.

The prompt is the two-token header the corpus rows start with,
``<KEY_*> <TEMPO_*>``; the model then emits its own section plan (``<SEC_*>``
runs with ``<PLAN_*>`` digit counts), the piano prefix, ``<S>``, and the Aria
stream including ``<SEC_*>`` states, until ``<E>``.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from taste import paths

os.environ.setdefault("HF_HUB_CACHE", str(paths.ckpt("hf_cache")))

import torch
from transformers import LogitsProcessor

from taste.models.ar import load_model
from taste.representation.tokenizer import get_tokenizer
from taste.representation.vocab import (
    KEY_BASE,
    MODEL_VOCAB_SIZE,
    PAD_ID,
    PLAN_IDS,
    key_token_id,
    tempo_token_id,
)

KEYS = [
    "C major", "G major", "A minor", "D minor",
    "Eb major", "F major", "E minor", "Bb major",
]
PROMPT_LEN = 2


class MidiGrammar(LogitsProcessor):
    """Section plan first, then ``<S>``, then sampling inside the MIDI stream.

    Two states per row: in-plan (only ``<SEC_*>``/``<PLAN_*>`` digits, plus the
    piano prefix the model uses to end the plan whenever it likes) and
    in-stream. The prefix forces ``<S>`` on the following step.
    """

    def __init__(
        self,
        batch_size: int,
        device: torch.device,
    ) -> None:
        midi = get_tokenizer()
        self.start_id = midi.start_id
        self.piano_prefix_id = midi.piano_prefix_id
        banned = {
            self.start_id,
            self.piano_prefix_id,
            # <P> is a real Aria vocab id here, so banning it is what makes
            # trailing generation padding unambiguous when trimming rows.
            PAD_ID,
            # key/tempo/plan/<S_MID> + unused tail; generation is always from
            # scratch, so <S_MID> must never be reachable -- the plan mask and
            # the forced-<S> start mask are explicit allow-lists that exclude it
            # and this range bans it inside the stream
            *range(KEY_BASE, MODEL_VOCAB_SIZE),
        }
        self.in_plan = torch.ones(batch_size, dtype=torch.bool, device=device)
        self.just_prefix = torch.zeros(batch_size, dtype=torch.bool, device=device)
        self._device = device
        self.plan_mask = self._mask(
            [*midi.section_ids, *PLAN_IDS, self.piano_prefix_id]
        )
        self.start_mask = self._mask([self.start_id])
        self.stream_mask = self._mask(
            [token for token in range(MODEL_VOCAB_SIZE) if token not in banned]
        )

    def _mask(self, ids) -> torch.Tensor:
        mask = torch.full((MODEL_VOCAB_SIZE,), -float("inf"), device=self._device)
        mask[torch.tensor(ids, device=self._device)] = 0.0
        return mask

    def __call__(self, input_ids, scores):
        last = input_ids[:, -1]
        self.just_prefix = self.in_plan & (last == self.piano_prefix_id)
        self.in_plan &= ~self.just_prefix
        mask = torch.where(
            self.just_prefix[:, None],
            self.start_mask,
            torch.where(self.in_plan[:, None], self.plan_mask, self.stream_mask),
        )
        return scores + mask.to(scores.dtype)


def parse_plan(plan_tokens, midi_tokenizer) -> list[list]:
    """``[[letter, n_windows], ...]`` from the sampled ``<SEC_*>``/``<PLAN_*>`` run."""
    id_to_letter = {
        token_id: token[5]
        for token, token_id in midi_tokenizer.section_to_id.items()
    }
    runs: list[list] = []
    digits = ""
    for token in plan_tokens:
        if token in id_to_letter:
            if runs:
                runs[-1][1] = int(digits) if digits else 0
            runs.append([id_to_letter[token], 0])
            digits = ""
        else:
            digits += str(token - PLAN_IDS[0])
    if runs:
        runs[-1][1] = int(digits) if digits else 0
    return runs


def window_states(stream, midi_tokenizer) -> list[str]:
    """Section letter per 5 s window: the state follows ``<S>`` and every ``<T>``."""
    id_to_letter = {
        token_id: token[5]
        for token, token_id in midi_tokenizer.section_to_id.items()
    }
    states: list[str] = []
    expect = False
    for token in stream:
        if token in (midi_tokenizer.start_id, midi_tokenizer.time_id):
            states.append(states[-1] if states else "?")
            expect = True
        elif expect:
            if token in id_to_letter:
                states[-1] = id_to_letter[token]
            expect = False
    return states


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--keys", default=",".join(KEYS))
    parser.add_argument("--tempo", default="moderate")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.98)
    parser.add_argument("--max-new-tokens", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rotation = [key.strip() for key in args.keys.split(",")]
    keys = [rotation[index % len(rotation)] for index in range(args.num_samples)]
    tempo_id = tempo_token_id(args.tempo)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(
        args.model_dir, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda")
    model.eval()
    model.config.use_cache = True
    midi_tokenizer = get_tokenizer()

    prompts = torch.tensor(
        [[key_token_id(key), tempo_id] for key in keys], device="cuda"
    )
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    max_new_tokens = min(args.max_new_tokens, 8192 - PROMPT_LEN)
    with torch.inference_mode():
        output = model.generate(
            input_ids=prompts,
            attention_mask=torch.ones_like(prompts),
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=0,
            max_new_tokens=max_new_tokens,
            eos_token_id=midi_tokenizer.end_id,
            pad_token_id=PAD_ID,
            use_cache=True,
            logits_processor=[MidiGrammar(len(keys), prompts.device)],
        )

    records = []
    for index, key in enumerate(keys):
        name = f"uncond_s{index + 1}"
        generated = output[index, PROMPT_LEN:].tolist()
        while generated and generated[-1] == PAD_ID:
            generated.pop()
        ended_naturally = midi_tokenizer.end_id in generated
        if ended_naturally:
            generated = generated[:generated.index(midi_tokenizer.end_id) + 1]
        # the plan header is not Aria vocab; the stream starts at the prefix
        plan_len = generated.index(midi_tokenizer.piano_prefix_id)
        stream = generated[plan_len:]
        midi_path = output_dir / f"{name}.mid"
        # detokenize_midi drops <SEC_*> and <D> itself, so the section states
        # in the stream need no filtering here.
        music = midi_tokenizer.detokenize_midi(stream)
        music.to_midi().save(midi_path)
        duration = max(
            (note["data"]["end"] for note in music.note_msgs), default=0
        ) / 1000.0
        record = {
            "name": name,
            "key": key,
            "tempo": args.tempo,
            "seed": args.seed,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "ended_naturally": ended_naturally,
            "hit_token_cap": len(generated) >= max_new_tokens,
            "n_generated_tokens": len(generated),
            "n_plan_tokens": plan_len,
            "n_sections": sum(
                1 for token in stream if token in midi_tokenizer.section_ids
            ),
            "midi_path": str(midi_path),
            "duration_sec": duration,
        }
        plan = parse_plan(generated[:plan_len], midi_tokenizer)
        states = window_states(stream, midi_tokenizer)
        sidecar = {
            "key": key,
            "tempo_class": args.tempo,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "plan": plan,
            "window_states": states,
            "truncated": not ended_naturally,
            "n_tokens": len(generated),
        }
        (output_dir / f"{name}.json").write_text(
            json.dumps(sidecar, indent=2) + "\n", encoding="utf-8"
        )
        record["plan"] = plan
        record["n_windows"] = len(states)
        records.append(record)
        print(json.dumps(record), flush=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
