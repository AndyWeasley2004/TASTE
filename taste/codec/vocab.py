"""Codec vocabulary: joint piano note heads (pitch x velocity), onsets, durations.

Matches ``tokenizer/midi_vocab.json`` inside the codec checkpoint directory. Each note is
``piano(pitch, vel) onset dur`` (3 tokens). Onsets in a 5 s store window are
relative to that window's start; the online 1 s split rebases them to
``[0, 1000)`` ms.
"""
from __future__ import annotations

import json

from taste import paths

VOCAB_SIZE = 2176

S_ID = 0
E_ID = 1
PAD_ID = 2
UNK_ID = 3
D_ID = 4
T_ID = 5
PREFIX_PIANO_ID = 6
PED_ON_ID = 2072
PED_OFF_ID = 2073
S_MID_ID = 2119

PIANO_LO, PIANO_HI = 7, 1670
DUR_LO, DUR_HI = 1671, 1871
ONSET_LO, ONSET_HI = 1872, 2071
SEC_LO, SEC_HI = 2074, 2081
KEY_LO, KEY_HI = 2082, 2105
TEMPO_LO, TEMPO_HI = 2106, 2108
PLAN_LO, PLAN_HI = 2109, 2118

TIME_STEP_MS = 25
WINDOW_MS = 5000
SUBWINDOW_MS = 1000
VELOCITY_STEP = 10
VELOCITY_BINS = 13
N_PITCHES = 128

FORMAT_VERSION = "taste_vocab_25ms"
ARIA_VOCAB_SIZE = 2082
UNUSED_LO, UNUSED_HI = 2120, 2175

VOCAB_DIR = paths.ckpt("codec", "window_vae", "tokenizer")


def piano_id(pitch: int, velocity_bin: int) -> int:
    return PIANO_LO + pitch * VELOCITY_BINS + velocity_bin


def piano_fields(token_id: int) -> tuple[int, int]:
    offset = token_id - PIANO_LO
    return offset // VELOCITY_BINS, (offset % VELOCITY_BINS) * VELOCITY_STEP


def onset_ms(token_id: int) -> int:
    return (token_id - ONSET_LO) * TIME_STEP_MS


def dur_ms(token_id: int) -> int:
    return (token_id - DUR_LO) * TIME_STEP_MS


def onset_id(onset_ms_value: int) -> int:
    return ONSET_LO + onset_ms_value // TIME_STEP_MS


def is_event_head(token_id: int) -> bool:
    return (
        PIANO_LO <= token_id <= PIANO_HI
        or token_id == PED_ON_ID
        or token_id == PED_OFF_ID
    )


def event_length(token_id: int) -> int:
    return 3 if PIANO_LO <= token_id <= PIANO_HI else 2


def load_vocab_json() -> dict:
    path = VOCAB_DIR / "midi_vocab.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_id_to_token() -> list:
    return load_vocab_json()["id_to_token"]


def describe(token_id: int, id_to_token: list) -> str:
    token = id_to_token[token_id]
    if token is None:
        return f"<UNUSED_{token_id}>"
    if isinstance(token, str):
        return token
    kind = token[0]
    if kind == "piano":
        return f"piano(p{token[1]},v{token[2]})"
    if kind == "onset":
        return f"onset+{token[1]}ms"
    if kind == "dur":
        return f"dur={token[1]}ms"
    return "/".join(str(part) for part in token)


def verify() -> None:
    """Fail loudly if the vocabulary file disagrees with the constants."""
    meta = load_vocab_json()
    id_to_token = meta["id_to_token"]
    if len(id_to_token) != VOCAB_SIZE:
        raise ValueError(f"vocab has {len(id_to_token)} ids, expected {VOCAB_SIZE}")

    declared = {
        "format_version": FORMAT_VERSION,
        "model_vocab_size": VOCAB_SIZE,
        "aria_vocab_size": ARIA_VOCAB_SIZE,
        "bos_id": S_ID,
        "eos_id": E_ID,
        "pad_id": PAD_ID,
        "s_mid_id": S_MID_ID,
        "key_base": KEY_LO,
        "tempo_base": TEMPO_LO,
        "plan_base": PLAN_LO,
    }
    for field, expected in declared.items():
        if meta[field] != expected:
            raise ValueError(f"midi_vocab.json {field}={meta[field]!r}, expected {expected!r}")
    if meta["unused_ids"] != list(range(UNUSED_LO, UNUSED_HI + 1)):
        raise ValueError(f"unused ids are not exactly [{UNUSED_LO}, {UNUSED_HI}]")

    specials = {
        "<S>": S_ID,
        "<E>": E_ID,
        "<P>": PAD_ID,
        "<U>": UNK_ID,
        "<D>": D_ID,
        "<T>": T_ID,
        "<PED_ON>": PED_ON_ID,
        "<PED_OFF>": PED_OFF_ID,
        "<S_MID>": S_MID_ID,
    }
    for name, token_id in specials.items():
        if id_to_token[token_id] != name:
            raise ValueError(f"id {token_id} is {id_to_token[token_id]!r}, expected {name}")
    if id_to_token[PREFIX_PIANO_ID] != ["prefix", "instrument", "piano"]:
        raise ValueError(f"id {PREFIX_PIANO_ID} is not the piano prefix")

    for token_id in range(PIANO_LO, PIANO_HI + 1):
        pitch, velocity = piano_fields(token_id)
        if id_to_token[token_id] != ["piano", pitch, velocity]:
            raise ValueError(f"note-head arithmetic wrong at id {token_id}")
    for token_id in range(ONSET_LO, ONSET_HI + 1):
        if id_to_token[token_id] != ["onset", onset_ms(token_id)]:
            raise ValueError(f"onset arithmetic wrong at id {token_id}")
    for token_id in range(DUR_LO, DUR_HI + 1):
        if id_to_token[token_id] != ["dur", dur_ms(token_id)]:
            raise ValueError(f"duration arithmetic wrong at id {token_id}")


if __name__ == "__main__":
    verify()
    print(f"vocab layout matches {VOCAB_DIR / 'midi_vocab.json'}")
