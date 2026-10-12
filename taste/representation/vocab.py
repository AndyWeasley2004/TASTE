"""Model vocabulary of the piano generators: the Aria piano tokens plus key, tempo-class, section-plan and ``<S_MID>`` specials.

Layout::

    0    .. 2081   Aria piano vocab (includes <S>, <E>, <P>, <T>, <D>, <SEC_A>..<SEC_H>)
    2082 .. 2105   24 key tokens, 12 majors then 12 minors
    2106 .. 2108   3 tempo-class tokens
    2109 .. 2118   10 section-plan digit tokens <PLAN_0>..<PLAN_9>
    2119          <S_MID>, the start token of a crop that does not begin at the
                  piece's true beginning
    2120 .. 2175   unused padding to a multiple of 64
"""
from __future__ import annotations

import json
from pathlib import Path

from taste.representation.tokenizer import get_tokenizer
from taste.representation.key import KEY_NAMES, key_token

# order defines the tempo token ids
TEMPO_CLASSES = ("slow", "moderate", "fast")

ARIA_VOCAB_SIZE = 2082
KEY_BASE = ARIA_VOCAB_SIZE
TEMPO_BASE = KEY_BASE + len(KEY_NAMES)
PLAN_BASE = TEMPO_BASE + len(TEMPO_CLASSES)
S_MID_ID = PLAN_BASE + 10
MODEL_VOCAB_SIZE = 2176  # 2,120 used ids padded to a multiple of 64

FORMAT_VERSION = "taste_vocab_25ms"
VOCAB_META_NAME = "midi_vocab.json"

KEY_TOKENS = tuple(key_token(name) for name in KEY_NAMES)
TEMPO_TOKENS = tuple(f"<TEMPO_{name.upper()}>" for name in TEMPO_CLASSES)
PLAN_TOKENS = tuple(f"<PLAN_{digit}>" for digit in range(10))
S_MID_TOKEN = "<S_MID>"

KEY_TO_ID = {name: KEY_BASE + index for index, name in enumerate(KEY_NAMES)}
TEMPO_TO_ID = {name: TEMPO_BASE + index for index, name in enumerate(TEMPO_CLASSES)}
PLAN_TO_ID = {digit: PLAN_BASE + digit for digit in range(10)}
PLAN_IDS = tuple(PLAN_TO_ID.values())

ID_TO_NAME = {
    **{KEY_BASE + index: token for index, token in enumerate(KEY_TOKENS)},
    **{TEMPO_BASE + index: token for index, token in enumerate(TEMPO_TOKENS)},
    **{PLAN_BASE + index: token for index, token in enumerate(PLAN_TOKENS)},
    S_MID_ID: S_MID_TOKEN,
}
NAME_TO_ID = {token: token_id for token_id, token in ID_TO_NAME.items()}

UNUSED_IDS = tuple(range(S_MID_ID + 1, MODEL_VOCAB_SIZE))

_ARIA = get_tokenizer()
if _ARIA.vocab_size != ARIA_VOCAB_SIZE:
    raise ValueError(f"aria vocab is {_ARIA.vocab_size}, expected {ARIA_VOCAB_SIZE}")
BOS_ID = _ARIA.start_id
EOS_ID = _ARIA.end_id
PAD_ID = _ARIA.tok_to_id["<P>"]


def key_token_id(key: str) -> int:
    if key not in KEY_TO_ID:
        raise ValueError(f"unknown key {key!r}; expected one of {KEY_NAMES}")
    return KEY_TO_ID[key]


def tempo_token_id(tempo: str) -> int:
    if tempo not in TEMPO_TO_ID:
        raise ValueError(f"unknown tempo class {tempo!r}; expected one of {TEMPO_CLASSES}")
    return TEMPO_TO_ID[tempo]


def plan_count_ids(count: int) -> list[int]:
    """Decimal digit tokens of a section-plan run length, most significant first."""
    if count < 1:
        raise ValueError(f"section-plan run length must be >= 1, got {count}")
    return [PLAN_TO_ID[int(digit)] for digit in str(count)]


def vocab_meta() -> dict:
    return {
        "format_version": FORMAT_VERSION,
        "aria_vocab_size": ARIA_VOCAB_SIZE,
        "model_vocab_size": MODEL_VOCAB_SIZE,
        "key_base": KEY_BASE,
        "tempo_base": TEMPO_BASE,
        "plan_base": PLAN_BASE,
        "s_mid_id": S_MID_ID,
        "bos_id": BOS_ID,
        "eos_id": EOS_ID,
        "pad_id": PAD_ID,
        "key_to_id": KEY_TO_ID,
        "tempo_to_id": TEMPO_TO_ID,
        "plan_to_id": {str(digit): token_id for digit, token_id in PLAN_TO_ID.items()},
        "id_to_name": {str(key): value for key, value in ID_TO_NAME.items()},
        "unused_ids": list(UNUSED_IDS),
    }


def save_vocab_meta(directory: str | Path) -> Path:
    path = Path(directory) / VOCAB_META_NAME
    path.write_text(json.dumps(vocab_meta(), indent=2) + "\n", encoding="utf-8")
    return path


def load_vocab_meta(directory: str | Path) -> dict:
    path = Path(directory) / VOCAB_META_NAME
    meta = json.loads(path.read_text(encoding="utf-8"))
    if meta["format_version"] != FORMAT_VERSION:
        raise ValueError(
            f"{path} uses format {meta['format_version']!r}; expected {FORMAT_VERSION!r}"
        )
    return meta
