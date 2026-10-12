"""Cut full-piece token rows into 5-second windows, and those into 1-second ones.

A corpus row is ``<KEY_*> <TEMPO_*> <plan> <prefix> <S> ... <E>``; ``<T>`` is a
hard 5-second boundary and onsets inside a span are relative to its start. The
header and in-stream ``<SEC_*>`` tokens are moved into metadata so the encoder
sees note content only (``piano onset dur`` / pedal events). ``split_piece``
produces the 5 s store rows; ``split_5s_to_1s`` is the online train-time split.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import vocab


@dataclass(frozen=True)
class Piece:
    """One parquet row, split into 5 s windows with header conditioning."""

    piece_id: str
    key_id: int
    tempo_id: int
    windows: list[np.ndarray]
    sections: np.ndarray  # uint8, 1..8 for <SEC_A>..<SEC_H>, 0 if absent
    plan: list[tuple[int, int]]

    @property
    def n_windows(self) -> int:
        return len(self.windows)


def parse_header(ids: np.ndarray) -> tuple[int, int, list[tuple[int, int]], int]:
    """Read the pre-``<S>`` header; returns key, tempo, plan runs and body start."""
    start = np.flatnonzero((ids == vocab.S_ID) | (ids == vocab.S_MID_ID))
    if start.size == 0:
        raise ValueError("row has no <S>/<S_MID>")
    body_start = int(start[0])
    header = ids[:body_start]

    key_id = tempo_id = -1
    plan: list[tuple[int, int]] = []
    index = 0
    while index < header.size:
        token = int(header[index])
        if vocab.KEY_LO <= token <= vocab.KEY_HI:
            key_id = token
            index += 1
        elif vocab.TEMPO_LO <= token <= vocab.TEMPO_HI:
            tempo_id = token
            index += 1
        elif vocab.SEC_LO <= token <= vocab.SEC_HI:
            section = token - vocab.SEC_LO + 1
            index += 1
            digits = ""
            while index < header.size and vocab.PLAN_LO <= header[index] <= vocab.PLAN_HI:
                digits += str(int(header[index]) - vocab.PLAN_LO)
                index += 1
            plan.append((section, int(digits) if digits else 0))
        else:
            index += 1
    return key_id, tempo_id, plan, body_start


def truncate_to_events(window: np.ndarray, max_tokens: int) -> np.ndarray:
    """Drop trailing tokens so the window fits ``max_tokens`` whole events."""
    if window.size <= max_tokens:
        return window
    heads = np.flatnonzero(
        ((window >= vocab.PIANO_LO) & (window <= vocab.PIANO_HI))
        | (window == vocab.PED_ON_ID)
        | (window == vocab.PED_OFF_ID)
    )
    cut = np.searchsorted(heads, max_tokens, side="right")
    return window[: int(heads[cut - 1])] if cut > 0 else window[:0]


def split_piece(
    piece_id: str,
    input_ids,
    max_tokens: int | None = None,
) -> Piece:
    """Split one corpus row into per-window note-content sequences (5 s)."""
    ids = np.asarray(input_ids, dtype=np.int32)
    key_id, tempo_id, plan, body_start = parse_header(ids)
    body = ids[body_start + 1 :]

    bounds = np.flatnonzero(body == vocab.T_ID)
    starts = np.concatenate(([0], bounds + 1))
    stops = np.concatenate((bounds, [body.size]))

    windows: list[np.ndarray] = []
    sections = np.zeros(starts.size, dtype=np.uint8)
    for index, (lo, hi) in enumerate(zip(starts, stops)):
        window = body[lo:hi]
        if window.size and vocab.SEC_LO <= window[0] <= vocab.SEC_HI:
            sections[index] = int(window[0]) - vocab.SEC_LO + 1
            window = window[1:]
        if window.size and (window[-1] == vocab.E_ID or window[-1] == vocab.D_ID):
            window = window[(window != vocab.E_ID) & (window != vocab.D_ID)]
        if max_tokens is not None:
            window = truncate_to_events(window, max_tokens)
        windows.append(window)

    return Piece(
        piece_id=piece_id,
        key_id=key_id,
        tempo_id=tempo_id,
        windows=windows,
        sections=sections,
        plan=plan,
    )


def iter_events(window: np.ndarray):
    """Yield ``(kind, pitch, velocity, onset_ms, dur_ms)`` for a joint-piano window."""
    index = 0
    while index < window.size:
        head = int(window[index])
        if vocab.PIANO_LO <= head <= vocab.PIANO_HI:
            if index + 2 >= window.size:
                raise ValueError("truncated note event")
            pitch, velocity = vocab.piano_fields(head)
            yield (
                "note",
                pitch,
                velocity,
                vocab.onset_ms(int(window[index + 1])),
                vocab.dur_ms(int(window[index + 2])),
            )
            index += 3
        elif head in (vocab.PED_ON_ID, vocab.PED_OFF_ID):
            if index + 1 >= window.size:
                raise ValueError("truncated pedal event")
            kind = "ped_on" if head == vocab.PED_ON_ID else "ped_off"
            yield (kind, None, None, vocab.onset_ms(int(window[index + 1])), None)
            index += 2
        else:
            raise ValueError(f"id {head} at position {index} does not open an event")


def split_5s_to_1s(
    window,
    subwindow_ms: int = vocab.SUBWINDOW_MS,
    parent_ms: int = vocab.WINDOW_MS,
) -> list[np.ndarray]:
    """Split one joint-piano 5 s window into fixed 1 s subwindows.

    Events are assigned by onset bucket; onsets are rebased to the subwindow
    start. Empty subwindows are kept so time stays aligned. Duration tokens are
    unchanged. Always returns ``parent_ms // subwindow_ms`` arrays (5 by default).
    """
    if parent_ms % subwindow_ms != 0:
        raise ValueError(f"parent_ms={parent_ms} not divisible by subwindow_ms={subwindow_ms}")
    n_subs = parent_ms // subwindow_ms
    ids = np.asarray(window, dtype=np.int64)
    buckets: list[list[int]] = [[] for _ in range(n_subs)]
    index = 0
    while index < ids.size:
        head = int(ids[index])
        if vocab.PIANO_LO <= head <= vocab.PIANO_HI:
            if index + 2 >= ids.size:
                raise ValueError("truncated note event")
            onset = vocab.onset_ms(int(ids[index + 1]))
            which = min(onset // subwindow_ms, n_subs - 1)
            local = onset - which * subwindow_ms
            buckets[which].extend([head, vocab.onset_id(local), int(ids[index + 2])])
            index += 3
        elif head in (vocab.PED_ON_ID, vocab.PED_OFF_ID):
            if index + 1 >= ids.size:
                raise ValueError("truncated pedal event")
            onset = vocab.onset_ms(int(ids[index + 1]))
            which = min(onset // subwindow_ms, n_subs - 1)
            local = onset - which * subwindow_ms
            buckets[which].extend([head, vocab.onset_id(local)])
            index += 2
        else:
            raise ValueError(f"id {head} at position {index} does not open an event")
    return [np.asarray(bucket, dtype=np.int64) for bucket in buckets]


def pad_batch(
    windows: list[np.ndarray],
    max_tokens: int,
    add_cls: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Pack windows into ``(tokens, mask)`` for a Transformer encoder."""
    width = max_tokens + 1 if add_cls else max_tokens
    tokens = np.full((len(windows), width), vocab.PAD_ID, dtype=np.int64)
    mask = np.zeros((len(windows), width), dtype=bool)
    for row, window in enumerate(windows):
        window = truncate_to_events(window, max_tokens)
        offset = 0
        if add_cls:
            tokens[row, 0] = vocab.T_ID
            offset = 1
        end = offset + window.size
        tokens[row, offset:end] = window
        mask[row, :end] = True
    return tokens, mask
