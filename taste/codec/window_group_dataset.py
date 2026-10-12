"""Window-group dataset for codec training, adapted from MuseTok's event dataset.

Serves groups of consecutive 1-second windows, split online from the memmapped
5-second store built by ``taste.codec.build_windows``. Each item holds:

- ``enc_input``        ``(max_windows, enc_seqlen)``, one padded 1 s window per row
- ``enc_padding_mask`` ``(max_windows, enc_seqlen)``, True where padded
- ``dec_input`` / ``dec_target``  the group's 1 s windows concatenated, shifted by one
- ``dec_window_idx``   1 s window owning each decoder position

``<T>`` is prepended to every 1 s row as the encoder readout slot. Pitch
augmentation transposes joint piano heads by ``n * VELOCITY_BINS`` ids.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from . import vocab
from .windows import split_5s_to_1s, truncate_to_events

MIN_PITCH, MAX_PITCH = 21, 108
SUBS_PER_PARENT = vocab.WINDOW_MS // vocab.SUBWINDOW_MS  # 5


def transpose_ids(tokens: np.ndarray, n_keys: int) -> np.ndarray:
    """Transpose joint-piano note heads by ``n_keys`` semitones."""
    if n_keys == 0:
        return tokens
    heads = (tokens >= vocab.PIANO_LO) & (tokens <= vocab.PIANO_HI)
    out = tokens.copy()
    out[heads] += n_keys * vocab.VELOCITY_BINS
    return out


def extreme_pitch(tokens: np.ndarray) -> tuple[int, int]:
    """Lowest and highest pitch in ``tokens``; ``(128, 0)`` when there are none."""
    heads = tokens[(tokens >= vocab.PIANO_LO) & (tokens <= vocab.PIANO_HI)]
    if heads.size == 0:
        return 128, 0
    pitches = (heads - vocab.PIANO_LO) // vocab.VELOCITY_BINS
    return int(pitches.min()), int(pitches.max())


class WindowGroupDataset(Dataset):
    """Groups of ``max_windows`` consecutive 1 s windows, split online from 5 s rows."""

    def __init__(
        self,
        store_dir: str | Path,
        pieces: np.ndarray | list[int] | None = None,
        model_enc_seqlen: int = 96,
        model_dec_seqlen: int = 768,
        model_max_windows: int = 20,
        do_augment: bool = True,
        augment_range: range = range(-6, 7),
        min_pitch: int = MIN_PITCH,
        max_pitch: int = MAX_PITCH,
        pad_to_same: bool = True,
        appoint_st_window: int | None = None,
        split_to_1s: bool = True,
        keep_empty_subwindows: bool = True,
    ) -> None:
        self.store_dir = Path(store_dir)
        self.meta = json.loads((self.store_dir / "meta.json").read_text(encoding="utf-8"))
        if self.meta["vocab_format"] != vocab.FORMAT_VERSION:
            raise ValueError(f"{self.store_dir} has vocab_format "
                             f"{self.meta['vocab_format']!r}, expected {vocab.FORMAT_VERSION!r}")

        index = np.load(self.store_dir / "index.npz")
        self.offsets = index["offsets"]
        self.section = index["section"]

        piece_table = np.load(self.store_dir / "pieces.npz", allow_pickle=False)
        self.piece_id = piece_table["piece_id"]
        self.key_id = piece_table["key_id"]
        self.tempo_id = piece_table["tempo_id"]
        self.piece_n_windows = piece_table["n_windows"]  # count of 5 s store rows
        self.piece_start = np.concatenate(([0], np.cumsum(self.piece_n_windows))).astype(np.int64)

        self.tokens = np.memmap(
            self.store_dir / "tokens.u16",
            dtype=np.uint16,
            mode="r",
            shape=(int(self.offsets[-1]),),
        )

        if pieces is None:
            pieces = np.arange(self.piece_n_windows.size)
        self.pieces = np.asarray(pieces, dtype=np.int64)
        self.pieces = self.pieces[self.piece_n_windows[self.pieces] > 0]

        self.model_enc_seqlen = model_enc_seqlen
        self.model_dec_seqlen = model_dec_seqlen
        self.model_max_windows = model_max_windows
        self.split_to_1s = split_to_1s
        self.keep_empty_subwindows = keep_empty_subwindows
        self.subs_per_parent = SUBS_PER_PARENT if split_to_1s else 1

        self.do_augment = do_augment
        self.augment_range = augment_range
        self.min_pitch, self.max_pitch = min_pitch, max_pitch
        self.pad_to_same = pad_to_same
        # appoint_st_window is in *encoder* window units (1 s when splitting)
        self.appoint_st_window = appoint_st_window

        self.pad_token = vocab.PAD_ID
        self.vocab_size = vocab.VOCAB_SIZE

    def __len__(self) -> int:
        return int(self.pieces.size)

    def window_at(self, row: int) -> np.ndarray:
        lo, hi = self.offsets[row], self.offsets[row + 1]
        return np.asarray(self.tokens[lo:hi], dtype=np.int64)

    def n_encoder_windows(self, piece: int) -> int:
        """How many encoder slots a piece yields (1 s slots if splitting)."""
        return int(self.piece_n_windows[piece]) * self.subs_per_parent

    def subwindows_of_parent(self, parent_tokens: np.ndarray) -> list[np.ndarray]:
        if not self.split_to_1s:
            return [parent_tokens]
        subs = split_5s_to_1s(parent_tokens)
        if self.keep_empty_subwindows:
            return subs
        return [sub for sub in subs if sub.size > 0]

    def get_sample(self, item: int):
        """Pick a contiguous run of encoder windows (1 s when ``split_to_1s``)."""
        piece = int(self.pieces[item])
        total = self.n_encoder_windows(piece)
        span = self.model_max_windows

        if total > span and self.appoint_st_window is None:
            start = random.randrange(0, total - span + 1)
        elif self.appoint_st_window is not None and self.appoint_st_window <= total - span:
            start = self.appoint_st_window
        else:
            start = 0

        n_windows = min(span, total)
        base = int(self.piece_start[piece])
        # cache split parents so each 5 s row is parsed once
        parent_cache: dict[int, list[np.ndarray]] = {}
        windows: list[np.ndarray] = []
        for offset in range(n_windows):
            enc_idx = start + offset
            parent_idx, sub_idx = divmod(enc_idx, self.subs_per_parent)
            if parent_idx not in parent_cache:
                parent_cache[parent_idx] = self.subwindows_of_parent(
                    self.window_at(base + parent_idx)
                )
            subs = parent_cache[parent_idx]
            # sub_idx is aligned with the parent's sub-windows only when
            # keep_empty_subwindows=True; a missing sub-window yields an empty window
            if sub_idx >= len(subs):
                windows.append(np.zeros(0, dtype=np.int64))
            else:
                windows.append(subs[sub_idx])

        is_tail = start + n_windows == total
        return piece, windows, start, n_windows, is_tail

    def pitch_augment(self, windows: list[np.ndarray]) -> list[np.ndarray]:
        joined = np.concatenate(windows) if windows else np.zeros(0, dtype=np.int64)
        low, high = extreme_pitch(joined)
        if high == 0:
            return windows

        allowed = [
            keys
            for keys in self.augment_range
            if self.min_pitch <= low + keys and high + keys <= self.max_pitch
        ]
        if not allowed:
            return windows
        n_keys = random.choice(allowed)
        return [transpose_ids(window, n_keys) for window in windows]

    def get_encoder_input_data(self, windows: list[np.ndarray]):
        enc_padding_mask = np.ones((self.model_max_windows, self.model_enc_seqlen), dtype=bool)
        enc_padding_mask[:, :1] = False
        padded_enc_input = np.full(
            (self.model_max_windows, self.model_enc_seqlen), self.pad_token, dtype=np.int64
        )
        padded_enc_input[:, 0] = vocab.T_ID
        enc_lens = np.ones((self.model_max_windows,), dtype=np.int64)

        for row, window in enumerate(windows):
            window = truncate_to_events(window, self.model_enc_seqlen - 1)
            length = window.size + 1
            padded_enc_input[row, 1:length] = window
            enc_padding_mask[row, :length] = False
            enc_lens[row] = length

        return padded_enc_input, enc_padding_mask, enc_lens

    def __getitem__(self, item: int) -> dict:
        piece, windows, start, n_windows, is_tail = self.get_sample(item)

        if self.do_augment:
            windows = self.pitch_augment(windows)

        enc_input, enc_padding_mask, enc_lens = self.get_encoder_input_data(windows)

        stream: list[int] = []
        window_of: list[int] = []
        for index, window in enumerate(windows):
            stream.append(vocab.T_ID)
            window_of.append(index)
            stream.extend(int(token) for token in window)
            window_of.extend([index] * window.size)
        if is_tail:
            stream.append(vocab.E_ID)
            window_of.append(n_windows - 1)

        length = min(len(stream) - 1, self.model_dec_seqlen)
        width = self.model_dec_seqlen + 1 if self.pad_to_same else len(stream)
        padded = np.full(width, self.pad_token, dtype=np.int64)
        owners = np.full(width, self.model_max_windows, dtype=np.int64)
        keep = min(len(stream), width)
        padded[:keep] = stream[:keep]
        owners[:keep] = window_of[:keep]

        dec_input = padded[:-1][: self.model_dec_seqlen]
        dec_target = padded[1:][: self.model_dec_seqlen]
        dec_window_idx = owners[:-1][: self.model_dec_seqlen]

        return {
            "id": item,
            "piece_row": piece,
            "piece_id": str(self.piece_id[piece]),
            "st_window_id": start,
            "enc_input": enc_input,
            "enc_padding_mask": enc_padding_mask,
            "enc_n_windows": n_windows,
            "enc_length": enc_lens,
            "dec_input": dec_input,
            "dec_target": dec_target,
            "dec_window_idx": dec_window_idx,
            "dec_length": length,
            "key_id": int(self.key_id[piece]),
            "tempo_id": int(self.tempo_id[piece]),
        }


def split_pieces(n_pieces: int, val_fraction: float = 0.02, seed: int = 1234, val_cap: int = 3000):
    """Deterministic train/val piece split."""
    order = np.random.default_rng(seed).permutation(n_pieces)
    n_val = min(val_cap, max(1, int(round(n_pieces * val_fraction))))
    return order[n_val:], order[:n_val]


class ConcatWindowDataset(Dataset):
    """Union of several per-stage stores, tagging each item with its stage."""

    def __init__(self, datasets: list[WindowGroupDataset], names: list[str]) -> None:
        if not datasets:
            raise ValueError("need at least one store")
        self.datasets = datasets
        self.names = names
        self.sizes = np.array([len(dataset) for dataset in datasets], dtype=np.int64)
        self.bounds = np.concatenate(([0], np.cumsum(self.sizes)))
        self.stage_of_item = np.repeat(np.arange(len(datasets)), self.sizes)

        first = datasets[0]
        self.vocab_size = first.vocab_size
        self.pad_token = first.pad_token
        self.model_max_windows = first.model_max_windows
        self.model_enc_seqlen = first.model_enc_seqlen
        self.model_dec_seqlen = first.model_dec_seqlen

    def __len__(self) -> int:
        return int(self.bounds[-1])

    @property
    def n_windows(self) -> int:
        """Encoder windows (1 s slots) across every piece in this split."""
        return int(sum(d.n_encoder_windows(int(p)) for d in self.datasets for p in d.pieces))

    def __getitem__(self, item: int) -> dict:
        which = int(np.searchsorted(self.bounds, item, side="right") - 1)
        entry = self.datasets[which][item - int(self.bounds[which])]
        entry["stage"] = which
        entry["stage_name"] = self.names[which]
        return entry


def build_split_datasets(
    store_dirs: list[str | Path],
    val_fraction: float = 0.02,
    split_seed: int = 1234,
    val_cap: int = 3000,
    do_augment: bool = True,
    **dataset_kwargs,
) -> tuple[ConcatWindowDataset, ConcatWindowDataset]:
    """Train/val datasets over one or more stores, split per store by piece."""
    train_sets, val_sets, names = [], [], []
    for store in store_dirs:
        store = Path(store)
        n_pieces = np.load(store / "pieces.npz")["n_windows"].size
        train_pieces, val_pieces = split_pieces(
            n_pieces,
            val_fraction=val_fraction,
            seed=split_seed,
            val_cap=max(1, val_cap // len(store_dirs)),
        )
        train_sets.append(
            WindowGroupDataset(store, pieces=train_pieces, do_augment=do_augment, **dataset_kwargs)
        )
        val_sets.append(
            WindowGroupDataset(
                store, pieces=val_pieces, do_augment=False, appoint_st_window=0, **dataset_kwargs
            )
        )
        names.append(store.name)
    return ConcatWindowDataset(train_sets, names), ConcatWindowDataset(val_sets, names)


def stage_sample_weights(dataset: ConcatWindowDataset, weights: list[float]) -> np.ndarray:
    """Per-item sampling weights that realise the requested stage mixture."""
    if len(weights) != len(dataset.datasets):
        raise ValueError(
            f"got {len(weights)} stage weights for {len(dataset.datasets)} stores"
        )
    share = np.asarray(weights, dtype=np.float64)
    share = share / share.sum()
    per_item = share / np.maximum(dataset.sizes, 1)
    return per_item[dataset.stage_of_item]
