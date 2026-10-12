"""Latent-store dataset, length-bucketed sampler and collation for the latent DiT.

A *store* is one ``(stage, pass)`` directory written by the latent builder:

``mu.f16``      flat float16 array of shape ``(n_latents_total, 128)``
``index.parquet`` one row per training row (see ``INDEX_COLUMNS``)
``meta.json``   store metadata; ``latents_per_second`` sets how many latents one
                second of music holds

``LatentStoreDataset`` concatenates several stores and yields standardised fp32
latents plus every conditioning field. The imagery states stay on disk and are
read one row at a time (see ``_ContextReader``).

``collate`` pads to ``max_len`` latents, builds the validity mask, and applies
classifier-free condition dropout in three groups: attributes + imagery,
{key, tempo, complete, start_mid}, and the section plan.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, Sampler

from taste.models.ar_conditional import ATTRIBUTE_FIELDS, load_attribute_classes
from taste.models.dit import CONTEXT_DIM, D_LATENT, N_KEYS, N_SECTIONS, N_TEMPOS

SECTION_NULL = N_SECTIONS
KEY_NULL = N_KEYS
TEMPO_NULL = N_TEMPOS
COMPLETE_NULL = 2
START_MID_NULL = 2

SECONDS_PER_WINDOW = 5

INDEX_COLUMNS = (
    "piece_id", "n_seconds", "n_latents", "offset", "sections", "key_id", "tempo_id",
    "complete", "start_mid", "prefix_dropped", "context_shard", "context_length",
    *ATTRIBUTE_FIELDS,
)

# imagery-state readers are opened lazily and kept per dataloader worker process
_STATE_HANDLES: dict[str, "_ContextReader"] = {}


class _ContextReader:
    """One imagery-state safetensors shard, read row by row with ``pread``.

    The header (8-byte little-endian length, then that many bytes of JSON) is
    parsed once; each row is then one ``pread`` of exactly its bytes, which avoids
    the page-by-page faults and readahead of a memmap slice. Descriptors are
    inherited by forked dataloader workers, and ``pread`` carries its own offset,
    so sharing one is safe.
    """

    def __init__(self, path: str):
        self.path = path
        self.fd = os.open(path, os.O_RDONLY)
        os.posix_fadvise(self.fd, 0, 0, os.POSIX_FADV_RANDOM)
        header_len = int.from_bytes(os.pread(self.fd, 8, 0), "little")
        header = json.loads(os.pread(self.fd, header_len, 8))
        self.base = 8 + header_len
        self.entries = {}
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            if entry["dtype"] != "BF16":
                raise ValueError(f"{path}: {name} has dtype {entry['dtype']}, expected BF16")
            self.entries[name] = (entry["shape"], entry["data_offsets"][0])

    def read(self, name: str, length: int) -> torch.Tensor:
        shape, start = self.entries[name]
        n_bytes = length * shape[1] * 2
        raw = os.pread(self.fd, n_bytes, self.base + start)
        if len(raw) != n_bytes:
            raise IOError(f"{self.path}: short read of {name} ({len(raw)} of {n_bytes} bytes)")
        # bytearray so the tensor owns writable memory (torch warns otherwise)
        return torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).view(length, shape[1])


def context_reader(path: str) -> _ContextReader:
    reader = _STATE_HANDLES.get(path)
    if reader is None:
        reader = _ContextReader(path)
        _STATE_HANDLES[path] = reader
    return reader


@dataclass
class DropoutProbs:
    """Per-group classifier-free dropout, applied on top of the natural nulls."""

    attributes: float = 0.1  # attributes + imagery together
    globals: float = 0.1  # key + tempo + complete + start_mid together
    plan: float = 0.1  # sections + within-section index


class _Store:
    """One ``(stage, pass)`` latent directory, decoded into flat arrays."""

    def __init__(self, store_dir: str, corpus_dir: str, classes: dict[str, list[str]]):
        self.store_dir = Path(store_dir)
        self.context_dir = Path(corpus_dir) / "qwen3_4b_imagery_states"
        meta = json.loads((self.store_dir / "meta.json").read_text())
        self.latents_per_second = int(meta["latents_per_second"])
        columns = pq.read_table(self.store_dir / "index.parquet", columns=list(INDEX_COLUMNS)).to_pydict()

        self.n_seconds = np.asarray(columns["n_seconds"], dtype=np.int64)
        self.n_latents = np.asarray(columns["n_latents"], dtype=np.int64)
        self.offset = np.asarray(columns["offset"], dtype=np.int64)
        self.piece_id = columns["piece_id"]

        total = self.offset[-1] + self.n_latents[-1] if len(self.offset) else 0
        path = self.store_dir / "mu.f16"
        n_stored = path.stat().st_size // (D_LATENT * 2)
        if n_stored != total:
            raise ValueError(
                f"{path}: holds {n_stored} latents, index accounts for {total}"
            )
        # One pread per row: a memmap slice would fault each 4 KB
        # page separately and every miss pulls the device readahead window.
        # POSIX_FADV_RANDOM turns readahead off for this descriptor.
        self.mu_fd = os.open(path, os.O_RDONLY)
        os.posix_fadvise(self.mu_fd, 0, 0, os.POSIX_FADV_RANDOM)

        sections = columns["sections"]
        lengths = np.fromiter((len(entry) for entry in sections), dtype=np.int64, count=len(sections))
        per_window = SECONDS_PER_WINDOW * self.latents_per_second
        if (lengths * per_window != self.n_latents).any():
            raise ValueError(
                f"{self.store_dir}: sections length does not match n_latents/{per_window}")
        self.section_offset = np.zeros(len(sections) + 1, dtype=np.int64)
        np.cumsum(lengths, out=self.section_offset[1:])
        self.sections = np.concatenate(
            [np.asarray(entry, dtype=np.int8) for entry in sections]
        ) if len(sections) else np.zeros(0, dtype=np.int8)

        dropped = np.asarray(columns["prefix_dropped"], dtype=bool)
        self.attrs = np.empty((len(dropped), len(ATTRIBUTE_FIELDS)), dtype=np.int16)
        for index, name in enumerate(ATTRIBUTE_FIELDS):
            lookup = {value: position for position, value in enumerate(classes[name])}
            null = len(classes[name])
            self.attrs[:, index] = [
                null if (not value or is_dropped) else lookup[value]
                for value, is_dropped in zip(columns[name], dropped)
            ]
        key = np.asarray(columns["key_id"], dtype=np.int16)
        tempo = np.asarray(columns["tempo_id"], dtype=np.int16)
        self.key = np.where(key < 0, KEY_NULL, key)
        self.tempo = np.where(tempo < 0, TEMPO_NULL, tempo)
        self.complete = np.asarray(columns["complete"], dtype=np.int16)
        self.start_mid = np.asarray(columns["start_mid"], dtype=np.int16)

        shard_files = sorted({name for name in columns["context_shard"] if name})
        self.shard_paths = [str(self.context_dir / name) for name in shard_files]
        shard_id = {name: position for position, name in enumerate(shard_files)}
        self.context_shard = np.asarray(
            [shard_id[name] if name else -1 for name in columns["context_shard"]],
            dtype=np.int32,
        )
        length = np.asarray(columns["context_length"], dtype=np.int64)
        # a dropped prefix or an absent shard keeps only the model's null token
        self.context_length = np.where(dropped | (self.context_shard < 0), 0, length)

    def __len__(self) -> int:
        return len(self.n_seconds)


def section_index(sections_per_latent: np.ndarray) -> np.ndarray:
    """Latents elapsed since the current section run started; 0 on null."""
    changed = np.empty(len(sections_per_latent), dtype=bool)
    changed[0] = True
    changed[1:] = sections_per_latent[1:] != sections_per_latent[:-1]
    run_start = np.maximum.accumulate(np.where(changed, np.arange(len(changed)), 0))
    index = np.arange(len(changed)) - run_start
    return np.where(sections_per_latent == SECTION_NULL, 0, index)


class LatentStoreDataset(Dataset):
    def __init__(self, stage_dirs: list, stats: tuple, attributes_path=None):
        classes = (
            load_attribute_classes(attributes_path)
            if attributes_path
            else load_attribute_classes()
        )
        self.stores = [_Store(store, corpus, classes) for store, corpus in stage_dirs]
        self.mean, self.std = stats
        sizes = np.fromiter((len(store) for store in self.stores), dtype=np.int64,
                            count=len(self.stores))
        self.bounds = np.zeros(len(self.stores) + 1, dtype=np.int64)
        np.cumsum(sizes, out=self.bounds[1:])

    def __len__(self) -> int:
        return int(self.bounds[-1])

    def row_lengths(self) -> np.ndarray:
        """Latents per row, in dataset index order, for the length-bucketed sampler."""
        return np.concatenate([store.n_latents for store in self.stores])

    def __getitem__(self, index: int) -> dict:
        store_index = int(np.searchsorted(self.bounds, index, side="right") - 1)
        store = self.stores[store_index]
        row = index - int(self.bounds[store_index])

        offset, n_latents = int(store.offset[row]), int(store.n_latents[row])
        row_bytes = n_latents * D_LATENT * 2
        raw = os.pread(store.mu_fd, row_bytes, offset * D_LATENT * 2)
        if len(raw) != row_bytes:
            raise IOError(f"{store.store_dir}: short read at latent {offset} ({len(raw)} of {row_bytes} bytes)")
        mu = np.frombuffer(raw, dtype=np.float16).reshape(n_latents, D_LATENT).astype(np.float32)
        latents = (mu - self.mean) / self.std

        windows = store.sections[store.section_offset[row] : store.section_offset[row + 1]]
        sections = np.repeat(
            np.where(windows < 0, SECTION_NULL, windows).astype(np.int64),
            SECONDS_PER_WINDOW * store.latents_per_second,
        )

        length = int(store.context_length[row])
        context = None
        if length:
            reader = context_reader(store.shard_paths[int(store.context_shard[row])])
            context = reader.read(store.piece_id[row], length)

        return {
            "latents": torch.from_numpy(latents),
            "n_latents": n_latents,
            "sections": torch.from_numpy(sections),
            "sec_index": torch.from_numpy(section_index(sections)),
            "attrs": torch.from_numpy(store.attrs[row].astype(np.int64)),
            "key": int(store.key[row]),
            "tempo": int(store.tempo[row]),
            "complete": int(store.complete[row]),
            "start_mid": int(store.start_mid[row]),
            "context": context,
        }


class LengthBucketedBatchSampler(Sampler):
    """Per-rank batches of similar-length rows, which keeps padding small.

    Rows are reshuffled every epoch, then sorted inside pools so that one batch is
    length-homogeneous. Each rank-group holds ``num_replicas`` adjacent batches, so
    the ranks get near-equal work and neither waits on the other at the gradient
    all-reduce. The order of those groups is then shuffled, so short and long
    batches stay mixed across training and the model sees short and long pieces
    throughout.

    All ranks draw from one generator seeded the same way, so they agree on the
    grouping and the order and only differ in which slice of each group they take.
    """

    def __init__(self, lengths, batch_size: int, num_replicas: int, rank: int,
                 seed: int, pool_batches: int = 50):
        self.lengths = np.asarray(lengths)
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.pool_batches = pool_batches
        self.epoch = 0
        self.group = batch_size * num_replicas
        self.n_groups = len(self.lengths) // self.group
        if self.n_groups == 0:
            raise ValueError(
                f"{len(self.lengths)} rows cannot fill one group of {self.group}"
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.n_groups

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        order = rng.permutation(len(self.lengths))[: self.n_groups * self.group]
        pool = self.pool_batches * self.group
        order = np.concatenate([
            block[np.argsort(self.lengths[block], kind="stable")]
            for block in (order[i : i + pool] for i in range(0, len(order), pool))
        ])
        groups = order.reshape(self.n_groups, self.num_replicas, self.batch_size)
        for group in rng.permutation(self.n_groups):
            yield groups[group, self.rank].tolist()


def collate(features, *, max_len: int, dropout: DropoutProbs, attribute_nulls: torch.Tensor):
    batch = len(features)
    longest = max(feature["n_latents"] for feature in features)
    if longest > max_len:
        raise ValueError(f"row of {longest} latents exceeds max_len {max_len}")
    # pad to the batch's longest row, rounded up to a multiple of 8; padded
    # positions are masked out of both attention and the loss, so this is exact
    width = min(int(np.ceil(longest / 8) * 8), max_len)
    latents = torch.zeros(batch, width, D_LATENT)
    valid_mask = torch.zeros(batch, width, dtype=torch.bool)
    sections = torch.full((batch, width), SECTION_NULL, dtype=torch.long)
    sec_index = torch.zeros(batch, width, dtype=torch.long)
    for row, feature in enumerate(features):
        length = feature["n_latents"]
        latents[row, :length] = feature["latents"]
        valid_mask[row, :length] = True
        sections[row, :length] = feature["sections"]
        sec_index[row, :length] = feature["sec_index"]

    attrs = torch.stack([feature["attrs"] for feature in features])
    key = torch.tensor([feature["key"] for feature in features], dtype=torch.long)
    tempo = torch.tensor([feature["tempo"] for feature in features], dtype=torch.long)
    complete = torch.tensor([feature["complete"] for feature in features], dtype=torch.long)
    start_mid = torch.tensor([feature["start_mid"] for feature in features], dtype=torch.long)

    context_lengths = torch.tensor(
        [0 if feature["context"] is None else feature["context"].shape[0]
         for feature in features],
        dtype=torch.long,
    )
    # attributes and imagery are dropped as one group, on top of the corpus's
    # own prefix_dropped rows
    drop_attributes = torch.rand(batch) < dropout.attributes
    drop_globals = torch.rand(batch) < dropout.globals
    drop_plan = torch.rand(batch) < dropout.plan

    attrs = torch.where(drop_attributes[:, None], attribute_nulls[None, :], attrs)
    context_lengths = torch.where(drop_attributes, 0, context_lengths)
    key = torch.where(drop_globals, KEY_NULL, key)
    tempo = torch.where(drop_globals, TEMPO_NULL, tempo)
    complete = torch.where(drop_globals, COMPLETE_NULL, complete)
    start_mid = torch.where(drop_globals, START_MID_NULL, start_mid)
    sections = torch.where(drop_plan[:, None], SECTION_NULL, sections)
    sec_index = torch.where(drop_plan[:, None], 0, sec_index)

    context_max = int(context_lengths.max())
    context = torch.zeros(batch, context_max, CONTEXT_DIM, dtype=torch.bfloat16)
    positions = torch.arange(context_max)
    for row, feature in enumerate(features):
        length = int(context_lengths[row])
        if length:
            context[row, :length] = feature["context"].to(torch.bfloat16)
    context_mask = positions[None, :] < context_lengths[:, None]

    return {
        "latents": latents,
        "valid_mask": valid_mask,
        "sections": sections,
        "sec_index": sec_index,
        "attrs": attrs,
        "key": key,
        "tempo": tempo,
        "complete": complete,
        "start_mid": start_mid,
        "context": context,
        "context_mask": context_mask,
    }


def attribute_null_ids(attributes_path=None) -> torch.Tensor:
    classes = (
        load_attribute_classes(attributes_path) if attributes_path else load_attribute_classes()
    )
    return torch.tensor([len(classes[name]) for name in ATTRIBUTE_FIELDS], dtype=torch.long)
