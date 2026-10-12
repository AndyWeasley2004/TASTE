"""Latent to MIDI: grammar-constrained free-running decode of the window VAE.

The codec (``taste.codec``) maps every second of music to one 128-d latent. Its
decoder is causal and reconstructs the concatenated stream of up to
``MAX_WINDOWS`` seconds (``<T>`` + the second's events, per-second latent added
as a segment embedding before every layer). Unconstrained greedy decoding in
``eval()`` mode collapses, so decoding is grammar constrained: the logits are
masked to

    head  -> piano head | pedal head | <T>
    piano head -> onset
    pedal head -> onset
    onset (after a piano head) -> duration

with onsets non-decreasing inside a second. ``<E>`` is masked out entirely: the
piece length is known, so a chunk ends when its ``(n + 1)``-th ``<T>`` would be
emitted.

Chunks of a whole cohort are decoded in lockstep with per-row grammar state,
and :class:`CachedDecoder` keeps per-layer K/V caches so each step costs one
token (``use_cache=False`` runs the codec's own full-prefix decoder).

``rebuild_row`` turns the decoded seconds back into a corpus-format row
(``<KEY> <TEMPO> plan <prefix> <S> ... <E>``) and ``write_midi`` detokenizes it
with the shared tokenizer.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import torch.nn.functional as F

from taste import paths
from taste.codec import vocab
from taste.codec.infer_reconstruct import load_run
from taste.representation.tokenizer import get_tokenizer

CODEC_DIR = paths.ckpt("codec", "window_vae")

MAX_WINDOWS = 20          # seconds per decoder chunk
ENC_SEQLEN = 96
DEC_SEQLEN = 1024         # token cap per chunk
SUBWINDOWS = vocab.WINDOW_MS // vocab.SUBWINDOW_MS          # 5 seconds per window
ONSET_STRIDE = vocab.SUBWINDOW_MS // vocab.TIME_STEP_MS     # 40 onset ids per second

# grammar states; MASK_OF_STATE indexes the mask table below
HEAD, ONSET_NOTE, ONSET_PED, DUR = 0, 1, 2, 3
MASK_OF_STATE = (0, 1, 1, 2)
NEXT_STATE = {HEAD: HEAD, ONSET_NOTE: DUR, ONSET_PED: HEAD, DUR: HEAD}


def load_codec(device: str = "cuda"):
    """Verify the vocabulary, then load the window VAE in eval mode."""
    vocab.verify()
    config, model, ckpt_path = load_run(CODEC_DIR, device)
    data = config["data"]
    if (data["max_windows"], data["enc_seqlen"], data["dec_seqlen"]) != (
        MAX_WINDOWS, ENC_SEQLEN, DEC_SEQLEN
    ):
        raise ValueError(f"{ckpt_path} was trained with a different window geometry")
    return config, model


# ------------------------------------------------------------------- grammar

def _mask_table(device) -> torch.Tensor:
    """``(3, V)`` boolean masks for the head / onset / duration states."""
    table = torch.zeros(3, vocab.VOCAB_SIZE, dtype=torch.bool)
    table[0, vocab.PIANO_LO:vocab.PIANO_HI + 1] = True
    table[0, vocab.PED_ON_ID] = True
    table[0, vocab.PED_OFF_ID] = True
    table[0, vocab.T_ID] = True
    table[1, vocab.ONSET_LO:vocab.ONSET_HI + 1] = True
    table[2, vocab.DUR_LO:vocab.DUR_HI + 1] = True
    return table.to(device)


def _advance(state: int, token: int) -> int:
    if state != HEAD:
        return NEXT_STATE[state]
    if vocab.PIANO_LO <= token <= vocab.PIANO_HI:
        return ONSET_NOTE
    if token in (vocab.PED_ON_ID, vocab.PED_OFF_ID):
        return ONSET_PED
    return HEAD  # <T>


# ------------------------------------------------------------- cached decoder

class CachedDecoder:
    """Incremental (KV-cached) form of ``WindowSegmentCodec.generate``.

    The codec's ``TransformerDecoder`` re-runs the whole prefix every step. Here
    each of its ``nn.TransformerEncoderLayer`` blocks is evaluated by hand from
    its own parameters (post-LN, relu, dropout off) for the single new position,
    with per-layer K/V caches standing in for the prefix. Causality is implicit:
    a step only ever attends to keys already in the cache.

    The cache holds exactly the prefix the uncached path would feed, padding
    included, so rows that have finished keep stepping harmlessly and the two
    paths agree row by row.
    """

    def __init__(self, model, batch: int, device: str) -> None:
        self.model = model
        self.layers = model.decoder.decoder_layers
        self.n_head = model.dec_n_head
        self.d_model = model.dec_d_model
        self.head_dim = self.d_model // self.n_head
        self.batch = batch
        self.device = device
        self.dtype = model.dec_out_proj.weight.dtype
        self.length = 0
        self.capacity = 0
        self.cache_k: list[torch.Tensor] = []
        self.cache_v: list[torch.Tensor] = []

    def _reserve(self, needed: int) -> None:
        if needed <= self.capacity:
            return
        capacity = max(128, self.capacity * 2)
        while capacity < needed:
            capacity *= 2
        shape = (self.batch, self.n_head, capacity, self.head_dim)
        grown_k, grown_v = [], []
        for layer in range(len(self.layers)):
            new_k = torch.empty(shape, device=self.device, dtype=self.dtype)
            new_v = torch.empty(shape, device=self.device, dtype=self.dtype)
            if self.length:
                new_k[:, :, :self.length] = self.cache_k[layer][:, :, :self.length]
                new_v[:, :, :self.length] = self.cache_v[layer][:, :, :self.length]
            grown_k.append(new_k)
            grown_v.append(new_v)
        self.cache_k, self.cache_v = grown_k, grown_v
        self.capacity = capacity

    def step(self, token: torch.Tensor, seg_raw: torch.Tensor) -> torch.Tensor:
        """One token per row; returns ``(batch, V)`` logits for the next token.

        ``token`` is ``(batch,)`` and ``seg_raw`` the ``(batch, d_latent)``
        unprojected segment embedding owned by this position.
        """
        model = self.model
        position = self.length
        self._reserve(position + 1)

        out = model.token_emb(token) + model.pe.pe[position, 0]
        seg = model.decoder.seg_emb_proj(seg_raw)

        for index, layer in enumerate(self.layers):
            hidden = out + seg
            attn = layer.self_attn
            qkv = F.linear(hidden, attn.in_proj_weight, attn.in_proj_bias)
            query, key, value = qkv.chunk(3, dim=-1)
            shape = (self.batch, self.n_head, 1, self.head_dim)
            self.cache_k[index][:, :, position] = key.view(
                self.batch, self.n_head, self.head_dim)
            self.cache_v[index][:, :, position] = value.view(
                self.batch, self.n_head, self.head_dim)
            context = F.scaled_dot_product_attention(
                query.view(shape),
                self.cache_k[index][:, :, :position + 1],
                self.cache_v[index][:, :, :position + 1],
            )
            hidden = layer.norm1(
                hidden + attn.out_proj(context.reshape(self.batch, self.d_model))
            )
            out = layer.norm2(hidden + layer.linear2(F.relu(layer.linear1(hidden))))

        self.length = position + 1
        return model.dec_out_proj(out)


# -------------------------------------------------------------------- decode

@torch.no_grad()
def _decode_chunk_batch(model, latents: np.ndarray, n_seconds: list[int],
                        device: str, use_cache: bool = True) -> list[np.ndarray]:
    """Decode one batch of chunks; ``latents`` is ``(B, MAX_WINDOWS, d)``."""
    batch = len(n_seconds)
    latent = torch.from_numpy(latents).to(device)
    tokens = torch.full((batch, DEC_SEQLEN), vocab.PAD_ID, dtype=torch.long, device=device)
    owners = torch.full((batch, DEC_SEQLEN), MAX_WINDOWS, dtype=torch.long, device=device)
    tokens[:, 0] = vocab.T_ID
    owners[:, 0] = 0

    masks = _mask_table(device)
    ids = torch.arange(vocab.VOCAB_SIZE, device=device)

    state = [HEAD] * batch
    last_onset = [vocab.ONSET_LO] * batch
    n_t = [1] * batch
    length = [1] * batch
    alive = [True] * batch

    cache = CachedDecoder(model, batch, device) if use_cache else None
    rows = torch.arange(batch, device=device)
    n_windows = latent.size(1)

    for position in range(1, DEC_SEQLEN):
        if cache is not None:
            owner = owners[:, position - 1]
            seg_raw = latent[rows, owner.clamp(max=n_windows - 1)]
            seg_raw = seg_raw * (owner < n_windows).unsqueeze(-1)
            logits = cache.step(tokens[:, position - 1], seg_raw)
        else:
            seg = model.segment_embedding(latent, owners[:, :position], position)
            logits = model.generate(tokens[:, :position].t(), seg)

        state_t = torch.tensor(state, device=device)
        allowed = masks[torch.tensor(MASK_OF_STATE, device=device)[state_t]].clone()
        is_onset = (state_t == ONSET_NOTE) | (state_t == ONSET_PED)
        stale = (ids >= vocab.ONSET_LO) & (
            ids < torch.tensor(last_onset, device=device).unsqueeze(1)
        )
        allowed &= ~(stale & is_onset.unsqueeze(1))
        nxt = logits.masked_fill(~allowed, float("-inf")).argmax(dim=-1).tolist()

        step = torch.full((batch,), vocab.PAD_ID, dtype=torch.long, device=device)
        step_owner = torch.full((batch,), MAX_WINDOWS, dtype=torch.long, device=device)
        for row in range(batch):
            if not alive[row]:
                continue
            token = nxt[row]
            if token == vocab.T_ID:
                if n_t[row] == n_seconds[row]:
                    alive[row] = False
                    continue
                n_t[row] += 1
                last_onset[row] = vocab.ONSET_LO
            elif vocab.ONSET_LO <= token <= vocab.ONSET_HI:
                last_onset[row] = token
            state[row] = _advance(state[row], token)
            step[row] = token
            step_owner[row] = min(n_t[row] - 1, MAX_WINDOWS - 1)
            length[row] = position + 1
        if not any(alive):
            break
        tokens[:, position] = step
        owners[:, position] = step_owner

    out = tokens.cpu().numpy()
    return [out[row, :length[row]].copy() for row in range(batch)]


def _stream_to_seconds(stream: np.ndarray, n_seconds: int) -> list[np.ndarray]:
    """Split a decoded chunk on ``<T>``; pad to ``n_seconds`` if it hit the cap."""
    marks = np.flatnonzero(stream == vocab.T_ID)
    seconds = []
    for index, lo in enumerate(marks):
        hi = marks[index + 1] if index + 1 < marks.size else stream.size
        window = stream[lo + 1:hi]
        # a chunk cut at DEC_SEQLEN can end mid-event
        while window.size and not _well_formed(window):
            window = window[:-1]
        seconds.append(window)
    seconds += [np.zeros(0, dtype=np.int64)] * (n_seconds - len(seconds))
    return seconds[:n_seconds]


def _well_formed(window: np.ndarray) -> bool:
    index = 0
    while index < window.size:
        head = int(window[index])
        if not vocab.is_event_head(head):
            return False
        index += vocab.event_length(head)
    return index == window.size


@torch.no_grad()
def decode_pieces(model, mus: list[np.ndarray], device: str = "cuda",
                  batch: int = 32,
                  use_cache: bool = True) -> tuple[list[list[np.ndarray]], dict]:
    """Free-run the decoder over whole pieces.

    ``mus[i]`` is ``(n_seconds_i, 128)``; the result is one list of per-second
    token arrays per piece (onsets local to the second), plus decode stats.
    """
    jobs = []
    for piece, mu in enumerate(mus):
        for start in range(0, len(mu), MAX_WINDOWS):
            jobs.append((piece, start, mu[start:start + MAX_WINDOWS]))

    seconds: list[list[np.ndarray]] = [[] for _ in mus]
    n_capped = 0
    for offset in range(0, len(jobs), batch):
        group = jobs[offset:offset + batch]
        latents = np.zeros((len(group), MAX_WINDOWS, group[0][2].shape[1]), dtype=np.float32)
        for row, (_, _, mu) in enumerate(group):
            latents[row, :len(mu)] = mu
        counts = [len(mu) for _, _, mu in group]
        streams = _decode_chunk_batch(model, latents, counts, device, use_cache)
        for (piece, _, _), stream, count in zip(group, streams, counts):
            if stream.size >= DEC_SEQLEN:
                n_capped += 1
            seconds[piece].extend(_stream_to_seconds(stream, count))
    return seconds, {"n_chunks": len(jobs), "n_chunks_token_capped": n_capped}


# ------------------------------------------------------------------- rebuild

def _shift_onsets(window: np.ndarray, sub_index: int) -> list[int]:
    """Rebase a 1 s window's onsets into its slot of the parent 5 s window."""
    shift = sub_index * ONSET_STRIDE
    out = [int(token) for token in window]
    index = 0
    while index < len(out):
        head = out[index]
        out[index + 1] += shift
        index += vocab.event_length(head)
    return out


def plan_tokens(sections: list[int]) -> list[int]:
    """Run-length encode per-window section ids (1..8) into header tokens."""
    header: list[int] = []
    run_section, run_length = sections[0], 0
    for section in list(sections) + [None]:
        if section == run_section:
            run_length += 1
            continue
        header.append(vocab.SEC_LO + run_section - 1)
        header += [vocab.PLAN_LO + int(digit) for digit in str(run_length)]
        run_section, run_length = section, 1
    return header


def rebuild_row(seconds: list[np.ndarray], key_id: int, tempo_id: int,
                sections: list[int] | None, complete: bool,
                start_id: int = vocab.S_ID) -> list[int]:
    """Per-second windows → one corpus-format row.

    ``sections`` holds one id in 1..8 per 5 s window, or is ``None`` when the
    piece carries no section plan (then no plan header and no in-stream
    ``<SEC_*>`` tokens are written).
    """
    if len(seconds) % SUBWINDOWS:
        raise ValueError(f"{len(seconds)} seconds is not a multiple of {SUBWINDOWS}")
    n_windows = len(seconds) // SUBWINDOWS
    if sections is not None and len(sections) != n_windows:
        raise ValueError(f"{len(sections)} sections for {n_windows} windows")

    row = [key_id, tempo_id]
    if sections is not None:
        row += plan_tokens(list(sections))
    row += [vocab.PREFIX_PIANO_ID, start_id]

    for window in range(n_windows):
        if window:
            row.append(vocab.T_ID)
        if sections is not None:
            row.append(vocab.SEC_LO + int(sections[window]) - 1)
        for sub in range(SUBWINDOWS):
            row += _shift_onsets(seconds[window * SUBWINDOWS + sub], sub)
    if complete:
        row.append(vocab.E_ID)
    return row


def write_midi(row: list[int], path: str | Path) -> None:
    """Detokenize a corpus-format row (from the piano prefix on) and save it."""
    midi = get_tokenizer()
    stream = row[row.index(vocab.PREFIX_PIANO_ID):]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    midi.detokenize_midi(stream).to_midi().save(str(path))
