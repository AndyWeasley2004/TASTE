"""Latent DiT with global labels, section-plan embeddings and imagery attention.

Missing/dropped labels use null embeddings. Imagery always includes a learned
null token. Bare AdaLN gates disable branches at zero; the zero-initialized
final projection makes the initial output zero.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from taste.models.ar_conditional import ATTRIBUTE_FIELDS, load_attribute_classes

REPO = Path(__file__).resolve().parents[2]

N_KEYS = 24
N_TEMPOS = 3
N_SECTIONS = 8
D_LATENT = 128
CONTEXT_DIM = 2560
TIME_FREQ_DIM = 256
EMBED_INIT_STD = 0.02


@dataclass
class DiTConfig:
    d_model: int = 1024
    n_layers: int = 16
    n_heads: int = 16
    mlp_ratio: int = 4
    cond_dim: int = 512
    d_latent: int = D_LATENT
    max_len: int = 384
    context_dim: int = CONTEXT_DIM
    n_sections: int = N_SECTIONS
    n_keys: int = N_KEYS
    n_tempos: int = N_TEMPOS
    attribute_classes: dict[str, list[str]] = field(
        default_factory=lambda: load_attribute_classes(REPO / "attributes.json")
    )


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of ``t`` (shape ``[B]``, in [0, 1]) scaled to the usual 1000 range."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10_000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    angles = (1000.0 * t.float())[..., None] * freqs
    return torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)


class ConditionEmbedder(nn.Module):
    """Timestep + label embeddings -> the AdaLN conditioning code."""

    def __init__(self, cfg: DiTConfig):
        super().__init__()
        self.cond_dim = cfg.cond_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(TIME_FREQ_DIM, cfg.cond_dim),
            nn.SiLU(),
            nn.Linear(cfg.cond_dim, cfg.cond_dim),
        )
        # each table carries one trailing null row (unknown label / CFG dropout)
        sizes = [len(cfg.attribute_classes[name]) + 1 for name in ATTRIBUTE_FIELDS]
        self.attribute_embeddings = nn.ModuleList(
            nn.Embedding(size, cfg.cond_dim) for size in sizes
        )
        self.key_embedding = nn.Embedding(cfg.n_keys + 1, cfg.cond_dim)
        self.tempo_embedding = nn.Embedding(cfg.n_tempos + 1, cfg.cond_dim)
        self.complete_embedding = nn.Embedding(3, cfg.cond_dim)
        self.start_mid_embedding = nn.Embedding(3, cfg.cond_dim)
        for embedding in [
            *self.attribute_embeddings,
            self.key_embedding,
            self.tempo_embedding,
            self.complete_embedding,
            self.start_mid_embedding,
        ]:
            nn.init.normal_(embedding.weight, mean=0.0, std=EMBED_INIT_STD)
        self.label_mlp = nn.Sequential(
            nn.Linear(cfg.cond_dim, cfg.cond_dim),
            nn.SiLU(),
            nn.Linear(cfg.cond_dim, cfg.cond_dim),
        )

    def forward(self, t, attrs, key, tempo, complete, start_mid):
        labels = sum(
            embedding(attrs[:, index])
            for index, embedding in enumerate(self.attribute_embeddings)
        )
        labels = (
            labels
            + self.key_embedding(key)
            + self.tempo_embedding(tempo)
            + self.complete_embedding(complete)
            + self.start_mid_embedding(start_mid)
        )
        time_code = self.time_mlp(timestep_embedding(t, TIME_FREQ_DIM))
        return time_code + self.label_mlp(labels)


class SelfAttention(nn.Module):
    def __init__(self, cfg: DiTConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.n_heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4)
        query, key = self.q_norm(query), self.k_norm(key)
        attended = F.scaled_dot_product_attention(
            query, key, value, attn_mask=valid_mask[:, None, None, :]
        )
        return self.out(attended.transpose(1, 2).reshape(batch, length, -1))


class CrossAttention(nn.Module):
    def __init__(self, cfg: DiTConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.context_dim, cfg.d_model, bias=False)
        self.v_proj = nn.Linear(cfg.context_dim, cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)

    def forward(self, x, context, context_mask):
        batch, length, _ = x.shape
        context_length = context.shape[1]

        def split(tensor, seq):
            return tensor.view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)

        query = self.q_norm(split(self.q_proj(x), length))
        key = self.k_norm(split(self.k_proj(context), context_length))
        value = split(self.v_proj(context), context_length)
        attended = F.scaled_dot_product_attention(
            query, key, value, attn_mask=context_mask[:, None, None, :]
        )
        return self.out(attended.transpose(1, 2).reshape(batch, length, -1))


class DiTBlock(nn.Module):
    """Pre-norm AdaLN-Zero block: self-attention, cross-attention, MLP."""

    def __init__(self, cfg: DiTConfig):
        super().__init__()
        d = cfg.d_model
        self.norm_self = nn.LayerNorm(d, elementwise_affine=False)
        self.norm_cross = nn.LayerNorm(d, elementwise_affine=False)
        self.norm_mlp = nn.LayerNorm(d, elementwise_affine=False)
        self.self_attn = SelfAttention(cfg)
        self.cross_attn = CrossAttention(cfg)
        self.mlp = nn.Sequential(
            nn.Linear(d, cfg.mlp_ratio * d),
            nn.GELU(),
            nn.Linear(cfg.mlp_ratio * d, d),
        )
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(cfg.cond_dim, 9 * d))
        nn.init.zeros_(self.adaln[1].weight)
        nn.init.zeros_(self.adaln[1].bias)

    def forward(self, x, c, valid_mask, context, context_mask):
        (
            shift_self, scale_self, gate_self,
            shift_cross, scale_cross, gate_cross,
            shift_mlp, scale_mlp, gate_mlp,
        ) = self.adaln(c).unsqueeze(1).chunk(9, dim=-1)

        normed = self.norm_self(x) * (1 + scale_self) + shift_self
        x = x + gate_self * self.self_attn(normed, valid_mask)

        normed = self.norm_cross(x) * (1 + scale_cross) + shift_cross
        x = x + gate_cross * self.cross_attn(normed, context, context_mask)

        normed = self.norm_mlp(x) * (1 + scale_mlp) + shift_mlp
        return x + gate_mlp * self.mlp(normed)


class LatentDiT(nn.Module):
    def __init__(self, cfg: DiTConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.input_proj = nn.Linear(cfg.d_latent, d)
        self.position_embedding = nn.Embedding(cfg.max_len, d)
        # section null row is n_sections; padded positions also take it
        self.section_embedding = nn.Embedding(cfg.n_sections + 1, d)
        self.section_index_embedding = nn.Embedding(cfg.max_len, d)
        for embedding in (
            self.position_embedding,
            self.section_embedding,
            self.section_index_embedding,
        ):
            nn.init.normal_(embedding.weight, mean=0.0, std=EMBED_INIT_STD)

        self.cond = ConditionEmbedder(cfg)
        self.null_context = nn.Parameter(
            torch.normal(0.0, EMBED_INIT_STD, (1, 1, cfg.context_dim))
        )
        self.blocks = nn.ModuleList(DiTBlock(cfg) for _ in range(cfg.n_layers))

        self.final_norm = nn.LayerNorm(d, elementwise_affine=False)
        self.final_adaln = nn.Sequential(nn.SiLU(), nn.Linear(cfg.cond_dim, 2 * d))
        nn.init.zeros_(self.final_adaln[1].weight)
        nn.init.zeros_(self.final_adaln[1].bias)
        self.output_proj = nn.Linear(d, cfg.d_latent)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        valid_mask: torch.Tensor,
        attrs: torch.Tensor,
        key: torch.Tensor,
        tempo: torch.Tensor,
        complete: torch.Tensor,
        start_mid: torch.Tensor,
        sections: torch.Tensor,
        sec_index: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, length, _ = x.shape
        positions = torch.arange(length, device=x.device)
        h = (
            self.input_proj(x)
            + self.position_embedding(positions)[None]
            + self.section_embedding(sections)
            + self.section_index_embedding(sec_index)
        )
        c = self.cond(t, attrs, key, tempo, complete, start_mid)

        null_context = self.null_context.expand(batch, 1, self.cfg.context_dim)
        context = torch.cat((null_context, context.to(null_context.dtype)), dim=1)
        context_mask = torch.cat(
            (torch.ones(batch, 1, dtype=torch.bool, device=x.device), context_mask), dim=1
        )

        for block in self.blocks:
            h = block(h, c, valid_mask, context, context_mask)

        shift, scale = self.final_adaln(c).unsqueeze(1).chunk(2, dim=-1)
        return self.output_proj(self.final_norm(h) * (1 + scale) + shift)


# --------------------------------------------------------------------------
# latent standardisation
# --------------------------------------------------------------------------


def load_stats(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    stats = np.load(path)
    return stats["mean"].astype(np.float32), stats["std"].astype(np.float32)


def standardize(mu, stats: tuple):
    mean, std = stats
    return (mu - mean) / std


def destandardize(x, stats: tuple):
    mean, std = stats
    return x * std + mean
