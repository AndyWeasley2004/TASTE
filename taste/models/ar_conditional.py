"""Conditional wrapper that adds attribute and imagery conditioning to the AR generator.

Two pathways are grafted onto the decoder from ``taste.models.ar``:

* **Attributes** (emotion, genre, accompaniment, performance): one embedding
  table per field with a trailing learned *null* row, used for both a missing
  label and a dropped prefix. The four embeddings are summed and passed through
  a shared trunk; each decoder layer owns a zero-initialised head that emits
  ``(shift, scale, gate)`` for its self-attention and MLP branches (AdaLN-Zero).
  The gate enters as ``(1 + gate)`` so that a zero head leaves the pretrained
  branches at unit weight.
* **Imagery**: precomputed frozen Qwen3-Embedding-4B per-token final-layer
  states (``CONTEXT_DIM`` = 2560), read by a cross-attention block inserted in
  every decoder layer between self-attention and the MLP, with a
  zero-initialised output projection. A learned null context token is always
  prepended to the states; a dropped-prefix row carries only that token.

Both pathways are identities at initialisation, so a freshly wrapped model
reproduces the pretrained model's logits.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.qwen3 import modeling_qwen3
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

from taste.models.ar import ArForCausalLM, load_model

REPO = Path(__file__).resolve().parents[2]

# The order fixes which embedding table owns which attribute, so it must stay
# fixed across checkpoints.
ATTRIBUTE_FIELDS = ("emotion", "genre", "accompaniment", "performance")

COND_TRUNK_DIM = 512
CROSS_ATTN_HEADS = 16
CROSS_ATTN_HEAD_DIM = 64
CONTEXT_DIM = 2560
EMBED_INIT_STD = 0.02


def load_attribute_classes(path: str | Path = REPO / "attributes.json") -> dict[str, list[str]]:
    classes = json.load(open(path))
    return {field: list(classes[field]) for field in ATTRIBUTE_FIELDS}


def null_indices(classes: dict[str, list[str]]) -> dict[str, int]:
    """The null row is appended after the real classes of each field."""
    return {field: len(values) for field, values in classes.items()}


class AttributeConditioner(nn.Module):
    """Sum of four attribute embeddings, then a shared trunk to the AdaLN code."""

    def __init__(self, classes: dict[str, list[str]], hidden_size: int):
        super().__init__()
        self.embeddings = nn.ModuleList(
            nn.Embedding(len(classes[field]) + 1, hidden_size) for field in ATTRIBUTE_FIELDS
        )
        for embedding in self.embeddings:
            nn.init.normal_(embedding.weight, mean=0.0, std=EMBED_INIT_STD)
        self.trunk = nn.Sequential(
            nn.Linear(hidden_size, COND_TRUNK_DIM),
            nn.SiLU(),
        )

    def forward(self, attribute_ids: torch.LongTensor) -> torch.Tensor:
        """``attribute_ids`` is ``(batch, 4)`` in ``ATTRIBUTE_FIELDS`` order."""
        summed = sum(
            embedding(attribute_ids[:, index])
            for index, embedding in enumerate(self.embeddings)
        )
        return self.trunk(summed)


class ImageryCrossAttention(nn.Module):
    """Pre-norm multi-head cross-attention with a zero-initialised output."""

    def __init__(self, hidden_size: int, rms_norm_eps: float):
        super().__init__()
        self.n_heads = CROSS_ATTN_HEADS
        self.head_dim = CROSS_ATTN_HEAD_DIM
        inner = self.n_heads * self.head_dim
        # resolved at call time so the Liger RMSNorm class patch is picked up
        self.norm = modeling_qwen3.Qwen3RMSNorm(hidden_size, eps=rms_norm_eps)
        self.q_proj = nn.Linear(hidden_size, inner, bias=False)
        self.k_proj = nn.Linear(CONTEXT_DIM, inner, bias=False)
        self.v_proj = nn.Linear(CONTEXT_DIM, inner, bias=False)
        self.o_proj = nn.Linear(inner, hidden_size, bias=False)
        nn.init.zeros_(self.o_proj.weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, length, _ = hidden_states.shape
        context_length = context.shape[1]

        def split(tensor, seq):
            return tensor.view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)

        query = split(self.q_proj(self.norm(hidden_states)), length)
        key = split(self.k_proj(context), context_length)
        value = split(self.v_proj(context), context_length)
        # (batch, 1, 1, context_length) key-padding mask, broadcast over queries
        mask = context_mask[:, None, None, :]
        attended = F.scaled_dot_product_attention(query, key, value, attn_mask=mask)
        attended = attended.transpose(1, 2).reshape(batch, length, -1)
        return self.o_proj(attended)


class ConditionalQwen3DecoderLayer(Qwen3DecoderLayer):
    """Decoder layer with AdaLN on both norms and an imagery cross-attention.

    Instances are produced by re-typing pretrained layers in
    ``attach_conditioning``, which keeps the pretrained parameter names.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        cond: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        attention_mask=None,
        **kwargs,
    ) -> torch.Tensor:
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
            self.adaln(cond).unsqueeze(1).chunk(6, dim=-1)
        )

        residual = hidden_states
        normed = self.input_layernorm(hidden_states) * (1 + scale_attn) + shift_attn
        attended, _ = self.self_attn(
            hidden_states=normed, attention_mask=attention_mask, **kwargs
        )
        # (1 + gate): a zero-initialised head must leave the pretrained branch at unit weight
        hidden_states = residual + (1 + gate_attn) * attended

        hidden_states = hidden_states + self.cross_attn(
            hidden_states, context, context_mask
        )

        residual = hidden_states
        normed = self.post_attention_layernorm(hidden_states) * (1 + scale_mlp) + shift_mlp
        return residual + (1 + gate_mlp) * self.mlp(normed)


def attach_conditioning(model: ArForCausalLM, classes: dict[str, list[str]]) -> None:
    """Re-type every decoder layer in place and attach its new submodules.

    Re-typing keeps every pretrained parameter under its original state-dict
    name, so the wrapped model loads both pretrain and conditional checkpoints
    without key rewriting.
    """
    if type(model.model.layers[0]).__name__ != "Qwen3DecoderLayer":
        raise ValueError(
            f"expected transformers Qwen3DecoderLayer, found "
            f"{type(model.model.layers[0]).__name__}"
        )
    config = model.config
    dtype = model.dtype
    for layer in model.model.layers:
        layer.__class__ = ConditionalQwen3DecoderLayer
        adaln = nn.Linear(COND_TRUNK_DIM, 6 * config.hidden_size)
        nn.init.zeros_(adaln.weight)
        nn.init.zeros_(adaln.bias)
        layer.adaln = adaln.to(dtype)
        layer.cross_attn = ImageryCrossAttention(
            config.hidden_size, config.rms_norm_eps
        ).to(dtype)
    model.attribute_conditioner = AttributeConditioner(
        classes, config.hidden_size
    ).to(dtype)
    model.null_context = nn.Parameter(
        torch.normal(0.0, EMBED_INIT_STD, (1, 1, CONTEXT_DIM), dtype=dtype)
    )


class ArConditionalForCausalLM(ArForCausalLM):
    """AR generator plus the attribute and imagery pathways."""

    accepts_loss_kwargs = False

    def forward(
        self,
        input_ids=None,
        attribute_ids=None,
        context_states=None,
        context_lengths=None,
        attention_mask=None,
        labels=None,
        **kwargs,
    ):
        batch = input_ids.shape[0]
        cond = self.attribute_conditioner(attribute_ids)

        null_context = self.null_context.expand(batch, 1, CONTEXT_DIM)
        context = torch.cat((null_context, context_states.to(null_context.dtype)), dim=1)
        # the null token is always visible; real context positions only up to the
        # row's true length (0 for a dropped prefix)
        positions = torch.arange(context.shape[1], device=context.device)
        context_mask = positions[None, :] <= context_lengths[:, None]

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            cond=cond,
            context=context,
            context_mask=context_mask,
            **kwargs,
        )


def load_conditional_model(
    path: str | Path,
    *,
    dtype=torch.bfloat16,
    attributes_path: str | Path = REPO / "attributes.json",
    **kwargs,
):
    """Load a checkpoint and graft the conditioning pathways onto it.

    A pretrain checkpoint gets identity-initialised pathways; a conditional
    checkpoint (recognised by its saved conditioning tensors) has them reloaded
    after the graft, since ``from_pretrained`` drops those keys as unexpected.
    """
    model = load_model(path, dtype=dtype, **kwargs)
    model.__class__ = ArConditionalForCausalLM
    attach_conditioning(model, load_attribute_classes(attributes_path))

    from safetensors.torch import load_file

    shards = sorted(Path(path).glob("model*.safetensors"))
    state = {}
    for shard in shards:
        state.update(load_file(shard))
    if any(key.startswith("attribute_conditioner.") for key in state):
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected or not set(missing) <= {"lm_head.weight"}:
            raise ValueError(
                f"conditional checkpoint reload mismatch: missing {missing}, "
                f"unexpected {unexpected}"
            )
    return model


def parameter_report(model: ArConditionalForCausalLM) -> dict[str, int]:
    new = 0
    for layer in model.model.layers:
        new += sum(p.numel() for p in layer.adaln.parameters())
        new += sum(p.numel() for p in layer.cross_attn.parameters())
    new += sum(p.numel() for p in model.attribute_conditioner.parameters())
    new += model.null_context.numel()
    total = sum(p.numel() for p in model.parameters())
    return {"total": total, "new": new, "pretrained": total - new}
