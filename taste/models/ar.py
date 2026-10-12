"""Autoregressive MIDI generator: a 0.89B Qwen3 decoder over the MIDI vocabulary.

The model's token ids are the MIDI vocabulary ids from ``taste.representation.vocab``.
"""
from __future__ import annotations

from pathlib import Path

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from taste.representation.vocab import (
    BOS_ID,
    EOS_ID,
    MODEL_VOCAB_SIZE,
    PAD_ID,
    load_vocab_meta,
)

SEQ_LEN = 4096


def build_config() -> Qwen3Config:
    return Qwen3Config(
        vocab_size=MODEL_VOCAB_SIZE,
        hidden_size=1536,
        num_hidden_layers=26,
        num_attention_heads=24,
        num_key_value_heads=6,
        head_dim=64,
        intermediate_size=6144,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        rope_theta=100_000.0,
        max_position_embeddings=SEQ_LEN,
        tie_word_embeddings=True,
        attention_bias=False,
        bos_token_id=BOS_ID,
        eos_token_id=EOS_ID,
        pad_token_id=PAD_ID,
    )


class ArForCausalLM(Qwen3ForCausalLM):
    """Qwen3 decoder that computes its training loss with fused linear cross-entropy.

    Without labels the standard Hugging Face forward runs. With labels the loss is
    computed from the hidden states and the tied output weight, so the full
    sequence-by-vocabulary logits tensor is never materialised.
    """

    # The fused loss is a per-microbatch mean and ignores Transformers'
    # ``num_items_in_batch``; this makes Trainer divide by gradient accumulation.
    accepts_loss_kwargs = False

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        logits_to_keep=0,
        **kwargs,
    ):
        if labels is None:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=None,
                use_cache=use_cache,
                logits_to_keep=logits_to_keep,
                **kwargs,
            )

        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        hidden = outputs.last_hidden_state[:, :-1].contiguous()
        targets = labels[:, 1:].contiguous()
        loss = LigerFusedLinearCrossEntropyLoss(ignore_index=-100)(
            self.lm_head.weight,
            hidden.view(-1, hidden.shape[-1]),
            targets.view(-1),
            self.lm_head.bias,
        )
        return CausalLMOutputWithPast(
            loss=loss,
            logits=None,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def new_model(*, dtype=torch.bfloat16, **kwargs):
    """Randomly initialised generator."""
    return ArForCausalLM._from_config(build_config(), dtype=dtype, **kwargs)


def load_model(path: str | Path, *, dtype=torch.bfloat16, **kwargs):
    """Load a trained checkpoint directory; its ``midi_vocab.json`` must match the vocabulary."""
    load_vocab_meta(path)
    return ArForCausalLM.from_pretrained(str(path), dtype=dtype, **kwargs)
