# Adapted from MuseTok, "Symbolic Music Tokenization for Generation and
# Semantic Understanding" (ICASSP 2026): https://github.com/Yuer867/MuseTok.
"""Window codecs adapted from MuseTok: a residual-VQ tokenizer and a Gaussian VAE.

Both share one scaffolding: a Transformer encoder summarises each 1-second
window into one latent, which is quantized or sampled, and a causal Transformer
decoder reconstructs the concatenated token stream with each window's latent
added as a segment embedding before every layer. The window plays the role of
MuseTok's bar. ``pad_id`` is explicit because ``<P>`` is id 2, not the last id.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from vector_quantize_pytorch import ResidualFSQ, ResidualSimVQ, ResidualVQ

from .transformer_helpers import (
    PositionalEncoding,
    TokenEmbedding,
    generate_causal_mask,
    weights_init,
)


class TransformerEncoder(nn.Module):
    """Encodes a padded segment and reads the summary off position 0."""

    def __init__(self, n_layer, n_head, d_model, d_ff, dropout=0.1, activation="relu") -> None:
        super().__init__()
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.activation = activation

        self.tr_encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_head, d_ff, dropout, activation
        )
        self.tr_encoder = nn.TransformerEncoder(self.tr_encoder_layer, n_layer)

    def forward(self, x, padding_mask=None):
        out = self.tr_encoder(x, src_key_padding_mask=padding_mask)
        return out[0, :, :]


class TransformerDecoder(nn.Module):
    """Causal decoder that re-adds the segment embedding before every layer."""

    def __init__(
        self, n_layer, n_head, d_model, d_ff, d_seg_emb, dropout=0.1, activation="relu"
    ) -> None:
        super().__init__()
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_model = d_model
        self.d_ff = d_ff
        self.d_seg_emb = d_seg_emb
        self.dropout = dropout
        self.activation = activation

        self.seg_emb_proj = nn.Linear(d_seg_emb, d_model, bias=False)
        self.decoder_layers = nn.ModuleList(
            nn.TransformerEncoderLayer(d_model, n_head, d_ff, dropout, activation)
            for _ in range(n_layer)
        )

    def forward(self, x, seg_emb):
        attn_mask = generate_causal_mask(x.size(0)).to(x.device)
        seg_emb = self.seg_emb_proj(seg_emb)

        out = x
        for layer in self.decoder_layers:
            out = layer(out + seg_emb, src_mask=attn_mask)
        return out


class WindowSegmentCodec(nn.Module):
    """Encoder, decoder and segment-embedding scaffolding shared by both models.

    Shapes, with ``W`` windows per sample and ``L`` tokens per window:

    - ``enc_inp``         ``(L, batch, W)``
    - ``padding_mask``    ``(batch, W, L)``, True where padded
    - ``dec_inp``         ``(dec_len, batch)``
    - ``dec_window_idx``  ``(batch, dec_len)``, window owning each position,
      and ``W`` (out of range) for padding, whose segment embedding is zeroed
    """

    reg_name = "reg_loss"
    supports_codebook_usage = False

    def __init__(
        self,
        enc_n_layer,
        enc_n_head,
        enc_d_model,
        enc_d_ff,
        dec_n_layer,
        dec_n_head,
        dec_d_model,
        dec_d_ff,
        d_latent,
        d_embed,
        n_token,
        pad_id,
        enc_dropout=0.1,
        enc_activation="relu",
        dec_dropout=0.1,
        dec_activation="relu",
        is_training=True,
    ) -> None:
        super().__init__()
        self.enc_n_layer = enc_n_layer
        self.enc_n_head = enc_n_head
        self.enc_d_model = enc_d_model
        self.enc_d_ff = enc_d_ff
        self.enc_dropout = enc_dropout
        self.enc_activation = enc_activation

        self.dec_n_layer = dec_n_layer
        self.dec_n_head = dec_n_head
        self.dec_d_model = dec_d_model
        self.dec_d_ff = dec_d_ff
        self.dec_dropout = dec_dropout
        self.dec_activation = dec_activation

        self.d_latent = d_latent
        self.d_embed = d_embed
        self.n_token = n_token
        self.pad_id = pad_id
        self.is_training = is_training

        self.token_emb = TokenEmbedding(n_token, d_embed, enc_d_model)
        self.emb_dropout = nn.Dropout(enc_dropout)
        self.pe = PositionalEncoding(enc_d_model)

        self.encoder = TransformerEncoder(
            enc_n_layer, enc_n_head, enc_d_model, enc_d_ff, enc_dropout, enc_activation
        )
        self.decoder = TransformerDecoder(
            dec_n_layer, dec_n_head, dec_d_model, dec_d_ff, d_latent, dec_dropout, dec_activation
        )
        self.dec_out_proj = nn.Linear(dec_d_model, n_token)

    def finalize_init(self) -> None:
        """Apply MuseTok's init, then repair the parts that need real scale."""
        self.apply(weights_init)
        self.reset_parameters_standard()

    def reset_parameters_standard(self) -> None:
        """Undo ``weights_init``'s ``std=0.01`` on everything that needs scale.

        At ``std=0.01`` the position-0 readout is nearly identical across
        windows and the projected latent is far below the token embedding, so
        the transformer stacks and token embedding get standard initialisation,
        the latent heads get Xavier, and each subclass rescales its latent.
        """
        for module in (self.encoder, self.decoder.decoder_layers):
            for sub in module.modules():
                if isinstance(sub, nn.Linear):
                    nn.init.xavier_uniform_(sub.weight)
                    if sub.bias is not None:
                        nn.init.zeros_(sub.bias)
                elif isinstance(sub, nn.LayerNorm):
                    sub.reset_parameters()
                elif isinstance(sub, nn.MultiheadAttention):
                    sub._reset_parameters()

        # TokenEmbedding multiplies by d_proj ** 0.5, so unit-RMS embeddings
        # need std 1 / sqrt(d_embed) in the lookup table
        nn.init.normal_(self.token_emb.emb_lookup.weight, 0.0, self.d_embed ** -0.5)
        if self.token_emb.emb_proj is not None:
            nn.init.xavier_uniform_(self.token_emb.emb_proj.weight)

        nn.init.xavier_uniform_(self.decoder.seg_emb_proj.weight)
        nn.init.xavier_uniform_(self.dec_out_proj.weight)
        nn.init.zeros_(self.dec_out_proj.bias)

        self.reset_latent_head()

    def reset_latent_head(self) -> None:
        """Hook for the subclass's encoder-to-latent layers."""

    def encode_readout(self, enc_inp, padding_mask=None):
        """Windows to the encoder's position-0 readout, ``(batch, W, d_model)``."""
        batch_size, n_windows = enc_inp.size(1), enc_inp.size(2)
        token_emb = self.token_emb(enc_inp)
        token_emb = token_emb.reshape(enc_inp.size(0), -1, token_emb.size(-1))
        hidden = self.emb_dropout(token_emb) + self.pe(enc_inp.size(0))

        if padding_mask is not None:
            padding_mask = padding_mask.reshape(-1, padding_mask.size(-1))

        readout = self.encoder(hidden, padding_mask=padding_mask)
        return readout.reshape(batch_size, n_windows, -1)

    def segment_embedding(self, latent, dec_window_idx, dec_len):
        """Per-decoder-position segment embedding, zero on padded positions."""
        batch_size, n_windows = latent.size(0), latent.size(1)
        rows = torch.arange(batch_size, device=latent.device).unsqueeze(1)
        valid = dec_window_idx < n_windows
        gathered = latent[rows, dec_window_idx.clamp(max=n_windows - 1)]
        gathered = gathered * valid.unsqueeze(-1)
        return gathered.permute(1, 0, 2)[:dec_len]

    def decode(self, dec_inp, latent, dec_window_idx):
        dec_token_emb = self.token_emb(dec_inp)
        dec_hidden = self.emb_dropout(dec_token_emb) + self.pe(dec_inp.size(0))
        dec_seg_emb = self.segment_embedding(latent, dec_window_idx, dec_inp.size(0))
        return self.dec_out_proj(self.decoder(dec_hidden, dec_seg_emb))

    def reconstruction_loss(self, dec_logits, dec_tgt):
        return F.cross_entropy(
            dec_logits.view(-1, dec_logits.size(-1)),
            dec_tgt.contiguous().view(-1),
            ignore_index=self.pad_id,
            reduction="mean",
        ).float()

    def generate(self, inp, dec_seg_emb, keep_last_only=True):
        token_emb = self.token_emb(inp)
        dec_inp = self.emb_dropout(token_emb) + self.pe(inp.size(0))
        out = self.dec_out_proj(self.decoder(dec_inp, dec_seg_emb))
        return out[-1, ...] if keep_last_only else out


class TransformerResidualVQ(WindowSegmentCodec):
    """Window tokenizer with discrete residual-VQ codes."""

    reg_name = "commit_loss"
    supports_codebook_usage = True

    def __init__(
        self,
        enc_n_layer,
        enc_n_head,
        enc_d_model,
        enc_d_ff,
        dec_n_layer,
        dec_n_head,
        dec_d_model,
        dec_d_ff,
        d_latent,
        d_embed,
        n_token,
        num_quantizers,
        codebook_size,
        pad_id,
        rotation_trick=True,
        enc_dropout=0.1,
        enc_activation="relu",
        dec_dropout=0.1,
        dec_activation="relu",
        is_training=True,
        rvq_type="SimVQ",
        fsq_levels=(8, 5, 5, 5),
    ) -> None:
        super().__init__(
            enc_n_layer, enc_n_head, enc_d_model, enc_d_ff,
            dec_n_layer, dec_n_head, dec_d_model, dec_d_ff,
            d_latent, d_embed, n_token, pad_id,
            enc_dropout, enc_activation, dec_dropout, dec_activation, is_training,
        )
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size

        self.enc_out_proj = nn.Linear(enc_d_model, d_latent)
        self.enc_out_norm = nn.LayerNorm(d_latent)

        self.rvq_type = rvq_type
        if rvq_type == "SimVQ":
            self.residual_sim_vq = ResidualSimVQ(
                dim=d_latent,
                num_quantizers=num_quantizers,
                codebook_size=codebook_size,
                rotation_trick=rotation_trick,
            )
        elif rvq_type == "FSQ":
            self.residual_fsq = ResidualFSQ(
                dim=d_latent,
                num_quantizers=num_quantizers,
                levels=list(fsq_levels),
            )
        elif rvq_type == "RVQ":
            self.residual_vq = ResidualVQ(
                dim=d_latent,
                num_quantizers=num_quantizers,
                codebook_size=codebook_size,
                rotation_trick=rotation_trick,
                kmeans_init=True,
                kmeans_iters=10,
            )
        else:
            raise ValueError(
                f"unknown rvq_type {rvq_type!r}; choose from ['SimVQ', 'FSQ', 'RVQ']"
            )

        self.finalize_init()

    def reset_latent_head(self) -> None:
        nn.init.xavier_uniform_(self.enc_out_proj.weight)
        nn.init.zeros_(self.enc_out_proj.bias)
        nn.init.ones_(self.enc_out_norm.weight)
        nn.init.zeros_(self.enc_out_norm.bias)
        self.calibrate_quantizer()

    @torch.no_grad()
    def calibrate_quantizer(self, samples: int = 4096) -> float:
        """Rescale SimVQ's code transform to pass a unit-scale latent through.

        SimVQ quantizes against a *frozen* random codebook read through a
        learned linear map, so the achievable output scale is set at
        initialisation by that map. Probing it with unit-variance noise and
        scaling the map by the shortfall makes the quantized latent match the
        LayerNorm'd encoder output it has to reconstruct. Returns the measured
        output RMS after calibration, or ``0.0`` when the quantizer type has
        no such map (``RVQ`` learns its codebook and can use k-means init,
        ``FSQ`` fixes its scale by construction).
        """
        transforms = [
            module
            for name, module in self.named_modules()
            if name.endswith("code_transform") and isinstance(module, nn.Linear)
        ]
        if not transforms:
            return 0.0

        device = self.enc_out_proj.weight.device
        probe = torch.randn(samples, 1, self.d_latent, device=device)
        actual = self.quantize(probe)[0].float().pow(2).mean().sqrt().item()
        if actual > 0:
            for transform in transforms:
                transform.weight.mul_(1.0 / actual)
        return self.quantize(probe)[0].float().pow(2).mean().sqrt().item()

    def quantize(self, latent):
        if self.rvq_type == "SimVQ":
            return self.residual_sim_vq(latent)
        if self.rvq_type == "RVQ":
            return self.residual_vq(latent)
        quantized, indices = self.residual_fsq(latent)
        return quantized, indices, None

    def encode_continuous(self, enc_inp, padding_mask=None):
        """Pre-quantization latents, for probing or for a continuous encoder."""
        readout = self.encode_readout(enc_inp, padding_mask=padding_mask)
        return self.enc_out_norm(self.enc_out_proj(readout))

    def encode(self, enc_inp, padding_mask=None):
        """Windows to ``(quantized latent, codebook indices, commit loss)``.

        Latents come back as ``(batch, W, d_latent)`` and indices as
        ``(batch, W, num_quantizers)``.
        """
        return self.quantize(self.encode_continuous(enc_inp, padding_mask=padding_mask))

    def latent_for_windows(self, enc_inp, padding_mask=None):
        """Deterministic per-window embedding: the quantized latent."""
        return self.encode(enc_inp, padding_mask=padding_mask)[0]

    def code_indices(self, enc_inp, padding_mask=None):
        return self.encode(enc_inp, padding_mask=padding_mask)[1]

    def forward(self, enc_inp, dec_inp, dec_window_idx, padding_mask=None):
        latent, _, commit_loss = self.encode(enc_inp, padding_mask=padding_mask)
        return self.decode(dec_inp, latent, dec_window_idx), commit_loss

    def compute_loss(self, commit_loss, beta, dec_logits, dec_tgt):
        recons_loss = self.reconstruction_loss(dec_logits, dec_tgt)

        if commit_loss is None:
            commit_loss = torch.zeros_like(recons_loss)
        else:
            commit_loss = commit_loss.sum().float()

        return {
            "beta": beta,
            "total_loss": recons_loss + beta * commit_loss,
            "reg_loss": commit_loss,
            "commit_loss": commit_loss,
            "recons_loss": recons_loss,
        }


class TransformerVAE(WindowSegmentCodec):
    """Window codec with continuous Gaussian latents and a KL penalty.

    ``fc_mu`` and ``fc_logvar`` sit on the encoder's position-0 readout. The
    reparameterisation samples only in training mode and returns the posterior
    mean in ``eval()``; ``free_bits`` is stored on the model so both codecs share
    the trainer's loss signature.
    """

    reg_name = "kldiv_loss"

    def __init__(
        self,
        enc_n_layer,
        enc_n_head,
        enc_d_model,
        enc_d_ff,
        dec_n_layer,
        dec_n_head,
        dec_d_model,
        dec_d_ff,
        d_latent,
        d_embed,
        n_token,
        pad_id,
        free_bits=0.0,
        enc_dropout=0.1,
        enc_activation="relu",
        dec_dropout=0.1,
        dec_activation="relu",
        is_training=True,
    ) -> None:
        super().__init__(
            enc_n_layer, enc_n_head, enc_d_model, enc_d_ff,
            dec_n_layer, dec_n_head, dec_d_model, dec_d_ff,
            d_latent, d_embed, n_token, pad_id,
            enc_dropout, enc_activation, dec_dropout, dec_activation, is_training,
        )
        self.free_bits = free_bits

        self.fc_mu = nn.Linear(enc_d_model, d_latent)
        self.fc_logvar = nn.Linear(enc_d_model, d_latent)

        self.finalize_init()

    def reset_latent_head(self) -> None:
        nn.init.xavier_uniform_(self.fc_mu.weight)
        nn.init.zeros_(self.fc_mu.bias)
        # start near the unit prior: logvar ~ 0 means sigma ~ 1, so the sampled
        # latent lands on the same scale the decoder's other inputs use, which
        # is the scale the quantizer had to be calibrated to reach
        nn.init.zeros_(self.fc_logvar.weight)
        nn.init.zeros_(self.fc_logvar.bias)

    def reparameterize(self, mu, logvar, use_sampling=None, sampling_var=1.0):
        if use_sampling is None:
            use_sampling = self.training
        std = torch.exp(0.5 * logvar)
        if not use_sampling:
            return mu
        return mu + torch.randn_like(std) * std * sampling_var

    def encode(self, enc_inp, padding_mask=None):
        """Windows to ``(sampled latent, mu, logvar)``, each ``(batch, W, d_latent)``."""
        readout = self.encode_readout(enc_inp, padding_mask=padding_mask)
        mu, logvar = self.fc_mu(readout), self.fc_logvar(readout)
        return self.reparameterize(mu, logvar), mu, logvar

    def latent_for_windows(self, enc_inp, padding_mask=None):
        """Deterministic per-window embedding: the posterior mean."""
        readout = self.encode_readout(enc_inp, padding_mask=padding_mask)
        return self.fc_mu(readout)

    def forward(self, enc_inp, dec_inp, dec_window_idx, padding_mask=None):
        latent, mu, logvar = self.encode(enc_inp, padding_mask=padding_mask)
        return self.decode(dec_inp, latent, dec_window_idx), (mu, logvar)

    def compute_loss(self, moments, beta, dec_logits, dec_tgt):
        recons_loss = self.reconstruction_loss(dec_logits, dec_tgt)
        mu, logvar = moments

        # per-latent-dimension KL, averaged over every window in the batch, so
        # free bits act per dimension
        mu = mu.reshape(-1, mu.size(-1))
        logvar = logvar.reshape(-1, logvar.size(-1))
        kl_raw = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).mean(dim=0)
        kl_before_free_bits = kl_raw.mean()
        kldiv_loss = kl_raw.clamp(min=self.free_bits).mean()

        return {
            "beta": beta,
            "total_loss": recons_loss + beta * kldiv_loss,
            "reg_loss": kldiv_loss,
            "kldiv_loss": kldiv_loss,
            "kldiv_raw": kl_before_free_bits,
            "recons_loss": recons_loss,
        }


def build_codec(model_conf: dict, n_token: int, pad_id: int) -> WindowSegmentCodec:
    """Instantiate the codec named by ``model_conf['model_type']``."""
    model_type = model_conf.get("model_type", "ResidualVQ")
    shared = (
        model_conf["enc_n_layer"], model_conf["enc_n_head"],
        model_conf["enc_d_model"], model_conf["enc_d_ff"],
        model_conf["dec_n_layer"], model_conf["dec_n_head"],
        model_conf["dec_d_model"], model_conf["dec_d_ff"],
        model_conf["d_latent"], model_conf["d_embed"], n_token,
    )

    if model_type == "ResidualVQ":
        return TransformerResidualVQ(
            *shared,
            num_quantizers=model_conf["num_quantizers"],
            codebook_size=model_conf["codebook_size"],
            pad_id=pad_id,
            rotation_trick=model_conf["rotation_trick"],
            rvq_type=model_conf["rvq_type"],
        )
    if model_type == "VAE":
        return TransformerVAE(
            *shared,
            pad_id=pad_id,
            free_bits=model_conf.get("free_bits", 0.0),
        )
    raise ValueError(f"unknown model_type {model_type!r}; choose from ['ResidualVQ', 'VAE']")
