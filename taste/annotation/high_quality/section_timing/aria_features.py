# Uses Aria / aria-utils by EleutherAI:
# https://github.com/EleutherAI/aria and https://github.com/EleutherAI/aria-utils.
"""Frame-level features from the hidden states of the Aria base model (loubb/aria-medium-base).

Each note contributes the hidden state at its duration token (which causally sees the
whole note); notes are binned onto a fixed fps grid by onset, and every layer is returned.
"""
import sys

import numpy as np
import pretty_midi
import torch

from taste import paths

if str(paths.ARIA_ROOT) not in sys.path:
    sys.path.insert(0, str(paths.ARIA_ROOT))

from aria.model import Transformer, ModelConfig          # noqa: E402
from aria.config import load_model_config                 # noqa: E402
from ariautils.midi import MidiDict                       # noqa: E402
from ariautils.tokenizer import AbsTokenizer              # noqa: E402
from safetensors.torch import load_file                   # noqa: E402

_MODEL = None
_TOK = None
_LAYERS = None          # list populated by forward hooks


def _ckpt_path():
    from huggingface_hub import hf_hub_download
    # the repo also ships other variants; model.safetensors is the base model
    return hf_hub_download("loubb/aria-medium-base", "model.safetensors")


def _load(device="cuda:0"):
    global _MODEL, _TOK, _LAYERS
    if _MODEL is not None:
        return _MODEL, _TOK

    sd = load_file(_ckpt_path())
    # the checkpoint is a TransformerLM; keep only the inner Transformer
    inner = {k[len("model."):]: v for k, v in sd.items()
             if k.startswith("model.")}
    vocab = inner["tok_embeddings.weight"].shape[0]

    cfg = ModelConfig(**load_model_config("medium"))
    cfg.set_vocab_size(vocab)
    cfg.grad_checkpoint = False
    model = Transformer(cfg)
    model.load_state_dict(inner, strict=True)
    model = model.to(device).eval()

    _LAYERS = []
    for blk in model.encode_layers:
        blk.register_forward_hook(
            lambda _m, _i, out: _LAYERS.append(out.detach()))

    _MODEL, _TOK = model, AbsTokenizer()
    return _MODEL, _TOK


def _tokenize_with_note_times(midi_path, tok):
    """Return (seq, note_records) with note_records = [(dur_token_index, onset_sec), ...]
    in token order. Absolute onset = n_T * 5000 ms + onset."""
    md = MidiDict.from_midi(mid_path=str(midi_path))
    seq = tok.tokenize(md, add_dim_tok=False)
    step = tok.abs_time_step_ms

    n_T = 0
    records = []
    i = 0
    while i < len(seq):
        t = seq[i]
        if t == tok.time_tok:
            n_T += 1
            i += 1
            continue
        if (isinstance(t, tuple) and t[0] in tok.instruments_nd
                and i + 2 < len(seq)
                and isinstance(seq[i + 1], tuple) and seq[i + 1][0] == "onset"
                and isinstance(seq[i + 2], tuple) and seq[i + 2][0] == "dur"):
            onset_ms = n_T * step + seq[i + 1][1]
            records.append((i + 2, onset_ms / 1000.0))
            i += 3
            continue
        i += 1
    return seq, records


def _bin_notes(hidden, records, n, fps):
    """Per-note hidden vectors -> (n, dim) frame matrix, averaged per onset frame;
    empty frames repeat the previous frame."""
    dim = hidden.shape[1]
    acc = np.zeros((n, dim), dtype=np.float64)
    cnt = np.zeros(n, dtype=np.float64)
    for dur_idx, onset_sec in records:
        f = min(n - 1, int(onset_sec * fps))
        acc[f] += hidden[dur_idx]
        cnt[f] += 1

    X = np.zeros((n, dim), dtype=np.float32)
    last = None
    for f in range(n):
        if cnt[f] > 0:
            X[f] = acc[f] / cnt[f]
            last = X[f]
        elif last is not None:
            X[f] = last
    first = next((f for f in range(n) if cnt[f] > 0), 0)
    X[:first] = X[first]
    return X


@torch.inference_mode()
def extract_aria_features(midi_path, *, fps=2.0, device="cuda:0"):
    """MIDI file -> {"X_all": (n_layers, n_frames, dim), fps, n_frames, duration}.

    n_layers = 17: index 0 is the final layer-normed output, 1..16 the residual
    stream after each transformer block.
    """
    model, tok = _load(device)
    seq, records = _tokenize_with_note_times(midi_path, tok)
    duration = pretty_midi.PrettyMIDI(str(midi_path)).get_end_time()

    ids = torch.tensor(tok.encode(seq), device=device).view(1, -1)
    _LAYERS.clear()
    final = model(ids)                                  # (1, L, d_model)
    hiddens = [final[0]] + [h[0] for h in _LAYERS]      # 17 x (L, d_model)

    n = max(1, int(round(duration * fps)))
    X_all = np.stack([
        _bin_notes(h.float().cpu().numpy(), records, n, fps) for h in hiddens
    ])
    return {"X_all": X_all, "fps": fps, "n_frames": n, "duration": duration}
