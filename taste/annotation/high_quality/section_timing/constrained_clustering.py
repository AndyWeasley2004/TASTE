"""Section start times by two-stage constrained clustering.

Stage 1 segments hand-crafted chroma and texture features from an equally spaced start;
stage 2 repeats the segmentation on Aria hidden states (layer 7), initialised from the
stage-1 boundaries. Both stages tie sections that share a letter to one centroid.

    python -m taste.annotation.high_quality.section_timing.constrained_clustering <midi> A0,B1,A2,C3
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d

from taste.annotation.high_quality.section_timing.aria_features import extract_aria_features
from taste.annotation.high_quality.section_timing.infer import (
    build_matrix, extract_features, section_letter, segment)

FPS = 2.0               # one feature frame every 0.5 s
ARIA_LAYER = 7          # transformer layer read from aria-medium-base
SMOOTH_SIGMA = 1.5      # temporal smoothing of the Aria features
MIN_SEC = 8.0           # shortest allowed section
LAM_DUR = 0.1           # length-prior weight (both stages)
LAM_NOV_STAGE1 = 0.0    # stage 1: no change bonus
LAM_NOV_STAGE2 = 1.0    # stage 2: reward cutting on musical change


def _prep(X):
    """Per-piece standardise -> smooth over time -> L2-normalise each frame.

    Standardising removes the constant solo-piano component of the embeddings, which
    would otherwise dominate after the per-frame L2 normalisation."""
    X = np.asarray(X, dtype=np.float64)
    X = (X - X.mean(0, keepdims=True)) / np.maximum(X.std(0, keepdims=True), 1e-6)
    X = gaussian_filter1d(X, sigma=SMOOTH_SIGMA, axis=0, mode="nearest")
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)


def infer_section_starts(midi_path, section_sequence, *, device="cuda:0"):
    """MIDI + section-label sequence -> list of section start times (seconds).

    Only the letter pattern of section_sequence is used (repeated letters are tied).
    """
    letters = [section_letter(s) for s in section_sequence]

    X_hc = build_matrix(extract_features(midi_path, fps=FPS))
    bnds_stage1 = segment(X_hc, letters, FPS, min_sec=MIN_SEC,
                          lam_dur=LAM_DUR, lam_nov=LAM_NOV_STAGE1)

    X_aria = _prep(extract_aria_features(midi_path, fps=FPS,
                                         device=device)["X_all"][ARIA_LAYER])
    bnds = segment(X_aria, letters, FPS, min_sec=MIN_SEC, lam_dur=LAM_DUR,
                   lam_nov=LAM_NOV_STAGE2, init=bnds_stage1)

    hop = 1.0 / FPS
    starts = [round(float(b) * hop, 3) for b in bnds[:-1]]
    starts[0] = 0.0
    return starts


if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("midi")
    ap.add_argument("labels", help="comma-separated, e.g. A0,B1,A2,C3")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    seq = [s.strip() for s in args.labels.split(",") if s.strip()]
    print(json.dumps(
        infer_section_starts(args.midi, seq, device=args.device),
        indent=2))
