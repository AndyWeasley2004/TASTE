"""Constrained clustering primitives: section start times from piano MIDI and a section label sequence.

Frame-level MIDI features at a fixed rate are segmented so that sections sharing a
letter (e.g. A B A C) are tied to one feature centroid; a duration prior and a Foote
novelty curve shape the boundaries, which are refined by Lloyd-style alternation.
Only the letter pattern and the section count of the label sequence are used.
"""
import re
import numpy as np
import pretty_midi
from scipy.ndimage import gaussian_filter1d


# ----------------------------------------------------------------------
# labels
# ----------------------------------------------------------------------
def section_letter(label):
    """'B3' -> 'B'.  The repetition identity of a section header."""
    m = re.match(r"[A-Za-z]+", str(label))
    return m.group(0) if m else str(label)


# ----------------------------------------------------------------------
# features
# ----------------------------------------------------------------------
def extract_features(midi_path, fps=2.0):
    """MIDI -> (X, fps, n_frames, duration_sec).

    X is a per-frame feature matrix combining a sustain/velocity-weighted
    chroma vector with scalar texture descriptors. Frame rate is fixed; the
    method never inspects beats or bars.
    """
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    dur = pm.get_end_time()
    hop = 1.0 / fps
    n = max(1, int(round(dur * fps)))

    notes = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for nt in inst.notes:
            if nt.end > nt.start:
                notes.append((nt.start, nt.end, nt.pitch, nt.velocity))
    notes.sort()

    chroma = np.zeros((n, 12))
    poly = np.zeros(n)                 # sustained polyphony
    pitch_sum = np.zeros(n)            # weighted pitch accumulator
    pitch_w = np.zeros(n)
    lo = np.full(n, np.nan)            # lowest sounding pitch
    hi = np.full(n, np.nan)            # highest sounding pitch
    onset_cnt = np.zeros(n)
    onset_vel = np.zeros(n)
    onset_dur = np.zeros(n)

    for (s, e, p, v) in notes:
        f0 = max(0, int(s * fps))
        f1 = min(n, max(int(np.ceil(e * fps)), f0 + 1))
        for f in range(f0, f1):
            fs, fe = f * hop, f * hop + hop
            ov = min(e, fe) - max(s, fs)
            if ov <= 0:
                continue
            w = ov * v
            chroma[f, p % 12] += w
            poly[f] += ov / hop
            pitch_sum[f] += p * w
            pitch_w[f] += w
            lo[f] = p if np.isnan(lo[f]) else min(lo[f], p)
            hi[f] = p if np.isnan(hi[f]) else max(hi[f], p)
        of = int(s * fps)
        if 0 <= of < n:
            onset_cnt[of] += 1
            onset_vel[of] += v
            onset_dur[of] += (e - s)

    active = pitch_w > 0
    mean_pitch = np.where(active, pitch_sum / np.maximum(pitch_w, 1e-9), 60.0)
    lo = np.where(np.isnan(lo), 60.0, lo)
    hi = np.where(np.isnan(hi), 60.0, hi)
    spread = hi - lo
    o_vel = np.where(onset_cnt > 0, onset_vel / np.maximum(onset_cnt, 1e-9), 0.0)
    o_dur = np.where(onset_cnt > 0, onset_dur / np.maximum(onset_cnt, 1e-9), 0.0)

    scal = np.stack([
        np.log1p(poly),
        mean_pitch,
        spread,
        np.log1p(onset_cnt),       # onset density
        o_vel,
        np.log1p(o_dur),           # articulation proxy
    ], axis=1)

    return {"chroma": chroma, "scalar": scal, "fps": fps,
            "n_frames": n, "duration": dur}


def _norm_rows(M, weight):
    """L2-normalise each row, then scale."""
    return M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-9) * weight


def build_matrix(feat, chroma_weight=2.0, smooth_sigma=1.5):
    """Normalise + fuse harmony + scalar texture features into X."""
    blocks = [_norm_rows(feat["chroma"].copy(), chroma_weight)]

    scal = feat["scalar"].astype(float).copy()
    mu = scal.mean(0, keepdims=True)
    sd = scal.std(0, keepdims=True)
    blocks.append((scal - mu) / np.maximum(sd, 1e-6))

    X = np.concatenate(blocks, axis=1)
    if smooth_sigma > 0:
        X = gaussian_filter1d(X, sigma=smooth_sigma, axis=0, mode="nearest")
    # scale so a per-frame squared distance is O(1), keeps lambdas interpretable
    X = X / np.sqrt(X.shape[1])
    return X


def self_similarity(X):
    """Cosine self-similarity matrix of the frame features."""
    norm = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    return norm @ norm.T


def _checkerboard_kernel(L):
    """Gaussian-tapered checkerboard kernel, size (2L, 2L)."""
    ax = np.arange(-L, L)
    xx, yy = np.meshgrid(ax, ax)
    taper = np.exp(-0.5 * (xx ** 2 + yy ** 2) / (L / 2.0) ** 2)
    sign = np.sign(xx + 0.5) * np.sign(yy + 0.5)   # +diag blocks, -anti-diag
    return sign * taper


def novelty_curve(X, win_frames):
    """Foote checkerboard novelty on the frame self-similarity matrix.

    Sharper than a window-mean contrast: a boundary is where the local SSM
    flips from one homogeneous block to another. Beat-tracking-free.
    """
    n = len(X)
    L = max(2, win_frames)
    S = self_similarity(X)
    K = _checkerboard_kernel(L)
    nov = np.zeros(n)
    for t in range(n):
        a0, a1 = t - L, t + L
        ka0 = max(0, -a0)
        kb0 = max(0, -a0)
        sa0, sa1 = max(0, a0), min(n, a1)
        ka1 = ka0 + (sa1 - sa0)
        nov[t] = np.sum(S[sa0:sa1, sa0:sa1] * K[ka0:ka1, kb0:kb0 + (sa1 - sa0)])
    nov = np.maximum(nov, 0.0)
    if nov.max() > 0:
        nov = nov / nov.max()
    return nov


# ----------------------------------------------------------------------
# constrained segmentation
# ----------------------------------------------------------------------
def _dp_segment(X, lab, mu, nov, min_len, mean_len, lam_dur, lam_nov):
    """Optimal boundaries for a fixed label-class assignment per segment.

    dp[i][e] = min cost of placing segments 0..i-1 with boundary b_i = e.
    Segment i pays sum of squared distances of its frames to centroid mu[lab[i]],
    a quadratic duration penalty, and (for internal boundaries) a novelty bonus.
    """
    n = X.shape[0]
    N = len(lab)
    K = mu.shape[0]

    xx = (X ** 2).sum(1)
    dist = xx[:, None] - 2.0 * (X @ mu.T) + (mu ** 2).sum(1)[None, :]
    dist = np.maximum(dist, 0.0)                      # (n, K)
    pref = np.zeros((K, n + 1))
    pref[:, 1:] = np.cumsum(dist.T, axis=1)

    INF = 1e18
    dp = np.full((N + 1, n + 1), INF)
    bk = np.zeros((N + 1, n + 1), dtype=int)
    dp[0, 0] = 0.0

    for i in range(N):
        c = lab[i]
        for e in range(min_len, n + 1):
            s_hi = e - min_len
            if s_hi < 0:
                continue
            s = np.arange(0, s_hi + 1)
            prev = dp[i, s]
            seg = pref[c, e] - pref[c, s]
            d = e - s
            durp = lam_dur * (d - mean_len) ** 2 / mean_len
            val = prev + seg + durp
            j = int(np.argmin(val))
            best = val[j]
            if best >= INF:
                continue
            bcost = -lam_nov * nov[e] if (i < N - 1 and e < n) else 0.0
            dp[i + 1, e] = best + bcost
            bk[i + 1, e] = s[j]

    bnds = [n]
    e = n
    for i in range(N, 0, -1):
        s = bk[i, e]
        bnds.append(s)
        e = s
    return np.array(bnds[::-1]), float(dp[N, n])


def _lloyd(X, lab, K, nov, min_len, mean_len, lam_dur, lam_nov, init, n_iter):
    """One Lloyd run from a given initialisation; returns (bnds, objective)."""
    bnds = init
    cost = np.inf
    for _ in range(n_iter):
        mu = np.zeros((K, X.shape[1]))
        cnt = np.zeros(K)
        for i in range(len(lab)):
            seg = X[bnds[i]:bnds[i + 1]]
            mu[lab[i]] += seg.sum(0)
            cnt[lab[i]] += len(seg)
        mu /= np.maximum(cnt[:, None], 1e-9)
        new, cost = _dp_segment(X, lab, mu, nov, min_len, mean_len,
                                lam_dur, lam_nov)
        if np.array_equal(new, bnds):
            break
        bnds = new
    return bnds, cost


def segment(X, letters, fps, *, n_iter=12, min_sec=8.0, lam_dur=0.12,
            lam_nov=1.0, nov_win_sec=10.0, init=None):
    """Constrained tied-emission segmentation (Lloyd alternation).

    init: optional boundary frame indices (length N+1) to start Lloyd from;
    defaults to an equally-spaced partition. Returns boundary frame indices,
    length N+1 (0 ... n_frames).
    """
    n = X.shape[0]
    N = len(letters)
    uniq = {c: k for k, c in enumerate(dict.fromkeys(letters))}
    lab = np.array([uniq[c] for c in letters])
    K = len(uniq)
    min_len = max(1, int(round(min_sec * fps)))
    min_len = min(min_len, n // N if N > 0 else n)
    mean_len = n / N

    nov = novelty_curve(X, max(1, int(round(nov_win_sec * fps))))
    if init is None:
        init = np.round(np.linspace(0, n, N + 1)).astype(int)
    else:
        init = np.clip(np.asarray(init, dtype=int), 0, n)
    bnds, _ = _lloyd(X, lab, K, nov, min_len, mean_len,
                     lam_dur, lam_nov, init, n_iter)
    return bnds


def infer_section_starts(midi_path, section_sequence, *, fps=2.0,
                         chroma_weight=2.0, smooth_sigma=1.5,
                         lam_dur=0.12, lam_nov=1.0, min_sec=8.0,
                         nov_win_sec=10.0):
    """Top-level entry point: MIDI + label sequence -> list of start seconds.

    Tied-centroid constrained segmentation: sections sharing a letter are tied
    to one feature centroid (the repetition prior), a duration prior keeps
    segment lengths plausible, and a content-novelty curve nudges boundaries
    onto points of musical change. Training-free; no beat tracking.
    """
    letters = [section_letter(s) for s in section_sequence]
    feat = extract_features(midi_path, fps=fps)
    X = build_matrix(feat, chroma_weight=chroma_weight,
                     smooth_sigma=smooth_sigma)
    bnds = segment(X, letters, feat["fps"], min_sec=min_sec,
                   lam_dur=lam_dur, lam_nov=lam_nov, nov_win_sec=nov_win_sec)
    hop = 1.0 / feat["fps"]
    starts = [round(float(b) * hop, 3) for b in bnds[:-1]]
    starts[0] = 0.0
    return starts


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("midi")
    ap.add_argument("labels", help="comma-separated, e.g. A0,B1,A2,C3")
    args = ap.parse_args()
    seq = [s.strip() for s in args.labels.split(",") if s.strip()]
    print(json.dumps(infer_section_starts(args.midi, seq), indent=2))
