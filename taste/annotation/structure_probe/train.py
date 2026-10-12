"""Train the structure probe: a frame-level boundary head and a section-grouping head.

A small transformer encoder over the 26-d frame features predicts Gaussian-widened
boundary targets (BCE); a projection head gives segment embeddings whose pairwise
same-section logit `scale * cos(e_i, e_j) + bias` is trained on the teacher's lettered
spans with class-balanced BCE. The grouping head reads a stop-gradient copy of the
encoder output. High-quality rows are repeated three times in training.

    python -m taste.annotation.structure_probe.train
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from taste import paths

ROOT = paths.data("structure_probe")
RUN_DIR = paths.ckpt("probes", "structure_probe")
HOP_SEC = 0.5
SIGMA_SEC = 1.5
FEATURE_DIM = 26
VAL_FRACTION = 0.05
SPLIT_SEED = 20260812
MIN_DIST_FRAMES = 24  # 12 s duration floor
MAX_CLUSTERS = 8


def segment_frames(row: dict, n_frames: int) -> tuple[list[tuple[int, int]], list[str]]:
    """Teacher spans as [f0, f1) frame ranges; zero-length spans dropped."""
    spans, letters = [], []
    for start, end, letter in zip(row["section_starts_sec"],
                                  row["section_ends_sec"],
                                  row["section_letters"]):
        f0 = min(max(int(start / HOP_SEC), 0), n_frames)
        f1 = min(max(int(math.ceil(end / HOP_SEC)), 0), n_frames)
        if f1 <= f0:
            continue
        spans.append((f0, f1))
        letters.append(str(letter))
    return spans, letters


class PieceDataset(Dataset):
    def __init__(self, rows: list[dict], frames: dict[str, np.ndarray]):
        self.rows = rows
        self.frames = frames

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        row = self.rows[i]
        feats = torch.from_numpy(self.frames[row["piece_id"]])
        n = feats.shape[0]
        target = np.zeros(n, dtype=np.float32)
        times = np.arange(n) * HOP_SEC
        for b in row["boundaries_sec"]:
            target = np.maximum(
                target, np.exp(-((times - b) ** 2) / (2 * SIGMA_SEC ** 2)))
        spans, letters = segment_frames(row, n)
        return feats, torch.from_numpy(target), spans, letters


def collate(batch):
    lengths = [f.shape[0] for f, _, _, _ in batch]
    n = max(lengths)
    feats = torch.zeros(len(batch), n, FEATURE_DIM)
    target = torch.zeros(len(batch), n)
    mask = torch.zeros(len(batch), n, dtype=torch.bool)
    spans, letters = [], []
    for i, (f, t, sp, lt) in enumerate(batch):
        feats[i, :f.shape[0]] = f
        target[i, :f.shape[0]] = t
        mask[i, :f.shape[0]] = True
        spans.append(sp)
        letters.append(lt)
    return feats, target, mask, spans, letters


class StructureProbe(nn.Module):
    def __init__(self, d_model: int = 192, layers: int = 4, heads: int = 4,
                 proj_dim: int = 64):
        super().__init__()
        self.input = nn.Sequential(
            nn.LayerNorm(FEATURE_DIM), nn.Linear(FEATURE_DIM, d_model))
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model, heads, dim_feedforward=4 * d_model, dropout=0.1,
                batch_first=True, norm_first=True),
            num_layers=layers)
        self.head = nn.Linear(d_model, 1)
        self.proj = nn.Sequential(
            nn.Linear(d_model, 128), nn.GELU(), nn.Linear(128, proj_dim))
        self.scale = nn.Parameter(torch.tensor(10.0))
        self.bias = nn.Parameter(torch.tensor(-5.0))
        self.d_model = d_model

    def positional(self, n: int, device) -> torch.Tensor:
        pos = torch.arange(n, device=device).unsqueeze(1)
        div = torch.exp(torch.arange(0, self.d_model, 2, device=device)
                        * (-math.log(10000.0) / self.d_model))
        pe = torch.zeros(n, self.d_model, device=device)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    def encode(self, feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.input(feats) + self.positional(feats.shape[1], feats.device)
        return self.encoder(x, src_key_padding_mask=~mask)

    def forward(self, feats: torch.Tensor, mask: torch.Tensor):
        h = self.encode(feats, mask)
        # stop-grad isolates the grouping objective from the shared encoder
        h_proj = h.detach() if self.training else h
        return self.head(h).squeeze(-1), self.proj(h_proj)

    def segment_embeddings(self, proj_i: torch.Tensor,
                           spans: list[tuple[int, int]]) -> torch.Tensor:
        """[n_seg, proj_dim] L2-normalized mean-pooled segment embeddings."""
        pooled = torch.stack([proj_i[f0:f1].mean(0) for f0, f1 in spans])
        return nn.functional.normalize(pooled, dim=-1)

    def pair_logits(self, emb: torch.Tensor) -> torch.Tensor:
        return self.scale * (emb @ emb.t()) + self.bias


def pairwise_targets(letters: list[str]) -> torch.Tensor:
    arr = np.array(letters)
    return torch.from_numpy((arr[:, None] == arr[None, :]).astype(np.float32))


def piece_pair_loss(logits: torch.Tensor, target: torch.Tensor
                    ) -> torch.Tensor | None:
    """Class-balanced BCE over unordered pairs (i<j); None if a class is empty."""
    iu = torch.triu_indices(logits.shape[0], logits.shape[0], offset=1,
                            device=logits.device)
    if iu.shape[1] == 0:
        return None
    lg = logits[iu[0], iu[1]]
    tg = target[iu[0], iu[1]]
    n_pos = tg.sum()
    n_neg = (1 - tg).sum()
    if n_pos == 0 or n_neg == 0:
        return None
    w = tg / (2 * n_pos) + (1 - tg) / (2 * n_neg)
    bce = nn.functional.binary_cross_entropy_with_logits(lg, tg,
                                                         reduction="none")
    return (bce * w).sum()


def pick_peaks(prob: np.ndarray, threshold: float,
               min_dist_frames: int = MIN_DIST_FRAMES) -> list[float]:
    order = np.argsort(prob)[::-1]
    chosen: list[int] = []
    for idx in order:
        if prob[idx] < threshold:
            break
        if all(abs(idx - c) >= min_dist_frames for c in chosen):
            chosen.append(int(idx))
    return sorted(c * HOP_SEC for c in chosen)


def hit_rate_f1(pred: list[float], ref: list[float], tol: float
                ) -> tuple[int, int, int]:
    """One-to-one greedy matching within tolerance -> (hits, n_pred, n_ref)."""
    used = [False] * len(pred)
    hits = 0
    for r in ref:
        best, best_d = -1, tol
        for j, p in enumerate(pred):
            if not used[j] and abs(p - r) <= best_d:
                best, best_d = j, abs(p - r)
        if best >= 0:
            used[best] = True
            hits += 1
    return hits, len(pred), len(ref)


def boundary_metrics(probs: list[np.ndarray], rows: list[dict]) -> dict:
    best = {tol: {"f1": 0.0, "precision": 0.0, "recall": 0.0, "threshold": 0.0}
            for tol in (3.0, 5.0)}
    for tol in (3.0, 5.0):
        best_f1 = 0.0
        for threshold in np.arange(0.1, 0.95, 0.05):
            h = p_n = r_n = 0
            for prob, row in zip(probs, rows):
                pred = pick_peaks(prob, threshold)
                hits, np_, nr = hit_rate_f1(pred, row["boundaries_sec"], tol)
                h += hits; p_n += np_; r_n += nr
            precision = h / max(p_n, 1)
            recall = h / max(r_n, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-9)
            if f1 > best_f1:
                best_f1 = f1
                best[tol] = {"f1": f1, "precision": precision,
                             "recall": recall, "threshold": float(threshold)}
    return {str(int(tol)): best[tol] for tol in (3.0, 5.0)}


@torch.no_grad()
def collect(model, loader, device) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Per-piece (boundary prob [T], projected frames [T, proj_dim])."""
    model.eval()
    probs, projs = [], []
    for feats, _, mask, _, _ in loader:
        logits, proj = model(feats.to(device), mask.to(device))
        p = torch.sigmoid(logits).cpu().numpy()
        pr = proj.cpu().numpy()
        for i in range(len(feats)):
            n = int(mask[i].sum())
            probs.append(p[i, :n])
            projs.append(pr[i, :n])
    return probs, projs


def _pool(proj: np.ndarray, spans: list[tuple[int, int]]) -> np.ndarray:
    emb = np.stack([proj[f0:f1].mean(0) for f0, f1 in spans])
    return emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9)


def pairwise_teacher_eval(projs, rows, scale: float, bias: float) -> dict:
    """(a) same/different accuracy on teacher spans."""
    tp = fp = fn = tn = 0
    skipped = 0
    for proj, row in zip(projs, rows):
        spans, letters = segment_frames(row, proj.shape[0])
        if len(spans) < 2:
            skipped += 1
            continue
        emb = _pool(proj, spans)
        prob = 1 / (1 + np.exp(-(scale * (emb @ emb.T) + bias)))
        arr = np.array(letters)
        same = arr[:, None] == arr[None, :]
        iu = np.triu_indices(len(spans), k=1)
        pred = prob[iu] > 0.5
        ref = same[iu]
        tp += int((pred & ref).sum()); fp += int((pred & ~ref).sum())
        fn += int((~pred & ref).sum()); tn += int((~pred & ~ref).sum())
    total = tp + fp + fn + tn
    return {
        "n_pairs": total,
        "accuracy": (tp + tn) / max(total, 1),
        "same_precision": tp / max(tp + fp, 1),
        "same_recall": tp / max(tp + fn, 1),
        "pieces_skipped_lt2_segments": skipped,
    }


def agglomerate(emb: np.ndarray, scale: float, bias: float) -> list[int]:
    """Merge highest mean p(same) pair while > 0.5 or above the cluster cap."""
    prob = 1 / (1 + np.exp(-(scale * (emb @ emb.T) + bias)))
    clusters = [[i] for i in range(len(emb))]
    while len(clusters) > 1:
        best, best_pair = -1.0, None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                m = prob[np.ix_(clusters[a], clusters[b])].mean()
                if m > best:
                    best, best_pair = m, (a, b)
        if best <= 0.5 and len(clusters) <= MAX_CLUSTERS:
            break
        a, b = best_pair
        clusters[a] = clusters[a] + clusters[b]
        clusters.pop(b)
    assign = [0] * len(emb)
    for c, members in enumerate(clusters):
        for m in members:
            assign[m] = c
    # relabel by order of first appearance
    order: dict[int, int] = {}
    for i in range(len(assign)):
        order.setdefault(assign[i], len(order))
    return [order[a] for a in assign]


def _cluster_at_times(spans_sec, assign, times):
    """Cluster id per sampled time (-1 if outside every span)."""
    out = np.full(len(times), -1, dtype=np.int64)
    for (s, e), c in zip(spans_sec, assign):
        out[(times >= s) & (times < e)] = c
    return out


def end_to_end_eval(probs, projs, rows, threshold: float,
                    scale: float, bias: float) -> dict:
    pred_counts: dict[int, int] = {}
    teacher_counts: dict[int, int] = {}
    agree = total = 0
    for prob, proj, row in zip(probs, projs, rows):
        n = proj.shape[0]
        duration = n * HOP_SEC
        bounds = pick_peaks(prob, threshold)
        edges = [0.0] + [b for b in bounds if 0 < b < duration] + [duration]
        spans_sec = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
        spans = []
        kept = []
        for s, e in spans_sec:
            f0 = min(max(int(s / HOP_SEC), 0), n)
            f1 = min(max(int(math.ceil(e / HOP_SEC)), 0), n)
            if f1 > f0:
                spans.append((f0, f1))
                kept.append((s, e))
        emb = _pool(proj, spans)
        assign = agglomerate(emb, scale, bias)
        k = len(set(assign))
        pred_counts[k] = pred_counts.get(k, 0) + 1

        t_spans, t_letters = segment_frames(row, n)
        t_sec = [(row["section_starts_sec"][i], row["section_ends_sec"][i])
                 for i in range(len(row["section_letters"]))]
        # keep teacher spans aligned with the ones segment_frames retained
        t_sec = [(s, e) for s, e in t_sec
                 if min(max(int(math.ceil(e / HOP_SEC)), 0), n)
                 > min(max(int(s / HOP_SEC), 0), n)]
        letter_ids = {l: i for i, l in enumerate(dict.fromkeys(t_letters))}
        t_assign = [letter_ids[l] for l in t_letters]
        tk = len(set(t_assign))
        teacher_counts[tk] = teacher_counts.get(tk, 0) + 1

        times = np.arange(0.0, duration, 2.0)
        pc = _cluster_at_times(kept, assign, times)
        tc = _cluster_at_times(t_sec, t_assign, times)
        valid = (pc >= 0) & (tc >= 0)
        idx = np.nonzero(valid)[0]
        if len(idx) < 2:
            continue
        p_same = pc[idx][:, None] == pc[idx][None, :]
        t_same = tc[idx][:, None] == tc[idx][None, :]
        iu = np.triu_indices(len(idx), k=1)
        agree += int((p_same[iu] == t_same[iu]).sum())
        total += len(iu[0])
    return {
        "threshold": threshold,
        "pred_cluster_counts": {str(k): pred_counts[k]
                                for k in sorted(pred_counts)},
        "teacher_cluster_counts": {str(k): teacher_counts[k]
                                   for k in sorted(teacher_counts)},
        "frame_pair_agreement": agree / max(total, 1),
        "n_frame_pairs": total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    device = torch.device("cuda")
    rows = pq.read_table(ROOT / "manifest.parquet").to_pylist()
    keep = {r["piece_id"] for r in rows}
    frames: dict[str, np.ndarray] = {}
    for shard_path in sorted((ROOT / "frames").glob("shard_*.npz")):
        with np.load(shard_path) as z:
            for key in z.files:
                if key in keep:
                    frames[key] = z[key]
    print(f"{len(rows)} pieces, {len(frames)} feature arrays loaded")

    rng = np.random.default_rng(SPLIT_SEED)
    order = rng.permutation(len(rows))
    n_val = int(len(rows) * VAL_FRACTION)
    val_rows = [rows[i] for i in order[:n_val]]
    train_rows = [rows[i] for i in order[n_val:]]
    train_rows = train_rows + [r for r in train_rows
                               if r["source"] == "high_quality"] * 2
    val_curated = [r for r in val_rows if r["source"] == "curated"]
    val_hq = [r for r in val_rows if r["source"] == "high_quality"]
    print(f"train {len(train_rows)}, val {len(val_rows)} "
          f"(curated {len(val_curated)}, high_quality {len(val_hq)})")

    train_loader = DataLoader(
        PieceDataset(train_rows, frames), batch_size=args.batch_size,
        shuffle=True, collate_fn=collate, num_workers=4)
    val_loader = DataLoader(
        PieceDataset(val_rows, frames), batch_size=args.batch_size,
        shuffle=False, collate_fn=collate, num_workers=2)

    model = StructureProbe().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params / 1e6:.2f}M")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=0.01)
    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=0.05)

    run_dir = RUN_DIR
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({
        **vars(args), "sigma_sec": SIGMA_SEC, "hop_sec": HOP_SEC,
        "split_seed": SPLIT_SEED, "val_fraction": VAL_FRACTION,
        "min_dist_frames": MIN_DIST_FRAMES, "max_clusters": MAX_CLUSTERS,
        "n_train": len(train_rows), "n_val": len(val_rows),
        "params": n_params}, indent=2))

    log = (run_dir / "train_log.txt").open("a")
    best_score = -1.0
    skipped_pairwise = 0
    for epoch in range(args.epochs):
        model.train()
        run_b = run_p = 0.0
        for step, (feats, target, mask, spans, letters) in enumerate(train_loader):
            feats, target, mask = (feats.to(device), target.to(device),
                                   mask.to(device))
            logits, proj = model(feats, mask)
            b_loss = nn.functional.binary_cross_entropy_with_logits(
                logits, target, reduction="none")
            b_loss = (b_loss * mask).sum() / mask.sum()
            pair_terms = []
            for i, (sp, lt) in enumerate(zip(spans, letters)):
                if len(sp) < 2:
                    skipped_pairwise += 1
                    continue
                emb = model.segment_embeddings(proj[i], sp)
                tgt = pairwise_targets(lt).to(device)
                term = piece_pair_loss(model.pair_logits(emb), tgt)
                if term is None:
                    skipped_pairwise += 1
                    continue
                pair_terms.append(term)
            p_loss = (torch.stack(pair_terms).mean() if pair_terms
                      else torch.zeros((), device=device))
            loss = b_loss + p_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            run_b += b_loss.item(); run_p += float(p_loss)
            if (step + 1) % 200 == 0:
                msg = (f"epoch {epoch} step {step + 1}/{len(train_loader)} "
                       f"bce {run_b / 200:.4f} pair {run_p / 200:.4f}")
                print(msg); log.write(msg + "\n"); log.flush()
                run_b = run_p = 0.0

        probs, projs = collect(model, val_loader, device)
        scale = float(model.scale.detach()); bias = float(model.bias.detach())
        bm = boundary_metrics(probs, val_rows)
        idx_hq = [i for i, r in enumerate(val_rows) if r["source"] == "high_quality"]
        pw = pairwise_teacher_eval([projs[i] for i in idx_hq],
                                   [val_rows[i] for i in idx_hq], scale, bias)
        msg = (f"epoch {epoch} F1@3s {bm['3']['f1']:.4f} F1@5s {bm['5']['f1']:.4f} "
               f"high_quality_pairwise_acc {pw['accuracy']:.4f} "
               f"scale {scale:.2f} bias {bias:.2f}")
        print(msg); log.write(msg + "\n"); log.flush()
        score = 0.5 * (bm["3"]["f1"] + pw["accuracy"])
        if score > best_score:
            best_score = score
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "boundary": bm, "pairwise_high_quality": pw,
                        "score": score}, run_dir / "best.pt")
    torch.save({"model": model.state_dict(), "epoch": args.epochs - 1},
               run_dir / "last.pt")

    model.load_state_dict(torch.load(run_dir / "best.pt")["model"])
    probs, projs = collect(model, val_loader, device)
    scale = float(model.scale.detach()); bias = float(model.bias.detach())
    eval_out = {"scale": scale, "bias": bias, "best_score": best_score,
                "skipped_pairwise_pieces_train": skipped_pairwise,
                "subsets": {}}
    for name in ("overall", "curated", "high_quality"):
        idx = [i for i, r in enumerate(val_rows)
               if name == "overall" or r["source"] == name]
        sp = [probs[i] for i in idx]; sj = [projs[i] for i in idx]
        sr = [val_rows[i] for i in idx]
        bm = boundary_metrics(sp, sr)
        eval_out["subsets"][name] = {
            "n_pieces": len(sr),
            "boundary": bm,
            "pairwise_teacher_spans": pairwise_teacher_eval(sj, sr, scale, bias),
            "end_to_end": end_to_end_eval(sp, sj, sr, bm["3"]["threshold"],
                                          scale, bias),
        }
    (run_dir / "eval.json").write_text(json.dumps(eval_out, indent=2))
    log.write("final eval " + json.dumps(eval_out) + "\n")
    log.close()
    print(json.dumps(eval_out, indent=2))
    print(f"-> {run_dir}")


if __name__ == "__main__":
    main()
