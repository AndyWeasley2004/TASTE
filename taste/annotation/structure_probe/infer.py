"""Predict section boundaries and letters from corpus frame features.

PRODUCTION_CONFIG defines boundary and merge thresholds. Distributed runs use
--gpu-shard K/N followed by --merge.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from taste import paths
from taste.annotation.structure_probe.train import (
    FEATURE_DIM, HOP_SEC, MAX_CLUSTERS, ROOT, StructureProbe, _pool, pick_peaks)

CKPT = paths.ckpt("probes", "structure_probe", "best.pt")
LETTERS = "ABCDEFGH"
MAX_BATCH_ATTN_ELEMS = 5e7
PRODUCTION_CONFIG = {
    "checkpoint": str(CKPT.relative_to(paths.CKPT_ROOT)),
    "hop_sec": HOP_SEC,
    "boundary_threshold": 0.15,
    "min_dist_frames": 40,
    "min_section_sec": 40 * HOP_SEC,
    "merge_tau": 0.65,
    "max_clusters": 8,
    "merge_adjacent_same_letter": True,
    # a final section shorter than one <T> window cannot manifest downstream
    "final_section_absorb_sec": 5.0,
    "reletter_by_first_appearance": True,
    "letters": LETTERS,
}

SOURCES = {
    "teacher": ("manifest.parquet", "frames"),
    "large_public": ("manifest_large_public.parquet", "large_public_frames"),
    "aria": ("manifest_aria.parquet", "aria_frames"),
    "large_public_full": ("manifest_large_public_full.parquet",
                          "large_public_full_frames"),
}
CORPUS = {
    "aria": str(paths.data("aria_midi", "pruned")),
    "large_public_full": str(paths.data("aria_midi", "deduped")),
}


def agglomerate(emb: np.ndarray, scale: float, bias: float,
                tau: float) -> list[int]:
    """train.agglomerate with the merge threshold exposed."""
    prob = 1 / (1 + np.exp(-(scale * (emb @ emb.T) + bias)))
    clusters = [[i] for i in range(len(emb))]
    while len(clusters) > 1:
        best, best_pair = -1.0, None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                m = prob[np.ix_(clusters[a], clusters[b])].mean()
                if m > best:
                    best, best_pair = m, (a, b)
        if best <= tau and len(clusters) <= MAX_CLUSTERS:
            break
        a, b = best_pair
        clusters[a] = clusters[a] + clusters[b]
        clusters.pop(b)
    assign = [0] * len(emb)
    for c, members in enumerate(clusters):
        for m in members:
            assign[m] = c
    order: dict[int, int] = {}
    for i in range(len(assign)):
        order.setdefault(assign[i], len(order))
    return [order[a] for a in assign]


def merge_adjacent(spans_sec: list[tuple[float, float]], assign: list[int]):
    """Collapse runs of consecutive segments sharing a letter."""
    out_spans: list[tuple[float, float]] = []
    out_assign: list[int] = []
    for (s, e), c in zip(spans_sec, assign):
        if out_assign and out_assign[-1] == c:
            out_spans[-1] = (out_spans[-1][0], e)
        else:
            out_spans.append((s, e))
            out_assign.append(c)
    return out_spans, out_assign


def absorb_short_final(spans: list[tuple[float, float]], assign: list[int]):
    """Fold a sub-<T>-window final section into its predecessor."""
    if len(spans) < 2:
        return spans, assign
    start, end = spans[-1]
    if end - start >= PRODUCTION_CONFIG["final_section_absorb_sec"]:
        return spans, assign
    spans = spans[:-1]
    assign = assign[:-1]
    spans[-1] = (spans[-1][0], end)
    return spans, assign


def sections_for_piece(prob: np.ndarray, proj: np.ndarray, duration: float,
                       scale: float, bias: float, threshold: float | None = None
                       ) -> tuple[list[str], list[float], list[float], bool, int]:
    n = proj.shape[0]
    if threshold is None:
        threshold = PRODUCTION_CONFIG["boundary_threshold"]
    bounds = pick_peaks(prob, threshold, PRODUCTION_CONFIG["min_dist_frames"])
    edges = [0.0] + [b for b in bounds if 0 < b < duration] + [duration]
    spans, kept = [], []
    for s, e in zip(edges[:-1], edges[1:]):
        f0 = min(max(int(s / HOP_SEC), 0), n)
        f1 = min(max(int(math.ceil(e / HOP_SEC)), 0), n)
        if f1 > f0:
            spans.append((f0, f1))
            kept.append((s, e))
    emb = _pool(proj, spans)
    assign = agglomerate(emb, scale, bias, PRODUCTION_CONFIG["merge_tau"])
    merged_spans, merged_assign = merge_adjacent(kept, assign)
    fired = len(merged_spans) < len(kept)
    n_before = len(merged_spans)
    merged_spans, merged_assign = absorb_short_final(merged_spans, merged_assign)
    absorbed = n_before - len(merged_spans)
    # re-letter so letters stay contiguous from A after the adjacent merge
    order: dict[int, int] = {}
    for c in merged_assign:
        order.setdefault(c, len(order))
    letters = [LETTERS[order[c]] for c in merged_assign]
    starts = [float(s) for s, _ in merged_spans]
    ends = [float(e) for _, e in merged_spans]
    return letters, starts, ends, fired, absorbed


@torch.no_grad()
def run_shard(model, rows: list[dict], frames: dict[str, np.ndarray],
              device, batch_size: int, scale: float, bias: float) -> list[dict]:
    rows = sorted(rows, key=lambda r: r["n_frames"], reverse=True)
    out = []
    i = 0
    while i < len(rows):
        # encoder attention is O(batch * n^2); hour-long pieces reach 7,200
        # frames, so --batch-size is a ceiling and long rows batch smaller
        size = max(1, min(batch_size,
                          int(MAX_BATCH_ATTN_ELEMS / rows[i]["n_frames"] ** 2)))
        chunk = rows[i:i + size]
        i += size
        n = max(r["n_frames"] for r in chunk)
        feats = torch.zeros(len(chunk), n, FEATURE_DIM)
        mask = torch.zeros(len(chunk), n, dtype=torch.bool)
        for j, r in enumerate(chunk):
            f = torch.from_numpy(frames[r["piece_id"]])
            feats[j, :f.shape[0]] = f
            mask[j, :f.shape[0]] = True
        logits, proj = model(feats.to(device), mask.to(device))
        probs = torch.sigmoid(logits).float().cpu().numpy()
        projs = proj.float().cpu().numpy()
        for j, r in enumerate(chunk):
            t = r["n_frames"]
            prob = probs[j, :t]
            letters, starts, ends, fired, absorbed = sections_for_piece(
                prob, projs[j, :t], float(r["duration_sec"]), scale, bias)
            out.append({
                "piece_id": r["piece_id"],
                "duration_sec": float(r["duration_sec"]),
                "n_frames": int(t),
                "n_sections": len(letters),
                "section_letters": letters,
                "section_starts_sec": starts,
                "section_ends_sec": ends,
                "max_boundary_prob": float(prob.max()),
                "_merge_fired": fired,
                "_absorbed": absorbed,
            })
    return out


def build_stats(rows: list[dict], merge_fired: int, absorbed: int) -> dict:
    hist: dict[int, int] = {}
    durations = []
    for r in rows:
        hist[r["n_sections"]] = hist.get(r["n_sections"], 0) + 1
        durations += [e - s for s, e in zip(r["section_starts_sec"],
                                            r["section_ends_sec"])]
    d = np.array(durations)
    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    return {
        "n_pieces": len(rows),
        "n_sections_histogram": {str(k): hist[k] for k in sorted(hist)},
        "n_sections_mean": float(np.mean([r["n_sections"] for r in rows])),
        "n_sections_total": int(len(durations)),
        "section_duration_percentiles_sec": {
            f"p{p}": float(np.percentile(d, p)) for p in pcts},
        "section_duration_min_sec": float(d.min()),
        "section_duration_max_sec": float(d.max()),
        "sections_under_12s": int((d < 11.999).sum()),
        "pieces_with_adjacent_merge": merge_fired,
        "final_sections_absorbed": absorbed,
        "final_section_duration_min_sec": float(
            min(r["section_ends_sec"][-1] - r["section_starts_sec"][-1]
                for r in rows)),
    }


def write_outputs(out_dir: Path, rows: list[dict], source: str,
                  manifest_name: str, frames_dir: str, batch_size: int,
                  scale: float, bias: float) -> None:
    merge_fired = sum(r.pop("_merge_fired") for r in rows)
    absorbed = sum(r.pop("_absorbed") for r in rows)
    pq.write_table(pa.Table.from_pylist(rows), out_dir / "labels.parquet")
    (out_dir / "config.json").write_text(json.dumps({
        **PRODUCTION_CONFIG,
        "source": source,
        "source_counts": {source: len(rows)},
        "corpus": CORPUS.get(source),
        "manifest": str(ROOT / manifest_name),
        "frames_dir": str(ROOT / frames_dir),
        "batch_size": batch_size,
        "scale": scale,
        "bias": bias,
        "date": datetime.date.today().isoformat(),
    }, indent=2))
    stats = build_stats(rows, merge_fired, absorbed)
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"-> {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=tuple(SOURCES), default="large_public")
    parser.add_argument("--out-dir", default="large_public_structure_labels",
                        help="relative to the structure-probe data dir, or an "
                             "absolute path; the large_public_full labels live "
                             "at aria_midi/deduped/structure_labels")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gpu-shard", default=None,
                        help="K/N: take only shards with shard %% N == K and "
                             "write labels_part{K}.parquet")
    parser.add_argument("--merge", action="store_true",
                        help="combine the labels_part*.parquet in --out-dir "
                             "into the final label set")
    args = parser.parse_args()

    manifest_name, frames_dir = SOURCES[args.source]
    out_dir = ROOT / args.out_dir
    if args.merge:
        parts = sorted(out_dir.glob("labels_part*.parquet"))
        rows = pa.concat_tables([pq.read_table(p) for p in parts]).to_pylist()
        state = torch.load(CKPT, map_location="cpu")["model"]
        print(f"merging {len(parts)} parts, {len(rows)} pieces")
        write_outputs(out_dir, rows, args.source, manifest_name, frames_dir,
                      args.batch_size, float(state["scale"]),
                      float(state["bias"]))
        for p in parts:
            p.unlink()
        return

    rows = pq.read_table(ROOT / manifest_name,
                         columns=["piece_id", "shard", "duration_sec",
                                  "n_frames"]).to_pylist()
    by_shard: dict[int, list[dict]] = {}
    for r in rows:
        by_shard.setdefault(int(r["shard"]), []).append(r)
    if args.gpu_shard:
        k, n = (int(x) for x in args.gpu_shard.split("/"))
        by_shard = {s: v for s, v in by_shard.items() if s % n == k}
    print(f"{sum(len(v) for v in by_shard.values())} pieces over "
          f"{len(by_shard)} shards ({args.source})")

    device = torch.device("cuda")
    model = StructureProbe().to(device)
    model.load_state_dict(torch.load(CKPT, map_location="cpu")["model"])
    model.eval()
    scale = float(model.scale.detach())
    bias = float(model.bias.detach())
    print(f"scale {scale:.4f} bias {bias:.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for k, shard in enumerate(sorted(by_shard)):
        path = ROOT / frames_dir / f"shard_{shard:03d}.npz"
        keep = {r["piece_id"] for r in by_shard[shard]}
        with np.load(path) as z:
            frames = {key: z[key] for key in z.files if key in keep}
        shard_rows = run_shard(model, by_shard[shard], frames, device,
                               args.batch_size, scale, bias)
        del frames
        results += shard_rows
        print(f"  shard {shard:03d} ({k + 1}/{len(by_shard)}): "
              f"{len(shard_rows)} pieces, {len(results)} total", flush=True)

    if args.gpu_shard:
        part = args.gpu_shard.split("/")[0]
        pq.write_table(pa.Table.from_pylist(results),
                       out_dir / f"labels_part{part}.parquet")
        print(f"-> {out_dir / f'labels_part{part}.parquet'}")
        return

    write_outputs(out_dir, results, args.source, manifest_name, frames_dir,
                  args.batch_size, scale, bias)


if __name__ == "__main__":
    main()
