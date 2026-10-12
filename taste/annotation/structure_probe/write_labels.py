"""Write structure-probe labels into the curated and high-quality datasets.

Curated: structure.json/.lab are archived once as meta_archive/structure_section_fusion.*
and replaced by the probe's lettered sections (a single level-4 entry).
High quality: section letters come from the generation prompt and are kept; each internal
boundary time of `section_timing` is snapped to a nearby probe peak when the result stays
monotonic and well spaced. The pre-snap block is archived once as `section_timing_initial`,
and `section_sequence_timed` is updated to the snapped starts.

    python -m taste.annotation.structure_probe.write_labels --source curated --apply
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from taste import paths
from taste.annotation.structure_probe.infer import (
    CKPT, PRODUCTION_CONFIG, sections_for_piece)
from taste.annotation.structure_probe.train import (
    FEATURE_DIM, ROOT, StructureProbe, pick_peaks)

CURATED_ROOT = paths.data("curated", "pieces")
HIGH_QUALITY_META = paths.data("high_quality", "metadata")

SOURCE_THRESHOLD = {"curated": 0.15, "high_quality": 0.30}
# peak-picking distance for the high-quality snap; curated sections use
# PRODUCTION_CONFIG["min_dist_frames"] (40)
HIGH_QUALITY_MIN_DIST_FRAMES = 24
SNAP_WINDOW_SEC = 4.0
MIN_BOUNDARY_GAP_SEC = 12.0
METHOD_CURATED = "structure_probe"
METHOD_HIGH_QUALITY = "constrained_clustering+structure_probe_snap"
ARCHIVE_DIR = "meta_archive"
ARCHIVE_JSON = "structure_section_fusion.json"
ARCHIVE_LAB = "structure_section_fusion.lab"
ARCHIVE_KEY = "section_timing_initial"


def provenance(source: str) -> dict:
    return {"method": METHOD_CURATED,
            "checkpoint": str(CKPT.relative_to(paths.CKPT_ROOT)),
            "config": {**PRODUCTION_CONFIG,
                       "boundary_threshold": SOURCE_THRESHOLD[source]}}


def build_structure_doc(letters: list[str], starts: list[float],
                        ends: list[float], audio_duration_sec: float) -> dict:
    structure = " ".join(letters)
    sections = [
        {"start_sec": s, "end_sec": e, "label": l,
         "native_cluster_id": str(ord(l) - ord("A"))}
        for l, s, e in zip(letters, starts, ends)
    ]
    return {
        **provenance("curated"),
        "audio_duration_sec": audio_duration_sec,
        "recommended_level": 4,
        "recommended_structure": structure,
        "levels": [{
            "level": 4,
            "section_count": len(sections),
            "unique_label_count": len(set(letters)),
            "structure": structure,
            "sections": sections,
        }],
    }


def lab_text(letters: list[str], starts: list[float], ends: list[float]) -> str:
    return "".join(f"{s:.3f}\t{e:.3f}\t{l}\n"
                   for l, s, e in zip(letters, starts, ends))


def snap_boundaries(starts: list[float], duration: float,
                    peaks: list[float]) -> tuple[list[float], list[float]]:
    """Snap internal boundaries to nearby peaks; returns (new starts, shifts)."""
    used = [False] * len(peaks)
    new = list(starts)
    shifts = []
    for i in range(1, len(starts)):
        best, best_d = -1, SNAP_WINDOW_SEC
        for j, p in enumerate(peaks):
            if not used[j] and abs(p - starts[i]) <= best_d:
                best, best_d = j, abs(p - starts[i])
        if best < 0:
            continue
        candidate = peaks[best]
        prev = new[i - 1]
        nxt = starts[i + 1] if i + 1 < len(starts) else duration
        if candidate - prev < MIN_BOUNDARY_GAP_SEC:
            continue
        if nxt - candidate < MIN_BOUNDARY_GAP_SEC:
            continue
        if duration - candidate < PRODUCTION_CONFIG["final_section_absorb_sec"]:
            continue
        used[best] = True
        shifts.append(abs(candidate - starts[i]))
        new[i] = candidate
    return new, shifts


@torch.no_grad()
def run_batch(model, rows, frames, device):
    n = max(r["n_frames"] for r in rows)
    feats = torch.zeros(len(rows), n, FEATURE_DIM)
    mask = torch.zeros(len(rows), n, dtype=torch.bool)
    for j, r in enumerate(rows):
        f = torch.from_numpy(frames[r["piece_id"]])
        feats[j, :f.shape[0]] = f
        mask[j, :f.shape[0]] = True
    logits, proj = model(feats.to(device), mask.to(device))
    return (torch.sigmoid(logits).float().cpu().numpy(),
            proj.float().cpu().numpy())


def write_curated(row, prob, proj, scale, bias, out_root: Path | None) -> dict:
    piece_id = row["piece_id"]
    src_dir = CURATED_ROOT / piece_id
    dst_dir = src_dir if out_root is None else out_root / piece_id
    if out_root is not None:
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_dir / "structure.json", dst_dir / "structure.json")
        shutil.copy2(src_dir / "structure.lab", dst_dir / "structure.lab")

    original = json.loads((dst_dir / "structure.json").read_text())
    # archive once: a second run must never overwrite the section-fusion original
    archive_dir = dst_dir / ARCHIVE_DIR
    if not (archive_dir / ARCHIVE_JSON).exists():
        archive_dir.mkdir(exist_ok=True)
        shutil.copy2(dst_dir / "structure.json", archive_dir / ARCHIVE_JSON)
        shutil.copy2(dst_dir / "structure.lab", archive_dir / ARCHIVE_LAB)

    letters, starts, ends, fired, absorbed = sections_for_piece(
        prob, proj, float(row["duration_sec"]), scale, bias,
        SOURCE_THRESHOLD["curated"])
    doc = build_structure_doc(letters, starts, ends,
                             float(original["audio_duration_sec"]))
    (dst_dir / "structure.json").write_text(
        json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (dst_dir / "structure.lab").write_text(lab_text(letters, starts, ends),
                                           encoding="utf-8")
    return {"piece_id": piece_id, "n_sections": len(letters),
            "unique_letters": len(set(letters)), "merge_fired": fired,
            "absorbed": absorbed}


def write_high_quality(row, prob, out_root: Path | None) -> dict:
    piece_id = row["piece_id"]
    src = HIGH_QUALITY_META / f"{piece_id}.json"
    dst = src if out_root is None else out_root / f"{piece_id}.json"
    if out_root is not None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    meta = json.loads(dst.read_text(encoding="utf-8"))
    timing = meta["section_timing"]
    if ARCHIVE_KEY not in meta:
        meta[ARCHIVE_KEY] = json.loads(json.dumps(timing))

    sections = timing["sections"]
    duration = float(timing["duration_sec"])
    starts = [float(s["start_sec"]) for s in sections]
    peaks = pick_peaks(prob, SOURCE_THRESHOLD["high_quality"],
                       HIGH_QUALITY_MIN_DIST_FRAMES)
    new_starts, shifts = snap_boundaries(starts, duration, peaks)

    for i, section in enumerate(sections):
        section["start_sec"] = new_starts[i]
        section["end_sec"] = (new_starts[i + 1] if i + 1 < len(new_starts)
                              else duration)
    timing["method"] = METHOD_HIGH_QUALITY
    timing["probe_snap"] = {
        "checkpoint": str(CKPT.relative_to(paths.CKPT_ROOT)),
        "boundary_threshold": SOURCE_THRESHOLD["high_quality"],
        "min_dist_frames": HIGH_QUALITY_MIN_DIST_FRAMES,
        "snap_window_sec": SNAP_WINDOW_SEC,
        "min_boundary_gap_sec": MIN_BOUNDARY_GAP_SEC,
        "n_boundaries": len(starts) - 1, "n_snapped": len(shifts),
    }
    for entry, start in zip(meta["section_sequence_timed"], new_starts):
        entry["start_sec"] = start
    dst.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    return {"piece_id": piece_id, "n_boundaries": len(starts) - 1,
            "n_snapped": len(shifts), "shifts": shifts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("curated", "high_quality"),
                        required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--pilot-dir", default=None,
                        help="write copies here instead of the corpus")
    parser.add_argument("--apply", action="store_true",
                        help="required to write in place")
    args = parser.parse_args()

    if args.pilot_dir is None and not args.apply:
        raise SystemExit("refusing to write in place without --apply")
    out_root = Path(args.pilot_dir) if args.pilot_dir else None

    manifest = pq.read_table(
        ROOT / "manifest.parquet",
        columns=["piece_id", "source", "shard", "duration_sec", "n_frames"]
    ).to_pylist()
    rows = [r for r in manifest if r["source"] == args.source]
    by_shard: dict[int, list[dict]] = {}
    for r in rows:
        by_shard.setdefault(int(r["shard"]), []).append(r)
    print(f"{len(rows)} {args.source} pieces over {len(by_shard)} shards "
          f"-> {out_root or 'IN PLACE'}")

    device = torch.device("cuda")
    model = StructureProbe().to(device)
    model.load_state_dict(torch.load(CKPT, map_location="cpu")["model"])
    model.eval()
    scale = float(model.scale.detach())
    bias = float(model.bias.detach())

    out: list[dict] = []
    for k, shard in enumerate(sorted(by_shard)):
        keep = {r["piece_id"] for r in by_shard[shard]}
        with np.load(ROOT / "frames" / f"shard_{shard:03d}.npz") as z:
            frames = {key: z[key] for key in z.files if key in keep}
        chunkable = sorted(by_shard[shard], key=lambda r: r["n_frames"],
                           reverse=True)
        for i in range(0, len(chunkable), args.batch_size):
            chunk = chunkable[i:i + args.batch_size]
            probs, projs = run_batch(model, chunk, frames, device)
            for j, r in enumerate(chunk):
                t = r["n_frames"]
                if args.source == "curated":
                    out.append(write_curated(r, probs[j, :t], projs[j, :t],
                                             scale, bias, out_root))
                else:
                    out.append(write_high_quality(r, probs[j, :t], out_root))
        del frames
        print(f"  shard {shard:03d} ({k + 1}/{len(by_shard)}): "
              f"{len(out)} written", flush=True)

    if args.source == "curated":
        counts = [r["n_sections"] for r in out]
        uniq = [r["unique_letters"] for r in out]
        stats = {
            "n_pieces": len(out),
            "n_sections_p5_p50_p95": [float(np.percentile(counts, p))
                                      for p in (5, 50, 95)],
            "n_sections_mean": float(np.mean(counts)),
            "unique_letters_p5_p50_p95": [float(np.percentile(uniq, p))
                                          for p in (5, 50, 95)],
            "pieces_with_adjacent_merge": sum(r["merge_fired"] for r in out),
            "final_sections_absorbed": sum(r["absorbed"] for r in out),
        }
    else:
        all_shifts = [s for r in out for s in r["shifts"]]
        n_bounds = sum(r["n_boundaries"] for r in out)
        stats = {
            "n_pieces": len(out),
            "n_boundaries": n_bounds,
            "n_snapped": len(all_shifts),
            "snapped_fraction": len(all_shifts) / max(n_bounds, 1),
            "mean_abs_shift_sec": float(np.mean(all_shifts)) if all_shifts else 0.0,
            "max_abs_shift_sec": float(max(all_shifts)) if all_shifts else 0.0,
        }
    print(json.dumps(stats, indent=2))
    report = ROOT / f"write_{args.source}_stats{'_pilot' if out_root else ''}.json"
    report.write_text(json.dumps(stats, indent=2))
    print(f"-> {report}")


if __name__ == "__main__":
    main()
