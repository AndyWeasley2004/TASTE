"""Extract 26-d symbolic frames at 0.5 s hop, plus optional teacher targets.

Teacher output uses manifest.parquet and frames/; other sources use
manifest_<source>.parquet and <source>_frames/. Full Aria sweeps record skipped
long pieces and onset tempos; the pruned sweep also writes tempo sidecars.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import multiprocessing as mp
import time
from collections import defaultdict
from pathlib import Path

import mido
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from taste import paths
from taste.representation.midi_features import estimate_tempo_from_onsets, tempo_class

HOP_SEC = 0.5
N_SHARDS = 64
N_SHARDS_LARGE_PUBLIC = 128
N_SHARDS_ARIA = 256
N_SHARDS_LARGE_PUBLIC_FULL = 256
MAX_DURATION_SEC = 3600.0
CURATED_ROOT = paths.data("curated", "pieces")
HIGH_QUALITY_ROOT = paths.data("high_quality")
DEDUPED_ROOT = paths.data("aria_midi", "deduped")
# the pretrain corpus is the pruned release; its piece ids are file stems that
# collide with deduped ids while carrying different content
PRETRAIN_ROOT = paths.data("aria_midi", "pruned")
LARGE_PUBLIC_LABELS = DEDUPED_ROOT / "large_public_labels" / "labels.parquet"
OUT_ROOT = paths.data("structure_probe")

FEATURE_DIM = 26


def read_notes_and_pedal(path: Path) -> tuple[list[dict], list[tuple[float, int]]]:
    """Notes in performed seconds plus (time, value) sustain-pedal events."""
    midi = mido.MidiFile(path)
    tempo = 500_000
    elapsed = 0.0
    active: dict[tuple[int, int], list[tuple[float, int]]] = defaultdict(list)
    notes: list[dict] = []
    pedal: list[tuple[float, int]] = []
    for message in mido.merge_tracks(midi.tracks):
        elapsed += mido.tick2second(message.time, midi.ticks_per_beat, tempo)
        if message.type == "set_tempo":
            tempo = message.tempo
        elif message.type == "control_change" and message.control == 64:
            pedal.append((elapsed, message.value))
        elif message.type == "note_on" and message.velocity > 0:
            active[(message.channel, message.note)].append((elapsed, message.velocity))
        elif message.type in {"note_off", "note_on"}:
            key = (message.channel, message.note)
            if not active[key]:
                continue
            onset, velocity = active[key].pop(0)
            if elapsed > onset:
                notes.append({"onset": onset, "offset": elapsed,
                              "pitch": message.note, "velocity": velocity})
    notes.sort(key=lambda n: (n["onset"], n["pitch"]))
    return notes, pedal


def _explode_intervals(starts: np.ndarray, ends: np.ndarray, n_frames: int
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Frame index, overlap seconds and source index per (interval, frame) pair.

    Flattens the per-interval frame ranges into one vector so the per-frame
    accumulation below is a single bincount.
    """
    first = np.floor(starts / HOP_SEC).astype(np.int64)
    last = np.minimum(np.ceil(ends / HOP_SEC).astype(np.int64), n_frames)
    counts = np.maximum(last - first, 0)
    total = int(counts.sum())
    src = np.repeat(np.arange(len(counts)), counts)
    offset = np.repeat(np.cumsum(counts) - counts, counts)
    frame = np.repeat(first, counts) + (np.arange(total) - offset)
    lo = frame * HOP_SEC
    overlap = (np.minimum(ends[src], lo + HOP_SEC)
               - np.maximum(starts[src], lo))
    keep = overlap > 0
    return frame[keep], overlap[keep], src[keep]


def _group_minmax(frame: np.ndarray, value: np.ndarray, n_frames: int
                  ) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(frame, kind="stable")
    fs, vs = frame[order], value[order].astype(np.float64)
    uniq, start = np.unique(fs, return_index=True)
    vmin = np.full(n_frames, 128.0)
    vmax = np.full(n_frames, -1.0)
    vmin[uniq] = np.minimum.reduceat(vs, start)
    vmax[uniq] = np.maximum.reduceat(vs, start)
    return vmin, vmax


def frame_features(notes: list[dict], pedal: list[tuple[float, int]],
                   duration: float) -> np.ndarray:
    n_frames = max(1, int(np.ceil(duration / HOP_SEC)))
    feats = np.zeros((n_frames, FEATURE_DIM), dtype=np.float32)

    onset = np.fromiter((n["onset"] for n in notes), np.float64, len(notes))
    offset = np.fromiter((n["offset"] for n in notes), np.float64, len(notes))
    pitch = np.fromiter((n["pitch"] for n in notes), np.int64, len(notes))
    velocity = np.fromiter((n["velocity"] for n in notes), np.float64, len(notes))
    onset_frame = np.clip((onset / HOP_SEC).astype(np.int64), 0, n_frames - 1)

    # sustain-pedal down fraction per frame (piecewise-constant CC64 >= 64)
    pedal_frac = np.zeros(n_frames)
    if pedal:
        times = np.array([0.0] + [t for t, _ in pedal] + [duration])
        values = np.array([0] + [v for _, v in pedal])
        down = (values >= 64) & (times[1:] > times[:-1])
        pf, pov, _ = _explode_intervals(times[:-1][down], times[1:][down],
                                        n_frames)
        pedal_frac = np.bincount(pf, weights=pov, minlength=n_frames) / HOP_SEC

    # sounding-time aggregates: total sounding time within frame / hop
    nf, nov, nsrc = _explode_intervals(onset, offset, n_frames)
    weight = np.bincount(nf, weights=nov, minlength=n_frames)
    rep_pitch = pitch[nsrc].astype(np.float64)
    pitch_sum = np.bincount(nf, weights=nov * rep_pitch, minlength=n_frames)
    pitch_sq = np.bincount(nf, weights=nov * rep_pitch ** 2, minlength=n_frames)
    pitch_min, pitch_max = _group_minmax(nf, rep_pitch, n_frames)
    poly = weight / HOP_SEC
    sounding = weight > 0
    p_mean = np.where(sounding, pitch_sum / np.maximum(weight, 1e-9), 0.0)
    p_var = np.where(sounding,
                     pitch_sq / np.maximum(weight, 1e-9) - p_mean ** 2, 0.0)
    p_std = np.sqrt(np.maximum(p_var, 0.0))

    # per-frame onset aggregates
    count = np.bincount(onset_frame, minlength=n_frames)
    onsets_here = count > 0
    denom = np.maximum(count, 1)
    v_sum = np.bincount(onset_frame, weights=velocity, minlength=n_frames)
    v_sq = np.bincount(onset_frame, weights=velocity ** 2, minlength=n_frames)
    v_mean = v_sum / denom
    v_std = np.sqrt(np.maximum(v_sq / denom - v_mean ** 2, 0.0))
    _, v_max = _group_minmax(onset_frame, velocity, n_frames)
    low = np.bincount(onset_frame, weights=(pitch < 48).astype(np.float64),
                      minlength=n_frames) / denom
    high = np.bincount(onset_frame, weights=(pitch > 72).astype(np.float64),
                       minlength=n_frames) / denom
    chroma = np.bincount(onset_frame * 12 + pitch % 12,
                         minlength=n_frames * 12).reshape(n_frames, 12)
    chroma = chroma / denom[:, None]

    feats[:, 0] = np.log1p(count)
    feats[:, 1] = poly
    feats[:, 2] = p_mean / 127.0
    feats[:, 3] = p_std / 127.0
    feats[:, 4] = np.where(sounding, pitch_min / 127.0, 0.0)
    feats[:, 5] = np.where(sounding, pitch_max / 127.0, 0.0)
    feats[:, 6] = np.where(onsets_here, v_mean / 127.0, 0.0)
    feats[:, 7] = np.where(onsets_here, v_max / 127.0, 0.0)
    feats[:, 8] = np.where(onsets_here, v_std / 127.0, 0.0)
    feats[:, 9] = np.where(onsets_here, low, 0.0)
    feats[:, 10] = np.where(onsets_here, high, 0.0)
    feats[:, 11] = pedal_frac
    feats[:, 12] = ~sounding
    feats[:, 13:25] = np.where(onsets_here[:, None], chroma, 0.0)
    feats[:, 25] = np.where(onsets_here, v_mean / 127.0 * poly, 0.0)
    return feats


def piece_tempo(notes: list[dict]) -> tuple[float | None, str | None]:
    """Onset-IOI tempo estimate; None for pieces too degenerate to estimate.

    `estimate_tempo_from_onsets` signals that condition with ValueError, and a
    corpus of 821K scraped MIDIs does contain such pieces.
    """
    onsets = np.fromiter((n["onset"] for n in notes), np.float64, len(notes))
    try:
        bpm = estimate_tempo_from_onsets(onsets)
    except ValueError:
        return None, None
    return float(bpm), tempo_class(bpm)




def curated_sections(piece_dir: Path) -> tuple[list[dict], float]:
    # structure.json holds the probe's own labels; the section-fusion teacher is the archive
    archive = piece_dir / "meta_archive" / "structure_section_fusion.json"
    if not archive.exists():
        raise FileNotFoundError(f"missing section-fusion teacher archive: {archive}")
    doc = json.loads(archive.read_text())
    level = next(l for l in doc["levels"] if l["level"] == doc["recommended_level"])
    return level["sections"], float(doc["audio_duration_sec"])


def high_quality_sections(meta_path: Path) -> tuple[list[dict], float]:
    # the teacher is the constrained-clustering timing before the probe snap
    doc = json.loads(meta_path.read_text())
    timing = doc["section_timing_initial"]
    sections = [{"start_sec": s["start_sec"], "end_sec": s["end_sec"],
                 "label": s["letter"]} for s in timing["sections"]]
    return sections, float(timing["duration_sec"])


def process(job: tuple[str, str, str]) -> dict | None:
    piece_id, source, midi_path = job
    if source in ("large_public", "aria", "large_public_full"):
        # no teacher structure: features only
        notes, pedal = read_notes_and_pedal(Path(midi_path))
        if not notes:
            return None
        duration = max(n["offset"] for n in notes)
        if source in ("aria", "large_public_full") and duration > MAX_DURATION_SEC:
            return {"piece_id": piece_id, "duration_sec": float(duration),
                    "too_long": True}
        feats = frame_features(notes, pedal, duration)
        row = {
            "piece_id": piece_id,
            "source": source,
            "duration_sec": float(duration),
            "n_frames": int(feats.shape[0]),
            "features": feats,
        }
        if source in ("aria", "large_public_full"):
            # tempo rides along on the parse already done
            root = PRETRAIN_ROOT if source == "aria" else DEDUPED_ROOT
            row["midi_relpath"] = str(Path(midi_path).relative_to(root))
            row["tempo_bpm"], row["tempo_class"] = piece_tempo(notes)
        return row
    if source == "curated":
        sections, teacher_dur = curated_sections(CURATED_ROOT / piece_id)
    else:
        sections, teacher_dur = high_quality_sections(
            HIGH_QUALITY_ROOT / "metadata" / f"{piece_id}.json")
    notes, pedal = read_notes_and_pedal(Path(midi_path))
    if not notes:
        return None
    duration = max(teacher_dur, max(n["offset"] for n in notes))
    feats = frame_features(notes, pedal, duration)
    boundaries = [float(s["start_sec"]) for s in sections[1:]]
    return {
        "piece_id": piece_id,
        "source": source,
        "duration_sec": float(duration),
        "n_frames": int(feats.shape[0]),
        "boundaries_sec": boundaries,
        "section_letters": [s["label"] for s in sections],
        "section_starts_sec": [float(s["start_sec"]) for s in sections],
        "section_ends_sec": [float(s["end_sec"]) for s in sections],
        "features": feats,
    }


def write_tempo_estimates(rows: list[dict]) -> None:
    """Sidecar tempo table for the pretrain corpus, next to its key estimates."""
    out_dir = PRETRAIN_ROOT / "tempo_estimates"
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(
        [{k: r[k] for k in ("piece_id", "midi_relpath", "tempo_bpm",
                            "tempo_class")} for r in rows],
        schema=pa.schema([("piece_id", pa.string()),
                          ("midi_relpath", pa.string()),
                          ("tempo_bpm", pa.float32()),
                          ("tempo_class", pa.string())]))
    pq.write_table(table, out_dir / "tempo.parquet")
    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        counts[str(r["tempo_class"])] += 1
    (out_dir / "manifest.json").write_text(json.dumps({
        "method": "onset_intervals",
        "implementation": "taste.representation.midi_features.estimate_tempo_from_onsets",
        "class_bounds_bpm": {"slow": "< 75", "moderate": "75-115", "fast": ">= 115"},
        "corpus": str(PRETRAIN_ROOT),
        "n_rows": len(rows),
        "tempo_class_counts": dict(sorted(counts.items())),
        "date": datetime.date.today().isoformat(),
    }, indent=2))
    print(f"tempo estimates -> {out_dir} ({dict(sorted(counts.items()))})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--source",
                        choices=("teacher", "large_public", "aria",
                                 "large_public_full"),
                        default="teacher",
                        help="teacher = curated + high_quality; large_public = "
                             "pieces listed in the large-public labels; aria = "
                             "every MIDI in the pruned release; "
                             "large_public_full = every MIDI in the deduped "
                             "release")
    args = parser.parse_args()

    jobs: list[tuple[str, str, str]] = []
    if args.source == "aria":
        midi_paths = sorted((PRETRAIN_ROOT / "data").rglob("*.mid"))
        jobs += [(p.stem, "aria", str(p)) for p in midi_paths]
        print(f"{len(jobs)} pieces (aria)")
        n_shards = N_SHARDS_ARIA
        frames_dir, manifest_name = "aria_frames", "manifest_aria"
    elif args.source == "large_public_full":
        midi_paths = sorted((DEDUPED_ROOT / "data").rglob("*.mid"))
        jobs += [(p.stem, "large_public_full", str(p)) for p in midi_paths]
        print(f"{len(jobs)} pieces (large_public_full)")
        n_shards = N_SHARDS_LARGE_PUBLIC_FULL
        frames_dir, manifest_name = ("large_public_full_frames",
                                     "manifest_large_public_full")
    elif args.source == "large_public":
        table = pq.read_table(LARGE_PUBLIC_LABELS,
                              columns=["piece_id", "midi_relpath"])
        jobs += [(r["piece_id"], "large_public",
                  str(DEDUPED_ROOT / r["midi_relpath"]))
                 for r in table.to_pylist()]
        print(f"{len(jobs)} pieces (large_public)")
        n_shards = N_SHARDS_LARGE_PUBLIC
        frames_dir, manifest_name = "large_public_frames", "manifest_large_public"
    else:
        curated_ids = sorted(p.name for p in CURATED_ROOT.iterdir())
        hq_ids = sorted(p.stem for p in (HIGH_QUALITY_ROOT / "midi").glob("*.mid"))
        jobs += [(pid, "curated", str(CURATED_ROOT / pid / "performance.mid"))
                 for pid in curated_ids]
        jobs += [(pid, "high_quality", str(HIGH_QUALITY_ROOT / "midi" / f"{pid}.mid"))
                 for pid in hq_ids]
        print(f"{len(jobs)} pieces ({len(curated_ids)} curated, "
              f"{len(hq_ids)} high_quality)")
        n_shards = N_SHARDS
        frames_dir, manifest_name = "frames", "manifest"

    out_frames = OUT_ROOT / frames_dir
    out_frames.mkdir(parents=True, exist_ok=True)

    # process shard group by shard group so only one shard of features is
    # ever resident: the full corpus does not fit in RAM
    by_shard: dict[int, list[tuple[str, str, str]]] = defaultdict(list)
    for job in jobs:
        by_shard[int(hashlib.sha1(job[0].encode()).hexdigest(), 16)
                 % n_shards].append(job)

    rows: list[dict] = []
    outliers: list[dict] = []
    skipped = done = 0
    start = time.time()
    with mp.Pool(args.workers) as pool:
        for shard in sorted(by_shard):
            arrays: dict[str, np.ndarray] = {}
            for result in pool.imap_unordered(process, by_shard[shard],
                                              chunksize=16):
                done += 1
                if result is None:
                    skipped += 1
                    continue
                if "too_long" in result:
                    outliers.append(result)
                    continue
                arrays[result["piece_id"]] = result.pop("features")
                result["shard"] = shard
                rows.append(result)
            np.savez_compressed(out_frames / f"shard_{shard:03d}.npz", **arrays)
            rate = done / max(time.time() - start, 1e-9) * 60
            print(f"  shard {shard:03d}: {len(arrays)} pieces, "
                  f"{done}/{len(jobs)} done, {rate:.0f} pieces/min", flush=True)

    pq.write_table(pa.Table.from_pylist(rows), OUT_ROOT / f"{manifest_name}.parquet")
    if args.source == "aria":
        write_tempo_estimates(rows)
    if outliers:
        pq.write_table(pa.Table.from_pylist(outliers),
                       OUT_ROOT / f"{manifest_name}_too_long.parquet")
    print(f"done: {len(rows)} pieces, {skipped} skipped (empty MIDI), "
          f"{len(outliers)} skipped (> {MAX_DURATION_SEC:.0f} s), "
          f"{len(by_shard)} shards -> {OUT_ROOT}")


if __name__ == "__main__":
    main()
