#!/usr/bin/env python3
"""Estimate one global tempo per piece from its beat track and store it with the piece's labels.

The tempo is ``60 / median(inter-beat interval)``, mapped to a tempo class by
``midi_features.tempo_class``. ``--dataset curated`` reads
``pieces/<id>/audio_beats.txt`` and writes ``pieces/<id>/tempo.json``, keeping
the onset-IOI estimate of the performance MIDI as ``tempo_bpm_onset`` and using
it when the beat estimate falls outside ``PLAUSIBLE_BPM``. ``--dataset
high_quality`` reads ``beats/<id>.beats.txt`` and writes ``tempo_bpm`` and
``tempo_class`` into ``metadata/<id>.json``.

    python -m taste.representation.tempo --dataset curated --workers 8
    python -m taste.representation.tempo --dataset high_quality --workers 8
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from datetime import date
from pathlib import Path

import numpy as np

from taste import paths
from taste.representation.midi_features import estimate_performance_tempo, tempo_class

DEFAULT_ROOTS = {
    "curated": paths.data("curated"),
    "high_quality": paths.data("high_quality"),
}
OUT_NAME = "tempo.json"
METHOD = (
    "beat_this median inter-beat interval (audio_beats.txt); onset-IOI estimate "
    "kept for reference and used as the fallback outside PLAUSIBLE_BPM"
)
# a handful of beats spread over minutes is a failed beat track, not a tempo;
# outside this band the beat estimate is discarded for the onset one
PLAUSIBLE_BPM = (20.0, 400.0)

_ROOT = DEFAULT_ROOTS["curated"]


def beat_bpm(path: Path) -> float | None:
    """BPM from the median inter-beat interval; None if there is no interval."""
    beats = np.loadtxt(path, ndmin=1)
    if len(beats) < 2:
        return None
    return 60.0 / float(np.median(np.diff(beats)))


def resolve_tempo(
    beat: float | None, onset: float | None
) -> tuple[float | None, str | None]:
    """Authoritative BPM and the fallback tag, from the two estimates.

    An implausible beat estimate with no onset estimate to fall back on is kept
    unchanged so the piece stays in the corpus.
    """
    if beat is not None and PLAUSIBLE_BPM[0] <= beat <= PLAUSIBLE_BPM[1]:
        return beat, None
    if onset is None:
        return beat, None
    return onset, "onset"


def tempo_record(beat: float | None, onset: float | None) -> dict:
    bpm, fallback = resolve_tempo(beat, onset)
    return {
        "tempo_bpm": bpm,
        "tempo_bpm_beat": beat,
        "tempo_bpm_onset": onset,
        "tempo_class": tempo_class(bpm) if bpm is not None else None,
        "fallback": fallback,
        "method": METHOD,
        "date": date.today().isoformat(),
    }


def estimate(piece_id: str) -> tuple[str, float | None, float | None, str | None]:
    """Write ``pieces/<piece_id>/tempo.json`` and return its numbers."""
    directory = _ROOT / "pieces" / piece_id
    beat = beat_bpm(directory / "audio_beats.txt")
    note = None
    try:
        onset_bpm = float(estimate_performance_tempo(directory / "performance.mid")[0])
    except ValueError as error:
        onset_bpm, note = None, f"onset estimator failed: {error}"
    record = tempo_record(beat, onset_bpm)
    (directory / OUT_NAME).write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    return piece_id, record["tempo_bpm"], onset_bpm, note


def estimate_high_quality(sample_id: str) -> tuple[str, float | None]:
    """Write ``tempo_bpm``/``tempo_class`` into ``metadata/<sample_id>.json`` and return the BPM."""
    bpm = beat_bpm(_ROOT / "beats" / f"{sample_id}.beats.txt")
    path = _ROOT / "metadata" / f"{sample_id}.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["tempo_bpm"] = bpm
    metadata["tempo_class"] = tempo_class(bpm) if bpm is not None else None
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return sample_id, bpm


def run_high_quality(root: Path, workers: int) -> None:
    sample_ids = sorted(path.stem for path in (root / "metadata").glob("*.json"))
    with mp.Pool(workers, initializer=_init, initargs=(root,)) as pool:
        results = pool.map(estimate_high_quality, sample_ids, chunksize=64)
    no_beats = [sample_id for sample_id, bpm in results if bpm is None]
    print(
        f"wrote tempo_bpm/tempo_class into metadata/<sample_id>.json for "
        f"{len(results)} samples ({len(no_beats)} without a beat-based estimate)"
    )


def _init(root: Path) -> None:
    global _ROOT
    _ROOT = root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DEFAULT_ROOTS), default="curated")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("--workers must be between 1 and 8")
    root = args.root or DEFAULT_ROOTS[args.dataset]
    if args.dataset == "high_quality":
        run_high_quality(root, args.workers)
        return

    piece_ids = sorted(path.name for path in (root / "pieces").iterdir())
    started = time.monotonic()
    results: list[tuple[str, float | None, float | None, str | None]] = []
    with mp.Pool(args.workers, initializer=_init, initargs=(root,)) as pool:
        for done, result in enumerate(
            pool.imap_unordered(estimate, piece_ids, chunksize=64), 1
        ):
            results.append(result)
            if done % 2000 == 0:
                print(
                    f"[{done}/{len(piece_ids)}] "
                    f"{done / (time.monotonic() - started):.0f} pieces/s",
                    flush=True,
                )

    results.sort()
    notes = [(piece_id, note) for piece_id, _, _, note in results if note]
    no_beats = [piece_id for piece_id, bpm, _, _ in results if bpm is None]
    classes: dict[str, int] = {}
    agree = comparable = 0
    for _, bpm, onset, _ in results:
        if bpm is None:
            continue
        name = tempo_class(bpm)
        classes[name] = classes.get(name, 0) + 1
        if onset is not None:
            comparable += 1
            agree += name == tempo_class(onset)
    print(
        f"wrote pieces/<piece_id>/{OUT_NAME} for {len(results)} pieces "
        f"({len(no_beats)} without a beat-based estimate, {len(notes)} without an "
        f"onset estimate) in {time.monotonic() - started:.0f}s"
    )
    print(f"  class distribution: {classes}")
    print(
        f"  beat/onset class agreement: {agree}/{comparable} "
        f"({agree / max(1, comparable):.4f})"
    )
    for piece_id, note in notes[:20]:
        print(f"  {piece_id}: {note}")


if __name__ == "__main__":
    main()
