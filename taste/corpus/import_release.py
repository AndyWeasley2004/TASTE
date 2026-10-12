#!/usr/bin/env python3
"""Expand released labels and MIDI into the annotation and corpus-builder layout.

Curated MIDI uses midi/<xx>/<piece_id>.mid, where xx follows the piece_ prefix.
Aria-MIDI files are acquired separately under aria_midi/{pruned,deduped}/data/.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from taste import paths
from taste.representation.key import FIELD_SOURCE, key_token

ATTRIBUTE_FIELDS = ("emotion", "genre", "accompaniment", "performance")


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_labels(tier_dir: Path) -> list[dict]:
    return pq.read_table(tier_dir / "labels.parquet").to_pylist()


def copy_midi(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def labelled_sections(structure: list[dict]) -> list[dict]:
    return [
        {"start_sec": s["start_sec"], "end_sec": s["end_sec"], "label": s["letter"]}
        for s in structure
    ]


def timing(structure: list[dict]) -> dict:
    """Section timing in the ``metadata/<id>.json`` form of the high-quality tier."""
    return {
        "sections": [
            {"letter": s["letter"], "start_sec": s["start_sec"], "end_sec": s["end_sec"]}
            for s in structure
        ],
        "duration_sec": structure[-1]["end_sec"],
    }


def import_curated(release_dir: Path) -> int:
    root = paths.data("curated")
    tier_dir = release_dir / "curated"
    records = read_labels(tier_dir)
    for record in records:
        piece_id = record["piece_id"]
        directory = root / "pieces" / piece_id
        copy_midi(tier_dir / "midi" / piece_id[6:8] / f"{piece_id}.mid", directory / "performance.mid")
        write_json(directory / "label.json", {
            "piece_id": piece_id,
            "accepted_fields": {
                **{name: record[name] for name in ATTRIBUTE_FIELDS},
                "imagery": record["imagery"],
                "key": record["key"],
            },
            "field_sources": {"key": FIELD_SOURCE},
            "source_dataset": record["source_dataset"],
            "source_id": record["source_id"],
        })
        write_json(directory / "tempo.json", {
            "tempo_bpm": record["tempo_bpm"],
            "tempo_class": record["tempo_class"],
        })
        write_json(directory / "structure.json", {
            "recommended_level": 4,
            "levels": [{"level": 4, "sections": labelled_sections(record["structure"])}],
        })
        initial = record["structure_initial"]
        write_json(directory / "meta_archive" / "structure_section_fusion.json", {
            "recommended_level": 4,
            "audio_duration_sec": initial[-1]["end_sec"],
            "levels": [{"level": 4, "sections": labelled_sections(initial)}],
        })
    (root / "manifests").mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table({"piece_id": [record["piece_id"] for record in records]}),
        root / "manifests" / "pieces.parquet",
    )
    pq.write_table(
        pa.table({
            "piece_id": [record["piece_id"] for record in records],
            "split": [record["split"] for record in records],
        }),
        root / "manifests" / "split.parquet",
    )
    return len(records)


def import_high_quality(release_dir: Path) -> int:
    root = paths.data("high_quality")
    tier_dir = release_dir / "high_quality"
    records = read_labels(tier_dir)
    for record in records:
        sample_id = record["sample_id"]
        copy_midi(tier_dir / "midi" / f"{sample_id}.mid", root / "midi" / f"{sample_id}.mid")
        write_json(root / "metadata" / f"{sample_id}.json", {
            "sample_id": sample_id,
            "key": record["key"],
            "key_source": FIELD_SOURCE,
            "section_sequence": [
                f"{letter}{index}" for index, letter in enumerate(record["section_plan"])
            ],
            "section_sequence_timed": [
                {"name": f"{s['letter']}{index}", "start_sec": s["start_sec"]}
                for index, s in enumerate(record["structure"])
            ],
            "section_timing": timing(record["structure"]),
            "section_timing_initial": timing(record["structure_initial"]),
            "split": record["split"],
            "tempo_bpm": record["tempo_bpm"],
            "tempo_class": record["tempo_class"],
        })
        write_json(root / "prompts" / f"{sample_id}.json", {
            "sample_id": sample_id,
            "input_attributes": {name: record[name] for name in ATTRIBUTE_FIELDS},
            "metaphor": record["imagery"],
        })
    return len(records)


def import_large_public(release_dir: Path) -> int:
    root = paths.data("aria_midi", "deduped")
    records = read_labels(release_dir / "large_public")
    relpaths = [record["midi_path"] for record in records]
    for record in records:
        record["piece_id"] = Path(record["midi_path"]).stem

    piece_ids = [record["piece_id"] for record in records]
    column = lambda name: [record[name] for record in records]  # noqa: E731
    tables = {
        "large_public_labels/labels.parquet": {
            "piece_id": piece_ids,
            "midi_relpath": relpaths,
            **{name: column(name) for name in ATTRIBUTE_FIELDS},
            "imagery": column("imagery"),
            "key": column("key"),
            "tempo_bpm": column("tempo_bpm"),
            "tempo": column("tempo_class"),
        },
        "structure_labels/labels.parquet": {
            "piece_id": piece_ids,
            "section_letters": [[s["letter"] for s in r["structure"]] for r in records],
            "section_starts_sec": [[s["start_sec"] for s in r["structure"]] for r in records],
            "section_ends_sec": [[s["end_sec"] for s in r["structure"]] for r in records],
        },
        # Aria file stems are <composition_id>_<segment_id>
        "key_estimates/part-00000.parquet": {
            "midi_relpath": relpaths,
            "piece_id": piece_ids,
            "composition_id": [piece_id.rpartition("_")[0] for piece_id in piece_ids],
            "segment_id": [int(piece_id.rpartition("_")[2]) for piece_id in piece_ids],
            "key": column("key"),
            "key_token": [key_token(key) for key in column("key")],
        },
    }
    for relpath, columns in tables.items():
        (root / relpath).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(columns), root / relpath, compression="zstd")
    return len(records)


def import_large_public_annotation(release_dir: Path) -> int:
    destination = paths.data("large_public_annotation", "benchmark.parquet")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(release_dir / "large_public_annotation" / "benchmark.parquet", destination)
    return pq.read_metadata(destination).num_rows


IMPORTERS = {
    "curated": import_curated,
    "high_quality": import_high_quality,
    "large_public": import_large_public,
    "large_public_annotation": import_large_public_annotation,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", required=True, choices=sorted(IMPORTERS))
    parser.add_argument("--release-dir", type=Path, required=True)
    args = parser.parse_args()
    count = IMPORTERS[args.tier](args.release_dir)
    print(f"imported {count} {args.tier} records into {paths.DATA_ROOT}")


if __name__ == "__main__":
    main()
