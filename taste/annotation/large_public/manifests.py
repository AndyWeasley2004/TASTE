#!/usr/bin/env python3
"""Build the manifests for large-public tag labeling.

`benchmark` builds the 61,129-piece work-disjoint benchmark manifest from the
curated and high-quality tiers, on which the tag probes are trained and
calibrated. The released probes were fitted on the shipped
`large_public_annotation/benchmark.parquet`, whose `split` column is canonical
and is what `tag_probes fit` reads; rebuilding with `benchmark` hashes the
renamed group keys and yields a different train/val split. `deduped` builds the 371,053-piece Aria-deduped manifest, carrying
the Aria metadata genre and its mapping into the controlled vocabulary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from taste import paths
from taste.annotation.curated.common import CURATED_ROOT, read_parquet_rows
from taste.corpus.sft_format import parse_legacy_prompt

ROOT = paths.data("large_public_annotation")
MANIFEST = ROOT / "benchmark.parquet"
HIGH_QUALITY_ROOT = paths.data("high_quality")
FIELDS = ("emotion", "genre", "accompaniment")
EXPECTED_ROWS = 61_129

DEDUPED_ROOT = paths.data("aria_midi", "deduped")
DEDUPED_EXPECTED = 371_053

# Aria metadata genre -> controlled genre vocabulary.
GENRE_MAP = {
    "classical": "classical",
    "atonal": "classical",
    "jazz": "jazz",
    "blues": "jazz",
    "ragtime": "jazz",
    "pop": "pop",
    "rock": "pop",
    "soundtrack": "pop",
    "folk": "pop",
    "ambient": "pop",
}


def split_for(work_id: str) -> str:
    value = int(hashlib.sha1(work_id.encode()).hexdigest()[:8], 16)
    return "val" if value % 5 == 0 else "train"


def _normalized_identity(value: str | None) -> str:
    return re.sub(r"[^\w]+", " ", str(value or "").casefold()).strip()


def source_native_key(piece: dict) -> str:
    """Return a source-local performance/work key for clip families."""
    source = str(piece["source"])
    value = str(piece.get("source_id") or "").strip()
    if source == "piast_yt":
        value = re.sub(r"_part_\d+$", "", value)
    elif source == "emopia":
        value = re.sub(r"^Q[1-4]_", "", value)
        value = re.sub(r"(?:_\d+)+$", "", value)
    return f"native:{source}:{value or piece['piece_id']}"


def catalog_key(piece: dict) -> str | None:
    """Return a cross-source work key when creator and title are informative."""
    creator = _normalized_identity(piece.get("creator"))
    title = _normalized_identity(piece.get("title"))
    if len(creator) < 2 or len(title) < 4:
        return None
    if creator in {"unknown", "none", "anonymous"}:
        return None
    if title in {"unknown", "none", "untitled"}:
        return None
    return f"catalog:{creator}:{title}"


def connected_work_groups(piece_rows: list[dict]) -> dict[str, str]:
    """Union source-native clip families and cross-source catalog identities."""
    parent = {str(piece["piece_id"]): str(piece["piece_id"]) for piece in piece_rows}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        keep, merge = sorted((left_root, right_root))
        parent[merge] = keep

    first_for_key: dict[str, str] = {}
    for piece in piece_rows:
        piece_id = str(piece["piece_id"])
        keys = [source_native_key(piece)]
        cross_source = catalog_key(piece)
        if cross_source:
            keys.append(cross_source)
        for key in keys:
            previous = first_for_key.setdefault(key, piece_id)
            union(piece_id, previous)
    return {
        piece_id: f"curated:component:{find(piece_id)}"
        for piece_id in parent
    }


def performance_axes(value: str | None) -> tuple[str | None, str | None]:
    if not value:
        return None, None
    parts = value.split()
    if len(parts) != 2 or parts[0] not in {"steady", "flexible"} or parts[1] not in {"connected", "detached"}:
        raise ValueError(f"invalid performance label: {value!r}")
    return parts[0], parts[1]


def build_manifest() -> None:
    rows = []
    piece_rows = read_parquet_rows(CURATED_ROOT / "manifests" / "pieces.parquet")
    pieces = {row["piece_id"]: row for row in piece_rows}
    split_groups = connected_work_groups(piece_rows)
    for record in read_parquet_rows(CURATED_ROOT / "manifests" / "training_rows.parquet"):
        piece_dir = CURATED_ROOT / "pieces" / record["work_id"]
        piece = pieces[record["work_id"]]
        split_group = split_groups[str(record["work_id"])]
        label = json.loads((piece_dir / "label.json").read_text())
        fields = label.get("accepted_fields") or {}
        performance = str(fields.get("performance") or "").strip() or None
        expression, articulation = performance_axes(performance)
        genre = str(fields.get("genre") or "").strip() or None
        rows.append({
            "piece_id": record["performance_id"],
            "work_id": record["work_id"],
            "split_group": split_group,
            "source": "curated",
            "midi_path": str(Path(record["midi_path"])),
            "split": split_for(split_group),
            "emotion": str(fields.get("emotion") or "").strip() or None,
            "genre": "jazz" if genre == "blues" else genre,
            "accompaniment": str(fields.get("accompaniment") or "").strip() or None,
            "performance": performance,
            "expression": expression,
            "articulation": articulation,
            "imagery": str(fields.get("imagery") or "").strip() or None,
        })
    for path in sorted((HIGH_QUALITY_ROOT / "metadata").glob("*.json")):
        metadata = json.loads(path.read_text())
        fields = parse_legacy_prompt(metadata["prompt"], require_all=False)
        expression, articulation = performance_axes(fields.get("performance"))
        rows.append({
            "piece_id": path.stem,
            "work_id": path.stem,
            "split_group": f"high_quality:{path.stem}",
            "source": "high_quality",
            "midi_path": str(HIGH_QUALITY_ROOT / "midi" / f"{path.stem}.mid"),
            "split": split_for(f"high_quality:{path.stem}"),
            "emotion": fields.get("emotion"),
            "genre": "jazz" if fields.get("genre") == "blues" else fields.get("genre"),
            "accompaniment": fields.get("accompaniment"),
            "performance": fields.get("performance"),
            "expression": expression,
            "articulation": articulation,
            "imagery": fields.get("imagery"),
        })
    if len(rows) != EXPECTED_ROWS or len({row["piece_id"] for row in rows}) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS:,} unique benchmark rows, got {len(rows):,}")
    if any(not Path(row["midi_path"]).is_file() for row in rows):
        raise FileNotFoundError("benchmark contains a missing MIDI path")
    ROOT.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), MANIFEST, compression="zstd")
    train_groups = {row["split_group"] for row in rows if row["split"] == "train"}
    val_groups = {row["split_group"] for row in rows if row["split"] == "val"}
    if train_groups & val_groups:
        raise ValueError("source-work groups overlap train and validation")
    summary = {
        "rows": len(rows),
        "sources": {source: sum(row["source"] == source for row in rows)
                    for source in ("curated", "high_quality")},
        "splits": {split: sum(row["split"] == split for row in rows) for split in ("train", "val")},
        "split_groups": {
            "total": len(train_groups | val_groups),
            "train": len(train_groups),
            "val": len(val_groups),
            "overlap": 0,
        },
        "available_labels": {
            field: sum(bool(str(row.get(field) or "").strip()) for row in rows)
            for field in (*FIELDS, "performance", "imagery")
        },
        "split_policy": (
            "SHA1(connected work component) modulo 5; components union source-native "
            "performance/clip IDs with normalized creator+title identities across sources"
        ),
    }
    (ROOT / "benchmark_meta.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def build_deduped_manifest() -> None:
    metadata = json.loads((DEDUPED_ROOT / "metadata.json").read_text())
    midi_paths = sorted((DEDUPED_ROOT / "data").glob("*/*.mid"))
    rows = []
    missing = []
    for path in midi_paths:
        piece_id = path.stem
        metadata_id = piece_id.rsplit("_", 1)[0]
        if metadata_id.isdigit():
            metadata_id = str(int(metadata_id))
        record = metadata.get(metadata_id)
        if record is None:
            missing.append(piece_id)
            continue
        raw_genre = str(record.get("metadata", {}).get("genre") or "").strip().lower()
        rows.append({
            "piece_id": piece_id,
            "metadata_id": metadata_id,
            "midi_path": str(path),
            "midi_relpath": path.relative_to(DEDUPED_ROOT).as_posix(),
            "metadata_genre": raw_genre or None,
            "mapped_genre": GENRE_MAP.get(raw_genre),
        })
    if len(midi_paths) != DEDUPED_EXPECTED or len(rows) != DEDUPED_EXPECTED or missing:
        raise ValueError(
            f"deduped mismatch: files={len(midi_paths):,} rows={len(rows):,} "
            f"missing_metadata={len(missing):,}"
        )
    if len({row["piece_id"] for row in rows}) != DEDUPED_EXPECTED:
        raise ValueError("deduped piece IDs are not unique")
    ROOT.mkdir(parents=True, exist_ok=True)
    output = ROOT / "deduped_manifest.parquet"
    pq.write_table(pa.Table.from_pylist(rows), output, compression="zstd")
    mapped = Counter(row["mapped_genre"] or "<missing>" for row in rows)
    summary = {
        "rows": len(rows),
        "unique_piece_ids": DEDUPED_EXPECTED,
        "missing_metadata": 0,
        "mapped_genre_counts": dict(mapped),
        "exact_midi_path_match": True,
        "output": str(output),
    }
    (ROOT / "deduped_manifest_meta.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("benchmark", help="curated + high-quality benchmark manifest")
    subparsers.add_parser("deduped", help="Aria-deduped corpus manifest")
    args = parser.parse_args()
    if args.command == "benchmark":
        build_manifest()
    else:
        build_deduped_manifest()


if __name__ == "__main__":
    main()
