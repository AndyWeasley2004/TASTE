#!/usr/bin/env python3
"""Audio route: list the curated pieces whose label holds at most one field, indexed by piece ID, for audio captioning."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from taste.annotation.curated.common import CURATED_ROOT


DEFAULT_ROOT = CURATED_ROOT
DEFAULT_PROMPT = Path(__file__).with_name("caption_prompt.txt")
DEFAULT_JSON = DEFAULT_ROOT / "manifests/multimodal_annotation_manifest.json"
DEFAULT_CSV = DEFAULT_ROOT / "manifests/multimodal_annotation_manifest.csv"
AUDIO_FILENAME = "rendered_audio.mp3"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    columns = [
        "piece_id",
        "audio_path",
        "pipeline_tag",
        "selection_reason",
        "label_status",
        "accepted_fields",
    ]
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **row,
                "accepted_fields": ",".join(row["accepted_fields"]),
            })
    os.replace(temporary, path)


def pipeline_tag(
    labeling_route: str,
    label: dict[str, Any],
    agent_trace_exists: bool,
) -> str:
    if labeling_route == "multimodal_required":
        return "multimodal_required"
    if label.get("model") or label.get("revision") or agent_trace_exists:
        return "search_fallback_to_multimodal"
    return "agentic_search"


def build_manifest(root: Path, prompt_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pieces = pq.read_table(root / "manifests/pieces.parquet").to_pylist()
    piece_ids = [row["piece_id"] for row in pieces]
    if len(piece_ids) != len(set(piece_ids)):
        raise RuntimeError("pieces.parquet contains duplicate piece IDs")
    if not prompt_path.is_file():
        raise FileNotFoundError(f"missing shared prompt: {prompt_path}")

    indexed: dict[str, dict[str, Any]] = {}
    csv_rows: list[dict[str, Any]] = []
    for piece in sorted(pieces, key=lambda row: row["piece_id"]):
        piece_id = piece["piece_id"]
        piece_dir = root / "pieces" / piece_id
        label_path = piece_dir / "label.json"
        if not label_path.is_file():
            raise FileNotFoundError(f"missing label: {label_path}")

        label = json.loads(label_path.read_text(encoding="utf-8"))
        accepted = label.get("accepted_fields")
        accepted = accepted if isinstance(accepted, dict) else {}
        if len(accepted) > 1:
            continue

        audio_path = piece_dir / "meta_archive" / AUDIO_FILENAME
        if not audio_path.is_file() or audio_path.stat().st_size == 0:
            raise FileNotFoundError(f"missing or empty rendered audio: {audio_path}")

        row = {
            "piece_id": piece_id,
            "audio_path": str(audio_path),
            "pipeline_tag": pipeline_tag(
                piece["labeling_route"],
                label,
                (piece_dir / "meta_archive" / "agent_trace.json").is_file(),
            ),
            "selection_reason": "no_fields" if not accepted else "one_field",
            "label_status": label.get("status", ""),
            "accepted_fields": sorted(accepted),
        }
        indexed[piece_id] = {key: value for key, value in row.items() if key != "piece_id"}
        csv_rows.append(row)

    manifest = {
        "schema_version": 1,
        "index": "piece_id",
        "piece_count": len(indexed),
        "audio_filename": AUDIO_FILENAME,
        "shared_prompt": {
            "path": str(prompt_path.resolve()),
            "sha256": sha256(prompt_path),
        },
        "pieces": indexed,
    }
    return manifest, csv_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()

    manifest, rows = build_manifest(args.root, args.prompt)
    write_json_atomic(args.json_output, manifest)
    write_csv_atomic(args.csv_output, rows)

    print(json.dumps({
        "pieces": len(rows),
        "selection_reasons": Counter(row["selection_reason"] for row in rows),
        "pipeline_tags": Counter(row["pipeline_tag"] for row in rows),
        "json_output": str(args.json_output),
        "csv_output": str(args.csv_output),
        "prompt": manifest["shared_prompt"],
    }, ensure_ascii=False, indent=2, default=dict))


if __name__ == "__main__":
    main()
