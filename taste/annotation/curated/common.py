"""Shared helpers for the curated corpus: text and work-ID normalization, MIDI inspection, parquet/JSON I/O."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable

import mido
import pyarrow as pa
import pyarrow.parquet as pq

from taste import paths


CURATED_ROOT = paths.data("curated")

PERFORMANCE_SCHEMA = pa.schema([
    pa.field("performance_id", pa.string()),
    pa.field("source", pa.string()),
    pa.field("source_id", pa.string()),
    pa.field("work_id", pa.string()),
    pa.field("composer", pa.string()),
    pa.field("composition", pa.string()),
    pa.field("movement", pa.string()),
    pa.field("artist", pa.string()),
    pa.field("title", pa.string()),
    pa.field("performer", pa.string()),
    pa.field("midi_path", pa.string()),
    pa.field("quality_label", pa.string()),
    pa.field("capture_model", pa.string()),
    pa.field("is_transcription", pa.bool_()),
    pa.field("duration_sec", pa.float64()),
    pa.field("note_count", pa.int64()),
    pa.field("has_pedal", pa.bool_()),
    pa.field("pedal_event_count", pa.int64()),
    pa.field("sha256", pa.string()),
    pa.field("note_fingerprint", pa.string()),
    pa.field("genre", pa.string()),
    pa.field("emotion", pa.string()),
    pa.field("source_metadata", pa.string()),
])


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return re.sub(r"\s+", " ", text)


def normalize_name(value: Any) -> str:
    text = unicodedata.normalize("NFKD", clean_text(value)).casefold()
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.replace("♭", " flat ").replace("♯", " sharp ").replace("#", " sharp ").replace("&", " and ")
    text = re.sub(r"\b([a-g])b\b", r"\1 flat", text)
    text = re.sub(r"\b(live|remastered|mono|stereo|recording)\b", " ", text)
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE).replace("_", " ")
    return re.sub(r"\s+", " ", text).strip()


_COMPOSER_ALIASES = {
    "bach": "johann sebastian bach",
    "j s bach": "johann sebastian bach",
    "beethoven": "ludwig van beethoven",
    "chopin": "frederic chopin",
    "debussy": "claude debussy",
    "handel": "george frideric handel",
    "haydn": "joseph haydn",
    "liszt": "franz liszt",
    "mozart": "wolfgang amadeus mozart",
    "w a mozart": "wolfgang amadeus mozart",
    "rachmaninoff": "sergei rachmaninoff",
    "rachmaninov": "sergei rachmaninoff",
    "sergey rachmaninoff": "sergei rachmaninoff",
    "sergei rachmaninov": "sergei rachmaninoff",
    "sergey rachmaninov": "sergei rachmaninoff",
    "ravel": "maurice ravel",
    "schubert": "franz schubert",
    "schumann": "robert schumann",
    "scriabin": "alexander scriabin",
    "tchaikovsky": "pyotr ilyich tchaikovsky",
    "pyotr tchaikovsky": "pyotr ilyich tchaikovsky",
    "p i tchaikovsky": "pyotr ilyich tchaikovsky",
}


def normalize_composer(value: Any) -> str:
    name = normalize_name(value)
    return _COMPOSER_ALIASES.get(name, name)


def normalize_work_component(value: Any) -> str:
    text = normalize_name(value)
    substitutions = (
        (r"\bop\s+(\d+)", r"opus \1"),
        (r"\bopus\s+(\d+)", r"opus \1"),
        (r"\bnumber\s+(\d+)", r"no \1"),
        (r"\bno\s+(\d+)", r"no \1"),
        (r"\b(?:kochel|kv)\s+(\d+)", r"k \1"),
        (r"\bhoboken\s+", "hob "),
        (r"\bwoo\s+(\d+)", r"woo \1"),
    )
    for pattern, replacement in substitutions:
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"\boriginally\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def stable_id(prefix: str, *parts: Any, length: int = 16) -> str:
    key = "\x1f".join(normalize_name(p) for p in parts)
    return f"{prefix}_{hashlib.sha1(key.encode()).hexdigest()[:length]}"


def classical_work_id(composer: Any, composition: Any, movement: Any = "") -> str:
    return stable_id(
        "work",
        normalize_composer(composer),
        normalize_work_component(composition),
        normalize_work_component(movement),
    )


def pop_work_id(artist: Any, title: Any) -> str:
    return stable_id("work", artist, title)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_midi(path: Path) -> dict[str, Any]:
    midi = mido.MidiFile(path)
    tempo = 500_000
    sec = 0.0
    notes: dict[tuple[int, int], list[tuple[float, int]]] = {}
    normalized: list[tuple[int, int, int]] = []
    pedal_count = 0
    first_onset: float | None = None

    for msg in mido.merge_tracks(midi.tracks):
        sec += mido.tick2second(msg.time, midi.ticks_per_beat, tempo)
        if msg.type == "set_tempo":
            tempo = msg.tempo
        elif msg.type == "control_change" and msg.control == 64:
            pedal_count += 1
        elif msg.type == "note_on" and msg.velocity > 0:
            notes.setdefault((msg.channel, msg.note), []).append((sec, msg.velocity))
            first_onset = sec if first_onset is None else first_onset
        elif msg.type in {"note_off", "note_on"}:
            stack = notes.get((msg.channel, msg.note))
            if stack:
                onset, _velocity = stack.pop(0)
                origin = first_onset or 0.0
                normalized.append((
                    msg.note,
                    int(round((onset - origin) * 50)),
                    max(1, int(round((sec - onset) * 50))),
                ))

    digest = hashlib.sha256()
    for note in sorted(normalized, key=lambda x: (x[1], x[0], x[2])):
        digest.update(f"{note[0]}:{note[1]}:{note[2]};".encode())
    return {
        "duration_sec": round(sec, 6),
        "note_count": len(normalized),
        "pedal_event_count": pedal_count,
        "has_pedal": pedal_count > 0,
        "sha256": file_sha256(path),
        "note_fingerprint": digest.hexdigest(),
    }


def performance_row(**kwargs: Any) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for field in PERFORMANCE_SCHEMA:
        name = field.name
        value = kwargs.get(name)
        if pa.types.is_string(field.type):
            if name == "source_metadata":
                row[name] = value if isinstance(value, str) else json.dumps(value or {}, ensure_ascii=False)
            else:
                row[name] = clean_text(value)
        elif pa.types.is_boolean(field.type):
            row[name] = bool(value)
        elif pa.types.is_integer(field.type):
            row[name] = int(value or 0)
        elif pa.types.is_floating(field.type):
            row[name] = float(value or 0.0)
        else:
            row[name] = value
    return row


def write_parquet(rows: Iterable[dict[str, Any]], path: Path, schema: pa.Schema) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = {field.name: [row.get(field.name) for row in rows] for field in schema}
    table = pa.table(columns, schema=schema)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(path)


def read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)
