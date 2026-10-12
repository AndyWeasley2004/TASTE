"""Audio route: map each stored Gemini caption to the fixed attribute vocabulary plus one imagery sentence with a local Qwen3.6 server.

By default the script starts the vLLM server, normalizes every piece in the audio-route manifest, writes and audits each
piece's label.json, and stops the server even when the run fails.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from openai import OpenAI

from taste import paths
from taste.annotation.curated.common import CURATED_ROOT

REPO = Path(__file__).resolve().parents[4]
ROOT = CURATED_ROOT
DEFAULT_MANIFEST = ROOT / "manifests/multimodal_annotation_manifest.json"
DEFAULT_PROMPT = Path(__file__).with_name("normalize_prompt.txt")
DEFAULT_RUN_DIR = (
    ROOT / "manifests/gemini35_flash_normalization_v4_grounded_first_section"
)
OUTPUT_NAME = "label.json"
CAPTION_NAME = "gemini35_flash_caption.txt"
MODEL_NAME = "Qwen/Qwen3.6-35B-A3B"
MODEL_PATH = paths.ckpt(
    "hf_cache",
    "models--Qwen--Qwen3.6-35B-A3B",
    "snapshots",
    "995ad96eacd98c81ed38be0c5b274b04031597b0",
)
VLLM_BIN = Path("vllm")
REVISION = "gemini35_flash_qwen36_normalization_v4_grounded_first_section"
SOURCE_GENRE_POLICY_REVISION = "source_genre_prior_v1"
FIELDS = ("emotion", "genre", "accompaniment", "performance", "sentiments")
# The model answers under the key "sentiments"; label.json stores it as "imagery".
LABEL_FIELDS = ("emotion", "genre", "accompaniment", "performance", "imagery")
ATTRIBUTES = json.loads((REPO / "attributes.json").read_text())
PIECE_ID_RE = re.compile(r"^piece_[0-9a-f]{16}$")
WORD_RE = re.compile(r"\b[\w\u00c0-\uffff]+(?:[’'-][\w\u00c0-\uffff]+)*\b")
FORBIDDEN_RE = re.compile(
    r"\b(?:"
    r"pianos?|music(?:al|ally)?|pieces?|songs?|"
    r"melod(?:y|ies|ic|ically)|harmon(?:y|ies|ic|ically|ize[ds]?|izing)|"
    r"chords?|notes?|tempos?|keys?|rhythms?|rhythmic|beats?|pulses?|"
    r"phrases?|sections?|performances?|accompaniments?|dynamics?|"
    r"registers?|octaves?|cadences?|motifs?|meters?|arpeggios?|"
    r"legato|staccato|rubato|sounds?|listeners?|recordings?|instruments?|"
    r"pianists?|keyboards?|audible|tunes?|hums?|lyrics?|lyrical|scales?"
    r")\b",
    re.IGNORECASE,
)
IMAGERY_BAD_WORDS = tuple(
    sorted(
        {
            form
            for word in (
                "piano", "pianos", "music", "musical", "musically", "piece",
                "pieces", "song", "songs", "melody", "melodies", "melodic",
                "melodically", "harmony", "harmonies", "harmonic",
                "harmonically", "chord", "chords", "note", "notes", "tempo",
                "tempos", "key", "keys", "rhythm", "rhythms", "rhythmic",
                "beat", "beats", "pulse", "pulses", "phrase", "phrases",
                "section", "sections", "performance", "performances",
                "accompaniment", "accompaniments", "dynamic", "dynamics",
                "register", "registers", "octave", "octaves", "cadence",
                "cadences", "motif", "motifs", "meter", "meters", "arpeggio",
                "arpeggios", "legato", "staccato", "rubato", "sound", "sounds",
                "listener", "listeners", "recording", "recordings",
                "instrument", "instruments", "pianist", "pianists",
                "keyboard", "keyboards", "audible", "tune", "tunes", "hum",
                "hums", "lyrics", "lyrical", "scale", "scales",
            )
            for form in (word, word.title())
        }
    )
)
NORMALIZATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        name: {"type": "string", "enum": values}
        for name, values in ATTRIBUTES.items()
    }
    | {"sentiments": {"type": "string"}},
    "required": list(FIELDS),
}
IMAGERY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"sentiments": {"type": "string"}},
    "required": ["sentiments"],
}
ATTRIBUTE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        name: {"type": "string", "enum": ATTRIBUTES[name]}
        for name in FIELDS[:-1]
    },
    "required": list(FIELDS[:-1]),
}
REPAIR_PROMPT = """Create a concise narrative strictly grounded in the supplied
SOURCE CAPTION. The previous draft is only an editing reference; omit anything
in it that is not explicitly supported by the source caption.
Return exactly one JSON object and nothing else.

Write approximately 30–40 words, using one or two natural sentences. Preserve
the source caption's overall scene progression: opening atmosphere, main
change, and ending state. Every character, object, setting, weather condition,
action, and transition must be traceable to the source caption. Do not add a
new traveler, journey, room, landscape, storm, memory, or ending. Rephrase
concisely; do not reuse any sequence of five consecutive words from the
previous draft. Omit analysis and explanations. End with a complete thought.

Do not use musical language: piano, music, musical, piece, song, melody,
melodic, harmony, harmonic, chord, note, tempo, key, rhythm, rhythmic, beat,
pulse, phrase, section, performance, accompaniment, dynamic, register, octave,
cadence, motif, meter, arpeggio, legato, staccato, rubato, sound, listener,
recording, instrument, pianist, keyboard, or audible. Silently count every word
and keep the result concise."""
CAPTION_FALLBACK_PROMPT = """Write a concise narrative/imagery description
based only on the supplied source caption. Return exactly one JSON object and
nothing else.

Write approximately 30–40 words in one or two complete, natural sentences.
Represent the overall emotional and visual progression across the source:
opening atmosphere, important change, and ending state. Use concrete settings,
imagery, and actions only when explicitly present in the source. Every detail
must be traceable to the source caption. Do not invent a new character, setting,
object, weather condition, action, memory, journey, or ending. Do not discuss
or explain the source caption.

Use no musical language, including piano, music, musical, piece, song, melody,
melodic, harmony, harmonic, chord, note, tempo, key, rhythm, rhythmic, beat,
pulse, phrase, section, performance, accompaniment, dynamic, register, octave,
cadence, motif, meter, arpeggio, legato, staccato, rubato, sound, listener,
recording, instrument, pianist, keyboard, audible, tune, hum, lyrics, lyrical,
or scale. Silently count every word before returning the JSON."""
CLEAN_IMAGERY_PROMPT = """Return exactly one JSON object with the single key
"sentiments" and nothing else.

Using only the source caption, write 20–40 words of natural everyday prose that
summarizes its explicitly stated feelings, atmosphere, imagery, and progression.
Do not mention the art form, its construction, technique, creator, audience, or
analysis. Do not add any person, place, object, weather, action, memory, event,
transition, or ending absent from the source. If the source contains no concrete
scene, give only an abstract summary of its stated emotional progression. Use
one or two complete sentences."""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def imagery_errors(text: Any) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return ["sentiments must be a non-empty string"]
    value = " ".join(text.split())
    count = len(WORD_RE.findall(value))
    errors: list[str] = []
    if not 20 <= count <= 60:
        errors.append(f"sentiments has {count} words; hard range is 20-60")
    forbidden = sorted({match.group(0).lower() for match in FORBIDDEN_RE.finditer(value)})
    if forbidden:
        errors.append("forbidden musical terms: " + ", ".join(forbidden))
    return errors


def narrative_evidence(caption: str) -> str:
    if "***" in caption:
        evidence = caption.rsplit("***", 1)[1].strip()
    else:
        paragraphs = [
            paragraph.strip()
            for paragraph in re.split(r"\n\s*\n", caption)
            if paragraph.strip()
        ]
        evidence = paragraphs[-1] if paragraphs else caption
    return FORBIDDEN_RE.sub("[analytical reference removed]", evidence)


def validate_normalized(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return ["response is not an object"]
    errors: list[str] = []
    if set(value) != set(FIELDS):
        errors.append(f"keys must be exactly {FIELDS}")
    for field in FIELDS[:-1]:
        if value.get(field) not in ATTRIBUTES[field]:
            errors.append(f"invalid {field}: {value.get(field)!r}")
    errors.extend(imagery_errors(value.get("sentiments")))
    return errors


def training_prompt(fields: dict[str, str]) -> str:
    return (
        f"<emotion> {fields['emotion']}\n"
        f"<genre> {fields['genre']}\n"
        f"<accompaniment> {fields['accompaniment']}\n"
        f"<performance> {fields['performance']}\n"
        f"<imagery> {fields['imagery']}\n\n"
        "MIDI:\n"
    )


def load_items(manifest_path: Path, root: Path) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pieces = manifest.get("pieces")
    if not isinstance(pieces, dict):
        raise ValueError("manifest.pieces must be indexed by piece ID")
    if manifest.get("piece_count") != len(pieces):
        raise ValueError("manifest piece_count does not match pieces")

    items: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    for piece_id in sorted(pieces):
        entry = pieces[piece_id]
        if not PIECE_ID_RE.fullmatch(piece_id):
            raise ValueError(f"invalid piece ID: {piece_id}")
        audio_path = Path(entry["audio_path"])
        if audio_path.parent.name != "meta_archive" or audio_path.parent.parent.name != piece_id:
            raise ValueError(f"audio/piece ID mismatch: {piece_id} -> {audio_path}")
        piece_dir = audio_path.parent.parent
        try:
            piece_dir.relative_to(root / "pieces")
        except ValueError as error:
            raise ValueError(f"piece outside corpus root: {piece_dir}") from error
        caption_path = piece_dir / "meta_archive" / CAPTION_NAME
        output_path = piece_dir / OUTPUT_NAME
        metadata_path = piece_dir / "meta_archive" / "metadata.json"
        if output_path in seen_paths:
            raise ValueError(f"duplicate output path: {output_path}")
        seen_paths.add(output_path)
        if not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        source_genre = metadata.get("genre")
        if metadata.get("piece_id") != piece_id:
            raise ValueError(f"metadata/piece ID mismatch: {metadata_path}")
        if source_genre not in ATTRIBUTES["genre"]:
            raise ValueError(f"invalid source genre in {metadata_path}: {source_genre!r}")
        if piece_dir.parent != root / "pieces":
            raise ValueError(f"piece is not in the flat piece store: {piece_dir}")
        if not caption_path.is_file():
            raise FileNotFoundError(caption_path)
        caption = caption_path.read_text(encoding="utf-8").strip()
        if not caption:
            raise ValueError(f"empty caption: {caption_path}")
        items.append(
            {
                "piece_id": piece_id,
                "caption_path": caption_path,
                "output_path": output_path,
                "caption": caption,
                "caption_sha256": sha256_text(caption),
                "metadata_path": metadata_path,
                "source_dataset": metadata.get("source"),
                "source_genre": source_genre,
                "pipeline_tag": entry.get("pipeline_tag", ""),
                "selection_reason": entry.get("selection_reason", ""),
            }
        )
    return items


def valid_existing(item: dict[str, Any]) -> bool:
    path = item["output_path"]
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    fields = value.get("accepted_fields")
    return (
        value.get("piece_id") == item["piece_id"]
        and value.get("revision") == REVISION
        and value.get("model") == MODEL_NAME
        and value.get("source_caption", {}).get("sha256") == item["caption_sha256"]
        and value.get("label_policy_revision") == SOURCE_GENRE_POLICY_REVISION
        and fields.get("genre") == item["source_genre"]
        and value.get("field_sources", {}).get("genre") == "source_dataset_prior"
        and set(fields) == set(LABEL_FIELDS)
        and not validate_normalized(model_fields(fields))
    )


def model_fields(label_fields: dict[str, str]) -> dict[str, str]:
    """Rename a label.json accepted_fields dict back to the model-facing keys."""
    return {
        model: label_fields[label]
        for model, label in zip(FIELDS, LABEL_FIELDS)
        if label in label_fields
    }


class Normalizer:
    def __init__(
        self,
        base_url: str,
        system_prompt: str,
        timeout: float,
        repair_attempts: int,
    ) -> None:
        self.base_url = base_url
        self.system_prompt = system_prompt
        self.timeout = timeout
        self.repair_attempts = repair_attempts
        self.local = threading.local()
        vocabulary_start = system_prompt.index("FIXED VOCABULARY")
        imagery_start = system_prompt.index("SENTIMENTS —")
        self.attribute_prompt = (
            "Normalize the supplied caption into exactly four fixed-vocabulary "
            "attributes. Return exactly one JSON object containing only emotion, "
            "genre, accompaniment, and performance. Follow these rules:\n\n"
            + system_prompt[vocabulary_start:imagery_start].strip()
        )

    def client(self) -> OpenAI:
        client = getattr(self.local, "client", None)
        if client is None:
            client = OpenAI(
                base_url=self.base_url,
                api_key="local",
                timeout=self.timeout,
                max_retries=2,
            )
            self.local.client = client
        return client

    def request(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        schema_name: str,
        max_tokens: int,
        temperature: float = 0.0,
        bad_words: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        extra_body: dict[str, Any] = {
            "chat_template_kwargs": {"enable_thinking": False}
        }
        if bad_words:
            extra_body["bad_words"] = list(bad_words)
        response = self.client().chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=temperature,
            top_p=0.9,
            max_tokens=max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                },
            },
            extra_body=extra_body,
        )
        content = response.choices[0].message.content
        return json.loads(content) if content else {}

    def normalize(self, item: dict[str, Any]) -> tuple[dict[str, str], int]:
        value: dict[str, Any] | None = None
        last_error: Exception | None = None
        for _ in range(2):
            try:
                value = self.request(
                    [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": "CAPTION:\n" + item["caption"]},
                    ],
                    NORMALIZATION_SCHEMA,
                    "normalized_caption",
                    1024,
                )
                break
            except (json.JSONDecodeError, ValueError) as error:
                last_error = error
        if value is None:
            for _ in range(4):
                try:
                    attributes = self.request(
                        [
                            {"role": "system", "content": self.attribute_prompt},
                            {"role": "user", "content": "CAPTION:\n" + item["caption"]},
                        ],
                        ATTRIBUTE_SCHEMA,
                        "normalized_attributes",
                        256,
                    )
                    value = attributes | {"sentiments": ""}
                    break
                except (json.JSONDecodeError, ValueError) as error:
                    last_error = error
        if value is None:
            raise ValueError(f"could not decode normalization output: {last_error}")
        fixed_errors = [
            error
            for error in validate_normalized(value)
            if not error.startswith("sentiments")
            and not error.startswith("forbidden musical terms")
        ]
        if fixed_errors:
            raise ValueError("; ".join(fixed_errors))

        repairs = 0
        original = str(value.get("sentiments", ""))
        current = original
        errors = imagery_errors(current)
        while errors and repairs < self.repair_attempts:
            repairs += 1
            try:
                repaired = self.request(
                    [
                        {"role": "system", "content": REPAIR_PROMPT},
                        {
                            "role": "user",
                            "content": (
                                f"ORIGINAL DRAFT ({len(WORD_RE.findall(original))} words):\n"
                                + original
                                + "\n\nSOURCE CAPTION:\n"
                                + item["caption"]
                                + "\n\nPREVIOUS INVALID ATTEMPT:\n"
                                + current
                                + "\n\nVIOLATIONS:\n- "
                                + "\n- ".join(errors)
                            ),
                        },
                    ],
                    IMAGERY_SCHEMA,
                    "repaired_imagery",
                    256,
                    temperature=0.35,
                )
            except (json.JSONDecodeError, ValueError):
                continue
            current = str(repaired.get("sentiments", ""))
            errors = imagery_errors(current)
        # If iterative editing remains stuck, regenerate the narrative directly
        # from the source caption.
        if errors:
            for _ in range(6):
                repairs += 1
                try:
                    rewritten = self.request(
                        [
                            {"role": "system", "content": CAPTION_FALLBACK_PROMPT},
                            {
                                "role": "user",
                                "content": (
                                    "SOURCE CAPTION:\n"
                                    + item["caption"]
                                    + "\n\nPREVIOUS VIOLATIONS TO AVOID:\n- "
                                    + "\n- ".join(errors)
                                ),
                            },
                        ],
                        IMAGERY_SCHEMA,
                        "caption_based_imagery",
                        256,
                        temperature=0.65,
                    )
                except (json.JSONDecodeError, ValueError):
                    continue
                current = str(rewritten.get("sentiments", ""))
                errors = imagery_errors(current)
                if not errors:
                    break
        # Last-resort regeneration avoids priming the model with a long
        # blacklist. Validation below still enforces the same hard constraints.
        if errors:
            for _ in range(12):
                repairs += 1
                try:
                    rewritten = self.request(
                        [
                            {"role": "system", "content": CLEAN_IMAGERY_PROMPT},
                            {"role": "user", "content": "SOURCE CAPTION:\n" + item["caption"]},
                        ],
                        IMAGERY_SCHEMA,
                        "clean_imagery",
                        192,
                        temperature=0.75,
                        bad_words=IMAGERY_BAD_WORDS,
                    )
                except (json.JSONDecodeError, ValueError):
                    continue
                current = str(rewritten.get("sentiments", ""))
                errors = imagery_errors(current)
                if not errors:
                    break
        # Rare captions can contain so much analysis that the model repeatedly
        # copies prohibited terminology. Restrict those last retries to the
        # caption's own narrative tail, with analytical terms redacted.
        if errors:
            evidence = narrative_evidence(item["caption"])
            for _ in range(12):
                repairs += 1
                try:
                    rewritten = self.request(
                        [
                            {"role": "system", "content": CLEAN_IMAGERY_PROMPT},
                            {
                                "role": "user",
                                "content": "SOURCE NARRATIVE EVIDENCE:\n" + evidence,
                            },
                        ],
                        IMAGERY_SCHEMA,
                        "narrative_evidence_imagery",
                        192,
                        temperature=0.75,
                        bad_words=IMAGERY_BAD_WORDS,
                    )
                except (json.JSONDecodeError, ValueError):
                    continue
                current = str(rewritten.get("sentiments", ""))
                errors = imagery_errors(current)
                if not errors:
                    break
        if errors:
            raise ValueError("; ".join(errors))

        # A coherent 41-60 word result remains valid. Try one optional rewrite
        # toward the preferred range, but keep the original if the rewrite is
        # invalid or not actually shorter.
        if len(WORD_RE.findall(current)) > 40:
            try:
                repaired = self.request(
                    [
                        {"role": "system", "content": REPAIR_PROMPT},
                        {
                            "role": "user",
                            "content": (
                                f"ORIGINAL DRAFT ({len(WORD_RE.findall(current))} words):\n"
                                + current
                                + "\n\nPREFERRED TARGET: 30-40 words."
                            ),
                        },
                    ],
                    IMAGERY_SCHEMA,
                    "shorter_imagery",
                    256,
                    temperature=0.35,
                )
                candidate = str(repaired.get("sentiments", ""))
                if (
                    not imagery_errors(candidate)
                    and len(WORD_RE.findall(candidate))
                    < len(WORD_RE.findall(current))
                ):
                    current = candidate
                    repairs += 1
            except Exception:
                pass
        value["sentiments"] = current
        return {field: value[field] for field in FIELDS}, repairs


def normalize_one(
    item: dict[str, Any],
    normalizer: Normalizer,
    overwrite: bool,
) -> tuple[str, str, int, float]:
    started = time.monotonic()
    if not overwrite and valid_existing(item):
        return item["piece_id"], "skipped", 0, time.monotonic() - started
    fields, repairs = normalizer.normalize(item)
    fields["imagery"] = fields.pop("sentiments")
    normalized_genre = fields["genre"]
    fields["genre"] = item["source_genre"]
    record = {
        "piece_id": item["piece_id"],
        "status": "complete",
        "model": MODEL_NAME,
        "revision": REVISION,
        "label_policy_revision": SOURCE_GENRE_POLICY_REVISION,
        "pipeline_tag": item["pipeline_tag"],
        "selection_reason": item["selection_reason"],
        "source_caption": {
            "path": str(item["caption_path"]),
            "sha256": item["caption_sha256"],
            "model": "gemini-3.5-flash",
        },
        "accepted_fields": fields,
        "field_sources": {
            field: "gemini35_flash_audio_caption" for field in LABEL_FIELDS
        }
        | {"genre": "source_dataset_prior"},
        "genre_prior": {
            "value": item["source_genre"],
            "source_dataset": item["source_dataset"],
            "metadata_path": str(item["metadata_path"]),
            "previous_normalized_genre": normalized_genre,
        },
        "prompt": training_prompt(fields),
    }
    write_json_atomic(item["output_path"], record)
    return item["piece_id"], "completed", repairs, time.monotonic() - started


def port_is_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def wait_for_server(
    process: subprocess.Popen[str],
    base_url: str,
    timeout: float,
    log_path: Path,
) -> None:
    health_url = base_url.removesuffix("/v1") + "/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            tail = ""
            if log_path.is_file():
                tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-40:])
            raise RuntimeError(
                f"vLLM exited during startup with code {process.returncode}\n{tail}"
            )
        try:
            with urlopen(health_url, timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        time.sleep(2)
    raise TimeoutError(f"vLLM did not become ready within {timeout:.0f}s")


def launch_server(args: argparse.Namespace) -> tuple[subprocess.Popen[str], Any]:
    if port_is_open(args.host, args.port):
        raise RuntimeError(f"{args.host}:{args.port} is already in use")
    gpu_ids = [part.strip() for part in args.gpu_ids.split(",") if part.strip()]
    if not 1 <= len(gpu_ids) <= 4:
        raise ValueError("gpu-ids must select 1-4 GPUs")
    args.server_log.parent.mkdir(parents=True, exist_ok=True)
    log_handle = args.server_log.open("w", encoding="utf-8")
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": ",".join(gpu_ids),
            "HF_HUB_CACHE": str(MODEL_PATH.parents[2]),
            "HF_HUB_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    command = [
        str(args.vllm_bin),
        "serve",
        str(args.model_path),
        "--served-model-name",
        MODEL_NAME,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--tensor-parallel-size",
        str(len(gpu_ids)),
        "--max-model-len",
        "4096",
        "--max-num-seqs",
        str(args.workers),
        "--gpu-memory-utilization",
        "0.90",
        "--enable-prefix-caching",
        "--language-model-only",
        "--generation-config",
        "vllm",
    ]
    process = subprocess.Popen(
        command,
        cwd=REPO,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    return process, log_handle


def stop_server(process: subprocess.Popen[str] | None, log_handle: Any) -> None:
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=90)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=30)
    if log_handle is not None:
        log_handle.close()


def write_audit(
    items: list[dict[str, Any]],
    run_dir: Path,
    started: float,
    completed: int,
    skipped: int,
    repairs: int,
    failures: dict[str, str],
) -> dict[str, Any]:
    valid: list[dict[str, Any]] = []
    invalid: dict[str, str] = {}
    for item in items:
        if valid_existing(item):
            record = json.loads(item["output_path"].read_text(encoding="utf-8"))
            valid.append(
                {
                    "piece_id": item["piece_id"],
                    "caption_sha256": item["caption_sha256"],
                    "label_path": str(item["output_path"]),
                    "imagery_words": len(
                        WORD_RE.findall(record["accepted_fields"]["imagery"])
                    ),
                }
            )
        else:
            invalid[item["piece_id"]] = "missing or invalid output"
    run_dir.mkdir(parents=True, exist_ok=True)
    index_path = run_dir / "label_index.jsonl"
    temporary = index_path.with_name(index_path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in valid:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, index_path)
    word_counts = [row["imagery_words"] for row in valid]
    preferred_count = sum(20 <= count <= 40 for count in word_counts)
    audit = {
        "revision": REVISION,
        "model": MODEL_NAME,
        "expected_pieces": len(items),
        "valid_outputs": len(valid),
        "completed_this_run": completed,
        "skipped_this_run": skipped,
        "imagery_repairs_this_run": repairs,
        "failed_this_run": failures,
        "invalid_outputs": invalid,
        "all_valid": len(valid) == len(items) and not failures,
        "preferred_20_40_count": preferred_count,
        "preferred_20_40_fraction": (
            round(preferred_count / len(word_counts), 6) if word_counts else 0.0
        ),
        "imagery_word_count": {
            "min": min(word_counts) if word_counts else None,
            "median": statistics.median(word_counts) if word_counts else None,
            "max": max(word_counts) if word_counts else None,
        },
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "output_filename": OUTPUT_NAME,
        "caption_filename": CAPTION_NAME,
    }
    write_json_atomic(run_dir / "final_audit.json", audit)
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--gpu-ids", default="0,1")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8016)
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument("--request-timeout", type=float, default=300)
    parser.add_argument("--repair-attempts", type=int, default=6)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--vllm-bin", type=Path, default=VLLM_BIN)
    parser.add_argument(
        "--server-log",
        type=Path,
        default=REPO / "runs/curated_audio_route/vllm.log",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-launch-server",
        action="store_true",
        help="Use an already-running compatible server at host:port.",
    )
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("--workers must be between 1 and 8")
    return args


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    items = load_items(args.manifest, args.root)
    print(f"validated manifest: {len(items)} unique piece IDs and caption paths")
    system_prompt = args.prompt.read_text(encoding="utf-8").strip()
    base_url = f"http://{args.host}:{args.port}/v1"
    process: subprocess.Popen[str] | None = None
    log_handle = None
    completed = skipped = repairs = 0
    failures: dict[str, str] = {}
    try:
        if args.no_launch_server:
            if not port_is_open(args.host, args.port):
                raise RuntimeError(f"no server is listening at {args.host}:{args.port}")
        else:
            process, log_handle = launch_server(args)
            print(f"starting vLLM pid={process.pid} on GPUs {args.gpu_ids}")
            wait_for_server(process, base_url, args.startup_timeout, args.server_log)
            print("vLLM ready")

        normalizer = Normalizer(
            base_url,
            system_prompt,
            args.request_timeout,
            args.repair_attempts,
        )
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(normalize_one, item, normalizer, args.overwrite): item
                for item in items
            }
            for index, future in enumerate(as_completed(futures), 1):
                item = futures[future]
                try:
                    piece_id, status, repair_count, _ = future.result()
                    repairs += repair_count
                    if status == "completed":
                        completed += 1
                    else:
                        skipped += 1
                except Exception as error:
                    piece_id = item["piece_id"]
                    failures[piece_id] = f"{type(error).__name__}: {error}"
                if index % 100 == 0 or index == len(items):
                    elapsed = time.monotonic() - started
                    rate = index / elapsed if elapsed else 0.0
                    print(
                        f"[{index}/{len(items)}] completed={completed} "
                        f"skipped={skipped} failed={len(failures)} "
                        f"repairs={repairs} rate={rate:.2f} pieces/s",
                        flush=True,
                    )
    finally:
        if process is not None:
            print("stopping vLLM", flush=True)
        stop_server(process, log_handle)

    audit = write_audit(
        items, args.run_dir, started, completed, skipped, repairs, failures
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if not audit["all_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
