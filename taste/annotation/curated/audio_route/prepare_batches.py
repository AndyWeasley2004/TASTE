#!/usr/bin/env python3
"""Audio route: split the caption manifest into Gemini batch shards (`plan`) and upload and submit one shard (`submit`)."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from taste.annotation.curated.common import CURATED_ROOT


ROOT = CURATED_ROOT
MANIFEST = ROOT / "manifests/multimodal_annotation_manifest.json"
PROMPT = Path(__file__).with_name("caption_prompt.txt")
OUTPUT = ROOT / "manifests/gemini35_flash_batch_v1"
MODEL = "gemini-3.5-flash"
SHARD_SIZE = 1_600
MAX_WORKERS = 8
MIME_TYPE = "audio/mpeg"
MAX_OUTPUT_TOKENS = 16_384
THINKING_LEVEL = "low"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> tuple[str, str]:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest(), base64.b64encode(digest.digest()).decode("ascii")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def shard_dir(output: Path, index: int) -> Path:
    return output / "shards" / f"shard_{index:03d}"


def load_plan(output: Path) -> dict[str, Any]:
    path = output / "plan.json"
    if not path.is_file():
        raise FileNotFoundError(f"run the plan command first: {path}")
    return read_json(path)


def prompt_for_shard(plan_data: dict[str, Any], index: int) -> tuple[Path, str]:
    for entry in plan_data.get("prompt_schedule", []):
        if entry["start_shard"] <= index <= entry["end_shard"]:
            return Path(entry["prompt_file"]), entry["prompt_sha256"]
    return Path(plan_data["prompt_file"]), plan_data["prompt_sha256"]


def load_shard(output: Path, index: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    directory = shard_dir(output, index)
    summary = read_json(directory / "summary.json")
    rows = [
        json.loads(line)
        for line in (directory / "local_manifest.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line
    ]
    if len(rows) != summary["piece_count"]:
        raise RuntimeError(f"shard {index}: summary/row count mismatch")
    if len({row["piece_id"] for row in rows}) != len(rows):
        raise RuntimeError(f"shard {index}: duplicate piece IDs")
    return summary, rows


def validate_local_row(row: dict[str, Any]) -> None:
    piece_id = row["piece_id"]
    audio = Path(row["audio_path"])
    if audio.name != "rendered_audio.mp3" or audio.parent.name != "meta_archive" or audio.parent.parent.name != piece_id:
        raise RuntimeError(f"{piece_id}: audio path is not ID-aligned: {audio}")
    if not audio.is_file() or audio.stat().st_size != row["audio_bytes"]:
        raise RuntimeError(f"{piece_id}: audio missing or size changed: {audio}")


def plan(args: argparse.Namespace) -> None:
    source = read_json(args.manifest)
    prompt = args.prompt.read_text(encoding="utf-8")
    if not prompt.strip():
        raise RuntimeError(f"empty prompt: {args.prompt}")
    if source.get("index") != "piece_id":
        raise RuntimeError("source manifest is not indexed by piece_id")
    pieces = source["pieces"]
    if source.get("piece_count") != len(pieces):
        raise RuntimeError("source manifest piece_count mismatch")

    rows: list[dict[str, Any]] = []
    for piece_id in sorted(pieces):
        audio = Path(pieces[piece_id]["audio_path"]).resolve()
        row = {
            "piece_id": piece_id,
            "audio_path": str(audio),
            "audio_bytes": audio.stat().st_size,
        }
        validate_local_row(row)
        rows.append(row)
    if len({row["audio_path"] for row in rows}) != len(rows):
        raise RuntimeError("multiple piece IDs map to the same audio path")

    args.output.mkdir(parents=True, exist_ok=True)
    write_text_atomic(args.output / "prompt.txt", prompt)
    shard_summaries = []
    for index, start in enumerate(range(0, len(rows), args.shard_size)):
        selected = rows[start : start + args.shard_size]
        directory = shard_dir(args.output, index)
        body = "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in selected
        )
        write_text_atomic(directory / "local_manifest.jsonl", body)
        summary = {
            "shard_index": index,
            "piece_count": len(selected),
            "audio_bytes": sum(row["audio_bytes"] for row in selected),
            "first_piece_id": selected[0]["piece_id"],
            "last_piece_id": selected[-1]["piece_id"],
            "local_manifest_sha256": sha256_text(body),
        }
        write_json_atomic(directory / "summary.json", summary)
        shard_summaries.append(summary)

    flattened = []
    for index in range(len(shard_summaries)):
        _, shard_rows = load_shard(args.output, index)
        flattened.extend(row["piece_id"] for row in shard_rows)
    expected = [row["piece_id"] for row in rows]
    if flattened != expected:
        raise RuntimeError("shard sequence does not exactly reproduce sorted piece IDs")

    plan_data = {
        "schema_version": 1,
        "created_at": utc_now(),
        "model": args.model,
        "source_manifest": str(args.manifest.resolve()),
        "source_manifest_sha256": sha256_file(args.manifest)[0],
        "prompt_file": str((args.output / "prompt.txt").resolve()),
        "prompt_sha256": sha256_text(prompt),
        "piece_count": len(rows),
        "shard_size": args.shard_size,
        "shard_count": len(shard_summaries),
        "total_audio_bytes": sum(row["audio_bytes"] for row in rows),
        "shards": shard_summaries,
    }
    write_json_atomic(args.output / "plan.json", plan_data)
    validate_plan(args.output)
    print(json.dumps(plan_data, indent=2))


def validate_plan(output: Path) -> None:
    plan_data = load_plan(output)
    prompt_ranges = []
    for index in range(plan_data["shard_count"]):
        prompt_path, prompt_sha = prompt_for_shard(plan_data, index)
        prompt = prompt_path.read_text(encoding="utf-8")
        if sha256_text(prompt) != prompt_sha:
            raise RuntimeError(f"planned prompt changed for shard {index}")
        prompt_ranges.append((str(prompt_path), prompt_sha))
    if len(prompt_ranges) != plan_data["shard_count"]:
        raise RuntimeError("prompt schedule does not cover all shards")
    all_ids: list[str] = []
    total_bytes = 0
    for index, expected_summary in enumerate(plan_data["shards"]):
        summary, rows = load_shard(output, index)
        if summary != expected_summary:
            raise RuntimeError(f"shard {index}: plan summary mismatch")
        for row in rows:
            validate_local_row(row)
        all_ids.extend(row["piece_id"] for row in rows)
        total_bytes += sum(row["audio_bytes"] for row in rows)
    if len(all_ids) != plan_data["piece_count"]:
        raise RuntimeError("planned piece count mismatch")
    if all_ids != sorted(all_ids) or len(all_ids) != len(set(all_ids)):
        raise RuntimeError("planned piece IDs are not sorted and unique")
    if total_bytes != plan_data["total_audio_bytes"]:
        raise RuntimeError("planned audio byte total mismatch")
    source = read_json(Path(plan_data["source_manifest"]))
    if set(all_ids) != set(source["pieces"]):
        raise RuntimeError("planned IDs do not exactly match source manifest")


def load_api_key() -> str:
    if os.environ.get("GEMINI_API_KEY"):
        return os.environ["GEMINI_API_KEY"]
    raise RuntimeError("set GEMINI_API_KEY")


def load_uploads(path: Path) -> dict[str, dict[str, Any]]:
    uploads: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return uploads
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        row = json.loads(line)
        piece_id = row["piece_id"]
        if piece_id in uploads and uploads[piece_id] != row:
            raise RuntimeError(f"conflicting upload records for {piece_id} at line {number}")
        uploads[piece_id] = row
    return uploads


def remote_record(piece_id: str, local: dict[str, Any], uploaded: Any) -> dict[str, Any]:
    if uploaded.display_name != piece_id:
        raise RuntimeError(
            f"{piece_id}: remote display_name mismatch: {uploaded.display_name}"
        )
    if uploaded.mime_type not in (MIME_TYPE, "audio/mp3"):
        raise RuntimeError(f"{piece_id}: unexpected remote MIME: {uploaded.mime_type}")
    if int(uploaded.size_bytes) != local["audio_bytes"]:
        raise RuntimeError(f"{piece_id}: remote/local size mismatch")
    local_hex, local_b64 = sha256_file(Path(local["audio_path"]))
    local_hex_b64 = base64.b64encode(local_hex.encode("ascii")).decode("ascii")
    if uploaded.sha256_hash and uploaded.sha256_hash not in (
        local_b64,
        local_hex_b64,
        local_hex,
    ):
        raise RuntimeError(f"{piece_id}: remote/local SHA-256 mismatch")
    if not uploaded.name or not uploaded.uri:
        raise RuntimeError(f"{piece_id}: upload returned no name or URI")
    return {
        "piece_id": piece_id,
        "audio_path": local["audio_path"],
        "audio_bytes": local["audio_bytes"],
        "audio_sha256": local_hex,
        "remote_name": uploaded.name,
        "remote_uri": uploaded.uri,
        "remote_mime_type": uploaded.mime_type,
        "remote_expiration_time": (
            uploaded.expiration_time.isoformat()
            if uploaded.expiration_time
            else None
        ),
    }


def upload_one(client: Any, row: dict[str, Any]) -> dict[str, Any]:
    from google.genai import types

    piece_id = row["piece_id"]
    uploaded = client.files.upload(
        file=row["audio_path"],
        config=types.UploadFileConfig(
            display_name=piece_id,
            mime_type=MIME_TYPE,
        ),
    )
    return remote_record(piece_id, row, uploaded)


def build_request(piece_id: str, remote: dict[str, Any], prompt: str) -> dict[str, Any]:
    return {
        "key": piece_id,
        "request": {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "fileData": {
                                "fileUri": remote["remote_uri"],
                                "mimeType": MIME_TYPE,
                            }
                        },
                        {"text": prompt},
                    ],
                }
            ],
            "generationConfig": {
                "maxOutputTokens": MAX_OUTPUT_TOKENS,
                "responseModalities": ["TEXT"],
                "thinkingConfig": {"thinkingLevel": THINKING_LEVEL},
            },
        },
    }


def validate_requests(
    path: Path,
    local_rows: list[dict[str, Any]],
    uploads: dict[str, dict[str, Any]],
    prompt: str,
) -> None:
    requests = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    expected_ids = [row["piece_id"] for row in local_rows]
    request_ids = [row["key"] for row in requests]
    if request_ids != expected_ids or len(request_ids) != len(set(request_ids)):
        raise RuntimeError("request keys do not exactly match ordered shard piece IDs")
    for request in requests:
        piece_id = request["key"]
        parts = request["request"]["contents"][0]["parts"]
        if parts[0]["fileData"]["fileUri"] != uploads[piece_id]["remote_uri"]:
            raise RuntimeError(f"{piece_id}: request URI does not match upload mapping")
        if parts[0]["fileData"]["mimeType"] != MIME_TYPE:
            raise RuntimeError(f"{piece_id}: request MIME mismatch")
        if parts[1].get("text") != prompt:
            raise RuntimeError(f"{piece_id}: request prompt mismatch")


def submit(args: argparse.Namespace) -> None:
    from google import genai
    from google.genai import types

    validate_plan(args.output)
    plan_data = load_plan(args.output)
    if not 0 <= args.shard < plan_data["shard_count"]:
        raise ValueError(f"shard must be in [0, {plan_data['shard_count'] - 1}]")
    directory = shard_dir(args.output, args.shard)
    submission_path = directory / "submission.json"
    if submission_path.is_file():
        submission = read_json(submission_path)
        if submission.get("batch_job_name"):
            print(json.dumps(submission, indent=2))
            return

    _, local_rows = load_shard(args.output, args.shard)
    prompt_path, prompt_sha = prompt_for_shard(plan_data, args.shard)
    prompt = prompt_path.read_text(encoding="utf-8")
    upload_log = directory / "uploads.jsonl"
    uploads = load_uploads(upload_log)
    local_by_id = {row["piece_id"]: row for row in local_rows}
    if not set(uploads).issubset(local_by_id):
        raise RuntimeError("upload log contains IDs outside this shard")

    client = genai.Client(api_key=load_api_key())
    remaining = [row for row in local_rows if row["piece_id"] not in uploads]
    print(f"Uploading {len(remaining)} of {len(local_rows)} audio files...")
    with upload_log.open("a", encoding="utf-8") as log:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(upload_one, client, row): row["piece_id"]
                for row in remaining
            }
            completed = len(uploads)
            errors = []
            for future in as_completed(futures):
                try:
                    record = future.result()
                except Exception as error:
                    errors.append((futures[future], repr(error)))
                    print(
                        f"Upload failed for {futures[future]}: {error!r}",
                        flush=True,
                    )
                    continue
                log.write(json.dumps(record, separators=(",", ":")) + "\n")
                log.flush()
                os.fsync(log.fileno())
                uploads[record["piece_id"]] = record
                completed += 1
                if completed == len(local_rows) or completed % 25 == 0:
                    print(f"Uploaded {completed}/{len(local_rows)}", flush=True)
    if errors:
        raise RuntimeError(
            f"{len(errors)} uploads failed; rerun to resume missing IDs: {errors[:3]}"
        )

    if set(uploads) != set(local_by_id):
        raise RuntimeError("completed upload mapping does not exactly match shard IDs")
    if len({row["remote_name"] for row in uploads.values()}) != len(uploads):
        raise RuntimeError("duplicate remote file names in upload mapping")
    if len({row["remote_uri"] for row in uploads.values()}) != len(uploads):
        raise RuntimeError("duplicate remote URIs in upload mapping")

    request_path = directory / "requests.jsonl"
    request_body = "".join(
        json.dumps(
            build_request(row["piece_id"], uploads[row["piece_id"]], prompt),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
        for row in local_rows
    )
    write_text_atomic(request_path, request_body)
    validate_requests(request_path, local_rows, uploads, prompt)

    input_state_path = directory / "input_file.json"
    if input_state_path.is_file():
        input_state = read_json(input_state_path)
        if input_state["request_jsonl_sha256"] != sha256_text(request_body):
            raise RuntimeError("existing input file was built from different requests")
        uploaded_input_name = input_state["remote_name"]
    else:
        uploaded_input = client.files.upload(
            file=request_path,
            config=types.UploadFileConfig(
                display_name=f"gemini35-audio-shard-{args.shard:03d}",
                mime_type="jsonl",
            ),
        )
        input_state = {
            "shard_index": args.shard,
            "piece_count": len(local_rows),
            "request_jsonl": str(request_path.resolve()),
            "request_jsonl_sha256": sha256_text(request_body),
            "remote_name": uploaded_input.name,
            "remote_uri": uploaded_input.uri,
            "uploaded_at": utc_now(),
        }
        write_json_atomic(input_state_path, input_state)
        uploaded_input_name = uploaded_input.name

    batch_job = client.batches.create(
        model=plan_data["model"],
        src=uploaded_input_name,
        config={"display_name": f"piano-caption-shard-{args.shard:03d}"},
    )
    state = getattr(batch_job.state, "name", str(batch_job.state))
    submission = {
        "schema_version": 1,
        "shard_index": args.shard,
        "piece_count": len(local_rows),
        "first_piece_id": local_rows[0]["piece_id"],
        "last_piece_id": local_rows[-1]["piece_id"],
        "model": plan_data["model"],
        "prompt_sha256": prompt_sha,
        "request_jsonl_sha256": sha256_text(request_body),
        "input_file_name": uploaded_input_name,
        "batch_job_name": batch_job.name,
        "batch_job_state": state,
        "submitted_at": utc_now(),
    }
    write_json_atomic(submission_path, submission)
    print(json.dumps(submission, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--manifest", type=Path, default=MANIFEST)
    plan_parser.add_argument("--prompt", type=Path, default=PROMPT)
    plan_parser.add_argument("--model", default=MODEL)
    plan_parser.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    plan_parser.set_defaults(function=plan)

    submit_parser = subparsers.add_parser("submit")
    submit_parser.add_argument("--shard", type=int, required=True)
    submit_parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    submit_parser.set_defaults(function=submit)
    args = parser.parse_args()
    if getattr(args, "workers", 1) > MAX_WORKERS:
        parser.error(f"--workers cannot exceed {MAX_WORKERS}")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    parsed.function(parsed)
