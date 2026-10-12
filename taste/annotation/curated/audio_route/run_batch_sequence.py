#!/usr/bin/env python3
"""Audio route: submit, retrieve, store and cloud-clean the planned Gemini caption shards one after another."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from taste.annotation.curated.audio_route import prepare_batches as PREPARE


OUTPUT = PREPARE.OUTPUT
CAPTION_FILENAME = "gemini35_flash_caption.txt"
TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}
RETRY_TOKEN_LIMITS = (32_768, 65_536)
DELETE_WORKERS = 4


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def jsonl_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def api_client() -> Any:
    from google import genai

    return genai.Client(api_key=PREPARE.load_api_key())


def wait_for_job(client: Any, name: str, poll_seconds: int) -> Any:
    last_state = None
    while True:
        job = client.batches.get(name=name)
        state = job.state.name
        if state != last_state:
            print(f"{PREPARE.utc_now()} {name}: {state}", flush=True)
            last_state = state
        if state in TERMINAL_STATES:
            if state != "JOB_STATE_SUCCEEDED":
                raise RuntimeError(f"{name} ended in {state}: {job.error}")
            return job
        time.sleep(poll_seconds)


def parse_results(
    data: bytes,
    expected_ids: list[str],
) -> tuple[dict[str, str], dict[str, dict[str, Any]], list[str]]:
    rows = [
        json.loads(line)
        for line in data.decode("utf-8").splitlines()
        if line.strip()
    ]
    keys = [row.get("key") for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Gemini result contains duplicate response keys")
    missing = set(expected_ids) - set(keys)
    unexpected = set(keys) - set(expected_ids)
    if missing or unexpected or len(rows) != len(expected_ids):
        raise RuntimeError(
            f"result key mismatch: missing={len(missing)}, "
            f"unexpected={len(unexpected)}, rows={len(rows)}, "
            f"expected={len(expected_ids)}"
        )

    captions: dict[str, str] = {}
    metadata: dict[str, dict[str, Any]] = {}
    needs_retry: list[str] = []
    for row in rows:
        piece_id = row["key"]
        if row.get("error"):
            needs_retry.append(piece_id)
            continue
        response = row.get("response") or {}
        candidates = response.get("candidates") or []
        if not candidates:
            needs_retry.append(piece_id)
            continue
        candidate = candidates[0]
        text = "".join(
            part.get("text", "")
            for part in candidate.get("content", {}).get("parts", [])
        ).strip()
        if not text:
            needs_retry.append(piece_id)
            continue
        finish_reason = candidate.get("finishReason", "UNKNOWN")
        if finish_reason != "STOP":
            needs_retry.append(piece_id)
        captions[piece_id] = text + "\n"
        metadata[piece_id] = {
            "finish_reason": finish_reason,
            "model_version": response.get("modelVersion"),
            "response_id": response.get("responseId"),
        }
    return captions, metadata, needs_retry


def write_request_jsonl(
    path: Path,
    piece_ids: list[str],
    uploads: dict[str, dict[str, Any]],
    prompt: str,
    max_output_tokens: int,
) -> str:
    body = ""
    for piece_id in piece_ids:
        request = PREPARE.build_request(piece_id, uploads[piece_id], prompt)
        request["request"]["generationConfig"]["maxOutputTokens"] = max_output_tokens
        body += json.dumps(request, separators=(",", ":")) + "\n"
    PREPARE.write_text_atomic(path, body)
    if [row["key"] for row in jsonl_rows(path)] != piece_ids:
        raise RuntimeError(f"{path}: retry request key/order mismatch")
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def retry_truncated(
    client: Any,
    directory: Path,
    shard_index: int,
    piece_ids: list[str],
    uploads: dict[str, dict[str, Any]],
    prompt: str,
    poll_seconds: int,
) -> tuple[dict[str, str], dict[str, dict[str, Any]], list[dict[str, str]]]:
    from google.genai import types

    remaining = piece_ids
    replacements: dict[str, str] = {}
    replacement_meta: dict[str, dict[str, Any]] = {}
    cloud_resources: list[dict[str, str]] = []
    for attempt, token_limit in enumerate(RETRY_TOKEN_LIMITS, 1):
        if not remaining:
            break
        request_path = directory / f"retry_{attempt:02d}_requests.jsonl"
        request_sha = write_request_jsonl(
            request_path,
            remaining,
            uploads,
            prompt,
            token_limit,
        )
        uploaded_input = client.files.upload(
            file=request_path,
            config=types.UploadFileConfig(
                display_name=f"gemini35-shard-{shard_index:03d}-retry-{attempt:02d}",
                mime_type="jsonl",
            ),
        )
        job = client.batches.create(
            model=PREPARE.MODEL,
            src=uploaded_input.name,
            config={
                "display_name": (
                    f"piano-caption-shard-{shard_index:03d}-retry-{attempt:02d}"
                )
            },
        )
        submission = {
            "attempt": attempt,
            "piece_count": len(remaining),
            "max_output_tokens": token_limit,
            "request_sha256": request_sha,
            "input_file_name": uploaded_input.name,
            "batch_job_name": job.name,
            "submitted_at": PREPARE.utc_now(),
        }
        PREPARE.write_json_atomic(
            directory / f"retry_{attempt:02d}_submission.json",
            submission,
        )
        job = wait_for_job(client, job.name, poll_seconds)
        result_name = job.dest.file_name
        data = client.files.download(file=result_name)
        atomic_bytes(directory / f"retry_{attempt:02d}_results.jsonl", data)
        retry_captions, retry_meta, still_pending = parse_results(data, remaining)
        cloud_resources.append({
            "batch_job_name": job.name,
            "input_file_name": uploaded_input.name,
            "result_file_name": result_name,
        })
        for piece_id, caption in retry_captions.items():
            replacements[piece_id] = caption
            replacement_meta[piece_id] = {
                **retry_meta[piece_id],
                "retry_attempt": attempt,
                "max_output_tokens": token_limit,
            }
        remaining = still_pending

    if remaining:
        raise RuntimeError(
            f"{len(remaining)} captions still truncated at "
            f"{RETRY_TOKEN_LIMITS[-1]} tokens"
        )
    return replacements, replacement_meta, cloud_resources


def verify_stored_captions(directory: Path, local_rows: list[dict[str, Any]]) -> None:
    index_rows = jsonl_rows(directory / "caption_index.jsonl")
    if len(index_rows) != len(local_rows):
        raise RuntimeError("caption index count mismatch")
    expected_ids = [row["piece_id"] for row in local_rows]
    if [row["piece_id"] for row in index_rows] != expected_ids:
        raise RuntimeError("caption index piece-ID order mismatch")
    for row, indexed in zip(local_rows, index_rows):
        caption_path = Path(row["audio_path"]).parent / CAPTION_FILENAME
        if str(caption_path) != indexed["caption_path"]:
            raise RuntimeError(f"{row['piece_id']}: caption path mismatch")
        data = caption_path.read_bytes()
        if not data or hashlib.sha256(data).hexdigest() != indexed["caption_sha256"]:
            raise RuntimeError(f"{row['piece_id']}: caption hash mismatch")


def retrieve_and_store(
    client: Any,
    output: Path,
    shard_index: int,
    poll_seconds: int,
) -> dict[str, Any]:
    directory = PREPARE.shard_dir(output, shard_index)
    _, local_rows = PREPARE.load_shard(output, shard_index)
    expected_ids = [row["piece_id"] for row in local_rows]
    submission = PREPARE.read_json(directory / "submission.json")
    job = wait_for_job(client, submission["batch_job_name"], poll_seconds)

    result_path = directory / "results.jsonl"
    if result_path.is_file():
        data = result_path.read_bytes()
    else:
        data = client.files.download(file=job.dest.file_name)
        atomic_bytes(result_path, data)
    captions, metadata, needs_retry = parse_results(data, expected_ids)

    uploads = PREPARE.load_uploads(directory / "uploads.jsonl")
    if set(uploads) != set(expected_ids):
        raise RuntimeError(f"shard {shard_index}: upload/expected ID mismatch")
    retry_resources: list[dict[str, str]] = []
    if needs_retry:
        print(
            f"Shard {shard_index}: retrying {len(needs_retry)} incomplete captions",
            flush=True,
        )
        prompt_path, _ = PREPARE.prompt_for_shard(
            PREPARE.load_plan(output),
            shard_index,
        )
        prompt = prompt_path.read_text(encoding="utf-8")
        replacements, replacement_meta, retry_resources = retry_truncated(
            client,
            directory,
            shard_index,
            needs_retry,
            uploads,
            prompt,
            poll_seconds,
        )
        captions.update(replacements)
        metadata.update(replacement_meta)
    if set(captions) != set(expected_ids):
        raise RuntimeError(f"shard {shard_index}: final caption/ID set mismatch")

    index_rows = []
    for row in local_rows:
        piece_id = row["piece_id"]
        caption_path = Path(row["audio_path"]).parent / CAPTION_FILENAME
        PREPARE.write_text_atomic(caption_path, captions[piece_id])
        caption_bytes = caption_path.read_bytes()
        index_rows.append({
            "piece_id": piece_id,
            "caption_path": str(caption_path),
            "caption_sha256": hashlib.sha256(caption_bytes).hexdigest(),
            "caption_chars": len(captions[piece_id].rstrip("\n")),
            **metadata[piece_id],
        })
    index_body = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in index_rows
    )
    PREPARE.write_text_atomic(directory / "caption_index.jsonl", index_body)
    verify_stored_captions(directory, local_rows)

    report = {
        "schema_version": 1,
        "retrieved_at": PREPARE.utc_now(),
        "shard_index": shard_index,
        "piece_count": len(expected_ids),
        "result_file_name": job.dest.file_name,
        "result_sha256": hashlib.sha256(data).hexdigest(),
        "unique_response_keys": len(captions),
        "missing_piece_ids": 0,
        "unexpected_piece_ids": 0,
        "empty_captions": 0,
        "retried_incomplete_captions": len(needs_retry),
        "caption_filename": CAPTION_FILENAME,
        "caption_index": str((directory / "caption_index.jsonl").resolve()),
        "retry_cloud_resources": retry_resources,
    }
    PREPARE.write_json_atomic(directory / "retrieval.json", report)
    return report


def delete_file_if_present(client: Any, name: str) -> None:
    for attempt in range(5):
        try:
            client.files.delete(name=name)
            return
        except Exception as error:
            message = str(error)
            absent = (
                "404" in message
                or "NOT_FOUND" in message
                or ("403" in message and "may not exist" in message)
            )
            if absent:
                return
            transient = any(
                code in message for code in ("429", "500", "502", "503", "504")
            )
            if not transient or attempt == 4:
                raise
            time.sleep(2**attempt)


def delete_batch_if_present(client: Any, name: str) -> None:
    for attempt in range(5):
        try:
            client.batches.delete(name=name)
            return
        except Exception as error:
            message = str(error)
            absent = (
                "404" in message
                or "NOT_FOUND" in message
                or ("403" in message and "may not exist" in message)
            )
            if absent:
                return
            transient = any(
                code in message for code in ("429", "500", "502", "503", "504")
            )
            if not transient or attempt == 4:
                raise
            time.sleep(2**attempt)


def cleanup_cloud(client: Any, output: Path, shard_index: int) -> dict[str, Any]:
    directory = PREPARE.shard_dir(output, shard_index)
    retrieval = PREPARE.read_json(directory / "retrieval.json")
    submission = PREPARE.read_json(directory / "submission.json")
    uploads = PREPARE.load_uploads(directory / "uploads.jsonl")
    local_ids = set(uploads)

    batch_names = [submission["batch_job_name"]]
    file_names = {
        submission["input_file_name"],
        *(row["remote_name"] for row in uploads.values()),
    }
    batch_result_files = [retrieval["result_file_name"]]
    for resource in retrieval["retry_cloud_resources"]:
        batch_names.append(resource["batch_job_name"])
        file_names.add(resource["input_file_name"])
        batch_result_files.append(resource["result_file_name"])

    for name in batch_names:
        delete_batch_if_present(client, name)
    with ThreadPoolExecutor(max_workers=DELETE_WORKERS) as executor:
        list(executor.map(lambda name: delete_file_if_present(client, name), file_names))

    # Remove only unreferenced retry artifacts bearing this shard's piece IDs.
    for _ in range(3):
        remote = list(client.files.list(config={"page_size": 100}))
        extras = [item for item in remote if item.display_name in local_ids]
        if not extras:
            break
        with ThreadPoolExecutor(max_workers=DELETE_WORKERS) as executor:
            list(executor.map(
                lambda item: delete_file_if_present(client, item.name),
                extras,
            ))
        time.sleep(5)
    remote = list(client.files.list(config={"page_size": 100}))
    remaining_ids = sorted({
        item.display_name for item in remote if item.display_name in local_ids
    })
    remaining_names = sorted(
        name for name in file_names if any(item.name == name for item in remote)
    )
    if remaining_ids or remaining_names:
        raise RuntimeError(
            f"shard {shard_index}: cloud cleanup incomplete: "
            f"piece IDs={len(remaining_ids)}, explicit files={len(remaining_names)}"
        )

    report = {
        "cleaned_at": PREPARE.utc_now(),
        "shard_index": shard_index,
        "deleted_batch_jobs": batch_names,
        "deleted_explicit_files": len(file_names),
        "batch_result_files_removed_with_jobs": batch_result_files,
        "remaining_piece_files": 0,
        "remaining_explicit_files": 0,
    }
    PREPARE.write_json_atomic(directory / "cloud_cleanup.json", report)
    return report


def process_shard(
    client: Any,
    output: Path,
    shard_index: int,
    workers: int,
    poll_seconds: int,
) -> None:
    directory = PREPARE.shard_dir(output, shard_index)
    retrieval_path = directory / "retrieval.json"
    cleanup_path = directory / "cloud_cleanup.json"
    _, local_rows = PREPARE.load_shard(output, shard_index)

    if retrieval_path.is_file():
        verify_stored_captions(directory, local_rows)
    else:
        if not (directory / "submission.json").is_file():
            PREPARE.submit(SimpleNamespace(
                output=output,
                shard=shard_index,
                workers=workers,
            ))
        retrieve_and_store(client, output, shard_index, poll_seconds)
    if not cleanup_path.is_file():
        cleanup_cloud(client, output, shard_index)
    print(f"Shard {shard_index} complete and cloud-clean", flush=True)


def final_verify(output: Path) -> dict[str, Any]:
    plan = PREPARE.load_plan(output)
    all_ids = []
    all_caption_paths = []
    for shard_index in range(plan["shard_count"]):
        directory = PREPARE.shard_dir(output, shard_index)
        _, rows = PREPARE.load_shard(output, shard_index)
        if not (directory / "retrieval.json").is_file():
            continue
        verify_stored_captions(directory, rows)
        all_ids.extend(row["piece_id"] for row in rows)
        all_caption_paths.extend(
            str(Path(row["audio_path"]).parent / CAPTION_FILENAME)
            for row in rows
        )
    report = {
        "verified_at": PREPARE.utc_now(),
        "expected_pieces": plan["piece_count"],
        "captioned_pieces": len(all_ids),
        "unique_piece_ids": len(set(all_ids)),
        "unique_caption_paths": len(set(all_caption_paths)),
        "missing_pieces": plan["piece_count"] - len(all_ids),
        "complete": len(all_ids) == plan["piece_count"],
    }
    PREPARE.write_json_atomic(output / "caption_verification.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--start-shard", type=int, default=0)
    parser.add_argument("--end-shard", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--poll-seconds", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("--workers must be in [1, 8]")
    PREPARE.validate_plan(args.output)
    plan = PREPARE.load_plan(args.output)
    end = plan["shard_count"] - 1 if args.end_shard is None else args.end_shard
    if not 0 <= args.start_shard <= end < plan["shard_count"]:
        raise ValueError("invalid shard range")
    client = api_client()
    for shard_index in range(args.start_shard, end + 1):
        process_shard(
            client,
            args.output,
            shard_index,
            args.workers,
            args.poll_seconds,
        )
    print(json.dumps(final_verify(args.output), indent=2), flush=True)


if __name__ == "__main__":
    main()
