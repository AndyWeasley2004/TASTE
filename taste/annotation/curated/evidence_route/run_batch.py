"""Run the evidence route over every pending curated work, spreading workers across local model-server replicas."""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
from queue import Empty
from pathlib import Path
from typing import Any

from taste.annotation.curated.common import CURATED_ROOT, read_parquet_rows, write_json
from taste.annotation.curated.evidence_route.run import (
    DEFAULT_CONFIG, DEFAULT_PROMPT, JinaUsageExceeded, annotate_agentic_work,
    rebuild_training_manifest, normalize_annotation,
)

REPO = Path(__file__).resolve().parents[4]


def pending_work_ids(root: Path) -> tuple[list[str], int]:
    work_ids = sorted(
        row["piece_id"]
        for row in read_parquet_rows(root / "manifests/pieces.parquet")
        if row.get("labeling_route") == "online_search"
    )
    pending = [
        work_id for work_id in work_ids
        if not (
            (root / "pieces" / work_id / "meta_archive" / "agent_trace.json").exists()
            and (root / "pieces" / work_id / "label.json").exists()
        )
    ]
    return pending, len(work_ids) - len(pending)


def revision_work_ids(root: Path, revisions: set[str]) -> list[str]:
    selected = []
    for trace_path in sorted((root / "pieces").glob("piece_*/meta_archive/agent_trace.json")):
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        if trace.get("revision") in revisions:
            selected.append(trace_path.parent.parent.name)
    return selected


def archive_revision_targets(
    root: Path,
    archive_dir: Path,
    revisions: set[str],
) -> list[str]:
    manifest_path = archive_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if set(manifest.get("source_revisions", [])) != revisions:
            raise ValueError("archive revision filter does not match this run")
        return manifest["piece_ids"]

    piece_ids = revision_work_ids(root, revisions)
    if not piece_ids:
        raise ValueError("revision filter selected no pieces")
    archive_dir.mkdir(parents=True, exist_ok=False)
    for piece_id in piece_ids:
        source = root / "pieces" / piece_id
        destination = archive_dir / "pieces" / piece_id
        (destination / "meta_archive").mkdir(parents=True)
        shutil.copy2(source / "meta_archive" / "agent_trace.json",
                     destination / "meta_archive" / "agent_trace.json")
        shutil.copy2(source / "label.json", destination / "label.json")
    write_json(manifest_path, {
        "piece_count": len(piece_ids),
        "piece_ids": piece_ids,
        "source_revisions": sorted(revisions),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })
    return piece_ids


def restore_annotations(root: Path, config: dict[str, Any]) -> int:
    restored = 0
    for trace_path in root.glob("pieces/*/meta_archive/agent_trace.json"):
        label_path = trace_path.parent.parent / "label.json"
        if label_path.exists():
            continue
        trace = json.loads(trace_path.read_text())
        metadata = json.loads((trace_path.parent / "metadata.json").read_text())
        write_json(label_path, normalize_annotation(trace.get("annotation"), metadata, config))
        restored += 1
    return restored


def run_worker(
    worker_name: str,
    base_url: str,
    work_queue,
    root: Path,
    config: dict[str, Any],
    prompt: str,
    log_dir: Path,
    progress,
    lock,
    stop_event,
    overwrite: bool,
    archive_dir: Path | None,
) -> None:
    logging.getLogger("qwen_agent_logger").setLevel(logging.WARNING)
    worker_config = dict(config)
    worker_config["base_url"] = base_url
    log_path = log_dir / f"{worker_name}.log"
    with log_path.open("a", encoding="utf-8") as log:
        while True:
            if stop_event.is_set():
                break
            try:
                work_id = work_queue.get(timeout=1)
            except Empty:
                break
            started = time.time()
            returncode = 1
            usage_stopped = False
            try:
                annotate_agentic_work(
                    root, worker_config, prompt, work_id, overwrite=overwrite,
                )
                returncode = 0
            except JinaUsageExceeded:
                usage_stopped = True
                print(f"{work_id}: Jina API usage quota exceeded", file=log)
                log.flush()
            except Exception:
                print(f"{work_id}:", file=log)
                traceback.print_exc(file=log)
                log.flush()
            if returncode != 0 and archive_dir is not None:
                archived_label = archive_dir / "pieces" / work_id / "label.json"
                if archived_label.is_file():
                    shutil.copy2(archived_label, root / "pieces" / work_id / "label.json")
            if usage_stopped:
                stop_event.set()
            with lock:
                status = "completed" if returncode == 0 else "failed"
                progress[status] = progress[status] + 1
                progress["last_update"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                if usage_stopped:
                    progress["stop_reason"] = "jina_api_usage_exceeded"
                    progress["stop_work_id"] = work_id
                workers = dict(progress["workers"])
                workers[worker_name] = {
                    "base_url": base_url,
                    "last_work_id": work_id,
                    "last_returncode": returncode,
                    "last_elapsed_seconds": round(time.time() - started, 3),
                }
                progress["workers"] = workers
                processed = progress["completed"] + progress["failed"]
                if returncode != 0 or processed % 10 == 0 or processed == progress["queued"]:
                    (log_dir / "progress.json").write_text(
                        json.dumps(dict(progress), ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                if processed % 25 == 0 or processed == progress["queued"]:
                    print(
                        f"[{processed}/{progress['queued']}] "
                        f"ok={progress['completed']} failed={progress['failed']}",
                        flush=True,
                    )
            work_queue.task_done()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=CURATED_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--base-urls", nargs="+", default=["http://127.0.0.1:8000/v1", "http://127.0.0.1:8001/v1"])
    parser.add_argument("--workers-per-server", type=int, default=4)
    parser.add_argument("--log-dir", type=Path, default=REPO / "runs/curated_evidence_route")
    parser.add_argument("--restore-only", action="store_true")
    parser.add_argument(
        "--trace-revisions", nargs="+",
        help="Rerun only traces with these exact revisions",
    )
    parser.add_argument(
        "--archive-dir", type=Path,
        help="Archive source labels/traces before a revision-filtered rerun",
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    prompt = args.prompt_file.read_text()
    restored = restore_annotations(args.root, config)
    if args.restore_only:
        print(json.dumps({"restored_annotations": restored}, indent=2))
        return
    if args.trace_revisions:
        if args.archive_dir is None:
            parser.error("--archive-dir is required with --trace-revisions")
        revisions = set(args.trace_revisions)
        target_ids = archive_revision_targets(args.root, args.archive_dir, revisions)
        work_ids = []
        for work_id in target_ids:
            piece_dir = args.root / "pieces" / work_id
            try:
                trace = json.loads((piece_dir / "meta_archive" / "agent_trace.json").read_text(encoding="utf-8"))
                label = json.loads((piece_dir / "label.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                work_ids.append(work_id)
                continue
            if not (
                trace.get("revision") == config["revision"]
                and label.get("revision") == config["revision"]
            ):
                work_ids.append(work_id)
        already_complete = len(target_ids) - len(work_ids)
    else:
        work_ids, already_complete = pending_work_ids(args.root)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    worker_specs: list[tuple[str, str]] = []
    worker_count = len(args.base_urls) * args.workers_per_server
    for server_index, base_url in enumerate(args.base_urls):
        for local_index in range(args.workers_per_server):
            worker_specs.append((f"server{server_index}_worker{local_index}", base_url))

    initial_progress: dict[str, Any] = {
        "total_manifest_works": already_complete + len(work_ids),
        "already_complete": already_complete,
        "queued": len(work_ids),
        "completed": 0,
        "failed": 0,
        "trace_revisions": args.trace_revisions or [],
        "archive_dir": str(args.archive_dir) if args.archive_dir else "",
        "workers": {},
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with Manager() as manager:
        work_queue = manager.Queue()
        for work_id in work_ids:
            work_queue.put(work_id)
        progress = manager.dict(initial_progress)
        lock = manager.Lock()
        stop_event = manager.Event()
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    run_worker, name, url, work_queue,
                    args.root, config, prompt, args.log_dir, progress, lock, stop_event,
                    bool(args.trace_revisions), args.archive_dir,
                )
                for name, url in worker_specs
            ]
            for future in as_completed(futures):
                future.result()

        progress = dict(progress)

    progress["training_manifest"] = rebuild_training_manifest(args.root)
    progress["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    (args.log_dir / "progress.json").write_text(
        json.dumps(progress, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(progress, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
