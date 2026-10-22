# Uses PIAST-YT source IDs from https://github.com/Hayeonbang/PIAST.
# Metadata acquisition uses yt-dlp; this wrapper is local code.
"""Retrieve YouTube metadata (no audio) for the official PIAST-YT video IDs."""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from taste.annotation.curated.common import CURATED_ROOT


def official_video_ids(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = csv.DictReader(handle)
        return sorted({row["video_id"].strip()[:11] for row in rows if len(row["video_id"].strip()) >= 11})


def retrieve(video_id: str, output_dir: Path, downloader: str, cookies: Path | None) -> str:
    output = output_dir / f"{video_id}.json"
    if output.exists():
        return "skipped"
    command = [
        downloader, "--skip-download", "--no-playlist", "--dump-single-json",
        "--socket-timeout", "20", "--retries", "3", "--sleep-requests", "1",
        "--remote-components", "ejs:github",
    ]
    if cookies:
        command += ["--cookies", str(cookies)]
    command.append(f"https://www.youtube.com/watch?v={video_id}")
    result = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
    if result.returncode != 0 or not result.stdout.strip():
        return "failed"
    try:
        raw = json.loads(result.stdout)
    except json.JSONDecodeError:
        return "failed"
    metadata = {
        "video_id": video_id,
        "title": raw.get("title"),
        "channel": raw.get("channel") or raw.get("uploader"),
        "upload_date": raw.get("upload_date"),
        "description": raw.get("description"),
        "webpage_url": raw.get("webpage_url") or f"https://www.youtube.com/watch?v={video_id}",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{video_id}.", suffix=".json", dir=output_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(tmp_name, output)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    return "downloaded"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=CURATED_ROOT)
    parser.add_argument("--id-list", type=Path)
    parser.add_argument("--cookies", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise SystemExit("--workers must be between 1 and 8")
    source = args.root / "sources/piast_yt/raw"
    id_list = args.id_list or source / "yt_ids.csv"
    output_dir = source / "youtube_metadata"
    downloader = shutil.which("yt-dlp")
    if not downloader:
        raise SystemExit("yt-dlp is not installed")
    video_ids = official_video_ids(id_list)
    counts = {"downloaded": 0, "skipped": 0, "failed": 0}
    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(retrieve, video_id, output_dir, downloader, args.cookies): video_id
            for video_id in video_ids
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="PIAST metadata"):
            video_id = futures[future]
            try:
                status = future.result()
            except Exception:
                status = "failed"
            counts[status] += 1
            if status == "failed":
                failed.append(video_id)
    (source / "youtube_metadata_failures.txt").write_text("".join(x + "\n" for x in failed))
    print(json.dumps({**counts, "total": len(video_ids)}, indent=2))


if __name__ == "__main__":
    main()
