# Uses the Pop2Piano source index: https://github.com/sweetcocoa/pop2piano.
# Audio acquisition uses yt-dlp and FFmpeg; the download wrapper is local code.
"""Download the piano-cover audio of the Pop2Piano index and the original song's YouTube metadata."""

import argparse
import glob
import json
import multiprocessing
import os
import re
import shutil
import subprocess
import tempfile

import pandas as pd
from joblib import Parallel, delayed
from omegaconf import OmegaConf
from tqdm import tqdm


DOWNLOADER = (
    shutil.which("yt-dlp")
    or shutil.which("youtube-dl")
)
if DOWNLOADER is None:
    raise RuntimeError("Install yt-dlp (recommended) or youtube-dl before downloading")


def title_from_cover(cover_title):
    """Best-effort fallback when the original video's metadata is unavailable."""
    return re.sub(
        r"\s*[|｜:–—-]?\s*(piano (cover|arrangement|tutorial|version).*|sheet music.*)$",
        "",
        cover_title,
        flags=re.IGNORECASE,
    ).strip()


def _cookie_args(cookies):
    return ["--cookies", cookies] if cookies else []


def retrieve_song_metadata(pop_id, cover_title, cookies=None):
    """Retrieve compact metadata for the original song video without downloading it."""
    command = [
        DOWNLOADER,
        "--skip-download",
        "--no-playlist",
        "--retries",
        "3",
        "--socket-timeout",
        "15",
        "--sleep-requests",
        "1",
        "--remote-components",
        "ejs:github",
        "--dump-single-json",
    ] + _cookie_args(cookies) + [f"https://www.youtube.com/watch?v={pop_id}"]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=90, check=False
        )
        if result.returncode == 0 and result.stdout.strip():
            info = json.loads(result.stdout)
            return {
                "title": info.get("title"), "artist": info.get("artist") or info.get("uploader"),
                "upload_date": info.get("upload_date"), "title_source": "youtube",
            }
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        pass
    return {"title": title_from_cover(cover_title), "title_source": "piano_title_fallback"}


def download_piano(piano_id, pop_id, output_dir, cookies=None):
    """Download one piano cover and store it with metadata in its own folder."""
    song_dir = os.path.join(output_dir, piano_id)
    audio_path = os.path.join(song_dir, "piano.wav")
    metadata_path = os.path.join(song_dir, "metadata.yaml")

    if os.path.isfile(audio_path) and os.path.isfile(metadata_path):
        return "skipped"

    with tempfile.TemporaryDirectory() as tmpdir:
        output_template = os.path.join(tmpdir, "%(id)s.%(ext)s")
        command = [
            DOWNLOADER,
            "-o",
            output_template,
            "--format",
            "bestaudio",
            "--extract-audio",
            "--audio-quality",
            "0",
            "--audio-format",
            "wav",
            "--retries",
            "10",
            "--sleep-requests",
            "1",
            "--postprocessor-args",
            "ffmpeg:-ac 1 -ar 16000",
            "--force-ipv4",
            "--no-playlist",
            "--ignore-errors",
            "--write-info-json",
            "--remote-components",
            "ejs:github",
        ] + _cookie_args(cookies) + [f"https://www.youtube.com/watch?v={piano_id}"]
        subprocess.run(command, check=False)

        audio_files = glob.glob(os.path.join(tmpdir, "*.wav"))
        info_files = glob.glob(os.path.join(tmpdir, "*.info.json"))
        if not audio_files or not info_files:
            return "failed"

        with open(info_files[0], encoding="utf-8") as info_file:
            info = json.load(info_file)
        song = retrieve_song_metadata(pop_id, info["title"], cookies)
        song["ytid"] = pop_id

        metadata = OmegaConf.create(
            {
                "piano": {
                    "uploader": info.get("uploader"),
                    "title": info["title"],
                    "ytid": piano_id,
                    "duration": int(info["duration"]),
                },
                "song": song,
            }
        )

        os.makedirs(song_dir, exist_ok=True)
        shutil.move(audio_files[0], audio_path)
        OmegaConf.save(metadata, metadata_path)
        return "downloaded"


def download_all(df, output_dir, workers=None, cookies=None):
    os.makedirs(output_dir, exist_ok=True)
    workers = workers or min(4, multiprocessing.cpu_count())
    rows = list(df[["piano_ids", "pop_ids"]].itertuples(index=False, name=None))
    results = Parallel(n_jobs=workers, prefer="threads")(
        delayed(download_piano)(piano_id, pop_id, output_dir, cookies)
        for piano_id, pop_id in tqdm(rows, desc="Queueing piano covers")
    )
    for status in sorted(set(results)):
        print(f"{status}: {results.count(status)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pop2Piano piano-cover downloader")
    parser.add_argument("dataset", help="CSV containing piano_ids and pop_ids")
    parser.add_argument("output_dir", help="destination directory")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--cookies", help="Netscape-format cookies.txt for yt-dlp")
    args = parser.parse_args()

    dataframe = pd.read_csv(args.dataset)
    download_all(dataframe, args.output_dir, args.workers, args.cookies)
