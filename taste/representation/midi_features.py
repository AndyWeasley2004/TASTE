"""Performance-MIDI statistics: note reading, onset-based tempo estimation and the symbolic feature vector."""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import mido
import numpy as np


FEATURE_NAMES = (
    "duration_sec", "note_count", "notes_per_sec", "pitch_mean", "pitch_std",
    "pitch_range", "velocity_mean", "velocity_std", "velocity_range",
    "velocity_diff_mean", "velocity_diff_std", "duration_mean", "duration_std",
    "duration_median", "ioi_mean", "ioi_std", "ioi_cv", "ioi_diff_std",
    "simultaneous_ratio", "short_note_ratio", "long_note_ratio",
    "overlap_ratio", "mean_polyphony", "max_polyphony", "low_pitch_ratio",
    "high_pitch_ratio", "pitch_step_mean", "pitch_step_std", "tempo_bpm",
    "tempo_flexibility", "onset_density_std", "pedal_event_rate",
)


def read_performance(path: str | Path) -> tuple[list[dict], list[float]]:
    """Read notes in seconds while retaining performed timing and velocities."""
    midi = mido.MidiFile(path)
    tempo = 500_000
    elapsed = 0.0
    active: dict[tuple[int, int], list[tuple[float, int]]] = defaultdict(list)
    notes: list[dict] = []
    pedal_times: list[float] = []
    for message in mido.merge_tracks(midi.tracks):
        elapsed += mido.tick2second(message.time, midi.ticks_per_beat, tempo)
        if message.type == "set_tempo":
            tempo = message.tempo
        elif message.type == "control_change" and message.control == 64:
            pedal_times.append(elapsed)
        elif message.type == "note_on" and message.velocity > 0:
            active[(message.channel, message.note)].append((elapsed, message.velocity))
        elif message.type in {"note_off", "note_on"}:
            key = (message.channel, message.note)
            if not active[key]:
                continue
            onset, velocity = active[key].pop(0)
            if elapsed > onset:
                notes.append({
                    "onset": onset,
                    "offset": elapsed,
                    "duration": elapsed - onset,
                    "pitch": message.note,
                    "velocity": velocity,
                })
    notes.sort(key=lambda note: (note["onset"], note["pitch"]))
    return notes, pedal_times


def _scaled_bpm(ioi: float) -> float:
    bpm = 60.0 / ioi
    while bpm < 50.0:
        bpm *= 2.0
    while bpm > 180.0:
        bpm /= 2.0
    return bpm


def estimate_tempo_from_onsets(onsets: np.ndarray) -> float:
    """Dixon-style IOI clustering over the performed onsets; the MIDI tempo map is ignored."""
    unique = np.unique(np.round(onsets, 4))
    if len(unique) < 4:
        raise ValueError("not enough distinct onsets for tempo estimation")
    candidates: list[tuple[float, float]] = []
    for lag in range(1, min(9, len(unique))):
        intervals = unique[lag:] - unique[:-lag]
        for interval in intervals[(intervals >= 0.12) & (intervals <= 3.0)]:
            candidates.append((_scaled_bpm(float(interval)), 1.0 / math.sqrt(lag)))
    if not candidates:
        raise ValueError("no valid onset intervals for tempo estimation")
    bins = np.arange(49.5, 181.5, 1.0)
    histogram = np.zeros(len(bins) - 1, dtype=np.float64)
    for bpm, weight in candidates:
        index = int(np.clip(np.searchsorted(bins, bpm) - 1, 0, len(histogram) - 1))
        histogram[index] += weight
    smoothed = np.convolve(histogram, np.array([0.5, 1.0, 0.5]), mode="same")
    winner = int(smoothed.argmax())
    nearby = [
        (bpm, weight)
        for bpm, weight in candidates
        if abs(bpm - (bins[winner] + 0.5)) <= 2.0
    ]
    return float(np.average([bpm for bpm, _ in nearby], weights=[w for _, w in nearby]))


def tempo_class(bpm: float) -> str:
    return "slow" if bpm < 75.0 else "moderate" if bpm < 115.0 else "fast"


def estimate_performance_tempo(path: str | Path) -> tuple[float, str]:
    notes, _ = read_performance(path)
    bpm = estimate_tempo_from_onsets(np.array([note["onset"] for note in notes]))
    return bpm, tempo_class(bpm)


def _safe(values, operation, default=0.0) -> float:
    return float(operation(values)) if len(values) else float(default)


def extract_symbolic_features(path: str | Path) -> np.ndarray:
    notes, pedal_times = read_performance(path)
    if len(notes) < 4:
        raise ValueError("fewer than four completed notes")
    onset = np.array([note["onset"] for note in notes], dtype=np.float64)
    offset = np.array([note["offset"] for note in notes], dtype=np.float64)
    duration = offset - onset
    pitch = np.array([note["pitch"] for note in notes], dtype=np.float64)
    velocity = np.array([note["velocity"] for note in notes], dtype=np.float64)
    unique_onsets, onset_counts = np.unique(np.round(onset, 4), return_counts=True)
    ioi = np.diff(unique_onsets)
    positive_ioi = ioi[(ioi > 0.02) & (ioi < 5.0)]
    bpm = estimate_tempo_from_onsets(onset)
    performed_beat = 60.0 / bpm
    beat_ioi = positive_ioi / performed_beat if len(positive_ioi) else positive_ioi
    overlap = np.maximum(0.0, offset[:-1] - onset[1:])
    overlap_ratio = float(np.mean(overlap > 0)) if len(overlap) else 0.0
    events = sorted(
        [(value, 1) for value in onset] + [(value, -1) for value in offset],
        key=lambda item: (item[0], item[1]),
    )
    polyphony = current = 0
    polyphony_samples = []
    for _, delta in events:
        current += delta
        polyphony = max(polyphony, current)
        polyphony_samples.append(max(0, current))
    total_duration = max(float(offset.max() - onset.min()), 1e-6)
    density_bins = np.histogram(onset, bins=max(1, math.ceil(total_duration / 5.0)))[0]
    pitch_steps = np.diff(pitch)
    velocity_diff = np.diff(velocity)
    features = np.array([
        total_duration, len(notes), len(notes) / total_duration,
        pitch.mean(), pitch.std(), pitch.max() - pitch.min(),
        velocity.mean(), velocity.std(), velocity.max() - velocity.min(),
        _safe(np.abs(velocity_diff), np.mean), _safe(velocity_diff, np.std),
        duration.mean(), duration.std(), np.median(duration),
        _safe(positive_ioi, np.mean), _safe(positive_ioi, np.std),
        _safe(positive_ioi, np.std) / max(_safe(positive_ioi, np.mean), 1e-6),
        _safe(np.diff(positive_ioi), np.std),
        float(np.mean(onset_counts > 1)), float(np.mean(duration < 0.15)),
        float(np.mean(duration > 1.0)), overlap_ratio,
        _safe(np.array(polyphony_samples), np.mean), polyphony,
        float(np.mean(pitch < 48)), float(np.mean(pitch > 72)),
        _safe(np.abs(pitch_steps), np.mean), _safe(pitch_steps, np.std), bpm,
        _safe(beat_ioi, np.std), _safe(density_bins, np.std),
        len(pedal_times) / total_duration,
    ], dtype=np.float32)
    if len(features) != len(FEATURE_NAMES) or not np.isfinite(features).all():
        raise ValueError("invalid symbolic feature vector")
    return features
