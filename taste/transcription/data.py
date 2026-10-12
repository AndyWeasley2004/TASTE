"""Minimal Transkun inference data types and MIDI serialization."""

from collections import defaultdict

import mido


class Note:
    def __init__(self, start, end, pitch, velocity, hasOnset=True, hasOffset=True):
        self.start = start
        self.end = end
        self.pitch = pitch
        self.velocity = velocity
        self.hasOnset = hasOnset
        self.hasOffset = hasOffset

    def __repr__(self):
        return str(self.__dict__)


def resolveOverlapping(note_events):
    """Resolve overlapping estimates of the same MIDI pitch."""
    resolved = []
    previous_index = {}
    for event in sorted(note_events, key=lambda item: (item.start, item.end, item.pitch)):
        if event.pitch in previous_index:
            previous = resolved[previous_index[event.pitch]]
            if previous.end > event.start:
                previous.end = event.start
        previous_index[event.pitch] = len(resolved)
        resolved.append(event)
    resolved = [event for event in resolved if event.start < event.end]
    validateNotes(resolved)
    return sorted(resolved, key=lambda event: (event.start, event.end, event.pitch))


def validateNotes(notes):
    by_pitch = defaultdict(list)
    for note in sorted(notes, key=lambda item: (item.pitch, item.start, item.end)):
        if note.end <= note.start:
            raise ValueError(f"note does not have positive duration: {note}")
        if note.pitch > 0 and not 0 <= note.velocity <= 127:
            raise ValueError(f"invalid MIDI velocity: {note.velocity}")
        if by_pitch[note.pitch] and note.start < by_pitch[note.pitch][-1].end:
            raise ValueError(f"overlapping events for pitch {note.pitch}")
        by_pitch[note.pitch].append(note)


class _MidiOutput:
    def __init__(self, midi):
        self.midi = midi

    def write(self, path):
        self.midi.save(path)


def writeMidi(notes, resolution=960):
    """Serialize Transkun's second-based note events without FluidSynth."""
    validateNotes(notes)
    tempo = 500_000  # 120 BPM; preserves seconds after tick conversion.
    ticks_per_second = resolution * 1_000_000 / tempo
    midi = mido.MidiFile(ticks_per_beat=resolution)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("track_name", name="Transkun piano", time=0))
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))

    events = []
    for note in notes:
        start = max(0, round(note.start * ticks_per_second))
        end = max(start + 1, round(note.end * ticks_per_second))
        velocity = max(1, min(127, round(note.velocity)))
        if note.pitch > 0:
            events.append((start, 2, mido.Message(
                "note_on", note=int(note.pitch), velocity=velocity, time=0)))
            events.append((end, 1, mido.Message(
                "note_off", note=int(note.pitch), velocity=0, time=0)))
        else:
            control = int(-note.pitch)
            events.append((start, 0, mido.Message(
                "control_change", control=control, value=velocity, time=0)))
            events.append((end, 0, mido.Message(
                "control_change", control=control, value=0, time=0)))

    previous_tick = 0
    for tick, _, message in sorted(events, key=lambda event: (event[0], event[1])):
        message.time = tick - previous_tick
        track.append(message)
        previous_tick = tick
    track.append(mido.MetaMessage("end_of_track", time=0))
    return _MidiOutput(midi)
