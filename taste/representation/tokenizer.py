"""Piano MIDI tokenizer: Aria's absolute tokenizer on a 25 ms grid with 5-second windows.

Pedal events are explicit and ordered before simultaneous notes, velocities use
12 non-zero bins, and eight section-state tokens ``<SEC_A>``-``<SEC_H>`` are
appended to the vocabulary.
"""
from __future__ import annotations

import random
from pathlib import Path

from ariautils.midi import MidiDict, get_duration_ms
from ariautils.tokenizer import AbsTokenizer


CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "tokenizer_piano_25ms.json"
SECTION_TOKENS = tuple(f"<SEC_{letter}>" for letter in "ABCDEFGH")
_TOKENIZER: "PianoMidiTokenizer | None" = None


class PianoMidiTokenizer(AbsTokenizer):
    """Piano-only AbsTokenizer with pedal-first ordering."""

    def __init__(self, config_path: str | Path = CONFIG_PATH) -> None:
        super().__init__(config_path=str(config_path))
        self.base_vocab_size = self.vocab_size
        self.add_tokens_to_vocab(SECTION_TOKENS)
        self.start_id = self.tok_to_id[self.bos_tok]
        self.end_id = self.tok_to_id[self.eos_tok]
        self.end_approach_id = self.tok_to_id["<D>"]
        self.time_id = self.tok_to_id[self.time_tok]
        self.ped_on_id = self.tok_to_id[self.ped_on_tok]
        self.ped_off_id = self.tok_to_id[self.ped_off_tok]
        self.piano_prefix_id = self.tok_to_id[("prefix", "instrument", "piano")]
        self.section_to_id = {
            token: self.tok_to_id[token] for token in SECTION_TOKENS
        }
        self.section_ids = tuple(self.section_to_id.values())

    def _tokenize_midi_dict(
        self,
        midi_dict: MidiDict,
        remove_preceding_silence: bool = True,
        add_dim_tok: bool = True,
        add_eos_tok: bool = True,
    ):
        ticks_per_beat = midi_dict.ticks_per_beat
        midi_dict.remove_instruments(self.config["ignore_instruments"])
        if not midi_dict.note_msgs:
            raise ValueError("note_msgs is empty after ignoring instruments")

        channels_used = {message["channel"] for message in midi_dict.note_msgs}
        channel_to_instrument = {
            message["channel"]: midi_dict.program_to_instrument[message["data"]]
            for message in midi_dict.instrument_msgs
            if message["channel"] != 9
        }
        for channel in channels_used:
            if channel_to_instrument.get(channel) is None and channel != 9:
                channel_to_instrument[channel] = "piano"
        if self.include_pedal:
            assert set(channel_to_instrument.values()) == {"piano"}

        prefix = [
            ("prefix", "instrument", instrument)
            for instrument in set(channel_to_instrument.values())
        ]
        random.shuffle(prefix)

        # Stable sort keeps pedal events before simultaneous note events.
        messages = (
            midi_dict.pedal_msgs + midi_dict.note_msgs
            if self.include_pedal
            else list(midi_dict.note_msgs)
        )
        messages.sort(
            key=lambda message: (
                message["data"]["start"]
                if message["type"] == "note"
                else message["tick"]
            )
        )
        initial_tick = 0 if not remove_preceding_silence else messages[0]["tick"]
        unformatted = []
        current_ms = 0
        for message in messages:
            is_note = message["type"] == "note"
            start_tick = message["data"]["start"] if is_note else message["tick"]
            previous_ms = current_ms
            current_ms = get_duration_ms(
                start_tick=initial_tick,
                end_tick=start_tick,
                tempo_msgs=midi_dict.tempo_msgs,
                ticks_per_beat=ticks_per_beat,
            )
            for _ in range(
                current_ms // self.abs_time_step_ms
                - previous_ms // self.abs_time_step_ms
            ):
                unformatted.append(self.time_tok)

            onset = self._quantize_onset(current_ms % self.abs_time_step_ms)
            if not is_note:
                if message["data"] == 1:
                    unformatted.append(self.ped_on_tok)
                elif message["data"] == 0:
                    unformatted.append(self.ped_off_tok)
                else:
                    raise ValueError("invalid pedal message")
                unformatted.append(("onset", onset))
                continue

            note = message["data"]
            duration = get_duration_ms(
                start_tick=start_tick,
                end_tick=note["end"],
                tempo_msgs=midi_dict.tempo_msgs,
                ticks_per_beat=ticks_per_beat,
            )
            unformatted.extend([
                (
                    channel_to_instrument[message["channel"]],
                    note["pitch"],
                    self._quantize_velocity(note["velocity"]),
                ),
                ("onset", onset),
                ("dur", self._quantize_dur(duration)),
            ])

        return self._format(
            prefix=prefix,
            unformatted_seq=unformatted,
            add_dim_tok=add_dim_tok,
            add_eos_tok=add_eos_tok,
        )

    def tokenize_midi(self, path: str | Path) -> list[int]:
        midi = MidiDict.from_midi(mid_path=str(path))
        return self.encode(self.tokenize(midi, add_dim_tok=False))

    def detokenize_midi(self, ids: list[int]) -> MidiDict:
        ignored = {self.end_approach_id, *self.section_ids}
        clean = [token_id for token_id in ids if token_id not in ignored]
        return self.detokenize(self.decode(clean))


def get_tokenizer() -> PianoMidiTokenizer:
    """The shared tokenizer instance, built on first use."""
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = PianoMidiTokenizer(CONFIG_PATH)
    return _TOKENIZER
