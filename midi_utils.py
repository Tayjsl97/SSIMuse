"""MIDI-to-piano-roll processing used by the real-data evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def midi_to_piano_roll(
    path: str | Path,
    *,
    steps_per_quarter: int = 4,
    track_index: int | None = None,
    ignore_drums: bool = True,
    trim_leading_silence: bool = True,
) -> np.ndarray:
    """Load all pitched MIDI channels into a binary ``(time, 128)`` roll.

    The adapter quantizes note boundaries to the requested step grid. Set
    ``track_index`` for melody-only evaluation; ``None`` merges all tracks and
    must not be confused with the paper's selected-melody input.
    """
    try:
        import mido
    except ImportError as error:
        raise ImportError("MIDI support requires `pip install mido`") from error

    midi = mido.MidiFile(Path(path))
    if track_index is None:
        messages = mido.merge_tracks(midi.tracks)
    else:
        if not 0 <= track_index < len(midi.tracks):
            raise ValueError(f"track_index {track_index} is outside the MIDI track range")
        messages = midi.tracks[track_index]
    absolute_tick = 0
    active: dict[tuple[int, int], list[int]] = {}
    notes: list[tuple[int, int, int]] = []
    for message in messages:
        absolute_tick += int(message.time)
        if message.type not in {"note_on", "note_off"}:
            continue
        channel = int(getattr(message, "channel", 0))
        if ignore_drums and channel == 9:
            continue
        pitch = int(message.note)
        key = (channel, pitch)
        is_on = message.type == "note_on" and int(message.velocity) > 0
        if is_on:
            active.setdefault(key, []).append(absolute_tick)
        elif active.get(key):
            start = active[key].pop()
            if absolute_tick > start:
                notes.append((start, absolute_tick, pitch))
    if not notes:
        raise ValueError(f"no pitched notes found in {path}")

    quantized = []
    for start, end, pitch in notes:
        left = round(start * steps_per_quarter / midi.ticks_per_beat)
        right = max(left + 1, round(end * steps_per_quarter / midi.ticks_per_beat))
        quantized.append((left, right, pitch))
    origin = min(left for left, _, _ in quantized) if trim_leading_silence else 0
    length = max(right for _, right, _ in quantized) - origin
    roll = np.zeros((length, 128), dtype=np.float64)
    for left, right, pitch in quantized:
        left, right = max(0, left - origin), min(length, right - origin)
        if right > left:
            roll[left:right, pitch] = 1.0
    return roll
