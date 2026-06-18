from collections.abc import Sequence
from typing import Protocol


class TimedTrainingRow(Protocol):
    duration_seconds: float | None


def count_window_transcript_mismatches(rows: Sequence[TimedTrainingRow], max_audio_seconds: float) -> int:
    if max_audio_seconds <= 0:
        raise RuntimeError(f"max_audio_seconds must be positive: {max_audio_seconds}")
    return sum(
        1
        for row in rows
        if row.duration_seconds is None or row.duration_seconds > max_audio_seconds
    )


def duration_seconds(row: dict[str, object]) -> float | None:
    raw_value = row.get("duration_seconds")
    if isinstance(raw_value, int | float):
        return float(raw_value)
    if isinstance(raw_value, str) and _is_float(raw_value):
        return float(raw_value)
    return None


def _is_float(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True
