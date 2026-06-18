import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

JsonValue = dict[str, Any] | list[Any] | str | int | float | bool | None

AUDIO_EXTENSIONS = {".flac", ".m4a", ".mp3", ".ogg", ".wav", ".webm"}
TEXT_FIELD_HINTS = (
    "transcript",
    "transcription",
    "sentence",
    "utterance",
    "script",
    "text",
    "발화",
    "전사",
    "문장",
)
AUDIO_FIELD_HINTS = (
    "audio",
    "audiofile",
    "audiopath",
    "file",
    "filename",
    "filepath",
    "path",
    "sound",
    "wav",
    "음성",
)
DURATION_FIELD_HINTS = ("duration", "durationseconds", "lengthseconds", "playtime", "seconds")


@dataclass(frozen=True)
class FieldValue:
    field_path: str
    value: str | int | float | bool | None


def iter_json_files(root: Path) -> tuple[Path, ...]:
    if not root.exists():
        raise FileNotFoundError(f"Label root does not exist: {root}")
    return tuple(sorted(path for path in root.rglob("*.json") if path.is_file()))


def read_json(path: Path) -> JsonValue:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(f"{json.dumps(row, ensure_ascii=False)}\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def iter_field_values(value: JsonValue, path_parts: tuple[str, ...] = ()) -> Iterator[FieldValue]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield from iter_field_values(child, (*path_parts, str(key)))
        return

    if isinstance(value, list):
        for index, child in enumerate(value):
            yield from iter_field_values(child, (*path_parts, str(index)))
        return

    yield FieldValue(field_path=".".join(path_parts), value=value)


def find_transcript_candidates(payload: JsonValue) -> tuple[FieldValue, ...]:
    candidates: list[FieldValue] = []
    for field in iter_field_values(payload):
        if not isinstance(field.value, str):
            continue
        text = field.value.strip()
        if not text or looks_like_audio_reference(text):
            continue
        if _path_matches(field.field_path, TEXT_FIELD_HINTS):
            candidates.append(FieldValue(field.field_path, text))
    return tuple(candidates)


def find_audio_path_candidates(payload: JsonValue) -> tuple[FieldValue, ...]:
    candidates: list[FieldValue] = []
    for field in iter_field_values(payload):
        if not isinstance(field.value, str):
            continue
        text = field.value.strip()
        if not text:
            continue
        if _path_matches(field.field_path, AUDIO_FIELD_HINTS) and looks_like_audio_reference(text):
            candidates.append(FieldValue(field.field_path, text))
    return tuple(candidates)


def find_duration_seconds(payload: JsonValue) -> float | None:
    for field in iter_field_values(payload):
        if not _path_matches(field.field_path, DURATION_FIELD_HINTS):
            continue
        if isinstance(field.value, int | float):
            return float(field.value)
        if isinstance(field.value, str) and _is_float(field.value):
            return float(field.value)
    return None


def looks_like_audio_reference(value: str) -> bool:
    suffix = Path(value.strip().replace("\\", "/")).suffix.casefold()
    return suffix in AUDIO_EXTENSIONS


def top_level_keys(payload: JsonValue) -> tuple[str, ...]:
    if not isinstance(payload, dict):
        return ()
    return tuple(str(key) for key in payload)


def _path_matches(field_path: str, hints: tuple[str, ...]) -> bool:
    normalized = _normalize(field_path)
    return any(hint in normalized for hint in hints)


def _normalize(value: str) -> str:
    return value.casefold().replace("_", "").replace("-", "").replace(".", "")


def _is_float(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True
