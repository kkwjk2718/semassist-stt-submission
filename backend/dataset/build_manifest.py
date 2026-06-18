import argparse
from pathlib import Path
from typing import Sequence

if __package__:
    from dataset.common import (
        find_audio_path_candidates,
        find_duration_seconds,
        find_transcript_candidates,
        iter_json_files,
        read_json,
        write_jsonl,
    )
else:
    from common import (
        find_audio_path_candidates,
        find_duration_seconds,
        find_transcript_candidates,
        iter_json_files,
        read_json,
        write_jsonl,
    )


def build_manifest(
    label_root: Path,
    audio_root: Path,
    output_path: Path,
    disorder_type: str = "brain_neurologic",
    max_files: int | None = None,
) -> list[dict[str, object]]:
    audio_index = _index_audio_files(audio_root)
    entries: list[dict[str, object]] = []
    label_files = iter_json_files(label_root)
    if max_files is not None:
        label_files = label_files[:max_files]

    for label_file in label_files:
        payload = read_json(label_file)
        transcript = _first_transcript(payload)
        audio_path = _first_audio_path(payload, audio_root, audio_index)
        if transcript is None or audio_path is None:
            continue

        entry: dict[str, object] = {
            "id": f"sample_{len(entries) + 1:06d}",
            "audio_path": str(audio_path),
            "text": transcript,
            "disorder_type": disorder_type,
            "split": _split_for_label_path(label_file),
        }
        duration_seconds = find_duration_seconds(payload)
        if duration_seconds is not None:
            entry["duration_seconds"] = duration_seconds
        entries.append(entry)

    write_jsonl(output_path, entries)
    return entries


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-root", required=True, type=Path)
    parser.add_argument("--audio-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--disorder-type", default="brain_neurologic")
    parser.add_argument("--max-files", type=int)
    args = parser.parse_args(argv)

    entries = build_manifest(
        label_root=args.label_root,
        audio_root=args.audio_root,
        output_path=args.output,
        disorder_type=args.disorder_type,
        max_files=args.max_files,
    )
    print(f"Wrote {len(entries)} manifest rows to {args.output}")
    return 0


def _first_transcript(payload: object) -> str | None:
    candidates = find_transcript_candidates(payload)
    if not candidates:
        return None
    value = candidates[0].value
    return value if isinstance(value, str) else None


def _first_audio_path(
    payload: object,
    audio_root: Path,
    audio_index: dict[str, Path],
) -> Path | None:
    for candidate in find_audio_path_candidates(payload):
        if not isinstance(candidate.value, str):
            continue
        resolved_path = _resolve_audio_path(audio_root, candidate.value, audio_index)
        if resolved_path is not None:
            return resolved_path
    return None


def _resolve_audio_path(
    audio_root: Path,
    raw_value: str,
    audio_index: dict[str, Path],
) -> Path | None:
    normalized_value = raw_value.strip().replace("\\", "/")
    raw_path = Path(normalized_value)
    direct_candidates = (
        raw_path,
        audio_root / raw_path,
    )
    for candidate in direct_candidates:
        if candidate.is_file():
            return candidate
    for candidate_name in _audio_name_variants(raw_path.name):
        resolved_path = audio_index.get(candidate_name) or audio_index.get(candidate_name.casefold())
        if resolved_path is not None:
            return resolved_path
    return None


def _index_audio_files(audio_root: Path) -> dict[str, Path]:
    if not audio_root.exists():
        raise FileNotFoundError(f"Audio root does not exist: {audio_root}")

    index: dict[str, Path] = {}
    for path in sorted(audio_root.rglob("*")):
        if path.is_file():
            index.setdefault(path.name, path)
            index.setdefault(path.name.casefold(), path)
    return index


def _audio_name_variants(file_name: str) -> tuple[str, ...]:
    variants: list[str] = []

    def add(value: str) -> None:
        if value not in variants:
            variants.append(value)

    add(file_name)
    if "중복1" in file_name:
        add(file_name.replace("중복1", ""))
    if "중복2" in file_name:
        add(file_name.replace("중복2", "2"))
        add(file_name.replace("중복2", ""))
    for value in tuple(variants):
        if value.startswith("ID-02-26-M-"):
            add(value.replace("ID-02-26-M-", "ID-02-26-N-", 1))
    return tuple(variants)


def _split_for_label_path(label_path: Path) -> str:
    normalized_parts = {part.casefold() for part in label_path.parts}
    if "2.validation" in normalized_parts or "validation" in normalized_parts or "valid" in normalized_parts:
        return "validation"
    return "train"


if __name__ == "__main__":
    raise SystemExit(main())
