import argparse
import hashlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

if __package__:
    from dataset.common import read_jsonl, write_jsonl
else:
    from common import read_jsonl, write_jsonl


@dataclass(frozen=True, slots=True)
class ManifestSplitSummary:
    total_rows: int
    train_rows: int
    validation_rows: int
    speaker_groups: int


def split_manifest(
    input_path: Path,
    train_output_path: Path,
    validation_output_path: Path,
    validation_ratio: float = 0.2,
) -> ManifestSplitSummary:
    if not 0 < validation_ratio < 1:
        raise ValueError(f"validation_ratio must be between 0 and 1: {validation_ratio}")

    rows = read_jsonl(input_path)
    groups = _group_rows_by_speaker(rows)
    target_validation_rows = max(1, round(len(rows) * validation_ratio))
    validation_group_keys = _choose_validation_groups(groups, target_validation_rows)

    train_rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    for key in sorted(groups):
        target = validation_rows if key in validation_group_keys else train_rows
        for row in groups[key]:
            copied_row = dict(row)
            copied_row["split"] = "validation" if key in validation_group_keys else "train"
            target.append(copied_row)

    train_rows.sort(key=lambda row: str(row["id"]))
    validation_rows.sort(key=lambda row: str(row["id"]))
    write_jsonl(train_output_path, train_rows)
    write_jsonl(validation_output_path, validation_rows)
    return ManifestSplitSummary(
        total_rows=len(rows),
        train_rows=len(train_rows),
        validation_rows=len(validation_rows),
        speaker_groups=len(groups),
    )


def _group_rows_by_speaker(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[_speaker_key(Path(str(row["audio_path"])).stem)].append(row)
    return dict(groups)


def _speaker_key(audio_stem: str) -> str:
    parts = audio_stem.split("-")
    if len(parts) >= 10 and parts[0] == "ID":
        return "-".join((*parts[:5], *parts[-3:]))
    return audio_stem


def _choose_validation_groups(
    groups: dict[str, list[dict[str, object]]],
    target_validation_rows: int,
) -> set[str]:
    ordered_keys = sorted(groups, key=lambda key: hashlib.sha256(key.encode("utf-8")).hexdigest())
    selected_keys: set[str] = set()
    selected_rows = 0
    for key in ordered_keys:
        if selected_rows >= target_validation_rows and selected_keys:
            break
        selected_keys.add(key)
        selected_rows += len(groups[key])
    return selected_keys


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--train-output", required=True, type=Path)
    parser.add_argument("--validation-output", required=True, type=Path)
    parser.add_argument("--validation-ratio", default=0.2, type=float)
    args = parser.parse_args(argv)

    summary = split_manifest(
        input_path=args.input,
        train_output_path=args.train_output,
        validation_output_path=args.validation_output,
        validation_ratio=args.validation_ratio,
    )
    print(
        "split "
        f"total={summary.total_rows} "
        f"train={summary.train_rows} "
        f"validation={summary.validation_rows} "
        f"speaker_groups={summary.speaker_groups}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
